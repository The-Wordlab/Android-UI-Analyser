"""Per-device leases, so parallel agents stop fighting over the same emulator.

Three emulators attached, Claude testing search, Cursor testing delete — and both land on
``emulator-5554``, because ``connect(serial=None)`` takes "the only/first device". Each
agent then drives a screen the other is mutating. Nothing errors; the results are just
quietly wrong.

The design keeps two properties that matter more than features:

**No deadlock is possible.** Expiry is computed when a lease is *read*, not by a reaper
process. A lease owned by an agent process expires as soon as that process is gone. The sole
bounded exception is an explicitly pending handoff, which reserves the target only until its
five-minute token expires so a spawned receiver cannot lose it in the process gap. Friendly
explicit labels remain labels only; the caller pid + start time travel separately so a warm
daemon cannot accidentally keep the label alive. There is no cleanup process that can fail.

**Identity has to be stable across an agent's calls, or stickiness inverts into churn.**
Measured: a session id is *not* stable — consecutive tool calls from one agent reported sids
40966 then 40979, because each shell invocation gets its own session. Keying on that would
hand the agent a different emulator every command. Walking up past the shells *and past the
per-command launchers* (``uv run``, ``uvx``, ``npx``, ``env``, …) finds the agent process
itself (``claude``, ``cursor``, …), which lives for the whole run. Stopping at a launcher
gives a fresh name every invocation, which is worse than no lease: the agent is locked out of
the device it took a moment earlier, by a holder that no longer exists.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .atomic import atomic_create_text, atomic_write_text

# Long enough that a single blocking call cannot outlive it: `--until` waits legitimately run
# 90-120s on a slow backend, and a lease that expires mid-wait would let another agent seize a
# device that is actively in use — strictly worse than no lease at all. Process-bound leases live
# exactly as long as their owner process; the TTL is the fallback only for legacy/unbound owners.
DEFAULT_TTL_S = 900
HANDOFF_TTL_S = 300

_SHELLS = {
    "sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh", "login", "-zsh", "-bash",
    "cmd", "powershell", "pwsh",
}

# Wrappers that exist for the duration of one command. Naming an agent after one of these gives it
# a fresh identity per invocation, so it cannot re-acquire the device it just leased — the caller
# above the wrapper is the one that persists.
_LAUNCHERS = {
    "uv", "uvx", "pipx", "poetry", "pdm", "rye", "hatch", "pipenv", "conda", "micromamba",
    "npx", "pnpx", "bunx", "nix", "nix-shell", "direnv", "mise", "asdf",
    "env", "sudo", "doas", "nohup", "stdbuf", "xargs", "time", "timeout", "caffeinate", "arch",
}

_OWNER_GUARD_STATE = threading.local()
_LEASE_GUARD_STATE = threading.local()
_DEVICE_GUARD_STATE = threading.local()
_COMMAND_GUARD_STATE = threading.local()
_HOST_GUARD_STATE = threading.local()
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, Any] = {}
_THREAD_RW_LOCKS: dict[str, Any] = {}
_WINDOWS_LOCK_POLL_S = 0.05


class _ThreadRWLock:
    """Small writer-preferring RW lock paired with the host's cross-process file lock."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers: dict[int, int] = {}
        self._writer: int | None = None
        self._write_depth = 0
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            if self._writer == ident or ident in self._readers:
                self._readers[ident] = self._readers.get(ident, 0) + 1
                return
            while self._writer is not None or self._waiting_writers:
                self._condition.wait()
            self._readers[ident] = 1

    def release_read(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            count = self._readers.get(ident, 0)
            if count <= 1:
                self._readers.pop(ident, None)
            else:
                self._readers[ident] = count - 1
            self._condition.notify_all()

    def acquire_write(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            if self._writer == ident:
                self._write_depth += 1
                return
            if ident in self._readers:
                raise RuntimeError("a device-use lock cannot be upgraded in place")
            self._waiting_writers += 1
            try:
                while self._writer is not None or self._readers:
                    self._condition.wait()
                self._writer = ident
                self._write_depth = 1
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        ident = threading.get_ident()
        with self._condition:
            if self._writer != ident:
                raise RuntimeError("device transition lock released by a different thread")
            self._write_depth -= 1
            if self._write_depth == 0:
                self._writer = None
                self._condition.notify_all()


class LeaseOwner(str):
    """Human-readable owner label carrying its separate process identity in-process."""

    pid: int | None
    started: str | None

    def __new__(
        cls,
        label: str,
        *,
        pid: int | None = None,
        started: str | None = None,
    ) -> LeaseOwner:
        value = str.__new__(cls, label)
        value.pid = pid
        value.started = started
        return value


# --------------------------------------------------------------------------- identity


def _is_windows_host() -> bool:
    return os.name == "nt"


def _windows_process_snapshot(pid: int) -> tuple[str, int | None]:
    """Return ``(executable_name, parent_pid)`` using the Windows process snapshot API."""

    import ctypes
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        return "", None
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            if int(entry.th32ProcessID) == pid:
                return str(entry.szExeFile), int(entry.th32ParentProcessID) or None
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return "", None


def _windows_proc_started(pid: int) -> str:
    """Stable Windows creation-time token for detecting PID reuse."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return ""
        ticks = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return f"{ticks:x}"
    finally:
        kernel32.CloseHandle(handle)


def _windows_process_exists(pid: int) -> bool:
    """Check liveness without ``os.kill(pid, 0)``, which can terminate on Windows."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        # Access denied proves that a process exists even though its metadata is unavailable.
        return ctypes.get_last_error() == 5  # type: ignore[attr-defined]  # ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and (
            exit_code.value == 259  # STILL_ACTIVE
        )
    finally:
        kernel32.CloseHandle(handle)


def _proc_name(pid: int) -> str:
    if _is_windows_host():
        return _windows_process_snapshot(pid)[0]
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return ""
    return (out.stdout or "").strip().rsplit("/", 1)[-1]


def _proc_ppid(pid: int) -> int | None:
    if _is_windows_host():
        return _windows_process_snapshot(pid)[1]
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        raw = (out.stdout or "").strip()
        return int(raw) if raw.isdigit() else None
    except Exception:
        return None


def _proc_started(pid: int) -> str:
    """Process start time — pairs with the pid so reuse cannot alias two agents."""
    if _is_windows_host():
        return _windows_proc_started(pid)
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return "".join((out.stdout or "").split())[-8:]
    except Exception:
        return ""


def _process_exists(pid: int) -> bool:
    if _is_windows_host():
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _worker_scope() -> str:
    """Which of several agents inside one ancestor process this caller is, or ``""`` for the only one.

    :func:`_derived_owner` is right about *when* an owner starts and ends, and blind to *how many*
    agents live inside it. Three QA workers running as subagents of one ``claude`` process all
    resolved to ``claude-1708-2:242026``; :func:`select` handed two of them the same device through
    its sticky branch and they drove one screen for six overlapping minutes without an error.

    An explicit run-cache override is the discriminator because a parallel harness is already
    required to give each worker its own, so this asks callers nothing new.

    **This is deliberately not folded into the owner label.** A first attempt did that and broke
    eleven tests in ``test_an_agent_keeps_one_name_across_commands``, because ``tests/conftest.py``
    sets ``AUA_CACHE__DIR`` for every test — the real lesson being that an explicit run cache is far
    more common than "a parallel worker", so it must not rewrite an identity other machinery parses
    back (lease expiry reads the pid and start time out of the stored owner string). As a record
    field it leaves every label byte-identical and every existing lease valid.

    Not to be confused with :attr:`LeaseCfg.registry_dir`, which stays independent of ``cache.dir``
    on purpose. That is lease *storage*: if the registry followed the override, two agents would each
    read an empty registry and see the same device as free. Matching is the opposite case — two
    callers with different run caches are, by the harness contract, two different callers.

    Only the environment is read. A ``--cache-dir`` flag differing between siblings in one process is
    not covered; that gap is left open rather than papered over, because no such caller has been seen
    and the flag cannot be reached here without threading config through fifteen call sites.
    """

    # A warm daemon serves one caller and must hold the lease as that caller. It cannot derive
    # this for itself: its environment always carries a pinned ``AUA_CACHE__DIR`` (see
    # ``daemon._daemon_environment``, which is right about cache isolation), so a daemon started
    # by a caller who set nothing would invent a scope its own caller does not have — and the
    # caller then reads back its own lease as foreign and refuses the device it is holding.
    # Present-but-empty is a real answer, so the key's presence decides, not its truthiness.
    if "AUA_WORKER_SCOPE" in os.environ:
        return os.environ["AUA_WORKER_SCOPE"].strip()
    raw = (os.environ.get("AUA_CACHE__DIR") or "").strip()
    if not raw:
        return ""
    # By path, not by spelling: one worker naming its own directory two ways is one worker.
    return hashlib.sha256(str(Path(raw).expanduser()).encode("utf-8")).hexdigest()[:8]


def _derived_owner() -> LeaseOwner:
    """The first ancestor that outlives one command: the agent process, stable for its whole run.

    Falls back to this process when the walk finds nothing better, which is the right answer
    for a human at a terminal running one command.
    """
    pid: int | None = os.getpid()
    seen = 0
    while pid and pid > 1 and seen < 12:
        seen += 1
        parent = _proc_ppid(pid)
        if parent is None or parent <= 1:
            break
        name = _proc_name(parent)
        if name and not _is_transient(name):
            started = _proc_started(parent)
            return LeaseOwner(
                f"{name}-{parent}-{started}".strip("-"),
                pid=parent,
                started=started or None,
            )
        pid = parent
    current = os.getpid()
    started = _proc_started(current)
    return LeaseOwner(
        f"pid-{current}-{started}".strip("-"),
        pid=current,
        started=started or None,
    )


def derive_identity() -> str:
    return _derived_owner()


def _is_transient(name: str) -> bool:
    normalized = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    if normalized.endswith(".exe"):
        normalized = normalized[:-4]
    return (
        normalized in _SHELLS
        or normalized in _LAUNCHERS
        or normalized.startswith("python")
    )


def resolve_owner(explicit: str | None = None) -> str:
    """``--owner`` → ``$AUA_OWNER`` → derived. Never None: every caller is somebody."""
    if isinstance(explicit, LeaseOwner):
        return explicit
    derived = _derived_owner()
    if explicit and str(explicit).strip():
        return LeaseOwner(
            str(explicit).strip(), pid=derived.pid, started=derived.started
        )
    env = (os.environ.get("AUA_OWNER") or "").strip()
    if env:
        return LeaseOwner(env, pid=derived.pid, started=derived.started)
    return derived


def owner_caller(owner: str) -> dict[str, Any] | None:
    """Structured caller identity for daemon transport; never encoded into the label."""
    process = _owner_process(owner)
    if process is None:
        return None
    pid, started = process
    return {"pid": pid, "started": started}


def bind_owner_caller(owner: str | None, caller: Any) -> str | None:
    """Rebuild a process-bound owner received over a daemon request."""
    if not owner:
        return None
    if not isinstance(caller, dict):
        return owner
    pid = caller.get("pid")
    started = caller.get("started")
    if not isinstance(pid, int) or pid <= 1 or not isinstance(started, str) or not started:
        return owner
    return LeaseOwner(str(owner), pid=pid, started=started)


# --------------------------------------------------------------------------- storage


def lease_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "leases"


def _safe_serial(serial: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in serial)


def _lease_path(cache_dir: str | Path, serial: str) -> Path:
    return lease_dir(cache_dir) / f"{_safe_serial(serial)}.json"


def _thread_lock(key: str) -> Any:
    """One re-entrant in-process lock for the same host transaction key."""

    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _thread_rw_lock(key: str) -> _ThreadRWLock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_RW_LOCKS.setdefault(key, _ThreadRWLock())


def _host_lock_kind() -> str:
    """The cross-process primitive available on this host.

    POSIX ``flock`` supports shared readers. Windows' stdlib ``msvcrt`` primitive is
    exclusive-only, so device readers intentionally serialize there. That costs concurrency,
    but preserves the lease/transfer safety boundary instead of silently falling back to a
    process-local lock.
    """

    return "windows" if _is_windows_host() else "posix"


def _windows_lock_module() -> Any:
    import msvcrt

    return msvcrt


def _acquire_windows_file_lock(handle: Any) -> None:
    """Block until byte zero is exclusively locked through ``msvcrt``."""

    msvcrt = _windows_lock_module()
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        # ``msvcrt.locking`` cannot lock past EOF. Concurrent initializers may both write the
        # same sentinel byte; either result is the same one-byte lock file.
        handle.write("\0")
        handle.flush()
    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN} and getattr(
                exc, "winerror", None
            ) not in {32, 33}:
                raise
            time.sleep(_WINDOWS_LOCK_POLL_S)


def _release_windows_file_lock(handle: Any) -> None:
    msvcrt = _windows_lock_module()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _acquire_file_lock(handle: Any, *, exclusive: bool) -> str:
    """Acquire a process-level file lock and return the backend needed for release."""

    if _host_lock_kind() == "windows":
        # Windows has no shared msvcrt lock; an exclusive range lock is the safe fallback.
        _acquire_windows_file_lock(handle)
        return "windows"
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return "posix"


def _release_file_lock(handle: Any, backend: str) -> None:
    if backend == "windows":
        _release_windows_file_lock(handle)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _lease_guard(cache_dir: str | Path, serial: str) -> Iterator[None]:
    """Serialize mutations of one lease file across threads and processes.

    Lease reads stay lock-free. Writers need the guard so a transfer cannot be overwritten by
    a heartbeat that read the old owner just before acceptance. POSIX uses ``flock``; Windows
    uses a one-byte ``msvcrt`` range lock.
    """

    key = f"lease|{Path(cache_dir).expanduser()}|{serial}"
    with _thread_lock(key):
        active = getattr(_LEASE_GUARD_STATE, "keys", None)
        if active is None:
            active = set()
            _LEASE_GUARD_STATE.keys = active
        if key in active:
            yield
            return
        path = lease_dir(cache_dir) / ".locks" / f"{_safe_serial(serial)}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        backend: str | None = None
        try:
            backend = _acquire_file_lock(handle, exclusive=True)
            active.add(key)
            yield
        finally:
            active.discard(key)
            if backend is not None:
                with contextlib.suppress(Exception):
                    _release_file_lock(handle, backend)
            handle.close()


@contextlib.contextmanager
def _device_lock(
    cache_dir: str | Path, serial: str, *, exclusive: bool
) -> Iterator[None]:
    """Shared normal use or an exclusive ownership transition for one physical target.

    Windows conservatively serializes normal readers because ``msvcrt`` has no shared range
    lock. Ownership transitions still exclude every reader and command across processes.
    """

    key = f"device|{Path(cache_dir).expanduser()}|{serial}"
    active = getattr(_DEVICE_GUARD_STATE, "entries", None)
    if active is None:
        active = {}
        _DEVICE_GUARD_STATE.entries = active
    current = active.get(key)
    if current is not None:
        if exclusive and current["mode"] != "exclusive":
            raise RuntimeError("a device-use lock cannot be upgraded in place")
        current["depth"] += 1
        try:
            yield
        finally:
            current["depth"] -= 1
        return

    rw_lock = _thread_rw_lock(key)
    thread_exclusive = exclusive or _host_lock_kind() == "windows"
    acquire = rw_lock.acquire_write if thread_exclusive else rw_lock.acquire_read
    release = rw_lock.release_write if thread_exclusive else rw_lock.release_read
    acquire()
    lock_dir = lease_dir(cache_dir) / ".locks"
    path = lock_dir / f"device-{_safe_serial(serial)}.lock"
    gate_path = lock_dir / f"device-gate-{_safe_serial(serial)}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gate_handle = gate_path.open("a+", encoding="utf-8")
    except BaseException:
        release()
        raise
    try:
        handle = path.open("a+", encoding="utf-8")
    except BaseException:
        gate_handle.close()
        release()
        raise
    backend: str | None = None
    gate_backend: str | None = None
    try:
        file_exclusive = exclusive or _host_lock_kind() == "windows"
        gate_backend = _acquire_file_lock(gate_handle, exclusive=file_exclusive)
        backend = _acquire_file_lock(handle, exclusive=file_exclusive)
        if not exclusive:
            _release_file_lock(gate_handle, gate_backend)
            gate_backend = None
        active[key] = {"mode": "exclusive" if exclusive else "shared", "depth": 1}
        yield
    finally:
        active.pop(key, None)
        if backend is not None:
            with contextlib.suppress(Exception):
                _release_file_lock(handle, backend)
        if gate_backend is not None:
            with contextlib.suppress(Exception):
                _release_file_lock(gate_handle, gate_backend)
        handle.close()
        gate_handle.close()
        release()


@contextlib.contextmanager
def _device_guard(cache_dir: str | Path, serial: str) -> Iterator[None]:
    with _device_lock(cache_dir, serial, exclusive=True):
        yield


@contextlib.contextmanager
def device_use(cache_dir: str | Path, serial: str) -> Iterator[None]:
    """Hold a shared command-lifetime fence while this process may touch the target."""

    with _device_lock(cache_dir, serial, exclusive=False):
        yield


@contextlib.contextmanager
def _command_guard(cache_dir: str | Path, serial: str) -> Iterator[None]:
    """Serialize foreground commands while still allowing short background reads."""

    key = f"command|{Path(cache_dir).expanduser()}|{serial}"
    with _thread_lock(key):
        active = getattr(_COMMAND_GUARD_STATE, "keys", None)
        if active is None:
            active = set()
            _COMMAND_GUARD_STATE.keys = active
        if key in active:
            yield
            return
        path = lease_dir(cache_dir) / ".locks" / f"command-{_safe_serial(serial)}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        backend: str | None = None
        try:
            backend = _acquire_file_lock(handle, exclusive=True)
            active.add(key)
            yield
        finally:
            active.discard(key)
            if backend is not None:
                with contextlib.suppress(Exception):
                    _release_file_lock(handle, backend)
            handle.close()


@contextlib.contextmanager
def device_command(cache_dir: str | Path, serial: str) -> Iterator[None]:
    """One foreground command: exclusive against commands, shared with perception readers."""

    with _command_guard(cache_dir, serial), device_use(cache_dir, serial):
        yield


@contextlib.contextmanager
def device_transaction(cache_dir: str | Path, serial: str) -> Iterator[None]:
    """Keep a full device command or ownership transition exclusive on one target."""

    with _device_guard(cache_dir, serial):
        yield


@contextlib.contextmanager
def host_transaction(cache_dir: str | Path, key: str) -> Iterator[None]:
    """Serialize one host-wide operation across AUA processes.

    Device locks deliberately key on a target serial. Some platform facilities instead share
    one host resource — Android's ADB server endpoint is the motivating example — and racing
    their bootstrap can make every otherwise-independent target disappear at once. Keep this
    primitive in the coordination layer so platform adapters can share the same portable POSIX /
    Windows locking guarantees without pretending a host service is a device lease.
    """

    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    lock_key = f"host|{Path(cache_dir).expanduser()}|{digest}"
    with _thread_lock(lock_key):
        active = getattr(_HOST_GUARD_STATE, "digests", None)
        if active is None:
            active = set()
            _HOST_GUARD_STATE.digests = active
        if digest in active:
            yield
            return
        path = lease_dir(cache_dir) / ".locks" / f"host-{digest}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        backend: str | None = None
        try:
            backend = _acquire_file_lock(handle, exclusive=True)
            active.add(digest)
            yield
        finally:
            active.discard(digest)
            if backend is not None:
                with contextlib.suppress(Exception):
                    _release_file_lock(handle, backend)
            handle.close()


@contextlib.contextmanager
def _owner_guard(cache_dir: str | Path, owner: str) -> Iterator[None]:
    """Serialize target selection for one process-bound owner."""

    process = owner_caller(owner) or {}
    identity = f"{owner}|{process.get('pid')}|{process.get('started')}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    active = getattr(_OWNER_GUARD_STATE, "digests", None)
    if active is None:
        active = set()
        _OWNER_GUARD_STATE.digests = active
    if digest in active:
        yield
        return
    with _thread_lock(f"owner|{digest}"):
        path = lease_dir(cache_dir) / ".locks" / f"owner-{digest}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        backend: str | None = None
        try:
            backend = _acquire_file_lock(handle, exclusive=True)
            active.add(digest)
            yield
        finally:
            active.discard(digest)
            if backend is not None:
                with contextlib.suppress(Exception):
                    _release_file_lock(handle, backend)
            handle.close()


@contextlib.contextmanager
def owner_transaction(cache_dir: str | Path, owner: str) -> Iterator[None]:
    """Keep a multi-step replacement serialized against this owner's ordinary selection."""

    with _owner_guard(cache_dir, owner):
        yield


def _handoff_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


def _expired(entry: dict[str, Any], *, now: float | None = None) -> bool:
    now = _now() if now is None else now
    handoff = entry.get("handoff")
    if isinstance(handoff, dict) and now < float(handoff.get("expires") or 0):
        # A transfer token is a short reservation: the source may die after spawning the child,
        # but no third agent may take the device in the gap before that child accepts it.
        return False
    owner_pid = entry.get("owner_pid")
    owner_started = entry.get("owner_started")
    if isinstance(owner_pid, int) and owner_pid > 1:
        if not _process_exists(owner_pid):
            return True
        current_started = _proc_started(owner_pid)
        # If metadata is temporarily inaccessible, preserving the lease is safer than handing a
        # live device to a second process. A known different token still proves PID reuse.
        return bool(owner_started and current_started and current_started != owner_started)
    ttl = float(entry.get("ttl_s") or DEFAULT_TTL_S)
    return (now - float(entry.get("last_activity") or 0)) > ttl


# ``{name}-{pid}-{started}``, where *name* can itself contain "-". Anchoring on the
# digits-only pid and matching greedily makes the last valid pid segment win.
_OWNER_RE = re.compile(r"^(?P<name>.+)-(?P<pid>\d+)-(?P<started>.+)$")


def _owner_process(owner: str) -> tuple[int, str] | None:
    if isinstance(owner, LeaseOwner) and owner.pid and owner.started:
        if _proc_started(owner.pid) == owner.started:
            return owner.pid, owner.started
        return None
    match = _OWNER_RE.match(owner)
    if match is None:
        return None
    pid = int(match.group("pid"))
    started = match.group("started")
    if _proc_started(pid) != started:
        return None
    return pid, started


def _claims_process_identity(owner: str) -> bool:
    """Whether *owner* names a specific caller process, which may since have died."""

    if isinstance(owner, LeaseOwner):
        return bool(owner.pid and owner.started)
    return _OWNER_RE.match(owner) is not None


def same_owner_identity(left: str | None, right: str | None) -> bool:
    """True only for the same label and, when bound, the same live caller identity."""
    if left is None or right is None or str(left) != str(right):
        return False
    left_process = _owner_process(left)
    right_process = _owner_process(right)
    if left_process is None and right_process is None:
        return True
    return left_process == right_process


def entry_owned_by(entry: dict[str, Any] | None, owner: str) -> bool:
    """Whether a persisted lease belongs to this exact label and process identity."""

    return bool(entry and _entry_matches_owner(entry, owner))


def _entry_matches_owner(entry: dict[str, Any], owner: str) -> bool:
    if entry.get("owner") != str(owner):
        return False
    # Same label, different agent inside the same process. A lease written before scoping existed
    # carries no scope and cannot be attributed to one sibling over another, so a scoped caller
    # treats it as foreign and routes elsewhere; the orphan ages out with its process. Wasting a
    # device once on upgrade beats two workers sharing one screen.
    if str(entry.get("scope") or "") != _worker_scope():
        return False
    incoming = _owner_process(owner)
    stored_pid = entry.get("owner_pid")
    stored_started = entry.get("owner_started")
    if incoming is None:
        return stored_pid is None and stored_started is None
    if stored_pid is None and stored_started is None:
        return True  # Upgrade a legacy same-label lease to process-bound ownership.
    return incoming == (stored_pid, stored_started)


def read_lease(cache_dir: str | Path, serial: str) -> dict[str, Any] | None:
    """The live lease on *serial*, or None when free/expired.

    Expiry is evaluated here rather than swept elsewhere — that is what makes a crashed
    agent's lease self-healing and a permanent block impossible.
    """
    path = _lease_path(cache_dir, serial)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Missing is the only metadata state that proves the target is free.
        return None
    except json.JSONDecodeError:
        # Atomic writes make corruption unusual, but they cannot prove that no live holder
        # exists (the file may have been damaged after a valid claim). Fail closed until an
        # operator explicitly repairs/removes it instead of silently double-allocating.
        return {
            "serial": serial,
            "owner": "<corrupt lease metadata>",
            "inaccessible": True,
            "corrupt": True,
        }
    except OSError:
        # Unreadable is NOT free. Access control must fail closed: treating an EACCES/EIO
        # blip as "nobody holds this" would let a second owner claim a device that is
        # actively driven. The synthetic entry matches no owner, so acquisition, selection
        # and use-validation all refuse until the metadata can actually be read.
        return {
            "serial": serial,
            "owner": "<unreadable lease metadata>",
            "inaccessible": True,
        }
    if not isinstance(entry, dict) or not entry.get("owner") or not entry.get("serial"):
        return {
            "serial": serial,
            "owner": "<corrupt lease metadata>",
            "inaccessible": True,
            "corrupt": True,
        }
    if _expired(entry):
        return None
    return entry


def _lease_metadata_paths(cache_dir: str | Path) -> list[Path]:
    """Enumerate lease records without turning registry I/O failure into an empty pool."""

    directory = lease_dir(cache_dir)
    try:
        with os.scandir(directory) as entries:
            paths = [
                Path(entry.path)
                for entry in entries
                if entry.name.endswith(".json") and entry.is_file()
            ]
    except FileNotFoundError:
        return []
    return sorted(paths)


def live_leased_serials(cache_dir: str | Path) -> set[str]:
    """Every serial with a live lease in this registry; dead/expired owners drop out.

    Provisioning must treat these targets as occupied even when a transport snapshot
    momentarily omits them: adb briefly losing sight of a foreign worker's emulator must
    never make that emulator's console port look free to allocate.
    """
    out: set[str] = set()
    for path in _lease_metadata_paths(cache_dir):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            out.add(path.stem)
            continue
        except OSError:
            # Fail closed: an entry that cannot be read may be a live holder, and the stem
            # is its (sanitised) serial. Blocking one port beats double-allocating it.
            out.add(path.stem)
            continue
        if (
            not isinstance(entry, dict)
            or not entry.get("owner")
            or not entry.get("serial")
        ):
            out.add(path.stem)
            continue
        if _expired(entry):
            continue
        serial = entry.get("serial")
        if isinstance(serial, str) and serial:
            out.add(serial)
    return out


def _write(path: Path, entry: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(entry, indent=2) + "\n")


def acquire(
    cache_dir: str | Path,
    serial: str,
    *,
    owner: str,
    ttl_s: int = DEFAULT_TTL_S,
    needs: list[str] | None = None,
    app: str | None = None,
    allow_additional: bool = False,
) -> bool:
    """Claim *serial* for *owner*. True when held afterwards (including a sticky re-claim)."""

    with (
        _owner_guard(cache_dir, owner),
        _device_guard(cache_dir, serial),
        _lease_guard(cache_dir, serial),
    ):
        held_entries = held_entries_by(cache_dir, owner)
        existing = [str(entry["serial"]) for entry in held_entries]
        others = [held for held in existing if held != serial]
        if not allow_additional and others:
            return False
        if allow_additional and serial not in existing and any(
            entry.get("role") == "replacement" for entry in held_entries
        ):
            return False
        replacement = bool(allow_additional and others and serial not in existing)
        acquired = _acquire_unlocked(
            cache_dir,
            serial,
            owner=owner,
            ttl_s=ttl_s,
            needs=needs,
            app=app,
            role="replacement" if replacement else "primary",
            replacement_from=others if replacement else None,
        )
        return acquired


def _acquire_unlocked(
    cache_dir: str | Path,
    serial: str,
    *,
    owner: str,
    ttl_s: int = DEFAULT_TTL_S,
    needs: list[str] | None = None,
    app: str | None = None,
    role: str = "primary",
    replacement_from: list[str] | None = None,
) -> bool:
    path = _lease_path(cache_dir, serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "serial": serial,
        "owner": str(owner),
        "generation": secrets.token_hex(16),
        "role": role,
        "acquired": _now(),
        "last_activity": _now(),
        "ttl_s": int(ttl_s),
        "pid": os.getpid(),
        "needs": list(needs or []),
        "app": app,
    }
    scope = _worker_scope()
    if scope:
        entry["scope"] = scope
    if replacement_from:
        entry["replacement_from"] = list(replacement_from)
    owner_process = _owner_process(owner)
    if owner_process is None and _claims_process_identity(owner):
        # The caller names a specific process, and that process is gone (or its pid was
        # reused). Degrading to a TTL lease here would let a daemon keep a dead caller's
        # claim alive on its own clock — process-bound ownership must fail, not soften.
        return False
    if owner_process is not None:
        entry["owner_pid"], entry["owner_started"] = owner_process

    # Fast path: publish a complete entry or no entry at all. Opening the destination with
    # O_EXCL exposed an empty/partial lease to readers when os.write failed, then reported True.
    try:
        atomic_create_text(path, json.dumps(entry, indent=2) + "\n")
    except FileExistsError:
        pass
    except OSError:
        return False  # access control cannot claim success without a durable lease record
    else:
        confirmed = read_lease(cache_dir, serial)
        return bool(
            confirmed
            and _entry_matches_owner(confirmed, owner)
            and confirmed.get("generation") == entry["generation"]
        )

    current = read_lease(cache_dir, serial)
    if current is None:
        # Free or expired — take it over, then re-read to settle a race with another
        # taker. Optimistic on purpose: the loser simply moves to another device.
        entry["acquired"] = _now()
        _write(path, entry)
        confirmed = read_lease(cache_dir, serial)
        return bool(confirmed and _entry_matches_owner(confirmed, owner))
    if _entry_matches_owner(current, owner):
        if pending_handoff(current):
            return False
        current.pop("handoff", None)  # discard an expired, never-accepted offer
        current["last_activity"] = _now()
        current["ttl_s"] = int(ttl_s)
        if owner_process is not None:
            current["owner_pid"], current["owner_started"] = owner_process
        if app:
            current["app"] = app
        if needs:
            current["needs"] = list(needs)
        _write(path, current)
        return True
    return False


def renew(cache_dir: str | Path, serial: str, *, owner: str, app: str | None = None) -> bool:
    """Heartbeat. Called on every command, and from inside long waits."""
    with _lease_guard(cache_dir, serial):
        return _renew_unlocked(cache_dir, serial, owner=owner, app=app)


def _renew_unlocked(
    cache_dir: str | Path, serial: str, *, owner: str, app: str | None = None
) -> bool:
    current = read_lease(cache_dir, serial)
    if current is None or not _entry_matches_owner(current, owner):
        return False
    if pending_handoff(current):
        return False
    current.pop("handoff", None)
    current["last_activity"] = _now()
    if app:
        current["app"] = app
    _write(_lease_path(cache_dir, serial), current)
    return True


def release(cache_dir: str | Path, serial: str, *, owner: str | None = None) -> bool:
    """Drop the lease. A mismatched owner is refused so one agent cannot free another's."""
    with _device_guard(cache_dir, serial), _lease_guard(cache_dir, serial):
        return _release_unlocked(cache_dir, serial, owner=owner)


def _release_unlocked(
    cache_dir: str | Path, serial: str, *, owner: str | None = None
) -> bool:
    current = read_lease(cache_dir, serial)
    if current is not None and owner is not None and not _entry_matches_owner(current, owner):
        return False
    if current is not None and owner is not None and pending_handoff(current):
        return False
    try:
        _lease_path(cache_dir, serial).unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def list_leases(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Live leases only — expired entries are reported as free, never as holders."""
    out: list[dict[str, Any]] = []
    for path in _lease_metadata_paths(cache_dir):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            out.append(
                {
                    "serial": path.stem,
                    "owner": "<corrupt lease metadata>",
                    "inaccessible": True,
                    "corrupt": True,
                }
            )
            continue
        except OSError:
            out.append(
                {
                    "serial": path.stem,
                    "owner": "<unreadable lease metadata>",
                    "inaccessible": True,
                }
            )
            continue
        if not isinstance(entry, dict) or not entry.get("owner") or not entry.get("serial"):
            out.append(
                {
                    "serial": path.stem,
                    "owner": "<corrupt lease metadata>",
                    "inaccessible": True,
                    "corrupt": True,
                }
            )
        elif not _expired(entry):
            out.append(entry)
    return out


def holder(cache_dir: str | Path, serial: str) -> str | None:
    entry = read_lease(cache_dir, serial)
    return str(entry.get("owner")) if entry else None


def held_by(cache_dir: str | Path, owner: str) -> list[str]:
    """Every target this owner holds, including a crash-recoverable replacement reservation."""

    return [str(e["serial"]) for e in held_entries_by(cache_dir, owner)]


def held_entries_by(cache_dir: str | Path, owner: str) -> list[dict[str, Any]]:
    return [e for e in list_leases(cache_dir) if _entry_matches_owner(e, owner)]


def primary_held_by(cache_dir: str | Path, owner: str) -> list[str]:
    """Targets eligible for ordinary bare routing (normally exactly zero or one)."""

    return [
        str(entry["serial"])
        for entry in held_entries_by(cache_dir, owner)
        if entry.get("role") != "replacement"
    ]


def promote_replacement(cache_dir: str | Path, serial: str, *, owner: str) -> bool:
    """Make a reserved replacement the owner's ordinary sticky target."""

    with (
        _owner_guard(cache_dir, owner),
        _device_guard(cache_dir, serial),
        _lease_guard(cache_dir, serial),
    ):
        current = read_lease(cache_dir, serial)
        if current is None or not _entry_matches_owner(current, owner):
            return False
        if any(held != serial for held in primary_held_by(cache_dir, owner)):
            return False
        if current.get("role") != "replacement":
            # Migration from versions that allowed several ordinary leases: after the CLI has
            # cleaned and released every other primary, the selected survivor is already the
            # correct route. Promotion is intentionally idempotent in that final state.
            return current.get("role", "primary") == "primary"
        current["role"] = "primary"
        current.pop("replacement_from", None)
        current["last_activity"] = _now()
        _write(_lease_path(cache_dir, serial), current)
        return True


def _recover_completed_replacement(cache_dir: str | Path, owner: str) -> None:
    """Promote a reserved target when its cleaned predecessor is already gone after a crash."""

    entries = held_entries_by(cache_dir, owner)
    live = {str(entry.get("serial") or "") for entry in entries}
    for entry in entries:
        if entry.get("role") != "replacement":
            continue
        predecessors = {
            str(serial) for serial in (entry.get("replacement_from") or []) if serial
        }
        if predecessors and predecessors.isdisjoint(live):
            promote_replacement(cache_dir, str(entry["serial"]), owner=owner)


def _replacement_call(serial: str | None, needs: list[str] | None = None) -> str:
    prefix = f"aua --needs {','.join(needs)} " if needs else "aua "
    target = f" {serial}" if serial else ""
    return f"{prefix}lease acquire{target} --replace"


def _raise_switch_required(
    *, owner: str, held: list[str], requested: str | None, needs: list[str] | None = None
) -> None:
    from .errors import LeaseSwitchRequiredError

    current = ", ".join(held)
    target = requested or "another compatible device"
    raise LeaseSwitchRequiredError(
        f"{owner} already leases {current}; switching to {target} would release the current lease",
        hint=(
            "No lease was changed. If that handoff is intentional, acknowledge the cleanup and "
            f"release with `{_replacement_call(requested, needs)}`. Ordinary commands should "
            "omit --serial and stay on the existing lease."
        ),
    )


def pending_handoff(entry: dict[str, Any], *, now: float | None = None) -> dict[str, Any] | None:
    """The unexpired one-time handoff reservation embedded in a lease entry."""

    handoff = entry.get("handoff")
    if not isinstance(handoff, dict):
        return None
    now = _now() if now is None else now
    if now >= float(handoff.get("expires") or 0):
        return None
    return handoff


def _raise_handoff_pending(serial: str, entry: dict[str, Any]) -> None:
    from .errors import LeaseHandoffPendingError

    handoff = pending_handoff(entry) or {}
    remaining = max(0, int(float(handoff.get("expires") or 0) - _now()))
    raise LeaseHandoffPendingError(
        f"{serial} is frozen for a lease handoff ({remaining}s remaining)",
        hint=(
            "Let the receiving agent accept its token, or cancel without releasing the device "
            f"with `aua lease cancel-transfer {serial}`."
        ),
    )


def create_handoff(
    cache_dir: str | Path,
    serial: str,
    *,
    owner: str,
    ttl_s: int = HANDOFF_TTL_S,
) -> dict[str, Any]:
    with _owner_guard(cache_dir, owner):
        return _create_handoff_unlocked(
            cache_dir,
            serial,
            owner=owner,
            ttl_s=ttl_s,
        )


def _create_handoff_unlocked(
    cache_dir: str | Path,
    serial: str,
    *,
    owner: str,
    ttl_s: int = HANDOFF_TTL_S,
) -> dict[str, Any]:
    """Create a one-time token and reserve *serial* until acceptance or expiry."""

    from .errors import DeviceLeasedError, UsageError

    owned = held_by(cache_dir, owner)
    other = [held for held in owned if held != serial]
    if other:
        _raise_switch_required(owner=owner, held=other, requested=serial)
    with _device_guard(cache_dir, serial), _lease_guard(cache_dir, serial):
        current = read_lease(cache_dir, serial)
        if current is None or not _entry_matches_owner(current, owner):
            holder_name = str(current.get("owner")) if current else "nobody"
            raise DeviceLeasedError(
                f"{serial} is not leased by {owner}; current holder: {holder_name}",
                hint="Only the current holder can transfer a lease. Run `aua lease list`.",
            )
        if current.get("role") == "replacement":
            _raise_switch_required(owner=owner, held=[serial], requested=serial)
        if not current.get("owner_pid") or not current.get("owner_started"):
            raise UsageError(
                "lease transfer requires process-bound source ownership",
                hint="Reacquire from the live orchestrator process, then create the transfer.",
            )
        if not current.get("generation"):
            current["generation"] = secrets.token_hex(16)

        token = f"aua1_{secrets.token_urlsafe(24)}"
        digest = _handoff_digest(token)
        created = _now()
        current["handoff"] = {
            "version": 1,
            "token_hash": digest,
            "from_owner": str(owner),
            "from_owner_pid": current.get("owner_pid"),
            "from_owner_started": current.get("owner_started"),
            "from_generation": current.get("generation"),
            "created": created,
            "expires": created + max(1, int(ttl_s)),
        }
        _write(_lease_path(cache_dir, serial), current)
        return {
            "serial": serial,
            "token": token,
            "from_owner": str(owner),
            "expires_in_s": max(1, int(ttl_s)),
        }


def accept_handoff(
    cache_dir: str | Path,
    token: str,
    *,
    owner: str,
) -> dict[str, Any]:
    with _owner_guard(cache_dir, owner):
        return _accept_handoff_unlocked(cache_dir, token, owner=owner)


def _accept_handoff_unlocked(
    cache_dir: str | Path,
    token: str,
    *,
    owner: str,
) -> dict[str, Any]:
    """Atomically rebind a reserved lease to the process presenting its one-time token."""

    from .errors import LeaseSwitchRequiredError, UsageError

    clean_token = token.strip()
    if not clean_token:
        raise UsageError("lease accept needs a handoff token")
    digest = _handoff_digest(clean_token)
    matches: list[str] = []
    for path in _lease_metadata_paths(cache_dir):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(candidate, dict):
            continue
        handoff = pending_handoff(candidate)
        if handoff and secrets.compare_digest(str(handoff.get("token_hash") or ""), digest):
            matches.append(str(candidate.get("serial") or ""))
    if len(matches) != 1 or not matches[0]:
        raise UsageError(
            "lease handoff token is invalid, expired, or already used",
            hint="Ask the current holder to run `aua lease transfer <serial>` again.",
        )
    serial = matches[0]

    recipient_process = owner_caller(owner) or {}
    if not recipient_process.get("pid") or not recipient_process.get("started"):
        raise UsageError(
            "lease transfer requires a process-bound receiving agent",
            hint="Run `lease accept` from the spawned agent process, not a TTL-only owner label.",
        )
    previous = [held for held in held_by(cache_dir, owner) if held != serial]
    if previous:
        raise LeaseSwitchRequiredError(
            f"{owner} already leases {', '.join(previous)}; accepting needs a free owner",
            hint=(
                f"No ownership changed and the token remains usable. First run "
                f"`aua lease release {previous[0]}` to clean and release the current device, "
                "then retry the same accept token."
            ),
        )

    with _device_guard(cache_dir, serial), _lease_guard(cache_dir, serial):
        try:
            raw = json.loads(_lease_path(cache_dir, serial).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        handoff = pending_handoff(raw)
        if handoff is None or not secrets.compare_digest(
            str(handoff.get("token_hash") or ""), digest
        ):
            raise UsageError("lease handoff token is invalid, expired, or already used")
        source_process = (handoff.get("from_owner_pid"), handoff.get("from_owner_started"))
        if source_process == (
            recipient_process.get("pid"),
            recipient_process.get("started"),
        ) and source_process != (None, None):
            raise UsageError(
                "the lease handoff must be accepted by a different agent process",
                hint="Pass the token to the spawned agent; the current holder already owns it.",
            )
        if raw.get("generation") != handoff.get("from_generation"):
            raise UsageError("lease handoff token is invalid, expired, or already used")

        raw["owner"] = str(owner)
        raw.pop("owner_pid", None)
        raw.pop("owner_started", None)
        raw.pop("handoff", None)
        raw.pop("replacement_from", None)
        raw["role"] = "primary"
        if recipient_process:
            raw["owner_pid"] = recipient_process.get("pid")
            raw["owner_started"] = recipient_process.get("started")
        raw["acquired"] = _now()
        raw["last_activity"] = _now()
        raw["pid"] = os.getpid()
        raw["generation"] = secrets.token_hex(16)
        _write(_lease_path(cache_dir, serial), raw)
        confirmed = read_lease(cache_dir, serial)
        if confirmed is None or not _entry_matches_owner(confirmed, owner):
            raise UsageError("lease handoff could not be confirmed; retry with a fresh token")

    return {
        "serial": serial,
        "from_owner": handoff.get("from_owner"),
        "owner": str(owner),
        "previous_serials": previous,
    }


def cancel_handoff(cache_dir: str | Path, serial: str, *, owner: str) -> bool:
    """Cancel an offered handoff without releasing or resetting the device."""

    with (
        _owner_guard(cache_dir, owner),
        _device_guard(cache_dir, serial),
        _lease_guard(cache_dir, serial),
    ):
        try:
            current = json.loads(_lease_path(cache_dir, serial).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not _entry_matches_owner(current, owner) or not isinstance(
            current.get("handoff"), dict
        ):
            return False
        current.pop("handoff", None)
        current["last_activity"] = _now()
        _write(_lease_path(cache_dir, serial), current)
        return True


def validate_use(
    cache_dir: str | Path,
    serial: str,
    *,
    owner: str,
    expected_generation: str | None = None,
    renew: bool = True,
) -> str | None:
    """Fence one device command against transfer/release and return its lease generation."""

    from .errors import DeviceLeasedError, LeaseSwitchRequiredError

    with _lease_guard(cache_dir, serial):
        current = read_lease(cache_dir, serial)
        if current is None or not _entry_matches_owner(current, owner):
            holder_name = str(current.get("owner")) if current else "nobody"
            raise DeviceLeasedError(
                f"{serial} is no longer leased by {owner}; current holder: {holder_name}",
                hint="Stop this stale call and resolve ownership with `aua lease list`.",
            )
        if renew and not current.get("generation"):
            current["generation"] = secrets.token_hex(16)
            _write(_lease_path(cache_dir, serial), current)
        if expected_generation is not None and current.get("generation") != expected_generation:
            raise DeviceLeasedError(
                f"{serial} lease generation changed before background work could run",
                hint="Discard this stale background result; the current owner must schedule it again.",
            )
        if pending_handoff(current):
            _raise_handoff_pending(serial, current)
        if current.get("role") == "replacement":
            raise LeaseSwitchRequiredError(
                f"{serial} is only a replacement reservation, not the ordinary target",
                hint=f"Resume with `aua lease acquire {serial} --replace`.",
            )
        if renew:
            if not _renew_unlocked(cache_dir, serial, owner=owner):
                raise DeviceLeasedError(f"could not renew the lease on {serial}")
            current = read_lease(cache_dir, serial)
        return str(current.get("generation")) if current else None


def idle_seconds(entry: dict[str, Any]) -> float:
    return max(0.0, _now() - float(entry.get("last_activity") or 0))


def _lease_lifetime_hint(entries: list[dict[str, Any]], fallback_ttl_s: int) -> str:
    """Describe when these leases actually free without conflating process and TTL owners."""

    pending = [entry for entry in entries if pending_handoff(entry)]
    process_bound = [
        entry
        for entry in entries
        if isinstance(entry.get("owner_pid"), int) and entry not in pending
    ]
    unbound = [entry for entry in entries if entry not in pending and entry not in process_bound]
    facts = ["`aua lease list` shows idle_s and holder process status."]
    if pending:
        facts.append("A pending handoff stays reserved only until its listed transfer expiry.")
    if process_bound:
        facts.append(
            "Process-bound leases do not expire while their owner is alive; they free as soon "
            "as that process exits."
        )
    if unbound:
        ttls = {int(entry.get("ttl_s") or fallback_ttl_s) for entry in unbound}
        if len(ttls) == 1:
            facts.append(f"Each legacy/unbound lease expires after {ttls.pop()}s idle.")
        else:
            facts.append("Each legacy/unbound lease expires after its listed ttl_s idle.")
    return " ".join(facts)


# --------------------------------------------------------------------------- selection


def unmet_needs(capabilities: dict[str, Any] | None, needs: list[str] | None) -> list[str]:
    """Which of *needs* this device cannot satisfy.

    Unknown capability names are reported as unmet rather than ignored: silently satisfying
    ``--needs jellybean`` would hand back a device that does not do what was asked, and the
    caller would only find out several steps later.
    """
    if not needs:
        return []
    caps = capabilities or {}
    return [n for n in needs if not bool(caps.get(n))]


def choose_device(
    cache_dir: str | Path,
    *,
    owner: str,
    explicit: str | None,
    candidates: list[tuple[str, dict[str, Any]]],
    needs: list[str] | None = None,
    ttl_s: int = DEFAULT_TTL_S,
    allow_replacement: bool = False,
) -> tuple[str, str]:
    with _owner_guard(cache_dir, owner):
        return _choose_device_unlocked(
            cache_dir,
            owner=owner,
            explicit=explicit,
            candidates=candidates,
            needs=needs,
            ttl_s=ttl_s,
            allow_replacement=allow_replacement,
        )


def _choose_device_unlocked(
    cache_dir: str | Path,
    *,
    owner: str,
    explicit: str | None,
    candidates: list[tuple[str, dict[str, Any]]],
    needs: list[str] | None = None,
    ttl_s: int = DEFAULT_TTL_S,
    allow_replacement: bool = False,
) -> tuple[str, str]:
    """Pick and claim a device. Returns ``(serial, why)``; raises when nothing is available.

    ``candidates`` is ``[(serial, capabilities)]`` for *online* devices, in preference order.

    Explicit intent is never silently redirected — asking for a specific emulator and being
    moved to another one would quietly invalidate a test pinned to that device's state. So an
    explicit ``--serial`` that is held fails loudly, while an unspecified one is free to route.
    """
    from .errors import DeviceLeasedError

    known = dict(candidates)
    _recover_completed_replacement(cache_dir, owner)
    entries = held_entries_by(cache_dir, owner)
    all_owned = [str(entry["serial"]) for entry in entries]
    replacements = [
        str(entry["serial"]) for entry in entries if entry.get("role") == "replacement"
    ]
    owned = [
        str(entry["serial"]) for entry in entries if entry.get("role") != "replacement"
    ]

    if replacements and not allow_replacement:
        from .errors import LeaseSwitchRequiredError

        raise LeaseSwitchRequiredError(
            f"{owner} has an interrupted replacement reserved on {replacements[0]}",
            hint=(
                "The old device remains the only ordinary target. Resume its cleanup and promote "
                f"the reservation with `aua lease acquire {replacements[0]} --replace`."
            ),
        )
    if allow_replacement and len(replacements) > 1:
        from .errors import LeaseSwitchRequiredError

        raise LeaseSwitchRequiredError(
            f"{owner} has multiple replacement reservations: {', '.join(replacements)}",
            hint="Release the unintended reservations explicitly, then retry one replacement.",
        )

    def _free_report() -> str:
        free = [
            s
            for s, caps in candidates
            if (
                (current := read_lease(cache_dir, s)) is None
                or _entry_matches_owner(current, owner)
            )
            and not unmet_needs(caps, needs)
        ]
        return ", ".join(free) if free else "none"

    if explicit:
        current = read_lease(cache_dir, explicit)
        if current is not None and not _entry_matches_owner(current, owner):
            # This function's contract is that explicit intent is NEVER redirected — but the
            # hint used to open with "free now: <others> — omit --serial to auto-pick", which
            # is a redirect instruction. Agents followed it onto devices they were never
            # assigned, including one a human was actively driving, and two of them then drove
            # the same screen. A caller who named a device must never be handed someone else's;
            # the only safe moves are wait, bring your own, or prove you are the holder.
            lifetime = _lease_lifetime_hint([current], ttl_s)
            hint = (
                f"Do NOT switch devices — you asked for {explicit}, and another agent's screen "
                f"is not a substitute for it. Wait only if that holder is expected to finish. "
                f"{lifetime} Or bring your own with `aua emulator start --headless --parallel` "
                f"and pass "
                f"the serial it returns. You are `{owner}`: pass "
                f"`--owner {current.get('owner')}` only if that holder is you under another "
                f"name, or `aua lease release {explicit} --force` if you know it is a dead run."
            )
            raise DeviceLeasedError(
                f"{explicit} is leased by {current.get('owner')} "
                f"(active {idle_seconds(current):.0f}s ago)",
                hint=hint,
            )
        if current is not None and pending_handoff(current):
            _raise_handoff_pending(explicit, current)
        missing = unmet_needs(known.get(explicit), needs)
        if missing:
            raise DeviceLeasedError(
                f"{explicit} does not satisfy: {', '.join(missing)}",
                hint=f"free and matching: {_free_report()}",
            )
        previous = [serial for serial in all_owned if serial != explicit]
        if previous and not allow_replacement:
            _raise_switch_required(
                owner=owner,
                held=previous,
                requested=explicit,
                needs=needs,
            )
        if not acquire(
            cache_dir,
            explicit,
            owner=owner,
            ttl_s=ttl_s,
            needs=needs,
            allow_additional=allow_replacement,
        ):
            raise DeviceLeasedError(
                f"{explicit} lease changed while it was being selected",
                hint="Run `aua lease list`, then retry or select another free device.",
            )
        return explicit, "replacement_reserved" if previous else "explicit"

    if allow_replacement and replacements:
        reserved = replacements[0]
        if reserved not in known or unmet_needs(known.get(reserved), needs):
            raise DeviceLeasedError(
                f"reserved replacement {reserved} is unavailable or no longer satisfies --needs",
                hint=f"Retry with `aua lease acquire {reserved} --replace` once it is online.",
            )
        if not acquire(
            cache_dir,
            reserved,
            owner=owner,
            ttl_s=ttl_s,
            needs=needs,
            allow_additional=True,
        ):
            raise DeviceLeasedError(f"reserved replacement {reserved} could not be renewed")
        return reserved, "replacement_reserved"

    # Sticky: keep an agent on the device it already knows, so its element ids, app state and
    # learned screen map stay valid across calls.
    if len(owned) > 1 and not allow_replacement:
        from .errors import LeaseSwitchRequiredError

        raise LeaseSwitchRequiredError(
            f"{owner} has multiple legacy leases: {', '.join(owned)}",
            hint=(
                "Choose the one device to keep and acknowledge cleanup of the others with "
                f"`aua lease acquire {owned[0]} --replace`. No lease was changed."
            ),
        )
    for serial in owned:
        if serial in known and not unmet_needs(known[serial], needs):
            current = read_lease(cache_dir, serial)
            if current is not None and pending_handoff(current):
                _raise_handoff_pending(serial, current)
            renew(cache_dir, serial, owner=owner)
            return serial, "sticky"

    missing_owned = [serial for serial in owned if serial not in known]
    if missing_owned:
        from .errors import LeasedTargetUnavailableError

        serials = ", ".join(missing_owned)
        raise LeasedTargetUnavailableError(
            f"the leased target {serials} is temporarily unavailable",
            hint=(
                "The lease was retained and no device action ran. Retry the ordinary unpinned "
                "AUA command; do not select or release another device. If it remains unavailable, "
                "run `aua doctor` to inspect the shared transport."
            ),
        )

    if owned and not allow_replacement:
        _raise_switch_required(owner=owner, held=owned, requested=None, needs=needs)

    for serial, caps in candidates:
        if unmet_needs(caps, needs):
            continue
        if read_lease(cache_dir, serial) is not None:
            continue
        if acquire(
            cache_dir,
            serial,
            owner=owner,
            ttl_s=ttl_s,
            needs=needs,
            allow_additional=allow_replacement,
        ):
            return serial, "assigned"

    busy_entries = [
        e for e in list_leases(cache_dir) if e.get("serial") in known
    ]
    busy = [
        f"{e['serial']} ({e['owner']}, active {idle_seconds(e):.0f}s ago)"
        for e in busy_entries
    ]
    detail = "; ".join(busy) if busy else "no attached device matches"
    # The sibling branch above learned this the hard way; this one kept the old advice and
    # stranded the same caller three more times in one session. "wait" is not actionable
    # without naming the event that frees this kind of lease, and the command that actually frees
    # a device — `lease release`, which takes the holder's `--owner` — went unmentioned.
    holders = sorted(
        {str(e.get("owner")) for e in busy_entries}
    )
    # A daemon leases as its *caller*, so one agent legitimately appears under two owner names
    # and adopting the holder is identity reconciliation, not theft. That distinction has to be
    # spelled out: phrased as "a stale holder's lease is yours to hand back", it read as
    # permission to take any busy device, and agents did.
    adopt = (
        f" If a listed holder is you under another name (a daemon leases as its caller), rerun "
        f"with `--owner {holders[0]}`."
        if holders
        else ""
    )
    raise DeviceLeasedError(
        f"no free device{' matching ' + ','.join(needs) if needs else ''}: {detail}",
        hint=(
            f"Start your own rather than taking one: `aua emulator start --headless --parallel`"
            f"{', or widen --needs' if needs else ''}. "
            f"{_lease_lifetime_hint(busy_entries, ttl_s)}{adopt} Only if you know a "
            f"listed holder is a dead run: `aua lease release <serial> --force`."
        ),
    )


def wait_for_device(
    cache_dir: str | Path,
    *,
    owner: str,
    explicit: str | None,
    candidates: Callable[[], list[tuple[str, dict[str, Any]]]],
    needs: list[str] | None = None,
    ttl_s: int = DEFAULT_TTL_S,
    allow_replacement: bool = False,
    wait_s: float = 0,
    poll_s: float = 0.25,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str, int]:
    """Choose a device, optionally waiting without redirecting explicit intent.

    The candidate supplier is re-run on every poll so newly attached targets and process-death
    lease release are observed.  This helper never releases or rewrites another owner's lease;
    it is only a bounded retry around :func:`choose_device`.
    """

    from .errors import DeviceLeasedError, LeasedTargetUnavailableError

    timeout = max(0.0, float(wait_s))
    started = monotonic()
    deadline = started + timeout
    last_error: DeviceLeasedError | LeasedTargetUnavailableError | None = None
    while True:
        try:
            serial, why = choose_device(
                cache_dir,
                owner=owner,
                explicit=explicit,
                candidates=candidates(),
                needs=needs,
                ttl_s=ttl_s,
                allow_replacement=allow_replacement,
            )
            return serial, why, int(max(0.0, monotonic() - started) * 1000)
        except DeviceLeasedError as exc:
            last_error = exc
        except LeasedTargetUnavailableError as exc:
            last_error = exc
        now = monotonic()
        if timeout <= 0 or now >= deadline:
            assert last_error is not None
            waited_ms = 0 if timeout <= 0 else int(max(0.0, now - started) * 1000)
            target = f" --serial {explicit}" if explicit else ""
            error_type = (
                LeasedTargetUnavailableError
                if isinstance(last_error, LeasedTargetUnavailableError)
                else DeviceLeasedError
            )
            raise error_type(
                f"{last_error.message} after waiting {waited_ms}ms",
                hint=(
                    f"{last_error.hint or ''} Retry the bounded wait with "
                    f"`aua{target} session start --goal <goal> --wait-for-lease "
                    f"{max(1, int(timeout))}`."
                ).strip(),
            ) from last_error
        sleep(min(max(0.01, poll_s), max(0.0, deadline - now)))
