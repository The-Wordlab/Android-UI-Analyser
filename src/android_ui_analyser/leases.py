"""Per-device leases, so parallel agents stop fighting over the same emulator.

Three emulators attached, Claude testing search, Cursor testing delete — and both land on
``emulator-5554``, because ``connect(serial=None)`` takes "the only/first device". Each
agent then drives a screen the other is mutating. Nothing errors; the results are just
quietly wrong.

The design keeps two properties that matter more than features:

**No deadlock is possible.** Expiry is computed when a lease is *read*, not by a reaper
process. A lease owned by an agent process expires as soon as that process is gone. Friendly
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
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text

# Long enough that a single blocking call cannot outlive it: `--until` waits legitimately run
# 90-120s on a slow backend, and a lease that expires mid-wait would let another agent seize a
# device that is actively in use — strictly worse than no lease at all. Every command renews,
# so the TTL only ever measures the gap *since an agent stopped working*.
DEFAULT_TTL_S = 900

_SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh", "login", "-zsh", "-bash"}

# Wrappers that exist for the duration of one command. Naming an agent after one of these gives it
# a fresh identity per invocation, so it cannot re-acquire the device it just leased — the caller
# above the wrapper is the one that persists.
_LAUNCHERS = {
    "uv", "uvx", "pipx", "poetry", "pdm", "rye", "hatch", "pipenv", "conda", "micromamba",
    "npx", "pnpx", "bunx", "nix", "nix-shell", "direnv", "mise", "asdf",
    "env", "sudo", "doas", "nohup", "stdbuf", "xargs", "time", "timeout", "caffeinate", "arch",
}


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


def _proc_name(pid: int) -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return ""
    return (out.stdout or "").strip().rsplit("/", 1)[-1]


def _proc_ppid(pid: int) -> int | None:
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
    try:
        out = subprocess.run(  # noqa: S603
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return "".join((out.stdout or "").split())[-8:]
    except Exception:
        return ""


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
    return name in _SHELLS or name in _LAUNCHERS or name.startswith("python")


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


def _lease_path(cache_dir: str | Path, serial: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in serial)
    return lease_dir(cache_dir) / f"{safe}.json"


def _now() -> float:
    return time.time()


def _expired(entry: dict[str, Any], *, now: float | None = None) -> bool:
    owner_pid = entry.get("owner_pid")
    owner_started = entry.get("owner_started")
    if isinstance(owner_pid, int) and owner_pid > 1:
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # It exists, but this caller cannot signal it.
        if owner_started and _proc_started(owner_pid) != owner_started:
            return True
    now = _now() if now is None else now
    ttl = float(entry.get("ttl_s") or DEFAULT_TTL_S)
    return (now - float(entry.get("last_activity") or 0)) > ttl


# ``{name}-{pid}-{started}``, where *name* can itself contain "-". Anchoring on the
# digits-only pid and matching greedily makes the last valid pid segment win.
_OWNER_RE = re.compile(r"^(?P<name>.+)-(?P<pid>\d+)-(?P<started>.+)$")


def _owner_process(owner: str) -> tuple[int, str] | None:
    if isinstance(owner, LeaseOwner) and owner.pid and owner.started:
        return owner.pid, owner.started
    match = _OWNER_RE.match(owner)
    if match is None:
        return None
    pid = int(match.group("pid"))
    started = match.group("started")
    if _proc_started(pid) != started:
        return None
    return pid, started


def same_owner_identity(left: str | None, right: str | None) -> bool:
    """True only for the same label and, when bound, the same live caller identity."""
    if left is None or right is None or str(left) != str(right):
        return False
    left_process = _owner_process(left)
    right_process = _owner_process(right)
    if left_process is None and right_process is None:
        return True
    return left_process == right_process


def _entry_matches_owner(entry: dict[str, Any], owner: str) -> bool:
    if entry.get("owner") != str(owner):
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
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict) or _expired(entry):
        return None
    return entry


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
) -> bool:
    """Claim *serial* for *owner*. True when held afterwards (including a sticky re-claim)."""
    path = _lease_path(cache_dir, serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "serial": serial,
        "owner": str(owner),
        "acquired": _now(),
        "last_activity": _now(),
        "ttl_s": int(ttl_s),
        "pid": os.getpid(),
        "needs": list(needs or []),
        "app": app,
    }
    owner_process = _owner_process(owner)
    if owner_process is not None:
        entry["owner_pid"], entry["owner_started"] = owner_process

    # Fast path: nobody has ever held it.
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        pass
    except OSError:
        return True  # cannot coordinate on this filesystem; never block the caller
    else:
        with contextlib.suppress(Exception):
            os.write(fd, (json.dumps(entry, indent=2) + "\n").encode())
        with contextlib.suppress(Exception):
            os.close(fd)
        return True

    current = read_lease(cache_dir, serial)
    if current is None:
        # Free or expired — take it over, then re-read to settle a race with another
        # taker. Optimistic on purpose: the loser simply moves to another device.
        entry["acquired"] = _now()
        _write(path, entry)
        confirmed = read_lease(cache_dir, serial)
        return bool(confirmed and _entry_matches_owner(confirmed, owner))
    if _entry_matches_owner(current, owner):
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
    current = read_lease(cache_dir, serial)
    if current is None or not _entry_matches_owner(current, owner):
        return False
    current["last_activity"] = _now()
    if app:
        current["app"] = app
    _write(_lease_path(cache_dir, serial), current)
    return True


def release(cache_dir: str | Path, serial: str, *, owner: str | None = None) -> bool:
    """Drop the lease. A mismatched owner is refused so one agent cannot free another's."""
    current = read_lease(cache_dir, serial)
    if current is not None and owner is not None and not _entry_matches_owner(current, owner):
        return False
    with contextlib.suppress(OSError):
        _lease_path(cache_dir, serial).unlink()
    return True


def list_leases(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Live leases only — expired entries are reported as free, never as holders."""
    out: list[dict[str, Any]] = []
    for path in sorted(lease_dir(cache_dir).glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(entry, dict) and not _expired(entry):
            out.append(entry)
    return out


def holder(cache_dir: str | Path, serial: str) -> str | None:
    entry = read_lease(cache_dir, serial)
    return str(entry.get("owner")) if entry else None


def held_by(cache_dir: str | Path, owner: str) -> list[str]:
    """Serials this owner already holds — the basis for sticky assignment."""
    return [str(e["serial"]) for e in list_leases(cache_dir) if _entry_matches_owner(e, owner)]


def idle_seconds(entry: dict[str, Any]) -> float:
    return max(0.0, _now() - float(entry.get("last_activity") or 0))


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
) -> tuple[str, str]:
    """Pick and claim a device. Returns ``(serial, why)``; raises when nothing is available.

    ``candidates`` is ``[(serial, capabilities)]`` for *online* devices, in preference order.

    Explicit intent is never silently redirected — asking for a specific emulator and being
    moved to another one would quietly invalidate a test pinned to that device's state. So an
    explicit ``--serial`` that is held fails loudly, while an unspecified one is free to route.
    """
    from .errors import DeviceLeasedError

    known = dict(candidates)

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
            hint = (
                f"Do NOT switch devices — you asked for {explicit}, and another agent's screen "
                f"is not a substitute for it. Either wait (this lease expires after "
                f"{int(current.get('ttl_s') or ttl_s)}s idle; `aua lease list` shows idle_s), "
                f"or bring your own with `aua emulator start --headless --parallel` and pass "
                f"the serial it returns. You are `{owner}`: pass "
                f"`--owner {current.get('owner')}` only if that holder is you under another "
                f"name, or `aua lease release {explicit} --force` if you know it is a dead run."
            )
            raise DeviceLeasedError(
                f"{explicit} is leased by {current.get('owner')} "
                f"(active {idle_seconds(current):.0f}s ago)",
                hint=hint,
            )
        missing = unmet_needs(known.get(explicit), needs)
        if missing:
            raise DeviceLeasedError(
                f"{explicit} does not satisfy: {', '.join(missing)}",
                hint=f"free and matching: {_free_report()}",
            )
        if not acquire(cache_dir, explicit, owner=owner, ttl_s=ttl_s, needs=needs):
            raise DeviceLeasedError(
                f"{explicit} lease changed while it was being selected",
                hint="Run `aua lease list`, then retry or select another free device.",
            )
        return explicit, "explicit"

    # Sticky: keep an agent on the device it already knows, so its element ids, app state and
    # learned screen map stay valid across calls.
    for serial in held_by(cache_dir, owner):
        if serial in known and not unmet_needs(known[serial], needs):
            renew(cache_dir, serial, owner=owner)
            return serial, "sticky"

    for serial, caps in candidates:
        if unmet_needs(caps, needs):
            continue
        if read_lease(cache_dir, serial) is not None:
            continue
        if acquire(cache_dir, serial, owner=owner, ttl_s=ttl_s, needs=needs):
            return serial, "assigned"

    busy = [
        f"{e['serial']} ({e['owner']}, active {idle_seconds(e):.0f}s ago)"
        for e in list_leases(cache_dir)
        if e.get("serial") in known
    ]
    detail = "; ".join(busy) if busy else "no attached device matches"
    # The sibling branch above learned this the hard way; this one kept the old advice and
    # stranded the same caller three more times in one session. "wait" is not actionable
    # without saying how long, and the command that actually frees a device — `lease release`,
    # which takes the holder's `--owner` — went unmentioned, so it stayed invisible.
    holders = sorted(
        {str(e.get("owner")) for e in list_leases(cache_dir) if e.get("serial") in known}
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
            f"{', or widen --needs' if needs else ''}. `aua lease list` shows idle_s and a lease "
            f"expires after {ttl_s}s idle, so waiting also works.{adopt} Only if you know a "
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

    from .errors import DeviceLeasedError

    timeout = max(0.0, float(wait_s))
    started = monotonic()
    deadline = started + timeout
    last_error: DeviceLeasedError | None = None
    while True:
        try:
            serial, why = choose_device(
                cache_dir,
                owner=owner,
                explicit=explicit,
                candidates=candidates(),
                needs=needs,
                ttl_s=ttl_s,
            )
            return serial, why, int(max(0.0, monotonic() - started) * 1000)
        except DeviceLeasedError as exc:
            last_error = exc
        now = monotonic()
        if timeout <= 0 or now >= deadline:
            assert last_error is not None
            waited_ms = int(max(0.0, now - started) * 1000)
            target = f" --serial {explicit}" if explicit else ""
            raise DeviceLeasedError(
                f"{last_error.message} after waiting {waited_ms}ms",
                hint=(
                    f"{last_error.hint or ''} Retry the bounded wait with "
                    f"`aua{target} session start --goal <goal> --wait-for-lease "
                    f"{max(1, int(timeout))}`."
                ).strip(),
            ) from last_error
        sleep(min(max(0.01, poll_s), max(0.0, deadline - now)))
