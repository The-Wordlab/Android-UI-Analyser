"""Unix-socket warm daemon for android-ui-analyser (PRD §10).

The daemon holds a warm Engine (connected device + loaded vision models) and
dispatches newline-delimited JSON requests over a AF_UNIX SOCK_STREAM socket.

Protocol
--------
Each connection carries one or more request/response pairs, each on its own line::

    → {"cmd": "analyze", "args": {"source": "auto"}}\n
    ← {"ok": true, "result": {...}}\n

Errors::

    ← {"ok": false, "error": {"code": "...", "message": "...", "hint": "..."}}\n

Supported commands
------------------
ping, analyze, ask_screen, has, inspect, screenshot, tap, long_press, input, clear,
swipe, scroll_to, key, open_link, wait, wait_stable, memory_update, goto,
flow_run, flow_save, navigate, orient, list_devices, app, logcat,
logcat_mark, suite_run
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import AuaError

if TYPE_CHECKING:
    from .config import Config
    from .engine import Engine

logger = logging.getLogger("android_ui_analyser.daemon")

_SOCKET_BACKLOG = 5
_START_POLL_INTERVAL = 0.1  # seconds between is_running checks
_START_TIMEOUT = 5.0  # max seconds to wait after spawning
_LOG_ROLL_BYTES = 8 * 1024 * 1024


class _Activity:
    """Last-request clock, shared by the accept loop and the idle policy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at = time.monotonic()

    def touch(self) -> None:
        with self._lock:
            self._at = time.monotonic()

    def idle_s(self) -> float:
        with self._lock:
            return time.monotonic() - self._at


def _idle_tick(engine: Engine, activity: _Activity) -> bool:
    """Apply the idle policy; return True when the daemon should shut itself down."""
    idle = activity.idle_s()
    pause_after = int(getattr(engine.config.capture, "idle_pause_s", 0) or 0)
    if pause_after > 0 and idle >= pause_after:
        with contextlib.suppress(Exception):
            if engine.capture_idle_pause():
                logger.info("capture paused after %.0fs idle (frames kept)", idle)
    ttl = int(getattr(engine.config.daemon, "idle_ttl_s", 0) or 0)
    return ttl > 0 and idle >= ttl


# --------------------------------------------------------------------------- helpers


def effective_serial(config: Config, serial: str | None = None) -> str | None:
    """The serial this daemon is *for*: explicit arg, ``config.device.serial``, or ``AUA_SERIAL``.

    Single source of truth on purpose. ``socket_path`` used to resolve the fallback chain
    inline while ``start`` gated its ``--serial`` argv on the bare parameter, so
    ``start(config)`` with ``config.device.serial`` set spawned a **serial-less daemon on a
    serial-named socket**. It looked healthy — right socket, `daemon-start` ok — then failed
    every routed call with "multiple devices attached", because the child resolved its device
    with ``connect(None)``. Callers then stopped/restarted the daemon and gave up, losing the
    warm-state amortization the daemon exists to provide.
    """
    return serial or getattr(config.device, "serial", None) or os.environ.get("AUA_SERIAL")


def socket_path(config: Config, serial: str | None = None) -> str:
    """Return the expanded unix-socket path from *config*.

    When a device serial is known (explicit arg, ``config.device.serial``, or
    ``AUA_SERIAL``), append ``.<sanitized-serial>`` so multiple warm daemons can
    coexist — one per emulator. ``AUA_DAEMON_SOCKET`` still wins outright.
    """
    env = os.environ.get("AUA_DAEMON_SOCKET")
    if env:
        return os.path.expanduser(env)
    base = os.path.expanduser(config.daemon.socket)
    ser = effective_serial(config, serial)
    if not ser:
        return base
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ser)
    return f"{base}.{safe}"


# --------------------------------------------------------------------------- dispatch


def _result_ok(value: Any) -> dict[str, Any]:
    return {"ok": True, "result": value}


def _result_err(code: str, message: str, hint: str | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if hint:
        err["hint"] = hint
    return {"ok": False, "error": err}


def _serialize(obj: Any) -> Any:
    """Convert pydantic models and lists thereof to JSON-able types."""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


def dispatch(engine: Engine, request: dict[str, Any]) -> dict[str, Any]:
    """Map a request dict to an Engine call and return a response dict.

    Args:
        engine: The live Engine instance.
        request: ``{"cmd": <str>, "args": {...}}``.

    Returns:
        ``{"ok": True, "result": ...}`` or ``{"ok": False, "error": {...}}``.
    """
    cmd = request.get("cmd", "")
    args: dict[str, Any] = request.get("args") or {}

    try:
        lease_owner = request.get("lease_owner")
        if lease_owner:
            engine.bind_lease_owner(str(lease_owner))

        if cmd == "ping":
            return _result_ok({"pong": True, "version": _aua_version()})

        elif cmd == "analyze":
            result: Any = engine.analyze(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "ask_screen":
            result = engine.ask_screen(**args)
            return _result_ok(result)

        elif cmd == "has":
            result = engine.has(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "inspect":
            result = engine.inspect(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "screenshot":
            result = engine.screenshot(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "tap":
            result = engine.tap(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "long_press":
            result = engine.long_press(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "input":
            result = engine.input_text(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "clear":
            result = engine.clear(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "swipe":
            result = engine.swipe(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "scroll":
            result = engine.scroll(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "scroll_to":
            result = engine.scroll_to(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "expect":
            result = engine.expect(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "await_predicate":
            result = engine.await_predicate(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "target_report":
            # Already a plain dict, not a model — it reports two nodes, not one result.
            return _result_ok(engine.target_report(**args))

        elif cmd == "key":
            result = engine.key(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "double_tap":
            result = engine.double_tap(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "hide_keyboard":
            result = engine.hide_keyboard(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "paste":
            result = engine.paste(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "copy_text":
            result = engine.copy_text(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "erase":
            result = engine.erase(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "clipboard_set":
            result = engine.clipboard_set(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "clipboard_get":
            result = engine.clipboard_get()
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "location_set":
            result = engine.location_set(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "orientation_set":
            result = engine.orientation_set(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "orientation_get":
            result = engine.orientation_get()
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "airplane_set":
            result = engine.airplane_set(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "airplane_toggle":
            result = engine.airplane_toggle()
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "media_add":
            result = engine.media_add(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "record_start":
            result = engine.record_start(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "record_stop":
            result = engine.record_stop(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "clock_set":
            result = engine.clock_set(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "open_link":
            result = engine.open_link(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "resolve":
            result = engine.resolve(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "wait":
            # "for" is a Python keyword; remap "for_" ↔ "for" transparently.
            remapped = {("for_" if k == "for" else k): v for k, v in args.items()}
            result = engine.wait(**remapped)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "wait_stable":
            result = engine.wait_stable(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "wait_changed":
            result = engine.wait_changed(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "hierarchy_fingerprint":
            return _result_ok({"fingerprint": engine.hierarchy_fingerprint()})

        elif cmd == "memory_update":
            # Returns a plain dict (not a pydantic model).
            return _result_ok(engine.memory_update(**args))

        elif cmd == "goto":
            # Memory autopilot — returns a plain dict (route/hops/handoff).
            return _result_ok(engine.goto(**args))

        elif cmd == "flow_run":
            # Whole-journey replay — returns a plain dict (steps_run/handoff).
            return _result_ok(engine.flow_run(**args))

        elif cmd == "flow_save":
            return _result_ok(engine.flow_save(**args))

        elif cmd == "navigate":
            # Autonomous planner-driven navigation — returns a plain dict.
            return _result_ok(engine.navigate(**args))

        elif cmd == "explore_mine":
            return _result_ok(engine.explore_mine(**args))

        elif cmd == "explore_plan":
            return _result_ok(engine.explore_plan(**args))

        elif cmd == "orient":
            # What the tool already knows about the foreground app (plain dict).
            return _result_ok(engine.orient())

        elif cmd == "list_devices":
            devices = engine.list_devices()
            return _result_ok(_serialize(devices))

        elif cmd == "app":
            result = engine.app(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "logcat":
            return _result_ok(engine.logcat(**args))

        elif cmd == "logcat_mark":
            return _result_ok(engine.logcat_mark(**args))

        elif cmd == "suite_run":
            return _result_ok(engine.suite_run(**args))

        elif cmd == "dev_show":
            return _result_ok(engine.dev_show())

        elif cmd == "dev_anim":
            return _result_ok(engine.dev_anim(**args))

        elif cmd == "dev_crashes":
            return _result_ok(engine.dev_crashes(**args))

        elif cmd == "dev_profile":
            return _result_ok(engine.dev_profile(**args))

        elif cmd == "a11y_scroll":
            result = engine.a11y_scroll(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "a11y_action":
            result = engine.a11y_action(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "flags_set":
            return _result_ok(engine.flags_set(**args))

        elif cmd == "flags_apply":
            return _result_ok(engine.flags_apply(**args))

        elif cmd == "proxy_start":
            return _result_ok(engine.proxy_start(**args))

        elif cmd == "proxy_stop":
            return _result_ok(engine.proxy_stop())

        elif cmd == "mock_map":
            return _result_ok(engine.mock_map(**args))

        elif cmd == "mock_record":
            return _result_ok(engine.mock_record(**args))

        elif cmd == "mock_replay":
            return _result_ok(engine.mock_replay(**args))

        elif cmd == "capture_status":
            return _result_ok(engine.capture_status())

        elif cmd == "capture_last":
            return _result_ok(engine.capture_last(**args))

        elif cmd == "capture_export":
            return _result_ok(engine.capture_export(**args))

        elif cmd == "capture_explain":
            return _result_ok(engine.capture_explain(**args))

        elif cmd == "capture_on":
            return _result_ok(engine.capture_on())

        elif cmd == "capture_off":
            return _result_ok(engine.capture_off())

        elif cmd == "capture_prune":
            return _result_ok(engine.capture_prune())

        elif cmd == "capture_start":
            return _result_ok(engine.capture_start())

        elif cmd == "capture_stop":
            return _result_ok(engine.capture_stop())

        else:
            return _result_err(
                "unknown_command",
                f"unknown command: {cmd!r}",
                hint="Valid commands: ping, analyze, has, inspect, screenshot, "
                "tap, long_press, double_tap, input, clear, swipe, scroll, scroll_to, expect, key, "
                "hide_keyboard, paste, copy_text, erase, clipboard_set, clipboard_get, "
                "location_set, orientation_set, orientation_get, airplane_set, airplane_toggle, "
                "media_add, record_start, record_stop, clock_set, open_link, resolve, wait, wait_stable, "
                "memory_update, goto, flow_run, flow_save, navigate, orient, list_devices, app, "
                "logcat, logcat_mark, suite_run, dev_show, dev_anim, dev_crashes, dev_profile, "
                "a11y_scroll, a11y_action, flags_set, flags_apply, proxy_start, proxy_stop, "
                "mock_map, mock_record, mock_replay, capture_status, capture_last, capture_export, "
                "capture_explain, capture_on, capture_off, capture_prune, capture_start, capture_stop",
            )

    except AuaError as exc:
        err = exc.to_dict()["error"]
        return {"ok": False, "error": err}

    except Exception as exc:  # noqa: BLE001 — generic fallback
        logger.exception("unhandled error in dispatch cmd=%r", cmd)
        return _result_err("internal_error", str(exc))


# --------------------------------------------------------------------------- server


def serve(
    engine: Engine,
    sock_path: str,
    *,
    ready_event: threading.Event | None = None,
    _stop_event: threading.Event | None = None,
) -> None:
    """Run the daemon accept loop (blocking) on the unix socket at *sock_path*.

    Unlinks any stale socket file before binding.  Sets *ready_event* once the
    socket is listening.  Removes the socket file on shutdown.

    Args:
        engine: The warm Engine to dispatch requests to.
        sock_path: Path to create the AF_UNIX socket.
        ready_event: If given, set() once the server is listening.
        _stop_event: Internal test hook; stop loop when set.
    """
    # Remove stale socket.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock_path)

    # Ensure parent directory.
    Path(sock_path).parent.mkdir(parents=True, exist_ok=True)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(sock_path)
        srv.listen(_SOCKET_BACKLOG)
        srv.settimeout(0.5)  # non-blocking so we can check _stop_event

        logger.info("daemon listening on %s", sock_path)
        write_pidfile(sock_path + ".pid")  # so `daemon stop` / `daemon reap` can find us
        if engine.config.capture.enabled:
            with contextlib.suppress(Exception):
                engine.capture_start()
                logger.info("capture buffer started")
        push_hub = None
        push_port = int(getattr(engine.config.daemon, "push_ws_port", 0) or 0)
        if push_port > 0:
            from .push import PushHub

            push_hub = PushHub()
            with contextlib.suppress(Exception):
                push_hub.start(push_port)
                push_hub.start_watcher(
                    engine.hierarchy_fingerprint,
                    interval_ms=int(engine.config.daemon.watch_interval_ms),
                    serial=getattr(engine.device, "serial", None)
                    if engine._device is not None
                    else engine.config.device.serial,
                )
                logger.info("push WebSocket on 127.0.0.1:%d", push_port)
        if ready_event is not None:
            ready_event.set()

        activity = _Activity()
        while True:
            if _stop_event is not None and _stop_event.is_set():
                break

            try:
                conn, _ = srv.accept()
            except TimeoutError:
                if _idle_tick(engine, activity):
                    logger.info("daemon idle for %.0fs; shutting down", activity.idle_s())
                    break
                continue

            activity.touch()
            with contextlib.suppress(Exception):
                engine.capture_idle_resume()

            try:
                _handle_connection(engine, conn, activity)
            except Exception:  # noqa: BLE001
                logger.exception("error handling connection")
            finally:
                with contextlib.suppress(OSError):
                    conn.close()

    finally:
        # Release the device + its on-device uiautomator2 server so the UiAutomation slot
        # is free for adb/Maestro after the daemon exits (otherwise it leaks).
        with contextlib.suppress(Exception):
            if "push_hub" in locals() and push_hub is not None:
                push_hub.stop()
        with contextlib.suppress(Exception):
            engine.close()
        srv.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path + ".pid")
        logger.info("daemon stopped, socket removed: %s", sock_path)


def _journal_dispatch(
    engine: Engine,
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    duration_ms: float,
    source: str = "daemon",
) -> None:
    cmd = str(request.get("cmd") or "")
    if request.get("journal") is False:
        return
    if cmd in {"ping", "capture_status"}:
        return
    serial = None
    with contextlib.suppress(Exception):
        serial = getattr(engine.device, "serial", None) if engine._device is not None else None
    if not serial:
        with contextlib.suppress(Exception):
            serial = engine.config.device.serial
    from . import journal as journal_mod

    ok = bool(response.get("ok"))
    journal_mod.record(
        cache_dir=engine.config.cache.dir,
        serial=serial,
        source=source,
        cmd=cmd,
        args=request.get("args") if isinstance(request.get("args"), dict) else {},
        ok=ok,
        duration_ms=duration_ms,
        result=response.get("result") if ok else None,
        error=response.get("error") if not ok else None,
    )


def _handle_connection(
    engine: Engine, conn: socket.socket, activity: _Activity | None = None
) -> None:
    """Read newline-delimited JSON requests and write newline-delimited JSON responses."""
    buf = b""
    conn.settimeout(30.0)

    try:
        while True:
            try:
                chunk = conn.recv(4096)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk

            # Process all complete lines in the buffer.
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if activity is not None:
                    activity.touch()
                t0 = time.monotonic()
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    response = _result_err("parse_error", f"invalid JSON: {exc}")
                    request = {"cmd": "?", "args": {}}
                else:
                    response = dispatch(engine, request)
                with contextlib.suppress(Exception):
                    _journal_dispatch(
                        engine,
                        request if isinstance(request, dict) else {},
                        response,
                        duration_ms=(time.monotonic() - t0) * 1000.0,
                    )

                resp_bytes = json.dumps(response, ensure_ascii=False).encode() + b"\n"
                conn.sendall(resp_bytes)
    except OSError:
        pass  # connection closed by peer


# --------------------------------------------------------------------------- client


class DaemonClient:
    """Minimal client for the daemon unix socket.

    Usage::

        with DaemonClient(sock_path) as client:
            ok = client.ping()
            resp = client.call("analyze", source="auto")
    """

    def __init__(self, sock_path: str, *, timeout: float = 5.0) -> None:
        self._sock_path = sock_path
        self._timeout = timeout

    def __enter__(self) -> DaemonClient:
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # Each call() opens and closes its own connection for simplicity.

    def call(
        self,
        cmd: str,
        *,
        _lease_owner: str | None = None,
        **args: Any,
    ) -> dict[str, Any]:
        """Send one request and return the parsed response dict.

        Does NOT raise on ok=False — caller decides what to do.

        Pass ``journal=False`` to skip the agent I/O journal (dashboard / internal polls).
        """
        journal = args.pop("journal", True)
        request: dict[str, Any] = {"cmd": cmd, "args": args}
        if _lease_owner:
            request["lease_owner"] = _lease_owner
        if journal is False:
            request["journal"] = False
        payload = json.dumps(request, ensure_ascii=False).encode() + b"\n"

        # Long-poll commands need a client timeout above their own deadline.
        timeout = self._timeout
        if cmd in {"wait", "wait_stable", "wait_changed", "goto", "flow_run", "navigate"}:
            ms = args.get("timeout_ms")
            if isinstance(ms, (int, float)) and ms > 0:
                timeout = max(timeout, ms / 1000.0 + 5.0)
            else:
                timeout = max(timeout, 60.0)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(self._sock_path)
            sock.sendall(payload)

            # Read until newline.
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk

            line = buf.split(b"\n", 1)[0]
            return json.loads(line)
        finally:
            with contextlib.suppress(OSError):
                sock.close()

    def ping(self) -> bool:
        """Return True if the daemon responds to a ping, False otherwise."""
        return self.pong_version() is not False

    def pong_version(self) -> str | None | bool:
        """Ping and return the daemon's aua version: a string, ``None`` (old daemon that
        predates version reporting), or ``False`` if the daemon is down/unresponsive."""
        try:
            resp = self.call("ping")
            result = resp.get("result")
            if not resp.get("ok"):
                return False
            if result == "pong":  # pre-version-reporting daemon
                return None
            if isinstance(result, dict) and result.get("pong"):
                return result.get("version")
            return False
        except (OSError, json.JSONDecodeError):
            # A daemon mid-shutdown may accept the connection but send nothing (empty line
            # → JSONDecodeError); treat any non-response as "not running".
            return False


def _source_fingerprint() -> str:
    """Hash of the package's .py bytes, captured ONCE at import.

    A daemon holds its modules in memory, so editing a source file is invisible to it: it
    keeps serving the old code while the version string still matches, and the caller gets
    stale answers with no signal. Computing this at import time means the value describes
    what this process actually LOADED — a CLI started later sees a different fingerprint,
    and the existing version-skew path then does the right thing on its own.

    Content, not mtime: the CLI and the daemon routinely run from two different trees (an
    installed copy vs. the repo), and `install.sh` rewrites mtimes without changing a line.
    An mtime fingerprint therefore reported skew for byte-identical code, which silently
    dropped every call onto the in-process path — a ~6x slowdown announced only in a log line.
    """
    digest = hashlib.blake2b(digest_size=8)
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        with contextlib.suppress(OSError):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


_LOADED_SOURCE = _source_fingerprint()


def _aua_version() -> str:
    from . import __version__

    return f"{__version__}+src{_LOADED_SOURCE}"


# --------------------------------------------------------------------------- lifecycle


def _roll_log(path: Path) -> None:
    """Roll daemon.log once it gets large — it is raw stdout/stderr, so nothing bounds it."""
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > _LOG_ROLL_BYTES:
            path.replace(path.with_name(path.name + ".1"))


def _socket_alive(sock: str) -> bool:
    """Return True if a daemon answers at the explicit socket path *sock*."""
    try:
        with DaemonClient(sock, timeout=2.0) as client:
            return client.ping()
    except OSError:
        return False


def is_running(config: Config) -> bool:
    """Return True if a daemon is live at *config*'s socket path."""
    return _socket_alive(socket_path(config))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_pidfile(path: str | Path) -> None:
    """Record pid + interpreter so :func:`reap` can recognise our own orphans later."""
    with contextlib.suppress(OSError):
        Path(path).write_text(json.dumps({"pid": os.getpid(), "exe": sys.executable}))


def read_pidfile(path: str | Path) -> tuple[int | None, str | None]:
    """Return ``(pid, interpreter)``; interpreter is None for legacy bare-int pidfiles."""
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None, None
    try:
        data = json.loads(raw)
    except ValueError:
        return None, None
    # A legacy pidfile is a bare integer, which is itself valid JSON.
    if isinstance(data, int):
        return data, None
    if not isinstance(data, dict):
        return None, None
    pid = data.get("pid")
    exe = data.get("exe")
    return (int(pid) if isinstance(pid, int) else None), (exe if isinstance(exe, str) else None)


def reap(config: Config) -> dict[str, Any]:
    """Clean up daemons that outlived the session that spawned them.

    Drops pid/socket pairs whose process is gone, and terminates live daemons whose
    interpreter no longer exists — an agent's throwaway venv is deleted long before the
    daemon it spawned notices, and the survivor keeps polling the device forever.

    The interpreter path comes from the pidfile the daemon wrote about itself, not from
    ``ps``: ``comm`` truncates, and a truncated path that fails to resolve would read as an
    orphan and kill a healthy daemon. A legacy pidfile carries no interpreter, so a live
    process behind one is always left alone.
    """
    cache_dir = Path(config.cache.dir).expanduser()
    reaped: list[dict[str, Any]] = []
    for pid_file in sorted(cache_dir.glob("daemon.sock*.pid")):
        sock = str(pid_file)[: -len(".pid")]
        pid, exe = read_pidfile(pid_file)
        if pid is None:
            continue
        alive = _pid_alive(pid)
        orphaned = alive and exe is not None and not Path(exe).exists()
        if alive and not orphaned:
            continue
        if alive:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGTERM)
        for path in (sock, str(pid_file)):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        reaped.append(
            {"pid": pid, "socket": sock, "reason": "orphaned_venv" if alive else "dead"}
        )
    return {"ok": True, "action": "daemon-reap", "reaped": reaped, "count": len(reaped)}


def running_version(config: Config) -> str | None | bool:
    """The live daemon's aua version (string / None if unknown / False if down)."""
    try:
        with DaemonClient(socket_path(config), timeout=2.0) as client:
            return client.pong_version()
    except OSError:
        return False


def start(config: Config, *, serial: str | None = None) -> dict[str, Any]:
    """Start the daemon as a detached background process.

    Returns a dict with keys ``running``, ``pid``, and ``socket``.
    """
    # Resolve once, use for BOTH the socket name and the child's argv. Gating the argv on
    # the bare parameter while the socket fell back to config/env is what produced a
    # serial-less daemon answering on a serial-named socket.
    serial = effective_serial(config, serial)
    sock = socket_path(config, serial=serial)

    # Adopt on the socket we are about to bind, not on the config-derived one: with an
    # explicit --serial those differ, and checking the wrong one spawns a second daemon
    # for a device that already has one.
    if _socket_alive(sock):
        return {"running": True, "pid": None, "socket": sock, "status": "already_running"}

    with contextlib.suppress(Exception):
        reap(config)

    cache_dir = Path(config.cache.dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / "daemon.log"
    _roll_log(log_path)

    cmd = [sys.executable, "-m", "android_ui_analyser.daemon", "--socket", sock]
    if serial:
        cmd += ["--serial", serial]

    log_fh = open(log_path, "a")  # noqa: SIM115 — kept open for child stdout/stderr
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    # Wait until daemon is live or timeout. Poll the socket we spawned, at the poll
    # interval — the old arithmetic slept a flat 5s per turn, so the loop ran once and
    # reported "timeout" for a daemon that had in fact come up, and the caller spawned again.
    deadline = time.monotonic() + _START_TIMEOUT
    while time.monotonic() < deadline:
        if _socket_alive(sock):
            return {"running": True, "pid": proc.pid, "socket": sock, "status": "started"}
        time.sleep(_START_POLL_INTERVAL)

    return {"running": False, "pid": proc.pid, "socket": sock, "status": "timeout"}


def stop(config: Config) -> dict[str, Any]:
    """Stop the daemon by signalling its process, so it runs cleanup on the way out.

    SIGTERM is caught by the daemon and trips its stop-event; the accept loop exits and
    ``serve``'s finally releases the device + on-device uiautomator2 server (freeing the
    UiAutomation slot for adb/Maestro). Falls back to unlinking the socket if no pidfile.
    """
    sock = socket_path(config)
    pid_file = sock + ".pid"
    if not is_running(config):
        for path in (sock, pid_file):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        return {"running": False, "socket": sock, "status": "not_running"}
    pid, _ = read_pidfile(pid_file)
    if pid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0  # let it run engine.close() on the way out
        while time.monotonic() < deadline and is_running(config):
            time.sleep(0.1)
    for path in (sock, pid_file):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
    return {"running": is_running(config), "socket": sock, "status": "stopped"}


def status(config: Config) -> dict[str, Any]:
    """Return a status dict for the daemon."""
    sock = socket_path(config)
    running = is_running(config)
    return {"running": running, "socket": sock}


# --------------------------------------------------------------------------- __main__


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="android-ui-analyser daemon")
    parser.add_argument("--socket", required=True, help="unix socket path")
    parser.add_argument("--serial", default=None, help="device serial")
    ns = parser.parse_args()

    # The parent redirects our stdout/stderr into daemon.log, but nothing ever configured a
    # level — so the lifecycle (startup, idle pause, idle shutdown) was invisible in the one
    # file you would go read to find out why a daemon is gone.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides: dict[str, Any] = {"daemon": {"socket": ns.socket}}
    if ns.serial:
        overrides["device"] = {"serial": ns.serial}
    cfg = load_config(cli_overrides=overrides)

    from .engine import Engine

    eng = Engine(cfg)
    _stop = threading.Event()
    # SIGTERM/SIGINT trip the accept loop's stop-event so serve()'s finally runs (which
    # closes the device + on-device uiautomator2 server — releasing the UiAutomation slot).
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    serve(eng, ns.socket, _stop_event=_stop)
