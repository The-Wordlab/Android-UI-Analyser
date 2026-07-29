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
ping, analyze, has, inspect, screenshot, tap, long_press, input, clear,
swipe, scroll_to, key, open_link, wait, wait_stable, memory_update, goto,
flow_run, flow_save, navigate, orient, list_devices, app, logcat,
logcat_mark, suite_run
"""

from __future__ import annotations

import contextlib
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


# --------------------------------------------------------------------------- helpers


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
    ser = serial or getattr(config.device, "serial", None) or os.environ.get("AUA_SERIAL")
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
        if cmd == "ping":
            return _result_ok({"pong": True, "version": _aua_version()})

        elif cmd == "analyze":
            result: Any = engine.analyze(**args)
            return _result_ok(result.model_dump(mode="json"))

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
        with contextlib.suppress(OSError):  # pidfile so `daemon stop` can signal this process
            Path(sock_path + ".pid").write_text(str(os.getpid()))
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

        while True:
            if _stop_event is not None and _stop_event.is_set():
                break

            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue

            try:
                _handle_connection(engine, conn)
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


def _handle_connection(engine: Engine, conn: socket.socket) -> None:
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
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    response = _result_err("parse_error", f"invalid JSON: {exc}")
                else:
                    response = dispatch(engine, request)

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

    def call(self, cmd: str, **args: Any) -> dict[str, Any]:
        """Send one request and return the parsed response dict.

        Does NOT raise on ok=False — caller decides what to do.
        """
        request = {"cmd": cmd, "args": args}
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
    """Newest .py mtime in the package, captured ONCE at import.

    A daemon holds its modules in memory, so editing a source file is invisible to it: it
    keeps serving the old code while the version string still matches, and the caller gets
    stale answers with no signal. Computing this at import time means the value describes
    what this process actually LOADED — a CLI started later sees a newer fingerprint, and
    the existing version-skew path then does the right thing on its own.
    """
    newest = 0.0
    for path in Path(__file__).resolve().parent.rglob("*.py"):
        with contextlib.suppress(OSError):
            newest = max(newest, path.stat().st_mtime)
    return f"{newest:.0f}"


_LOADED_SOURCE = _source_fingerprint()


def _aua_version() -> str:
    from . import __version__

    return f"{__version__}+src{_LOADED_SOURCE}"


# --------------------------------------------------------------------------- lifecycle


def is_running(config: Config) -> bool:
    """Return True if a daemon is live at *config*'s socket path."""
    try:
        with DaemonClient(socket_path(config), timeout=2.0) as client:
            return client.ping()
    except OSError:
        return False


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
    sock = socket_path(config, serial=serial)

    if is_running(config):
        return {"running": True, "pid": None, "socket": sock, "status": "already_running"}

    cache_dir = Path(config.cache.dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / "daemon.log"

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

    # Wait until daemon is live or timeout.
    deadline = time.monotonic() + _START_TIMEOUT
    while time.monotonic() < deadline:
        if is_running(config):
            return {"running": True, "pid": proc.pid, "socket": sock, "status": "started"}
        time.sleep(_SOCKET_BACKLOG * _START_POLL_INTERVAL / _START_POLL_INTERVAL)
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
    try:
        pid = int(Path(pid_file).read_text().strip())
    except (OSError, ValueError):
        pid = None
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
