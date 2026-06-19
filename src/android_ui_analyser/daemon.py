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
swipe, scroll_to, key, wait, list_devices, app
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
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


def socket_path(config: Config) -> str:
    """Return the expanded unix-socket path from *config*."""
    return os.path.expanduser(config.daemon.socket)


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
            return _result_ok("pong")

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

        elif cmd == "scroll_to":
            result = engine.scroll_to(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "key":
            result = engine.key(**args)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "wait":
            # "for" is a Python keyword; remap "for_" ↔ "for" transparently.
            remapped = {("for_" if k == "for" else k): v for k, v in args.items()}
            result = engine.wait(**remapped)
            return _result_ok(result.model_dump(mode="json"))

        elif cmd == "list_devices":
            devices = engine.list_devices()
            return _result_ok(_serialize(devices))

        elif cmd == "app":
            result = engine.app(**args)
            return _result_ok(result.model_dump(mode="json"))

        else:
            return _result_err(
                "unknown_command",
                f"unknown command: {cmd!r}",
                hint="Valid commands: ping, analyze, has, inspect, screenshot, "
                "tap, long_press, input, clear, swipe, scroll_to, key, wait, "
                "list_devices, app",
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
        srv.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)
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

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
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
        try:
            resp = self.call("ping")
            return bool(resp.get("ok")) and resp.get("result") == "pong"
        except OSError:
            return False


# --------------------------------------------------------------------------- lifecycle


def is_running(config: Config) -> bool:
    """Return True if a daemon is live at *config*'s socket path."""
    try:
        with DaemonClient(socket_path(config), timeout=2.0) as client:
            return client.ping()
    except OSError:
        return False


def start(config: Config, *, serial: str | None = None) -> dict[str, Any]:
    """Start the daemon as a detached background process.

    Returns a dict with keys ``running``, ``pid``, and ``socket``.
    """
    sock = socket_path(config)

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
    """Ask the daemon to stop by sending a stop request (best-effort)."""
    sock = socket_path(config)
    if not is_running(config):
        return {"running": False, "socket": sock, "status": "not_running"}
    # We don't have a "shutdown" command wired into dispatch; just remove the socket
    # so new connections fail, then wait briefly.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock)
    # Give the daemon's accept loop a moment to notice the socket is gone.
    time.sleep(0.2)
    return {"running": False, "socket": sock, "status": "stopped"}


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
    serve(eng, ns.socket)
