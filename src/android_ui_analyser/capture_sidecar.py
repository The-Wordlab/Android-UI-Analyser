"""Host-side capture sidecar — keeps the ring buffer without the full warm daemon.

Survives one-shot CLI exits. Protocol: newline-delimited JSON over a unix socket under
``cache.dir/capture.sock``. Prefer the full daemon when available; this is the fallback
when ``capture.sidecar`` is true.
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
from typing import Any

from .capture import CaptureBuffer, CaptureCfgView
from .device import connect
from .errors import UsageError

logger = logging.getLogger(__name__)


def socket_path(cache_dir: Path) -> str:
    return str(Path(cache_dir).expanduser() / "capture.sock")


def pid_path(cache_dir: Path) -> Path:
    return Path(cache_dir).expanduser() / "capture_sidecar.pid"


def start(*, serial: str, cache_dir: Path, cfg: Any) -> dict[str, Any]:
    sock = socket_path(cache_dir)
    if _ping(sock):
        return {"ok": True, "action": "capture-sidecar-start", "status": "already_running", "socket": sock}
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / "capture_sidecar.log"
    cmd = [
        sys.executable,
        "-m",
        "android_ui_analyser.capture_sidecar",
        "--socket",
        sock,
        "--serial",
        serial,
        "--cache",
        str(cache_dir),
        "--idle-fps",
        str(cfg.idle_fps),
        "--burst-fps",
        str(cfg.burst_fps),
        "--burst-ms",
        str(cfg.burst_ms),
    ]
    with open(log, "a") as fh:  # noqa: SIM115
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=fh,
            start_new_session=True,
            close_fds=True,
        )
    pid_path(cache_dir).write_text(str(proc.pid), encoding="utf-8")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _ping(sock):
            return {
                "ok": True,
                "action": "capture-sidecar-start",
                "status": "started",
                "pid": proc.pid,
                "socket": sock,
            }
        time.sleep(0.1)
    return {
        "ok": False,
        "action": "capture-sidecar-start",
        "status": "timeout",
        "pid": proc.pid,
        "socket": sock,
        "hint": f"See {log}",
    }


def stop(cache_dir: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser()
    sock = socket_path(cache_dir)
    with contextlib.suppress(Exception):
        call(sock, "stop")
    pp = pid_path(cache_dir)
    if pp.is_file():
        try:
            pid = int(pp.read_text(encoding="utf-8").strip())
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGTERM)
        except ValueError:
            pass
        pp.unlink(missing_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock)
    return {"ok": True, "action": "capture-sidecar-stop", "running": False}


def call(sock: str, cmd: str, **args: Any) -> dict[str, Any]:
    payload = json.dumps({"cmd": cmd, **args}, ensure_ascii=False) + "\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10.0)
        s.connect(sock)
        s.sendall(payload.encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    if not buf:
        raise UsageError("capture sidecar returned empty response")
    return json.loads(buf.decode("utf-8"))


def _ping(sock: str) -> bool:
    try:
        out = call(sock, "ping")
        return bool(out.get("ok"))
    except Exception:
        return False


def serve(sock: str, *, serial: str, cache_dir: Path, view: CaptureCfgView) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock)
    Path(sock).parent.mkdir(parents=True, exist_ok=True)
    device = connect(serial)
    buf = CaptureBuffer(
        root=Path(cache_dir) / "captures",
        serial=serial,
        cfg=view,
        screenshot=device.screenshot,
    )
    buf.start()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock)
    srv.listen(8)
    srv.settimeout(0.5)
    stop_ev = threading.Event()

    def handle(conn: socket.socket) -> None:
        try:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            req = json.loads(data.decode("utf-8") or "{}")
            cmd = req.get("cmd")
            if cmd == "ping":
                resp: dict[str, Any] = {"ok": True, "pong": True}
            elif cmd == "status":
                resp = buf.status()
            elif cmd == "last":
                resp = buf.last(
                    seconds=req.get("seconds"),
                    since_ms=req.get("since_ms"),
                    region=req.get("region"),
                )
            elif cmd == "mark":
                buf.mark(str(req.get("action") or "action"))
                resp = {"ok": True, "action": "capture-mark"}
            elif cmd == "on":
                buf.resume()
                resp = buf.status()
            elif cmd == "off":
                buf.pause()
                resp = buf.status()
            elif cmd == "prune":
                resp = buf.prune()
            elif cmd == "export":
                resp = buf.export(
                    req["path"],
                    seconds=req.get("seconds"),
                    since_ms=req.get("since_ms"),
                    fmt=req.get("fmt") or "gif",
                    fps=float(req.get("fps") or 8),
                )
            elif cmd == "explain":
                resp = buf.explain_local(
                    seconds=req.get("seconds"), since_ms=req.get("since_ms")
                )
            elif cmd == "stop":
                stop_ev.set()
                resp = {"ok": True, "action": "stop"}
            else:
                resp = {"ok": False, "error": f"unknown cmd {cmd!r}"}
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                conn.sendall(
                    (json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8")
                )
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    try:
        while not stop_ev.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            handle(conn)
    finally:
        buf.stop()
        with contextlib.suppress(Exception):
            device.close()
        srv.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="aua capture sidecar")
    p.add_argument("--socket", required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--idle-fps", type=float, default=2.0)
    p.add_argument("--burst-fps", type=float, default=10.0)
    p.add_argument("--burst-ms", type=int, default=1500)
    ns = p.parse_args()
    view = CaptureCfgView(
        idle_fps=ns.idle_fps,
        burst_fps=ns.burst_fps,
        burst_ms=ns.burst_ms,
        extend_burst_on_change=True,
    )
    serve(ns.socket, serial=ns.serial, cache_dir=Path(ns.cache), view=view)
