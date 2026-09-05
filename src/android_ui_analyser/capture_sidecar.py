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
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .capture import CaptureBuffer, CaptureCfgView
from .errors import UsageError
from .platforms.identity import LEGACY_PLATFORM, TargetRef
from .platforms.options_transport import (
    encode_platform_options,
    platform_options_fingerprint,
    read_platform_options_fd,
    scrub_platform_option_environment,
)

logger = logging.getLogger(__name__)


def socket_path(
    cache_dir: Path,
    *,
    serial: str | None = None,
    platform: str = LEGACY_PLATFORM,
) -> str:
    base = Path(cache_dir).expanduser() / "capture.sock"
    if serial is None and str(platform).strip().lower() == LEGACY_PLATFORM:
        return str(base)
    suffix = TargetRef(platform, serial or "unbound").storage_key
    return f"{base}.{suffix}"


def pid_path(
    cache_dir: Path,
    *,
    serial: str | None = None,
    platform: str = LEGACY_PLATFORM,
) -> Path:
    base = Path(cache_dir).expanduser() / "capture_sidecar.pid"
    if serial is None and str(platform).strip().lower() == LEGACY_PLATFORM:
        return base
    suffix = TargetRef(platform, serial or "unbound").storage_key
    return base.with_name(f"{base.name}.{suffix}")


def start(
    *,
    serial: str,
    cache_dir: Path,
    cfg: Any,
    platform: str = "android",
    platform_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ref = TargetRef(platform, serial)
    sock = socket_path(cache_dir, serial=ref.target_id, platform=ref.platform)
    options = dict(platform_options or {})
    options_fingerprint = platform_options_fingerprint(options, key_dir=cache_dir)
    existing = _ping_response(sock)
    if _matches_identity(existing, ref, options_fingerprint):
        return {
            "ok": True,
            "action": "capture-sidecar-start",
            "status": "already_running",
            "socket": sock,
            **ref.to_json(),
        }
    if existing:
        # A sidecar owns a warm adapter/runtime. Reusing it with changed plugin options would
        # silently drive the old endpoint, so retire it before starting the requested runtime.
        with contextlib.suppress(Exception):
            call(sock, "stop")
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock)

    # Before target-scoped sockets, Android had one cache-wide sidecar with no target identity.
    # It cannot be safely adopted: requesting target B could otherwise reuse target A. Retire
    # unknown legacy state and start the requested scoped worker.
    legacy_sock = socket_path(cache_dir)
    if ref.platform == LEGACY_PLATFORM and legacy_sock != sock:
        legacy = _ping_response(legacy_sock)
        if legacy is not None:
            legacy_matches = _matches_identity(legacy, ref, options_fingerprint)
            if legacy_matches:
                return {
                    "ok": True,
                    "action": "capture-sidecar-start",
                    "status": "already_running",
                    "socket": legacy_sock,
                    "legacy_socket": True,
                    **ref.to_json(),
                }
            with contextlib.suppress(Exception):
                call(legacy_sock, "stop")
            with contextlib.suppress(FileNotFoundError):
                os.unlink(legacy_sock)
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / "capture_sidecar.log"
    with tempfile.TemporaryFile(mode="w+b") as options_file:
        options_file.write(encode_platform_options(options))
        options_file.seek(0)
        options_fd = options_file.fileno()
        cmd = [
            sys.executable,
            "-m",
            "android_ui_analyser.capture_sidecar",
            "--socket",
            sock,
            "--serial",
            serial,
            "--platform",
            ref.platform,
            "--platform-options-fd",
            str(options_fd),
            "--platform-options-fingerprint",
            options_fingerprint,
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
                pass_fds=(options_fd,),
                env=scrub_platform_option_environment(),
            )
    pid_path(cache_dir, serial=ref.target_id, platform=ref.platform).write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "options_fingerprint": options_fingerprint,
                **ref.to_json(),
            }
        ),
        encoding="utf-8",
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _matches_identity(_ping_response(sock), ref, options_fingerprint):
            return {
                "ok": True,
                "action": "capture-sidecar-start",
                "status": "started",
                "pid": proc.pid,
                "socket": sock,
                **ref.to_json(),
            }
        time.sleep(0.1)
    return {
        "ok": False,
        "action": "capture-sidecar-start",
        "status": "timeout",
        "pid": proc.pid,
        "socket": sock,
        **ref.to_json(),
        "hint": f"See {log}",
    }


def stop(
    cache_dir: Path,
    *,
    serial: str | None = None,
    platform: str = LEGACY_PLATFORM,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser()
    sock = socket_path(cache_dir, serial=serial, platform=platform)
    stopped_scoped = _ping(sock)
    with contextlib.suppress(Exception):
        call(sock, "stop")
    pp = pid_path(cache_dir, serial=serial, platform=platform)
    if pp.is_file():
        try:
            raw = pp.read_text(encoding="utf-8").strip()
            try:
                metadata = json.loads(raw)
            except ValueError:
                metadata = raw
            pid_value = metadata.get("pid") if isinstance(metadata, dict) else metadata
            pid = int(str(pid_value))
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGTERM)
        except (TypeError, ValueError):
            pass
        pp.unlink(missing_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock)
    # A pre-foundation Android sidecar used the unscoped path. Keep explicit stop capable of
    # retiring it, while every new start writes only the TargetRef-scoped path above.
    legacy_sock = socket_path(cache_dir)
    if (
        str(platform).strip().lower() == LEGACY_PLATFORM
        and legacy_sock != sock
        and not stopped_scoped
    ):
        with contextlib.suppress(Exception):
            call(legacy_sock, "stop")
        legacy_pid = pid_path(cache_dir)
        if legacy_pid.is_file():
            try:
                pid = int(legacy_pid.read_text(encoding="utf-8").strip())
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.kill(pid, signal.SIGTERM)
            except ValueError:
                pass
            legacy_pid.unlink(missing_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(legacy_sock)
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
    return _ping_response(sock) is not None


def _ping_response(sock: str) -> dict[str, Any] | None:
    try:
        out = call(sock, "ping")
        return out if out.get("ok") else None
    except Exception:
        return None


def _matches_identity(
    response: Mapping[str, Any] | None,
    ref: TargetRef,
    options_fingerprint: str,
) -> bool:
    """Whether a live worker is exactly the target/adapter instance requested."""

    return bool(
        response
        and response.get("platform") == ref.platform
        and response.get("target_id") == ref.target_id
        and response.get("options_fingerprint") == options_fingerprint
    )


def serve(
    sock: str,
    *,
    serial: str,
    cache_dir: Path,
    view: CaptureCfgView,
    platform: str = "android",
    platform_options: Mapping[str, Any] | None = None,
    options_fingerprint: str | None = None,
) -> None:
    from .config import load_config
    from .platforms import PlatformFactory

    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock)
    Path(sock).parent.mkdir(parents=True, exist_ok=True)
    platform_name = str(platform).strip().lower()
    config = load_config(
        cli_overrides={"device": {"platform": platform_name, "serial": serial}}
    )
    if platform_options is not None:
        # Replacement is essential: the detached process may discover unrelated cwd/user
        # options, while the parent already selected the exact mapping for this worker.
        config.platforms[platform_name] = dict(platform_options)
    adapter = PlatformFactory(config).create()
    device = adapter.validate_runtime(adapter.connect(serial))
    screenshots = adapter.adapter_capability("ui.screenshot")
    buf = CaptureBuffer(
        root=Path(cache_dir) / "captures",
        serial=serial,
        cfg=view,
        screenshot=lambda: screenshots.capture_screenshot(device),
        platform=platform_name,
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
                resp: dict[str, Any] = {
                    "ok": True,
                    "pong": True,
                    "platform": platform_name,
                    "target_id": serial,
                    "options_fingerprint": options_fingerprint,
                }
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
                resp = buf.explain_local(seconds=req.get("seconds"), since_ms=req.get("since_ms"))
            elif cmd == "stop":
                stop_ev.set()
                resp = {"ok": True, "action": "stop"}
            else:
                resp = {"ok": False, "error": f"unknown cmd {cmd!r}"}
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))
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


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="aua capture sidecar")
    p.add_argument("--socket", required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--platform", default="android")
    p.add_argument("--platform-options-fd", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--platform-options-fingerprint",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--idle-fps", type=float, default=2.0)
    p.add_argument("--burst-fps", type=float, default=10.0)
    p.add_argument("--burst-ms", type=int, default=1500)
    ns = p.parse_args(argv)
    transported_options: dict[str, Any] | None = None
    if ns.platform_options_fd is not None:
        try:
            transported_options = read_platform_options_fd(
                ns.platform_options_fd, consumer="capture sidecar"
            )
        except UsageError as exc:
            p.error(str(exc))
    if transported_options is not None:
        actual_fingerprint = platform_options_fingerprint(
            transported_options,
            key_dir=ns.cache,
        )
        if ns.platform_options_fingerprint != actual_fingerprint:
            p.error("capture sidecar platform-option identity mismatch")
    view = CaptureCfgView(
        idle_fps=ns.idle_fps,
        burst_fps=ns.burst_fps,
        burst_ms=ns.burst_ms,
        extend_burst_on_change=True,
    )
    serve(
        ns.socket,
        serial=ns.serial,
        cache_dir=Path(ns.cache),
        view=view,
        platform=ns.platform,
        platform_options=transported_options,
        options_fingerprint=ns.platform_options_fingerprint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
