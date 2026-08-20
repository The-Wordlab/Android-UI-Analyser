"""Smoke tests for the C ``aua-fast`` daemon shim (no real device required)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "aua-fast"
BINARY = NATIVE / "aua-fast"


@pytest.fixture(scope="module")
def aua_fast() -> Path:
    subprocess.run(["make", "-C", str(NATIVE)], check=True)
    assert BINARY.is_file()
    return BINARY


def _serve_once(sock_path: Path, handler) -> threading.Event:
    """Bind a one-shot unix server; return an Event that is set once listening."""
    if sock_path.exists():
        sock_path.unlink()
    ready = threading.Event()

    def run() -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(sock_path))
            srv.listen(1)
            srv.settimeout(5.0)
            ready.set()
            conn, _ = srv.accept()
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                line = buf.split(b"\n", 1)[0]
                req = json.loads(line.decode())
                resp = handler(req)
                conn.sendall((json.dumps(resp) + "\n").encode())
        finally:
            srv.close()
            if sock_path.exists():
                sock_path.unlink()

    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(timeout=2.0), f"mock daemon failed to bind {sock_path}"
    # Brief settle so the accept() is armed after listen().
    time.sleep(0.02)
    return ready


def test_aua_fast_ping(aua_fast: Path) -> None:
    sock = Path(f"/tmp/aua-fast-test-{os.getpid()}-ping.sock")

    def handler(req: dict) -> dict:
        assert req["cmd"] == "ping"
        return {"ok": True, "result": {"pong": True, "version": "test"}}

    _serve_once(sock, handler)
    env = {**os.environ, "AUA_DAEMON_SOCKET": str(sock)}
    proc = subprocess.run(
        [str(aua_fast), "ping"], capture_output=True, text=True, env=env, timeout=5
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    data = json.loads(proc.stdout)
    assert data["pong"] is True


def test_aua_fast_tap_request_shape(aua_fast: Path) -> None:
    sock = Path(f"/tmp/aua-fast-test-{os.getpid()}-tap.sock")
    seen: dict = {}

    def handler(req: dict) -> dict:
        seen.update(req)
        return {"ok": True, "result": {"ok": True, "action": "tap", "id": 4}}

    _serve_once(sock, handler)
    env = {**os.environ, "AUA_DAEMON_SOCKET": str(sock)}
    proc = subprocess.run(
        [str(aua_fast), "tap", "4"], capture_output=True, text=True, env=env, timeout=5
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert seen["cmd"] == "tap"
    assert seen["args"]["element_id"] == 4
    out = json.loads(proc.stdout)
    assert out["action"] == "tap"


def test_aua_fast_has_miss_exit_1(aua_fast: Path) -> None:
    sock = Path(f"/tmp/aua-fast-test-{os.getpid()}-has.sock")

    def handler(req: dict) -> dict:
        return {"ok": True, "result": {"found": False, "source": None}}

    _serve_once(sock, handler)
    env = {**os.environ, "AUA_DAEMON_SOCKET": str(sock)}
    proc = subprocess.run(
        [str(aua_fast), "has", "Nope"], capture_output=True, text=True, env=env, timeout=5
    )
    assert proc.returncode == 1


def test_aua_fast_analyze_default_source(aua_fast: Path) -> None:
    sock = Path(f"/tmp/aua-fast-test-{os.getpid()}-analyze.sock")
    seen: dict = {}

    def handler(req: dict) -> dict:
        seen.update(req)
        return {"ok": True, "result": {"elements": [], "meta": {}}}

    _serve_once(sock, handler)
    env = {**os.environ, "AUA_DAEMON_SOCKET": str(sock)}
    proc = subprocess.run(
        [str(aua_fast), "analyze"], capture_output=True, text=True, env=env, timeout=5
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert seen["cmd"] == "analyze"
    assert seen["args"]["source"] == "auto"
