"""Localhost WebSocket push of screen-changed events (phase 4).

Stdlib-only RFC6455 server (text frames). The daemon watches the hierarchy
fingerprint on an interval and broadcasts::

    {"event":"screen_changed","fingerprint":"...","serial":"...","ts":...}

Clients: ``websocat ws://127.0.0.1:<port>/`` or any WS library.
Security: bind 127.0.0.1 only — same threat model as the unix daemon socket.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import select
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("android_ui_analyser.push")

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept(key: str) -> str:
    digest = hashlib.sha1((key + _GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _handshake(conn: socket.socket) -> bool:
    data = b""
    conn.settimeout(5.0)
    try:
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                return False
            data += chunk
            if len(data) > 16_384:
                return False
    except OSError:
        return False
    head = data.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1", errors="replace")
    key = None
    for line in head.split("\r\n")[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip().lower() == "sec-websocket-key":
            key = v.strip()
            break
    if not key:
        return False
    accept = _ws_accept(key)
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    )
    try:
        conn.sendall(resp.encode())
    except OSError:
        return False
    return True


def _encode_text(msg: str) -> bytes:
    payload = msg.encode()
    n = len(payload)
    if n < 126:
        header = struct.pack("!BB", 0x81, n)
    elif n < 65536:
        header = struct.pack("!BBH", 0x81, 126, n)
    else:
        header = struct.pack("!BBQ", 0x81, 127, n)
    return header + payload


class PushHub:
    """Fan-out WebSocket hub + optional fingerprint watcher thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[socket.socket] = []
        self._stop = threading.Event()
        self._serve_thread: threading.Thread | None = None
        self._watch_thread: threading.Thread | None = None
        self._listen: socket.socket | None = None
        self.port = 0

    def start(self, port: int, *, host: str = "127.0.0.1") -> None:
        if port <= 0:
            return
        if self._serve_thread and self._serve_thread.is_alive():
            return
        self.port = port
        self._stop.clear()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        srv.settimeout(0.5)
        self._listen = srv

        def accept_loop() -> None:
            while not self._stop.is_set():
                try:
                    conn, _addr = srv.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not _handshake(conn):
                    conn.close()
                    continue
                conn.setblocking(False)
                with self._lock:
                    self._clients.append(conn)
                logger.info("push ws client connected (%d total)", len(self._clients))

        self._serve_thread = threading.Thread(target=accept_loop, name="aua-push-ws", daemon=True)
        self._serve_thread.start()

    def start_watcher(
        self,
        fingerprint_fn: Callable[[], str | None],
        *,
        interval_ms: int = 150,
        serial: str | None = None,
    ) -> None:
        if self._watch_thread and self._watch_thread.is_alive():
            return

        def loop() -> None:
            last: str | None = None
            while not self._stop.is_set():
                try:
                    fp = fingerprint_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("push watcher fingerprint failed: %s", exc)
                    fp = None
                if fp and last is not None and fp != last:
                    self.broadcast(
                        {
                            "event": "screen_changed",
                            "fingerprint": fp,
                            "serial": serial,
                            "ts": time.time(),
                        }
                    )
                if fp:
                    last = fp
                self._stop.wait(max(0.05, interval_ms / 1000.0))

        self._watch_thread = threading.Thread(target=loop, name="aua-push-watch", daemon=True)
        self._watch_thread.start()

    def broadcast(self, event: dict[str, Any]) -> None:
        payload = _encode_text(json.dumps(event, ensure_ascii=False))
        dead: list[socket.socket] = []
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                readable, _, _ = select.select([c], [], [], 0)
                if readable:
                    peek = c.recv(1024)
                    if not peek:
                        dead.append(c)
                        continue
                c.sendall(payload)
            except OSError:
                dead.append(c)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]
            for c in dead:
                with contextlib.suppress(OSError):
                    c.close()

    def stop(self) -> None:
        self._stop.set()
        if self._listen is not None:
            with contextlib.suppress(OSError):
                self._listen.close()
            self._listen = None
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for c in clients:
            with contextlib.suppress(OSError):
                c.close()
