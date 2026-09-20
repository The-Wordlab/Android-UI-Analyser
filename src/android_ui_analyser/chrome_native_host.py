"""Chrome native-messaging host for the AUA existing-tab extension.

Chrome starts this process.  It intentionally contains no browser automation logic: it only
authenticates to the user-private AUA Unix socket and forwards framed JSON in both directions.
Nothing except native-messaging frames is ever written to stdout.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO

from .platforms.chrome_extension import (
    BRIDGE_PROTOCOL,
    bridge_config_path,
)


def read_native_message(stream: BinaryIO) -> dict[str, Any] | None:
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise EOFError("truncated native-messaging header")
    length = struct.unpack("=I", header)[0]
    if length > 16 * 1024 * 1024:
        raise ValueError("native-messaging frame is too large")
    payload = stream.read(length)
    if len(payload) != length:
        raise EOFError("truncated native-messaging frame")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("native-messaging frame must be a JSON object")
    return value


def write_native_message(stream: BinaryIO, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("=I", len(payload)))
    stream.write(payload)
    stream.flush()


def _load_bridge_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("bridge config must be a JSON object")
    if int(value.get("protocol", 0)) != BRIDGE_PROTOCOL:
        raise ValueError("bridge protocol mismatch")
    if not value.get("socket") or not value.get("token"):
        raise ValueError("bridge config is incomplete")
    return value


def _socket_to_chrome(peer: socket.socket, output: BinaryIO, lock: threading.Lock) -> None:
    stream = peer.makefile("rb")
    try:
        while True:
            raw = stream.readline()
            if not raw:
                return
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                continue
            with lock:
                write_native_message(output, value)
    except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return


def main() -> int:
    try:
        config = _load_bridge_config(bridge_config_path())
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.connect(str(config["socket"]))
        peer.sendall(
            (
                json.dumps(
                    {
                        "type": "host_hello",
                        "protocol": BRIDGE_PROTOCOL,
                        "token": str(config["token"]),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
        # connectNative() returns a Port before Chrome knows whether this process reached AUA.
        # Confirm the authenticated socket first so the extension never attaches a debugger that
        # is immediately torn down by a late native-host startup failure.
        write_native_message(
            sys.stdout.buffer,
            {"type": "host_ready", "protocol": BRIDGE_PROTOCOL},
        )
    except Exception as exc:
        print(f"AUA Chrome bridge unavailable: {exc}", file=sys.stderr)
        return 1

    output_lock = threading.Lock()
    reader = threading.Thread(
        target=_socket_to_chrome,
        args=(peer, sys.stdout.buffer, output_lock),
        name="aua-chrome-native-reader",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            message = read_native_message(sys.stdin.buffer)
            if message is None:
                break
            peer.sendall(
                (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
            )
    except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"AUA Chrome bridge stopped: {exc}", file=sys.stderr)
    finally:
        with contextlib.suppress(OSError):
            peer.shutdown(socket.SHUT_RDWR)
        peer.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - Chrome is the entry point
    raise SystemExit(main())
