"""Single-attempt reads from an already running Android automation server.

uiautomator2's public RPC path retries by restarting the server. A bounded passive wait
must do neither. Use its existing endpoint through ADB's transport protocol, and shut down
the owned socket at the absolute deadline, including a stalled handshake or trickling body.
"""

from __future__ import annotations

import contextlib
import http.client
import ipaddress
import json
import socket
import threading
from typing import Any

from ..errors import DeviceError
from ..read_budget import ReadBudget, ReadDeadlineExceeded

_READ_METHODS = frozenset(
    {"dumpWindowHierarchy", "takeScreenshot", "deviceInfo", "exist", "objInfo", "waitForIdle"}
)


def rpc(
    host: str,
    port: int,
    serial: str,
    server_port: int,
    method: str,
    params: list[Any],
    budget: ReadBudget,
) -> Any:
    if method not in _READ_METHODS:
        raise DeviceError("operation is not a bounded UI read", code="unsupported_capability")
    # DNS resolution has no portable synchronous deadline. The default ADB address and
    # localhost are numeric; explicitly refuse a hostname instead of hiding an unbounded lookup.
    address = "127.0.0.1" if host == "localhost" else host
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as exc:
        raise DeviceError(
            "bounded waits require a numeric ADB server address", code="unsupported_capability"
        ) from exc
    remaining = budget.remaining()
    sock = socket.socket(socket.AF_INET6 if ip.version == 6 else socket.AF_INET, socket.SOCK_STREAM)

    def interrupt() -> None:
        # This timer owns no device runtime and issues no commands. shutdown wakes a
        # blocked read without closing/reusing the descriptor underneath the caller.
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)

    timer = threading.Timer(remaining, interrupt)
    timer.name = "aua-read-deadline"
    timer.start()
    try:
        sock.settimeout(budget.remaining())
        sock.connect((address, port))

        def receive(size: int) -> bytes:
            data = bytearray()
            while len(data) < size:
                sock.settimeout(budget.remaining())
                chunk = sock.recv(size - len(data))
                if not chunk:
                    budget.check()
                    raise DeviceError("bounded Android read connection closed")
                data.extend(chunk)
            return bytes(data)

        for command in (f"host:transport:{serial}", f"tcp:{server_port}"):
            encoded = command.encode()
            sock.settimeout(budget.remaining())
            sock.sendall(f"{len(encoded):04x}".encode() + encoded)
            status = receive(4)
            if status != b"OKAY":
                raise DeviceError("bounded Android read transport unavailable")
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        ).encode()
        connection = http.client.HTTPConnection(address, server_port)
        connection.sock = sock
        sock.settimeout(budget.remaining())
        connection.request(
            "POST",
            "/jsonrpc/0",
            payload,
            {"Content-Type": "application/json", "Accept-Encoding": "identity"},
        )
        response = connection.getresponse()
        with response:
            raw = response.read(32 * 1024 * 1024 + 1)
        budget.check()
        if response.status != 200 or len(raw) > 32 * 1024 * 1024:
            raise DeviceError("bounded Android read returned an invalid response")
        data = json.loads(raw)
        if not isinstance(data, dict) or "error" in data or "result" not in data:
            raise DeviceError("bounded Android UI read failed; no reconnect attempted")
        return data["result"]
    except (OSError, http.client.HTTPException) as exc:
        if budget.clock() >= budget.deadline:
            raise ReadDeadlineExceeded("Android UI-read deadline reached") from exc
        raise DeviceError(f"bounded Android UI read failed: {exc}") from exc
    finally:
        timer.cancel()
        timer.join()
        sock.close()
