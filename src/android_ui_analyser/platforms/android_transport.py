"""Android host-transport readiness, isolated behind :class:`AndroidPlatform`.

``adbutils`` starts ``adb start-server`` whenever its first socket connection fails. That is
fine for one process and unsafe for a cold burst: every process starts a daemon, the losers fail
to bind the shared smartsocket port, and callers see an empty or reset device list. AUA owns the
coordination here so every Android entry point observes one ready server before third-party code
gets a chance to race its implicit bootstrap.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .. import leases
from ..emulator import adb_bin, ensure_adb_on_path
from ..errors import DeviceError

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_T = TypeVar("_T")


def adb_server_endpoint() -> tuple[str, int]:
    """Return the endpoint used by adbutils and the Android SDK client."""

    host = os.environ.get("ANDROID_ADB_SERVER_HOST", "127.0.0.1")
    raw_port = os.environ.get("ANDROID_ADB_SERVER_PORT", "5037")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise DeviceError(
            f"invalid ANDROID_ADB_SERVER_PORT {raw_port!r}",
            code="adb_server_unavailable",
            hint="Use a numeric port, then run `aua doctor`.",
        ) from exc
    if not 1 <= port <= 65535:
        raise DeviceError(
            f"invalid Android transport port {port}",
            code="adb_server_unavailable",
            hint="Use a port from 1 to 65535, then run `aua doctor`.",
        )
    return host, port


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return b""
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _adb_server_ready(host: str, port: int, *, timeout_s: float = 0.35) -> bool:
    """Probe the ADB protocol rather than treating any listener on the port as ready."""

    request = b"host:version"
    framed = f"{len(request):04x}".encode("ascii") + request
    try:
        with socket.create_connection((host, port), timeout=max(0.05, timeout_s)) as sock:
            sock.settimeout(max(0.05, timeout_s))
            sock.sendall(framed)
            if _recv_exact(sock, 4) != b"OKAY":
                return False
            raw_size = _recv_exact(sock, 4)
            if len(raw_size) != 4:
                return False
            size = int(raw_size, 16)
            return bool(_recv_exact(sock, size))
    except (OSError, ValueError):
        return False


def _start_adb_server(adb: str, *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [adb, "start-server"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _adb_coordination_dir() -> Path:
    """One per-user host location, independent of per-run AUA configuration.

    The ADB endpoint is host-global. Keying its lock below a caller-selected lease registry
    allowed two otherwise-correct AUA configurations to coordinate different files while
    contending for the same TCP port. A fixed directory below the user's cache is both private
    and shared by every AUA configuration that can reach that user's ADB server.
    """

    return Path.home() / ".cache" / "android-ui-analyser" / "host-transactions"


def _transaction_key(host: str, port: int) -> str:
    return f"android-adb-server|{host.casefold()}|{port}"


def _ensure_ready_locked(host: str, port: int, *, startup_timeout_s: float) -> None:
    if _adb_server_ready(host, port):
        return
    if host not in _LOCAL_HOSTS:
        raise DeviceError(
            f"remote Android transport {host}:{port} is unavailable",
            code="adb_server_unavailable",
            hint="Restore the configured remote transport, then run `aua doctor`.",
        )
    try:
        result = _start_adb_server(adb_bin(), timeout_s=startup_timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceError(
            f"could not start the shared Android transport at {host}:{port}: {exc}",
            code="adb_server_unavailable",
            hint="Run `aua doctor`; AUA retained every existing device lease.",
        ) from exc

    deadline = time.monotonic() + min(3.0, max(0.25, startup_timeout_s))
    while time.monotonic() < deadline:
        if _adb_server_ready(host, port):
            return
        time.sleep(0.05)
    detail = (result.stderr or result.stdout or "").strip()
    suffix = f": {detail}" if detail else ""
    raise DeviceError(
        f"shared Android transport did not become ready at {host}:{port}{suffix}",
        code="adb_server_unavailable",
        hint="Run `aua doctor`; do not release or replace the current device lease.",
    )


def ensure_adb_server_ready(
    registry_dir: str | Path,
    *,
    startup_timeout_s: float = 20.0,
) -> None:
    """Ensure one healthy ADB server exists without racing another AUA process.

    The warm path is one bounded protocol probe and takes no lock. On a cold local endpoint,
    callers converge on one cross-process transaction, recheck after acquiring it, and only the
    first process starts the server. A failed ``start-server`` is still accepted when another
    process made the endpoint healthy meanwhile. AUA never kills or restarts a healthy server.
    """

    del registry_dir  # compatibility: transport locking is intentionally config-independent
    ensure_adb_on_path()
    host, port = adb_server_endpoint()
    if _adb_server_ready(host, port):
        return
    with leases.host_transaction(_adb_coordination_dir(), _transaction_key(host, port)):
        _ensure_ready_locked(host, port, startup_timeout_s=startup_timeout_s)


def run_adb_server_operation(
    registry_dir: str | Path | None,
    operation: Callable[[], _T],
    *,
    startup_timeout_s: float = 20.0,
) -> _T:
    """Run one adbutils-backed host operation behind the endpoint's process lock.

    ``adbutils`` may run its own bounded ``adb start-server`` between our readiness probe and
    its socket connection. Keeping the complete operation in the same endpoint transaction
    makes that third-party fallback single-file too: it may recover once, but it cannot race
    another AUA inventory process. Exceptions deliberately propagate as unknown inventory.
    """

    del registry_dir  # compatibility: callers cannot split one endpoint across lock roots
    ensure_adb_on_path()
    host, port = adb_server_endpoint()
    with leases.host_transaction(_adb_coordination_dir(), _transaction_key(host, port)):
        _ensure_ready_locked(host, port, startup_timeout_s=startup_timeout_s)
        return operation()


def run_adb_inventory_operation(
    registry_dir: str | Path | None,
    operation: Callable[[], _T],
    *,
    startup_timeout_s: float = 20.0,
) -> _T:
    """Run read-only inventory with one serialized retry after a transient reset.

    An emulator process can reset the shared transport while AUA holds its own endpoint lock.
    That lock prevents another AUA process from amplifying the reset, but cannot fence the
    emulator's internal ADB client. Inventory is safe to repeat, so stabilize the endpoint and
    retry it exactly once. Device connections and actions deliberately use the non-retrying
    operation above because their outcome may be stateful or unknown.
    """

    del registry_dir  # compatibility: callers cannot split one endpoint across lock roots
    ensure_adb_on_path()
    host, port = adb_server_endpoint()
    with leases.host_transaction(_adb_coordination_dir(), _transaction_key(host, port)):
        _ensure_ready_locked(host, port, startup_timeout_s=startup_timeout_s)
        try:
            return operation()
        except DeviceError as first_error:
            # Require a second successful protocol handshake after the observed failure. The
            # short pause lets an emulator's internal client finish replacing/resetting the
            # server before our final, bounded read-only attempt.
            time.sleep(0.1)
            _ensure_ready_locked(host, port, startup_timeout_s=startup_timeout_s)
            if not _adb_server_ready(host, port):
                _ensure_ready_locked(host, port, startup_timeout_s=startup_timeout_s)
            try:
                return operation()
            except DeviceError as retry_error:
                raise retry_error from first_error


__all__ = [
    "adb_server_endpoint",
    "ensure_adb_server_ready",
    "run_adb_inventory_operation",
    "run_adb_server_operation",
]
