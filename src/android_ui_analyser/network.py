"""Deterministic, reversible Android network controls.

Airplane mode is not an offline assertion: Android deliberately allows Wi-Fi to stay on.
This module reads every relevant control plus ConnectivityService's active default network,
then makes offline/restore decisions from observed state rather than command dispatch alone.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .atomic import atomic_write_text
from .errors import DeviceError, UsageError
from .schema import NetworkState

_ACTIVE_NETWORK_RE = re.compile(r"(?im)^Active default network:\s*(\S+)")
_TRANSPORTS_RE = re.compile(r"Transports:\s*([A-Z0-9_|]+)")
_CAPABILITIES_RE = re.compile(r"Capabilities:\s*([A-Z0-9_&]+)")
_NO_NETWORK = {"none", "null", "-1"}


class NetworkBackup(BaseModel):
    """A restore point tied to one serial and, when observable, one device boot."""

    model_config = ConfigDict(extra="forbid")

    serial: str
    instance_token: str | None = None
    saved_at: str
    state: NetworkState


def _shell_or_none(device: Any, command: str) -> str | None:
    with contextlib.suppress(Exception):
        return str(device.shell(command)).strip()
    return None


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().splitlines()[0].strip().lower() if raw.strip() else ""
    if value in {"1", "true", "enabled", "on", "yes"}:
        return True
    if value in {"0", "false", "disabled", "off", "no"}:
        return False
    return None


def _setting_bool(device: Any, namespace: str, key: str) -> bool | None:
    raw = _shell_or_none(device, f"settings get {namespace} {key}")
    if key == "wifi_on" and raw:
        # Android exposes additional Wi-Fi states here: 2 is the enabled airplane-mode
        # override and 3 is disabled. Connectivity remains the final verification oracle,
        # but preserving this control accurately is required for a reliable restore.
        value = raw.strip().splitlines()[0].strip()
        if value in {"1", "2"}:
            return True
        if value in {"0", "3"}:
            return False
    return _parse_bool(raw)


def _feature_bool(device: Any, feature: str) -> bool | None:
    return _parse_bool(_shell_or_none(device, f"pm has-feature {feature}"))


def parse_connectivity(raw: str | None) -> dict[str, Any]:
    """Extract only the active default network from ``dumpsys connectivity`` output."""
    if not raw:
        return {
            "active_network": None,
            "active_network_id": None,
            "active_transports": [],
            "internet_validated": None,
        }
    match = _ACTIVE_NETWORK_RE.search(raw)
    if match is None:
        return {
            "active_network": None,
            "active_network_id": None,
            "active_transports": [],
            "internet_validated": None,
        }
    network_id = match.group(1).strip()
    if network_id.lower() in _NO_NETWORK:
        return {
            "active_network": False,
            "active_network_id": None,
            "active_transports": [],
            "internet_validated": False,
        }

    line = next(
        (
            item
            for item in raw.splitlines()
            if f"NetworkAgentInfo{{network{{{network_id}}}" in item
        ),
        "",
    )
    transport_match = _TRANSPORTS_RE.search(line)
    transports = (
        [part.lower() for part in transport_match.group(1).split("|") if part]
        if transport_match
        else []
    )
    capabilities_match = _CAPABILITIES_RE.search(line)
    validated = (
        "VALIDATED" in capabilities_match.group(1).split("&")
        if capabilities_match is not None
        else None
    )
    return {
        "active_network": True,
        "active_network_id": network_id,
        "active_transports": transports,
        "internet_validated": validated,
    }


def read_network_state(device: Any) -> NetworkState:
    connectivity = parse_connectivity(_shell_or_none(device, "dumpsys connectivity"))
    active = connectivity["active_network"]
    return NetworkState(
        airplane_mode=device.get_airplane_mode(),
        wifi_supported=_feature_bool(device, "android.hardware.wifi"),
        wifi_enabled=_setting_bool(device, "global", "wifi_on"),
        cellular_supported=_feature_bool(device, "android.hardware.telephony"),
        mobile_data_enabled=_setting_bool(device, "global", "mobile_data"),
        **connectivity,
        offline=not active if active is not None else None,
    )


def apply_offline_controls(device: Any, initial: NetworkState) -> None:
    """Disable every ordinary internet transport, preserving failures for restore."""
    failures: list[str] = []
    operations: list[tuple[str, Callable[[], object]]] = [
        ("airplane mode", lambda: device.set_airplane_mode(True))
    ]
    if initial.wifi_supported is not False:
        operations.append(("Wi-Fi", lambda: device.shell("svc wifi disable")))
    if initial.cellular_supported is not False:
        operations.append(("mobile data", lambda: device.shell("svc data disable")))
    for label, operation in operations:
        try:
            operation()
        except Exception as exc:  # continue so one unavailable radio does not leave another live
            failures.append(f"{label}: {exc}")
    if failures:
        raise DeviceError(
            "could not apply every offline control: " + "; ".join(failures),
            hint="The original network state was saved; run `aua network restore`.",
        )


def restore_controls(device: Any, state: NetworkState) -> None:
    """Reapply the saved user controls; Android reconnects transports asynchronously."""
    failures: list[str] = []
    operations: list[tuple[str, Callable[[], object]]] = []
    if state.airplane_mode is not None:
        operations.append(
            ("airplane mode", lambda: device.set_airplane_mode(bool(state.airplane_mode)))
        )
    if state.wifi_enabled is not None:
        wifi = "enable" if state.wifi_enabled else "disable"
        operations.append(("Wi-Fi", lambda: device.shell(f"svc wifi {wifi}")))
    if state.mobile_data_enabled is not None:
        data = "enable" if state.mobile_data_enabled else "disable"
        operations.append(("mobile data", lambda: device.shell(f"svc data {data}")))

    for label, operation in operations:
        try:
            operation()
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    if failures:
        raise DeviceError(
            "could not restore every network control: " + "; ".join(failures),
            hint="The restore point was retained; fix device access and retry.",
        )


def offline_verified(state: NetworkState) -> bool:
    wifi_off = state.wifi_supported is False or state.wifi_enabled is False
    data_off = state.cellular_supported is False or state.mobile_data_enabled is False
    return bool(
        state.airplane_mode is True
        and wifi_off
        and data_off
        and state.active_network is False
    )


def restored_verified(current: NetworkState, saved: NetworkState) -> bool:
    controls = (
        ("airplane_mode", saved.airplane_mode),
        ("wifi_enabled", saved.wifi_enabled),
        ("mobile_data_enabled", saved.mobile_data_enabled),
    )
    for field, expected in controls:
        if expected is not None and getattr(current, field) != expected:
            return False
    if saved.active_network is not None and current.active_network != saved.active_network:
        return False
    # Android can bring cellular online before a saved Wi-Fi default reconnects. A validated
    # intermediate network is not the state we promised to restore, so require every transport
    # observed in the restore point before returning success and exposing the settled state.
    if saved.active_transports and not set(saved.active_transports).issubset(
        current.active_transports
    ):
        return False
    return saved.internet_validated is not True or current.internet_validated is True


def wait_for_state(
    device: Any,
    predicate: Callable[[NetworkState], bool],
    *,
    timeout_ms: int,
    poll_ms: int = 200,
) -> tuple[NetworkState, bool]:
    deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
    while True:
        state = read_network_state(device)
        if predicate(state):
            return state, True
        if time.monotonic() >= deadline:
            return state, False
        time.sleep(max(0.05, poll_ms / 1000.0))


def backup_path(cache_dir: str | Path, serial: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", serial)
    return Path(cache_dir).expanduser() / "network" / f"{safe}.json"


def load_backup(path: Path) -> NetworkBackup | None:
    if not path.is_file():
        return None
    try:
        return NetworkBackup.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise UsageError(
            f"network restore point is unreadable: {path}",
            hint="Move the invalid file aside, then run `aua network offline` again.",
        ) from exc


def save_backup(path: Path, *, device: Any, state: NetworkState) -> NetworkBackup:
    current_token = device.instance_token()
    existing = load_backup(path)
    if existing is not None:
        same_boot = (
            existing.instance_token is None
            or current_token is None
            or existing.instance_token == current_token
        )
        if same_boot:
            return existing
    backup = NetworkBackup(
        serial=device.serial,
        instance_token=current_token,
        saved_at=datetime.now(UTC).isoformat(),
        state=state,
    )
    atomic_write_text(path, backup.model_dump_json(indent=2))
    return backup


def require_current_backup(path: Path, *, device: Any) -> NetworkBackup:
    backup = load_backup(path)
    if backup is None:
        raise UsageError(
            "no saved network state to restore",
            hint="Run `aua network offline` first; it snapshots the current controls.",
        )
    current_token = device.instance_token()
    if (
        backup.instance_token is not None
        and current_token is not None
        and backup.instance_token != current_token
    ):
        raise UsageError(
            "saved network state belongs to a previous device boot",
            hint="Run `aua network offline` on this boot to create a fresh restore point.",
        )
    return backup
