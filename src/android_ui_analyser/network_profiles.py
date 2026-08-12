"""Reversible network-condition profiles for Android emulators and rooted devices."""

from __future__ import annotations

import contextlib
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .atomic import atomic_write_text
from .emulator import adb_bin
from .errors import DeviceError, UsageError
from .network import restore_controls, restored_verified, wait_for_state
from .schema import NetworkShaping, NetworkState

PROFILE_NAMES = ("wifi-only", "cellular-only", "slow", "lossy")
_RADIO_PROFILES = frozenset({"wifi-only", "cellular-only"})
_DEFAULT_QDISCS = frozenset({"mq", "pfifo_fast", "fq_codel", "noqueue"})
_ACTIVE_INTERFACE_RE = re.compile(r"\bdev\s+([A-Za-z0-9_.-]+)\b")
_ROOT_QDISC_RE = re.compile(r"(?m)^qdisc\s+(\S+)\s+\S+:\s+root\b")
_LOSS_RE = re.compile(r"\bloss\s+([0-9]+(?:\.[0-9]+)?)%")


class EmulatorShape(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_bps: int
    download_bps: int
    min_latency_ms: int
    max_latency_ms: int

    def evidence(self) -> NetworkShaping:
        return NetworkShaping(mechanism="emulator-console", **self.model_dump())


class ProfileBackup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serial: str
    instance_token: str | None = None
    saved_at: str
    profile: str
    network_state: NetworkState
    emulator_shape: EmulatorShape | None = None
    interface: str | None = None
    original_qdisc: str | None = None
    loss_percent: float | None = None
    root_was_enabled: bool | None = None


def normalize_profile(name: str) -> str:
    profile = name.strip().lower().replace("_", "-")
    if profile not in PROFILE_NAMES:
        raise UsageError(
            f"unknown network profile {name!r}",
            hint="Choose one of: " + ", ".join(PROFILE_NAMES),
        )
    return profile


def profile_path(cache_dir: str | Path, serial: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", serial)
    return Path(cache_dir).expanduser() / "network" / "profiles" / f"{safe}.json"


def load_profile(path: Path) -> ProfileBackup | None:
    if not path.is_file():
        return None
    try:
        return ProfileBackup.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise UsageError(
            f"network profile restore point is unreadable: {path}",
            hint="Move the invalid file aside before applying another profile.",
        ) from exc


def save_profile(
    path: Path,
    *,
    device: Any,
    profile: str,
    network_state: NetworkState,
    emulator_shape: EmulatorShape | None = None,
    interface: str | None = None,
    original_qdisc: str | None = None,
    loss_percent: float | None = None,
    root_was_enabled: bool | None = None,
) -> ProfileBackup:
    current_token = device.instance_token()
    existing = load_profile(path)
    if existing is not None:
        same_boot = (
            existing.instance_token is None
            or current_token is None
            or existing.instance_token == current_token
        )
        if same_boot:
            raise UsageError(
                f"network profile {existing.profile!r} is already active",
                hint="Run `aua network profile restore` before applying another profile.",
            )
    backup = ProfileBackup(
        serial=device.serial,
        instance_token=current_token,
        saved_at=datetime.now(UTC).isoformat(),
        profile=profile,
        network_state=network_state,
        emulator_shape=emulator_shape,
        interface=interface,
        original_qdisc=original_qdisc,
        loss_percent=loss_percent,
        root_was_enabled=root_was_enabled,
    )
    atomic_write_text(path, backup.model_dump_json(indent=2))
    return backup


def require_current_profile(path: Path, *, device: Any) -> ProfileBackup:
    backup = load_profile(path)
    if backup is None:
        raise UsageError(
            "no active network profile to restore",
            hint="Apply one with `aua network profile apply <name>` first.",
        )
    current_token = device.instance_token()
    if (
        backup.instance_token is not None
        and current_token is not None
        and backup.instance_token != current_token
    ):
        raise UsageError(
            "saved network profile belongs to a previous device boot",
            hint="Apply a fresh profile on this boot; stale settings cannot be restored safely.",
        )
    return backup


def _adb(
    serial: str,
    *args: str,
    check: bool = True,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(  # noqa: S603
            [adb_bin(), "-s", serial, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise DeviceError(f"adb command failed for {serial}: {exc}") from exc
    if check and result.returncode != 0:
        detail = ((result.stdout or "") + (result.stderr or "")).strip()
        raise DeviceError(f"adb {' '.join(args)} failed: {detail or result.returncode}")
    return result


def parse_emulator_shape(raw: str) -> EmulatorShape:
    patterns = {
        "download_bps": r"download speed:\s*(\d+)\s+bits/s",
        "upload_bps": r"upload speed:\s*(\d+)\s+bits/s",
        "min_latency_ms": r"minimum latency:\s*(\d+)\s+ms",
        "max_latency_ms": r"maximum latency:\s*(\d+)\s+ms",
    }
    values: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.IGNORECASE)
        if match is None:
            raise DeviceError(
                "could not read Android Emulator network shaping",
                hint="Slow profiles require an Android Emulator console, not a physical device.",
            )
        values[key] = int(match.group(1))
    return EmulatorShape(**values)


def read_emulator_shape(serial: str) -> EmulatorShape:
    result = _adb(serial, "emu", "network", "status", check=False)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "KO:" in output or "error:" in output.lower():
        raise DeviceError(
            f"Android Emulator network console is unavailable for {serial}",
            hint="The `slow` profile requires an emulator; radio-only profiles work elsewhere.",
        )
    return parse_emulator_shape(output)


def _shape_speed_arg(shape: EmulatorShape) -> str:
    if shape.upload_bps == 0 and shape.download_bps == 0:
        return "full"
    up_kbps = max(1, round(shape.upload_bps / 1000))
    down_kbps = max(1, round(shape.download_bps / 1000))
    return f"{up_kbps}:{down_kbps}"


def _shape_delay_arg(shape: EmulatorShape) -> str:
    if shape.min_latency_ms == 0 and shape.max_latency_ms == 0:
        return "none"
    if shape.min_latency_ms == shape.max_latency_ms:
        return str(shape.min_latency_ms)
    return f"{shape.min_latency_ms}:{shape.max_latency_ms}"


def set_emulator_shape(serial: str, *, speed: str, delay: str) -> EmulatorShape:
    _adb(serial, "emu", "network", "speed", speed)
    _adb(serial, "emu", "network", "delay", delay)
    return read_emulator_shape(serial)


def restore_emulator_shape(serial: str, shape: EmulatorShape) -> EmulatorShape:
    return set_emulator_shape(
        serial,
        speed=_shape_speed_arg(shape),
        delay=_shape_delay_arg(shape),
    )


def shape_matches(current: EmulatorShape, expected: EmulatorShape) -> bool:
    return bool(
        abs(current.upload_bps - expected.upload_bps) <= 1000
        and abs(current.download_bps - expected.download_bps) <= 1000
        and current.min_latency_ms == expected.min_latency_ms
        and current.max_latency_ms == expected.max_latency_ms
    )


def root_enabled(serial: str) -> bool:
    result = _adb(serial, "shell", "id", "-u", check=False, timeout=10)
    return result.returncode == 0 and (result.stdout or "").strip() == "0"


def ensure_root(serial: str) -> bool:
    was_root = root_enabled(serial)
    if was_root:
        return True
    result = _adb(serial, "root", check=False, timeout=30)
    output = ((result.stdout or "") + (result.stderr or "")).lower()
    if result.returncode != 0 or "cannot" in output or "production" in output:
        raise DeviceError(
            f"packet-loss shaping needs a rootable emulator; adb root refused on {serial}",
            hint=(
                "Use a Google APIs AVD: `aua emulator ensure-proxy`, then start it and apply "
                "the lossy profile with `--needs root`."
            ),
        )
    _adb(serial, "wait-for-device", timeout=60)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if root_enabled(serial):
            return False
        time.sleep(0.2)
    raise DeviceError(f"{serial} did not become root after adb root")


def restore_root(serial: str, *, was_root: bool) -> bool:
    if was_root:
        return root_enabled(serial)
    result = _adb(serial, "unroot", check=False, timeout=30)
    output = ((result.stdout or "") + (result.stderr or "")).lower()
    if result.returncode != 0 or "cannot" in output:
        return False
    _adb(serial, "wait-for-device", timeout=60)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not root_enabled(serial):
            return True
        time.sleep(0.2)
    return False


def active_interface(serial: str) -> str:
    result = _adb(serial, "shell", "ip", "route", "get", "1.1.1.1")
    match = _ACTIVE_INTERFACE_RE.search(result.stdout or "")
    if match is None:
        raise DeviceError(
            "could not identify the active network interface",
            hint="Make sure the device is online before applying the lossy profile.",
        )
    return match.group(1)


def qdisc_status(serial: str, interface: str) -> str:
    return (_adb(serial, "shell", "tc", "qdisc", "show", "dev", interface).stdout or "").strip()


def root_qdisc_kind(raw: str) -> str | None:
    match = _ROOT_QDISC_RE.search(raw)
    return match.group(1) if match is not None else None


def qdisc_evidence(serial: str, interface: str, *, root: bool) -> NetworkShaping:
    raw = qdisc_status(serial, interface)
    loss = _LOSS_RE.search(raw)
    return NetworkShaping(
        mechanism="tc-netem",
        interface=interface,
        loss_percent=float(loss.group(1)) if loss else None,
        qdisc=root_qdisc_kind(raw),
        root_enabled=root,
    )


def prepare_loss(serial: str) -> tuple[str, str, bool]:
    was_root = root_enabled(serial)
    try:
        ensure_root(serial)
        interface = active_interface(serial)
        original = qdisc_status(serial, interface)
        kind = root_qdisc_kind(original)
        if kind not in _DEFAULT_QDISCS:
            raise DeviceError(
                f"refusing to replace existing root qdisc {kind or 'unknown'!r} on {interface}",
                hint=(
                    "Remove or restore the existing traffic shaper before applying AUA's profile."
                ),
            )
        return interface, kind, was_root
    except Exception:
        safe_unroot_after_failed_apply(serial, was_root=was_root)
        raise


def set_loss(serial: str, *, interface: str, loss_percent: float) -> NetworkShaping:
    _adb(
        serial,
        "shell",
        "tc",
        "qdisc",
        "replace",
        "dev",
        interface,
        "root",
        "netem",
        "loss",
        f"{loss_percent:g}%",
    )
    return qdisc_evidence(serial, interface, root=True)


def remove_loss(serial: str, backup: ProfileBackup) -> tuple[NetworkShaping, bool]:
    if backup.interface is None:
        raise UsageError("lossy profile restore point has no interface")
    ensure_root(serial)
    try:
        current = qdisc_evidence(serial, backup.interface, root=True)
        if current.qdisc == "netem":
            _adb(serial, "shell", "tc", "qdisc", "del", "dev", backup.interface, "root")
        elif current.qdisc not in _DEFAULT_QDISCS:
            raise DeviceError(
                f"refusing to delete changed qdisc {current.qdisc or 'unknown'!r}",
                hint=(
                    "The restore point was retained because another shaper now owns the "
                    "interface."
                ),
            )
        after = qdisc_evidence(serial, backup.interface, root=True)
        expected_qdisc = backup.original_qdisc
        removed = bool(
            after.qdisc != "netem"
            and after.loss_percent is None
            and (expected_qdisc is None or after.qdisc == expected_qdisc)
        )
    except Exception:
        if not backup.root_was_enabled:
            with contextlib.suppress(Exception):
                restore_root(serial, was_root=False)
        raise
    root_restored = restore_root(serial, was_root=bool(backup.root_was_enabled))
    after.root_enabled = root_enabled(serial)
    return after, removed and root_restored


def profile_verified(profile: str, state: NetworkState) -> bool:
    if profile == "wifi-only":
        return bool(
            state.airplane_mode is False
            and state.wifi_enabled is True
            and state.mobile_data_enabled is False
            and state.active_network is True
            and "wifi" in state.active_transports
            and "cellular" not in state.active_transports
        )
    if profile == "cellular-only":
        return bool(
            state.airplane_mode is False
            and state.wifi_enabled is False
            and state.mobile_data_enabled is True
            and state.active_network is True
            and "cellular" in state.active_transports
            and "wifi" not in state.active_transports
        )
    return False


def profile_restored_verified(current: NetworkState, saved: NetworkState) -> bool:
    if not restored_verified(current, saved):
        return False
    if saved.active_transports:
        return bool(set(current.active_transports) & set(saved.active_transports))
    return not current.active_transports


def apply_radio_profile(device: Any, profile: str) -> None:
    device.set_airplane_mode(False)
    if profile == "wifi-only":
        device.shell("svc wifi enable")
        device.shell("svc data disable")
    else:
        device.shell("svc wifi disable")
        device.shell("svc data enable")


def wait_for_radio_profile(
    device: Any,
    profile: str,
    *,
    timeout_ms: int,
) -> tuple[NetworkState, bool]:
    return wait_for_state(
        device,
        lambda state: profile_verified(profile, state),
        timeout_ms=timeout_ms,
    )


def restore_radio_profile(
    device: Any,
    saved: NetworkState,
    *,
    timeout_ms: int,
) -> tuple[NetworkState, bool]:
    restore_controls(device, saved)
    return wait_for_state(
        device,
        lambda state: profile_restored_verified(state, saved),
        timeout_ms=timeout_ms,
    )


def stale_profile(backup: ProfileBackup, device: Any) -> bool:
    current = device.instance_token()
    return bool(
        backup.instance_token is not None
        and current is not None
        and backup.instance_token != current
    )


def safe_unroot_after_failed_apply(serial: str, *, was_root: bool) -> None:
    if not was_root:
        with contextlib.suppress(Exception):
            restore_root(serial, was_root=False)
