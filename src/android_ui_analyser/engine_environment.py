"""The conditions the app runs under: network and airplane state and network profiles, the mock proxy and its rules, clock, location, orientation, clipboard, media, and developer options.

Engine methods for environment. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_environment.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .engine_support import _ResolvedCassetteResource, logger
from .errors import DeviceError, UsageError
from .memory import RouteStep
from .platforms.identity import TargetRef
from .platforms.runtime import TargetRuntime as Device
from .schema import ActionResult, AppContext, NetworkResult

if TYPE_CHECKING:
    from .engine import Engine


def teardown_discard(
    self: Engine, *, serial: str, keys: list[str], reason: str, confirmed: bool = False,
) -> dict[str, Any]:
    """Explicitly abandon named stale undos; no platform loading or target connection."""
    from . import device_ledger

    name = self._platform.name if self._platform is not None else self.config.device.platform
    return device_ledger.discard(
        TargetRef(name, serial), keys=keys, reason=reason, confirmed=confirmed,
        lease_registry_dir=self._lease_registry_dir,
    )


def _runtime_capability(self: Engine, capability: str) -> Device:
    return self.platform.runtime_capability(capability, self.device)


def clipboard_set(self: Engine, text: str) -> ActionResult:
    _runtime_capability(self, "device.clipboard").set_clipboard(text)
    return ActionResult(ok=True, action="clipboard-set", detail=text)


def clipboard_get(self: Engine) -> ActionResult:
    text = _runtime_capability(self, "device.clipboard").get_clipboard()
    return ActionResult(ok=True, action="clipboard-get", detail=text)


def location_set(self: Engine, lat: float, lon: float) -> ActionResult:
    _runtime_capability(self, "device.location").set_location(lat, lon)
    return ActionResult(ok=True, action="location-set", detail=f"{lat},{lon}")


def orientation_set(self: Engine, mode: str) -> ActionResult:
    runtime = _runtime_capability(self, "device.orientation")
    requested = str(mode or "").strip()
    if requested.casefold() == "restore":
        existing = self._pending_device_change(
            "orientation", serial=runtime.target_id
        )
        previous = str(existing.args.get("mode") or "") if existing is not None else ""
        if not previous or previous.casefold() == "unknown":
            raise UsageError(
                "no saved orientation to restore",
                hint="Change orientation through AUA first, then use `orientation set restore`.",
            )
        with self._acting():
            runtime.set_orientation(previous)
        self.forget_device_change("orientation")
        return ActionResult(ok=True, action="orientation-restore", detail=previous)

    previous = runtime.get_orientation()
    if not previous or previous.casefold() == "unknown":
        raise DeviceError(
            "cannot change orientation without reading its current state",
            code="orientation_state_unknown",
            hint="No change was made; repair target orientation access and retry.",
        )
    existing = self._pending_device_change("orientation", serial=runtime.target_id)
    if existing is None:
        self.record_device_change(
            key="orientation",
            kind="orientation",
            op="set_orientation",
            args={"mode": previous},
            detail=f"orientation changed from {previous}",
        )
    with self._acting():
        runtime.set_orientation(requested)
    return ActionResult(ok=True, action="orientation-set", detail=requested)


def orientation_get(self: Engine) -> ActionResult:
    mode = _runtime_capability(self, "device.orientation").get_orientation()
    return ActionResult(ok=True, action="orientation-get", detail=mode)


def airplane_set(self: Engine, enabled: bool) -> ActionResult:
    runtime = _runtime_capability(self, "device.airplane")
    previous = runtime.get_airplane_mode()
    if previous is None:
        raise DeviceError(
            "cannot change airplane mode without reading its current state",
            code="airplane_state_unknown",
            hint="No change was made; repair target settings access and retry.",
        )

    target = runtime.target_id
    existing = self._pending_device_change("airplane_mode", serial=target)
    original = (
        bool(existing.args.get("enabled"))
        if existing is not None and "enabled" in existing.args
        else bool(previous)
    )
    if previous != enabled and existing is None:
        self.record_device_change(
            key="airplane_mode",
            kind="airplane_mode",
            op="set_airplane_mode",
            args={"enabled": bool(previous)},
            detail=f"airplane mode changed from {'on' if previous else 'off'}",
        )
    if previous != enabled:
        runtime.set_airplane_mode(enabled)
    if enabled == original:
        self.forget_device_change("airplane_mode")
    return ActionResult(
        ok=True,
        action="airplane-set",
        detail="on" if enabled else "off",
        note=(
            "airplane mode is not proof of offline connectivity because Wi-Fi may remain "
            "active; for offline tests use `aua network offline --verify` and always "
            "restore with `aua network restore`"
            if enabled
            else None
        ),
    )


def airplane_toggle(self: Engine) -> ActionResult:
    current = _runtime_capability(self, "device.airplane").get_airplane_mode()
    if current is None:
        raise DeviceError(
            "cannot toggle airplane mode without reading its current state",
            code="airplane_state_unknown",
            hint="No change was made; repair target settings access and retry.",
        )
    result = airplane_set(self, not current)
    result.action = "airplane-toggle"
    return result


def network_status(self: Engine) -> NetworkResult:
    network = self.platform.capability("network")

    device = _runtime_capability(self, "device.proxy")
    pending = self._pending_device_change("network_controls", serial=device.target_id)
    cache_dir = (
        str(pending.args.get("cache_dir"))
        if pending is not None and pending.args.get("cache_dir")
        else self.config.cache.dir
    )
    path = network.backup_path(cache_dir, device.serial)
    backup = network.load_backup(path)
    state = network.read_network_state(device)
    return NetworkResult(
        ok=True,
        action="network-status",
        state=state,
        saved_state=backup.state if backup is not None else None,
        verified=state.active_network is not None,
        detail="restore point available" if backup is not None else "no restore point",
    )


def network_offline(self: Engine, *, verify: bool = True, timeout_ms: int = 10_000) -> NetworkResult:
    network = self.platform.capability("network")
    network_profiles = self.platform.capability("network_profiles")

    device = self.device
    pending = self._pending_device_change("network_controls", serial=device.target_id)
    current_token = device.instance_token()
    if (
        pending is not None
        and pending.instance_token is not None
        and current_token is not None
        and pending.instance_token != current_token
    ):
        # Rebooting clears the controls the old entry owned. Drop that stale ownership before
        # replacing its host-side restore point for the new device instance.
        self.forget_device_change("network_controls")
        pending = self._pending_device_change("network_controls", serial=device.target_id)
        if pending is not None:
            raise DeviceError(
                "could not retire network state from the previous device boot",
                code="network_restore_point_cleanup_failed",
                hint="No network controls were changed; repair the device ledger and retry.",
            )
    if self._pending_device_change("radio_profile", serial=device.target_id) is not None:
        raise UsageError(
            "a network profile recorded by AUA is active for this target",
            hint="Run `aua network profile restore` on the original boot. If it is gone, inspect "
                 "`aua teardown status` and explicitly archive its stale undo with `aua teardown discard`.",
        )
    profile = network_profiles.load_profile(
        network_profiles.profile_path(self.config.cache.dir, device.serial)
    )
    if profile is not None and not network_profiles.stale_profile(profile, device):
        raise UsageError(
            f"network profile {profile.profile!r} is active",
            hint="Run `aua network profile restore` before entering offline mode.",
        )
    cache_dir = (
        str(pending.args.get("cache_dir"))
        if pending is not None and pending.args.get("cache_dir")
        else self.config.cache.dir
    )
    path = network.backup_path(cache_dir, device.serial)
    initial = network.read_network_state(device)
    if pending is not None:
        # A repeated request on the same boot is idempotent. In particular, an Engine using a
        # different cache directory must retain the first agent's original online baseline.
        backup = network.require_current_backup(path, device=device)
    else:
        backup = network.save_backup(path, device=device, state=initial)
        self.record_device_change(
            key="network_controls",
            kind="network_controls",
            op="restore_network_controls",
            args={"cache_dir": str(self.config.cache.dir)},
            detail="Wi-Fi / mobile data / airplane forced offline",
        )
    network.apply_offline_controls(device, initial)
    if verify:
        state, verified = network.wait_for_state(
            device,
            network.offline_verified,
            timeout_ms=timeout_ms,
        )
    else:
        state = network.read_network_state(device)
        verified = None
    result = NetworkResult(
        ok=bool(verified) if verify else True,
        action="network-offline",
        state=state,
        saved_state=backup.state,
        verified=verified,
        detail=(
            "offline verified"
            if verified
            else (
                "offline controls applied without verification"
                if not verify
                else "offline verification timed out; restore point retained"
            )
        ),
    )
    self._record_action_safe(RouteStep(kind="network-offline"))
    return result


def network_restore(self: Engine, *, timeout_ms: int = 15_000) -> NetworkResult:
    network = self.platform.capability("network")

    device = self.device
    pending = self._pending_device_change("network_controls", serial=device.target_id)
    cache_dir = (
        str(pending.args.get("cache_dir"))
        if pending is not None and pending.args.get("cache_dir")
        else self.config.cache.dir
    )
    path = network.backup_path(cache_dir, device.serial)
    backup = network.require_current_backup(path, device=device)
    network.restore_controls(device, backup.state)
    state, verified = network.wait_for_state(
        device,
        lambda current: network.restored_verified(current, backup.state),
        timeout_ms=timeout_ms,
    )
    if verified:
        path.unlink(missing_ok=True)
        self.forget_device_change("network_controls")
    result = NetworkResult(
        ok=verified,
        action="network-restore",
        state=state,
        saved_state=backup.state,
        verified=verified,
        detail=(
            "original network state restored"
            if verified
            else "restore verification timed out; restore point retained"
        ),
    )
    self._record_action_safe(RouteStep(kind="network-restore"))
    return result


def network_profile_list(self: Engine) -> dict[str, Any]:
    network_profiles = self.platform.capability("network_profiles")

    return {
        "ok": True,
        "action": "network-profile-list",
        "profiles": [
            {
                "name": "wifi-only",
                "effect": "enable Wi-Fi and disable mobile data",
                "needs": [],
            },
            {
                "name": "cellular-only",
                "effect": "disable Wi-Fi and enable mobile data",
                "needs": [],
            },
            {
                "name": "slow",
                "effect": "EDGE bandwidth with 80-400 ms latency",
                "needs": ["emulator"],
            },
            {
                "name": "lossy",
                "effect": "outbound packet loss on the active interface",
                "needs": ["root"],
            },
        ],
        "names": list(network_profiles.PROFILE_NAMES),
    }


def network_profile_status(self: Engine) -> NetworkResult:
    network = self.platform.capability("network")
    network_profiles = self.platform.capability("network_profiles")

    device = self.device
    pending = self._pending_device_change("radio_profile", serial=device.target_id)
    cache_dir = (
        str(pending.args.get("cache_dir"))
        if pending is not None and pending.args.get("cache_dir")
        else self.config.cache.dir
    )
    path = network_profiles.profile_path(cache_dir, device.serial)
    backup = network_profiles.load_profile(path)
    state = network.read_network_state(device)
    if backup is None:
        return NetworkResult(
            ok=True,
            action="network-profile-status",
            state=state,
            verified=True,
            detail="no active network profile",
        )
    if network_profiles.stale_profile(backup, device):
        return NetworkResult(
            ok=True,
            action="network-profile-status",
            profile=backup.profile,
            state=state,
            saved_state=backup.network_state,
            verified=False,
            detail="profile restore point belongs to a previous device boot",
        )

    shaping = None
    if backup.profile in ("wifi-only", "cellular-only"):
        verified = network_profiles.profile_verified(backup.profile, state)
    elif backup.profile == "slow":
        current = network_profiles.read_emulator_shape(device.serial)
        shaping = current.evidence()
        verified = bool(
            current.upload_bps > 0
            and current.download_bps > 0
            and current.min_latency_ms >= 80
            and current.max_latency_ms >= 400
        )
    else:
        if backup.interface is None:
            verified = False
        else:
            shaping = network_profiles.qdisc_evidence(
                device.serial,
                backup.interface,
                root=network_profiles.root_enabled(device.serial),
            )
            verified = bool(
                shaping.qdisc == "netem"
                and shaping.loss_percent is not None
                and backup.loss_percent is not None
                and abs(shaping.loss_percent - backup.loss_percent) < 0.01
            )
    return NetworkResult(
        ok=True,
        action="network-profile-status",
        profile=backup.profile,
        state=state,
        saved_state=backup.network_state,
        shaping=shaping,
        verified=verified,
        detail="profile verified" if verified else "profile could not be verified",
    )


def network_profile_apply(
    self: Engine,
    profile: str,
    *,
    loss_percent: float = 10.0,
    timeout_ms: int = 15_000,
) -> NetworkResult:
    network = self.platform.capability("network")
    network_profiles = self.platform.capability("network_profiles")

    name = network_profiles.normalize_profile(profile)
    if not 0.1 <= loss_percent <= 100:
        raise UsageError("--loss-percent must be between 0.1 and 100")
    device = self.device
    if self._pending_device_change("radio_profile", serial=device.target_id) is not None:
        raise UsageError(
            "a network profile is already active for this target",
            hint="Run `aua network profile restore` on the original boot. If it is gone, inspect "
                 "`aua teardown status` and explicitly archive its stale undo with `aua teardown discard`.",
        )
    if self._pending_device_change("network_controls", serial=device.target_id) is not None:
        raise UsageError(
            "verified offline mode is active for this target",
            hint="Run `aua network restore` on the original boot. If it is gone, inspect "
                 "`aua teardown status` and explicitly archive its stale undo with `aua teardown discard`.",
        )
    path = network_profiles.profile_path(self.config.cache.dir, device.serial)
    initial = network.read_network_state(device)
    if (
        network.load_backup(network.backup_path(self.config.cache.dir, device.serial))
        is not None
    ):
        raise UsageError(
            "verified offline mode is active",
            hint="Run `aua network restore` before applying a network profile.",
        )
    active_profile = network_profiles.load_profile(path)
    if active_profile is not None and not network_profiles.stale_profile(
        active_profile,
        device,
    ):
        raise UsageError(
            f"network profile {active_profile.profile!r} is already active",
            hint="Run `aua network profile restore` before applying another profile.",
        )

    # One record covers all three branches: the restore point names which kind was applied,
    # so the reaper undoes the right one without the ledger having to know.
    self.record_device_change(
        key="radio_profile",
        kind="radio_profile",
        op="restore_network_profile",
        args={"cache_dir": str(self.config.cache.dir), "timeout_ms": int(timeout_ms)},
        detail=f"network profile {name!r} applied",
    )

    if name in ("wifi-only", "cellular-only"):
        backup = network_profiles.save_profile(
            path,
            device=device,
            profile=name,
            network_state=initial,
        )
        network_profiles.apply_radio_profile(device, name)
        state, verified = network_profiles.wait_for_radio_profile(
            device,
            name,
            timeout_ms=timeout_ms,
        )
        shaping = None
    elif name == "slow":
        original_shape = network_profiles.read_emulator_shape(device.serial)
        backup = network_profiles.save_profile(
            path,
            device=device,
            profile=name,
            network_state=initial,
            emulator_shape=original_shape,
        )
        current_shape = network_profiles.set_emulator_shape(
            device.serial,
            speed="edge",
            delay="edge",
        )
        state = network.read_network_state(device)
        shaping = current_shape.evidence()
        verified = bool(
            current_shape.upload_bps > 0
            and current_shape.download_bps > 0
            and current_shape.min_latency_ms >= 80
            and current_shape.max_latency_ms >= 400
        )
    else:
        was_root = network_profiles.root_enabled(device.serial)
        self.record_device_change(
            key="radio_profile_root",
            kind="temporary_adbd_root",
            op="restore_adbd_root",
            args={"was_root": was_root},
            detail="adb root may be enabled while packet-loss shaping is prepared",
        )
        interface, original_qdisc, was_root = network_profiles.prepare_loss(device.serial)
        try:
            backup = network_profiles.save_profile(
                path,
                device=device,
                profile=name,
                network_state=initial,
                interface=interface,
                original_qdisc=original_qdisc,
                loss_percent=loss_percent,
                root_was_enabled=was_root,
            )
        except Exception:
            network_profiles.safe_unroot_after_failed_apply(
                device.serial,
                was_root=was_root,
            )
            raise
        # The durable profile now contains the same root baseline plus qdisc ownership, so it
        # supersedes the narrow crash-window entry. If forgetting fails, replaying both is
        # idempotent and still ordered profile first, root state second.
        self.forget_device_change("radio_profile_root")
        shaping = network_profiles.set_loss(
            device.serial,
            interface=interface,
            loss_percent=loss_percent,
        )
        state = network.read_network_state(device)
        verified = bool(
            shaping.qdisc == "netem"
            and shaping.loss_percent is not None
            and abs(shaping.loss_percent - loss_percent) < 0.01
        )

    result = NetworkResult(
        ok=verified,
        action="network-profile-apply",
        profile=name,
        state=state,
        saved_state=backup.network_state,
        shaping=shaping,
        verified=verified,
        detail=(
            f"{name} profile verified"
            if verified
            else f"{name} profile verification timed out; restore point retained"
        ),
    )
    self._record_action_safe(RouteStep(kind="network-profile", arg=name))
    return result


def network_profile_restore(self: Engine, *, timeout_ms: int = 20_000) -> NetworkResult:
    network = self.platform.capability("network")
    network_profiles = self.platform.capability("network_profiles")

    device = self.device
    pending = self._pending_device_change("radio_profile", serial=device.target_id)
    cache_dir = (
        str(pending.args.get("cache_dir"))
        if pending is not None and pending.args.get("cache_dir")
        else self.config.cache.dir
    )
    path = network_profiles.profile_path(cache_dir, device.serial)
    backup = network_profiles.require_current_profile(path, device=device)
    shaping = None
    if backup.profile in ("wifi-only", "cellular-only"):
        state, verified = network_profiles.restore_radio_profile(
            device,
            backup.network_state,
            timeout_ms=timeout_ms,
        )
    elif backup.profile == "slow":
        if backup.emulator_shape is None:
            raise UsageError("slow profile restore point has no original emulator shape")
        current = network_profiles.restore_emulator_shape(
            device.serial,
            backup.emulator_shape,
        )
        shaping = current.evidence()
        verified = network_profiles.shape_matches(current, backup.emulator_shape)
        state = network.read_network_state(device)
    else:
        shaping, verified = network_profiles.remove_loss(device.serial, backup)
        state = network.read_network_state(device)
    if verified:
        path.unlink(missing_ok=True)
        self.forget_device_change("radio_profile")
    result = NetworkResult(
        ok=verified,
        action="network-profile-restore",
        profile=backup.profile,
        state=state,
        saved_state=backup.network_state,
        shaping=shaping,
        verified=verified,
        detail=(
            "original network conditions restored"
            if verified
            else "profile restore verification failed; restore point retained"
        ),
    )
    self._record_action_safe(RouteStep(kind="network-profile-restore"))
    return result


def media_add(self: Engine, path: str, *, remote_dir: str | None = None) -> ActionResult:
    runtime = _runtime_capability(self, "device.media")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise UsageError(f"media file not found: {source}")
    remote_dir = runtime.media_directory(remote_dir)
    identity = hashlib.sha256(f"{source}\0{remote_dir}".encode()).hexdigest()[:20]
    key = f"added_media:{identity}"
    already_recorded = self._pending_device_change(key, serial=runtime.target_id) is not None
    if not already_recorded:
        self.record_device_change(
            key=key,
            kind="added_media",
            op="remove_added_media",
            args={"local_path": str(source), "remote_dir": remote_dir},
            detail=f"media file {source.name} added to the target gallery",
        )
    try:
        remote = runtime.add_media(str(source), remote_dir=remote_dir)
    except DeviceError as exc:
        # Android detects a destination collision before pushing any bytes. Keeping the
        # write-ahead undo in that proven no-mutation case would later delete somebody else's
        # pre-existing gallery file.
        if exc.code == "media_already_exists" and not already_recorded:
            self.forget_device_change(key)
        raise
    return ActionResult(ok=True, action="media-add", detail=remote)


def clock_set(self: Engine, *, timestamp_ms: int | None = None, restore: bool = False) -> ActionResult:
    """Set or restore the device wall clock (Maestro ``travel``).

        Moving the clock often invalidates auth tokens (401s). Always ``clock restore``
        (or ``clock set --restore``) when the test is done — never leave the device in
        a time-traveled state.
        """
    path = self._clock_backup_path()
    if restore:
        if not path.is_file():
            raise UsageError(
                "no saved clock to restore",
                hint="Run `aua clock set --ms …` first; it saves the prior wall clock.",
            )
        previous = int(path.read_text(encoding="utf-8").strip())
        _runtime_capability(self, "device.clock").set_clock(previous)
        path.unlink(missing_ok=True)
        self.forget_device_change("wall_clock")
        return ActionResult(ok=True, action="clock-restore", detail=str(previous))
    if timestamp_ms is None:
        raise UsageError("clock set needs --ms <unix-ms> (or --restore)")
    # Save current clock once so restore is possible.
    runtime = _runtime_capability(self, "device.clock")
    current = runtime.get_clock_ms()
    if current is None:
        raise DeviceError(
            "cannot change the device clock without reading its current time",
            code="clock_state_unknown",
            hint="No change was made; repair target clock access and retry.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(str(current), encoding="utf-8")
    if self._pending_device_change("wall_clock", serial=runtime.target_id) is None:
        # The undo carries the value *and* when it was taken: a device restored an hour later
        # must land on now, not on the instant the backup was written, or every token is
        # stale again for a different reason. Repeated time travel preserves this first value.
        self.record_device_change(
            key="wall_clock",
            kind="wall_clock",
            op="set_clock",
            args={"timestamp_ms": int(current), "saved_at": time.time()},
            detail=f"wall clock moved to {timestamp_ms} (was {current})",
        )
    runtime.set_clock(timestamp_ms)
    return ActionResult(
        ok=True,
        action="clock-set",
        detail=str(timestamp_ms),
    )


def _clock_backup_path(self: Engine) -> Path:
    serial = self._device.serial if self._device else (self.config.device.serial or "default")
    key = TargetRef(self.platform.name, str(serial)).storage_key
    return Path(self.config.cache.dir).expanduser() / f"clock_backup_{key}.txt"


def _dev_backup_path(self: Engine) -> Path:
    serial = self._device.serial if self._device else (self.config.device.serial or "default")
    key = TargetRef(self.platform.name, str(serial)).storage_key
    return Path(self.config.cache.dir).expanduser() / f"devopts_backup_{key}.json"


def _developer_restore_point(self: Engine) -> tuple[Path, bool]:
    """First pending developer snapshot, even when a later caller uses another cache."""

    pending = self._pending_device_change("developer_settings")
    if pending is None:
        return self._dev_backup_path(), False
    path = Path(str(pending.args.get("backup_path") or ""))
    if not path.is_file():
        raise DeviceError(
            "the pending developer-settings restore point is unavailable",
            code="developer_settings_restore_missing",
            hint=(
                "Do not apply another developer profile; recover the recorded backup path "
                "or explicitly clear the pending teardown entry after inspecting the target."
            ),
        )
    return path, True


def _proxy_port(self: Engine) -> int | None:
    """Listen port this AUA positively owns, or ``None``.

        The device setting names where traffic goes, not who owns the tunnel. Using it as an
        ownership proof let ``proxy stop`` remove another process's reverse mapping. Unpointing
        the device is always safe; removing a tunnel requires our per-device ownership record.
        """
    pm = self.platform.capability("proxy")

    serial = self._device.serial if self._device else self.config.device.serial
    if serial:
        state = pm.read_state(serial)
        if isinstance(state, dict) and int(state.get("port") or 0) > 0:
            return int(state["port"])
    return None


def dev_show(self: Engine) -> dict[str, Any]:
    devopts = self.platform.capability("developer_settings")

    state = devopts.read_state(self.device)
    return {"ok": True, "action": "dev-show", **state}


def dev_anim(self: Engine, mode: str) -> dict[str, Any]:
    devopts = self.platform.capability("developer_settings")

    path, already_recorded = self._developer_restore_point()
    m = (mode or "").lower()
    if m in {"on", "off"}:
        if not already_recorded:
            self.record_device_change(
                key="developer_settings",
                kind="developer_settings",
                op="restore_developer_settings",
                args={"backup_path": str(path)},
                detail=f"animation scales set to {'1' if m == 'on' else '0'}",
            )
        state = (
            devopts.anim_on(self.device, path)
            if m == "on"
            else devopts.anim_off(self.device, path)
        )
    elif m == "restore":
        if not path.is_file():
            raise UsageError(
                "no saved developer settings to restore",
                hint="Change developer settings through AUA before asking to restore them.",
            )
        # One write-ahead snapshot covers every developer knob AUA owns. Restoring only the
        # animation subset would forget the ledger while a prior `dev crashes` remained active.
        state = devopts.profile_default(self.device, path)
        self.forget_device_change("developer_settings")
    else:
        raise UsageError(
            f"unknown anim mode {mode!r}",
            hint="Use `aua dev anim on`, `aua dev anim off`, or `aua dev anim restore`.",
        )
    return {"ok": True, "action": f"dev-anim-{m}", **state}


def dev_crashes(self: Engine, enabled: bool) -> dict[str, Any]:
    devopts = self.platform.capability("developer_settings")

    path, already_recorded = self._developer_restore_point()
    if not already_recorded:
        self.record_device_change(
            key="developer_settings",
            kind="developer_settings",
            op="restore_developer_settings",
            args={"backup_path": str(path)},
            detail=f"crash and ANR dialogs {'shown' if enabled else 'hidden'}",
        )
    state = devopts.crashes_set(self.device, enabled, path)
    return {
        "ok": True,
        "action": "dev-crashes-on" if enabled else "dev-crashes-off",
        **state,
    }


def dev_profile(self: Engine, name: str) -> dict[str, Any]:
    devopts = self.platform.capability("developer_settings")

    path, already_recorded = self._developer_restore_point()
    n = (name or "").lower()
    if n == "ac":
        if not already_recorded:
            self.record_device_change(
                key="developer_settings",
                kind="developer_settings",
                op="restore_developer_settings",
                args={"backup_path": str(path)},
                detail="AC developer profile enabled",
            )
        state = devopts.profile_ac(self.device, path)
    elif n == "default":
        if not path.is_file():
            raise UsageError(
                "no saved developer settings to restore",
                hint="Apply `aua dev profile ac` before restoring the previous profile.",
            )
        state = devopts.profile_default(self.device, path)
        self.forget_device_change("developer_settings")
    else:
        raise UsageError(
            f"unknown dev profile {name!r}",
            hint="Use `ac` (anim off + crashes on) or `default` (restore).",
        )
    self._record_action_safe(RouteStep(kind="dev-profile", arg=n))
    return {"ok": True, "action": f"dev-profile-{n}", **state}


def proxy_start(
    self: Engine,
    *,
    port: int | None = None,
    install_ca: bool = True,
) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    # Touch the device first so a dead serial does not leave a stray mitmdump.
    device = self.device
    self._claim_or_reap_proxy(device)
    ca_info: dict[str, Any] | None = None
    if install_ca:
        try:
            ca_info = pm.install_system_ca(device.serial)
        except (DeviceError, UsageError) as exc:
            # Still start the proxy — but surface why HTTPS will likely fail.
            ca_info = {"ok": False, "error": str(exc), "hint": getattr(exc, "hint", None)}
            logger.warning("system CA install failed: %s", exc)
    # ``port<=0`` / omitted → random free high port (never hardcodes 8080).
    preferred = port if port and port > 0 else None
    pid, listen = pm.start_mitm(
        cache_dir=cache, port=preferred, mode="map", serial=device.serial
    )
    # Journal the undos *before* the device is rewired. A crash between the record and the
    # mutation leaves a redundant undo, which is harmless; a crash the other way leaves a
    # device pointed at a dead port with nothing on disk that says so — every app reports
    # "Offline" and the next agent has no way to learn why.
    self.record_device_change(
        key="host_proxy_process",
        kind="host_proxy_process",
        op="kill_host_process",
        args={"pid": int(pid), "match": "mitmdump"},
        detail=f"mitmdump pid {pid} listening on 127.0.0.1:{listen}",
    )
    self.record_device_change(
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        detail=f"device http_proxy set to 127.0.0.1:{listen}",
    )
    self.record_device_change(
        key=f"reverse_port:{listen}",
        kind="reverse_port",
        op="remove_reverse_port",
        args={"port": int(listen)},
        detail=f"reverse tcp:{listen} → host tcp:{listen}",
    )
    self.record_device_change(
        key="proxy_ownership",
        kind="proxy_ownership",
        op="clear_proxy_ownership",
        detail="cross-process record naming this agent as the proxy's owner",
    )
    try:
        device.reverse_port(listen, listen)
        device.set_http_proxy(f"127.0.0.1:{listen}")
    except Exception:
        with contextlib.suppress(Exception):
            pm.stop_mitm(cache)
        self.forget_device_change(
            "host_proxy_process",
            "http_proxy",
            "proxy_ownership",
            f"reverse_port:{listen}",
        )
        raise
    # Device-global ownership, at a path every process can read: who owns the proxy, on which
    # port, under which boot. Without it a parallel agent inherits a proxied emulator it
    # cannot see, and its own `proxy stop` silently empties this one's recordings.
    boot_id: str | None = None
    with contextlib.suppress(Exception):
        boot_id = device.instance_token()
    with contextlib.suppress(Exception):
        pm.write_state(
            device.serial,
            {
                "pid": int(pid),
                "port": int(listen),
                "boot_id": boot_id,
                "owner": self._ledger_identity().get("owner"),
                "cache_dir": str(cache),
            },
        )
    # Relaunching the foreground app makes it inherit Zygote CA mounts.
    pkg = None
    with contextlib.suppress(Exception):
        tree = self.platform.runtime_capability("ui.tree", device)
        pkg = AppContext.coerce(tree.current_app()).app_id
    if pkg and ca_info and ca_info.get("ok"):
        with contextlib.suppress(Exception):
            self._app_process_replaced(pkg)
            lifecycle = self.platform.runtime_capability("app.lifecycle", device)
            lifecycle.stop_app(pkg)
            lifecycle.launch_app(pkg)
    out: dict[str, Any] = {
        "ok": True,
        "action": "proxy-start",
        "pid": pid,
        "port": listen,
        "ca": ca_info,
        "hint": (
            f"Device http_proxy is 127.0.0.1:{listen} (via adb reverse). "
            + (
                "System CA installed — force-stop/relaunch done for foreground app."
                if ca_info and ca_info.get("ok")
                else "CA install failed or skipped: HTTPS apps that only trust system "
                "CAs will produce EMPTY cassettes until the mitm CA is a "
                "system trust anchor. Fix: `aua emulator ensure-proxy` → start "
                "`aua_proxy` → `aua proxy start` on that serial."
            )
        ),
    }
    if pkg:
        out["relaunched"] = pkg
    return out


def _claim_or_reap_proxy(self: Engine, device: Device) -> None:
    """Refuse to overwrite a healthy foreign proxy; clean up a dead one first.

        ``http_proxy`` is a single device-global setting. Two agents on one emulator means the
        second silently redirects the first's traffic into its own mitmdump, and the first keeps
        writing an empty cassette while its assertions quietly pass. So a live foreign owner is
        an error with a way out, and a dead one is licence to reap.
        """
    pm = self.platform.capability("proxy")

    state = None
    with contextlib.suppress(Exception):
        state = pm.read_state(device.serial)
    if not isinstance(state, dict):
        return
    boot_id: str | None = None
    with contextlib.suppress(Exception):
        boot_id = device.instance_token()
    reason: str | None = "unknown"
    with contextlib.suppress(Exception):
        reason = pm.orphan_reason(state, boot_id=boot_id)
    if reason is None:
        owner = state.get("owner") or "another agent"
        raise UsageError(
            f"{device.serial} is already proxied through 127.0.0.1:{state.get('port')} "
            f"by {owner}",
            hint=(
                "Do NOT take it over — that agent's traffic would land in your mitmdump and "
                "its cassette would come out empty. Use the running proxy (`aua proxy status`, "
                "`aua mock map …`), or start your own emulator with "
                "`aua emulator start --headless --parallel`. If you know that holder is a "
                "dead run, `aua teardown run --serial-target "
                f"{device.serial} --force` cleans it up first."
            ),
        )
    # Positive evidence of death — hand the device back before wiring a new proxy onto it.
    logger.warning("reaping orphaned proxy on %s: %s", device.serial, reason)
    from . import teardown

    with contextlib.suppress(Exception):
        teardown.reap(
            device.serial,
            platform=self.platform,
            platform_name=self.platform.name,
            cache_dir=self.config.cache.dir,
            grace_s=float(self.config.teardown.grace_s),
            force=True,
        )
    with contextlib.suppress(Exception):
        pm.clear_state(device.serial)


def proxy_stop(self: Engine) -> dict[str, Any]:
    pm = self.platform.capability("proxy")
    device = _runtime_capability(self, "device.proxy")

    cache = Path(self.config.cache.dir).expanduser()
    p = self._proxy_port()
    with contextlib.suppress(Exception):
        device.set_http_proxy(None)
    if p is not None:
        with contextlib.suppress(Exception):
            device.remove_reverse_port(p)
    stopped = pm.stop_mitm(cache)
    # Undone deliberately, so the journal must forget it: a pending undo the reaper would
    # replay later is a promise to un-point a device that some *later* proxy may own.
    self.forget_device_change("http_proxy", "host_proxy_process", "proxy_ownership")
    if p is not None:
        self.forget_device_change(f"reverse_port:{p}")
    with contextlib.suppress(Exception):
        pm.clear_state(device.serial)
    return {"ok": True, "action": "proxy-stop", "stopped": stopped, "port": p}


def proxy_status(self: Engine, *, heal: bool = True, serial: str | None = None) -> dict[str, Any]:
    """Is the proxy an agent thinks is armed ACTUALLY working end to end — not just one of
        the pieces that have to hold together for it to be.

        Measured 2026-08-19: the device's ``http_proxy`` setting and the mitmdump process each
        looked fine on their own, but the ``adb reverse`` tunnel between them was gone — every
        app's network call failed with ``ConnectException``, visible only in logcat, and no
        `aua` surface said so because nothing had ever checked the three pieces together. This
        is that check, in the same ``{"ok", "detail"/"hint", "checks": {...}}`` shape `aua
        doctor` already uses, so an agent asking "is my interception actually working?" gets an
        answer it can branch on rather than three unrelated fields to cross-reference by hand.

        ``heal`` (default on) re-establishes a dropped tunnel automatically, but only when the
        process and the device setting both already check out — a dropped ``adb reverse`` is a
        normal consequence of an adb restart or a device reconnect, not user error, and it is
        the one piece safe to fix without guessing at what the caller wanted. The undo for
        touching the device is journalled *before* the tunnel is re-created, same as every
        other device mutation here, under the identical ``reverse_port:<port>`` key
        ``proxy_start`` already uses — so this never doubles up the ledger, it just refreshes a
        record that may otherwise have gone stale.
        """
    pm = self.platform.capability("proxy")
    cache = Path(self.config.cache.dir).expanduser()
    # An explicit *serial* is how a read-only observer (the dashboard) asks this
    # question. Connecting would attach uiautomator2 and take the UiAutomation slot
    # away from whichever agent is actually driving the device, to learn a string the
    # caller already had.
    target = serial or self.device.serial

    report = pm.proxy_health(target, cache, self_heal=False)
    adopted = False
    if heal and report.get("adoptable"):
        self._adopt_own_proxy(pm, self.device, cache, report)
        adopted = True
        report = pm.proxy_health(target, cache, self_heal=False)
    checks = report.get("checks") or {}
    tunnel = checks.get("tunnel")
    process = checks.get("process")
    listener = checks.get("listener")
    device_setting = checks.get("device_setting")
    port = report.get("port")
    safe_to_heal = bool(
        heal
        and port
        and tunnel is not None
        and not tunnel.get("ok")
        and process is not None
        and process.get("ok")
        and listener is not None
        and listener.get("ok")
        and device_setting is not None
        and device_setting.get("ok")
    )
    if safe_to_heal:
        self.record_device_change(
            key=f"reverse_port:{port}",
            kind="reverse_port",
            op="remove_reverse_port",
            args={"port": int(port)},
            detail=(
                f"`aua proxy status` self-healed a dropped adb reverse tcp:{port} "
                f"tcp:{port} tunnel (process + device setting were already fine)"
            ),
        )
        with contextlib.suppress(Exception):
            pm.ensure_reverse_tunnel(target, port)
        report = pm.proxy_health(target, cache, self_heal=False)
        healed_tunnel = (report.get("checks") or {}).get("tunnel")
        if healed_tunnel is not None and healed_tunnel.get("ok"):
            healed_tunnel["healed"] = True
            healed_tunnel["detail"] = str(healed_tunnel["detail"]) + " (just re-established)"
            # `ok` at the top level was computed before the heal; recompute now that the
            # tunnel check has flipped.
            report["ok"] = all(bool(c.get("ok")) for c in report["checks"].values())
            report["state"] = "healthy" if report["ok"] else report.get("state")
            report["intercepting"] = bool(report["ok"]) and bool(report.get("owned"))
            report.pop("hint", None)
    if adopted:
        report["adopted"] = True
        report["warning"] = (
            "rebuilt this device's missing proxy ownership record from this session's own "
            "mitm sidecars (mitmproxy.port/mitmproxy.pid)"
        )
    report["action"] = "proxy-status"
    return report


def _adopt_own_proxy(
    self: Engine, pm: Any, device: Device, cache: Path, report: dict[str, Any]
) -> None:
    """Rebuild the ownership record for a proxy this session can PROVE is its own.

        The premise is not hypothetical: ``proxy_start`` wraps both the boot-id read and its
        ``pm.write_state`` in ``contextlib.suppress(Exception)``, so a perfectly healthy aua
        proxy can exist with no ownership record at all. The record's absence is then the bug,
        and adoption restores a fact that was already true.

        The proof is two sidecars this cache dir wrote and a live pid (``proxy_mock._self_proof``)
        — never "a port answered TCP". Ownership here is *executable*: ``proxy_stop``,
        ``teardown.reap`` and the watchdog all act on it, so fabricating it would let a
        diagnostic kill another agent's mitmdump and un-point their device.

        Write-ahead, like every other mutation: the ``proxy_ownership`` undo is journalled
        before the record is written. A crash between them leaves a redundant undo (harmless); a
        crash the other way leaves a device claimed by an owner nothing can retract. The boot id
        is read fresh rather than assumed — the port sidecar can outlive a reboot.
        """
    self.record_device_change(
        key="proxy_ownership",
        kind="proxy_ownership",
        op="clear_proxy_ownership",
        detail=(
            "`aua proxy status` rebuilt the missing cross-process record naming this agent "
            f"as owner of the proxy on 127.0.0.1:{report.get('port')}"
        ),
    )
    boot_id: str | None = None
    with contextlib.suppress(Exception):
        boot_id = device.instance_token()
    pm.write_state(
        device.serial,
        {
            "pid": int(report["adoptable_pid"]),
            "port": int(report["port"]),
            "boot_id": boot_id,
            "owner": self._ledger_identity().get("owner"),
            "cache_dir": str(cache),
            "adopted": True,
        },
    )


def proxy_survey(self: Engine) -> dict[str, Any]:
    """Proxy health for every attached target, read-only — what `aua doctor` reports.

        This matters more than `aua proxy status` does. An agent inheriting a device runs `aua
        doctor`; it does not run `aua proxy status --serial X` for a serial it has not yet
        thought about. Before this, a black-holed device was invisible to every `aua` surface an
        arriving agent would plausibly try.

        Deliberately never heals and never connects: ``self_heal=False`` unconditionally
        (doctor reports, it does not mutate — the same rule ``_installed_skill_check`` follows),
        and it goes by serial rather than through ``self.device``, which would connect and can
        raise. That is also what makes it safe to sweep serials the caller never pointed at:
        two adb reads per device, no writes.

        Group ``ok`` is false for ``blackholed``, ``degraded``, and ``unknown``. An unproxied device is
        the normal case and must not fail doctor; a ``foreign`` proxy is someone else's working
        setup and gets a hint at most.
        """
    pm = self.platform.capability("proxy")
    cache = Path(self.config.cache.dir).expanduser()

    devices: list[dict[str, Any]] = []
    try:
        infos = self.list_devices()
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": True, "detail": f"could not list targets: {exc}", "devices": []}

    for info in infos:
        serial = info.serial
        try:
            health = pm.proxy_health(serial, cache, self_heal=False)
        except Exception as exc:  # pragma: no cover - defensive
            devices.append({"serial": serial, "state": "unknown", "detail": str(exc)})
            continue
        entry: dict[str, Any] = {
            "serial": serial,
            "state": health.get("state"),
            "owned": health.get("owned"),
            "intercepting": health.get("intercepting"),
            "target": health.get("target"),
        }
        for key in ("detail", "hint", "warning"):
            if health.get(key):
                entry[key] = health[key]
        devices.append(entry)

    bad = [d for d in devices if d.get("state") in {"blackholed", "degraded", "unknown"}]
    noteworthy = [d for d in devices if d.get("state") == "foreign"]
    if not devices:
        detail = "no attached target to check"
    elif bad:
        detail = "; ".join(f"{d['serial']}: {d.get('state')}" for d in bad)
    else:
        detail = ", ".join(f"{d['serial']}: {d.get('state')}" for d in devices)
    out: dict[str, Any] = {"ok": not bad, "detail": detail, "devices": devices}
    if bad:
        out["hint"] = " ".join(str(d.get("hint") or d.get("detail") or "") for d in bad).strip()
    elif noteworthy:
        out["hint"] = " ".join(
            str(d.get("warning") or d.get("detail") or "") for d in noteworthy
        ).strip()
    return out


def _refresh_proxy_ownership_pid(self: Engine, pm: Any, port: int, pid: int) -> None:
    """Keep the shared ownership record's pid current across an internal mitm restart.

        ``mock record start``/``stop`` restart mitmdump under a fresh pid on the same port (to
        flip map/record mode), but never touched the ownership record `proxy_start` wrote —
        so it kept naming the *original* pid forever. `pid_alive(old_pid)` goes false the
        instant that first process exits, even though a new one already owns the same socket,
        which would make `proxy_health`'s process check report "dead" right after a perfectly
        healthy mode flip. Best-effort and silent: this is bookkeeping, not the mutation, and
        must never fail the recording action it rides along with.
        """
    with contextlib.suppress(Exception):
        serial = self.device.serial
        state = pm.read_state(serial)
        if isinstance(state, dict) and int(state.get("port") or 0) == int(port):
            pm.write_state(serial, {**state, "pid": int(pid), "port": int(port)})


def _proxy_health_warning(self: Engine, serial: str | None = None) -> str | None:
    """A one-line warning when the armed proxy is not actually reachable, else ``None``.

        Called only from the two points an agent is most likely to be misled by a clean
        response: arming a mock rule and starting a recording — both look identical whether
        the device can reach the proxy or not, and both are exactly where a caller most needs
        to know before spending a whole flow on traffic that never arrives. Does not run on
        every proxy command: `proxy_start` just built everything fresh, and `mock list`/`mock
        rm`/`mock clear` do not touch the device at all, so a device round trip there would be
        pure overhead for a question nobody asked. It is diagnostic only: arming a mock rule
        must not mutate persistent device state as an incidental side effect.

        The gate is the *device's* setting, not an ownership record. It used to be the record,
        which meant this went silent in exactly the state it exists for: a device black-holed
        by a partial teardown has no record, so `mock map` and `mock record start` said nothing
        while every request the recording was supposed to capture failed with
        ``ConnectException``. It also warns on a `foreign` proxy — traffic flows there, so
        nothing looks wrong, but these rules are not the rules that proxy reads. It stays
        silent on `unproxied`, which is the normal case of arming rules before `proxy start`:
        crying wolf there would train agents to ignore this line.
        """
    status: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        status = self.proxy_status(heal=False, serial=serial)
    if not isinstance(status, dict):
        return None
    state = status.get("state")
    if state in (None, "unproxied", "healthy"):
        return None
    message = status.get("hint") or status.get("warning") or status.get("detail")
    return f"proxy health check: {message}" if message else None


def _proxy_serial(self: Engine, serial: str | None = None) -> str | None:
    """Which target this proxy/mock call belongs to, resolved without connecting.

        Proxy state is per-serial: two agents on two devices each get their own rules and
        their own traffic log. Falls back to the lease this engine already holds, so an
        ordinary `aua mock map` still lands on the device the agent is driving without the
        caller having to name it.
        """
    if serial:
        return str(serial)
    if self._device is not None:
        return self._device.serial
    with contextlib.suppress(Exception):
        leased = self._leased_serial()
        if leased:
            return str(leased)
    return getattr(self.config.device, "serial", None) or None


def _arm_mock_rule(
    self: Engine,
    rule: dict[str, Any],
    *,
    action: str,
    detail: str,
    serial: str | None = None,
) -> dict[str, Any]:
    """Append one built rule to the armed set, undo journalled first.

        Shared by :meth:`mock_map` and :meth:`mock_rewrite` so a stub and a rewrite are
        armed, owned, journalled and warned about identically — the two differ only in
        the rule they build, and letting that difference leak into the arming step is how
        one of them ends up without an undo record.
        """
    from . import leases

    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    target = self._proxy_serial(serial)
    rules_file = pm.rules_path(cache, target)
    doc = pm.load_doc(rules_file)
    existing = list(doc["rules"])
    owner = str(leases.resolve_owner(None))
    warning: str | None = None
    if existing and doc.get("owner") != owner:
        who = f"owner {doc['owner']!r}" if doc.get("owner") else "an untagged earlier session"
        warning = (
            f"appending onto {len(existing)} pre-existing mock rule(s) armed by {who}, "
            f"not this session ({owner!r}). Another agent's stubs may still be live — "
            "run `aua mock list` to inspect them, or `aua mock clear` to start clean."
        )
        logger.warning(warning)
    rules = existing + [rule]
    doc["rules"] = rules
    doc["owner"] = doc.get("owner") or owner
    # Journal the undo *before* the rule is armed: a crash right after this call must
    # still leave a stranger enough to clear it, or a left-armed stub silently poisons
    # whichever agent inherits this cache dir next.
    # `serial` matters here: record_device_change falls back to this engine's own
    # device and silently records nothing when there is none. The dashboard's engine
    # deliberately never connects, so without this the rule it just armed would be
    # unretractable by the reaper that exists for exactly that case.
    self.record_device_change(
        key="mock_rules",
        kind="mock_rules",
        op="clear_mock_rules",
        args={"cache_dir": str(cache), "serial": target},
        detail=detail,
        serial=target,
    )
    pm.save_doc(rules_file, doc)
    out: dict[str, Any] = {"ok": True, "action": action, "rule": rule, "count": len(rules)}
    health_warning = self._proxy_health_warning(serial)
    if warning and health_warning:
        warning = f"{warning} Also: {health_warning}"
    elif health_warning:
        warning = health_warning
    if warning:
        out["warning"] = warning
    return out


def mock_map(
    self: Engine,
    method: str,
    path: str,
    *,
    status: int = 200,
    body: str | None = None,
    host: str | None = None,
    times: int = 0,
    serial: str | None = None,
) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    rule = pm.map_rule(method, path, status=status, body=body, host=host, times=times)
    pm.guard_rule_scope(rule)
    return self._arm_mock_rule(
        rule,
        action="mock-map",
        detail=f"mock stub rule armed via `aua mock map` ({method} {path})",
        serial=serial,
    )


def mock_rewrite(
    self: Engine,
    method: str,
    path: str,
    *,
    host: str | None = None,
    query: str | None = None,
    request_body: str | None = None,
    status: int | None = None,
    headers: dict[str, str] | None = None,
    body: Any = None,
    set_json: dict[str, Any] | None = None,
    delete_json: list[str] | None = None,
    replace: list[tuple[str, str]] | None = None,
    times: int = 0,
    serial: str | None = None,
) -> dict[str, Any]:
    """Arm a rule that lets the request reach the server, then patches the response.

        The complement to :meth:`mock_map`. A stub answers from the rule and the server
        never hears about it; a rewrite is how you keep the real exchange and change one
        thing about the answer — the status an app sees, a header, a JSON field — which is
        what you want when reproducing a server-side condition you cannot trigger on demand.
        """
    pm = self.platform.capability("proxy")

    rule = pm.rewrite_rule(
        host=host,
        method=method,
        path=path,
        query=query,
        request_body=request_body,
        status=status,
        headers=headers,
        body=body,
        set_json=set_json,
        delete_json=delete_json,
        replace=replace,
        times=times,
    )
    # A rewrite with no host and a catch-all path matches Android's own connectivity
    # probes as well as the app's traffic, and the device just looks offline.
    pm.guard_rule_scope(rule)
    return self._arm_mock_rule(
        rule,
        action="mock-rewrite",
        detail=f"mock rewrite rule armed via `aua mock rewrite` ({method} {path})",
        serial=serial,
    )


def mock_list(self: Engine, *, serial: str | None = None) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    target = self._proxy_serial(serial)
    rules_file = pm.rules_path(cache, target)
    doc = pm.load_doc(rules_file)
    rules, changed = pm.backfill_rule_ids(doc["rules"])
    if changed:
        pm.write_rules(rules_file, rules)
    # How many times each rule actually fired. The addon spends a rule's `times` budget
    # in its own process and deliberately never writes it back, so the flow log is the
    # only place this is knowable — and without it a caller cannot tell a rule that is
    # armed from one that has already been used up, or one that never matched at all.
    fired: dict[str, int] = {}
    with contextlib.suppress(Exception):
        for entry in pm.read_flows_since(cache, 0, target):
            rid = entry.get("rule")
            if rid:
                fired[str(rid)] = fired.get(str(rid), 0) + 1
    listed = [dict(rule, fired=fired.get(str(rule.get("id")), 0)) for rule in rules]
    return {
        "ok": True,
        "action": "mock-list",
        "mode": doc["mode"],
        "owner": doc.get("owner"),
        "serial": target,
        "count": len(listed),
        "rules": listed,
    }


def mock_clear(self: Engine, *, serial: str | None = None) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    removed = pm.clear_rules(cache, self._proxy_serial(serial))
    # The change is undone right here, deliberately — forget the pending journal entry or
    # a reaper replays a no-op undo against a device that has already moved on.
    self.forget_device_change("mock_rules")
    return {"ok": True, "action": "mock-clear", "removed": removed}


def mock_rm(self: Engine, rule_id: str, *, serial: str | None = None) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    rules_file = pm.rules_path(cache, self._proxy_serial(serial))
    rules, _changed = pm.backfill_rule_ids(pm.load_rules(rules_file))
    kept = [r for r in rules if str(r.get("id")) != str(rule_id)]
    if len(kept) == len(rules):
        raise UsageError(
            f"no mock rule with id {rule_id!r}",
            hint="`aua mock list` to see current ids.",
        )
    pm.write_rules(rules_file, kept)
    return {"ok": True, "action": "mock-rm", "id": rule_id, "count": len(kept)}


def mock_record(
    self: Engine, action: str, name: str | None = None, *, serial: str | None = None
) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    target = self._proxy_serial(serial)
    a = (action or "").lower()
    if a == "start":
        if not name:
            raise UsageError("mock record start needs a NAME")
        # The window this recording covers, captured *before* anything else touches the
        # log or the record file, so `stop` can later tell a stale line (an earlier,
        # unrelated run) from evidence about this recording — see
        # `proxy_mock.diagnose_empty_recording`.
        log = cache / "mitmdump.log"
        log_offset = log.stat().st_size if log.is_file() else 0
        pm.save_record_window(cache, since_ts=time.time(), log_offset=log_offset, serial=target)
        # Clean JSONL seed: the addon appends one `json.dumps(entry) + "\n"` per completed
        # flow directly to disk as it happens (see `AuaMock.response()`), so there is
        # nothing to lose here — this only has to not corrupt that stream (see
        # `proxy_mock.reset_record`).
        pm.reset_record(cache, target)
        # Recording is persistent, device-pointing proxy state: a crash here must still
        # leave a stranger enough to disarm it, or the next agent silently inherits
        # `record` mode.
        self.record_device_change(
            key="mock_rules",
            kind="mock_rules",
            op="clear_mock_rules",
            args={"cache_dir": str(cache)},
            detail=f"mock record mode armed via `aua mock record start {name}`",
        )
        # Restart mitm in record mode if running; otherwise just arm the sidecar.
        env_mode = cache / "mock_mode.txt"
        env_mode.write_text("record", encoding="utf-8")
        (cache / "mock_record_name.txt").write_text(name, encoding="utf-8")
        # Live addon reads AUA_MOCK_MODE from process env — restart to flip mode.
        # Keep the same listen port when one is already bound so adb reverse stays valid.
        prev = pm.load_listen_port(cache)
        pm.stop_mitm(cache)
        pid, listen = pm.start_mitm(cache_dir=cache, port=prev, mode="record", serial=target)
        self._refresh_proxy_ownership_pid(pm, listen, pid)
        runtime = _runtime_capability(self, "device.proxy")
        with contextlib.suppress(Exception):
            runtime.reverse_port(listen, listen)
            runtime.set_http_proxy(f"127.0.0.1:{listen}")
        rec_out: dict[str, Any] = {
            "ok": True,
            "action": "mock-record-start",
            "name": name,
            "port": listen,
        }
        # The reverse/proxy calls just above are best-effort and swallow their own
        # exceptions — a recording that silently never sees a single flow because the
        # tunnel or the setting quietly failed to apply is exactly the failure this
        # confirms did not happen before an agent spends a whole flow capturing nothing.
        health_warning = self._proxy_health_warning()
        if health_warning:
            rec_out["warning"] = health_warning
        return rec_out
    if a == "stop":
        name_path = cache / "mock_record_name.txt"
        rec_name = name or (
            name_path.read_text(encoding="utf-8").strip() if name_path.is_file() else ""
        )
        if not rec_name:
            raise UsageError("mock record stop needs the cassette NAME")
        window = pm.load_record_window(cache, target)
        entries = pm.load_record(cache, target)
        dest = pm.cassette_dir(self.config.memory.dir) / f"{rec_name}.yaml"
        pm.save_cassette(dest, rec_name, entries)
        # Flip back to map mode on the same port.
        prev = pm.load_listen_port(cache)
        pm.stop_mitm(cache)
        pid, listen = pm.start_mitm(cache_dir=cache, port=prev, mode="map", serial=target)
        self._refresh_proxy_ownership_pid(pm, listen, pid)
        runtime = _runtime_capability(self, "device.proxy")
        with contextlib.suppress(Exception):
            runtime.reverse_port(listen, listen)
            runtime.set_http_proxy(f"127.0.0.1:{listen}")
        out: dict[str, Any] = {
            "ok": True,
            "action": "mock-record-stop",
            "name": rec_name,
            "path": str(dest),
            "entries": len(entries),
            "port": listen,
        }
        if not entries:
            since_ts = window["since_ts"] if window else 0.0
            log_offset = window["log_offset"] if window else 0
            diag = pm.diagnose_empty_recording(
                cache, since_ts=since_ts, log_offset=log_offset, serial=target
            )
            out["ok"] = False
            out["diagnosis"] = diag
            diagnosis = diag["diagnosis"]
            if diagnosis == "tls_failed":
                out["code"] = "proxy_tls"
                out["hint"] = (
                    "Recorded 0 HTTP flows. The app under test's own traffic failed the "
                    "TLS handshake against the mitm CA during this recording — it does "
                    "not trust it (its NSC is system-only). Re-run `aua proxy start` on "
                    "a rootable emulator so the system CA overlay is installed, then "
                    "force-stop + relaunch the app."
                )
            elif diagnosis == "decrypted_not_recorded":
                out["code"] = "proxy_record_lost"
                out["hint"] = (
                    f"Recorded 0 HTTP flows, but {diag['decrypted_flows_app']} flow(s) "
                    "for the app under test decrypted fine during this window (see the "
                    "flow log). This is not a CA trust problem — it looks like an aua "
                    "bug in the recording pipeline itself."
                )
            elif diagnosis == "system_traffic_only":
                out["code"] = "proxy_no_app_traffic"
                out["hint"] = (
                    "Recorded 0 HTTP flows. Only OS/Google-services traffic was seen "
                    "while recording — expected to fail TLS against this CA and not "
                    "evidence about the app under test — which made no HTTPS calls "
                    "during this window."
                )
            else:
                out["code"] = "proxy_no_traffic"
                out["hint"] = (
                    "Recorded 0 HTTP flows. Mitm saw no CONNECT or TLS activity at all "
                    "while recording. Check the device is actually pointed at this "
                    "proxy (`aua proxy status`) and that the app under test made HTTPS "
                    "calls during this window."
                )
        pm.clear_record_window(cache, target)
        return out
    raise UsageError(
        f"unknown mock record action {action!r}",
        hint="Use `aua mock record start NAME` or `aua mock record stop`.",
    )


def mock_replay(
    self: Engine,
    name: str,
    *,
    _snapshot: _ResolvedCassetteResource | None = None,
) -> dict[str, Any]:
    pm = self.platform.capability("proxy")

    cache = Path(self.config.cache.dir).expanduser()
    if _snapshot is None:
        path = pm.cassette_dir(self.config.memory.dir) / f"{name}.yaml"
        if not path.is_file():
            # also accept a direct path
            alt = Path(name).expanduser()
            path = alt if alt.is_file() else path
        entries = pm.load_cassette(path)
    else:
        path = _snapshot.source_path
        entries = deepcopy(_snapshot.entries)
    from . import leases

    owner = str(leases.resolve_owner(None))
    # Journal before the whole rule set is replaced: a crash right after this call must
    # still leave a stranger enough to clear it, same as `mock map`.
    self.record_device_change(
        key="mock_rules",
        kind="mock_rules",
        op="clear_mock_rules",
        args={"cache_dir": str(cache)},
        detail=f"cassette {name!r} loaded as live mock rules via `aua mock replay`",
    )
    pm.write_rules(pm.rules_path(cache, self._proxy_serial()), entries, owner=owner)
    self._record_action_safe(RouteStep(kind="mock-replay", arg=name))
    return {
        "ok": True,
        "action": "mock-replay",
        "name": name,
        "entries": len(entries),
        "path": str(path),
    }
