"""Engine operations for optional virtual-target discovery and provisioning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

from .errors import InvalidPlatformCapabilityError, UnsupportedPlatformCapabilityError
from .platforms.services import VirtualTargetsService
from .platforms.virtual_targets import (
    OwnedVirtualTargetStopRequest,
    VirtualTargetCreateRequest,
    VirtualTargetCreateResult,
    VirtualTargetDeleteRequest,
    VirtualTargetDeleteResult,
    VirtualTargetInstance,
    VirtualTargetList,
    VirtualTargetProvisionRequest,
    VirtualTargetReclaimRequest,
    VirtualTargetStartRequest,
    VirtualTargetStatus,
    VirtualTargetStopRequest,
    VirtualTargetStopResult,
    compatibility_result,
    neutral_result,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import Engine


_ResultT = TypeVar("_ResultT")


def _service(self: Engine) -> VirtualTargetsService:
    return self.platform.capability("virtual_targets")


def _typed_result(
    self: Engine,
    result: object,
    expected: type[_ResultT],
    *,
    operation: str,
) -> _ResultT:
    if not isinstance(result, expected):
        raise InvalidPlatformCapabilityError(
            self.platform.name,
            "virtual_targets",
            [f"{operation} return type {expected.__name__}"],
        )
    return result


def _owned_instance(self: Engine, result: object, *, operation: str) -> VirtualTargetInstance:
    instance = _typed_result(self, result, VirtualTargetInstance, operation=operation)
    if not instance.instance_token:
        raise InvalidPlatformCapabilityError(
            self.platform.name,
            "virtual_targets",
            [f"{operation} result.instance_token"],
        )
    return instance


def virtual_target_list(self: Engine, *, compatibility: bool = False) -> dict[str, Any]:
    """List reusable definitions exposed by the selected platform."""

    result = _typed_result(
        self,
        _service(self).list_virtual_targets(),
        VirtualTargetList,
        operation="list_virtual_targets",
    )
    neutral = neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-list",
    )
    if not compatibility:
        return neutral
    return compatibility_result(
        result,
        fallback={**neutral, "action": "emulator-list"},
    )


def virtual_target_status(self: Engine, *, compatibility: bool = False) -> dict[str, Any]:
    """Return configured, running, and AUA-owned virtual targets."""

    result = _typed_result(
        self,
        _service(self).virtual_target_status(cache_dir=self.config.cache.dir),
        VirtualTargetStatus,
        operation="virtual_target_status",
    )
    neutral = neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-status",
    )
    if not compatibility:
        return neutral
    return compatibility_result(
        result,
        fallback={**neutral, "action": "emulator-status"},
    )


def virtual_target_start(
    self: Engine,
    definition_id: str | None = None,
    *,
    headless: bool = True,
    audio: bool = False,
    animations: bool = False,
    wait_s: float = 120.0,
    owner: object | None = None,
    parallel: bool = False,
    options: dict[str, Any] | None = None,
    compatibility: bool = False,
) -> dict[str, Any]:
    """Start one virtual target and retain its exact rollback identity."""

    result = _owned_instance(
        self,
        _service(self).start_virtual_target(
            VirtualTargetStartRequest(
                definition_id=definition_id,
                headless=headless,
                audio=audio,
                animations=animations,
                wait_s=wait_s,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self.config.lease.registry_dir,
                owner=owner,
                parallel=parallel,
                options=dict(options or {}),
            )
        ),
        operation="start_virtual_target",
    )
    neutral = neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-start",
    )
    if not compatibility:
        return neutral
    return compatibility_result(
        result,
        fallback={
            **neutral,
            "action": "emulator-start",
            "serial": result.target_id,
            "avd": result.definition_id,
            "instance": result.instance_token,
        },
    )


def virtual_target_provision(
    self: Engine,
    definition_id: str | None = None,
    *,
    needs: list[str] | tuple[str, ...] | None = None,
    headless: bool = True,
    audio: bool = False,
    animations: bool = False,
    wait_s: float = 120.0,
    owner: object | None = None,
    parallel: bool = True,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select and start a compatible virtual target as one platform operation."""

    result = _owned_instance(
        self,
        _service(self).provision_virtual_target(
            VirtualTargetProvisionRequest(
                definition_id=definition_id,
                needs=tuple(str(item) for item in (needs or ())),
                headless=headless,
                audio=audio,
                animations=animations,
                wait_s=wait_s,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self.config.lease.registry_dir,
                owner=owner,
                parallel=parallel,
                options=dict(options or {}),
            )
        ),
        operation="provision_virtual_target",
    )
    return neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-provision",
    )


def virtual_target_create(
    self: Engine,
    definition_id: str,
    *,
    replace: bool = False,
    options: dict[str, Any] | None = None,
    compatibility: bool = False,
) -> dict[str, Any]:
    """Create or idempotently reuse a virtual-target definition."""

    result = _typed_result(
        self,
        _service(self).create_virtual_target(
            VirtualTargetCreateRequest(
                definition_id=definition_id,
                replace=replace,
                options=dict(options or {}),
            )
        ),
        VirtualTargetCreateResult,
        operation="create_virtual_target",
    )
    neutral = neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-create",
    )
    if not compatibility:
        return neutral
    return compatibility_result(
        result,
        fallback={**neutral, "action": "emulator-ensure-proxy"},
    )


def virtual_target_delete(
    self: Engine,
    definition_id: str,
    *,
    confirmed: bool = False,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Delete one stopped virtual-target definition after confirmation."""

    result = _typed_result(
        self,
        _service(self).delete_virtual_target(
            VirtualTargetDeleteRequest(
                definition_id=definition_id,
                confirmed=confirmed,
                options=dict(options or {}),
            )
        ),
        VirtualTargetDeleteResult,
        operation="delete_virtual_target",
    )
    return neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-delete",
    )


def virtual_target_stop(
    self: Engine,
    *,
    target_id: str | None = None,
    definition_id: str | None = None,
    owner: str | None = None,
    mine: bool = False,
    all_targets: bool = False,
    requested_by: str | None = None,
    lease_owner: object | None = None,
    compatibility: bool = False,
) -> dict[str, Any]:
    """Stop explicitly selected targets while respecting live foreign leases."""

    result = _typed_result(
        self,
        _service(self).stop_virtual_targets(
            VirtualTargetStopRequest(
                target_id=target_id,
                definition_id=definition_id,
                owner=owner,
                mine=mine,
                all_targets=all_targets,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self.config.lease.registry_dir,
                lease_owner=lease_owner,
                requested_by=requested_by,
            )
        ),
        VirtualTargetStopResult,
        operation="stop_virtual_targets",
    )
    neutral = neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-stop",
    )
    if not compatibility:
        return neutral
    return compatibility_result(
        result,
        fallback={
            **neutral,
            "action": "emulator-stop",
            "stopped": list(result.stopped_target_ids),
        },
    )


def virtual_target_stop_instance(
    self: Engine,
    instance_token: str,
    *,
    expected_pid: int | None = None,
    owner: object | None = None,
    requested_by: str | None = None,
) -> dict[str, Any]:
    """Roll back only the exact virtual-target instance this operation created."""

    result = _typed_result(
        self,
        _service(self).stop_virtual_target_instance(
            OwnedVirtualTargetStopRequest(
                instance_token=instance_token,
                expected_pid=expected_pid,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self.config.lease.registry_dir,
                owner=owner,
                requested_by=requested_by,
            )
        ),
        VirtualTargetStopResult,
        operation="stop_virtual_target_instance",
    )
    return neutral_result(
        result,
        platform=self.platform.name,
        action="virtual-target-stop-instance",
    )


def virtual_target_reclaim(self: Engine, *, idle_timeout_s: float) -> dict[str, Any]:
    """Re-arm platform retirement supervision for owned orphan instances."""

    raw = _service(self).reclaim_virtual_targets(
        VirtualTargetReclaimRequest(
            cache_dir=self.config.cache.dir,
            idle_timeout_s=idle_timeout_s,
            lease_registry_dir=self.config.lease.registry_dir,
        )
    )
    if not isinstance(raw, tuple) or not all(isinstance(item, VirtualTargetInstance) for item in raw):
        raise InvalidPlatformCapabilityError(
            self.platform.name,
            "virtual_targets",
            ["reclaim_virtual_targets return type tuple[VirtualTargetInstance, ...]"],
        )
    return {
        "ok": True,
        "action": "virtual-target-reclaim",
        "platform": self.platform.name,
        "reclaimed": [item.to_dict() for item in raw],
    }


# -- Established Android-shaped compatibility methods ---------------------


def emulator_list(self: Engine) -> dict[str, Any]:
    return virtual_target_list(self, compatibility=True)


def emulator_status(self: Engine) -> dict[str, Any]:
    return virtual_target_status(self, compatibility=True)


def emulator_start(
    self: Engine,
    avd: str | None = None,
    *,
    headless: bool = True,
    audio: bool = False,
    animations: bool = False,
    wait_s: float = 120.0,
    owner: object | None = None,
    parallel: bool = False,
    gpu: str | None = None,
    idle_timeout_s: float | None = None,
    port: int | None = None,
    read_only: bool | None = None,
) -> dict[str, Any]:
    options = {
        key: value
        for key, value in {
            "gpu": gpu,
            "idle_timeout_s": idle_timeout_s,
            "port": port,
            "read_only": read_only,
        }.items()
        if value is not None
    }
    return virtual_target_start(
        self,
        avd,
        headless=headless,
        audio=audio,
        animations=animations,
        wait_s=wait_s,
        owner=owner,
        parallel=parallel,
        options=options,
        compatibility=True,
    )


def emulator_stop(
    self: Engine,
    *,
    serial: str | None = None,
    avd: str | None = None,
    owner: str | None = None,
    mine: bool = False,
    all_devices: bool = False,
    requested_by: str | None = None,
    lease_owner: object | None = None,
) -> dict[str, Any]:
    return virtual_target_stop(
        self,
        target_id=serial,
        definition_id=avd,
        owner=owner,
        mine=mine,
        all_targets=all_devices,
        requested_by=requested_by,
        lease_owner=lease_owner,
        compatibility=True,
    )


def emulator_recommend_proxy(
    self: Engine, *, api: int, name: str
) -> dict[str, Any]:
    """Legacy Android-only recommendation, retained behind the selected capability."""

    service = _service(self)
    operation = getattr(service, "recommend_proxy_avd", None)
    if not callable(operation):
        raise UnsupportedPlatformCapabilityError(self.platform.name, "emulator")
    return cast(dict[str, Any], operation(api=api, name=name))


def emulator_ensure_proxy(
    self: Engine,
    *,
    name: str,
    api: int,
    force: bool,
) -> dict[str, Any]:
    return virtual_target_create(
        self,
        name,
        replace=force,
        options={"profile": "proxy", "api": api},
        compatibility=True,
    )
