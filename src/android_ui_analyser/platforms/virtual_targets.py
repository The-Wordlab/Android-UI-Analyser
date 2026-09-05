"""Platform-neutral contracts for optional virtual-target provisioning.

The selected platform owns how a virtual target is defined and started.  Shared code only
needs three stable identities:

* ``definition_id`` identifies a reusable simulator/emulator definition;
* ``target_id`` identifies the attached automation target after startup;
* ``instance_token`` identifies the exact process/boot AUA created.

The last value is deliberately mandatory for a successful start.  A target id can be reused or
collide with another worker, so failure rollback must never stop by target id alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JsonObject = Mapping[str, Any]


def _required(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _json_copy(value: JsonObject | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True, slots=True)
class VirtualTargetDefinition:
    """One reusable platform-owned virtual-target definition."""

    definition_id: str
    capabilities: Mapping[str, bool] = field(default_factory=dict)
    details: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_id",
            _required(self.definition_id, field_name="definition_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "capabilities": {
                str(name): bool(enabled) for name, enabled in self.capabilities.items()
            },
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetInstance:
    """One running target known to a virtual-target service.

    ``instance_token`` may be absent for a target discovered rather than created by AUA. A
    successful ``start`` or ``provision`` result must provide one; the Engine enforces that at
    the operation boundary before it can become rollback state.
    """

    target_id: str
    instance_token: str | None = None
    definition_id: str | None = None
    owner: str | None = None
    pid: int | None = None
    details: JsonObject = field(default_factory=dict)
    # Existing Android commands have a long-lived JSON contract.  Android wrappers keep their
    # exact payload here so compatibility aliases can return it byte-for-byte in field content,
    # while neutral callers see only the fields above.
    legacy_result: JsonObject | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _required(self.target_id, field_name="target_id"))
        if self.instance_token is not None:
            object.__setattr__(
                self,
                "instance_token",
                _required(self.instance_token, field_name="instance_token"),
            )
        if self.definition_id is not None:
            object.__setattr__(
                self,
                "definition_id",
                _required(self.definition_id, field_name="definition_id"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "instance_token": self.instance_token,
            "definition_id": self.definition_id,
            "owner": self.owner,
            "pid": self.pid,
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetList:
    definitions: tuple[VirtualTargetDefinition, ...]
    details: JsonObject = field(default_factory=dict)
    legacy_result: JsonObject | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [item.to_dict() for item in self.definitions],
            "count": len(self.definitions),
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetStatus:
    definitions: tuple[VirtualTargetDefinition, ...] = ()
    running: tuple[VirtualTargetInstance, ...] = ()
    owned: tuple[VirtualTargetInstance, ...] = ()
    details: JsonObject = field(default_factory=dict)
    legacy_result: JsonObject | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [item.to_dict() for item in self.definitions],
            "running": [item.to_dict() for item in self.running],
            "owned": [item.to_dict() for item in self.owned],
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetStopResult:
    stopped_target_ids: tuple[str, ...] = ()
    preserved_target_ids: tuple[str, ...] = ()
    details: JsonObject = field(default_factory=dict)
    legacy_result: JsonObject | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped_target_ids": list(self.stopped_target_ids),
            "preserved_target_ids": list(self.preserved_target_ids),
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetCreateResult:
    definition: VirtualTargetDefinition
    created: bool
    details: JsonObject = field(default_factory=dict)
    legacy_result: JsonObject | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "target": self.definition.to_dict(),
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetDeleteResult:
    definition_id: str
    deleted: bool
    details: JsonObject = field(default_factory=dict)
    legacy_result: JsonObject | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_id",
            _required(self.definition_id, field_name="definition_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "deleted": self.deleted,
            "details": _json_copy(self.details),
        }


@dataclass(frozen=True, slots=True)
class VirtualTargetStartRequest:
    definition_id: str | None = None
    headless: bool = True
    audio: bool = False
    animations: bool = False
    wait_s: float = 120.0
    cache_dir: str | Path = "."
    lease_registry_dir: str | Path | None = None
    owner: object | None = None
    parallel: bool = False
    options: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VirtualTargetProvisionRequest(VirtualTargetStartRequest):
    needs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VirtualTargetCreateRequest:
    definition_id: str
    replace: bool = False
    options: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_id",
            _required(self.definition_id, field_name="definition_id"),
        )


@dataclass(frozen=True, slots=True)
class VirtualTargetDeleteRequest:
    definition_id: str
    confirmed: bool = False
    options: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_id",
            _required(self.definition_id, field_name="definition_id"),
        )


@dataclass(frozen=True, slots=True)
class VirtualTargetStopRequest:
    target_id: str | None = None
    definition_id: str | None = None
    owner: str | None = None
    mine: bool = False
    all_targets: bool = False
    cache_dir: str | Path = "."
    lease_registry_dir: str | Path | None = None
    lease_owner: object | None = None
    requested_by: str | None = None


@dataclass(frozen=True, slots=True)
class OwnedVirtualTargetStopRequest:
    """Exact rollback request for one boot returned by ``start``/``provision``."""

    instance_token: str
    expected_pid: int | None = None
    cache_dir: str | Path = "."
    lease_registry_dir: str | Path | None = None
    owner: object | None = None
    requested_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instance_token",
            _required(self.instance_token, field_name="instance_token"),
        )


@dataclass(frozen=True, slots=True)
class VirtualTargetReclaimRequest:
    cache_dir: str | Path
    idle_timeout_s: float
    lease_registry_dir: str | Path | None = None


def neutral_result(
    result: VirtualTargetList
    | VirtualTargetStatus
    | VirtualTargetInstance
    | VirtualTargetStopResult
    | VirtualTargetCreateResult
    | VirtualTargetDeleteResult,
    *,
    platform: str,
    action: str,
) -> dict[str, Any]:
    """Serialize a typed service result at the shared Engine boundary."""

    return {
        "ok": True,
        "action": action,
        "platform": platform,
        **result.to_dict(),
    }


def compatibility_result(
    result: VirtualTargetList
    | VirtualTargetStatus
    | VirtualTargetInstance
    | VirtualTargetStopResult
    | VirtualTargetCreateResult
    | VirtualTargetDeleteResult,
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Use an adapter's exact historical payload when it supplied one."""

    if result.legacy_result is not None:
        return dict(result.legacy_result)
    return fallback


__all__ = [
    "OwnedVirtualTargetStopRequest",
    "VirtualTargetCreateRequest",
    "VirtualTargetCreateResult",
    "VirtualTargetDefinition",
    "VirtualTargetDeleteRequest",
    "VirtualTargetDeleteResult",
    "VirtualTargetInstance",
    "VirtualTargetList",
    "VirtualTargetProvisionRequest",
    "VirtualTargetReclaimRequest",
    "VirtualTargetStartRequest",
    "VirtualTargetStatus",
    "VirtualTargetStopRequest",
    "VirtualTargetStopResult",
    "compatibility_result",
    "neutral_result",
]
