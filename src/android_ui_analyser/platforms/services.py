"""Stable structural contracts for optional platform capability services.

The neutral ``TargetRuntime`` covers common screen/input/app operations. These named services
cover operations that are host-wide or not universal. A plugin may implement only the services
it supports, but claiming a service means implementing its complete structural surface below.
"""

from __future__ import annotations

from collections.abc import Sequence
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from .supervision import TargetSupervisionStatus
from .virtual_targets import (
    OwnedVirtualTargetStopRequest,
    VirtualTargetCreateRequest,
    VirtualTargetCreateResult,
    VirtualTargetDefinition,
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
)

APP_DATABASE: Final = "app_database"
DEVICE_AGENT: Final = "device_agent"
DEVELOPER_SETTINGS: Final = "developer_settings"
FEATURE_FLAGS: Final = "feature_flags"
MICROPHONE: Final = "microphone"
NETWORK: Final = "network"
NETWORK_PROFILES: Final = "network_profiles"
PROXY: Final = "proxy"
TARGET_SUPERVISION: Final = "target_supervision"
# ``virtual_targets`` is the API-v1 capability.  The capability normalizer accepts the
# historical ``virtual_devices``/``emulator`` spellings, and the public constants remain so
# downstream imports do not need a flag day.
VIRTUAL_TARGETS: Final = "virtual_targets"
VIRTUAL_DEVICES: Final = "virtual_devices"
WEBVIEW: Final = "webview"


@runtime_checkable
class VirtualTargetsService(Protocol):
    """Typed host-service contract for simulator/emulator provisioning.

    The request and result values deliberately contain no native process or command grammar.
    Plugins implement this protocol structurally and return it from
    :meth:`PlatformAdapter.load_capability` when declaring ``virtual_targets``.
    """

    def list_virtual_targets(self) -> VirtualTargetList: ...

    def select_virtual_target(
        self,
        definition_id: str | None = None,
        *,
        needs: Sequence[str] | None = None,
    ) -> VirtualTargetDefinition: ...

    def start_virtual_target(
        self, request: VirtualTargetStartRequest
    ) -> VirtualTargetInstance: ...

    def provision_virtual_target(
        self, request: VirtualTargetProvisionRequest
    ) -> VirtualTargetInstance: ...

    def virtual_target_status(
        self, *, cache_dir: str | Path | None
    ) -> VirtualTargetStatus: ...

    def stop_virtual_targets(
        self, request: VirtualTargetStopRequest
    ) -> VirtualTargetStopResult: ...

    def stop_virtual_target_instance(
        self, request: OwnedVirtualTargetStopRequest
    ) -> VirtualTargetStopResult: ...

    def reclaim_virtual_targets(
        self, request: VirtualTargetReclaimRequest
    ) -> tuple[VirtualTargetInstance, ...]: ...

    def create_virtual_target(
        self, request: VirtualTargetCreateRequest
    ) -> VirtualTargetCreateResult: ...

    def delete_virtual_target(
        self, request: VirtualTargetDeleteRequest
    ) -> VirtualTargetDeleteResult: ...


@runtime_checkable
class TargetSupervisionService(Protocol):
    """Typed host-service contract for optional AUA-owned target lifecycle metadata."""

    def target_supervision_status(
        self,
        target_id: str,
        *,
        cache_dir: str | Path,
    ) -> TargetSupervisionStatus | None: ...


CAPABILITY_PROTOCOLS: dict[str, type[Any]] = {
    VIRTUAL_TARGETS: VirtualTargetsService,
    TARGET_SUPERVISION: TargetSupervisionService,
}

CAPABILITY_METHODS: dict[str, frozenset[str]] = {
    APP_DATABASE: frozenset(
        {
            "backup_database",
            "database_schema",
            "execute_database",
            "list_backups",
            "list_databases",
            "query_database",
            "restore_database",
        }
    ),
    # An optional in-target agent: a platform-side process AUA can talk to instead of
    # polling from the host. Android backs it with an AccessibilityService APK; another
    # platform could back it with whatever its runtime offers, or not claim it at all.
    DEVICE_AGENT: frozenset(
        {
            "HelperUnavailableError",
            "disable",
            "discard_touch_capture",
            "enable",
            "install",
            "is_bound",
            "is_enabled",
            "is_installed",
            "open_channel",
            "release_uiautomation",
            "remove",
            "restore_state",
            "root_available",
            "rootable",
            "snapshot_state",
            "start_touch_capture",
            "status",
            "stop_touch_capture",
            "tree_to_xml",
            "uiautomation_held",
        }
    ),
    DEVELOPER_SETTINGS: frozenset(
        {"anim_off", "anim_restore", "crashes_set", "profile_ac", "profile_default", "read_state"}
    ),
    FEATURE_FLAGS: frozenset(
        {
            "build_uri",
            "dump_result",
            "load_flags_file",
            "parse_assignments",
            "read_context_flags",
            "read_prefs",
            "restore_prefs",
            "save_prefs_backup",
            "snapshot_prefs",
            "write_prefs",
        }
    ),
    MICROPHONE: frozenset(
        {
            "MicDeliveredReleaseError",
            "MicDeliveryUncertainError",
            "MicToggleStartUncertainError",
            "MicToggleStopUncertainError",
            "claim_injection_attempt",
            "inject_prepared",
            "inspect_pcm_wav",
            "prepare_injection",
            "synthesize_speech",
            "validate_control_mode",
        }
    ),
    NETWORK: frozenset(
        {
            "apply_offline_controls",
            "backup_path",
            "load_backup",
            "offline_verified",
            "read_network_state",
            "require_current_backup",
            "restore_controls",
            "restored_verified",
            "save_backup",
            "wait_for_state",
        }
    ),
    NETWORK_PROFILES: frozenset(
        {
            "PROFILE_NAMES",
            "apply_radio_profile",
            "load_profile",
            "normalize_profile",
            "prepare_loss",
            "profile_path",
            "profile_verified",
            "qdisc_evidence",
            "read_emulator_shape",
            "remove_loss",
            "require_current_profile",
            "restore_emulator_shape",
            "restore_root",
            "restore_radio_profile",
            "root_enabled",
            "safe_unroot_after_failed_apply",
            "save_profile",
            "set_emulator_shape",
            "set_loss",
            "shape_matches",
            "stale_profile",
            "wait_for_radio_profile",
        }
    ),
    # The ownership members are part of the contract, not an optional extra: the device's proxy
    # is a *device-global* setting pointing at a *non-persistent* host process, so a platform
    # claiming this capability must be able to say who owns it and whether that owner is dead.
    # Without them a parallel agent silently inherits a proxied device it cannot see or fix.
    PROXY: frozenset(
        {
            "backfill_rule_ids",
            "cassette_dir",
            "clear_record_window",
            "clear_rules",
            "clear_state",
            "diagnose_empty_recording",
            # Health-check trio: a platform claiming PROXY must be able to say — together, not
            # separately — whether the process is alive, the tunnel is reachable, and the
            # device setting points at it. See `proxy_mock.proxy_health`.
            "ensure_reverse_tunnel",
            "flow_matches",
            "guard_rule_scope",
            "install_system_ca",
            "load_cassette",
            "load_doc",
            "load_listen_port",
            "load_record",
            "load_record_window",
            "load_rules",
            "map_rule",
            "orphan_reason",
            # A device pointed at a proxy nobody owns is diagnosable *only* from the device's
            # own setting, so a platform claiming PROXY must be able to say what that setting
            # means — which host, which port, and whether the host is even on this machine.
            # Without it the generic layer cannot tell a black hole from a clean device.
            "parse_proxy_target",
            "proxy_health",
            "read_device_http_proxy",
            "read_flow_bodies",
            "read_flows_since",
            "read_state",
            "record_path",
            "reset_record",
            "reverse_tunnel_active",
            "rewrite_rule",
            "rules_path",
            "save_cassette",
            "save_doc",
            "save_record_window",
            "start_mitm",
            "stop_mitm",
            "tls_failures_in_log",
            "write_rules",
            "write_state",
        }
    ),
    # Optional lifecycle metadata for targets started and monitored by AUA. This is separate
    # from virtual-target provisioning: the dashboard needs to describe ownership and automatic
    # retirement without knowing how a platform stores process records or names its monitor.
    TARGET_SUPERVISION: frozenset({"target_supervision_status"}),
    VIRTUAL_TARGETS: frozenset(
        {
            "create_virtual_target",
            "delete_virtual_target",
            "list_virtual_targets",
            "provision_virtual_target",
            # A platform that can boot throwaway targets must also be able to reclaim the ones
            # its own supervisor lost track of, or "aua started it" becomes "aua leaked it".
            "reclaim_virtual_targets",
            "select_virtual_target",
            "start_virtual_target",
            "stop_virtual_targets",
            # Rollback teardown is scoped to the opaque token returned by the exact boot. A bare
            # target id can name a foreign instance after a provisioning collision.
            "stop_virtual_target_instance",
            "virtual_target_status",
        }
    ),
    WEBVIEW: frozenset({"enrich", "should_try_webview"}),
}

# Structural services are operations except for these deliberately published data attributes.
# A present ``None`` operation is not an implementation and must fail at capability resolution.
CAPABILITY_DATA_MEMBERS: dict[str, frozenset[str]] = {
    NETWORK_PROFILES: frozenset({"PROFILE_NAMES"}),
}


def missing_members(capability: str, service: Any) -> list[str]:
    """Members absent or non-callable on a service that claims *capability*."""

    data_members = CAPABILITY_DATA_MEMBERS.get(capability, frozenset())
    missing: list[str] = []
    for name in CAPABILITY_METHODS.get(capability, ()):
        value = getattr(service, name, None)
        if value is None or (name not in data_members and not callable(value)):
            missing.append(name)
            continue
        protocol = CAPABILITY_PROTOCOLS.get(capability)
        contract = getattr(protocol, name, None) if protocol is not None else None
        if contract is not None and not _signature_compatible(value, contract):
            missing.append(f"{name} signature")
    return sorted(missing)


def _signature_compatible(implementation: Any, contract: Any) -> bool:
    """Whether a bound service operation accepts the Protocol's public call shape."""

    try:
        actual = list(signature(implementation).parameters.values())
        expected = list(signature(contract).parameters.values())
    except (TypeError, ValueError):
        return False
    if expected and expected[0].name == "self":
        expected = expected[1:]
    actual_by_name = {parameter.name: parameter for parameter in actual}
    has_args = any(parameter.kind is Parameter.VAR_POSITIONAL for parameter in actual)
    has_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in actual)

    for parameter in expected:
        if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
            continue
        candidate = actual_by_name.get(parameter.name)
        if candidate is None:
            if parameter.kind is Parameter.POSITIONAL_ONLY and has_args:
                continue
            if parameter.kind is Parameter.KEYWORD_ONLY and has_kwargs:
                continue
            if parameter.kind is Parameter.POSITIONAL_OR_KEYWORD and has_args and has_kwargs:
                continue
            return False
        allowed_kinds: set[Any]
        if parameter.kind is Parameter.POSITIONAL_ONLY:
            allowed_kinds = {Parameter.POSITIONAL_ONLY}
        elif parameter.kind is Parameter.POSITIONAL_OR_KEYWORD:
            allowed_kinds = {Parameter.POSITIONAL_OR_KEYWORD}
        elif parameter.kind is Parameter.KEYWORD_ONLY:
            allowed_kinds = {
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
            }
        else:  # pragma: no cover - variadic kinds were skipped above
            return False
        if candidate.kind not in allowed_kinds:
            return False
        if parameter.default is not Parameter.empty and candidate.default is Parameter.empty:
            return False

    expected_names = {parameter.name for parameter in expected}
    for parameter in actual:
        if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
            continue
        if parameter.name not in expected_names and parameter.default is Parameter.empty:
            return False
    return True


__all__ = [
    "APP_DATABASE",
    "CAPABILITY_METHODS",
    "CAPABILITY_PROTOCOLS",
    "DEVELOPER_SETTINGS",
    "DEVICE_AGENT",
    "FEATURE_FLAGS",
    "MICROPHONE",
    "NETWORK",
    "NETWORK_PROFILES",
    "PROXY",
    "TARGET_SUPERVISION",
    "VIRTUAL_DEVICES",
    "VIRTUAL_TARGETS",
    "WEBVIEW",
    "TargetSupervisionService",
    "VirtualTargetsService",
]
