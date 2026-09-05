from __future__ import annotations

from types import SimpleNamespace

import android_ui_analyser.platforms as public_platforms
from android_ui_analyser.platforms import (
    TARGET_SUPERVISION,
    VIRTUAL_TARGETS,
    TargetSupervisionService,
    VirtualTargetsService,
)
from android_ui_analyser.platforms.contracts import (
    ADAPTER_CAPABILITIES,
    RUNTIME_CAPABILITIES,
    CapabilityScope,
    missing_structural_members,
    normalize_capability,
)
from android_ui_analyser.platforms.services import CAPABILITY_METHODS, missing_members


class _Defaults:
    def operation(self) -> None:
        raise NotImplementedError


class _InheritedDefault(_Defaults):
    pass


class _Implemented(_Defaults):
    def operation(self) -> None:
        return None


def test_direct_capability_names_have_one_canonical_spelling() -> None:
    specs = {**RUNTIME_CAPABILITIES, **ADAPTER_CAPABILITIES}

    assert specs
    assert all(name == spec.name == normalize_capability(name) for name, spec in specs.items())
    assert all(spec.members for spec in specs.values())
    assert all(spec.scope in {CapabilityScope.RUNTIME, CapabilityScope.ADAPTER} for spec in specs.values())


def test_structural_validation_rejects_absent_and_non_callable_members() -> None:
    class Partial:
        present = None

        def callable(self) -> None:
            return None

    assert missing_structural_members(Partial(), frozenset({"missing", "present", "callable"})) == [
        "missing",
        "present",
    ]


def test_capability_names_are_normalized_once() -> None:
    assert normalize_capability(" Virtual-Targets ") == "virtual_targets"


def test_structural_validation_rejects_inherited_optional_stubs() -> None:
    members = frozenset({"operation"})

    assert missing_structural_members(
        _InheritedDefault(), members, default_owner=_Defaults
    ) == ["operation"]
    assert missing_structural_members(_Implemented(), members, default_owner=_Defaults) == []


def test_typed_host_service_protocols_match_the_runtime_capability_catalogue() -> None:
    virtual_operations = {
        name
        for name, value in VirtualTargetsService.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    supervision_operations = {
        name
        for name, value in TargetSupervisionService.__dict__.items()
        if not name.startswith("_") and callable(value)
    }

    assert virtual_operations == CAPABILITY_METHODS[VIRTUAL_TARGETS]
    assert supervision_operations == CAPABILITY_METHODS[TARGET_SUPERVISION]


def test_typed_host_service_protocols_are_structural_at_runtime() -> None:
    virtual = SimpleNamespace(
        **{name: (lambda *args, **kwargs: None) for name in CAPABILITY_METHODS[VIRTUAL_TARGETS]}
    )
    supervision = SimpleNamespace(target_supervision_status=lambda *args, **kwargs: None)

    assert isinstance(virtual, VirtualTargetsService)
    assert isinstance(supervision, TargetSupervisionService)
    assert not isinstance(SimpleNamespace(), VirtualTargetsService)
    assert not isinstance(SimpleNamespace(target_supervision_status=None), TargetSupervisionService)


def test_typed_service_contract_rejects_a_callable_with_the_wrong_signature() -> None:
    service = SimpleNamespace(
        **{name: (lambda *args, **kwargs: None) for name in CAPABILITY_METHODS[VIRTUAL_TARGETS]}
    )
    service.start_virtual_target = lambda: None

    assert missing_members(VIRTUAL_TARGETS, service) == ["start_virtual_target signature"]


def test_public_platform_facade_exports_plugin_contract_types_and_service_values() -> None:
    expected = {
        "AppBundle",
        "AppContext",
        "AppExitEvidence",
        "AttachedTargetCase",
        "AttachedTargetReport",
        "Bounds",
        "DiagnosticEvent",
        "DiagnosticLevel",
        "DiagnosticSourcePolicy",
        "DiagnosticWindow",
        "DiscoveredTarget",
        "DisplayGeometry",
        "InstalledApp",
        "Element",
        "MatchMode",
        "NormalizedTree",
        "PlatformAdapter",
        "PlatformConformanceError",
        "ScreenImage",
        "ShellResult",
        "TargetInfo",
        "TargetRuntime",
        "TargetSupervisionService",
        "TargetSupervisionStatus",
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
        "VirtualTargetsService",
    }

    assert expected <= set(public_platforms.__all__)
    assert all(getattr(public_platforms, name, None) is not None for name in expected)
