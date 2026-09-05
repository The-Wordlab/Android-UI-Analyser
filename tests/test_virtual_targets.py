"""Platform-neutral virtual-target contract, transports, and Android compatibility."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import emulator, leases, mcp_server
from android_ui_analyser.cli import GlobalOpts, app
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    DeviceError,
    InvalidPlatformCapabilityError,
    UnsupportedPlatformCapabilityError,
)
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from android_ui_analyser.mcp_server import _tool_definitions
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.virtual_targets import (
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
from android_ui_analyser.schema import TargetInfo
from conftest import make_config


class _StrictVirtualTargets:
    """A complete fake service with no AVD, serial, pid, or Android convenience methods."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.definitions = (
            VirtualTargetDefinition(
                definition_id="small-simulator",
                capabilities={"camera": True},
                details={"family": "phone"},
            ),
        )

    def list_virtual_targets(self) -> VirtualTargetList:
        self.calls.append(("list", None))
        return VirtualTargetList(self.definitions)

    def select_virtual_target(
        self, definition_id: str | None = None, *, needs: Sequence[str] = ()
    ) -> VirtualTargetDefinition:
        self.calls.append(("select", (definition_id, tuple(needs))))
        return self.definitions[0]

    def start_virtual_target(
        self, request: VirtualTargetStartRequest
    ) -> VirtualTargetInstance:
        self.calls.append(("start", request))
        return VirtualTargetInstance(
            target_id="attached-target",
            definition_id=request.definition_id or self.definitions[0].definition_id,
            instance_token="owned-boot-token",
            owner=str(request.owner) if request.owner is not None else None,
        )

    def provision_virtual_target(
        self, request: VirtualTargetProvisionRequest
    ) -> VirtualTargetInstance:
        self.calls.append(("provision", request))
        return VirtualTargetInstance(
            target_id="attached-target",
            definition_id=request.definition_id or self.definitions[0].definition_id,
            instance_token="owned-boot-token",
            owner=str(request.owner) if request.owner is not None else None,
        )

    def virtual_target_status(self, *, cache_dir: str | Path) -> VirtualTargetStatus:
        self.calls.append(("status", cache_dir))
        return VirtualTargetStatus(definitions=self.definitions)

    def stop_virtual_targets(
        self, request: VirtualTargetStopRequest
    ) -> VirtualTargetStopResult:
        self.calls.append(("stop", request))
        return VirtualTargetStopResult(
            stopped_target_ids=((request.target_id,) if request.target_id else ())
        )

    def stop_virtual_target_instance(
        self, request: OwnedVirtualTargetStopRequest
    ) -> VirtualTargetStopResult:
        self.calls.append(("stop_instance", request))
        return VirtualTargetStopResult(stopped_target_ids=("attached-target",))

    def reclaim_virtual_targets(
        self, request: VirtualTargetReclaimRequest
    ) -> tuple[VirtualTargetInstance, ...]:
        self.calls.append(("reclaim", request))
        return ()

    def create_virtual_target(
        self, request: VirtualTargetCreateRequest
    ) -> VirtualTargetCreateResult:
        self.calls.append(("create", request))
        return VirtualTargetCreateResult(
            definition=VirtualTargetDefinition(request.definition_id),
            created=True,
        )

    def delete_virtual_target(
        self, request: VirtualTargetDeleteRequest
    ) -> VirtualTargetDeleteResult:
        self.calls.append(("delete", request))
        return VirtualTargetDeleteResult(request.definition_id, deleted=True)


class _FakePlatform(PlatformAdapter):
    name = "strict-fake"
    capabilities = frozenset({"virtual_targets"})

    def __init__(self, config: Config, service: _StrictVirtualTargets | None = None) -> None:
        super().__init__(config)
        self.service = service or _StrictVirtualTargets()

    def connect(self, target_id: str | None = None) -> Any:
        raise AssertionError("virtual-target host operations must not connect")

    def list_targets(self) -> list[TargetInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree([])

    def load_capability(self, capability: str) -> object | None:
        return self.service if capability == "virtual_targets" else None


class _AttachedOnlyPlatform(_FakePlatform):
    name = "attached-only"
    capabilities = frozenset()

    def load_capability(self, capability: str) -> object | None:
        raise AssertionError(f"unsupported capability attempted to load: {capability}")


def _engine(tmp_path: Path, service: _StrictVirtualTargets | None = None) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        lease={"enabled": False, "registry_dir": str(tmp_path / "leases")},
        memory={"enabled": False},
    )
    return Engine(cfg, platform=_FakePlatform(cfg, service))


def test_fake_platform_provisions_with_neutral_request_and_owned_token(tmp_path: Path) -> None:
    service = _StrictVirtualTargets()
    engine = _engine(tmp_path, service)

    result = engine.virtual_target_provision(
        "small-simulator",
        needs=["camera"],
        headless=False,
        audio=True,
        owner="worker-a",
        options={"runtime": "latest"},
    )

    assert result == {
        "ok": True,
        "action": "virtual-target-provision",
        "platform": "strict-fake",
        "target_id": "attached-target",
        "instance_token": "owned-boot-token",
        "definition_id": "small-simulator",
        "owner": "worker-a",
        "pid": None,
        "details": {},
    }
    request = dict(service.calls)["provision"]
    assert isinstance(request, VirtualTargetProvisionRequest)
    assert request.needs == ("camera",)
    assert request.options == {"runtime": "latest"}


def test_engine_exposes_the_complete_neutral_virtual_target_lifecycle(tmp_path: Path) -> None:
    service = _StrictVirtualTargets()
    engine = _engine(tmp_path, service)

    assert engine.virtual_target_list()["action"] == "virtual-target-list"
    assert engine.virtual_target_status()["action"] == "virtual-target-status"
    assert engine.virtual_target_start("small-simulator")["action"] == "virtual-target-start"
    assert engine.virtual_target_create("new-simulator")["action"] == "virtual-target-create"
    assert (
        engine.virtual_target_delete("old-simulator", confirmed=True)["action"]
        == "virtual-target-delete"
    )
    assert (
        engine.virtual_target_stop(target_id="attached-target")["action"]
        == "virtual-target-stop"
    )
    assert (
        engine.virtual_target_stop_instance("owned-boot-token")["action"]
        == "virtual-target-stop-instance"
    )
    assert engine.virtual_target_reclaim(idle_timeout_s=30)["action"] == "virtual-target-reclaim"

    calls = dict(service.calls)
    assert isinstance(calls["start"], VirtualTargetStartRequest)
    assert isinstance(calls["create"], VirtualTargetCreateRequest)
    assert isinstance(calls["delete"], VirtualTargetDeleteRequest)
    assert isinstance(calls["stop"], VirtualTargetStopRequest)
    assert isinstance(calls["stop_instance"], OwnedVirtualTargetStopRequest)
    assert isinstance(calls["reclaim"], VirtualTargetReclaimRequest)


def test_claimed_virtual_target_service_must_return_the_published_result_types(
    tmp_path: Path,
) -> None:
    class WrongResultService(_StrictVirtualTargets):
        def list_virtual_targets(self) -> Any:
            return []

    engine = _engine(tmp_path, WrongResultService())

    with pytest.raises(InvalidPlatformCapabilityError, match="VirtualTargetList"):
        engine.virtual_target_list()


def test_mcp_publishes_every_neutral_virtual_target_transport() -> None:
    names = {tool.name for tool in _tool_definitions()}
    assert {
        "virtual_target_list",
        "virtual_target_status",
        "virtual_target_start",
        "virtual_target_provision",
        "virtual_target_create",
        "virtual_target_delete",
        "virtual_target_stop",
        "virtual_target_reclaim",
    } <= names


def test_attached_only_platform_gets_typed_refusal_without_loading_android(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        lease={"enabled": False},
        memory={"enabled": False},
    )
    engine = Engine(cfg, platform=_AttachedOnlyPlatform(cfg))

    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        engine.virtual_target_list()

    assert caught.value.code == "platform_capability_unsupported"
    assert "virtual_targets" in caught.value.message


def test_cli_and_mcp_preserve_typed_refusal_for_attached_only_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        lease={"enabled": False},
        memory={"enabled": False},
    )
    engine = Engine(cfg, platform=_AttachedOnlyPlatform(cfg))
    monkeypatch.setattr(GlobalOpts, "engine", lambda _self: engine)

    cli = CliRunner().invoke(app, ["virtual-target", "list"])
    assert cli.exit_code == 2
    assert "platform_capability_unsupported" in cli.stderr

    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        mcp_dispatch(engine, "virtual_target_list", {})
    assert caught.value.code == "platform_capability_unsupported"


def test_cli_and_mcp_list_use_the_same_engine_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _StrictVirtualTargets()
    engine = _engine(tmp_path, service)
    monkeypatch.setattr(GlobalOpts, "engine", lambda _self: engine)
    monkeypatch.setattr(
        emulator,
        "list_avds",
        lambda: pytest.fail("neutral transports imported Android's virtual-target backend"),
    )

    cli = CliRunner().invoke(app, ["--format", "compact", "virtual-target", "list"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout)["action"] == "virtual-target-list"

    mcp = mcp_dispatch(engine, "virtual_target_list", {})
    assert mcp["action"] == "virtual-target-list"
    assert [name for name, _request in service.calls] == ["list", "list"]


def test_mcp_cleanup_checks_the_selected_platform_lease_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        lease={"enabled": False, "registry_dir": str(tmp_path / "leases")},
        memory={"enabled": False},
    )
    service = _StrictVirtualTargets()
    platform = _FakePlatform(cfg, service)
    checked_platforms: list[str] = []

    def fake_read_lease(
        _registry: str | Path,
        _target_id: object,
        *,
        platform: str = "android",
    ) -> None:
        checked_platforms.append(platform)
        return None

    monkeypatch.setattr(leases, "read_lease", fake_read_lease)
    mcp_server._MCP_STARTED_SERIALS.clear()
    mcp_server._MCP_STARTED_OWNERS.clear()
    mcp_server._MCP_STARTED_SERIALS.add("attached-target")

    result = mcp_server.cleanup_mcp_emulators(
        tmp_path / "cache",
        platform=platform,
        lease_registry_dir=tmp_path / "leases",
    )

    assert checked_platforms == ["strict-fake"]
    assert result["stopped"] == ["attached-target"]


def test_session_provisioning_uses_neutral_service_and_exact_rollback_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _StrictVirtualTargets()
    engine = _engine(tmp_path, service)
    claimed = iter([None, "foreign-target"])
    monkeypatch.setattr(engine, "_lease_device", lambda **_kwargs: next(claimed))

    with pytest.raises(DeviceError, match="started attached-target but leased foreign-target"):
        engine._prepare_session_target(
            wait_for_lease_s=0,
            provision_target=True,
            headed=False,
            audio=False,
            virtual_target="small-simulator",
        )

    rollback = [request for name, request in service.calls if name == "stop_instance"]
    assert len(rollback) == 1
    assert isinstance(rollback[0], OwnedVirtualTargetStopRequest)
    assert rollback[0].instance_token == "owned-boot-token"
    assert all(name != "stop" for name, _request in service.calls)


def test_android_neutral_list_preserves_exact_emulator_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = {
        "ok": True,
        "action": "emulator-list",
        "emulator": "/sdk/emulator",
        "avds": ["rootable-api34"],
        "details": [
            {
                "name": "rootable-api34",
                "rootable": True,
                "play_store": False,
            }
        ],
        "rootable": ["rootable-api34"],
        "play_store": [],
        "count": 1,
        "hint": None,
        "recommend_proxy": None,
    }
    monkeypatch.setattr(emulator, "list_avds", lambda: dict(legacy))

    neutral = emulator.list_virtual_targets()

    assert neutral.definitions[0].definition_id == "rootable-api34"
    assert neutral.definitions[0].capabilities == {
        "root": True,
        "proxy": True,
        "play": False,
    }
    assert dict(neutral.legacy_result or {}) == legacy


def test_android_engine_alias_returns_the_exact_existing_list_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = {
        "ok": True,
        "action": "emulator-list",
        "emulator": "/sdk/emulator",
        "avds": [],
        "details": [],
        "rootable": [],
        "play_store": [],
        "count": 0,
        "hint": "unchanged",
        "recommend_proxy": None,
    }
    monkeypatch.setattr(emulator, "list_avds", lambda: dict(legacy))
    engine = Engine(
        make_config(
            cache={"dir": str(tmp_path / "cache")},
            memory={"enabled": False},
        )
    )

    assert engine.emulator_list() == legacy


def test_android_start_translation_requires_and_preserves_owned_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = {
        "ok": True,
        "action": "emulator-start",
        "serial": "emulator-5592",
        "avd": "rootable-api34",
        "instance": "rootable-api34.p5592",
        "pid": 5592,
        "owner": "worker-a",
        "headless": True,
    }
    monkeypatch.setattr(emulator, "start", lambda *_args, **_kwargs: dict(legacy))

    result = emulator.start_virtual_target(
        VirtualTargetStartRequest(
            definition_id="rootable-api34",
            cache_dir=tmp_path,
            owner="worker-a",
        )
    )

    assert result.target_id == "emulator-5592"
    assert result.instance_token == "rootable-api34.p5592"
    assert dict(result.legacy_result or {}) == legacy


def test_android_stop_translation_preserves_process_bound_lease_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = leases.LeaseOwner("worker-a", pid=123, started="boot-a")
    calls: list[dict[str, Any]] = []

    def fake_stop(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True, "action": "emulator-stop", "stopped": []}

    monkeypatch.setattr(emulator, "stop", fake_stop)

    emulator.stop_virtual_targets(
        VirtualTargetStopRequest(
            target_id="emulator-5592",
            cache_dir=tmp_path,
            lease_registry_dir=tmp_path / "leases",
            lease_owner=owner,
        )
    )

    assert calls[0]["lease_owner"] is owner
