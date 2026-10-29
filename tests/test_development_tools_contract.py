"""Optional developer tools stay platform-neutral and keep durable, target-owned cleanup."""

from __future__ import annotations

import builtins
import contextlib
import copy
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import device_ledger, leases
from android_ui_analyser.cli import GlobalOpts, app
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    DeviceError,
    UnsupportedPlatformCapabilityError,
    UsageError,
)
from android_ui_analyser.mcp_server import _dispatch_tool
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.runtime import TargetRuntime
from android_ui_analyser.schema import (
    ActionResult,
    AnalyzeResult,
    Element,
    Meta,
    Screen,
    Source,
    TargetInfo,
)
from conftest import make_config


class _Tools:
    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls
        self.specs = {
            "project.describe": {"scope": "host"},
            "project.remove": {"scope": "host", "needs_confirmation": True},
            "target.inspect": {"scope": "target"},
            "target.mutate": {"scope": "target", "needs_confirmation": True},
            "target.tap": {"scope": "target", "observe": True},
            "trace.start": {"scope": "target", "profile_action": "begin"},
            "trace.continue": {"scope": "target", "profile_action": "continue"},
            "trace.stop": {"scope": "target", "profile_action": "end"},
            "trace.status": {"scope": "target", "profile_action": "read"},
            "heap.capture": {"scope": "target", "profile_action": "capture"},
            "virtual.start": {"scope": "host", "route": "virtual.start"},
            "virtual.stop": {"scope": "host", "route": "virtual.stop"},
            "virtual.delete": {
                "scope": "host", "route": "virtual.delete", "needs_confirmation": True,
            },
        }

    def status(self) -> dict[str, Any]:
        self.calls.append(("status",))
        return {"ok": True, "installed": False}

    def setup(self) -> dict[str, Any]:
        self.calls.append(("setup",))
        return {"ok": True, "installed": True}

    def catalogue(self) -> list[dict[str, Any]]:
        return [{"name": name, **spec} for name, spec in self.specs.items()]

    def validate(
        self, operation: str, arguments: list[str], options: dict[str, Any],
        project_dir: str | None, timeout_s: float,
    ) -> dict[str, Any]:
        self.calls.append(("validate", operation))
        if operation not in self.specs:
            raise UsageError("unknown operation")
        if set(options) & {"session", "device", "sdk"}:
            raise UsageError("reserved target or session override")
        return {"name": operation, **self.specs[operation]}

    def execute(
        self, operation: str, arguments: list[str], options: dict[str, Any],
        project_dir: str | None, timeout_s: float, target_id: str | None = None,
    ) -> dict[str, Any]:
        assert self.specs[operation]["scope"] == "host"
        assert target_id is None
        self.calls.append(("host-execute", operation, arguments, options, project_dir, timeout_s))
        return {"ok": True, "operation": operation, "data": {"target": "fixture-target"}}


class _Runtime(TargetRuntime):
    def __init__(self, target_id: str, calls: list[Any]) -> None:
        self.target_id = target_id
        self.calls = calls
        self.result: dict[str, Any] = {"ok": True}
        self.error: Exception | None = None
        self.before_execute: Callable[[], None] | None = None

    def instance_token(self) -> str:
        return "fixture-boot"

    def execute_development_tool(
        self, operation: str, arguments: list[str], options: dict[str, Any],
        project_dir: str | None, timeout_s: float,
    ) -> dict[str, Any]:
        if self.before_execute:
            self.before_execute()
        self.calls.append(("target-execute", self.target_id, operation, arguments,
                           dict(options), project_dir, timeout_s))
        if self.error:
            raise self.error
        return {"operation": operation, **copy.deepcopy(self.result)}

    def restore_development_tool(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("restore", self.target_id, session_id))
        return {"ok": True, "session_id": session_id}

    def prepare_development_ui(self) -> dict[str, Any]:
        self.calls.append(("ui-baseline", self.target_id))
        return {"remove_installed_helper": True}

    def restore_development_ui(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("ui-restore", self.target_id, state))
        return {"ok": True}


class _Platform(PlatformAdapter):
    name = "fixture-development"
    capabilities = frozenset({"development_tools", "device.development_tools"})

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[Any] = []
        self.tools = _Tools(self.calls)
        self.runtime = _Runtime("owned-target", self.calls)

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        self.calls.append(("connect", target_id))
        assert target_id == self.runtime.target_id
        return self.runtime

    def list_targets(self) -> list[Any]:
        raise AssertionError("this test pins its lease; discovery must not choose another target")

    def normalize_tree(
        self, raw_tree: str, screen_size: tuple[int, int], *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree([])

    def load_capability(self, capability: str) -> object | None:
        return self.tools if capability == "development_tools" else None


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        lease={"enabled": False, "registry_dir": str(tmp_path / "leases")},
        teardown={"enabled": True, "sweep_on_command": False},
        memory={"enabled": False},
    )
    platform = _Platform(cfg)
    engine = Engine(cfg, platform=platform)

    def leased() -> str:
        platform.calls.append(("lease", "owned-target"))
        return platform.runtime.target_id

    monkeypatch.setattr(engine, "_lease_device", leased)
    monkeypatch.setattr(engine, "_ledger_identity", lambda: {
        "owner": "fixture-owner", "owner_pid": None, "owner_started": None,
        "cache_dir": str(cfg.cache.dir), "leased": False,
    })
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"adbutils", "uiautomator2"} or name.startswith(
            "android_ui_analyser.platforms.android"
        ):
            raise AssertionError(f"generic developer tools imported native backend {name}")
        return original_import(name, *args, **kwargs)

    def native_forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a fake platform must never invoke a native process")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", native_forbidden)
    monkeypatch.setattr(subprocess, "Popen", native_forbidden)
    yield engine, platform
    engine.release_device_use()


def _entries(platform: _Platform) -> list[Any]:
    return device_ledger.read_ledger(platform.runtime.target_id, platform=platform.name)


def test_host_discovery_and_execution_never_connect_or_load_native_tools(rig) -> None:
    engine, platform = rig
    assert engine.development_tools_status() == {"ok": True, "installed": False}
    catalogue = engine.development_tools_catalogue(operation="project.describe")
    assert catalogue["platform"] == platform.name
    assert [item["name"] for item in catalogue["operations"]] == ["project.describe"]
    result = engine.development_tool(
        "project.describe", arguments=[":fixture"], options={"format": "json"},
        project_dir="/fixture/workspace", timeout_s=40,
    )
    assert result["data"] == {"target": "fixture-target"}
    assert ("host-execute", "project.describe", [":fixture"], {"format": "json"},
            "/fixture/workspace", 40) in platform.calls
    assert not [call for call in platform.calls if call[0] in {"connect", "lease", "setup"}]
    assert engine._device is None


def test_setup_is_only_an_explicit_host_call(rig) -> None:
    engine, platform = rig
    engine.development_tools_status()
    engine.development_tools_catalogue()
    assert ("setup",) not in platform.calls
    assert engine.development_tools_setup()["installed"] is True
    assert platform.calls.count(("setup",)) == 1
    assert engine._device is None


@pytest.mark.parametrize("operation,options", [
    ("unknown", {}), ("target.inspect", {"device": "foreign-target"}),
    ("trace.start", {"session": "foreign-session"}),
    ("target.inspect", {"sdk": "/unreviewed/sdk"}),
])
def test_validation_refuses_bad_calls_before_claiming_a_device(rig, operation, options) -> None:
    engine, platform = rig
    with pytest.raises(UsageError):
        engine.development_tool(operation, options=options)
    assert engine._device is None
    assert not [call for call in platform.calls if call[0] != "validate"]


@pytest.mark.parametrize("operation", ["project.remove", "target.mutate"])
def test_confirmation_is_checked_before_any_execution_or_target_claim(rig, operation) -> None:
    engine, platform = rig
    with pytest.raises(UsageError) as caught:
        engine.development_tool(operation)
    assert caught.value.code == "development_tool_confirmation_required"
    assert engine._device is None
    assert platform.calls == [("validate", operation)]


def test_target_operations_use_the_exact_leased_runtime_and_not_the_host_service(rig) -> None:
    engine, platform = rig
    result = engine.development_tool("target.inspect", arguments=["fixture"], timeout_s=17)
    assert result["ok"] is True
    assert platform.calls == [
        ("validate", "target.inspect"), ("lease", "owned-target"), ("connect", "owned-target"),
        ("target-execute", "owned-target", "target.inspect", ["fixture"], {}, None, 17),
    ]


def test_live_lease_selection_does_not_touch_a_target_held_by_another_worker(rig, monkeypatch) -> None:
    engine, platform = rig
    engine.config.lease.enabled = True
    engine._lease_owner = "test-tools-worker"
    monkeypatch.setattr(engine, "_lease_device", Engine._lease_device.__get__(engine, Engine))
    monkeypatch.setattr(platform, "list_targets", lambda: [
        TargetInfo(target_id="busy-target", platform=platform.name, status="online"),
        TargetInfo(target_id="owned-target", platform=platform.name, status="online"),
    ])
    leases.choose_device(
        engine._lease_registry_dir, owner="other-worker", explicit="busy-target",
        candidates=[("busy-target", {})], platform=platform.name,
    )

    engine.development_tool("target.inspect")

    assert ("connect", "owned-target") in platform.calls
    assert ("connect", "busy-target") not in platform.calls
    assert leases.held_by(
        engine._lease_registry_dir, "test-tools-worker", platform=platform.name,
    ) == ["owned-target"]


def test_profile_undo_is_on_disk_before_target_mutation(rig) -> None:
    engine, platform = rig

    def prove_write_ahead() -> None:
        entries = _entries(platform)
        assert len(entries) == 1
        assert entries[0].op == "restore_development_tool"
        assert entries[0].args["session_id"].startswith("aua-")

    platform.runtime.before_execute = prove_write_ahead
    result = engine.development_tool("trace.start")
    entry, = _entries(platform)
    assert entry.args == {"session_id": result["session_id"]}
    assert entry.key == "development-tool:" + result["session_id"]
    assert result["cleanup_pending"] is True


def test_profile_identity_is_stable_for_owner_target_and_options_are_not_mutated(rig) -> None:
    engine, platform = rig
    options = {"package": "com.example.fixture"}
    original = dict(options)
    start = engine.development_tool("trace.start", options=options)
    continued = engine.development_tool("trace.continue")
    status = engine.development_tool("trace.status")
    assert start["session_id"] == continued["session_id"] == status["session_id"]
    assert options == original
    assert len(_entries(platform)) == 1


def test_completed_heap_capture_cannot_retire_an_active_trace_undo(rig) -> None:
    engine, platform = rig
    trace = engine.development_tool("trace.start")
    heap = engine.development_tool("heap.capture")
    assert trace["session_id"] != heap["session_id"]
    remaining, = _entries(platform)
    assert remaining.args["session_id"] == trace["session_id"]


def test_profile_status_does_not_claim_pending_cleanup_has_finished(rig) -> None:
    engine, platform = rig
    engine.development_tool("trace.start")
    status = engine.development_tool("trace.status")
    assert status.get("cleanup_pending") is not False
    assert len(_entries(platform)) == 1


def test_durable_undo_replays_through_the_selected_runtime(rig) -> None:
    engine, platform = rig
    result = engine.development_tool("trace.start")
    entry, = _entries(platform)
    context = device_ledger.UndoContext(
        serial=platform.runtime.target_id, device=platform.runtime,
        platform=platform.name, runtime_capability=platform.runtime_capability,
        capability=platform.capability,
    )
    handler = device_ledger.UNDO_OPS[entry.op]
    handler.handler(context, entry.args)
    assert platform.calls[-1] == ("restore", "owned-target", result["session_id"])


def test_another_owner_gets_a_different_profile_identity(rig, monkeypatch) -> None:
    engine, platform = rig
    first = engine.development_tool("trace.start")
    original = engine._ledger_identity()
    monkeypatch.setattr(engine, "_ledger_identity", lambda: {**original, "owner": "second-owner"})
    second = engine.development_tool("trace.start")
    assert first["session_id"] != second["session_id"]
    assert len(_entries(platform)) == 2


@pytest.mark.parametrize("raises", [True, False])
def test_failed_profile_attempt_keeps_the_undo_and_is_never_retried(rig, raises) -> None:
    engine, platform = rig
    if raises:
        platform.runtime.error = DeviceError("transport failed after dispatch", code="timeout")
    else:
        platform.runtime.result = {"ok": False, "stderr": "recording could have started"}
    with pytest.raises(DeviceError):
        engine.development_tool("trace.start")
    assert len(_entries(platform)) == 1
    assert len([call for call in platform.calls if call[0] == "target-execute"]) == 1


def test_successful_stop_forgets_only_its_owned_profile_key(rig, monkeypatch) -> None:
    engine, platform = rig
    original_identity = engine._ledger_identity()
    own = engine.development_tool("trace.start")
    monkeypatch.setattr(engine, "_ledger_identity", lambda: {
        **original_identity, "owner": "other-worker",
    })
    other = engine.development_tool("trace.start")
    monkeypatch.setattr(engine, "_ledger_identity", lambda: original_identity)
    result = engine.development_tool("trace.stop")
    assert result["session_id"] == own["session_id"]
    assert result["cleanup_pending"] is False
    remaining, = _entries(platform)
    assert remaining.args["session_id"] == other["session_id"]


def test_failed_stop_retains_cleanup_and_keep_session_does_not_forget_it(rig) -> None:
    engine, platform = rig
    engine.development_tool("trace.start")
    kept = engine.development_tool("trace.stop", options={"keep-session": True})
    assert kept["cleanup_pending"] is True
    platform.runtime.error = DeviceError("stop failed")
    with pytest.raises(DeviceError):
        engine.development_tool("trace.stop")
    assert len(_entries(platform)) == 1


@pytest.mark.parametrize("operation,options", [
    ("heap.capture", {}), ("trace.start", {"duration": 1, "wait": True}),
])
def test_completed_bounded_capture_retires_its_cleanup(rig, operation, options) -> None:
    engine, platform = rig
    result = engine.development_tool(operation, options=options)
    assert result["cleanup_pending"] is False
    assert not _entries(platform)


def test_profile_refuses_to_mutate_without_durable_teardown(rig) -> None:
    engine, platform = rig
    engine.config.teardown.enabled = False
    with pytest.raises(UsageError, match="teardown"):
        engine.development_tool("trace.start")
    assert not [call for call in platform.calls if call[0] == "target-execute"]
    assert not _entries(platform)


def test_observation_failure_preserves_successful_action_without_repeating_it(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(engine, "_acting", lambda **kw: contextlib.nullcontext())

    def cannot_observe(action: ActionResult, observe: bool) -> Any:
        assert observe is True
        raise DeviceError("fresh frame unavailable", code="capture_failed")

    monkeypatch.setattr(engine, "_observe", cannot_observe)
    result = engine.development_tool("target.tap", arguments=["fixture-control"])
    assert result["action_succeeded"] is True
    assert result["observation_error"]["code"] == "capture_failed"
    assert "do not repeat" in result["next_action"]
    assert len([call for call in platform.calls if call[0] == "target-execute"]) == 1


def test_native_action_reuses_settled_observation_and_compact_stable_ids(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(engine, "_acting", lambda **kw: contextlib.nullcontext())
    platform.tools.specs["target.tap"]["action_kind"] = "app-launch"
    platform.runtime.result = {"ok": True, "data": {"native": "retained"}, "duration_ms": 17}

    def observe(action: ActionResult, enabled: bool) -> ActionResult:
        assert enabled and action.action == "app-launch"
        action.observation = AnalyzeResult(
            screen=Screen(width=400, height=800, package="com.example.fixture", source="hierarchy"),
            elements=[
                Element(id=1, stable_key="el:fixture", text="Continue", type="Button",
                        bounds=(0, 20, 200, 70), center=(100, 45), source=Source.hierarchy,
                        clickable=True),
                Element(id=2, text="Status", resource_id="com.android.systemui:id/clock",
                        type="TextView", bounds=(0, 0, 100, 15), center=(50, 7), source=Source.hierarchy),
            ],
            meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy",
                      fingerprint="fixture-frame", device_serial="owned-target"),
        )
        action.observation_present = True
        action.arrival = {"state": "transitioning", "evidence": ["settle_timeout"]}
        action.stale_risk = "Destination has not rendered yet."
        return action

    monkeypatch.setattr(engine, "_observe", observe)
    result = engine.development_tool("target.tap")
    assert result["data"] == {"native": "retained"}
    assert result["duration_ms"] == 17
    assert result["action_succeeded"] and result["observation_present"]
    assert result["arrival"]["state"] == "transitioning"
    assert result["stale_risk"] == "Destination has not rendered yet."
    assert len(result["observation"]["elements"]) == 1
    assert result["observation"]["elements"][0]["id"] == "el:fixture"
    assert "content_desc" not in result["observation"]["elements"][0]
    assert len([call for call in platform.calls if call[0] == "target-execute"]) == 1


def test_suppressed_observation_failure_does_not_claim_a_frame(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(engine, "_acting", lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(engine, "_observe", lambda action, observe: action)
    result = engine.development_tool("target.tap")
    assert result["ok"] and result["action_succeeded"]
    assert result["observation_present"] is False
    assert result["observation_error"]["code"] == "observation_unavailable"
    assert "do not repeat" in result["next_action"]


@pytest.mark.parametrize("background_idle,capture_idle", [(False, True), (True, False)])
def test_native_ui_automation_refuses_busy_handover_before_execution(
    rig, monkeypatch, background_idle, capture_idle,
) -> None:
    engine, platform = rig
    platform.tools.specs["target.inspect"]["ui_automation"] = True
    events = []
    monkeypatch.setattr(engine, "_quiesce_background_device_work", lambda: background_idle)

    @contextlib.contextmanager
    def stand_down():
        events.append("paused")
        try:
            yield capture_idle
        finally:
            events.append("resumed")

    monkeypatch.setattr(engine, "_device_stood_down", stand_down)
    with pytest.raises(DeviceError) as caught:
        engine.development_tool("target.inspect")
    assert caught.value.code == "development_tool_handover_refused"
    assert not [call for call in platform.calls if call[0] == "target-execute"]
    assert events == (["paused", "resumed"] if background_idle else [])


@pytest.mark.parametrize("native_fails", [False, True])
def test_native_ui_automation_resumes_capture_before_post_action_work(
    rig, monkeypatch, native_fails,
) -> None:
    engine, platform = rig
    platform.tools.specs["target.tap"]["ui_automation"] = True
    events = []

    @contextlib.contextmanager
    def acting(**kwargs):
        events.append("pre-action")
        try:
            yield
        finally:
            assert engine._stood_down is False
            events.append("post-action")

    @contextlib.contextmanager
    def stand_down():
        events.append("paused")
        engine._stood_down = True
        engine._device = None
        try:
            yield True
        finally:
            engine._stood_down = False
            events.append("resumed")

    def native_call():
        assert engine._stood_down is True
        assert engine._device is None
        events.append("native")

    monkeypatch.setattr(engine, "_acting", acting)
    monkeypatch.setattr(engine, "_device_stood_down", stand_down)
    monkeypatch.setattr(engine, "_quiesce_background_device_work", lambda: events.append("quiesced") or True)
    monkeypatch.setattr(engine, "_observe", lambda action, observe: action)
    platform.runtime.before_execute = native_call
    if native_fails:
        platform.runtime.error = DeviceError("native failed", code="fixture_failed")
        with pytest.raises(DeviceError, match="native failed"):
            engine.development_tool("target.tap")
    else:
        assert engine.development_tool("target.tap")["action_succeeded"]
    assert events == ["pre-action", "quiesced", "paused", "native", "resumed", "post-action"]


def test_unsupported_adapter_refuses_explicitly_without_connecting(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(platform, "capabilities", frozenset())
    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        engine.development_tool("project.describe")
    assert caught.value.code == "platform_capability_unsupported"
    assert not platform.calls


def test_host_service_does_not_imply_target_runtime_support(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(platform, "capabilities", frozenset({"development_tools"}))
    assert engine.development_tools_status()["installed"] is False
    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        engine.development_tool("target.inspect")
    assert caught.value.code == "platform_capability_unsupported"
    assert not [call for call in platform.calls if call[0].endswith("execute")]


@pytest.mark.parametrize("route,method,arguments,options,expected", [
    ("virtual.start", "virtual_target_start", ["fixture-avd"], {"cold": True, "headless": True},
     (("fixture-avd",), {"headless": True, "wait_s": 80, "options": {"cold": True}})),
    ("virtual.stop", "virtual_target_stop", ["fixture-target"], {},
     ((), {"target_id": "fixture-target"})),
    ("virtual.delete", "virtual_target_delete", ["fixture-avd"], {},
     (("fixture-avd",), {"confirmed": True})),
])
def test_virtual_lifecycle_reuses_owned_engine_operations(
    rig, monkeypatch, route, method, arguments, options, expected,
) -> None:
    engine, platform = rig
    calls = []

    def routed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(engine, method, routed)
    result = engine.development_tool(
        route, arguments=arguments, options=options, timeout_s=80, confirmed=True,
    )
    assert calls == [expected]
    assert result["implementation"] == "aua-owned-lifecycle"
    assert engine._device is None
    assert not [call for call in platform.calls if call[0].endswith("execute")]


def test_catalogue_rejects_an_unknown_operation_without_executing(rig) -> None:
    engine, platform = rig
    with pytest.raises(UsageError, match="unknown"):
        engine.development_tools_catalogue(operation="unknown")
    assert not platform.calls


def test_cli_and_mcp_use_the_same_host_engine_contract(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(GlobalOpts, "engine", lambda self: engine)
    cli = CliRunner().invoke(app, [
        "--format", "json", "devtools", "run", "project.describe", ":fixture",
        "--options", '{"format":"json"}', "--project", "/fixture/workspace",
        "--timeout", "41",
    ])
    assert cli.exit_code == 0, cli.output
    cli_payload = json.loads(cli.stdout)
    cli_call = [call for call in platform.calls if call[0] == "host-execute"][-1]
    platform.calls.clear()
    mcp_payload = _dispatch_tool(engine, "development_tool", {
        "operation": "project.describe", "arguments": [":fixture"],
        "options": {"format": "json"}, "project_dir": "/fixture/workspace", "timeout_s": 41,
    })
    mcp_call = [call for call in platform.calls if call[0] == "host-execute"][-1]
    assert cli_call == mcp_call
    assert cli_payload == mcp_payload
    assert engine._device is None


def test_cli_and_mcp_refuse_the_same_unconfirmed_mutation(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(GlobalOpts, "engine", lambda self: engine)
    cli = CliRunner().invoke(app, ["devtools", "run", "target.mutate"])
    assert cli.exit_code == UsageError.exit_code, cli.output
    with pytest.raises(UsageError) as caught:
        _dispatch_tool(engine, "development_tool", {"operation": "target.mutate"})
    assert caught.value.code == "development_tool_confirmation_required"
    assert caught.value.code in cli.output
    assert engine._device is None
    assert not [call for call in platform.calls if call[0].endswith("execute")]
    assert not [call for call in platform.calls if call[0] == "lease"]


def test_cli_and_mcp_dispatch_confirmed_calls_to_the_same_target(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(GlobalOpts, "engine", lambda self: engine)
    cli = CliRunner().invoke(app, [
        "--format", "json", "devtools", "run", "target.mutate", "fixture-control",
        "--options", '{"value":"fixture-value"}', "--timeout", "19", "--yes",
    ])
    assert cli.exit_code == 0, cli.output
    cli_payload = json.loads(cli.stdout)
    cli_call = [call for call in platform.calls if call[0] == "target-execute"][-1]
    platform.calls.clear()

    mcp_payload = _dispatch_tool(engine, "development_tool", {
        "operation": "target.mutate", "arguments": ["fixture-control"],
        "options": {"value": "fixture-value"}, "timeout_s": 19, "confirmed": True,
    })

    mcp_call = [call for call in platform.calls if call[0] == "target-execute"][-1]
    assert cli_call == mcp_call
    assert cli_call[1] == "owned-target"
    assert cli_payload == mcp_payload
    assert not [call for call in platform.calls if call[0] == "host-execute"]


def test_cli_and_mcp_reject_target_overrides_before_leasing(rig, monkeypatch) -> None:
    engine, platform = rig
    monkeypatch.setattr(GlobalOpts, "engine", lambda self: engine)
    cli = CliRunner().invoke(app, [
        "devtools", "run", "target.inspect", "--options", '{"device":"foreign-target"}',
    ])
    assert cli.exit_code == UsageError.exit_code, cli.output
    with pytest.raises(UsageError, match="reserved"):
        _dispatch_tool(engine, "development_tool", {
            "operation": "target.inspect", "options": {"device": "foreign-target"},
        })
    assert engine._device is None
    assert not [call for call in platform.calls if call[0] != "validate"]


def test_ui_helper_baseline_is_durable_before_native_execution_and_removed_after(rig):
    engine, platform = rig
    platform.tools.specs["target.inspect"]["ui_automation"] = True

    def check_ledger():
        entries = _entries(platform)
        assert len(entries) == 1
        assert entries[0].op == "restore_development_ui"
        assert entries[0].args == {"state": {"remove_installed_helper": True}}

    platform.runtime.before_execute = check_ledger
    result = engine.development_tool("target.inspect")
    assert result["ok"] is True
    assert not _entries(platform)
    assert platform.calls[-1] == ("ui-restore", "owned-target", {"remove_installed_helper": True})


def test_native_ui_failure_restores_helper_without_repeating_action(rig):
    engine, platform = rig
    platform.tools.specs["target.inspect"]["ui_automation"] = True
    platform.runtime.error = DeviceError("native read failed")
    with pytest.raises(DeviceError, match="native read failed"):
        engine.development_tool("target.inspect")
    assert len([call for call in platform.calls if call[0] == "target-execute"]) == 1
    assert len([call for call in platform.calls if call[0] == "ui-restore"]) == 1
    assert not _entries(platform)


def test_failed_ui_cleanup_retains_undo_and_native_success(rig, monkeypatch):
    from android_ui_analyser.errors import DevelopmentToolError

    engine, platform = rig
    platform.tools.specs["target.inspect"]["ui_automation"] = True

    def fail_restore(state):
        raise DeviceError("cleanup transport failed")

    monkeypatch.setattr(platform.runtime, "restore_development_ui", fail_restore)
    with pytest.raises(DevelopmentToolError) as caught:
        engine.development_tool("target.inspect")
    assert caught.value.result["operation_succeeded"] is True
    assert caught.value.result["cleanup_pending"] is True
    assert _entries(platform)[0].op == "restore_development_ui"
    assert len([call for call in platform.calls if call[0] == "target-execute"]) == 1


def test_ui_helper_is_not_started_without_durable_cleanup(rig):
    engine, platform = rig
    platform.tools.specs["target.inspect"]["ui_automation"] = True
    engine.config.teardown.enabled = False
    with pytest.raises(UsageError, match="durable"):
        engine.development_tool("target.inspect")
    assert not [call for call in platform.calls if call[0] in {"ui-baseline", "target-execute"}]


def test_heap_status_and_cancel_use_only_this_callers_heap_session(rig):
    engine, platform = rig
    platform.tools.specs["trace.status"]["profile_kinds"] = ["trace", "heap"]
    platform.tools.specs["trace.stop"]["profile_kinds"] = ["trace", "heap"]
    trace = engine.development_tool("trace.start")
    platform.runtime.error = DeviceError("heap transport uncertain")
    with pytest.raises(DeviceError):
        engine.development_tool("heap.capture")
    platform.runtime.error = None
    status = engine.development_tool("trace.status", profile_kind="heap")
    assert status["session_id"] != trace["session_id"]
    cancelled = engine.development_tool("trace.stop", profile_kind="heap")
    assert cancelled["session_id"] == status["session_id"]
    assert not cancelled["cleanup_pending"]
    assert [entry.args["session_id"] for entry in _entries(platform)] == [trace["session_id"]]


@pytest.mark.parametrize("operation,kind", [("project.describe", "heap"), ("trace.start", "foreign-session"), ("heap.capture", "trace")])
def test_invalid_profile_kind_is_rejected_before_leasing(rig, operation, kind):
    engine, platform = rig
    with pytest.raises(UsageError, match="profile_kind"):
        engine.development_tool(operation, profile_kind=kind)
    assert engine._device is None
    assert not [call for call in platform.calls if call[0] == "lease"]


@pytest.mark.parametrize("invalid", [{"ok": False}, {}, None, [], {"ok": 1}])
def test_ui_cleanup_requires_explicit_true_and_retains_pending_undo(rig, monkeypatch, invalid):
    from android_ui_analyser.errors import DevelopmentToolError

    engine, platform = rig
    platform.tools.specs["target.inspect"]["ui_automation"] = True
    monkeypatch.setattr(platform.runtime, "restore_development_ui", lambda state: invalid)
    with pytest.raises(DevelopmentToolError) as caught:
        engine.development_tool("target.inspect")
    assert caught.value.result["cleanup_pending"] is True
    assert caught.value.result["operation_succeeded"] is True
    assert _entries(platform)[0].op == "restore_development_ui"


@pytest.mark.parametrize("op,method,args", [
    ("restore_development_tool", "restore_development_tool", {"session_id": "aua-fixture-owned"}),
    ("restore_development_ui", "restore_development_ui", {"state": {"remove_installed_helper": True}}),
])
@pytest.mark.parametrize("invalid", [{"ok": False}, {}, None, [], {"ok": 1}])
def test_watchdog_replay_does_not_forget_unconfirmed_restore(rig, monkeypatch, op, method, args, invalid):
    _engine, platform = rig
    device_ledger.record(
        platform.runtime.target_id, key="pending-fixture", kind="fixture", op=op,
        args=args, platform=platform.name,
    )
    monkeypatch.setattr(platform.runtime, method, lambda value: invalid)
    context = device_ledger.UndoContext(
        serial=platform.runtime.target_id, device=platform.runtime,
        platform=platform.name, runtime_capability=platform.runtime_capability,
        capability=platform.capability,
    )
    result = device_ledger.replay(platform.runtime.target_id, context=context, platform=platform.name)
    assert result["remaining"] == 1
    assert len(result["failed"]) == 1
    assert not result["undone"]
    assert _entries(platform)[0].key == "pending-fixture"


@pytest.mark.parametrize("uncertain", [True, False])
def test_failed_native_result_and_failed_cleanup_never_claim_operation_success(rig, monkeypatch, uncertain):
    from android_ui_analyser.errors import DevelopmentToolError

    engine, platform = rig
    platform.tools.specs["target.tap"]["ui_automation"] = True
    platform.runtime.result = {"ok": False, "outcome_unknown": uncertain}
    monkeypatch.setattr(engine, "_acting", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(platform.runtime, "restore_development_ui", lambda state: {"ok": False})
    with pytest.raises(DevelopmentToolError) as caught:
        engine.development_tool("target.tap")
    assert caught.value.result["operation_succeeded"] is (None if uncertain else False)
    assert caught.value.result["action_succeeded"] is (None if uncertain else False)
    assert caught.value.result["outcome_unknown"] is uncertain
    assert caught.value.result["cleanup_pending"] is True
    assert _entries(platform)


@pytest.mark.parametrize("structured", [True, False])
def test_native_exception_and_cleanup_failure_preserve_both_diagnostics(rig, monkeypatch, structured):
    from android_ui_analyser.errors import DevelopmentToolError

    engine, platform = rig
    platform.tools.specs["target.inspect"].update(ui_automation=True, effect="read")
    native = (
        DevelopmentToolError("native uncertain", result={"ok": False, "outcome_unknown": True})
        if structured else DeviceError("native read transport failed", code="fixture_native_failure")
    )
    platform.runtime.error = native
    monkeypatch.setattr(platform.runtime, "restore_development_ui", lambda state: None)
    with pytest.raises(DevelopmentToolError) as caught:
        engine.development_tool("target.inspect")
    assert caught.value.result["operation_succeeded"] is None
    assert caught.value.result["cleanup_pending"] is True
    assert "cleanup" in caught.value.result["cleanup_error"]
    if structured:
        assert caught.value is native
    else:
        assert caught.value.result["native_error"]["code"] == "fixture_native_failure"
        assert caught.value.__cause__ is native
    assert _entries(platform)
    assert len([call for call in platform.calls if call[0] == "target-execute"]) == 1
