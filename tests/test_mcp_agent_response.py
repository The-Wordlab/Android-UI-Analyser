"""Opt-in MCP normalization preserves one dispatch and the evidence already returned."""

from __future__ import annotations

import base64
import json
from copy import deepcopy

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from android_ui_analyser import journal
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import AuaError, SelectorNotFoundError, UsageError
from android_ui_analyser.mcp_server import _dispatch, _tool_definitions, build_server
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.schema import Element
from android_ui_analyser.session import create_session_state
from android_ui_analyser.session_artifacts import SessionArtifactStore
from conftest import FakeDevice, make_config, make_png


class AgentPlatform(PlatformAdapter):
    name = "agent-fixture"
    capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot"})

    def connect(self, target_id=None):
        raise AssertionError("response normalization must not connect")

    def list_targets(self):
        raise AssertionError("response normalization must not discover targets")

    def normalize_tree(self, raw_tree, screen_size, **kwargs):
        return NormalizedTree(
            [
                Element(
                    id=1,
                    type="Button",
                    text="Continue",
                    resource_id="example:id/continue",
                    bounds=(0, 0, 40, 40),
                    center=(20, 20),
                    clickable=True,
                    enabled=True,
                    source="hierarchy",
                ),
            ]
        )

    def capture_screenshot(self, runtime):
        return runtime.screenshot()


def _engine(tmp_path):
    config = make_config(
        cache={"dir": str(tmp_path)},
        capture={"enabled": False},
        lease={"enabled": False},
        logs={"enabled": False},
        perf={"prefetch": False, "predictive_prefetch": False},
        output={"observation_fields": "all", "observation_meta": "all"},
    )
    device = FakeDevice(width=80, height=120, serial="agent-target")
    engine = Engine(config, device=device, platform=AgentPlatform(config))
    engine._session_id = "active-goal"
    engine._lease_owner_resolved = "agent-owner"
    return engine


def _payload(response):
    value = json.loads(next(block.text for block in response.content if block.type == "text"))
    if "context" in value and "result" in value:
        assert response.structuredContent == value
        assert response.isError is (not value["ok"])
    return value


def test_agent_mode_persists_and_disables_without_changing_legacy_schema(tmp_path):
    engine = _engine(tmp_path)
    server = build_server(engine)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            legacy = _payload(await client.call_tool("configure", {}))
            assert "agent_response" not in legacy and "schema_version" not in legacy
            enabled = _payload(await client.call_tool("configure", {"agent_response": True}))
            assert enabled["ok"] and enabled["result"]["agent_response"] is True
            assert enabled["context"] == {
                "platform": "agent-fixture",
                "target_id": "agent-target",
                "owner": "agent-owner",
                "session_id": "active-goal",
                "cache_dir": str(tmp_path),
            }
            continued = _payload(await client.call_tool("configure", {"with_image": True}))
            assert continued["result"]["agent_response"] is True
            assert continued["result"]["with_image"] is True
            disabled = _payload(await client.call_tool("configure", {"agent_response": False}))
            assert disabled["agent_response"] is False and "schema_version" not in disabled
            restored = _payload(await client.call_tool("configure", {"with_image": False}))
            assert restored == {**legacy, "with_image": False}
            assert engine._device.hierarchy_calls == engine._device.screenshot_calls == 0

    try:
        anyio.run(run)
    finally:
        engine.close()


def test_action_normalization_adds_no_acquisitions_or_dispatch_and_preserves_image(
    tmp_path,
    monkeypatch,
):
    def no_android(*args, **kwargs):
        raise AssertionError("agent response reached Android tooling")

    for method in ("connect", "dump_tree", "capture_screenshot"):
        monkeypatch.setattr(AndroidPlatform, method, no_android)
    records = []
    monkeypatch.setattr(journal, "record", lambda **kwargs: records.append(deepcopy(kwargs)))

    def exercise(mode):
        engine = _engine(tmp_path / str(mode))
        # This comparison measures the response wrapper, not how many pixel polls the
        # scheduler fits into a settle window. Use one fixed settled result in both lanes;
        # the real action, analyze, screenshot and response paths remain exercised.
        monkeypatch.setattr(
            engine,
            "_await_post_action_ready",
            lambda **_kwargs: {"changed": True, "timeout": False, "via": "hierarchy", "ms": 1},
        )
        server = build_server(engine)

        async def run():
            async with create_connected_server_and_client_session(server) as client:
                if mode:
                    await client.call_tool("configure", {"agent_response": True})
                analyzed = _payload(
                    await client.call_tool(
                        "analyze_screen",
                        {
                            "source": "hierarchy",
                            "with_image": True,
                        },
                    )
                )
                assert len((analyzed["observation"] if mode else analyzed)["elements"]) == 1
                result = await client.call_tool(
                    "key_and_analyze",
                    {
                        "name": "back",
                        "with_image": True,
                    },
                )
                payload = _payload(result)
                observation = payload["observation"]
                assert observation["elements"][0]["text"] == "Continue"
                images = [block for block in result.content if block.type == "image"]
                assert len(images) == 1
                assert base64.b64decode(images[0].data) == engine._device._png
                if mode:
                    assert payload["ok"] is True
                    assert payload["observation_contract"]["elements_available"] is True
                return (
                    engine._device.hierarchy_calls,
                    engine._device.screenshot_calls,
                    engine._device.calls,
                )

        try:
            return anyio.run(run)
        finally:
            engine.close()

    legacy = exercise(False)
    normalized = exercise(True)
    assert normalized == legacy
    assert normalized[0] > 0 and normalized[1] > 0
    assert sum(name == "press" for name, _ in normalized[2]) == 1
    mcp_actions = [
        item
        for item in records
        if item.get("source") == "mcp" and item.get("cmd") == "key_and_analyze"
    ]
    assert len(mcp_actions) == 2
    assert all("schema_version" not in item["result"] for item in mcp_actions)
    assert all(item["result"]["observation"]["elements"] for item in mcp_actions)


@pytest.mark.parametrize("agent_response", [False, True])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("config_fields", ["all", "id,text,rid"])
def test_selector_miss_returns_usable_existing_screen_without_another_read(
    tmp_path,
    monkeypatch,
    agent_response,
    projected,
    config_fields,
):
    def no_android(*args, **kwargs):
        raise AssertionError("selector recovery reached Android tooling")

    for method in ("connect", "dump_tree", "capture_screenshot"):
        monkeypatch.setattr(AndroidPlatform, method, no_android)

    # Compare against the resolver itself: publication of its already-read screen must
    # add neither a hierarchy read nor a screenshot, in either MCP response mode.
    baseline = _engine(tmp_path / "resolver")
    try:
        with pytest.raises(SelectorNotFoundError):
            baseline.resolve_selector(rid="example:id/missing", prefer_clickable=True)
        resolver_reads = (baseline._device.hierarchy_calls, baseline._device.screenshot_calls)
        resolver_calls = list(baseline._device.calls)
        assert resolver_reads[0] == 1
        assert [name for name, _ in resolver_calls] == ["find_text"]
    finally:
        baseline.close()

    engine = _engine(tmp_path / "mcp")
    engine.config.output.observation_fields = config_fields
    state = create_session_state(
        engine.config.cache.dir,
        goal="Verify Continue is visible",
        serial=engine._device.serial,
        platform=engine.platform.name,
        owner="agent-owner",
        recommended_kind="inspect",
        recommended_cli="aua analyze",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
        artifact_dir=str(tmp_path / "artifacts"),
        evidence="all",
    )
    engine._session_id = state.session_id
    SessionArtifactStore.create(
        state.artifact_dir,
        session_id=state.session_id,
        goal=state.goal,
        evidence="all",
        junit=False,
        contract_yaml=None,
    )
    closed_fingerprints = []
    close_turn = engine.close_caller_turn

    def close(fingerprint=None):
        closed_fingerprints.append(fingerprint)
        close_turn(fingerprint)

    monkeypatch.setattr(engine, "close_caller_turn", close)
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "timeout": False, "via": "hierarchy", "ms": 1},
    )
    server = build_server(engine)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            if agent_response:
                await client.call_tool("configure", {"agent_response": True})
            arguments = {"rid": "example:id/missing"}
            if projected:
                arguments["observe_fields"] = "id,text,resource_id"
            response = await client.call_tool("tap_and_analyze", arguments)
            payload = _payload(response)
            assert response.isError is agent_response
            assert payload["error"]["code"] == "selector_not_found"
            assert "Continue" in payload["error"]["hint"]
            if agent_response:
                assert payload["ok"] is False
                observation = payload["observation"]
                contract = payload["observation_contract"]
            else:
                observation = payload["error"]["observation"]
                contract = observation["meta"]["observation_contract"]
            (control,) = observation["elements"]
            assert control["id"].startswith("el:")
            assert control["text"] == "Continue"
            if projected or config_fields == "all":
                assert control["resource_id"] == "example:id/continue"
            else:
                assert control["rid"] == "continue"
            if projected:
                assert set(control) == {"id", "text", "resource_id"}
            elif config_fields == "all":
                assert control["bounds"] == [0, 0, 40, 40]
            assert contract["reusable"] is True
            assert contract["analyze_needed"] is False
            assert contract["elements_available"] is True
            assert contract.get("action_succeeded") is not True
            assert (
                engine._device.hierarchy_calls,
                engine._device.screenshot_calls,
            ) == resolver_reads
            assert engine._device.calls == resolver_calls
            assert closed_fingerprints[-1] == engine._last_analyze_result.meta.fingerprint

            # An agent can immediately address the returned ID. No compensating analyze
            # call is needed, and the failed selector must not have dispatched a gesture.
            recovered = await client.call_tool("tap_and_analyze", {"id": control["id"]})
            assert not recovered.isError
            assert _payload(recovered)["ok"] is True
            mutations = [call for call in engine._device.calls if call[0] != "find_text"]
            assert mutations == [("click", (20, 20))]

    try:
        anyio.run(run)
    finally:
        engine.close()


@pytest.mark.parametrize("attachment", ["result", "observation"])
def test_failed_action_keeps_recovery_screen_and_image_but_remains_error(
    tmp_path,
    monkeypatch,
    attachment,
):
    engine = _engine(tmp_path)
    image_path = tmp_path / "already-produced.png"
    image_path.write_bytes(make_png(80, 120))
    observation = {
        "screen": {"width": 80, "height": 120},
        "elements": [{"id": "rid:continue", "text": "Continue", "enabled": True}],
        "meta": {"raw_image": str(image_path)},
    }
    if attachment == "observation":
        observation["meta"]["observation_contract"] = {
            "reusable": False,
            "evidence_fresh": False,
            "analyze_needed": True,
            "reason": "The producer could not confirm freshness.",
        }
    calls = []
    records = []
    monkeypatch.setattr(journal, "record", lambda **kwargs: records.append(deepcopy(kwargs)))

    class FailedAction(AuaError):
        code = "action_outcome_unknown"

        def to_dict(self):
            value = super().to_dict()
            value["error"][attachment] = (
                {"ok": True, "observation": observation, "action": "key"}
                if attachment == "result"
                else observation
            )
            return value

    def failed_key(*args, **kwargs):
        calls.append((args, kwargs))
        raise FailedAction("Delivery is uncertain; do not replay.")

    monkeypatch.setattr(engine, "key", failed_key)
    server = build_server(engine)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("configure", {"agent_response": True})
            response = await client.call_tool(
                "key_and_analyze", {"name": "back", "observe_fields": "id,text"}
            )
            payload = _payload(response)
            assert payload["ok"] is False and response.isError
            assert payload["error"]["code"] == "action_outcome_unknown"
            assert payload["observation"]["elements"][0]["text"] == "Continue"
            if attachment == "observation":
                assert payload["observation_contract"]["reusable"] is False
                assert payload["observation_contract"]["evidence_fresh"] is False
                assert payload["observation_contract"]["analyze_needed"] is True
            images = [block for block in response.content if block.type == "image"]
            assert len(images) == 1
            assert base64.b64decode(images[0].data) == image_path.read_bytes()
            assert len(calls) == 1
            assert engine._device.hierarchy_calls == engine._device.screenshot_calls == 0
            recorded = next(item for item in records if item.get("cmd") == "key_and_analyze")
            assert recorded["ok"] is False
            assert recorded["error"]["code"] == "action_outcome_unknown"
            assert "schema_version" not in (recorded["result"] or {})

    try:
        anyio.run(run)
    finally:
        engine.close()


def test_unexpected_exception_does_not_substitute_previous_observation(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    server = build_server(engine)
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("fixture transport exploded")

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("configure", {"agent_response": True})
            previous = _payload(await client.call_tool("analyze_screen", {"source": "hierarchy"}))
            assert previous["observation"]["elements"]
            counts = (engine._device.hierarchy_calls, engine._device.screenshot_calls)
            monkeypatch.setattr(engine, "key", broken)
            failed = _payload(await client.call_tool("key_and_analyze", {"name": "back"}))
            assert failed["ok"] is False
            assert failed["error"]["code"] == "internal_error"
            assert failed["observation"] is None
            assert len(calls) == 1
            assert (engine._device.hierarchy_calls, engine._device.screenshot_calls) == counts

    try:
        anyio.run(run)
    finally:
        engine.close()


def test_capture_session_id_never_replaces_goal_context(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine,
        "capture_status",
        lambda: {
            "ok": True,
            "session_id": "capture-buffer",
            "running": False,
        },
    )
    server = build_server(engine)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("configure", {"agent_response": True})
            payload = _payload(await client.call_tool("capture_status", {}))
            assert payload["result"]["session_id"] == "capture-buffer"
            assert payload["context"]["session_id"] == "active-goal"
            engine._session_id = None
            payload = _payload(await client.call_tool("capture_status", {}))
            assert payload["context"]["session_id"] is None
            assert engine._device.hierarchy_calls == engine._device.screenshot_calls == 0

    try:
        anyio.run(run)
    finally:
        engine.close()


def test_configure_schema_failure_remains_transport_error_and_cannot_enable_mode(tmp_path):
    engine = _engine(tmp_path)
    schema = next(tool.inputSchema for tool in _tool_definitions() if tool.name == "configure")
    assert schema["properties"]["agent_response"]["type"] == "boolean"
    assert schema["properties"]["agent_response"]["default"] is False
    server = build_server(engine)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            invalid = await client.call_tool("configure", {"agent_response": "false"})
            assert invalid.isError
            assert not getattr(engine, "_mcp_agent_response", False)
            await client.call_tool("configure", {"agent_response": True})
            invalid = await client.call_tool("configure", {"agent_response": "false"})
            # SDK validation precedes the handler: callers must always check isError.
            assert invalid.isError and invalid.structuredContent is None
            assert getattr(engine, "_mcp_agent_response", False) is True
            assert _payload(await client.call_tool("configure", {}))["ok"] is True
            assert engine._device.hierarchy_calls == engine._device.screenshot_calls == 0

    try:
        anyio.run(run)
        for value in ("false", 0, None):
            with pytest.raises(UsageError, match="must be a boolean"):
                _dispatch(engine, "configure", {"agent_response": value})
        assert getattr(engine, "_mcp_agent_response", False) is True
    finally:
        engine.close()
