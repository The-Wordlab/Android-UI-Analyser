"""MCP wrapper tests (PRD §13.1 AC8).

Drives the MCP server **in-process** with the SDK's in-memory client/server session
helper, so no subprocess and no device are needed. We list the tools (assert the core
ones are present) and call ``analyze_screen`` against an :class:`Engine` backed by a
:class:`FakeDevice`, asserting the returned content is schema-valid JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import (
    _fold_action_until,
    _mcp_started_serials,
    _tool_definitions,
    build_server,
)
from android_ui_analyser.schema import ActionResult, AnalyzeResult
from conftest import FakeDevice, make_config

# A labeled hierarchy so the forced/auto path yields elements without real providers.
HIERARCHY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Hello" bounds="[0,0][1080,120]"/>
  <node index="1" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
  <node index="2" class="android.widget.EditText" content-desc="Email"
        resource-id="com.test.app:id/email" clickable="true" enabled="true"
        bounds="[40,400][1040,500]"/>
</hierarchy>"""


def _engine() -> Engine:
    cfg = make_config()
    device = FakeDevice(
        hierarchy_xml=HIERARCHY_XML,
        text_index={"Continue": (40, 200, 1040, 320)},
    )
    return Engine(cfg, device=device)


def _first_text(result) -> str:  # type: ignore[no-untyped-def]
    """Extract the first text-content block from a CallToolResult."""
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise AssertionError(f"no text content in {result!r}")


def test_mcp_lists_core_tools() -> None:
    server = build_server(_engine())

    async def run() -> list[str]:
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            return [t.name for t in listed.tools]

    names = anyio.run(run)
    assert "analyze_screen" in names
    assert "tap_and_analyze" in names
    assert "tap" not in names
    assert "has" in names
    # Full 1:1 surface (PRD §11) + Maestro/device parity tools.
    assert {
        "analyze_screen",
        "tap_and_analyze",
        "input_and_analyze",
        "swipe_and_analyze",
        "key_and_analyze",
        "back_until_and_analyze",
        "wait_and_analyze",
        "wait_changed_and_analyze",
        "has",
        "screenshot",
        "inspect",
        "long_press_and_analyze",
        "scroll_to_and_analyze",
        "wait_stable_and_analyze",
        "goto",
        "flow_run",
        "navigate",
        "list_devices",
        "emulator_list",
        "emulator_status",
        "emulator_start",
        "emulator_stop",
        "double_tap_and_analyze",
        "clear_and_analyze",
        "scroll_and_analyze",
        "expect_and_analyze",
        "hide_keyboard_and_analyze",
        "open_link_and_analyze",
        "clipboard_set",
        "clipboard_get",
        "paste_and_analyze",
        "copy_text",
        "erase_and_analyze",
        "location_set",
        "orientation_set",
        "orientation_get",
        "airplane_set",
        "airplane_toggle",
        "network_status",
        "network_offline",
        "network_restore",
        "network_profile_list",
        "network_profile_status",
        "network_profile_apply",
        "network_profile_restore",
        "media_add",
        "screen_record_start",
        "screen_record_stop",
        "clock_set",
        "app",
        "app_status",
        "app_launch_and_analyze",
        "shell_read_only",
        "install_app",
        "database_list",
        "database_schema",
        "database_query",
        "database_execute",
        "database_backup",
        "database_backups",
        "database_restore",
        "a11y_scroll_and_analyze",
        "flags_apply_and_analyze",
        "resolve",
        "configure",
        "map_audit",
        "reconcile_plan",
        "reconcile_submit",
        "reconcile_status",
        "reconcile_apply",
        "reconcile_rollback",
        "knowledge_list",
        "knowledge_add",
        "knowledge_stale",
        "capabilities",
        "session_start",
        "session_autopilot",
        "session_review",
        "session_finish",
        "orient",
        "reach",
        "await_and_analyze",
        "flow_list",
        "flow_save",
        "flow_delete",
    } <= set(names)


def test_mcp_emulator_start_does_not_require_serial_pinning() -> None:
    tool = next(item for item in _tool_definitions() if item.name == "emulator_start")

    assert "automatic sticky lease" in (tool.description or "")
    assert "do not repeat a serial" in (tool.description or "")
    assert "configure/device serial" not in (tool.description or "")


def test_mcp_back_until_contract_is_bounded_and_never_allows_package_change() -> None:
    tool = next(item for item in _tool_definitions() if item.name == "back_until_and_analyze")
    properties = tool.inputSchema["properties"]

    assert properties["max_steps"]["default"] == 4
    assert properties["max_steps"]["maximum"] == 12
    assert properties["back_selector"]["maxProperties"] == 1
    assert "allow_package_change" not in properties
    assert tool.inputSchema["allOf"] == [{"not": {"required": ["back_id", "back_selector"]}}]


def test_mcp_accepts_intuitive_session_and_network_verification_options() -> None:
    tools = {item.name: item for item in _tool_definitions()}

    assert tools["session_start"].inputSchema["properties"]["headed"]["default"] is False
    assert tools["session_autopilot"].inputSchema["properties"]["max_steps"]["default"] == 6
    assert tools["network_status"].inputSchema["properties"]["verify"]["default"] is True
    assert tools["analyze_screen"].inputSchema["properties"]["no_cache"] == {
        "type": "boolean",
        "default": False,
        "description": "Force a fresh capture instead of reusing cached hierarchy data.",
    }


def test_mcp_analyze_screen_forwards_uncached_refresh() -> None:
    engine = _engine()
    original = engine.analyze
    calls: list[dict[str, object]] = []

    def analyze(**kwargs: object) -> AnalyzeResult:
        calls.append(kwargs)
        return original(**kwargs)

    engine.analyze = analyze  # type: ignore[method-assign]
    server = build_server(engine)

    async def run() -> None:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(
                "analyze_screen",
                {"source": "hierarchy", "no_cache": True},
            )
            assert json.loads(_first_text(result))["elements"]

    anyio.run(run)

    assert calls == [
        {
            "source": "hierarchy",
            "with_ocr": None,
            "no_cache": True,
            "query": None,
            "with_image": False,
        }
    ]


def test_mcp_flow_save_is_preview_first_with_explicit_commit() -> None:
    tool = next(item for item in _tool_definitions() if item.name == "flow_save")
    props = tool.inputSchema["properties"]
    assert props["save"]["default"] is False
    assert "writes nothing" in props["save"]["description"]
    assert "Deprecated" in props["dry_run"]["description"]
    assert "selector_resilience" in tool.description
    assert "action-bound until" in tool.description


def test_mcp_session_review_defines_exact_accounting_scope() -> None:
    tool = next(item for item in _tool_definitions() if item.name == "session_review")

    for field in (
        "top_level_calls",
        "lifecycle_calls",
        "task_calls",
        "journal_events",
        "folded_internal_events",
        "reporting_call_included",
    ):
        assert field in tool.description


def test_mcp_expected_error_annotation_is_machine_readable() -> None:
    tool = next(item for item in _tool_definitions() if item.name == "flow_save")

    assert tool.inputSchema["properties"]["expect_error"]["type"] == "string"
    assert "exact" in tool.inputSchema["properties"]["expect_error"]["description"].casefold()


def test_mcp_initialization_teaches_goal_first_protocol() -> None:
    instructions = build_server(_engine()).create_initialization_options().instructions

    assert instructions is not None
    assert "session_start(goal)" in instructions
    assert "goto" in instructions and "matching saved flow" in instructions
    assert "network_offline" in instructions and "never airplane mode" in instructions
    assert "lifecycle_calls plus task_calls" in instructions
    assert "folded_internal_events" in instructions
    assert "session_autopilot" in instructions and "do not relay" in instructions


def test_mcp_capabilities_progressively_discover_goal_features() -> None:
    server = build_server(_engine())

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(
                "capabilities", {"goal": "verify a cached screen while offline"}
            )
            return json.loads(_first_text(result))

    data = anyio.run(run)
    ids = {item["id"] for item in data["capabilities"]}
    assert {"session", "goto", "flow", "network_offline", "action_until"} <= ids


def test_mcp_host_only_capabilities_does_not_claim_or_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine(make_config(cache={"dir": str(tmp_path)}))
    device_access: list[str] = []

    def unexpected_device_access(*_args: object, **_kwargs: object) -> None:
        device_access.append("attempted")
        raise AssertionError("host-only capability discovery touched a device")

    monkeypatch.setattr(engine, "_lease_device", unexpected_device_access)
    monkeypatch.setattr(engine, "_connect_target", unexpected_device_access)
    server = build_server(engine)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("capabilities", {"goal": "inspect available tools"})
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["capabilities"]
    assert device_access == []
    assert engine._device is None
    assert not list((tmp_path / "leases").glob("*.json"))


def test_mcp_session_and_reach_dispatch_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine()
    calls: list[tuple[str, dict[str, object]]] = []

    def session_start(goal: str, **kwargs: object) -> dict[str, object]:
        calls.append(("session_start", {"goal": goal, **kwargs}))
        return {"ok": True, "session_id": "session-1"}

    def session_review(*, session_id: str | None = None) -> dict[str, object]:
        calls.append(("session_review", {"session_id": session_id}))
        return {"ok": True, "session_id": session_id}

    def session_autopilot(**kwargs: object) -> dict[str, object]:
        calls.append(("session_autopilot", kwargs))
        return {"ok": True, "autopilot": {"steps_executed": 2}}

    def session_finish(*, session_id: str | None = None) -> dict[str, object]:
        calls.append(("session_finish", {"session_id": session_id}))
        return {"ok": True, "session_id": session_id}

    def reach(goal: str, **kwargs: object) -> dict[str, object]:
        calls.append(("reach", {"goal": goal, **kwargs}))
        return {"ok": True, "goal": goal, "via": "goto"}

    monkeypatch.setattr(eng, "session_start", session_start, raising=False)
    monkeypatch.setattr(eng, "session_review", session_review, raising=False)
    monkeypatch.setattr(eng, "session_autopilot", session_autopilot, raising=False)
    monkeypatch.setattr(eng, "session_finish", session_finish, raising=False)
    monkeypatch.setattr(eng, "reach", reach, raising=False)
    server = build_server(eng)

    async def run() -> list[dict[str, object]]:
        async with create_connected_server_and_client_session(server) as client:
            results = [
                await client.call_tool("session_start", {"goal": "open settings", "headed": True}),
                await client.call_tool("reach", {"goal": "settings", "poll_ms": 25}),
                await client.call_tool("session_review", {"session_id": "session-1"}),
                await client.call_tool(
                    "session_autopilot",
                    {"session_id": "session-1", "max_steps": 4, "max_duration_ms": 12000},
                ),
                await client.call_tool("session_finish", {"session_id": "session-1"}),
            ]
            return [json.loads(_first_text(result)) for result in results]

    outputs = anyio.run(run)
    assert all(output["ok"] for output in outputs)
    assert calls == [
        (
            "session_start",
            {
                "goal": "open settings",
                "start_emulator": True,
                "headed": True,
                "audio": False,
                "avd": None,
                "needs": None,
                "package": None,
                "activity": None,
            },
        ),
        (
            "reach",
            {
                "goal": "settings",
                "until": None,
                "timeout_ms": 30000,
                "interval_ms": 25,
                "allow_unsafe": False,
                "allow_destructive": False,
                "assist": False,
            },
        ),
        ("session_review", {"session_id": "session-1"}),
        (
            "session_autopilot",
            {"session_id": "session-1", "max_steps": 4, "max_duration_ms": 12000},
        ),
        ("session_finish", {"session_id": "session-1"}),
    ]


def test_mcp_finish_forgets_the_stopped_serial_even_when_other_cleanup_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "session_finish",
        lambda **_kwargs: {
            "ok": False,
            "cleanup": [
                {
                    "action": "owned_emulator_stop",
                    "ok": True,
                    "result": {"ok": True, "stopped": ["emulator-5592"]},
                }
            ],
            "errors": [{"action": "network_restore", "message": "still offline"}],
        },
    )
    tracked = _mcp_started_serials()
    previous = set(tracked)
    tracked.clear()
    tracked.update({"emulator-5592", "emulator-5594"})

    async def run() -> dict[str, object]:
        server = build_server(engine)
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("session_finish", {"session_id": "session-1"})
            return json.loads(_first_text(result))

    try:
        output = anyio.run(run)
        assert output["ok"] is False
        assert tracked == {"emulator-5594"}
    finally:
        tracked.clear()
        tracked.update(previous)


def test_mcp_analyze_screen_returns_schema_valid_json() -> None:
    server = build_server(_engine())

    async def run() -> str:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            assert not result.isError, result
            return _first_text(result)

    text = anyio.run(run)
    data = json.loads(text)
    assert {"schema_version", "screen", "elements", "meta"} <= set(data)
    assert data["schema_version"] == 1
    # Schema-valid against the pydantic source of truth.
    parsed = AnalyzeResult.model_validate(data)
    assert parsed.screen.source.value == "hierarchy"
    assert len(parsed.elements) == 3


def test_mcp_closes_each_caller_turn_and_compares_the_emitted_screen(tmp_path: Path) -> None:
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        cache={"dir": str(tmp_path / "cache")},
    )
    device = FakeDevice(hierarchy_xml=HIERARCHY_XML)
    engine = Engine(cfg, device=device)
    server = build_server(engine)

    async def run() -> tuple[dict, dict]:  # type: ignore[type-arg]
        async with create_connected_server_and_client_session(server) as client:
            first_result = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            assert engine._caller_turn is None  # noqa: SLF001
            device._xml = HIERARCHY_XML.replace("Hello", "Different screen")  # noqa: SLF001
            second_result = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            assert engine._caller_turn is None  # noqa: SLF001
            return json.loads(_first_text(first_result)), json.loads(_first_text(second_result))

    first, second = anyio.run(run)

    assert first["meta"]["fingerprint"] != second["meta"]["fingerprint"]
    assert second["meta"]["caller"]["previous_screen_gone"] is True


def test_mcp_network_offline_and_restore_roundtrip() -> None:
    server = build_server(_engine())

    async def run() -> tuple[dict, dict]:  # type: ignore[type-arg]
        async with create_connected_server_and_client_session(server) as client:
            offline = await client.call_tool("network_offline", {"verify": True, "timeout_ms": 0})
            restored = await client.call_tool("network_restore", {"timeout_ms": 0})
            return json.loads(_first_text(offline)), json.loads(_first_text(restored))

    offline, restored = anyio.run(run)
    assert offline["ok"] is True
    assert offline["state"]["offline"] is True
    assert restored["ok"] is True
    assert restored["state"]["offline"] is False


def test_mcp_network_profile_roundtrip() -> None:
    server = build_server(_engine())

    async def run() -> tuple[dict, dict, dict]:  # type: ignore[type-arg]
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.call_tool("network_profile_list", {})
            applied = await client.call_tool(
                "network_profile_apply",
                {"profile": "wifi-only", "timeout_ms": 0},
            )
            restored = await client.call_tool("network_profile_restore", {"timeout_ms": 0})
            return (
                json.loads(_first_text(listed)),
                json.loads(_first_text(applied)),
                json.loads(_first_text(restored)),
            )

    listed, applied, restored = anyio.run(run)
    assert "lossy" in listed["names"]
    assert applied["verified"] is True
    assert restored["verified"] is True


def test_mcp_observed_actions_are_named_and_force_analysis() -> None:
    eng = _engine()
    server = build_server(eng)

    async def run() -> tuple[dict, dict, dict]:  # type: ignore[type-arg]
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            tap_tool = next(t for t in listed.tools if t.name == "tap_and_analyze")
            before = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            button = next(
                e for e in json.loads(_first_text(before))["elements"] if e["text"] == "Continue"
            )
            tapped = await client.call_tool("tap_and_analyze", {"rid": "continue_btn"})
            old = await client.call_tool("tap", {"id": button["id"]})
            return (
                json.loads(_first_text(tapped)),
                json.loads(_first_text(old)),
                tap_tool.inputSchema,
            )

    tapped, old, schema = anyio.run(run)
    assert tapped["observation_present"] is True
    assert tapped["observation"]["elements"]
    assert {"rid", "text", "desc", "index", "first"} <= set(schema["properties"])
    assert {tuple(branch["required"]) for branch in schema["oneOf"]} == {
        ("id",),
        ("rid",),
        ("text",),
        ("desc",),
    }
    assert "observe" not in schema["properties"], "the renamed tool cannot contradict its name"
    assert old["error"]["code"] == "usage"
    assert "tap_and_analyze" in old["error"]["message"]


def test_mcp_analyzed_actions_advertise_and_fold_until(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine()
    awaited: dict[str, object] = {}

    def fake_await(predicate: str, **kwargs: object) -> ActionResult:
        awaited.update(predicate=predicate, **kwargs)
        return ActionResult(
            ok=True,
            action="await",
            await_outcome="satisfied",
            await_terms=[{"term": "rid:destination", "present": True, "satisfied": True}],
            elapsed_ms=7,
            observation=eng.analyze(source="hierarchy"),
            observation_present=True,
        )

    monkeypatch.setattr(eng, "await_predicate", fake_await)
    server = build_server(eng)

    async def run() -> tuple[dict, dict]:  # type: ignore[type-arg]
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            tap_tool = next(tool for tool in listed.tools if tool.name == "tap_and_analyze")
            before = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            button = next(
                element
                for element in json.loads(_first_text(before))["elements"]
                if element["text"] == "Continue"
            )
            result = await client.call_tool(
                "tap_and_analyze",
                {
                    "id": button["id"],
                    "until": "rid:destination",
                    "until_timeout": 1234,
                    "until_poll": 42,
                },
            )
            return json.loads(_first_text(result)), tap_tool.inputSchema

    data, schema = anyio.run(run)
    assert {"until", "until_timeout", "until_poll"} <= set(schema["properties"])
    assert data["action"] == "tap" and data["await_outcome"] == "satisfied"
    assert data["observation_present"] is True
    assert awaited == {
        "predicate": "rid:destination",
        "timeout_ms": 1234,
        "poll_ms": 42,
        "observe": True,
        "adopt_action": True,
    }


def test_mcp_until_is_validated_before_action() -> None:
    eng = _engine()
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            before = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            button = next(
                element
                for element in json.loads(_first_text(before))["elements"]
                if element["text"] == "Continue"
            )
            result = await client.call_tool(
                "tap_and_analyze", {"id": button["id"], "until": "rid:"}
            )
            return json.loads(_first_text(result))

    before_calls = list(eng.device.calls)  # type: ignore[attr-defined]
    data = anyio.run(run)
    assert data["error"]["code"] == "usage"
    assert eng.device.calls == before_calls  # type: ignore[attr-defined]


def test_mcp_negative_only_until_is_rejected_before_action() -> None:
    eng = _engine()
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            before = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            button = next(
                element
                for element in json.loads(_first_text(before))["elements"]
                if element["text"] == "Continue"
            )
            result = await client.call_tool(
                "tap_and_analyze", {"id": button["id"], "until": "!text:Loading"}
            )
            return json.loads(_first_text(result))

    before_calls = list(eng.device.calls)  # type: ignore[attr-defined]
    data = anyio.run(run)
    assert "positive arrival" in data["error"]["message"]
    assert eng.device.calls == before_calls  # type: ignore[attr-defined]


def test_mcp_timeout_preserves_valid_delta_and_hints_regex_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine()
    previous = {
        "changed": True,
        "node_count_before": 2,
        "node_count_after": 3,
        "text_added": ["Intermediate"],
    }

    monkeypatch.setattr(
        eng,
        "await_predicate",
        lambda *_a, **_k: ActionResult(
            ok=False,
            action="await",
            await_outcome="timeout",
            change={
                "changed": False,
                "node_count_before": None,
                "node_count_after": 3,
                "detail": "no pre-action snapshot — deltas unavailable",
            },
        ),
    )

    out = _fold_action_until(
        eng,
        "tap_and_analyze",
        {"until": "text:^Destination.*$", "until_timeout": 5},
        {"ok": True, "observation_present": True, "change": previous},
    )

    assert out["change"] == previous
    assert "looks regex-like" in out["note"]
    assert "literal contains" in out["note"]
    assert "aua await-and-analyze" in out["note"]
    assert "--match regex" in out["note"]


def test_mcp_folded_action_preserves_structured_settled_arrival_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine()
    mismatch = {
        "code": "arrival_mismatch",
        "original_predicate": "text:Old title,!text:Loading",
        "suggested_positive_predicates": ["rid:recentItem"],
        "recommended_call": ("aua await-and-analyze 'rid:recentItem,!text:Loading' --observe"),
        "recommended_mcp_call": {
            "tool": "await_and_analyze",
            "arguments": {"predicate": "rid:recentItem,!text:Loading"},
        },
        "action_repeated": False,
    }
    monkeypatch.setattr(
        eng,
        "await_predicate",
        lambda *_args, **_kwargs: ActionResult(
            ok=False,
            action="await",
            await_outcome="settled-unmet",
            arrival_mismatch=mismatch,
            note="The action ran once; reuse this destination and do not repeat it.",
        ),
    )

    out = _fold_action_until(
        eng,
        "tap_and_analyze",
        {"until": "text:Old title,!text:Loading"},
        {"ok": True, "action": "tap", "observation_present": True},
    )

    assert out["action"] == "tap"
    assert out["await_outcome"] == "settled-unmet"
    assert out["arrival_mismatch"] == mismatch
    assert "do not repeat" in out["note"]


def test_mcp_await_and_analyze_returns_term_evidence() -> None:
    server = build_server(_engine())

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(
                "await_and_analyze",
                {"predicate": "text:Continue,!text:Loading", "timeout_ms": 0},
            )
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["await_outcome"] == "satisfied"
    assert [term["satisfied"] for term in data["await_terms"]] == [True, True]
    assert data["observation_present"] is True


def test_mcp_top_level_call_is_redacted_and_journaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import journal

    eng = _engine()

    def database_query(*args: object, **kwargs: object) -> dict[str, object]:
        return {"ok": True, "rows": [["value"]]}

    monkeypatch.setattr(eng, "database_query", database_query)
    server = build_server(eng)

    async def run() -> None:
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool(
                "database_query",
                {
                    "package": "com.example.fixture",
                    "database": "fixture.db",
                    "sql": "SELECT value FROM records WHERE id = ?",
                    "parameters": ["private-value"],
                },
            )

    anyio.run(run)
    event = journal.read_since(eng.config.cache.dir, eng.device.serial, limit=5)[-1]
    assert event["source"] == "mcp" and event["cmd"] == "database_query"
    assert event["owner"]
    assert event["duration_ms"] >= 0
    assert event["args"]["sql"].startswith("<redacted SQL:")
    assert event["args"]["parameters"] == "<redacted>"
    assert event["extra"]["invocation_id"]
    assert event["result"]["ok"] is True
    detail = journal.read_detail(
        eng.config.cache.dir, eng.device.serial, event["detail_id"]
    )
    assert detail is not None
    assert detail["request"]["args"]["sql"].startswith("<redacted SQL:")
    assert detail["request"]["args"]["parameters"] == "<redacted>"
    assert detail["response"]["result"] == {"ok": True, "rows": [["value"]]}


def test_mcp_journal_marks_only_an_exact_expected_error_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import journal
    from android_ui_analyser.errors import UsageError

    eng = _engine()
    monkeypatch.setattr(
        eng,
        "flow_delete",
        lambda _name: (_ for _ in ()).throw(UsageError("intentional probe")),
    )
    server = build_server(eng)

    async def run() -> None:
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool(
                "flow_delete",
                {"name": "missing", "expect_error": "usage"},
            )

    anyio.run(run)
    event = journal.read_since(eng.config.cache.dir, eng.device.serial, limit=5)[-1]
    assert event["ok"] is False
    assert event["extra"]["expected_error_code"] == "usage"
    assert event["extra"]["expected_error_matched"] is True
    detail = journal.read_detail(
        eng.config.cache.dir, eng.device.serial, event["detail_id"]
    )
    assert detail is not None
    assert detail["request"]["args"]["expect_error"] == "usage"


def test_mcp_map_audit_and_reconcile_plan_roundtrip() -> None:
    server = build_server(_engine())

    async def run() -> tuple[dict, dict]:  # type: ignore[type-arg]
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("analyze_screen", {"source": "hierarchy"})
            audited = await client.call_tool("map_audit", {"package": "com.test.app"})
            planned = await client.call_tool("reconcile_plan", {"package": "com.test.app"})
            return json.loads(_first_text(audited)), json.loads(_first_text(planned))

    audit, plan = anyio.run(run)
    assert audit["package"] == "com.test.app"
    assert plan["package"] == "com.test.app"
    assert isinstance(plan["tasks"], list)


def test_mcp_has_tool_roundtrip() -> None:
    server = build_server(_engine())

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("has", {"text": "Continue"})
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["found"] is True
    assert data["source"] == "hierarchy"


def test_mcp_analyze_via_monkeypatched_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same as AC8 but exercising the lazy connect path (engine.connect patched)."""
    device = FakeDevice(hierarchy_xml=HIERARCHY_XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    server = build_server(Engine(make_config()))  # no device passed → lazy connect

    async def run() -> str:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            assert not result.isError, result
            return _first_text(result)

    data = json.loads(anyio.run(run))
    AnalyzeResult.model_validate(data)


def test_mcp_long_press_drives_device() -> None:
    eng = _engine()
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("analyze_screen", {"source": "hierarchy"})
            result = await client.call_tool("long_press_and_analyze", {"id": 1})
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] is True and data["action"] == "long-press"
    assert any(c[0] == "long_click" for c in eng.device.calls)  # type: ignore[attr-defined]


def test_mcp_wait_stable_settles_on_static_screen() -> None:
    server = build_server(_engine())  # FakeDevice returns identical frames

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(
                "wait_stable_and_analyze", {"settle": 100, "interval": 10}
            )
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] is True and data["action"] == "wait-stable"


def test_mcp_goto_error_payload_when_memory_disabled() -> None:
    cfg = make_config(memory={"enabled": False})
    server = build_server(Engine(cfg, device=FakeDevice(hierarchy_xml=HIERARCHY_XML)))

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("goto", {"goal": "settings"})
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["error"]["code"] == "usage"  # structured AuaError payload, not a crash


def test_mcp_flow_run_dry_run_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from android_ui_analyser.flows import Flow, FlowStore
    from android_ui_analyser.memory import RouteStep

    cfg = make_config(memory={"dir": str(tmp_path / "home")})
    FlowStore(cfg.memory).save(
        Flow(name="mcpflow", app="com.test.app", steps=[RouteStep(kind="tap", label="Continue")])
    )
    eng = Engine(cfg, device=FakeDevice(hierarchy_xml=HIERARCHY_XML))
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("flow_run", {"name": "mcpflow", "dry_run": True})
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] and data["dry_run"] and data["steps"][0]["step"] == "tap 'Continue'"


def test_mcp_flow_run_accepts_inline_yaml_and_artifact_options(tmp_path: Path) -> None:
    server = build_server(_engine())
    artifacts = tmp_path / "mcp-flow-artifacts"

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(
                "flow_run",
                {
                    "yaml": "steps:\n  - assert: {text: Hello, count: 1}\n",
                    "dry_run": True,
                    "artifacts_dir": str(artifacts),
                    "evidence": "none",
                    "junit": True,
                },
            )
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] is True and data["source"] == "inline_yaml"
    assert Path(data["artifacts"]["junit"]).is_file()


def test_mcp_navigate_requires_planner(monkeypatch) -> None:
    # navigate without planner.enabled → structured usage error, not a crash.
    from android_ui_analyser.engine import Engine

    device = FakeDevice(hierarchy_xml=HIERARCHY_XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    server = build_server(Engine(make_config()))

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("navigate", {"goal": "open reports"})
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["error"]["code"] == "usage"


def test_mcp_goto_accepts_assist_param() -> None:
    # The assist param is accepted by the schema (memory disabled → route_unknown, no crash).
    from android_ui_analyser.engine import Engine

    cfg = make_config(memory={"enabled": False})
    server = build_server(Engine(cfg, device=FakeDevice(hierarchy_xml=HIERARCHY_XML)))

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("goto", {"goal": "x", "assist": True})
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert "error" in data  # memory disabled → usage error, but the param was accepted


def test_mcp_hide_keyboard_drives_device() -> None:
    eng = _engine()
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("hide_keyboard_and_analyze", {})
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] is True and data["action"] == "hide-keyboard"
    assert data["observation_present"] is True
    assert ("hide_keyboard", ()) in eng.device.calls  # type: ignore[attr-defined]


def test_mcp_clipboard_set() -> None:
    eng = _engine()
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("clipboard_set", {"text": "mcp-clip"})
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] is True and data["detail"] == "mcp-clip"
    assert eng.device.get_clipboard() == "mcp-clip"  # type: ignore[attr-defined]


def test_mcp_resolve_dispatch_stub() -> None:
    """MCP resolve calls getattr(engine, 'resolve') — smoke-test the wiring with a stub."""
    from android_ui_analyser.schema import ResolveResult

    eng = _engine()

    def _fake_resolve(target: int | str) -> ResolveResult:
        return ResolveResult(ok=True, from_id=1 if isinstance(target, int) else None, to_id=2)

    eng.resolve = _fake_resolve  # type: ignore[method-assign]
    server = build_server(eng)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("resolve", {"target": 1})
            assert not result.isError, result
            return json.loads(_first_text(result))

    data = anyio.run(run)
    assert data["ok"] is True and data.get("to_id") == 2


def test_mcp_emulator_cleanup_tracks_serials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import mcp_server as mcp

    mcp._MCP_STARTED_SERIALS.clear()
    mcp._MCP_STARTED_OWNERS.clear()
    mcp._MCP_STARTED_SERIALS.add("emulator-9998")
    stopped: list[str] = []

    def fake_stop(**kwargs):  # type: ignore[no-untyped-def]
        ser = kwargs.get("serial")
        if ser:
            stopped.append(str(ser))
            return {"ok": True, "stopped": [ser]}
        return {"ok": True, "stopped": []}

    monkeypatch.setattr("android_ui_analyser.emulator.stop", fake_stop)
    out = mcp.cleanup_mcp_emulators(tmp_path)
    assert "emulator-9998" in out["stopped"]
    assert "emulator-9998" not in mcp._MCP_STARTED_SERIALS


def test_mcp_emulator_cleanup_preserves_a_transferred_live_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser import mcp_server as mcp

    starts = {111: "source", 222: "child"}
    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    source = leases.LeaseOwner("orchestrator", pid=111, started="source")
    child = leases.LeaseOwner("child", pid=222, started="child")
    serial = "emulator-9998"
    assert leases.acquire(tmp_path, serial, owner=source)
    offered = leases.create_handoff(tmp_path, serial, owner=source)
    leases.accept_handoff(tmp_path, offered["token"], owner=child)
    mcp._MCP_STARTED_SERIALS.clear()
    mcp._MCP_STARTED_OWNERS.clear()
    mcp._MCP_STARTED_SERIALS.add(serial)
    mcp._MCP_STARTED_OWNERS.add("started-by-orchestrator")
    stopped: list[dict[str, object]] = []

    class EmulatorService:
        def stop(self, **kwargs: object) -> dict[str, object]:
            stopped.append(kwargs)
            return {"ok": True, "stopped": [serial]}

    class Platform:
        def capability(self, name: str) -> EmulatorService:
            assert name == "virtual_devices"
            return EmulatorService()

    result = mcp.cleanup_mcp_emulators(tmp_path, platform=Platform())

    assert result["stopped"] == []
    assert result["preserved"] == [serial]
    assert stopped == []
    assert leases.holder(tmp_path, serial) == "child"
    assert not mcp._MCP_STARTED_SERIALS
    assert not mcp._MCP_STARTED_OWNERS
