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
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.schema import AnalyzeResult
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
        "media_add",
        "record_start",
        "record_stop",
        "clock_set",
        "app",
        "app_launch_and_analyze",
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
    } <= set(names)


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
            tapped = await client.call_tool("tap_and_analyze", {"id": button["id"]})
            old = await client.call_tool("tap", {"id": button["id"]})
            return (
                json.loads(_first_text(tapped)),
                json.loads(_first_text(old)),
                tap_tool.inputSchema,
            )

    tapped, old, schema = anyio.run(run)
    assert tapped["observation_present"] is True
    assert tapped["observation"]["elements"]
    assert "observe" not in schema["properties"], "the renamed tool cannot contradict its name"
    assert old["error"]["code"] == "usage"
    assert "tap_and_analyze" in old["error"]["message"]


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


def test_mcp_navigate_requires_planner(monkeypatch) -> None:
    # navigate without planner.enabled → structured usage error, not a crash.
    from android_ui_analyser.engine import Engine

    device = FakeDevice(hierarchy_xml=HIERARCHY_XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    server = build_server(Engine(make_config()))

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("navigate", {"goal": "open images"})
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
