"""MCP wrapper tests (PRD §13.1 AC8).

Drives the MCP server **in-process** with the SDK's in-memory client/server session
helper, so no subprocess and no device are needed. We list the tools (assert the core
ones are present) and call ``analyze_screen`` against an :class:`Engine` backed by a
:class:`FakeDevice`, asserting the returned content is schema-valid JSON.
"""

from __future__ import annotations

import json

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
    assert "tap" in names
    assert "has" in names
    # Full 1:1 surface (PRD §11).
    assert {
        "analyze_screen",
        "tap",
        "input",
        "swipe",
        "key",
        "wait",
        "has",
        "screenshot",
        "inspect",
        "list_devices",
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
