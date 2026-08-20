"""The HTTP mock surface must be reachable over MCP, and not confusable with screen video.

An MCP-only agent could start the proxy (``proxy_start``) and replay a cassette (``mock_replay``)
but had no tool to *author* one: ``mock record`` and ``mock map`` existed only on the CLI and the
daemon socket, so the whole write side of the mock surface was CLI-only. Worse, the screen-video
recorder was published as ``record_start``/``record_stop`` — names that sit beside ``mock_replay``
in a flat tool list and read as traffic capture.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _dispatch, _tool_definitions
from conftest import FakeDevice, make_config

_HIERARCHY = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<hierarchy rotation="0">'
    '<node index="0" class="android.widget.TextView" text="Hi" bounds="[0,0][1080,120]"/>'
    "</hierarchy>"
)


def _tools() -> dict[str, Any]:
    return {tool.name: tool for tool in _tool_definitions()}


def test_mock_record_is_published_with_the_cli_arguments() -> None:
    tool = _tools().get("mock_record")
    assert tool is not None, "MCP has no mock_record tool; a cassette can only be authored via CLI"
    props = tool.inputSchema["properties"]
    assert "action" in props, "mock_record must take the CLI's start|stop action"
    assert "name" in props, "mock_record must take the cassette name"
    assert set(props["action"].get("enum") or []) == {"start", "stop"}


def test_mock_map_is_published_with_the_cli_arguments() -> None:
    tool = _tools().get("mock_map")
    assert tool is not None, "MCP has no mock_map tool; no ad-hoc route stub is possible over MCP"
    props = tool.inputSchema["properties"]
    for key in ("method", "path", "status", "body"):
        assert key in props, f"mock_map is missing {key!r}"
    assert set(tool.inputSchema.get("required") or []) >= {"method", "path"}


def test_the_screen_video_recorder_cannot_be_mistaken_for_traffic_capture() -> None:
    names = set(_tools())
    assert {"screen_record_start", "screen_record_stop"} <= names, (
        "the MP4 screen recorder must say `screen` in its published name"
    )
    # A description does not save a model at tool-selection time — that is the whole lesson
    # behind `_ANALYZED_TOOL_NAMES`. The bare verbs read as "record traffic" next to mock_replay.
    assert "record_start" not in names
    assert "record_stop" not in names


def test_dispatch_routes_the_mock_tools_to_the_engine(monkeypatch: Any) -> None:
    """The published tools must reach the real engine methods, with the CLI's arguments."""
    engine = Engine(make_config(), device=FakeDevice(hierarchy_xml=_HIERARCHY))
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_mock_record(action: str, name: str | None = None) -> dict[str, Any]:
        calls.append(("mock_record", (action, name), {}))
        return {"ok": True, "action": "mock-record"}

    def fake_mock_map(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(("mock_map", (method, path), kwargs))
        return {"ok": True, "action": "mock-map"}

    monkeypatch.setattr(engine, "mock_record", fake_mock_record)
    monkeypatch.setattr(engine, "mock_map", fake_mock_map)

    _dispatch(engine, "mock_record", {"action": "start", "name": "hub"})
    _dispatch(engine, "mock_map", {"method": "GET", "path": "/hub", "status": 500, "body": "{}"})

    assert calls[0] == ("mock_record", ("start", "hub"), {})
    assert calls[1][0] == "mock_map"
    assert calls[1][1] == ("GET", "/hub")
    assert calls[1][2]["status"] == 500
    assert calls[1][2]["body"] == "{}"
