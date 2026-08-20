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


def test_the_screen_video_recorder_keeps_explicit_names_and_unambiguous_compatibility_aliases() -> None:
    tools = _tools()
    names = set(tools)
    assert {"screen_record_start", "screen_record_stop"} <= names, (
        "the MP4 screen recorder must say `screen` in its published name"
    )
    assert {"record_start", "record_stop"} <= names, "renaming must not break existing clients"
    for alias in ("record_start", "record_stop"):
        description = tools[alias].description.lower()
        assert "deprecated alias" in description
        assert "screen" in description


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


def test_deprecated_record_aliases_reach_the_same_screen_video_engine_path(
    monkeypatch: Any,
) -> None:
    engine = Engine(make_config(), device=FakeDevice(hierarchy_xml=_HIERARCHY))
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        engine,
        "record_start",
        lambda path=None: calls.append(("start", path)) or {"ok": True},
    )
    monkeypatch.setattr(
        engine,
        "record_stop",
        lambda path: calls.append(("stop", path)) or {"ok": True},
    )

    _dispatch(engine, "record_start", {"path": "screen.mp4"})
    _dispatch(engine, "record_stop", {"path": "screen.mp4"})

    assert calls == [("start", "screen.mp4"), ("stop", "screen.mp4")]


def test_mock_rewrite_is_published_so_the_write_side_is_not_stub_only() -> None:
    """`rewrite_rule` shipped implemented and addon-tested with no caller-facing surface
    at all, so an agent could stub an endpoint but never patch a real response."""
    tool = _tools().get("mock_rewrite")
    assert tool is not None, "MCP has no mock_rewrite tool; response patching is unreachable"
    props = tool.inputSchema["properties"]
    for field in ("method", "path", "host", "status", "headers", "set_json", "delete_json"):
        assert field in props, f"mock_rewrite must accept {field!r}"
    assert tool.inputSchema["required"] == ["method", "path"]


def test_mock_rewrite_dispatches_to_the_same_engine_method_as_the_cli(
    monkeypatch: Any, tmp_path: Any
) -> None:
    seen: dict[str, Any] = {}

    def fake(self: Engine, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update({"method": method, "path": path, **kwargs})
        return {"ok": True, "action": "mock-rewrite"}

    monkeypatch.setattr(Engine, "mock_rewrite", fake)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=FakeDevice())
    _dispatch(
        engine,
        "mock_rewrite",
        {
            "method": "GET",
            "path": "/v1/feed",
            "host": "api.example.test",
            "status": 429,
            "set_json": {"items[0].title": "patched"},
        },
    )
    assert seen["host"] == "api.example.test"
    assert seen["status"] == 429
    assert seen["set_json"] == {"items[0].title": "patched"}
    # Unset options must arrive as None, not as an empty dict the rule builder would
    # read as "the caller asked for no headers".
    assert seen["headers"] is None
    assert seen["delete_json"] is None
