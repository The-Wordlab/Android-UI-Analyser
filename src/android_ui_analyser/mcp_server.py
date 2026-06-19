"""MCP server wrapper — a *thin* adapter exposing the engine over the official MCP SDK.

Tools map 1:1 to :class:`~android_ui_analyser.engine.Engine` methods (PRD §11). Each
tool builds nothing of its own: it calls a single shared :class:`Engine` and returns the
pydantic result serialised to JSON (``model_dump``) as a text content block — the exact
same schema the CLI emits. No perception logic lives here.

``build_server(engine)`` returns a configured low-level :class:`mcp.server.Server` so
tests can drive it in-process; ``run_stdio()`` is what ``aua mcp`` invokes.
"""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from .config import load_config
from .engine import Engine
from .errors import AuaError

SERVER_NAME = "android-ui-analyser"


# --------------------------------------------------------------------------- tool specs


def _tool_definitions() -> list[types.Tool]:
    """The MCP tool catalogue (input schemas only; output is JSON text content)."""
    match_enum = ["exact", "contains", "regex"]
    source_enum = ["auto", "hierarchy", "vision"]
    return [
        types.Tool(
            name="analyze_screen",
            description="Analyze the current screen and return Set-of-Marks JSON "
            "(elements with stable ids, bounds, centers).",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": source_enum, "default": "auto"},
                    "with_ocr": {"type": "boolean"},
                    "query": {
                        "type": "string",
                        "description": "Return the single best-matching element.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="has",
            description="Quick check: is this text on screen right now? Returns "
            "{found, source, bounds?}.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "match": {"type": "string", "enum": match_enum, "default": "contains"},
                    "ignore_case": {"type": "boolean", "default": False},
                    "ocr_fallback": {"type": "boolean", "default": True},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="tap",
            description="Tap the element with the given id (from the last analyze).",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="input",
            description="Type text into the element with the given id; optional IME submit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="swipe",
            description="Swipe a direction (up|down|left|right) or by explicit "
            "[x1,y1,x2,y2] coordinates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "coords": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="key",
            description="Press a hardware/navigation key (back|home|enter|recents|KEYCODE_*).",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="wait",
            description="Wait for text to appear (for_) or for the UI to go idle.",
            inputSchema={
                "type": "object",
                "properties": {
                    "for_": {"type": "string"},
                    "idle": {"type": "boolean", "default": False},
                    "timeout": {"type": "integer", "default": 5000},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="screenshot",
            description="Save a screenshot; set annotate=true to overlay Set-of-Marks numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "annotate": {"type": "boolean", "default": False},
                    "path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="inspect",
            description="Return full attributes for one element id from the last analyze.",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="list_devices",
            description="List attached devices (serial, model, android version, state).",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    ]


# --------------------------------------------------------------------------- dispatch


def _dispatch(engine: Engine, name: str, args: dict[str, Any]) -> Any:
    """Call the engine method for ``name`` and return a JSON-serialisable payload."""
    if name == "analyze_screen":
        result = engine.analyze(
            source=args.get("source", "auto"),
            with_ocr=args.get("with_ocr"),
            query=args.get("query"),
        )
        return result.model_dump(mode="json")
    if name == "has":
        return engine.has(
            args["text"],
            match=args.get("match", "contains"),
            ignore_case=args.get("ignore_case", False),
            ocr_fallback=args.get("ocr_fallback", True),
        ).model_dump(mode="json")
    if name == "tap":
        return engine.tap(int(args["id"])).model_dump(mode="json")
    if name == "input":
        return engine.input_text(
            int(args["id"]), args["text"], submit=args.get("submit", False)
        ).model_dump(mode="json")
    if name == "swipe":
        coords = args.get("coords")
        coord_tuple: tuple[int, int, int, int] | None = None
        if coords:
            x1, y1, x2, y2 = (int(c) for c in coords)
            coord_tuple = (x1, y1, x2, y2)
        return engine.swipe(direction=args.get("direction"), coords=coord_tuple).model_dump(
            mode="json"
        )
    if name == "key":
        return engine.key(args["name"]).model_dump(mode="json")
    if name == "wait":
        return engine.wait(
            for_=args.get("for_"),
            idle=args.get("idle", False),
            timeout_ms=args.get("timeout", 5000),
        ).model_dump(mode="json")
    if name == "screenshot":
        return engine.screenshot(args.get("path"), annotate=args.get("annotate", False)).model_dump(
            mode="json"
        )
    if name == "inspect":
        return engine.inspect(int(args["id"])).model_dump(mode="json")
    if name == "list_devices":
        return [d.model_dump(mode="json") for d in engine.list_devices()]
    raise AuaError(f"unknown tool '{name}'", code="usage")


# --------------------------------------------------------------------------- server


def build_server(engine: Engine) -> Server:
    """Build a low-level MCP :class:`Server` bound to ``engine`` (for stdio + tests)."""
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        try:
            payload = _dispatch(engine, name, arguments or {})
            text = json.dumps(payload, ensure_ascii=False)
        except AuaError as err:
            text = json.dumps(err.to_dict(), ensure_ascii=False)
            return [types.TextContent(type="text", text=text)]
        return [types.TextContent(type="text", text=text)]

    return server


def build_default_engine() -> Engine:
    """Build an :class:`Engine` from the standard layered config (device connects lazily)."""
    return Engine(load_config())


def run_stdio() -> None:
    """Run the MCP server over stdio — the entry point used by ``aua mcp``."""
    import anyio
    from mcp.server.stdio import stdio_server

    engine = build_default_engine()
    server = build_server(engine)

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_serve)


__all__ = ["SERVER_NAME", "build_default_engine", "build_server", "run_stdio"]
