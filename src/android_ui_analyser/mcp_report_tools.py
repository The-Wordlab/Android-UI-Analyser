"""MCP tools for evidence pages; none of them touch a device."""

from __future__ import annotations

from typing import Any

import mcp.types as types


def tool_definitions() -> list[types.Tool]:
    text = {"type": "string"}
    texts = {"type": "array", "items": text}
    return [
        types.Tool(
            name="report_guide",
            description=(
                "Read how to turn a run's recording and screenshots into a page people can open: "
                "what to collect, the video and chapter markup, then report_player and "
                "report_publish. How the page looks is up to you."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="report_player",
            description=(
                "Put AUA's video player and image viewer into HTML pages just before </body>; "
                "running it again replaces the earlier copy. Every <video controls> becomes a "
                "player that seeks from memory, elements with data-seek become chapters, and every "
                "<img> opens larger with zoom when clicked."
            ),
            inputSchema={
                "type": "object",
                "properties": {"pages": {**texts, "minItems": 1, "description": "HTML page paths."}},
                "required": ["pages"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="report_publish",
            description=(
                "Publish a folder (index.html at the top) to Drophere and check the result: link, "
                "expiry, who can open it, and that every file loads. With allow_domains or "
                "allow_emails, access is set before the upload. Without a Drophere API key the "
                "page is public for 24 hours. This uploads to an external host: publish only what "
                "the user means to share."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {**text, "description": "Absolute path of the folder to publish."},
                    "ttl": {**text, "description": "Expiry such as 24h or 7d (default 7d); needs an API key."},
                    "allow_domains": {**texts, "description": "Email domains whose people may open it."},
                    "allow_emails": {**texts, "description": "Email addresses that may open it."},
                    "slug": {**text, "description": "Publish a new version at this existing link."},
                },
                "required": ["folder"],
                "additionalProperties": False,
            },
        ),
    ]


TOOL_NAMES = frozenset(tool.name for tool in tool_definitions())


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    from . import report

    if name == "report_guide":
        return {"ok": True, "guide": report.guide()}
    if name == "report_player":
        return report.add_player(list(args.get("pages") or []))
    return report.publish(
        str(args.get("folder") or ""),
        ttl=args.get("ttl"),
        allow_domains=args.get("allow_domains"),
        allow_emails=args.get("allow_emails"),
        slug=args.get("slug"),
    )
