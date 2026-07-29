"""MCP server wrapper — a *thin* adapter exposing the engine over the official MCP SDK.

Tools map 1:1 to :class:`~android_ui_analyser.engine.Engine` methods (PRD §11). Each
tool builds nothing of its own: it calls a single shared :class:`Engine` and returns the
pydantic result serialised to JSON (``model_dump``) as a text content block — the exact
same schema the CLI emits. No perception logic lives here.

``build_server(engine)`` returns a configured low-level :class:`mcp.server.Server` so
tests can drive it in-process; ``run_stdio()`` is what ``aua mcp`` invokes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from .config import load_config
from .engine import Engine
from .errors import AuaError

SERVER_NAME = "android-ui-analyser"


def _with_image(engine: Engine, args: dict[str, Any]) -> bool | str | None:
    """Per-call ``with_image``, else the engine default set by ``configure``."""
    if "with_image" in args:
        return args["with_image"]
    return getattr(engine, "_default_with_image", None)


def _selector_from_args(args: dict[str, Any]) -> dict[str, Any] | None:
    """Build ``resolve_selector`` kwargs from optional rid/text/desc (+ index/first)."""
    rid, text, desc = args.get("rid"), args.get("text"), args.get("desc")
    if rid is None and text is None and desc is None:
        return None
    sel: dict[str, Any] = {"rid": rid, "text": text, "desc": desc}
    if args.get("index") is not None:
        sel["index"] = int(args["index"])
    if args.get("first"):
        sel["first"] = True
    return sel


def _dump(result: Any) -> Any:
    return result.model_dump(mode="json") if hasattr(result, "model_dump") else result


_WITH_IMAGE_PROP: dict[str, Any] = {
    "type": "boolean",
    "description": "Also attach a post-action screenshot (overrides configure default).",
}
_OBSERVE_PROP: dict[str, Any] = {
    "type": "boolean",
    "default": True,
    "description": "Also return the post-action screen analysis.",
}
_SELECTOR_PROPS: dict[str, Any] = {
    "rid": {"type": "string", "description": "Match by resource-id."},
    "text": {"type": "string", "description": "Match by visible text."},
    "desc": {"type": "string", "description": "Match by content-desc."},
    "index": {"type": "integer", "description": "0-based index when the selector is ambiguous."},
    "first": {"type": "boolean", "default": False, "description": "Take the first match."},
}


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
                    "with_image": {
                        "type": "boolean",
                        "default": False,
                        "description": "Also return the raw screenshot as an image block.",
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
                    "by": {
                        "type": "string",
                        "enum": ["text", "id", "desc"],
                        "default": "text",
                        "description": "Match by text, resource-id (finds pruned containers), or desc.",
                    },
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
                "properties": {
                    "id": {"type": "integer"},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
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
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
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
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="key",
            description="Press a hardware/navigation key (back|home|enter|recents|KEYCODE_*).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
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
            name="wait_changed",
            description="Block until the accessibility hierarchy fingerprint changes "
            "(any UI tree change). Host-polled stand-in for a11y event push.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout_ms": {"type": "integer", "default": 15000},
                    "interval_ms": {"type": "integer", "default": 150},
                    "observe": _OBSERVE_PROP,
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
            name="long_press",
            description="Long-press the element with the given id (context menus).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "ms": {"type": "integer", "default": 600},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="scroll_to",
            description="Scroll until the given text is visible; returns whether it was found.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "match": {"type": "string", "enum": match_enum, "default": "contains"},
                    "ignore_case": {"type": "boolean", "default": False},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="wait_stable",
            description="Wait until the screen stops visually changing (perceptual hash; "
            "works on opaque/Compose/video screens) — for loading / image generation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "interval": {"type": "integer", "default": 200},
                    "settle": {"type": "integer", "default": 600},
                    "timeout": {"type": "integer", "default": 30000},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="goto",
            description="Drive to a remembered screen via the learned app map: replays "
            "the recorded steps of each route edge (including cross-app auth legs) and "
            "verifies each hop; hands back remaining steps on divergence. plan=true "
            "previews without acting. assist=true lets the opt-in planner recover a "
            "divergence (needs planner.enabled).",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Screen name or fuzzy goal, e.g. 'settings'.",
                    },
                    "plan": {"type": "boolean", "default": False},
                    "max_steps": {"type": "integer", "default": 8},
                    "allow_destructive": {"type": "boolean", "default": False},
                    "assist": {"type": "boolean", "default": False},
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="flow_run",
            description="Replay a saved flow (whole journey — launch, taps, waits, "
            "asserts, cross-app auth) in one call; on divergence returns the failing "
            "step index + remaining steps, resumable via from_step. assist=true lets the "
            "opt-in planner clear a blocker and resume (needs planner.enabled).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "${NAME} placeholder values.",
                        "additionalProperties": {"type": "string"},
                    },
                    "dry_run": {"type": "boolean", "default": False},
                    "from_step": {"type": "integer", "default": 0},
                    "allow_destructive": {"type": "boolean", "default": True},
                    "assist": {"type": "boolean", "default": False},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="navigate",
            description="Drive to a natural-language goal with the opt-in planner LLM "
            "(needs planner.enabled), recording the path so a later goto replays it for "
            "free. until=<text> stops deterministically; save_flow saves the path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "e.g. 'open the image generator'."},
                    "until": {"type": "string"},
                    "max_steps": {"type": "integer", "default": 12},
                    "allow_destructive": {"type": "boolean", "default": False},
                    "save_flow": {"type": "string"},
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="list_devices",
            description="List attached devices (serial, model, android version, state).",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="double_tap",
            description="Double-tap the element with the given id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="clear",
            description="Focus an element and clear its text field.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="scroll",
            description="Scroll a direction (up|down|left|right); optional percent of the "
            "scrollable container.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "percent": {"type": "integer", "default": 70},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["direction"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="expect",
            description="Assert something about the screen (exists/absent/text/state). "
            "ok=false means the assertion failed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rid": {"type": "string"},
                    "text": {"type": "string"},
                    "desc": {"type": "string"},
                    "exists": {"type": "boolean", "default": False},
                    "absent": {"type": "boolean", "default": False},
                    "text_is": {"type": "string"},
                    "text_contains": {"type": "string"},
                    "checked": {"type": "boolean"},
                    "enabled": {"type": "boolean"},
                    "selected": {"type": "boolean"},
                    "focused": {"type": "boolean"},
                    "index": {"type": "integer"},
                    "first": {"type": "boolean", "default": False},
                    "timeout_ms": {"type": "integer", "default": 0},
                    "poll_ms": {"type": "integer", "default": 250},
                    "observe": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="hide_keyboard",
            description="Dismiss the soft keyboard (prefer over key back when the IME is up).",
            inputSchema={
                "type": "object",
                "properties": {
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="open_link",
            description="Open a deeplink URI. Pass package/prefer to skip the system "
            "'Open with…' chooser.",
            inputSchema={
                "type": "object",
                "properties": {
                    "uri": {"type": "string"},
                    "package": {
                        "type": "string",
                        "description": "Target package — pins the VIEW intent.",
                    },
                    "prefer": {
                        "type": "string",
                        "description": "If a chooser appears, auto-pick this package/label.",
                    },
                    "pin_package": {
                        "type": "boolean",
                        "default": True,
                        "description": "Pin to foreground/package (false = allow chooser).",
                    },
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["uri"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="clipboard_set",
            description="Set the device clipboard to the given text.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="clipboard_get",
            description="Read the device clipboard.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="paste",
            description="Paste the clipboard into the focused field.",
            inputSchema={
                "type": "object",
                "properties": {
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="copy_text",
            description="Copy an element's text/content-desc to the clipboard "
            "(by id or rid/text/desc selector).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    **_SELECTOR_PROPS,
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="erase",
            description="Erase text in a field: focus (optional id) then delete chars "
            "or clear all.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "chars": {
                        "type": "integer",
                        "description": "Delete this many characters; omit to clear all.",
                    },
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                    **_SELECTOR_PROPS,
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="location_set",
            description="Set GPS mock location (lat, lon).",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["lat", "lon"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="orientation_set",
            description="Set screen orientation (portrait|landscape|…).",
            inputSchema={
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "required": ["mode"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="orientation_get",
            description="Get the current screen orientation.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="airplane_set",
            description="Enable or disable airplane mode.",
            inputSchema={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="airplane_toggle",
            description="Toggle airplane mode.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="media_add",
            description="Push a local media file onto the device (DCIM/Camera by default).",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="record_start",
            description="Start screen recording on the device.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Remote path on device (default /sdcard/aua_recording.mp4).",
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="record_stop",
            description="Stop screen recording and pull the MP4 to a local path.",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="capture_status",
            description="Status of the rolling screencap buffer (daemon-warm).",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="capture_last",
            description="Return recent capture frames + cheap local diff summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "seconds": {"type": "number"},
                    "since": {
                        "type": "string",
                        "description": "Use 'last-action' for frames since the last tap/input.",
                    },
                    "region": {
                        "type": "string",
                        "description": "Filter diff to a 3×3 cell (center, upper, left, …).",
                    },
                    "where_rid": {
                        "type": "string",
                        "description": "Infer region from a resource-id's last-known center.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="capture_export",
            description="Export recent capture frames to a GIF (or MP4 with imageio).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "seconds": {"type": "number"},
                    "since": {"type": "string"},
                    "format": {"type": "string", "description": "gif|mp4"},
                    "fps": {"type": "number"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="capture_explain",
            description="Narrate recent capture (local summary; optional LLM).",
            inputSchema={
                "type": "object",
                "properties": {
                    "seconds": {"type": "number"},
                    "since": {"type": "string"},
                    "llm": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="clock_set",
            description="Set the device clock to a Unix timestamp in milliseconds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ms": {"type": "integer", "description": "Unix timestamp in milliseconds."},
                },
                "required": ["ms"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="dev_profile",
            description="Apply developer-option profile: ac (anim off + crashes on) or default.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string", "enum": ["ac", "default"]}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="a11y_scroll",
            description="Accessibility scroll on an element (forward or backward).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "rid": {"type": "string"},
                    "text": {"type": "string"},
                    "direction": {"type": "string", "enum": ["forward", "backward"]},
                    "observe": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="flags_apply",
            description=(
                "Apply feature flags from a YAML file via the package's configured deeplink "
                "template, verify them against the app's shared_prefs, and restart the app "
                "(flags read at cold start need the restart)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "package": {"type": "string"},
                    "restart": {"type": "boolean", "default": True},
                    "verify": {"type": "boolean", "default": True},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="map_audit",
            description="Audit a learned map for poor names, duplicate variants, stale "
            "screens, route conflicts, and source/runtime research questions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="reconcile_plan",
            description="Create canonical research tasks for an external agent. AUA does "
            "not spawn the agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="reconcile_submit",
            description="Submit a structured external-agent report. verdict=apply is "
            "validated, committed atomically, and given a rollback id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "report": {"type": "object", "additionalProperties": True},
                },
                "required": ["package", "report"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="reconcile_status",
            description="List research tasks, queued reports, and correction events.",
            inputSchema={
                "type": "object",
                "properties": {"package": {"type": "string"}},
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="reconcile_apply",
            description="Apply a queued review report by task id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "task_id": {"type": "string"},
                },
                "required": ["package", "task_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="reconcile_rollback",
            description="Restore the transaction snapshot for a correction event.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "rollback_id": {"type": "string"},
                },
                "required": ["package", "rollback_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="knowledge_list",
            description="List provenance-bearing app knowledge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "status": {"type": "string"},
                },
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="knowledge_add",
            description="Save feedback or source/runtime research for future agents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "kind": {"type": "string"},
                    "text": {"type": "string"},
                    "name": {"type": "string"},
                    "context": {"type": "string"},
                    "source": {"type": "string"},
                    "agent": {"type": "string"},
                    "session": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["package", "text"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="knowledge_stale",
            description="Mark one knowledge item stale while retaining its evidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "id": {"type": "string"},
                },
                "required": ["package", "id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="proxy_start",
            description="Start headless mitmproxy + device http_proxy (needs [proxy] extra).",
            inputSchema={
                "type": "object",
                "properties": {"port": {"type": "integer", "default": 8080}},
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="proxy_stop",
            description="Stop mitmproxy and clear the device HTTP proxy.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="mock_replay",
            description="Load a YAML cassette as live HTTP mock rules.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="app",
            description="Inspect or control the foreground app "
            "(foreground|launch|stop|kill|clear|grant|current).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "package": {"type": "string"},
                    "activity": {"type": "string"},
                    "clear_state": {"type": "boolean", "default": False},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="resolve",
            description="Remap a previous-frame id or stable_key onto the current screen.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "description": "Previous-frame integer id or stable_key string.",
                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                    },
                },
                "required": ["target"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="configure",
            description="Set session defaults for subsequent action tools "
            "(e.g. with_image on every observation).",
            inputSchema={
                "type": "object",
                "properties": {
                    "with_image": {
                        "type": "boolean",
                        "description": "Default with_image for action tools that observe.",
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]


# --------------------------------------------------------------------------- dispatch


def _dispatch(engine: Engine, name: str, args: dict[str, Any]) -> Any:
    """Call the engine method for ``name`` and return a JSON-serialisable payload."""
    img = _with_image(engine, args)

    if name == "analyze_screen":
        result = engine.analyze(
            source=args.get("source", "auto"),
            with_ocr=args.get("with_ocr"),
            query=args.get("query"),
            with_image=args.get("with_image", False),
        )
        return _dump(result)
    if name == "has":
        return _dump(
            engine.has(
                args["text"],
                match=args.get("match", "contains"),
                ignore_case=args.get("ignore_case", False),
                ocr_fallback=args.get("ocr_fallback", True),
                by=args.get("by", "text"),
            )
        )
    if name == "tap":
        return _dump(
            engine.tap(
                int(args["id"]),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "input":
        return _dump(
            engine.input_text(
                int(args["id"]),
                args["text"],
                submit=args.get("submit", False),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "swipe":
        coords = args.get("coords")
        coord_tuple: tuple[int, int, int, int] | None = None
        if coords:
            x1, y1, x2, y2 = (int(c) for c in coords)
            coord_tuple = (x1, y1, x2, y2)
        return _dump(
            engine.swipe(
                direction=args.get("direction"),
                coords=coord_tuple,
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "key":
        return _dump(
            engine.key(
                args["name"],
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "wait":
        return _dump(
            engine.wait(
                for_=args.get("for_"),
                idle=args.get("idle", False),
                timeout_ms=args.get("timeout", 5000),
            )
        )
    if name == "wait_changed":
        return _dump(
            engine.wait_changed(
                timeout_ms=int(args.get("timeout_ms", 15000)),
                interval_ms=args.get("interval_ms"),
                observe=args.get("observe", False),
            )
        )
    if name == "screenshot":
        return _dump(engine.screenshot(args.get("path"), annotate=args.get("annotate", False)))
    if name == "inspect":
        return _dump(engine.inspect(int(args["id"])))
    if name == "long_press":
        return _dump(
            engine.long_press(
                int(args["id"]),
                ms=int(args.get("ms", 600)),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "scroll_to":
        return _dump(
            engine.scroll_to(
                args["text"],
                match=args.get("match", "contains"),
                ignore_case=args.get("ignore_case", False),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "wait_stable":
        return _dump(
            engine.wait_stable(
                interval_ms=int(args.get("interval", 200)),
                settle_ms=int(args.get("settle", 600)),
                timeout_ms=int(args.get("timeout", 30000)),
            )
        )
    if name == "goto":
        # Plain dict (route/hops/handoff) — same payload the CLI/daemon emit.
        return engine.goto(
            args["goal"],
            plan=args.get("plan", False),
            max_steps=int(args.get("max_steps", 8)),
            allow_destructive=args.get("allow_destructive", False),
            assist=args.get("assist", False),
        )
    if name == "flow_run":
        return engine.flow_run(
            args["name"],
            params={str(k): str(v) for k, v in (args.get("params") or {}).items()},
            dry_run=args.get("dry_run", False),
            from_step=int(args.get("from_step", 0)),
            allow_destructive=args.get("allow_destructive", True),
            assist=args.get("assist", False),
        )
    if name == "navigate":
        return engine.navigate(
            args["goal"],
            until=args.get("until"),
            max_steps=int(args.get("max_steps", 12)),
            allow_destructive=args.get("allow_destructive", False),
            save_flow=args.get("save_flow"),
        )
    if name == "list_devices":
        return [d.model_dump(mode="json") for d in engine.list_devices()]
    if name == "double_tap":
        return _dump(
            engine.double_tap(
                int(args["id"]),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "clear":
        return _dump(
            engine.clear(
                int(args["id"]),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "scroll":
        return _dump(
            engine.scroll(
                args.get("direction"),
                percent=int(args.get("percent", 70)),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "expect":
        return _dump(
            engine.expect(
                rid=args.get("rid"),
                text=args.get("text"),
                desc=args.get("desc"),
                exists=args.get("exists", False),
                absent=args.get("absent", False),
                text_is=args.get("text_is"),
                text_contains=args.get("text_contains"),
                checked=args.get("checked"),
                enabled=args.get("enabled"),
                selected=args.get("selected"),
                focused=args.get("focused"),
                index=args.get("index"),
                first=args.get("first", False),
                timeout_ms=int(args.get("timeout_ms", args.get("timeout", 0))),
                poll_ms=int(args.get("poll_ms", 250)),
                observe=args.get("observe", False),
            )
        )
    if name == "hide_keyboard":
        return _dump(
            engine.hide_keyboard(
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "open_link":
        return _dump(
            engine.open_link(
                args["uri"],
                package=args.get("package"),
                prefer=args.get("prefer"),
                pin_package=args.get("pin_package", True),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "clipboard_set":
        return _dump(engine.clipboard_set(args["text"]))
    if name == "clipboard_get":
        return _dump(engine.clipboard_get())
    if name == "paste":
        return _dump(
            engine.paste(
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "copy_text":
        selector = _selector_from_args(args)
        element_id = int(args["id"]) if args.get("id") is not None else None
        return _dump(engine.copy_text(element_id, selector=selector))
    if name == "erase":
        selector = _selector_from_args(args)
        element_id = int(args["id"]) if args.get("id") is not None else None
        return _dump(
            engine.erase(
                element_id,
                selector=selector,
                chars=args.get("chars"),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "location_set":
        return _dump(engine.location_set(float(args["lat"]), float(args["lon"])))
    if name == "orientation_set":
        return _dump(engine.orientation_set(args["mode"]))
    if name == "orientation_get":
        return _dump(engine.orientation_get())
    if name == "airplane_set":
        return _dump(engine.airplane_set(bool(args["enabled"])))
    if name == "airplane_toggle":
        return _dump(engine.airplane_toggle())
    if name == "media_add":
        return _dump(engine.media_add(args["path"]))
    if name == "record_start":
        return _dump(engine.record_start(args.get("path")))
    if name == "record_stop":
        return _dump(engine.record_stop(args["path"]))
    if name == "clock_set":
        ms = args.get("ms", args.get("timestamp_ms"))
        return _dump(engine.clock_set(timestamp_ms=int(ms) if ms is not None else None))
    if name == "capture_status":
        return _dump(engine.capture_status())
    if name == "capture_last":
        return _dump(
            engine.capture_last(
                seconds=args.get("seconds"),
                since=args.get("since"),
                region=args.get("region"),
                where_rid=args.get("where_rid"),
            )
        )
    if name == "capture_export":
        return _dump(
            engine.capture_export(
                args["path"],
                seconds=args.get("seconds"),
                since=args.get("since"),
                fmt=args.get("format", "gif"),
                fps=float(args.get("fps") or 8.0),
            )
        )
    if name == "capture_explain":
        return _dump(
            engine.capture_explain(
                seconds=args.get("seconds"),
                since=args.get("since"),
                llm=bool(args.get("llm", False)),
            )
        )
    if name == "dev_profile":
        return _dump(engine.dev_profile(args["name"]))
    if name == "a11y_scroll":
        sel = _selector_from_args(args)
        eid = int(args["id"]) if args.get("id") is not None else None
        return _dump(
            engine.a11y_scroll(
                eid,
                selector=sel,
                direction=args.get("direction", "forward"),
                observe=args.get("observe", True),
            )
        )
    if name == "flags_apply":
        return _dump(
            engine.flags_apply(
                args["path"],
                package=args.get("package"),
                observe=True,
                restart=args.get("restart", True),
                verify=args.get("verify", True),
            )
        )
    if name in {
        "map_audit",
        "reconcile_plan",
        "reconcile_submit",
        "reconcile_status",
        "reconcile_apply",
        "reconcile_rollback",
        "knowledge_list",
        "knowledge_add",
        "knowledge_stale",
    }:
        from .memory import AppMap, KnowledgeEvidence
        from .reconcile import ReconciliationStore, ResearchReport, audit_map

        store = engine._memory
        if store is None:
            raise AuaError("memory is disabled", code="usage")
        package = str(args["package"])
        app_map = store.load(package) or AppMap(package=package)
        reconciliation = ReconciliationStore(store)
        try:
            if name == "map_audit":
                context = (
                    args.get("context")
                    or store.load_session(engine.device.serial).active_context_id
                )
                return audit_map(app_map, context_id=context).model_dump(mode="json")
            if name == "reconcile_plan":
                context = (
                    args.get("context")
                    or store.load_session(engine.device.serial).active_context_id
                )
                tasks = reconciliation.plan(package, context_id=context)
                return {
                    "package": package,
                    "tasks": [task.model_dump(mode="json") for task in tasks],
                }
            if name == "reconcile_submit":
                report = ResearchReport.model_validate(args["report"])
                return reconciliation.submit(package, report)
            if name == "reconcile_status":
                return reconciliation.status(package)
            if name == "reconcile_apply":
                raw = next(
                    (
                        item
                        for item in app_map.pending_reports
                        if item.get("task_id") == args["task_id"]
                    ),
                    None,
                )
                if raw is None:
                    raise ValueError(f"no queued report for task: {args['task_id']}")
                report = ResearchReport.model_validate({**raw, "verdict": "apply"})
                return reconciliation.apply(package, report).model_dump(mode="json")
            if name == "reconcile_rollback":
                return reconciliation.rollback(package, str(args["rollback_id"])).model_dump(
                    mode="json"
                )
            if name == "knowledge_list":
                status = args.get("status")
                return {
                    "package": package,
                    "knowledge": [
                        item.model_dump(mode="json")
                        for item in app_map.knowledge
                        if status is None or item.status == status
                    ],
                }
            if name == "knowledge_add":
                item = store.remember_knowledge(
                    package,
                    kind=args.get("kind", "claim"),
                    text=args["text"],
                    name=args.get("name"),
                    context_id=args.get("context"),
                    source=args.get("source", "agent"),
                    agent=args.get("agent"),
                    session=args.get("session"),
                    evidence=[
                        KnowledgeEvidence(kind="agent", ref=ref) for ref in args.get("evidence", [])
                    ],
                )
                return item.model_dump(mode="json") if item else {}
            if name == "knowledge_stale":
                item = next(
                    (known for known in app_map.knowledge if known.id == args["id"]),
                    None,
                )
                if item is None:
                    raise ValueError(f"unknown knowledge item: {args['id']}")
                item.status = "stale"
                store.save(app_map)
                return {"ok": True, "id": args["id"], "status": "stale"}
        except (ValueError, OSError) as exc:
            raise AuaError(str(exc), code="usage") from exc
    if name == "proxy_start":
        return _dump(engine.proxy_start(port=args.get("port")))
    if name == "proxy_stop":
        return _dump(engine.proxy_stop())
    if name == "mock_replay":
        return _dump(engine.mock_replay(args["name"]))
    if name == "app":
        return _dump(
            engine.app(
                args["action"],
                package=args.get("package"),
                activity=args.get("activity"),
                clear_state=args.get("clear_state", False),
            )
        )
    if name == "resolve":
        # Engine.resolve may land soon; call through getattr so MCP stays ahead of the method.
        result = getattr(engine, "resolve")(args["target"])  # noqa: B009
        return _dump(result)
    if name == "configure":
        if "with_image" in args:
            engine._default_with_image = args["with_image"]  # type: ignore[attr-defined]
        return {
            "ok": True,
            "with_image": getattr(engine, "_default_with_image", None),
        }
    raise AuaError(f"unknown tool '{name}'", code="usage")


def _image_block(name: str, payload: Any) -> types.ImageContent | None:
    """Return the produced screenshot as an inline image block, when there is one.

    Vision-capable MCP clients get the actual pixels alongside the JSON; text-only
    clients simply ignore the extra block. The path stays in the JSON either way.
    """
    if not isinstance(payload, dict):
        return None
    path: str | None = None
    if name == "screenshot" and payload.get("ok"):
        path = payload.get("detail")
    else:
        meta = payload.get("meta") or (payload.get("observation") or {}).get("meta") or {}
        path = meta.get("raw_image") or meta.get("annotated_image")
    if not path:
        return None
    try:
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return None
    return types.ImageContent(type="image", data=data, mimeType="image/png")


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
        blocks: list[types.ContentBlock] = [types.TextContent(type="text", text=text)]
        image = _image_block(name, payload)
        if image is not None:
            blocks.append(image)
        return blocks

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
