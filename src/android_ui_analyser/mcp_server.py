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
import contextlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from . import __version__
from .capabilities import capabilities_for_goal, capability_manifest, render_mcp_instructions
from .config import load_config
from .engine import Engine, _parse_await_terms, _regex_literal_hint, _safe_adopted_change
from .errors import AuaError, UsageError
from .projection import Projection, trim_observation_payload
from .schema import OutputFormat

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


def _engine_method(engine: Engine, name: str) -> Any:
    """Late-bind a new parity method while CLI/engine and MCP land concurrently."""
    return getattr(engine, name)


_WITH_IMAGE_PROP: dict[str, Any] = {
    "type": "boolean",
    "description": "Also attach a post-action screenshot (overrides configure default).",
}
_OBSERVE_FIELDS_PROP: dict[str, Any] = {
    "type": "string",
    "description": (
        "Columns to keep in the returned `observation` ('all' = full dump). Defaults to a "
        "compact view. This is the cost dial the renamed tools kept instead of `observe`: the "
        "screen always comes back — that is the point of the name — but you choose how much of "
        "it, which is what the CLI's --observe-fields already offered."
    ),
}
_OBSERVE_PROP: dict[str, Any] = {
    "type": "boolean",
    "default": True,
    "description": "Also return the post-action screen analysis.",
}
_UNTIL_PROPS: dict[str, dict[str, Any]] = {
    "until": {
        "type": "string",
        "description": (
            "After the action, wait for comma-separated semantic terms such as "
            "'rid:result,!text:Loading', then return that settled screen in this same call. "
            "At least one term must be positive arrival evidence."
        ),
    },
    "until_timeout": {
        "type": "integer",
        "default": 30000,
        "minimum": 0,
        "description": "Maximum milliseconds to wait for until.",
    },
    "until_poll": {
        "type": "integer",
        "default": 500,
        "minimum": 10,
        "description": "Milliseconds between until predicate checks.",
    },
}
_DATABASE_PARAMS_PROP: dict[str, Any] = {
    "oneOf": [
        {"type": "array"},
        {"type": "object", "additionalProperties": True},
    ],
    "description": "SQLite bind parameters as a JSON array or object.",
}
_SELECTOR_PROPS: dict[str, Any] = {
    "rid": {"type": "string", "description": "Match by resource-id."},
    "text": {"type": "string", "description": "Match by visible text."},
    "desc": {"type": "string", "description": "Match by content-desc."},
    "index": {"type": "integer", "description": "0-based index when the selector is ambiguous."},
    "first": {"type": "boolean", "default": False, "description": "Take the first match."},
}

# These are the public agent-facing names. The shorter verbs used to advertise an optional
# ``observe`` switch even though their default response already analyzed the resulting screen.
# Models repeatedly ignored that response and called ``analyze_screen`` anyway. Make the
# behavior impossible to miss at tool-selection time, and keep the terse names internal only.
_ANALYZED_TOOL_NAMES: dict[str, str] = {
    "tap": "tap_and_analyze",
    "input": "input_and_analyze",
    "swipe": "swipe_and_analyze",
    "key": "key_and_analyze",
    "wait": "wait_and_analyze",
    "wait_changed": "wait_changed_and_analyze",
    "long_press": "long_press_and_analyze",
    "scroll_to": "scroll_to_and_analyze",
    "wait_stable": "wait_stable_and_analyze",
    "double_tap": "double_tap_and_analyze",
    "clear": "clear_and_analyze",
    "scroll": "scroll_and_analyze",
    "expect": "expect_and_analyze",
    "hide_keyboard": "hide_keyboard_and_analyze",
    "open_link": "open_link_and_analyze",
    "paste": "paste_and_analyze",
    "erase": "erase_and_analyze",
    "a11y_scroll": "a11y_scroll_and_analyze",
    "flags_apply": "flags_apply_and_analyze",
}
_ANALYZED_TOOL_BASES = {public: base for base, public in _ANALYZED_TOOL_NAMES.items()}
_POST_ACTION_WAIT_TOOLS = frozenset({*_ANALYZED_TOOL_BASES, "app_launch_and_analyze"})
_OBSERVATION_TOOL_NAMES = frozenset(
    {*_POST_ACTION_WAIT_TOOLS, "await_and_analyze", "back_until_and_analyze", "session_start"}
)
_PHASE_DONE_PROP = {
    "type": "object",
    "description": (
        "Optional checkpoint from the previous result. It advances the current goal phase "
        "without an extra tool call before this tool runs."
    ),
    "properties": {
        "id": {"type": "string"},
        "evidence": {"type": "string", "minLength": 1},
    },
    "required": ["id", "evidence"],
    "additionalProperties": False,
}


def _validate_until(name: str, args: dict[str, Any]) -> None:
    """Validate a folded action wait before the action can mutate device state."""
    predicate = args.get("until")
    dangling = [key for key in ("until_timeout", "until_poll") if key in args]
    if predicate:
        _parse_await_terms(str(predicate), require_positive=True)
        return
    if name in _POST_ACTION_WAIT_TOOLS and dangling:
        raise UsageError(
            f"{' and '.join(dangling)} only bounds `until`, and no until predicate was given",
            hint="Name the semantic arrival evidence in `until`, for example "
            "'rid:result,!text:Loading'.",
        )


def _fold_action_until(
    engine: Engine,
    name: str,
    args: dict[str, Any],
    payload: Any,
) -> Any:
    """Adopt predicate evidence as an action's final observation, like CLI ``--until``."""
    predicate = args.get("until")
    if name not in _POST_ACTION_WAIT_TOOLS or not predicate or not isinstance(payload, dict):
        return payload
    if not payload.get("ok") or payload.get("observation_present") is None:
        return payload
    awaited = _dump(
        engine.await_predicate(
            str(predicate),
            timeout_ms=int(args.get("until_timeout", 30000)),
            poll_ms=int(args.get("until_poll", 500)),
            observe=True,
            adopt_action=True,
        )
    )
    if not isinstance(awaited, dict):
        return payload
    previous_change = payload.get("change")
    for key in (
        "await_outcome",
        "await_terms",
        "elapsed_ms",
        "observation",
        "observation_present",
        "known_screen",
        "stable_elements",
        "action_diff_summary",
        "change",
        "next_actions",
        "routes",
        "note",
    ):
        value = awaited.get(key)
        if key == "change":
            value = _safe_adopted_change(previous_change, value)
        payload[key] = value
    if awaited.get("await_outcome") == "timeout":
        payload["note"] = (
            f"the action landed; `until` timed out after "
            f"{int(args.get('until_timeout', 30000))}ms. Match the exact screen label or "
            "resource-id; this is a predicate timeout, not proof that the action failed."
        )
        regex_hint = _regex_literal_hint(str(predicate))
        if regex_hint:
            payload["note"] += " " + regex_hint
    elif awaited.get("await_outcome") == "satisfied":
        payload["stale_risk"] = None
    return payload


def _as_analyzed_tool(tool: types.Tool) -> types.Tool:
    """Rename an observed tool and remove the now-contradictory ``observe`` input."""
    public_name = _ANALYZED_TOOL_NAMES.get(tool.name)
    if public_name is None:
        return tool
    schema = dict(tool.inputSchema)
    properties = dict(schema.get("properties") or {})
    properties.pop("observe", None)
    # `observe` contradicted the name; a *width* control does not. Without it MCP had no cost
    # control at all and returned the whole tree on every action, while the CLI trimmed by default.
    properties["observe_fields"] = _OBSERVE_FIELDS_PROP
    properties.update(_UNTIL_PROPS)
    schema["properties"] = properties
    return types.Tool(
        name=public_name,
        description=(
            f"{tool.description or 'Perform the action.'} Returns the analyzed resulting screen "
            "in `observation`; use its fresh ids directly and do not call analyze_screen next."
        ),
        inputSchema=schema,
    )


def _with_phase_checkpoint(tool: types.Tool) -> types.Tool:
    """Give every MCP operation the same no-extra-call phase checkpoint as the CLI."""
    if tool.name == "session_start":
        return tool
    schema = dict(tool.inputSchema)
    properties = dict(schema.get("properties") or {})
    properties["phase_done"] = _PHASE_DONE_PROP
    properties["expect_error"] = {
        "type": "string",
        "description": (
            "Exact machine-readable error code intentionally probed by this invocation; "
            "session review treats it as expected only when the returned code matches."
        ),
    }
    schema["properties"] = properties
    return types.Tool(name=tool.name, description=tool.description, inputSchema=schema)


# --------------------------------------------------------------------------- tool specs


def _tool_definitions() -> list[types.Tool]:
    """The MCP tool catalogue (input schemas only; output is JSON text content)."""
    match_enum = ["exact", "contains", "regex"]
    source_enum = ["auto", "hierarchy", "vision"]
    tools = [
        types.Tool(
            name="capabilities",
            description=(
                "Discover AUA capabilities, ranked for an optional goal. Start a new kind of "
                "task here only when session_start did not already surface what you need."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Optional natural-language task used to rank the catalogue.",
                    }
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="session_start",
            description=(
                "Start goal-aware Android work: observe once, surface relevant capabilities, "
                "and return the safest exact recommended call. Use this first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The end-to-end test goal."},
                    "start_emulator": {
                        "type": "boolean",
                        "default": False,
                        "description": "Explicitly permit booting an AVD when no device is attached.",
                    },
                    "headed": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Require a visible emulator; if AUA starts one, show its window."
                        ),
                    },
                    "avd": {"type": "string", "description": "AVD name when several exist."},
                    "package": {
                        "type": "string",
                        "description": "Launch and observe this package before planning.",
                    },
                    "activity": {
                        "type": "string",
                        "description": "Optional launcher Activity to pin with package.",
                    },
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="session_review",
            description=(
                "Review this AUA session's top-level calls, elapsed time, failures, redundant "
                "patterns, recommendation use, and estimated savings."
            ),
            inputSchema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="session_progress",
            description=(
                "Return ordered goal phases, the active checkpoint, and one exact next call. "
                "Analyzed tool results include this automatically; call explicitly only after "
                "reconnecting."
            ),
            inputSchema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="session_finish",
            description=(
                "Finish the session, restore session-owned reversible state, stop only emulators "
                "started by it, and return the final efficiency review."
            ),
            inputSchema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="job_start",
            description=(
                "Start one durable read-only wait and return immediately with a reconnectable "
                "job id. Ordinary device tools are serialized until it finishes or is cancelled."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["await", "wait-stable", "wait-changed", "wait-after-change"],
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Required for await, e.g. 'rid:result,!text:Loading'.",
                    },
                    "timeout_ms": {"type": "integer", "minimum": 1, "default": 60000},
                    "poll_ms": {"type": "integer", "minimum": 10, "default": 120},
                    "settle_ms": {"type": "integer", "minimum": 1, "default": 1200},
                    "confirmation_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 1800,
                    },
                    "observe": {"type": "boolean", "default": True},
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="job_status",
            description="Reconnect to a durable wait by id without restarting it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "recent_output": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include lifecycle events and latest result/error.",
                    },
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="job_wait",
            description=(
                "Wait at most 10 seconds for a durable job; a running result keeps the same id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10000,
                        "default": 5000,
                    },
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="job_cancel",
            description="Cancel a durable wait and briefly await terminal acknowledgement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "wait_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10000,
                        "default": 1000,
                    },
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="job_list",
            description="List recent durable waits visible to this owner.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="orient",
            description=(
                "Return the current app's known screens, verified route suggestions, recipes, "
                "deeplinks, notes, and research tasks without navigating."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="reach",
            description=(
                "Reach a semantic destination in one call using the safest verified option: "
                "goto, then a matching safe flow, then an explicitly permitted deeplink or assist."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "until": {
                        "type": "string",
                        "description": "Arrival predicate, for example 'rid:result,!text:Loading'.",
                    },
                    "timeout_ms": {"type": "integer", "default": 30000, "minimum": 0},
                    "poll_ms": {"type": "integer", "default": 300, "minimum": 10},
                    "allow_unsafe": {"type": "boolean", "default": False},
                    "allow_destructive": {"type": "boolean", "default": False},
                    "assist": {
                        "type": "boolean",
                        "default": False,
                        "description": "Permit configured planner assistance after deterministic options.",
                    },
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="await_and_analyze",
            description=(
                "Wait for all comma-separated positive/negative semantic terms and return the "
                "settled analyzed screen plus per-term evidence in one call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "predicate": {
                        "type": "string",
                        "description": "For example 'rid:result,!text:Loading'.",
                    },
                    "timeout_ms": {"type": "integer", "default": 60000, "minimum": 0},
                    "poll_ms": {"type": "integer", "default": 500, "minimum": 10},
                    "match": {
                        "type": "string",
                        "enum": match_enum,
                        "default": "contains",
                    },
                    "ignore_case": {"type": "boolean", "default": False},
                    "observe_fields": _OBSERVE_FIELDS_PROP,
                },
                "required": ["predicate"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="flow_list",
            description=(
                "List saved flows with app/context compatibility, step count, parameters, "
                "arrival proof, description, and path."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="flow_save",
            description=(
                "Preview recent recorded actions as an editable reusable flow without writing. "
                "Set save=true only after reviewing scope, selectors, and arrival proof."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "last": {"type": "integer", "default": 12, "minimum": 1},
                    "save": {
                        "type": "boolean",
                        "default": False,
                        "description": "Write the previewed flow; default false writes nothing.",
                    },
                    "force": {"type": "boolean", "default": False},
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Deprecated non-writing alias; preview is already default.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="flow_delete",
            description=(
                "Idempotently delete a saved flow. Returns deleted=false and "
                "status=already_absent when it was already gone."
            ),
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="map_find",
            description="Find a context-compatible learned route and return its exact goto call without acting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "package": {"type": "string"},
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="suite_run",
            description="Run a YAML acceptance checklist as one grouped assertion call.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "text": {"type": "string", "description": "Inline YAML when path is '-'."},
                    "continue_on_fail": {"type": "boolean", "default": False},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="logcat_mark",
            description="Set a named device-clock logcat mark before the action under test.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "default": "default"},
                    "clear": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="logcat_dump",
            description="Read only logcat lines since a named mark, with optional grep/tag filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "since": {"type": "string"},
                    "grep": {"type": "string"},
                    "tag": {"type": "string"},
                    "lines": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        ),
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
            name="back_until_and_analyze",
            description=(
                "Navigate back in one bounded call until semantic destination evidence is "
                "present. Re-resolves a visible Back/Navigate-up control on every fresh frame, "
                "then falls back to hardware Back. Returns the final analyzed screen and always "
                "stops if the foreground leaves the starting package. A bare mapped "
                "known_screen name is accepted directly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "predicate": {
                        "type": "string",
                        "description": (
                            "Bare mapped known_screen name, or ANDed text:/rid:/desc: terms."
                        ),
                    },
                    "back_id": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Fresh frame-local id for an unlabeled Back control; first step only."
                        ),
                    },
                    "back_selector": {
                        "type": "object",
                        "description": (
                            "Optional app-owned Back selector, re-resolved on every frame."
                        ),
                        "properties": {
                            "rid": {"type": "string"},
                            "text": {"type": "string"},
                            "desc": {"type": "string"},
                        },
                        "minProperties": 1,
                        "maxProperties": 1,
                        "additionalProperties": False,
                    },
                    "max_steps": {
                        "type": "integer",
                        "default": 4,
                        "minimum": 1,
                        "maximum": 12,
                    },
                    "step_timeout_ms": {
                        "type": "integer",
                        "default": 1200,
                        "minimum": 0,
                    },
                    "poll_ms": {"type": "integer", "default": 200, "minimum": 10},
                    "observe_fields": _OBSERVE_FIELDS_PROP,
                },
                "required": ["predicate"],
                "allOf": [{"not": {"required": ["back_id", "back_selector"]}}],
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
                    "observe": _OBSERVE_PROP,
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
            description="Scroll until the given text is visible; returns whether it was found. "
            "It searches ONE way: `direction` up (default) looks further down the list, down "
            "looks back up. A list that opens already scrolled past the target needs `down`.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "match": {"type": "string", "enum": match_enum, "default": "contains"},
                    "ignore_case": {"type": "boolean", "default": False},
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "default": "up",
                    },
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
                    "observe": _OBSERVE_PROP,
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="goto",
            description="Drive to a remembered screen via the learned app map: replays "
            "the recorded steps of each route edge and verifies each hop. Risky deeplink, "
            "external, settings/data/lifecycle, and non-navigation effects are previewed and "
            "refused before execution unless allow_unsafe=true. plan=true previews without "
            "acting. assist=true lets the opt-in planner recover a "
            "divergence (needs planner.enabled). from_here=true resumes mid-edge when "
            "you already navigated part of the way.",
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
                    "allow_unsafe": {
                        "type": "boolean",
                        "default": False,
                        "description": "Permit disclosed non-navigation route effects after reviewing the preview.",
                    },
                    "assist": {"type": "boolean", "default": False},
                    "from_here": {
                        "type": "boolean",
                        "default": False,
                        "description": "Resume mid-edge from the first matching step on "
                        "the current screen.",
                    },
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
            name="emulator_list",
            description="List configured Android Virtual Devices (AVD names, Play Store vs rootable).",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="emulator_status",
            description="SDK/AVD tooling + running emulator serials + what aua started "
            "(owner/port for parallel agents).",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="emulator_start",
            description="Boot an AVD headless (default) for unattended verify. "
            "Use parallel=true when multiple agents share a host (unique port + read-only + owner). "
            "Pin later tools with configure/device serial from the response. "
            "REQUIRED: call emulator_stop when done (or stop_mine) — orphaned AVDs burn CPU. "
            "Idle auto-stop is only a safety net.",
            inputSchema={
                "type": "object",
                "properties": {
                    "avd": {"type": "string", "description": "AVD name (omit if only one exists)."},
                    "headless": {"type": "boolean", "default": True},
                    "parallel": {
                        "type": "boolean",
                        "default": False,
                        "description": "Safe multi-agent boot: free -port, -read-only, owner tag.",
                    },
                    "owner": {
                        "type": "string",
                        "description": "Owner tag for scoped stop (default $AUA_OWNER or auto).",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Even console port 5554–5682 (auto with parallel).",
                    },
                    "read_only": {"type": "boolean"},
                    "gpu": {"type": "string"},
                    "idle_stop": {
                        "type": "integer",
                        "default": 900,
                        "description": "Auto-stop after N seconds idle (0=never). Headless only.",
                    },
                    "wait": {"type": "integer", "default": 120},
                    "animations": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="emulator_stop",
            description="Stop emulator(s) aua started. ALWAYS call before ending a session that "
            "booted an AVD. Prefer serial= for parallel agents; mine=true stops aua-started "
            "(scoped by owner / $AUA_OWNER when set).",
            inputSchema={
                "type": "object",
                "properties": {
                    "serial": {"type": "string", "description": "Kill this emulator serial only."},
                    "avd": {"type": "string"},
                    "owner": {"type": "string", "description": "Stop only this owner's instances."},
                    "mine": {
                        "type": "boolean",
                        "default": False,
                        "description": "Stop aua-started records (filter by owner if set).",
                    },
                    "all": {
                        "type": "boolean",
                        "default": False,
                        "description": "Kill EVERY running emulator (dangerous).",
                    },
                },
                "additionalProperties": False,
            },
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
            name="network_status",
            description=(
                "Read and verify airplane, Wi-Fi, mobile-data, and active default-network state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "verify": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Accepted for CLI/MCP command-family parity; status always reads "
                            "back the current Android state."
                        ),
                    }
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="network_offline",
            description=(
                "Save current network controls, disable transports, and verify the device "
                "has no active default network. Always pair with network_restore."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "verify": {"type": "boolean", "default": True},
                    "timeout_ms": {"type": "integer", "minimum": 0, "default": 10000},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="network_restore",
            description="Restore and verify the network controls saved by network_offline.",
            inputSchema={
                "type": "object",
                "properties": {"timeout_ms": {"type": "integer", "minimum": 0, "default": 15000}},
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="network_profile_list",
            description="List reversible Wi-Fi, cellular, slow, and lossy profiles.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="network_profile_status",
            description="Read the active network profile and verify its current evidence.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="network_profile_apply",
            description=(
                "Apply one reversible profile: wifi-only, cellular-only, slow, or lossy. "
                "Restore the active profile before applying another."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "enum": ["wifi-only", "cellular-only", "slow", "lossy"],
                    },
                    "loss_percent": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 100,
                        "default": 10,
                    },
                    "timeout_ms": {"type": "integer", "minimum": 0, "default": 15000},
                },
                "required": ["profile"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="network_profile_restore",
            description="Restore and verify conditions saved before the active profile.",
            inputSchema={
                "type": "object",
                "properties": {"timeout_ms": {"type": "integer", "minimum": 0, "default": 20000}},
                "additionalProperties": False,
            },
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
                    "observe": _OBSERVE_PROP,
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="map_audit",
            description="Audit a learned map and persist source/runtime research tasks for "
            "poor names, variants/states, contexts, and untrusted routes.",
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
                "properties": {
                    "port": {
                        "type": "integer",
                        "default": 0,
                        "description": "mitmdump listen port; 0 = free random high port.",
                    }
                },
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
            "(foreground|stop|kill|clear|grant|current). Use app_launch_and_analyze to launch.",
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
            name="app_launch_and_analyze",
            description="Launch an app and return the analyzed resulting screen in `observation`; "
            "use its fresh ids directly and do not call analyze_screen next.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "activity": {"type": "string"},
                    "clear_state": {"type": "boolean", "default": False},
                    "confirmed": {
                        "type": "boolean",
                        "default": False,
                        "description": "Required when clear_state=true because it wipes app data.",
                    },
                    "with_image": _WITH_IMAGE_PROP,
                    **_UNTIL_PROPS,
                },
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_list",
            description="List a debuggable package's private SQLite databases and WAL/SHM sizes.",
            inputSchema={
                "type": "object",
                "properties": {"package": {"type": "string"}},
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_schema",
            description="Return tables/views, columns, indexes, foreign keys, and CREATE SQL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "database": {"type": "string"},
                    "table": {"type": "string"},
                    "restart": {"type": "boolean", "default": True},
                },
                "required": ["package", "database"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_query",
            description=(
                "Run one read-only SQLite statement against a coherent private-database "
                "snapshot and return columns plus JSON rows."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "database": {"type": "string"},
                    "sql": {"type": "string"},
                    "parameters": _DATABASE_PARAMS_PROP,
                    "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000},
                    "timeout_ms": {"type": "integer", "default": 5000, "minimum": 1},
                    "restart": {"type": "boolean", "default": True},
                },
                "required": ["package", "database", "sql"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_execute",
            description=(
                "Execute confirmed INSERT/UPDATE/DELETE/REPLACE/WITH statements. Stops the "
                "app, creates a restore point, validates integrity/foreign keys, replaces the "
                "database without stale sidecars, and relaunches by default."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "database": {"type": "string"},
                    "sql": {"type": "string"},
                    "parameters": _DATABASE_PARAMS_PROP,
                    "timeout_ms": {"type": "integer", "default": 5000, "minimum": 1},
                    "restart": {"type": "boolean", "default": True},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Must be true after reviewing the data mutation.",
                    },
                },
                "required": ["package", "database", "sql", "confirmed"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_backup",
            description="Create a private host restore point with the database and sidecars.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "database": {"type": "string"},
                    "restart": {"type": "boolean", "default": True},
                },
                "required": ["package", "database"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_backups",
            description="List restore points scoped to this device, package, and database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "database": {"type": "string"},
                },
                "required": ["package", "database"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="database_restore",
            description=(
                "Restore a confirmed database backup after preserving the current state as a "
                "new safety backup."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "database": {"type": "string"},
                    "backup_id": {"type": "string"},
                    "restart": {"type": "boolean", "default": True},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Must be true after reviewing the restore point id.",
                    },
                },
                "required": ["package", "database", "backup_id", "confirmed"],
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
    return [_with_phase_checkpoint(_as_analyzed_tool(tool)) for tool in tools]


# --------------------------------------------------------------------------- dispatch


def _dispatch(engine: Engine, name: str, args: dict[str, Any]) -> Any:
    """Call the engine method for ``name`` and return a JSON-serialisable payload."""
    args = dict(args)
    internal_name = _ANALYZED_TOOL_BASES.get(name)
    if internal_name is not None:
        name = internal_name
        args = {**args, "observe": True}
        # Dropped rather than read: no engine method takes it. The caller's copy in `args_in`
        # is what trims the folded observation, at the boundary every tool returns through.
        args.pop("observe_fields", None)
        for wrapper_arg in _UNTIL_PROPS:
            args.pop(wrapper_arg, None)
    elif name in _ANALYZED_TOOL_NAMES:
        raise UsageError(
            f"MCP tool '{name}' was renamed to '{_ANALYZED_TOOL_NAMES[name]}'",
            hint="The renamed tool already returns the analyzed resulting screen; use its "
            "`observation` and do not call `analyze_screen` afterward.",
        )
    from .jobs import manager_for, reject_if_active

    jobs = manager_for(engine)
    if name == "job_start":
        operation = str(args.pop("operation", ""))
        return jobs.start(operation, args)
    if name == "job_status":
        return jobs.status(
            str(args.get("job_id") or ""),
            recent_output=bool(args.get("recent_output", False)),
        )
    if name == "job_wait":
        return jobs.wait(
            str(args.get("job_id") or ""),
            timeout_ms=int(args.get("timeout_ms", 5_000)),
        )
    if name == "job_cancel":
        return jobs.cancel(
            str(args.get("job_id") or ""),
            wait_ms=int(args.get("wait_ms", 1_000)),
        )
    if name == "job_list":
        return jobs.list(limit=int(args.get("limit", 20)))
    if name not in {"capabilities", "session_progress", "session_review", "configure"}:
        reject_if_active(engine, name)

    img = _with_image(engine, args)

    if name == "capabilities":
        goal = args.get("goal")
        return {"capabilities": capabilities_for_goal(str(goal)) if goal else capability_manifest()}
    if name == "session_start":
        started = _dump(
            _engine_method(engine, "session_start")(
                str(args["goal"]),
                start_emulator=bool(args.get("start_emulator", False)),
                headed=bool(args.get("headed", False)),
                avd=args.get("avd"),
                package=args.get("package"),
                activity=args.get("activity"),
            )
        )
        if isinstance(started, dict) and started.get("emulator_started"):
            serial = started.get("serial")
            if isinstance(serial, str) and serial:
                _mcp_started_serials().add(serial)
            owner = started.get("owner")
            if owner:
                _mcp_started_owners().add(str(owner))
        return started
    if name == "session_review":
        return _dump(_engine_method(engine, "session_review")(session_id=args.get("session_id")))
    if name == "session_progress":
        return _dump(_engine_method(engine, "session_progress")(session_id=args.get("session_id")))
    if name == "session_finish":
        finished = _dump(
            _engine_method(engine, "session_finish")(session_id=args.get("session_id"))
        )
        if isinstance(finished, dict):
            for item in finished.get("cleanup") or []:
                if not isinstance(item, dict) or item.get("action") != "owned_emulator_stop":
                    continue
                result = item.get("result")
                if not isinstance(result, dict):
                    continue
                for serial in result.get("stopped") or []:
                    if isinstance(serial, str):
                        _mcp_started_serials().discard(serial)
        return finished
    if name == "orient":
        return _dump(engine.orient())
    if name == "reach":
        return _dump(
            _engine_method(engine, "reach")(
                str(args["goal"]),
                until=args.get("until"),
                timeout_ms=int(args.get("timeout_ms", 30000)),
                interval_ms=int(args.get("poll_ms", 300)),
                allow_unsafe=bool(args.get("allow_unsafe", False)),
                allow_destructive=bool(args.get("allow_destructive", False)),
                assist=bool(args.get("assist", False)),
            )
        )
    if name == "await_and_analyze":
        return _dump(
            engine.await_predicate(
                str(args["predicate"]),
                timeout_ms=int(args.get("timeout_ms", 60000)),
                poll_ms=int(args.get("poll_ms", 500)),
                match=args.get("match", "contains"),
                ignore_case=bool(args.get("ignore_case", False)),
                observe=True,
            )
        )
    if name == "flow_list":
        return engine.flow_list()
    if name == "flow_save":
        return _dump(
            engine.flow_save(
                str(args["name"]),
                last=int(args.get("last", 12)),
                save=bool(args.get("save", False)),
                force=bool(args.get("force", False)),
                dry_run=bool(args.get("dry_run", False)),
            )
        )
    if name == "flow_delete":
        return _dump(engine.flow_delete(str(args["name"])))
    if name == "map_find":
        return _dump(
            engine.map_find(
                str(args["goal"]),
                package=args.get("package"),
            )
        )
    if name == "suite_run":
        return _dump(
            engine.suite_run(
                str(args["path"]),
                text=args.get("text"),
                continue_on_fail=bool(args.get("continue_on_fail", False)),
            )
        )
    if name == "logcat_mark":
        return _dump(
            engine.logcat_mark(
                str(args.get("name", "default")),
                clear=bool(args.get("clear", False)),
            )
        )
    if name == "logcat_dump":
        return _dump(
            engine.logcat(
                since=args.get("since"),
                grep=args.get("grep"),
                tag=args.get("tag"),
                lines=args.get("lines"),
            )
        )
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
                observe=args.get("observe", True),
            )
        )
    if name == "back_until_and_analyze":
        return _dump(
            engine.back_until(
                args["predicate"],
                back_id=args.get("back_id"),
                back_selector=args.get("back_selector"),
                max_steps=int(args.get("max_steps", 4)),
                step_timeout_ms=int(args.get("step_timeout_ms", 1200)),
                poll_ms=int(args.get("poll_ms", 200)),
            )
        )
    if name == "wait_changed":
        return _dump(
            engine.wait_changed(
                timeout_ms=int(args.get("timeout_ms", 15000)),
                interval_ms=args.get("interval_ms"),
                observe=args.get("observe", True),
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
                direction=args.get("direction", "up"),
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
                observe=args.get("observe", True),
            )
        )
    if name == "goto":
        # Plain dict (route/hops/handoff) — same payload the CLI/daemon emit.
        return engine.goto(
            args["goal"],
            plan=args.get("plan", False),
            max_steps=int(args.get("max_steps", 8)),
            allow_destructive=args.get("allow_destructive", False),
            allow_unsafe=args.get("allow_unsafe", False),
            assist=args.get("assist", False),
            from_here=args.get("from_here", False),
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
    if name == "emulator_list":
        from . import emulator as emulator_mod

        return emulator_mod.list_avds()
    if name == "emulator_status":
        from . import emulator as emulator_mod

        return emulator_mod.status(cache_dir=engine.config.cache.dir)
    if name == "emulator_start":
        from . import emulator as emulator_mod

        headless = bool(args.get("headless", True))
        idle = args.get("idle_stop")
        if idle is None:
            idle = 900 if headless else 0
        out = emulator_mod.start(
            args.get("avd"),
            headless=headless,
            animations=bool(args.get("animations", False)),
            wait_s=float(args.get("wait", 120)),
            cache_dir=engine.config.cache.dir,
            gpu=args.get("gpu"),
            idle_timeout_s=float(idle) if headless else 0.0,
            parallel=bool(args.get("parallel", False)),
            port=args.get("port"),
            read_only=args.get("read_only"),
            owner=args.get("owner"),
        )
        # Track for MCP process exit cleanup (agents that forget emulator_stop).
        ser = out.get("serial") if isinstance(out, dict) else None
        if isinstance(ser, str) and ser:
            _mcp_started_serials().add(ser)
            if out.get("owner"):
                # Also remember owner so --mine scoped cleanup works at exit.
                _mcp_started_owners().add(str(out["owner"]))
        return out
    if name == "emulator_stop":
        from . import emulator as emulator_mod

        out = emulator_mod.stop(
            serial=args.get("serial"),
            avd=args.get("avd"),
            owner=args.get("owner"),
            mine=bool(args.get("mine", False)),
            all_devices=bool(args.get("all", False)),
            cache_dir=engine.config.cache.dir,
        )
        stopped = out.get("stopped") if isinstance(out, dict) else None
        if isinstance(stopped, list):
            tracked = _mcp_started_serials()
            for s in stopped:
                tracked.discard(s)
            # A long-lived MCP Engine may otherwise keep a Device and memory cursor bound to
            # the boot that was just stopped. Drop only when its attached serial was stopped;
            # the next tool call reconnects and claims the new boot instance before reading.
            attached = getattr(engine, "_device", None)
            if attached is not None and attached.serial in stopped:
                engine.close()
        return out
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
    if name == "network_status":
        return _dump(engine.network_status())
    if name == "network_offline":
        return _dump(
            engine.network_offline(
                verify=bool(args.get("verify", True)),
                timeout_ms=int(args.get("timeout_ms", 10_000)),
            )
        )
    if name == "network_restore":
        return _dump(engine.network_restore(timeout_ms=int(args.get("timeout_ms", 15_000))))
    if name == "network_profile_list":
        return _dump(engine.network_profile_list())
    if name == "network_profile_status":
        return _dump(engine.network_profile_status())
    if name == "network_profile_apply":
        return _dump(
            engine.network_profile_apply(
                str(args["profile"]),
                loss_percent=float(args.get("loss_percent", 10.0)),
                timeout_ms=int(args.get("timeout_ms", 15_000)),
            )
        )
    if name == "network_profile_restore":
        return _dump(engine.network_profile_restore(timeout_ms=int(args.get("timeout_ms", 20_000))))
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
                observe=args.get("observe", True),
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
                payload = audit_map(app_map, context_id=context).model_dump(mode="json")
                payload["research_tasks"] = [
                    task.model_dump(mode="json")
                    for task in reconciliation.plan(package, context_id=context)
                ]
                return payload
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
        return _dump(engine.proxy_start(port=args.get("port") or None))
    if name == "proxy_stop":
        return _dump(engine.proxy_stop())
    if name == "mock_replay":
        return _dump(engine.mock_replay(args["name"]))
    if name == "app":
        if str(args["action"]).lower() == "launch":
            raise UsageError(
                "MCP app launch is named 'app_launch_and_analyze'",
                hint="That tool returns the launched screen in `observation`; use its fresh ids "
                "without calling `analyze_screen` afterward.",
            )
        return _dump(
            engine.app(
                args["action"],
                package=args.get("package"),
                activity=args.get("activity"),
                clear_state=args.get("clear_state", False),
            )
        )
    if name == "app_launch_and_analyze":
        for wrapper_arg in _UNTIL_PROPS:
            args.pop(wrapper_arg, None)
        return _dump(
            engine.app(
                "launch",
                package=args["package"],
                activity=args.get("activity"),
                clear_state=args.get("clear_state", False),
                confirmed=args.get("confirmed", False),
                observe=True,
                with_image=img,
            )
        )
    if name == "database_list":
        return engine.database_list(args["package"])
    if name == "database_schema":
        return engine.database_schema(
            args["package"],
            args["database"],
            table=args.get("table"),
            restart=args.get("restart", True),
        )
    if name == "database_query":
        return engine.database_query(
            args["package"],
            args["database"],
            args["sql"],
            parameters=args.get("parameters"),
            limit=int(args.get("limit", 100)),
            timeout_ms=int(args.get("timeout_ms", 5000)),
            restart=args.get("restart", True),
        )
    if name == "database_execute":
        return engine.database_execute(
            args["package"],
            args["database"],
            args["sql"],
            parameters=args.get("parameters"),
            timeout_ms=int(args.get("timeout_ms", 5000)),
            restart=args.get("restart", True),
            confirmed=args.get("confirmed", False),
        )
    if name == "database_backup":
        return engine.database_backup(
            args["package"],
            args["database"],
            restart=args.get("restart", True),
        )
    if name == "database_backups":
        return engine.database_backups(args["package"], args["database"])
    if name == "database_restore":
        return engine.database_restore(
            args["package"],
            args["database"],
            args["backup_id"],
            restart=args.get("restart", True),
            confirmed=args.get("confirmed", False),
        )
    if name == "resolve":
        # Engine.resolve may land soon; call through getattr so MCP stays ahead of the method.
        result = getattr(engine, "resolve")(args["target"])  # noqa: B009
        return _dump(result)
    if name == "configure":
        if "with_image" in args:
            engine._default_with_image = args["with_image"]
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


# Emulators started via MCP in this process — stopped on stdio exit if the agent forgot.
_MCP_STARTED_SERIALS: set[str] = set()
_MCP_STARTED_OWNERS: set[str] = set()


def _mcp_started_serials() -> set[str]:
    return _MCP_STARTED_SERIALS


def _mcp_started_owners() -> set[str]:
    return _MCP_STARTED_OWNERS


def cleanup_mcp_emulators(cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Best-effort stop of emulators this MCP process started and forgot to tear down."""
    import contextlib

    from . import emulator as emulator_mod
    from .config import load_config

    cache = cache_dir or load_config().cache.dir
    stopped: list[str] = []
    serials = list(_MCP_STARTED_SERIALS)
    for ser in serials:
        with contextlib.suppress(Exception):
            out = emulator_mod.stop(serial=ser, cache_dir=cache)
            if isinstance(out, dict):
                stopped.extend(str(s) for s in (out.get("stopped") or []))
        _MCP_STARTED_SERIALS.discard(ser)
    # Owner-scoped leftover (parallel) if serial kill missed a record.
    for owner in list(_MCP_STARTED_OWNERS):
        with contextlib.suppress(Exception):
            out = emulator_mod.stop(mine=True, owner=owner, cache_dir=cache)
            if isinstance(out, dict):
                stopped.extend(str(s) for s in (out.get("stopped") or []))
        _MCP_STARTED_OWNERS.discard(owner)
    return {"ok": True, "action": "mcp-emulator-cleanup", "stopped": stopped}


def build_server(engine: Engine) -> Server:
    """Build a low-level MCP :class:`Server` bound to ``engine`` (for stdio + tests)."""
    server: Server = Server(
        SERVER_NAME,
        version=__version__,
        instructions=render_mcp_instructions(),
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        from . import journal as journal_mod

        args_in = dict(arguments or {})
        phase_done = args_in.pop("phase_done", None)
        expected_error_code = args_in.pop("expect_error", None)
        annotation_warnings: list[dict[str, Any]] = []
        invocation_id = uuid.uuid4().hex
        started_at = time.monotonic()
        payload: Any = None

        def journal_call(*, ok: bool, error: dict[str, Any] | None = None) -> None:
            from . import leases

            device = getattr(engine, "_device", None)
            serial = getattr(device, "serial", None) or getattr(
                engine.config.device, "serial", None
            )
            session_id = (
                (payload.get("session_id") if isinstance(payload, dict) else None)
                or args_in.get("session_id")
                or getattr(engine, "_session_id", None)
            )
            extra: dict[str, Any] = {"invocation_id": invocation_id}
            if session_id:
                extra["session_id"] = str(session_id)
            if isinstance(expected_error_code, str) and expected_error_code:
                extra["expected_error_code"] = expected_error_code
                extra["expected_error_matched"] = (
                    isinstance(error, dict) and error.get("code") == expected_error_code
                )
            with contextlib.suppress(Exception):
                journal_mod.record(
                    cache_dir=engine.config.cache.dir,
                    serial=serial,
                    source="mcp",
                    cmd=name,
                    args=args_in,
                    ok=ok,
                    duration_ms=(time.monotonic() - started_at) * 1000.0,
                    result=payload,
                    error=error,
                    extra=extra,
                    owner=leases.resolve_owner(
                        getattr(engine, "_lease_owner_resolved", None)
                        or getattr(engine, "_lease_owner", None)
                    ),
                )

        try:
            if isinstance(phase_done, dict):
                try:
                    _engine_method(engine, "session_mark_phase")(
                        str(phase_done.get("id") or ""),
                        str(phase_done.get("evidence") or ""),
                    )
                except (AuaError, OSError, ValueError) as err:
                    if isinstance(err, AuaError):
                        raw = err.to_dict().get("error")
                        warning = dict(raw) if isinstance(raw, dict) else {"message": str(err)}
                    else:
                        warning = {"code": "annotation_failed", "message": str(err)}
                    warning["annotation"] = "phase_done"
                    annotation_warnings.append(warning)
            _validate_until(name, args_in)
            payload = _dispatch(engine, name, args_in)
            payload = _fold_action_until(engine, name, args_in, payload)
            from .coaching import decorate_result

            payload = decorate_result(
                engine,
                name,
                payload,
                args=args_in,
                current_recorded=False,
            )
            # Trim the folded observation the same way the CLI does. Applied here, at the one
            # boundary every tool returns through, rather than at ~40 `_dump` sites — and via the
            # shared helper, because these two surfaces had already drifted once: MCP was
            # returning every field of every element on every action while the CLI trimmed.
            if isinstance(payload, dict) and name in _OBSERVATION_TOOL_NAMES:
                spec = args_in.get("observe_fields")
                if spec is None:
                    spec = getattr(engine.config.output, "observation_fields", None)
                view = Projection.for_observation(spec, fmt=OutputFormat.json)
                payload = trim_observation_payload(payload, view, fmt=OutputFormat.json)
            if annotation_warnings and isinstance(payload, dict):
                payload["annotation_warnings"] = annotation_warnings
            text = json.dumps(payload, ensure_ascii=False)
        except AuaError as err:
            error = err.to_dict().get("error")
            journal_call(ok=False, error=error if isinstance(error, dict) else None)
            text = json.dumps(err.to_dict(), ensure_ascii=False)
            return [types.TextContent(type="text", text=text)]
        except Exception as err:
            journal_call(ok=False, error={"code": "error", "message": str(err)})
            raise
        journal_call(ok=not (isinstance(payload, dict) and payload.get("ok") is False))
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
    import atexit
    import contextlib

    import anyio
    from mcp.server.stdio import stdio_server

    engine = build_default_engine()
    server = build_server(engine)
    # If the MCP client disconnects without emulator_stop, tear down what we started.
    atexit.register(cleanup_mcp_emulators, engine.config.cache.dir)

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        anyio.run(_serve)
    finally:
        with contextlib.suppress(Exception):
            from .jobs import manager_for

            manager_for(engine).shutdown()
        with contextlib.suppress(Exception):
            engine.close()
        with contextlib.suppress(Exception):
            cleanup_mcp_emulators(engine.config.cache.dir)


__all__ = [
    "SERVER_NAME",
    "build_default_engine",
    "build_server",
    "cleanup_mcp_emulators",
    "run_stdio",
]
