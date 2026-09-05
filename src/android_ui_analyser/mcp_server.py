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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from . import __version__
from .capabilities import capabilities_for_goal, capability_manifest, render_mcp_instructions
from .config import Config, load_config
from .engine import (
    _AWAIT_PREDICATE_HELD,
    Engine,
    _parse_await_terms,
    _regex_literal_hint,
    _safe_adopted_change,
)
from .errors import AuaError, UsageError
from .platforms import PlatformAdapter, TargetRef
from .projection import Projection, trim_observation_payload
from .schema import OutputFormat, publish_ids
from .selectors import normalize_selector_prefix

SERVER_NAME = "android-ui-analyser"


def _with_image(engine: Engine, args: dict[str, Any]) -> bool | str | None:
    """Per-call ``with_image``, else the engine default set by ``configure``."""
    if "with_image" in args:
        return args["with_image"]
    return getattr(engine, "_default_with_image", None)


def _ordinal(raw: Any) -> int | None:
    """*raw* as a frame-local ordinal, or ``None`` when it is a published stable id.

    A stable id is resolved through the selector instead (see :func:`_selector_from_args`), so
    coercing it here would turn a valid target into a ValueError from inside the dispatcher.
    """
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _element_target(engine: Engine, args: dict[str, Any], *, verb: str) -> Any:
    """The id for a call that takes an id and no selector.

    ``inspect`` reads one element rather than acting on one, so it has no selector parameter to
    carry an identity — but the id a caller holds is a published stable id, and refusing it here
    would mean the one read-only lookup could not accept the ids every other call hands out.
    Resolving through the engine's own key path keeps a single definition of what an id means.
    """
    raw = args.get("id")
    ordinal = _ordinal(raw)
    if ordinal is not None:
        return ordinal
    if isinstance(raw, str) and raw.strip():
        return engine._resolve_action_key(raw.strip(), verb=verb).id
    raise UsageError(f"{verb} needs an element id")


def _selector_from_args(args: dict[str, Any]) -> dict[str, Any] | None:
    """Build the engine selector from an optional stable id, or rid/text/desc (+ index/first).

    ``id`` is accepted here as well as ``stable_key``: ids are *published* as stable ids, so a
    caller pasting one back sends it in the field it came out of. A numeric ``id`` stays an
    ordinal and is handled by the caller of this function.
    """
    key = args.get("stable_key")
    if not (isinstance(key, str) and key.strip()):
        raw = args.get("id")
        if isinstance(raw, str) and raw.strip() and not raw.strip().lstrip("-").isdigit():
            key = raw
    if isinstance(key, str) and key.strip():
        # A stable_key is an identity, not a query: it needs no index/first, and it is the
        # only element name that stays meaningful outside the frame it was published in.
        identity: dict[str, Any] = {"key": key.strip()}
        bounds = args.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            identity["bounds"] = [int(value) for value in bounds]
        return identity
    rid, text, desc = args.get("rid"), args.get("text"), args.get("desc")
    if rid is None and text is None and desc is None:
        return None
    sel: dict[str, Any] = {
        "rid": normalize_selector_prefix("rid", rid),
        "text": normalize_selector_prefix("text", text),
        "desc": normalize_selector_prefix("desc", desc),
    }
    if args.get("index") is not None:
        sel["index"] = int(args["index"])
    if args.get("first"):
        sel["first"] = True
    return sel


def _optional_mic_target(
    args: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None]:
    """Return one optional mic control target while refusing ambiguous addressing."""

    target_keys = [key for key in ("id", "rid", "text", "desc") if args.get(key) is not None]
    if len(target_keys) > 1:
        raise UsageError(
            "microphone control accepts only one id/rid/text/desc target",
            hint="Pass one fresh id or one stable selector; omit all four for audio-only input.",
        )
    if not target_keys:
        if args.get("index") is not None or args.get("first"):
            raise UsageError("--index/first needs a microphone control selector")
        return None, None
    if target_keys[0] == "id":
        if args.get("index") is not None or args.get("first"):
            raise UsageError("index/first cannot modify a numeric microphone control id")
        return _ordinal(args.get("id")), _selector_from_args(args)
    return None, _selector_from_args(args)


def _dump(result: Any) -> Any:
    # Publish at the boundary: `model_dump` is the internal form and still carries frame
    # ordinals, and this is the last place before the payload reaches an agent.
    return publish_ids(
        result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    )


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
_OBSERVE_META_PROP: dict[str, Any] = {
    "type": "string",
    "description": (
        "`meta` keys to keep in the returned `observation`: 'changed' (default — did the screen "
        "move, where am I, plus anything you must not miss), 'all', or a comma-separated key "
        "list. The second cost dial, independent of `observe_fields` on purpose: wanting every "
        "column is not the same as wanting every hint. Research tasks, deeplink suggestions, "
        "capture hints and locale are not in 'changed' — call analyze_screen when you want them."
    ),
}
_OBSERVE_PROP: dict[str, Any] = {
    "type": "boolean",
    "default": True,
    "description": "Also return the post-action screen analysis.",
}
_RELATION_SELECTOR_PROP: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rid": {"type": "string"},
        "text": {"type": "string"},
        "desc": {"type": "string"},
        "index": {"type": "integer", "minimum": 0},
    },
    "oneOf": [
        {"required": ["rid"]},
        {"required": ["text"]},
        {"required": ["desc"]},
    ],
    "additionalProperties": False,
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
    "stable_key": {
        "type": "string",
        "description": (
            "Match by an element's stable_key from any observation (e.g. rid:continueBtn). "
            "It is what `id` already carries, and it outlives the frame it was read in, so it is the safe "
            "way to act on an observation this process did not produce."
        ),
    },
    "bounds": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 4,
        "maxItems": 4,
        "description": (
            "Where stable_key was seen, as [x1,y1,x2,y2]. Only used to pick between several "
            "elements sharing that key, as reusable list rows do."
        ),
    },
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
    "mic_inject": "mic_inject_and_analyze",
    "mic_speak": "mic_speak_and_analyze",
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
    {
        *_POST_ACTION_WAIT_TOOLS,
        "await_and_analyze",
        "back_until_and_analyze",
        "session_start",
        "session_autopilot",
        # `install_app` folds in a screen only when asked to launch. Trimming is a no-op on the
        # responses that carry none, and leaving it out is how MCP and the CLI drifted before:
        # MCP returned every field of every element while the CLI trimmed.
        "install_app",
    }
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
        "arrival_mismatch",
        "elapsed_ms",
        "observation",
        "observation_present",
        "known_screen",
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
    elif awaited.get("await_outcome") in _AWAIT_PREDICATE_HELD:
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
    properties["observe_meta"] = _OBSERVE_META_PROP
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
            name="policy_status",
            description=(
                "Host-only readiness for the optional guarded local policy: effective config, "
                "dependency and artifact/hash checks, and warm-daemon compatibility. Never "
                "loads a model or touches an Android device."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="session_start",
            description=(
                "Start goal-aware target work: observe once, surface relevant capabilities, "
                "and return the safest exact recommended call. Use this first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The end-to-end test goal."},
                    "contract_yaml": {
                        "type": "string",
                        "description": (
                            "Authored version-1 YAML checkpoints; only fresh assertion proof "
                            "can complete them."
                        ),
                    },
                    "artifacts_dir": {
                        "type": "string",
                        "description": "Absolute directory for the cross-command evidence bundle.",
                    },
                    "evidence": {
                        "type": "string",
                        "enum": ["none", "failures", "all"],
                        "default": "failures",
                    },
                    "junit": {"type": "boolean", "default": False},
                    "wait_for_lease_s": {
                        "type": "number",
                        "minimum": 0,
                        "default": 0,
                        "description": "Bounded device-lease wait in seconds; never steals.",
                    },
                    "start_emulator": {
                        "type": "boolean",
                        "default": True,
                        "description": "Android compatibility alias for provision_target.",
                    },
                    "provision_target": {
                        "type": "boolean",
                        "description": (
                            "Start a compatible selected-platform virtual target when no "
                            "matching target is free."
                        ),
                    },
                    "needs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Required capabilities interpreted by the selected platform; "
                            "animations remains a shared session-environment request."
                        ),
                    },
                    "headed": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Require a visible virtual target; if AUA starts one, show its window."
                        ),
                    },
                    "audio": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "When provisioning boots one, keep host audio enabled so "
                            "microphone injection is available."
                        ),
                    },
                    "animations": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Enable Android animations for this session and restore prior scales "
                            "at finish; animation goals infer this automatically."
                        ),
                    },
                    "avd": {"type": "string", "description": "AVD name when several exist."},
                    "virtual_target": {
                        "type": "string",
                        "description": (
                            "Selected-platform virtual-target definition; avd is the Android alias."
                        ),
                    },
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
                "Review exact caller-visible top_level_calls (lifecycle_calls + task_calls), "
                "journal_events including folded_internal_events, elapsed time, failures, "
                "redundant patterns, and savings. reporting_call_included=false means the "
                "embedded snapshot precedes this review/finish invocation."
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
            name="session_autopilot",
            description=(
                "Let the configured warm local policy choose and directly execute a bounded "
                "stretch of fresh-frame, guard-approved navigation taps. AUA re-observes every "
                "action and hands control back on uncertainty, no progress, repetition, input, "
                "mutation, or proof work. Requires policy advisory mode."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "max_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 6,
                    },
                    "max_duration_ms": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": 300000,
                        "default": 30000,
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="session_finish",
            description=(
                "Finish the session, restore session-owned reversible state, release its lease, "
                "keep an AUA-started emulator warm until the lease-gated idle timeout, and return "
                "a compact final verdict. Set summary=false only when the full review/evidence "
                "payload is needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "allow_incomplete": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Explicitly terminate despite incomplete authored checkpoints."
                        ),
                    },
                    "summary": {
                        "type": "boolean",
                        "default": True,
                        "description": "Compact verdict/cleanup/accounting instead of full detail.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="session_candidate_flow",
            description=(
                "Preview a contract-proven action path. Replay/save require an explicit reset "
                "flow; saving occurs only after both reset and candidate replay pass."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "session_id": {"type": "string"},
                    "reset_flow": {
                        "type": "string",
                        "description": "Saved flow name or absolute YAML path for deterministic reset.",
                    },
                    "replay": {"type": "boolean", "default": False},
                    "save": {"type": "boolean", "default": False},
                },
                "required": ["name"],
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
                    "observe_meta": _OBSERVE_META_PROP,
                },
                "required": ["predicate"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="flow_list",
            description=(
                "List saved flows with app/context compatibility, step count, parameters, "
                "arrival proof, description, and path. Flows are filed per app; pass app to "
                "narrow to one package, and replay a flow by its `ref` (a bare name, or "
                "`<package>:<name>` when two apps share one)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app": {
                        "type": "string",
                        "description": "Package whose flows to list (plus app-agnostic ones).",
                    }
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="flow_save",
            description=(
                "Preview recent recorded actions as an editable reusable flow without writing. "
                "Review scope, value-free selector_resilience, and arrival proof. An unmapped "
                "arrival is captured only from a positive action-bound until satisfied on the "
                "same frame; otherwise it remains unverified. Set save=true only after review."
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
                "status=already_absent when it was already gone. Qualify the name as "
                "`<package>:<flow>` when two apps own flows of that name."
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
            name="app_log_prefs_get",
            description=(
                "Read one app's persisted app_logs preferences: which tags it ignores, which "
                "it reports despite the built-in noise list, any only-list, and the line, "
                "per-tag and priority settings — plus what they resolve to and what the "
                "built-in list already hides."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "App id to read."},
                },
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="app_log_prefs_set",
            description=(
                "Persist one app's app_logs preferences: ignore a chatty tag, stop ignoring one "
                "(including a tag the built-in noise list hides), keep only the tags you are "
                "chasing, or raise the 20-line budget and the 5-per-tag cap. Stored per app id "
                "beside its map on this host, so every later session inherits it — unlike "
                "`configure`, which lasts this session and applies to every app. Needs no "
                "device. `F` lines survive every tag filter, so this can never hide a crash, "
                "and an ignored tag stays ignored even when only_tags names it too. `configure` "
                "set in this session outranks what is stored here."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "App id these apply to."},
                    "ignore_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to stop reporting for this app (prefix match).",
                    },
                    "unignore_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Tags to report again — reaches the built-in noise list too. Tags "
                            "that were not being ignored come back in `not_ignored`."
                        ),
                    },
                    "only_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Report ONLY these tags for this app; [] clears the list. A narrowed "
                            "window says so, as `only` in the digest."
                        ),
                    },
                    "levels": {
                        "type": "string",
                        "description": (
                            "Priority SET, not a floor (default 'DWEF'). 'I' is noisier than 'D' "
                            "on Android, so widen to 'DIWEF' only when chasing a library."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Lines attached per action (default 20), head+tail on overflow.",
                    },
                    "per_tag": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Lines one tag may contribute before it is capped (default 5). "
                        "A tag named in only_tags is never capped.",
                    },
                    "scan_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5000,
                        "description": "Lines read from the window before filtering (default 600).",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Fold this one app's logs into its observations, or not.",
                    },
                    "reset": {
                        "type": "boolean",
                        "default": False,
                        "description": "Forget this app's preferences; cannot be combined with a change.",
                    },
                },
                "required": ["package"],
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
                    "no_cache": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force a fresh capture instead of reusing cached hierarchy data.",
                    },
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
            description=(
                "Tap by a fresh observation id or by one stable rid/text/desc selector. "
                "Prefer a stable selector when one is available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": ["integer", "string"]},
                    **_SELECTOR_PROPS,
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "oneOf": [
                    {"required": ["id"]},
                    {"required": ["rid"]},
                    {"required": ["text"]},
                    {"required": ["desc"]},
                ],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="input",
            description=(
                "Type text into the element with the given id; optionally use the IME submit "
                "action or name one explicit semantic app send control."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": ["integer", "string"]},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                    "send": {
                        "type": "string",
                        "description": "Stable id of the app's explicit send control.",
                    },
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
                    "observe_meta": _OBSERVE_META_PROP,
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
                "properties": {"id": {"type": ["integer", "string"]}},
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
                    "id": {"type": ["integer", "string"]},
                    "ms": {"type": "integer", "default": 600},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="mic_inject",
            description=(
                "Inject a host U8/S16 PCM WAV into an Android Emulator microphone. An optional "
                "control defaults to push-to-talk hold, or can tap once to start and once to stop."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Host path to a PCM WAV."},
                    "id": {"type": "integer", "minimum": 0},
                    **_SELECTOR_PROPS,
                    "control_mode": {
                        "type": "string",
                        "enum": ["hold", "toggle"],
                        "default": "hold",
                        "description": (
                            "hold = DOWN/audio/UP; toggle requires an initially-off target and "
                            "uses one non-retrying tap to start and one to stop."
                        ),
                    },
                    "pre_roll_ms": {"type": "integer", "minimum": 0, "default": 250},
                    "post_roll_ms": {"type": "integer", "minimum": 0, "default": 250},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="mic_speak",
            description=(
                "On macOS, synthesize text with /usr/bin/say and inject it into an Android "
                "Emulator microphone; optionally hold one control or toggle it on then off."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "speech": {"type": "string", "minLength": 1},
                    "id": {"type": "integer", "minimum": 0},
                    **_SELECTOR_PROPS,
                    "control_mode": {
                        "type": "string",
                        "enum": ["hold", "toggle"],
                        "default": "hold",
                        "description": (
                            "hold = DOWN/audio/UP; toggle requires an initially-off target and "
                            "uses one non-retrying tap to start and one to stop."
                        ),
                    },
                    "voice": {"type": "string", "description": "Installed macOS say voice."},
                    "rate": {"type": "integer", "minimum": 1},
                    "pre_roll_ms": {"type": "integer", "minimum": 0, "default": 250},
                    "post_roll_ms": {"type": "integer", "minimum": 0, "default": 250},
                    "observe": _OBSERVE_PROP,
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["speech"],
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
            description="Replay a saved, file-backed, or inline-YAML flow (whole journey — launch, taps, waits, "
            "asserts, cross-app auth) in one call; on divergence returns the failing "
            "step index + remaining steps, resumable via from_step. assist=true lets the "
            "opt-in planner clear a blocker and resume (needs planner.enabled). Optional "
            "artifacts_dir writes portable evidence and JUnit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "file": {"type": "string"},
                    "yaml": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "${NAME} placeholder values.",
                        "additionalProperties": {"type": "string"},
                    },
                    "dry_run": {"type": "boolean", "default": False},
                    "from_step": {"type": "integer", "default": 0},
                    "allow_destructive": {"type": "boolean", "default": True},
                    "assist": {"type": "boolean", "default": False},
                    "artifacts_dir": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "enum": ["none", "failures", "all"],
                        "default": "failures",
                    },
                    "junit": {"type": "boolean", "default": False},
                },
                "oneOf": [
                    {"required": ["name"]},
                    {"required": ["file"]},
                    {"required": ["yaml"]},
                ],
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
            name="teardown_status",
            description="List pending undos and blocked ledger files without connecting to a target.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="teardown_run",
            description="Replay safe pending undos; force bypasses leases, never boot/configuration proof.",
            inputSchema={
                "type": "object", "properties": {
                    "target_id": {"type": "string"},
                    "force": {"type": "boolean", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                }, "additionalProperties": False,
            },
        ),
        types.Tool(
            name="teardown_discard",
            description=("Explicitly abandon named stale undos on an unleased target. Archives evidence "
                         "but does not connect to or restore the device. Requires human authorization."),
            inputSchema={
                "type": "object", "properties": {
                    "target_id": {"type": "string", "minLength": 1},
                    "keys": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "confirmed": {"type": "boolean", "default": False},
                }, "required": ["target_id", "keys", "reason", "confirmed"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="virtual_target_list",
            description="List reusable virtual-target definitions for the selected platform.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="virtual_target_status",
            description="Show configured, running, and AUA-owned virtual targets.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="virtual_target_start",
            description="Start one selected-platform virtual target without opening a session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "definition_id": {"type": "string"},
                    "headless": {"type": "boolean", "default": True},
                    "audio": {"type": "boolean", "default": False},
                    "animations": {"type": "boolean", "default": False},
                    "wait": {"type": "number", "minimum": 0, "default": 120},
                    "owner": {"type": "string"},
                    "parallel": {"type": "boolean", "default": False},
                    "options": {
                        "type": "object",
                        "description": "Opaque options validated by the selected adapter.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="virtual_target_provision",
            description="Select and start a compatible virtual target as one Engine operation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "definition_id": {"type": "string"},
                    "needs": {"type": "array", "items": {"type": "string"}},
                    "headless": {"type": "boolean", "default": True},
                    "audio": {"type": "boolean", "default": False},
                    "animations": {"type": "boolean", "default": False},
                    "wait": {"type": "number", "minimum": 0, "default": 120},
                    "owner": {"type": "string"},
                    "parallel": {"type": "boolean", "default": True},
                    "options": {
                        "type": "object",
                        "description": "Opaque options validated by the selected adapter.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="virtual_target_create",
            description="Create or idempotently reuse a selected-platform target definition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "definition_id": {"type": "string", "minLength": 1},
                    "replace": {"type": "boolean", "default": False},
                    "confirmed": {
                        "type": "boolean",
                        "default": False,
                        "description": "Required when replace=true because saved data may be lost.",
                    },
                    "options": {"type": "object"},
                },
                "required": ["definition_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="virtual_target_delete",
            description="Delete a stopped virtual-target definition after confirmation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "definition_id": {"type": "string", "minLength": 1},
                    "confirmed": {"type": "boolean", "default": False},
                    "options": {"type": "object"},
                },
                "required": ["definition_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="virtual_target_stop",
            description="Stop selected virtual targets while preserving foreign live leases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_id": {"type": "string"},
                    "definition_id": {"type": "string"},
                    "owner": {"type": "string"},
                    "mine": {"type": "boolean", "default": False},
                    "all": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="virtual_target_reclaim",
            description="Re-arm retirement supervision for AUA-owned orphan instances.",
            inputSchema={
                "type": "object",
                "properties": {
                    "idle_stop": {"type": "number", "minimum": 0},
                },
                "additionalProperties": False,
            },
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
            "Automatically clears a confirmed unowned proxy black hole inherited from the AVD; "
            "reachable foreign and AUA-owned proxies are preserved. "
            "This provisions an AVD but does not open a goal session; agents should call "
            "session_start for automatic selection/provisioning/leasing. Ordinary device tools follow the "
            "owner's one automatic sticky lease and do not repeat a serial. "
            "REQUIRED: call emulator_stop when done (or stop_mine) — orphaned AVDs burn CPU. "
            "Idle auto-stop is only a safety net.",
            inputSchema={
                "type": "object",
                "properties": {
                    "avd": {"type": "string", "description": "AVD name (omit if only one exists)."},
                    "headless": {"type": "boolean", "default": True},
                    "audio": {
                        "type": "boolean",
                        "default": False,
                        "description": "Keep host audio enabled for emulator microphone injection.",
                    },
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
                    "id": {"type": ["integer", "string"]},
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
                    "id": {"type": ["integer", "string"]},
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
                    "count": {"type": "integer", "minimum": 0},
                    "within": _RELATION_SELECTOR_PROP,
                    "same_parent_as": _RELATION_SELECTOR_PROP,
                    "contains_all": {
                        "type": "array",
                        "items": _RELATION_SELECTOR_PROP,
                        "minItems": 1,
                    },
                    "index": {"type": "integer", "minimum": 0},
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
                    "id": {"type": ["integer", "string"]},
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
                    "id": {"type": ["integer", "string"]},
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
            name="screen_record_start",
            description=(
                "Start an MP4 SCREEN VIDEO recording on the device. This captures pixels, not "
                "HTTP traffic — for traffic use mock_record."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target path override; the selected platform chooses its default."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="screen_record_stop",
            description="Stop the screen video recording and pull the MP4 to a local path.",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        # Compatibility aliases for clients generated before the clearer screen_* names. Keep
        # them explicit and unmistakable; removing MCP tools breaks cached agent tool calls.
        types.Tool(
            name="record_start",
            description=(
                "Deprecated alias for screen_record_start. Starts MP4 SCREEN VIDEO, never HTTP "
                "traffic; use mock_record for traffic."
            ),
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="record_stop",
            description="Deprecated alias for screen_record_stop; stops and pulls MP4 screen video.",
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
            name="capture_sheet",
            description=(
                "Export an evenly sampled PNG contact sheet with relative timestamps; "
                "does not require ffmpeg."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "seconds": {"type": "number"},
                    "since": {
                        "type": "string",
                        "description": "Use 'last-action' for post-action frames.",
                    },
                    "max_frames": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 24,
                        "default": 6,
                    },
                    "columns": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 3,
                    },
                    "timestamps": {"type": "boolean", "default": True},
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
                    "id": {"type": ["integer", "string"]},
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
            name="proxy_status",
            description=(
                "Is interception ACTUALLY working end to end? Returns state="
                "unproxied|healthy|degraded|foreign|blackholed plus owned/intercepting. "
                "Diagnoses a device pointed at a proxy no aua owns — including the black-hole "
                "state where every app request fails with ConnectException."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "heal": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Re-establish a dropped adb reverse tunnel when the process and "
                            "device setting already check out. Never heals a proxy this "
                            "session cannot prove is its own."
                        ),
                    }
                },
                "additionalProperties": False,
            },
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
        # Without these two, MCP could only *consume* the mock surface: start the proxy and
        # replay a cassette somebody else authored with the CLI. Recording and ad-hoc stubbing
        # are how a cassette comes to exist, so an MCP-only agent could never begin.
        types.Tool(
            name="mock_record",
            description=(
                "Record live HTTP traffic into a named YAML cassette (start|stop). Records "
                "requests/responses through the proxy, not screen video — for video use "
                "screen_record_start. Requires proxy_start first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop"],
                        "description": "Begin or end the recording window.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Cassette name; required for start.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="mock_map",
            description=(
                "Arm one ad-hoc HTTP mock rule: answer METHOD PATH with a fixed status and body. "
                "Appends to the live rules the proxy already reloads, so no restart is needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method, for example GET or POST.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Request path to match, for example /v1/hub.",
                    },
                    "status": {
                        "type": "integer",
                        "default": 200,
                        "description": "Status code to answer with.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Response body to answer with.",
                    },
                },
                "required": ["method", "path"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="mock_rewrite",
            description=(
                "Arm one HTTP rule that lets the request REACH the server and then patches "
                "the real response — status, headers, or individual JSON fields. Use this "
                "instead of mock_map to reproduce a server-side condition (a 429, a missing "
                "field) on real data. A rule with no host and a catch-all path is refused: "
                "it would also intercept the platform's connectivity probes and the device "
                "would look offline."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method, for example GET or POST. `*` matches any.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Request path to match, for example /v1/hub.",
                    },
                    "host": {
                        "type": "string",
                        "description": "Restrict the rule to one host.",
                    },
                    "status": {
                        "type": "integer",
                        "description": "Replace the response status code.",
                    },
                    "headers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Response headers to set.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Replace the whole response body.",
                    },
                    "set_json": {
                        "type": "object",
                        "description": (
                            "JSON fields to set, keyed by path: `items[0].title` or "
                            "`items.0.title`, values as-is."
                        ),
                    },
                    "delete_json": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "JSON field paths to delete.",
                    },
                    "times": {
                        "type": "integer",
                        "default": 0,
                        "description": "Fire at most N times; 0 means every time.",
                    },
                },
                "required": ["method", "path"],
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
            name="app_status",
            description="Read package presence and version from AUA's leased target. This is "
            "read-only; the result includes the exact selected serial.",
            inputSchema={
                "type": "object",
                "properties": {"package": {"type": "string", "minLength": 1}},
                "required": ["package"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="shell_read_only",
            description="Run one bounded read-only diagnostic command on AUA's leased target. "
            "Every argv item is quoted before Android's remote shell parses it; unknown or "
            "mutating verbs are refused, and each output stream is capped at 256 KiB.",
            inputSchema={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Command argv, for example ['pm', 'path', 'com.example.app'].",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "default": 30000,
                        "minimum": 100,
                        "maximum": 120000,
                    },
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="install_app",
            description="Install an app bundle (.apk) on the device instead of shelling out to "
            "`adb install`. Idempotent: an app already there at the bundle's version is left "
            "alone. With launch=true it also opens the app and returns the resulting screen in "
            "`observation` — use its fresh ids and do not call analyze_screen next.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bundle": {
                        "type": "string",
                        "description": "Host path to the .apk to install.",
                    },
                    "package": {
                        "type": "string",
                        "description": "Optional assertion: fail if the bundle declares another "
                        "package id.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["if-needed", "reinstall", "fresh"],
                        "default": "if-needed",
                        "description": "Differs only when the app is already installed. "
                        "if-needed: skip unless the version differs. reinstall: push anyway, "
                        "keeping app data. fresh: uninstall first (wipes data; the only mode that "
                        "survives a signing-key change).",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "default": False,
                        "description": "Required when mode=fresh because it wipes app data.",
                    },
                    "grant_permissions": {
                        "type": "boolean",
                        "default": False,
                        "description": "Pre-grant runtime permissions. Off by default: it skips "
                        "the permission dialog a scenario may be there to verify.",
                    },
                    "launch": {"type": "boolean", "default": False},
                    "activity": {"type": "string"},
                    "timeout_ms": {"type": "integer", "default": 300000, "minimum": 1000},
                    "with_image": _WITH_IMAGE_PROP,
                },
                "required": ["bundle"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="helper_status",
            description=(
                "Report whether the selected platform's optional on-device helper is "
                "installed, enabled, bound, and ready."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="helper_install",
            description="Explicitly install the selected platform's optional device helper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "reinstall": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="helper_enable",
            description=(
                "Enable the optional device helper through the shared Engine path, with "
                "write-ahead restoration of device state."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="helper_disable",
            description="Disable the optional device helper and restore its saved setup state.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        types.Tool(
            name="helper_remove",
            description="Disable and explicitly uninstall the optional device helper.",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
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
                "Run one read-only SQLite statement against a private-database snapshot and "
                "return columns plus JSON rows. By default this keeps the app running and "
                "preserves navigation/UI state. Pass `live: false` only when a transactionally "
                "coherent snapshot is required; that force-stops the app and the result's "
                "`warning` says the previous UI state is gone."
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
                    "live": {
                        "type": "boolean",
                        "default": True,
                        "description": "Keep the app running and preserve UI state (default). "
                        "Set false for a coherent stop-first snapshot. Ignores `restart` when "
                        "true.",
                    },
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
                        "description": "A published stable id, or a legacy frame-local integer id.",
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
            "(e.g. with_image on every observation, or which app log levels they fold in).",
            inputSchema={
                "type": "object",
                "properties": {
                    "with_image": {
                        "type": "boolean",
                        "description": "Default with_image for action tools that observe.",
                    },
                    "app_logs": {
                        "type": "boolean",
                        "description": (
                            "Fold what the app logged during an action into its observation "
                            "(default true). Set false to stop paying for it."
                        ),
                    },
                    "app_log_levels": {
                        "type": "string",
                        "description": (
                            "Log priorities to fold in, as a SET not a floor (default 'DWEF'). "
                            "'I' is noisier than 'D' on Android — measured, every 'I' line in a "
                            "real launch window came from an SDK or the runtime — so widen to "
                            "'DIWEF' only when chasing a library. 'F' is always included."
                        ),
                    },
                    "app_log_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Lines folded into each action (default 20). Raise it for one stretch "
                            "of a session; use app_log_prefs_set to keep it for an app."
                        ),
                    },
                    "app_log_per_tag": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Lines one tag may contribute before it is capped (default 5).",
                    },
                    "app_log_ignore_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Tags to drop this session, on top of the built-in noisy list. "
                            "Replaces the current session list; [] clears it."
                        ),
                    },
                    "app_log_keep_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Tags to report despite the built-in noisy list, this session. "
                            "Replaces the current session list; [] clears it."
                        ),
                    },
                    "app_log_only_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Fold in ONLY these tags this session; [] clears the list. 'F' lines "
                            "survive it, so this cannot hide a crash."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]
    return [_with_phase_checkpoint(_as_analyzed_tool(tool)) for tool in tools]


# --------------------------------------------------------------------------- dispatch


def _dispatch(engine: Engine, name: str, args: dict[str, Any]) -> Any:
    """Call the engine method for ``name``; the server brackets the complete response turn."""
    return _dispatch_tool(engine, name, args)


def _dispatch_tool(engine: Engine, name: str, args: dict[str, Any]) -> Any:
    """Call the engine method for ``name`` and return a JSON-serialisable payload."""
    args = dict(args)
    internal_name = _ANALYZED_TOOL_BASES.get(name)
    if internal_name is not None:
        name = internal_name
        args = {**args, "observe": True}
        # Dropped rather than read: no engine method takes it. The caller's copy in `args_in`
        # is what trims the folded observation, at the boundary every tool returns through.
        args.pop("observe_fields", None)
        args.pop("observe_meta", None)
        for wrapper_arg in _UNTIL_PROPS:
            args.pop(wrapper_arg, None)
    elif name in _ANALYZED_TOOL_NAMES:
        raise UsageError(
            f"MCP tool '{name}' was renamed to '{_ANALYZED_TOOL_NAMES[name]}'",
            hint="The renamed tool already returns the analyzed resulting screen; use its "
            "`observation` and do not call `analyze_screen` afterward.",
        )
    if name == "policy_status":
        from .daemon import policy_runtime_status

        return policy_runtime_status(engine.config)
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
    if name not in {
        "app_log_prefs_get",
        "app_log_prefs_set",
        "capabilities",
        "session_progress",
        "session_autopilot",
        "session_review",
        "session_candidate_flow",
        "configure",
    }:
        reject_if_active(engine, name)

    img = _with_image(engine, args)

    if name == "capabilities":
        goal = args.get("goal")
        return {"capabilities": capabilities_for_goal(str(goal)) if goal else capability_manifest()}
    if name == "session_start":
        start_kwargs: dict[str, Any] = {
            "start_emulator": bool(args.get("start_emulator", True)),
            "headed": bool(args.get("headed", False)),
            "audio": bool(args.get("audio", False)),
            "animations": bool(args.get("animations", False)),
            "avd": args.get("avd"),
            "needs": args.get("needs"),
            "package": args.get("package"),
            "activity": args.get("activity"),
        }
        if "provision_target" in args:
            start_kwargs["provision_target"] = bool(args["provision_target"])
        if "virtual_target" in args:
            start_kwargs["virtual_target"] = args["virtual_target"]
        for key in ("contract_yaml", "artifacts_dir", "evidence", "junit", "wait_for_lease_s"):
            if key in args:
                start_kwargs[key] = args[key]
        started = _dump(
            _engine_method(engine, "session_start")(
                str(args["goal"]),
                **start_kwargs,
            )
        )
        if isinstance(started, dict) and (
            started.get("virtual_target_started") or started.get("emulator_started")
        ):
            _track_mcp_target(engine, started)
        return started
    if name == "session_review":
        return _dump(_engine_method(engine, "session_review")(session_id=args.get("session_id")))
    if name == "session_progress":
        return _dump(_engine_method(engine, "session_progress")(session_id=args.get("session_id")))
    if name == "session_autopilot":
        return _dump(
            _engine_method(engine, "session_autopilot")(
                session_id=args.get("session_id"),
                max_steps=int(args.get("max_steps", 6)),
                max_duration_ms=int(args.get("max_duration_ms", 30_000)),
            )
        )
    if name == "session_finish":
        finish_kwargs: dict[str, Any] = {
            "session_id": args.get("session_id"),
            "summary": bool(args.get("summary", True)),
        }
        if "allow_incomplete" in args:
            finish_kwargs["allow_incomplete"] = bool(args["allow_incomplete"])
        return _dump(_engine_method(engine, "session_finish")(**finish_kwargs))
    if name == "session_candidate_flow":
        return _dump(
            _engine_method(engine, "session_candidate_flow")(
                str(args["name"]),
                session_id=args.get("session_id"),
                reset_flow=args.get("reset_flow"),
                replay=bool(args.get("replay", False)),
                save=bool(args.get("save", False)),
            )
        )
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
        return engine.flow_list(app=args.get("app"))
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
    if name == "app_log_prefs_get":
        return _dump(engine.app_log_prefs(app=str(args["package"])))
    if name == "app_log_prefs_set":
        return _dump(
            engine.app_log_prefs_set(
                app=str(args["package"]),
                ignore_tags=args.get("ignore_tags"),
                unignore_tags=args.get("unignore_tags"),
                only_tags=args.get("only_tags"),
                levels=args.get("levels"),
                limit=args.get("limit"),
                per_tag=args.get("per_tag"),
                scan_lines=args.get("scan_lines"),
                enabled=args.get("enabled"),
                reset=bool(args.get("reset", False)),
            )
        )
    if name == "analyze_screen":
        result = engine.analyze(
            source=args.get("source", "auto"),
            with_ocr=args.get("with_ocr"),
            no_cache=bool(args.get("no_cache", False)),
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
                _ordinal(args.get("id")),
                selector=_selector_from_args(args),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "input":
        return _dump(
            engine.input_text(
                _ordinal(args.get("id")),
                args["text"],
                selector=_selector_from_args(args),
                submit=args.get("submit", False),
                send_key=args.get("send"),
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
        return _dump(engine.inspect(_element_target(engine, args, verb="inspect")))
    if name == "long_press":
        return _dump(
            engine.long_press(
                _ordinal(args.get("id")),
                selector=_selector_from_args(args),
                ms=int(args.get("ms", 600)),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "mic_inject":
        element_id, selector = _optional_mic_target(args)
        return _dump(
            engine.mic_inject(
                args["path"],
                element_id,
                selector=selector,
                control_mode=str(args.get("control_mode", "hold")),
                pre_roll_ms=int(args.get("pre_roll_ms", 250)),
                post_roll_ms=int(args.get("post_roll_ms", 250)),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "mic_speak":
        element_id, selector = _optional_mic_target(args)
        return _dump(
            engine.mic_speak(
                str(args["speech"]),
                element_id,
                selector=selector,
                control_mode=str(args.get("control_mode", "hold")),
                voice=args.get("voice"),
                rate=int(args["rate"]) if args.get("rate") is not None else None,
                pre_roll_ms=int(args.get("pre_roll_ms", 250)),
                post_roll_ms=int(args.get("post_roll_ms", 250)),
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
            args.get("name"),
            file=args.get("file"),
            yaml=args.get("yaml"),
            params={str(k): str(v) for k, v in (args.get("params") or {}).items()},
            dry_run=args.get("dry_run", False),
            from_step=int(args.get("from_step", 0)),
            allow_destructive=args.get("allow_destructive", True),
            assist=args.get("assist", False),
            artifacts_dir=args.get("artifacts_dir"),
            evidence=args.get("evidence", "failures"),
            junit=args.get("junit", False),
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
    if name == "teardown_status":
        return engine.teardown_status()
    if name == "teardown_run":
        return engine.teardown_run(
            serial=args.get("target_id"), force=bool(args.get("force", False)),
            dry_run=bool(args.get("dry_run", False)),
        )
    if name == "teardown_discard":
        return engine.teardown_discard(
            serial=str(args["target_id"]), keys=list(args["keys"]), reason=str(args["reason"]),
            confirmed=bool(args.get("confirmed", False)),
        )
    if name == "virtual_target_list":
        return engine.virtual_target_list()
    if name == "virtual_target_status":
        return engine.virtual_target_status()
    if name in {"virtual_target_start", "virtual_target_provision"}:
        common: dict[str, Any] = {
            "headless": bool(args.get("headless", True)),
            "audio": bool(args.get("audio", False)),
            "animations": bool(args.get("animations", False)),
            "wait_s": float(args.get("wait", 120)),
            "owner": args.get("owner"),
            "parallel": bool(
                args.get("parallel", name == "virtual_target_provision")
            ),
            "options": dict(args.get("options") or {}),
        }
        if name == "virtual_target_provision":
            out = engine.virtual_target_provision(
                args.get("definition_id"),
                needs=args.get("needs") or [],
                **common,
            )
        else:
            out = engine.virtual_target_start(args.get("definition_id"), **common)
        _track_mcp_target(engine, out)
        return out
    if name == "virtual_target_create":
        replace_existing = bool(args.get("replace", False))
        if replace_existing and not bool(args.get("confirmed", False)):
            raise UsageError(
                "virtual_target_create replace=true needs confirmed=true",
                hint="Replacement may discard the existing virtual target's saved data.",
            )
        return engine.virtual_target_create(
            str(args["definition_id"]),
            replace=replace_existing,
            options=dict(args.get("options") or {}),
        )
    if name == "virtual_target_delete":
        return engine.virtual_target_delete(
            str(args["definition_id"]),
            confirmed=bool(args.get("confirmed", False)),
            options=dict(args.get("options") or {}),
        )
    if name == "virtual_target_stop":
        from . import leases as leases_mod

        out = engine.virtual_target_stop(
            target_id=args.get("target_id"),
            definition_id=args.get("definition_id"),
            owner=args.get("owner"),
            mine=bool(args.get("mine", False)),
            all_targets=bool(args.get("all", False)),
            lease_owner=leases_mod.resolve_owner(args.get("owner")),
        )
        stopped = out.get("stopped_target_ids") if isinstance(out, dict) else None
        if isinstance(stopped, list):
            _forget_mcp_targets(engine, stopped)
            attached = getattr(engine, "_device", None)
            if attached is not None and attached.target_id in stopped:
                engine.close()
        return out
    if name == "virtual_target_reclaim":
        idle = args.get("idle_stop")
        if idle is None:
            idle = float(getattr(engine.config.teardown, "emulator_idle_stop_s", 1200.0))
        return engine.virtual_target_reclaim(idle_timeout_s=float(idle))
    if name == "emulator_list":
        return engine.emulator_list()
    if name == "emulator_status":
        return engine.emulator_status()
    if name == "emulator_start":
        headless = bool(args.get("headless", True))
        idle = args.get("idle_stop")
        if idle is None:
            # Same default as the CLI, windowed included: MCP and CLI must share one policy or
            # an agent's emulator outlives it on whichever surface was forgotten.
            idle = float(getattr(engine.config.teardown, "emulator_idle_stop_s", 1200.0))
        out = engine.emulator_start(
            args.get("avd"),
            headless=headless,
            audio=bool(args.get("audio", False)),
            animations=bool(args.get("animations", False)),
            wait_s=float(args.get("wait", 120)),
            gpu=args.get("gpu"),
            idle_timeout_s=float(idle),
            parallel=bool(args.get("parallel", False)),
            port=args.get("port"),
            read_only=args.get("read_only"),
            owner=args.get("owner"),
        )
        _track_mcp_target(engine, out)
        return out
    if name == "emulator_stop":
        from . import leases as leases_mod

        out = engine.emulator_stop(
            serial=args.get("serial"),
            avd=args.get("avd"),
            owner=args.get("owner"),
            mine=bool(args.get("mine", False)),
            all_devices=bool(args.get("all", False)),
            lease_owner=leases_mod.resolve_owner(args.get("owner")),
        )
        stopped = out.get("stopped") if isinstance(out, dict) else None
        if isinstance(stopped, list):
            _forget_mcp_targets(engine, stopped)
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
                _ordinal(args.get("id")),
                selector=_selector_from_args(args),
                observe=args.get("observe", True),
                with_image=img,
            )
        )
    if name == "clear":
        return _dump(
            engine.clear(
                _ordinal(args.get("id")),
                selector=_selector_from_args(args),
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
                count=args.get("count"),
                within=args.get("within"),
                same_parent_as=args.get("same_parent_as"),
                contains_all=args.get("contains_all"),
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
        element_id = _ordinal(args.get("id"))
        return _dump(engine.copy_text(element_id, selector=selector))
    if name == "erase":
        selector = _selector_from_args(args)
        element_id = _ordinal(args.get("id"))
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
    if name in {"screen_record_start", "record_start"}:
        return _dump(engine.record_start(args.get("path")))
    if name in {"screen_record_stop", "record_stop"}:
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
    if name == "capture_sheet":
        return _dump(
            engine.capture_sheet(
                args["path"],
                seconds=args.get("seconds"),
                since=args.get("since"),
                max_frames=int(args.get("max_frames", 6)),
                columns=int(args.get("columns", 3)),
                timestamps=bool(args.get("timestamps", True)),
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
        eid = _ordinal(args.get("id"))
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
    if name == "proxy_status":
        return _dump(engine.proxy_status(heal=bool(args.get("heal", True))))
    if name == "mock_replay":
        return _dump(engine.mock_replay(args["name"]))
    if name == "mock_record":
        return _dump(engine.mock_record(str(args["action"]), args.get("name")))
    if name == "mock_map":
        return _dump(
            engine.mock_map(
                str(args["method"]),
                str(args["path"]),
                status=int(args.get("status", 200)),
                body=args.get("body"),
            )
        )
    if name == "mock_rewrite":
        raw_status = args.get("status")
        return _dump(
            engine.mock_rewrite(
                str(args["method"]),
                str(args["path"]),
                host=args.get("host"),
                status=int(raw_status) if raw_status is not None else None,
                headers=args.get("headers") or None,
                body=args.get("body"),
                set_json=args.get("set_json") or None,
                delete_json=args.get("delete_json") or None,
                times=int(args.get("times", 0)),
            )
        )
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
    if name == "app_status":
        return _dump(engine.app_status(args["package"]))
    if name == "shell_read_only":
        return _dump(
            engine.shell(
                list(args["argv"]),
                timeout_ms=int(args.get("timeout_ms", 30000)),
            )
        )
    if name == "install_app":
        launch = bool(args.get("launch", False))
        return _dump(
            engine.install_app(
                args["bundle"],
                package=args.get("package"),
                mode=str(args.get("mode", "if-needed")),
                confirmed=bool(args.get("confirmed", False)),
                grant_permissions=bool(args.get("grant_permissions", False)),
                launch=launch,
                activity=args.get("activity"),
                observe=launch,
                with_image=img,
                timeout_ms=int(args.get("timeout_ms", 300000)),
            )
        )
    if name == "helper_status":
        return _dump(engine.helper_status())
    if name == "helper_install":
        return _dump(
            engine.helper_install(
                reinstall=bool(args.get("reinstall", False)),
                force=bool(args.get("force", False)),
            )
        )
    if name == "helper_enable":
        return _dump(engine.helper_enable())
    if name == "helper_disable":
        return _dump(engine.helper_disable())
    if name == "helper_remove":
        return _dump(engine.helper_remove())
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
            live=bool(args.get("live", True)),
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
        if "app_logs" in args:
            engine.config.logs.enabled = bool(args["app_logs"])
        # Every log field set here is recorded as a SESSION field, which is what makes it beat a
        # stored per-app preference. An agent asking for 60 lines while chasing a library must get
        # 60 even on an app whose remembered preference says 20 — otherwise this tool quietly does
        # nothing for exactly the apps that use the persisted one.
        for arg, field in (
            ("app_log_levels", "levels"),
            ("app_log_limit", "limit"),
            ("app_log_per_tag", "per_tag"),
            ("app_log_ignore_tags", "deny_tags"),
            ("app_log_keep_tags", "keep_tags"),
            ("app_log_only_tags", "only_tags"),
        ):
            if arg not in args:
                continue
            if field in {"deny_tags", "keep_tags", "only_tags"}:
                value: Any = [str(tag) for tag in (args[arg] or []) if str(tag).strip()]
            elif field == "levels":
                value = str(args[arg])
            else:
                value = int(args[arg])
            setattr(engine.config.logs, field, value)
            engine._session_log_fields.add(field)
        return {
            "ok": True,
            "with_image": getattr(engine, "_default_with_image", None),
            "app_logs": engine.config.logs.enabled,
            "app_log_levels": engine.config.logs.levels,
            "app_log_limit": engine.config.logs.limit,
            "app_log_per_tag": engine.config.logs.per_tag,
            "app_log_ignore_tags": list(engine.config.logs.deny_tags),
            "app_log_keep_tags": list(engine.config.logs.keep_tags),
            "app_log_only_tags": list(engine.config.logs.only_tags),
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


@dataclass(frozen=True)
class _McpStartedTarget:
    target: TargetRef
    instance_token: str
    expected_pid: int | None
    config: Config
    platform: PlatformAdapter


# A native id and an owner label can both be reused. Retain the exact adapter/configuration
# and boot token from startup; no exit path may widen this to target-id or owner-wide stop.
_MCP_STARTED_TARGETS: dict[tuple[int, str, str], _McpStartedTarget] = {}
_LEASE_FREE_TOOLS = frozenset(
    {
        "app_log_prefs_get",
        "app_log_prefs_set",
        "capabilities",
        "capture_explain",
        "capture_export",
        "capture_last",
        "capture_sheet",
        "capture_status",
        "configure",
        "emulator_list",
        "emulator_start",
        "emulator_status",
        "emulator_stop",
        "flow_delete",
        "flow_list",
        "flow_save",
        "job_cancel",
        "job_list",
        "job_status",
        "job_wait",
        "knowledge_add",
        "knowledge_list",
        "knowledge_stale",
        "list_devices",
        "map_audit",
        "map_find",
        "policy_status",
        "reconcile_apply",
        "reconcile_plan",
        "reconcile_rollback",
        "reconcile_status",
        "reconcile_submit",
        "session_candidate_flow",
        "session_start",
        "session_progress",
        "session_review",
        "teardown_status",
        "teardown_run",
        "teardown_discard",
        "virtual_target_create",
        "virtual_target_delete",
        "virtual_target_list",
        "virtual_target_provision",
        "virtual_target_reclaim",
        "virtual_target_start",
        "virtual_target_status",
        "virtual_target_stop",
    }
)


def _track_mcp_target(engine: Engine, result: Any) -> None:
    if not isinstance(result, dict):
        return
    target_id = result.get("target_id") or result.get("serial")
    token = (
        result.get("instance_token")
        or result.get("virtual_target_instance_token")
        or result.get("instance")
    )
    if not isinstance(target_id, str) or not isinstance(token, str) or not token:
        return  # No exact ownership proof: never manufacture a broad cleanup request.
    selected = engine.platform
    _MCP_STARTED_TARGETS[(id(selected), target_id, token)] = _McpStartedTarget(
        target=TargetRef(selected.name, target_id),
        instance_token=token,
        expected_pid=result.get("pid") if isinstance(result.get("pid"), int) else None,
        config=engine.config.model_copy(deep=True),
        platform=selected,
    )


def _forget_mcp_targets(engine: Engine, target_ids: list[Any]) -> None:
    for key, started in list(_MCP_STARTED_TARGETS.items()):
        if started.platform is engine.platform and started.target.target_id in target_ids:
            del _MCP_STARTED_TARGETS[key]


def cleanup_mcp_emulators(
    cache_dir: str | Path | None = None,
    *,
    platform: Any | None = None,
    lease_registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Stop only exact owned boots, through their original Engine strategy.

    Optional arguments filter the historical cleanup surface; they never substitute another
    platform or configuration for a tracked boot. Failed cleanup remains retryable at atexit.
    """
    from . import leases

    stopped: list[str] = []
    preserved: list[str] = []
    caller = leases.owner_caller(leases.resolve_owner()) or {}
    caller_process = (caller.get("pid"), caller.get("started"))
    for key, started in list(_MCP_STARTED_TARGETS.items()):
        if platform is not None and started.platform is not platform:
            continue
        if cache_dir is not None and Path(started.config.cache.dir) != Path(cache_dir):
            continue
        registry = started.config.lease.registry_dir or started.config.cache.dir
        if lease_registry_dir is not None and Path(registry) != Path(lease_registry_dir):
            continue
        ser = started.target.target_id
        lease = leases.read_lease(registry, started.target)
        lease_process = (
            lease.get("owner_pid") if lease else None,
            lease.get("owner_started") if lease else None,
        )
        if lease is not None and (
            leases.pending_handoff(lease)
            or (lease_process != (None, None) and lease_process != caller_process)
        ):
            # The orchestrator may have handed the running emulator to a child. Its lifecycle
            # safety net must not tear down a device that now has a different live agent.
            preserved.append(ser)
            del _MCP_STARTED_TARGETS[key]
            continue
        try:
            cleanup_engine = Engine(started.config, platform=started.platform)
            out = cleanup_engine.virtual_target_stop_instance(
                started.instance_token,
                expected_pid=started.expected_pid,
                owner=leases.resolve_owner(),
                requested_by="mcp-exit-cleanup",
            )
        except Exception:
            preserved.append(ser)
            continue
        stopped.extend(out.get("stopped_target_ids", []))
        preserved.extend(out.get("preserved_target_ids", []))
        del _MCP_STARTED_TARGETS[key]
    return {
        "ok": True,
        "action": "mcp-emulator-cleanup",
        "stopped": stopped,
        "preserved": preserved,
    }


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
        journal_args = dict(args_in)
        phase_done = args_in.pop("phase_done", None)
        expected_error_code = args_in.pop("expect_error", None)
        annotation_warnings: list[dict[str, Any]] = []
        invocation_id = uuid.uuid4().hex
        started_at = time.monotonic()
        payload: Any = None

        # One MCP tool call is one caller turn. Keep it open through folded waits and response
        # decoration so the report can compare the fingerprint actually returned to the caller.
        engine.open_caller_turn()

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
                    platform=engine.platform.name,
                    source="mcp",
                    cmd=name,
                    args=journal_args,
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
            # The long-lived MCP Engine may already cache a Device from a previous tool call.
            # Revalidate and fence that lease before any new dispatch can touch it.
            if name not in _LEASE_FREE_TOOLS:
                engine.begin_device_use()
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
                invocation_id=invocation_id,
                duration_ms=(time.monotonic() - started_at) * 1000.0,
            )
            # Trim the folded observation the same way the CLI does. Applied here, at the one
            # boundary every tool returns through, rather than at ~40 `_dump` sites — and via the
            # shared helper, because these two surfaces had already drifted once: MCP was
            # returning every field of every element on every action while the CLI trimmed.
            if isinstance(payload, dict) and name in _OBSERVATION_TOOL_NAMES:
                spec = args_in.get("observe_fields")
                if spec is None:
                    spec = getattr(engine.config.output, "observation_fields", None)
                meta_spec = args_in.get("observe_meta")
                if meta_spec is None:
                    meta_spec = getattr(engine.config.output, "observation_meta", None)
                view = Projection.for_observation(
                    spec, meta=meta_spec, fmt=OutputFormat.json
                )
                payload = trim_observation_payload(payload, view, fmt=OutputFormat.json)
            if annotation_warnings and isinstance(payload, dict):
                payload["annotation_warnings"] = annotation_warnings
            text = json.dumps(payload, ensure_ascii=False)
        except AuaError as err:
            error = err.to_dict().get("error")
            journal_call(ok=False, error=error if isinstance(error, dict) else None)
            text = json.dumps(err.to_dict(), ensure_ascii=False)
            engine.close_caller_turn()
            return [types.TextContent(type="text", text=text)]
        except Exception as err:
            journal_call(ok=False, error={"code": "error", "message": str(err)})
            engine.close_caller_turn()
            raise
        finally:
            engine.release_device_use()
        from .coaching import emitted_fingerprint

        engine.close_caller_turn(emitted_fingerprint(payload))
        journal_call(ok=not (isinstance(payload, dict) and payload.get("ok") is False))
        blocks: list[types.ContentBlock] = [types.TextContent(type="text", text=text)]
        image = _image_block(name, payload)
        if image is not None:
            blocks.append(image)
        return blocks

    return server


def build_default_engine(config: Config | None = None) -> Engine:
    """Build an :class:`Engine` from an effective config (device connects lazily).

    ``aua mcp`` has already resolved global ``--config``/``--profile``/CLI overrides.  Accepting
    that object here keeps the stdio server on the same effective configuration instead of
    silently doing a second, context-free discovery pass.
    """
    return Engine(config if config is not None else load_config())


def run_stdio(config: Config | None = None) -> None:
    """Run the MCP server over stdio — the entry point used by ``aua mcp``."""
    import atexit
    import contextlib

    import anyio
    from mcp.server.stdio import stdio_server

    engine = build_default_engine(config)
    server = build_server(engine)
    # If the MCP client disconnects without emulator_stop, tear down what we started.
    atexit.register(
        cleanup_mcp_emulators,
        engine.config.cache.dir,
        lease_registry_dir=engine.config.lease.registry_dir,
    )

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
            cleanup_mcp_emulators(
                engine.config.cache.dir,
                platform=engine.platform,
                lease_registry_dir=engine.config.lease.registry_dir,
            )


__all__ = [
    "SERVER_NAME",
    "build_default_engine",
    "build_server",
    "cleanup_mcp_emulators",
    "run_stdio",
]
