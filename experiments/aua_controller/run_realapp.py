"""Drive one unauthored goal on a real application, then judge it from the frames.

The fixture runner needs an authored contract to accept ``session_finish``. A real
application usually has none, so the model's completion claim is accepted as a *signal*
that stops the loop, and a separate bounded judgement (``judgement.py``) decides the
outcome from the observed frames. Screens seen along the way can be named for the map.

Everything the model reads is compacted (``compaction.py``) after the hosted privacy
projection. Raw evidence stays untouched under ``<output>/controller/evidence``.

The runner is application-agnostic: goal, package and optional setup flow come from the
caller. Paid model calls happen only through the injected sender; the result names the
model, provider and reported cost of every tier that ran.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.compaction import FrameCompactor
from experiments.aua_controller.hosted import BACKENDS, validate_endpoint, validate_request_config
from experiments.aua_controller.hosted_projection import hosted_model_view
from experiments.aua_controller.judgement import (
    MAX_IMAGES,
    Decider,
    ScreenNamer,
    encode_image,
    frame_fingerprint,
    image_frame_sample,
    judge_outcome_votes,
    judged_frame_sample,
    screenshot_for,
    screenshot_index,
    summarize_route,
)
from experiments.aua_controller.run_live import (
    COMPACT_SYSTEM,
    SYSTEM,
    RunError,
    _error_text,
    compact_schema,
    offered_schema,
    tool_result,
)
from experiments.aua_controller.session_state import observation_frame
from experiments.aua_controller.transport import resilient_request, retryable_http_status

from android_ui_analyser.engine_support import _parse_await_terms
from android_ui_analyser.errors import UsageError

FORMAT = "aua-realapp-run-v1"
CONTROLLER_TOOLS = (
    "analyze_screen", "tap_and_analyze", "long_press_and_analyze", "input_and_analyze",
    # scroll sits beside swipe because the engine refuses to save a flow containing a raw swipe:
    # "swipe capture omits coordinates/container/percentage; author `scroll: up` by hand instead -
    # a scroll names its container and replays, where a raw swipe is positional" (flows.py). With
    # swipe as the only way to move the screen, a controller could only produce routes this engine
    # would not save, so any journey needing to scroll could never leave a replayable flow behind.
    "scroll_and_analyze",
    "swipe_and_analyze", "back_gesture_and_analyze", "wait_and_analyze",
    # A contract that names a deeplink cannot be judged by a controller with no way to follow one.
    # Measured 2026-09-15: a tools scenario asserting "the tool's deeplink opens the same launchpad
    # as the tap route" left three criteria unverifiable, and the judge said why in its own words --
    # "there was no OS intent/URL-launch tool available, so the URI was never triggered". The
    # engine has always exposed `open_link`; only the controller's list left it out.
    #
    # `pin_package` defaults true, so the VIEW intent stays on the app under test rather than
    # wandering into a chooser or another product.
    "open_link_and_analyze",
    "key_and_analyze", "session_progress", "session_finish",
)
# Real-app-only additions to compact-v1. Keep them here rather than widening run_live's fixed
# comparison profile: these close product-scenario capability gaps without changing an existing
# model benchmark. Long-press accepts only a fresh AUA id; edge-back accepts no geometry at all.
REALAPP_COMPACT_PROPERTIES = {
    "long_press_and_analyze": frozenset({"id"}),
    "back_gesture_and_analyze": frozenset(),
    # Scroll takes a direction and nothing else. `percent` is deliberately withheld: a scroll that
    # names only its direction replays against whatever the container is, while a baked percentage
    # is the same positional trap that makes a raw swipe unsaveable -- and the promoter has already
    # refused flows carrying a fixed scroll distance (the same carousel needed 7, then 12, then 17).
    "scroll_and_analyze": frozenset({"direction"}),
    # The URI is the whole action: which link was followed is the evidence a deeplink bullet is
    # judged on. `package`/`prefer` stay out -- they are routing detail, and a flow that bakes a
    # chooser preference is describing this host rather than the journey.
    "open_link_and_analyze": frozenset({"uri"}),
}
#: The one extra tool a contract-driven run needs. A checkpoint completes only on fresh
#: assertion proof, so without a way to assert, a loaded contract can never be satisfied and
#: every verdict falls back to a model reading frames - the weaker answer, from a run that was
#: given the stronger one. Added only when there is a contract: a run with no checkpoints has
#: nothing to assert against, and one more tool in the list is one more way to spend a step.
CONTRACT_TOOL = "expect_and_analyze"
#: One selector and one predicate is the whole vocabulary a checkpoint assertion needs, and
#: every extra argument is another one a small model can get wrong.
CONTRACT_TOOL_PROPERTIES = ("rid", "text", "desc", "exists", "absent", "text_contains")
CONTROLLER_CAPABILITIES = frozenset({
    "network", "wall-clock-wait", "async-ui-wait", "app-lifecycle",
})
WALL_CLOCK_WAIT_TOOL = "wait_uninterrupted_620_seconds"
WALL_CLOCK_WAIT_SECONDS = 620
# Boundary clock reads, detached-job dispatch and the first/last status poll sit outside the
# duration itself. This timeout applies only to the harness-owned wait tool; model calls and
# ordinary UI tools retain the normal request timeout.
WALL_CLOCK_WAIT_TOOL_TIMEOUT_S = WALL_CLOCK_WAIT_SECONDS + 120
ASYNC_UI_WAIT_TOOL = "wait_for_ui_condition"
ASYNC_UI_WAIT_DEFAULT_SECONDS = 120
ASYNC_UI_WAIT_MAX_SECONDS = 900
ASYNC_UI_WAIT_GRACE_SECONDS = 60
ASYNC_UI_WAIT_TOOL_TIMEOUT_S = ASYNC_UI_WAIT_MAX_SECONDS + ASYNC_UI_WAIT_GRACE_SECONDS
ASYNC_UI_WAIT_STATUS_POLL_SECONDS = 5.0
ASYNC_UI_WAIT_STATUS_CALL_SECONDS = 15.0
FINISH_OUTCOMES = ("achieved", "already_satisfied", "blocked", "not_achievable")
REALAPP_SYSTEM = """
Real-application mode. There is no authored checklist; you decide when the goal is met.
If the initial observation already shows the requested end state, call session_finish at once
with outcome "already_satisfied". After you observe the requested end state, call session_finish
once with outcome "achieved" and a one-line note; do not spend steps collecting extra evidence.
If a login wall, permission prompt, network failure or missing precondition stops you, call
session_finish with outcome "blocked" and say what blocked you. If the app cannot do what is
asked, use "not_achievable". A separate reviewer verifies your claim from the screens.
"""
CONTRACT_SYSTEM = """
This run adds expect_and_analyze to the compact-v1 subset: the note above about expect being
unavailable does not apply here, because proving an assertion is the whole job.
This run has an authored contract, so you are not the one who decides it is met. session_progress
names the current checkpoint and the assertions it needs. Drive the app to that state, then prove
each assertion on the live screen with expect_and_analyze - a checkpoint completes only on fresh
proof, and nothing you say completes one. When session_progress reports every checkpoint complete,
call session_finish with outcome "achieved". If you cannot reach a checkpoint, finish with
"blocked" and say which assertion you could not prove.
"""


KNOWLEDGE_SHOWN = 5
_BARE_ELEMENT_UUID = re.compile(r"^[0-9a-f]{32}$")


def controller_tool_timeouts(capabilities: Sequence[str]) -> dict[str, float]:
    """Exact harness-owned tools allowed to outlive an ordinary request window."""

    requested = set(capabilities)
    timeouts: dict[str, float] = {}
    if "wall-clock-wait" in requested:
        timeouts[WALL_CLOCK_WAIT_TOOL] = WALL_CLOCK_WAIT_TOOL_TIMEOUT_S
    if "async-ui-wait" in requested:
        timeouts[ASYNC_UI_WAIT_TOOL] = ASYNC_UI_WAIT_TOOL_TIMEOUT_S
    return timeouts


def async_ui_wait_spec(arguments: Mapping[str, Any]) -> tuple[str, int]:
    """Build one positive-arrival/negative-pending predicate from bounded model input."""

    anchor = arguments.get("anchor")
    pending = arguments.get("pending_text")
    timeout_seconds = arguments.get("timeout_seconds", ASYNC_UI_WAIT_DEFAULT_SECONDS)
    if (
        not isinstance(anchor, str)
        or not 1 <= len(anchor.strip()) <= 160
        or any(ord(char) < 32 for char in anchor)
    ):
        raise RunError("async UI wait anchor must be a 1-160 character semantic selector")
    if not isinstance(pending, str) or not 1 <= len(pending.strip()) <= 120:
        raise RunError("async UI wait pending_text must be 1-120 characters")
    if any(ord(char) < 32 for char in pending):
        raise RunError("async UI wait pending_text cannot contain control characters")
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= ASYNC_UI_WAIT_MAX_SECONDS
    ):
        raise RunError(
            f"async UI wait timeout_seconds must be an integer from 1 to {ASYNC_UI_WAIT_MAX_SECONDS}"
        )
    try:
        terms = _parse_await_terms(anchor, require_positive=True)
    except UsageError as exc:
        raise RunError("async UI wait anchor is not a valid semantic selector") from exc
    if len(terms) != 1 or terms[0].negated or terms[0].by not in {"rid", "text", "desc"}:
        raise RunError("async UI wait anchor must be exactly one positive rid:, text:, or desc: selector")
    if terms[0].by in {"text", "desc"} and terms[0].value.casefold() == pending.strip().casefold():
        raise RunError("async UI wait anchor and pending_text cannot describe the same label")

    def escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace(",", "\\,")

    positive = f"{terms[0].by}:{escape(terms[0].value)}"
    negative = f"!text:{escape(pending.strip())}"
    predicate = f"{positive},{negative}"
    _parse_await_terms(predicate, require_positive=True)
    return predicate, timeout_seconds


async def run_async_ui_wait(
    *,
    call: Callable[[str, dict[str, Any], str], Awaitable[dict[str, Any]]],
    arguments: Mapping[str, Any],
    result: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Run one durable AUA await while this controller spends no additional model turns."""

    predicate, timeout_seconds = async_ui_wait_spec(arguments)
    started_job = await call(
        "job_start",
        {
            "operation": "await",
            "predicate": predicate,
            "timeout_ms": timeout_seconds * 1_000,
            "poll_ms": 500,
            "observe": True,
        },
        "harness-ui-wait-start",
    )
    job_id = started_job.get("job_id")
    if started_job.get("ok") is not True or not isinstance(job_id, str) or not job_id:
        raise RunError("AUA could not detach the UI wait: " + json.dumps(started_job)[:500])
    wait_started_at = datetime.now().astimezone()
    receipt: dict[str, Any] = {
        "job_id": job_id,
        "operation": "await",
        "predicate": predicate,
        "requested_seconds": timeout_seconds,
        "status": started_job.get("status"),
        "started_at": wait_started_at.isoformat(),
        "deadline_at": (wait_started_at + timedelta(seconds=timeout_seconds)).isoformat(),
        "model_calls_during_wait": 0,
        "reconnect": {"tool": "job_status", "arguments": {"job_id": job_id}},
    }
    result["deferred_waits"].append(receipt)

    def persist() -> None:
        (output / "deferred-waits.json").write_text(
            json.dumps(result["deferred_waits"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    persist()
    host_deadline = time.monotonic() + timeout_seconds + ASYNC_UI_WAIT_GRACE_SECONDS
    status = started_job
    try:
        while status.get("terminal") is not True:
            remaining = host_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("durable AUA UI wait did not reach a terminal status")
            status = await asyncio.wait_for(
                call("job_status", {"job_id": job_id}, "harness-ui-wait-status"),
                timeout=min(ASYNC_UI_WAIT_STATUS_CALL_SECONDS, max(0.001, remaining)),
            )
            receipt["status"] = status.get("status")
            receipt["progress_percent"] = status.get("progress_percent")
            persist()
            if status.get("terminal") is not True:
                remaining = host_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("durable AUA UI wait did not reach a terminal status")
                await asyncio.sleep(min(ASYNC_UI_WAIT_STATUS_POLL_SECONDS, remaining))
    except BaseException:
        receipt["cancel_requested"] = True
        try:
            cancelled = await asyncio.shield(
                asyncio.wait_for(
                    call(
                        "job_cancel",
                        {"job_id": job_id, "wait_ms": 10_000},
                        "harness-ui-wait-cancel",
                    ),
                    timeout=ASYNC_UI_WAIT_STATUS_CALL_SECONDS,
                )
            )
            receipt["status"] = cancelled.get("status")
            receipt["cancelled"] = cancelled.get("status") == "cancelled"
        except BaseException as cleanup_error:  # preserve the original timeout/cancellation
            receipt["cancel_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        persist()
        raise
    if status.get("status") != "succeeded":
        raise RunError("AUA UI wait job failed: " + json.dumps(status)[:500])
    completed = status.get("result")
    if not isinstance(completed, dict):
        raise RunError("AUA UI wait job returned no structured result")
    receipt.update({
        "status": "succeeded",
        "finished_at": datetime.now().astimezone().isoformat(),
        "run_ok": status.get("run_ok"),
        "await_outcome": completed.get("await_outcome"),
        "capture_evidence": completed.get("capture_evidence"),
    })
    persist()
    return {
        **completed,
        "deferred_job_id": job_id,
        "harness_owned_wait": True,
        "model_calls_during_wait": 0,
    }


def normalize_element_id_argument(arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Restore the unambiguous ``el:`` namespace a hosted model occasionally drops."""

    value = arguments.get("id")
    if not isinstance(value, str) or _BARE_ELEMENT_UUID.fullmatch(value) is None:
        return arguments, False
    return {**arguments, "id": f"el:{value}"}, True


def device_epoch_seconds(result: Mapping[str, Any]) -> int | None:
    """Read an integer epoch from AUA's bounded read-only shell result."""
    try:
        return int(str(result.get("stdout") or "").strip())
    except (TypeError, ValueError):
        return None


def judge_intermediate_frame_limit(requested: int, *, vision: bool) -> int:
    """Retain compact text evidence independently of the rendered-image window."""

    return requested


def host_knowledge(start: dict[str, Any]) -> list[dict[str, Any]]:
    """The facts ``session_start`` ranked against the goal: recorded advice, never verified state."""
    items = start.get("relevant_knowledge")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and str(item.get("text") or "").strip()][:KNOWLEDGE_SHOWN]


def goal_prompt(
    goal: str,
    knowledge: list[dict[str, Any]],
    setup_notes: Sequence[str] = (),
    setup_facts: Sequence[str] = (),
    authored_context: str | None = None,
) -> str:
    """The first user message: the goal, what the host already knows, what setup did not finish.

    Without this the model re-derives facts the store already holds (where a setting lives, that a
    fresh install overwrites it, the route to it). Ids stay out: they are host bookkeeping.

    *setup_notes* carries any setup flow that diverged. The alternative - saying nothing - makes the
    model start from a screen the harness expected to be somewhere else, with no idea which part of
    the precondition is missing.
    """
    lines = ["Goal: " + goal]
    if setup_facts:
        lines += [
            "",
            "Harness-owned setup already completed and verified before this controller turn. "
            "Treat these as established facts; do not reopen host-only setup screens or block "
            "because the compact controller has no infrastructure tools:",
        ]
        lines += [f"- {fact}" for fact in setup_facts]
    if authored_context and authored_context.strip():
        lines += [
            "",
            "Authored precondition context follows for classification and constraints. It is not "
            "a list of additional session phases; host-owned checks named here are already covered "
            "by the verified setup facts above:",
            authored_context.strip(),
        ]
    if setup_notes:
        lines += ["", "Setup did not finish as written. Establish the rest yourself before judging, "
                      "and say so if you cannot:"]
        lines += [f"- {note}" for note in setup_notes]
    if knowledge:
        lines += ["", "Recorded knowledge about this app that matches the goal. It is advice with provenance, "
                      "possibly stale: prefer it over rediscovering, but confirm on screen before you rely on it."]
        for item in knowledge:
            head = str(item.get("kind") or "fact")
            if item.get("name"):
                head += " " + str(item["name"])
            lines.append(f"- [{head}] {str(item['text']).strip()}")
    return "\n".join(lines)


def finish_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": list(FINISH_OUTCOMES)},
            "note": {"type": "string", "maxLength": 240},
        },
        "required": ["outcome"],
        "additionalProperties": False,
    }


def contract_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The assert tool, trimmed to one selector and one predicate.

    Built here rather than through `compact_schema`, because compact-v1 is `run_live`'s
    benchmark profile and deliberately offers no `expect`. Widening it there would change what
    every benchmark run is handed, to serve a contract those runs do not have.
    """
    properties = {
        key: value
        for key, value in (schema.get("properties") or {}).items()
        if key in CONTRACT_TOOL_PROPERTIES
    }
    missing = set(schema.get("required") or ()) - properties.keys()
    if missing:
        raise RunError(f"{CONTRACT_TOOL} now requires {', '.join(sorted(missing))}")
    return {"type": "object", "properties": properties, "additionalProperties": False}


def realapp_compact_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Trim real-app-only actions without widening the fixed compact-v1 benchmark."""

    allowed = REALAPP_COMPACT_PROPERTIES.get(name)
    if allowed is None:
        return compact_schema(name, schema)
    compact = offered_schema(name, schema)
    compact["properties"] = {
        key: value for key, value in compact["properties"].items() if key in allowed
    }
    if set(compact.get("required", ())) - compact["properties"].keys():
        raise RunError(f"{name} requires an argument outside the real-app compact schema")
    return compact


def realapp_tools(
    schemas: dict[str, dict[str, Any]],
    *,
    contract: bool = False,
    capabilities: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """compact-v1 tools, with session_finish carrying the model's outcome claim."""
    tools = []
    names = CONTROLLER_TOOLS if contract else tuple(
        name for name in CONTROLLER_TOOLS if name != "session_progress"
    )
    for name in (*names, *((CONTRACT_TOOL,) if contract else ())):
        if name not in schemas:
            raise RunError(f"AUA MCP does not offer {name}")
        if name == "session_finish":
            parameters = finish_schema()
        elif name == CONTRACT_TOOL:
            parameters = contract_tool_schema(schemas[name])
        else:
            parameters = realapp_compact_schema(name, schemas[name])
        description = str(schemas[name].get("description") or "")[:300]
        if name == "session_finish":
            description = "Claim the goal is finished (or blocked) with an outcome and a short note."
        elif name == "long_press_and_analyze":
            description = (
                "Long-press one element using its fresh id from the current AUA observation, "
                "then return the resulting screen."
            )
        elif name == "back_gesture_and_analyze":
            description = (
                "Perform Android's left-edge back gesture and return the resulting screen; "
                "coordinates are intentionally unavailable."
            )
        tools.append({"type": "function", "function": {"name": name, "description": description,
                                                       "parameters": parameters}})
    requested = set(capabilities)
    unknown = requested - CONTROLLER_CAPABILITIES
    if unknown:
        raise RunError("unknown controller capabilities: " + ", ".join(sorted(unknown)))
    if "network" in requested:
        for name in ("network_offline", "network_restore"):
            if name not in schemas:
                raise RunError(f"AUA MCP does not offer {name}")
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(schemas[name].get("description") or "")[:300],
                    "parameters": offered_schema(name, schemas[name]),
                },
            })
    if "wall-clock-wait" in requested:
        tools.append({
            "type": "function",
            "function": {
                "name": WALL_CLOCK_WAIT_TOOL,
                "description": (
                    "Wait exactly 620 uninterrupted seconds while the app is already backgrounded; "
                    "records host-monotonic and device-clock boundary evidence."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            })
    if "async-ui-wait" in requested:
        for name in ("job_start", "job_status", "job_cancel"):
            if name not in schemas:
                raise RunError(f"AUA MCP does not offer {name}")
        tools.append({
            "type": "function",
            "function": {
                "name": ASYNC_UI_WAIT_TOOL,
                "description": (
                    "Wait once for a positive semantic AUA anchor to appear while temporary "
                    "pending text disappears. The harness owns one durable AUA job and polls it "
                    "without model turns; do not call this tool again while it runs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 160,
                            "description": (
                                "Exactly one positive rid:, text:, or desc: selector; escape a "
                                "literal comma as \\,."
                            ),
                        },
                        "pending_text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 120,
                            "description": "Temporary visible text that must disappear.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": ASYNC_UI_WAIT_MAX_SECONDS,
                            "default": ASYNC_UI_WAIT_DEFAULT_SECONDS,
                        },
                    },
                    "required": ["anchor", "pending_text"],
                    "additionalProperties": False,
                },
            },
        })
    if "app-lifecycle" in requested:
        for name, description in (
            ("app_force_stop", "Force-stop only the package under test without clearing its data."),
            ("app_relaunch_and_analyze", "Relaunch the pinned package/activity and return its screen."),
        ):
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            })
    return tools


def goal_progress_of(progress: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The counts inside a `session_progress` reply, whichever level they arrive at."""
    if not isinstance(progress, Mapping) or progress.get("ok") is False:
        return None
    inner = progress.get("goal_progress")
    if isinstance(inner, Mapping):
        return inner
    return progress if "total" in progress else None


def unmet_checkpoint(progress: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The checkpoint a contract run stopped on, so a reader knows what to look at.

    An authored contract can be unsatisfiable rather than unsatisfied - an assertion naming a
    selector the app never publishes can never match, no matter how well the run went. Measured
    on 2026-09-14: a contract asserting `desc:"Create, New"` against a label the accessibility
    tree carries as *text* left both checkpoints open, and the run came back `unverified` with
    nothing pointing at the reason. This does not judge which it was; it names the checkpoint
    and what it asked for, which is the difference between a re-run and a fix.
    """
    counts = goal_progress_of(progress)
    if counts is None or counts.get("done") is True:
        return None
    current = counts.get("current")
    if not isinstance(current, Mapping):
        return None
    return {
        "id": current.get("id"),
        "objective": current.get("objective"),
        "completed": counts.get("completed"),
        "total": counts.get("total"),
        "hint": (
            "no checkpoint completed, so check the contract before the app: an assertion whose "
            "selector the app never publishes can never match. Compare it against the elements "
            "in the captured observations."
        ) if counts.get("completed") == 0 else
        "the run stopped part-way through the contract; this checkpoint was still open.",
    }


def contract_satisfied(progress: Mapping[str, Any] | None) -> bool:
    """True only when AUA itself reports every authored checkpoint complete.

    Deliberately strict about shape: a missing or malformed progress block means the contract
    was not proven, and the judge answers instead. Reading "probably fine" out of a payload we
    did not understand is how a harness reports a verdict nobody produced.
    """
    counts = goal_progress_of(progress)
    if counts is None:
        return False
    total, completed = counts.get("total"), counts.get("completed")
    if not isinstance(total, int) or not isinstance(completed, int) or total <= 0:
        return False
    return completed == total and counts.get("done") is True


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def export_primary_flow(call, output: Path) -> dict[str, Any]:
    """Export only a clean controller action suffix, never write shared flow memory."""
    action_tools = {
        "tap_and_analyze", "long_press_and_analyze", "input_and_analyze",
        "scroll_and_analyze", "swipe_and_analyze", "back_gesture_and_analyze",
        "key_and_analyze", "open_link_and_analyze", "app_force_stop",
        "app_relaunch_and_analyze",
    }
    read_tools = {"analyze_screen", "wait_and_analyze", "session_progress", "session_finish"}
    path = output / "controller/tool-calls.jsonl"
    if not path.is_file():
        raise RunError("primary flow requires the controller action journal")
    count = 0
    for entry in _load_jsonl(path):
        if not isinstance(entry, dict):
            raise RunError("primary flow action journal is malformed")
        if entry.get("dispatch_started") and entry.get("executed") is not True:
            raise RunError("primary flow has an action with unknown execution outcome")
        if entry.get("executed") is not True:
            continue
        name = entry.get("tool")
        if name in read_tools:
            continue
        if name not in action_tools:
            raise RunError("primary flow contains an unsupported controller action")
        action = entry.get("result")
        if (not isinstance(action, dict) or action.get("ok") is not True
                or action.get("error") or action.get("mcp_is_error")):
            raise RunError("primary flow contains a failed or unverified controller action")
        count += 1
    if count == 0:
        return {"route_action_count": 0, "flow_not_applicable_reason": "observation_only_no_actions"}
    preview = await call("flow_save", {"name": "controller-route", "last": count, "save": False}, "evidence")
    scope = preview.get("scope") or {}
    body = preview.get("preview")
    if (preview.get("ok") is not True or preview.get("steps") != count
            or scope.get("requested_last") != count or scope.get("selected") != count
            or scope.get("boundary_omitted") not in (0, None)
            or not isinstance(body, str) or not body.strip()):
        raise RunError("primary flow preview could not prove the exact clean controller action suffix")
    (output / "flow.yaml").write_text(body.rstrip() + "\n", encoding="utf-8")
    return {"route_action_count": count, "primary_flow": "flow.yaml"}


def controller_cost(report: dict[str, Any]) -> float:
    accounting = report.get("cost_accounting") or {}
    if isinstance(accounting.get("reported_usd"), (int, float)):
        return float(accounting["reported_usd"])
    return sum(float((usage or {}).get("cost") or 0) for usage in report.get("usage", []))


def verdict_markdown(result: dict[str, Any]) -> str:
    verdict = result["verdict"]
    lines = [f"# {verdict['verdict'].upper()}  ·  {result['goal']}", ""]
    stop = (result.get("controller") or {}).get("stop_reason")
    lines.append(f"Oracle: `{verdict['oracle']}` · verified: {verdict['verified']} · controller stop: `{stop}`")
    if result.get("error"):
        lines.append(f"Run error: {result['error']}")
    for warning in result.get("warnings") or []:
        # A run that adapted around a stale precondition is not the same run as one that did
        # not, and a reader who cannot see the difference will read the verdict as stronger
        # than it is.
        lines.append(f"Adapted: {warning}")
    claim = result.get("claim")
    if claim:
        lines.append(f"Controller claim: **{claim.get('outcome')}** — {claim.get('note') or ''}".rstrip(" —"))
    lines.append("")
    for reason in verdict.get("reasons", []):
        lines.append(f"- {reason}")
    cost = result["cost"]
    lines += ["", "| tier | model | provider | reported USD |", "|---|---|---|---|"]
    for tier in ("controller", "judge", "map"):
        entry = cost.get(tier)
        if entry:
            lines.append(f"| {tier} | {entry.get('model')} | {entry.get('provider')} | {entry.get('usd'):.6f} |")
    lines.append(f"| total | | | {cost['total_usd']:.6f} |")
    screens = result.get("screens") or []
    if screens:
        lines += ["", "## Screens", ""]
        for screen in screens:
            lines.append(f"- `{screen['logical_name']}` ({screen['kind']}): {screen['purpose']}")
    return "\n".join(lines) + "\n"


def _target_start_failed(payload: dict[str, Any]) -> bool:
    """Return true only for lease/provision failures that are safe to retry by waiting."""
    error = payload.get("error")
    code = payload.get("code")
    if isinstance(error, dict):
        code = error.get("code") or code
    normalized = str(code or "").strip().casefold()
    if normalized in {
        "device_unavailable",
        "device_leased",
        "lease_unavailable",
        "virtual_target_start_failed",
        "virtual_target_provision_failed",
        "emulator_start_failed",
    }:
        return True
    if normalized != "device":
        return False
    text = json.dumps(payload, ensure_ascii=False, default=str).casefold()
    return any(word in text for word in ("lease", "emulator", "virtual target", "boot", "disk"))


def _mark_cleanup_failure(result: dict[str, Any], message: str) -> None:
    """Never leave a successful verdict on a run whose evidence or lease cleanup failed."""
    prior = str(result.get("error") or "").strip()
    result["error"] = f"{prior}; {message}".strip("; ")
    verdict = result.get("verdict")
    if isinstance(verdict, dict):
        reasons = [str(item) for item in (verdict.get("reasons") or [])]
        if message not in reasons:
            reasons.append(message)
        verdict["reasons"] = reasons
        verdict["cleanup_verified"] = False
        if verdict.get("verdict") in {"pass", "pass_with_warning"}:
            verdict["verdict"] = "unverified"
            verdict["verified"] = False
    else:
        result["verdict"] = {
            "oracle": "none",
            "verified": False,
            "verdict": "unverified",
            "reasons": [message],
            "cleanup_verified": False,
        }


def _retryable_recording_stop_failure(value: Any) -> bool:
    """A recorder stop is safe to retry after the Android transport only timed out reading."""

    text = json.dumps(value, ensure_ascii=False, default=str).casefold()
    return "adb read timeout" in text or "adb read timed out" in text


SETUP_FLOW_RESUMES = 4
"""How often a setup flow may re-issue a wait the ceiling cut short.

`perf.max_wait_ms` ends every observation wait at 5s, so a flow that writes
`timeout_ms: 60000` gets 5s and reports `wait_timeout` whether the screen is late or
genuinely absent. The remedy AUA documents is to ask again, not to pass a bigger number the
ceiling ignores. Four resumes buy a precondition about 25s without letting a wrong flow spin:
a setup flow is the fast path, not the oracle, and a run abandoned because a cold start took
seven seconds is a verdict about the harness rather than about the product.
"""



async def capture_setup_proof(
    call: Callable[[str, dict[str, Any], str], Awaitable[dict[str, Any]]],
    setup_proof: tuple[str, str, str],
    *, regex: bool = False, since: str | None = None,
) -> dict[str, Any] | None:
    """Whether a caller-supplied pattern is in the device log, read through this session.

    A precondition a caller must be able to *prove* rather than infer from the screen. The
    pattern and the value it stands for belong to the caller: this engine is app-agnostic and
    must not know what any product calls its tiers or entitlements.

    Read over the session's own tool channel, never a second `aua` process. A subprocess derives
    its own worker scope and is refused with `device_leased` against the very device this session
    holds -- which is why an earlier subprocess version worked when a scenario ran alone and
    returned nothing under the panel's parallel workers, silently failing a row whose login had
    succeeded.

    The proving line is written once, moments after the step that causes it, so a single look can
    miss it while the response is still in flight; this polls briefly and stops the moment it is
    proved. Only the boolean and the caller's label come back. **The matching line is never
    returned or stored** -- a body carrying this kind of field usually carries the account address
    beside it, and setup evidence is copied into published reports.

    Returns ``None`` when the log cannot be read at all, so the caller records nothing and keeps
    whatever it already does about an unproved precondition. A capture that cannot run must not be
    mistaken for a precondition that failed, nor invent one that held.
    """
    _, pattern, value = setup_proof
    matcher = re.compile(pattern if regex else re.escape(pattern))
    deadline = time.monotonic() + 12
    while True:
        try:
            arguments: dict[str, Any] = {"grep": matcher.pattern, "lines": 20}
            if since is not None:
                arguments["since"] = since
            payload = await call("logcat_dump", arguments, "setup")
        except Exception:
            return None
        lines = (payload.get("lines") if isinstance(payload, dict)
                 and payload.get("ok") is not False else None)
        if not isinstance(lines, list):
            return None
        matches = [match for line in lines for match in matcher.finditer(str(line))]
        matched = bool(matches)
        if matches and "value" in matcher.groupindex:
            matched = matches[-1].group("value") == value
            # A later negative state overrides an earlier positive one. Never return the
            # captured text: only the caller-supplied expected label is safe to publish.
            return {"verified": matched, "actual": value if matched else None,
                    "source": "logcat"}
        if matched or time.monotonic() >= deadline:
            return {"verified": matched, "actual": value if matched else None,
                    "source": "logcat"}
        await asyncio.sleep(2)


async def replay_setup_flow(
    call: Callable[[str, dict[str, Any], str], Awaitable[dict[str, Any]]],
    flow_yaml: str,
    flow_params: Mapping[str, str] | None,
    *,
    actor: str = "setup",
) -> tuple[dict[str, Any], int]:
    """Replay one setup flow, re-issuing a wait that ran out of budget rather than screen.

    Only `wait_timeout` resumes, and only from the step the engine names: every other
    divergence - a missing element, an unsafe step, a failed assertion - is information about
    the flow or the app, and repeating it just spends a device on the same answer.
    """
    arguments: dict[str, Any] = {"yaml": flow_yaml, "assist": False}
    if flow_params:
        arguments["params"] = {str(k): str(v) for k, v in flow_params.items()}
    flow = await call("flow_run", dict(arguments), actor)
    resumes = 0
    while (
        flow.get("ok") is not True
        and flow.get("code") == "wait_timeout"
        and isinstance(flow.get("resume_from_step"), int)
        and resumes < SETUP_FLOW_RESUMES
    ):
        resumes += 1
        flow = await call(
            "flow_run",
            {**arguments, "from_step": int(flow["resume_from_step"])},
            actor,
        )
    return flow, resumes


async def run_realapp(
    *,
    call_tool: Callable[[str, dict[str, Any]], Awaitable[Any]],
    list_tools: Callable[[], Awaitable[dict[str, dict[str, Any]]]],
    send: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    goal: str,
    session_goal: str | None = None,
    package: str,
    output: Path,
    model: str,
    request_config: dict[str, Any] | None = None,
    backend: str = "openrouter",
    controller_fallbacks: Sequence[tuple[str, dict[str, Any] | None]] = (),
    launch: bool = False,
    launch_app: bool = True,
    activity: str | None = None,
    headed: bool = False,
    fresh_app: bool = False,
    apk: str | None = None,
    grant_permissions: bool = False,
    forbidden_packages: Sequence[str] = (),
    record: bool = False,
    recording_required: bool = True,
    lease_wait_s: float = 0,
    fallback_lease_wait_s: float = 600,
    provision_target: bool = True,
    needs: Sequence[str] = (),
    controller_capabilities: Sequence[str] = (),
    flags: dict[str, str] | None = None,
    prelaunch_setup_flows: Sequence[tuple[str, dict[str, str]]] = (),
    setup_flows: Sequence[tuple[str, dict[str, str]]] = (),
    setup_proof: tuple[str, str, str] | None = None,
    setup_proof_regex: bool = False,
    save_primary_flow: bool = False,
    contract: str | None = None,
    authored_context: str | None = None,
    session_contract: str | None = None,
    vision: bool = False,
    judge_model: str | None = None,
    judge_request_config: dict[str, Any] | None = None,
    judge_fallbacks: Sequence[tuple[str, dict[str, Any] | None]] = (),
    judge: bool = True,
    judge_votes: int = 2,
    judge_frames: int = 8,
    name_screens: bool = False,
    max_named_screens: int = 8,
    max_steps: int = 24,
    time_limit_s: float = 300,
    max_tokens: int = 32768,
    max_request_bytes: int = 200_000,
    cost_limit_usd: float = 0.15,
    judge_cost_limit_usd: float = 0.10,
    judge_max_tokens: int = 8192,
    terminal_claim_limit: int = 1,
    no_progress_limit: int = 4,
    max_elements: int = 60,
    request_timeout_s: float = 90,
    existing_session_id: str | None = None,
    finish_session: bool = True,
    retain_started_target: bool = True,
    inherited_setup_facts: Sequence[str] = (),
    session_artifacts_dir: str | Path | None = None,
    record_after_host_setup: bool = False,
) -> dict[str, Any]:
    """Return the run result; also written to ``<output>/result.json`` and ``verdict.md``."""
    if backend not in BACKENDS:
        raise RunError("unknown backend")
    if fresh_app and not apk:
        raise RunError("fresh_app needs --apk: a booted AVD has no app to clear")
    if existing_session_id and (fresh_app or apk or prelaunch_setup_flows or flags):
        raise RunError(
            "an existing session cannot repeat fresh install, APK bootstrap, prelaunch flows, or flags"
        )
    if lease_wait_s < 0 or fallback_lease_wait_s < 0:
        raise RunError("lease waits must be non-negative")
    if any(not str(item).strip() for item in forbidden_packages):
        raise RunError("forbidden package names must be non-empty")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RunError("real-app output directory must be empty")
    effective_session_goal = (session_goal or goal).strip()
    if not effective_session_goal:
        raise RunError("session_goal must be non-empty when supplied")
    settings = validate_request_config(request_config or {}) if backend == "openrouter" else copy.deepcopy(request_config or {})
    judging_model = judge_model or model
    judge_settings = (
        validate_request_config(judge_request_config or request_config or {})
        if backend == "openrouter"
        else copy.deepcopy(judge_request_config or request_config or {})
    )
    started = time.monotonic()
    result: dict[str, Any] = {
        "format": FORMAT, "goal": goal, "session_goal": effective_session_goal,
        "package": package, "model": model, "backend": backend,
        "request_config": settings, "judge_model": judging_model,
        "controller_ladder": [model, *(name for name, _ in controller_fallbacks)],
        "judge_request_config": judge_settings,
        "judge_ladder": [judging_model, *(name for name, _ in judge_fallbacks)],
        "judge_fallbacks": [
            {"model": name, "request_config": copy.deepcopy(config or {})}
            for name, config in judge_fallbacks
        ],
        "session_id": None, "serial": None, "setup": [],
        "controller": None, "claim": None, "verdict": None, "screens": [], "route": None,
        "recording": None, "deferred_waits": [],
        "lifecycle": {"lease_strategy": "reuse_or_provision"},
        "cost": {"total_usd": 0.0}, "error": None, "warnings": [], "report_is_untrusted": True,
    }
    setup_log = output / "setup-calls.jsonl"

    async def call(name: str, arguments: dict[str, Any], actor: str) -> dict[str, Any]:
        tick = time.monotonic()
        record: dict[str, Any] = {"actor": actor, "tool": name, "arguments": arguments}
        try:
            decoded = tool_result(await call_tool(name, arguments))
            record["ok"] = decoded.get("ok")
            if decoded.get("ok") is not True and name != "logcat_dump":
                # A failing AUA call does not always populate `error`; keep a bounded copy of the
                # payload so the reason survives in the log instead of reading as `error: null`.
                record["payload"] = json.dumps(decoded, ensure_ascii=False, default=str)[:2000]
            return decoded
        except Exception as exc:
            record["error"] = "logcat proof unavailable" if name == "logcat_dump" else _error_text(exc)
            raise
        finally:
            record["duration_ms"] = (time.monotonic() - tick) * 1000
            with setup_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    async def stop_recording(actor: str) -> dict[str, Any]:
        """Stop/export once, retrying one known transient device read timeout."""

        attempts = 0
        while True:
            attempts += 1
            try:
                stopped = await call(
                    "screen_record_stop", {"path": str(recording_path)}, actor
                )
                retryable = _retryable_recording_stop_failure(stopped)
            except Exception as exc:
                stopped = {"ok": False, "error": _error_text(exc)}
                retryable = _retryable_recording_stop_failure(stopped)
            if stopped.get("ok") is True or not retryable or attempts >= 2:
                if result.get("recording") is not None:
                    result["recording"]["stop_attempts"] = attempts
                return stopped
            result["warnings"].append(
                "AUA recording stop hit an adb read timeout; retried once after the device settled."
            )
            await asyncio.sleep(1.0)

    session_id: str | None = None
    proof_mark: str | None = None
    setup_notes: list[str] = []
    setup_facts: list[str] = [str(item) for item in inherited_setup_facts]
    recording_started = False
    recording_path = (output / "journey.mp4").resolve()
    aua_artifacts_dir = (
        Path(session_artifacts_dir).resolve()
        if session_artifacts_dir is not None
        else (output / "aua").resolve()
    )

    try:
        if existing_session_id:
            session_id = existing_session_id
            result["session_id"] = session_id
            result["lifecycle"]["lease_strategy"] = "existing_session"
            result["setup"].append({"session_bootstrap": False, "existing_session": True})
            knowledge = []
        else:
            start_arguments: dict[str, Any] = {
                "goal": effective_session_goal,
                "package": package,
                "headed": headed,
                "artifacts_dir": str(aua_artifacts_dir),
                "evidence": "all",
                "launch_app": False,
                "grant_permissions": grant_permissions,
                "wait_for_lease_s": lease_wait_s,
                "provision_target": provision_target,
            }
            if needs:
                start_arguments["needs"] = [str(item) for item in needs]
            if session_contract:
                # AUA's own checkpoints, not just the judge's reading material. With them the
                # session can only be finished on fresh assertion proof, so the verdict comes back
                # `aua_session_contract` / verified rather than a model reading frames. Without
                # them AUA derives one phase from the goal sentence, with no assertions - which is
                # what `aua prepare` writes a contract to avoid.
                start_arguments["contract_yaml"] = session_contract
            if activity:
                start_arguments["activity"] = activity
            if apk:
                start_arguments["apk"] = apk
            if fresh_app:
                start_arguments.update({"fresh": True, "confirmed": True})
            start = await call("session_start", start_arguments, "setup")
            session_id = start.get("session_id")
            if (
                (not isinstance(session_id, str) or not session_id)
                and provision_target
                and fallback_lease_wait_s > 0
                and _target_start_failed(start)
            ):
                retry_arguments = dict(start_arguments)
                retry_arguments["provision_target"] = False
                retry_arguments["wait_for_lease_s"] = fallback_lease_wait_s
                result["lifecycle"]["lease_strategy"] = "waited_after_provision_failure"
                start = await call("session_start", retry_arguments, "setup")
                session_id = start.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise RunError("session_start returned no session_id: " + json.dumps(start)[:500])
            result["session_id"], result["serial"] = session_id, start.get("serial")
            knowledge = host_knowledge(start)
            result["knowledge_shown"] = [item.get("id") for item in knowledge]
            result["setup"].append({
                "session_bootstrap": True,
                "fresh_app": fresh_app,
                "apk": bool(apk),
                "provision_target": provision_target,
            })
            if fresh_app and apk:
                setup_facts.append("AUA completed the requested fresh APK bootstrap.")
            for forbidden_package in forbidden_packages:
                package_status = await call(
                    "app_status", {"package": str(forbidden_package)}, "setup"
                )
                installed = package_status.get("installed")
                result["setup"].append({
                    "forbidden_package": str(forbidden_package),
                    "installed": installed,
                    "status_ok": package_status.get("ok"),
                })
                if package_status.get("ok") is not True or not isinstance(installed, bool):
                    raise RunError(
                        "forbidden package status could not be verified: "
                        + json.dumps(package_status)[:400]
                    )
                if installed:
                    raise RunError(
                        f"forbidden package is installed on the leased target: {forbidden_package}"
                    )
                setup_facts.append(
                    f"AUA verified forbidden package {forbidden_package} is not installed."
                )
        if record and not record_after_host_setup:
            started_recording = await call("screen_record_start", {}, "setup")
            if started_recording.get("ok") is not True:
                message = "screen recording failed to start: " + json.dumps(started_recording)[:400]
                if recording_required:
                    raise RunError(message)
                result["recording"] = {"path": str(recording_path), "started": False, "stop_ok": False}
                result["recording_warning"] = message
                result["warnings"].append(message + "; continuing with AUA frame evidence.")
            else:
                recording_started = True
                result["recording"] = {"path": str(recording_path), "started": True, "stop_ok": None}
        if setup_proof is not None:
            # Device-clock mark excludes old-account log lines on a reused emulator. Failure
            # leaves proof unavailable rather than silently searching the entire log buffer.
            mark_name = "harness-setup-" + str(time.monotonic_ns())
            try:
                marked = await call("logcat_mark", {"name": mark_name}, "setup")
                if marked.get("ok") is True:
                    proof_mark = mark_name
            except Exception:
                pass
        for index, (flow_yaml, flow_params) in enumerate(prelaunch_setup_flows):
            flow, resumes = await replay_setup_flow(call, flow_yaml, flow_params)
            result["setup"].append({"prelaunch_setup_flow": index,
                                    "params": dict(flow_params or {}),
                                    "wait_resumes": resumes,
                                    "flow_run_ok": flow.get("ok"), "error": flow.get("error")})
            if flow.get("ok") is not True:
                raise RunError(
                    f"prelaunch setup flow {index} failed: " + json.dumps(flow)[:400]
                )
            fact_params = ", ".join(
                f"{key}={value}" for key, value in sorted((flow_params or {}).items())
            )
            setup_facts.append(
                f"Prelaunch setup flow {index} completed"
                + (f" with {fact_params}." if fact_params else ".")
            )
        if flags:
            # Feature-flag context is part of the oracle: a scenario judged in the wrong arm is
            # a verdict about a different product. flags_apply verifies each key against the
            # app's own stored prefs and restarts, so a key the build no longer knows is a
            # loud failure here rather than a silently dropped precondition.
            flag_file = output / "feature-flags.yaml"
            flag_file.write_text(
                "app: " + package + "\nflags:\n"
                + "".join(f"  {key}: {value}\n" for key, value in flags.items()),
                encoding="utf-8")
            # The tool is `flags_apply_and_analyze`; `flags_apply` was its name before the rename
            # and now fails with a usage error, which arrives here as "flags not applied and
            # verified" - a precondition failure that reads like a product one.
            applied = await call("flags_apply_and_analyze",
                                 {"path": str(flag_file.resolve()), "package": package,
                                  "restart": True, "verify": True}, "setup")
            result["setup"].append({"flags": dict(flags), "flags_ok": applied.get("ok")})
            result["flag_context"] = dict(flags)
            if applied.get("ok") is not True or applied.get("verified") is not True:
                raise RunError("feature flags not applied and verified: " + json.dumps(applied)[:400])
            setup_facts.append(
                "AUA applied and read-back verified feature flags: "
                + ", ".join(f"{key}={value}" for key, value in sorted(flags.items()))
                + "."
            )
        if record and record_after_host_setup:
            started_recording = await call("screen_record_start", {}, "setup")
            if started_recording.get("ok") is not True:
                message = "screen recording failed to start: " + json.dumps(started_recording)[:400]
                if recording_required:
                    raise RunError(message)
                result["recording"] = {"path": str(recording_path), "started": False, "stop_ok": False}
                result["recording_warning"] = message
                result["warnings"].append(message + "; continuing with AUA frame evidence.")
            else:
                recording_started = True
                result["recording"] = {"path": str(recording_path), "started": True, "stop_ok": None}
        launched: dict[str, Any] | None = None
        if launch_app:
            launch_arguments: dict[str, Any] = {"package": package}
            if activity:
                launch_arguments["activity"] = activity
            launched = await call("app_launch_and_analyze", launch_arguments, "setup")
            if launched.get("ok") is not True:
                raise RunError("app launch failed: " + json.dumps(launched)[:400])
        for index, (flow_yaml, flow_params) in enumerate(setup_flows):
            flow, resumes = await replay_setup_flow(call, flow_yaml, flow_params)
            result["setup"].append({"setup_flow": index, "params": dict(flow_params or {}),
                                    "wait_resumes": resumes,
                                    "flow_run_ok": flow.get("ok"), "error": flow.get("error")})
            if flow.get("ok") is not True:
                # A setup flow is the fast path to a precondition, not the oracle. When it
                # diverges the app is still running and still on a screen, so ending the run
                # here throws away a device to say nothing: measured on 2026-09-14, a guest
                # entry that had plainly succeeded returned `unverified` because the flow's
                # arrival marker had moved. AUA's own rule is that a divergence is recovery
                # information - hand the controller where it stopped and let it finish the
                # precondition semantically. What it must not do is hide, so the divergence
                # rides on the result, into the goal prompt, and past the judge.
                remaining = ", ".join(
                    str(step) for step in flow.get("remaining_steps") or []
                )
                setup_notes.append(
                    f"the setup flow stopped at step {flow.get('step_index')} "
                    f"({flow.get('code')}) on screen "
                    f"{flow.get('current_screen') or 'unknown'}; it still owed: "
                    f"{remaining or 'nothing'}"
                )
                result.setdefault("warnings", []).append(
                    f"setup flow {index} diverged: " + json.dumps(flow)[:400]
                )
                # Keep the divergence in a shape something can act on, not only prose. A stale
                # flow and a broken app are the same "a step did not land" from here, and the
                # caller is the only one who knows which - so record what it needs to decide:
                # the step that stopped, where the app went instead, and what that screen does
                # publish.
                failed = flow.get("failed_step")
                result.setdefault("setup_divergences", []).append({
                    "setup_flow": index,
                    "flow": flow.get("flow"),
                    "code": flow.get("code"),
                    "step_index": flow.get("step_index"),
                    "step": (failed or {}).get("display") if isinstance(failed, dict) else None,
                    "reached_screen": flow.get("current_screen"),
                    "remaining_steps": [str(s) for s in flow.get("remaining_steps") or []],
                    "markers_on_the_screen_reached": [
                        str(e.get("id"))
                        for e in (flow.get("elements") or [])
                        if isinstance(e, Mapping) and e.get("id")
                    ][:12],
                })
        # After every setup flow, while the session is still live: a precondition the caller
        # must be able to prove rather than infer from the screen.
        if setup_proof is not None and proof_mark is not None:
            proved = await capture_setup_proof(
                call, setup_proof, regex=setup_proof_regex, since=proof_mark,
            )
            if proved is not None:
                result[setup_proof[0]] = proved
        if flags or setup_flows or observation_frame(launched) is None:
            initial = await call("analyze_screen", {"source": "hierarchy", "no_cache": True}, "setup")
        else:
            assert launched is not None
            initial = launched
        if observation_frame(initial) is None:
            raise RunError("initial analyze_screen returned no fresh frame")
        schemas = await list_tools()
        tools = realapp_tools(
            schemas,
            contract=bool(session_contract),
            capabilities=controller_capabilities,
        )
        result["setup_facts"] = list(setup_facts)
        claims: list[dict[str, Any]] = []

        async def controller_call(name: str, arguments: dict[str, Any]) -> Any:
            arguments, repaired_id = normalize_element_id_argument(arguments)
            if repaired_id:
                repair = (
                    f"controller omitted the el: namespace for {name}; the harness restored it "
                    "before AUA validation"
                )
                result.setdefault("controller_argument_repairs", []).append(repair)
                if repair not in result["warnings"]:
                    result["warnings"].append(repair)
            if name == "session_finish":
                claims.append(copy.deepcopy(arguments))
                # This is a model completion claim, not infrastructure authority. Keep the AUA
                # session and its lease alive through the final observation, judges, recording
                # export and finally cleanup.
                return {"ok": False, "finished": False, "claim_recorded": True}
            if name == "session_progress":
                arguments = {"session_id": session_id}
            if name == "app_force_stop":
                return await call(
                    "app", {"action": "stop", "package": package}, "controller"
                )
            if name == "app_relaunch_and_analyze":
                relaunch_arguments: dict[str, Any] = {"package": package}
                if activity:
                    relaunch_arguments["activity"] = activity
                return await call(
                    "app_launch_and_analyze", relaunch_arguments, "controller"
                )
            if name == ASYNC_UI_WAIT_TOOL:
                return await run_async_ui_wait(
                    call=call,
                    arguments=arguments,
                    result=result,
                    output=output,
                )
            if name == WALL_CLOCK_WAIT_TOOL:
                before_host = time.monotonic()
                before_device = await call(
                    "shell_read_only", {"argv": ["date", "+%s"]}, "harness-wait-boundary"
                )
                started_job = await call(
                    "job_start",
                    {
                        "operation": "idle-duration",
                        "timeout_ms": WALL_CLOCK_WAIT_SECONDS * 1_000,
                        "observe": False,
                    },
                    "harness-wait-start",
                )
                job_id = started_job.get("job_id")
                if started_job.get("ok") is not True or not isinstance(job_id, str) or not job_id:
                    raise RunError("AUA could not detach the inactivity timer: " + json.dumps(started_job)[:500])
                wait_started_at = datetime.now().astimezone()
                receipt: dict[str, Any] = {
                    "job_id": job_id,
                    "operation": "idle-duration",
                    "requested_seconds": WALL_CLOCK_WAIT_SECONDS,
                    "status": started_job.get("status"),
                    "started_at": wait_started_at.isoformat(),
                    "deadline_at": (
                        wait_started_at + timedelta(seconds=WALL_CLOCK_WAIT_SECONDS)
                    ).isoformat(),
                    "device_clock_before": device_epoch_seconds(before_device),
                    "model_calls_during_wait": 0,
                    "device_reads_during_wait": 0,
                    "reconnect": {"tool": "job_status", "arguments": {"job_id": job_id}},
                }
                result["deferred_waits"].append(receipt)

                def persist_waits() -> None:
                    (output / "deferred-waits.json").write_text(
                        json.dumps(result["deferred_waits"], ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

                persist_waits()
                try:
                    while True:
                        # No model turn and no device operation happens here. Yielding the event
                        # loop keeps the harness available to supervise parallel lanes on their own
                        # leases; this guarded target remains exclusively idle.
                        await asyncio.sleep(30)
                        status = await call(
                            "job_status", {"job_id": job_id}, "harness-wait-status"
                        )
                        receipt["status"] = status.get("status")
                        receipt["progress_percent"] = status.get("progress_percent")
                        persist_waits()
                        if status.get("terminal") is True:
                            break
                except BaseException:
                    with contextlib.suppress(Exception):
                        cancelled = await call(
                            "job_cancel", {"job_id": job_id, "wait_ms": 1_000},
                            "harness-wait-cancel",
                        )
                        receipt["status"] = cancelled.get("status")
                        receipt["cancelled"] = True
                        persist_waits()
                    raise
                if status.get("status") != "succeeded" or status.get("run_ok") is not True:
                    raise RunError("AUA inactivity timer did not complete: " + json.dumps(status)[:500])
                after_device = await call(
                    "shell_read_only", {"argv": ["date", "+%s"]}, "harness-wait-boundary"
                )
                elapsed = time.monotonic() - before_host
                before_epoch = device_epoch_seconds(before_device)
                after_epoch = device_epoch_seconds(after_device)
                device_elapsed = (
                    after_epoch - before_epoch
                    if before_epoch is not None and after_epoch is not None
                    else None
                )
                receipt.update({
                    "status": "succeeded",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "host_monotonic_elapsed_seconds": elapsed,
                    "device_clock_after": after_epoch,
                    "device_clock_elapsed_seconds": device_elapsed,
                    "job_result": status.get("result"),
                })
                persist_waits()
                return {
                    "ok": elapsed > 600 and device_elapsed is not None and device_elapsed > 600,
                    "action": "wait-uninterrupted",
                    "requested_seconds": WALL_CLOCK_WAIT_SECONDS,
                    "host_monotonic_elapsed_seconds": elapsed,
                    "device_clock_before": before_epoch,
                    "device_clock_after": after_epoch,
                    "device_clock_elapsed_seconds": device_elapsed,
                    "deferred_job_id": job_id,
                    "model_calls_during_wait": 0,
                    "device_reads_during_wait": 0,
                    "note": (
                        "AUA's detached idle-duration job guarded the leased device; no UI "
                        "observation, screenshot, foreground action, or model call ran in the interval."
                    ),
                }
            return await call_tool(name, arguments)

        compactor = FrameCompactor(max_elements=max_elements)
        report = await run_agent(
            send=send, call_tool=controller_call, tools=tools,
            system_prompt=SYSTEM + COMPACT_SYSTEM + (
                CONTRACT_SYSTEM if session_contract else REALAPP_SYSTEM
            ),
            user_prompt=goal_prompt(
                goal, knowledge, setup_notes, setup_facts, authored_context
            ),
            initial_observation=initial, model=model, output=output / "controller",
            request_config=settings, backend=backend, max_tokens=max_tokens, max_steps=max_steps,
            model_fallbacks=controller_fallbacks,
            time_limit_s=time_limit_s, max_request_bytes=max_request_bytes,
            cost_limit_usd=cost_limit_usd, observation_filter=hosted_model_view,
            model_observation_filter=compactor, request_timeout_s=request_timeout_s,
            tool_timeouts_s=controller_tool_timeouts(controller_capabilities),
            terminal_tools=frozenset({"session_finish"}), terminal_claim_limit=terminal_claim_limit,
            no_progress_limit=no_progress_limit,
        )
        result["controller"] = {
            key: report.get(key) for key in (
                "stop_reason", "error", "steps_consumed", "model_requests", "tool_calls_executed",
                "tool_errors", "schema_repairs", "terminal_claims", "no_progress_streak",
                "returned_models", "providers", "model_http_seconds", "tool_seconds", "duration_seconds",
                "final_model_text", "model_ladder", "model_escalations", "model_failures",
            )
        }
        result["controller"]["prompt_tokens"] = [
            (usage or {}).get("prompt_tokens") for usage in report.get("usage", [])]
        result["controller"]["compaction"] = {"frames": compactor.frames_seen, "unchanged_hits": compactor.unchanged_hits}
        for warning in report.get("warnings") or []:
            message = "controller warning: " + str(warning)[:500]
            if message not in result["warnings"]:
                result["warnings"].append(message)
        result["claim"] = claims[-1] if claims else None
        result["cost"]["controller"] = {
            "model": (report.get("returned_models") or [model])[0],
            "provider": (report.get("providers") or [None])[0], "usd": controller_cost(report),
        }

        final = await call("analyze_screen", {"source": "hierarchy", "no_cache": True}, "final")
        # Kept raw so a verdict can be re-judged offline from the same evidence later.
        (output / "final-observation.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
        frames = []
        for entry in report.get("evidence", []):
            raw = json.loads((output / "controller" / entry["path"]).read_text(encoding="utf-8"))
            if observation_frame(raw) is not None:
                frames.append({"ref": entry["ref"], "tool": entry.get("tool"), "raw": raw})
        actions = [
            {"step": call_record["step"], "tool": call_record["tool"], "arguments": call_record.get("arguments")}
            for call_record in _load_jsonl(output / "controller" / "tool-calls.jsonl")
            if call_record.get("executed") is True
        ]
        # Recording is journey evidence, not judge latency evidence. Export it while the exact
        # device boot is still unquestionably alive; a slow pair of hosted judges used to keep
        # the encoder running long enough for a provisioned target failure to turn an otherwise
        # valid row into recording_identity_mismatch.
        if recording_started:
            stopped_recording = await stop_recording("evidence")
            stop_ok = stopped_recording.get("ok") is True and recording_path.is_file()
            if result["recording"] is not None:
                result["recording"]["stop_ok"] = stop_ok
            if not stop_ok:
                message = "recording cleanup failed: " + str(
                    stopped_recording.get("error") or stopped_recording
                )[:500]
                if recording_required:
                    result["recording_cleanup_error"] = message
                else:
                    result["recording_warning"] = message
                    result["warnings"].append(
                        message + "; product judgement continues from AUA frame evidence."
                    )
                recording_started = False
            recording_started = False
        stop = report.get("stop_reason")
        contract_progress = None
        if session_contract:
            # The model's session_finish never reaches AUA here - it is a claim, and the
            # session has to outlive it for the final observation, the judge and the recording.
            # So ask AUA directly instead: `session_progress` is the same authority that would
            # have accepted or refused the finish, and every checkpoint in it was completed on
            # fresh assertions rather than on anything the model said.
            contract_progress = await call("session_progress", {}, "judgement")
            result["contract_progress"] = contract_progress
            result["contract_unmet"] = unmet_checkpoint(contract_progress)
        if stop == "terminal_tool":
            result["verdict"] = {"oracle": "aua_session_contract", "verified": True, "verdict": "pass",
                                 "reasons": ["AUA accepted session_finish against its own contract."]}
        elif contract_satisfied(contract_progress):
            counts = goal_progress_of(contract_progress) or {}
            result["verdict"] = {
                "oracle": "aua_session_contract", "verified": True, "verdict": "pass",
                "reasons": [
                    f"AUA completed all {counts.get('total')} authored checkpoints on fresh "
                    "assertions; no model read the frames to decide this."
                ],
            }
        elif stop in (
            "terminal_claimed", "model_text", "no_progress", "conversation_budget", "step_budget"
        ) and judge:
            decider = Decider(send, model=judging_model, backend=backend,
                              request_config=judge_settings, fallbacks=judge_fallbacks,
                              max_tokens=judge_max_tokens, cost_limit_usd=judge_cost_limit_usd,
                              output=output / "judge")
            # AUA's goal_progress without an authored contract is always 0/1 active; it would
            # only mislead a judge, so real-app mode does not pass it.
            context_actions = list(actions)
            if result["claim"]:
                context_actions.append({"step": len(actions), "tool": "session_finish",
                                        "arguments": {"controller_claim_untrusted": result["claim"]}})
            judged_frames = judged_frame_sample(
                [frame["raw"] for frame in frames],
                judge_intermediate_frame_limit(judge_frames, vision=vision),
            )
            images: list[str] = []
            if vision:
                # Element text cannot answer a question about appearance. Pair each judged
                # frame with the screenshot AUA already captured for it, oldest first, so the
                # final screen is the last image the judge sees.
                shot_index = screenshot_index(aua_artifacts_dir / "manifest.json")
                image_frames = image_frame_sample(judged_frames, MAX_IMAGES - 1)
                for frame in image_frames:
                    shot = screenshot_for(shot_index, frame_fingerprint(frame))
                    encoded = encode_image(shot) if shot else None
                    if encoded:
                        images.append(encoded)
                final_shot = screenshot_for(shot_index, frame_fingerprint(final))
                final_image = encode_image(final_shot) if final_shot else None
                if final_image:
                    images.append(final_image)
                result["vision"] = {"requested": True, "frames": len(judged_frames) + 1,
                                    "image_frames_selected": len(image_frames) + 1,
                                    "images_attached": len(images),
                                    "final_image_attached": bool(final_image)}
            verdict = await judge_outcome_votes(
                decider, votes=judge_votes, goal=goal, final_frame=final,
                frames=judged_frames, actions=context_actions,
                images=images, contract=contract,
            )
            if stop == "no_progress" and verdict["verdict"] == "pass":
                verdict["verdict"] = "pass_with_warning"
                verdict["reasons"].insert(0, "Controller stalled on an unchanged screen before finishing.")
            if stop in {"conversation_budget", "step_budget"} and verdict["verdict"] == "pass":
                verdict["verdict"] = "pass_with_warning"
                verdict["reasons"].insert(
                    0,
                    "Controller reached its bounded "
                    + ("conversation" if stop == "conversation_budget" else "step")
                    + " budget; the saved frames nevertheless prove the contract.",
                )
            verdict["controller_stop_reason"] = stop
            result["verdict"] = verdict
            result["cost"]["judge"] = {"model": judging_model, "provider": (verdict["votes"][0].get("provider") if verdict["votes"] else None),
                                       "usd": verdict["cost"], "decider": decider.report()}
        else:
            reason = report.get("error") or f"controller stopped with {stop}"
            result["verdict"] = {"oracle": "none", "verified": False, "verdict": "unverified",
                                 "reasons": [str(reason)[:300]], "controller_stop_reason": stop}
        if save_primary_flow and (result.get("verdict") or {}).get("verdict") in {"pass", "pass_with_warning"}:
            try:
                result.update(await export_primary_flow(call, output))
            except Exception:
                message = "Primary flow export could not prove a clean replayable controller route."
                result["primary_flow_error"] = message
                _mark_cleanup_failure(result, message)
        if result.get("recording_cleanup_error"):
            _mark_cleanup_failure(result, str(result["recording_cleanup_error"]))
        if name_screens:
            namer_decider = Decider(send, model=model, backend=backend, request_config=settings,
                                    max_tokens=4096, cost_limit_usd=judge_cost_limit_usd, output=output / "map")
            namer = ScreenNamer(namer_decider)
            ordered = [{"ref": "initial", "tool": None, "raw": initial}] + frames + [{"ref": "final", "tool": None, "raw": final}]
            transitions: list[dict[str, Any]] = []
            previous_name: str | None = None
            for frame in ordered:
                if len(namer.distinct()) >= max_named_screens and namer.fingerprint(frame["raw"]) not in namer.by_fingerprint:
                    transitions.append({"after": frame["tool"], "to": "unnamed(limit)"})
                    continue
                meta = (observation_frame(frame["raw"]) or {}).get("meta") or {}
                known = meta.get("known_screen") if isinstance(meta.get("known_screen"), str) else None
                entry = await namer.name(frame["raw"], known_name=known)
                name = entry["logical_name"] if entry else None
                if name and name != previous_name:
                    transitions.append({"after": frame["tool"], "from": previous_name, "to": name, "evidence_ref": frame["ref"]})
                    previous_name = name
            result["screens"] = namer.distinct()
            if result["screens"]:
                result["route"] = await summarize_route(namer_decider, goal=goal, screens=result["screens"],
                                                        transitions=transitions)
                result["route"]["transitions"] = transitions
            result["cost"]["map"] = {"model": model, "provider": (result["cost"]["controller"] or {}).get("provider"),
                                     "usd": namer_decider.total_cost, "decider": namer_decider.report()}
            (output / "screens.json").write_text(json.dumps(result["screens"], ensure_ascii=False, indent=2) + "\n")
            if result["route"]:
                (output / "route.json").write_text(json.dumps(result["route"], ensure_ascii=False, indent=2) + "\n")
    except Exception as exc:
        result["error"] = _error_text(exc)
        if result["verdict"] is None:
            result["verdict"] = {"oracle": "none", "verified": False, "verdict": "unverified",
                                 "reasons": [result["error"][:300]]}
    finally:
        if recording_started:
            try:
                stopped_recording = await stop_recording("cleanup")
                stop_ok = stopped_recording.get("ok") is True and recording_path.is_file()
                if result["recording"] is not None:
                    result["recording"]["stop_ok"] = stop_ok
                if not stop_ok:
                    message = "recording cleanup failed: " + str(
                        stopped_recording.get("error") or stopped_recording
                    )[:500]
                    if recording_required:
                        result["recording_cleanup_error"] = message
                        _mark_cleanup_failure(result, message)
                    else:
                        result["recording_warning"] = message
                        result["warnings"].append(
                            message + "; product judgement continues from AUA frame evidence."
                        )
                recording_started = False
            except Exception as exc:
                message = "recording cleanup failed: " + _error_text(exc)
                if recording_required:
                    result["recording_cleanup_error"] = message
                    _mark_cleanup_failure(result, message)
                else:
                    result["recording_warning"] = message
                    result["warnings"].append(
                        message + "; product judgement continues from AUA frame evidence."
                    )
        # Refresh while the session is still live: a later state change must supersede setup
        # proof. Missing/unreadable evidence cannot preserve an earlier positive attestation.
        if setup_proof is not None and proof_mark is not None:
            retried = await capture_setup_proof(
                call, setup_proof, regex=setup_proof_regex, since=proof_mark,
            )
            result[setup_proof[0]] = retried

        if session_id and finish_session:
            try:
                finish_arguments: dict[str, Any] = {
                    "session_id": session_id,
                    "allow_incomplete": True,
                    "summary": False,
                }
                if not retain_started_target:
                    finish_arguments["retain_started_target"] = False
                finished_session = await call("session_finish", finish_arguments, "cleanup")
                finish_ok = finished_session.get("ok") is True and (
                    finished_session.get("finished") is True
                    or finished_session.get("terminated") is True
                )
                if not finish_ok:
                    message = "session cleanup failed: " + str(
                        finished_session.get("error") or finished_session
                    )[:500]
                    result["cleanup_error"] = message
                    _mark_cleanup_failure(result, message)
            except Exception as exc:
                message = "session cleanup failed: " + _error_text(exc)
                result["cleanup_error"] = message
                _mark_cleanup_failure(result, message)
        result["cost"]["total_usd"] = round(sum(
            float(entry["usd"]) for key, entry in result["cost"].items()
            if key != "total_usd" and isinstance(entry, dict)), 8)
        result["duration_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        (output / "verdict.md").write_text(verdict_markdown(result))
    return result


def parse_pairs(items: Sequence[str], *, what: str) -> dict[str, str]:
    """Parse repeated or comma-joined ``K=V`` arguments into one mapping."""
    pairs: dict[str, str] = {}
    for item in items or ():
        for chunk in str(item).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise RunError(f"{what} needs K=V, got {chunk!r}")
            key, _, value = chunk.partition("=")
            key, value = key.strip(), value.strip()
            if not key:
                raise RunError(f"{what} needs a name before '=', got {chunk!r}")
            pairs[key] = value
    return pairs


def build_setup_flows(paths: Sequence[Any], params: Sequence[str]) -> list[tuple[str, dict[str, str]]]:
    """Pair each --setup-flow with the --setup-params given at the same position.

    Position, not name, is what binds them: two flows can legitimately take a parameter of
    the same name with different values, and a flow that needs none is skipped with ''.
    """
    if len(params) > len(paths):
        raise RunError("more --setup-params than --setup-flow")
    flows: list[tuple[str, dict[str, str]]] = []
    for index, path in enumerate(paths):
        raw = params[index] if index < len(params) else ""
        flows.append((Path(path).read_text(encoding="utf-8"),
                      parse_pairs([raw], what="--setup-params")))
    return flows


def mcp_read_timeout(lease_wait_s: float, fallback_lease_wait_s: float) -> timedelta:
    """Cover both sequential lease waits plus bounded bootstrap overhead."""
    seconds = max(180.0, lease_wait_s + fallback_lease_wait_s + 120.0)
    return timedelta(seconds=seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goal", required=True)
    parser.add_argument("--package", required=True, help="Application package under test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("openrouter-comparison.json"))
    parser.add_argument("--model", required=True, help="Candidate id or repository from the manifest")
    parser.add_argument("--provider", help="Override the pinned provider slug (for example when a pin rots)")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPEN_ROUTER_API_KEY")
    parser.add_argument("--aua-command", default="aua")
    parser.add_argument("--launch", action="store_true", help="Compatibility flag; app launch is automatic")
    parser.add_argument("--fresh", action="store_true", help="Clear app data and reinstall the selected APK")
    parser.add_argument("--apk", help="Build installed by session bootstrap when provided")
    parser.add_argument(
        "--grant-permissions",
        action="store_true",
        help="Grant every declared runtime permission after install (off by default)",
    )
    parser.add_argument(
        "--forbid-package",
        action="append",
        default=[],
        help="Fail before controller navigation when this package is installed; repeatable",
    )
    parser.add_argument("--activity")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Expose the virtual Android device while the controller is running",
    )
    parser.add_argument("--record", action="store_true", help="Record from the first app launch through judgement")
    parser.add_argument("--lease-wait", type=float, default=0,
                        help="Seconds to wait for a free device before provisioning another")
    parser.add_argument("--fallback-lease-wait", type=float, default=600,
                        help="Seconds to wait for an existing lease after provisioning fails")
    parser.add_argument("--no-provision", action="store_true",
                        help="Never create a new virtual target; only wait for an existing device")
    parser.add_argument(
        "--stop-started-target",
        action="store_true",
        help="After cleanup, stop only the exact virtual target boot this run started.",
    )
    parser.add_argument(
        "--needs",
        action="append",
        default=[],
        help="Required target capability passed to AUA session bootstrap; repeatable",
    )
    parser.add_argument(
        "--controller-capability",
        action="append",
        default=[],
        choices=sorted(CONTROLLER_CAPABILITIES),
        help="Add a narrow scenario-declared controller capability; repeatable",
    )
    parser.add_argument("--prelaunch-setup-flow", type=Path, action="append", default=[],
                        help="AUA flow YAML run before flags and the first product launch; repeatable")
    parser.add_argument("--prelaunch-setup-params", action="append", default=[],
                        help="K=V[,K=V] for the prelaunch flow at the same position")
    parser.add_argument("--setup-flow", type=Path, action="append", default=[],
                        help="AUA flow YAML run before the goal; repeat to chain flows in order")
    parser.add_argument("--setup-params", action="append", default=[],
                        help="K=V[,K=V] for the --setup-flow at the same position; use '' to skip one")
    parser.add_argument("--flags", action="append", default=[],
                        help="Feature flag K=V applied and verified before the setup flows; repeatable")
    parser.add_argument("--setup-proof-name",
                        help="Result key to record a proved setup precondition under "
                             "(e.g. a persona tier). Requires --setup-proof-grep or --setup-proof-regex and "
                             "--setup-proof-value.")
    parser.add_argument("--setup-proof-grep",
                        help="Substring searched in the device log after the setup flows. Only "
                             "whether it matched is recorded -- never the matching line, which "
                             "may carry account data.")
    parser.add_argument("--setup-proof-regex",
                        help="Regex alternative to --setup-proof-grep. A named 'value' group "
                             "compares the latest matching value with --setup-proof-value; "
                             "captured text is never returned.")
    parser.add_argument("--save-primary-flow", action="store_true",
                        help="Export a proved controller journal suffix to output/flow.yaml; never save globally")
    parser.add_argument("--setup-proof-value",
                        help="Value recorded as `actual` when --setup-proof-grep matches.")
    parser.add_argument("--contract", type=Path,
                        help="Authored acceptance criteria the judge answers against")
    parser.add_argument(
        "--context",
        type=Path,
        help="Authored controller context that must not become session goal phases",
    )
    parser.add_argument("--session-contract", type=Path,
                        help="Version-1 checkpoint YAML AUA itself holds the session to, so "
                             "session_finish is accepted on fresh assertion proof rather than "
                             "on the model's word. Usually the same file as --contract.")
    parser.add_argument("--vision", action="store_true",
                        help="Show the judge the captured screenshots; needs an image-capable model")
    parser.add_argument("--judge-model",
                        help="Manifest candidate used only for judgement; defaults to --model")
    parser.add_argument(
        "--controller-fallback",
        action="append",
        default=[],
        metavar="CANDIDATE",
        help=(
            "Ordered manifest candidate to continue controller inference after a model request "
            "fails before returning a usable response; repeatable."
        ),
    )
    parser.add_argument("--judge-fallback", action="append", default=[], metavar="CANDIDATE",
                        help="Stronger manifest candidate to ask when the judge cannot produce "
                             "its schema. Repeatable; tried in the order given.")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-votes", type=int, default=2, choices=[1, 2])
    parser.add_argument("--judge-frames", type=int, default=8,
                        help="How many observations to show the judge, spread across the whole "
                             "journey. A contract bullet about the route is unverifiable from "
                             "the tail alone.")
    parser.add_argument("--map", action="store_true", help="Name screens and summarise the route (paid)")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--cost-limit-usd", type=float, default=0.05)
    parser.add_argument("--judge-cost-limit-usd", type=float, default=0.02)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--terminal-claim-limit", type=int, default=1)
    parser.add_argument("--no-progress-limit", type=int, default=4)
    parser.add_argument("--max-elements", type=int, default=60)
    args = parser.parse_args()

    prelaunch_setup_flows = build_setup_flows(
        args.prelaunch_setup_flow, args.prelaunch_setup_params
    )
    setup_flows = build_setup_flows(args.setup_flow, args.setup_params)
    if args.setup_proof_grep and args.setup_proof_regex:
        parser.error("use only one of --setup-proof-grep and --setup-proof-regex")
    proof_parts = (args.setup_proof_name, args.setup_proof_regex or args.setup_proof_grep,
                   args.setup_proof_value)
    if any(proof_parts) and not all(proof_parts):
        parser.error("--setup-proof-name, one proof pattern, and --setup-proof-value go together")
    setup_proof = tuple(proof_parts) if all(proof_parts) else None
    if args.setup_proof_regex:
        try:
            re.compile(args.setup_proof_regex)
        except re.error:
            parser.error("--setup-proof-regex is not a valid regular expression")
    manifest = json.loads(args.manifest.read_text())
    candidate = next((item for item in manifest["models"] if args.model in {item["id"], item["repository"]}), None)
    if candidate is None:
        parser.error("model must be a candidate in the manifest")
    request_config = copy.deepcopy(candidate.get("request_config", {}))
    if args.provider:
        request_config.setdefault("provider", {})
        request_config["provider"]["only"] = [args.provider]
        request_config["provider"]["order"] = [args.provider]
    controller_fallbacks: list[tuple[str, dict[str, Any]]] = []
    for name in args.controller_fallback:
        rung = next(
            (
                item
                for item in manifest["models"]
                if name in {item["id"], item["repository"]}
            ),
            None,
        )
        if rung is None:
            parser.error(f"--controller-fallback {name} is not a candidate in the manifest")
        controller_fallbacks.append(
            (
                rung["repository"],
                validate_request_config(copy.deepcopy(rung.get("request_config", {}))),
            )
        )
    judge_candidate = candidate
    judge_config = None
    if args.judge_model:
        judge_candidate = next(
            (item for item in manifest["models"]
             if args.judge_model in {item["id"], item["repository"]}),
            None,
        )
        if judge_candidate is None:
            parser.error("--judge-model must be a candidate in the manifest")
        judge_config = validate_request_config(
            copy.deepcopy(judge_candidate.get("request_config", {}))
        )
    judge_fallbacks: list[tuple[str, dict[str, Any]]] = []
    for name in args.judge_fallback:
        rung = next((item for item in manifest["models"]
                     if name in {item["id"], item["repository"]}), None)
        if rung is None:
            parser.error(f"--judge-fallback {name} is not a candidate in the manifest")
        if args.vision and not rung.get("vision"):
            parser.error(f"--vision needs image-capable fallbacks; {rung['id']} is text-only")
        judge_fallbacks.append((rung["repository"],
                                validate_request_config(copy.deepcopy(rung.get("request_config", {})))))
    if args.vision and not judge_candidate.get("vision"):
        parser.error(f"--vision needs an image-capable judge; {judge_candidate['id']} is text-only. "
                     "The manifest records which candidates accept images; pass --judge-model.")
    request_config = validate_request_config(request_config)
    key = os.environ.get(args.api_key_env)
    validate_endpoint(args.base_url, key)

    async def execute() -> dict[str, Any]:
        import httpx
        from experiments.aua_controller.run_live import mcp_server
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        headers = {"Authorization": f"Bearer {key}"}
        retries: list[dict[str, Any]] = []
        server = mcp_server(args.aua_command)
        async with httpx.AsyncClient(headers=headers, timeout=120, follow_redirects=False) as http:
            def classify(exc: BaseException):
                """(status, headers) for a transport failure; None for anything else."""
                if isinstance(exc, httpx.HTTPStatusError):
                    return (
                        retryable_http_status(exc.response.status_code, exc.response.text),
                        exc.response.headers,
                    )
                if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
                    return None, None  # Nothing was served, so nothing was charged.
                return None

            def note_retry(attempt: int, delay: float, status: int | None) -> None:
                retries.append({"attempt": attempt, "delay_s": round(delay, 2), "status": status})
                print(f"provider busy (HTTP {status}); retry {attempt} in {delay:.1f}s",
                      file=sys.stderr, flush=True)

            async def send(payload: dict[str, Any]) -> dict[str, Any]:
                async def once() -> dict[str, Any]:
                    response = await http.post(
                        args.base_url.rstrip("/") + "/chat/completions", json=payload)
                    response.raise_for_status()
                    return response.json()

                body = await resilient_request(
                    once, classify=classify, sleep=asyncio.sleep, on_retry=note_retry)
                if not isinstance(body, dict):
                    raise RunError("endpoint returned non-object JSON")
                return body

            async with (
                stdio_client(server) as (read, write),
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=mcp_read_timeout(
                        args.lease_wait, args.fallback_lease_wait
                    ),
                ) as session,
            ):
                await session.initialize()

                async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                    return await session.call_tool(name, arguments)

                async def list_tools() -> dict[str, dict[str, Any]]:
                    listing = await session.list_tools()
                    schemas = {}
                    for tool in listing.tools:
                        data = tool.model_dump(mode="json")
                        schema = dict(data["inputSchema"])
                        schema.setdefault("description", data.get("description"))
                        schemas[data["name"]] = schema
                    return schemas

                outcome = await run_realapp(
                    call_tool=call_tool, list_tools=list_tools, send=send, goal=args.goal,
                    package=args.package, output=args.output.resolve(), model=candidate["repository"],
                    request_config=request_config, controller_fallbacks=controller_fallbacks,
                    launch=args.launch, activity=args.activity,
                    headed=args.headed,
                    fresh_app=args.fresh, apk=args.apk,
                    grant_permissions=args.grant_permissions,
                    forbidden_packages=args.forbid_package,
                    record=args.record,
                    lease_wait_s=args.lease_wait, fallback_lease_wait_s=args.fallback_lease_wait,
                    provision_target=not args.no_provision,
                    needs=args.needs,
                    controller_capabilities=args.controller_capability,
                    flags=parse_pairs(args.flags, what="--flags"),
                    prelaunch_setup_flows=prelaunch_setup_flows,
                    setup_flows=setup_flows,
                    setup_proof=setup_proof,
                    setup_proof_regex=bool(args.setup_proof_regex),
                    save_primary_flow=args.save_primary_flow,
                    contract=args.contract.read_text(encoding="utf-8") if args.contract else None,
                    authored_context=(
                        args.context.read_text(encoding="utf-8") if args.context else None
                    ),
                    session_contract=(
                        args.session_contract.read_text(encoding="utf-8")
                        if args.session_contract
                        else None
                    ),
                    vision=args.vision,
                    judge_model=judge_candidate["repository"] if args.judge_model else None,
                    judge_request_config=judge_config,
                    judge_fallbacks=judge_fallbacks,
                    judge=not args.no_judge, judge_votes=args.judge_votes,
                    judge_frames=args.judge_frames, name_screens=args.map,
                    max_steps=args.max_steps, time_limit_s=args.time_limit, max_tokens=args.max_tokens,
                    cost_limit_usd=args.cost_limit_usd, judge_cost_limit_usd=args.judge_cost_limit_usd,
                    judge_max_tokens=args.judge_max_tokens,
                    terminal_claim_limit=args.terminal_claim_limit, no_progress_limit=args.no_progress_limit,
                    max_elements=args.max_elements,
                    retain_started_target=not args.stop_started_target,
                )
                # A run that needed four retries to finish is healthy but degrading, and the
                # only place that shows is here.
                outcome["provider_retries"] = retries
                return outcome

    result = asyncio.run(execute())
    print(json.dumps({
        "verdict": result["verdict"]["verdict"], "oracle": result["verdict"]["oracle"],
        "stop_reason": (result["controller"] or {}).get("stop_reason"), "claim": result.get("claim"),
        "steps": (result["controller"] or {}).get("steps_consumed"),
        "total_usd": result["cost"]["total_usd"], "error": result["error"],
        "provider_retries": len(result.get("provider_retries") or []),
        "output": str(args.output),
    }, default=str))
    return 0 if result["verdict"]["verdict"] in {"pass", "pass_with_warning"} else 1


if __name__ == "__main__":
    sys.exit(main())
