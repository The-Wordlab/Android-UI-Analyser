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
import copy
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.compaction import FrameCompactor
from experiments.aua_controller.hosted import BACKENDS, validate_endpoint, validate_request_config
from experiments.aua_controller.hosted_projection import hosted_model_view
from experiments.aua_controller.judgement import (
    Decider,
    ScreenNamer,
    encode_image,
    frame_fingerprint,
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
    tool_result,
)
from experiments.aua_controller.session_state import observation_frame
from experiments.aua_controller.transport import resilient_request

FORMAT = "aua-realapp-run-v1"
CONTROLLER_TOOLS = (
    "analyze_screen", "tap_and_analyze", "input_and_analyze", "swipe_and_analyze",
    "wait_and_analyze", "key_and_analyze", "session_progress", "session_finish",
)
#: The one extra tool a contract-driven run needs. A checkpoint completes only on fresh
#: assertion proof, so without a way to assert, a loaded contract can never be satisfied and
#: every verdict falls back to a model reading frames - the weaker answer, from a run that was
#: given the stronger one. Added only when there is a contract: a run with no checkpoints has
#: nothing to assert against, and one more tool in the list is one more way to spend a step.
CONTRACT_TOOL = "expect_and_analyze"
#: One selector and one predicate is the whole vocabulary a checkpoint assertion needs, and
#: every extra argument is another one a small model can get wrong.
CONTRACT_TOOL_PROPERTIES = ("rid", "text", "desc", "exists", "absent", "text_contains")
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


def host_knowledge(start: dict[str, Any]) -> list[dict[str, Any]]:
    """The facts ``session_start`` ranked against the goal: recorded advice, never verified state."""
    items = start.get("relevant_knowledge")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and str(item.get("text") or "").strip()][:KNOWLEDGE_SHOWN]


def goal_prompt(
    goal: str, knowledge: list[dict[str, Any]], setup_notes: Sequence[str] = ()
) -> str:
    """The first user message: the goal, what the host already knows, what setup did not finish.

    Without this the model re-derives facts the store already holds (where a setting lives, that a
    fresh install overwrites it, the route to it). Ids stay out: they are host bookkeeping.

    *setup_notes* carries any setup flow that diverged. The alternative - saying nothing - makes the
    model start from a screen the harness expected to be somewhere else, with no idea which part of
    the precondition is missing.
    """
    lines = ["Goal: " + goal]
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


def realapp_tools(
    schemas: dict[str, dict[str, Any]], *, contract: bool = False
) -> list[dict[str, Any]]:
    """compact-v1 tools, with session_finish carrying the model's outcome claim."""
    tools = []
    for name in (*CONTROLLER_TOOLS, *((CONTRACT_TOOL,) if contract else ())):
        if name not in schemas:
            raise RunError(f"AUA MCP does not offer {name}")
        if name == "session_finish":
            parameters = finish_schema()
        elif name == CONTRACT_TOOL:
            parameters = contract_tool_schema(schemas[name])
        else:
            parameters = compact_schema(name, schemas[name])
        description = str(schemas[name].get("description") or "")[:300]
        if name == "session_finish":
            description = "Claim the goal is finished (or blocked) with an outcome and a short note."
        tools.append({"type": "function", "function": {"name": name, "description": description,
                                                       "parameters": parameters}})
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


SETUP_FLOW_RESUMES = 4
"""How often a setup flow may re-issue a wait the ceiling cut short.

`perf.max_wait_ms` ends every observation wait at 5s, so a flow that writes
`timeout_ms: 60000` gets 5s and reports `wait_timeout` whether the screen is late or
genuinely absent. The remedy AUA documents is to ask again, not to pass a bigger number the
ceiling ignores. Four resumes buy a precondition about 25s without letting a wrong flow spin:
a setup flow is the fast path, not the oracle, and a run abandoned because a cold start took
seven seconds is a verdict about the harness rather than about the product.
"""


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
    package: str,
    output: Path,
    model: str,
    request_config: dict[str, Any] | None = None,
    backend: str = "openrouter",
    launch: bool = False,
    activity: str | None = None,
    fresh_app: bool = False,
    apk: str | None = None,
    grant_permissions: bool = False,
    record: bool = False,
    lease_wait_s: float = 0,
    fallback_lease_wait_s: float = 600,
    provision_target: bool = True,
    flags: dict[str, str] | None = None,
    prelaunch_setup_flows: Sequence[tuple[str, dict[str, str]]] = (),
    setup_flows: Sequence[tuple[str, dict[str, str]]] = (),
    contract: str | None = None,
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
) -> dict[str, Any]:
    """Return the run result; also written to ``<output>/result.json`` and ``verdict.md``."""
    if backend not in BACKENDS:
        raise RunError("unknown backend")
    if fresh_app and not apk:
        raise RunError("fresh_app needs --apk: a booted AVD has no app to clear")
    if lease_wait_s < 0 or fallback_lease_wait_s < 0:
        raise RunError("lease waits must be non-negative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RunError("real-app output directory must be empty")
    settings = validate_request_config(request_config or {}) if backend == "openrouter" else copy.deepcopy(request_config or {})
    judging_model = judge_model or model
    judge_settings = (
        validate_request_config(judge_request_config or request_config or {})
        if backend == "openrouter"
        else copy.deepcopy(judge_request_config or request_config or {})
    )
    started = time.monotonic()
    result: dict[str, Any] = {
        "format": FORMAT, "goal": goal, "package": package, "model": model, "backend": backend,
        "request_config": settings, "judge_model": judging_model,
        "judge_ladder": [judging_model, *(name for name, _ in judge_fallbacks)],
        "session_id": None, "serial": None, "setup": [],
        "controller": None, "claim": None, "verdict": None, "screens": [], "route": None,
        "recording": None,
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
            if decoded.get("ok") is not True:
                # A failing AUA call does not always populate `error`; keep a bounded copy of the
                # payload so the reason survives in the log instead of reading as `error: null`.
                record["payload"] = json.dumps(decoded, ensure_ascii=False, default=str)[:2000]
            return decoded
        except Exception as exc:
            record["error"] = _error_text(exc)
            raise
        finally:
            record["duration_ms"] = (time.monotonic() - tick) * 1000
            with setup_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    session_id: str | None = None
    setup_notes: list[str] = []
    recording_started = False
    recording_path = (output / "journey.mp4").resolve()

    try:
        start_arguments: dict[str, Any] = {
            "goal": goal,
            "package": package,
            "headed": False,
            "artifacts_dir": str((output / "aua").resolve()),
            "evidence": "all",
            "launch_app": False,
            "grant_permissions": grant_permissions,
            "wait_for_lease_s": lease_wait_s,
            "provision_target": provision_target,
        }
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
        if record:
            started_recording = await call("screen_record_start", {}, "setup")
            if started_recording.get("ok") is not True:
                raise RunError("screen recording failed to start: " + json.dumps(started_recording)[:400])
            recording_started = True
            result["recording"] = {"path": str(recording_path), "started": True, "stop_ok": None}
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
        if flags or setup_flows or observation_frame(launched) is None:
            initial = await call("analyze_screen", {"source": "hierarchy", "no_cache": True}, "setup")
        else:
            assert launched is not None
            initial = launched
        if observation_frame(initial) is None:
            raise RunError("initial analyze_screen returned no fresh frame")
        schemas = await list_tools()
        tools = realapp_tools(schemas, contract=bool(session_contract))
        claims: list[dict[str, Any]] = []

        async def controller_call(name: str, arguments: dict[str, Any]) -> Any:
            if name == "session_finish":
                claims.append(copy.deepcopy(arguments))
                # This is a model completion claim, not infrastructure authority. Keep the AUA
                # session and its lease alive through the final observation, judges, recording
                # export and finally cleanup.
                return {"ok": False, "finished": False, "claim_recorded": True}
            if name == "session_progress":
                arguments = {"session_id": session_id}
            return await call_tool(name, arguments)

        compactor = FrameCompactor(max_elements=max_elements)
        report = await run_agent(
            send=send, call_tool=controller_call, tools=tools,
            system_prompt=SYSTEM + COMPACT_SYSTEM + (
                CONTRACT_SYSTEM if session_contract else REALAPP_SYSTEM
            ),
            user_prompt=goal_prompt(goal, knowledge, setup_notes),
            initial_observation=initial, model=model, output=output / "controller",
            request_config=settings, backend=backend, max_tokens=max_tokens, max_steps=max_steps,
            time_limit_s=time_limit_s, max_request_bytes=max_request_bytes,
            cost_limit_usd=cost_limit_usd, observation_filter=hosted_model_view,
            model_observation_filter=compactor, request_timeout_s=request_timeout_s,
            terminal_tools=frozenset({"session_finish"}), terminal_claim_limit=terminal_claim_limit,
            no_progress_limit=no_progress_limit,
        )
        result["controller"] = {
            key: report.get(key) for key in (
                "stop_reason", "error", "steps_consumed", "model_requests", "tool_calls_executed",
                "tool_errors", "schema_repairs", "terminal_claims", "no_progress_streak",
                "returned_models", "providers", "model_http_seconds", "tool_seconds", "duration_seconds",
                "final_model_text",
            )
        }
        result["controller"]["prompt_tokens"] = [
            (usage or {}).get("prompt_tokens") for usage in report.get("usage", [])]
        result["controller"]["compaction"] = {"frames": compactor.frames_seen, "unchanged_hits": compactor.unchanged_hits}
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
        elif stop in ("terminal_claimed", "model_text", "no_progress") and judge:
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
            judged_frames = judged_frame_sample([frame["raw"] for frame in frames], judge_frames)
            images: list[str] = []
            if vision:
                # Element text cannot answer a question about appearance. Pair each judged
                # frame with the screenshot AUA already captured for it, oldest first, so the
                # final screen is the last image the judge sees.
                shot_index = screenshot_index((output / "aua" / "manifest.json").resolve())
                for frame in [*judged_frames, final]:
                    shot = screenshot_for(shot_index, frame_fingerprint(frame))
                    encoded = encode_image(shot) if shot else None
                    if encoded:
                        images.append(encoded)
                result["vision"] = {"requested": True, "frames": len(judged_frames) + 1,
                                    "images_attached": len(images)}
            verdict = await judge_outcome_votes(
                decider, votes=judge_votes, goal=goal, final_frame=final,
                frames=judged_frames, actions=context_actions,
                images=images, contract=contract,
            )
            if stop == "no_progress" and verdict["verdict"] == "pass":
                verdict["verdict"] = "pass_with_warning"
                verdict["reasons"].insert(0, "Controller stalled on an unchanged screen before finishing.")
            verdict["controller_stop_reason"] = stop
            result["verdict"] = verdict
            result["cost"]["judge"] = {"model": judging_model, "provider": (verdict["votes"][0].get("provider") if verdict["votes"] else None),
                                       "usd": verdict["cost"], "decider": decider.report()}
        else:
            reason = report.get("error") or f"controller stopped with {stop}"
            result["verdict"] = {"oracle": "none", "verified": False, "verdict": "unverified",
                                 "reasons": [str(reason)[:300]], "controller_stop_reason": stop}
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
                stopped_recording = await call(
                    "screen_record_stop", {"path": str(recording_path)}, "cleanup"
                )
                stop_ok = stopped_recording.get("ok") is True and recording_path.is_file()
                if result["recording"] is not None:
                    result["recording"]["stop_ok"] = stop_ok
                if not stop_ok:
                    message = "recording cleanup failed: " + str(
                        stopped_recording.get("error") or stopped_recording
                    )[:500]
                    result["recording_cleanup_error"] = message
                    _mark_cleanup_failure(result, message)
                recording_started = False
            except Exception as exc:
                message = "recording cleanup failed: " + _error_text(exc)
                result["recording_cleanup_error"] = message
                _mark_cleanup_failure(result, message)
        if session_id:
            try:
                finished_session = await call(
                    "session_finish",
                    {"session_id": session_id, "allow_incomplete": True, "summary": False},
                    "cleanup",
                )
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
    parser.add_argument("--activity")
    parser.add_argument("--record", action="store_true", help="Record from the first app launch through judgement")
    parser.add_argument("--lease-wait", type=float, default=0,
                        help="Seconds to wait for a free device before provisioning another")
    parser.add_argument("--fallback-lease-wait", type=float, default=600,
                        help="Seconds to wait for an existing lease after provisioning fails")
    parser.add_argument("--no-provision", action="store_true",
                        help="Never create a new virtual target; only wait for an existing device")
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
    parser.add_argument("--contract", type=Path,
                        help="Authored acceptance criteria the judge answers against")
    parser.add_argument("--session-contract", type=Path,
                        help="Version-1 checkpoint YAML AUA itself holds the session to, so "
                             "session_finish is accepted on fresh assertion proof rather than "
                             "on the model's word. Usually the same file as --contract.")
    parser.add_argument("--vision", action="store_true",
                        help="Show the judge the captured screenshots; needs an image-capable model")
    parser.add_argument("--judge-model",
                        help="Manifest candidate used only for judgement; defaults to --model")
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
    manifest = json.loads(args.manifest.read_text())
    candidate = next((item for item in manifest["models"] if args.model in {item["id"], item["repository"]}), None)
    if candidate is None:
        parser.error("model must be a candidate in the manifest")
    request_config = copy.deepcopy(candidate.get("request_config", {}))
    if args.provider:
        request_config.setdefault("provider", {})
        request_config["provider"]["only"] = [args.provider]
        request_config["provider"]["order"] = [args.provider]
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
                    return exc.response.status_code, exc.response.headers
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
                    request_config=request_config, launch=args.launch, activity=args.activity,
                    fresh_app=args.fresh, apk=args.apk,
                    grant_permissions=args.grant_permissions, record=args.record,
                    lease_wait_s=args.lease_wait, fallback_lease_wait_s=args.fallback_lease_wait,
                    provision_target=not args.no_provision,
                    flags=parse_pairs(args.flags, what="--flags"),
                    prelaunch_setup_flows=prelaunch_setup_flows,
                    setup_flows=setup_flows,
                    contract=args.contract.read_text(encoding="utf-8") if args.contract else None,
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
