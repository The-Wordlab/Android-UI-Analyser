"""Generic bounded native-tool controller; callers own lifecycle and judgement.

This module does not start sessions, prepare apps, choose test contracts, or issue
QA verdicts. The injected transport must validate its endpoint/authentication
before callers acquire a device. Use a dedicated private output subdirectory.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypedDict

import jsonschema
from experiments.aua_controller.hosted import (
    BACKENDS,
    CostGuard,
    assistant_message,
    configure_payload,
    validate_request_config,
)
from experiments.aua_controller.hosted_projection import hosted_model_view
from experiments.aua_controller.run_live import RunError, _error_text, completion, tool_result
from experiments.aua_controller.session_state import observation_frame

SCHEMA_REPAIR_BUDGET = 3


class HostAction(TypedDict):
    """One fully bound host choice; no authority beyond the offered tools."""

    tool: str
    arguments: dict[str, Any]
    reason: str


class AgentConversation:
    """One caller-owned transcript, reusable only across closed run boundaries.

    The system/model/backend/request settings remain bound. Offered tools may
    change at an explicit new run, leaving prior calls as historical records.
    A failed or interrupted run poisons reuse; this holder never repairs history.
    The caller owns its device/session lifetime; this is not a session lease.
    """

    def __init__(self):
        self._messages: list[dict[str, Any]] = []
        self._binding: dict[str, Any] | None = None
        self._active = False
        self._blocked = False

    def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._messages)

    def _closed(self) -> bool:
        pending = set()
        for message in self._messages:
            if message.get("role") == "tool":
                identity = message.get("tool_call_id")
                if not isinstance(identity, str) or identity not in pending:
                    return False
                pending.remove(identity)
            else:
                if pending:
                    return False
                calls = message.get("tool_calls") or []
                if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
                    return False
                ids = [call.get("id") for call in calls]
                if any(not isinstance(identity, str) for identity in ids) or len(set(ids)) != len(ids):
                    return False
                pending.update(ids)
        return not pending

    def _validate(self, binding):
        if self._active:
            raise RunError("conversation is already in use")
        if self._blocked or not self._closed():
            raise RunError("conversation cannot resume after an unresolved or failed run")
        if self._binding is not None and self._binding != binding:
            raise RunError("conversation system/model/backend/request_config binding mismatch")

    def _start(self, binding):
        self._validate(binding)
        self._binding = copy.deepcopy(binding)
        self._active = True
        return self._messages

    def _finish(self, completed):
        self._active = False
        self._blocked = not completed or not self._closed()


def context_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Return claim changes and changed host facts, without repeating UI/history.

    Check fields merge by stable ID; other listed fields replace their prior
    values. Append the patch to a new result, never rewrite a cached prefix.
    """
    before = {item["id"]: item for item in previous.get("checks", [])}
    checks = []
    for item in current.get("checks", []):
        changed = {key: copy.deepcopy(item[key]) for key in ("status", "evidence_refs", "note")
                   if key in item and (key not in before.get(item["id"], {})
                                      or item[key] != before[item["id"]][key])}
        if changed:
            checks.append({"id": item["id"], **changed})
    delta = {key: copy.deepcopy(current[key]) for key in ("phase_id", "knowledge", "route_attempts")
             if key in current and (key not in previous or current[key] != previous[key])}
    if checks:
        delta.update({"checks_are_untrusted_claims": True, "checks": checks})
    return delta


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _multiple_calls(response: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Identify repairable multi-call envelopes without selecting or executing a call."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None  # The ordinary completion validator reports the malformed envelope.
    choice = choices[0]
    message = choice.get("message")
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or len(calls) <= 1:
        return None
    native_calls = []
    ids = set()
    for call in calls:
        # Reuse the same role, finish reason, ID and serialized-JSON validation.
        _, native = completion({"choices": [{**choice, "message": {**message, "tool_calls": [call]}}]})
        if native is None or not isinstance(native["name"], str) or not native["name"].strip():
            raise RunError("invalid native function name in multi-call response")
        if native["id"] in ids:
            raise RunError("duplicate native tool call IDs in multi-call response")
        ids.add(native["id"])
        native_calls.append(native)
    return message, native_calls


async def run_agent(
    *,
    send: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    call_tool: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    initial_observation: dict[str, Any],
    model: str,
    output: Path,
    request_config: dict[str, Any] | None = None,
    backend: str = "openrouter",
    max_tokens: int = 32768,
    max_steps: int = 64,
    time_limit_s: float = 600,
    max_request_bytes: int = 200_000,
    cost_limit_usd: float = 0.10,
    observation_filter: Callable[[Any], Any] = hosted_model_view,
    model_observation_filter: Callable[[Any], Any] | None = None,
    request_timeout_s: float = 60,
    terminal_tools: frozenset[str] = frozenset(),
    session_state: Any = None,
    evidence_namespace: str = "",
    host_next: Callable[[dict[str, Any]], Awaitable[HostAction | None]] | None = None,
    conversation: AgentConversation | None = None,
    terminal_claim_limit: int | None = None,
    no_progress_limit: int | None = None,
) -> dict[str, Any]:
    """Return trace metadata and an untrusted report, never a scenario verdict.

    ``terminal_tools`` are caller-defined submission tools. A valid call stops
    only when the caller returns ``ok: true``; a rejected submission is feedback.
    Raw decoded tool results stay in numbered local evidence files. The model
    receives filtered JSON with an ``evidence_ref`` pointing to that exact file.
    A reference identifies a record, not proof that the record verifies a claim.
    ``model_observation_filter`` optionally narrows new model-facing results after
    privacy filtering, without changing the observation filter used by host state.
    ``host_next`` receives the latest raw result before each model opportunity.
    A host action uses the same execution path and consumes one step; None asks
    the model. Host decisions append user events, never invented native calls.
    ``conversation`` retains exact prior messages across explicit row invocations;
    each row adds its objective/observation while budgets remain invocation-local.
    ``terminal_claim_limit`` stops the run with ``stop_reason == "terminal_claimed"`` once
    the model has called a terminal tool that many times without the caller accepting it:
    on a real application there is often no machine contract to accept, so the claim
    itself is the signal and the caller judges it afterwards. ``no_progress_limit`` stops
    with ``"no_progress"`` when that many consecutive executed calls return the same
    screen fingerprint. Both default to off and report their counters.
    """
    if backend not in BACKENDS:
        raise RunError("unknown controller backend")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0
           for value in (max_tokens, max_steps, time_limit_s, max_request_bytes, request_timeout_s)):
        raise RunError("controller budgets must be positive finite numbers")
    if any(type(value) is not int for value in (max_tokens, max_steps, max_request_bytes)):
        raise RunError("token, step and byte budgets must be integers")
    if not isinstance(initial_observation, dict):
        raise RunError("initial observation must be an object")
    if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
        raise RunError("tools must be a list of native function schemas")
    if request_config is not None and not isinstance(request_config, dict):
        raise RunError("request_config must be an object")
    if host_next is not None and not callable(host_next):
        raise RunError("host_next must be an asynchronous callable")
    if model_observation_filter is not None and not callable(model_observation_filter):
        raise RunError("model_observation_filter must be callable")
    if conversation is not None and not isinstance(conversation, AgentConversation):
        raise RunError("conversation must be an AgentConversation holder")
    for value in (terminal_claim_limit, no_progress_limit):
        if value is not None and (type(value) is not int or value <= 0):
            raise RunError("terminal_claim_limit and no_progress_limit must be positive integers")
    if terminal_claim_limit is not None and not terminal_tools:
        raise RunError("terminal_claim_limit needs terminal tools")
    offered = copy.deepcopy(tools)
    schemas: dict[str, dict[str, Any]] = {}
    for definition in offered:
        function = definition.get("function") or {}
        if not isinstance(function, dict):
            raise RunError("tools must contain native function objects")
        name = function.get("name")
        schema = function.get("parameters")
        if (definition.get("type") != "function" or not isinstance(name, str) or not name
                or name in schemas or not isinstance(schema, dict)):
            raise RunError("tools must have unique native function names and parameter schemas")
        jsonschema.validators.validator_for(schema).check_schema(schema)
        schemas[name] = schema
    if not schemas or not set(terminal_tools) <= schemas.keys():
        raise RunError("terminal tools must belong to a nonempty offered tool set")
    hosted = backend == "openrouter"
    settings = validate_request_config(request_config or {}) if hosted else copy.deepcopy(request_config or {})
    if not hosted and set(settings) - {"temperature", "chat_template_kwargs"}:
        raise RunError("local request_config supports only temperature and chat_template_kwargs")
    binding = {"system_prompt": system_prompt, "model": model, "backend": backend, "request_config": settings}
    if conversation is not None:
        conversation._validate(binding)
    guard = CostGuard(cost_limit_usd) if hosted else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RunError("controller output directory must be empty")
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    started = time.monotonic()
    deadline = started + time_limit_s
    report: dict[str, Any] = {
        "format": "aua-generic-controller-v1", "model_requested": model, "backend": backend,
        "request_config": settings,
        "limits": {"steps": max_steps, "seconds": time_limit_s, "request_seconds": request_timeout_s,
                   "max_tokens": max_tokens, "request_bytes": max_request_bytes},
        "schema_repair_budget": SCHEMA_REPAIR_BUDGET, "schema_repairs": 0,
        "repair_budget": SCHEMA_REPAIR_BUDGET, "repair_count": 0, "protocol_repairs": 0,
        "model_requests": 0, "model_responses": 0, "tool_calls_executed": 0,
        "host_actions_selected": 0, "host_tool_calls_executed": 0, "model_tool_calls_executed": 0,
        "host_next_calls": 0, "host_decision_ms": [], "host_schema_repairs": 0, "steps_consumed": 0,
        "conversation_reused": conversation is not None and bool(conversation._messages),
        "tool_errors": 0, "unknown_tool_outcomes": 0,
        "host_rejections": 0, "session_managed": session_state is not None,
        "model_request_ms": [], "tool_call_ms": [], "usage": [], "evidence": [],
        "returned_models": [], "providers": [], "warnings": [],
        "stop_reason": None, "error": None, "final_model_text": None, "terminal_submission": None,
        "terminal_claims": 0, "terminal_claim_limit": terminal_claim_limit,
        "no_progress_streak": 0, "no_progress_limit": no_progress_limit, "last_fingerprint": None,
        "report_is_untrusted": True, "caller_owns_verification_and_cleanup": True,
    }

    def record_evidence(result: dict[str, Any], kind: str, tool: str | None = None) -> dict[str, Any]:
        ref = f"E{len(report['evidence']):04d}"
        path = evidence_dir / f"{ref}.json"
        data = json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n"
        path.write_bytes(data)
        report["evidence"].append({"ref": ref, "path": str(path.relative_to(output)), "kind": kind,
                                   "tool": tool, "sha256": hashlib.sha256(data).hexdigest()})
        projected = observation_filter(copy.deepcopy(result))
        if model_observation_filter is not None:
            projected = model_observation_filter(projected)
        if not isinstance(projected, dict):
            raise RunError("observation filter must return an object")
        citable = observation_frame(result) is not None
        return {**projected, "evidence_ref": ref, "citable_observation": citable,
                "evidence_kind": "observation" if citable else "receipt"}

    def remaining() -> float:
        available = min(request_timeout_s, deadline - time.monotonic())
        if available <= 0:
            raise RunError("controller elapsed time budget exhausted")
        return available

    def append_result(visible, actor, action, native):
        nonlocal last_context
        if session_state is not None:
            current_context = observation_filter(session_state.context())
            delta = context_delta(last_context, current_context)
            if delta:
                visible["host_context_delta"] = delta
            last_context = current_context
        if actor == "host":
            event = {"actor": "host", "host_action": observation_filter(copy.deepcopy(action)),
                     "returned_evidence": visible}
            messages.append({"role": "user", "content": "Host-selected action and returned evidence (not a model tool call):\n"
                             + json.dumps(event, ensure_ascii=False)})
        else:
            messages.append({"role": "tool", "tool_call_id": native["id"], "name": native["name"],
                             "content": json.dumps(visible, ensure_ascii=False)})

    conversation_started = False
    try:
        messages = conversation._start(binding) if conversation is not None else []
        conversation_started = conversation is not None
        initial = record_evidence(initial_observation, "initial_observation")
        latest_result = copy.deepcopy(initial_observation)
        latest_ref = initial["evidence_ref"]
        initial_content = user_prompt
        if conversation is not None or evidence_namespace:
            initial_content += ("\n\nEvidence namespace for this invocation: "
                                + json.dumps(observation_filter(evidence_namespace), ensure_ascii=False)
                                + ". Bare E#### references below are local to this invocation.")
        initial_content += "\n\nInitial observation:\n" + json.dumps(initial, ensure_ascii=False)
        last_context = None
        if session_state is not None:
            session_state.observe("initial_observation", {}, observation_filter(initial_observation), evidence_namespace + initial["evidence_ref"])
            last_context = observation_filter(session_state.context())
            initial_content += (
                "\n\nLater tool results may include host_context_delta: merge check fields by stable ID; "
                "replace listed phase_id, knowledge and route_attempts. Omitted fields are unchanged. "
                "Checklist claims are never independent verdicts. Use the latest returned observation for targets.\n"
                "Host-owned session ledger (claims are not independent verdicts):\n"
                + json.dumps(last_context, ensure_ascii=False)
            )
        if not messages:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": initial_content})
        for step in range(max_steps):
            report["steps_consumed"] = step + 1
            remaining()
            action = None
            if host_next is not None:
                decision = {"step": step, "actor": "host", "source_evidence_ref": latest_ref}
                tick = time.monotonic()
                report["host_next_calls"] += 1
                try:
                    action = await asyncio.wait_for(host_next(copy.deepcopy(latest_result)), remaining())
                    if action is not None:
                        if (not isinstance(action, dict) or set(action) != {"tool", "arguments", "reason"}
                                or not isinstance(action.get("tool"), str) or not action["tool"].strip()
                                or not isinstance(action.get("arguments"), dict)
                                or not isinstance(action.get("reason"), str) or not action["reason"].strip()
                                or len(action["reason"]) > 1024):
                            raise RunError("invalid host action envelope")
                        action = copy.deepcopy(action)
                        json.dumps(action, allow_nan=False)
                        report["host_actions_selected"] += 1
                    decision["action"] = action
                except Exception as exc:
                    decision["error"] = _error_text(exc)
                    raise
                finally:
                    decision["duration_ms"] = (time.monotonic() - tick) * 1000
                    report["host_decision_ms"].append(decision["duration_ms"])
                    _append(output / "host-decisions.jsonl", decision)
            actor = "host" if action is not None else "model"
            native = None
            if action is not None:
                name, arguments = action["tool"], action["arguments"]
            else:
                payload = {"model": model, "messages": copy.deepcopy(messages), "tools": offered,
                           "tool_choice": "auto", "parallel_tool_calls": False, "stream": False,
                           "max_tokens": max_tokens, "temperature": 0}
                if hosted:
                    payload = configure_payload(payload, settings)
                    guard.before_request()
                else:
                    payload.update(copy.deepcopy(settings))
                if len(json.dumps(payload).encode()) > max_request_bytes:
                    # The device evidence collected so far remains valid even when the hosted
                    # conversation cannot safely grow again. Return a bounded stop so the caller
                    # can judge that evidence; treating this as an infrastructure exception used
                    # to suppress the judge and block every later consumer of an otherwise healthy
                    # continuous session.
                    report["stop_reason"] = "conversation_budget"
                    report["warnings"].append(
                        "controller conversation reached its byte budget before another model turn"
                    )
                    break
                turn: dict[str, Any] = {"step": step, "actor": "model", "request": copy.deepcopy(payload)}
                tick = time.monotonic()
                report["model_requests"] += 1
                try:
                    response = await asyncio.wait_for(send(payload), remaining())
                    if not isinstance(response, dict):
                        raise RunError("model response must be a JSON object")
                    turn["response"] = response
                    report["model_responses"] += 1
                    report["usage"].append(response.get("usage"))
                    for field, collection in (("model", "returned_models"), ("provider", "providers")):
                        value = response.get(field)
                        if isinstance(value, str) and value not in report[collection]:
                            report[collection].append(value)
                    if response.get("warnings"):
                        report["warnings"].append(response["warnings"])
                    if guard is not None:
                        guard.consume(response)
                except Exception as exc:
                    turn["error"] = _error_text(exc)
                    raise
                finally:
                    turn["request_ms"] = (time.monotonic() - tick) * 1000
                    report["model_request_ms"].append(turn["request_ms"])
                    _append(output / "model-turns.jsonl", turn)
                multiple = _multiple_calls(response)
                if multiple is not None:
                    message, native_calls = multiple
                    if report["repair_count"] >= SCHEMA_REPAIR_BUDGET:
                        raise RunError("controller repair budget exhausted")
                    report["repair_count"] += 1
                    report["protocol_repairs"] += 1
                    messages.append(assistant_message(message))
                    feedback = {"ok": False, "error": {
                        "code": "multiple_tool_calls", "executed": False,
                        "message": "None of the calls in this response were executed. Retry with exactly one native tool call; wait for its result before choosing another.",
                    }}
                    for native in native_calls:
                        _append(output / "controller-feedback.jsonl", {
                            "step": step, "tool_call_id": native["id"], "tool": native["name"], "feedback": feedback,
                        })
                        messages.append({"role": "tool", "tool_call_id": native["id"], "name": native["name"],
                                         "content": json.dumps(feedback)})
                    continue
                message, native = completion(response)
                messages.append(assistant_message(message))
                if native is None:
                    report["final_model_text"] = message["content"]
                    report["stop_reason"] = "model_text"
                    break
                name, arguments = native["name"], native["arguments"]
            invalid = None
            if not isinstance(name, str) or name not in schemas:
                invalid = "Unavailable tool; use a supplied tool name."
            else:
                try:
                    jsonschema.validate(arguments, schemas[name])
                except jsonschema.ValidationError as exc:
                    invalid = str(exc.message)[:768]
            if invalid is not None:
                if report["repair_count"] >= SCHEMA_REPAIR_BUDGET:
                    raise RunError("controller repair budget exhausted")
                report["repair_count"] += 1
                report["schema_repairs"] += 1
                feedback = {"ok": False, "error": {"code": "invalid_tool_arguments", "message": invalid,
                                                   "executed": False}}
                entry = {"step": step, "actor": actor, "tool": name, "feedback": feedback}
                if actor == "model":
                    entry["tool_call_id"] = native["id"]
                else:
                    report["host_schema_repairs"] += 1
                _append(output / "controller-feedback.jsonl", entry)
                if actor == "model":
                    messages.append({"role": "tool", "tool_call_id": native["id"], "name": name if isinstance(name, str) else "invalid_tool",
                                     "content": json.dumps(feedback)})
                    continue
            call: dict[str, Any] = {"step": step, "actor": actor, "tool": name,
                                    "arguments": copy.deepcopy(arguments)}
            if actor == "model":
                call["tool_call_id"] = native["id"]
            else:
                call["reason"] = action["reason"]
            tick = time.monotonic()
            try:
                timeout = remaining()
                refusal = session_state.rejection(name, arguments) if session_state is not None and invalid is None else None
                if invalid is not None:
                    result = feedback
                    call["executed"] = False
                elif refusal is not None:
                    result = refusal
                    report["host_rejections"] += 1
                    call["executed"] = False
                else:
                    report["tool_calls_executed"] += 1
                    report[f"{actor}_tool_calls_executed"] += 1
                    call["dispatch_started"] = True
                    result = tool_result(await asyncio.wait_for(call_tool(name, copy.deepcopy(arguments)), timeout))
                    call["executed"] = True
                call["result"] = result
                if result.get("ok") is False or result.get("error") or result.get("mcp_is_error"):
                    report["tool_errors"] += 1
                visible = record_evidence(result, "tool_result", name)
                call["evidence_ref"] = visible["evidence_ref"]
                latest_result, latest_ref = copy.deepcopy(result), visible["evidence_ref"]
                if session_state is not None:
                    session_state.observe(name, arguments, observation_filter(result), evidence_namespace + visible["evidence_ref"])
            except asyncio.CancelledError:
                call["error"] = "CancelledError"
                if call.get("dispatch_started") and call.get("executed") is not True:
                    call["execution_outcome"] = "unknown"
                    report["unknown_tool_outcomes"] += 1
                raise
            except Exception as exc:
                call["error"] = _error_text(exc)
                if call.get("dispatch_started") and call.get("executed") is not True:
                    call["execution_outcome"] = "unknown"
                    report["unknown_tool_outcomes"] += 1
                raise
            finally:
                call["duration_ms"] = (time.monotonic() - tick) * 1000
                report["tool_call_ms"].append(call["duration_ms"])
                _append(output / "tool-calls.jsonl", call)
            append_result(visible, actor, action, native)
            if (call.get("executed") is True and name in terminal_tools and result.get("ok") is True
                    and not result.get("error") and not result.get("mcp_is_error")):
                report["terminal_submission"] = {"tool": name, "arguments": arguments,
                                                  "result": result, "evidence_ref": visible["evidence_ref"], "actor": actor}
                report["stop_reason"] = "terminal_tool"
                break
            if call.get("executed") is True and name in terminal_tools:
                report["terminal_claims"] += 1
                if terminal_claim_limit is not None and report["terminal_claims"] >= terminal_claim_limit:
                    report["terminal_submission"] = {"tool": name, "arguments": arguments, "result": result,
                                                      "evidence_ref": visible["evidence_ref"], "actor": actor,
                                                      "accepted": False}
                    report["stop_reason"] = "terminal_claimed"
                    break
            if call.get("executed") is True and no_progress_limit is not None:
                frame = observation_frame(result)
                fingerprint = frame["meta"]["fingerprint"] if frame is not None else None
                if fingerprint is not None and fingerprint == report["last_fingerprint"]:
                    report["no_progress_streak"] += 1
                else:
                    report["no_progress_streak"] = 0
                if fingerprint is not None:
                    report["last_fingerprint"] = fingerprint
                if report["no_progress_streak"] >= no_progress_limit:
                    report["stop_reason"] = "no_progress"
                    break
        else:
            # A step limit is a controller boundary, not proof that the product failed. Preserve
            # the collected frames for the independent judge just like a no-progress stop.
            report["stop_reason"] = "step_budget"
            report["warnings"].append("controller reached its step budget before finishing")
    except asyncio.CancelledError:
        report["error"] = "CancelledError"
        report["stop_reason"] = "cancelled"
        raise
    except Exception as exc:
        report["error"] = _error_text(exc)
        report["stop_reason"] = "error"
    finally:
        if conversation_started:
            conversation._finish(report["stop_reason"] in {
                "terminal_tool", "model_text", "terminal_claimed", "no_progress",
                "conversation_budget", "step_budget",
            })
            report["conversation_messages"] = len(messages)
        if guard is not None:
            report["cost_accounting"] = guard.report()
        report["model_http_seconds"] = sum(report["model_request_ms"]) / 1000
        report["host_decision_seconds"] = sum(report["host_decision_ms"]) / 1000
        report["tool_seconds"] = sum(report["tool_call_ms"]) / 1000
        report["duration_seconds"] = time.monotonic() - started
        (output / "controller-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
