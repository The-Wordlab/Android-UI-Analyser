"""Host/model dispatch contracts with fictional tools and no external calls."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.agent_loop import HostAction, run_agent
from experiments.aua_controller.session_state import SessionState

TOOLS = [{"type": "function", "function": {"name": "tap", "parameters": {
    "type": "object", "properties": {"id": {"type": "string"}},
    "required": ["id"], "additionalProperties": False}}}]
SETTINGS = {"provider": {"only": ["fictional"], "allow_fallbacks": False,
                         "require_parameters": True, "max_price": {"prompt": 1, "completion": 1}}}


def observation(label="Start"):
    return {"screen": {"width": 400, "height": 800}, "meta": {"fingerprint": label},
            "elements": [{"id": "el:1", "text": label}]}


def action(target="el:1") -> HostAction:
    return {"tool": "tap", "arguments": {"id": target}, "reason": "The host resolved the current operation."}


def response(*, tool=False, cost=0.001):
    message = {"role": "assistant", "content": "Review requested."}
    if tool:
        message.update(content=None, reasoning_content="retained reasoning",
                       reasoning_details=[{"type": "reasoning.encrypted", "data": "opaque==", "index": 0}],
                       tool_calls=[{"type": "function", "id": "model-advice-1", "function": {
                           "name": "tap", "arguments": '{"id":"model-choice"}'},
                           "extra_content": {"google": {"thought_signature": "signature=="}}}])
    return {"choices": [{"finish_reason": "tool_calls" if tool else "stop", "message": message}],
            "usage": {"cost": cost}}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(tmp_path, host_next, execute, *, replies=None, **kwargs):
    requests = []
    responses = iter(replies or [response()])
    async def send(payload):
        requests.append(copy.deepcopy(payload))
        return copy.deepcopy(next(responses))
    state = SessionState(tmp_path / "session.json", "fictional-session", ["Reach the requested screen"])
    options = {"send": send, "call_tool": execute, "host_next": host_next, "tools": TOOLS,
               "system_prompt": "Synthetic host/model test", "user_prompt": "Follow the authored objective.",
               "initial_observation": {"ok": True, "observation": observation()}, "model": "fictional",
               "backend": "openai-compatible", "output": tmp_path / "run", "max_steps": 12,
               "session_state": state}
    options.update(kwargs)
    report = asyncio.run(run_agent(**options))
    return report, requests, state


def test_host_runs_before_model_then_resumes_from_advisory_action_result(tmp_path):
    initial = {"ok": True, "observation": observation(), "native_progress": "host-a", "session_id": "private-fictional"}
    initial_copy = copy.deepcopy(initial)
    seen, calls = [], []
    async def host_next(latest):
        seen.append(copy.deepcopy(latest))
        progress = latest["native_progress"]
        latest["tampered_by_callback"] = True
        return action(progress) if progress.startswith("host-") else None
    async def execute(tool, arguments):
        calls.append((tool, copy.deepcopy(arguments)))
        target = arguments["id"]
        arguments["id"] = "caller mutation must not alter the audit"
        next_phase = {"host-a": "host-b", "host-b": "needs-advice",
                      "model-choice": "host-c", "host-c": "review"}[target]
        return {"ok": True, "observation": observation(target), "native_progress": next_phase,
                "terminal_submission": {"fabricated_metadata": True}}
    first = response(tool=True)
    report, requests, _ = run(tmp_path, host_next, execute, replies=[first, response()], initial_observation=initial)
    assert initial == initial_copy
    assert [item[1]["id"] for item in calls] == ["host-a", "host-b", "model-choice", "host-c"]
    assert [item["native_progress"] for item in seen] == ["host-a", "host-b", "needs-advice", "host-c", "review"]
    assert seen[0]["session_id"] == "private-fictional"  # Host receives raw data, the model receives projected data.
    assert report["host_actions_selected"] == report["host_tool_calls_executed"] == 3
    assert report["model_tool_calls_executed"] == 1 and report["tool_calls_executed"] == 4
    assert report["model_requests"] == report["model_responses"] == 2
    assert report["host_next_calls"] == report["steps_consumed"] == 5
    assert len(report["host_decision_ms"]) == 5 and report["host_decision_seconds"] >= 0
    assert report["terminal_submission"] is None and report["stop_reason"] == "model_text"
    first_messages, last_messages = (request["messages"] for request in requests)
    assert [message["role"] for message in first_messages] == ["system", "user", "user", "user"]
    assert json.dumps(last_messages[:len(first_messages)]).encode() == json.dumps(first_messages).encode()
    host_messages = [message for message in last_messages if message["content"] and message["content"].startswith("Host-selected action")]
    host_events = [json.loads(message["content"].split("\n", 1)[1]) for message in host_messages]
    assert [event["host_action"]["arguments"]["id"] for event in host_events] == ["host-a", "host-b", "host-c"]
    assert [event["returned_evidence"]["evidence_ref"] for event in host_events] == ["E0001", "E0002", "E0004"]
    assert [message for message in last_messages if message["role"] == "assistant"] == [first["choices"][0]["message"]]
    assert [message["tool_call_id"] for message in last_messages if message["role"] == "tool"] == ["model-advice-1"]
    assert "private-fictional" not in json.dumps(requests) and "tampered_by_callback" not in json.dumps(requests)
    trace = records(tmp_path / "run/tool-calls.jsonl")
    assert [row["actor"] for row in trace] == ["host", "host", "model", "host"]
    assert [row["arguments"]["id"] for row in trace] == ["host-a", "host-b", "model-choice", "host-c"]
    assert all("tool_call_id" not in row for row in trace if row["actor"] == "host")
    decisions = records(tmp_path / "run/host-decisions.jsonl")
    assert [row["source_evidence_ref"] for row in decisions] == [f"E{n:04}" for n in range(5)]
    assert all(row["actor"] == "model" for row in records(tmp_path / "run/model-turns.jsonl"))
    assert json.loads((tmp_path / "run/evidence/E0000.json").read_text()) == initial_copy


@pytest.mark.parametrize("invalid", [
    {"tool": "unavailable", "arguments": {}, "reason": "A host mistake"},
    {"tool": "tap", "arguments": {}, "reason": "An unresolved binding"},
])
def test_host_schema_feedback_allows_repair_without_dispatching_invalid_action(tmp_path, invalid):
    choices, calls, seen = iter([invalid, action(), None]), [], []
    async def host_next(latest):
        seen.append(latest)
        return next(choices)
    async def execute(name, arguments):
        calls.append((name, arguments))
        return {"ok": True, "observation": observation()}
    report, requests, _ = run(tmp_path, host_next, execute)
    assert len(calls) == 1 and len(requests) == 1
    assert report["schema_repairs"] == report["host_schema_repairs"] == 1
    assert seen[1]["error"]["code"] == "invalid_tool_arguments" and seen[1]["error"]["executed"] is False
    trace = records(tmp_path / "run/tool-calls.jsonl")
    assert [row["executed"] for row in trace] == [False, True]
    assert report["host_tool_calls_executed"] == 1 and report["model_tool_calls_executed"] == 0


def test_host_invalid_repair_budget_is_shared_and_finite(tmp_path):
    async def host_next(_):
        return {"tool": "tap", "arguments": {}, "reason": "Missing binding"}
    async def forbidden(*_):
        pytest.fail("Invalid action dispatched")
    report, requests, _ = run(tmp_path, host_next, forbidden)
    assert not requests and report["tool_calls_executed"] == 0
    assert report["repair_count"] == report["host_schema_repairs"] == 3
    assert report["error"] == "controller repair budget exhausted"


def test_no_progress_guard_blocks_host_and_model_retry_without_duplicate_dispatch(tmp_path):
    host_calls, dispatches = 0, []
    async def host_next(_):
        nonlocal host_calls
        host_calls += 1
        return action("model-choice") if host_calls <= 4 else None
    async def execute(name, arguments):
        dispatches.append((name, arguments))
        return {"ok": True, "observation": observation()}
    report, requests, state = run(tmp_path, host_next, execute, replies=[response(tool=True), response()])
    assert len(dispatches) == 3
    assert report["host_rejections"] == 2 and report["host_tool_calls_executed"] == 3
    assert report["model_tool_calls_executed"] == 0 and len(requests) == 2
    trace = records(tmp_path / "run/tool-calls.jsonl")
    assert [row["executed"] for row in trace] == [True, True, True, False, False]
    assert trace[-1]["actor"] == "model" and trace[-2]["actor"] == "host"
    assert state.context()["checks"][0]["status"] == "pending"
    assert json.loads(requests[-1]["messages"][-1]["content"])["error"]["executed"] is False


def test_host_action_timeout_is_unknown_and_never_retried(tmp_path):
    called = []
    async def host_next(_):
        return action()
    async def slow(name, arguments):
        called.append((name, arguments))
        await asyncio.sleep(10)
    report, requests, _ = run(tmp_path, host_next, slow, request_timeout_s=0.01)
    assert len(called) == 1 and not requests
    assert report["unknown_tool_outcomes"] == report["host_tool_calls_executed"] == 1
    assert report["error"] == "TimeoutError" and report["caller_owns_verification_and_cleanup"] is True
    call = records(tmp_path / "run/tool-calls.jsonl")[0]
    assert call["dispatch_started"] is True and call["execution_outcome"] == "unknown"
    assert "result" not in call and "executed" not in call


def test_callback_exception_after_dispatch_is_unknown_and_never_retried(tmp_path):
    called = []
    async def host_next(_):
        return action()
    async def interrupted(name, arguments):
        called.append((name, arguments))
        raise RuntimeError("Transport lost after dispatch")
    report, requests, _ = run(tmp_path, host_next, interrupted)
    assert len(called) == 1 and not requests
    assert report["unknown_tool_outcomes"] == 1
    assert records(tmp_path / "run/tool-calls.jsonl")[0]["execution_outcome"] == "unknown"


def test_host_decision_timeout_is_not_an_unknown_device_outcome(tmp_path):
    async def slow(_):
        await asyncio.sleep(10)
    async def forbidden(*_):
        pytest.fail("No host action was returned")
    report, requests, _ = run(tmp_path, slow, forbidden, request_timeout_s=0.01)
    assert not requests and report["tool_calls_executed"] == report["unknown_tool_outcomes"] == 0
    assert report["error"] == "TimeoutError"
    assert records(tmp_path / "run/host-decisions.jsonl")[0]["error"] == "TimeoutError"


def test_host_steps_are_bounded_and_byte_limit_blocks_only_the_next_model_request(tmp_path):
    async def host_next(_):
        return action()
    async def execute(*_):
        return {"ok": True, "observation": observation(), "detail": "x" * 20_000}
    report, requests, _ = run(tmp_path, host_next, execute, max_steps=2, max_request_bytes=100)
    assert not requests and report["host_tool_calls_executed"] == report["steps_consumed"] == 2
    assert report["error"] == "controller step budget exhausted"
    choices = iter([action(), action(), None])
    async def host_then_advice(_):
        return next(choices)
    report, requests, _ = run(tmp_path / "bytes", host_then_advice, execute, max_request_bytes=10_000)
    assert not requests and report["host_tool_calls_executed"] == 2
    assert "history was not truncated" in report["error"]


def test_inference_cost_limit_allows_host_resume_but_blocks_next_model_request(tmp_path):
    decisions, dispatched = 0, []
    async def host_next(_):
        nonlocal decisions
        decisions += 1
        return action() if decisions == 2 else None
    async def execute(name, arguments):
        dispatched.append((name, arguments))
        return {"ok": True, "observation": observation()}
    report, requests, _ = run(tmp_path, host_next, execute, replies=[response(tool=True, cost=0.02)],
                              backend="openrouter", request_config=SETTINGS, cost_limit_usd=0.01)
    assert len(requests) == 1 and len(dispatched) == 2 and decisions == 3
    assert report["host_tool_calls_executed"] == 1 and "cost limit reached" in report["error"]


def test_host_can_finish_known_work_after_inference_cost_limit(tmp_path):
    decisions = 0
    async def host_next(_):
        nonlocal decisions
        decisions += 1
        return None if decisions == 1 else {"tool": "submit_result", "arguments": {}, "reason": "Host work is complete; request review."}
    async def execute(*_):
        return {"ok": True, "accepted_for_independent_review": True}
    terminal = {"type": "function", "function": {"name": "submit_result", "parameters": {"type": "object"}}}
    report, requests, _ = run(tmp_path, host_next, execute, replies=[response(tool=True, cost=0.02)],
        tools=TOOLS + [terminal], terminal_tools=frozenset({"submit_result"}),
        backend="openrouter", request_config=SETTINGS, cost_limit_usd=0.01)
    assert len(requests) == report["model_tool_calls_executed"] == report["host_tool_calls_executed"] == 1
    assert report["stop_reason"] == "terminal_tool" and report["terminal_submission"]["actor"] == "host"
    assert report["cost_accounting"]["limit_reached"] is True and report["error"] is None


def test_only_explicit_executed_terminal_tool_can_finish_host_run(tmp_path):
    async def host_next(_):
        return action()
    async def execute(*_):
        return {"ok": True, "accepted_for_independent_review": True}
    report, requests, _ = run(tmp_path, host_next, execute, terminal_tools=frozenset({"tap"}))
    assert not requests and report["host_tool_calls_executed"] == 1
    assert report["terminal_submission"]["actor"] == "host"
    assert report["stop_reason"] == "terminal_tool" and report["report_is_untrusted"] is True


@pytest.mark.parametrize("invalid", [[], {"tool": "tap"},
    {"tool": "tap", "arguments": {}, "reason": "", "terminal": True}])
def test_malformed_host_envelope_fails_without_model_or_action(tmp_path, invalid):
    async def host_next(_):
        return invalid
    async def forbidden(*_):
        pytest.fail("Malformed host envelope dispatched")
    report, requests, _ = run(tmp_path, host_next, forbidden)
    assert not requests and report["tool_calls_executed"] == 0
    assert report["error"] == "invalid host action envelope"
