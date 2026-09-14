"""Closed row-boundary continuity with synthetic native conversations."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.agent_loop import AgentConversation, run_agent
from experiments.aua_controller.run_live import RunError
from experiments.aua_controller.session_state import SessionState


def tools(row="one"):
    return [{"type": "function", "function": {"name": "submit_result", "parameters": {
        "type": "object", "properties": {"row": {"const": row}}, "required": ["row"],
        "additionalProperties": False}}}]


def frame(label="Ready"):
    return {"screen": {"width": 400, "height": 800}, "meta": {"fingerprint": label},
            "elements": [{"id": "el:1", "text": label}]}


def options(tmp_path, holder, send, call_tool, **changes):
    return {"send": send, "call_tool": call_tool, "tools": tools(),
            "system_prompt": "Fixed synthetic group system", "user_prompt": "First synthetic objective.",
            "initial_observation": frame(), "model": "fictional-model", "backend": "openai-compatible",
            "output": tmp_path, "conversation": holder, "max_steps": 8,
            "terminal_tools": frozenset({"submit_result"}), **changes}


def model_call(row="one"):
    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
        "reasoning_content": "Preserved native reasoning", "reasoning_details": [{"data": "opaque=="}],
        "tool_calls": [{"type": "function", "id": "native-" + row, "function": {
            "name": "submit_result", "arguments": json.dumps({"row": row})},
            "extra_content": {"google": {"thought_signature": "signature=="}}}]}}]}


def model_text():
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "Ready for independent review.", "tool_calls": None}}]}


def test_two_rows_keep_exact_transcript_and_close_terminal_pair_before_new_objective(tmp_path):
    holder, requests, dispatched = AgentConversation(), [], []
    state = SessionState(tmp_path / "session.json", "fictional-session", ["First authored clause", "Second authored clause"])
    replies = iter([model_call("one"), model_call("two")])
    async def send(payload):
        requests.append(copy.deepcopy(payload))
        return next(replies)
    async def execute(name, arguments):
        dispatched.append((name, arguments))
        return {"ok": True, "accepted_for_independent_review": True, "observation": frame(arguments["row"])}
    state.begin_phase("fictional-row-one")
    first = asyncio.run(run_agent(**options(tmp_path / "first", holder, send, execute,
        evidence_namespace="fictional-row-one/", session_state=state)))
    first_snapshot = holder.snapshot()
    assert first["stop_reason"] == "terminal_tool" and first["conversation_reused"] is False
    assert first_snapshot[-1]["role"] == "tool" and first_snapshot[-1]["tool_call_id"] == "native-one"
    first_snapshot[0]["content"] = "Detached snapshot mutation"
    assert holder.snapshot()[0]["content"] == "Fixed synthetic group system"
    first_snapshot = holder.snapshot()
    review = "Independent first-row review: BLOCKED; missing evidence remains unverified."
    state.begin_phase("fictional-row-two")
    second = asyncio.run(run_agent(**options(tmp_path / "second", holder, send, execute,
        tools=tools("two"), user_prompt="Second synthetic objective. " + review,
        evidence_namespace="fictional-row-two/", initial_observation=frame("Follow-up"), session_state=state)))
    assert second["stop_reason"] == "terminal_tool" and second["conversation_reused"] is True
    assert second["model_requests"] == first["model_requests"] == 1  # Invocation-local budgets/accounting.
    second_request = requests[1]["messages"]
    assert json.dumps(second_request[:len(first_snapshot)]).encode() == json.dumps(first_snapshot).encode()
    assert sum(message["role"] == "system" for message in holder.snapshot()) == 1
    assert review in second_request[-1]["content"] and "Follow-up" in second_request[-1]["content"]
    assert '"fictional-row-one/"' in second_request[1]["content"]
    assert '"fictional-row-two/"' in second_request[-1]["content"]
    assert "fictional-row-two/E0000" in second_request[-1]["content"]
    assert second_request[2] == model_call("one")["choices"][0]["message"]
    assert requests[0]["tools"] == tools("one") and requests[1]["tools"] == tools("two")
    assert dispatched == [("submit_result", {"row": "one"}), ("submit_result", {"row": "two"})]
    assert [m["tool_call_id"] for m in holder.snapshot() if m["role"] == "tool"] == ["native-one", "native-two"]


@pytest.mark.parametrize("changes", [
    {"model": "different-model"}, {"system_prompt": "Different system"},
    {"request_config": {"temperature": 1}},
    {"backend": "openrouter", "request_config": {"provider": {
        "only": ["fictional"], "allow_fallbacks": False, "require_parameters": True,
        "max_price": {"prompt": 1, "completion": 1}}}},
])
def test_bound_request_identity_mismatch_fails_before_next_request(tmp_path, changes):
    holder, requests = AgentConversation(), []
    async def send(payload):
        requests.append(payload)
        return model_text()
    async def forbidden(*_):
        pytest.fail("No action requested")
    first = asyncio.run(run_agent(**options(tmp_path / "first", holder, send, forbidden)))
    assert first["stop_reason"] == "model_text"
    snapshot = holder.snapshot()
    with pytest.raises(RunError, match="binding mismatch"):
        asyncio.run(run_agent(**options(tmp_path / "second", holder, send, forbidden, **changes)))
    assert len(requests) == 1 and holder.snapshot() == snapshot
    assert not (tmp_path / "second").exists()


def test_unknown_action_outcome_poisoned_conversation_cannot_resume(tmp_path):
    holder, requests, actions = AgentConversation(), [], []
    async def send(payload):
        requests.append(payload)
        return model_call()
    async def slow(name, arguments):
        actions.append((name, arguments))
        await asyncio.sleep(10)
    report = asyncio.run(run_agent(**options(tmp_path / "first", holder, send, slow, request_timeout_s=0.01)))
    assert report["unknown_tool_outcomes"] == 1 and report["stop_reason"] == "error"
    assert holder.snapshot()[-1]["role"] == "assistant"  # No invented tool completion.
    with pytest.raises(RunError, match="unresolved or failed run"):
        asyncio.run(run_agent(**options(tmp_path / "second", holder, send, slow)))
    assert len(requests) == len(actions) == 1


def test_same_holder_cannot_be_used_by_concurrent_invocations(tmp_path):
    holder = AgentConversation()
    async def test():
        started, finish = asyncio.Event(), asyncio.Event()
        async def send(_):
            started.set()
            await finish.wait()
            return model_text()
        async def forbidden(*_):
            pytest.fail("No action requested")
        first = asyncio.create_task(run_agent(**options(tmp_path / "first", holder, send, forbidden)))
        await started.wait()
        with pytest.raises(RunError, match="already in use"):
            await run_agent(**options(tmp_path / "second", holder, send, forbidden))
        finish.set()
        return await first
    assert asyncio.run(test())["stop_reason"] == "model_text"


def test_second_row_byte_budget_counts_prior_history_without_truncation(tmp_path):
    holder, requests = AgentConversation(), []
    async def send(payload):
        requests.append(payload)
        return model_text()
    async def forbidden(*_):
        pytest.fail("No action requested")
    asyncio.run(run_agent(**options(tmp_path / "first", holder, send, forbidden, user_prompt="x" * 20_000)))
    prior = holder.snapshot()
    second = asyncio.run(run_agent(**options(tmp_path / "second", holder, send, forbidden, max_request_bytes=10_000)))
    assert len(requests) == 1
    assert second["error"] is None and second["stop_reason"] == "conversation_budget"
    assert "conversation reached its byte budget" in second["warnings"][0]
    assert holder.snapshot()[:len(prior)] == prior
