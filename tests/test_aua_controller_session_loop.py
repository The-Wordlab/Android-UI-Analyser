from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.agent_loop import context_delta, run_agent
from experiments.aua_controller.session_state import SessionState

LEDGER_LABEL = "Host-owned session ledger (claims are not independent verdicts):\n"


def frame():
    return {"screen": {"width": 400, "height": 800},
            "elements": [{"id": "el:1", "text": "Continue"}],
            "meta": {"fingerprint": "stable"}}


def native(name, parameters):
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def test_delta_reports_only_changed_claim_fields_and_host_facts():
    before = {"checks": [{"id": "C001", "clause": "Reach the next screen", "status": "pending",
                          "evidence_refs": [], "note": ""},
                         {"id": "C002", "clause": "Return safely", "status": "pending"}],
              "phase_id": "phase-1", "knowledge": {"routes": []}, "route_attempts": [],
              "current_observation": {"evidence_ref": "E0000"}, "sequence": 1, "recent_history": []}
    after = copy.deepcopy(before)
    after["checks"][0].update(status="claimed_verified", evidence_refs=["fictional-row/E0001"], note="Observed earlier")
    after.update(phase_id="phase-2", knowledge={"routes": ["route-1"]},
                 route_attempts=[{"action_digest": "fictional", "count": 1}],
                 sequence=9, current_observation={"evidence_ref": "E0009"}, recent_history=[{"sequence": 9}])
    original_before, original_after = copy.deepcopy(before), copy.deepcopy(after)
    delta = context_delta(before, after)
    assert delta == {"checks_are_untrusted_claims": True,
        "checks": [{"id": "C001", "status": "claimed_verified", "evidence_refs": ["fictional-row/E0001"],
                    "note": "Observed earlier"}],
        "phase_id": "phase-2", "knowledge": {"routes": ["route-1"]},
        "route_attempts": [{"action_digest": "fictional", "count": 1}]}
    assert before == original_before and after == original_after
    assert context_delta(after, after) == {}
    delta["checks"][0]["evidence_refs"].append("E9999")
    delta["knowledge"]["routes"].clear()
    assert after == original_after
    # Cleared claim fields and host state are explicit, not interpreted as omissions.
    reset = context_delta(after, before)
    assert reset["checks"] == [{"id": "C001", "status": "pending", "evidence_refs": [], "note": ""}]
    assert reset["route_attempts"] == []


def test_observation_bookkeeping_alone_does_not_repeat_context():
    before = {"checks": [], "phase_id": None, "knowledge": {}, "route_attempts": [], "sequence": 1}
    after = {**before, "sequence": 2, "current_observation": {"evidence_ref": "E0002"},
             "recent_history": [{"sequence": 2}], "history_events": 2}
    assert context_delta(before, after) == {}


def run_managed(tmp_path, *, max_request_bytes=200_000):
    state = SessionState(tmp_path / "session.json", "fictional-session",
                         ["Reach the next screen", "Return safely"])
    state.begin_phase("fictional-phase")
    state.set_knowledge({"routes": ["fictional-route"]})
    requests, calls, assistant_responses = [], [], []
    actions = [("tap", {"id": "el:1"})] * 4 + [
        ("update_checks", {"updates": [{"id": "C001", "status": "claimed_verified",
                                        "evidence_refs": ["E0001"], "note": "A model claim"}]}),
        ("read_status", {})]
    async def send(payload):
        requests.append(copy.deepcopy(payload))
        if len(requests) > len(actions):
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Review requested."}}]}
        name, arguments = actions[len(requests) - 1]
        message = {"role": "assistant", "content": None, "reasoning": "preserved reasoning",
                   "reasoning_content": "preserved native reasoning",
                   "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque==", "index": 0}],
                   "tool_calls": [{"type": "function", "id": f"call-{len(requests)}",
                                   "function": {"name": name, "arguments": json.dumps(arguments)},
                                   "extra_content": {"google": {"thought_signature": "signature=="}}}]}
        assistant_responses.append(copy.deepcopy(message))
        return {"choices": [{"finish_reason": "tool_calls", "message": message}]}
    async def execute(name, arguments):
        calls.append((name, arguments))
        if name == "update_checks":
            return state.update_checks(arguments["updates"])
        return {"ok": True, "observation": frame(), "earlier_detail": "x" * 20_000}
    report = asyncio.run(run_agent(
        send=send, call_tool=execute,
        tools=[native("tap", {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
               native("update_checks", {"type": "object", "properties": {"updates": {"type": "array"}}, "required": ["updates"]}),
               native("read_status", {"type": "object"})],
        system_prompt="Synthetic test", user_prompt="Proceed", initial_observation=frame(),
        model="fictional", backend="openai-compatible", output=tmp_path / "run", max_steps=8,
        session_state=state, max_request_bytes=max_request_bytes,
    ))
    return report, requests, calls, assistant_responses, state


def test_managed_history_is_exact_append_only_prefix_through_guard_and_check_updates(tmp_path):
    report, requests, calls, assistants, state = run_managed(tmp_path)
    assert report["stop_reason"] == "model_text"
    assert len(requests) == 7 and len(calls) == 5
    assert report["host_rejections"] == 1 and report["tool_calls_executed"] == 5
    assert len(report["evidence"]) == 7
    for previous, current in zip(requests, requests[1:], strict=False):
        # Compare serialized bytes, including string content and dict order.
        old = json.dumps(previous["messages"], ensure_ascii=False).encode()
        prefix = json.dumps(current["messages"][:len(previous["messages"])], ensure_ascii=False).encode()
        assert prefix == old
    first = requests[0]["messages"]
    assert [message["role"] for message in first] == ["system", "user"]
    ledger = json.loads(first[1]["content"].split(LEDGER_LABEL)[1])
    assert ledger["checks_are_untrusted_claims"] is True
    assert [(check["id"], check["status"]) for check in ledger["checks"]] == [("C001", "pending"), ("C002", "pending")]
    assert ledger["phase_id"] == "fictional-phase" and ledger["knowledge"] == {"routes": ["fictional-route"]}
    assert ledger["current_observation"]["evidence_ref"] == "E0000"
    final = requests[-1]["messages"]
    assert sum(LEDGER_LABEL in (message.get("content") or "") for message in final) == 1
    assert [message for message in final if message["role"] == "assistant"] == assistants
    tool_messages = [message for message in final if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [f"call-{n}" for n in range(1, 7)]
    assert [message["name"] for message in tool_messages] == ["tap"] * 4 + ["update_checks", "read_status"]
    results = [json.loads(message["content"]) for message in tool_messages]
    assert results[0]["observation"] == frame() and results[0]["earlier_detail"] == "x" * 20_000
    assert all("archived_result" not in result for result in results)
    assert results[3]["error"]["code"] == "repeated_action_without_progress"
    assert results[3]["error"]["executed"] is False
    assert results[3]["host_context_delta"]["route_attempts"][0]["count"] == 4
    assert results[4]["host_context_delta"] == {"checks_are_untrusted_claims": True,
        "checks": [{"id": "C001", "status": "claimed_verified", "evidence_refs": ["E0001"], "note": "A model claim"}]}
    assert "host_context_delta" not in results[5]
    assert state.context()["checks"][1]["status"] == "pending"
    assert "verdict" not in report and report["report_is_untrusted"] is True
    # Deltas belong to request messages; recorded source evidence stays exact.
    source = json.loads((tmp_path / "run/evidence/E0001.json").read_text())
    assert source == {"ok": True, "observation": frame(), "earlier_detail": "x" * 20_000}
    turns = [json.loads(line) for line in (tmp_path / "run/model-turns.jsonl").read_text().splitlines()]
    assert [turn["request"] for turn in turns] == requests


def test_managed_full_history_exceeding_byte_budget_stops_before_next_request(tmp_path):
    report, requests, calls, _, _ = run_managed(tmp_path, max_request_bytes=10_000)
    assert len(requests) == len(calls) == 1
    assert report["error"] is None
    assert report["stop_reason"] == "conversation_budget"
    assert "conversation reached its byte budget" in report["warnings"][0]
    assert len(report["evidence"]) == 2
    assert json.loads((tmp_path / "run/evidence/E0001.json").read_text())["earlier_detail"] == "x" * 20_000
