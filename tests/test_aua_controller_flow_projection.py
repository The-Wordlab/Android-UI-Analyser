"""New-message flow width control preserves execution and evidence authority."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.flow_projection import ELEMENT_FIELDS, compact_flow_observations


def flow_result():
    element = {"id": "el:opaque/one", "type": "EditText", "text": "Fictional unsent draft",
               "resource_id": "example.app:id/composer", "content_desc": "Message field",
               "bounds": [10, 20, 100, 60], "clickable": True, "enabled": True,
               "focused": True, "checked": False, "selected": False, "confidence": 0.9,
               "stable_key": "rid:composer", "cost": {"tap": {"samples": 1}}}
    legacy = {key: element[key] for key in ("id", "type", "resource_id", "text", "bounds", "clickable")}
    return {"candidate_id": "route-fictional", "source_sha256": "f" * 64, "executed": True, "ok": True,
            "result": {"ok": True, "flow": "fictional-phase", "steps_run": [{"index": 0, "kind": "input"}],
                       "elements": [legacy], "observation_present": True,
                       "goal_progress": {"current": {"id": "next_input"}, "done": False},
                       "observation_contract": {"reusable": True, "evidence_fresh": True},
                       "observation": {"schema_version": 2, "screen": {"width": 400, "height": 800},
                                       "elements": [element], "meta": {"fingerprint": "exact-fp", "stale_risk": False}}}}


def test_recognized_flow_loses_only_duplicate_summary_and_unrequested_element_fields():
    original = flow_result()
    before = copy.deepcopy(original)
    projected = compact_flow_observations(original)
    source, shown = original["result"], projected["result"]
    assert original == before and "elements" not in shown
    expected = {k: v for k, v in source["observation"]["elements"][0].items() if k in ELEMENT_FIELDS}
    assert shown["observation"]["elements"] == [expected]
    assert {k: v for k, v in shown.items() if k != "observation"} == {k: v for k, v in source.items() if k not in {"observation", "elements"}}
    assert shown["observation"]["meta"] == source["observation"]["meta"]
    assert compact_flow_observations(projected) == projected
    projected["result"]["observation"]["elements"][0]["text"] = "changed copy"
    assert original == before


def test_full_and_short_selector_aliases_are_preserved_exactly():
    value = flow_result()
    frame = value["result"]["observation"]
    element = frame["elements"][0]
    element.update(rid=element["resource_id"], desc=element["content_desc"])
    value["result"]["elements"] = [{"id": element["id"], "rid": element["rid"], "desc": element["desc"]}]
    result = compact_flow_observations(value)["result"]
    assert "elements" not in result
    for key in ("id", "text", "resource_id", "rid", "content_desc", "desc"):
        assert result["observation"]["elements"][0][key] == element[key]


@pytest.mark.parametrize("mutation", [
    lambda v: v["elements"].append(copy.deepcopy(v["elements"][0])),
    lambda v: v["elements"][0].update(id="el:another-frame"),
    lambda v: v["elements"][0].update(text="Different text"),
    lambda v: v["elements"][0].update(confidence=0.3),
    lambda v: v["elements"][0].update(label="Distinct diagnostic summary"),
    lambda v: v["elements"][0].update(clickable=1),
    lambda v: v["elements"][0].update(rid="conflicting alias"),
    lambda v: v["elements"][0].update(cost={"tap": {"samples": True}}),
])
def test_nonidentical_legacy_summary_is_preserved_including_confidence_conflicts(mutation):
    value = flow_result()
    mutation(value["result"])
    original = copy.deepcopy(value)
    projected = compact_flow_observations(value)
    assert projected["result"]["elements"] == original["result"]["elements"]
    assert "cost" not in projected["result"]["observation"]["elements"][0]
    assert value == original


def test_failed_flow_keeps_every_failure_resume_progress_and_unreusable_flag():
    value = flow_result()
    value.update(ok=False)
    inner = value["result"]
    inner.update(ok=False, code="assert_failed", failed_step={"assert": {"rid": "reply"}},
                 step_index=4, remaining_steps=[{"clear": "composer"}], failure_detail="reply absent",
                 resume_call="resume from step 4", error={"code": "unmet"}, stale_risk=True,
                 readiness="unmet", elements=[{"id": "diagnostic-id", "label": "Loading"}])
    inner["observation_contract"].update(reusable=False, evidence_fresh=True)
    inner["observation"]["meta"].update(stale_risk=True, readiness="loading")
    expected = copy.deepcopy(inner)
    actual = compact_flow_observations(value)["result"]
    for key in expected:
        if key != "observation":
            assert actual[key] == expected[key]
    assert actual["observation"]["meta"] == expected["observation"]["meta"]
    assert "cost" not in actual["observation"]["elements"][0]


@pytest.mark.parametrize("mutation", [
    lambda v: v.pop("flow"), lambda v: v.pop("steps_run"),
    lambda v: v.update(observation_present=False),
    lambda v: v["observation"]["meta"].pop("fingerprint"),
    lambda v: v["observation"]["screen"].update(width=0),
    lambda v: v["observation"]["elements"][0].update(id=True),
    lambda v: v.update(result={"observation": {"screen": {"width": 400, "height": 800},
                                              "elements": [], "meta": {"fingerprint": "conflicting"}}}),
])
def test_unrecognized_or_conflicting_full_frames_remain_unchanged(mutation):
    value = flow_result()
    mutation(value["result"])
    assert compact_flow_observations(value) == value


def test_projection_does_not_traverse_arguments_steps_history_or_serialized_strings():
    flow = flow_result()["result"]
    value = {"arguments": flow, "steps_run": [flow], "history": [flow], "text": json.dumps(flow)}
    assert compact_flow_observations(value) == value
    ordinary = {"ok": True, "observation": flow["observation"], "elements": flow["elements"]}
    assert compact_flow_observations(ordinary) == ordinary


def test_optional_loop_projection_leaves_raw_state_native_arguments_and_prior_messages_intact(tmp_path):
    original = flow_result()
    requests, calls, observations = [], [], []
    assistant = {"role": "assistant", "content": None, "reasoning": "native reasoning retained",
                 "tool_calls": [{"id": "call-1", "type": "function", "function": {
                     "name": "run_phase_flow", "arguments": '{"checkpoint_id":"fictional-phase"}'}}]}

    class State:
        def context(self):
            return {"checks": [], "phase_id": "fictional", "knowledge": {}, "route_attempts": []}

        def observe(self, tool, arguments, result, reference):
            observations.append((tool, copy.deepcopy(arguments), copy.deepcopy(result), reference))

        def rejection(self, *_):
            return None

    async def send(payload):
        requests.append(copy.deepcopy(payload))
        if len(requests) == 1:
            return {"choices": [{"finish_reason": "tool_calls", "message": copy.deepcopy(assistant)}]}
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Await independent review"}}]}

    async def execute(name, args):
        calls.append((name, copy.deepcopy(args)))
        return copy.deepcopy(original)

    report = asyncio.run(run_agent(send=send, call_tool=execute,
        tools=[{"type": "function", "function": {"name": "run_phase_flow", "parameters": {
            "type": "object", "properties": {"checkpoint_id": {"type": "string"}}, "required": ["checkpoint_id"]}}}],
        system_prompt="Fictional test", user_prompt="Proceed", initial_observation=original["result"]["observation"],
        model="fictional", backend="openai-compatible", output=tmp_path, observation_filter=copy.deepcopy,
        model_observation_filter=compact_flow_observations, session_state=State()))
    assert report["stop_reason"] == "model_text"
    assert calls == [("run_phase_flow", {"checkpoint_id": "fictional-phase"})]
    assert observations[1][2] == original
    assert json.loads((tmp_path / "evidence/E0001.json").read_text()) == original
    assert json.loads((tmp_path / "tool-calls.jsonl").read_text())["result"] == original
    assert requests[1]["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]
    assert requests[1]["messages"][-2] == assistant
    visible = json.loads(requests[1]["messages"][-1]["content"])
    assert visible["evidence_ref"] == "E0001" and visible["citable_observation"] is True
    assert visible["evidence_kind"] == "observation"
    assert visible["result"]["observation"]["elements"] == compact_flow_observations(original)["result"]["observation"]["elements"]
    assert "elements" not in visible["result"]
