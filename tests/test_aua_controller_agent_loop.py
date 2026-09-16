from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.run_live import RunError

TOOLS = [{"type": "function", "function": {"name": "tap", "parameters": {
    "type": "object", "properties": {"id": {"type": "string"}},
    "required": ["id"], "additionalProperties": False,
}}}]
SETTINGS = {"provider": {"only": ["fictional"], "allow_fallbacks": False,
                         "require_parameters": True,
                         "max_price": {"prompt": 0.3, "completion": 1.2}},
            "reasoning": {"effort": "low"}}


def response(name="tap", arguments=None, *, content=None, cost=0.001):
    message = {"role": "assistant", "content": content}
    if content is None:
        message["tool_calls"] = [{"type": "function", "id": "native-call-1", "function": {
            "name": name, "arguments": json.dumps(arguments if arguments is not None else {"id": "el:1"})}}]
    return {"model": "fictional/model", "provider": "fictional",
            "usage": {"cost": cost, "prompt_tokens": 12, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop" if content is not None else "tool_calls", "message": message}]}


def run(tmp_path, responses, results=None, **kwargs):
    requests, calls = [], []
    results = iter(results if results is not None else [{"ok": True, "observation": {"elements": []}}] * 10)
    replies = iter(responses)

    async def send(payload):
        requests.append(copy.deepcopy(payload))
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    async def call_tool(name, arguments):
        calls.append((name, copy.deepcopy(arguments)))
        return next(results)

    options = {"send": send, "call_tool": call_tool, "tools": TOOLS,
               "system_prompt": "Act only through tools; your report is untrusted.",
               "user_prompt": "Inspect the application.", "initial_observation": {"elements": []},
               "model": "fictional/model", "output": tmp_path / "controller", "request_config": SETTINGS}
    options.update(kwargs)
    report = asyncio.run(run_agent(**options))
    return report, requests, calls


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_action_records_resolve_pin_unpin_save_from_prior_host_observations(tmp_path):
    def observed(label):
        return {"screen": {"width": 100, "height": 200},
                "meta": {"fingerprint": "screen-" + label},
                "elements": [{"id": "el:1", "text": label, "type": "Button"}]}

    run(tmp_path, [response(), response(), response(), response(content="done")],
        [observed("Unpin"), observed("Save"), observed("Done")],
        initial_observation=observed("Pin"))
    calls = records(tmp_path / "controller/tool-calls.jsonl")
    assert [call["step"] for call in calls] == [0, 1, 2]
    assert [call["resolved_target"]["text"] for call in calls] == ["Pin", "Unpin", "Save"]
    assert [call["resolved_target"]["source_evidence_ref"] for call in calls] == ["E0000", "E0001", "E0002"]


def test_native_continuity_and_numbered_raw_evidence(tmp_path):
    first = response()
    message = first["choices"][0]["message"]
    message.update({"reasoning": "reasoned", "reasoning_content": "native",
                    "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque==", "index": 0}]})
    message["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": "signature=="}}
    raw = {"ok": True, "observation": {"elements": [{"id": "el:1", "text": "Ready"}]},
           "private_marker": "local-only"}

    def project(value):
        value.pop("private_marker", None)
        return value

    report, requests, calls = run(tmp_path, [first, response(content="I think it passed. E0001")],
                                  [raw], observation_filter=project)
    assert calls == [("tap", {"id": "el:1"})]
    assert requests[1]["messages"][2] == message
    feedback = requests[1]["messages"][3]
    assert feedback["tool_call_id"] == "native-call-1"
    assert json.loads(feedback["content"])["evidence_ref"] == "E0001"
    assert "local-only" not in json.dumps(requests)
    assert report["stop_reason"] == "model_text"
    assert report["final_model_text"] == "I think it passed. E0001"
    assert report["report_is_untrusted"] is True
    assert "passed" not in report and "verdict" not in report
    assert report["providers"] == ["fictional"]
    assert report["returned_models"] == ["fictional/model"]
    assert report["cost_accounting"]["reported_usd"] == 0.002
    assert len(report["usage"]) == 2
    # Missing reasoning-token counts remain missing, never invented as zero.
    assert "completion_tokens_details" not in report["usage"][0]
    saved = json.loads((tmp_path / "controller/controller-result.json").read_text())
    assert saved == report
    for evidence in report["evidence"]:
        data = (tmp_path / "controller" / evidence["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == evidence["sha256"]
    assert json.loads((tmp_path / "controller/evidence/E0001.json").read_text()) == raw
    assert records(tmp_path / "controller/model-turns.jsonl")[0]["request"] == requests[0]


@pytest.mark.parametrize("invalid", [response(arguments={"coords": [1, 2]}), response(name="session_start")])
def test_schema_repair_never_executes_invalid_call_and_preserves_id(tmp_path, invalid):
    report, requests, calls = run(tmp_path, [invalid, response(), response(content="Done")])
    assert report["schema_repairs"] == 1
    assert len(calls) == 1
    feedback = requests[1]["messages"][-1]
    assert feedback["tool_call_id"] == "native-call-1"
    assert json.loads(feedback["content"])["error"]["executed"] is False


def test_three_repairs_then_fourth_invalid_call_stops(tmp_path):
    report, requests, calls = run(tmp_path, [response(arguments={})] * 4)
    assert report["schema_repairs"] == 3
    assert len(requests) == 4 and calls == []
    assert report["error"] == "controller repair budget exhausted"
    assert len(records(tmp_path / "controller/controller-feedback.jsonl")) == 3


@pytest.mark.parametrize("kind", ["multiple", "truncated", "malformed", "missing_id"])
def test_unsafe_native_envelopes_are_terminal_without_tool_execution(tmp_path, kind):
    item = response()
    choice = item["choices"][0]
    if kind == "multiple":
        choice["message"]["tool_calls"] *= 2
    elif kind == "truncated":
        choice["finish_reason"] = "length"
    elif kind == "malformed":
        choice["message"]["tool_calls"][0]["function"]["arguments"] = "{"
    else:
        choice["message"]["tool_calls"][0].pop("id")
    report, requests, calls = run(tmp_path, [item])
    assert report["stop_reason"] == "error" and report["error"]
    assert len(requests) == 1 and calls == []
    assert report["cost_accounting"]["reported_usd"] == 0.001


def multiple_response():
    item = response(name="complete_setup", arguments={"route": "guest"})
    message = item["choices"][0]["message"]
    message["tool_calls"][0]["id"] = "native-setup"
    second = response(name="capture_evidence", arguments={"purpose": "guest home"})["choices"][0]["message"]["tool_calls"][0]
    second["id"] = "native-capture"
    message["tool_calls"].append(second)
    message["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "native-opaque==", "index": 0}]
    message["tool_calls"][0]["extra_content"] = {"provider": {"signature": "keep-me=="}}
    return item


def multiple_tools():
    return TOOLS + [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {field: {"type": "string"}},
        "required": [field], "additionalProperties": False,
    }}} for name, field in (("complete_setup", "route"), ("capture_evidence", "purpose"))]


def test_multiple_calls_receive_all_native_error_results_then_single_call_recovers(tmp_path):
    item = multiple_response()
    original = copy.deepcopy(item)
    report, requests, calls = run(tmp_path, [item, response(arguments={"id": "el:recovery"}), response(content="Done")], tools=multiple_tools())
    assert item == original
    assert calls == [("tap", {"id": "el:recovery"})]
    assert records(tmp_path / "controller/tool-calls.jsonl")[0]["step"] == 1
    history = requests[1]["messages"]
    assert history[2] == original["choices"][0]["message"]
    assert [entry["tool_call_id"] for entry in history[3:]] == ["native-setup", "native-capture"]
    assert [entry["name"] for entry in history[3:]] == ["complete_setup", "capture_evidence"]
    for feedback in history[3:]:
        error = json.loads(feedback["content"])["error"]
        assert error["code"] == "multiple_tool_calls" and error["executed"] is False
        assert "exactly one" in error["message"]
    assert report["repair_count"] == report["protocol_repairs"] == 1
    assert report["schema_repairs"] == 0 and report["error"] is None
    assert report["cost_accounting"]["reported_usd"] == 0.003
    feedback = records(tmp_path / "controller/controller-feedback.jsonl")
    assert [entry["tool_call_id"] for entry in feedback] == ["native-setup", "native-capture"]


def test_repeated_multi_calls_exhaust_the_same_three_repair_budget(tmp_path):
    report, requests, calls = run(tmp_path, [multiple_response()] * 4, tools=multiple_tools())
    assert len(requests) == 4 and calls == []
    assert report["repair_count"] == report["protocol_repairs"] == 3
    assert report["schema_repairs"] == 0
    assert report["error"] == "controller repair budget exhausted"
    assert len(records(tmp_path / "controller/controller-feedback.jsonl")) == 6
    assert report["cost_accounting"]["reported_usd"] == 0.004


def test_schema_and_protocol_repairs_share_one_budget(tmp_path):
    report, requests, calls = run(tmp_path, [response(arguments={}), multiple_response(), response(arguments={}), multiple_response()], tools=multiple_tools())
    assert len(requests) == 4 and calls == []
    assert report["repair_count"] == 3
    assert report["schema_repairs"] == 2 and report["protocol_repairs"] == 1
    assert report["error"] == "controller repair budget exhausted"


@pytest.mark.parametrize("kind", ["duplicate_id", "blank_id", "missing_id", "invalid_name", "malformed_arguments", "truncated"])
def test_malformed_multi_call_envelopes_stop_without_repair_or_execution(tmp_path, kind):
    item = multiple_response()
    choice = item["choices"][0]
    call = choice["message"]["tool_calls"][1]
    if kind == "duplicate_id":
        call["id"] = "native-setup"
    elif kind == "blank_id":
        call["id"] = " "
    elif kind == "missing_id":
        call.pop("id")
    elif kind == "invalid_name":
        call["function"]["name"] = ""
    elif kind == "malformed_arguments":
        call["function"]["arguments"] = "{"
    else:
        choice["finish_reason"] = "length"
    report, requests, calls = run(tmp_path, [item])
    assert len(requests) == 1 and calls == []
    assert report["stop_reason"] == "error" and report["repair_count"] == 0
    assert not (tmp_path / "controller/controller-feedback.jsonl").exists()
    assert report["cost_accounting"]["reported_usd"] == 0.001


def test_missing_cost_in_multi_response_stops_before_protocol_repair(tmp_path):
    item = multiple_response()
    item["usage"].pop("cost")
    report, requests, calls = run(tmp_path, [item])
    assert len(requests) == 1 and calls == [] and report["repair_count"] == 0
    assert report["cost_accounting"]["missing_or_invalid_cost"] is True
    assert not (tmp_path / "controller/controller-feedback.jsonl").exists()


def test_missing_cost_is_logged_before_any_action_and_blocks_execution(tmp_path):
    item = response()
    item["usage"].pop("cost")
    report, requests, calls = run(tmp_path, [item])
    assert len(requests) == 1 and calls == []
    assert report["cost_accounting"]["missing_or_invalid_cost"] is True
    assert "usage.cost" in report["error"]
    assert records(tmp_path / "controller/model-turns.jsonl")[0]["response"] == item


def test_reported_cost_cap_stops_next_request_after_paid_action(tmp_path):
    report, requests, calls = run(tmp_path, [response(cost=0.02)], cost_limit_usd=0.01)
    assert len(requests) == len(calls) == 1
    assert report["cost_accounting"]["reported_usd"] == 0.02
    assert "cost limit reached" in report["error"]


def test_openrouter_exact_config_and_local_defaults_are_separate(tmp_path):
    _, requests, _ = run(tmp_path, [response(content="Done")])
    assert "parallel_tool_calls" not in requests[0]
    assert "temperature" not in requests[0]
    assert "chat_template_kwargs" not in requests[0]
    assert requests[0]["provider"] == SETTINGS["provider"]
    local = response(content="Done")
    local["usage"].pop("cost")
    report, requests, _ = run(tmp_path / "local", [local], backend="openai-compatible",
                              request_config={"chat_template_kwargs": {"enable_thinking": True}})
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["temperature"] == 0
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "cost_accounting" not in report
    assert report["error"] is None


def test_rejected_submit_is_feedback_and_accepted_submit_remains_untrusted(tmp_path):
    submission = {"type": "function", "function": {"name": "submit_result", "parameters": {
        "type": "object", "properties": {"claim": {"type": "string"}},
        "required": ["claim"], "additionalProperties": False}}}
    replies = [response(name="submit_result", arguments={"claim": "PASS"})] * 2
    outcomes = [{"ok": False, "error": "missing evidence"}, {"ok": True, "accepted_for_review": True}]
    report, requests, calls = run(tmp_path, replies, outcomes, tools=[submission],
                                  terminal_tools=frozenset({"submit_result"}))
    assert len(requests) == len(calls) == 2
    assert json.loads(requests[1]["messages"][-1]["content"])["error"] == "missing evidence"
    assert report["tool_errors"] == 1
    assert report["stop_reason"] == "terminal_tool"
    assert report["terminal_submission"]["result"]["accepted_for_review"] is True
    assert report["final_model_text"] is None and report["report_is_untrusted"]
    assert "passed" not in report and "verdict" not in report


def test_mcp_error_cannot_terminate_submission(tmp_path):
    raw = {"isError": True, "content": [{"type": "text", "text": '{"ok":true}'}]}
    report, _, _ = run(tmp_path, [response(), response(content="Stopped")], [raw],
                       terminal_tools=frozenset({"tap"}))
    assert report["terminal_submission"] is None and report["tool_errors"] == 1
    assert report["stop_reason"] == "model_text"


def test_http_failure_preserves_bounded_diagnostic_and_redacts_auth(tmp_path):
    request = httpx.Request("POST", "https://example.invalid", headers={"Authorization": "Bearer test-only-key"})
    remote = httpx.Response(429, request=request, json={"error": {"message": "pool full test-only-key"}})
    report, requests, calls = run(tmp_path, [httpx.HTTPStatusError("failure", request=request, response=remote)])
    assert len(requests) == 1 and calls == []
    assert report["model_responses"] == 0
    assert "HTTP 429" in report["error"] and "pool full" in report["error"]
    assert "test-only-key" not in report["error"]
    assert report["cost_accounting"]["responses_with_cost"] == 0


def test_byte_and_step_budgets_stop_without_truncating_history(tmp_path):
    report, requests, calls = run(tmp_path, [], max_request_bytes=100)
    assert not requests and not calls
    assert report["error"] is None
    assert report["stop_reason"] == "conversation_budget"
    assert "conversation reached its byte budget" in report["warnings"][0]
    report, requests, calls = run(tmp_path / "step", [response()], max_steps=1)
    assert len(requests) == len(calls) == 1
    assert report["error"] is None
    assert report["stop_reason"] == "step_budget"
    assert "reached its step budget" in report["warnings"][0]


def test_send_timeout_records_attempt_and_does_not_call_tools(tmp_path):
    async def slow_send(payload):
        await asyncio.sleep(10)

    report, _, calls = run(tmp_path, [], send=slow_send, request_timeout_s=0.01)
    assert report["model_requests"] == 1 and report["model_responses"] == 0
    assert report["error"] == "TimeoutError" and calls == []


def test_tool_timeout_is_an_unknown_outcome_for_caller_cleanup(tmp_path):
    async def slow_tool(name, arguments):
        await asyncio.sleep(10)

    report, requests, _ = run(tmp_path, [response()], call_tool=slow_tool, request_timeout_s=0.01)
    assert len(requests) == 1
    assert report["tool_calls_executed"] == report["unknown_tool_outcomes"] == 1
    assert report["caller_owns_verification_and_cleanup"] is True
    assert report["error"] == "TimeoutError"


def test_exact_tool_timeout_can_exceed_request_timeout_without_widening_model_calls(tmp_path):
    calls = []

    async def slow_tool(name, arguments):
        calls.append((name, arguments))
        await asyncio.sleep(0.03)
        return {"ok": True, "observation": {"elements": []}}

    report, requests, _ = run(
        tmp_path,
        [response(), response(content="Done")],
        call_tool=slow_tool,
        request_timeout_s=0.01,
        tool_timeouts_s={"tap": 0.1},
        time_limit_s=1,
    )
    assert len(requests) == 2 and len(calls) == 1
    assert report["error"] is None and report["stop_reason"] == "model_text"
    assert report["limits"]["request_seconds"] == 0.01
    assert report["limits"]["tool_seconds"] == {"tap": 0.1}
    call = records(tmp_path / "controller/tool-calls.jsonl")[0]
    assert call["executed"] is True and 0.01 < call["timeout_s"] <= 0.1


def test_model_request_failure_escalates_once_and_stays_on_ordered_fallback(tmp_path):
    report, requests, calls = run(
        tmp_path,
        [TimeoutError(), response(), response(content="Done")],
        model_fallbacks=[("fictional/fallback", SETTINGS)],
    )
    assert calls == [("tap", {"id": "el:1"})]
    assert [payload["model"] for payload in requests] == [
        "fictional/model",
        "fictional/fallback",
        "fictional/fallback",
    ]
    assert report["model_ladder"] == ["fictional/model", "fictional/fallback"]
    assert report["model_escalations"] == 1
    assert report["model_failures"] == [
        {"model": "fictional/model", "error": "TimeoutError"}
    ]
    assert report["stop_reason"] == "model_text" and report["error"] is None
    turns = records(tmp_path / "controller/model-turns.jsonl")
    assert [turn["model_rung"] for turn in turns] == [0, 1, 1]


@pytest.mark.parametrize("kind", ["abnormal", "truncated", "malformed_arguments", "missing_id"])
def test_unusable_response_envelope_escalates_before_any_device_dispatch(tmp_path, kind):
    unusable = response()
    choice = unusable["choices"][0]
    if kind == "abnormal":
        choice["finish_reason"] = "content_filter"
    elif kind == "truncated":
        choice["finish_reason"] = "length"
    elif kind == "malformed_arguments":
        choice["message"]["tool_calls"][0]["function"]["arguments"] = "{"
    else:
        choice["message"]["tool_calls"][0].pop("id")

    report, requests, calls = run(
        tmp_path,
        [unusable, response(), response(content="Done")],
        model_fallbacks=[("fictional/fallback", SETTINGS)],
    )

    assert calls == [("tap", {"id": "el:1"})]
    assert [payload["model"] for payload in requests] == [
        "fictional/model",
        "fictional/fallback",
        "fictional/fallback",
    ]
    assert report["model_escalations"] == 1
    assert report["tool_calls_executed"] == 1
    assert report["unknown_tool_outcomes"] == 0
    assert report["cost_accounting"]["reported_usd"] == 0.003
    turns = records(tmp_path / "controller/model-turns.jsonl")
    assert turns[0]["model_rung"] == 0 and turns[0]["error"]
    assert "response" in turns[0], "the unusable but billable response remains auditable"


def test_valid_multi_call_protocol_repair_does_not_advance_the_model_ladder(tmp_path):
    report, requests, calls = run(
        tmp_path,
        [multiple_response(), response(), response(content="Done")],
        tools=multiple_tools(),
        model_fallbacks=[("fictional/fallback", SETTINGS)],
    )

    assert calls == [("tap", {"id": "el:1"})]
    assert [payload["model"] for payload in requests] == ["fictional/model"] * 3
    assert report["protocol_repairs"] == 1
    assert report["model_escalations"] == 0


def test_exhausted_schema_repairs_continue_on_fallback_with_matched_feedback(tmp_path):
    invalid = response(arguments={})
    report, requests, calls = run(
        tmp_path,
        [invalid, invalid, invalid, invalid, response(), response(content="Done")],
        model_fallbacks=[("fictional/fallback", SETTINGS)],
    )

    assert [payload["model"] for payload in requests] == [
        "fictional/model",
        "fictional/model",
        "fictional/model",
        "fictional/model",
        "fictional/fallback",
        "fictional/fallback",
    ]
    assert calls == [("tap", {"id": "el:1"})]
    assert report["repair_count"] == report["schema_repairs"] == 3
    assert report["model_escalations"] == 1
    assert report["model_failures"][-1]["error"] == "controller schema repair budget exhausted"
    fallback_history = requests[4]["messages"]
    assert fallback_history[-2]["role"] == "assistant"
    assert fallback_history[-1]["role"] == "tool"
    assert json.loads(fallback_history[-1]["content"])["error"]["executed"] is False
    assert len(records(tmp_path / "controller/controller-feedback.jsonl")) == 4


def test_exhausted_protocol_repairs_continue_on_fallback_without_device_replay(tmp_path):
    invalid = multiple_response()
    report, requests, calls = run(
        tmp_path,
        [invalid, invalid, invalid, invalid, response(), response(content="Done")],
        tools=multiple_tools(),
        model_fallbacks=[("fictional/fallback", SETTINGS)],
    )

    assert [payload["model"] for payload in requests[-2:]] == [
        "fictional/fallback",
        "fictional/fallback",
    ]
    assert calls == [("tap", {"id": "el:1"})]
    assert report["repair_count"] == report["protocol_repairs"] == 3
    assert report["model_escalations"] == 1
    assert report["model_failures"][-1]["error"] == "controller protocol repair budget exhausted"
    fallback_history = requests[4]["messages"]
    assert fallback_history[-3]["role"] == "assistant"
    assert [entry["role"] for entry in fallback_history[-2:]] == ["tool", "tool"]
    assert len(records(tmp_path / "controller/controller-feedback.jsonl")) == 8


def test_model_fallback_never_replays_a_tool_with_unknown_outcome(tmp_path):
    async def slow_tool(name, arguments):
        await asyncio.sleep(10)

    report, requests, _ = run(
        tmp_path,
        [response()],
        call_tool=slow_tool,
        request_timeout_s=0.01,
        model_fallbacks=[("fictional/fallback", SETTINGS)],
    )
    assert [payload["model"] for payload in requests] == ["fictional/model"]
    assert report["model_escalations"] == 0
    assert report["unknown_tool_outcomes"] == 1


@pytest.mark.parametrize(
    "tool_timeouts_s",
    [[], {"unknown": 10}, {"tap": 0}, {"tap": True}, {"tap": float("inf")}],
)
def test_invalid_exact_tool_timeouts_fail_before_transport(tmp_path, tool_timeouts_s):
    with pytest.raises(RunError):
        run(tmp_path, [], tool_timeouts_s=tool_timeouts_s)
    assert not (tmp_path / "controller").exists()


@pytest.mark.parametrize(
    "model_fallbacks",
    ["fallback", [("", SETTINGS)], [("model", [])], [("model",)]],
)
def test_invalid_model_fallbacks_fail_before_transport(tmp_path, model_fallbacks):
    with pytest.raises(RunError):
        run(tmp_path, [], model_fallbacks=model_fallbacks)
    assert not (tmp_path / "controller").exists()


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_tokens": True}, {"time_limit_s": float("nan")},
                                   {"tools": [None]}, {"tools": []}, {"request_config": []},
                                   {"terminal_tools": frozenset({"unknown"})}])
def test_invalid_configuration_fails_before_transport_or_tools(tmp_path, kwargs):
    with pytest.raises(RunError):
        run(tmp_path, [], **kwargs)
    assert not (tmp_path / "controller").exists()


def test_output_refuses_to_overwrite_existing_trace(tmp_path):
    directory = tmp_path / "controller"
    directory.mkdir()
    existing = directory / "keep.txt"
    existing.write_text("existing evidence")
    with pytest.raises(RunError, match="empty"):
        run(tmp_path, [])
    assert existing.read_text() == "existing evidence"
