from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.hosted import HostedError
from experiments.aua_controller.judgement import (
    ORACLE,
    OUTCOME_SCHEMA,
    Decider,
    ScreenNamer,
    combine_votes,
    contract_criteria,
    contract_max_tokens,
    evidence_frame,
    judge_outcome_votes,
    judge_route_budget,
    normalize_optional_summaries,
    outcome_schema,
    relaxed_json_answer,
    summarize_route,
)
from experiments.aua_controller.run_live import RunError

SETTINGS = {"provider": {"only": ["fictional"], "allow_fallbacks": False,
                         "max_price": {"prompt": 0.3, "completion": 1.2}},
            "reasoning": {"effort": "low"}}


def frame(fingerprint="fp-1", texts=("Theme", "Light")):
    return {"ok": True, "observation": {
        "screen": {"package": "com.example.fictional", "width": 720, "height": 1280},
        "elements": [{"id": f"el:{i}", "text": text, "clickable": True, "bounds": [0, i, 1, i + 1]}
                     for i, text in enumerate(texts)],
        "meta": {"fingerprint": fingerprint},
    }}


def tool_reply(name, arguments, *, cost=0.0005):
    return {"model": "fictional/model", "provider": "fictional",
            "usage": {"cost": cost, "prompt_tokens": 300, "completion_tokens": 40},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"type": "function", "id": "decide-1", "function": {
                    "name": name, "arguments": json.dumps(arguments)}}]}}]}


def verdict(value, confidence=0.9, reasons=("final frame shows Light selected",)):
    return {"verdict": value, "confidence": confidence, "reasons": list(reasons)}


class Sender:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    async def __call__(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)


def decider(sender, **kwargs):
    options = {"model": "fictional/model", "request_config": SETTINGS, "max_tokens": 512}
    options.update(kwargs)
    return Decider(sender, **options)


def test_decide_opens_a_fresh_window_and_forces_the_structured_tool():
    sender = Sender([tool_reply("record_verdict", verdict("pass"))])
    result = asyncio.run(decider(sender).decide(
        role="outcome judge", instructions="decide", question="done?", context={"goal": "g"},
        schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert result["result"]["verdict"] == "pass"
    assert result["cost"] == 0.0005 and result["repairs"] == 0
    payload = sender.payloads[0]
    assert [message["role"] for message in payload["messages"]] == ["system", "user"], "no controller history"
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "record_verdict"}}
    assert payload["max_tokens"] == 512
    assert payload["provider"]["only"] == ["fictional"] and payload["reasoning"] == {"effort": "low"}
    assert "temperature" not in payload and "parallel_tool_calls" not in payload


def test_judge_keeps_requested_text_checkpoints_beyond_eight_and_image_positions():
    sender = Sender([tool_reply("record_verdict", verdict("unverified"))])
    frames = [frame(f"fp-{index}", (f"Checkpoint {index}",)) for index in range(13)]
    image_position = {"image_index": 1, "ref": "E12", "after_step": 11, "lifecycle_epoch": 1}
    asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="audit", final_frame=frame(),
                                   frames=frames, image_evidence=[image_position]))
    context = json.loads(sender.payloads[0]["messages"][1]["content"].split("Evidence:\n", 1)[1])
    assert len(context["intermediate_frames"]) == 13
    assert context["image_evidence"] == [image_position]
    assert "fresh observation after that numbered action" in context["evidence_selection_note"]
    assert "not a controller claim" in context["evidence_selection_note"]
    assert context["intermediate_frames"][-1]["observation"]["elements"][0]["text"] == "Checkpoint 12"


def test_disabled_reasoning_stays_disabled_under_the_unchanged_vote_deadline():
    settings = copy.deepcopy(SETTINGS)
    settings["reasoning"] = {"enabled": False, "exclude": False}
    sender = Sender([tool_reply("record_verdict", verdict("unverified"))])
    subject = decider(sender, request_config=settings, max_tokens=8192)
    asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                              schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert sender.payloads[0]["reasoning"] == {"enabled": False, "exclude": False}
    assert subject.report()["decision_timeout_s"] == 90
    assert judge_route_budget(90, 3, 45) == 30
    assert judge_route_budget(75, 2, 45) == 37.5


def test_decide_repairs_once_then_fails(tmp_path):
    sender = Sender([tool_reply("record_verdict", {"verdict": "maybe", "confidence": 2, "reasons": []}),
                     tool_reply("record_verdict", {"verdict": "fail", "confidence": 0.4, "reasons": ["x"]})])
    result = asyncio.run(decider(sender, output=tmp_path).decide(
        role="outcome judge", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert result["result"]["verdict"] == "fail" and result["repairs"] == 1
    assert "previous answer was invalid" in sender.payloads[1]["messages"][-1]["content"]
    logged = [json.loads(line) for line in (tmp_path / "judgements.jsonl").read_text().splitlines()]
    assert logged[0]["repairs"] == 1 and logged[0]["result"]["verdict"] == "fail"

    exhausted = Sender([tool_reply("record_verdict", {"verdict": "maybe"}), tool_reply("other_tool", {})])
    with pytest.raises(RunError, match="valid structured answer"):
        asyncio.run(decider(exhausted).decide(
            role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))


def test_decider_drops_forced_tool_choice_when_no_rung_can_honour_it():
    """Forcing the tool is an optimisation, not a requirement -- do not lose a row over routing.

    On 2026-09-15 `threads-new-chat-from-character-card` died with "No endpoints found that
    support the provided 'tool_choice' value" after the ladder was exhausted, and the whole
    scenario was recorded as a provider failure. A model that chooses the tool itself answers
    exactly the same, and a reply without the call already falls into the repair loop.
    """
    route_missing = RuntimeError(
        "HTTP 404: No endpoints found that support the provided 'tool_choice' value"
    )
    sender = Sender([route_missing, tool_reply("record_verdict", verdict("pass"))])
    judge = decider(sender)  # single rung: nothing to escalate to

    result = asyncio.run(judge.decide(
        role="outcome judge", instructions="decide", question="done?",
        context={"goal": "g"}, schema=OUTCOME_SCHEMA, name="record_verdict",
    ))

    assert result["result"]["verdict"] == "pass", "the row must still get its verdict"
    assert sender.payloads[0]["tool_choice"] != "auto", "the first attempt still forces it"
    assert sender.payloads[1]["tool_choice"] == "auto", "the retry lets the model choose"
    assert len(sender.payloads) == 2, "and it retries the same rung, not a new one"


@pytest.mark.parametrize("decision_seconds,expected_budget", [(90, 45), (50, 20)])
def test_rejected_tool_probe_does_not_consume_relaxed_answer_budget(
    tmp_path, monkeypatch, decision_seconds, expected_budget,
):
    from experiments.aua_controller import judgement as module

    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    real_timeout = asyncio.timeout
    budgets = []

    def capture_timeout(delay):
        budgets.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(module.asyncio, "timeout", capture_timeout)

    async def send(payload):
        if payload["tool_choice"] != "auto":
            clock[0] += 30  # deterministic transport latency, without a real sleep
            raise RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value")
        return tool_reply("record_verdict", verdict("pass"), cost=0.002)

    subject = decider(send, route_timeout_s=45, decision_timeout_s=decision_seconds, output=tmp_path)
    result = asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                                       schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert budgets == [45, expected_budget]
    assert result["cost"] == pytest.approx(0.002)
    assert subject.requests == 2
    events = [json.loads(line) for line in (tmp_path / "judge-events.jsonl").read_text().splitlines()]
    relaxed = next(item for item in events if item.get("event") == "tool_choice_relaxed")
    assert relaxed["request_budget_s"] == expected_budget
    assert relaxed["decision_remaining_s"] == decision_seconds - 30


def test_relaxed_answer_is_still_cancellable_without_retry_or_lost_diagnostics(tmp_path):
    async def exercise():
        relaxed = asyncio.Event()

        async def send(payload):
            if payload["tool_choice"] != "auto":
                raise RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value")
            relaxed.set()
            await asyncio.Event().wait()

        subject = decider(send, output=tmp_path)
        task = asyncio.create_task(subject.decide(role="judge", instructions="i", question="q", context={},
                                                  schema=OUTCOME_SCHEMA, name="record_verdict"))
        await asyncio.wait_for(relaxed.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert subject.requests == 2
        assert subject.report()["unreported_cost_requests"] == 1

    asyncio.run(exercise())
    logs = (tmp_path / "judge-events.jsonl").read_text()
    assert "tool_choice_relaxed" in logs and "decision_cancelled" in logs


def test_decider_uses_its_fallback_when_the_tool_choice_route_stays_unavailable():
    route_missing = RuntimeError(
        "HTTP 404: No endpoints found that support the provided 'tool_choice' value"
    )
    sender = Sender([route_missing, tool_reply("record_verdict", verdict("pass"))])
    judge = decider(sender, fallbacks=[("fictional/fallback", SETTINGS)])

    result = asyncio.run(judge.decide(
        role="outcome judge",
        instructions="decide",
        question="done?",
        context={"goal": "g"},
        schema=OUTCOME_SCHEMA,
        name="record_verdict",
    ))

    assert result["result"]["verdict"] == "pass"
    assert result["escalations"] == 1
    assert [payload["model"] for payload in sender.payloads] == [
        "fictional/model",
        "fictional/fallback",
    ]


def test_decider_has_its_own_spend_stop():
    sender = Sender([tool_reply("record_verdict", verdict("pass"), cost=0.02),
                     tool_reply("record_verdict", verdict("pass"), cost=0.02)])
    judge = decider(sender, cost_limit_usd=0.03)
    asyncio.run(judge.decide(role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    asyncio.run(judge.decide(role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    with pytest.raises(HostedError, match="cost limit"):
        asyncio.run(judge.decide(role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert judge.report()["decisions"] == 2 and judge.report()["reported_usd"] == pytest.approx(0.04)


def test_reasoning_exhaustion_route_relaxation_and_schema_repair_have_separate_budgets():
    contract = "- Theme selected.\n- Device default follows system."
    truncated = {"model": "reasoning/model", "usage": {"cost": 0.01},
                 "choices": [{"finish_reason": "length", "message": {"role": "assistant"}}]}
    invalid = {**verdict("unverified"), "criteria": [{
        "criterion": contract_criteria(contract), "result": "not_verified", "evidence": "not seen",
    }]}
    usable = {**verdict("pass"), "criteria": {
        "theme selected": {"result": "verified", "evidence": "selection is visible"},
        "Device default follows system.": {"result": "not_verified", "evidence": "system setting unseen"},
    }}
    sender = Sender([truncated, RuntimeError(
        "HTTP 404: No endpoints found that support the provided 'tool_choice' value"),
        tool_reply("record_verdict", invalid), tool_reply("record_verdict", usable)])
    subject = decider(sender, fallbacks=[("fictional/fallback", SETTINGS)])
    result = asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                                       schema=outcome_schema(contract), name="record_verdict",
                                       criteria_order=contract_criteria(contract)))
    assert result["result"]["verdict"] == "unverified"
    assert result["result"]["criteria"][1]["result"] == "not_verified"
    assert [item["model"] for item in sender.payloads] == [
        "fictional/model", "fictional/fallback", "fictional/fallback", "fictional/fallback",
    ]
    assert sender.payloads[-1]["tool_choice"] == "auto"
    assert "exactly one authored bullet" in sender.payloads[-1]["messages"][-1]["content"]
    assert result["cost"] == pytest.approx(0.011)


def test_route_deadline_covers_transport_backoff_and_advances_without_repair(tmp_path):
    from experiments.aua_controller.transport import resilient_request

    calls, transport_calls, cancelled = [], [], []

    async def send(payload):
        calls.append(payload["model"])
        if payload["model"] == "fictional/fallback":
            return tool_reply("record_verdict", verdict("pass"))

        async def once():
            transport_calls.append(True)
            raise RuntimeError("temporary provider failure")

        async def backoff(_seconds):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        return await resilient_request(once, classify=lambda exc: (503, {}) if isinstance(exc, RuntimeError)
                                       else None, sleep=backoff)

    subject = decider(send, route_timeout_s=0.01, decision_timeout_s=1,
                      fallbacks=[("fictional/fallback", SETTINGS)], output=tmp_path)
    result = asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                                       schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert result["result"]["verdict"] == "pass"
    assert calls == ["fictional/model", "fictional/fallback"]
    assert len(transport_calls) == len(cancelled) == 1
    report = subject.report()
    assert report["cost_complete"] is False and report["unreported_cost_requests"] == 1
    assert report["reported_usd"] == pytest.approx(0.0005)
    logs = [json.loads(line) for line in (tmp_path / "judge-events.jsonl").read_text().splitlines()]
    assert logs[0]["event"] == "route_timeout" and logs[0]["cost_unreported"] is True


@pytest.mark.parametrize("first_response_seconds", [15, 45])
def test_route_repairs_share_one_deadline_and_keep_already_reported_spend(
    tmp_path, monkeypatch, first_response_seconds,
):
    from experiments.aua_controller import judgement as module

    # Schema validation and CI scheduling can consume a 10ms real deadline before
    # the repair starts. Advance the judge's clock explicitly; cancellation itself
    # is covered by the transport-backoff deadline test above.
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    real_timeout = asyncio.timeout
    budgets = []

    def capture_timeout(delay):
        budgets.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(module.asyncio, "timeout", capture_timeout)
    requests = []

    async def send(payload):
        requests.append(payload["model"])
        if len(requests) == 1:
            clock[0] += first_response_seconds
            return tool_reply("record_verdict", {"verdict": "invalid"}, cost=0.003)
        if payload["model"] == "fictional/model":
            clock[0] += 30
            raise TimeoutError
        return tool_reply("record_verdict", verdict("pass"), cost=0.004)

    subject = decider(send, route_timeout_s=45, decision_timeout_s=90,
                      fallbacks=[("fictional/fallback", SETTINGS)], output=tmp_path)
    result = asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                                       schema=OUTCOME_SCHEMA, name="record_verdict"))
    repair = first_response_seconds < 45
    assert requests == ["fictional/model"] * (2 if repair else 1) + ["fictional/fallback"]
    assert budgets == ([45, 30, 45] if repair else [45, 45])
    assert result["cost"] == pytest.approx(0.007)
    assert subject.report()["unreported_cost_requests"] == int(repair)
    logs = [json.loads(line) for line in (tmp_path / "judge-events.jsonl").read_text().splitlines()]
    timeout = next(item for item in logs if item.get("event") == "route_timeout")
    assert timeout["cost"] == pytest.approx(0.003)
    if repair:
        assert timeout["reported_cost_usd"] == pytest.approx(0.003)
        assert timeout["deadline_s"] == 30


@pytest.mark.parametrize("fenced", [False, True])
def test_relaxed_tool_route_recovers_only_schema_valid_json_content(tmp_path, fenced):
    answer = {**verdict("unverified"), "criteria": [
        {"criterion_index": 0, "result": "not_verified", "evidence": "system mode unavailable"},
    ]}
    body = json.dumps(answer)
    if fenced:
        body = "```json\n" + body + "\n```"
    response = {"model": "fictional/model", "usage": {"cost": 0.001}, "choices": [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": body}},
    ]}
    sender = Sender([RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value"),
                     response])
    result = asyncio.run(judge_outcome_votes(decider(sender, output=tmp_path), votes=1,
        goal="verify", final_frame=frame(), contract="- Device default follows system."))
    assert sender.payloads[-1]["tool_choice"] == "auto"
    assert result["verdict"] == "unverified"
    assert result["criteria"][0]["criterion"] == "Device default follows system."
    assert result["criteria"][0]["result"] == "not_verified"
    logged = json.loads((tmp_path / "judgements.jsonl").read_text())
    assert logged["response_format"] == "relaxed_json_content"


@pytest.mark.parametrize("content", [
    'Here is the answer: {"verdict":"pass"}', '{"verdict":"pass"} {}',
    '```json\n{}\n```\n```json\n{}\n```', '{invalid}', '[]', 'null',
    '{"verdict":"fail","verdict":"pass"}', '{"confidence":NaN}',
    '```python\n{}\n```', '{} trailing explanation',
])
def test_relaxed_json_rejects_prose_ambiguity_and_invalid_json(content):
    with pytest.raises(RunError):
        relaxed_json_answer(content)


def test_content_recovery_is_disabled_until_tool_choice_was_relaxed():
    response = {"usage": {"cost": 0.001}, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(verdict("pass")),
    }}]}
    sender = Sender([response])
    with pytest.raises(RunError, match="required tool call"):
        asyncio.run(decider(sender, repair_budget=0).decide(
            role="judge", instructions="i", question="q", context={},
            schema=OUTCOME_SCHEMA, name="record_verdict"))


def test_native_tool_is_preferred_over_message_content_after_route_relaxation():
    response = tool_reply("record_verdict", verdict("fail"))
    response["choices"][0]["message"]["content"] = json.dumps(verdict("pass"))
    sender = Sender([RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value"),
                     response])
    result = asyncio.run(decider(sender).decide(role="judge", instructions="i", question="q", context={},
                                              schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert result["result"]["verdict"] == "fail"


@pytest.mark.parametrize("field", ["satisfied", "unsatisfied"])
@pytest.mark.parametrize("text_response", [False, True])
def test_optional_summary_overflow_never_invalidates_strict_criteria(tmp_path, field, text_response):
    answer = {**verdict("unverified"), field: ["private-summary@example.test"] * 11,
              "criteria": [{"criterion_index": 0, "result": "not_verified",
                            "evidence": "Independent system state unavailable."}]}
    response = tool_reply("record_verdict", answer)
    if text_response:
        response["choices"][0]["message"] = {"role": "assistant", "content": json.dumps(answer)}
        response["choices"][0]["finish_reason"] = "stop"
    sender = Sender([RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value"),
                     response])
    result = asyncio.run(decider(sender, repair_budget=0, output=tmp_path).decide(
        role="judge", instructions="i", question="q", context={}, name="record_verdict",
        schema=outcome_schema("- Device default follows system."),
        criteria_order=["Device default follows system."]))
    assert result["repairs"] == 0 and len(sender.payloads) == 2
    assert len(result["result"][field]) == 8
    assert result["result"]["verdict"] == "unverified"
    assert result["result"]["criteria"] == [{"criterion": "Device default follows system.",
        "result": "not_verified", "evidence": "Independent system state unavailable."}]
    events = (tmp_path / "judge-events.jsonl").read_text()
    normalized = [json.loads(line) for line in events.splitlines()
                  if json.loads(line).get("event") == "optional_summary_normalized"]
    assert normalized[0]["fields"] == [{"field": field, "original_count": 11, "retained_count": 8}]
    assert "private-summary@example.test" not in events


@pytest.mark.parametrize("field,value", [
    ("reasons", ["reason"] * 7), ("confidence", 2), ("verdict", "maybe"),
    ("criteria", [{"criterion_index": 0, "result": "verified", "evidence": "observed"}] * 2),
    ("criteria", [{"criterion_index": 1, "result": "verified", "evidence": "observed"}]),
    ("criteria", [{"criterion_index": 0, "result": "maybe", "evidence": "observed"}]),
    ("criteria", [{"criterion_index": 0, "result": "verified", "evidence": "x" * 401}]),
    ("satisfied", ["valid"] * 8 + [42]), ("satisfied", ["valid"] * 8 + ["x" * 201]),
])
def test_optional_summary_normalization_never_repairs_authoritative_or_malformed_fields(field, value):
    schema = outcome_schema("- One criterion.")
    answer = {**verdict("pass"), "satisfied": ["summary"] * 11,
              "criteria": [{"criterion_index": 0, "result": "verified", "evidence": "observed"}],
              field: value}
    normalized, _ = normalize_optional_summaries(answer, schema)
    assert normalized[field] == value
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(normalized, schema)


def test_optional_summary_normalization_preserves_missing_and_required_fields():
    schema = copy.deepcopy(OUTCOME_SCHEMA)
    schema["required"].append("satisfied")
    answer = {"satisfied": ["summary"] * 11}
    assert normalize_optional_summaries(answer, schema) == (answer, [])
    assert normalize_optional_summaries({}, OUTCOME_SCHEMA) == ({}, [])


def test_schema_repair_diagnostics_never_include_private_model_content(tmp_path):
    private = "private-account@example.test"
    response = {"usage": {"cost": 0.001}, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps({"verdict": private}),
    }}]}
    sender = Sender([RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value"),
                     response])
    with pytest.raises(RunError):
        asyncio.run(decider(sender, repair_budget=0, output=tmp_path).decide(
            role="judge", instructions="i", question="q", context={},
            schema=OUTCOME_SCHEMA, name="record_verdict"))
    events = (tmp_path / "judge-events.jsonl").read_text()
    assert "schema_repair" in events and "violates" in events
    assert private not in events
    assert private not in (tmp_path / "judgements.jsonl").read_text()


def test_total_decision_deadline_bounds_all_routes(tmp_path):
    calls = []

    async def send(payload):
        calls.append(payload["model"])
        await asyncio.Event().wait()

    subject = decider(send, route_timeout_s=0.02, decision_timeout_s=0.03,
                      fallbacks=[("fictional/second", SETTINGS), ("fictional/third", SETTINGS)],
                      output=tmp_path)
    with pytest.raises(RunError, match="decision deadline"):
        asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                                   schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert calls == ["fictional/model", "fictional/second", "fictional/third"]
    assert subject.report()["unreported_cost_requests"] == 3


def test_remaining_routes_receive_a_fair_slice_without_extending_total_deadline():
    assert judge_route_budget(90, 4, 45) == 22.5
    assert judge_route_budget(60, 3, 45) == 20
    assert judge_route_budget(50, 1, 45) == 45


@pytest.mark.parametrize("completion,reasoning,finish", [(9285, 9285, "stop"),
                                                        (429, 429, "stop"), (8192, 8192, "length")])
def test_reasoning_only_usage_skips_schema_repair_and_counts_upstream_charge(completion, reasoning, finish):
    exhausted = {"model": "fictional/model", "usage": {
        "cost": 0, "cost_details": {"upstream_inference_cost": 0.0146829},
        "completion_tokens": completion, "completion_tokens_details": {"reasoning_tokens": reasoning},
    }, "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": ""}}]}
    sender = Sender([exhausted, tool_reply("record_verdict", verdict("pass"), cost=0.0005)])
    subject = decider(sender, fallbacks=[("fictional/fallback", SETTINGS)])
    result = asyncio.run(subject.decide(role="judge", instructions="i", question="q", context={},
                                       schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert [item["model"] for item in sender.payloads] == ["fictional/model", "fictional/fallback"]
    assert result["cost"] == pytest.approx(0.0151829)
    assert subject.report()["reported_usd"] == pytest.approx(0.0151829)


def test_judge_requests_reasoning_budget_without_mutating_controller_profile():
    original = copy.deepcopy(SETTINGS)
    sender = Sender([tool_reply("record_verdict", verdict("pass"))])
    asyncio.run(decider(sender, max_tokens=8192).decide(
        role="judge", instructions="i", question="q", context={},
        schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert sender.payloads[0]["reasoning"] == {"max_tokens": 2048, "exclude": False}
    assert original == SETTINGS
    assert sender.payloads[0]["max_tokens"] == 8192


def test_default_judge_deadlines_do_not_change_controller_request_timeout():
    subject = decider(Sender([]))
    assert subject.route_timeout_s == 45 and subject.decision_timeout_s == 90


@pytest.mark.parametrize("failure", [RuntimeError("provider timeout"), None])
def test_transport_or_nonobject_reply_can_reach_valid_fallback(failure):
    replies = ([failure] if isinstance(failure, Exception) else [None, None])
    sender = Sender([*replies, tool_reply("record_verdict", verdict("pass"))])
    result = asyncio.run(decider(sender, fallbacks=[("fictional/fallback", SETTINGS)]).decide(
        role="judge", instructions="i", question="q", context={},
        schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert result["result"]["verdict"] == "pass" and result["escalations"] == 1


def test_criterion_reordering_and_missing_evidence_never_invent_a_pass():
    contract = "- First surface.\n- Second surface.\n- Device default."
    answer = {**verdict("pass"), "criteria": [
        {"name": "second  surface", "result": "verified", "evidence": "visible second"},
        {"criterion": ["First surface."], "result": "verified", "evidence": "visible first"},
    ]}
    sender = Sender([tool_reply("record_verdict", answer)])
    result = asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="theme", final_frame=frame(),
                                             contract=contract))
    assert [item["criterion"] for item in result["criteria"]] == contract_criteria(contract)
    assert result["criteria"][-1]["result"] == "not_verified"
    assert result["verdict"] == "unverified"


def test_wire_schema_uses_only_indexes_and_host_restores_exact_authored_labels():
    labels = ["The first long authored product criterion must remain byte-for-byte intact.",
              "The second criterion includes apostrophes, punctuation, and exact casing."]
    contract = "\n".join("- " + label for label in labels)
    answer = {**verdict("pass"), "criteria": [
        {"criterion_index": 1, "result": "verified", "evidence": "second visible"},
        {"criterion_index": 0, "result": "verified", "evidence": "first visible"},
    ]}
    sender = Sender([tool_reply("record_verdict", answer)])
    result = asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="verify", final_frame=frame(),
                                             contract=contract))
    wire_schema = sender.payloads[0]["tools"][0]["function"]["parameters"]
    properties = wire_schema["properties"]["criteria"]["items"]["properties"]
    assert properties["criterion_index"] == {"type": "integer", "minimum": 0, "maximum": 1}
    assert "criterion" not in properties
    assert not any(label in json.dumps(wire_schema) for label in labels)
    assert [item["criterion"] for item in result["criteria"]] == labels
    assert [item["criterion"] for item in result["votes"][0]["criteria"]] == labels
    assert "criterion_index" not in json.dumps(result)
    assert "Never repeat the criterion text" in sender.payloads[0]["messages"][1]["content"]


@pytest.mark.parametrize("indexes", [[0, 0], [0, 2], [-1, 1], [True, 1], ["0", 1]])
def test_invalid_or_duplicate_indexes_use_schema_repair_before_a_verdict(indexes):
    contract = "- First.\n- Second."

    def indexed_answer(values):
        return {**verdict("pass"), "criteria": [
            {"criterion_index": index, "result": "verified", "evidence": "visible"}
            for index in values
        ]}

    sender = Sender([tool_reply("record_verdict", indexed_answer(indexes)),
                     tool_reply("record_verdict", indexed_answer([0, 1]))])
    result = asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="verify", final_frame=frame(),
                                             contract=contract))
    assert len(sender.payloads) == 2 and result["verdict"] == "pass"
    assert [item["criterion"] for item in result["criteria"]] == ["First.", "Second."]


def test_omitted_index_maps_to_not_verified_not_an_invented_pass():
    answer = {**verdict("pass"), "criteria": [{"criterion_index": 0, "result": "verified", "evidence": "visible"}]}
    sender = Sender([tool_reply("record_verdict", answer)])
    result = asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="verify", final_frame=frame(),
                                             contract="- First.\n- Device default."))
    assert result["verdict"] == "unverified"
    assert result["criteria"][1]["criterion"] == "Device default."
    assert result["criteria"][1]["result"] == "not_verified"
    assert "omitted" in result["criteria"][1]["evidence"]


def test_duplicate_criteria_fail_inside_the_repair_loop_not_after_it():
    contract = "- First surface.\n- Second surface."
    entry = {"criterion": "First surface.", "result": "verified", "evidence": "visible"}
    bad = {**verdict("pass"), "criteria": [entry, entry]}
    fixed = {**verdict("fail"), "criteria": [entry, {
        "criterion": "Second surface.", "result": "failed", "evidence": "wrong palette visible",
    }]}
    sender = Sender([tool_reply("record_verdict", bad), tool_reply("record_verdict", fixed)])
    result = asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="theme", final_frame=frame(),
                                             contract=contract))
    assert result["verdict"] == "fail" and len(sender.payloads) == 2


def test_decider_rejects_bad_budgets_and_backends():
    with pytest.raises(RunError):
        Decider(Sender([]), model="m", backend="nowhere")
    with pytest.raises(RunError):
        Decider(Sender([]), model="m", request_config=SETTINGS, max_tokens=0)


def test_two_stances_must_agree_and_never_see_element_ids():
    sender = Sender([tool_reply("record_verdict", verdict("pass")),
                     tool_reply("record_verdict", verdict("pass", 0.7, ["Light row is selected"]))])
    result = asyncio.run(judge_outcome_votes(
        decider(sender), goal="Switch the theme to Light", final_frame=frame("fp-final"),
        frames=[frame("fp-1"), frame("fp-2")],
        actions=[{"step": 0, "tool": "tap_and_analyze", "arguments": {"id": "el:1"}, "reason": "secret controller thoughts"}]))
    assert result["verdict"] == "pass" and result["agreement"] is True
    assert result["oracle"] == ORACLE and result["verified"] is False
    assert result["confidence"] == 0.7
    assert [vote["stance"] for vote in result["votes"]] == ["neutral", "skeptical"]
    assert result["cost"] == pytest.approx(0.001)
    for payload in sender.payloads:
        evidence = payload["messages"][1]["content"]
        assert '"el:' not in evidence, "judges do not act, so they do not get handles"
        assert "secret controller thoughts" not in evidence
        assert "bounds" not in evidence
    assert "refute" in sender.payloads[1]["messages"][0]["content"].lower()


def test_judges_receive_resolved_target_names_in_order_without_handles_or_claims():
    sender = Sender([tool_reply("record_verdict", verdict("pass"))])
    actions = [{"step": step, "tool": "tap_and_analyze", "arguments": {"id": "el:opaque"},
                "resolved_target": {"text": label, "source": "previous_fresh_observation",
                                    "source_evidence_ref": f"E{step:04d}"},
                "reason": "untrusted controller narrative"}
               for step, label in enumerate(("Pin", "Unpin", "Save"))]
    asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="Review menu states",
                                    final_frame=frame(), actions=actions))
    evidence = sender.payloads[0]["messages"][1]["content"]
    assert evidence.index('"text": "Pin"') < evidence.index('"text": "Unpin"') < evidence.index('"text": "Save"')
    assert "E0000" in evidence and "E0002" in evidence
    assert "el:opaque" not in evidence and "untrusted controller narrative" not in evidence


def test_judge_relative_positions_prove_order_changes_without_pixel_bounds():
    frames = []
    for step, labels in ((20, ("First item", "Named item")),
                         (22, ("Named item", "First item")),
                         (24, ("First item", "Named item"))):
        raw = frame(f"fp-{step}", labels)
        raw["_judge_evidence"] = {"ref": f"E{step}", "after_step": step}
        raw["observation"]["elements"][0]["bounds"] = [10, 380, 600, 424]
        raw["observation"]["elements"][1]["bounds"] = [10, 496, 600, 540]
        frames.append(raw)
    originals = copy.deepcopy(frames)
    compact = [evidence_frame(raw) for raw in frames]
    for value in compact:
        elements = value["observation"]["elements"]
        assert elements[0]["center_pct"][1] < elements[1]["center_pct"][1]
        assert elements[0]["center_pct"] == [42.4, 31.4]
        assert all("id" not in element and "bounds" not in element for element in elements)
    assert [value["observation"]["elements"][0]["text"] for value in compact] == [
        "First item", "Named item", "First item"]
    assert frames == originals

    sender = Sender([tool_reply("record_verdict", verdict("unverified"))])
    actions = [{"step": step, "tool": "tap_and_analyze", "resolved_target": {
        "text": label, "source": "previous_fresh_observation"}}
        for step, label in ((22, "Pin"), (24, "Unpin"))]
    asyncio.run(judge_outcome_votes(decider(sender), votes=1, goal="Check row order",
                                    final_frame=frame(), frames=frames, actions=actions,
                                    images=[f"data:image/png;base64,{i}" for i in range(5)],
                                    image_evidence=[{"image_index": i + 1} for i in range(5)]))
    content = sender.payloads[0]["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 5
    context = json.loads(content[0]["text"].split("Evidence:\n", 1)[1].split("\n\nThe attached", 1)[0])
    assert len(context["image_evidence"]) == 5
    checkpoints = context["observed_order_transitions"][0]["checkpoints"]
    assert [item["evidence_position"]["ref"] for item in checkpoints] == ["E20", "E22", "E24"]
    assert [item["rows"][0]["text"] for item in checkpoints] == ["First item", "Named item", "First item"]
    assert "chronological host-captured post-action observations" in (
        context["observed_order_transitions"][0]["note"]
    )


@pytest.mark.parametrize("bounds", [[0, 0, float("nan"), 20], [0, 0, 0, 0],
                                    [0, -200, 10, -100], [0, 0, True, 5], [0, 1]])
def test_judge_rejects_invalid_relative_geometry(bounds):
    raw = frame()
    raw["observation"]["elements"][0]["bounds"] = bounds
    raw["observation"]["elements"][0]["center_pct"] = [99, 99]  # never trust supplied values
    assert "center_pct" not in evidence_frame(raw)["observation"]["elements"][0]


def test_judge_does_not_invent_positions_for_unlabeled_noninteractive_or_sizeless_frames():
    raw = frame()
    raw["observation"]["elements"][0].pop("text")
    raw["observation"]["elements"][1].pop("clickable")
    assert all("center_pct" not in item for item in evidence_frame(raw)["observation"]["elements"])
    raw = frame()
    raw["observation"]["screen"].pop("height")
    assert all("center_pct" not in item for item in evidence_frame(raw)["observation"]["elements"])


def test_contract_token_budget_scales_for_exact_per_criterion_output():
    contract = "\n".join(f"- criterion {index}" for index in range(21))
    assert contract_max_tokens(1200, contract) == 2612
    assert contract_max_tokens(1200, None) == 1200


def test_contract_criteria_preserve_wrapped_markdown_bullets():
    contract = (
        "## UX contract\n"
        "- The first interactive screen offers a way to\n"
        "  continue without an account.\n"
        "- Home is usable.\n"
        "\n"
        "Judgement guidance: a slow launch is only a warning."
    )

    assert contract_criteria(contract) == [
        "The first interactive screen offers a way to continue without an account.",
        "Home is usable.",
    ]


def test_failed_required_criterion_overrides_unverified_top_level_votes():
    criteria = [{
        "criterion": "Home is usable.",
        "result": "failed",
        "evidence": "Home stayed blank.",
    }]
    votes = [
        {"stance": stance, "criteria_order": ["Home is usable."], "cost": 0.0,
         "result": {"verdict": "unverified", "confidence": 0.8, "reasons": ["blank"],
                    "criteria": criteria}}
        for stance in ("neutral", "skeptical")
    ]

    assert combine_votes(votes)["verdict"] == "fail"


def test_contract_votes_report_each_exact_criterion_with_evidence():
    contract = "## UX contract\n- Theme row shows Light selected.\n- Main surfaces use a light palette."
    checks_neutral = [
        {"criterion": "Theme row shows Light selected.", "result": "verified", "evidence": "Theme row reads Light."},
        {"criterion": "Main surfaces use a light palette.", "result": "verified", "evidence": "The visible surface is light."},
    ]
    checks_skeptical = [
        {"criterion": "Theme row shows Light selected.", "result": "verified", "evidence": "Selected Light label is visible."},
        {"criterion": "Main surfaces use a light palette.", "result": "verified", "evidence": "No dark surface is shown."},
    ]
    first, second = verdict("pass"), verdict("pass", 0.8)
    first["criteria"], second["criteria"] = checks_neutral, checks_skeptical
    sender = Sender([
        tool_reply("record_verdict", first),
        tool_reply("record_verdict", second),
    ])

    result = asyncio.run(judge_outcome_votes(
        decider(sender),
        goal="Switch the theme to Light",
        final_frame=frame("fp-final"),
        contract=contract,
    ))

    assert [item["criterion"] for item in result["criteria"]] == [
        "Theme row shows Light selected.",
        "Main surfaces use a light palette.",
    ]
    judge_prompt = sender.payloads[0]["messages"][1]["content"]
    assert "A negative criterion is verified by evidence that the forbidden state is absent" in judge_prompt
    system_prompt = sender.payloads[0]["messages"][0]["content"]
    assert "A detour through another screen is a route warning" in system_prompt
    assert "conventional unlabeled controls" in system_prompt
    assert all(item["result"] == "verified" for item in result["criteria"])
    assert sender.payloads[0]["max_tokens"] == contract_max_tokens(512, contract)
    assert "neutral:" in result["criteria"][0]["evidence"]
    assert "skeptical:" in result["criteria"][0]["evidence"]


def test_disagreement_is_unverified_and_soft_pass_combinations():
    pass_warn = combine_votes([{"stance": "neutral", "result": verdict("pass"), "cost": 0},
                               {"stance": "skeptical", "result": verdict("pass_with_warning", 0.6), "cost": 0}])
    assert pass_warn["verdict"] == "pass_with_warning" and pass_warn["agreement"] is False
    split = combine_votes([{"stance": "neutral", "result": verdict("pass"), "cost": 0},
                           {"stance": "skeptical", "result": verdict("fail"), "cost": 0}])
    assert split["verdict"] == "unverified"
    blocked = combine_votes([{"stance": "neutral", "result": verdict("blocked"), "cost": 0},
                             {"stance": "skeptical", "result": verdict("fail"), "cost": 0}])
    assert blocked["verdict"] == "blocked"
    with pytest.raises(RunError):
        asyncio.run(judge_outcome_votes(decider(Sender([])), votes=3, goal="g", final_frame=frame()))


def test_screen_namer_caches_by_fingerprint_and_route_summary_uses_names():
    sender = Sender([
        tool_reply("record_screen_name", {"logical_name": "settings_theme", "kind": "settings",
                                          "purpose": "Pick the app theme.", "landmarks": ["Theme", "Light"]}),
        tool_reply("record_screen_name", {"logical_name": "home_feed", "kind": "home",
                                          "purpose": "Landing screen.", "landmarks": ["Chats"]}),
        tool_reply("record_route_summary", {"summary": "Home → Settings → Theme.", "landmarks": ["Theme"], "pitfalls": []}),
    ])
    judge = decider(sender)
    namer = ScreenNamer(judge)
    first = asyncio.run(namer.name(frame("fp-settings"), known_name="settings__1a2b"))
    again = asyncio.run(namer.name(frame("fp-settings"), known_name="settings__1a2b"))
    # Same heuristic label, new fingerprint (a row's value changed): no new call, no new entry.
    changed = asyncio.run(namer.name(frame("fp-settings-dark", ("Theme", "Dark")), known_name="settings__1a2b"))
    home = asyncio.run(namer.name(frame("fp-home", ("Chats",)), known_name="screen_3_ab12"))
    assert first["logical_name"] == "settings_theme" and again is first and changed is first
    assert first["fingerprints"] == ["fp-settings", "fp-settings-dark"]
    assert first["oracle"] == ORACLE and first["verified"] is False and first["fingerprint"] == "fp-settings"
    assert home["heuristic_name"] == "screen_3_ab12"
    assert "screen_3_ab12" in sender.payloads[1]["messages"][1]["content"]
    assert "names_already_assigned" in sender.payloads[1]["messages"][1]["content"]
    assert asyncio.run(namer.name({"ok": False})) is None
    assert len(sender.payloads) == 2, "the repeated label and fingerprint cost nothing"
    assert [item["logical_name"] for item in namer.distinct()] == ["settings_theme", "home_feed"]
    route = asyncio.run(summarize_route(judge, goal="g", screens=namer.distinct(),
                                        transitions=[{"from": "home_feed", "to": "settings_theme", "after": "tap_and_analyze"}]))
    assert route["summary"].startswith("Home") and route["oracle"] == ORACLE
    assert "settings_theme" in sender.payloads[2]["messages"][1]["content"]
    assert judge.report()["decisions"] == 3


def test_screen_namer_merges_when_the_model_reuses_an_existing_name():
    sender = Sender([
        tool_reply("record_screen_name", {"logical_name": "settings_theme", "kind": "settings",
                                          "purpose": "Pick the app theme.", "landmarks": ["Theme"]}),
        tool_reply("record_screen_name", {"logical_name": "settings_theme", "kind": "settings",
                                          "purpose": "Pick the app theme.", "landmarks": ["Theme", "Dark"]}),
    ])
    namer = ScreenNamer(decider(sender))
    first = asyncio.run(namer.name(frame("fp-a")))
    merged = asyncio.run(namer.name(frame("fp-b", ("Theme", "Dark"))))  # no heuristic label: model decides
    assert merged is first and first["fingerprints"] == ["fp-a", "fp-b"]
    assert len(namer.distinct()) == 1 and first["cost"] == pytest.approx(0.001)
