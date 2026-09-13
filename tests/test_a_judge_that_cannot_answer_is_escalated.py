"""When the judge cannot produce its own schema, ask a better model rather than asking again.

2026-09-14 run 3 of the controller harness::

    decider could not obtain a valid structured answer: 'reasons' is a required property

The default repair budget is one, so the judge got two tries at the same model and the run
came back BLOCKED. Repeating the same ask a third time is the least likely thing to work: a
model that cannot emit a required field is usually not one field away from it.

So the decider walks a ladder. Each rung spends its full repair budget, then the next, stronger
model reads the same thread - including the message saying what was wrong - and answers. The
ladder is recorded, because a judge that is quietly being carried by its fallback is a finding
about the cheap model, not a detail.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from experiments.aua_controller.judgement import Decider
from experiments.aua_controller.run_live import RunError

SCHEMA = {"type": "object", "properties": {"reasons": {"type": "string"}},
          "required": ["reasons"], "additionalProperties": False}
CAPS = {"prompt": 0.3, "completion": 1.2}
CHEAP = {"provider": {"allow_fallbacks": True, "sort": "throughput", "max_price": CAPS}}
STRONG = {"provider": {"allow_fallbacks": True, "max_price": CAPS}}


def answer(name, arguments, *, model="cheap/model", provider="DeepInfra"):
    return {"model": model, "provider": provider,
            "usage": {"cost": 0.0001, "prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": "tool_calls",
                         "message": {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def decider(send, **kwargs):
    return Decider(send, model="cheap/model", request_config=CHEAP,
                   cost_limit_usd=1.0, **kwargs)


def decide(subject):
    return asyncio.run(subject.decide(
        role="judge", instructions="Judge it.", name="verdict", schema=SCHEMA,
        question="Did the guest reach home?", context={"goal": "land home"},
    ))


def test_a_good_answer_never_touches_the_ladder():
    calls = []

    async def send(payload):
        calls.append(payload["model"])
        return answer("verdict", {"reasons": "it landed home"})

    subject = decider(send, fallbacks=[("strong/model", STRONG)])
    out = decide(subject)
    assert out["result"]["reasons"] == "it landed home"
    assert calls == ["cheap/model"] and out["escalations"] == 0


def test_the_cheap_model_spends_its_whole_repair_budget_first():
    calls = []

    async def send(payload):
        calls.append(payload["model"])
        return answer("verdict", {"wrong": "field"})

    subject = decider(send, repair_budget=1, fallbacks=[("strong/model", STRONG)])
    with pytest.raises(RunError):
        decide(subject)
    assert calls == ["cheap/model", "cheap/model", "strong/model", "strong/model"]


def test_the_stronger_model_rescues_the_run():
    calls = []

    async def send(payload):
        calls.append(payload["model"])
        if payload["model"] == "cheap/model":
            return answer("verdict", {"wrong": "field"})
        return answer("verdict", {"reasons": "guest reached home"}, model="strong/model")

    subject = decider(send, repair_budget=1, fallbacks=[("strong/model", STRONG)])
    out = decide(subject)
    assert out["result"]["reasons"] == "guest reached home"
    assert out["escalations"] == 1
    assert out["model"] == "strong/model", "the answer must be attributed to who actually gave it"


def test_each_rung_carries_its_own_routing_settings():
    seen = {}

    async def send(payload):
        seen[payload["model"]] = payload["provider"]
        if payload["model"] == "cheap/model":
            return answer("verdict", {"wrong": "field"})
        return answer("verdict", {"reasons": "ok"}, model="strong/model")

    decide(decider(send, repair_budget=0, fallbacks=[("strong/model", STRONG)]))
    assert seen["cheap/model"].get("sort") == "throughput"
    assert "sort" not in seen["strong/model"], "a rung must not inherit the rung below's routing"


def test_a_ladder_that_all_fails_names_every_model_it_asked():
    async def send(payload):
        return answer("verdict", {"wrong": "field"})

    subject = decider(send, repair_budget=0, fallbacks=[("strong/model", STRONG)])
    with pytest.raises(RunError, match="cheap/model -> strong/model"):
        decide(subject)


def test_the_report_shows_a_cheap_judge_being_carried_by_its_fallback():
    async def send(payload):
        if payload["model"] == "cheap/model":
            return answer("verdict", {"wrong": "field"})
        return answer("verdict", {"reasons": "ok"}, model="strong/model")

    subject = decider(send, repair_budget=0, fallbacks=[("strong/model", STRONG)])
    decide(subject)
    decide(subject)
    report = subject.report()
    assert report["ladder"] == ["cheap/model", "strong/model"]
    assert report["escalations"] == 2 and report["decisions"] == 2


def test_no_ladder_behaves_exactly_as_before():
    calls = []

    async def send(payload):
        calls.append(payload["model"])
        return answer("verdict", {"wrong": "field"})

    with pytest.raises(RunError):
        decide(decider(send, repair_budget=1))
    assert calls == ["cheap/model", "cheap/model"]


def test_a_nameless_rung_is_refused_at_construction_not_mid_judgement():
    async def send(payload):
        return answer("verdict", {"reasons": "ok"})

    with pytest.raises(RunError, match="fallback"):
        decider(send, fallbacks=[("  ", STRONG)])
