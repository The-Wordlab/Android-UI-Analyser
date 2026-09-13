from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.hosted import HostedError
from experiments.aua_controller.judgement import (
    ORACLE,
    OUTCOME_SCHEMA,
    Decider,
    ScreenNamer,
    combine_votes,
    judge_outcome_votes,
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


def test_decider_has_its_own_spend_stop():
    sender = Sender([tool_reply("record_verdict", verdict("pass"), cost=0.02),
                     tool_reply("record_verdict", verdict("pass"), cost=0.02)])
    judge = decider(sender, cost_limit_usd=0.03)
    asyncio.run(judge.decide(role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    asyncio.run(judge.decide(role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    with pytest.raises(HostedError, match="cost limit"):
        asyncio.run(judge.decide(role="r", instructions="i", question="q", context={}, schema=OUTCOME_SCHEMA, name="record_verdict"))
    assert judge.report()["decisions"] == 2 and judge.report()["reported_usd"] == pytest.approx(0.04)


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
