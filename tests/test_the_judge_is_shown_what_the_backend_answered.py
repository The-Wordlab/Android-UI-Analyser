"""A 200 from the app's own backend is evidence, and the judge was never shown it.

Measured on a real run. The contract asked whether a language change was still in effect after
leaving Settings; the judge marked it `not_verified` -- "no frame captures the Settings screen
after the change". The window between two observations held
`PUT /v1/profile -> 200`: the app telling its backend exactly that, and the backend
agreeing. The proxy had it the whole time and nothing carried it to the judge.

It is evidence about the server, not about pixels, and the instruction says so: a 200 proves the
app asked and was answered, never that anything rendered.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from experiments.aua_controller.judgement import evidence_frame, judge_outcome

CALLS = ["PUT /v1/profile -> 200", "POST /v1/send -> no answer yet"]


def _reply(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"model": "fictional/model", "provider": "fictional",
            "usage": {"cost": 0.0005, "prompt_tokens": 300, "completion_tokens": 40},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"type": "function", "id": "decide-1", "function": {
                    "name": name, "arguments": json.dumps(arguments)}}]}}]}


def _frame(calls: list[str] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"fingerprint": "fp-1"}
    if calls is not None:
        meta["network_calls"] = calls
    return {"observation": {"screen": {"package": "com.example.app"}, "meta": meta,
                            "elements": [{"id": "el:a", "text": "Idioma", "clickable": True,
                                          "bounds": [0, 0, 10, 10]}]}}


class _Decider:
    """Captures the payload instead of answering it."""

    max_tokens = 2048

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def decide(self, **kwargs: Any) -> Any:
        self.seen.append(kwargs)
        criteria = list(kwargs.get("criteria_order") or [])
        return {"result": {"verdict": "unverified", "confidence": 0.5, "reasons": [],
                           "criteria": [{"criterion": name, "result": "not_verified",
                                         "evidence": ""} for name in criteria]}}


def test_what_the_backend_answered_survives_into_a_judged_frame() -> None:
    kept = evidence_frame(_frame(CALLS))
    assert kept["observation"]["meta"]["network_calls"] == CALLS


def test_a_frame_with_no_backend_traffic_carries_no_such_key() -> None:
    assert "network_calls" not in evidence_frame(_frame())["observation"]["meta"]


def _asked(frames: list[dict[str, Any]]) -> dict[str, Any]:
    decider = _Decider()
    asyncio.run(judge_outcome(decider, goal="change the app language",
                              final_frame=frames[-1], frames=frames, actions=[],
                              contract="- the language change is saved"))
    assert decider.seen, "the judge was never asked"
    return decider.seen[0]


def test_the_judge_is_told_what_those_lines_are() -> None:
    """Unlabelled, `PUT /v1/profile -> 200` is a string the judge may simply skip."""
    asked = _asked([_frame(CALLS), _frame(CALLS)])
    assert any(entry.get("network") == CALLS for entry in asked["context"]["journey"])
    note = asked["context"].get("network_evidence_note")
    assert note and "backend" in note.lower()


def test_the_judge_is_warned_a_status_is_not_a_rendering() -> None:
    """The failure mode this invites: crediting a 200 as proof the screen showed something."""
    note = _asked([_frame(CALLS), _frame(CALLS)])["context"]["network_evidence_note"].lower()
    assert "render" in note or "drawn" in note or "screen" in note


def test_a_quiet_run_is_not_given_the_explanation() -> None:
    """Every sentence of instruction costs tokens on every judged run that cannot use it."""
    asked = _asked([_frame(), _frame()])
    assert "network_evidence_note" not in asked["context"]
    assert "network_calls" not in json.dumps(asked, default=str)


# ------------------------------------------------ the judge's own request, on the record


def test_the_judgement_log_keeps_the_request_that_produced_it(tmp_path: Any) -> None:
    """A verdict nobody can check the inputs of is an opinion with a number on it.

    The navigator writes its state, its questions and the raw answer for every call; the judge
    wrote only what came back. So when a judge marked a criterion unevidenced, there was no way
    to tell a model that reasoned badly from a model that was handed the wrong frames -- and on
    a real run it was the second: the frame that proved the clause had been dropped before the
    judge ever saw it.
    """
    from experiments.aua_controller.judgement import Decider

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return _reply("record_verdict",
                      {"verdict": "pass", "confidence": 0.9, "reasons": [], "criteria": []})

    out = tmp_path / "judge"
    decider = Decider(send, model="m", backend="openai-compatible", output=out)
    schema = {"type": "object", "properties": {"verdict": {"type": "string"},
                                               "confidence": {"type": "number"},
                                               "reasons": {"type": "array"},
                                               "criteria": {"type": "array"}},
              "required": ["verdict", "confidence", "reasons", "criteria"],
              "additionalProperties": False}
    asyncio.run(decider.decide(role="outcome judge (neutral)", instructions="judge it",
                               question="did it work?", context={"goal": "open settings"},
                               schema=schema, name="record_verdict"))

    entry = json.loads((out / "judgements.jsonl").read_text().strip())
    assert entry["request"]["question"] == "did it work?"
    assert entry["request"]["context"] == {"goal": "open settings"}
    assert "judge it" in entry["request"]["instructions"]
    assert entry["result"]["verdict"] == "pass"


def test_the_recorded_request_does_not_carry_the_screenshots(tmp_path: Any) -> None:
    """A base64 frame is megabytes and proves nothing a reader of the log can check.

    The count is what matters -- whether the judge was looking at pictures at all.
    """
    from experiments.aua_controller.judgement import Decider

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return _reply("r", {"ok": True})

    out = tmp_path / "judge"
    decider = Decider(send, model="m", backend="openai-compatible", output=out)
    asyncio.run(decider.decide(
        role="r", instructions="i", question="q", context={}, name="r",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"], "additionalProperties": False},
        images=["data:image/jpeg;base64," + "A" * 5000]))
    raw = (out / "judgements.jsonl").read_text()
    assert "AAAAA" not in raw
    assert json.loads(raw.strip())["images"] == 1
