"""A 200 from the app's own backend is evidence, and the judge was never shown it.

Measured on a real run. The contract asked whether a language change was still in effect after
leaving Settings; the judge marked it `not_verified` -- "no frame captures the Settings screen
after the change". The window between two observations held
`PUT /api/v4.0/user/profile -> 200`: the app telling its backend exactly that, and the backend
agreeing. The proxy had it the whole time and nothing carried it to the judge.

It is evidence about the server, not about pixels, and the instruction says so: a 200 proves the
app asked and was answered, never that anything rendered.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from experiments.aua_controller.judgement import evidence_frame, judge_outcome

CALLS = ["PUT /api/v4.0/user/profile -> 200", "POST /api/v4.0/send -> no answer yet"]


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
    """Unlabelled, `PUT /api/v4.0/user/profile -> 200` is a string the judge may simply skip."""
    asked = _asked([_frame(CALLS), _frame(CALLS)])
    assert "network_calls" in json.dumps(asked, default=str)
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
