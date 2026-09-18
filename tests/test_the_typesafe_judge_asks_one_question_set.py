"""The judge spends one request per run, and reads the evidence the incumbent judge reads.

These tests use the real SDK question types and the real frame projection, with a stand-in
client, so the wiring is exercised without reaching the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("typesafe_sdk", reason="install the `typesafe` extra to exercise the SDK wiring")

from experiments.aua_controller.typesafe_judge import (  # noqa: E402
    EVIDENCE_LEVELS,
    MAX_FRAMES,
    TypeSafeJudge,
    build_questions,
    build_state,
)

CRITERIA = [
    "The notifications screen is reachable from settings",
    "The promotional messages switch is off",
]

FINAL_FRAME = {
    "ok": True,
    "elements": [
        {"id": "rid:sw_promos", "text": "Promotional messages", "resource_id": "com.example.demo:id/sw_promos",
         "clickable": True, "bounds": [0, 100, 720, 200]},
        {"id": "t:1", "text": "Notifications", "bounds": [0, 0, 720, 100]},
    ],
}


class FakeClient:
    """Records the one call it is given and replays a canned answer set."""

    def __init__(self, levels: list[float], *, blocked: float = 0.02, reached: float = 0.98) -> None:
        self.levels = levels
        self.blocked = blocked
        self.reached = reached
        self.calls: list[dict[str, object]] = []

    def system_one(self, *, state, questions, model):
        self.calls.append({"state": state, "questions": questions, "model": model})
        answers = {
            f"c{index}": SimpleNamespace(score=level, confidence=0.9)
            for index, level in enumerate(self.levels)
        }
        answers["blocked"] = SimpleNamespace(noul=self.blocked)
        answers["reached"] = SimpleNamespace(noul=self.reached)
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=612, output_tokens=40))


def test_one_run_costs_one_request_carrying_every_question() -> None:
    client = FakeClient([3.0, 3.0])
    judge = TypeSafeJudge(client)
    result = judge.judge(goal="Turn off promotional messages", criteria=CRITERIA, final_frame=FINAL_FRAME)

    assert len(client.calls) == 1
    # One Score per authored bullet, plus the two run-level Nouls, in a single parallel pass.
    assert set(client.calls[0]["questions"]) == {"c0", "c1", "blocked", "reached"}
    assert result["verdict"] == "pass"
    assert judge.report() == {
        "oracle": result["oracle"], "model": "jev-latest", "requests": 1, "input_tokens": 612,
    }


def test_the_questions_are_real_sdk_types_on_the_authored_ladder() -> None:
    questions = build_questions(CRITERIA)
    assert questions["c0"].type == "score"
    assert list(questions["c0"].criteria) == list(EVIDENCE_LEVELS)
    assert questions["blocked"].type == "noul"
    assert questions["reached"].type == "noul"


def test_a_contract_with_no_bullets_is_refused_before_a_request_is_spent() -> None:
    with pytest.raises(ValueError):
        build_questions([])


def test_the_state_carries_projected_frames_not_raw_ones() -> None:
    state = build_state(
        goal="Turn off promotional messages",
        criteria=CRITERIA,
        final_frame=FINAL_FRAME,
        actions=[{"step": 1, "tool": "tap_and_analyze", "arguments": {"id": "rid:row_notifications"}}],
    )
    assert state["contract"] == CRITERIA
    # Element handles are an actor's vocabulary; a judge that can cite them starts arguing
    # about the route instead of the evidence.
    assert "rid:sw_promos" not in repr(state["final_screen"])
    # The action log keeps what happened, never the ids it happened to.
    assert state["actions_taken"] == [{"step": 1, "tool": "tap_and_analyze"}]


def test_a_long_run_sends_a_bounded_number_of_screens() -> None:
    state = build_state(
        goal="Turn off promotional messages",
        criteria=CRITERIA,
        final_frame=FINAL_FRAME,
        frames=[FINAL_FRAME] * (MAX_FRAMES + 5),
    )
    # Jev's ceiling is 32k tokens of state, and its accuracy falls as unrelated material
    # crowds the decision, so an unbounded run must not send an unbounded state.
    assert len(state["intermediate_screens"]) == MAX_FRAMES


def test_a_blocked_run_is_reported_without_consulting_the_criteria() -> None:
    client = FakeClient([3.0, 3.0], blocked=0.97, reached=0.02)
    result = TypeSafeJudge(client).judge(
        goal="Turn off promotional messages", criteria=CRITERIA, final_frame=FINAL_FRAME
    )
    assert result["verdict"] == "blocked"
    assert result["blocker"]


def test_thresholds_reach_compose_so_a_caller_can_tighten_the_bar() -> None:
    client = FakeClient([3.0, 2.4])
    common = {"goal": "Turn off promotional messages", "criteria": CRITERIA, "final_frame": FINAL_FRAME}
    assert TypeSafeJudge(client).judge(**common)["verdict"] == "pass_with_warning"
    # Demanding less proof turns the same answers into a pass; the model never changed.
    assert TypeSafeJudge(client).judge(**common, verified_at=0.75)["verdict"] == "pass"
