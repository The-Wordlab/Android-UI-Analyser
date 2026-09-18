"""The run picks its judge explicitly, and a System One verdict fits where the other one did.

The two judges answer the same question with different machinery: the chat judge writes its
own reasons and can read screenshots, the System One judge scores authored bullets and does
neither. Downstream code reads one verdict shape regardless, so that shape is pinned here.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller import run_realapp  # noqa: E402
from experiments.aua_controller.judgement import VERDICTS, contract_criteria  # noqa: E402
from experiments.aua_controller.typesafe_judge import TypeSafeJudge  # noqa: E402

CONTRACT = """
- The notifications screen is reachable from settings
- The promotional messages switch is off
"""
FRAME = {
    "ok": True,
    "observation": {
        "screen": {"package": "com.example.demo", "activity": ".NotificationSettings"},
        "meta": {"fingerprint": "fp-final"},
        "elements": [
            {"id": "rid:sw_promos", "text": "Promotional messages", "checkable": True,
             "checked": False, "clickable": True, "bounds": [0, 100, 720, 200]},
        ],
    },
}


class FakeClient:
    def __init__(self, levels: list[float]) -> None:
        self.levels = levels

    def system_one(self, *, state, questions, model):
        answers = {f"c{i}": SimpleNamespace(score=level, confidence=0.9)
                   for i, level in enumerate(self.levels)}
        answers["blocked"] = SimpleNamespace(noul=0.02)
        answers["reached"] = SimpleNamespace(noul=0.97)
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=700))


def test_the_run_takes_a_judge_engine_and_keeps_the_chat_one_as_default() -> None:
    # Swapping the judge is an opt-in: an existing invocation keeps the judge it had.
    parameter = inspect.signature(run_realapp.run_realapp).parameters["judge_engine"]
    assert parameter.default == "chat"


def test_the_command_line_offers_both_judges(capsys: pytest.CaptureFixture[str]) -> None:
    sys.argv = ["run_realapp", "--help"]
    with pytest.raises(SystemExit) as exit_code:
        run_realapp.main()
    assert exit_code.value.code == 0
    help_text = capsys.readouterr().out
    assert "--judge-engine" in help_text
    assert "typesafe" in help_text


def test_a_system_one_verdict_carries_everything_the_harness_reads() -> None:
    criteria = contract_criteria(CONTRACT)
    verdict = TypeSafeJudge(FakeClient([3.0, 3.0])).judge(
        goal="Turn off promotional messages", criteria=criteria, final_frame=FRAME
    )
    # run_realapp adds these two before handing the verdict on, so the shape matches the
    # chat judge's for every reader downstream.
    verdict["votes"] = []
    verdict["cost"] = 0.0

    assert verdict["verdict"] in VERDICTS
    # The stop-reason adjustments insert into reasons, and the evidence-gap check reads
    # criteria and votes; a missing key there would only surface on a real run.
    verdict["reasons"].insert(0, "Controller stalled on an unchanged screen before finishing.")
    assert isinstance(verdict["criteria"], list) and verdict["criteria"]
    assert all(set(entry) == {"criterion_index", "result", "evidence"} for entry in verdict["criteria"])
    assert verdict["votes"] == [] and verdict["cost"] == 0.0
    assert verdict["verified"] is False


def test_the_contract_bullets_are_what_the_judge_scores() -> None:
    criteria = contract_criteria(CONTRACT)
    assert criteria == [
        "The notifications screen is reachable from settings",
        "The promotional messages switch is off",
    ]
    client = FakeClient([3.0, 3.0])
    TypeSafeJudge(client).judge(goal="g", criteria=criteria, final_frame=FRAME)


def test_a_run_with_no_authored_contract_has_nothing_for_this_judge_to_score() -> None:
    # The guard in run_realapp refuses rather than falling back, so a result never hides
    # which model gave it.
    assert contract_criteria(None) == []
    assert contract_criteria("a goal sentence with no bullets") == []
