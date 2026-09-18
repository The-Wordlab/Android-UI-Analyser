"""A System One judge returns numbers; the verdict is composed here, in ordinary code.

These tests never reach the network and never import the TypeSafe SDK. ``compose`` is the
whole decision, so it is exercised directly with plain answer stand-ins: anything carrying
``.score``/``.confidence`` for a level judgement and ``.noul`` for a yes/no one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.typesafe_judge import (  # noqa: E402
    EVIDENCE_LEVELS,
    ORACLE,
    compose,
)

CRITERIA = [
    "The notifications screen is reachable from settings",
    "The promotional messages switch is off",
    "No error dialog is shown",
]
TOP = len(EVIDENCE_LEVELS) - 1


def score(level: float, confidence: float = 0.9) -> SimpleNamespace:
    """A Score answer sits on the authored level ladder, and may land between levels."""
    return SimpleNamespace(score=level, confidence=confidence)


def noul(probability: float) -> SimpleNamespace:
    return SimpleNamespace(noul=probability)


def answers(levels: list[float], *, blocked: float = 0.02, reached: float = 0.97,
            confidences: list[float] | None = None) -> dict[str, SimpleNamespace]:
    confs = confidences or [0.9] * len(levels)
    out: dict[str, SimpleNamespace] = {
        f"c{index}": score(level, conf) for index, (level, conf) in enumerate(zip(levels, confs, strict=True))
    }
    out["blocked"] = noul(blocked)
    out["reached"] = noul(reached)
    return out


def test_every_criterion_directly_evidenced_is_a_pass() -> None:
    result = compose(answers([TOP, TOP, TOP]), CRITERIA)
    assert result["verdict"] == "pass"
    assert [entry["result"] for entry in result["criteria"]] == ["verified"] * 3


def test_a_contradicted_criterion_fails_the_whole_run() -> None:
    # Level 0 is "an observed screen plainly shows the OPPOSITE of this".
    result = compose(answers([TOP, 0.1, TOP]), CRITERIA)
    assert result["verdict"] == "fail"
    assert result["criteria"][1]["result"] == "failed"
    assert CRITERIA[1] in result["unsatisfied"]


def test_evidence_that_is_only_implied_passes_with_a_warning() -> None:
    # Level 2 is "implied by what was observed, but not shown outright": short of proof,
    # but not absent and not contradicted.
    result = compose(answers([TOP, TOP, 2.0]), CRITERIA)
    assert result["verdict"] == "pass_with_warning"
    assert result["criteria"][2]["result"] == "not_verified"


def test_evidence_that_is_simply_absent_is_unverified_not_a_pass() -> None:
    # Level 1 is "nothing on the observed screens speaks to this either way". Silence is
    # never a pass, and it is never a failure either.
    result = compose(answers([TOP, 1.0, TOP]), CRITERIA)
    assert result["verdict"] == "unverified"
    assert result["criteria"][1]["result"] == "not_verified"


def test_a_blocker_outside_the_feature_outranks_the_criteria() -> None:
    result = compose(answers([TOP, TOP, TOP], blocked=0.98, reached=0.03), CRITERIA)
    assert result["verdict"] == "blocked"
    assert result["blocker"]


def test_a_run_that_never_reached_the_screen_is_unverified() -> None:
    result = compose(answers([1.0, 1.0, 2.6], blocked=0.10, reached=0.12), CRITERIA)
    assert result["verdict"] == "unverified"
    assert result["blocker"] is None


def test_criteria_come_back_once_each_in_authored_order() -> None:
    result = compose(answers([TOP, 0.1, 2.0]), CRITERIA)
    assert [entry["criterion_index"] for entry in result["criteria"]] == [0, 1, 2]


def test_confidence_is_the_weakest_judgement_in_the_chain() -> None:
    result = compose(answers([TOP, TOP, TOP], confidences=[0.99, 0.41, 0.88]), CRITERIA)
    assert result["confidence"] == pytest.approx(0.41)


def test_the_judge_never_invents_quoted_evidence() -> None:
    # A System One model returns no text. Every string here must be derived from the
    # authored criterion and the level it landed on, never presented as an observation.
    result = compose(answers([TOP, 0.1, 2.0]), CRITERIA)
    for entry, criterion in zip(result["criteria"], CRITERIA, strict=True):
        assert criterion in entry["evidence"]
        assert any(level.split(":")[0] in entry["evidence"] for level in EVIDENCE_LEVELS)


def test_a_result_is_an_opinion_and_says_so() -> None:
    result = compose(answers([TOP, TOP, TOP]), CRITERIA)
    assert result["oracle"] == ORACLE
    assert result["verified"] is False


def test_an_answer_set_that_misses_a_criterion_is_refused() -> None:
    incomplete = answers([TOP, TOP, TOP])
    del incomplete["c2"]
    with pytest.raises(KeyError):
        compose(incomplete, CRITERIA)


def test_an_empty_contract_is_refused_rather_than_silently_passing() -> None:
    with pytest.raises(ValueError):
        compose(answers([]), [])
