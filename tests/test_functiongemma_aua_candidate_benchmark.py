"""Regression gate for the independent AUA candidate-compiler benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.aua_candidate_benchmark import benchmark_cases, evaluate_cases


def test_candidate_benchmark_is_broad_balanced_and_strict() -> None:
    cases = benchmark_cases()

    assert len(cases) == 208
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.oracle_call is not None for case in cases) == 192
    assert sum(case.expect_policy_candidate for case in cases) == 128
    assert (
        sum(case.oracle_call is not None and not case.expect_policy_candidate for case in cases)
        == 64
    )
    assert sum(case.oracle_call is None for case in cases) == 16
    assert {case.family for case in cases} == {
        "open_from_choices",
        "tap_from_choices",
        "choose_among",
        "origin_then_select",
        "stale_refresh",
        "named_loading_wait",
        "unlabeled_progress_wait",
        "scroll_to_reveal",
        "disabled_target_abstain",
    }

    report = evaluate_cases(cases)

    assert report["passed"] is True, report["failures"][:5]
    assert report["failures"] == []
    assert report["metrics"]["target_extraction_accuracy"] == 1.0
    assert report["metrics"]["oracle_action_offered_rate"] == 1.0
    assert report["metrics"]["deterministic_action_accuracy"] == 1.0
    assert report["metrics"]["deterministic_recovery_accuracy"] == 1.0
    assert report["metrics"]["safe_abstention_rate"] == 1.0
