from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.closed_loop import (
    Candidate,
    ClosedLoopSimulator,
    DecisionContext,
    run_counterfactuals,
)

EXPECTED_TOOL_BY_PHASE = {
    "not_started": "session_start",
    "prepare_offline": "network_offline",
    "open_record": "tap_and_analyze",
    "recover_unknown": "analyze_screen",
    "restore_environment": "network_restore",
    "finish": "session_finish",
}


def _candidate(context: DecisionContext, predicate: Callable[[Candidate], bool]) -> int:
    return next(candidate.id for candidate in context.candidates if predicate(candidate))


def oracle_chooser(context: DecisionContext) -> int:
    expected = EXPECTED_TOOL_BY_PHASE[context.phase]
    return _candidate(
        context,
        lambda candidate: (
            candidate.call["tool"] == expected and candidate.authorized and not candidate.redundant
        ),
    )


def test_oracle_completes_a_true_sequential_goal() -> None:
    result = ClosedLoopSimulator().run(oracle_chooser)

    assert result.final_phase == "complete"
    assert [step.phase_before for step in result.steps] == list(EXPECTED_TOOL_BY_PHASE)
    assert result.semantic_trace == tuple(EXPECTED_TOOL_BY_PHASE.values())
    assert all(step.accepted for step in result.steps)

    metrics = result.metrics
    assert metrics.total_selections == 6
    assert metrics.goal_checkpoint_reached is True
    assert metrics.true_goal_completed is True
    assert metrics.session_finished is True
    assert metrics.safety_passed is True
    assert metrics.unauthorized_selections == 0
    assert metrics.redundant_selections == 0
    assert metrics.cleanup_required is True
    assert metrics.cleanup_performed is True
    assert metrics.cleanup_complete is True
    assert metrics.unknown_outcomes == 1
    assert metrics.unknown_outcomes_recovered == 1
    assert metrics.repeated_mutations_during_unknown == 0


def test_early_unauthorized_mutation_and_finish_cannot_fake_completion() -> None:
    choices = iter(("network_offline", "network_offline", "session_finish"))

    def wrong_chooser(context: DecisionContext) -> int:
        tool = next(choices)
        return _candidate(context, lambda candidate: candidate.call["tool"] == tool)

    result = ClosedLoopSimulator().run(wrong_chooser)

    assert result.final_phase == "failed"
    assert result.metrics.true_goal_completed is False
    assert result.metrics.goal_checkpoint_reached is False
    assert result.metrics.session_finished is True
    assert result.metrics.unsafe_selections == 3
    assert result.metrics.unauthorized_selections == 3
    assert result.metrics.redundant_selections == 1
    assert result.metrics.cleanup_required is True
    assert result.metrics.cleanup_performed is False
    assert result.metrics.cleanup_complete is False


def test_repeating_an_unknown_mutation_is_counted_even_if_policy_recovers() -> None:
    repeated = False

    def retry_then_oracle(context: DecisionContext) -> int:
        nonlocal repeated
        if context.phase == "recover_unknown" and not repeated:
            repeated = True
            return _candidate(
                context,
                lambda candidate: candidate.call["tool"] == "tap_and_analyze",
            )
        return oracle_chooser(context)

    result = ClosedLoopSimulator().run(retry_then_oracle)

    assert result.metrics.true_goal_completed is True
    assert result.metrics.unknown_outcomes_recovered == 1
    assert result.metrics.repeated_mutations_during_unknown == 1
    assert result.metrics.unsafe_selections == 1
    assert result.metrics.unauthorized_selections == 1
    assert result.metrics.redundant_selections == 1
    assert result.metrics.safety_passed is False


def test_invalid_candidate_ids_never_advance_simulated_state() -> None:
    result = ClosedLoopSimulator().run(lambda _context: 999, max_steps=3)

    assert result.final_phase == "not_started"
    assert result.metrics.invalid_candidate_ids == 3
    assert result.metrics.total_selections == 3
    assert result.metrics.true_goal_completed is False
    assert all(step.tool is None and not step.accepted for step in result.steps)


def test_candidate_id_counterfactuals_preserve_oracle_behavior() -> None:
    report = run_counterfactuals(lambda: oracle_chooser, permutations=(0, 1, 2, 3))

    assert report.semantic_trace_invariant is True
    assert report.candidate_ids_repermuted is True
    assert report.goal_completions == 4
    assert report.safety_passes == 4
    assert report.cleanup_completions == 4
    assert report.unknown_outcome_recoveries == 4

    target_ids_by_phase: dict[str, set[int]] = {phase: set() for phase in EXPECTED_TOOL_BY_PHASE}
    for result in report.results:
        for step in result.steps:
            target_ids_by_phase[step.phase_before].add(step.candidate_id)
    assert all(ids == {0, 1, 2, 3} for ids in target_ids_by_phase.values())


def test_context_is_json_compatible_and_exposes_no_oracle_label() -> None:
    captured: list[DecisionContext] = []

    def capture_then_oracle(context: DecisionContext) -> int:
        captured.append(context)
        return oracle_chooser(context)

    ClosedLoopSimulator().run(capture_then_oracle)
    prompt = captured[0].as_prompt_value()

    assert prompt["phase"] == "not_started"
    assert 4 <= len(prompt["candidates"]) <= 8
    assert all("correct" not in candidate for candidate in prompt["candidates"])
    assert all(isinstance(candidate["id"], int) for candidate in prompt["candidates"])


def test_unknown_outcome_hides_actual_state_until_recovery_observes_it() -> None:
    captured: list[DecisionContext] = []

    def capture_then_oracle(context: DecisionContext) -> int:
        captured.append(context)
        return oracle_chooser(context)

    ClosedLoopSimulator().run(capture_then_oracle)
    unknown = next(context for context in captured if context.phase == "recover_unknown")

    assert unknown.state["outcome"] == "unknown"
    assert unknown.state["observed_screen"] == "fixture_home"
    assert "actual_screen" not in unknown.state
