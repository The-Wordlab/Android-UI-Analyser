"""Semantic goal checkpoints own lifecycle clauses and require relevant proof."""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.session import (
    GoalPhase,
    complete_environment_phase,
    create_session_state,
    finish_session_state,
    goal_phases,
    mark_phase_complete,
    phase_progress,
)

# This preserves the evaluator's exact grammar while using public, fictional placeholders.
_EVALUATOR_SHAPED_GOAL = (
    "In Example App, reuse visible recent Vocabulary and Equation threads if present; "
    "create online fixtures only if missing. Switch offline with AUA's verified reversible "
    "command. From Tool Shelf/tool recents, open Vocabulary's threaded recent exactly through "
    "user-visible taps and verify cached thread content opens without remaining behind Loading. "
    "Return through UI using bounded back navigation where applicable, open Equation's recent "
    "through user-visible taps, and verify cached content plus an enabled interaction affordance "
    "offline. Never use a deep link to bypass either recents-item tap. Finish session cleanup "
    "and verify connectivity restored."
)


def _state(tmp_path: Path, goal: str):
    return create_session_state(
        tmp_path,
        goal=goal,
        serial="example-serial",
        owner="example-agent",
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )


def test_legacy_phase_payload_remains_valid_without_semantic_metadata() -> None:
    phase = GoalPhase.model_validate(
        {"id": "phase_1", "objective": "Inspect the catalog", "status": "active"}
    )

    assert phase.intent is None
    assert phase.satisfaction is None
    assert phase.source_span is None
    assert phase.branches == []
    assert phase.proof is None


def test_evaluator_shaped_goal_compiles_to_six_semantic_checkpoints() -> None:
    phases = goal_phases(_EVALUATOR_SHAPED_GOAL)

    assert [phase.kind for phase in phases] == [
        "verify",
        "environment",
        "verify",
        "verify",
        "verify",
        "cleanup",
    ]
    assert [phase.intent for phase in phases] == [
        "alternative",
        "offline_transition",
        "ui_verification",
        "ui_verification",
        "ui_verification",
        "cleanup_finalizer",
    ]
    assert [(branch.id, branch.condition) for branch in phases[0].branches] == [
        ("branch_present", "present"),
        ("branch_missing", "missing"),
    ]
    assert phases[1].satisfaction == "verified_offline"
    assert phases[-1].satisfaction == "session_cleanup"
    assert phases[-1].terminal is True
    assert all(not phase.terminal for phase in phases[:-1])
    assert phases[4].constraints == [
        "Never use a deep link to bypass either recents-item tap"
    ]
    assert not any(
        phrase in phase.objective.casefold()
        for phase in phases
        for phrase in ("verified reversible command", "never use a deep link", "finish session cleanup")
    )

    offline_start, offline_end = phases[1].source_span or (-1, -1)
    cleanup_start, cleanup_end = phases[-1].source_span or (-1, -1)
    assert _EVALUATOR_SHAPED_GOAL[offline_start:offline_end].startswith("Switch offline")
    assert _EVALUATOR_SHAPED_GOAL[cleanup_start:cleanup_end].startswith("Finish session cleanup")


@pytest.mark.parametrize(
    "goal",
    [
        (
            "Switch offline using AUA's verified reversible method; inspect account cache; "
            "restore connectivity and finish cleanup"
        ),
        (
            "Enter airplane mode via the safe AUA network command. Inspect account cache. "
            "Re-enable Wi-Fi before ending the session."
        ),
        (
            "Go fully offline; then use AUA's verified reversible method; inspect account "
            "cache; reconnect internet then complete the session"
        ),
    ],
)
def test_transition_and_finalizer_paraphrases_do_not_leak_verify_phases(goal: str) -> None:
    phases = goal_phases(goal)

    assert [phase.intent for phase in phases] == [
        "offline_transition",
        "ui_verification",
        "cleanup_finalizer",
    ]
    assert phases[1].objective.casefold() == "inspect account cache"


def test_runtime_scope_plus_restoring_cleanup_is_one_terminal_phase() -> None:
    phases = goal_phases(
        "Repeat offline regression: verify cached Vocabulary recent opens offline, then "
        "Equation recent opens with an enabled affordance; use already-running "
        "emulator-5554 and clean up restoring connectivity"
    )

    assert [phase.intent for phase in phases] == [
        "offline_transition",
        "ui_verification",
        "ui_verification",
        "cleanup_finalizer",
    ]
    assert phases[-1].terminal is True
    assert "emulator-5554" not in phases[-1].objective


@pytest.mark.parametrize(
    ("goal", "conditions"),
    [
        (
            "Reuse saved conversations if available; create seed fixtures only if absent",
            ["present", "missing"],
        ),
        (
            "Use the existing profile if present; provision one only if missing",
            ["present", "missing"],
        ),
    ],
)
def test_if_present_if_missing_pair_is_one_alternative(goal: str, conditions: list[str]) -> None:
    phases = goal_phases(goal)

    assert len(phases) == 1
    assert phases[0].intent == "alternative"
    assert [branch.condition for branch in phases[0].branches] == conditions


def test_manual_phase_evidence_rejects_unrelated_claim_and_records_relevant_proof(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, "Verify the account balance on the details page")

    with pytest.raises(ValueError, match="evidence does not substantiate"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="Battery health is excellent",
        )

    persisted = phase_progress(state)
    assert persisted["done"] is False
    completed = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence="Account balance is shown on the details page",
    )
    phase = completed.phases[0]
    assert phase.status == "completed"
    assert phase.proof is not None
    assert phase.proof.source == "manual_evidence"
    assert {"account", "balance", "detail"} <= set(phase.proof.matched_terms)


def test_manual_phase_evidence_rejects_a_relevant_but_explicitly_unfinished_claim(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, "Verify the cached Vocabulary recent opens correctly")

    with pytest.raises(ValueError, match="remains unfinished"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence=(
                "Equation cached recent is visible offline; Vocabulary still needs verification"
            ),
        )


def test_long_alternative_rejects_one_weak_overlap_and_advertises_proof_contract(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, _EVALUATOR_SHAPED_GOAL)
    checkpoint = phase_progress(state)["checkpoint"]

    assert checkpoint["proof_required"] is True
    assert checkpoint["minimum_relevant_terms"] == 2
    with pytest.raises(ValueError, match="at least 2 distinct"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="Example App launcher visible",
        )


@pytest.mark.parametrize(
    ("evidence", "branch_id"),
    [
        ("Saved conversations are present and were reused", "branch_present"),
        ("Seed fixtures were missing and were created", "branch_missing"),
    ],
)
def test_evidence_for_either_alternative_completes_the_group_without_demanding_other_branch(
    tmp_path: Path,
    evidence: str,
    branch_id: str,
) -> None:
    state = _state(
        tmp_path,
        "Reuse saved conversations if present; create seed fixtures only if missing",
    )

    completed = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence=evidence,
    )

    assert phase_progress(completed)["done"] is True
    assert completed.phases[0].proof is not None
    assert completed.phases[0].proof.branch_id == branch_id


def test_verified_environment_event_and_cleanup_complete_the_semantic_goal(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, _EVALUATOR_SHAPED_GOAL)
    state = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence="Vocabulary and Equation recent threads were present and reused",
    )
    assert phase_progress(state)["checkpoint"] is None
    state = complete_environment_phase(
        tmp_path,
        state,
        command="network_offline",
        result={"ok": True, "verified": True, "state": {"offline": True}},
    )
    assert state.phases[1].proof is not None
    assert state.phases[1].proof.source == "verified_event"

    for phase_id, evidence in (
        ("phase_3", "Vocabulary cached thread content had no Loading state"),
        ("phase_4", "Bounded back navigation returned through the UI"),
        ("phase_5", "Equation cached content and interaction affordance were available offline"),
    ):
        state = mark_phase_complete(
            tmp_path,
            state,
            phase_id=phase_id,
            evidence=evidence,
        )

    finished = finish_session_state(tmp_path, state)
    progress = phase_progress(finished)
    assert progress["done"] is True
    assert progress["status"] == "completed"
    assert finished.phases[-1].proof is not None
    assert finished.phases[-1].proof.source == "session_cleanup"


def test_offline_checkpoint_requires_the_verified_structured_event(tmp_path: Path) -> None:
    state = _state(tmp_path, "Switch offline using AUA's reversible command")

    with pytest.raises(ValueError, match="verified network_offline result"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="The device is offline",
        )
    unchanged = complete_environment_phase(
        tmp_path,
        state,
        command="network_offline",
        result={"ok": True, "verified": False, "state": {"offline": True}},
    )
    assert phase_progress(unchanged)["done"] is False

    completed = complete_environment_phase(
        tmp_path,
        state,
        command="network_offline",
        result={"ok": True, "verified": True, "state": {"offline": True}},
    )
    assert phase_progress(completed)["done"] is True


def test_finish_preserves_a_genuinely_unfinished_ui_checkpoint(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        "Inspect the catalog; then verify item details; finally restore network",
    )
    state = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence="Catalog items were inspected",
    )

    finished = finish_session_state(tmp_path, state)
    progress = phase_progress(finished)

    assert progress["status"] == "terminated_incomplete"
    assert progress["done"] is False
    assert progress["current"]["objective"] == "verify item details"
    assert finished.phases[-1].status == "completed"
