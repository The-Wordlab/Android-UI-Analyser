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
    load_session_state,
    mark_phase_complete,
    phase_progress,
    update_phase_recommendation,
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
    assert phase.requirements == []


@pytest.mark.parametrize(
    "goal",
    [
        (
            "Record the verified active network transport; make the Example Emulator "
            "verifiably offline; restore the original connectivity on finish."
        ),
        (
            "Capture the current connection transport; take the Example Device provably "
            "offline; reconnect its original network when the session ends."
        ),
        (
            "Observe current connectivity state; enter airplane mode using AUA; return "
            "networking to its starting state as part of session completion."
        ),
    ],
)
def test_network_only_paraphrases_compile_to_three_typed_phases(goal: str) -> None:
    phases = goal_phases(goal)

    assert [phase.kind for phase in phases] == ["environment", "environment", "cleanup"]
    assert [phase.intent for phase in phases] == [
        "network_observation",
        "offline_transition",
        "cleanup_finalizer",
    ]
    assert [phase.satisfaction for phase in phases] == [
        "verified_network_status",
        "verified_offline",
        "session_cleanup",
    ]
    assert len({phase.source_span for phase in phases}) == 3
    assert not any(phase.intent == "ui_verification" for phase in phases)
    assert not any(phase.objective.casefold() == "on finish" for phase in phases)


def test_compound_offline_clause_preserves_independent_ui_work() -> None:
    phases = goal_phases(
        "Record the active network transport; go offline and verify cached Example Catalog "
        "content is readable; restore network on finish."
    )

    assert [phase.intent for phase in phases] == [
        "network_observation",
        "offline_transition",
        "ui_verification",
        "cleanup_finalizer",
    ]
    assert phases[2].objective == "verify cached Example Catalog content is readable"
    assert phases[1].source_span == phases[2].source_span


@pytest.mark.parametrize(
    "goal",
    [
        "Return to Network & internet settings and inspect the Example toggle",
        "Record a video showing network settings and inspect the Example toggle",
    ],
)
def test_network_words_in_ui_instructions_do_not_become_environment_phases(goal: str) -> None:
    phases = goal_phases(goal)

    assert [phase.intent for phase in phases] == ["ui_verification"]


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
    assert phases[4].constraints == ["Never use a deep link to bypass either recents-item tap"]
    assert not any(
        phrase in phase.objective.casefold()
        for phase in phases
        for phrase in (
            "verified reversible command",
            "never use a deep link",
            "finish session cleanup",
        )
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


@pytest.mark.parametrize(
    "evidence",
    [
        "Saved conversations and seed fixtures were checked",
        "Saved conversations were checked twice",
        "Saved conversations were not present but were reused",
        "Seed fixtures were not missing but were created",
    ],
)
def test_alternative_requires_one_substantiated_present_or_missing_branch(
    tmp_path: Path,
    evidence: str,
) -> None:
    state = _state(
        tmp_path,
        "Reuse saved conversations if present; create seed fixtures only if missing",
    )

    with pytest.raises(ValueError, match="one exact alternative branch"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence=evidence,
        )

    assert phase_progress(state)["done"] is False


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


def test_verified_network_status_and_offline_events_complete_network_only_goal(
    tmp_path: Path,
) -> None:
    state = _state(
        tmp_path,
        "Record the verified active network transport; make the Example Emulator verifiably "
        "offline; restore the original connectivity on finish.",
    )

    with pytest.raises(ValueError, match="verified network_status result"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="Active network transport is Wi-Fi",
        )
    unchanged = complete_environment_phase(
        tmp_path,
        state,
        command="network_status",
        result={
            "ok": True,
            "verified": False,
            "state": {"active_network": True, "active_transports": ["wifi"]},
        },
    )
    assert phase_progress(unchanged)["completed"] == 0

    state = complete_environment_phase(
        tmp_path,
        state,
        command="network_status",
        result={
            "ok": True,
            "verified": True,
            "state": {
                "active_network": True,
                "active_transports": ["wifi"],
                "internet_validated": True,
                "offline": False,
            },
        },
    )
    assert state.phases[0].proof is not None
    assert state.phases[0].proof.command == "network_status"
    assert "active_transports=wifi" in (state.phases[0].evidence or "")
    assert phase_progress(state)["current"]["intent"] == "offline_transition"

    state = complete_environment_phase(
        tmp_path,
        state,
        command="network_offline",
        result={"ok": True, "verified": True, "state": {"offline": True}},
    )
    assert phase_progress(state)["current"]["intent"] == "cleanup_finalizer"

    finished = finish_session_state(tmp_path, state)
    progress = phase_progress(finished)
    assert progress["completed"] == 3
    assert progress["total"] == 3
    assert progress["done"] is True
    assert progress["status"] == "completed"
    assert progress["next_call"] is None
    assert progress["blocking_phases"] == []


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


def test_manual_proof_requires_every_explicit_absence_and_control_state(
    tmp_path: Path,
) -> None:
    state = _state(
        tmp_path,
        "Verify cached Example thread content with no Loading and an enabled Reply control",
    )
    assert [(item.terms, item.expected) for item in state.phases[0].requirements] == [
        (["loading"], "absent"),
        (["control", "reply"], "enabled"),
        (["cache", "content", "example", "thread"], "present"),
    ]

    with pytest.raises(ValueError, match="loading=absent"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="Example thread content and Reply control are visible",
        )
    with pytest.raises(ValueError, match="control/reply=enabled"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="Example thread content is visible and Loading is absent",
        )

    completed = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence=(
            "Example thread content is visible, Loading is absent, and Reply control is enabled"
        ),
    )
    assert phase_progress(completed)["done"] is True


def test_manual_proof_rejects_an_explicitly_negated_required_presence(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, "Verify the Example summary panel is visible")

    with pytest.raises(ValueError, match="example/panel/summary=present"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence="Example summary panel is not visible",
        )

    completed = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence="Example summary panel is visible",
    )
    assert phase_progress(completed)["done"] is True


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("Reply control", "enabled"),
        ("Reply control", "disabled"),
        ("Example checkbox", "checked"),
        ("Example checkbox", "unchecked"),
        ("Example option", "selected"),
        ("Example option", "unselected"),
    ],
)
def test_manual_proof_rejects_explicit_negation_of_every_named_state(
    tmp_path: Path,
    subject: str,
    expected: str,
) -> None:
    state = _state(tmp_path, f"Verify the {subject} is {expected}")

    with pytest.raises(ValueError, match=f"={expected}"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence=f"The {subject} is not {expected}",
        )


@pytest.mark.parametrize(
    ("goal", "wrong_evidence", "right_evidence", "expected"),
    [
        (
            "Verify the Example Reply control is disabled",
            "Example Reply control is enabled",
            "Example Reply control is disabled",
            "disabled",
        ),
        (
            "Verify the Example checkbox is checked",
            "Example checkbox is visible",
            "Example checkbox is checked",
            "checked",
        ),
        (
            "Verify the Example option is selected",
            "Example option is visible",
            "Example option is selected",
            "selected",
        ),
    ],
)
def test_manual_proof_preserves_named_state_polarity(
    tmp_path: Path,
    goal: str,
    wrong_evidence: str,
    right_evidence: str,
    expected: str,
) -> None:
    state = _state(tmp_path, goal)

    assert state.phases[0].requirements[0].expected == expected
    with pytest.raises(ValueError, match="matching polarity"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="phase_1",
            evidence=wrong_evidence,
        )
    completed = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence=right_evidence,
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
    state = update_phase_recommendation(
        tmp_path,
        state,
        phase_id="phase_2",
        call={"kind": "analyze", "cli": "aua analyze"},
    )

    finished = finish_session_state(tmp_path, state)
    progress = phase_progress(finished)

    assert progress["status"] == "terminated_incomplete"
    assert progress["done"] is False
    assert progress["current"]["objective"] == "verify item details"
    assert "recommended_call" not in progress["current"]
    assert progress["next_call"] is None
    assert progress["blocking_phases"] == [
        {
            "id": "phase_2",
            "objective": "verify item details",
            "required_evidence": ("phase-specific observable evidence including detail=present"),
        }
    ]
    assert finished.phases[-1].status == "completed"
    unchanged = update_phase_recommendation(
        tmp_path,
        finished,
        phase_id="phase_2",
        call={"kind": "tap", "cli": "aua tap 1"},
    )
    assert unchanged == finished
    assert unchanged.phases[1].recommended_call is None


def test_manual_phase_completion_refuses_a_finished_session(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        "Inspect the Example catalog; then verify Example item details; finally restore network",
    )
    state = mark_phase_complete(
        tmp_path,
        state,
        phase_id="phase_1",
        evidence="The Example catalog was inspected",
    )
    finished = finish_session_state(tmp_path, state)

    with pytest.raises(ValueError, match="session has finished"):
        mark_phase_complete(
            tmp_path,
            finished,
            phase_id="phase_2",
            evidence="Example item details are visible",
        )

    persisted = load_session_state(tmp_path, session_id=state.session_id)
    assert persisted == finished
    assert persisted.phases[1].status == "active"
