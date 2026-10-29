"""Pure call-accounting contract for session reviews."""

from __future__ import annotations

from android_ui_analyser.session import SessionState, review_session_events


def test_review_partitions_top_level_calls_and_discloses_reporting_scope() -> None:
    state = SessionState(
        session_id="accounting-contract",
        goal="Open saved items",
        goal_hash="goal-hash",
        serial="contract-emulator",
        started_ms=1,
        recommended_kind="goto",
        recommended_cli="aua goto 'saved items'",
    )
    common = {"session_id": state.session_id, "source": "mcp", "ok": True}
    events = [
        {**common, "cmd": "session_start", "args": {}, "result": {"ok": True}},
        {
            **common,
            "cmd": "tap",
            "args": {"selector": {"rid": "saved_items"}},
            "result": {"ok": True, "observation": {"elements": []}},
        },
        {
            **common,
            "cmd": "await_predicate",
            "args": {"predicate": "rid:saved_items", "adopt_action": True},
            "result": {"ok": True},
        },
    ]

    accounting = review_session_events(state, events)["accounting"]

    assert accounting["journal_events"] == 3
    assert accounting["top_level_calls"] == 2
    assert accounting["folded_internal_events"] == 1
    assert accounting["lifecycle_calls"] == 1
    assert accounting["task_calls"] == 1
    assert accounting["lifecycle_calls"] + accounting["task_calls"] == accounting["top_level_calls"]
    assert accounting["reporting_call_included"] is False
    assert accounting["top_level_calls_including_reporting_call"] == 3


def test_journey_lifecycle_keeps_checks_and_reviews_as_tasks_and_preserves_failures() -> None:
    state = SessionState(
        session_id="journey-accounting", goal="Inspect the fixture screen",
        goal_hash="fixture-goal", serial="fixture-target", started_ms=1,
        recommended_kind="manual", recommended_cli="aua journey check",
    )
    events = [
        {"session_id": state.session_id, "source": "cli", "cmd": command,
         "ok": ok, "args": {}, "result": {"ok": ok}}
        for command, ok in [
            ("journey_start", True),
            ("journey_check", False),
            ("journey_record", False),
            ("journey_status", True),
            ("journey_finish", False),
            ("journey_finish", True),
        ]
    ]

    review = review_session_events(state, events)
    accounting = review["accounting"]

    assert accounting["top_level_calls"] == 6
    assert accounting["lifecycle_calls"] == 4
    assert accounting["task_calls"] == 2
    assert accounting["reporting_call_included"] is False
    assert accounting["top_level_calls_including_reporting_call"] == 7
    assert accounting["unexpected_failures"] == review["failures"] == 3
    assert accounting["expected_error_probes"] == 0
    assert review["run_ok"] is False
    assert review["high_level_navigation_ratio"] == round(1 / 6, 3)
    assert review["commands"]["journey_start"] == 1
    assert review["commands"]["journey_finish"] == 2
