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
