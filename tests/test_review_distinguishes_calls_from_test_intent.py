"""Efficiency accounting does not turn intentional QA or engine internals into waste."""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.session import SessionState, review_session_events


def _state() -> SessionState:
    return SessionState(
        session_id="example-session",
        goal="Verify the example catalog",
        goal_hash="example-goal",
        serial="example-target",
        started_ms=0,
        recommended_kind="manual",
        recommended_cli="reuse the returned observation",
    )


def _event(command: str, index: int, **extra: Any) -> dict[str, Any]:
    return {
        "session_id": "example-session",
        "serial": "example-target",
        "owner": "example-owner",
        "invocation_id": f"call-{index}",
        "ts_ms": index * 1000,
        "cmd": command,
        "ok": True,
        "duration_ms": 20,
        "result": {
            "ok": True,
            "action": command,
            "observation": {
                "elements": [{"id": "el:next", "text": "Next"}],
                "meta": {"fingerprint": "same-frame"},
            },
        },
        **extra,
    }


def _component(name: str, index: int, **extra: Any) -> dict[str, Any]:
    return _event(
        f"app_{name}",
        index,
        invocation_id="restart-call",
        extra={
            "parent_command": "app_restart",
            "component": name,
            "component_index": index,
            "component_count": 2,
        },
        **extra,
    )


def test_explicit_restart_components_are_one_successful_caller_invocation() -> None:
    review = review_session_events(_state(), [_component("stop", 0), _component("launch", 1)])

    assert review["calls"] == 1
    assert review["run_ok"] is True
    assert review["commands"] == {"app_restart": 1}
    assert review["accounting"]["folded_internal_events"] == 1
    assert "ambiguous_invocation" not in review["patterns"]
    assert review["duration_ms"] == 40


def test_replayed_component_remains_ambiguous() -> None:
    review = review_session_events(
        _state(), [_component("stop", 0), _component("launch", 1), _component("launch", 1)]
    )

    assert review["calls"] == 1
    assert review["run_ok"] is None
    assert review["patterns"]["ambiguous_invocation"]


def test_unmarked_duplicate_invocation_is_not_assumed_to_be_a_composite() -> None:
    review = review_session_events(
        _state(),
        [
            _event("app_stop", 0, invocation_id="old-call"),
            _event("app_launch", 1, invocation_id="old-call"),
        ],
    )

    assert review["run_ok"] is None
    assert review["patterns"]["ambiguous_invocation"]


def test_composite_cannot_merge_another_owners_operation() -> None:
    review = review_session_events(
        _state(), [_component("stop", 0), _component("launch", 1, owner="another-owner")]
    )

    assert review["run_ok"] is None


def test_one_failed_component_keeps_the_whole_call_failed() -> None:
    review = review_session_events(
        _state(), [_component("stop", 0, ok=False), _component("launch", 1)]
    )

    assert review["calls"] == 1
    assert review["failures"] == 1
    assert review["run_ok"] is False


def test_manual_navigation_is_an_opportunity_without_claiming_a_flow_already_exists() -> None:
    review = review_session_events(_state(), [_event("tap", index) for index in range(5)])

    assert review["avoidable_calls"] == 0
    assert review["estimated_calls_saved_next_run"] == 0
    assert review["potential_calls_saved_next_run"] == 4
    assert review["patterns"]["manual_path"]


def test_wait_for_later_behavior_is_not_a_confirmed_avoidable_call() -> None:
    review = review_session_events(
        _state(),
        [_event("tap", 0), _event("await_predicate", 1, args={"predicate": "!text:Celebration"})],
    )

    assert review["avoidable_calls"] == 0
    assert review["potential_calls_saved_next_run"] == 1


@pytest.mark.parametrize(
    "request_details",
    [
        {"args": {"source": "hierarchy"}},
        {"args": {"fields": "id,text"}},
        {"args": {"with_ocr": False}},
        {"args": {"query": "id:el:example"}},
        {"args": {"with_image": True}},
        {"client": {"projection": {"where_text": ["Next"]}}},
    ],
)
def test_explicit_diagnostic_reads_are_not_penalized(request_details) -> None:
    review = review_session_events(
        _state(), [_event("tap", 0), _event("analyze", 1, **request_details)]
    )

    assert "redundant_analyze" not in review["patterns"]
    assert review["avoidable_calls"] == 0


def test_a_read_that_finds_a_different_frame_is_not_redundant() -> None:
    current = _event("analyze", 1)
    current["result"]["observation"]["meta"]["fingerprint"] = "later-frame"

    review = review_session_events(_state(), [_event("tap", 0), current])

    assert "redundant_analyze" not in review["patterns"]


def test_missing_frame_identity_is_only_a_possible_duplicate() -> None:
    events = [_event("tap", 0), _event("analyze", 1)]
    for event in events:
        event["result"]["observation"]["meta"].clear()

    review = review_session_events(_state(), events)

    assert review["avoidable_calls"] == 0
    assert review["potential_calls_saved_next_run"] == 1


def test_a_declared_probe_is_counted_but_not_treated_as_waste() -> None:
    probe = _event(
        "analyze",
        1,
        ok=False,
        extra={"expected_error_code": "usage", "expected_error_matched": True},
    )

    review = review_session_events(_state(), [_event("tap", 0), probe])

    assert review["calls"] == 2
    assert review["run_ok"] is True
    assert review["avoidable_calls"] == 0
    assert "redundant_analyze" not in review["patterns"]


def test_export_keeps_the_previous_view_available_for_review() -> None:
    events = [
        _event("tap", 0),
        _event("capture_sheet", 1, result={"ok": True, "path": "/example/sheet.png"}),
        _event("analyze", 2),
    ]

    review = review_session_events(_state(), events)

    assert review["avoidable_calls"] == 1
    assert review["patterns"]["redundant_analyze"][0]["after"] == "tap"
