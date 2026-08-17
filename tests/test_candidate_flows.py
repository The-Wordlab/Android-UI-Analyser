from __future__ import annotations

import pytest

from android_ui_analyser.candidate_flows import build_candidate_flow
from android_ui_analyser.errors import UsageError
from android_ui_analyser.memory import RouteStep


def _tap(order: int, *, segment: int = 4) -> RouteStep:
    return RouteStep(
        kind="tap",
        resource_id=f"control{order}",
        origin_package="com.example.catalog",
        context_id="default",
        capture_segment=segment,
        capture_order=order,
    )


def test_candidate_uses_watermark_and_inserts_checkpoint_assertions() -> None:
    assertion = RouteStep(
        kind="assert",
        assertion={"rid": "result", "exists": True},
    )

    candidate = build_candidate_flow(
        name="verified-example",
        app="com.example.catalog",
        context_id="default",
        recent=[_tap(2), _tap(3), _tap(4)],
        start_capture_order=3,
        capture_segment=4,
        checkpoints=[
            {
                "id": "result-visible",
                "capture_order": 4,
                "assertions": [assertion],
            }
        ],
    )

    assert candidate.source_steps == 2
    assert candidate.checkpoint_ids == ("result-visible",)
    assert [step.kind for step in candidate.flow.steps] == ["tap", "tap", "assert"]
    assert "assert:" in candidate.yaml
    assert "capture_order" not in candidate.yaml


def test_candidate_stops_at_the_last_checkpoint_boundary() -> None:
    candidate = build_candidate_flow(
        name="bounded-example",
        app="com.example.catalog",
        context_id="default",
        recent=[_tap(3), _tap(4), _tap(5)],
        start_capture_order=3,
        capture_segment=4,
        checkpoints=[
            {
                "id": "cleanup",
                "capture_order": 4,
                "assertions": [
                    RouteStep(kind="assert", resource_id="home", assertion={"exists": True})
                ],
            }
        ],
    )

    assert candidate.source_steps == 2
    assert [step.resource_id for step in candidate.flow.steps if step.kind == "tap"] == [
        "control3",
        "control4",
    ]


def test_candidate_refuses_checkpoint_outside_watermark() -> None:
    with pytest.raises(UsageError, match="outside the captured"):
        build_candidate_flow(
            name="invalid-example",
            app="com.example.catalog",
            context_id="default",
            recent=[_tap(4)],
            start_capture_order=4,
            capture_segment=4,
            checkpoints=[
                {
                    "id": "old-proof",
                    "capture_order": 2,
                    "assertions": [
                        RouteStep(kind="assert", assertion={"rid": "old", "exists": True})
                    ],
                }
            ],
        )


def test_candidate_refuses_cross_segment_actions() -> None:
    with pytest.raises(UsageError, match="no replayable actions"):
        build_candidate_flow(
            name="invalid-example",
            app="com.example.catalog",
            context_id="default",
            recent=[_tap(4, segment=3)],
            start_capture_order=4,
            capture_segment=4,
            checkpoints=[],
        )
