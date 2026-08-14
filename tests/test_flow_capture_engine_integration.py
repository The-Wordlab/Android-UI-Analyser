"""Action-bound arrival proof survives into a trustworthy flow preview without another guess."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine, _AwaitTerm
from android_ui_analyser.flows import FlowStore
from android_ui_analyser.memory import AppMemoryStore, RouteStep, SessionState
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from android_ui_analyser.session import plan_goal_session
from conftest import FakeDevice, make_config

PACKAGE = "com.example.catalog"


def _observation(serial: str) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=4,
            tier_used="hierarchy",
            path="hierarchy",
            fingerprint="destination-fingerprint",
            device_serial=serial,
        ),
    )


def test_satisfied_action_until_becomes_unmapped_flow_arrival_proof(
    monkeypatch: Any, tmp_path: Path
) -> None:
    serial = "flow-proof-engine"
    config = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"dir": str(tmp_path / "memory")},
    )
    engine = Engine(config, device=FakeDevice(serial=serial, package=PACKAGE))
    store = AppMemoryStore(config.memory)
    store.save_session(
        serial,
        SessionState(
            package=PACKAGE,
            active_context_id="default",
            capture_segment=3,
            next_capture_order=2,
            recent=[
                RouteStep(
                    kind="tap",
                    resource_id="openDetails",
                    by="id",
                    package=PACKAGE,
                    origin_package=PACKAGE,
                    context_id="default",
                    capture_segment=3,
                    capture_order=1,
                )
            ],
        ),
    )
    observed = _observation(serial)

    def fold(result: Any, *_args: Any, **_kwargs: Any) -> Any:
        result.observation = observed
        result.observation_present = True
        return result

    monkeypatch.setattr(engine, "_observe", fold)
    monkeypatch.setattr(engine, "_join_memory_writers", lambda **_kwargs: True)
    settled = engine._await_result(
        "satisfied",
        [{"term": "rid:detailPanel", "present": True, "satisfied": True}],
        time.monotonic(),
        1,
        (PACKAGE, ".MainActivity"),
        (PACKAGE, ".MainActivity"),
        True,
        True,
        capture_terms=[_AwaitTerm("rid:detailPanel", "rid", "detailPanel", False)],
    )
    assert settled.ok is True
    assert store.load_session(serial).recent[-1].arrival_proof is not None

    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)
    preview = engine.flow_save("open_details", last=1)

    assert preview["arrival_proof"] == {
        "status": "verified",
        "screen": None,
        "reason": "fresh snapshot matches satisfied action-bound arrival proof",
        "predicate": "rid:detailPanel",
        "source": "satisfied_action_until",
        "fingerprint": "destination-fingerprint",
    }
    assert "arrival: rid:detailPanel" in preview["preview"]
    assert "arrival_status: predicate_verified" in preview["preview"]
    assert preview["arrival_status"] == "predicate_verified"
    assert preview["selector_resilience"][0]["selector"] == "rid"
    assert preview["selector_resilience"][0]["cross_frame"] is True

    saved = engine.flow_save("open_details", last=1, save=True)
    assert saved["arrival_status"] == "predicate_verified"

    store = FlowStore(config.memory)
    loaded = store.load("open_details")
    assert loaded.arrival == "rid:detailPanel"
    assert loaded.arrival_status == "predicate_verified"
    assert store.list()[0]["arrival_status"] == "predicate_verified"

    dry_run = engine.flow_run("open_details", dry_run=True)
    assert dry_run["arrival"] == "rid:detailPanel"
    assert dry_run["arrival_status"] == "predicate_verified"

    plan = plan_goal_session("open details", observed, flows=[loaded])
    candidate = next(item for item in plan.candidates if item.id == "flow:open_details")
    assert candidate.evidence["arrival_status"] == "predicate_verified"
    assert candidate.safe is True
    assert candidate.call.executes is True
