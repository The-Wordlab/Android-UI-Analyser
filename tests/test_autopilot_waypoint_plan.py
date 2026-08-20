"""The bounded local navigator may only steer toward the phase the run is actually on.

Observed live: a session whose own state said phase 1 was active had autopilot select a
waypoint authored in phase 3, report ``skipped_waypoints: []``, and drive the app to a screen
the goal never asked for. Two mechanisms produced that, both of them silent:

* waypoints were collected from **every** remaining verify phase in source order, so a phase
  that authors no navigation verb (a proof-only checkpoint) or is not a verify phase at all
  (an offline transition) was crossed without a word;
* a waypoint passed over because nothing on screen matched it was afterwards recorded in
  ``completed_waypoints``, i.e. reported as reached.

An autopilot that cannot steer has to say so once, not pick a destination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.autopilot import plan_waypoints
from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import ActionResult, Element, Source
from conftest import make_config  # noqa: F401 - re-exported by the shared fixtures module
from test_functiongemma_engine_policy import _element, _engine, _observation, _Selector


class _Phase:
    def __init__(self, phase_id: str, objective: str, *, kind: str = "verify", status: str = "pending") -> None:
        self.id = phase_id
        self.objective = objective
        self.kind = kind
        self.status = status


def _plan(
    phases: list[_Phase],
    *,
    active: str,
    completed: tuple[str, ...] = (),
    skipped: tuple[str, ...] = (),
    reached: tuple[str, ...] = (),
) -> Any:
    return plan_waypoints(
        phases,
        active_phase_id=active,
        completed=completed,
        skipped=skipped,
        waypoints_of=Engine._policy_navigation_waypoints,  # noqa: SLF001 - the authored compiler
        arrived=lambda waypoint: waypoint in reached,
    )


def test_a_proof_only_active_phase_is_never_crossed_for_a_later_waypoint() -> None:
    plan = _plan(
        [
            _Phase("phase_1", "From Example home, confirm the greeting card", status="active"),
            _Phase("phase_2", "delete the saved record"),
            _Phase("phase_3", "open Profile and prove the header"),
        ],
        active="phase_1",
    )

    assert plan.objectives == ()
    assert plan.blocked_reason == "phase_not_navigable"
    assert "phase_1" in plan.blocked_detail
    assert plan.phase_id is None


def test_a_non_verify_active_phase_is_never_crossed_for_a_later_waypoint() -> None:
    plan = _plan(
        [
            _Phase(
                "phase_1",
                "Establish and verify the requested offline network state",
                kind="environment",
                status="active",
            ),
            _Phase("phase_2", "open Apps and prove the list"),
        ],
        active="phase_1",
    )

    assert plan.objectives == ()
    assert plan.blocked_reason == "phase_not_navigable"


def test_the_active_phase_waypoints_are_offered_in_authored_order() -> None:
    plan = _plan(
        [
            _Phase("phase_1", "open Catalog, then open Archive", status="active"),
            _Phase("phase_2", "open Profile and prove the header"),
        ],
        active="phase_1",
    )

    assert plan.phase_id == "phase_1"
    assert plan.objectives == ("Catalog", "Archive")
    assert plan.crossed_phases == ()


def test_a_fully_reached_active_phase_lets_the_next_phase_run_and_records_the_crossing() -> None:
    plan = _plan(
        [
            _Phase("phase_1", "open Catalog", status="active"),
            _Phase("phase_2", "open Archive and prove the header"),
        ],
        active="phase_1",
        reached=("Catalog",),
    )

    assert plan.arrived_waypoints == ("Catalog",)
    assert plan.phase_id == "phase_2"
    assert plan.objectives == ("Archive",)
    # The jump past an incomplete proof checkpoint is explicit, never implied.
    assert plan.crossed_phases == ("phase_1",)


def test_a_skipped_waypoint_is_not_offered_again_and_does_not_become_completed() -> None:
    plan = _plan(
        [_Phase("phase_1", "open Catalog, then open Archive", status="active")],
        active="phase_1",
        skipped=("Catalog",),
    )

    assert plan.objectives == ("Archive",)
    assert plan.arrived_waypoints == ()


def test_autopilot_refuses_when_the_active_phase_authors_no_navigation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The reported run: phase 1 active, a phase-3 waypoint selected, nothing recorded."""

    selector = _Selector()
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine.config.policy.candidate_scope = "safe_visible"
    home = _observation(
        engine.device.serial,
        [
            _element(1, "Profile", rid="com.example.catalog:id/openProfile"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
        fingerprint="frame-home",
    )

    def forbidden_tap(*_args: Any, **_kwargs: Any) -> ActionResult:
        raise AssertionError("autopilot must not act for a phase the run has not reached")

    monkeypatch.setattr(engine, "tap", forbidden_tap)
    started = engine.session_start(
        "From Example home, confirm the greeting card; then delete the saved record; "
        "then open Profile and prove the header.",
        observation=home,
    )

    before = selector.select_calls  # session_start already paid for one advisory decision
    result = engine.session_autopilot(started["session_id"], max_steps=4, observation=home)

    assert result["autopilot"]["steps_executed"] == 0
    assert result["autopilot"]["terminal_reason"] == "phase_not_navigable"
    assert result["autopilot"]["handoff_required"] is True
    assert result["autopilot"]["completed_waypoints"] == []
    # It refuses before spending a single inference, which is the whole point: the chain costs
    # real seconds per call.
    assert selector.select_calls == before


def test_autopilot_reports_a_passed_over_waypoint_as_skipped_not_completed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A waypoint nothing on screen matched was being filed as completed navigation."""

    def choose(context: Any) -> int:
        return next(
            candidate.candidate_id
            for candidate in context.candidates
            if "Archive" in candidate.purpose
        )

    selector = _Selector(choose)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    home = _observation(
        engine.device.serial,
        [
            _element(1, "Archive", rid="com.example.catalog:id/openArchive"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
        fingerprint="frame-home",
    )
    archive = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Archive",
                bounds=(20, 100, 900, 160),
                center=(460, 130),
                source=Source.hierarchy,
            )
        ],
        fingerprint="frame-archive",
    )

    def execute_tap(
        _element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        observe: bool = True,
        **_kwargs: Any,
    ) -> ActionResult:
        engine._last_analyze_result = archive  # noqa: SLF001 - fake folded action result
        return ActionResult(ok=True, action="tap", observation=archive, observation_present=True)

    monkeypatch.setattr(engine, "tap", execute_tap)
    started = engine.session_start(
        "From Example home, open Catalog, then open Archive and prove the header.",
        observation=home,
    )

    result = engine.session_autopilot(started["session_id"], max_steps=2, observation=home)

    autopilot = result["autopilot"]
    assert autopilot["skipped_waypoints"] == ["Catalog"]
    assert "Catalog" not in autopilot["completed_waypoints"]
    assert autopilot["trace"][0]["skipped_waypoints"] == ["Catalog"]
    assert autopilot["trace"][0]["phase"] == "phase_1"


def test_autopilot_records_the_phase_it_ran_ahead_into(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Following a later phase's waypoint is allowed, but only as a recorded crossing.

    A title-only arrival leaves the first proof checkpoint open on purpose — the parent agent
    owns that proof. Autopilot may still take the next authored tap, and the step that does it
    names both the phase the run is on and the phase the waypoint came from.
    """

    def choose(context: Any) -> int:
        target = "Settings" if "Settings" in context.goal else "Archive"
        return next(
            candidate.candidate_id
            for candidate in context.candidates
            if target in candidate.purpose
        )

    engine, _factory = _engine(tmp_path, "advisory", _Selector(choose))
    engine.config.policy.candidate_scope = "safe_visible"
    home = _observation(
        engine.device.serial,
        [
            _element(1, "Settings", rid="com.example.catalog:id/openSettings"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
        fingerprint="frame-home",
    )
    settings = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Settings",
                bounds=(20, 100, 900, 160),
                center=(460, 130),
                source=Source.hierarchy,
            ),
            _element(2, "Archive", rid="com.example.catalog:id/openArchive"),
        ],
        fingerprint="frame-settings",
    )
    archive = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Archive",
                bounds=(20, 100, 900, 160),
                center=(460, 130),
                source=Source.hierarchy,
            )
        ],
        fingerprint="frame-archive",
    )
    returned = {"openSettings": settings, "openArchive": archive}

    def execute_tap(
        _element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        observe: bool = True,
        **_kwargs: Any,
    ) -> ActionResult:
        observed = returned[str((selector or {}).get("rid"))]
        engine._last_analyze_result = observed  # noqa: SLF001 - fake folded action result
        return ActionResult(ok=True, action="tap", observation=observed, observation_present=True)

    monkeypatch.setattr(engine, "tap", execute_tap)
    started = engine.session_start(
        "From Example home, open Settings; then open Archive and prove the destination.",
        observation=home,
    )

    result = engine.session_autopilot(started["session_id"], max_steps=4, observation=home)

    trace = result["autopilot"]["trace"]
    assert [item["waypoint"] for item in trace] == ["Settings", "Archive"]
    assert trace[0]["phase"] == "phase_1"
    assert trace[0]["crossed_phases"] == []
    # The second step follows phase 2's waypoint while phase 1 still awaits its proof, and says so.
    assert trace[1]["active_phase"] == "phase_1"
    assert trace[1]["phase"] == "phase_2"
    assert trace[1]["crossed_phases"] == ["phase_1"]


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
