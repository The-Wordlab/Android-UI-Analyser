"""Which authored waypoint a bounded local navigator is allowed to steer toward.

``Engine.session_autopilot`` lets a small local model take the next few navigation steps by
itself. The model only ever picks one opaque candidate ID; *which destination it is asked
about* is decided here, deterministically, from the session's own compiled phases.

That decision used to be made inline by folding every remaining verify phase into one flat
list of waypoints. Observed live, that produced a run whose session state said phase 1 was
active while autopilot steered toward a waypoint authored in phase 3 — a screen the goal
never asked for — and reported ``skipped_waypoints: []`` while doing it. Three rules keep
that from happening again:

1. Only one phase supplies objectives per step, and it is the phase the run is actually on.
2. A phase may be crossed only when every waypoint it authored is already reached, and the
   crossing is named in :attr:`WaypointPlan.crossed_phases`.
3. A phase that authors no navigation at all — a proof-only checkpoint, an offline
   transition, a cleanup — is never crossed. Autopilot has nothing to steer and says so once
   instead of choosing a destination for the user.

The module is deliberately dependency-free and works on any phase object exposing ``id``,
``objective``, ``kind`` and ``status``, so the planner is testable without a device, an
engine, or a model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["PhaseLike", "WaypointPlan", "plan_waypoints"]

# Autopilot only knows how to reach a destination by tapping. Environment phases (an offline
# transition) and cleanup phases are satisfied by deterministic engine work, never by a tap.
NAVIGABLE_PHASE_KIND = "verify"


class PhaseLike(Protocol):
    """The compiled goal-phase surface the planner reads."""

    id: str
    objective: str
    kind: str
    status: str


@dataclass(frozen=True)
class WaypointPlan:
    """One step's worth of steering, or an explicit reason there is none."""

    phase_id: str | None = None
    objectives: tuple[str, ...] = ()
    #: Waypoints proven reached while planning; the caller records them as completed.
    arrived_waypoints: tuple[str, ...] = ()
    #: Phases whose waypoints are all reached but whose proof is still the parent's job.
    crossed_phases: tuple[str, ...] = ()
    blocked_reason: str | None = None
    blocked_detail: str = ""

    @property
    def can_steer(self) -> bool:
        return bool(self.objectives)


def _folded(values: Iterable[str]) -> set[str]:
    return {" ".join(str(value).casefold().split()) for value in values}


def plan_waypoints(
    phases: Sequence[Any],
    *,
    active_phase_id: str,
    completed: Sequence[str] = (),
    skipped: Sequence[str] = (),
    waypoints_of: Callable[[str], Sequence[str]],
    arrived: Callable[[str], bool],
) -> WaypointPlan:
    """Return the objectives autopilot may pursue right now.

    *completed* are waypoints already reached in this run, *skipped* are waypoints already
    passed over because nothing on screen matched them — neither is offered again, and the
    two are kept apart because filing a skip as a completion is how a run reports navigation
    it never performed.
    """

    done = _folded(completed)
    passed = _folded(skipped)
    remaining_phases = [phase for phase in phases if str(getattr(phase, "status", "")) != "completed"]
    index = next(
        (
            position
            for position, phase in enumerate(remaining_phases)
            if str(getattr(phase, "id", "")) == active_phase_id
        ),
        None,
    )
    if index is None:
        return WaypointPlan(
            blocked_reason="phase_not_navigable",
            blocked_detail=(
                f"The active phase {active_phase_id!r} is not among the incomplete phases, so "
                "no authored waypoint can be attributed to where the run actually is."
            ),
        )

    arrived_waypoints: list[str] = []
    crossed_phases: list[str] = []
    for phase in remaining_phases[index:]:
        phase_id = str(getattr(phase, "id", ""))
        objective = str(getattr(phase, "objective", ""))
        kind = str(getattr(phase, "kind", ""))
        if kind != NAVIGABLE_PHASE_KIND:
            return WaypointPlan(
                arrived_waypoints=tuple(arrived_waypoints),
                crossed_phases=tuple(crossed_phases),
                blocked_reason="phase_not_navigable",
                blocked_detail=(
                    f"Phase {phase_id} ({kind}) is not reached by tapping: {objective!r}. "
                    "Autopilot will not navigate past it."
                ),
            )
        authored = [str(item) for item in waypoints_of(objective)]
        if not authored:
            return WaypointPlan(
                arrived_waypoints=tuple(arrived_waypoints),
                crossed_phases=tuple(crossed_phases),
                blocked_reason="phase_not_navigable",
                blocked_detail=(
                    f"Phase {phase_id} authors no navigation waypoint: {objective!r}. "
                    "Autopilot has no destination for this checkpoint and will not choose one."
                ),
            )
        pending = [
            waypoint
            for waypoint in authored
            if " ".join(waypoint.casefold().split()) not in done | passed
        ]
        # A prior action may already have landed on the next authored waypoint. Only a
        # prefix can be consumed this way: a later waypoint normally lives behind an
        # earlier one, so a match further down is not evidence of arrival.
        while pending and arrived(pending[0]):
            reached = pending.pop(0)
            arrived_waypoints.append(reached)
            done.add(" ".join(reached.casefold().split()))
        if pending:
            return WaypointPlan(
                phase_id=phase_id,
                objectives=tuple(pending),
                arrived_waypoints=tuple(arrived_waypoints),
                crossed_phases=tuple(crossed_phases),
            )
        # Every authored waypoint of this phase is reached; its proof checkpoint is the
        # parent agent's job, so the next phase may run — as a recorded crossing.
        crossed_phases.append(phase_id)

    return WaypointPlan(
        arrived_waypoints=tuple(arrived_waypoints),
        crossed_phases=tuple(crossed_phases),
        blocked_reason="navigation_complete",
        blocked_detail=(
            "Every authored safe-tap waypoint is reached; the parent agent must continue with "
            "input, mutation, waiting, or proof."
        ),
    )
