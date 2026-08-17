"""Build a replay candidate from one goal session's proven action interval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import UsageError
from .flows import (
    Flow,
    check_saveable,
    recorded_step_blockers,
    render_flow_yaml,
    steps_from_recent,
)
from .memory import RouteStep


@dataclass(frozen=True)
class CandidateFlow:
    flow: Flow
    yaml: str
    source_steps: int
    checkpoint_ids: tuple[str, ...]


def build_candidate_flow(
    *,
    name: str,
    app: str,
    context_id: str | None,
    recent: Sequence[RouteStep],
    start_capture_order: int,
    capture_segment: int,
    checkpoints: Sequence[Mapping[str, Any]],
) -> CandidateFlow:
    """Materialize recorded actions and insert proof assertions at their evidence boundary.

    ``checkpoints`` entries contain ``id``, ``capture_order`` and already validated flow
    ``assertions`` (:class:`RouteStep` instances).  Keeping assertion parsing in the authored
    contract module prevents this capture helper from inventing another assertion language.
    """

    assertions_by_order: dict[int, list[RouteStep]] = {}
    checkpoint_ids: list[str] = []
    for checkpoint in checkpoints:
        raw_order = checkpoint.get("capture_order")
        if not isinstance(raw_order, int):
            raise UsageError(
                f"checkpoint {checkpoint.get('id')!r} has no correlated action boundary",
                hint="re-observe the checkpoint after a session action before capturing a flow",
            )
        assertions = checkpoint.get("assertions")
        if not isinstance(assertions, list) or not all(
            isinstance(step, RouteStep) for step in assertions
        ):
            raise UsageError(f"checkpoint {checkpoint.get('id')!r} has invalid assertions")
        assertions_by_order.setdefault(raw_order, []).extend(assertions)
        checkpoint_ids.append(str(checkpoint.get("id") or f"checkpoint-{len(checkpoint_ids) + 1}"))

    checkpoint_orders = set(assertions_by_order)
    if checkpoint_orders and max(checkpoint_orders) < start_capture_order:
        raise UsageError(
            f"checkpoint action boundaries are outside the captured session path: "
            f"{sorted(checkpoint_orders)}",
            hint="do not save; repeat the journey inside one app/context capture segment",
        )
    end_capture_order = max(checkpoint_orders) if checkpoint_orders else None
    selected = [
        step
        for step in recent
        if step.capture_segment == capture_segment
        and step.capture_order is not None
        and step.capture_order >= start_capture_order
        and (end_capture_order is None or step.capture_order <= end_capture_order)
    ]
    if not selected:
        raise UsageError(
            "the session has no replayable actions after its capture watermark",
            hint="complete the UI journey before asking for a candidate flow",
        )
    blockers = recorded_step_blockers(selected)
    if blockers:
        raise UsageError(
            "the session path cannot be replayed losslessly",
            hint="; ".join(blockers),
        )

    materialized = [
        step.model_copy(update={"package": None}) if step.package == app else step
        for step in selected
    ]
    materialized, params = steps_from_recent(materialized)
    candidate_steps: list[RouteStep] = []
    inserted_orders: set[int] = set()
    for step in materialized:
        order = step.capture_order
        candidate_steps.append(
            step.model_copy(
                update={
                    "origin_package": None,
                    "context_id": None,
                    "capture_segment": None,
                    "capture_order": None,
                    "arrival_proof": None,
                }
            )
        )
        if isinstance(order, int) and order in assertions_by_order:
            candidate_steps.extend(assertions_by_order[order])
            inserted_orders.add(order)
    missing = sorted(set(assertions_by_order) - inserted_orders)
    if missing:
        raise UsageError(
            f"checkpoint action boundaries are outside the captured session path: {missing}",
            hint="do not save; repeat the journey inside one app/context capture segment",
        )

    flow = Flow(
        name=name,
        app=app,
        context_id=context_id,
        description=(
            f"Verified candidate captured from {len(selected)} goal-session actions and "
            f"{len(checkpoint_ids)} checkpoint(s)"
        ),
        arrival_status="unverified",
        params=params,
        steps=candidate_steps,
    )
    warnings = check_saveable(flow)
    if warnings:
        raise UsageError("candidate flow is not saveable", hint="; ".join(warnings))
    return CandidateFlow(
        flow=flow,
        yaml=render_flow_yaml(flow),
        source_steps=len(selected),
        checkpoint_ids=tuple(checkpoint_ids),
    )
