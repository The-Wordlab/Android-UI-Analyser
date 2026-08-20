"""Turn what the on-device recorder announced into steps, without overstating it.

The device's recorder listens to accessibility events, which makes it a better source than
polling snapshots and guessing what changed between them — the framework names the node at
the moment it is acted on, so there is no sampling gap and no inference. What it is *not* is
complete: a view only emits a click event if it calls ``performClick``, and plenty do not.

That incompleteness is the entire design problem here, because silence is ambiguous. "No
event" means either "the user did nothing" or "the user did something I cannot see", and a
draft flow that quietly omits a step is indistinguishable from one that is finished. So the
device reports the *shadow* a missed action leaves — the screen changed with nothing announced
before it — and this module's job is to carry that admission all the way to the reader rather
than dropping it on the floor. A draft that says "I went blind between step 4 and step 5" is
useful. A draft that silently skips step 5 is worse than no draft at all.

Deliberately platform-neutral: it takes plain dictionaries and returns
:class:`~android_ui_analyser.memory.RouteStep` objects, so it is reachable from any adapter
and testable without a device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .flows import recorded_step_blockers, steps_from_recent
from .memory import RouteStep

# Kinds the device is able to announce. Anything else is from a newer helper than this host,
# and is skipped rather than guessed at.
_ACTION_KINDS = frozenset({"tap", "long-press", "input", "scroll"})

# The device's marker for "the screen moved and nothing told me why".
_GAP_KIND = "gap"


@dataclass(frozen=True)
class RecordingGap:
    """A point where the recording is known to be missing something."""

    after_step: int
    """How many steps had been recorded when the screen changed unannounced.

    Zero would mean "before anything", which is why a leading gap is dropped rather than
    reported — see :func:`steps_from_recording`.
    """

    reason: str
    package: str | None = None


@dataclass(frozen=True)
class UnnamedControl:
    """A control that was pressed but has nothing to identify it by.

    Not a shortcoming of the recording — the node was found, it simply has no text, no content
    description and no resource id, so no selector can name it and the step falls back to
    coordinates. On one real screen four of the eight pressable controls were like this, the
    back arrow among them. That is worth saying out loud: it is why the step is brittle, it is
    what to fix to make it durable, and it is an accessibility defect in its own right.
    """

    step: int
    """1-based index of the step in the draft."""

    x: int
    y: int
    bounds: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class RecordingDraft:
    """Steps the device saw, plus an honest account of what it did not."""

    steps: list[RouteStep] = field(default_factory=list)
    gaps: list[RecordingGap] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
    recovered: int = 0
    """How many steps came from the raw touch stream rather than an announced event."""

    unnamed_controls: list[UnnamedControl] = field(default_factory=list)
    app_initiated_changes: int = 0
    """Screen changes with no press behind them — the app moving on its own, not a miss."""

    @property
    def complete(self) -> bool:
        """Can this be replayed as-is, or does a human have to fill something in first?

        Both halves matter and they fail differently. A *gap* means a step is missing
        outright. A *blocker* means a step is present but was captured without enough detail
        to replay it — a scroll whose container and extent were never recorded, say. Either
        way the answer to "can I just run this" is no, so they share one flag.
        """

        return not self.gaps and not self.blockers


def _identity(row: dict[str, Any]) -> tuple[str, ...] | None:
    """What makes two input events the same field, or None when nothing does.

    Read off the *raw* row, before redaction, because redaction is what erases the
    distinguishing detail. Returning None for an unidentifiable field is the important case:
    two adjacent unlabelled text boxes would otherwise share the key "nothing at all",
    collapse into a single step, and have the draft type one value into the wrong place.
    """

    for key in ("resource_id", "content_desc", "label"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return (key, value)
    return None


# A press and the event it did (or did not) produce are not simultaneous. Wide enough to
# cover a slow frame, narrow enough that two deliberate taps are never confused for one.
_SAME_PRESS_MS = 900

# How long after a press its consequence may still arrive. A gap later than this belongs to
# something else and must keep being reported.
_EXPLAINS_GAP_MS = 2500

# How long after an *unattributed* press a screen change may still be that press's doing.
# Wider than the window above, because this only decides whether a human was involved at all,
# not which step to attribute it to.
_UNATTRIBUTED_PRESS_MS = 6000


def _node_at(snapshot: dict[str, Any], x: int, y: int) -> dict[str, Any] | None:
    """The smallest pressable node containing the point, or None.

    Smallest wins because rows nest inside containers and the container is almost never what
    the finger meant. Ties go to the first seen, which is document order.
    """

    best: dict[str, Any] | None = None
    best_area: float = float("inf")
    for node in snapshot.get("nodes") or []:
        bounds = node.get("bounds")
        if not isinstance(bounds, list | tuple) or len(bounds) != 4:
            continue
        left, top, right, bottom = bounds
        if not (left <= x <= right and top <= y <= bottom):
            continue
        area = max(0, right - left) * max(0, bottom - top)
        if area < best_area:
            best, best_area = node, area
    return best


def _snapshot_before(snapshots: Sequence[dict[str, Any]], when_ms: int) -> dict[str, Any] | None:
    """The newest snapshot taken at or before *when_ms*.

    Before, not nearest: a tap moves the screen, so the snapshot that follows it describes
    where the journey went rather than what was pressed to get there.
    """

    best: dict[str, Any] | None = None
    for snapshot in snapshots:
        ts = snapshot.get("ts")
        if not isinstance(ts, int | float):
            continue
        if ts <= when_ms and (best is None or ts > best["ts"]):
            best = snapshot
    return best


def _recover_touch(
    touch: Any, snapshots: Sequence[dict[str, Any]]
) -> tuple[RouteStep, dict[str, Any] | None]:
    """Build a step for a press nothing announced.

    Second value is the pressable node that could not be named, when there was one. That
    distinguishes "you pressed a control nobody can address" — worth reporting, and worth
    fixing in the app — from "you pressed the padding between two rows", which is not.

    Falling back to coordinates rather than inventing a name matters — a tap can legitimately
    land on padding between rows, and a step naming the nearest control would replay as a
    press on something the person never touched.
    """

    snapshot = _snapshot_before(snapshots, touch.down_ms)
    node = _node_at(snapshot, touch.x, touch.y) if snapshot else None
    if node is None:
        # Nothing pressable was under the finger. Not a control anyone can go and label, so
        # it is not reported as one.
        return RouteStep(kind="tap-point", arg=f"{touch.x},{touch.y}"), None
    fields: dict[str, Any] = {"kind": "tap"}
    for key in ("resource_id", "label", "content_desc"):
        value = node.get(key)
        if isinstance(value, str) and value:
            fields[key] = value
    if len(fields) == 1:  # a pressable node with nothing to name it by
        return RouteStep(kind="tap-point", arg=f"{touch.x},{touch.y}"), node
    return RouteStep(**fields), None


def steps_from_recording(
    rows: Sequence[Any],
    *,
    touches: Sequence[Any] | None = None,
    snapshots: Sequence[dict[str, Any]] | None = None,
    touch_capture: bool = False,
) -> RecordingDraft:
    """Build a draft from the device's recorded rows, and from what it could not announce.

    Typed values never survive: the device does not capture them, and every ``input`` becomes
    a named parameter for the person editing the draft to fill in. ``content_desc`` is dropped
    from an input step for the same reason — a widget that mirrors what was typed into its own
    description would leak past the suppression that exists to stop exactly that.

    *touches* and *snapshots* are the second source. Accessibility only reports a press when
    the view announces one; the kernel touch stream reports every press there was. A press with
    no announcement is looked up against the pressable nodes captured just before it, which
    turns a coordinate into a selector — and closes the gap that press would otherwise leave.
    """

    announced: list[tuple[int, RouteStep]] = []
    gaps_raw: list[tuple[int, int, str, str | None]] = []
    last_input_identity: tuple[str, ...] | None = None
    seen_input = False
    order = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if not isinstance(kind, str):
            continue
        when = row.get("ts")
        when_ms = int(when) if isinstance(when, int | float) else order
        order += 1

        if kind == _GAP_KIND:
            gaps_raw.append(
                (when_ms, len(announced), str(row.get("reason") or "unknown"), row.get("package"))
            )
            continue

        if kind not in _ACTION_KINDS:
            continue

        if kind == "input":
            identity = _identity(row)
            # Every keystroke fires its own event, so consecutive typing into one field is one
            # step. Only collapse when the field can actually be identified.
            if seen_input and identity is not None and identity == last_input_identity:
                continue
            last_input_identity = identity
            seen_input = True
        else:
            last_input_identity = None
            seen_input = False

        fields: dict[str, Any] = {"kind": kind}
        for key in ("resource_id", "label", "content_desc", "arg"):
            value = row.get(key)
            if isinstance(value, str) and value:
                fields[key] = value
        if kind == "input":
            fields.pop("content_desc", None)
            fields.pop("label", None)
        announced.append((when_ms, RouteStep(**fields)))

    # -- fold in the presses nothing announced -----------------------------
    recovered: list[tuple[int, RouteStep]] = []
    explained_ms: list[int] = []
    unnameable: list[tuple[int, Any, dict[str, Any]]] = []
    # Presses that did NOT become a step. Only these can leave a hole: once a press has been
    # turned into a step it is, by definition, not missing from the recording.
    unattributed_ms: list[int] = []
    for touch in touches or []:
        if not getattr(touch, "is_tap", False):
            # A drag is a finger too, and replaying it as a tap presses whatever it started
            # over. But a press read as a drag is exactly how a real action could still go
            # missing, so it is remembered as something that might explain a hole.
            unattributed_ms.append(touch.down_ms)
            continue
        if any(abs(when - touch.down_ms) <= _SAME_PRESS_MS for when, _ in announced):
            continue   # both sources saw the same press
        step, blank = _recover_touch(touch, snapshots or [])
        recovered.append((touch.down_ms, step))
        explained_ms.append(touch.down_ms)
        if blank is not None:
            unnameable.append((touch.down_ms, touch, blank))

    merged = sorted(announced + recovered, key=lambda pair: pair[0])
    steps = [step for _, step in merged]

    # A hole with a known cause is not a hole. Anything still unexplained keeps being said.
    gaps: list[RecordingGap] = []
    app_initiated = 0
    for when_ms, _position, reason, package in gaps_raw:
        if any(0 <= when_ms - press <= _EXPLAINS_GAP_MS for press in explained_ms):
            continue
        if touch_capture and not any(
            0 <= when_ms - press <= _UNATTRIBUTED_PRESS_MS for press in unattributed_ms
        ):
            # Every press became a step, so nothing was missed — this is the app moving on its
            # own, a toast landing or a request returning. That is evidence, not absence of
            # it, and reporting it sends a reader hunting for a tap that never happened. One
            # showed up on the first real journey, a couple of seconds after a copy.
            #
            # Deliberately keyed on unattributed presses rather than elapsed time: a hole only
            # means "an action went missing" if there is an action left over to have gone
            # missing. (A hardware key is the one thing this cannot see — it is not a touch.)
            app_initiated += 1
            continue
        # Position comes from where the gap sat among the ANNOUNCED rows, plus any press
        # recovered before it. Deriving it purely from timestamps instead moved a gap past
        # steps that merely shared its millisecond — and the whole value of a gap is that it
        # says *where* the recording went blind.
        after = _position + sum(1 for stamp, _ in recovered if stamp <= when_ms)
        # One navigation announces several window changes. The device debounces them, but
        # this must not rely on that alone: the two versions travel separately, an older
        # helper does not debounce at all, and a hole count that grows with animation
        # timing sends a reader hunting for taps that were never missed.
        if gaps and gaps[-1].after_step == after:
            continue
        gaps.append(RecordingGap(after_step=after, reason=reason, package=package or None))

    # One implementation of "a recorded step must never carry a typed value": the same
    # materializer `flow save` uses, so a recorded draft and a saved capture agree.
    materialized, params = steps_from_recent(steps)
    # Position the findings against the steps the reader will actually see.
    positions = {id(step): index for index, (_, step) in enumerate(merged, start=1)}
    findings = [
        UnnamedControl(
            step=positions.get(id(step), 0),
            x=touch.x,
            y=touch.y,
            bounds=tuple(node["bounds"]) if isinstance(node.get("bounds"), list) else None,
        )
        for (when, touch, node) in unnameable
        for step in [next((s for w, s in recovered if w == when), None)]
        if step is not None
    ]
    return RecordingDraft(
        steps=materialized,
        gaps=gaps,
        blockers=recorded_step_blockers(materialized),
        params=params,
        recovered=len(recovered),
        unnamed_controls=findings,
        app_initiated_changes=app_initiated,
    )
