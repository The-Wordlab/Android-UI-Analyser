"""Saved flows and step execution: flow_run/save/list/delete, the step executor and its on-device offload, nested-flow preflight and arrival evidence, demo recording of a person's journey, and suite_run for AC checklists.

Engine methods for flows. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_flows.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import shlex
import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .assertions import evaluate_order
from .engine_support import (
    _ASSIST_MAX_STEPS,
    _HandoverRefused,
    _parse_await_terms,
    _ResolvedCassetteResource,
    _ResolvedFlagsResource,
    logger,
)
from .errors import DeviceError, StabilityTimeout, UsageError
from .memory import (
    DEFAULT_CONTEXT_ID,
    LEGACY_CONTEXT_ID,
    RouteStep,
    is_destructive_step,
    matches_any,
    resolve_goal,
    route_step_risks,
    step_display,
)
from .schema import ActionResult, AnalyzeResult
from .selectors import _match_step

if TYPE_CHECKING:
    from .engine import Engine


_MAX_FLOW_DEPTH = 5  # bound on nested `flow:` sub-flow composition (cycle backstop)


class _ResolvedFlowNode(NamedTuple):
    """One immutable nested-flow snapshot validated before device mutation."""

    flow: Any
    directory: Path | None
    source_id: str
    steps: list[RouteStep]


class _ResolvedFlowPlan(NamedTuple):
    """The complete filesystem snapshot authorized by one flow preflight."""

    flow_graph: dict[tuple[str | None, str], _ResolvedFlowNode]
    flags: dict[int, _ResolvedFlagsResource]
    cassettes: dict[int, _ResolvedCassetteResource]


_FLOW_SELF_CLEARING_STEPS = frozenset({"clear-data", "stop-app"})


def _parse_point(arg: str | None) -> tuple[int, int] | None:
    """``"x,y"`` → ``(x, y)``; None when it is not a usable pair of coordinates."""
    if not arg:
        return None
    parts = arg.replace(" ", "").split(",")
    if len(parts) != 2:
        return None
    try:
        x, y = (int(round(float(p))) for p in parts)
    except ValueError:
        return None
    return (x, y) if x >= 0 and y >= 0 else None


class StepFailure(NamedTuple):
    """Why (and where) a step sequence stopped — the executor's divergence signal."""

    code: str  # destructive_step | input_required | input_not_applied | element_not_found |
    #            unsupported_action | wait_timeout | assert_failed
    at: int  # failing step index within the executed list
    step: RouteStep
    detail: str | None = None


def _flows_for(self: Engine, package: str | None) -> list[str]:
    """Saved journeys for *package*, as ``name(PARAM, …)``.

        A flow replays a whole sequence — launch, taps, waits, cross-app auth — in one call,
        and one had been sitting saved and parameterised for this project with no agent ever
        running it. `flow` appeared 19 times in the long guide and zero times in anything an
        agent actually reads: not the orientation block, not the analyze header. That is the
        same omission that kept `goto` unused across five runs.

        Cached per package/context and directory fingerprint. Flows are deliberately editable
        YAML, so a long-lived daemon must notice save/delete/rename/manual edits without a
        restart; statting the small library is cheaper than parsing it every frame.
        """
    if not package:
        return []
    # A package with no package-matching cursor is in its default context until runtime
    # flags prove otherwise. Treating that as None hid every auto-recorded default-context
    # flow on the first frame after returning from a foreign app.
    context_id: str | None = DEFAULT_CONTEXT_ID
    if self._memory is not None and self._device is not None:
        with contextlib.suppress(Exception):
            session = self._memory.load_session(self._device.serial)
            if session.package == package:
                context_id = session.active_context_id
    from .flows import FlowStore

    store = FlowStore(self.config.memory)
    fingerprint: tuple[tuple[str, int, int], ...] = ()
    with contextlib.suppress(OSError):
        fingerprint = tuple(
            (str(path), stat.st_mtime_ns, stat.st_size)
            for path in store.files()
            if (stat := path.stat())
        )
    cache_key = f"{package}\0{context_id or ''}\0{fingerprint!r}"
    cached = self._flows_cache.get(cache_key)
    if cached is not None:
        return cached
    names: list[str] = []
    with contextlib.suppress(Exception):
        for flow in store.list(app=package):
            if flow.get("error"):
                continue
            if flow.get("context_id") not in (None, context_id):
                continue
            # A flow whose name two apps share needs the qualified spelling, and one the
            # store cannot address at all is left to `flow list` — the header must never
            # advertise a call that fails to load.
            runnable = flow.get("ref")
            if not runnable:
                continue
            params = ", ".join(flow.get("params") or [])
            names.append(f"{runnable}({params})" if params else str(runnable))
    self._flows_cache[cache_key] = names
    return names


def _source_for(self: Engine, steps: list[RouteStep], index: int, origin_package: str | None) -> str:
    """Analyze source between steps: ``auto`` when the NEXT step runs in a foreign
        (transit) package — its screen may be vision-tier — else the fast hierarchy path."""
    nxt = steps[index] if index < len(steps) else None
    if nxt is not None and nxt.package and nxt.package != origin_package:
        return "auto"
    return "hierarchy"


def _analyze_route_step(
    self: Engine,
    steps: list[RouteStep],
    index: int,
    origin_package: str | None,
    *,
    hierarchy_ocr: bool,
) -> AnalyzeResult:
    """Observe between route steps without taxing ordinary native hops with OCR.

        Foreign/transit screens keep ``source=auto`` and its normal OCR behavior. Inside
        the origin app, ``goto`` can request hierarchy-only observations and explicitly
        retry OCR only when a remembered selector is absent.
        """
    source = self._source_for(steps, index, origin_package)
    with_ocr = None if hierarchy_ocr or source == "auto" else False
    return self.analyze(source=source, with_ocr=with_ocr)


def _run_flow_assertion(self: Engine, step: RouteStep) -> ActionResult:
    """Evaluate the rich flow ``assert:`` step through the public expect primitive."""

    predicates = dict(step.assertion)
    first = bool(predicates.pop("first", False))
    count = predicates.pop("count", None)
    return self.expect(
        rid=step.resource_id,
        text=step.label,
        desc=step.content_desc,
        exists=bool(predicates.pop("exists", False)),
        absent=bool(predicates.pop("absent", False)),
        text_is=predicates.pop("text_is", None),
        text_contains=predicates.pop("text_contains", None),
        checked=predicates.pop("checked", None),
        enabled=predicates.pop("enabled", None),
        selected=predicates.pop("selected", None),
        focused=predicates.pop("focused", None),
        count=count,
        within=predicates.pop("within", None),
        same_parent_as=predicates.pop("same_parent_as", None),
        contains_all=predicates.pop("contains_all", None),
        index=step.index,
        first=first,
        timeout_ms=step.timeout_ms or 0,
        observe=False,
    )


def _run_flow_order_assertion(self: Engine, step: RouteStep) -> tuple[bool, str]:
    """Assert explicit horizontal/vertical ordering without guessing grid semantics."""

    assertion = step.assertion
    axis = assertion.get("axis")
    selectors = assertion.get("selectors")
    if axis not in {"horizontal", "vertical", "reading"} or not isinstance(selectors, list):
        return False, "invalid assert_order payload"
    timeout_ms, _clamped_from, _ceiling = self._bounded_wait_ms(step.timeout_ms or 0)
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        raw_tree = self.platform.dump_tree(self.device)
        elements = self.platform.normalize_tree(
            raw_tree,
            self.device.window_size(),
            ignored_app_ids=self.config.memory.ignore_packages,
        ).elements
        order = evaluate_order(elements, axis=axis, selectors=selectors)
        if order.ok:
            return True, order.detail
        detail = order.detail
        if time.monotonic() >= deadline:
            return False, detail
        self._sleep_between_polls(250.0, deadline)


def _device_runnable_step(self: Engine, step: RouteStep) -> bool:
    if step.kind not in self._DEVICE_STEP_KINDS:
        return False
    if getattr(step, "substeps", None):
        return False
    if step.kind == "key":
        return (step.arg or "").strip().lower() in self._DEVICE_KEY_ARGS
    if step.kind in self._DEVICE_PREDICATE_KINDS:
        # The helper matches text/desc/rid; a regex or another matcher stays on the host.
        if (step.by or "text") not in self._DEVICE_BY_FIELDS:
            return False
        return bool(step.arg)
    if step.kind in {"swipe", "scroll"}:
        return (step.arg or "").strip().lower() in self._DEVICE_DIRECTIONS
    if step.kind == "scroll-to":
        if (step.by or "text") not in self._DEVICE_BY_FIELDS:
            return False
        if (step.direction or "up").strip().lower() not in self._DEVICE_DIRECTIONS:
            return False
        return bool(step.arg)
    if step.kind == "tap-point":
        return _parse_point(step.arg) is not None
    if step.kind in {"paste", "wait-stable"}:
        return True  # no selector to resolve
    if step.kind == "input" and step.text is None:
        return False
    # An acting step with no selector at all cannot be matched on-device.
    return bool(step.resource_id or step.label or step.content_desc)


def _device_step_payload(self: Engine, step: RouteStep) -> dict[str, Any]:
    """The wire form of one step, with any wait the host would apply made explicit.

        ``model_dump`` drops an unset ``timeout_ms``, which left the device free to apply its
        own default — five seconds, against the host's none for an assertion. That gap is a
        false-pass generator: an element that turns up 400ms after the check was made passes
        on the device and would have failed on the host, and a device pass is never re-run.
        """

    row = step.model_dump(exclude_none=True)
    default = self._HOST_STEP_TIMEOUT_MS.get(step.kind)
    if default is not None:
        timeout_ms, _clamped_from, _ceiling = self._bounded_wait_ms(step.timeout_ms or default)
        row["timeout_ms"] = timeout_ms
    return row


def _device_runnable_run(
    self: Engine, steps: list[RouteStep], start: int, *, allow_destructive: bool
) -> int:
    """How many consecutive steps from *start* the device could run. Free, host-side only.

        Deciding this without touching the device is the whole point. Handing a run over
        costs a fixed handover, so the engine has to know how long the run is *before* it
        commits to paying for one — otherwise a flow the helper cannot help with pays the
        cost anyway and comes out slower.
        """

    lexicon = self.config.memory.destructive_labels
    length = 0
    for step in steps[start:]:
        if not self._device_runnable_step(step):
            break
        # The device cannot weigh a destructive label the way the host does, so a run that
        # is not explicitly allowed to be destructive simply stops before one.
        if is_destructive_step(step, lexicon) and not allow_destructive:
            break
        length += 1
    return length


def _pick_offload_start(
    self: Engine, steps: list[RouteStep], *, allow_destructive: bool, start: int = 0
) -> int | None:
    """Which index, if any, is worth handing to the device. Returns None for "none of them".

        Two different prices are on offer here and conflating them is what made the feature
        lose time. A run starting at index 0, before anything has connected, is cheap: the
        helper is already bound and the handover is 682ms. A run starting later is not: the
        host has been driving with uiautomator2, so the slot has to be taken away from it and
        then given back afterwards, and that costs several times as much. The same two-step
        run is therefore a clear win at the front of a flow and a clear loss in the middle,
        which is why there are two floors rather than one.

        Only the first run that clears its floor is chosen. Probing again at the next index
        after a refusal — which is what an earlier version did — is not free once the device
        has been contacted, and re-paying the handover per step is exactly the regression
        this method exists to prevent.

        *start* is where to begin looking, and it is what lets a flow hand over more than
        once. A refusal still ends the matter for the whole flow; a run that *worked* has
        proved the device will take the work, so the stretch after the next host-only step is
        worth the same question. Without that, a flow with a check in the middle — which is
        every real QA flow — handed over its opening steps and drove the entire remainder by
        hand, however long it was.
        """

    cfg = self.config.helper
    if not cfg.enabled:
        return None
    i = max(0, start)
    while i < len(steps):
        length = self._device_runnable_run(steps, i, allow_destructive=allow_destructive)
        if length == 0:
            i += 1
            continue
        # Only the flow's opening run gets the cheap price. A later search cannot reach
        # index 0, so this stays exactly as strict when the question is re-asked.
        cheap = i == 0 and self._device is None
        floor = max(1, cfg.min_flow_steps if cheap else cfg.min_midflow_steps)
        if length >= floor:
            return i
        i += length
    return None


def _offload_steps_to_device(
    self: Engine,
    steps: list[RouteStep],
    *,
    executed: list[dict[str, Any]] | None,
    allow_destructive: bool,
    index_offset: int = 0,
) -> int:
    """Run the leading UI-only stretch of *steps* on the device. Returns how many ran.

        Strictly an optimisation. Returning 0 — because the helper is off, absent, unbindable,
        the run is too short, or anything at all went wrong — leaves the caller's normal path
        untouched, which is why every failure here is swallowed rather than raised.

        Measured on an 8-step Settings flow: 4092ms host-driven against 606ms on-device, so
        ~436ms saved per step against the ~1.8s cost of passing the UiAutomation slot back and
        forth. ``helper.min_flow_steps`` is where those two meet.
        """

    cfg = self.config.helper
    if not cfg.enabled:
        return 0

    runnable = self._device_runnable_run(steps, 0, allow_destructive=allow_destructive)
    prefix = steps[:runnable]
    # A floor against pointless handovers only. Whether a *mid-flow* run is worth its
    # much larger handover is decided by :meth:`_pick_offload_start` before we get here.
    if len(prefix) < max(1, cfg.min_flow_steps):
        self._journal_helper(
            "skipped",
            None,
            reason="run_too_short",
            runnable=len(prefix),
            total=len(steps),
            min_flow_steps=cfg.min_flow_steps,
        )
        return 0

    began = time.perf_counter()
    try:
        with self._device_agent_borrowed(purpose="flow.run") as loan:
            serial = loan.serial
            was_connected = loan.u2_was_connected
            payload = [self._device_step_payload(step) for step in prefix]
            result = loan.channel.request(
                "flow.run",
                {"steps": payload},
                timeout=max(30.0, 5.0 * len(prefix)),
            )
    except _HandoverRefused:
        # Already journalled, at the point of refusal, by the borrow. This path is strictly an
        # optimisation, so a refusal is not an error: returning 0 leaves the caller's normal
        # host path untouched.
        return 0
    except Exception as exc:  # noqa: BLE001 - never let the shortcut break the run
        logger.debug("device flow offload unavailable (%s); running on the host", exc)
        self._journal_helper(
            "skipped", self._leased_serial(), reason="offload_failed", error=str(exc)[:160]
        )
        return 0

    completed = int(result.get("completed") or 0)
    total = int(result.get("total") or len(prefix))
    # A partial run is the interesting case and the one the host silently absorbs: it
    # simply picks up where the device stopped and the flow still passes. Without the
    # device's own reason for stopping there is nothing to tell a genuinely impossible
    # step apart from a helper that lost the screen, so carry the first failing row.
    stopped_on = next((row for row in (result.get("steps") or []) if not row.get("ok")), None)
    self._journal_helper(
        "offloaded" if completed == total else "partial",
        serial,
        failed_step=stopped_on if stopped_on else None,
        completed=completed,
        total=total,
        offered=len(prefix),
        steps_in_run=len(steps),
        starts_at=index_offset,
        ms=round((time.perf_counter() - began) * 1000, 1),
        u2_was_connected=was_connected,
        stopped_reason=result.get("stopped_reason"),
    )
    if executed is not None:
        for row, step in zip(result.get("steps") or [], prefix, strict=False):
            if not row.get("ok"):
                break
            executed.append(
                {
                    # The device numbers its own slice; the caller thinks in whole-flow
                    # positions, and a report that disagrees with the flow is worse than
                    # no report.
                    "index": (row.get("index") or 0) + index_offset,
                    "step": step_display(step),
                    "duration_ms": int(row.get("ms") or 0),
                    "ran_on": "device",
                }
            )
    return completed


def _offload_from(
    self: Engine,
    steps: list[RouteStep],
    *,
    at: int,
    executed: list[dict[str, Any]] | None,
    allow_destructive: bool,
) -> tuple[int, int | None]:
    """Hand the run beginning at *at* to the device. Returns (steps run, where to try next).

        The second half of that pair is the whole reason this is a method rather than two
        lines in the loop, because the two outcomes are not symmetrical:

        * A run that worked has proved the device will take this flow's work, so the stretch
          after the next host-only step deserves the same question.
        * A run that was refused ends offloading for the flow. Asking is only free until the
          device has been contacted, and re-probing at every subsequent gap re-pays the setup
          cost per gap — the exact regression :meth:`_pick_offload_start` was written to stop.

        Returning ``None`` for "do not ask again" rather than letting the caller decide keeps
        that asymmetry in one place, where it can be tested on its own.
        """

    ran = self._offload_steps_to_device(
        steps[at:],
        executed=executed,
        allow_destructive=allow_destructive,
        index_offset=at,
    )
    if not ran:
        return 0, None
    return ran, self._pick_offload_start(
        steps, allow_destructive=allow_destructive, start=at + ran
    )


def _recorder(self: Engine) -> tuple[Any, str]:
    """The device_agent capability and a serial, without connecting the device.

        Connecting is the one thing this path must not do. Android suppresses every
        accessibility service while uiautomator2 holds the UiAutomation slot, so touching
        ``self.device`` here would tear down the very service being asked to record and the
        journey would come back empty — with no error to explain why. ``_leased_serial``
        answers "which device" without attaching to it.
        """

    try:
        agent = self.platform.capability("device_agent")
    except Exception as exc:  # noqa: BLE001 - surfaced as a usage error below
        raise UsageError(
            "recording needs the on-device helper, which this platform does not provide",
            hint="Recording is Android-only today.",
        ) from exc
    serial = self._leased_serial()
    if serial is None:
        raise DeviceError(
            "no target device for recording",
            hint="Connect a device or pass --serial.",
        )
    self.begin_device_use(serial)
    if not agent.is_enabled(serial):
        if not agent.rootable(serial):
            raise DeviceError(
                f"{serial} cannot run the on-device helper, which recording needs",
                hint="The helper needs `adb root`; use a debuggable emulator image.",
            )
        self._record_device_agent_change(serial)
        agent.enable(serial)
    # Something else may be holding the slot from an earlier command in this session.
    agent.release_uiautomation(serial)
    # And check it actually let go. Android suppresses every accessibility service while
    # uiautomator2 holds UiAutomation, and a warm daemon holds it merely by existing — so
    # without this the recorder arms against a service that is not running, the human
    # walks the whole journey, and `demo stop` returns nothing while reporting the
    # recording complete. An empty journey and a journey nobody could see are
    # indistinguishable to the person who just performed one, which is the one outcome
    # this command must never produce.
    if not agent.is_bound(serial):
        raise DeviceError(
            f"the helper's accessibility service is not running on {serial}, so nothing "
            "would be recorded",
            hint=(
                "Something is holding the UiAutomation slot — usually a warm daemon. "
                "Run `aua daemon stop` (and keep other aua commands off this device "
                "while recording), then start again."
            ),
        )
    return agent, serial


def demo_record_start(self: Engine) -> dict[str, Any]:
    """Arm the device's recorder, then get out of the way.

        Nothing is driven from here. The point of this path is that a *person* demonstrates
        the journey — no agent turn per step, no selector guessing — so this call exists only
        to arm the device and release the slot, and the process then exits so that whatever
        the human does next is theirs alone.
        """

    agent, serial = self._recorder()
    channel = agent.open_channel(serial)
    try:
        result = channel.request("record.start", None)
    finally:
        channel.close()
    # The second source, and the one that catches what accessibility cannot: a view only
    # announces a click if it calls performClick, while every finger appears in the kernel
    # touch stream. Best effort — a target that will not give it up simply records what it
    # always did, with the gaps still reported honestly.
    touches = False
    with contextlib.suppress(Exception):
        agent.start_touch_capture(serial)
        touches = True
    return {
        "ok": True,
        "action": "demo-record-start",
        "serial": serial,
        "recording": bool((result or {}).get("recording", True)),
        "touch_capture": touches,
    }


def demo_record_stop(self: Engine, *, save: str | None = None, force: bool = False) -> dict[str, Any]:
    """Stop recording and return the journey as steps, with its holes named.

        ``save`` refuses an incomplete draft on purpose. The device cannot see every tap —
        a view only announces a click if it calls ``performClick`` — so a recording may be
        missing steps, and a saved flow that skips one is worse than no flow: it fails later,
        somewhere else, as though the product were broken. An incomplete draft is still
        returned for a human to finish; it just is not written out as though it were ready.
        """

    from .flows import Flow, FlowStore
    from .recordings import steps_from_recording

    agent, serial = self._recorder()
    channel = agent.open_channel(serial)
    try:
        # Ask whether it is still armed BEFORE draining. Anything that connects
        # uiautomator2 takes the UiAutomation slot back and Android tears the service
        # down; it restarts having forgotten it was recording, and drains an empty list.
        # "Nothing happened" and "nobody was watching" are otherwise the same JSON, and
        # only one of them means the journey has to be walked again.
        armed = channel.request("record.peek", None) or {}
        result = channel.request("record.stop", None)
    finally:
        channel.close()

    if not armed.get("recording", True):
        raise DeviceError(
            "the recording was lost: the helper's accessibility service was torn down "
            "part-way through the journey",
            hint=(
                "Something connected to the device while recording — usually another aua "
                "command or a daemon warming up. Run `aua daemon stop`, keep other "
                "commands off this device, and walk the journey again."
            ),
        )

    touches: list[Any] = []
    captured = False
    with contextlib.suppress(Exception):
        touches = list(agent.stop_touch_capture(serial))
        captured = True
    draft = steps_from_recording(
        (result or {}).get("steps") or [],
        touches=touches,
        snapshots=(result or {}).get("snapshots") or [],
        touch_capture=captured,
    )
    payload: dict[str, Any] = {
        "ok": True,
        "action": "demo-record-stop",
        "serial": serial,
        "count": len(draft.steps),
        "complete": draft.complete,
        "recovered_from_touches": draft.recovered,
        "steps": [step.model_dump(exclude_none=True) for step in draft.steps],
        "gaps": [
            {"after_step": gap.after_step, "reason": gap.reason, "package": gap.package}
            for gap in draft.gaps
        ],
        "blockers": draft.blockers,
        "params": draft.params,
        # Pressed, found, and impossible to name: no text, no description, no resource id.
        # A defect in the app rather than in the recording, and the reason those steps are
        # coordinates — so it is reported where someone can act on it.
        "unnamed_controls": [
            {
                "step": found.step,
                "x": found.x,
                "y": found.y,
                "bounds": list(found.bounds) if found.bounds else None,
            }
            for found in draft.unnamed_controls
        ],
        "app_initiated_changes": draft.app_initiated_changes,
    }
    if save is None:
        return payload

    if not draft.complete and not force:
        raise UsageError(
            f"refusing to save '{save}': the recording has "
            f"{len(draft.gaps)} gap(s) and {len(draft.blockers)} unreplayable step(s)",
            hint=(
                "The device cannot see every tap. Review the returned steps, fill in what "
                "is missing, and save with `aua flow save`, or pass --force to write the "
                "draft as-is."
            ),
        )
    flow = Flow(name=save, steps=draft.steps, params=draft.params)
    path = FlowStore(self.config.memory).save(flow, force=force)
    payload["saved"] = str(path)
    return payload


def _run_steps(
    self: Engine,
    steps: list[RouteStep],
    *,
    origin_package: str | None,
    allow_destructive: bool,
    allow_goto_steps: bool = False,
    scroll_fallback: bool = False,
    res: AnalyzeResult | None = None,
    executed: list[dict[str, Any]] | None = None,
    flow_depth: int = 0,
    hierarchy_ocr: bool = True,
    flow_dir: Path | None = None,
    allow_unsafe_route_effects: bool = True,
    flow_plan: _ResolvedFlowPlan | None = None,
    flow_artifacts: Any | None = None,
) -> tuple[StepFailure | None, AnalyzeResult]:
    """Execute *steps* with selector matching, settle waits, and re-perception.

        The single replay engine behind ``goto`` edge replay and ``flow run``. Between
        state-changing steps it settles (suppressed ``wait_stable``) and re-analyzes with
        a package-aware source (:meth:`_source_for`). Verification is lazy — a wrong
        screen surfaces as the next step's ``element_not_found`` — terminal verification
        (``known_screen`` / asserts) is the caller's job. Returns
        ``(failure | None, last analyze result)``.

        ``flow_dir`` is the directory of the flow file these steps came from, and is what makes
        a nested ``flow:`` reference resolvable by path — "next to me" has no meaning without
        it. It is passed down each nesting level, so a sub-flow's own references are relative
        to the sub-flow.
        """
    # Optional: let the device run a stretch of UI-only steps itself. Purely a shortcut —
    # it reports how far it got, and any refusal, absence or error leaves the whole run to
    # the host path below.
    #
    # The run does not have to start at index 0. It used to, which sounded harmless and
    # was not: real flows open with `launch_app`, a host-only step, so the prefix was
    # always empty and nothing was ever handed over — the repo's one saved flow is exactly
    # that shape. A later start is allowed, but it is a different and much more expensive
    # trade, so the opening question is answered up front from the step list alone.
    #
    # There can be more than one handover. Only a *refusal* ends them for the flow; after
    # a run that worked, ``_offload_from`` asks again from where the device stopped. Every
    # real QA flow checks what it just did, and a check is a host step — so with a single
    # handover, everything past the first check was driven one round trip at a time.
    offload_at = (
        self._pick_offload_start(steps, allow_destructive=allow_destructive)
        if flow_depth == 0
        else None
    )
    skip_until = 0

    # A run starting at 0 must be handed over *before* the opening analyze, because that
    # analyze is what connects uiautomator2, and connecting it is most of what a handover
    # costs. Doing it in the loop instead turned the cheap path into the expensive one.
    if offload_at == 0:
        skip_until, offload_at = self._offload_from(
            steps, at=0, executed=executed, allow_destructive=allow_destructive
        )

    if res is None:
        res = self._analyze_route_step(
            steps, skip_until, origin_package, hierarchy_ocr=hierarchy_ocr
        )
    lexicon = self.config.memory.destructive_labels
    for i, s in enumerate(steps):
        if i < skip_until:
            continue
        if i == offload_at:
            ran, offload_at = self._offload_from(
                steps, at=i, executed=executed, allow_destructive=allow_destructive
            )
            if ran:
                skip_until = i + ran
                # The device moved the screen; the host's view of it is now stale.
                res = self._analyze_route_step(
                    steps, skip_until, origin_package, hierarchy_ocr=hierarchy_ocr
                )
                continue
        step_started = time.perf_counter()
        step_extra: dict[str, Any] = {}
        if is_destructive_step(s, lexicon) and not allow_destructive:
            return StepFailure("destructive_step", i, s), res
        non_destructive_risks = [
            risk
            for risk in route_step_risks(
                s,
                origin_package=origin_package,
                destructive_labels=lexicon,
            )
            if risk["code"] != "destructive"
        ]
        if non_destructive_risks and not allow_unsafe_route_effects:
            return StepFailure("unsafe_route_step", i, s), res
        kind = s.kind
        reanalyze = True  # most kinds change state → settle + re-perceive
        settle = True
        if kind in ("tap", "long-press", "clear", "input"):
            if kind == "input" and s.text is None:
                # auto-recorded inputs never store the value — the caller supplies it
                return StepFailure("input_required", i, s), res
            el = _match_step(res.elements, s)
            if el is None and not hierarchy_ocr:
                source = self._source_for(steps, i, origin_package)
                if source == "hierarchy":
                    # Known native routes stay hierarchy-fast. OCR is paid only when
                    # the remembered selector is not present in accessibility data.
                    retry = self.analyze(source="hierarchy", with_ocr=True)
                    retry_el = _match_step(retry.elements, s)
                    if retry_el is not None:
                        res, el = retry, retry_el
            selector_value = s.resource_id or s.content_desc or s.label
            if el is None and scroll_fallback and selector_value:
                self.scroll_to(
                    selector_value,
                    observe=False,
                    by=s.by
                    or ("id" if s.resource_id else "desc" if s.content_desc else "text"),
                )
                res = self._analyze_route_step(
                    steps, i, origin_package, hierarchy_ocr=hierarchy_ocr
                )
                el = _match_step(res.elements, s)
            if el is None:
                return StepFailure("element_not_found", i, s), res
            if kind == "tap":
                self.tap(el.id, observe=False)
            elif kind == "long-press":
                self.long_press(el.id, observe=False)
            elif kind == "clear":
                self.clear(el.id, observe=False)
            else:
                # A step that typed nothing must diverge, not pass quietly: a flow whose
                # input never landed goes on to assert against a screen it never reached,
                # and reports the app's fault instead of its own.
                if not self.input_text(el.id, s.text or "", submit=s.submit, observe=False).ok:
                    return StepFailure("input_not_applied", i, s), res
        elif kind == "tap-point":
            point = _parse_point(s.arg)
            if point is None:
                return StepFailure("unsupported_action", i, s), res
            self.tap_point(*point, observe=False)
        elif kind == "key":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            self.key(s.arg, observe=False)
        elif kind == "swipe":
            if s.arg not in ("up", "down", "left", "right"):
                return StepFailure("unsupported_action", i, s), res
            self.swipe(s.arg, observe=False)
        elif kind == "scroll":
            # A recorded scroll used to be unsaveable *and* unreplayable: the engine
            # records kind="scroll", the flow schema had no such kind, and rendering it
            # raised KeyError('scroll') - surfacing as `internal_error: 'scroll'` out of
            # `flow save`. Any journey containing a scroll therefore could not be
            # captured at all, which is most journeys worth capturing.
            if s.arg not in ("up", "down", "left", "right"):
                return StepFailure("unsupported_action", i, s), res
            self.scroll(s.arg, observe=False)
        elif kind == "scroll-to":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            if not self.scroll_to(
                s.arg,
                observe=False,
                by=s.by or "text",
                # Default matches the CLI's `--direction up`: keep looking further down.
                direction=s.direction or "up",
            ).ok:
                return StepFailure("element_not_found", i, s), res
        elif kind == "launch-app":
            pkg = s.arg or origin_package  # bare launch_app → the flow's own app
            if not pkg:
                return StepFailure("unsupported_action", i, s), res
            # `activity:` pins the entry component on multi-launcher builds.
            self.app("launch", package=pkg, activity=s.activity)
        elif kind == "stop-app":
            pkg = s.arg or origin_package
            if not pkg:
                return StepFailure("unsupported_action", i, s), res
            self.app("stop", package=pkg)
            reanalyze = False  # app is gone; nothing to perceive until relaunch
        elif kind == "open-link":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            self.open_link(s.arg, observe=False)
        elif kind == "wait-for":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            if not self.wait(
                for_=s.arg, timeout_ms=s.timeout_ms or 10000, by=s.by or "text"
            ).ok:
                return StepFailure("wait_timeout", i, s), res
            settle = False  # the wait already absorbed the transition
        elif kind == "wait-stable":
            try:
                self.wait_stable(settle_ms=600, timeout_ms=s.timeout_ms or 15000)
            except StabilityTimeout:
                return StepFailure("wait_timeout", i, s), res
            settle = False
        elif kind == "wait-ms":
            # A deliberate fixed pause — not a UI-condition wait like `wait-for`/
            # `wait-stable` above. Exists for background work a UI signal cannot observe
            # (e.g. an async preferences flush after a deep link): nothing on screen
            # proves "the write landed", only time does. Bounded through the same
            # ceiling every other wait already goes through, so a flow cannot use it to
            # block a caller indefinitely.
            delay_ms, clamped_from, ceiling = self._bounded_wait_ms(s.timeout_ms)
            time.sleep(delay_ms / 1000)
            if clamped_from is not None:
                step_extra["wait_clamped_from_ms"] = clamped_from
                step_extra["wait_ceiling_ms"] = ceiling
            reanalyze = False
        elif kind == "assert-visible":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            if not self.has(s.arg, timeout_ms=s.timeout_ms or 0, by=s.by or "text").found:
                return StepFailure("assert_failed", i, s), res
            reanalyze = False  # pure check, screen unchanged
        elif kind == "assert-not-visible":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            if self.has(s.arg, timeout_ms=s.timeout_ms or 0, by=s.by or "text").found:
                return StepFailure("assert_failed", i, s), res
            reanalyze = False
        elif kind == "assert":
            assertion = self._run_flow_assertion(s)
            if not assertion.ok:
                return StepFailure("assert_failed", i, s, assertion.detail), res
            step_extra["assertion"] = assertion.detail
            reanalyze = False
        elif kind == "assert-order":
            ok, detail = self._run_flow_order_assertion(s)
            if not ok:
                return StepFailure("assert_failed", i, s, detail), res
            step_extra["assertion"] = detail
            reanalyze = False
        elif kind == "screenshot":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            try:
                if flow_artifacts is not None:
                    screenshot_path = flow_artifacts.capture_checkpoint(s.arg)
                else:
                    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", s.arg).strip("-._")
                    path = self._default_annotate_path(
                        self.device.serial,
                        suffix=f"flow-{safe_name or 'checkpoint'}",
                        timestamped=True,
                    )
                    screenshot_path = self.screenshot(path).detail
            except Exception as exc:  # noqa: BLE001 - preserve a resumable flow failure
                return StepFailure(
                    "screenshot_failed", i, s, f"{type(exc).__name__}: {exc}"
                ), res
            if screenshot_path:
                step_extra["screenshot"] = screenshot_path
            reanalyze = False
            settle = False
        elif kind == "hide-keyboard":
            self.hide_keyboard(observe=False)
        elif kind == "paste":
            self.paste(observe=False)
        elif kind == "dev-profile":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            self.dev_profile(s.arg)
            reanalyze = False
        elif kind == "a11y-scroll":
            el = _match_step(res.elements, s)
            if el is None:
                return StepFailure("element_not_found", i, s), res
            direction = (s.arg or "forward").lower()
            self.a11y_scroll(el.id, direction=direction, observe=False)
        elif kind == "flags-apply":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            flags_snapshot = flow_plan.flags.get(id(s)) if flow_plan is not None else None
            if flow_plan is not None and flags_snapshot is None:
                return StepFailure("resource_snapshot_missing", i, s), res
            if not self.flags_apply(
                s.arg,
                observe=False,
                _snapshot=flags_snapshot,
            ).get("ok", True):
                return StepFailure("assert_failed", i, s), res
        elif kind == "network-offline":
            if not self.network_offline(verify=True, timeout_ms=s.timeout_ms or 10_000).ok:
                return StepFailure("assert_failed", i, s), res
            reanalyze = False
        elif kind == "network-restore":
            if not self.network_restore(timeout_ms=s.timeout_ms or 15_000).ok:
                return StepFailure("assert_failed", i, s), res
            reanalyze = False
        elif kind == "network-profile":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            if not self.network_profile_apply(s.arg, timeout_ms=s.timeout_ms or 20_000).ok:
                return StepFailure("assert_failed", i, s), res
            reanalyze = False
        elif kind == "network-profile-restore":
            if not self.network_profile_restore(timeout_ms=s.timeout_ms or 20_000).ok:
                return StepFailure("assert_failed", i, s), res
            reanalyze = False
        elif kind == "clear-data":
            # `confirmed=True` because the flow itself is the confirmation: a step that says
            # `clear_data` cannot mean anything else, and it is destructive *by kind*, so it
            # has already been gated out of speculative `goto` replay.
            target = s.arg or origin_package
            if not target:
                return StepFailure("unsupported_action", i, s), res
            clear_result = self.app("clear", package=target, confirmed=True, observe=False)
            # `detail` is `target` alone unless the post-wipe settle barrier timed out
            # without proof (see `Device.clear_app`) — non-fatal, but worth surfacing on
            # this step rather than letting it vanish, which is what silently discarding
            # the action result used to do.
            if clear_result.detail and clear_result.detail != target:
                step_extra["warning"] = clear_result.detail
        elif kind == "db-execute":
            target = s.arg or origin_package
            database = str(s.data.get("database") or "")
            sql = str(s.data.get("sql") or "")
            if not (target and database and sql):
                return StepFailure("unsupported_action", i, s), res
            outcome = self.database_execute(
                target,
                database,
                sql,
                parameters=s.data.get("parameters"),
                restart=bool(s.data.get("restart", True)),
                confirmed=True,
            )
            if not outcome.get("ok", False):
                return StepFailure("assert_failed", i, s), res
        elif kind == "prefs-write":
            values = s.data.get("values")
            target = s.package or origin_package
            if not s.arg or not isinstance(values, dict) or not values or not target:
                return StepFailure("unsupported_action", i, s), res
            relaunch = bool(s.data.get("relaunch", True))
            if not self.prefs_write(target, s.arg, values, relaunch=relaunch).get("ok", True):
                return StepFailure("assert_failed", i, s), res
            # The write force-stops the app; without a relaunch there is no screen left to
            # perceive, and re-analyzing would report the launcher as the flow's state.
            reanalyze = relaunch
        elif kind == "proxy-start":
            self.proxy_start()
            reanalyze = False
        elif kind == "proxy-stop":
            self.proxy_stop()
            reanalyze = False
        elif kind == "mock-replay":
            if not s.arg:
                return StepFailure("unsupported_action", i, s), res
            cassette_snapshot = (
                flow_plan.cassettes.get(id(s)) if flow_plan is not None else None
            )
            if flow_plan is not None and cassette_snapshot is None:
                return StepFailure("resource_snapshot_missing", i, s), res
            self.mock_replay(s.arg, _snapshot=cassette_snapshot)
            reanalyze = False
        elif kind == "repeat":
            times = max(1, s.repeat or 1)
            for iteration in range(times):
                sub_executed: list[dict[str, Any]] = []
                subfail, res = self._run_steps(
                    s.substeps,
                    origin_package=origin_package,
                    allow_destructive=allow_destructive,
                    allow_goto_steps=allow_goto_steps,
                    scroll_fallback=scroll_fallback,
                    res=res,
                    executed=sub_executed,
                    flow_depth=flow_depth,
                    hierarchy_ocr=hierarchy_ocr,
                    # substeps came from the same file, so "next to me" is unchanged
                    flow_dir=flow_dir,
                    allow_unsafe_route_effects=allow_unsafe_route_effects,
                    flow_plan=flow_plan,
                    flow_artifacts=flow_artifacts,
                )
                if executed is not None:
                    for row in sub_executed:
                        child_path = row.get("path")
                        if not isinstance(child_path, list):
                            child_path = [row.get("index")]
                        executed.append(
                            {
                                **row,
                                "index": i,
                                "path": [i, iteration, *child_path],
                            }
                        )
                if subfail is not None:
                    return StepFailure(subfail.code, i, s, subfail.detail), res
            reanalyze = False
            settle = False
        elif kind == "retry":
            attempts = max(1, s.max_retries or 3)
            subfail = StepFailure("assert_failed", i, s)
            for attempt in range(attempts):
                sub_executed = []
                subfail, res = self._run_steps(
                    s.substeps,
                    origin_package=origin_package,
                    allow_destructive=allow_destructive,
                    allow_goto_steps=allow_goto_steps,
                    scroll_fallback=scroll_fallback,
                    res=res,
                    executed=sub_executed,
                    flow_depth=flow_depth,
                    hierarchy_ocr=hierarchy_ocr,
                    # substeps came from the same file, so "next to me" is unchanged
                    flow_dir=flow_dir,
                    allow_unsafe_route_effects=allow_unsafe_route_effects,
                    flow_plan=flow_plan,
                    flow_artifacts=flow_artifacts,
                )
                if executed is not None:
                    for row in sub_executed:
                        child_path = row.get("path")
                        if not isinstance(child_path, list):
                            child_path = [row.get("index")]
                        executed.append(
                            {
                                **row,
                                "index": i,
                                "path": [i, attempt, *child_path],
                            }
                        )
                if subfail is None:
                    break
            if subfail is not None:
                return StepFailure(subfail.code, i, s, subfail.detail), res
            reanalyze = False
            settle = False
        elif kind == "goto":
            if not allow_goto_steps or not s.arg:
                return StepFailure("unsupported_action", i, s), res
            out = self.goto(
                s.arg,
                allow_destructive=allow_destructive,
                # A goto nested in an explicitly authored flow is already deliberate
                # execution; keep flow semantics while standalone learned goto stays safe.
                allow_unsafe=True,
            )
            if not out.get("ok"):
                return StepFailure(str(out.get("code") or "route_unknown"), i, s), res
            settle = False  # goto verified arrival; just refresh our view
        elif kind == "flow":
            # Run a saved flow inline (Maestro's runFlow) — reuse shared recipes.
            if not allow_goto_steps or not s.arg or flow_depth >= _MAX_FLOW_DEPTH:
                return StepFailure("unsupported_action", i, s), res
            try:
                key = self._flow_ref_key(s.arg, flow_dir)
                node = flow_plan.flow_graph.get(key) if flow_plan is not None else None
                if node is None:
                    node = self._resolve_nested_flow_node(s.arg, flow_dir)
            except UsageError:
                return StepFailure("route_unknown", i, s), res
            sub, sub_dir, _source_id, sub_steps = node
            nested_executed: list[dict[str, Any]] = []
            subfail, res = self._execute_flow_steps(
                sub,
                sub_steps,
                res=res,
                allow_destructive=allow_destructive,
                scroll_fallback=scroll_fallback,
                executed=nested_executed,
                flow_depth=flow_depth + 1,
                hierarchy_ocr=hierarchy_ocr,
                flow_dir=sub_dir,
                allow_unsafe_route_effects=allow_unsafe_route_effects,
                flow_plan=flow_plan,
                flow_artifacts=flow_artifacts,
            )
            if executed is not None:
                for row in nested_executed:
                    child_path = row.get("path")
                    if not isinstance(child_path, list):
                        child_path = [row.get("index")]
                    prior_flow_path = row.get("flow_path")
                    if not isinstance(prior_flow_path, list):
                        prior_flow_path = []
                    executed.append(
                        {
                            **row,
                            "index": i,
                            "path": [i, *child_path],
                            "flow_path": [s.arg, *prior_flow_path],
                        }
                    )
            if subfail is not None:
                return StepFailure(subfail.code, i, s, subfail.detail), res
            arrival_verified, arrival_code, res, _arrival_evidence = (
                self._flow_arrival_evidence(sub, res)
            )
            if arrival_verified is False:
                return StepFailure(arrival_code or "arrival_unverified", i, s), res
            settle = False  # the sub-flow already settled
        else:
            return StepFailure("unsupported_action", i, s), res

        if reanalyze:
            if settle:
                nxt = steps[i + 1] if i + 1 < len(steps) else None
                if not self._settle_for_next_step(nxt):
                    with contextlib.suppress(StabilityTimeout):
                        self.wait_stable(settle_ms=500, timeout_ms=8000)
            res = self._analyze_route_step(
                steps, i + 1, origin_package, hierarchy_ocr=hierarchy_ocr
            )
        if executed is not None:
            step_row: dict[str, Any] = {
                "index": i,
                "step": step_display(s),
                "duration_ms": max(0, int((time.perf_counter() - step_started) * 1000)),
                **step_extra,
            }
            if flow_artifacts is not None:
                flow_artifacts.record_step(step_row, kind=kind, observation=res)
            executed.append(step_row)
    return None, res


def _flow_ref_key(ref: str, flow_dir: Path | None) -> tuple[str | None, str]:
    return (str(flow_dir.resolve()) if flow_dir is not None else None, ref)


def _resolve_nested_flow_node(self: Engine, ref: str, flow_dir: Path | None) -> _ResolvedFlowNode:
    """Load and resolve one nested flow as an immutable execution snapshot.

        Nested flows resolved by *name* from AUA's own memory directory only, so a promoted
        flow that referenced a sibling broke for anyone whose memory directory did not happen
        to contain a flow of that name. Factoring shared preconditions into ``flows/common/``
        was therefore impossible: nine shared routes had to be inlined into ~35 derived flows,
        so a fix to one does not propagate.

        A path-looking reference that resolves nowhere is **refused**, not retried as a name.
        Falling back would look up a sanitised spelling of the path in the memory directory
        (``common/auth.yaml`` → ``common_auth.yaml``), where a chance match would silently run
        a different journey. Failing to find the file the author named is recoverable; running
        somebody else's flow instead is not. The searched candidates are logged, because the
        executor's ``StepFailure`` carries a code and a step but no message.
        """
    from .flows import (
        FlowStore,
        anchor_paths,
        looks_like_path,
        nested_flow_candidates,
        parse_flow_yaml,
        resolve_params,
    )

    store = FlowStore(self.config.memory)
    if not looks_like_path(ref):
        # Names repeat across apps now, so the referring flow's own directory decides which
        # sibling is meant; an unqualified name matching two apps is refused, not guessed.
        path = store.resolve(ref, referring_dir=flow_dir).resolve()
        flow = store.load_file(path)
        directory = path.parent
        steps = anchor_paths(resolve_params(flow, {}), directory)
        return _ResolvedFlowNode(flow, directory, str(path), steps)
    candidates = nested_flow_candidates(ref, flow_dir, store.flows_dir())
    for cand in candidates:
        if cand.is_file():
            path = cand.resolve()
            flow = parse_flow_yaml(path.read_text(encoding="utf-8"), name=path.stem)
            directory = path.parent
            steps = anchor_paths(resolve_params(flow, {}), directory)
            return _ResolvedFlowNode(flow, directory, str(path), steps)
    logger.warning(
        "nested flow %r not found; looked in: %s",
        ref,
        ", ".join(str(c) for c in candidates) or "(nowhere — no referring directory)",
    )
    raise UsageError(
        f"no flow file for nested reference {ref!r}",
        hint="Tried: " + ", ".join(str(c) for c in candidates),
    )


def _resolve_nested_flow(self: Engine, ref: str, flow_dir: Path | None) -> tuple[Any, Path | None]:
    """Compatibility wrapper returning the parsed flow and its source directory."""
    node = self._resolve_nested_flow_node(ref, flow_dir)
    return node.flow, node.directory


def _flow_graph_identity(flow: Any, flow_dir: Path | None) -> str:
    directory = str(flow_dir.resolve()) if flow_dir is not None else "<memory>"
    return f"{directory}::{flow.name}"


def _preflight_nested_flow_graph(
    self: Engine,
    steps: list[RouteStep],
    *,
    flow_dir: Path | None,
    flow_app: str | None = None,
    context_id: str | None = None,
    flow_depth: int = 0,
    ancestors: tuple[str, ...] = (),
    plan: _ResolvedFlowPlan | None = None,
    goto_allowed: bool = True,
) -> _ResolvedFlowPlan:
    """Resolve and validate every composed flow before the first device mutation.

        Nested flows used to be loaded only when execution reached their step. A missing file,
        unbound parameter, invalid arrival predicate, or cycle could therefore be discovered
        after earlier parent steps had already changed the device. This filesystem-only walk
        proves the whole graph is runnable first; runtime app/context checks still happen on
        the fresh observation at the point each child begins.
        """
    if plan is None:
        plan = _ResolvedFlowPlan({}, {}, {})

    for index, step in enumerate(steps):
        if step.substeps:
            self._preflight_nested_flow_graph(
                step.substeps,
                flow_dir=flow_dir,
                flow_app=flow_app,
                context_id=context_id,
                flow_depth=flow_depth,
                ancestors=ancestors,
                plan=plan,
                goto_allowed=False,
            )
        if step.kind == "flags-apply":
            if not step.arg:
                raise UsageError("flags_apply step needs a flags file")
            flags = self.platform.capability("feature_flags")

            app, pairs = flags.load_flags_file(step.arg)
            plan.flags[id(step)] = _ResolvedFlagsResource(
                str(Path(step.arg).expanduser().resolve()),
                app,
                deepcopy(pairs),
            )
        elif step.kind == "mock-replay":
            if not step.arg:
                raise UsageError("mock_replay step needs a cassette name or path")
            pm = self.platform.capability("proxy")

            cassette = pm.cassette_dir(self.config.memory.dir) / f"{step.arg}.yaml"
            alternate = Path(step.arg).expanduser()
            selected = alternate if alternate.is_file() else cassette
            plan.cassettes[id(step)] = _ResolvedCassetteResource(
                step.arg,
                selected.resolve(),
                deepcopy(pm.load_cassette(selected)),
            )
        elif step.kind == "goto":
            if not step.arg:
                raise UsageError("goto step needs a mapped screen goal")
            mem = self._memory
            mapped_app = mem.load(flow_app) if mem is not None and flow_app else None
            if (
                mapped_app is None
                or resolve_goal(
                    mapped_app,
                    step.arg,
                    context_id=context_id,
                    destructive_labels=self.config.memory.destructive_labels,
                )
                is None
            ):
                raise UsageError(
                    f"goto target {step.arg!r} is not mapped for {flow_app or 'this flow'}",
                    hint="record/map the destination before composing it into a flow",
                )
            if not goto_allowed or flow_depth > 0 or index != 0:
                raise UsageError(
                    "goto inside a flow must be its first top-level step",
                    hint=(
                        "A later goto's route origin depends on earlier mutations and cannot "
                        "be authorized atomically; capture the exact route steps instead."
                    ),
                )
        if step.kind != "flow":
            continue
        if not step.arg:
            raise UsageError("nested flow step needs a flow name or path")
        if flow_depth >= _MAX_FLOW_DEPTH:
            raise UsageError(
                f"nested flow depth exceeds {_MAX_FLOW_DEPTH}",
                hint="remove the recursive reference or flatten the composed flow",
            )
        key = self._flow_ref_key(step.arg, flow_dir)
        node = plan.flow_graph.get(key)
        if node is None:
            node = self._resolve_nested_flow_node(step.arg, flow_dir)
            plan.flow_graph[key] = node
        child, child_dir, identity, child_steps = node
        child_app = child.app or flow_app
        child_context = child.context_id or context_id
        if child.app is not None and child.app != flow_app:
            raise UsageError(
                f"nested flow {step.arg!r} belongs to {child.app}, not parent app {flow_app}",
                hint=(
                    "Keep cross-app transit as explicit package-stamped steps in one flow "
                    "so its entry contract can be verified before the first mutation."
                ),
            )
        if child.context_id is not None and child.context_id != context_id:
            raise UsageError(
                f"nested flow {step.arg!r} uses context {child.context_id}, not {context_id}",
                hint="compose only flows recorded for the same app context",
            )
        self._validate_flow_arrival_screen(child, child_app, child_context)
        if identity in ancestors:
            chain = " -> ".join((*ancestors, identity))
            raise UsageError(
                f"nested flow cycle detected: {chain}",
                hint="remove one of the recursive flow references",
            )
        if child.arrival:
            _parse_await_terms(child.arrival, require_positive=True)
        self._preflight_nested_flow_graph(
            child_steps,
            flow_dir=child_dir,
            flow_app=child_app,
            context_id=child_context,
            flow_depth=flow_depth + 1,
            ancestors=(*ancestors, identity),
            plan=plan,
            goto_allowed=False,
        )
    return plan


def _resolved_flow_disclosure(
    self: Engine,
    steps: list[RouteStep],
    *,
    flow_dir: Path | None,
    flow_app: str | None,
    plan: _ResolvedFlowPlan,
    path_prefix: str = "steps",
    index_offset: int = 0,
) -> dict[str, Any]:
    """Describe the exact recursively resolved graph without exposing resource payloads.

        A ``flow`` step is itself conservatively non-authorizable, but that generic fact is not
        enough for review: the referenced child may change settings, data, or the environment.
        The preflight plan already pins every child file, so disclosure walks those same nodes
        instead of reopening YAML. Parsed flag values and cassette bodies deliberately remain
        private to the execution plan and never enter this result (or rendered flow YAML).
        """
    lexicon = self.config.memory.destructive_labels
    all_risks: list[dict[str, str]] = []
    graph: list[dict[str, Any]] = []

    def add_risks(items: Sequence[dict[str, str]]) -> None:
        for item in items:
            if item not in all_risks:
                all_risks.append(item)

    def walk(
        current: list[RouteStep],
        *,
        directory: Path | None,
        origin_package: str | None,
        prefix: str,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, step in enumerate(current):
            absolute_index = index + offset
            path = f"{prefix}[{absolute_index}]"
            risks = route_step_risks(
                step,
                origin_package=origin_package,
                destructive_labels=lexicon,
                path=path,
            )
            add_risks(risks)
            row: dict[str, Any] = {
                "index": absolute_index,
                "path": path,
                "step": step_display(step),
                "kind": step.kind,
                "risks": risks,
            }
            if step.substeps:
                row["substeps"] = walk(
                    step.substeps,
                    directory=directory,
                    origin_package=origin_package,
                    prefix=f"{path}.substeps",
                )
            if step.kind == "flow" and step.arg:
                node = plan.flow_graph.get(self._flow_ref_key(step.arg, directory))
                if node is not None:
                    child, child_dir, source_id, child_steps = node
                    edge = {
                        "path": path,
                        "reference": step.arg,
                        "name": child.name,
                        "source": source_id,
                        "app": child.app or origin_package,
                        "context_id": child.context_id,
                    }
                    graph.append(edge)
                    child_rows = walk(
                        child_steps,
                        directory=child_dir,
                        origin_package=child.app or origin_package,
                        prefix=f"{path}.resolved_flow.steps",
                    )
                    row["resolved_flow"] = {
                        **edge,
                        "arrival": child.arrival,
                        "arrival_screen": child.arrival_screen,
                        "arrival_status": child.arrival_status or "unverified",
                        "steps": child_rows,
                    }
            nested_rows = [
                *row.get("substeps", []),
                *((row.get("resolved_flow") or {}).get("steps") or []),
            ]
            row["destructive"] = any(
                risk.get("code") == "destructive" for risk in risks
            ) or any(bool(nested.get("destructive")) for nested in nested_rows)
            row["effects"] = sorted(
                {str(risk.get("code")) for risk in risks if risk.get("code") is not None}
                | {
                    str(effect)
                    for nested in nested_rows
                    for effect in nested.get("effects", [])
                }
            )
            rows.append(row)
        return rows

    resolved_steps = walk(
        steps,
        directory=flow_dir,
        origin_package=flow_app,
        prefix=path_prefix,
        offset=index_offset,
    )
    return {
        "steps": resolved_steps,
        "risks": all_risks,
        "effects": sorted(
            {str(risk.get("code")) for risk in all_risks if risk.get("code") is not None}
        ),
        "flow_graph": graph,
    }


def _validate_flow_arrival_screen(
    self: Engine,
    flow: Any,
    package: str | None,
    context_id: str | None,
) -> None:
    """Prove a declared mapped arrival exists, is fresh, and fits the known context."""
    if not flow.arrival_screen:
        return
    mem = self._memory
    app = mem.load(package) if mem is not None and package else None
    record = app.screens.get(flow.arrival_screen) if app is not None else None
    context_ok = bool(
        record is not None
        and (context_id is None or record.context_id in {context_id, LEGACY_CONTEXT_ID})
    )
    if record is None or record.stale or not context_ok:
        raise UsageError(
            f"flow '{flow.name}' claims unavailable mapped arrival {flow.arrival_screen!r}",
            hint="record a fresh same-context destination or use a positive arrival predicate",
        )


def _flow_leading_launch_establishes_origin(flow: Any, steps: list[RouteStep]) -> int:
    """How many leading steps bring the flow's own app to the foreground on their own.

        ``0`` means the flow depends on whatever is already in the foreground, so the
        precondition this backs must hold. A flow that opens with ``launch_app`` for its own
        package obviously satisfies it — the flow is *about* to make itself true (returns
        ``1``). The same holds for a flow that opens with one or more ``clear_data`` or
        ``stop_app`` steps immediately followed by ``launch_app``: both kill the app and drop
        the device on the launcher — so *by design* the very first run of such a setup flow
        leaves nothing in the foreground for a *second* run to match. Without this, a setup
        flow could run exactly once, ever (returns the number of leading self-clearing steps,
        plus the ``launch_app`` that follows them).

        ``stop_app`` was missing here at first, and the same docstring already justified its
        inclusion. Twenty-seven of one suite's fifty-four flows open with it, so they ran
        whenever a previous scenario happened to leave the app in the foreground and were
        refused whenever a lane started cold — the worst shape a precondition can have.

        Only a leading, uninterrupted run of the flow's OWN such steps followed by its OWN
        ``launch_app`` counts — any other step first, or a stop/clear/launch of a different
        package, still has to prove the precondition normally. A flow that genuinely depends
        on a specific starting foreground is therefore not silently let through, and killing
        an app without relaunching it is not a route to that app.
        """
    if not flow.app:
        return 0
    for offset, step in enumerate(steps):
        if step.kind in _FLOW_SELF_CLEARING_STEPS and (step.arg or flow.app) == flow.app:
            continue
        if step.kind == "launch-app" and (step.arg or flow.app) == flow.app:
            return offset + 1
        return 0
    return 0


def _flow_runtime_state(
    self: Engine,
    flow: Any,
    observation: AnalyzeResult,
    *,
    refresh_context: bool,
    transit_step: RouteStep | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return active context and an entry-contract mismatch, if any.

        The observation is the foreground truth.  Session state contributes a feature-flag
        context only when it belongs to that same package; a cursor from another app must not
        authorize a replay.  Explicit flow execution forces one runtime flag read so a recent
        out-of-band flag change cannot be hidden by the normal short refresh cache.  A resumed
        flow may continue inside configured transit only when the origin-owned session and the
        next step's explicit package both corroborate that exact foreground.
        """
    observed_package = observation.screen.package or self.current_package()
    mem = self._memory
    active_context: str | None = DEFAULT_CONTEXT_ID if mem is None else None
    owner_mismatch = bool(flow.app and observed_package != flow.app)
    transit_resume = False
    if mem is not None and self._device is not None and observed_package:
        # Serialize with async screen recording: otherwise an older background session write
        # can land after this forced flag read and put the stale context back.
        with self._mem_lock:
            session = mem.load_session(self._device.serial)
            transit_resume = bool(
                owner_mismatch
                and transit_step is not None
                and transit_step.package == observed_package
                and matches_any(observed_package, self.config.memory.transit_packages)
                and session.package == flow.app
            )
            if refresh_context and not owner_mismatch:
                self._sync_runtime_flag_context(
                    self._device,
                    observed_package,
                    mem,
                    force=True,
                )
                session = mem.load_session(self._device.serial)
            if session.package == observed_package or transit_resume:
                active_context = session.active_context_id

    if owner_mismatch and not transit_resume:
        return None, {
            "code": "flow_app_mismatch",
            "expected_package": flow.app,
            "observed_package": observed_package,
            "reason": "the flow's owning app is not in the foreground",
        }

    if flow.context_id and active_context != flow.context_id:
        return active_context, {
            "code": "flow_context_mismatch",
            "expected_context_id": flow.context_id,
            "active_context_id": active_context,
            "observed_package": observed_package,
            "reason": "the active app context does not match the recorded flow context",
        }
    return active_context, None


def _execute_flow_steps(
    self: Engine,
    flow: Any,
    steps: list[RouteStep],
    *,
    res: AnalyzeResult,
    allow_destructive: bool,
    scroll_fallback: bool,
    flow_depth: int,
    hierarchy_ocr: bool,
    flow_dir: Path | None,
    allow_unsafe_route_effects: bool,
    executed: list[dict[str, Any]],
    allow_transit_resume: bool = False,
    flow_plan: _ResolvedFlowPlan | None = None,
    flow_artifacts: Any | None = None,
) -> tuple[StepFailure | None, AnalyzeResult]:
    """Execute a resolved flow after enforcing its package/context entry contract."""

    def run_chunk(
        chunk: list[RouteStep], start: int, current: AnalyzeResult
    ) -> tuple[StepFailure | None, AnalyzeResult]:
        chunk_executed: list[dict[str, Any]] = []
        failure, latest = self._run_steps(
            chunk,
            origin_package=flow.app,
            allow_destructive=allow_destructive,
            allow_goto_steps=True,
            scroll_fallback=scroll_fallback,
            res=current,
            executed=chunk_executed,
            flow_depth=flow_depth,
            hierarchy_ocr=hierarchy_ocr,
            flow_dir=flow_dir,
            allow_unsafe_route_effects=allow_unsafe_route_effects,
            flow_plan=flow_plan,
            flow_artifacts=flow_artifacts,
        )
        for row in chunk_executed:
            row["index"] += start
        executed.extend(chunk_executed)
        if failure is not None:
            failure = StepFailure(
                failure.code, failure.at + start, failure.step, failure.detail
            )
        return failure, latest

    _active_context, mismatch = self._flow_runtime_state(
        flow,
        res,
        refresh_context=True,
        transit_step=steps[0] if allow_transit_resume and steps else None,
    )
    start = 0
    if mismatch is not None:
        establishing = self._flow_leading_launch_establishes_origin(flow, steps)
        if not (mismatch["code"] == "flow_app_mismatch" and establishing):
            return StepFailure(mismatch["code"], 0, steps[0]), res

        # A wrong foreground is allowed only for these explicit establishing steps (a
        # setup flow's leading `clear_data`s and the `launch_app` that follows them).
        # Verify the observed result (including flags) before a further flow action is
        # authorized.
        failure, res = run_chunk(steps[:establishing], 0, res)
        if failure is not None:
            return failure, res
        _active_context, mismatch = self._flow_runtime_state(flow, res, refresh_context=True)
        if mismatch is not None:
            return StepFailure(mismatch["code"], 0, steps[0]), res
        start = establishing

    if start == len(steps):
        return None, res
    return run_chunk(steps[start:], start, res)


def _flow_arrival_evidence(
    self: Engine,
    flow: Any,
    res: AnalyzeResult,
) -> tuple[bool | None, str | None, AnalyzeResult, dict[str, Any]]:
    """Verify every declared arrival condition against one terminal observation.

        ``None`` means the flow declared no arrival proof.  That remains executable when the
        caller names the flow directly, but it is never presented as verified arrival.
        """
    declared = bool(flow.arrival_screen or flow.arrival)
    predicate_ok = True
    predicate_code: str | None = None
    evidence: dict[str, Any] = {}
    terminal = res
    terminal_is_fresh = False
    if flow.arrival:
        arrival = self.await_predicate(
            flow.arrival,
            timeout_ms=30_000,
            poll_ms=300,
            observe=True,
        )
        evidence["arrival"] = arrival.model_dump(mode="json")
        predicate_ok = arrival.ok
        if not arrival.ok:
            predicate_code = f"arrival_{arrival.await_outcome or 'unverified'}"
        # When a legacy predicate and mapped screen are both present, this exact folded
        # observation is the screen proof too.  Checking the earlier last-step frame would
        # combine two different moments and could accept a transient destination.
        if arrival.observation is not None:
            terminal = arrival.observation
            terminal_is_fresh = True

    screen_ok = True
    if flow.arrival_screen:
        if not terminal_is_fresh:
            # Some valid flow steps intentionally do not re-perceive (notably stop_app,
            # because the app has just disappeared).  Their returned ``res`` describes the
            # screen from before the step and must never satisfy mapped arrival proof.  A
            # no-cache hierarchy read makes the proof about the device now; predicate-based
            # arrivals already supplied their own terminal observation above.
            terminal = self.analyze(
                source="hierarchy",
                with_ocr=False,
                no_cache=True,
            )
        expected = flow.arrival_screen
        recognized = terminal.meta.known_screen
        _active_context, runtime_mismatch = self._flow_runtime_state(
            flow,
            terminal,
            refresh_context=True,
        )
        record_ok = runtime_mismatch is None
        mem = self._memory
        package = terminal.screen.package
        if record_ok and mem is not None and package:
            from .memory import LEGACY_CONTEXT_ID

            app = mem.load(package)
            record = app.screens.get(expected) if app is not None else None
            session = mem.load_session(self.device.serial)
            allowed_contexts = {session.active_context_id, LEGACY_CONTEXT_ID}
            record_ok = bool(
                record is not None
                and not record.stale
                and record.context_id in allowed_contexts
                and session.package == package
            )
        elif record_ok:
            # ``known_screen`` is map-derived. With no usable map/session there is no way
            # to prove that a supplied or cached name is fresh for this app context.
            record_ok = False
        screen_ok = recognized == expected and record_ok
        evidence["arrival_screen"] = {
            "expected": expected,
            "recognized": recognized,
            "verified": screen_ok,
        }
    else:
        evidence["arrival_screen"] = {
            "expected": None,
            "recognized": terminal.meta.known_screen,
            "verified": False,
            "status": flow.arrival_status or "unverified",
        }

    verified = declared and predicate_ok and screen_ok
    evidence["arrival_verified"] = bool(verified)
    evidence["arrival_status"] = "verified" if verified else "unverified"
    code = None
    if declared and not screen_ok:
        code = "arrival_screen_unverified"
    elif declared and not predicate_ok:
        code = predicate_code
    return (bool(verified) if declared else None), code, terminal, evidence


def _settle_for_next_step(self: Engine, nxt: RouteStep | None) -> bool:
    """Synchronize on the next step's known selector instead of a full pixel settle.

        Returns True when the next target already appeared (caller skips ``wait_stable``).
        Falls back to False for swipes/keys/unknown labels so the conservative settle runs.
        """
    if nxt is None:
        return False
    timeout_ms = min(int(nxt.timeout_ms or 3000), 4000)
    if nxt.kind in ("wait-for", "assert-visible") and nxt.arg:
        return self.has(nxt.arg, timeout_ms=timeout_ms, by=nxt.by or "text").found
    if nxt.kind in ("tap", "long-press", "input", "clear", "a11y-scroll"):
        if nxt.resource_id:
            return self.has(nxt.resource_id, timeout_ms=timeout_ms, by="id").found
        if nxt.content_desc:
            return self.has(nxt.content_desc, timeout_ms=timeout_ms, by="desc").found
        label = nxt.label
        if label and label not in ("<filled>", "<redacted>"):
            return self.has(label, timeout_ms=timeout_ms, by="text").found
    return False


def flow_run(
    self: Engine,
    name: str | None = None,
    *,
    file: str | None = None,
    yaml: str | None = None,
    params: dict[str, str] | None = None,
    dry_run: bool = False,
    from_step: int = 0,
    allow_destructive: bool = True,
    assist: bool = False,
    allow_unsafe: bool = True,
    artifacts_dir: str | None = None,
    evidence: str = "failures",
    junit: bool = False,
    _observation: AnalyzeResult | None = None,
) -> dict[str, Any]:
    """Replay a named (or ``--file``) flow in one call — the whole journey.

        Runs through the same executor as ``goto``; on divergence returns the failing
        step's index + the remaining steps so the caller can fix or finish manually and
        resume with ``from_step``. Authored flows are deliberate intent, so destructive
        steps are ALLOWED by default (unlike goto's auto-learned replay). With *assist*
        (opt-in planner), a divergence triggers one recovery attempt (dismiss a blocking
        dialog) then resumes from the failed step before handing off.
        """
    from .flow_artifacts import FlowArtifactWriter, validate_evidence_mode
    from .flows import (
        FlowStore,
        anchor_paths,
        parse_flow_yaml,
        render_flow_yaml,
        resolve_params,
    )

    run_started = time.perf_counter()
    try:
        evidence = validate_evidence_mode(evidence)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    if junit and not artifacts_dir:
        raise UsageError("--junit needs --artifacts-dir")
    if not artifacts_dir and evidence != "failures":
        raise UsageError("--evidence needs --artifacts-dir")
    sources = [name is not None, file is not None, yaml is not None]
    if sum(sources) != 1:
        raise UsageError(
            "flow run needs exactly one of NAME, --file, or --yaml",
            hint="use a saved name, a YAML path, or an inline YAML body",
        )

    base_dir: Path | None = None
    flow_file: Path | None = None
    if file is not None:
        path = Path(file).expanduser()
        if not path.is_file():
            # Name the absolute location, always. Reporting the relative path back is
            # what hid this bug: "no flow file at flows/x.yaml" looks like a typo, while
            # "no flow file at /Users/daemon-was-started-here/flows/x.yaml" tells you
            # immediately that the lookup happened somewhere you did not expect.
            raise UsageError(
                f"no flow file at {path.resolve()}",
                hint=(
                    "That is where a relative path resolves for the process running the "
                    "flow, which is not always the shell you typed in."
                )
                if not path.is_absolute()
                else None,
            )
        flow = parse_flow_yaml(path.read_text(encoding="utf-8"), name=path.stem)
        flow_file = path.resolve()
        root_source_id = str(flow_file)
        base_dir = flow_file.parent
    elif yaml is not None:
        flow = parse_flow_yaml(yaml, name="inline")
        root_source_id = "inline:" + hashlib.sha256(yaml.encode("utf-8")).hexdigest()
    elif name is not None:
        store = FlowStore(self.config.memory)
        # The flow's own directory — not the library root — is the base a composed `flow:`
        # and a relative host path resolve against, so an app's flows can reference each
        # other by bare name once they are filed together.
        source = store.resolve(name)
        flow = store.load_file(source)
        root_source_id = str(source.resolve())
        base_dir = source.resolve().parent
    # A flow's optional YAML `name:` is display metadata. Named replay is addressed by
    # its storage key (the filename), while file replay must keep using the exact file.
    # Returning or suggesting the display name made failed journeys impossible to resume
    # whenever those identities differed.
    runnable_name = name or flow.name
    resume_prefix: str | None
    if flow_file is not None:
        resume_prefix = f"aua flow run --file {shlex.quote(str(flow_file))}"
    elif yaml is not None:
        resume_prefix = None
    else:
        resume_prefix = f"aua flow run {shlex.quote(runnable_name)}"
    identity: dict[str, Any] = {"flow": runnable_name}
    if flow.name != runnable_name:
        identity["declared_name"] = flow.name
    if flow_file is not None:
        identity["file"] = str(flow_file)
    if yaml is not None:
        identity["source"] = "inline_yaml"

    artifact_writer: FlowArtifactWriter | None = None
    if artifacts_dir:
        artifact_writer = FlowArtifactWriter(
            artifacts_dir,
            flow_name=runnable_name,
            evidence=evidence,
            junit=junit,
            screenshot=lambda path: str(self.screenshot(str(path)).detail or path),
            diagnostics=lambda: (
                self.platform.diagnostic_logs(self.device, lines=400)
                if self.platform.supports("device.logs")
                else None
            ),
        )

    def finish(
        result: dict[str, Any], observation: AnalyzeResult | None = None
    ) -> dict[str, Any]:
        duration_ms = max(0, int((time.perf_counter() - run_started) * 1000))
        result["duration_ms"] = duration_ms
        if artifact_writer is None:
            return result
        if result.get("ok") is False:
            if observation is not None:
                artifact_writer.record_failure(result, observation)
            else:
                artifact_writer.record_preflight_failure(result)
        return artifact_writer.finalize(
            result,
            canonical_flow_yaml=render_flow_yaml(flow),
            duration_ms=duration_ms,
        )

    active_context: str | None = None
    if flow.context_id and self._memory is not None and self._device is not None:
        session = self._memory.load_session(self._device.serial)
        # Dry-run is intentionally device-read-free. Disclose compatibility only when the
        # persisted context belongs to this flow's app; another foreground's cursor is not
        # evidence either way.
        if flow.app is None or session.package == flow.app:
            active_context = session.active_context_id
    if flow.arrival:
        _parse_await_terms(flow.arrival, require_positive=True)
    self._validate_flow_arrival_screen(flow, flow.app, flow.context_id)
    steps = resolve_params(flow, params or {})
    if base_dir is not None:
        # A path *inside* a flow belongs to the flow, not to the caller's cwd.
        steps = anchor_paths(steps, base_dir)
    if not 0 <= from_step < len(steps):
        raise UsageError(f"--from-step {from_step} out of range (flow has {len(steps)} steps)")
    steps_slice = steps[from_step:]
    flow_plan = self._preflight_nested_flow_graph(
        steps_slice,
        flow_dir=base_dir,
        flow_app=flow.app,
        context_id=flow.context_id,
        ancestors=(root_source_id,),
    )
    disclosure = self._resolved_flow_disclosure(
        steps_slice,
        flow_dir=base_dir,
        flow_app=flow.app,
        plan=flow_plan,
        index_offset=from_step,
    )
    lexicon = self.config.memory.destructive_labels

    def step_is_destructive(step: RouteStep, directory: Path | None) -> bool:
        if is_destructive_step(step, lexicon):
            return True
        if any(step_is_destructive(child, directory) for child in step.substeps):
            return True
        if step.kind == "flow" and step.arg:
            node = flow_plan.flow_graph.get(self._flow_ref_key(step.arg, directory))
            return bool(
                node and any(step_is_destructive(child, node.directory) for child in node.steps)
            )
        return False

    destructive_indices = [
        from_step + index
        for index, step in enumerate(steps_slice)
        if step_is_destructive(step, base_dir)
    ]
    if dry_run:
        return finish(
            {
                "ok": True,
                **identity,
                "dry_run": True,
                "app": flow.app,
                "context_id": flow.context_id,
                "arrival": flow.arrival,
                "arrival_screen": flow.arrival_screen,
                "arrival_status": flow.arrival_status or "unverified",
                "active_context_id": active_context,
                "context_compatible": (
                    None
                    if flow.context_id is not None and active_context is None
                    else flow.context_id in (None, active_context)
                ),
                "would_execute": False,
                "params_declared": sorted(flow.params),
                "steps": disclosure["steps"],
                "risks": disclosure["risks"],
                "effects": disclosure["effects"],
                "flow_graph": disclosure["flow_graph"],
                "note": "not executed (--dry-run)",
            }
        )

    if destructive_indices and not allow_destructive:
        index = destructive_indices[0]
        return finish(
            {
                "ok": False,
                "code": "destructive_step",
                **identity,
                "step_index": index,
                "failed_step": {
                    "display": step_display(steps[index]),
                    **steps[index].model_dump(),
                },
                "steps_run": [],
                "remaining_steps": [step_display(step) for step in steps[index:]],
                "hint": "review the full flow, then rerun with --allow-destructive",
            }
        )

    # Execution always begins from a current foreground observation. ``reach`` hands its
    # just-captured frame through the private seam; direct flow_run pays for exactly one.
    res = _observation or self.analyze(source="hierarchy", with_ocr=False)
    active_context, entry_mismatch = self._flow_runtime_state(
        flow,
        res,
        refresh_context=True,
        transit_step=steps_slice[0] if from_step > 0 else None,
    )
    if entry_mismatch is not None and not (
        entry_mismatch["code"] == "flow_app_mismatch"
        and self._flow_leading_launch_establishes_origin(flow, steps_slice)
    ):
        if entry_mismatch["code"] == "flow_context_mismatch":
            raise UsageError(
                f"flow '{flow.name}' was recorded for context {flow.context_id}, but the "
                f"active context is {active_context}",
                hint="activate the recorded feature/flag context before replaying this flow",
            )
        raise UsageError(
            f"flow '{flow.name}' belongs to {flow.app}, but the foreground package is "
            f"{res.screen.package or 'unknown'}",
            hint=(
                "bring the owning app to foreground, or make launch_app for that exact "
                "package the flow's first step"
            ),
        )
    executed: list[dict[str, Any]] = []

    def _exec(
        slice_start: int, res_in: AnalyzeResult | None
    ) -> tuple[Any, AnalyzeResult, int | None]:
        ex: list[dict[str, Any]] = []
        # ``res_in`` is always present: either the fresh top-level entry observation or
        # the fresh handoff returned by a failed/assisted attempt.
        assert res_in is not None
        f, r = self._execute_flow_steps(
            flow,
            steps[slice_start:],
            res=res_in,
            allow_destructive=allow_destructive,
            scroll_fallback=True,
            executed=ex,
            flow_depth=0,
            hierarchy_ocr=True,
            flow_dir=base_dir,
            allow_unsafe_route_effects=allow_unsafe,
            allow_transit_resume=slice_start > 0,
            flow_plan=flow_plan,
            flow_artifacts=artifact_writer,
        )
        for e in ex:
            e["index"] += slice_start  # absolute flow indices
            path = e.get("path")
            if isinstance(path, list) and path and isinstance(path[0], int):
                e["path"] = [path[0] + slice_start, *path[1:]]
        executed.extend(ex)
        return f, r, (slice_start + f.at if f is not None else None)

    fail, res, idx = _exec(from_step, res)
    if fail is not None and assist and self.factory.is_enabled("planner"):
        objective = (
            f"A UI automation step could not run: {step_display(fail.step)}. If a "
            "dialog, permission prompt, or popup is blocking the screen, dismiss it "
            "(Allow, Not now, Skip, Close, Continue) so the flow can proceed."
        )
        recovered, res = self._drive_with_planner(
            objective, res=res, max_steps=_ASSIST_MAX_STEPS, allow_destructive=allow_destructive
        )
        if recovered and idx is not None:
            fail, res, idx = _exec(idx, res)  # resume from the failed step
    if fail is not None:
        assert idx is not None
        if resume_prefix is not None:
            hint = (
                "fix the flow or finish the step manually, then resume with "
                f"`{resume_prefix} --from-step {idx}`"
            )
        else:
            hint = (
                "fix the flow or finish the step manually, then submit the same inline "
                f"YAML again with from_step={idx}"
            )
        if not assist:
            hint += (
                "; or add `--assist` to let a fast model clear blockers "
                "(needs `planner.enabled` + its API key)"
            )
        failure_result = {
            "ok": False,
            "code": fail.code,
            **identity,
            "step_index": idx,
            "failed_step": {"display": step_display(fail.step), **fail.step.model_dump()},
            "steps_run": executed,
            "remaining_steps": [step_display(s) for s in steps[idx:]],
            "current_screen": res.meta.known_screen,
            "elements": [
                {"id": e.id, "label": e.text or e.content_desc, "clickable": e.clickable}
                for e in res.elements
                if (e.text or e.content_desc)
            ][:20],
            "hint": hint,
        }
        if fail.detail:
            failure_result["failure_detail"] = fail.detail
        if resume_prefix is not None:
            failure_result["resume_call"] = f"{resume_prefix} --from-step {idx}"
        else:
            failure_result["resume_from_step"] = idx
        return finish(failure_result, res)
    arrival_verified, arrival_code, res, arrival_evidence = self._flow_arrival_evidence(
        flow,
        res,
    )
    out = {
        "ok": True,
        **identity,
        "steps_run": executed,
        "final_screen": res.meta.known_screen,
        # destination elements (ids) so the caller can act without a re-analyze
        "elements": [e.compact() for e in res.elements],
        **arrival_evidence,
    }
    if arrival_verified is False:
        out["ok"] = False
        out["code"] = arrival_code or "arrival_unverified"
    return finish(out, res if out.get("ok") is False else None)


def flow_save(
    self: Engine,
    name: str,
    *,
    last: int = 12,
    force: bool = False,
    save: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Preview or save a trustworthy homogeneous suffix of recorded actions.

        Redacted inputs/labels become required ``${PARAM_n}`` placeholders — typed
        values are never recorded, so the agent fills them in the saved YAML.
        """
    from .flows import (
        Flow,
        FlowStore,
        check_saveable,
        recorded_selector_resilience,
        recorded_step_blockers,
        render_flow_yaml,
        steps_from_recent,
    )
    from .memory import capture_arrival_for_current, capture_arrival_predicate

    mem = self._memory
    if mem is None:
        raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
    if force and not save:
        raise UsageError(
            "--force only applies when --save writes the previewed flow",
            hint="preview first, then add --save --force to replace the existing file",
        )
    if dry_run and save:
        raise UsageError("--dry-run and --save cannot be combined")

    # Terminal proof must describe the screen on the device now, never the cursor left by
    # an older observation.  This read also applies any foreground/context boundary before
    # the journal suffix is selected.
    current = self.analyze(source="hierarchy", with_ocr=False)
    if not self._join_memory_writers(timeout_s=5.0):
        raise UsageError(
            "recorded-flow provenance is still being finalized",
            hint="retry `aua flow save` after the current memory update completes",
        )
    sess = mem.load_session(self.device.serial)
    if last < 1:
        raise UsageError("flow save --last must be at least 1")
    requested = last
    journal = list(sess.recent)
    if not journal:
        raise UsageError(
            "no recorded actions to save",
            hint="drive the app first (tap/input/…), then `aua flow save <name>`",
        )

    warnings: list[str] = []
    newest = journal[-1]
    segment_id: int | None
    origin: str | None
    context_id: str | None
    if newest.capture_segment is not None:
        if not newest.origin_package:
            raise UsageError(
                "recorded actions have no proven origin package",
                hint="analyze the app first, then repeat the intended actions before saving",
            )
        if newest.capture_segment != sess.capture_segment:
            raise UsageError(
                "no actions exist in the current capture segment",
                hint=(
                    f"the segment was reset because {sess.capture_boundary_reason}; drive the "
                    "intended app/context again before saving"
                    if sess.capture_boundary_reason
                    else "drive the intended app/context again before saving"
                ),
            )
        segment_id = newest.capture_segment
        segment = [step for step in journal if step.capture_segment == segment_id]
        origin = newest.origin_package
        context_id = newest.context_id
    else:
        # One-release compatibility for journals captured before per-action provenance.
        # Transit packages remain folded into the owning app; a foreign non-transit action
        # starts the suffix.  The uncertainty is disclosed rather than presented as proof.
        segment_id = None
        origin = (
            sess.package
            if newest.package is None
            or matches_any(newest.package, self.config.memory.transit_packages)
            else newest.package
        )
        suffix: list[RouteStep] = []
        for step in reversed(journal):
            pkg = step.package
            if pkg not in (None, origin) and not matches_any(
                pkg, self.config.memory.transit_packages
            ):
                break
            suffix.append(step)
        segment = list(reversed(suffix))
        # A legacy step has no capture-time context.  The session's *current* context may
        # have changed since the action, so attaching it would turn uncertainty into false
        # typed provenance (and could make the flow replay in the wrong variant).
        context_id = None
        warnings.append(
            "legacy recorded actions have no per-action origin/context provenance; "
            "the newest package-compatible suffix was selected conservatively"
        )

    selected = segment[-requested:]
    boundary_omitted = max(0, min(requested, len(journal)) - len(selected))
    if boundary_omitted:
        warnings.append(
            f"requested --last {requested} crosses a capture boundary; omitted "
            f"{boundary_omitted} older action(s) and used only the newest homogeneous suffix"
        )
    if not selected:
        raise UsageError(
            "no actions exist in the current capture segment",
            hint=(
                f"the segment was reset because {sess.capture_boundary_reason}; drive the "
                "intended app/context again before saving"
                if sess.capture_boundary_reason
                else "drive the intended app/context again before saving"
            ),
        )
    if segment_id is not None:
        expected_provenance = (segment_id, origin, context_id)
        if any(
            (step.capture_segment, step.origin_package, step.context_id) != expected_provenance
            for step in selected
        ):
            raise UsageError(
                "selected recorded actions have mixed origin/context provenance",
                hint=(
                    "nothing was saved; drive the journey again after a clean app/context "
                    "boundary, or request a smaller homogeneous --last suffix"
                ),
            )

    captured_arrival = capture_arrival_for_current(
        selected,
        session=sess,
        observation_package=current.screen.package,
        observation_fingerprint=current.meta.fingerprint,
    )
    captured_predicate = (
        capture_arrival_predicate(captured_arrival.proof)
        if captured_arrival.proof is not None
        else None
    )
    arrival_screen: str | None = None
    arrival_reason: str
    if current.screen.package != origin:
        arrival_reason = (
            f"current package {current.screen.package or 'unknown'} is not the selected "
            f"segment origin {origin or 'unknown'}"
        )
    elif sess.active_context_id != context_id:
        arrival_reason = (
            f"current context {sess.active_context_id} does not match selected context "
            f"{context_id}"
        )
    elif current.meta.known_screen:
        from .memory import LEGACY_CONTEXT_ID

        app = mem.load(origin) if origin else None
        mapped = app.screens.get(current.meta.known_screen) if app is not None else None
        if (
            mapped is not None
            and not mapped.stale
            and mapped.context_id in {context_id, LEGACY_CONTEXT_ID}
        ):
            arrival_screen = current.meta.known_screen
            arrival_reason = "current destination was freshly recognized as a mapped screen"
        else:
            arrival_reason = (
                "current known_screen has no fresh map record in the selected capture context"
            )
    else:
        arrival_reason = "current destination is not a mapped known_screen"
    if arrival_screen is None and captured_predicate is not None:
        arrival_reason = captured_arrival.reason

    def arrival_payload() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "verified" if arrival_screen or captured_predicate else "unverified",
            "screen": arrival_screen,
            "reason": arrival_reason,
        }
        if captured_predicate is not None and arrival_screen is None:
            payload.update(
                predicate=captured_predicate,
                source="satisfied_action_until",
                fingerprint=current.meta.fingerprint,
            )
        return payload

    selector_resilience = [
        item.model_dump(mode="json") for item in recorded_selector_resilience(selected)
    ]

    store = FlowStore(self.config.memory)
    # Collision is per app: the same name under a different package is a different flow.
    path = store.path(name, app=origin)
    existed_before = path.is_file()
    required_save_mode = "force" if existed_before else "create"
    save_call = f"aua flow save {shlex.quote(name)} --last {requested} --save"
    if existed_before:
        save_call += " --force"
    invalid_force_probe = {
        "case": "force_without_save",
        "error_code": "usage",
        "cli": (
            f"aua --expect-error usage flow save {shlex.quote(name)} --last {requested} --force"
        ),
        "mcp": {
            "tool": "flow_save",
            "arguments": {
                "name": name,
                "last": requested,
                "force": True,
                "expect_error": "usage",
            },
        },
    }
    capture_blockers = recorded_step_blockers(selected)
    if capture_blockers:
        return {
            "ok": False,
            "action": "flow-save-preview",
            "flow": name,
            "path": str(path),
            "exists": existed_before,
            "collision": existed_before,
            "status": "not_saveable",
            "required_save_mode": required_save_mode,
            "saved": False,
            "saveable": False,
            "steps": len(selected),
            "scope": {
                "requested_last": requested,
                "selected": len(selected),
                "origin_package": origin,
                "context_id": context_id,
                "capture_segment": segment_id,
                "boundary_omitted": boundary_omitted,
            },
            "capture_warnings": capture_blockers,
            # One-release response alias for callers that handled selector refusals.
            "selector_warnings": capture_blockers,
            "arrival_proof": arrival_payload(),
            "selector_resilience": selector_resilience,
            "hint": (
                "Nothing was written. Re-record with fully captured replay arguments and "
                "a unique stable, privacy-safe selector, or author the step explicitly in YAML."
            ),
            **({"warnings": warnings} if warnings else {}),
        }

    materialized = [
        step.model_copy(update={"package": None}) if step.package == origin else step
        for step in selected
    ]
    steps, params = steps_from_recent(materialized)

    flow = Flow(
        name=name,
        app=origin,
        context_id=context_id,
        description=f"Recorded from the last {len(steps)} session actions",
        arrival_screen=arrival_screen,
        arrival=(captured_predicate if arrival_screen is None else None),
        arrival_status=(
            "mapped"
            if arrival_screen
            else "predicate_verified"
            if captured_predicate
            else "unverified"
        ),
        params=params,
        steps=steps,
    )
    preview = render_flow_yaml(flow)
    warnings.extend(check_saveable(flow))
    should_save = save and not dry_run
    if should_save:
        path = store.save(flow, force=force)
        self._flows_cache.clear()
    out = {
        "ok": True,
        "action": "flow-save" if should_save else "flow-save-preview",
        "flow": name,
        "path": str(path),
        "exists": path.is_file(),
        "collision": existed_before,
        "status": (
            "overwritten"
            if should_save and existed_before
            else "created"
            if should_save
            else "preview_existing"
            if existed_before
            else "preview_new"
        ),
        "required_save_mode": required_save_mode,
        "steps": len(steps),
        "params_needed": sorted(params),
        "saved": should_save,
        "saveable": True,
        "dry_run": dry_run,
        "scope": {
            "requested_last": requested,
            "selected": len(selected),
            "origin_package": origin,
            "context_id": context_id,
            "capture_segment": segment_id,
            "boundary_omitted": boundary_omitted,
        },
        "arrival_proof": arrival_payload(),
        "arrival_status": flow.arrival_status or "unverified",
        "selector_resilience": selector_resilience,
        "preview": preview,
        "hint": (
            "saved; edit/fill ${PARAM_n}, then preview replay with the run_preview_call"
            if should_save
            else "nothing written; review the scope, selectors, and arrival proof, then run save_call"
        ),
    }
    if should_save:
        out["run_preview_call"] = f"aua flow run {shlex.quote(name)} --dry-run"
    else:
        out["save_call"] = save_call
        if existed_before:
            out["invalid_mode_probe"] = invalid_force_probe
    if dry_run:
        warnings.append(
            "--dry-run remains a deprecated non-writing alias; flow save previews by default"
        )
    if warnings:
        out["warnings"] = warnings
    return out


def flow_delete(self: Engine, name: str) -> dict[str, Any]:
    """Idempotently delete one named flow through the shared engine boundary.

        *name* may be qualified as ``<package>:<flow>``; a bare name two apps claim is refused
        rather than resolved, because the wrong deletion is the one nobody can undo.
        """
    from .flows import FlowStore, split_flow_ref

    store = FlowStore(self.config.memory)
    found = store.find(name)
    deleted = store.delete(name)
    # Report the file that was removed; an absent flow still names where it would have been.
    path = found[0] if found else store.path(name, app=split_flow_ref(name)[0])
    if deleted:
        self._flows_cache.clear()
    return {
        "ok": True,
        "action": "flow-delete",
        "flow": name,
        "path": str(path),
        "deleted": deleted,
        "status": "deleted" if deleted else "already_absent",
    }


def flow_list(self: Engine, *, app: str | None = None) -> dict[str, Any]:
    """List flows and disclose compatibility with the attached session when known.

        *app* narrows the library to one package's flows — the question the per-app layout
        exists to answer. It is an explicit filter, never inferred from the foreground: a
        listing that silently hid another app's journeys would be indistinguishable from an
        empty library.
        """
    from .flows import FlowStore

    package: str | None = None
    context_id: str | None = None
    mem = self._memory
    can_observe_foreground = self._device is not None
    if not can_observe_foreground:
        # CLI invocations and a newly constructed MCP engine are fresh here.  Discover an
        # already-online target before using the lazy device property: an absent/offline
        # device must leave this read-only listing available rather than paying a failed u2
        # connection (or changing the device state to make one available).
        with contextlib.suppress(Exception):
            configured_serial = self.config.device.serial
            can_observe_foreground = any(
                info.state == "device"
                and (configured_serial is None or info.serial == configured_serial)
                for info in self.list_devices()
            )
    if can_observe_foreground:
        with contextlib.suppress(Exception):
            package = self.current_package()
            device = self._device
            if mem is not None and device is not None and package is not None:
                session = mem.load_session(device.serial)
                if session.package == package:
                    context_id = session.active_context_id
    return {
        "flows": FlowStore(self.config.memory).list(
            app=app,
            active_package=package if context_id is not None else None,
            active_context_id=context_id,
        ),
        "app": app,
        "active_package": package,
        "active_context_id": context_id,
    }


def suite_run(
    self: Engine,
    path: str,
    *,
    continue_on_fail: bool = False,
    text: str | None = None,
) -> dict[str, Any]:
    """Run an AC checklist YAML (path, or *text* when path is ``-``)."""
    from . import suite as suite_mod

    if text is not None:
        suite = suite_mod.parse_suite(text, source=path or "<stdin>")
    elif path == "-":
        raise UsageError(
            "suite run from stdin needs the YAML body passed as text",
            hint="CLI reads stdin when PATH is `-`.",
        )
    else:
        suite = suite_mod.load_suite(path)
    result = suite_mod.run_suite(self, suite, continue_on_fail=continue_on_fail)
    return result.as_dict()
