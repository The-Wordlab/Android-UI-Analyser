"""Getting somewhere in the app: goto over the learned map with its planner fallback, navigate and reach, the goal-driven drive lanes, back_until with map-screen recognition, open_link deeplinks with chooser handling, and map_find route previews.

Engine methods for navigation. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_navigation.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from typing import TYPE_CHECKING, Any

from .device import Device
from .engine_support import (
    _ASSIST_MAX_STEPS,
    _SYSTEM_BAR_BAND,
    _HandoverRefused,
    _parse_await_terms,
    logger,
)
from .errors import (
    DeviceError,
    ElementNotFoundError,
    ProviderError,
    SelectorAmbiguousError,
    SelectorNotFoundError,
    StabilityTimeout,
    StaleElementIdError,
    UsageError,
)
from .memory import (
    AppMap,
    RouteEdge,
    RouteStep,
    _shortest_path,
    context_view,
    is_destructive_step,
    matches_any,
    resolve_goal,
    route_step_risks,
    same_screen_family,
    screen_is_root,
    step_display,
    target_arrival_evidence,
)
from .providers.base import ScreenImage
from .providers.registry import run_chain
from .schema import ActionResult, AnalyzeResult, Element, ElementId, MatchMode
from .selectors import _match_step, is_back_resource_id

if TYPE_CHECKING:
    from .engine import Engine


def _parse_legacy_steps(action: str) -> list[RouteStep] | None:
    """Replay steps for a pre-v2 string-only edge: strictly a single ``tap 'X'``.

    Anything else — compound joins, ``tap [View]``, key/input/swipe — is unreplayable
    and returns ``None`` (a clean ``unsupported_action``, never a garbage label).
    """
    m = re.fullmatch(r"tap '([^']+)'", action)
    if m is None:
        return None
    return [RouteStep(kind="tap", label=m.group(1))]


def _goto_handoff(
    goal: str,
    target: str,
    code: str,
    hops: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    res: AnalyzeResult,
    *,
    failed_step: RouteStep | None = None,
    remaining_steps: list[RouteStep] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Stop driving and return enough state for the caller to continue manually."""
    out = {
        "ok": False,
        "code": code,
        "goal": goal,
        "target": target,
        "arrived": False,
        "hops": hops,
        "remaining_route": remaining,
        "current_screen": res.meta.known_screen,
        "suggested_gotos": res.meta.suggested_gotos,
        "elements": [
            {"id": e.id, "label": e.text or e.content_desc, "clickable": e.clickable}
            for e in res.elements
            if (e.text or e.content_desc)
        ][:20],
        "hint": hint
        or 'route diverged — finish the failed step, then `aua goto "…" --from-here` '
        "(or continue with `aua analyze` + `aua tap-and-analyze`)",
    }
    if failed_step is not None:
        out["step"] = {"display": step_display(failed_step), **failed_step.model_dump()}
    if remaining_steps:
        out["remaining_steps"] = [step_display(s) for s in remaining_steps]
        pkg = next((s.package for s in remaining_steps if s.package), None)
        if pkg:
            out["expected_package"] = pkg
    return out


_HANDOVER_HINTS = {
    "platform_has_no_helper": "The on-device helper is Android-only today.",
    "no_target_serial": "Connect a device or pass --serial.",
    "auto_setup_disabled": "Set helper.auto_setup, or run `aua helper enable` yourself.",
    "known_unavailable": "This target already refused `adb root` once this session.",
    "not_rootable": "The helper needs `adb root`; use a debuggable emulator image.",
    "setup_failed": "Try `aua helper install --reinstall`, then `aua helper enable`.",
    "device_busy_in_background": "Something is still talking to the device; retry in a moment.",
    "job_running": "A background job owns the device. Wait for it, or `aua job cancel`.",
    "daemon_owns_device": "Set helper.offload_under_daemon, or run `aua daemon stop` first.",
    "another_process_owns_device": "Another aua process owns this device.",
    "capture_would_not_settle": "A screen capture is still in flight; retry in a moment.",
    "not_bound_after_release": (
        "Something is holding the UiAutomation slot — usually a warm daemon. "
        "Run `aua daemon stop`, then try again."
    ),
}


def drive_on_device(self: Engine, goal: str, *, budget: int = 8) -> dict[str, Any]:
    """Hand a *goal* to the device and let it decide its own steps. Returns what it did.

        This is the only way ``drive.run`` is reachable. Until it existed the feature was
        registered in the APK and called by nothing, so the scoring rule inside it — the one
        component of the policy experiment that has ever driven a real device — could not be run
        against a device at all, and every change to it shipped unvalidated.

        **Not the same trade as the flow offload, in two ways that matter.**

        *There is no step floor.* ``helper.min_flow_steps`` exists because that path is strictly an
        optimisation: the host can execute the very same steps, so a handover costing more than it
        saves is pure loss, and two steps is where saving ~436ms each covers a ~682ms handover.
        Nothing here is comparable. The host has no implementation of this rule, so there is no
        cheaper path to be slower than — the choice is the device or nothing. The budget bounds the
        run instead, and one step is a legitimate answer to a one-step goal.

        *A refusal is fatal.* The flow offload swallows every refusal and runs on the host, because
        the run still completes. Somebody who asked for this by name has no fallback, so a refusal
        is raised with the reason the handover recorded — silently returning "nothing happened" to
        an explicit request is the one behaviour this must not have.

        ``helper.enabled`` is deliberately not consulted, matching every other ``aua helper``
        command: that switch governs whether AUA reaches for the device *on its own*, and this is
        somebody asking for it directly.
        """

    goal = (goal or "").strip()
    if not goal:
        raise UsageError(
            "driving needs a goal",
            hint='Say what to reach, e.g. `aua helper drive "open the display settings"`.',
        )
    budget = max(1, int(budget))

    began = time.perf_counter()
    try:
        with self._device_agent_borrowed(purpose="drive.run") as loan:
            result = loan.channel.request(
                "drive.run",
                # Expanded here, so the device receives a goal already in its own app's words.
                # The helper cannot read the app map; this is how it gets the vocabulary anyway.
                {"goal": self._goal_in_the_apps_words(goal), "budget": budget},
                # The device settles after every tap, so a step costs far more than a replayed
                # one; budget the wait per step rather than for the run.
                timeout=max(30.0, 6.0 * budget),
            )
            serial = loan.serial
            was_connected = loan.u2_was_connected
    except _HandoverRefused as refused:
        # The borrow already journalled *why* it refused, under its own cmd. This adds the
        # caller's line: without it the dashboard shows a helper decision and no record that a
        # `helper drive` was ever asked for, which is the question a reader starts from.
        self._journal_helper(
            "refused",
            # From the refusal, not from `loan`: the borrow can decline before a serial is even
            # resolved, and reading `loan.serial` here raised UnboundLocalError on every refused
            # drive — turning a clear error into a crash inside the error path.
            refused.serial,
            cmd="helper.drive",
            ok=False,
            args={"goal": goal, "budget": budget},
            result={"ok": False, "reason": refused.reason, "detail": refused.detail},
            reason=refused.reason,
            detail=refused.detail,
            ms=round((time.perf_counter() - began) * 1000, 1),
        )
        raise DeviceError(
            f"the device could not be handed the goal ({refused.reason})",
            hint=_HANDOVER_HINTS.get(
                refused.reason,
                "Run `aua helper status` to see whether the helper is installed and bound.",
            ),
        ) from refused

    steps = list(result.get("steps") or [])
    stop_reason = str(result.get("stop_reason") or "unknown")
    payload = {
        "ok": True,
        "action": "helper-drive",
        "goal": goal,
        "budget": budget,
        "serial": serial,
        "ran_on": "device",
        "stop_reason": stop_reason,
        "step_count": len(steps),
        "steps": steps,
    }
    self._journal_helper(
        "drove",
        serial,
        # `helper.drive` is the command a reader can search for; `helper.drove` is not. And `ok`
        # is stated rather than derived — a drive that reached its goal was showing FAIL.
        cmd="helper.drive",
        ok=True,
        args={"goal": goal, "budget": budget},
        result=payload,
        stop_reason=stop_reason,
        step_count=len(steps),
        ms=round((time.perf_counter() - began) * 1000, 1),
        u2_was_connected=was_connected,
    )
    # The same object the journal recorded, so what the dashboard shows and what the caller
    # receives cannot drift apart.
    return payload


def _goal_in_the_apps_words(self: Engine, goal: str) -> str:
    """Fold the current app's own vocabulary into *goal*, if any has been taught.

        Both lanes call this before deciding anything, so a term learned once is spent once. The
        expansion happens here rather than inside the rule because the helper's word tables are
        compiled into the APK: teaching `score()` would give the host lane a vocabulary the device
        lane lacks, and the two lanes are tested for agreeing about a goal.

        Best-effort by design — a missing memory backend, an unknown package or an unreadable map
        must degrade to the goal as written, never to an error. Nothing here is required for driving.
        """

    with contextlib.suppress(Exception):
        mem = self._memory
        package = self.current_package()
        app = mem.load(package) if mem is not None and package else None
        if app is not None and app.vocabulary:
            from .drive_rule import expand_goal

            return expand_goal(goal, app.vocabulary)
    return goal


def drive_on_host(self: Engine, goal: str, *, budget: int = 8) -> dict[str, Any]:
    """Drive to *goal* from the host, one round trip per step. Returns what it did.

        The same rule as :meth:`drive_on_device`, running here instead of inside the helper, because
        the helper cannot run everywhere. Android will not bind a sideloaded accessibility service
        unless adbd can run as root, which rules out every retail phone and every Play-image
        emulator — and on such a device the *only* other autopilot, ``session autopilot``, needs a
        local policy model that measured 17.5% correct node selection against this rule's 82.2%. So
        the fast lane was the only lane, and it was unavailable on the most ordinary targets there
        are.

        The trade is plain and worth stating rather than hiding: a round trip per step, against none
        for the device lane. That is the whole difference. It calls the same
        :func:`drive_rule.decide`, so the two lanes cannot disagree about what to do — only about how
        long it takes.

        Progress is keyed by ``stable_key`` rather than by position. Element ids are frame-local
        ordinals that renumber on every analyze, so keying by them would lose track of a node the
        moment the screen redrew, and ``no_progress`` would never fire.
        """

    from .drive_projection import project
    from .drive_rule import decide

    goal = (goal or "").strip()
    if not goal:
        raise UsageError(
            "driving needs a goal",
            hint='Say what to reach, e.g. `aua drive "open the display settings"`.',
        )
    budget = max(1, int(budget))
    # The app's own words for what the goal names, if any were taught. Kept separate from `goal`
    # so what is reported back is what the caller asked for, not the expansion.
    driving_goal = self._goal_in_the_apps_words(goal)

    tried: dict[str, int] = {}
    last: dict[str, str] = {}
    scrolls = 0
    steps: list[dict[str, Any]] = []
    stop_reason = "budget_exhausted"
    began = time.perf_counter()

    for index in range(budget):
        observation = self.analyze()
        elements = list(observation.elements)
        # The screen as a set of identities, so "did anything change" is answerable without the
        # meta fingerprint — that folds in the status-bar clock and ticks on its own.
        before = {e.stable_key for e in elements if e.stable_key}

        nodes: list[dict[str, Any]] = []
        for element in elements:
            key = element.stable_key
            node: dict[str, Any] = {
                "text": element.text,
                "desc": element.content_desc,
                "rid": element.resource_id,
                "clickable": bool(element.clickable),
                "scrollable": bool(element.scrollable),
                # `project` copies `id` into its `keys`, so this is how the chosen node maps
                # back to something tappable. Frame-local and renumbered every analyze, which
                # is why progress is keyed by `stable_key` and not by this.
                "id": element.id,
            }
            if key and tried.get(key):
                node["tried"] = tried[key]
                node["last"] = last.get(key, "changed")
            nodes.append(node)

        key_of = {e.id: e.stable_key for e in elements}
        projection = project(nodes)
        shown = list(projection["nodes"])
        ids = list(projection["keys"])
        chosen = decide(driving_goal, projection, scrolls_used=scrolls)

        record: dict[str, Any] = {
            "step": index,
            "shown": len(shown),
            "more": bool(projection.get("more")),
            "decision": chosen["call"],
        }

        if chosen["call"] == "done":
            record["ok"] = True
            steps.append(record)
            stop_reason = "done"
            break

        if chosen["call"] == "handoff":
            record["reason"] = chosen.get("reason")
            record["why"] = chosen.get("why")
            steps.append(record)
            stop_reason = "handoff"
            break

        if chosen["call"] == "scroll":
            record["best_score"] = chosen.get("why")
            action = self.scroll("up", observe=False)
            moved = bool(getattr(action, "ok", False))
            record["ok"] = moved
            steps.append(record)
            scrolls += 1
            if not moved:
                # Nothing moved, so there is no more screen to reveal and refusing is honest.
                stop_reason = "handoff"
                record["reason"] = "target_absent"
                break
            continue

        position = next(
            (i for i, n in enumerate(shown) if n.get("n") == chosen["n"]), None
        )
        if position is None:  # pragma: no cover - decide() only returns a listed node
            stop_reason = "internal"
            break
        target = shown[position]
        element_id = ids[position]
        record["n"] = chosen["n"]
        record["label"] = " ".join(
            str(target.get(k) or "") for k in ("text", "desc")
        ).strip()
        record["score"] = chosen.get("score")
        self.tap(element_id, observe=False)

        after = {e.stable_key for e in self.analyze().elements if e.stable_key}
        outcome = "changed" if after != before else "unchanged"
        record["outcome"] = outcome
        record["ok"] = True
        steps.append(record)
        stable = key_of.get(element_id)
        if stable:
            tried[stable] = tried.get(stable, 0) + 1
            last[stable] = outcome

    return {
        "ok": True,
        "action": "drive",
        "goal": goal,
        "budget": budget,
        "ran_on": "host",
        "stop_reason": stop_reason,
        "step_count": len(steps),
        "steps": steps,
        "ms": round((time.perf_counter() - began) * 1000, 1),
    }


def _mid_edge_path(
    self: Engine,
    app: AppMap,
    target: str,
    elements: list[Element],
    *,
    context_id: str | None = None,
) -> tuple[list[RouteEdge], int] | None:
    """Find a multi-step edge to *target* whose steps already match the current UI.

        Used by ``--from-here`` when recognition has already named a mid-journey screen
        (so shortest-path from that screen is empty) but a remembered edge still has
        remaining steps visible — e.g. edge home→images ``[Apps, Images]`` while the
        map now says ``apps``.
        """
    from .memory import DEFAULT_CONTEXT_ID, LEGACY_CONTEXT_ID

    best: tuple[RouteEdge, int, int] | None = None  # edge, resume_from, remaining
    for edge in app.routes:
        if edge.to_screen != target:
            continue
        if context_id and edge.context_id not in (
            context_id,
            DEFAULT_CONTEXT_ID,
            LEGACY_CONTEXT_ID,
        ):
            continue
        steps = edge.steps or _parse_legacy_steps(edge.action)
        if not steps:
            continue
        matches = [j for j, s in enumerate(steps) if _match_step(elements, s)]
        if not matches:
            continue
        resume = matches[-1]
        remaining = len(steps) - resume
        if best is None or remaining < best[2] or (remaining == best[2] and resume > best[1]):
            best = (edge, resume, remaining)
    if best is None:
        return None
    return [best[0]], best[1]


def _planner_view(self: Engine, res: AnalyzeResult) -> tuple[list[dict[str, Any]], ScreenImage | None]:
    """Token-light element list for the planner (+ a screenshot only if weakly labelled)."""
    elements = [
        {
            "id": e.id,
            "label": e.text or e.content_desc,
            "clickable": e.clickable,
            "input": "edittext" in (e.type or "").lower(),
        }
        for e in res.elements
    ]
    labeled = sum(1 for e in res.elements if e.text or e.content_desc)
    img: ScreenImage | None = None
    if res.elements and (labeled < 3 or labeled / len(res.elements) < 0.3):
        with contextlib.suppress(Exception):  # image is a bonus; text-only still works
            img = self.device.screenshot()
    return elements, img


def _drive_with_planner(
    self: Engine,
    objective: str,
    *,
    res: AnalyzeResult,
    max_steps: int,
    allow_destructive: bool,
    until: str | None = None,
) -> tuple[bool, AnalyzeResult]:
    """Let the opt-in planner choose actions toward *objective* until done/until/cap.

        Bounded and safe: the planner may only target an id from the list we hand it
        (validated here), its taps pass the destructive guard, and it runs at most
        *max_steps* times. Returns ``(reached, last analyze result)``. Never the happy
        path — callers gate on ``factory.is_enabled("planner")`` + an explicit opt-in.
        """
    if not self.factory.is_enabled("planner"):
        return False, res
    chain = self.factory.build_chain("planner")
    if not chain.providers:
        return False, res
    lexicon = self.config.memory.destructive_labels
    for _ in range(max(1, max_steps)):
        if until and self.has(until).found:
            return True, res
        elements, img = self._planner_view(res)
        try:
            decision, name = run_chain(
                chain,
                lambda p: p.decide(objective, elements, img),  # type: ignore[attr-defined]  # noqa: B023
                is_empty=lambda r: r is None,
                timeout_s=self.config.timeouts.planner_ms / 1000.0,
            )
        except ProviderError as exc:
            logger.info("planner unavailable: %s", exc)
            return False, res
        action = decision.action
        if action == "done":
            return True, res
        if action == "give-up":
            return False, res
        el = res.element_by_id(decision.target_id) if decision.target_id is not None else None
        if action in ("tap", "input") and el is None:
            return False, res  # invalid/off-screen id → hand off rather than guess
        if el is not None:  # destructive guard applies to the planner too
            probe = RouteStep(
                kind="tap",
                # This probe is transient policy evidence, not persisted memory. Include
                # every semantic surface so a resource-only `deleteAccount`/`signOut`
                # control cannot bypass a label-only guard (including when copy redacts).
                label=el.text,
                content_desc=el.content_desc,
                resource_id=el.resource_id,
            )
            if is_destructive_step(probe, lexicon) and not allow_destructive:
                return False, res
        if action == "tap" and el is not None:
            self.tap(el.id, observe=False)
        elif action == "input" and el is not None:
            self.input_text(el.id, decision.text or "", observe=False)
        elif action == "key" and decision.arg:
            self.key(decision.arg, observe=False)
        elif action == "swipe" and decision.arg in ("up", "down", "left", "right"):
            self.swipe(decision.arg, observe=False)
        elif action == "scroll-to" and decision.arg:
            self.scroll_to(decision.arg, observe=False)
        else:
            return False, res  # unusable decision → hand off
        with contextlib.suppress(StabilityTimeout):
            self.wait_stable(settle_ms=500, timeout_ms=8000)
        res = self.analyze(source="auto")  # planner may land on unlabeled screens
    return False, res


def _goto_assist_recover(
    self: Engine, target: str, res: AnalyzeResult, *, allow_destructive: bool
) -> tuple[bool, AnalyzeResult]:
    """On a diverged goto, let the planner try to reach *target*. Verified by
        target-specific mapped identity plus fresh screen evidence, not the planner verdict."""
    objective = (
        f"Reach the app screen named '{target}'. If a dialog, permission prompt, or "
        "popup is blocking the screen, dismiss it (Allow, Not now, Skip, Close, "
        "Continue) to make progress toward that screen."
    )
    _, res = self._drive_with_planner(
        objective, res=res, max_steps=_ASSIST_MAX_STEPS, allow_destructive=allow_destructive
    )
    memory = self._memory
    app = memory.load(res.screen.package) if memory is not None and res.screen.package else None
    recognized = res.meta.known_screen
    proof = (
        target_arrival_evidence(
            app,
            recognized or target,
            target,
            res.elements,
            screen_height=res.screen.height,
        )
        if app is not None
        and same_screen_family(app, recognized, target)
        else None
    )
    return proof is not None, res


def _assist_suggestion(self: Engine, assist: bool) -> str | None:
    """Handoff hint: suggest --assist when it wasn't used; note it was tried if it was."""
    if not assist:
        return (
            "route diverged — continue manually, or re-run with `--assist` to let a "
            "fast model try to recover (needs `planner.enabled` + its API key)"
        )
    return "route diverged and assisted recovery could not reach the target — continue manually"


def goto(
    self: Engine,
    goal: str,
    *,
    plan: bool = False,
    max_steps: int = 8,
    allow_destructive: bool = False,
    allow_unsafe: bool = False,
    assist: bool = False,
    from_here: bool = False,
    _attempted_route_ids: set[str] | None = None,
    _observation: AnalyzeResult | None = None,
) -> dict[str, Any]:
    """Drive to a remembered screen via the app map (PRD §6b).

        Resolves *goal* to a known screen, then replays the recorded steps of each edge
        on the shortest route, re-analyzing and verifying ``known_screen`` after each hop.
        On any mismatch it stops and hands back the remaining route/steps + the current
        screen, so the caller can continue manually. ``plan=True`` returns the annotated
        route without acting. Destructive steps (config ``memory.destructive_labels``)
        are refused unless *allow_destructive*. Deeplinks, cross-package actions, settings/data
        mutation, app lifecycle changes, environment changes, and other actions not provably
        limited to navigation are refused unless *allow_unsafe*. A refusal includes the full
        route/risk preview and occurs before the first state-changing step.

        ``from_here=True`` (``--from-here``): you already opened part of the journey —
        scan the first edge for the last step that still matches the *current* screen and
        resume from there (same idea as mid-auth transit resume, but for any route). When
        recognition already named a mid-journey screen so shortest-path is empty, also
        search multi-step edges that still lead to the goal and resume mid-edge.
        """
    mem = self._memory
    if mem is None:
        raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
    # Known routes normally replay stable hierarchy selectors. Keep the happy path free
    # of OCR; `_run_steps` retries with it only when a remembered label is absent.
    # ``reach`` already paid for one bootstrap observation.  Accept it through a private
    # seam so the high-level one-call path does not immediately read the same screen again.
    # Public CLI/MCP goto behavior is unchanged.
    res = _observation or self.analyze(source="hierarchy", with_ocr=False)
    serial = res.meta.device_serial or self.device.serial
    package = res.screen.package or self.current_package()
    if not package:
        return {
            "ok": False,
            "code": "no_package",
            "goal": goal,
            "hint": "could not determine the foreground app",
        }
    # Transit resume: stranded mid-auth (foreground is a transit package while the
    # session journey belongs to another app) → resolve the goal against the ORIGIN
    # app's map and continue its transit edge from the first step that matches here.
    transit_resume = False
    sess_probe = mem.load_session(serial)
    if (
        sess_probe.package
        and package != sess_probe.package
        and matches_any(package, self.config.memory.transit_packages)
        and mem.load(sess_probe.package) is not None
    ):
        package = sess_probe.package
        transit_resume = True
    app = mem.load(package)
    if app is None or not app.screens:
        return {
            "ok": False,
            "code": "route_unknown",
            "goal": goal,
            "package": package,
            "hint": "no map for this app yet — explore with `aua analyze`",
        }
    sess = mem.load_session(serial)
    # The cursor is a memory of the last screen aua *wrote down*; the analyze above is the
    # screen the device is on now. They diverge whenever a write was lost, so replaying a
    # route planned from the cursor pressed `back` twice on the Android home screen. Mid-
    # transit the observed screen belongs to another app, and there the cursor is correct.
    current = sess.current_screen
    if not transit_resume and res.meta.known_screen:
        current = res.meta.known_screen
    lexicon = self.config.memory.destructive_labels
    target = resolve_goal(
        app,
        goal,
        start=current,
        half_life_days=self.config.memory.rank_half_life_days,
        last_goal=sess.last_goal,
        context_id=sess.active_context_id,
        destructive_labels=lexicon,
    )
    if target is None:
        return {
            "ok": False,
            "code": "route_unknown",
            "goal": goal,
            "package": package,
            "known_screens": list(context_view(app, sess.active_context_id).screens),
            "hint": "no known screen matches; explore with `aua analyze`",
        }

    def arrival_proof(observation: AnalyzeResult) -> dict[str, str] | None:
        recognized = observation.meta.known_screen
        if not same_screen_family(app, recognized, target):
            return None
        return target_arrival_evidence(
            app,
            recognized or target,
            goal,
            observation.elements,
            screen_height=observation.screen.height,
        )

    mem.set_last_goal(serial, goal)  # remember intent for ranking even if we divert
    if same_screen_family(app, current, target) and not transit_resume:
        proof = arrival_proof(res)
        if proof is None:
            return {
                "ok": False,
                "code": "arrival_unproven",
                "goal": goal,
                "target": target,
                "arrived": False,
                "package": package,
                "current_screen": current,
                "elements": [element.compact() for element in res.elements],
                "hint": (
                    "The map cursor names this screen, but the requested destination is "
                    "not proven by its mapped identity or a fresh non-clickable title/anchor. "
                    "A matching clickable row is navigation evidence, not arrival."
                ),
            }
        return {
            "ok": True,
            "goal": goal,
            "target": target,
            "arrived": True,
            "already_there": True,
            "package": package,
            "route": [],
            "hops": [],
            "arrival_proof": proof,
        }
    path = _shortest_path(
        app,
        target,
        start=current,
        context_id=sess.active_context_id,
        exclude_route_ids=_attempted_route_ids,
        destructive_labels=lexicon,
    )
    resume_from = 0
    from_here_preset = False
    if not path and from_here and not transit_resume:
        # Recognition may already name a mid-journey screen while a multi-step edge
        # toward the goal still has remaining selectors on screen.
        mid = self._mid_edge_path(app, target, res.elements, context_id=sess.active_context_id)
        if mid is not None:
            path, resume_from = mid
            from_here_preset = True

    def edge_preview(edge: RouteEdge) -> dict[str, Any]:
        steps = edge.steps or _parse_legacy_steps(edge.action)
        risk_rows: list[dict[str, Any]] = []
        for step_index, step in enumerate(steps or []):
            for risk in route_step_risks(
                step,
                origin_package=package,
                destructive_labels=lexicon,
                path=f"steps[{step_index}]",
            ):
                risk_rows.append(
                    {
                        "step_index": step_index,
                        "step": step_display(step),
                        **risk,
                    }
                )
        return {
            "from": edge.from_screen,
            "action": edge.action,
            "to": edge.to_screen,
            "steps": [step_display(step) for step in (steps or [])],
            "replayable": steps is not None,
            "legacy": not edge.steps,
            "risk": "requires_opt_in" if risk_rows else "safe_navigation",
            "risks": risk_rows,
            # Kept for compatibility with callers that consumed the original plan field.
            "destructive": [
                step.label for step in (steps or []) if is_destructive_step(step, lexicon)
            ],
        }

    route = [edge_preview(edge) for edge in path]
    if not path:
        return {
            "ok": False,
            "code": "route_unknown",
            "goal": goal,
            "target": target,
            "package": package,
            "current_screen": current,
            "hint": (
                'no known route from here — try `aua goto "…" --from-here` if you '
                "already opened part of a remembered edge, or explore with `aua analyze`"
                if not from_here
                else "no known route / mid-edge match from here — explore with `aua analyze`"
            ),
        }
    if plan:
        return {
            "ok": True,
            "goal": goal,
            "target": target,
            "plan": True,
            "package": package,
            "route": route,
            "note": "not executed (--plan)",
        }

    # Preflight the WHOLE learned route before even trying to resume within it. An observed
    # edge proves that its actions preceded the destination; it does not prove that a
    # deeplink, cross-package action, or configuration step was navigation-only. Doing this
    # before transit/from-here selector matching also guarantees a blind caller sees the
    # side-effect reason rather than an unrelated element-miss from inside a risky route.
    blocked: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(path):
        edge_steps = edge.steps or _parse_legacy_steps(edge.action) or []
        start = resume_from if edge_index == 0 and from_here_preset else 0
        for step_index, step in enumerate(edge_steps[start:], start=start):
            for risk in route_step_risks(
                step,
                origin_package=package,
                destructive_labels=lexicon,
                path=f"route[{edge_index}].steps[{step_index}]",
            ):
                code = risk["code"]
                # Learned routes never execute another route/flow. This is not an opt-in
                # side effect: `_run_steps` has no safe semantics for it, so author an
                # explicit flow instead of mutating earlier hops and failing late.
                if code == "nested_execution":
                    blocked.append(
                        {
                            "edge_index": edge_index,
                            "step_index": step_index,
                            "step": step_display(step),
                            **risk,
                        }
                    )
                    continue
                if code == "destructive" and allow_destructive:
                    continue
                if code != "destructive" and allow_unsafe:
                    continue
                blocked.append(
                    {
                        "edge_index": edge_index,
                        "step_index": step_index,
                        "step": step_display(step),
                        **risk,
                    }
                )
    if blocked:
        codes = {str(item["code"]) for item in blocked}
        required: list[str] = []
        if codes - {"destructive"}:
            required.append("--allow-unsafe")
        if "destructive" in codes:
            required.append("--allow-destructive")
        first = blocked[0]
        first_edge = path[int(first["edge_index"])]
        blocked_first_steps = first_edge.steps or _parse_legacy_steps(first_edge.action) or []
        first_step = blocked_first_steps[int(first["step_index"])]
        return {
            "ok": False,
            "code": "destructive_step" if codes == {"destructive"} else "unsafe_route",
            "goal": goal,
            "target": target,
            "package": package,
            "current_screen": current,
            "route": route,
            "risks": blocked,
            "required_opt_in": required,
            "step": {"display": step_display(first_step), **first_step.model_dump()},
            "hint": (
                "No route step was executed. Review `route[].risks`, then re-run with "
                + " and ".join(required)
                + " only if every disclosed side effect is intended. For setup or mutation, "
                "prefer an explicitly authored `flow run` journey."
            ),
        }
    if not from_here_preset:
        resume_from = 0
        if transit_resume or from_here:
            first_steps = path[0].steps or _parse_legacy_steps(path[0].action)
            if first_steps is None:
                return _goto_handoff(
                    goal,
                    target,
                    "unsupported_action",
                    [],
                    route,
                    res,
                    hint=(
                        "mid-transit on a pre-v2 edge — finish manually, then re-run goto"
                        if transit_resume
                        else "first edge is not replayable — walk it once to re-record, "
                        "or author a flow"
                    ),
                )
            if transit_resume:
                res = self.analyze(source="auto")  # transit screens may be vision-tier
            matches = [j for j, s in enumerate(first_steps) if _match_step(res.elements, s)]
            if not matches:
                if transit_resume:
                    return _goto_handoff(
                        goal,
                        target,
                        "element_not_found",
                        [],
                        route,
                        res,
                        remaining_steps=first_steps,
                        hint="mid-transit, but no remembered step matches this screen — "
                        "finish it manually (`aua analyze` + `aua tap-and-analyze`), then re-run `aua goto`",
                    )
                # --from-here with no matching step: still try from the start of the edge
                # (current screen is the edge's from_screen). Agents that are mid-edge
                # with no selector visible yet fall through to full replay.
                resume_from = 0
            else:
                # Transit: first match is the auth step to perform now.
                # --from-here: last match skips already-passed prefix taps when several
                # remembered selectors are still on screen (e.g. Apps + Settings).
                resume_from = matches[0] if transit_resume else matches[-1]

    hops: list[dict[str, Any]] = []
    attempted_route_ids = set(_attempted_route_ids or ())

    def arrived_result(*, early: bool = False) -> dict[str, Any]:
        proof = arrival_proof(res)
        if proof is None:
            return {
                "ok": False,
                "code": "arrival_unproven",
                "goal": goal,
                "target": target,
                "arrived": False,
                "package": package,
                "final_screen": res.meta.known_screen,
                "hops": hops,
                "route": route,
                "elements": [element.compact() for element in res.elements],
                "hint": (
                    "Recognition named the target, but this frame does not prove the "
                    "goal-specific destination. Inspect the returned observation instead "
                    "of treating a clickable destination label as arrival."
                ),
            }
        out: dict[str, Any] = {
            "ok": True,
            "goal": goal,
            "target": target,
            "arrived": True,
            "package": package,
            "final_screen": res.meta.known_screen,
            "hops": hops,
            "route": route,
            "elements": [e.compact() for e in res.elements],
            "arrival_proof": proof,
        }
        if early:
            out["early_arrival"] = True
        return out

    def replan_from(
        reached: str | None, *, attempted_route: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Continue from a recognized divergence without replaying an attempted edge."""
        remaining_budget = max_steps - len(hops)
        if not reached or remaining_budget <= 0:
            return None
        latest = mem.load(package)
        latest_sess = mem.load_session(serial)
        if latest is None or not _shortest_path(
            latest,
            target,
            start=reached,
            context_id=latest_sess.active_context_id,
            exclude_route_ids=attempted_route_ids,
            destructive_labels=lexicon,
        ):
            return None
        follow = self.goto(
            target,
            max_steps=remaining_budget,
            allow_destructive=allow_destructive,
            allow_unsafe=allow_unsafe,
            assist=assist,
            _attempted_route_ids=attempted_route_ids,
        )
        follow["goal"] = goal
        follow["replanned_from"] = reached
        follow["hops"] = [*hops, *follow.get("hops", [])]
        follow["route"] = [*attempted_route, *follow.get("route", [])]
        return follow

    for i, edge in enumerate(path):
        if i >= max_steps:
            return _goto_handoff(goal, target, "max_steps", hops, route[i:], res)
        all_steps = edge.steps or _parse_legacy_steps(edge.action)
        if all_steps is None:
            return _goto_handoff(
                goal,
                target,
                "unsupported_action",
                hops,
                route[i:],
                res,
                hint="edge recorded before v2 — walk it once to re-record it "
                "(or author a flow), then goto can replay it",
            )
        steps = all_steps[resume_from:] if i == 0 else all_steps
        if edge.id:
            attempted_route_ids.add(edge.id)
        edge_executed: list[dict[str, Any]] = []
        fail, res = self._run_steps(
            steps,
            origin_package=package,
            allow_destructive=allow_destructive,
            allow_unsafe_route_effects=allow_unsafe,
            res=res,
            executed=edge_executed,
            hierarchy_ocr=False,
        )
        if fail is not None:
            reached = res.meta.known_screen
            if edge_executed:
                hops.append(
                    {
                        "action": edge.action,
                        "expected": edge.to_screen,
                        "known_screen": reached,
                        "ok": same_screen_family(app, reached, target),
                        "partial": True,
                        "executed_steps": edge_executed,
                        "failed_step": step_display(fail.step),
                    }
                )
            # A policy refusal or a manual handoff mid-transit did not test the edge.
            # Demote only when a mutation produced a different recognized screen inside
            # the origin app; that is actual contradictory route evidence.
            if (
                edge_executed
                and edge.id
                and reached
                and not same_screen_family(app, reached, edge.from_screen)
                and res.screen.package == package
            ):
                with contextlib.suppress(Exception):
                    mem.record_route_outcome(package, edge.id, ok=False, reached=reached)
            if same_screen_family(app, reached, target):
                return arrived_result(early=True)
            if edge_executed:
                replanned = replan_from(reached, attempted_route=route[: i + 1])
                if replanned is not None:
                    return replanned
            if assist:
                recovered, res = self._goto_assist_recover(
                    target, res, allow_destructive=allow_destructive
                )
                if recovered:
                    break  # post-loop confirms arrival from known_screen
            return _goto_handoff(
                goal,
                target,
                fail.code,
                hops,
                route[i:],
                res,
                failed_step=fail.step,
                remaining_steps=steps[fail.at :],
                hint=self._assist_suggestion(assist),
            )
        reached = res.meta.known_screen
        if not same_screen_family(app, reached, edge.to_screen) and self._observation_is_loading(
            res
        ):
            # An analyzed action can legitimately return the app's settled loading shell.
            # That is evidence the tap landed, not evidence the learned route diverged.
            # Reuse the read-only mapped-screen recognizer for one bounded arrival wait
            # before demoting the route or asking an agent to recover manually.
            with contextlib.suppress(UsageError):
                awaited = self._await_known_screen(
                    edge.to_screen,
                    timeout_ms=5_000,
                    poll_ms=200,
                )
                if awaited.ok and awaited.observation is not None:
                    res = awaited.observation
                    reached = res.meta.known_screen
        if (
            not same_screen_family(app, reached, edge.to_screen)
            and "apple_vision" not in res.meta.providers_used
        ):
            # A custom-rendered destination may not be recognisable from accessibility
            # alone. Pay for one OCR retry before declaring that the route diverged.
            retry = self.analyze(source="hierarchy", with_ocr=True)
            if same_screen_family(app, retry.meta.known_screen, edge.to_screen):
                res = retry
                reached = retry.meta.known_screen
        hops.append(
            {
                "action": edge.action,
                "expected": edge.to_screen,
                "known_screen": reached,
                "ok": same_screen_family(app, reached, edge.to_screen)
                or same_screen_family(app, reached, target),
                **({"executed_steps": edge_executed} if len(edge_executed) > 1 else {}),
            }
        )
        # Replaying an edge IS the check on it, and the device is the ground truth. This
        # outcome was computed and then thrown away on every hop of every `goto` ever run,
        # so a route that had stopped working stayed `verified` forever and no amount of
        # driving could clean the map. Measured 2026-08-10: 118 of 636 rows contradicted
        # another row on the same origin+action+context. Nothing here needs an agent.
        if edge.id:
            with contextlib.suppress(Exception):
                mem.record_route_outcome(
                    package,
                    edge.id,
                    ok=same_screen_family(app, reached, edge.to_screen),
                    reached=reached,
                )
        if same_screen_family(app, reached, target):
            return arrived_result(
                early=not same_screen_family(app, reached, edge.to_screen)
            )
        if not same_screen_family(app, reached, edge.to_screen):
            replanned = replan_from(reached, attempted_route=route[: i + 1])
            if replanned is not None:
                return replanned
            if assist:
                recovered, res = self._goto_assist_recover(
                    target, res, allow_destructive=allow_destructive
                )
                if recovered:
                    break
            return _goto_handoff(
                goal,
                target,
                "wrong_screen",
                hops,
                route[i + 1 :],
                res,
                hint=self._assist_suggestion(assist),
            )
    arrived = same_screen_family(app, res.meta.known_screen, target)
    if arrived:
        return arrived_result()
    return {
        "ok": False,
        "goal": goal,
        "target": target,
        "arrived": False,
        "package": package,
        "final_screen": res.meta.known_screen,
        "hops": hops,
        "route": route,
        # destination elements (ids) so the caller can act without a re-analyze;
        # the id cache is already warm from goto's final analyze.
        "elements": [e.compact() for e in res.elements],
    }


def reach(
    self: Engine,
    goal: str,
    *,
    until: str | None = None,
    timeout_ms: int = 30_000,
    interval_ms: int = 300,
    allow_unsafe: bool = False,
    allow_destructive: bool = False,
    assist: bool = False,
) -> dict[str, Any]:
    """Use the safest known route or flow, then optionally verify arrival evidence.

        Selection is the same pure plan returned by :meth:`session_start`.  A safe verified
        goto wins, followed by a safe matching flow.  A deeplink or risky journey is never
        selected unless its exact risk class was explicitly authorized.  The initial
        observation is reused by route/flow execution instead of being immediately repeated.
        """
    if not goal.strip():
        raise UsageError("reach needs a non-empty goal")
    if until is not None:
        _parse_await_terms(until, require_positive=True)  # preflight before any action
    observation = self.analyze(source="hierarchy", with_ocr=False)
    plan = self._goal_session_plan(goal, observation)

    def authorized(candidate: Any) -> bool:
        if candidate.safe:
            return True
        codes = {risk.get("code") for risk in candidate.risks}
        # A positive caller-owned predicate is stronger than an old flow's absent arrival
        # metadata. It authorizes only that missing-proof risk; every side effect retains
        # its normal opt-in requirement.
        if until is not None:
            codes.discard("arrival_unverified")
        if (
            "required_params" in codes
            or "legacy_route" in codes
            or "arrival_unverified" in codes
            or "arrival_invalid" in codes
            or "arrival_screen_invalid" in codes
            or "nested_execution" in codes
        ):
            return False
        if "destructive" in codes and not allow_destructive:
            return False
        return not (codes - {"destructive"}) or allow_unsafe

    candidate = next(
        (
            item
            for item in plan.candidates
            if item.kind in {"arrived", "goto", "flow", "deeplink"} and authorized(item)
        ),
        None,
    )
    if candidate is None:
        return {
            "ok": False,
            "code": "navigation_unavailable",
            "goal": goal,
            "observation": observation.model_dump(mode="json"),
            "candidates": [item.model_dump(mode="json") for item in plan.candidates],
            "recommended_call": plan.recommended_call.model_dump(mode="json"),
            "warnings": plan.warnings,
        }

    navigation: dict[str, Any]
    if candidate.kind == "arrived":
        proof = candidate.evidence.get("arrival_proof")
        navigation = {
            "ok": isinstance(proof, dict),
            "arrived": isinstance(proof, dict),
            "already_there": isinstance(proof, dict),
            "target": candidate.target,
            "final_screen": observation.meta.known_screen,
            "elements": [element.compact() for element in observation.elements],
            "arrival_proof": proof,
        }
        if not isinstance(proof, dict):
            navigation.update(
                code="arrival_unproven",
                hint=(
                    "A mapped cursor alone is not arrival proof; inspect this observation "
                    "for a non-clickable destination title/anchor."
                ),
            )
    elif candidate.kind == "goto":
        navigation = self.goto(
            goal,
            allow_unsafe=allow_unsafe,
            allow_destructive=allow_destructive,
            assist=assist,
            _observation=observation,
        )
    elif candidate.kind == "flow":
        navigation = self.flow_run(
            candidate.name,
            allow_destructive=allow_destructive,
            allow_unsafe=allow_unsafe,
            assist=assist,
            _observation=observation,
        )
    else:
        action = self.open_link(candidate.name, observe=True)
        landed = action.observation.meta.known_screen if action.observation else None
        expected = candidate.target
        proven = expected is not None and landed == expected
        navigation = {
            "ok": action.ok and (proven or until is not None),
            "action": action.model_dump(mode="json"),
            "expected_screen": expected,
            "final_screen": landed,
            "arrival_proven": proven,
        }
        if action.ok and not proven and until is None:
            navigation.update(
                code="arrival_unproven",
                hint="Intent delivery is not arrival; provide --until semantic evidence.",
            )

    out: dict[str, Any] = {
        "ok": bool(navigation.get("ok")),
        "goal": goal,
        "strategy": candidate.kind,
        "candidate": candidate.model_dump(mode="json"),
        "navigation": navigation,
    }
    if out["ok"] and until is not None:
        awaited = self.await_predicate(
            until,
            timeout_ms=timeout_ms,
            poll_ms=interval_ms,
            observe=True,
        )
        out["await"] = awaited.model_dump(mode="json")
        out["ok"] = awaited.ok
        if not awaited.ok:
            out["code"] = f"arrival_{awaited.await_outcome or 'unverified'}"
    return out


def navigate(
    self: Engine,
    goal: str,
    *,
    max_steps: int = 12,
    allow_destructive: bool = False,
    until: str | None = None,
    save_flow: str | None = None,
) -> dict[str, Any]:
    """Drive to *goal* from scratch with the opt-in planner — the self-improving path.

        No prior map needed: the planner chooses each action; because those actions run
        through the normal tap/input/… methods, the journey is **recorded into memory**,
        so a later ``aua goto <that screen>`` replays it deterministically for free. Stop
        early on ``until`` text. ``save_flow`` also materializes the taken path as a flow.
        Requires ``planner.enabled`` (this command IS the explicit opt-in).
        """
    if not self.factory.is_enabled("planner"):
        raise UsageError(
            "navigate needs the planner enabled",
            hint="set `planner.enabled: true` + the model's API key (e.g. GEMINI_API_KEY)",
        )
    mem = self._memory
    serial = self.device.serial
    capture_before = mem.load_session(serial).next_capture_order if mem else None
    res = self.analyze(source="auto")  # perceive + record the starting screen
    arrived, res = self._drive_with_planner(
        goal,
        res=res,
        max_steps=max_steps,
        allow_destructive=allow_destructive,
        until=until,
    )

    def save_refusal(code: str, reason: str, hint: str) -> dict[str, Any]:
        """Report that navigation finished but its requested artifact was not trustworthy."""
        return {
            "ok": False,
            "code": code,
            "goal": goal,
            "arrived": arrived,
            "final_screen": res.meta.known_screen,
            "package": res.screen.package,
            "elements": [e.compact() for e in res.elements],
            "flow_save": {
                "name": save_flow,
                "saved": False,
                "reason": reason,
            },
            "hint": hint,
        }

    flow_saved: str | None = None
    if save_flow:
        from .flows import Flow, FlowStore, recorded_step_blockers, steps_from_recent

        if mem is None:
            return save_refusal(
                "flow_capture_memory_disabled",
                "memory is disabled, so the planner path was not recorded",
                "Enable memory before using `navigate --save-flow`.",
            )
        if not self._join_memory_writers(timeout_s=5.0):
            return save_refusal(
                "flow_capture_pending",
                "recorded-flow provenance is still being finalized",
                "Retry `navigate --save-flow` after the current memory update completes.",
            )

        # The final observation can complete asynchronously and establish an app/context
        # boundary.  Read the journal only after that write has landed, then require every
        # selected action to belong to the finalized current segment.
        session = mem.load_session(serial)
        retained_orders = sorted(
            step.capture_order for step in session.recent if step.capture_order is not None
        )
        if (
            capture_before is not None
            and retained_orders
            and retained_orders[0] > capture_before
        ):
            return save_refusal(
                "flow_capture_overflow",
                "the planner journey exceeded the rolling action journal and its beginning was dropped",
                "Capture a shorter journey (40 actions or fewer), or split it into composed flows.",
            )
        taken = [
            step
            for step in session.recent
            if capture_before is not None
            and step.capture_order is not None
            and step.capture_order >= capture_before
        ]
        if not taken:
            return save_refusal(
                "flow_capture_empty",
                "the finalized planner journal contains no new replayable actions",
                "Drive at least one action, or omit --save-flow when already at the goal.",
            )
        newest = taken[-1]
        homogeneous = bool(
            newest.capture_segment is not None
            and newest.origin_package is not None
            and all(
                step.capture_segment == newest.capture_segment
                and step.origin_package == newest.origin_package
                and step.context_id == newest.context_id
                for step in taken
            )
        )
        if not homogeneous:
            return save_refusal(
                "flow_capture_mixed",
                "the planner path crosses an app/context boundary or lacks provenance",
                "Save a smaller homogeneous journey with `flow save`.",
            )
        if not (
            newest.capture_segment == session.capture_segment
            and newest.origin_package == session.package
            and newest.context_id == session.active_context_id
        ):
            boundary = session.capture_boundary_reason or "the foreground app/context changed"
            return save_refusal(
                "flow_capture_boundary",
                f"the recorded actions belong to an older capture segment ({boundary})",
                "Drive the intended app/context again before saving a flow.",
            )
        blockers = recorded_step_blockers(taken)
        if blockers:
            return save_refusal(
                "flow_capture_lossy",
                "the planner path cannot be replayed exactly: " + "; ".join(blockers),
                "Author the missing replay details explicitly in a flow YAML file.",
            )
        origin = newest.origin_package
        materialized = [
            step.model_copy(update={"package": None}) if step.package == origin else step
            for step in taken
        ]
        steps, params = steps_from_recent(materialized)
        arrival_screen: str | None = None
        if (
            arrived
            and res.screen.package == origin
            and session.package == origin
            and session.active_context_id == newest.context_id
            and res.meta.known_screen
        ):
            from .memory import LEGACY_CONTEXT_ID

            app = mem.load(origin) if origin else None
            record = app.screens.get(res.meta.known_screen) if app is not None else None
            if (
                record is not None
                and not record.stale
                and record.context_id in {newest.context_id, LEGACY_CONTEXT_ID}
            ):
                arrival_screen = res.meta.known_screen
        flow_store = FlowStore(self.config.memory)
        # Per app: another package owning a flow of this name is not this app's collision.
        if flow_store.path(save_flow, app=origin).exists():
            return save_refusal(
                "flow_capture_exists",
                f"flow '{save_flow}' already exists and was not overwritten",
                "Choose a new name or explicitly manage the existing flow first.",
            )
        try:
            path = flow_store.save(
                Flow(
                    name=save_flow,
                    app=origin,
                    context_id=newest.context_id,
                    description=f"Recorded by `aua navigate`: {goal}",
                    arrival_screen=arrival_screen,
                    arrival_status="mapped" if arrival_screen else "unverified",
                    params=params,
                    steps=steps,
                ),
                force=False,
            )
        except UsageError as exc:
            return save_refusal(
                "flow_capture_save_refused",
                str(exc),
                "Nothing was overwritten; choose another name or repair the existing flow.",
            )
        flow_saved = str(path)
        # Long-lived daemon/MCP engines may already have rendered flow hints for this app.
        # The newly saved journey must be discoverable on the very next observation.
        self._flows_cache.clear()
    out: dict[str, Any] = {
        "ok": arrived,
        "goal": goal,
        "arrived": arrived,
        "final_screen": res.meta.known_screen,
        "package": res.screen.package,
        "elements": [e.compact() for e in res.elements],
        "hint": (
            "goal reached — the path was recorded; next time use `aua goto` (free/fast)"
            if arrived
            else "planner could not confirm the goal — finish manually or refine the goal"
        ),
    }
    if flow_saved:
        out["flow_saved"] = flow_saved
    return out


def map_find(self: Engine, goal: str, *, package: str | None = None) -> dict[str, Any]:
    """Return a context-compatible route preview for a goal without executing it."""
    mem = self._memory
    if mem is None:
        raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
    pkg = package or self.current_package()
    if not pkg:
        raise UsageError("could not determine the foreground package")
    app = mem.load(pkg)
    if app is None:
        return {"ok": False, "goal": goal, "package": pkg, "code": "map_unknown"}
    session = mem.load_session(self.device.serial)
    context_id = session.active_context_id
    start = session.current_screen
    lexicon = self.config.memory.destructive_labels
    target = resolve_goal(
        app,
        goal,
        start=start,
        context_id=context_id,
        destructive_labels=lexicon,
    )
    path = (
        _shortest_path(
            app,
            target,
            start=start,
            context_id=context_id,
            destructive_labels=lexicon,
        )
        if target
        else None
    )
    if not target or not path:
        return {
            "ok": False,
            "goal": goal,
            "package": pkg,
            "current_screen": start,
            "context_id": context_id,
            "code": "route_unknown",
        }
    risks: list[dict[str, Any]] = []
    route: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(path):
        edge_risks: list[dict[str, Any]] = []
        if not edge.steps:
            edge_risks.append(
                {
                    "code": "legacy_route",
                    "reason": "route has no inspectable structured steps",
                    "path": f"route[{edge_index}]",
                }
            )
        else:
            for step_index, step in enumerate(edge.steps):
                edge_risks.extend(
                    route_step_risks(
                        step,
                        origin_package=app.package,
                        destructive_labels=lexicon,
                        path=f"route[{edge_index}].steps[{step_index}]",
                    )
                )
        risks.extend(edge_risks)
        route.append(
            {
                "from": edge.from_screen,
                "to": edge.to_screen,
                "status": edge.status,
                "steps": [step_display(step) for step in edge.steps],
                "risk": "requires_review" if edge_risks else "safe_navigation",
                "risks": edge_risks,
            }
        )
    safe = not risks
    required_opt_in: list[str] = []
    codes = {str(item["code"]) for item in risks}
    if codes - {"destructive", "legacy_route"}:
        required_opt_in.append("--allow-unsafe")
    if "destructive" in codes:
        required_opt_in.append("--allow-destructive")
    arguments: dict[str, Any] = {"goal": goal}
    if not safe:
        arguments["plan"] = True
    return {
        "ok": True,
        "goal": goal,
        "package": pkg,
        "current_screen": start,
        "target": target,
        "context_id": context_id,
        "safe": safe,
        "status": "ready" if safe else "requires_review",
        "route": route,
        "risks": risks,
        "required_opt_in": required_opt_in,
        "recommended_call": {
            "cli": f"aua goto {goal!r}" + (" --plan" if not safe else ""),
            "mcp": {"tool": "goto", "arguments": arguments},
            "executes": safe,
            "reason": (
                "A safe structured navigation route is ready to run."
                if safe
                else "Review the route risks before authorizing any disclosed side effect."
            ),
        },
    }


def back_until(
    self: Engine,
    predicate: str,
    *,
    back_id: int | None = None,
    back_selector: dict[str, str] | None = None,
    max_steps: int = 4,
    step_timeout_ms: int = 1_200,
    poll_ms: int = 200,
) -> ActionResult:
    """Navigate back until mapped-screen or semantic UI evidence is present.

        Each fresh frame is checked for an unambiguous toolbar Back/Navigate-up affordance and
        that stable selector is preferred; hardware Back is the fallback. This matters on nested
        Compose screens that consume the hardware event. The predicate is validated before the
        first mutation, every step is observed, and leaving the starting package stops the
        journey. Cross-package traversal belongs in a risk-preflighted route or flow.
        """
    raw_destination = (predicate or "").strip()
    known_screen_target = (
        raw_destination if re.fullmatch(r"[A-Za-z0-9_.-]+", raw_destination or "") else None
    )
    terms = [] if known_screen_target else _parse_await_terms(predicate)
    unsupported = sorted({term.by for term in terms if term.by in {"net", "log"}})
    if unsupported:
        raise UsageError(
            "back-until needs screen evidence, not off-screen evidence",
            hint="Use text:, rid:, or desc: terms so AUA knows where Back arrived.",
        )
    if not known_screen_target and not any(not term.negated for term in terms):
        raise UsageError(
            "back-until needs at least one positive destination term",
            hint="Add text:, rid:, or desc: evidence for the screen you want to reach.",
        )
    if back_selector is not None and not isinstance(back_selector, dict):
        raise UsageError("back_selector must be an object with one of rid, text, or desc")
    selector = {key: value for key, value in (back_selector or {}).items() if value}
    if back_id is not None and selector:
        raise UsageError("choose either back_id or back_selector, not both")
    if back_id is not None and back_id < 0:
        raise UsageError("back_id must be a non-negative id from the current observation")
    if selector and (len(selector) != 1 or next(iter(selector)) not in {"rid", "text", "desc"}):
        raise UsageError("back_selector must contain exactly one of rid, text, or desc")
    if not 1 <= max_steps <= 12:
        raise UsageError("back-until --max-steps must be between 1 and 12")
    if step_timeout_ms < 0 or poll_ms < 10:
        raise UsageError("back-until timeouts must be non-negative and poll at least 10ms")

    # Bind an explicit ordinal to the caller's cached frame before the predicate precheck
    # analyzes again. The ordinal itself is never identity: it must be remapped from this
    # original element onto the fresh precheck observation before any action is authorized.
    back_binding: Element | None = None
    if back_id is not None:
        cached = self._read_cache()
        back_binding = cached.element_by_id(back_id) if cached is not None else None

    started_at = time.monotonic()
    requested_total_ms = max(0, step_timeout_ms) * max_steps
    total_budget_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(requested_total_ms)
    operation_deadline = started_at + total_budget_ms / 1000.0
    self._job_context.back_wait_clamp = (clamped_from, ceiling_ms)
    device = self.device

    def wait_destination(timeout_ms: int) -> ActionResult:
        if known_screen_target:
            return self._await_known_screen(
                known_screen_target, timeout_ms=timeout_ms, poll_ms=poll_ms
            )
        return self.await_predicate(
            predicate,
            timeout_ms=timeout_ms,
            poll_ms=poll_ms,
            observe=True,
            rich_ui=False,
            hierarchy_only=True,
        )

    def refresh_weak_terminal(
        result: ActionResult,
        before_observation: AnalyzeResult | None,
    ) -> ActionResult:
        """Replace a half-attached success frame with one authoritative hierarchy read."""
        if not result.ok or not self._back_terminal_frame_is_weak(
            before_observation, result.observation
        ):
            return result
        try:
            fresh = self.analyze(
                source="hierarchy",
                with_ocr=False,
                no_cache=True,
                record=False,
            )
        except Exception:  # noqa: BLE001 - the already-proven result remains valid evidence
            return result
        if known_screen_target:
            actual = self._recognize_screen_read_only(fresh)
            fresh.meta.known_screen = actual
            still_satisfied = bool(
                actual and actual.casefold() == known_screen_target.casefold()
            )
            refreshed_terms = [
                {
                    "term": f"screen:{known_screen_target}",
                    "present": still_satisfied,
                    "satisfied": still_satisfied,
                }
            ]
        else:
            refreshed_terms = self._await_terms_on_observation(
                terms,
                [{} for _term in terms],
                fresh,
                mode=MatchMode.contains,
                ignore_case=True,
            )
            still_satisfied = all(row["satisfied"] for row in refreshed_terms)
        if still_satisfied:
            result.observation = fresh
            result.observation_present = True
            result.await_terms = refreshed_terms
            return result
        # The tiny frame's positive evidence disappeared on the authoritative reread.
        # Returning the original ok=True would certify a transient title that is no longer
        # present, exactly the false-success this refresh exists to prevent.
        result.ok = False
        result.detail = (
            "authoritative terminal reread no longer satisfies the destination evidence"
        )
        result.observation = fresh
        result.observation_present = True
        result.await_terms = refreshed_terms
        result.await_outcome = "settled-unmet"
        result.verified = False
        return result

    current = wait_destination(0)
    origin_package = str(
        current.observation.screen.package if current.observation is not None else ""
    )
    if not origin_package:
        origin_package = str((device.current_app() or {}).get("package") or "")
    if current.ok:
        current = refresh_weak_terminal(current, None)
        if not current.ok:
            return self._back_until_result(
                current,
                ok=False,
                reason="terminal_evidence_unmet",
                detail=current.detail or "terminal destination evidence disappeared",
                steps_run=[],
                started_at=started_at,
            )
        current.action = "back-until"
        current.detail = "destination already satisfied; steps=0"
        current.stop_reason = "already_satisfied"
        current.steps_run = []
        current.verified = True
        return current

    steps_run: list[dict[str, Any]] = []
    for steps in range(1, max_steps + 1):
        remaining_ms = max(0, int((operation_deadline - time.monotonic()) * 1000))
        if requested_total_ms > 0 and remaining_ms == 0:
            return self._back_until_result(
                current,
                ok=False,
                reason="wait_ceiling",
                detail="destination unmet before the command wait ceiling expired",
                steps_run=steps_run,
                started_at=started_at,
            )
        before_observation = current.observation
        before = self._back_observation_identity(current.observation)
        requested_id: ElementId | None = None
        explicit_id_invalid = False
        if steps == 1 and back_id is not None:
            if back_binding is None or current.observation is None:
                explicit_id_invalid = True
            else:
                from .identity import remap_ids

                requested_id = remap_ids(
                    [back_binding], current.observation.elements
                ).get(
                    back_binding.id
                )
                explicit_id_invalid = requested_id is None
        status, selected, frame_id = self._semantic_back_selector(
            current.observation,
            selector or None,
            frame_id=requested_id,
        )
        if explicit_id_invalid:
            status = "invalid"
        if status == "ambiguous":
            return self._back_until_result(
                current,
                ok=False,
                reason="ambiguous_back_affordance",
                detail="several Back/Navigate-up controls are visible; supply --back-rid/text/desc",
                steps_run=steps_run,
                started_at=started_at,
            )
        if status == "invalid":
            return self._back_until_result(
                current,
                ok=False,
                reason="no_back_affordance",
                detail="the explicit Back id is not a fresh enabled app-owned control",
                steps_run=steps_run,
                started_at=started_at,
            )
        if selected is None and self._mapped_screen_is_root(current.observation):
            # Never spend a hardware Back from a mapped route root on the strength of a
            # transient predicate miss. Recheck the exact destination once on a fresh frame;
            # if it remains unmet, return the in-package boundary instead of crossing it.
            rechecked = wait_destination(0)
            package = self._back_observed_package(rechecked, device)
            if origin_package and package and package != origin_package:
                return self._back_until_result(
                    rechecked,
                    ok=False,
                    reason="package_changed",
                    detail=f"foreground left {origin_package} for {package}",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if rechecked.ok:
                rechecked = refresh_weak_terminal(rechecked, before_observation)
                if not rechecked.ok:
                    return self._back_until_result(
                        rechecked,
                        ok=False,
                        reason="terminal_evidence_unmet",
                        detail=(
                            rechecked.detail or "terminal destination evidence disappeared"
                        ),
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                return self._back_until_result(
                    rechecked,
                    ok=True,
                    reason=("already_satisfied" if not steps_run else "predicate_satisfied"),
                    detail=(
                        "destination already satisfied; steps=0"
                        if not steps_run
                        else f"satisfied after {len(steps_run)} back-navigation step(s)"
                    ),
                    steps_run=steps_run,
                    started_at=started_at,
                )
            root_still_visible = self._mapped_screen_is_root(rechecked.observation)
            return self._back_until_result(
                rechecked,
                ok=False,
                reason="package_boundary_risk" if root_still_visible else "screen_unstable",
                detail=(
                    "destination evidence is still unmet on a mapped route root; stopped "
                    "before hardware Back could leave the app"
                    if root_still_visible
                    else "the mapped root changed during its safety recheck; stopped before "
                    "hardware Back could act on an unchecked frame"
                ),
                steps_run=steps_run,
                started_at=started_at,
            )
        if selected is not None:
            try:
                if frame_id is not None:
                    # The unlabeled top-left navigation affordance has no semantic selector.
                    # Its id came from this exact observation, and `tap` immediately remaps
                    # that binding against a fresh hierarchy before touching the device.
                    self.tap(element_id=frame_id, observe=False, _hierarchy_settle=True)
                else:
                    self.tap(selector=selected, observe=False, _hierarchy_settle=True)
            except SelectorAmbiguousError:
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="ambiguous_back_affordance",
                    detail="the selected Back affordance became ambiguous before action",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            except (ElementNotFoundError, SelectorNotFoundError, StaleElementIdError):
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="no_back_affordance",
                    detail="the selected Back affordance disappeared before action",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            via = "affordance"
        else:
            self.key("back", observe=False, _hierarchy_settle=True)
            via = "hardware"
        step_budget_ms = min(step_timeout_ms, remaining_ms)
        step_deadline = min(operation_deadline, time.monotonic() + (step_budget_ms / 1000.0))
        current = wait_destination(step_budget_ms)

        # `await_predicate` deliberately returns screen-changed before trusting terms from
        # a newly resumed Activity. Re-evaluate on that Activity before sending another
        # navigation action, or a destination that just appeared could be overshot. A chain
        # of Activity transitions is observed only within this step's original deadline;
        # if it remains unstable, stop rather than act on an unchecked screen.
        transition_rechecks = 0
        while not current.ok and current.await_outcome == "screen-changed":
            package = self._back_observed_package(current, device)
            if origin_package and package and package != origin_package:
                steps_run.append(
                    self._back_step_evidence(
                        index=steps - 1,
                        via=via,
                        selector=selected,
                        before=before,
                        observation=current.observation,
                    )
                )
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="package_changed",
                    detail=f"foreground left {origin_package} for {package}",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            remaining_ms = max(0, int((step_deadline - time.monotonic()) * 1000))
            if transition_rechecks >= 3 or (transition_rechecks and remaining_ms == 0):
                steps_run.append(
                    self._back_step_evidence(
                        index=steps - 1,
                        via=via,
                        selector=selected,
                        before=before,
                        observation=current.observation,
                    )
                )
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="screen_unstable",
                    detail="screen kept changing before destination evidence could be checked",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            current = wait_destination(remaining_ms)
            transition_rechecks += 1
        package = self._back_observed_package(current, device)
        steps_run.append(
            self._back_step_evidence(
                index=steps - 1,
                via=via,
                selector=selected,
                before=before,
                observation=current.observation,
            )
        )
        if origin_package and package and package != origin_package:
            return self._back_until_result(
                current,
                ok=False,
                reason="package_changed",
                detail=f"foreground left {origin_package} for {package}",
                steps_run=steps_run,
                started_at=started_at,
            )
        after = self._back_observation_identity(current.observation)
        if current.ok:
            current = refresh_weak_terminal(current, before_observation)
            if not current.ok:
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="terminal_evidence_unmet",
                    detail=current.detail or "terminal destination evidence disappeared",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            return self._back_until_result(
                current,
                ok=True,
                reason="predicate_satisfied",
                detail=f"satisfied after {steps} back-navigation step(s)",
                steps_run=steps_run,
                started_at=started_at,
            )
        if known_screen_target and (
            current.observation is None or not current.observation.meta.known_screen
        ):
            return self._back_until_result(
                current,
                ok=False,
                reason="screen_unrecognized",
                detail=(
                    "the post-Back frame was not recognized by the app map; stopped "
                    "rather than risk overshooting the requested screen"
                ),
                steps_run=steps_run,
                started_at=started_at,
            )
        if known_screen_target and self._mapped_screen_state(current.observation) == "loading":
            return self._back_until_result(
                current,
                ok=False,
                reason="screen_unstable",
                detail=(
                    "the post-Back frame is still a mapped loading state; stopped rather "
                    "than navigate again before it settles"
                ),
                steps_run=steps_run,
                started_at=started_at,
            )
        if before and after and before == after:
            retry_hint = (
                "; reuse this returned observation and retry once with --back-id <fresh-id> "
                "only if that id is visibly the app-owned unlabeled Back control, or use "
                "--back-rid/--back-desc for a semantic Back control"
                if via == "hardware"
                else ""
            )
            return self._back_until_result(
                current,
                ok=False,
                reason="no_progress",
                detail=f"{via} Back produced no semantic screen change{retry_hint}",
                steps_run=steps_run,
                started_at=started_at,
            )

    return self._back_until_result(
        current,
        ok=False,
        reason="max_steps",
        detail=f"destination unmet after max_steps={max_steps}",
        steps_run=steps_run,
        started_at=started_at,
    )


def _recognize_screen_read_only(self: Engine, observation: AnalyzeResult) -> str | None:
    """Recognize one hierarchy frame without recording or mutating app memory."""
    package = observation.screen.package or ""
    memory = self._memory
    if memory is None or not package or memory.load(package) is None:
        return None
    return memory.recognize_screen(
        self.device.serial,
        package=package,
        elements=observation.elements,
        activity=observation.screen.activity,
        screen_height=observation.screen.height,
    )


def _await_known_screen(self: Engine, target: str, *, timeout_ms: int, poll_ms: int) -> ActionResult:
    """Observe hierarchy frames until memory recognizes *target*, within one Back step."""
    timeout_ms, clamped_from, ceiling_ms = self._bounded_wait_ms(timeout_ms)
    started_at = time.monotonic()
    deadline = started_at + max(0.0, timeout_ms / 1000.0)
    checks = 0
    while True:
        checks += 1
        observation = self.analyze(source="hierarchy", with_ocr=False, record=False)
        package = observation.screen.package or ""
        memory = self._memory
        app = memory.load(package) if memory is not None and package else None
        if memory is None or app is None or target not in app.screens:
            scope = package or "the foreground app"
            raise UsageError(
                f"{target!r} is not a mapped screen for {scope}",
                hint=(
                    "Use `aua map --find <goal>` to discover exact screen names, or pass "
                    "positive text:/rid:/desc: destination evidence."
                ),
            )
        resolved_target = target
        actual = self._recognize_screen_read_only(observation)
        # `record=False` intentionally leaves map metadata blank. Surface the result of
        # this read-only anchor recognition so the caller can reuse the final frame and so
        # the next Back is allowed only from a stable mapped intermediate screen.
        observation.meta.known_screen = actual
        satisfied = same_screen_family(app, actual, resolved_target)
        elapsed = int((time.monotonic() - started_at) * 1000)
        if satisfied or time.monotonic() >= deadline:
            outcome = "satisfied" if satisfied else "timeout"
            return self._say_the_wait_was_shortened(
                ActionResult(
                    ok=satisfied,
                    action="await",
                    detail=(
                        f"{outcome} after {elapsed}ms ({checks} checks)"
                        + ("" if satisfied else f"; current screen: {actual or 'unknown'}")
                    ),
                    observation=observation,
                    observation_present=True,
                    await_outcome=outcome,
                    await_terms=[
                        {
                            "term": f"screen:{resolved_target}",
                            "present": satisfied,
                            "satisfied": satisfied,
                        }
                    ],
                    elapsed_ms=elapsed,
                ),
                clamped_from,
                ceiling_ms,
            )
        self._sleep_between_polls(max(10.0, float(poll_ms)), deadline)


def _mapped_screen_state(self: Engine, observation: AnalyzeResult | None) -> str | None:
    """Return a recognized screen's remembered state without changing app memory."""
    if observation is None or not observation.meta.known_screen:
        return None
    package = observation.screen.package or ""
    memory = self._memory
    app = memory.load(package) if memory is not None and package else None
    if app is None:
        return None
    record = app.screens.get(observation.meta.known_screen)
    return record.state if record is not None else None


def _mapped_screen_is_root(self: Engine, observation: AnalyzeResult | None) -> bool:
    """Whether this exact frame is a recognized in-app route root.

        Hardware Back from a mapped root can leave the package. The route map gives us a
        conservative boundary without guessing from toolbar geometry; absent or incomplete
        memory returns False and preserves the existing bounded hardware behavior.
        """
    if observation is None or not observation.meta.known_screen:
        return False
    package = observation.screen.package or ""
    memory = self._memory
    app = memory.load(package) if memory is not None and package else None
    if memory is None or app is None:
        return False
    context_id: str | None = None
    with contextlib.suppress(Exception):
        session = memory.load_session(observation.meta.device_serial or self.device.serial)
        if session.package == package:
            context_id = session.active_context_id
    return screen_is_root(app, observation.meta.known_screen, context_id)


def _back_terminal_frame_is_weak(
    before: AnalyzeResult | None,
    after: AnalyzeResult | None,
) -> bool:
    """Detect a half-attached terminal hierarchy without penalising truly sparse screens."""
    if after is None:
        return True
    before_count = len(before.elements) if before is not None else 0
    after_count = len(after.elements)
    if before_count >= 8 and after_count * 3 < before_count:
        return True
    if after_count == 0:
        return False
    usable = any(
        element.window in {None, "app"}
        and (
            element.clickable is True
            or bool((element.text or "").strip())
            or bool((element.content_desc or "").strip())
            or bool((element.resource_id or "").strip())
        )
        for element in after.elements
    )
    return not usable


def _semantic_back_selector(
    observation: AnalyzeResult | None,
    override: dict[str, str] | None = None,
    *,
    frame_id: ElementId | None = None,
) -> tuple[str, dict[str, str] | None, ElementId | None]:
    """One app-owned Back selector, plus none/ambiguous status."""
    if override:
        return "one", override, None
    if observation is None:
        return "none", None, None
    if frame_id is not None:
        element = observation.element_by_id(frame_id)
        if element is None:
            return "invalid", None, None
        if (
            element.clickable is not True
            or element.enabled is False
            or element.window not in {None, "app"}
        ):
            return "invalid", None, None
        return "one", {"frame_id": str(frame_id)}, frame_id
    candidates: list[tuple[dict[str, str], int | None]] = []
    bottom = int(observation.screen.height * _SYSTEM_BAR_BAND)
    for element in observation.elements:
        if element.clickable is not True or element.enabled is False:
            continue
        if element.window not in {None, "app"}:
            continue
        rid = (element.resource_id or "").strip()
        if (
            ":" in rid
            and observation.screen.package
            and not rid.startswith(observation.screen.package + ":")
        ):
            continue
        if element.bounds[1] >= bottom:
            continue
        desc = (element.content_desc or "").strip()
        text = (element.text or "").strip()
        if is_back_resource_id(rid):
            candidates.append(({"rid": rid}, None))
        elif desc.casefold() in {"back", "navigate up", "up"}:
            candidates.append(({"desc": desc}, None))
        elif text.casefold() == "back":
            candidates.append(({"text": text}, None))
    if not candidates:
        return "none", None, None
    if len(candidates) > 1:
        return "ambiguous", None, None
    selected, frame_id = candidates[0]
    return "one", selected, frame_id


def _back_observation_identity(observation: AnalyzeResult | None) -> str | None:
    if observation is None:
        return None
    labels = tuple(
        (
            (element.resource_id or "")[:80],
            (element.content_desc or "")[:80],
            (element.text or "")[:80],
            element.bounds,
        )
        for element in observation.elements
        if element.resource_id or element.content_desc or element.text
    )
    fingerprint = hashlib.sha256(
        repr((observation.screen.package, observation.meta.known_screen, labels)).encode()
    ).hexdigest()[:12]
    if observation.meta.known_screen:
        return f"{observation.meta.known_screen}:{fingerprint}"
    return fingerprint


def _back_observed_package(current: ActionResult, device: Device) -> str:
    return str(
        (current.observation.screen.package if current.observation is not None else "")
        or (device.current_app() or {}).get("package")
        or ""
    )


def _back_until_result(
    self: Engine,
    current: ActionResult,
    *,
    ok: bool,
    reason: str,
    detail: str,
    steps_run: list[dict[str, Any]],
    started_at: float,
) -> ActionResult:
    current.ok = ok
    current.action = "back-until"
    current.detail = detail
    current.stop_reason = reason
    current.steps_run = steps_run
    current.elapsed_ms = int((time.monotonic() - started_at) * 1000)
    current.verified = ok
    # Every other observed action reports arrival in the top-level `known_screen`; this one
    # hid it inside `await_terms`, so a caller reading the documented field got None for a
    # call that fully succeeded. Fall back to the observation's own answer.
    if current.known_screen is None and current.observation is not None:
        meta = getattr(current.observation, "meta", None)
        if meta is not None:
            current.known_screen = meta.known_screen
    clamp = getattr(self._job_context, "back_wait_clamp", None)
    if clamp is not None:
        current = self._say_the_wait_was_shortened(current, clamp[0], clamp[1])
    return current


def open_link(
    self: Engine,
    uri: str,
    *,
    package: str | None = None,
    prefer: str | None = None,
    pin_package: bool = True,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    """Open a deeplink URI (jump straight to a screen / trigger an app action).

        By default pins the VIEW intent to the foreground/known package so Android's
        "Open with…" chooser never appears when both prod + dev builds are installed.
        Pass ``pin_package=False`` (CLI ``--no-package-pin``) to deliberately exercise
        the chooser. If a chooser still appears after open, raises :class:`DeviceError`
        naming the competing app rows — never leaves the caller stranded on the dialog.
        """
    target_pkg = package or prefer
    if pin_package and not target_pkg:
        target_pkg = self.current_package() or self._cached_package()
    step = self._step("open-link", arg=uri)
    with self._acting():
        self.device.open_link(uri, package=target_pkg if pin_package else None)
    time.sleep(0.35)  # chooser / activity settle
    detail = uri if not target_pkg else f"{uri} → {target_pkg}"
    if self._is_chooser():
        competitors = self._chooser_app_labels()
        if pin_package and target_pkg and self._dismiss_chooser(prefer=target_pkg):
            time.sleep(0.25)
        if self._is_chooser():
            listing = ", ".join(competitors) if competitors else "(unknown handlers)"
            raise DeviceError(
                "deeplink opened the system 'Open with…' chooser",
                hint=(
                    f"Competing apps on screen: {listing}. "
                    f"Re-run with `--package <id>` (e.g. the foreground "
                    f"`{self.current_package() or 'com.example.app'}`), or "
                    f"`--no-package-pin` only when you intentionally want the chooser."
                ),
            )
        detail = f"{uri} (chooser→{target_pkg or 'picked'})"
    self._record_action_safe(step)
    self._remember_deeplink_safe(uri, package=target_pkg)
    self._remember_pending_flag_context(uri, target_pkg)
    result = self._observe(
        ActionResult(ok=True, action="open-link", detail=detail), observe, with_image
    )
    self._flag_deeplink_that_did_not_land(result, uri)
    return result


def _flag_deeplink_that_did_not_land(self: Engine, result: ActionResult, uri: str) -> None:
    """Say so — structurally, not just in prose — whether the deeplink's arrival was
        confirmed, confirmed absent, or genuinely unknown.

        `am start` returning cleanly only means the intent was delivered — an app is free to
        ignore it, and several only honour a deeplink across a restart. The result still read
        `ok: true` with `detail: "<uri> → <package>"` and an all-zero `action_diff_summary`,
        which a caller checking only those two fields cannot tell apart from a jump that
        worked. Measured 2026-08-19: exactly that — `ok: true`, `action_diff_summary: {added:
        0, removed: 0, changed: 0}`, offline app, fresh id — was read as the target having been
        reached. `stale_risk` alone already said so in prose (measured 2026-08-10, a different
        incident), but a prose field nobody is told to check is not a contract.

        `verified` is the field built for exactly this ("True = confirmed effect, False =
        confirmed no effect, None = genuinely could not tell" — see its docstring), so it is
        set here rather than inventing a new one. `ok` is deliberately left alone in every
        branch: an unresolvable deeplink and one that legitimately leaves you exactly where you
        already were produce an IDENTICAL before/after diff — aua cannot and does not try to
        tell them apart — so the only way to "flag" the no-op case without crying wolf on the
        legitimate one is to report the fact (no confirmed arrival) without asserting which of
        the two it is. Only a hard failure signal (`ok: false`) would cry wolf here; an honest
        `verified: false` does not, because it never claims the deeplink was wrong to no-op.
        """
    change = result.change if isinstance(result.change, dict) else None
    if not change or change.get("activity_changed") is None:
        return  # no usable baseline — "could not tell" must stay untouched, not False
    if change.get("changed"):
        # A confirmed, real destination change: close the loop on the tri-state rather
        # than leaving a genuine landing indistinguishable from "never checked".
        result.verified = True
        return
    result.verified = False
    result.stale_risk = (
        f"the app accepted {uri} but did not move: same activity, identical tree — "
        "`verified: false`. The intent was delivered (`am start` succeeded); this is either "
        "the app ignoring it from the current state, or you were already on the target "
        "screen and there was nothing to navigate to — a before/after diff cannot tell "
        "those apart, so neither is asserted. Some deeplinks only apply across a restart — "
        "`aua app restart-and-analyze <pkg>` then re-open — otherwise navigate normally."
    )


def _remember_pending_flag_context(self: Engine, uri: str, package: str | None) -> None:
    """Recognize configured raw set-flags links so a later manual launch is scoped."""
    if not package or self._memory is None:
        return
    template = self.config.flags.templates.get(package)
    if not template or "{query}" not in template:
        return
    prefix = template.split("{query}", 1)[0]
    if not uri.startswith(prefix):
        return
    from urllib.parse import parse_qsl, urlsplit

    flags = dict(parse_qsl(urlsplit(uri).query))
    if flags:
        self._memory.set_pending_flags(self.device.serial, package, flags)


def _is_chooser(self: Engine) -> bool:
    """True when the system resolver / 'Open with…' UI is in the foreground."""
    device = self.device
    try:
        app = device.current_app() or {}
    except Exception:
        return False
    pkg = (app.get("package") or "").lower()
    activity = (app.get("activity") or "").lower()
    if (
        "resolver" in activity
        or "intentresolver" in pkg
        or pkg in {"android", "com.android.intentresolver", "com.android.internal.app"}
    ):
        return True
    with contextlib.suppress(Exception):
        xml = self.platform.dump_tree(device)
        if "Open with" in xml or ("Just once" in xml and "Always" in xml):
            return True
    return False


def _chooser_app_labels(self: Engine) -> list[str]:
    """Clickable app-row labels on a chooser screen (best-effort)."""
    skip = {"Just once", "Always", "Open with", "Cancel", "Open"}
    labels: list[str] = []
    with contextlib.suppress(Exception):
        result = self.analyze(source="hierarchy", record=False)
        for el in result.elements:
            label = (el.text or el.content_desc or "").strip()
            if label and label not in skip and el.clickable:
                labels.append(label)
    return labels


def _dismiss_chooser(self: Engine, *, prefer: str | None = None) -> bool:
    """If the system 'Open with…' resolver is foreground, pick an app and continue."""
    if not self._is_chooser():
        return False
    device = self.device
    # Prefer an explicit package label match, else tap "Just once" on first row.
    try:
        result = self.analyze(source="hierarchy", record=False)
    except Exception:
        return False
    prefer_tail = (prefer or "").rsplit(".", 1)[-1].lower() if prefer else ""
    candidates = [
        el
        for el in result.elements
        if el.clickable
        and (
            (prefer_tail and prefer_tail in (el.text or el.content_desc or "").lower())
            or (el.text or "").strip() not in {"Just once", "Always", "Open with"}
        )
    ]
    target = None
    if prefer_tail:
        for el in candidates:
            hay = f"{el.text or ''} {el.content_desc or ''}".lower()
            if prefer_tail in hay or (prefer or "").lower() in hay:
                target = el
                break
    if target is None:
        # First non-chrome row that looks like an app
        for el in result.elements:
            label = (el.text or el.content_desc or "").strip()
            if (
                label
                and label not in {"Just once", "Always", "Open with", "Cancel"}
                and el.clickable
            ):
                target = el
                break
    if target is None:
        return False
    x, y = target.center
    device.click(x, y)
    # Confirm "Just once" if still on chooser.
    time.sleep(0.3)
    with contextlib.suppress(Exception):
        again = self.analyze(source="hierarchy", record=False)
        for el in again.elements:
            if (el.text or "").strip() == "Just once" and el.clickable:
                device.click(*el.center)
                break
    # Cache only — deliberately NOT `_acting()`. This runs mid-`open_link`, whose window
    # is already open; re-stamping here would move it past the deeplink's own output.
    self._invalidate_cache()
    return True


def _remember_deeplink_safe(self: Engine, uri: str, *, package: str | None = None) -> None:
    mem = self._memory
    if mem is None or self._device is None:
        return
    pkg = package or self._cached_package() or self.current_package()
    if not pkg:
        return
    with contextlib.suppress(Exception):  # playbook is a bonus; never fail the action
        mem.remember_deeplink(pkg, uri, probed=True)
