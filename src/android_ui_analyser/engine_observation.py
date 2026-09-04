"""The post-action `observation` every action returns: the shared _observe pipeline, the loading/readiness predicate and settle waits, arrival and stale-risk verdicts, the before/after change summary, and crash and app-log evidence.

Engine methods for observation. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_observation.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import re
import time
from typing import TYPE_CHECKING, Any

from .engine_support import _label, logger
from .errors import AuaError
from .memory import _id_tail
from .schema import ActionResult, AnalyzeResult
from .selectors import app_elements

if TYPE_CHECKING:
    from .engine import Engine


# A hierarchy dump quicker than this can outrun the screen it is reading, so a post-action
# sample may catch a half-attached tree; a slower one cannot (the render has finished by the
# time it returns). Measured ~150ms headless vs 600-1200ms windowed on the same emulator.
_FAST_DUMP_MS = 250.0


_CHANGE_TEXT_CAP = 12  # text deltas echoed back per direction; a list screen would dump hundreds


_CRASH_LOG_SCAN_LINES = 600  # bounded read from one already-short last-action window


_CRASH_EVIDENCE_LINES = 60  # enough for exception + causes without dumping the full device log


def _compact_action_diff(element_diff: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep inline diffs token-cheap and machine-readable."""
    if not isinstance(element_diff, dict):
        return None
    added = element_diff.get("added", [])
    removed = element_diff.get("removed", [])
    changed = element_diff.get("changed", [])
    out: dict[str, Any] = {
        "added": len(added) if isinstance(added, list) else added,
        "removed": len(removed) if isinstance(removed, list) else removed,
        "changed": len(changed) if isinstance(changed, list) else changed,
    }
    if "prev_count" in element_diff:
        out["prev_count"] = element_diff["prev_count"]
    if "curr_count" in element_diff:
        out["curr_count"] = element_diff["curr_count"]
    if element_diff.get("unchanged") is not None:
        out["unchanged"] = bool(element_diff["unchanged"])
    return out


def _analyze_post_action(
    self: Engine,
    with_image: bool | str | None,
    *,
    record_screen: bool = False,
) -> AnalyzeResult:
    """Read the post-action screen, escalating thin trees exactly as ``analyze`` would."""
    obs = self.analyze(
        source="hierarchy",
        record=record_screen,
        with_image=self._effective_with_image(with_image),
    )
    # Hierarchy first because it is tens of milliseconds and answers for most screens.
    # But pinning the folded observation to hierarchy made every Compose/canvas/WebView
    # caller pay for a second explicit analyze. Escalate once through the normal gate.
    if self.config.perception.observe_escalates_to_vision:
        with contextlib.suppress(Exception):
            decision = self._gate_decide(
                obs.elements,
                package=obs.screen.package,
                activity=obs.screen.activity,
            )
            if decision.use_vision:
                richer = self.analyze(
                    source="auto",
                    record=record_screen,
                    with_image=self._effective_with_image(with_image),
                )
                # Keep the hierarchy answer unless escalation actually found more.
                if len(richer.elements) > len(obs.elements):
                    obs = richer
    return obs


def _change_has_semantic_effect(change: dict[str, Any] | None) -> bool:
    """Whether the readback names an effect beyond node/layout churn."""
    if not change:
        return False
    return bool(
        change.get("activity_changed") is True
        or change.get("focus_moved") is True
        or change.get("text_added")
        or change.get("text_removed")
    )


def _unready_destination_risk(
    self: Engine,
    change: dict[str, Any] | None,
    obs: AnalyzeResult,
    *,
    before_state: dict[str, Any] | None,
    destination_confirmed: bool,
) -> tuple[str, str] | None:
    """``(detector name, why)`` for a loading/unrendered destination, or ``None``.

        The journalled failure this exists for: a tap started a new Activity that then waited
        on the network before drawing. The settle wait truthfully reported quiet pixels and a
        stable tree (``via=hierarchy``), the change summary said the Activity moved with
        nothing added — and the result still cleared every caveat. The screen was *physically
        settled and semantically loading*, so no settle-loop tuning can catch it; only
        classification can. This is Phase 0 of the arrival design: classification only, from
        evidence already in hand — no new waits, no new device reads, the same facts finally
        consulted.

        Three detectors, most explicit first:

        * the screen *says* it is loading — a progress widget, loading text, or a mapped
          ``loading`` screen (:meth:`_observation_is_loading`, until now consulted only by
          goto/back/await). Explicit beats inference, so this fires even on a recognized or
          additive destination: a loading shell that arrived is still a loading shell.
        * departure without arrival — the Activity changed while nothing was added: no new
          text, a tree that did not grow, and no new actionable control. Removal-only change
          proves the old screen was *left*, not that the new one has rendered
          (``_change_has_semantic_effect`` treats those symmetrically, which is exactly how
          the journalled frame passed for arrival). Suppressed when recognition confirmed a
          different known screen — the map is stronger arrival evidence than this inference —
          and when the app left the foreground, which is its own, stronger report.
        * the same departure without the Activity evidence — a strong subtractive transition
          landing on a content-bare tree. Single-Activity (Compose) apps never change
          Activity for in-app navigation, which measured live made the detector above inert
          for the dominant modern app shape.

        Additive arrival is measured on the app's own elements, never on
        ``change.text_added``: that field counts every window, and the status-bar clock
        ticking over vetoed the verdict live on exactly the frame that needed it.
        """
    if self._observation_is_loading(obs):
        return (
            "loading_indicator",
            "the post-action screen shows an explicit loading state (progress indicator, "
            "loading text, or a mapped loading screen). Content may replace these ids when "
            "it lands — run `aua wait-and-analyze --after-change` or wait for an exact "
            "destination predicate instead of acting on this frame.",
        )
    if destination_confirmed and (before_state or {}).get("known_screen"):
        # Recognition suppresses the inference only when the *origin* was also known:
        # `destination_confirmed` is "recognized name differs from the before name", and
        # with an unstamped before name (cold session, async memory) it is True whenever
        # anything is recognized at all. Under this detector's own preconditions the after
        # tree barely differs from the before tree, so that recognition is at least as
        # likely to be the origin's map entry as a reached destination — no differential,
        # no suppression.
        return None
    if not isinstance(change, dict):
        return None
    if change.get("app_left_foreground"):
        return None
    # Additive arrival, measured on the app's own elements only. `change.text_added`
    # counts every window, and measured live the status-bar clock ticking over ("8:15")
    # and the Wi-Fi icon returning counted as "content arrived" — vetoing the verdict on
    # exactly the frame that needed it. A new app label or a new actionable app control
    # is arrival evidence; system chrome is not.
    before_labels = set((before_state or {}).get("labels") or [])
    before_rids = set((before_state or {}).get("rids") or [])
    # Bareness outranks additive chrome: a loading shell's own nav-back button is a NEW
    # actionable control, and measured live it counted as "content arrived" on a screen
    # with nothing readable on it. Only a non-bare tree gets to prove arrival additively.
    bare = self._content_bare(obs)
    if not bare and self._destination_rendered(obs, before_labels, before_rids):
        return None
    delta = change.get("node_count_delta")
    if change.get("activity_changed") is True:
        if not isinstance(delta, int) or delta > 0:
            return None
        return (
            "departure_without_arrival",
            "the action left the previous screen (the Activity changed) but the new one "
            "has rendered nothing yet: no text or actionable control was added and the "
            "tree did not grow. This observation is a transitional/loading frame, not "
            "arrival evidence — run `aua wait-and-analyze --after-change` or wait for an "
            "exact destination predicate before acting on ids from it.",
        )
    # Single-Activity (Compose) apps navigate without an Activity change — the dominant
    # modern shape, and the one the recorded live miss came from: a strong subtractive
    # transition that lands on a tree with no readable app content is the same departure,
    # minus the Activity evidence. The bareness test keeps a thin-but-labelled screen
    # (an empty-state message, an image viewer with OCR-legible text) out of it.
    removed = change.get("text_removed") or []
    strongly_subtractive = len(removed) >= 3 or (isinstance(delta, int) and delta <= -8)
    if strongly_subtractive and bare:
        return (
            "shell_only_tree",
            "the action left the previous screen, but the destination shows no readable "
            "app content yet — only bare or unlabelled containers/controls. It may still "
            "be loading (single-Activity apps navigate without an Activity change), or it "
            "may be visual-only content the tree cannot describe. Run `aua "
            "wait-and-analyze --after-change`, or `aua analyze --source vision` if this "
            "screen is an image/canvas.",
        )
    return None


def _content_bare(obs: AnalyzeResult) -> bool:
    """No readable app content and at most one (unlabelled) affordance.

        The launch-shell test refuses *any* clickable node; a navigated-to loading shell
        usually keeps its one nav-back affordance, so a single unlabelled clickable is
        allowed here. OCR-augmented reads put legible pixels into ``elements`` too, so a
        canvas screen with readable text does not count as bare.
        """
    affordances = 0
    for element in app_elements(obs.elements):
        if element.window in {"system", "ime", "overlay"}:
            continue
        label = (element.text or "").strip() or (element.content_desc or "").strip()
        if label and _readable_label(label):
            return False
        if (
            element.clickable is True
            or element.checkable is True
            or element.long_clickable is True
            or bool(element.focused)
        ):
            affordances += 1
            if affordances > 1:
                return False
    return True


def _destination_confirmed(known_after: str | None, before_known: str | None) -> bool:
    """Recognition is arrival evidence only with a known origin to differ from.

        The weak form — "recognised name differs from the before name" — is True on a cold
        session (async memory, first action after the first analyze) whenever *anything* is
        recognised, including the origin's own map entry, because the before name is still
        unstamped. That let recognition clear stale caveats and suppress the unready verdict
        on exactly the frames that need them. No known origin, no differential, no claim.
        """
    return bool(known_after and before_known and known_after != before_known)


def _post_action_change(
    self: Engine,
    obs: AnalyzeResult,
    before_state: dict[str, Any] | None,
    *,
    hierarchy_only: bool,
    ready: dict[str, Any] | None,
    confirmed_stable: bool,
) -> tuple[dict[str, Any] | None, bool]:
    """Change summary + passive recognition for one post-action readback.

        One method because the content wait can adopt a later, better readback, and the
        second readback must be summarised and recognised exactly like the first — the
        launch content poll grew a subtle drift bug from duplicating this by hand.

        Post-action analyzes deliberately do not write memory: their frame may still be
        transitional. Recognition against the existing map is safe, and is strong
        destination evidence — evaluated before stale risk so a looping animation cannot
        make a correctly recognised, semantically different destination look unsafe merely
        because its extended quiet-window timed out.
        """
    change: dict[str, Any] | None = None
    if not hierarchy_only:
        with contextlib.suppress(Exception):
            change = self._change_summary(before_state, obs)
    if ready is not None and (confirmed_stable or ready.get("confirmation_timeout")):
        ready["semantic_confirmation"] = self._change_has_semantic_effect(change)
    mem = self._memory
    if mem is not None and self._device is not None:
        with contextlib.suppress(Exception):
            known = mem.observe_screen_passive(
                self._device.serial,
                package=obs.screen.package,
                elements=obs.elements,
                activity=obs.screen.activity,
                screen_height=obs.screen.height,
            )
            if known:
                obs.meta.known_screen = known
    destination_confirmed = self._destination_confirmed(
        obs.meta.known_screen, (before_state or {}).get("known_screen")
    )
    return change, destination_confirmed


def _await_rendered_destination(
    self: Engine,
    initial: AnalyzeResult,
    before_state: dict[str, Any] | None,
    *,
    budget_ms: int,
    poll_ms: int = 120,
) -> AnalyzeResult | None:
    """Hold the call until the unready destination renders, bounded; None on expiry.

        The launch content poll, generalised to actions: instead of handing back a frame the
        classifier just called unready — sending the caller into a blind wait plus a
        re-analyze, or worse, a tap on a control that no longer exists — spend a bounded
        budget re-reading the cheap hierarchy until content that provably was not there
        arrives. Only ever entered when the answer in hand is already wrong, so a settled
        action pays nothing.

        The polls are internal freshness reads (``record_ids=False``): none of them is
        published, so a caller's ids always match the observation it was actually handed.
        The accepted candidate is only a trigger — the caller-facing re-read happens in
        ``_observe`` through the same ``_analyze_post_action`` path as the first readback.
        """
    package = initial.screen.package
    if not package or budget_ms <= 0:
        return None
    before_labels = set((before_state or {}).get("labels") or [])
    before_rids = set((before_state or {}).get("rids") or [])
    started = time.monotonic()
    deadline = started + budget_ms / 1000.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        self._job_sleep(min(poll_ms / 1000.0, remaining))
        with contextlib.suppress(Exception):
            candidate = self.analyze(
                source="hierarchy",
                with_ocr=False,
                no_cache=True,
                record=False,
                record_ids=False,
                with_image=False,
            )
            # Never adopt a transition owned by SystemUI or another app, and never call
            # a still-loading frame rendered — a spinner over partial content keeps the
            # wait alive until the budget rules.
            if candidate.screen.package != package:
                continue
            if self._observation_is_loading(candidate):
                continue
            if self._content_bare(candidate):
                # A shell's own chrome (a new nav-back control) is not arrival — the
                # additive test below would accept it on the first poll and hand the
                # wait's whole budget to a frame with nothing readable on it.
                continue
            if self._destination_rendered(candidate, before_labels, before_rids):
                return candidate
    return None


def _readable_label(label: str) -> bool:
    """Whether *label* is readable content, not decoration.

        Measured live: a loading shell's page-indicator dot ("•") was the destination's only
        text, and it counted both as content (not bare) and as additive arrival. Punctuation
        and dingbats are decoration; one alphanumeric character in any script is the bar.
        """
    return any(ch.isalnum() for ch in label)


def _destination_rendered(
    candidate: AnalyzeResult,
    before_labels: set[str],
    before_rids: set[str],
) -> bool:
    """Additive evidence relative to the pre-action snapshot: new text or a new control."""
    for element in app_elements(candidate.elements):
        label = _label((element.text or element.content_desc or "").strip())
        if label and label not in before_labels and _readable_label(label):
            return True
        if element.clickable is True:
            rid = _id_tail(element.resource_id)
            if rid and rid not in before_rids:
                return True
    return False


def _arrival_report(
    *,
    settle: bool,
    hierarchy_only: bool,
    ready: dict[str, Any] | None,
    destination_confirmed: bool,
    semantic_change_confirmed: bool,
    unready: tuple[str, str] | None,
    entry_kind: str | None,
    content_wait_ms: int,
    content_arrived: bool,
    launch_transitional: bool,
) -> dict[str, Any] | None:
    """The machine-readable arrival verdict, or ``None`` for a silently settled action.

        Mirrors :meth:`_stale_observation_risk`'s branch order — the prose caveat and this
        state must never disagree, and that method's docstring owns the reasoning for each
        branch. Emitted only when there is something to say (a non-settled state, or a
        settled one that had to be waited for): like ``screen_moved`` and ``stale_risk``,
        absence is the cheap, healthy answer. Named evidence, never a confidence score.
        """
    if not settle or hierarchy_only:
        return None

    def report(state: str, evidence: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"state": state, "evidence": evidence}
        if content_wait_ms:
            out["waited_ms"] = content_wait_ms
        return out

    if launch_transitional:
        return report("unconfirmed", ["shell_only_tree"])
    if ready is None:
        return report("unconfirmed", ["no_settle_wait"])
    if unready is not None:
        evidence = [unready[0]]
        if content_wait_ms:
            evidence.append("content_wait_expired")
        state = "loading" if unready[0] == "loading_indicator" else "unconfirmed"
        return report(state, evidence)
    if ready.get("confirmation_timeout"):
        if destination_confirmed and ready.get("semantic_confirmation") is True:
            pass  # recognition + semantic delta outrank a looping animation
        else:
            return report("transitioning", ["confirmation_timeout"])
    if ready.get("semantic_confirmation") is False:
        return report("unconfirmed", ["visual_only_change"])
    if ready.get("timeout") and not semantic_change_confirmed:
        return report("transitioning", ["settle_timeout"])
    if not ready.get("changed"):
        return report("no_change", ["no_confirmed_change"])
    if content_arrived:
        evidence = [entry_kind] if entry_kind else []
        evidence.append("content_arrived")
        return report("settled", evidence)
    return None


def _tap_settle_needs_confirmation(
    action_kind: str | None, ready: dict[str, Any] | None
) -> bool:
    """Whether an early tap settle needs a longer quiet window before analysis.

        The synthetic transition fixtures defeat content heuristics in three different ways: an
        old screen plus one pager node, OCR destination text over the old hierarchy, and a mixed hierarchy with
        both old and new screens. Therefore no single early frame certifies arrival. Confirm every
        tap-like fast hierarchy/pixel settle; slower double-sampled hierarchy settles keep their
        existing path — but only when the tree *grew*.
        """
    if action_kind not in {"tap", "tap-point", "double-tap", "long-press"}:
        return False
    if not ready or ready.get("timeout") or not ready.get("changed"):
        return False
    if ready.get("via") in {"hierarchy-fast", "pixels"}:
        return True
    # Double-sampling earns its exemption only by growth. Two agreeing dumps of a tree
    # that only LOST parts are the recorded transitional shape: uiautomator serving the
    # old window frozen minus one label while the new Activity waits on the network.
    # An absent count is a legacy/monkeypatched settle — keep its old exemption rather
    # than inventing evidence it never measured.
    return ready.get("via") == "hierarchy" and ready.get("tree_added") == 0


def _observe(
    self: Engine,
    result: ActionResult,
    observe: bool,
    with_image: bool | str | None = None,
    *,
    settle: bool = True,
    record_screen: bool = False,
    hierarchy_only: bool = False,
    adopt_action: bool = False,
    finalize: bool = True,
) -> ActionResult:
    """Attach the post-action screen so callers skip a separate ``analyze`` round-trip.

        The folded ``analyze`` also re-populates the id cache, so the agent can act on an id
        from ``result.observation`` immediately (e.g. type → tap send) in one fewer call.

        When ``settle`` is True (default for actions), wait until pixels differ from the
        pre-action frame and the non-animated region is idle — otherwise agents get the
        previous screen and burn a second ``wait --for-stable`` + re-analyze.
        """
    if observe:
        with contextlib.suppress(Exception):  # observation is a bonus; never fail the action
            # Read before the settle consumes the pre-action bookkeeping.
            if adopt_action:
                before_state = self._action_observation_baseline
                self._action_observation_baseline = None
            else:
                before_state = self._pre_action_state
                self._pre_action_state = None
                if before_state is not None:
                    self._action_observation_baseline = before_state
            ready: dict[str, Any] | None = None
            confirmed_stable = False
            # A deliberate pause before anything is read. The poll loop's 45ms quiet
            # window cannot tell a splash that has gone quiet while loading from a screen
            # that is finished, and in the field it returned `shown=0` as a settled
            # result. This is the knob that trade-off is tuned on; see
            # `perf.stable_delay_ms`.
            spent_delay = self._spend_stable_delay()
            if spent_delay:
                result.stable_delay_ms = spent_delay
            if settle:
                settle_ms, total_ms = 45, 1100
                if self.config.perf.settle_profiles and self._last_action_kind:
                    settle_ms, total_ms = self._settle_profiles.budget(
                        self._last_action_kind,
                        total_max_ms=self.config.perf.settle_total_max_ms,
                    )
                # A control with its own history overrides the per-kind guess. This is the
                # only path that can exceed `settle_total_max_ms`: that ceiling exists to stop
                # a *blind* timer taxing every same-screen tap, and it is the wrong instrument
                # for a control we have actually measured at 18s.
                learned = self._learned_action_budget(total_ms)
                if learned is not None:
                    total_ms = learned
                ready = self._await_post_action_ready(
                    settle_ms=settle_ms, total_timeout_ms=total_ms
                )
                # Only learn from real transitions — same-screen / timeouts poison the EMA
                # and made subsequent taps ~2× slower (450→900ms) in the field.
                if (
                    ready
                    and self.config.perf.settle_profiles
                    and self._last_action_kind
                    and ready.get("changed")
                    and ready.get("via") in {"hierarchy-fast", "hierarchy", "pixels"}
                    and ready.get("ms") is not None
                ):
                    self._settle_profiles.observe(
                        self._last_action_kind,
                        min(float(ready["ms"]), self.config.perf.settle_learn_cap_ms),
                    )
                # Persist the per-site cost regardless of whether the EMA accepted it: the
                # coarse profile refuses timeouts because they poison an app-wide average,
                # but "this control timed out" is precisely what a future run needs told.
                if ready is not None and ready.get("ms") is not None:
                    self._record_action_timing_safe(
                        float(ready["ms"]),
                        outcome=("changed" if ready.get("changed") else "unchanged"),
                    )
            if self._tap_settle_needs_confirmation(self._last_action_kind, ready):
                confirm_t0 = time.monotonic()
                try:
                    # A 350ms quiet window outlives a ripple and a single Compose pager
                    # frame, while the 1.4s ceiling keeps a looping surface bounded.
                    self.wait_stable(
                        interval_ms=80,
                        settle_ms=350,
                        timeout_ms=1400,
                        observe=False,
                    )
                    confirmed_stable = True
                except Exception:  # noqa: BLE001 — observation remains best-effort
                    if ready is not None:
                        ready["confirmation_timeout"] = True
                if ready is not None:
                    ready["confirmation_ms"] = int((time.monotonic() - confirm_t0) * 1000)

            # Analyze only after the confirmation. Besides being safer, this avoids paying
            # for and returning an OCR-enriched read of a frame we already distrust.
            if hierarchy_only:
                # Explicitly frame-free, not merely defaulted so. This branch is the cheap
                # read for *intermediate* navigation, and it can run once per poll — letting
                # it inherit the session `with_image` default would put a screenshot on the
                # inside of a wait loop, which is exactly the visual work this path exists
                # to skip. An explicit per-call request still reaches the returned
                # observation through the non-hierarchy-only branch.
                obs = self.analyze(
                    source="hierarchy",
                    with_ocr=False,
                    record=record_screen,
                    with_image=False,
                )
            else:
                obs = self._analyze_post_action(with_image, record_screen=record_screen)
            launch_content_wait_ms = 0
            if (
                result.action == "app-launch"
                and obs.screen.package
                and self._launch_observation_is_transitional(obs)
            ):
                obs, launch_content_wait_ms = self._await_meaningful_launch_observation(obs)
            change, destination_confirmed = self._post_action_change(
                obs,
                before_state,
                hierarchy_only=hierarchy_only,
                ready=ready,
                confirmed_stable=confirmed_stable,
            )
            caveat = self._stale_observation_risk(
                settle,
                ready,
                destination_confirmed=destination_confirmed,
                semantic_change_confirmed=self._change_has_semantic_effect(change),
            )
            launch_transitional = bool(
                result.action == "app-launch" and self._launch_observation_is_transitional(obs)
            )
            launch_content_ready = bool(
                result.action == "app-launch"
                and launch_content_wait_ms
                and not launch_transitional
            )
            if launch_transitional:
                caveat = (
                    "the app reached the foreground, but launch produced only framework shell "
                    "nodes and no meaningful app content. This observation is transitional, "
                    "not arrival evidence. Use `aua wait-and-analyze --after-change` or wait "
                    "for an exact destination predicate."
                )
            elif launch_content_ready:
                # The initial pixel settle may have called the framework shell unchanged.
                # The bounded package-owned poll subsequently found semantic app content,
                # which is newer and stronger evidence than that early frame.
                caveat = None
            arrival_unready: tuple[str, str] | None = None
            if caveat is None and settle and not hierarchy_only and not launch_transitional:
                # Only where the settle machinery cleared every caveat: an existing caveat
                # (unchanged / timeout / unconfirmed) is already honest, and the repeat-
                # mutation warning on `unchanged` must never be replaced by a softer one.
                # Its own suppress: the surrounding blanket suppress would otherwise let a
                # classifier bug silently discard the whole observation attachment.
                with contextlib.suppress(Exception):
                    arrival_unready = self._unready_destination_risk(
                        change,
                        obs,
                        before_state=before_state,
                        destination_confirmed=destination_confirmed,
                    )
                if arrival_unready:
                    caveat = arrival_unready[1]
            # ---- the arrival extension: hold the call while the wrong answer fixes itself.
            # Entered ONLY when the classifier said the frame is unready — a settled action
            # never reaches this line, which is what keeps the hot path at +0ms. Launch has
            # its own (longer) content poll above; an adopted-baseline observe belongs to an
            # await that manages its own evidence wait. The budget shares the extended
            # confirmation's ceiling: whatever the confirmation already spent comes off it,
            # so the two phases can never stack past max(confirmation_cap, extension).
            content_wait_ms = 0
            content_arrived = False
            entry_kind = arrival_unready[0] if arrival_unready else None
            if (
                arrival_unready
                and result.action != "app-launch"
                and not adopt_action
            ):
                from .perf import arrival_extension_for

                budget = arrival_extension_for(self.config) - int(
                    (ready or {}).get("confirmation_ms") or 0
                )
                if budget > 0:
                    wait_t0 = time.monotonic()
                    rendered = self._await_rendered_destination(
                        obs, before_state, budget_ms=budget
                    )
                    if rendered is not None:
                        # One caller-facing re-read through the same path as the first
                        # readback (image, vision gate, published ids), then re-derive the
                        # verdict from scratch — the wait proved nothing by itself.
                        obs = self._analyze_post_action(
                            with_image, record_screen=record_screen
                        )
                        change, destination_confirmed = self._post_action_change(
                            obs,
                            before_state,
                            hierarchy_only=hierarchy_only,
                            ready=ready,
                            confirmed_stable=confirmed_stable,
                        )
                        caveat = self._stale_observation_risk(
                            settle,
                            ready,
                            destination_confirmed=destination_confirmed,
                            semantic_change_confirmed=self._change_has_semantic_effect(
                                change
                            ),
                        )
                        arrival_unready = None
                        if caveat is None:
                            with contextlib.suppress(Exception):
                                arrival_unready = self._unready_destination_risk(
                                    change,
                                    obs,
                                    before_state=before_state,
                                    destination_confirmed=destination_confirmed,
                                )
                            if arrival_unready:
                                caveat = arrival_unready[1]
                        content_arrived = caveat is None
                    content_wait_ms = int((time.monotonic() - wait_t0) * 1000)
                    if arrival_unready and content_wait_ms:
                        # The prose must say the wait already happened, or the caller's
                        # obvious next move is the very wait that just expired.
                        caveat = (
                            f"{arrival_unready[1]} AUA already waited {content_wait_ms}ms "
                            "beyond the settle for content; none arrived."
                        )
            result.observation = obs
            result.change = change
            if caveat:
                obs.meta.stale_risk = caveat
                # Also at the top level of the action result, because a runner reading only the
                # terse form must not have to know the caveat exists to find it. It gets its
                # own field rather than being appended to `detail`: `detail` carries a
                # *semantic value* for several actions — `app launch` puts the launched
                # package/activity there — and appending a marker to it corrupts the thing a
                # caller parses. The caveat text, not a bare flag, so the reason travels too.
                result.stale_risk = caveat
            # The launch readback cannot be stood behind: its ids may be gone before the
            # caller's next command reaches them.
            launch_ids_unstable = bool(
                result.action == "app-launch"
                and not launch_content_ready
                and (
                    launch_transitional
                    or ready is None
                    or ready.get("timeout")
                    or not ready.get("changed")
                    # hierarchy-fast proves departure from the old tree with one sample. It
                    # is enough for an observation, but not for advertising numeric ids that
                    # may disappear before the next command reaches them.
                    or ready.get("via") not in {"pixels", "hierarchy"}
                )
            )
            if ready and ready.get("ms") is not None and ready.get("via") != "unchanged":
                # Surface settle cost so agents/tests can see why a tap took >50 ms — in its own
                # field. This used to be appended to `detail` as a "settle=295ms via=pixels"
                # tag, which corrupts the semantic value `detail` carries for some actions:
                # `app launch` puts the launched package/activity there, so an observed launch
                # answered `detail: "<pkg>/<activity> settle=295ms via=pixels"` and a caller
                # parsing it got the timing glued onto the component name. Structured here, so
                # it can be read as a number rather than scraped out of prose.
                # Named apart from the `settle: bool` parameter this used to shadow. Nothing
                # read the flag again after the rebind, so the behaviour was right, but any
                # later `if settle:` would have tested a dict that is always truthy.
                settle_report: dict[str, Any] = {"ms": ready["ms"]}
                if ready.get("via"):
                    settle_report["via"] = ready["via"]
                if ready.get("masked"):
                    settle_report["anim"] = ready["masked"]
                if ready.get("confirmation_ms") is not None:
                    settle_report["confirmation_ms"] = ready["confirmation_ms"]
                if ready.get("semantic_confirmation") is not None:
                    settle_report["semantic_confirmation"] = ready["semantic_confirmation"]
                if launch_content_wait_ms or content_wait_ms:
                    settle_report["content_ms"] = launch_content_wait_ms or content_wait_ms
                result.settle = settle_report
            elif launch_content_wait_ms or content_wait_ms:
                result.settle = {"content_ms": launch_content_wait_ms or content_wait_ms}
            result.observation_present = True
            # The learned cost belongs on the control it describes, and this is where it
            # reaches the caller: `meta.slow_controls` is not in the `changed` preset a
            # folded observation is trimmed to.
            self._price_elements(obs)
            # Three independent reasons to withhold the derived list, and the third is the
            # honesty one: an unready destination's ids may vanish when content finally
            # lands, so advertising next actions from it invites exactly the wrong tap the
            # caveat exists to prevent. That suppression is best-effort — the list is now
            # off by default anyway, so `stale_risk` and the note carry the verdict.
            result.next_actions = (
                self._next_actions(obs)
                if (
                    self.config.output.next_actions
                    and not launch_ids_unstable
                    and not arrival_unready
                )
                else None
            )
            nav = list(obs.meta.known_routes or []) + list(obs.meta.suggested_gotos or [])
            result.routes = nav or None
            result.known_screen = obs.meta.known_screen
            result.action_diff_summary = self._compact_action_diff(obs.meta.element_diff)
            arrival = self._arrival_report(
                settle=settle,
                hierarchy_only=hierarchy_only,
                ready=ready,
                destination_confirmed=destination_confirmed,
                semantic_change_confirmed=self._change_has_semantic_effect(change),
                unready=arrival_unready,
                entry_kind=entry_kind,
                content_wait_ms=content_wait_ms,
                content_arrived=content_arrived,
                launch_transitional=launch_transitional,
            )
            if arrival:
                result.arrival = arrival
                obs.meta.arrival_state = str(arrival["state"])
            if launch_transitional:
                result.note = (
                    "The app is foreground, but its launch readback contains only framework "
                    "shell nodes, so it is not a settled/reusable destination. Run `aua "
                    "wait-and-analyze --after-change` or wait for an exact destination "
                    "predicate; do not act on ids from this frame."
                )
            elif launch_ids_unstable:
                result.note = (
                    "The app is foreground, but its launch screen has not produced a stable "
                    "readback yet, so its ids may not survive until your next call. Run "
                    "`aua analyze` once before acting on an id."
                )
            elif arrival_unready:
                waited = (
                    f" AUA already held this call {content_wait_ms}ms waiting for it."
                    if content_wait_ms
                    else ""
                )
                result.note = (
                    "The action dispatched, but the destination has not rendered usable "
                    f"content yet (see stale_risk).{waited} Do not act on ids from this "
                    "frame; run `aua wait-and-analyze --after-change` or wait for an exact "
                    "destination predicate."
                )
            elif ready and ready.get("timeout") and self._change_has_semantic_effect(change):
                result.note = (
                    "Fresh hierarchy confirms the action changed the screen. Use this "
                    "observation; if an exact destination is still absent, run one exact "
                    "predicate wait instead of a predicate-less settle wait."
                )
            else:
                result.note = "No separate analyze needed; state is in observation."
            # The world arriving on its own is not about this action at all, so it is not in
            # the chain above — it is prepended to whatever that chain concluded. In the
            # note as well as in `meta` because a field a caller must remember to check is
            # weaker than a sentence in the place it is already reading, which is the whole
            # reason `stale_risk` says itself twice too. Free when it does not fire.
            moved = self._consume_screen_moved()
            if moved:
                obs.meta.screen_moved = moved
                result.note = f"WARNING: {moved} {result.note}"
            # Say it in the note too, not only in `change`: the screen this observation
            # describes belongs to a different app, so every id in it is a dead end for
            # whatever the caller was doing.
            # What the app said while it was doing this, before the crash branch below:
            # that branch reads the same window for a different, stronger purpose, and
            # reading it twice would pay for two dumps to say one thing.
            left = change.get("app_left_foreground") if isinstance(change, dict) else None
            if not left:
                with contextlib.suppress(Exception):
                    # Order matters. The app that was in front when the action was dispatched
                    # is the one that logged the response to it, so it wins over the screen we
                    # landed on. Both are ignored when they are the launcher or system UI, and
                    # the remembered app under test then carries a cold launch — the window
                    # that has no previous package at all and the most to say.
                    for candidate in (
                        (before_state or {}).get("package"),
                        obs.screen.package,
                    ):
                        self._note_app_under_test(candidate)
                    scope = next(
                        (
                            pkg
                            for pkg in (
                                (before_state or {}).get("package"),
                                obs.screen.package,
                                self._app_under_test,
                            )
                            if self._could_be_app_under_test(pkg)
                        ),
                        None,
                    )
                    if scope:
                        result.app_logs = self._app_logs(str(scope))
            if left:
                result.crash_evidence = self._crash_evidence(str(left["from"]))
                dialog = (
                    " A system crash dialog is on screen." if left.get("crash_dialog") else ""
                )
                evidence = result.crash_evidence
                if evidence.get("available") and evidence.get("count"):
                    log_note = "The crash/error log block is attached in `crash_evidence`."
                elif evidence.get("available"):
                    log_note = (
                        "AUA checked the action's diagnostic-log window but found no "
                        "fatal, ANR, or error-priority lines; the checked window is in "
                        "`crash_evidence`."
                    )
                else:
                    log_note = (
                        "AUA could not read this platform's diagnostic logs; the structured "
                        "reason is in `crash_evidence`."
                    )
                result.note = (
                    f"WARNING: {left['from']} left the foreground — {left['to']} is in front "
                    f"now, so this observation is NOT your app.{dialog} {log_note} Then "
                    "relaunch with `aua app restart-and-analyze "
                    f"{left['from']}` instead of navigating this screen. {result.note}"
                )
    else:
        self._pre_action_sig = None
        if self.config.perf.prefetch:
            self._kick_hierarchy_prefetch()
        result.observation_present = False
    if self._frame_history_matters(result):
        hint = self._capture_hint()
        if hint:
            result.capture_hint = hint
    if finalize:
        result = self._finalize_observed_action(result)
    return result


def _frame_history_matters(result: ActionResult) -> bool:
    """Is there anything about this response the rolling frame buffer could explain?

        `capture_hint` names the one artefact that shows *what happened in between* — an
        interstitial sliding in, a screen replaced twice — and it was trimmed out of the
        observation because on a settled, successful action it answered a question nobody had
        asked. Three verdicts do ask it: the action failed, the settle never confirmed the
        screen had moved (`stale_risk` / `settled_unmet`), or the screen came back empty. A
        healthy arrival raises none of them and pays nothing.
        """
    observation = result.observation
    return bool(
        not result.ok
        or result.stale_risk
        or result.settled_unmet
        or (observation is not None and not observation.elements)
    )


def _finalize_observed_action(self: Engine, result: ActionResult) -> ActionResult:
    """Attach final timing/emptiness and journal the response the caller will receive."""
    result = self._note_empty_observation(result)
    if result.wall_ms is None:
        result.wall_ms = self._wall_ms()
    self._journal_call_answer(result)
    return result


def _stale_observation_risk(
    settle: bool,
    ready: dict[str, Any] | None,
    *,
    destination_confirmed: bool = False,
    semantic_change_confirmed: bool = False,
) -> str | None:
    """Why this post-action observation may describe the screen as it was *before* the action.

        Observed: a `tap` succeeded and the device advanced, yet the returned observation reported
        an empty `element_diff` with `unchanged=true` — measured against a snapshot taken before
        the screen changed. A screenshot plus a fresh `analyze` showed the app had in fact moved
        on.

        The mechanism is the settle wait giving up early. `_await_post_action_ready` returns
        `via=unchanged` after ~80ms of identical frames, and `via=hierarchy-same` on two matching
        trees; the folded `analyze` then dumps a tree that still matches the previous one, so
        `skip_unchanged_analyze` reuses the *previous* payload and stamps `unchanged=true`. The
        device advances a few milliseconds later. Nothing was wrong with any individual step —
        the claim is just older than it looks.

        This is the dangerous direction. The other ways a tap can look inert risk an agent giving
        up too early; this one risks it **repeating an action that already happened** — a second
        submit, a second message, a second purchase attempt. So the engine cannot report
        `unchanged` as a fact here: it genuinely cannot tell "no effect" from "not yet", and
        saying which one it is would be a guess presented as evidence.

        Deliberately conservative: only a *confirmed* transition (the wait saw the screen change
        and then stop, without timing out) clears the caveat. A real in-screen no-op therefore
        carries it too — correct, because the engine cannot distinguish that case either, and the
        expensive mistake is the other direction.
        """
    if not settle:
        # No wait was performed, so there is nothing to be stale relative to; the caller
        # asked for a raw read.
        return None
    if ready is None:
        return (
            "post-action wait did not run, so `unchanged` / `element_diff` may describe the "
            "pre-action screen. Re-analyze before concluding the action had no effect."
        )
    via = str(ready.get("via") or "?")
    if (
        ready.get("confirmation_timeout")
        and destination_confirmed
        and ready.get("semantic_confirmation") is True
    ):
        # The stability wait answers whether every pixel stopped moving. Recognition plus a
        # semantic before/after delta answers the question agents actually need: whether the
        # action reached a different known destination. Persistent animation must not turn
        # that stronger evidence into a false stale warning.
        return None
    if ready.get("confirmation_timeout"):
        return (
            "post-action read looked transitional and its extended stability confirmation "
            "timed out. The later observation is safer but may still be in flight — wait or "
            "re-analyze; never repeat a mutating action from this readback alone."
        )
    if ready.get("semantic_confirmation") is False:
        return (
            "post-action screen stabilized, but AUA observed no semantic destination beyond "
            "layout/node movement. The action may have a visual-only effect — do not repeat a "
            "mutating action from this readback alone."
        )
    if ready.get("timeout") and semantic_change_confirmed:
        # The visual settle budget expired, but the fresh hierarchy has different text,
        # focus, or Activity. It may still be rendering, yet it demonstrably does not
        # predate the action. Calling that stale made fresh agents start a predicate-less
        # 60s wait even when the requested content was already present.
        return None
    if ready.get("timeout"):
        return (
            f"post-action wait timed out (via={via}) — this observation may be mid-transition "
            "or predate the action. Re-analyze before concluding anything from it."
        )
    if not ready.get("changed"):
        return (
            f"post-action wait saw no confirmed screen change (via={via}), so `unchanged` and "
            "`element_diff` may be measured against a frame that predates the action. NOT "
            "evidence the action had no effect — re-analyze, and never retry a mutating "
            "action on `unchanged` alone."
        )
    return None


def _spend_stable_delay(self: Engine) -> int:
    """Sleep the configured post-action pause for the current action kind.

        Deliberately blunt: a fixed pause the operator can sweep, rather than another
        heuristic. Returns the milliseconds actually spent so a caller can attribute latency
        to this knob instead of guessing at it.
        """
    from .perf import stable_delay_for

    delay_ms = stable_delay_for(self._last_action_kind, self.config)
    if delay_ms <= 0:
        return 0
    self._job_sleep(delay_ms / 1000.0)
    return delay_ms


def _note_empty_observation(self: Engine, result: ActionResult) -> ActionResult:
    """Say so when the folded observation has nothing in it.

        An action that reports ``ok`` while returning a screen with no visible elements sends
        the caller away to wait for something that may already have arrived — which is how a
        5s launch turned into a 42s wait downstream. Naming it costs one field and lets the
        caller re-read instead of blocking.
        """
    obs = result.observation
    if obs is None:
        return result
    if obs.elements:
        return result
    result.observation_empty = True
    hint = (
        "the observation is empty — nothing was visible yet. Re-read with `analyze` "
        "rather than waiting for a change: the screen may already have arrived."
    )
    result.note = f"{result.note} {hint}".strip() if result.note else hint
    return result


def _await_post_action_ready(
    self: Engine,
    *,
    change_timeout_ms: int = 500,
    settle_ms: int = 45,
    total_timeout_ms: int = 1100,
    poll_ms: int = 28,
) -> dict[str, Any]:
    """Wait for post-action content change, then pixel-idle (animation-aware).

        Runs pixel settle and hierarchy double-sample in one loop so a Compose
        transition that updates the tree early can return before pixels fully idle.
        """
    from . import imaging

    device = self.device
    pre = self._pre_action_sig
    pre_tree = self._pre_action_tree_fp
    self._pre_action_sig = None
    self._pre_action_tree_fp = None
    t0 = time.monotonic()
    deadline = t0 + total_timeout_ms / 1000.0
    change_deadline = t0 + change_timeout_ms / 1000.0
    changed = pre is None
    gs = imaging.GridSettle(streak=imaging.ANIMATION_STREAK)
    stable_since: float | None = None
    last_tree: tuple[str, ...] | None = None
    next_hier_at = t0 + 0.04
    hier_checks = 0
    identical_polls = 0
    same_tree_hits = 0

    while time.monotonic() < deadline:
        try:
            # Fresh frames only — reusing a capture-buffer JPEG (~2 fps) falsely
            # reports idle / change and stretches same-screen taps.
            img = device.screenshot()
        except Exception:
            break
        now = time.monotonic()
        sig = imaging.frame_signature(img)
        if not changed and pre is not None:
            if imaging.frames_differ(pre, sig):
                changed = True
                identical_polls = 0
            else:
                identical_polls += 1
                # No pixel movement and no tree rewrite → action was a visual no-op
                # (or FakeDevice). Don't burn the full change_timeout.
                if identical_polls >= 3 and now - t0 >= 0.08:
                    return {
                        "changed": False,
                        "masked": 0,
                        "ms": int((now - t0) * 1000),
                        "timeout": False,
                        "via": "unchanged",
                    }
        if not changed and now > change_deadline:
            changed = True  # give up waiting for a pixel delta; settle what we have

        visually_idle = gs.feed(img)
        if visually_idle and changed:
            if stable_since is None:
                stable_since = now
            if (now - stable_since) * 1000.0 >= settle_ms:
                return {
                    "changed": changed,
                    "masked": len(gs.masked_cells),
                    "ms": int((now - t0) * 1000),
                    "timeout": False,
                    "via": "pixels",
                }
        else:
            stable_since = None

        if pre_tree is not None and hier_checks < 8 and now >= next_hier_at:
            hier_checks += 1
            next_hier_at = now + 0.06
            with contextlib.suppress(Exception):
                dump_started = time.monotonic()
                xml = self.platform.dump_tree(
                    device,
                    compact=bool(self.config.device.compressed_hierarchy),
                )
                dump_ms = (time.monotonic() - dump_started) * 1000.0
                w, h = device.window_size()
                els = self.platform.normalize_tree(xml, (w, h)).elements
                parts: list[str] = []
                for e in els:
                    if getattr(e, "window", None) == "system":
                        continue
                    rid = (e.resource_id or "").split("/")[-1]
                    label = (e.text or e.content_desc or "")[:40]
                    if rid or label:
                        parts.append(f"{rid}:{label}")
                cur = tuple(parts[:60])
                if not cur:
                    pass
                elif cur == pre_tree:
                    # Same accessibility tree as pre-action — in-screen tap / ripple /
                    # selected-state. Element IDs are still valid; don't wait for pixels
                    # (GridSettle stays busy on animations and was the 2× regression).
                    same_tree_hits += 1
                    if same_tree_hits >= 2 or (same_tree_hits >= 1 and now - t0 >= 0.12):
                        return {
                            "changed": False,
                            "masked": len(gs.masked_cells),
                            "ms": int((time.monotonic() - t0) * 1000),
                            "timeout": False,
                            "via": "hierarchy-same",
                        }
                else:
                    same_tree_hits = 0
                    changed = True
                    s_cur, s_pre = set(cur), set(pre_tree)
                    delta = len(s_cur ^ s_pre)
                    union = max(1, len(s_cur | s_pre))
                    # A big delta is measured against the PRE-action tree, so it only says we
                    # LEFT the old screen — a header-only frame differs from where we came
                    # from maximally, and accepting it on sight handed the caller an
                    # observation with the list body missing (measured on a fast device: one
                    # run in four read zero rows off a screen that has five).
                    #
                    # "Arrived" is distinguished from "still painting" by the tree having
                    # stopped GROWING, which takes two samples to see. Whether that second
                    # sample is affordable depends entirely on the device: a hierarchy dump
                    # measured ~150ms headless but ~600-1200ms windowed, and on the slow one
                    # the render has always finished before the first dump even returns. So
                    # spend a confirming dump only when the remaining budget can absorb one —
                    # otherwise take what we have. Requiring it unconditionally turned 662ms
                    # actions into 1276ms ones with 9 of 12 hitting the deadline, which buys
                    # nothing on a device whose dumps are already slower than its rendering.
                    # A tree that is a small fraction of the one we left is the shape of a
                    # half-drawn screen — a header whose body has not been attached yet.
                    # That, not "differs from before", is what has to hold us.
                    #
                    # Only devices with FAST dumps can land in that state: a dump measured
                    # ~150ms headless but 600-1200ms windowed, and on the slow one the render
                    # has always finished before the first dump even returns (measured: rows
                    # present in the first sample every time, 807-1205ms after the tap). So
                    # the confirming sample is worth its cost exactly when dumps are cheap;
                    # spending it anywhere else buys nothing and cost +614ms per action.
                    thin = len(pre_tree) >= 6 and len(cur) * 2 < len(pre_tree)
                    settled = visually_idle or not thin or dump_ms > _FAST_DUMP_MS
                    # How much the tree GAINED relative to the pre-action fingerprint.
                    # Growth is what distinguishes "rendered something" from "left the old
                    # screen": a shrink-only double-sample is the recorded transitional
                    # shape and costs its confirmation exemption downstream.
                    tree_added = len(s_cur - s_pre)
                    if settled and delta >= max(4, union // 3):
                        return {
                            "changed": True,
                            "masked": len(gs.masked_cells),
                            "ms": int((time.monotonic() - t0) * 1000),
                            "timeout": False,
                            "tree_added": tree_added,
                            "via": "hierarchy-fast",
                        }
                    if settled and cur == last_tree:
                        return {
                            "changed": True,
                            "masked": len(gs.masked_cells),
                            "ms": int((time.monotonic() - t0) * 1000),
                            "timeout": False,
                            "tree_added": tree_added,
                            "via": "hierarchy",
                        }
                    last_tree = cur
        time.sleep(poll_ms / 1000.0)

    return {
        "changed": changed,
        "masked": len(gs.masked_cells),
        "ms": int((time.monotonic() - t0) * 1000),
        "timeout": True,
        "via": "timeout",
    }


def _observation_is_loading(self: Engine, observation: AnalyzeResult | None) -> bool:
    """Conservative signal that a wrong-screen verdict would be premature."""
    if observation is None:
        return False
    if self._mapped_screen_state(observation) == "loading":
        return True
    for element in observation.elements:
        kind = (element.type or "").casefold()
        if kind.endswith("progressbar"):
            return True
        label = " ".join(
            value for value in (element.text, element.content_desc) if value
        ).strip()
        if re.search(r"\b(?:loading|please wait)\b", label, re.IGNORECASE):
            return True
    return False


def _app_left_foreground(
    activity_before: str | None, activity_after: str | None, obs: AnalyzeResult
) -> dict[str, Any] | None:
    """Report the app under test vanishing from the foreground — nearly always a crash.

        A tap that kills the app answered ``ok: true`` with a cheerful observation of the
        launcher, leaving the caller to infer the crash from ``activity_after`` by hand.
        A weaker caller does not make that leap: it concludes the button "navigated home"
        and then spends its whole budget trying to navigate back inside a dead app.

        Both signals Android gives us are checked — the system's ``aerr_*`` crash dialog, and
        the foreground falling back to a launcher. An ordinary app-to-app hand-off (a share
        sheet, a browser) is deliberately NOT reported: the package changing is normal there.
        """

    def package_of(activity: str | None) -> str | None:
        if not activity or "/" not in activity:
            return None
        return activity.split("/", 1)[0] or None

    before_pkg = package_of(activity_before)
    after_pkg = package_of(activity_after)
    if not before_pkg or not after_pkg or before_pkg == after_pkg:
        return None
    crash_dialog = any("aerr_" in str(e.resource_id or "") for e in obs.elements)
    to_launcher = any(hint in after_pkg.lower() for hint in ("launcher", "home"))
    if not crash_dialog and not to_launcher:
        return None
    return {"from": before_pkg, "to": after_pkg, "crash_dialog": crash_dialog}


def _crash_evidence(self: Engine, app_id: str) -> dict[str, Any]:
    """Read and reduce the diagnostic window already opened before the failed action."""
    from . import logcat as logcat_mod

    source = "device.logs"
    if not self.platform.supports(source):
        return {
            "available": False,
            "source": source,
            "app_id": app_id,
            "code": "platform_capability_unsupported",
            "detail": (
                f"platform {self.platform.name!r} does not support capability {source!r}"
            ),
        }

    device = self.device
    path = logcat_mod.marks_path(self.config.cache.dir, device.serial)
    marks = logcat_mod.load_marks(path)
    clock = logcat_mod.resolve_clock(device, self.config.cache.dir)
    since_ms, since_label = logcat_mod.resolve_since_ms(marks, None, clock=clock)
    base: dict[str, Any] = {
        "available": True,
        "source": source,
        "app_id": app_id,
        "since": since_label,
        "since_unix_ms": since_ms,
        "clock": clock.name,
    }
    try:
        raw = self.platform.diagnostic_logs(
            device,
            lines=_CRASH_LOG_SCAN_LINES,
            since_ms=since_ms,
        )
    except AuaError as exc:
        return {
            **base,
            "available": False,
            "code": exc.code,
            "detail": exc.message,
        }
    except Exception as exc:  # noqa: BLE001 — diagnostic evidence must not hide the action
        return {
            **base,
            "available": False,
            "code": "diagnostic_logs_failed",
            "detail": str(exc),
        }
    return {
        **base,
        **logcat_mod.extract_crash_evidence(
            raw,
            app_id=app_id,
            limit=_CRASH_EVIDENCE_LINES,
        ),
    }


def _app_logs(self: Engine, app_id: str) -> dict[str, Any] | None:
    """What *app_id* logged inside this action's own window, or None when it said nothing.

        Reuses the ``last-action`` mark ``_acting`` already stamps before the device is touched,
        so this costs one scoped dump and no extra bookkeeping. Returns None rather than an
        "empty" structure for a quiet window: a field that appears on every action to say
        nothing is a tax on every step of every flow, and measured on a real app most actions
        are quiet — an idle window logged 0 lines and an ordinary tap 0 after filtering.
        """
    if not self.config.logs.enabled or not app_id:
        return None
    # Per-app first: what this app was told to keep or drop outranks the host-wide default,
    # and it is re-read per action on purpose — a preference another process just wrote (the
    # CLI, or an agent through MCP) has to take effect on the very next observation.
    cfg = self._effective_app_logs(app_id)
    if not cfg.enabled:
        return None
    if not self.platform.supports("device.logs"):
        # Silent, deliberately. An unsupported optional extra is not the caller's problem
        # on every single action; `aua logcat` reports it properly when asked directly.
        return None
    from . import logcat as logcat_mod

    device = self.device
    marks = logcat_mod.load_marks(logcat_mod.marks_path(self.config.cache.dir, device.serial))
    if "last-action" not in marks:
        # No window means no action bracketed this observation — a bare `analyze` must not
        # re-report the previous action's lines as if they were new.
        return None
    clock = logcat_mod.resolve_clock(device, self.config.cache.dir)
    since_ms, since_label = logcat_mod.resolve_since_ms(marks, "last-action", clock=clock)
    if since_ms is not None and since_ms == self._app_logs_reported_ms:
        # A wait does not stamp a new mark, so its observation would otherwise re-report the
        # previous action's lines — reading as though the app had just said all of it again.
        return None
    self._app_logs_reported_ms = since_ms
    try:
        raw = self.platform.diagnostic_logs(
            device,
            lines=cfg.scan_lines,
            since_ms=since_ms,
            app_id=app_id,
        )
    except AuaError as exc:
        logger.debug("app log window unavailable: %s", exc.message)
        return None
    digest = logcat_mod.digest_app_logs(
        raw,
        app_id=app_id,
        levels=cfg.levels,
        deny_tag_prefixes=logcat_mod.DEFAULT_DENY_TAG_PREFIXES,
        # Held apart from the built-in list on purpose: what a caller said about THIS app
        # outranks an allow-list, while the generic guess about apps in general does not.
        drop_tag_prefixes=cfg.ignore_tags,
        keep_tag_prefixes=cfg.keep_tags,
        allow_tag_prefixes=cfg.only_tags,
        limit=cfg.limit,
        per_tag=cfg.per_tag,
    )
    if not digest["count"]:
        return None
    return {
        "app_id": app_id,
        "since": since_label,
        "since_unix_ms": since_ms,
        **digest,
    }


def _change_summary(self: Engine, before: dict[str, Any] | None, obs: AnalyzeResult) -> dict[str, Any]:
    """Structured before/after deltas, with "nothing changed" stated rather than implied.

        ``changed`` is an explicit boolean so a caller can branch on it without re-deriving the
        answer from four other fields — and so "nothing changed" is machine-checkable, which is
        the half that was missing. An unknown baseline is reported as ``None``, never as False:
        "I could not compare" and "they are the same" are different claims.
        """
    # Not shortened. These become `change.text_added` / `text_removed`, which is the field a
    # caller reads to decide whether the action did what it was for — so a cut here lands in
    # the verdict itself. Saving a few dozen characters there is a false economy: the reader
    # cannot tell a clipped line from a complete one, and the way out is another `analyze`,
    # which costs a round trip and kilobytes to recover the tens of bytes that were withheld.
    after_labels = [
        (e.text or e.content_desc or "").strip()
        for e in obs.elements
        if (e.text or e.content_desc or "").strip()
    ]
    after_focus = next((e.id for e in obs.elements if e.focused), None)
    activity_before = (before or {}).get("activity") or self._last_activity
    activity_after = self._read_activity()
    if activity_after is not None:
        self._last_activity = activity_after

    out: dict[str, Any] = {
        "activity_before": activity_before,
        "activity_after": activity_after,
        "activity_changed": (
            None
            if activity_before is None or activity_after is None
            else activity_before != activity_after
        ),
        "node_count_after": len(obs.elements),
    }
    left = self._app_left_foreground(activity_before, activity_after, obs)
    if left is not None:
        out["app_left_foreground"] = left
    if before is None:
        # No baseline: say so instead of implying stability from silence.
        out.update(
            {
                "changed": None if out["activity_changed"] is None else out["activity_changed"],
                "node_count_before": None,
                "node_count_delta": None,
                "focus_moved": None,
                "text_added": [],
                "text_removed": [],
                "detail": "no pre-action snapshot — deltas unavailable",
            }
        )
        return out

    added = [t for t in dict.fromkeys(after_labels) if t not in set(before["labels"])]
    removed = [t for t in dict.fromkeys(before["labels"]) if t not in set(after_labels)]
    focus_moved = before["focused"] != after_focus
    out.update(
        {
            "node_count_before": before["count"],
            "node_count_delta": len(obs.elements) - before["count"],
            "focus_moved": focus_moved,
            "text_added": added[:_CHANGE_TEXT_CAP],
            "text_removed": removed[:_CHANGE_TEXT_CAP],
        }
    )
    out["changed"] = bool(
        out["activity_changed"] or added or removed or out["node_count_delta"] or focus_moved
    )
    if not out["changed"]:
        out["detail"] = (
            "nothing changed: same activity, same node count, no text added or removed, "
            "focus unmoved"
        )
    return out
