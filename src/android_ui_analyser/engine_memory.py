"""Learning into per-app memory and reading it back: recording screens and actions (and their timings) into the app map, the runtime flag context, learned control costs and next-action hints, the `aua memory update` command, and the knowledge read-back commands orient and explore mine/plan (source-tree deeplink mining, exploration worklist).

Engine methods for memory. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_memory.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .device import Device
from .engine_support import _label, logger
from .errors import UsageError
from .memory import (
    DEFAULT_CONTEXT_ID,
    AppMemoryStore,
    AppStrings,
    NavHints,
    RouteStep,
    _id_tail,
    launch_payload,
    playbook_view,
)
from .schema import AnalyzeResult, Element, Tier, drop_default_flags

if TYPE_CHECKING:
    from .engine import Engine


def _version_for(self: Engine, device: Device, package: str) -> str | None:
    """App versionName, fetched at most once per package (kept off the hot path)."""
    if package not in self._version_cache:
        try:
            with self.device_use_context(device.serial):
                self._version_cache[package] = device.app_version(package)
        except Exception:  # pragma: no cover - best effort
            self._version_cache[package] = None
    return self._version_cache[package]


def _sync_runtime_flag_context(
    self: Engine,
    device: Device,
    package: str,
    mem: AppMemoryStore,
    *,
    force: bool = False,
) -> bool:
    """Discover already-active feature flags before assigning a screen context."""
    cfg = self.config.flags
    configured = package in cfg.prefs_files or package in cfg.context_keys
    if not cfg.auto_context or not configured:
        return False
    now = time.monotonic()
    if not force and now - self._flag_context_checked_at.get(package, float("-inf")) < max(
        0.0, cfg.context_refresh_s
    ):
        return False
    self._flag_context_checked_at[package] = now
    flags = self.platform.capability("feature_flags")

    result = flags.read_context_flags(
        device,
        package,
        prefs_file=cfg.prefs_files.get(package),
        keys=cfg.context_keys.get(package),
        key_patterns=cfg.context_key_patterns,
    )
    if not result.verified:
        logger.debug("runtime flag context unavailable for %s: %s", package, result.reason)
        return False
    previous = mem.load_session(device.serial)
    previous_identity = (
        previous.package,
        previous.active_context_id,
        tuple(sorted(previous.active_flags.items())),
    )
    mem.activate_flag_context(
        device.serial,
        package,
        result.flags,
        app_version=self._version_for(device, package),
        verified=True,
        replace=True,
        evidence=[f"shared_prefs:{name}" for name in result.files],
    )
    current = mem.load_session(device.serial)
    changed = previous_identity != (
        current.package,
        current.active_context_id,
        tuple(sorted(current.active_flags.items())),
    )
    if changed:
        self._last_mem_fp = None
        self._last_known_screen = None
    return changed


def _record_screen_safe(
    self: Engine,
    device: Device,
    package: str | None,
    activity: str | None,
    elements: list[Element],
    tier: Tier,
    height: int | None = None,
    *,
    ocr_helped: bool | None = None,
) -> tuple[str | None, NavHints | None]:
    """Auto-record the current screen + derive navigation hints; never break analyze.

        Returns ``(known_screen, hints)``. ``hints`` carries the inline affordances
        (known_routes / suggested_gotos / map_hint) so the agent gets them on the analyze
        it already runs, instead of having to remember to call ``aua map``.
        ``ocr_helped`` records whether parallel OCR contributed kept elements (for
        experience-based OCR skip on later visits).
        """
    mem = self._memory
    if mem is None or not package:
        return None, None
    perf = self.config.perf
    try:
        # Context discovery precedes the unchanged-screen fast path: a flag may have
        # changed outside AUA while the rendered hierarchy stayed temporarily equal.
        context_changed = self._sync_runtime_flag_context(device, package, mem)
        from .perf import elements_fingerprint

        fp = elements_fingerprint(elements)
        if perf.skip_unchanged_memory and not context_changed and fp == self._last_mem_fp:
            mcfg = self.config.memory
            hints = (
                mem.navigation_hints(
                    device.serial,
                    package,
                    max_suggest=mcfg.suggest_max,
                    max_research=mcfg.research_suggest_max,
                    include_navigation=mcfg.suggest,
                    half_life_days=mcfg.rank_half_life_days,
                )
                if mcfg.suggest or mcfg.auto_research
                else None
            )
            return self._last_known_screen, hints

        # Fetch this while the foreground command still owns its lease generation. The
        # async map writer must never wake after transfer and query the old Device as the
        # newly adopted owner.
        app_version = self._version_for(device, package)

        def _do_record() -> str | None:
            return mem.observe_screen(
                device.serial,
                package=package,
                elements=elements,
                activity=activity,
                app_version=app_version,
                tier=tier.value,
                screen_height=height,
                ocr_helped=ocr_helped,
            )

        mcfg = self.config.memory
        if perf.async_memory:
            # Hints come from the map as it stands; the write happens off-path.
            hints = (
                mem.navigation_hints(
                    device.serial,
                    package,
                    max_suggest=mcfg.suggest_max,
                    max_research=mcfg.research_suggest_max,
                    include_navigation=mcfg.suggest,
                    half_life_days=mcfg.rank_half_life_days,
                )
                if mcfg.suggest or mcfg.auto_research
                else None
            )

            def _bg() -> None:
                with self._mem_lock:
                    try:
                        known = _do_record()
                        self._last_known_screen = known
                        self._last_mem_fp = fp
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("async memory record failed: %s", exc)

            t = threading.Thread(target=_bg, name="aua-mem-record", daemon=True)
            # Register the writer before exposing it to an action/save caller.  Pruning
            # after append but before start used to immediately drop the new thread
            # because ``is_alive()`` is false until ``start()`` returns, recreating the
            # exact provenance race this list is meant to prevent.
            with self._mem_threads_lock:
                self._mem_threads = [
                    thread for thread in self._mem_threads if thread.is_alive()
                ]
                self._mem_thread = t
                self._mem_threads.append(t)
                t.start()
            # Recognise synchronously rather than reusing the remembered name: the write
            # above has not landed yet, so `self._last_known_screen` still holds the
            # PREVIOUS screen's name. Reporting it labelled the device launcher and a
            # system ANR dialog with names from the app under test's own map, and told a
            # caller that had just navigated back that it was still on the screen it left.
            # Recognition is a read of a map `navigation_hints` just loaded on this same
            # path, so it costs ~nothing; an unmapped screen answers None, which is honest
            # rather than wrong.
            return (
                mem.recognize_screen(
                    device.serial,
                    package=package,
                    elements=elements,
                    activity=activity,
                    screen_height=height,
                ),
                hints,
            )

        known = _do_record()
        self._last_known_screen = known
        self._last_mem_fp = fp
        hints = (
            mem.navigation_hints(
                device.serial,
                package,
                max_suggest=mcfg.suggest_max,
                max_research=mcfg.research_suggest_max,
                include_navigation=mcfg.suggest,
                half_life_days=mcfg.rank_half_life_days,
            )
            if mcfg.suggest or mcfg.auto_research
            else None
        )
        return known, hints
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("memory record_screen failed: %s", exc)
        return None, None


def _join_memory_writers(self: Engine, *, timeout_s: float = 5.0) -> bool:
    """Wait for every queued async screen write within one bounded deadline.

        ``_mem_thread`` retained only the newest writer.  When observations arrived faster
        than their asynchronous map writes, an older queued writer could outlive it and stamp
        a package/context boundary *after* the next action had already been journaled.  Keep
        all outstanding writers ordered ahead of action capture and artifact materialisation.
        """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        with self._mem_threads_lock:
            pending = [
                thread
                for thread in self._mem_threads
                if thread is not threading.current_thread() and thread.is_alive()
            ]
            latest = self._mem_thread
            if (
                latest is not None
                and latest is not threading.current_thread()
                and latest.is_alive()
                and latest not in pending
            ):
                # Compatibility for integrations/tests that set the historical singular
                # writer handle directly. Production writers are also kept in the list.
                pending.append(latest)
        if not pending:
            with self._mem_threads_lock:
                self._mem_threads = [
                    thread for thread in self._mem_threads if thread.is_alive()
                ]
            return True
        for thread in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            thread.join(timeout=remaining)


def _record_action_safe(self: Engine, step: RouteStep) -> None:
    if self._action_recording_suppression:
        return
    mem = self._memory
    if mem is None or self._device is None:
        return
    try:
        self._claim_memory_session()
        # Async screen recording and the action journal both update SessionState.  Let the
        # screen writer finish first so the action is stamped with its newly established
        # origin/context/segment and cannot be overwritten by a stale save.
        if not self._join_memory_writers(timeout_s=5.0):
            raise RuntimeError("memory screen provenance is still being finalized")
        with self._mem_lock:
            # Open this call's access-log line here, the one moment both the start
            # instant and the resolved selector are in hand; `_journal_call_answer`
            # closes it with the cost once the caller is about to be answered. With
            # no stamp there is nothing to open, and the whole line is written there.
            started_at_ms = self._call_started_epoch_ms
            mem.observe_action(
                self._device.serial,
                step,
                started_at_ms=started_at_ms,
                outcome="ok" if started_at_ms is not None else None,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("memory record_action failed: %s", exc)


def _next_actions(
    self: Engine, obs: AnalyzeResult, *, limit: int | None = None
) -> list[dict[str, Any]] | None:
    """The actionable subset of *obs*, pre-filtered — emitted only on explicit opt-in.

        Not on by default, and the default is measured. This existed to remove a *reasoning*
        step: the post-action screen already came back inline, but an agent had to scan
        `observation.elements` for which of ~50 nodes it could act on. That scan is gone — the
        folded observation is trimmed to ~20 app-owned rows carrying `clickable`, so filtering
        it is one comprehension over a list the caller already holds. What the duplicate cost
        on one real journalled response: 1384 bytes / 346 tokens, 25% of the whole response,
        for a filtered restatement of a 1301-byte `elements` list.

        Two faults were fixed rather than shipped behind the switch. The rows go through
        :func:`schema.drop_default_flags`, the same rule `elements` uses, because 12 of 12 rows
        were reporting `checked: false, selected: false` and meaning nothing by it. And the
        list is complete: it used to cap at 12, so a 15-control screen silently hid three real
        controls from a list an agent reads as "what I can do here". A caller that wants a
        bound passes *limit* and knows it asked for one.

        The learned per-control cost — the one thing `elements` could not express — moved onto
        the element itself; see :meth:`_price_elements`.
        """
    from .identity import stable_key as _stable_key

    rows: list[dict[str, Any]] = []
    for e in obs.elements:
        if not (e.clickable or e.checkable or e.long_clickable or e.scrollable):
            continue
        label = (e.text or e.content_desc or "").strip()
        rid = _id_tail(e.resource_id)
        # The same id the observation publishes. Naming the frame ordinal here would hand
        # the caller a number that does not appear anywhere in the payload it came with —
        # and the shared trim, which filters this list to the ids that survived the view,
        # would match nothing and quietly return no next actions at all.
        row: dict[str, Any] = {"id": e.stable_key or _stable_key(e)}
        if label:
            row["label"] = _label(label)
        if rid:
            row["rid"] = rid
        # One rule for "a flag at its default says nothing", shared with `elements`, and it
        # is the reason `checkable` is written here at all: without it the trim cannot tell
        # an off switch (whose `checked: false` IS the reading) from a plain button.
        row.update(
            drop_default_flags(
                {"checkable": e.checkable, "checked": e.checked, "selected": e.selected}
            )
        )
        rows.append(row)
    if not rows:
        return None
    # Labelled first: an unlabelled container is kept (it may be the only thing that acts —
    # see `keep_actionable`) but it is not what a caller reaches for first.
    rows.sort(key=lambda r: 0 if r.get("label") or r.get("rid") else 1)
    return rows if limit is None else rows[:limit]


def _price_elements(self: Engine, obs: AnalyzeResult) -> None:
    """Attach each control's learned cost to the element it belongs to, in place.

        A cost is learned per (screen, control) and spent two ways already — as a deadline when
        acting, and as `meta.slow_controls` on arrival. Neither reaches an agent choosing what
        to tap from a folded observation: `slow_controls` is not in the `changed` meta preset
        every action response is trimmed to, so the only route was the derived `next_actions`
        list, which cost more than the whole element list to carry it.

        Priced onto the row it describes, "tap this next, and it takes ~4.8s" stays one read,
        and it survives every projection because `cost` is a default observation column. In
        place, and on the observation about to be returned, because the timing may have been
        recorded *after* the analyze that produced this frame — a still screen is exactly when
        a caller is deciding, and a price that only appears on the next fresh tree is stale
        guidance for as long as the screen sits still.
        """
    timings = self._screen_timings_safe(obs.meta.known_screen if obs.meta else None)
    if not timings:
        return
    for e in obs.elements:
        known = timings.get(e.stable_key or _id_tail(e.resource_id) or "")
        if known is None:
            continue
        e.cost = {
            "avg_ms": round(known.ema_ms),
            "max_ms": round(known.max_ms),
            "n": known.n,
        }


def _screen_timings_safe(self: Engine, screen: str | None) -> dict[str, Any]:
    """The timing map for *screen*, keyed by control — empty when unknown."""
    mem = self._memory
    if not screen or mem is None:
        return {}
    with contextlib.suppress(Exception):
        package = self._cached_package()
        if package:
            app = mem.load(package)
            rec = app.screens.get(screen) if app else None
            if rec is not None:
                return dict(rec.timings)
    return {}


def _slow_controls_safe(
    self: Engine, screen: str | None, *, package: str | None = None
) -> list[dict[str, Any]]:
    """Slow controls on *screen*, for `meta` — told on arrival, not discovered on timeout.

        This is the half the coarse profile could never provide: a per-kind average cannot say
        *which* control on the screen in front of you costs 6s. An agent that knows before acting
        can plan the wait (or pick `--until`) instead of reading a timeout as a broken product.

        Callers that already know the package pass it: recovering it from the id cache costs a
        read and a full re-validation of the previous payload, which is the wrong price to pay
        on the unchanged-frame path whose entire purpose is to be cheap.
        """
    mem = self._memory
    if not screen or mem is None:
        return []
    with contextlib.suppress(Exception):
        package = (
            package
            or self._cached_package()
            or (self.device.current_app().get("package") if self._device is not None else None)
        )
        if package:
            return mem.slow_controls(package, screen=screen)
    return []


def _record_action_timing_safe(self: Engine, ms: float, *, outcome: str) -> None:
    """Never let bookkeeping break an action — the same contract as the observation itself."""
    site = getattr(self, "_last_action_site", None)
    mem = self._memory
    if not site or mem is None:
        return
    with contextlib.suppress(Exception):
        if site.package:
            mem.record_action_timing(
                site.package,
                screen=site.screen,
                control=site.control,
                ms=ms,
                outcome=outcome,
            )


def _learned_action_budget(self: Engine, default_total_ms: int) -> int | None:
    """The deadline this control has earned from history, or None to use the coarse profile.

        Built from ``max_ms`` rather than the average: a deadline set from the mean is by
        construction too short half the time, and the cost of being too short is a false
        "nothing changed" — which this suite has already mistaken for a product defect. Padded
        by 50% and floored at the caller's default so history can only ever *extend* the wait.
        """
    site = getattr(self, "_last_action_site", None)
    mem = self._memory
    if not site or mem is None or self._device is None:
        return None
    with contextlib.suppress(Exception):
        if not site.package:
            return None
        timing = mem.action_timing(site.package, screen=site.screen, control=site.control)
        if timing is None or timing.n < 1:
            return None
        return max(default_total_ms, int(timing.max_ms * 1.5))
    return None


def memory_update(self: Engine, screen_name: str | None = None) -> dict[str, Any]:
    """Force-record the current screen now (PRD §5 ``aua memory update``)."""
    mem = self._memory
    if mem is None:
        raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
    device, w, h = self._context()
    elements, package, _xml_hash = self._capture_hierarchy(device, w, h)
    app = device.current_app()
    package = app.get("package") or package
    if not package:
        raise UsageError("could not determine the foreground package to record")
    # ``memory update --screen`` is the explicit correction path for a bad generated
    # name. Keep that correction in the same feature-flag context as normal analyze;
    # recording into ``default`` creates a disconnected duplicate and leaves every
    # route pointing at the bad name.
    self._sync_runtime_flag_context(device, package, mem)
    sess = mem.load_session(device.serial)
    same_context = sess.package in (None, package)
    context_id = sess.active_context_id if same_context else DEFAULT_CONTEXT_ID
    context_flags = sess.active_flags if same_context else {}
    outcome = mem.record_screen(
        package=package,
        elements=elements,
        activity=app.get("activity") or None,
        app_version=self._version_for(device, package),
        tier="hierarchy",
        name_hint=screen_name,
        screen_height=h,
        context_id=context_id,
        context_flags=context_flags,
        context_verified=sess.context_verified if same_context else False,
    )
    sess = mem.load_session(device.serial)
    sess.current_screen = outcome.name
    sess.package = package
    sess.pending = []
    mem.save_session(device.serial, sess)
    return {
        "ok": True,
        "action": "memory-update",
        "package": package,
        "screen": outcome.name,
        "known": outcome.was_known,
        "stale": outcome.stale,
        "created": outcome.created,
    }


def explore_mine(
    self: Engine, source: str, *, package: str | None = None, save: bool = True
) -> dict[str, Any]:
    """Mine an app's source tree for deeplinks and save them to its playbook (§6b).

        Deeplinks are shortcuts — jump straight to a screen instead of navigating — and
        the app declares them in its source. Found links are recorded so the agent can
        reuse them (`aua open-and-analyze <uri>`); templated ones (``$id``/``{id}``) are flagged.
        String resources (``values*/strings.xml``) are recorded per locale so text
        lookups can bridge the device's UI language.
        """
    from .explore import mine_deeplinks, mine_strings

    result = mine_deeplinks(Path(source))
    strings = mine_strings(Path(source))
    pkg = package or self.current_package()
    saved = 0
    strings_saved = 0
    mem = self._memory
    if save and pkg and mem is not None:
        for d in result.deeplinks:
            note = "mined from source" + (f" ({d.source})" if d.source else "")
            if d.templated:
                note += " — templated, fill the placeholder"
            mem.remember_deeplink(pkg, d.uri, note=note)
            saved += 1
        if strings.entries:
            mem.save_strings(
                AppStrings(package=pkg, locales=strings.locales, entries=strings.entries)
            )
            strings_saved = len(strings.entries)
            self._strings_cache.pop(pkg, None)
    return {
        "ok": True,
        "action": "explore-mine",
        "package": pkg,
        "source": source,
        "schemes": result.schemes,
        "found": len(result.deeplinks),
        "saved": saved,
        "strings_found": len(strings.entries),
        "string_locales": strings.locales,
        "strings_saved": strings_saved,
        "deeplinks": result.as_dict()["deeplinks"],
    }


def explore_plan(self: Engine, *, package: str | None = None, max_tasks: int = 12) -> dict[str, Any]:
    """A prioritized exploration worklist for the calling agent (the offline-agent mode).

        Reads the app's map + playbook and returns concrete next actions. Existing map debt is
        more valuable than speculative shortcuts: unresolved research comes first, then dead-end
        screens, then unprobed deeplinks. Every task declares whether it can leave the app or
        mutate state so an agent never turns "explore" into accidental authorization.
        """
    from .explore import _is_templated as _is_templated_uri
    from .reconcile import ReconciliationStore

    mem = self._memory
    pkg = package or self.current_package()
    out: dict[str, Any] = {"ok": True, "action": "explore-plan", "package": pkg, "tasks": []}
    if mem is None or not pkg:
        out["hint"] = "no memory/package — run on a device with memory enabled"
        return out
    app = mem.load(pkg)
    tasks: list[dict[str, Any]] = []
    if app is None or (not app.screens and not app.deeplinks):
        out["known"] = {"screens": 0, "routes": 0, "deeplinks": 0}
        out["bootstrap"] = (
            "no map yet — mine deeplinks first (`aua explore mine <repo> --app "
            f"{pkg}`), then launch + log in (`aua about` for the recipe) and `aua open-and-analyze` "
            "each concrete deeplink, analyzing after each to seed screens fast."
        )
        out["hint"] = "then re-run `aua explore plan`"
        return out
    out["known"] = {
        "screens": len(app.screens),
        "routes": len(app.routes),
        "deeplinks": len(app.deeplinks),
    }

    def add_task(
        *,
        kind: str,
        do: str,
        why: str,
        risk: str,
        external: bool = False,
        destructive: bool = False,
    ) -> None:
        tasks.append(
            {
                "kind": kind,
                "do": do,
                "why": why,
                "risk": risk,
                "external": external,
                "destructive": destructive,
                "requires_explicit_authorization": destructive,
            }
        )

    # 1) Current structural questions. `plan` also refreshes stale task materialization, so
    # this worklist cannot ignore hundreds of audit findings merely because deeplinks exist.
    latest = mem.latest_session(pkg)
    active_context = latest.active_context_id if latest else DEFAULT_CONTEXT_ID
    research = ReconciliationStore(mem).plan(pkg, context_id=active_context)
    issue_rank = {
        "orphan_route": 0,
        "route_conflict": 1,
        "unreplayable_route": 2,
        "stale_screen": 3,
        "duplicate_screen": 4,
        "poor_name": 5,
        "provisional_route": 6,
        "unverified_context": 7,
        "legacy_context": 8,
    }
    open_research = sorted(
        (task for task in research if task.status == "open"),
        key=lambda task: (issue_rank.get(task.issue_type, 99), task.id),
    )
    for task in open_research:
        affected = ", ".join(task.affected_ids[:3]) or "map"
        add_task(
            kind="resolve_map_issue",
            do=(
                f'aua reconcile plan --app "{pkg}"; investigate task {task.id} '
                "with source or a fresh runtime observation, then submit evidence"
            ),
            why=f"{task.issue_type} affects {affected}",
            risk="read_only_research",
        )

    # 2) Dead ends before shortcuts. The instruction is explicitly non-destructive: a screen
    # with an unknown Delete/Pay/Send control is not permission to press it.
    outgoing = {edge.from_screen for edge in app.routes if edge.status == "verified"}
    for name in app.screens:
        if name not in outgoing:
            add_task(
                kind="expand_screen",
                do=(
                    f'aua goto "{name}"; aua analyze; inspect unvisited controls one at a '
                    "time, skipping destructive or external actions unless authorized"
                ),
                why="screen has no verified routes out — map safe navigation from it",
                risk="interactive_navigation",
            )

    # 3) Deeplinks are speculative external intents, not the first exploration strategy.
    # Flag obviously destructive URI vocabulary and refuse to turn it into a ready-to-run
    # command. Other links remain runnable, but their metadata tells an orchestrator that the
    # action crosses the normal UI boundary and arrival must be verified.
    destructive_words = {
        "delete",
        "remove",
        "logout",
        "signout",
        "purchase",
        "subscribe",
        "payment",
        "erase",
    }
    destructive_words.update(
        word.casefold().replace(" ", "") for word in self.config.memory.destructive_labels
    )
    for d in app.deeplinks:
        if d.probed:
            continue
        normalized_uri = d.uri.casefold().replace("-", "").replace("_", "")
        destructive = any(word and word in normalized_uri for word in destructive_words)
        templated = _is_templated_uri(d.uri)
        if destructive:
            do = (
                f'inspect the source/handler for "{d.uri}"; do not open it without explicit '
                "authorization for the destructive effect"
            )
            risk = "destructive_external_intent"
        elif templated:
            do = (
                f'fill the placeholder in "{d.uri}" with a non-sensitive fixture, then '
                "`aua open-and-analyze` it and verify arrival"
            )
            risk = "templated_external_intent"
        else:
            do = f'aua open-and-analyze "{d.uri}"'
            risk = "external_intent"
        add_task(
            kind="probe_template" if templated else "probe_deeplink",
            do=do,
            why=(
                "unprobed deeplink — delivered intent is not proof of arrival"
                if not templated
                else "templated deeplink needs a safe fixture and verified destination"
            ),
            risk=risk,
            external=True,
            destructive=destructive,
        )

    out["tasks"] = tasks[: max(0, max_tasks)]
    out["remaining"] = len(tasks)
    out["remaining_by_kind"] = dict(Counter(task["kind"] for task in tasks))
    out["hint"] = (
        "resolve map debt and safe dead ends before speculative intents; results auto-record. "
        "Re-run `aua explore plan` as the map improves, and never treat a listed task as "
        "authorization for destructive or external side effects."
    )
    return out


def orient(self: Engine) -> dict[str, Any]:
    """What the tool already knows about the foreground app (for ``daemon start``).

        Surfaces the app **playbook** (description, deeplinks, login recipes, quirks) up
        front so the agent starts informed — the durable knowledge the tool learned.
        """
    mem = self._memory
    pkg = self.current_package()
    out: dict[str, Any] = {"package": pkg, "known": False}
    if mem is None or not pkg:
        return out
    app = mem.load(pkg)
    if app is None:
        return out
    session = mem.load_session(self.device.serial)
    playbook = playbook_view(
        app,
        context_id=session.active_context_id,
        max_deeplinks=8,
        max_notes=10,
    )
    launch = launch_payload(app)
    has_playbook = bool(
        any(playbook[key] for key in ("description", "deeplinks", "recipes", "notes")) or launch
    )
    if not app.screens and not has_playbook:
        return out
    hints = mem.navigation_hints(
        self.device.serial,
        pkg,
        max_suggest=self.config.memory.suggest_max,
        max_research=self.config.memory.research_suggest_max,
        include_navigation=self.config.memory.suggest,
        half_life_days=self.config.memory.rank_half_life_days,
    )
    out.update(
        known=True,
        screens=len(app.screens),
        routes=len(app.routes),
        suggested_gotos=hints.suggested_gotos,
        research_tasks=hints.research_tasks,
    )
    if playbook["description"]:
        out["description"] = playbook["description"]
    out.update(launch)
    if playbook["recipes"]:
        out["recipes"] = {r.name: r.note for r in playbook["recipes"]}
    if playbook["deeplinks"]:
        out["deeplinks"] = [
            {"uri": link.uri, "note": link.note} for link in playbook["deeplinks"]
        ]
    if playbook["notes"]:
        out["notes"] = playbook["notes"]
    counts = playbook["counts"]
    if counts["deeplinks"] > len(playbook["deeplinks"]) or counts["notes"] > len(
        playbook["notes"]
    ):
        out["playbook_more"] = {
            "deeplinks": counts["deeplinks"] - len(playbook["deeplinks"]),
            "notes": counts["notes"] - len(playbook["notes"]),
            "hint": "Run `aua about` for the complete current playbook.",
        }
    if counts["stale_or_scoped_out"]:
        out["playbook_filtered"] = counts["stale_or_scoped_out"]
    return out
