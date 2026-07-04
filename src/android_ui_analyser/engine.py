"""The interface-agnostic perception + action engine (PRD §6, §6a).

The engine orchestrates the analyze pipeline and the cost-aware escalation ladder. It
depends only on: the schema, the config, the device ABC, the provider *factory* +
interfaces, and the routing helpers. It NEVER imports a concrete provider, and the
hierarchy/gate/merge/annotate modules are imported lazily so a fresh checkout imports
cleanly. The CLI, MCP server, and daemon are all thin adapters over this class.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from . import routing
from .config import Config
from .device import Device, connect, list_devices
from .errors import ElementNotFoundError, ProviderError, StabilityTimeout, UsageError
from .memory import (
    REDACT_TOKENS,
    AppMemoryStore,
    NavHints,
    RouteStep,
    _id_tail,
    _shortest_path,
    is_destructive_step,
    matches_any,
    redact_label,
    resolve_goal,
    step_display,
)
from .providers.base import DetBox, Point, ScreenImage, TextBox
from .providers.registry import ProviderFactory, registered_names, run_chain
from .schema import (
    ActionResult,
    AnalyzeResult,
    DeviceInfo,
    Element,
    HasResult,
    MatchMode,
    Meta,
    PathKind,
    Screen,
    ScreenSource,
    Source,
    Tier,
    center_of,
)

logger = logging.getLogger("android_ui_analyser.engine")

QUERY_CONFIDENT = 1.0  # all salient tokens / exact phrase present
QUERY_SOFT = 0.5  # best-effort threshold when escalation is exhausted
_ASSIST_MAX_STEPS = 6  # bound on planner actions per recovery attempt (opt-in only)
_MAX_FLOW_DEPTH = 5  # bound on nested `flow:` sub-flow composition (cycle backstop)

_PACKAGE_RE = re.compile(r'package="([^"]+)"')


def _package_from_xml(
    xml: str, ignore: Sequence[str] = ("com.android.systemui",)
) -> str | None:
    """Cheap foreground-package guess from a hierarchy dump (avoids an app_current RPC).

    Picks the most common ``package=`` among nodes, excluding *ignore* globs — system
    chrome and IMEs overlay every app, so an open keyboard must never win the vote.
    Falls back to the overall majority when every node is ignorable.
    """
    pkgs = _PACKAGE_RE.findall(xml)
    if not pkgs:
        return None
    counts = Counter(p for p in pkgs if p and not matches_any(p, ignore))
    if not counts:
        counts = Counter(pkgs)
    return counts.most_common(1)[0][0]


def _parse_legacy_steps(action: str) -> list[RouteStep] | None:
    """Replay steps for a pre-v2 string-only edge: strictly a single ``tap 'X'``.

    Anything else — compound joins, ``tap [View]``, key/input/swipe — is unreplayable
    and returns ``None`` (a clean ``unsupported_action``, never a garbage label).
    """
    m = re.fullmatch(r"tap '([^']+)'", action)
    if m is None:
        return None
    return [RouteStep(kind="tap", label=m.group(1))]


def _match_step(elements: list[Element], step: RouteStep) -> Element | None:
    """Resolve a step's target element: resource-id tail first, then label.

    Redacted labels never match — a step whose only identity was PII hands off rather
    than guessing. Label matching keeps the legacy tolerance (exact, then
    prefix/substring for truncation drift).
    """
    rid = (step.resource_id or "").lower()
    if rid:
        matches = [
            e
            for e in elements
            if e.resource_id and e.resource_id.split("/")[-1].strip().lower() == rid
        ]
        if matches:
            matches.sort(
                key=lambda e: (
                    not e.clickable,
                    (e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]),
                )
            )
            return matches[0]
    label = (step.label or "").strip()
    if not label or label in REDACT_TOKENS:
        return None
    for e in elements:  # exact text / content-desc match first
        if (e.text or e.content_desc or "") == label:
            return e
    low = label.lower()
    for e in elements:  # tolerate truncation / case drift on long labels
        t = (e.text or e.content_desc or "").lower()
        if t and (t.startswith(low) or low in t):
            return e
    return None


class StepFailure(NamedTuple):
    """Why (and where) a step sequence stopped — the executor's divergence signal."""

    code: str  # destructive_step | input_required | element_not_found |
    #            unsupported_action | wait_timeout | assert_failed
    at: int  # failing step index within the executed list
    step: RouteStep


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
        "hint": hint or "route diverged — continue with `aua analyze` + `aua tap`",
    }
    if failed_step is not None:
        out["step"] = {"display": step_display(failed_step), **failed_step.model_dump()}
    if remaining_steps:
        out["remaining_steps"] = [step_display(s) for s in remaining_steps]
        pkg = next((s.package for s in remaining_steps if s.package), None)
        if pkg:
            out["expected_package"] = pkg
    return out


class Engine:
    def __init__(
        self,
        config: Config,
        *,
        device: Device | None = None,
        factory: ProviderFactory | None = None,
    ) -> None:
        self.config = config
        self._device = device
        self.factory = factory or ProviderFactory(config)
        self._mem: AppMemoryStore | None = None
        self._version_cache: dict[str, str | None] = {}

    # ----------------------------------------------------------------- device

    @property
    def device(self) -> Device:
        """Lazily connect; doctor/devices/config work without ever touching this."""
        if self._device is None:
            self._device = connect(self.config.device.serial)
        return self._device

    def list_devices(self) -> list[DeviceInfo]:
        return list_devices()

    # ----------------------------------------------------------------- capture

    def _context(self) -> tuple[Device, int, int]:
        # window_size is memoized on the device; no app_current RPC on the hot path.
        device = self.device
        w, h = device.window_size()
        return device, w, h

    def _capture_hierarchy(
        self, device: Device, w: int, h: int
    ) -> tuple[list[Element], str | None]:
        from . import hierarchy

        xml = device.dump_hierarchy()
        pkg = _package_from_xml(xml, self.config.memory.ignore_packages)
        return hierarchy.parse_hierarchy(xml, (w, h)), pkg

    def _run_vision(
        self, device: Device, *, with_ocr: bool | None, start_id: int = 0
    ) -> tuple[list[Element], list[str], ScreenImage]:
        from . import merge

        img = device.screenshot()
        providers_used: list[str] = []
        detections: list[DetBox] = []
        if self.factory.is_enabled("detection"):
            chain = self.factory.build_chain("detection")
            if chain.providers:
                try:
                    detections, name = run_chain(
                        chain,
                        lambda p: p.detect(img),  # type: ignore[attr-defined]
                        timeout_s=self.config.timeouts.detection_ms / 1000.0,
                    )
                    providers_used.append(name)
                except ProviderError as exc:
                    logger.info("detection unavailable, continuing OCR-only: %s", exc)

        texts: list[TextBox] = []
        want_ocr = self.config.ocr.enabled if with_ocr is None else with_ocr
        if want_ocr and self.factory.is_enabled("ocr"):
            chain = self.factory.build_chain("ocr")
            if chain.providers:
                try:
                    texts, name = run_chain(
                        chain,
                        lambda p: p.recognize(img),  # type: ignore[attr-defined]
                        timeout_s=self.config.timeouts.vision_ms / 1000.0,
                    )
                    providers_used.append(name)
                except ProviderError as exc:
                    logger.info("ocr unavailable: %s", exc)

        elements = merge.merge_vision(detections, texts, iou_threshold=0.5, start_id=start_id)
        return elements, providers_used, img

    # ----------------------------------------------------------------- analyze

    def _resolve_pins(self, source: str | None, strategy: str | None) -> tuple[bool, bool, bool]:
        """Return (force_hierarchy, force_vision, pin_grounding). strategy > source."""
        s = (strategy or "").lower()
        if s in ("text", "selector", "hierarchy"):
            return True, False, False
        if s == "vision":
            return False, True, False
        if s == "grounding":
            return False, True, True
        src = (source or "auto").lower()
        if src == "hierarchy":
            return True, False, False
        if src == "vision":
            return False, True, False
        return False, False, False

    def analyze(
        self,
        *,
        source: str = "auto",
        with_ocr: bool | None = None,
        query: str | None = None,
        annotate: bool | str | None = None,
        strategy: str | None = None,
        cheap: bool = False,
        deep: bool = False,
        no_cache: bool = False,
        record: bool = True,
    ) -> AnalyzeResult:
        ceiling = routing.resolve_ceiling(self.config.routing.max_tier, cheap=cheap, deep=deep)
        force_hier, force_vis, pin_grounding = self._resolve_pins(source, strategy)
        # An explicit --strategy pin is a per-call opt-in: raise the ceiling so the pinned
        # tier is actually reachable even if routing.max_tier is lower (still never an
        # *implicit* paid escalation — the user named the tier).
        if pin_grounding:
            ceiling = Tier.grounding
        elif force_vis and not routing.allows(Tier.vision, ceiling):
            ceiling = Tier.vision
        if query:
            return self._analyze_query(
                query,
                ceiling=ceiling,
                force_hierarchy=force_hier,
                force_vision=force_vis,
                pin_grounding=pin_grounding,
                with_ocr=with_ocr,
                annotate=annotate,
                no_cache=no_cache,
            )
        return self._analyze_screen(
            ceiling=ceiling,
            force_hierarchy=force_hier,
            force_vision=force_vis,
            with_ocr=with_ocr,
            annotate=annotate,
            no_cache=no_cache,
            record=record,
        )

    def _analyze_screen(
        self,
        *,
        ceiling: Tier,
        force_hierarchy: bool,
        force_vision: bool,
        with_ocr: bool | None,
        annotate: bool | str | None,
        no_cache: bool,
        record: bool = True,
    ) -> AnalyzeResult:
        from . import gate

        t0 = time.perf_counter()
        device, w, h = self._context()
        providers_used: list[str] = []
        img: ScreenImage | None = None
        package: str | None = None
        activity: str | None = None

        elements: list[Element] = []
        screen_source = ScreenSource.hierarchy
        tier_used = Tier.hierarchy
        path = PathKind.hierarchy

        if not force_vision:
            elements, package = self._capture_hierarchy(device, w, h)

        use_vision = force_vision
        if not force_vision and not force_hierarchy:
            decision = gate.decide(
                elements, package=package, activity=activity, cfg=self.config.perception.gate
            )
            if decision.use_vision and routing.allows(Tier.vision, ceiling):
                use_vision = True
                logger.info("gate → vision: %s", decision.reason)
            elif decision.use_vision:
                logger.info("gate wants vision but ceiling=%s; staying hierarchy", ceiling.value)

        if use_vision:
            # slow fallback path: fetch full app context (incl. activity)
            app = device.current_app()
            package = app.get("package") or package
            activity = app.get("activity") or None
            vis_elements, providers_used, img = self._run_vision(device, with_ocr=with_ocr)
            elements = vis_elements
            screen_source = ScreenSource.vision
            tier_used = Tier.vision
            path = PathKind.vision

        if record:
            known_screen, hints = self._record_screen_safe(
                device, package, activity, elements, tier_used, h
            )
        else:
            # An observe snapshot taken right after an action can be mid-transition; never
            # let it pollute memory with a transient screen (it's just fresh ids for the agent).
            known_screen, hints = None, None
        annotated = self._maybe_annotate(annotate, device, elements, img)

        result = AnalyzeResult(
            screen=Screen(
                width=w, height=h, package=package, activity=activity, source=screen_source
            ),
            elements=elements,
            meta=Meta(
                duration_ms=int((time.perf_counter() - t0) * 1000),
                tier_used=tier_used,
                path=path,
                providers_used=providers_used,
                known_screen=known_screen,
                known_routes=hints.known_routes if hints else [],
                suggested_gotos=hints.suggested_gotos if hints else [],
                map_hint=hints.map_hint if hints else None,
                annotated_image=annotated,
                device_serial=device.serial,
            ),
        )
        if not no_cache:
            self._write_cache(result)
        return result

    def _analyze_query(
        self,
        query: str,
        *,
        ceiling: Tier,
        force_hierarchy: bool,
        force_vision: bool,
        pin_grounding: bool,
        with_ocr: bool | None,
        annotate: bool | str | None,
        no_cache: bool,
    ) -> AnalyzeResult:
        from . import gate

        t0 = time.perf_counter()
        device, w, h = self._context()
        package: str | None = None
        activity: str | None = None
        providers_used: list[str] = []
        pool: list[Element] = []
        img: ScreenImage | None = None
        tier_used = Tier.hierarchy
        screen_source = ScreenSource.hierarchy
        path = PathKind.hierarchy
        best: Element | None = None
        best_score = 0.0
        known_screen: str | None = None
        hints: NavHints | None = None

        # --- T1/T2: satisfy from the hierarchy first (cheap-first) ---
        if not force_vision:
            pool, package = self._capture_hierarchy(device, w, h)
            tier_used = Tier.selector
            known_screen, hints = self._record_screen_safe(
                device, package, activity, pool, Tier.hierarchy, h
            )
            cand, score = self._match_query(query, pool)
            if cand is not None and score > best_score:
                best, best_score = cand, score
            if best_score >= QUERY_CONFIDENT and not pin_grounding:
                return self._finish_query(
                    best,
                    w,
                    h,
                    package,
                    activity,
                    ScreenSource.hierarchy,
                    Tier.selector,
                    PathKind.hierarchy,
                    providers_used,
                    device,
                    annotate,
                    img,
                    no_cache,
                    t0,
                    known_screen,
                    hints,
                )

        # --- T3: vision, if allowed and useful ---
        want_vision = force_vision
        if not force_vision and routing.allows(Tier.vision, ceiling):
            decision = gate.decide(
                pool, package=package, activity=activity, cfg=self.config.perception.gate
            )
            kind = routing.classify_query(query)
            want_vision = decision.use_vision or kind is routing.QueryKind.visual or pin_grounding

        if want_vision and routing.allows(Tier.vision, ceiling):
            app = device.current_app()
            package = app.get("package") or package
            activity = app.get("activity") or None
            vis_elements, vprov, img = self._run_vision(
                device, with_ocr=with_ocr, start_id=len(pool)
            )
            providers_used.extend(vprov)
            pool = pool + vis_elements
            screen_source = ScreenSource.mixed if pool and vis_elements else ScreenSource.vision
            tier_used = Tier.vision
            path = PathKind.vision
            if force_vision:  # hierarchy block was skipped → record the screen from vision
                known_screen, hints = self._record_screen_safe(
                    device, package, activity, pool, Tier.vision, h
                )
            cand, score = self._match_query(query, vis_elements)
            if cand is not None and score > best_score:
                best, best_score = cand, score
            if best_score >= QUERY_CONFIDENT and not pin_grounding:
                return self._finish_query(
                    best,
                    w,
                    h,
                    package,
                    activity,
                    screen_source,
                    Tier.vision,
                    path,
                    providers_used,
                    device,
                    annotate,
                    img,
                    no_cache,
                    t0,
                    known_screen,
                    hints,
                )

        # --- T4: grounding VLM, only if explicitly allowed (never silent/paid by default) ---
        grounding_ok = (
            routing.allows(Tier.grounding, ceiling)
            and self.factory.is_enabled("grounding")
            and (
                pin_grounding or routing.classify_query(query) is not routing.QueryKind.resource_id
            )
        )
        if best_score < QUERY_CONFIDENT and grounding_ok:
            chain = self.factory.build_chain("grounding")
            if chain.providers:
                if img is None:
                    img = device.screenshot()
                try:
                    loc, name = run_chain(
                        chain,
                        lambda p: p.locate(img, query),  # type: ignore[attr-defined]
                        is_empty=lambda r: r is None,
                        timeout_s=self.config.timeouts.grounding_ms / 1000.0,
                    )
                    providers_used.append(name)
                    grounded = self._map_grounding(loc, pool, w, h)
                    if grounded is not None:
                        return self._finish_query(
                            grounded,
                            w,
                            h,
                            package,
                            activity,
                            ScreenSource.mixed,
                            Tier.grounding,
                            PathKind.vision,
                            providers_used,
                            device,
                            annotate,
                            img,
                            no_cache,
                            t0,
                        )
                except ProviderError as exc:
                    logger.info("grounding unavailable: %s", exc)
        elif best_score < QUERY_CONFIDENT and self.factory.is_enabled("grounding"):
            logger.info(
                "not escalating to grounding: ceiling=%s (use --deep or raise routing.max_tier)",
                ceiling.value,
            )

        # --- best-effort or not-found ---
        chosen = best if best is not None and best_score >= QUERY_SOFT else None
        return self._finish_query(
            chosen,
            w,
            h,
            package,
            activity,
            screen_source,
            tier_used,
            path,
            providers_used,
            device,
            annotate,
            img,
            no_cache,
            t0,
            known_screen,
            hints,
        )

    def _finish_query(
        self,
        element: Element | None,
        w: int,
        h: int,
        package: str | None,
        activity: str | None,
        screen_source: ScreenSource,
        tier_used: Tier,
        path: PathKind,
        providers_used: list[str],
        device: Device,
        annotate: bool | str | None,
        img: ScreenImage | None,
        no_cache: bool,
        t0: float,
        known_screen: str | None = None,
        hints: NavHints | None = None,
    ) -> AnalyzeResult:
        elements = [element] if element is not None else []
        annotated = self._maybe_annotate(annotate, device, elements, img)
        result = AnalyzeResult(
            screen=Screen(
                width=w, height=h, package=package, activity=activity, source=screen_source
            ),
            elements=elements,
            meta=Meta(
                duration_ms=int((time.perf_counter() - t0) * 1000),
                tier_used=tier_used,
                path=path,
                providers_used=providers_used,
                known_screen=known_screen,
                known_routes=hints.known_routes if hints else [],
                suggested_gotos=hints.suggested_gotos if hints else [],
                map_hint=hints.map_hint if hints else None,
                annotated_image=annotated,
                device_serial=device.serial,
            ),
        )
        if not no_cache:
            self._write_cache(result)
        return result

    # ----------------------------------------------------------------- query match

    def _match_query(self, query: str, elements: list[Element]) -> tuple[Element | None, float]:
        tokens = routing.salient_tokens(query)
        phrase = " ".join(tokens)
        ql = query.strip().lower()
        best: Element | None = None
        best_score = -1.0
        for el in elements:
            parts: list[str] = []
            if el.text:
                parts.append(el.text)
            if el.content_desc:
                parts.append(el.content_desc)
            if el.resource_id:
                parts.append(el.resource_id.split("/")[-1].replace("_", " "))
            hay = " ".join(parts).lower().strip()
            if not hay:
                continue
            if el.text and el.text.strip().lower() == ql or phrase and phrase in hay:
                score = 1.0
            elif tokens:
                score = sum(1 for t in tokens if t in hay) / len(tokens)
            else:
                score = 0.0
            # tie-break: prefer clickable, then smaller area
            adj = score + (0.001 if el.clickable else 0.0)
            if adj > best_score:
                best, best_score = el, adj
        if best is None:
            return None, 0.0
        return best, min(1.0, best_score)

    def _map_grounding(
        self, loc: Point | DetBox | None, pool: list[Element], w: int, h: int
    ) -> Element | None:
        from . import merge

        if loc is None:
            return None
        if isinstance(loc, Point):
            px, py = loc.x, loc.y
            # element containing the point, else nearest center
            containing = [
                e
                for e in pool
                if e.bounds[0] <= px <= e.bounds[2] and e.bounds[1] <= py <= e.bounds[3]
            ]
            if containing:
                return min(
                    containing,
                    key=lambda e: (e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]),
                )
            if pool:
                return min(pool, key=lambda e: (e.center[0] - px) ** 2 + (e.center[1] - py) ** 2)
            box = (max(0, px - 24), max(0, py - 24), min(w, px + 24), min(h, py + 24))
            return Element(
                id=0,
                type="GroundedPoint",
                bounds=box,
                center=(px, py),
                source=Source.grounding,
                confidence=loc.confidence,
                clickable=True,
            )
        # DetBox
        if pool:
            scored = [(merge.iou(loc.bounds, e.bounds), e) for e in pool]
            scored.sort(key=lambda t: t[0], reverse=True)
            if scored and scored[0][0] > 0.1:
                return scored[0][1]
        return Element(
            id=len(pool),
            type="GroundedBox",
            text=loc.label,
            bounds=loc.bounds,
            center=center_of(loc.bounds),
            source=Source.grounding,
            confidence=loc.confidence,
            clickable=loc.interactable,
        )

    # ----------------------------------------------------------------- memory (§6b)

    @property
    def _memory(self) -> AppMemoryStore | None:
        if not self.config.memory.enabled:
            return None
        if self._mem is None:
            self._mem = AppMemoryStore(self.config.memory)
        return self._mem

    def _version_for(self, device: Device, package: str) -> str | None:
        """App versionName, fetched at most once per package (kept off the hot path)."""
        if package not in self._version_cache:
            try:
                self._version_cache[package] = device.app_version(package)
            except Exception:  # pragma: no cover - best effort
                self._version_cache[package] = None
        return self._version_cache[package]

    def _record_screen_safe(
        self,
        device: Device,
        package: str | None,
        activity: str | None,
        elements: list[Element],
        tier: Tier,
        height: int | None = None,
    ) -> tuple[str | None, NavHints | None]:
        """Auto-record the current screen + derive navigation hints; never break analyze.

        Returns ``(known_screen, hints)``. ``hints`` carries the inline affordances
        (known_routes / suggested_gotos / map_hint) so the agent gets them on the analyze
        it already runs, instead of having to remember to call ``aua map``.
        """
        mem = self._memory
        if mem is None or not package:
            return None, None
        try:
            known = mem.observe_screen(
                device.serial,
                package=package,
                elements=elements,
                activity=activity,
                app_version=self._version_for(device, package),
                tier=tier.value,
                screen_height=height,
            )
            mcfg = self.config.memory
            hints = (
                mem.navigation_hints(
                    device.serial,
                    package,
                    max_suggest=mcfg.suggest_max,
                    half_life_days=mcfg.rank_half_life_days,
                )
                if mcfg.suggest
                else None
            )
            return known, hints
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("memory record_screen failed: %s", exc)
            return None, None

    def _record_action_safe(self, step: RouteStep) -> None:
        mem = self._memory
        if mem is None or self._device is None:
            return
        try:
            mem.observe_action(self._device.serial, step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("memory record_action failed: %s", exc)

    def _cached_package(self) -> str | None:
        """Package of the last analyze (call BEFORE the action invalidates the cache)."""
        cached = self._read_cache()
        return cached.screen.package if cached else None

    def _step(
        self,
        kind: str,
        element: Element | None = None,
        *,
        arg: str | None = None,
        submit: bool = False,
    ) -> RouteStep:
        """The structured record of one action (selector + redacted label, never a value)."""
        label = redact_label(element, redact=self.config.memory.redact) if element else None
        return RouteStep(
            kind=kind,
            label=label,
            resource_id=_id_tail(element.resource_id) if element else None,
            arg=arg,
            submit=submit,
            package=self._cached_package(),
        )

    def current_package(self) -> str | None:
        """Best-effort foreground package (for ``aua map`` without ``--app``)."""
        try:
            pkg = self.device.current_app().get("package")
        except Exception:  # pragma: no cover - device hiccup
            pkg = None
        if pkg:
            return pkg
        try:
            return _package_from_xml(
                self.device.dump_hierarchy(), self.config.memory.ignore_packages
            )
        except Exception:  # pragma: no cover
            return None

    def memory_update(self, screen_name: str | None = None) -> dict[str, Any]:
        """Force-record the current screen now (PRD §5 ``aua memory update``)."""
        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        device, w, h = self._context()
        elements, package = self._capture_hierarchy(device, w, h)
        app = device.current_app()
        package = app.get("package") or package
        if not package:
            raise UsageError("could not determine the foreground package to record")
        outcome = mem.record_screen(
            package=package,
            elements=elements,
            activity=app.get("activity") or None,
            app_version=self._version_for(device, package),
            tier="hierarchy",
            name_hint=screen_name,
            screen_height=h,
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

    # ----------------------------------------------------------------- step executor

    def _source_for(
        self, steps: list[RouteStep], index: int, origin_package: str | None
    ) -> str:
        """Analyze source between steps: ``auto`` when the NEXT step runs in a foreign
        (transit) package — its screen may be vision-tier — else the fast hierarchy path."""
        nxt = steps[index] if index < len(steps) else None
        if nxt is not None and nxt.package and nxt.package != origin_package:
            return "auto"
        return "hierarchy"

    def _run_steps(
        self,
        steps: list[RouteStep],
        *,
        origin_package: str | None,
        allow_destructive: bool,
        allow_goto_steps: bool = False,
        scroll_fallback: bool = False,
        res: AnalyzeResult | None = None,
        executed: list[dict[str, Any]] | None = None,
        flow_depth: int = 0,
    ) -> tuple[StepFailure | None, AnalyzeResult]:
        """Execute *steps* with selector matching, settle waits, and re-perception.

        The single replay engine behind ``goto`` edge replay and ``flow run``. Between
        state-changing steps it settles (suppressed ``wait_stable``) and re-analyzes with
        a package-aware source (:meth:`_source_for`). Verification is lazy — a wrong
        screen surfaces as the next step's ``element_not_found`` — terminal verification
        (``known_screen`` / asserts) is the caller's job. Returns
        ``(failure | None, last analyze result)``.
        """
        if res is None:
            res = self.analyze(source=self._source_for(steps, 0, origin_package))
        lexicon = self.config.memory.destructive_labels
        for i, s in enumerate(steps):
            if is_destructive_step(s, lexicon) and not allow_destructive:
                return StepFailure("destructive_step", i, s), res
            kind = s.kind
            reanalyze = True  # most kinds change state → settle + re-perceive
            settle = True
            if kind in ("tap", "long-press", "clear", "input"):
                if kind == "input" and s.text is None:
                    # auto-recorded inputs never store the value — the caller supplies it
                    return StepFailure("input_required", i, s), res
                el = _match_step(res.elements, s)
                if el is None and scroll_fallback and (s.label or s.resource_id):
                    self.scroll_to(s.label or s.resource_id or "", observe=False)
                    res = self.analyze(source=self._source_for(steps, i, origin_package))
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
                    self.input_text(el.id, s.text or "", submit=s.submit, observe=False)
            elif kind == "key":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                self.key(s.arg, observe=False)
            elif kind == "swipe":
                if s.arg not in ("up", "down", "left", "right"):
                    return StepFailure("unsupported_action", i, s), res
                self.swipe(s.arg, observe=False)
            elif kind == "scroll-to":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if not self.scroll_to(s.arg, observe=False).ok:
                    return StepFailure("element_not_found", i, s), res
            elif kind == "launch-app":
                pkg = s.arg or origin_package  # bare launch_app → the flow's own app
                if not pkg:
                    return StepFailure("unsupported_action", i, s), res
                self.app("launch", package=pkg)
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
                if not self.wait(for_=s.arg, timeout_ms=s.timeout_ms or 10000).ok:
                    return StepFailure("wait_timeout", i, s), res
                settle = False  # the wait already absorbed the transition
            elif kind == "wait-stable":
                try:
                    self.wait_stable(settle_ms=600, timeout_ms=s.timeout_ms or 15000)
                except StabilityTimeout:
                    return StepFailure("wait_timeout", i, s), res
                settle = False
            elif kind == "assert-visible":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if not self.has(s.arg, timeout_ms=s.timeout_ms or 0).found:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False  # pure check, screen unchanged
            elif kind == "goto":
                if not allow_goto_steps or not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                out = self.goto(s.arg, allow_destructive=allow_destructive)
                if not out.get("ok"):
                    return StepFailure(str(out.get("code") or "route_unknown"), i, s), res
                settle = False  # goto verified arrival; just refresh our view
            elif kind == "flow":
                # Run a saved flow inline (Maestro's runFlow) — reuse shared recipes.
                if not allow_goto_steps or not s.arg or flow_depth >= _MAX_FLOW_DEPTH:
                    return StepFailure("unsupported_action", i, s), res
                from .flows import FlowStore, resolve_params

                try:
                    sub = FlowStore(self.config.memory).load(s.arg)
                except UsageError:
                    return StepFailure("route_unknown", i, s), res
                subfail, res = self._run_steps(
                    resolve_params(sub, {}),
                    origin_package=sub.app or origin_package,
                    allow_destructive=allow_destructive,
                    allow_goto_steps=True,
                    scroll_fallback=scroll_fallback,
                    res=res,
                    executed=executed,
                    flow_depth=flow_depth + 1,
                )
                if subfail is not None:
                    return StepFailure(subfail.code, i, s), res  # surface sub-failure here
                settle = False  # the sub-flow already settled
            else:
                return StepFailure("unsupported_action", i, s), res

            if executed is not None:
                executed.append({"index": i, "step": step_display(s)})
            if reanalyze:
                if settle:
                    with contextlib.suppress(StabilityTimeout):
                        self.wait_stable(settle_ms=500, timeout_ms=8000)
                res = self.analyze(source=self._source_for(steps, i + 1, origin_package))
        return None, res

    # ----------------------------------------------------------------- planner (§7.3)

    def _planner_view(
        self, res: AnalyzeResult
    ) -> tuple[list[dict[str, Any]], ScreenImage | None]:
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
        self,
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
            el = (
                res.element_by_id(decision.target_id)
                if decision.target_id is not None
                else None
            )
            if action in ("tap", "input") and el is None:
                return False, res  # invalid/off-screen id → hand off rather than guess
            if el is not None:  # destructive guard applies to the planner too
                probe = RouteStep(
                    kind="tap", label=redact_label(el, redact=self.config.memory.redact)
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
        self, target: str, res: AnalyzeResult, *, allow_destructive: bool
    ) -> tuple[bool, AnalyzeResult]:
        """On a diverged goto, let the planner try to reach *target*. Verified by
        ``known_screen`` (deterministic), not the planner's own verdict."""
        objective = (
            f"Reach the app screen named '{target}'. If a dialog, permission prompt, or "
            "popup is blocking the screen, dismiss it (Allow, Not now, Skip, Close, "
            "Continue) to make progress toward that screen."
        )
        _, res = self._drive_with_planner(
            objective, res=res, max_steps=_ASSIST_MAX_STEPS, allow_destructive=allow_destructive
        )
        return res.meta.known_screen == target, res

    def _assist_suggestion(self, assist: bool) -> str | None:
        """Handoff hint: suggest --assist when it wasn't used; note it was tried if it was."""
        if not assist:
            return (
                "route diverged — continue manually, or re-run with `--assist` to let a "
                "fast model try to recover (needs `planner.enabled` + its API key)"
            )
        return "route diverged and assisted recovery could not reach the target — continue manually"

    def goto(
        self,
        goal: str,
        *,
        plan: bool = False,
        max_steps: int = 8,
        allow_destructive: bool = False,
        assist: bool = False,
    ) -> dict[str, Any]:
        """Drive to a remembered screen via the app map (PRD §6b).

        Resolves *goal* to a known screen, then replays the recorded steps of each edge
        on the shortest route, re-analyzing and verifying ``known_screen`` after each hop.
        On any mismatch it stops and hands back the remaining route/steps + the current
        screen, so the caller can continue manually. ``plan=True`` returns the annotated
        route without acting. Destructive steps (config ``memory.destructive_labels``)
        are refused unless *allow_destructive*.
        """
        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        res = self.analyze(source="hierarchy")  # perceive current screen (writes the id cache)
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
        current = sess.current_screen
        target = resolve_goal(
            app,
            goal,
            start=current,
            half_life_days=self.config.memory.rank_half_life_days,
            last_goal=sess.last_goal,
        )
        if target is None:
            return {
                "ok": False,
                "code": "route_unknown",
                "goal": goal,
                "package": package,
                "known_screens": list(app.screens),
                "hint": "no known screen matches; explore with `aua analyze`",
            }
        mem.set_last_goal(serial, goal)  # remember intent for ranking even if we divert
        if current == target and not transit_resume:  # mid-transit we are NOT on target
            return {
                "ok": True,
                "goal": goal,
                "target": target,
                "arrived": True,
                "already_there": True,
                "package": package,
                "route": [],
                "hops": [],
            }
        path = _shortest_path(app, target, start=current)
        route = [{"from": e.from_screen, "action": e.action, "to": e.to_screen} for e in path]
        if not path:
            return {
                "ok": False,
                "code": "route_unknown",
                "goal": goal,
                "target": target,
                "package": package,
                "current_screen": current,
                "hint": "no known route from here — explore with `aua analyze`",
            }
        lexicon = self.config.memory.destructive_labels
        if plan:
            annotated = []
            for e in path:
                steps = e.steps or _parse_legacy_steps(e.action)
                annotated.append(
                    {
                        "from": e.from_screen,
                        "action": e.action,
                        "to": e.to_screen,
                        "steps": [step_display(s) for s in (steps or [])],
                        "replayable": steps is not None,
                        "legacy": not e.steps,
                        "destructive": [
                            s.label for s in (steps or []) if is_destructive_step(s, lexicon)
                        ],
                    }
                )
            return {
                "ok": True,
                "goal": goal,
                "target": target,
                "plan": True,
                "package": package,
                "route": annotated,
                "note": "not executed (--plan)",
            }
        resume_from = 0
        if transit_resume:
            first_steps = path[0].steps or _parse_legacy_steps(path[0].action)
            if first_steps is None:
                return _goto_handoff(
                    goal,
                    target,
                    "unsupported_action",
                    [],
                    route,
                    res,
                    hint="mid-transit on a pre-v2 edge — finish manually, then re-run goto",
                )
            res = self.analyze(source="auto")  # transit screens may be vision-tier
            matched = next(
                (j for j, s in enumerate(first_steps) if _match_step(res.elements, s)),
                None,
            )
            if matched is None:
                return _goto_handoff(
                    goal,
                    target,
                    "element_not_found",
                    [],
                    route,
                    res,
                    remaining_steps=first_steps,
                    hint="mid-transit, but no remembered step matches this screen — "
                    "finish it manually (`aua analyze` + `aua tap`), then re-run `aua goto`",
                )
            resume_from = matched
        hops: list[dict[str, Any]] = []
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
            fail, res = self._run_steps(
                steps,
                origin_package=package,
                allow_destructive=allow_destructive,
                res=res,
            )
            if fail is not None:
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
            hops.append(
                {
                    "action": edge.action,
                    "expected": edge.to_screen,
                    "known_screen": reached,
                    "ok": reached == edge.to_screen,
                }
            )
            if reached != edge.to_screen:
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
        arrived = res.meta.known_screen == target
        return {
            "ok": arrived,
            "goal": goal,
            "target": target,
            "arrived": arrived,
            "package": package,
            "final_screen": res.meta.known_screen,
            "hops": hops,
            "route": route,
            # destination elements (ids) so the caller can act without a re-analyze;
            # the id cache is already warm from goto's final analyze.
            "elements": [e.compact() for e in res.elements],
        }

    # ----------------------------------------------------------------- flows (§6b)

    def flow_run(
        self,
        name: str | None = None,
        *,
        file: str | None = None,
        params: dict[str, str] | None = None,
        dry_run: bool = False,
        from_step: int = 0,
        allow_destructive: bool = True,
        assist: bool = False,
    ) -> dict[str, Any]:
        """Replay a named (or ``--file``) flow in one call — the whole journey.

        Runs through the same executor as ``goto``; on divergence returns the failing
        step's index + the remaining steps so the caller can fix or finish manually and
        resume with ``from_step``. Authored flows are deliberate intent, so destructive
        steps are ALLOWED by default (unlike goto's auto-learned replay). With *assist*
        (opt-in planner), a divergence triggers one recovery attempt (dismiss a blocking
        dialog) then resumes from the failed step before handing off.
        """
        from .flows import FlowStore, parse_flow_yaml, resolve_params

        if file:
            path = Path(file).expanduser()
            if not path.is_file():
                raise UsageError(f"no flow file at {path}")
            flow = parse_flow_yaml(path.read_text(encoding="utf-8"), name=path.stem)
        elif name:
            flow = FlowStore(self.config.memory).load(name)
        else:
            raise UsageError("flow run needs a NAME or --file", hint="see `aua flow list`")
        steps = resolve_params(flow, params or {})
        if not 0 <= from_step < len(steps):
            raise UsageError(
                f"--from-step {from_step} out of range (flow has {len(steps)} steps)"
            )
        steps_slice = steps[from_step:]
        lexicon = self.config.memory.destructive_labels
        if dry_run:
            return {
                "ok": True,
                "flow": flow.name,
                "dry_run": True,
                "app": flow.app,
                "params_declared": sorted(flow.params),
                "steps": [
                    {
                        "index": from_step + i,
                        "step": step_display(s),
                        "destructive": is_destructive_step(s, lexicon),
                    }
                    for i, s in enumerate(steps_slice)
                ],
                "note": "not executed (--dry-run)",
            }
        executed: list[dict[str, Any]] = []

        def _exec(slice_start: int, res_in: AnalyzeResult | None) -> tuple[Any, AnalyzeResult, int | None]:
            ex: list[dict[str, Any]] = []
            f, r = self._run_steps(
                steps[slice_start:],
                origin_package=flow.app,
                allow_destructive=allow_destructive,
                allow_goto_steps=True,
                scroll_fallback=True,
                res=res_in,
                executed=ex,
            )
            for e in ex:
                e["index"] += slice_start  # absolute flow indices
            executed.extend(ex)
            return f, r, (slice_start + f.at if f is not None else None)

        fail, res, idx = _exec(from_step, None)
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
            hint = (
                "fix the flow or finish the step manually, then resume with "
                f"`aua flow run {flow.name} --from-step {idx}`"
            )
            if not assist:
                hint += (
                    "; or add `--assist` to let a fast model clear blockers "
                    "(needs `planner.enabled` + its API key)"
                )
            return {
                "ok": False,
                "code": fail.code,
                "flow": flow.name,
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
        return {
            "ok": True,
            "flow": flow.name,
            "steps_run": executed,
            "final_screen": res.meta.known_screen,
            # destination elements (ids) so the caller can act without a re-analyze
            "elements": [e.compact() for e in res.elements],
        }

    def flow_save(self, name: str, *, last: int = 12, force: bool = False) -> dict[str, Any]:
        """Materialize the session's recent recorded actions into an editable flow file.

        Redacted inputs/labels become required ``${PARAM_n}`` placeholders — typed
        values are never recorded, so the agent fills them in the saved YAML.
        """
        from .flows import Flow, FlowStore, steps_from_recent

        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        sess = mem.load_session(self.device.serial)
        recent = sess.recent[-max(1, last) :]
        if not recent:
            raise UsageError(
                "no recorded actions to save",
                hint="drive the app first (tap/input/…), then `aua flow save <name>`",
            )
        steps, params = steps_from_recent(recent)
        flow = Flow(
            name=name,
            app=sess.package,
            description=f"Recorded from the last {len(steps)} session actions",
            params=params,
            steps=steps,
        )
        path = FlowStore(self.config.memory).save(flow, force=force)
        return {
            "ok": True,
            "action": "flow-save",
            "flow": name,
            "path": str(path),
            "steps": len(steps),
            "params_needed": sorted(params),
            "hint": "edit the YAML to fill ${PARAM_n} values / trim steps, then `aua flow run`",
        }

    def navigate(
        self,
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
        recent_before = len(mem.load_session(serial).recent) if mem else 0
        res = self.analyze(source="auto")  # perceive + record the starting screen
        arrived, res = self._drive_with_planner(
            goal,
            res=res,
            max_steps=max_steps,
            allow_destructive=allow_destructive,
            until=until,
        )
        flow_saved: str | None = None
        if save_flow and mem is not None:
            from .flows import Flow, FlowStore, steps_from_recent

            taken = mem.load_session(serial).recent[recent_before:]
            if taken:
                steps, params = steps_from_recent(taken)
                path = FlowStore(self.config.memory).save(
                    Flow(
                        name=save_flow,
                        app=res.screen.package,
                        description=f"Recorded by `aua navigate`: {goal}",
                        params=params,
                        steps=steps,
                    ),
                    force=True,
                )
                flow_saved = str(path)
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

    def close(self) -> None:
        """Release the device (and its on-device uiautomator2 server). Idempotent."""
        dev = self._device
        if dev is not None:
            with contextlib.suppress(Exception):
                dev.close()
            self._device = None

    def orient(self) -> dict[str, Any]:
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
        has_playbook = bool(app.description or app.deeplinks or app.recipes or app.notes)
        if not app.screens and not has_playbook:
            return out
        hints = mem.navigation_hints(
            self.device.serial,
            pkg,
            max_suggest=self.config.memory.suggest_max,
            half_life_days=self.config.memory.rank_half_life_days,
        )
        out.update(
            known=True,
            screens=len(app.screens),
            routes=len(app.routes),
            suggested_gotos=hints.suggested_gotos,
        )
        if app.description:
            out["description"] = app.description
        if app.recipes:
            out["recipes"] = {r.name: r.note for r in app.recipes}
        if app.deeplinks:
            out["deeplinks"] = [
                {"uri": d.uri, "note": d.note} for d in app.deeplinks
            ]
        if app.notes:
            out["notes"] = list(app.notes)
        return out

    # ----------------------------------------------------------------- wait --for-stable

    def wait_stable(
        self, *, interval_ms: int = 200, settle_ms: int = 600, timeout_ms: int = 30000
    ) -> ActionResult:
        """Return once the screen stops changing for ``settle_ms`` (PRD §5, AC14).

        Cheap perceptual-hash over screenshots only — NO OCR, NO hierarchy parse. Works on
        opaque/Compose/video screens; ideal for waiting on image generation / loading.
        """
        from . import imaging

        device = self.device
        deadline = time.monotonic() + timeout_ms / 1000.0
        last: int | None = None
        stable_since: float | None = None
        samples = 0
        while True:
            current = imaging.dhash(device.screenshot())
            samples += 1
            now = time.monotonic()
            if last is not None and imaging.is_stable(current, last):
                if stable_since is None:
                    stable_since = now
                if (now - stable_since) * 1000.0 >= settle_ms:
                    return ActionResult(
                        ok=True, action="wait-stable", detail=f"settled after {samples} samples"
                    )
            else:
                stable_since = None
            last = current
            if now >= deadline:
                raise StabilityTimeout(
                    f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                    hint="Increase --timeout/--settle, or the screen is still animating.",
                )
            time.sleep(interval_ms / 1000.0)

    # ----------------------------------------------------------------- has (T0)

    def has(
        self,
        text: str,
        *,
        match: str = "contains",
        ignore_case: bool = False,
        ocr_fallback: bool = True,
        source: str = "auto",
        timeout_ms: int = 0,
    ) -> HasResult:
        """Quick presence check — NOT the full pipeline (PRD §5, §6a T0)."""
        mode = MatchMode(match)
        device = self.device
        src = (source or "auto").lower()

        # T0: hierarchy selector (short-circuits on first hit)
        if src in ("auto", "hierarchy"):
            if timeout_ms and timeout_ms > 0:
                bounds = device.wait_for(
                    text, match=mode, ignore_case=ignore_case, timeout_ms=timeout_ms
                )
            else:
                bounds = device.find_text(text, match=mode, ignore_case=ignore_case)
            if bounds is not None:
                return HasResult(found=True, source="hierarchy", bounds=bounds, text=text)
            if src == "hierarchy":
                return HasResult(found=False, source="hierarchy")

        # T0→T3: OCR fallback (only on a hierarchy miss)
        if (src in ("auto", "vision")) and (ocr_fallback or src == "vision"):
            hit = self._ocr_contains(device, text, mode, ignore_case)
            if hit is not None:
                return HasResult(found=True, source="ocr", bounds=hit, text=text)

        return HasResult(found=False, source="hierarchy" if src != "vision" else "ocr")

    def _ocr_contains(
        self, device: Device, text: str, mode: MatchMode, ignore_case: bool
    ) -> tuple[int, int, int, int] | None:
        if not self.factory.is_enabled("ocr"):
            return None
        chain = self.factory.build_chain("ocr")
        if not chain.providers:
            return None
        img = device.screenshot()
        try:
            boxes, _ = run_chain(
                chain,
                lambda p: p.recognize(img),  # type: ignore[attr-defined]
                timeout_s=self.config.timeouts.vision_ms / 1000.0,
            )
        except ProviderError as exc:
            logger.info("ocr fallback unavailable: %s", exc)
            return None
        import re as _re

        needle = text if not ignore_case else text.lower()
        for tb in boxes:
            hay = tb.text if not ignore_case else tb.text.lower()
            ok = False
            if mode is MatchMode.exact:
                ok = hay.strip() == needle.strip()
            elif mode is MatchMode.regex:
                flags = _re.IGNORECASE if ignore_case else 0
                ok = _re.search(text, tb.text, flags) is not None
            else:
                ok = needle in hay
            if ok:
                return tb.bounds
        return None

    # ----------------------------------------------------------------- inspect

    def inspect(self, element_id: int) -> Element:
        return self._resolve(element_id)

    def screenshot(self, path: str | None = None, *, annotate: bool = False) -> ActionResult:
        device = self.device
        img = device.screenshot()
        if annotate:
            cached = self._read_cache()
            elements = cached.elements if cached else []
            out = path or self._default_annotate_path(device.serial)
            from . import annotate as annotate_mod

            saved = annotate_mod.annotate(img, elements, out)
            return ActionResult(ok=True, action="screenshot", detail=saved)
        out = path or self._default_annotate_path(device.serial, suffix="screenshot")
        img.save(out)
        return ActionResult(ok=True, action="screenshot", detail=out)

    # ----------------------------------------------------------------- actions

    def _observe(self, result: ActionResult, observe: bool) -> ActionResult:
        """Attach the post-action screen so callers skip a separate ``analyze`` round-trip.

        The folded ``analyze`` also re-populates the id cache, so the agent can act on an id
        from ``result.observation`` immediately (e.g. type → tap send) in one fewer call.
        """
        if observe:
            with contextlib.suppress(Exception):  # observation is a bonus; never fail the action
                obs = self.analyze(source="hierarchy", record=False)
                result.observation = obs
                mem = self._memory
                if mem is not None and self._device is not None:
                    # Recognition-only pass: draws the single-action edge when the
                    # snapshot lands on a known screen; never creates screens.
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
        return result

    def tap(self, element_id: int, *, observe: bool = True) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        step = self._step("tap", el)  # built pre-action: needs the cached package
        self.device.click(cx, cy)
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="tap", id=element_id, target=[cx, cy]), observe
        )

    def long_press(self, element_id: int, *, ms: int = 600, observe: bool = True) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        step = self._step("long-press", el)
        self.device.long_click(cx, cy, ms)
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="long-press", id=element_id, target=[cx, cy]), observe
        )

    def input_text(
        self, element_id: int, text: str, *, submit: bool = False, observe: bool = True
    ) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        # The step records the field's SHAPE only — the typed value is never persisted
        # (PRD §6b privacy; observe_action strips `text` defensively too).
        step = self._step("input", el, submit=submit)
        self.device.input_text(cx, cy, text, clear=True, submit=submit)
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="input", id=element_id, detail=text), observe
        )

    def clear(self, element_id: int, *, observe: bool = True) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        step = self._step("clear", el)
        self.device.click(cx, cy)
        self.device.clear_text()
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(ActionResult(ok=True, action="clear", id=element_id), observe)

    def swipe(
        self,
        direction: str | None = None,
        *,
        from_id: int | None = None,
        percent: int = 50,
        coords: tuple[int, int, int, int] | None = None,
        observe: bool = True,
    ) -> ActionResult:
        device = self.device
        if coords is not None:
            x1, y1, x2, y2 = coords
            step = self._step("swipe", arg="coords")
            device.swipe(x1, y1, x2, y2)
            self._invalidate_cache()
            self._record_action_safe(step)
            return self._observe(
                ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]), observe
            )
        if direction is None:
            raise UsageError("swipe needs a direction or --coords", hint="e.g. `aua swipe up`")
        w, h = device.window_size()
        if from_id is not None:
            cx, cy = self._resolve(from_id).center
        else:
            cx, cy = w // 2, h // 2
        ax = int(w * percent / 200)
        ay = int(h * percent / 200)
        d = direction.lower()
        if d == "up":
            x1, y1, x2, y2 = cx, cy + ay, cx, cy - ay
        elif d == "down":
            x1, y1, x2, y2 = cx, cy - ay, cx, cy + ay
        elif d == "left":
            x1, y1, x2, y2 = cx + ax, cy, cx - ax, cy
        elif d == "right":
            x1, y1, x2, y2 = cx - ax, cy, cx + ax, cy
        else:
            raise UsageError(f"unknown swipe direction '{direction}'", hint="up|down|left|right")
        clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
        x1, x2 = clamp(x1, 0, w - 1), clamp(x2, 0, w - 1)
        y1, y2 = clamp(y1, 0, h - 1), clamp(y2, 0, h - 1)
        step = self._step("swipe", arg=d)
        device.swipe(x1, y1, x2, y2)
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]), observe
        )

    def scroll_to(
        self,
        query: str,
        *,
        match: str = "contains",
        ignore_case: bool = False,
        observe: bool = True,
    ) -> ActionResult:
        step = self._step("scroll-to", arg=query)
        found = self.device.scroll_to(query, match=MatchMode(match), ignore_case=ignore_case)
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(
            ActionResult(
                ok=found is not None,
                action="scroll-to",
                detail=query,
                target=list(found) if found else None,
            ),
            observe,
        )

    def key(self, name: str, *, observe: bool = True) -> ActionResult:
        step = self._step("key", arg=name)
        self.device.press(name)
        self._invalidate_cache()
        self._record_action_safe(step)
        return self._observe(ActionResult(ok=True, action="key", detail=name), observe)

    def open_link(self, uri: str, *, observe: bool = True) -> ActionResult:
        """Open a deeplink URI (jump straight to a screen / trigger an app action).

        A latency shortcut over tapping through the UI. The deeplink is remembered in the
        app's playbook (§6b) so it can be suggested next time.
        """
        step = self._step("open-link", arg=uri)
        self.device.open_link(uri)
        self._invalidate_cache()
        self._record_action_safe(step)
        self._remember_deeplink_safe(uri)
        return self._observe(ActionResult(ok=True, action="open-link", detail=uri), observe)

    def _remember_deeplink_safe(self, uri: str) -> None:
        mem = self._memory
        if mem is None or self._device is None:
            return
        pkg = self._cached_package() or self.current_package()
        if not pkg:
            return
        with contextlib.suppress(Exception):  # playbook is a bonus; never fail the action
            mem.remember_deeplink(pkg, uri)

    def wait(
        self,
        *,
        for_: str | None = None,
        idle: bool = False,
        timeout_ms: int = 5000,
        match: str = "contains",
        ignore_case: bool = False,
    ) -> ActionResult:
        device = self.device
        if idle:
            device.wait_idle(timeout_ms)
            return ActionResult(ok=True, action="wait", detail="idle")
        if not for_:
            raise UsageError("wait needs --for <text> or --idle")
        found = device.wait_for(
            for_, match=MatchMode(match), ignore_case=ignore_case, timeout_ms=timeout_ms
        )
        return ActionResult(
            ok=found is not None,
            action="wait",
            detail=for_,
            target=list(found) if found else None,
        )

    def app(self, action: str, *, package: str | None = None) -> ActionResult:
        device = self.device
        a = action.lower()
        if a in ("foreground", "current"):
            info = device.current_app()
            return ActionResult(ok=True, action=f"app-{a}", detail=json.dumps(info))
        if a == "launch":
            if not package:
                raise UsageError("app launch needs a package name")
            device.launch_app(package)
            self._invalidate_cache()
            return ActionResult(ok=True, action="app-launch", detail=package)
        if a == "stop":
            if not package:
                raise UsageError("app stop needs a package name")
            device.stop_app(package)
            self._invalidate_cache()
            return ActionResult(ok=True, action="app-stop", detail=package)
        raise UsageError(f"unknown app action '{action}'", hint="foreground|launch|stop|current")

    # ----------------------------------------------------------------- doctor

    def provider_status(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for kind in ("ocr", "detection", "grounding", "planner"):
            chain_names = self.factory.chain_names(kind)
            enabled = self.factory.is_enabled(kind)
            items: list[dict[str, Any]] = []
            for name in registered_names(kind):
                try:
                    prov = self.factory.create(kind, name)
                    avail = prov.is_available()
                    items.append(
                        {
                            "name": name,
                            "available": avail.ok,
                            "reason": avail.reason,
                            "in_chain": name in chain_names,
                            "kind_enabled": enabled,
                        }
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    items.append(
                        {
                            "name": name,
                            "available": False,
                            "reason": f"init error: {exc}",
                            "in_chain": name in chain_names,
                            "kind_enabled": enabled,
                        }
                    )
            out[kind] = items
        return out

    # ----------------------------------------------------------------- annotate

    def _maybe_annotate(
        self,
        annotate: bool | str | None,
        device: Device,
        elements: list[Element],
        img: ScreenImage | None,
    ) -> str | None:
        if not annotate:
            return None
        from . import annotate as annotate_mod

        if img is None:
            img = device.screenshot()
        out = annotate if isinstance(annotate, str) else self._default_annotate_path(device.serial)
        return annotate_mod.annotate(img, elements, out)

    def _default_annotate_path(self, serial: str, *, suffix: str = "annotated") -> str:
        run_dir = Path(self.config.cache.dir).expanduser() / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        safe = serial.replace(":", "_")
        return str(run_dir / f"{safe}_{suffix}.png")

    # ----------------------------------------------------------------- cache

    def _cache_path(self, serial: str | None = None) -> Path:
        # Resolve the real connected serial on reads (config serial may be null =
        # auto-detected) so a `tap`/`inspect` process keys the same file `analyze`
        # wrote. Writes pass the serial explicitly and never trigger a connect here.
        if serial is None:
            serial = self._device.serial if self._device else self.device.serial
        cache_dir = Path(self.config.cache.dir).expanduser()
        safe = str(serial).replace(":", "_")
        return cache_dir / f"analyze_{safe}.json"

    def _write_cache(self, result: AnalyzeResult) -> None:
        if not self.config.cache.enabled:
            return
        path = self._cache_path(result.meta.device_serial)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.model_dump_json(), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues
            logger.warning("could not write analyze cache: %s", exc)

    def _read_cache(self) -> AnalyzeResult | None:
        path = self._cache_path()
        if not path.is_file():
            return None
        try:
            return AnalyzeResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupt cache
            logger.warning("ignoring corrupt analyze cache: %s", exc)
            return None

    def _invalidate_cache(self) -> None:
        path = self._cache_path()
        with contextlib.suppress(OSError):  # pragma: no cover
            path.unlink(missing_ok=True)

    def _resolve(self, element_id: int) -> Element:
        cached = self._read_cache()
        if cached is None:
            raise ElementNotFoundError(
                "no cached analyze result", hint="Run `aua analyze` first to assign element ids."
            )
        el = cached.element_by_id(element_id)
        if el is None:
            valid = ", ".join(str(e.id) for e in cached.elements[:20]) or "(none)"
            raise ElementNotFoundError(
                f"element id {element_id} is not in the last analyze (valid: {valid})",
                hint="Re-run `aua analyze`; ids change when the screen changes.",
            )
        return el
