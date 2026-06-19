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
from pathlib import Path
from typing import Any

from . import routing
from .config import Config
from .device import Device, connect, list_devices
from .errors import ElementNotFoundError, ProviderError, StabilityTimeout, UsageError
from .memory import AppMemoryStore, NavHints, _shortest_path, redact_label, resolve_goal
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

_PACKAGE_RE = re.compile(r'package="([^"]+)"')


def _package_from_xml(xml: str) -> str | None:
    """Cheap foreground-package guess from a hierarchy dump (avoids an app_current RPC).

    Picks the most common ``package=`` among nodes, ignoring the system UI overlay.
    """
    pkgs = _PACKAGE_RE.findall(xml)
    if not pkgs:
        return None
    counts = Counter(p for p in pkgs if p and p != "com.android.systemui")
    if not counts:
        return pkgs[0]
    return counts.most_common(1)[0][0]


def _parse_tap_label(action: str) -> str | None:
    """Pull the label out of a route action like ``tap 'Apps'`` (None if not a tap)."""
    m = re.match(r"tap '(.+)'$", action)
    return m.group(1) if m else None


def _match_element(elements: list[Element], label: str) -> Element | None:
    """Find the on-screen element a route's ``tap '<label>'`` refers to."""
    for e in elements:  # exact text / content-desc match first
        if (e.text or e.content_desc or "") == label:
            return e
    low = label.lower()
    for e in elements:  # tolerate truncation / case drift on long labels
        t = (e.text or e.content_desc or "").lower()
        if t and (t.startswith(low) or low in t):
            return e
    return None


def _goto_handoff(
    goal: str,
    target: str,
    code: str,
    hops: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    res: AnalyzeResult,
) -> dict[str, Any]:
    """Stop driving and return enough state for the caller to continue manually."""
    return {
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
        "hint": "route diverged — continue with `aua analyze` + `aua tap`",
    }


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
        return hierarchy.parse_hierarchy(xml, (w, h)), _package_from_xml(xml)

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

        known_screen, hints = self._record_screen_safe(
            device, package, activity, elements, tier_used, h
        )
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

    def _record_action_safe(self, summary: str) -> None:
        mem = self._memory
        if mem is None or self._device is None:
            return
        try:
            mem.observe_action(self._device.serial, summary)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("memory record_action failed: %s", exc)

    def _action_label(self, element: Element | None) -> str:
        if element is None:
            return ""
        lab = redact_label(element, redact=self.config.memory.redact)
        return f"'{lab}'" if lab else f"[{element.type}]"

    def current_package(self) -> str | None:
        """Best-effort foreground package (for ``aua map`` without ``--app``)."""
        try:
            pkg = self.device.current_app().get("package")
        except Exception:  # pragma: no cover - device hiccup
            pkg = None
        if pkg:
            return pkg
        try:
            return _package_from_xml(self.device.dump_hierarchy())
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

    def goto(self, goal: str, *, plan: bool = False, max_steps: int = 8) -> dict[str, Any]:
        """Drive to a remembered screen via the app map (PRD §6b).

        Resolves *goal* to a known screen, then taps along the shortest route from the
        current screen, re-analyzing and verifying ``known_screen`` after each hop. On any
        mismatch it stops and hands back the remaining route + current screen, so the caller
        can continue manually. ``plan=True`` returns the route without acting.
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
        if current == target:
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
        hops: list[dict[str, Any]] = []
        for i, edge in enumerate(path):
            if i >= max_steps:
                return _goto_handoff(goal, target, "max_steps", hops, route[i:], res)
            label = _parse_tap_label(edge.action)
            if label is None:
                return _goto_handoff(goal, target, "unsupported_action", hops, route[i:], res)
            el = _match_element(res.elements, label)
            if el is None:
                return _goto_handoff(goal, target, "element_not_found", hops, route[i:], res)
            self.tap(el.id)
            with contextlib.suppress(StabilityTimeout):
                self.wait_stable(settle_ms=500, timeout_ms=8000)
            res = self.analyze(source="hierarchy")
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
                return _goto_handoff(goal, target, "wrong_screen", hops, route[i + 1 :], res)
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

    def close(self) -> None:
        """Release the device (and its on-device uiautomator2 server). Idempotent."""
        dev = self._device
        if dev is not None:
            with contextlib.suppress(Exception):
                dev.close()
            self._device = None

    def orient(self) -> dict[str, Any]:
        """What the tool already knows about the foreground app (for ``daemon start``)."""
        mem = self._memory
        pkg = self.current_package()
        out: dict[str, Any] = {"package": pkg, "known": False}
        if mem is None or not pkg:
            return out
        app = mem.load(pkg)
        if app is None or not app.screens:
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
                result.observation = self.analyze(source="hierarchy")
        return result

    def tap(self, element_id: int, *, observe: bool = False) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        self.device.click(cx, cy)
        self._invalidate_cache()
        self._record_action_safe(f"tap {self._action_label(el)}")
        return self._observe(
            ActionResult(ok=True, action="tap", id=element_id, target=[cx, cy]), observe
        )

    def long_press(self, element_id: int, *, ms: int = 600, observe: bool = False) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        self.device.long_click(cx, cy, ms)
        self._invalidate_cache()
        self._record_action_safe(f"long-press {self._action_label(el)}")
        return self._observe(
            ActionResult(ok=True, action="long-press", id=element_id, target=[cx, cy]), observe
        )

    def input_text(
        self, element_id: int, text: str, *, submit: bool = False, observe: bool = False
    ) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        self.device.input_text(cx, cy, text, clear=True, submit=submit)
        self._invalidate_cache()
        # Record the action SHAPE only — the typed value is never persisted (PRD §6b privacy).
        self._record_action_safe("input '<filled>'" + (" + send" if submit else ""))
        return self._observe(
            ActionResult(ok=True, action="input", id=element_id, detail=text), observe
        )

    def clear(self, element_id: int, *, observe: bool = False) -> ActionResult:
        el = self._resolve(element_id)
        cx, cy = el.center
        self.device.click(cx, cy)
        self.device.clear_text()
        self._invalidate_cache()
        self._record_action_safe(f"clear {self._action_label(el)}")
        return self._observe(ActionResult(ok=True, action="clear", id=element_id), observe)

    def swipe(
        self,
        direction: str | None = None,
        *,
        from_id: int | None = None,
        percent: int = 50,
        coords: tuple[int, int, int, int] | None = None,
        observe: bool = False,
    ) -> ActionResult:
        device = self.device
        if coords is not None:
            x1, y1, x2, y2 = coords
            device.swipe(x1, y1, x2, y2)
            self._invalidate_cache()
            self._record_action_safe("swipe coords")
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
        device.swipe(x1, y1, x2, y2)
        self._invalidate_cache()
        self._record_action_safe(f"swipe {d}")
        return self._observe(
            ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]), observe
        )

    def scroll_to(
        self,
        query: str,
        *,
        match: str = "contains",
        ignore_case: bool = False,
        observe: bool = False,
    ) -> ActionResult:
        found = self.device.scroll_to(query, match=MatchMode(match), ignore_case=ignore_case)
        self._invalidate_cache()
        self._record_action_safe(f"scroll-to '{query}'")
        return self._observe(
            ActionResult(
                ok=found is not None,
                action="scroll-to",
                detail=query,
                target=list(found) if found else None,
            ),
            observe,
        )

    def key(self, name: str, *, observe: bool = False) -> ActionResult:
        self.device.press(name)
        self._invalidate_cache()
        self._record_action_safe(f"key '{name}'")
        return self._observe(ActionResult(ok=True, action="key", detail=name), observe)

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
        d: Any = getattr(device, "_d", None)
        a = action.lower()
        if a in ("foreground", "current"):
            info = device.current_app()
            return ActionResult(ok=True, action=f"app-{a}", detail=json.dumps(info))
        if d is None:
            raise UsageError("app launch/stop requires a real device")
        if a == "launch":
            if not package:
                raise UsageError("app launch needs a package name")
            d.app_start(package)
            return ActionResult(ok=True, action="app-launch", detail=package)
        if a == "stop":
            if not package:
                raise UsageError("app stop needs a package name")
            d.app_stop(package)
            return ActionResult(ok=True, action="app-stop", detail=package)
        raise UsageError(f"unknown app action '{action}'", hint="foreground|launch|stop|current")

    # ----------------------------------------------------------------- doctor

    def provider_status(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for kind in ("ocr", "detection", "grounding"):
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
