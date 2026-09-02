"""Perceiving the screen for `aua analyze`: hierarchy, OCR and vision capture, the analyze pipeline and its semantic query path, screenshot/inspect/annotate, and perception-provider status.

Engine methods for analyze. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_analyze.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from . import routing
from .device import Device
from .engine_support import logger
from .errors import ElementNotFoundError, ProviderError, UsageError
from .memory import NavHints, _id_tail, screen_skips_ocr
from .providers.base import DetBox, OcrProvider, Point, ScreenAnalysisResult, ScreenImage, TextBox
from .providers.registry import registered_names, run_chain
from .schema import (
    ActionResult,
    AnalyzeResult,
    Element,
    ElementId,
    Meta,
    PathKind,
    Screen,
    ScreenSource,
    Source,
    Tier,
    center_of,
)
from .selectors import drop_redundant_ocr, ocr_added_app_content

if TYPE_CHECKING:
    from .engine import Engine


QUERY_CONFIDENT = 1.0  # all salient tokens / exact phrase present


QUERY_SOFT = 0.5  # best-effort threshold when escalation is exhausted


class _PendingOcr(NamedTuple):
    """Apple Vision work running while the main thread captures the hierarchy."""

    image: ScreenImage
    provider: OcrProvider
    future: Future[list[TextBox]]
    executor: ThreadPoolExecutor
    started_at: float


class _HierarchyObservation(NamedTuple):
    elements: list[Element]
    package: str | None
    # Optional in fact: `_capture_hierarchy` returns None when the dump could not be hashed, and
    # the unchanged-frame check at the read site already guards on the value being falsy.
    xml_hash: str | None
    ocr_texts: list[TextBox]
    ocr_elements: list[Element]
    ocr_provider: str | None
    image: ScreenImage | None


def _detect_lossy_text(elements: list[Any]) -> tuple[bool, str | None]:
    """Did the accessibility tree hand us text it could not represent?

    A U+FFFD in a label means the real glyph never reached us. It happens on
    formula/equation rendering, some custom fonts, and WebView content: prose survives,
    the interesting part becomes "?". Returning that silently is the worst outcome - the
    agent believes it read the screen, reports an observation that omits the very thing
    under test, or starts eyeballing screenshots on its own. Flag it and name the recovery.
    """
    hits = 0
    for e in elements:
        for attr in ("text", "content_desc"):
            v = getattr(e, attr, None)
            if isinstance(v, str) and "\ufffd" in v:
                hits += v.count("\ufffd")
    if not hits:
        return False, None
    return True, (
        f"{hits} unrepresentable character(s) in hierarchy text (U+FFFD): formula/WebView "
        "content did not survive the accessibility tree. Re-read with "
        "`aua analyze --source vision` (OCR) before judging anything that depends on that text."
    )


def _effective_with_image(self: Engine, with_image: bool | str | None) -> bool | str | None:
    """Per-call ``with_image`` overrides the session default; ``False`` forces off."""
    if with_image is False:
        return None
    if with_image is not None:
        return with_image
    return self._default_with_image


def _context(self: Engine) -> tuple[Device, int, int]:
    # window_size is memoized on the device; no app_current RPC on the hot path.
    device = self.device
    w, h = device.window_size()
    return device, w, h


def _capture_hierarchy(
    self: Engine, device: Device, w: int, h: int
) -> tuple[list[Element], str | None, str]:
    perf = self.config.perf
    if perf.prefetch:
        slot = self._prefetch.take()
        if slot is not None:
            xml_hash = hashlib.sha1(slot.xml.encode()).hexdigest()
            return slot.elements, slot.package, xml_hash

    compressed = bool(self.config.device.compressed_hierarchy)
    raw_tree = self.platform.dump_tree(device, compact=compressed)
    tree_hash = hashlib.sha1(raw_tree.encode()).hexdigest()
    normalized = self.platform.normalize_tree(
        raw_tree,
        (w, h),
        ignored_app_ids=self.config.memory.ignore_packages,
    )
    return normalized.elements, normalized.app_id, tree_hash


def _kick_hierarchy_prefetch(self: Engine) -> None:
    """Speculatively dump+parse the hierarchy for the next analyze."""
    if not self.config.perf.prefetch:
        return
    if self._device is None:
        return
    device = self._device
    platform = self.platform
    compressed = bool(self.config.device.compressed_hierarchy)
    owner = self._lease_owner_resolved
    generation = getattr(self._device_use_context, "generation", None)
    try:
        w, h = device.window_size()
    except Exception:  # pragma: no cover - device mid-disconnect
        return

    def dump() -> str:
        with self.device_use_context(
            device.serial,
            owner=owner,
            generation=generation,
        ):
            return platform.dump_tree(device, compact=compressed)

    def parse(raw_tree: str) -> tuple[list[Element], str | None]:
        normalized = platform.normalize_tree(
            raw_tree,
            (w, h),
            ignored_app_ids=self.config.memory.ignore_packages,
        )
        return normalized.elements, normalized.app_id

    self._prefetch.kick(dump, parse)


def _screenshot(self: Engine, *, max_reuse_ms: float = 50.0) -> ScreenImage:
    """Prefer a fresh-enough capture-buffer frame; else take a device screenshot."""
    perf = self.config.perf
    if perf.reuse_capture_frames and self._capture is not None:
        with contextlib.suppress(Exception):
            age = self._capture.latest_age_ms()
            img = self._capture.latest_frame()
            if img is not None and age is not None and age <= max_reuse_ms:
                return img
    return self.device.screenshot()


def _start_hierarchy_ocr(self: Engine, *, with_ocr: bool | None) -> _PendingOcr | None:
    """Start the macOS OCR augmenter before the hierarchy capture begins.

        This intentionally selects only Apple Vision from the configured OCR chain. A
        heavyweight cross-platform OCR fallback must not silently run on every hierarchy
        call; those providers remain available to the ordinary vision fallback.
        """
    want_ocr = self.config.ocr.enabled if with_ocr is None else with_ocr
    if (
        not want_ocr
        or not self.config.ocr.augment_hierarchy
        or not self.factory.is_enabled("ocr")
    ):
        return None
    chain = self.factory.build_chain("ocr")
    provider = next(
        (
            item
            for item in chain.providers
            if item.name == "apple_vision" and isinstance(item, OcrProvider)
        ),
        None,
    )
    if provider is None or not provider.is_available().ok:
        return None
    try:
        # The daemon's rolling capture normally makes this a memory read. Keeping the
        # screenshot on the caller thread avoids concurrent ADB/uiautomator RPCs.
        image = self._screenshot(max_reuse_ms=250.0)
    except Exception as exc:
        logger.info("parallel hierarchy OCR could not capture a screenshot: %s", exc)
        return None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aua-apple-ocr")
    started_at = time.perf_counter()
    future = executor.submit(provider.recognize, image)
    return _PendingOcr(image, provider, future, executor, started_at)


def _finish_hierarchy_ocr(
    self: Engine, pending: _PendingOcr | None
) -> tuple[list[TextBox], str | None, ScreenImage | None]:
    if pending is None:
        return [], None, None
    timed_out = False
    try:
        elapsed = time.perf_counter() - pending.started_at
        timeout = max(0.0, self.config.timeouts.vision_ms / 1000.0 - elapsed)
        texts = pending.future.result(timeout=timeout)
        return texts, pending.provider.name, pending.image
    except FuturesTimeout:
        timed_out = True
        pending.future.cancel()
        logger.warning("parallel hierarchy OCR timed out")
        return [], None, pending.image
    except Exception as exc:
        logger.info("parallel hierarchy OCR unavailable: %s", exc)
        return [], None, pending.image
    finally:
        pending.executor.shutdown(wait=not timed_out, cancel_futures=timed_out)


def _fuse_hierarchy_ocr(
    self: Engine,
    elements: list[Element],
    package: str | None,
    xml_hash: str | None,
    pending: _PendingOcr | None,
) -> _HierarchyObservation:
    """Finish a pending OCR job and fuse kept boxes onto the hierarchy elements."""
    texts, provider, image = self._finish_hierarchy_ocr(pending)
    ocr_elements: list[Element] = []
    if provider is not None:
        from . import merge

        # These elements were just built by this process, so their ids are ordinals.
        start_id = max((int(element.id) for element in elements), default=-1) + 1
        ocr_elements = merge.merge_vision([], texts, start_id=start_id)
        if self.config.ocr.drop_redundant:
            # Withhold readings of text the tree already reports. Provenance is worth
            # keeping where OCR *adds* something - web content, a lossy-text repair, a
            # surface the tree cannot see - but a second copy of text already described
            # is not evidence to reconcile. Measured on one app screen: 14 of 16
            # readings were pure duplication, and one of the remaining two was a misread
            # ("Al" for "AI") that survived only because it differed from the truth.
            # Those cost tokens on every observation and let a wrong label be quoted as
            # fact. See selectors.drop_redundant_ocr for what counts as redundant.
            keep = {id(el) for el in drop_redundant_ocr([*elements, *ocr_elements])}
            ocr_elements = [el for el in ocr_elements if id(el) in keep]
    return _HierarchyObservation(
        elements,
        package,
        xml_hash,
        texts,
        ocr_elements,
        provider,
        image,
    )


def _capture_hierarchy_with_ocr(
    self: Engine,
    device: Device,
    w: int,
    h: int,
    *,
    with_ocr: bool | None,
) -> _HierarchyObservation:
    """Capture hierarchy, optionally fused with Apple OCR.

        ``with_ocr=True`` keeps the parallel overlap (OCR starts before hierarchy).
        Auto mode (``None``) captures hierarchy first, then skips OCR entirely when the
        map already knows this screen is hierarchy-sufficient — experience-based cheap
        analyze without risking unknown screens. Forced ``False`` is hierarchy-only.
        """
    if with_ocr is False:
        elements, package, xml_hash = self._capture_hierarchy(device, w, h)
        return _HierarchyObservation(elements, package, xml_hash, [], [], None, None)

    if with_ocr is True:
        # Caller forced OCR — overlap screenshot OCR with the hierarchy dump.
        pending = self._start_hierarchy_ocr(with_ocr=True)
        try:
            elements, package, xml_hash = self._capture_hierarchy(device, w, h)
        except BaseException:
            if pending is not None:
                pending.future.cancel()
                pending.executor.shutdown(wait=False, cancel_futures=True)
            raise
        return self._fuse_hierarchy_ocr(elements, package, xml_hash, pending)

    # Auto: hierarchy first so we can consult the map before paying for OCR.
    elements, package, xml_hash = self._capture_hierarchy(device, w, h)
    if self._map_skips_ocr(device, package, elements, h):
        return _HierarchyObservation(elements, package, xml_hash, [], [], None, None)
    pending = self._start_hierarchy_ocr(with_ocr=True)
    return self._fuse_hierarchy_ocr(elements, package, xml_hash, pending)


def _map_skips_ocr(
    self: Engine,
    device: Device,
    package: str | None,
    elements: list[Element],
    height: int,
) -> bool:
    """Skip parallel OCR only when map evidence says this known screen never needed it."""
    mem = self._memory
    if mem is None or not package:
        return False
    try:
        name = mem.recognize_screen(
            device.serial,
            package=package,
            elements=elements,
            screen_height=height,
        )
        if not name:
            return False
        app = mem.load(package)
        rec = app.screens.get(name) if app else None
        return bool(rec and screen_skips_ocr(rec))
    except Exception as exc:
        logger.debug("map OCR-skip check failed: %s", exc)
        return False


def _run_vision(
    self: Engine,
    device: Device,
    *,
    with_ocr: bool | None,
    start_id: int = 0,
    image: ScreenImage | None = None,
    ocr_result: tuple[list[TextBox], str] | None = None,
) -> tuple[list[Element], list[str], ScreenImage]:
    from . import merge

    img = image or self._screenshot(max_reuse_ms=80.0)
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
    if ocr_result is not None:
        texts, name = ocr_result
        providers_used.append(name)
    elif want_ocr and self.factory.is_enabled("ocr"):
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


def _repair_lossy_text(self: Engine, device: Device, elements: list[Element]) -> tuple[int, str | None]:
    """Fill in hierarchy labels the accessibility tree could not represent, using OCR.

        The tree sometimes hands back U+FFFD instead of the real glyphs - formula and
        WebView content especially. Element *structure* is fine, only the text is lost, so
        replacing the whole observation with a vision pass would be wasteful and would
        double the payload. Instead run OCR once and graft the recognised text onto the
        elements that are broken, matched by geometric overlap.

        Costs one OCR pass (~145ms with apple_vision) and only when something is actually
        broken. Returns how many labels were repaired and which provider performed it.
        """
    broken = [e for e in elements if e.text is not None and "\ufffd" in e.text]
    if not broken:
        return 0, None
    try:
        img = self._screenshot(max_reuse_ms=250.0)
        chain = self.factory.build_chain("ocr")
        if not chain.providers:
            return 0, None
        texts, name = run_chain(
            chain,
            lambda p: p.recognize(img),  # type: ignore[attr-defined]
            timeout_s=self.config.timeouts.vision_ms / 1000.0,
        )
    except Exception as exc:  # never let a repair attempt break the analyze
        logger.info("lossy-text repair unavailable: %s", exc)
        return 0, None

    def overlap(a: Any, b: Any) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / area_b

    repaired = 0
    for el in broken:
        box = getattr(el, "bounds", None)
        if not box:
            continue
        hits = []
        for tb in texts:
            tbb = getattr(tb, "bounds", None) or getattr(tb, "box", None)
            if not tbb:
                continue
            # Keep OCR text whose box sits mostly inside the broken element.
            if overlap(tuple(box), tuple(tbb)) >= 0.5 and (tb.text or "").strip():
                hits.append((tbb[1], tbb[0], tb.text.strip()))
        if not hits:
            continue
        hits.sort()  # reading order: top-to-bottom, then left-to-right
        merged = " ".join(h[2] for h in hits)
        if merged and merged != el.text:
            el.text = merged
            repaired += 1
    if repaired:
        logger.info("repaired %d lossy label(s) with OCR (%s)", repaired, name)
    return repaired, (name if repaired else None)


def ask_screen(self: Engine, question: str) -> dict[str, Any]:
    """Ask the configured grounding model about screenshot + current element graph."""
    question = question.strip()
    if not question:
        raise UsageError("ask needs a non-empty screen question")
    if not self.factory.is_enabled("grounding"):
        raise UsageError(
            "screen questions need grounding.enabled: true",
            hint="Enable grounding and configure a screen-analysis provider such as gemini or openai.",
        )

    t0 = time.perf_counter()
    device, width, height = self._context()
    observation = self._capture_hierarchy_with_ocr(device, width, height, with_ocr=None)
    elements = observation.elements + observation.ocr_elements
    package = observation.package
    app = device.current_app()
    package = app.get("package") or package
    activity = app.get("activity") or None
    image = observation.image or self._screenshot(max_reuse_ms=80.0)
    graph: list[dict[str, Any]] = []
    for element in elements:
        item: dict[str, Any] = {
            "id": element.id,
            "type": element.type,
            "bounds": list(element.bounds),
        }
        if element.text:
            item["text"] = element.text
        if element.content_desc:
            item["desc"] = element.content_desc
        if element.resource_id:
            item["rid"] = _id_tail(element.resource_id)
        if element.source is not Source.hierarchy:
            item["source"] = element.source.value
        if element.confidence is not None:
            item["confidence"] = round(element.confidence, 4)
        for flag in ("clickable", "checkable", "checked", "selected", "scrollable"):
            value = getattr(element, flag)
            if value:
                item[flag] = True
        graph.append(item)
    chain = self.factory.build_chain("grounding")
    result, provider = run_chain(
        chain,
        lambda item: item.ask(image, question, graph),  # type: ignore[attr-defined]
        timeout_s=self.config.timeouts.grounding_ms / 1000.0,
    )
    if not isinstance(result, ScreenAnalysisResult):  # pragma: no cover - contract guard
        raise ProviderError("grounding", [(provider, "invalid screen-analysis result")])
    return {
        "question": question,
        "screen": {
            "width": width,
            "height": height,
            "package": package,
            "activity": activity,
        },
        "provider": provider,
        "model": result.model,
        "perception_providers": (
            [observation.ocr_provider] if observation.ocr_provider else []
        ),
        "duration_ms": round((time.perf_counter() - t0) * 1000),
        "usage": result.usage,
        "input_image": result.input_image,
        "graph_elements": len(graph),
        "analysis": result.analysis,
    }


def _resolve_pins(self: Engine, source: str | None, strategy: str | None) -> tuple[bool, bool, bool]:
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


def _attach_visual_identity(
    device: Device,
    elements: list[Element],
    image: ScreenImage | None,
    width: int,
    height: int,
) -> tuple[list[Element], ScreenImage | None, bool]:
    """Give unlabeled actionable controls pixel keys from one shared screenshot.

        ``Device.screenshot`` is the platform-neutral runtime contract used by every other
        perception path. A capture failure must degrade to the existing geometry key rather
        than turn a hierarchy analysis into an error.
        """
    from .identity import attach_visual_stable_keys, needs_visual_stable_key

    needed = any(needs_visual_stable_key(element) for element in elements)
    if not needed:
        return elements, image, False
    if image is None:
        try:
            image = device.screenshot()
        except Exception as exc:
            logger.info("visual element identity could not capture a screenshot: %s", exc)
            return elements, None, True
    try:
        return (
            attach_visual_stable_keys(elements, image, screen_size=(width, height)),
            image,
            True,
        )
    except Exception as exc:
        logger.info("visual element identity could not fingerprint crops: %s", exc)
        return elements, image, True


def analyze(
    self: Engine,
    *,
    source: str = "auto",
    with_ocr: bool | None = None,
    query: str | None = None,
    annotate: bool | str | None = None,
    with_image: bool | str | None = None,
    strategy: str | None = None,
    cheap: bool = False,
    deep: bool = False,
    no_cache: bool = False,
    record: bool = True,
    record_ids: bool = True,
) -> AnalyzeResult:
    wi = self._effective_with_image(with_image)
    if wi:
        return self._with_raw_image(
            self.analyze(
                source=source,
                with_ocr=with_ocr,
                query=query,
                annotate=annotate,
                strategy=strategy,
                cheap=cheap,
                deep=deep,
                no_cache=no_cache,
                record=record,
                record_ids=record_ids,
                with_image=False,  # already applying session/per-call image below
            ),
            wi,
            image=self._last_analyze_image,
        )
    self._last_analyze_image = None
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
        record_ids=record_ids,
    )


def _analyze_screen(
    self: Engine,
    *,
    ceiling: Tier,
    force_hierarchy: bool,
    force_vision: bool,
    with_ocr: bool | None,
    annotate: bool | str | None,
    no_cache: bool,
    record: bool = True,
    record_ids: bool = True,
) -> AnalyzeResult:
    t0 = time.perf_counter()
    device, w, h = self._context()
    if self.config.memory.enabled:
        # Re-check a long-lived daemon's boot identity before it reads or writes any
        # serial-scoped cursor. This is a cheap no-op for the same token and retries a
        # transiently unreadable first token; a reboot on the same serial clears history.
        _ = self._memory
        self._claim_memory_session()
    providers_used: list[str] = []
    img: ScreenImage | None = None
    package: str | None = None
    activity: str | None = None

    elements: list[Element] = []
    hierarchy_elements: list[Element] = []
    hierarchy_observation: _HierarchyObservation | None = None
    screen_source = ScreenSource.hierarchy
    tier_used = Tier.hierarchy
    path = PathKind.hierarchy

    if not force_vision:
        hierarchy_observation = self._capture_hierarchy_with_ocr(
            device, w, h, with_ocr=with_ocr
        )
        hierarchy_elements = hierarchy_observation.elements
        elements = hierarchy_elements + hierarchy_observation.ocr_elements
        package = hierarchy_observation.package
        xml_hash = hierarchy_observation.xml_hash
        img = hierarchy_observation.image
        elements, img, visual_identity_needed = self._attach_visual_identity(
            device, elements, img, w, h
        )
        hierarchy_elements = elements[: len(hierarchy_elements)]
        if hierarchy_observation.ocr_provider:
            providers_used.append(hierarchy_observation.ocr_provider)
            screen_source = ScreenSource.mixed
        if (
            self.config.perf.skip_unchanged_analyze
            and xml_hash
            and xml_hash == self._last_hierarchy_hash
            and self._last_analyze_result is not None
            and not no_cache
            # A warm daemon can move between apps whose accessibility dump happens to
            # hash the same during a transition. Reusing the previous payload in that case
            # creates an impossible observation: the new hierarchy under the old package.
            # Package identity is part of a screen observation, not optional metadata.
            and package == self._last_analyze_result.screen.package
            # Identical accessibility XML does not imply identical pixels (canvas,
            # charts, video, custom rendering). Current OCR must reach the caller.
            and not hierarchy_observation.ocr_provider
            # XML equality cannot reuse visual identities when an unlabeled control's
            # crop changed underneath the same hierarchy. Exact keys need not match for
            # later fuzzy remapping, but a changed key must run the normal result path.
            and (
                not visual_identity_needed
                or tuple(element.stable_key for element in elements)
                == tuple(element.stable_key for element in self._last_analyze_result.elements)
            )
        ):
            prev = self._last_analyze_result
            # Reusing the PAYLOAD is fine — the tree really is identical — but the memory
            # side-effects are not the tree's properties and must still run:
            #  - `known_screen`: the map learns between calls, so the first analyze of a new
            #    screen answers None and every later one would repeat that None forever.
            #  - a pending route deferred by a mid-transition observe snapshot is drawn by
            #    the NEXT recording analyze; skipping it dropped the edge silently, so the
            #    map stopped learning exactly when the screen sat still.
            # `_record_screen_safe` has its own unchanged-screen fast path, so this is a
            # map read rather than a re-record.
            #  - `slow_controls`, `flows`, and the map hints: same category as `known_screen`.
            #    A cost is learned when an action is measured and a flow can be saved at any
            #    moment, both of which happen after the analyze whose payload is being
            #    reused, so carrying the previous copies over reported stale memory for as
            #    long as the screen sat still — and a still screen is exactly when a caller
            #    is choosing what to do next.
            known = prev.meta.known_screen
            hints = None
            if record:
                known, hints = self._record_screen_safe(
                    device, package, activity, prev.elements, tier_used, h
                )
            # The hints this call already produced answer every remaining memory-derived
            # field, so refreshing them costs nothing — they were being discarded. A
            # non-recording snapshot has no hints, and then the previous values stand:
            # refreshing must not become erasing, or a mid-transition observe would report
            # that the map had forgotten its routes.
            learned: dict[str, Any] = {
                "slow_controls": self._slow_controls_safe(known, package=package),
                "flows": self._flows_for(package),
            }
            if hints is not None:
                learned.update(
                    known_routes=hints.known_routes,
                    suggested_gotos=hints.suggested_gotos,
                    suggested_deeplinks=hints.suggested_deeplinks,
                    research_tasks=hints.research_tasks,
                    ask=hints.ask,
                    map_hint=hints.map_hint,
                )
            reused = prev.model_copy(
                update={
                    "meta": prev.meta.model_copy(
                        update={
                            "duration_ms": int((time.perf_counter() - t0) * 1000),
                            "unchanged": True,
                            "fingerprint": xml_hash,
                            "known_screen": known,
                            **learned,
                            "via": "hierarchy-unchanged",
                            "element_diff": {
                                "added": [],
                                "removed": [],
                                "changed": [],
                                "prev_count": len(prev.elements),
                                "curr_count": len(prev.elements),
                            }
                            if self.config.perf.differential
                            else prev.meta.element_diff,
                        }
                    )
                }
            )
            # Reusing the payload must not skip the side effect callers depend on: every
            # action invalidates the id cache, so an unchanged screen returned straight
            # from memory left NOTHING on disk and the next `tap <id>` died with "no
            # cached analyze result". The ids are only usable because analyze persists them.
            if not no_cache:
                self._write_cache(reused)
            self._last_analyze_image = img
            return reused
    else:
        xml_hash = None
        visual_identity_needed = False

    use_vision = force_vision
    xml_dump: str | None = None
    if not force_vision and not force_hierarchy:
        decision = self._gate_decide(hierarchy_elements, package=package, activity=activity)
        if decision.use_vision and routing.allows(Tier.vision, ceiling):
            # Prefer WebView DOM/a11y enrichment over OCR when the tree looks hollow.
            wv_cfg = self.config.perception.webview
            if wv_cfg.enabled and self.platform.supports("webview"):
                webview_mod = self.platform.capability("webview")

                xml_dump = self.platform.dump_tree(
                    device,
                    compact=bool(self.config.device.compressed_hierarchy),
                )
                if webview_mod.should_try_webview(hierarchy_elements, xml_dump):
                    shell = None
                    if wv_cfg.cdp:
                        shell = lambda cmd: str(  # noqa: E731
                            device.shell(cmd) if hasattr(device, "shell") else ""
                        )
                    enriched = webview_mod.enrich(
                        xml_dump,
                        screen_size=(w, h),
                        shell=shell,
                        cdp=wv_cfg.cdp,
                    )
                    if len(enriched) >= wv_cfg.min_elements:
                        hierarchy_elements = enriched
                        elements = enriched + (
                            hierarchy_observation.ocr_elements
                            if hierarchy_observation is not None
                            else []
                        )
                        screen_source = (
                            ScreenSource.mixed
                            if hierarchy_observation is not None
                            and hierarchy_observation.ocr_provider
                            else ScreenSource.hierarchy
                        )
                        path = PathKind.hierarchy
                        providers_used.append("webview")
                        logger.info(
                            "webview enrichment: %d elements (skipping vision)", len(enriched)
                        )
                        use_vision = False
                    else:
                        use_vision = True
                        logger.info("gate → vision: %s", decision.reason)
                else:
                    use_vision = True
                    logger.info("gate → vision: %s", decision.reason)
            else:
                use_vision = True
                logger.info("gate → vision: %s", decision.reason)
        elif decision.use_vision:
            logger.info("gate wants vision but ceiling=%s; staying hierarchy", ceiling.value)

    if use_vision:
        # slow fallback path: fetch full app context (incl. activity)
        app = device.current_app()
        package = app.get("package") or package
        activity = app.get("activity") or None
        precomputed_ocr = None
        if hierarchy_observation is not None and hierarchy_observation.ocr_provider:
            precomputed_ocr = (
                hierarchy_observation.ocr_texts,
                hierarchy_observation.ocr_provider,
            )
        # These elements were just built by this process, so their ids are ordinals.
        start_id = max((int(element.id) for element in elements), default=-1) + 1
        vis_elements, vision_providers, img = self._run_vision(
            device,
            with_ocr=with_ocr,
            start_id=start_id,
            image=img,
            ocr_result=precomputed_ocr,
        )
        for provider in vision_providers:
            if provider not in providers_used:
                providers_used.append(provider)
        if hierarchy_observation is not None:
            # OCR already exists as its own raw pool. Keep detection elements (including
            # labels associated from OCR), but do not append the same OCR boxes a third time.
            vis_elements = [el for el in vis_elements if el.source is not Source.ocr]
            elements = elements + vis_elements
            screen_source = ScreenSource.mixed
        else:
            elements = vis_elements
            screen_source = ScreenSource.vision
        tier_used = Tier.vision
        path = PathKind.vision

    # WebView enrichment and vision can add controls after the first hierarchy pass. Reuse
    # its screenshot when possible and fingerprint only the metadata-empty actionable ones.
    elements, img, _visual_identity_needed = self._attach_visual_identity(
        device, elements, img, w, h
    )

    if record:
        ocr_helped: bool | None = None
        if hierarchy_observation is not None and with_ocr is not False:
            # Evidence for experience-based OCR skip: only count visits where OCR was
            # allowed. Forced hierarchy-only paths (goto hops) must not inflate the score.
            if hierarchy_observation.ocr_provider is not None:
                # Not `bool(ocr_elements)`: the status-bar clock is read on every
                # screen and never matches the tree's digits, so it survives every
                # redundancy test and would score every visit as "OCR helped" -
                # pinning hierarchy_only_ok at zero and disabling the skip forever.
                ocr_helped = ocr_added_app_content(
                    [*hierarchy_observation.elements, *hierarchy_observation.ocr_elements]
                )
            else:
                ocr_helped = False  # skipped (map) or unavailable → hierarchy alone
        known_screen, hints = self._record_screen_safe(
            device,
            package,
            activity,
            elements,
            tier_used,
            h,
            ocr_helped=ocr_helped,
        )
    else:
        # An observe snapshot taken right after an action can be mid-transition; never
        # let it pollute memory with a transient screen (it's just fresh ids for the agent).
        known_screen, hints = None, None
    annotated = self._maybe_annotate(annotate, device, elements, img)
    ediff = None
    if self.config.perf.differential and self._last_analyze_elements is not None:
        from .perf import element_diff as _element_diff

        with contextlib.suppress(Exception):
            ediff = _element_diff(self._last_analyze_elements, elements)
    self._last_analyze_elements = list(elements)

    from .perf import elements_fingerprint

    fp = (
        None
        if hierarchy_observation is not None and hierarchy_observation.ocr_provider
        else xml_hash
    )
    if not fp:
        with contextlib.suppress(Exception):
            fp = elements_fingerprint(elements)

    # Auto-recover before reporting: a fast answer that is not usable is not an answer.
    # Only pays the OCR cost when the tree actually handed back broken text.
    # When parallel OCR ran, keep the hierarchy text untouched and expose raw OCR boxes
    # alongside it. The older repair fallback remains for hosts without Apple Vision.
    _repaired, _repair_provider = (
        self._repair_lossy_text(device, elements)
        if not use_vision
        and not (hierarchy_observation is not None and hierarchy_observation.ocr_provider)
        else (0, None)
    )
    if _repair_provider and _repair_provider not in providers_used:
        # Provenance matters: text that came from OCR is not text the app exposed.
        providers_used.append(_repair_provider)
    _lossy, _lossy_hint = _detect_lossy_text(elements)
    if _lossy and hierarchy_observation is not None and hierarchy_observation.ocr_provider:
        _lossy_hint = (
            "The hierarchy contains unrepresentable text; raw Apple Vision OCR elements "
            "are included alongside it with source=ocr. Compare both observations."
        )
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
            slow_controls=self._slow_controls_safe(known_screen, package=package),
            suggested_deeplinks=hints.suggested_deeplinks if hints else [],
            research_tasks=hints.research_tasks if hints else [],
            flows=self._flows_for(package),
            ask=hints.ask if hints else None,
            map_hint=hints.map_hint if hints else None,
            capture_hint=self._capture_hint(),
            lossy_text=_lossy,
            lossy_hint=_lossy_hint,
            ocr_repaired=_repaired,
            annotated_image=annotated,
            device_serial=device.serial,
            device_locale=device.device_locale(),
            element_diff=ediff,
            unchanged=False,
            fingerprint=fp,
            via=path.value if hasattr(path, "value") else str(path),
        ),
    )
    if xml_hash:
        self._last_hierarchy_hash = xml_hash
    self._last_analyze_result = result
    self._last_analyze_image = img
    if record_ids:
        # Recording is not the same decision as reuse. `no_cache` means "do not hand me a
        # stale result"; it used to also mean "do not record this one", so a caller who
        # asked for a fresh screen received ids that the next numeric action validated
        # against the *previous* screen. Only AUA's own internal freshness reads opt out,
        # because their observation is never published to a caller.
        self._write_cache(result)
    if self.config.perf.prefetch:
        self._kick_hierarchy_prefetch()
    return result


def _gate_decide(
    self: Engine,
    elements: list[Element],
    *,
    package: str | None,
    activity: str | None,
) -> Any:
    from . import gate
    from .perf import GateCache

    cfg = self.config.perception.gate
    if self.config.perf.gate_cache:
        key = GateCache.key(elements, package=package, activity=activity)
        hit = self._gate_cache.get(key)
        if hit is not None:
            return hit
        decision = gate.decide(elements, package=package, activity=activity, cfg=cfg)
        self._gate_cache.put(key, decision)
        return decision
    return gate.decide(elements, package=package, activity=activity, cfg=cfg)


def _analyze_query(
    self: Engine,
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
    t0 = time.perf_counter()
    device, w, h = self._context()
    package: str | None = None
    activity: str | None = None
    providers_used: list[str] = []
    pool: list[Element] = []
    hierarchy_elements: list[Element] = []
    hierarchy_observation: _HierarchyObservation | None = None
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
        hierarchy_observation = self._capture_hierarchy_with_ocr(
            device, w, h, with_ocr=with_ocr
        )
        hierarchy_elements = hierarchy_observation.elements
        pool = hierarchy_elements + hierarchy_observation.ocr_elements
        package = hierarchy_observation.package
        img = hierarchy_observation.image
        if hierarchy_observation.ocr_provider:
            providers_used.append(hierarchy_observation.ocr_provider)
            screen_source = ScreenSource.mixed
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
                screen_source,
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
        decision = self._gate_decide(hierarchy_elements, package=package, activity=activity)
        kind = routing.classify_query(query)
        want_vision = decision.use_vision or kind is routing.QueryKind.visual or pin_grounding

    if want_vision and routing.allows(Tier.vision, ceiling):
        app = device.current_app()
        package = app.get("package") or package
        activity = app.get("activity") or None
        precomputed_ocr = None
        if hierarchy_observation is not None and hierarchy_observation.ocr_provider:
            precomputed_ocr = (
                hierarchy_observation.ocr_texts,
                hierarchy_observation.ocr_provider,
            )
        # Built in this process: ordinals by construction.
        start_id = max((int(element.id) for element in pool), default=-1) + 1
        vis_elements, vprov, img = self._run_vision(
            device,
            with_ocr=with_ocr,
            start_id=start_id,
            image=img,
            ocr_result=precomputed_ocr,
        )
        for provider in vprov:
            if provider not in providers_used:
                providers_used.append(provider)
        if hierarchy_observation is not None:
            vis_elements = [el for el in vis_elements if el.source is not Source.ocr]
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
    self: Engine,
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
    elements, img, _visual_identity_needed = self._attach_visual_identity(
        device, elements, img, w, h
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
            slow_controls=self._slow_controls_safe(known_screen, package=package),
            suggested_deeplinks=hints.suggested_deeplinks if hints else [],
            research_tasks=hints.research_tasks if hints else [],
            flows=self._flows_for(package),
            ask=hints.ask if hints else None,
            map_hint=hints.map_hint if hints else None,
            capture_hint=self._capture_hint(),
            annotated_image=annotated,
            device_serial=device.serial,
            device_locale=device.device_locale(),
        ),
    )
    if not no_cache:
        self._write_cache(result)
    self._last_analyze_image = img
    return result


def _match_query(self: Engine, query: str, elements: list[Element]) -> tuple[Element | None, float]:
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
    self: Engine, loc: Point | DetBox | None, pool: list[Element], w: int, h: int
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


def inspect(self: Engine, element_id: ElementId) -> Element:
    return self._resolve(element_id)


def screenshot(self: Engine, path: str | None = None, *, annotate: bool = False) -> ActionResult:
    device = self.device
    img = self.platform.capture_screenshot(device)
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


def provider_status(self: Engine) -> dict[str, list[dict[str, Any]]]:
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


def _with_raw_image(
    self: Engine,
    result: AnalyzeResult,
    with_image: bool | str,
    *,
    image: ScreenImage | None = None,
) -> AnalyzeResult:
    img = image or self._screenshot(max_reuse_ms=80.0)
    explicit = isinstance(with_image, str)
    out = (
        with_image
        if isinstance(with_image, str)
        else self._default_annotate_path(self.device.serial, suffix="screen", timestamped=True)
    )
    img.save(out)
    if not explicit:
        # Frames are timestamped so a before/after pair cannot clobber itself, which makes
        # the directory grow without a bound — fine while this was opt-in, not fine now that
        # every analyze writes one. A caller-named path is left alone: that file is the
        # caller's, and pruning someone else's directory is not this function's business.
        self._prune_run_frames(self.device.serial, suffix="screen")
    result.meta.raw_image = out
    return result


def _prune_run_frames(self: Engine, serial: str, *, suffix: str) -> None:
    """Keep only the newest :attr:`MAX_RUN_FRAMES` auto-named frames for this device.

        Best-effort by design: a frame that cannot be deleted is a housekeeping problem, never
        a reason to fail the analyze that produced it.
        """
    try:
        run_dir = Path(self.config.cache.dir).expanduser() / "runs"
        safe = serial.replace(":", "_")
        frames = sorted(
            run_dir.glob(f"{safe}_{suffix}_*.png"),
            key=lambda f: f.name,
            reverse=True,
        )
        for stale in frames[self.MAX_RUN_FRAMES :]:
            with contextlib.suppress(OSError):
                stale.unlink()
    except Exception:  # noqa: BLE001 - housekeeping must never break perception
        logger.debug("could not prune run frames for %s", serial, exc_info=True)


def _maybe_annotate(
    self: Engine,
    annotate: bool | str | None,
    device: Device,
    elements: list[Element],
    img: ScreenImage | None,
) -> str | None:
    if not annotate:
        return None
    from . import annotate as annotate_mod

    if img is None:
        img = self._screenshot(max_reuse_ms=80.0)
    out = annotate if isinstance(annotate, str) else self._default_annotate_path(device.serial)
    return annotate_mod.annotate(img, elements, out)


def _default_annotate_path(
    self: Engine, serial: str, *, suffix: str = "annotated", timestamped: bool = False
) -> str:
    run_dir = Path(self.config.cache.dir).expanduser() / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    safe = serial.replace(":", "_")
    if timestamped:
        # Sequential captures (before/after an action) must never clobber each other.
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
        return str(run_dir / f"{safe}_{suffix}_{stamp}.png")
    return str(run_dir / f"{safe}_{suffix}.png")


def _resolve(self: Engine, element_id: ElementId) -> Element:
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
