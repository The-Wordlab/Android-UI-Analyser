"""Acting on elements by id: target and selector resolution, tap/long-press/double-tap, text input, clear and erase, mic audio injection, swipe/scroll/key gestures, keyboard, clipboard paste/copy, a11y actions, and the RouteStep record each action emits.

Engine methods for actions. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_actions.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import re
import shlex
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .engine_support import (
    _SYSTEM_BAR_BAND,
    _ActionSite,
    _is_resource_id_lookup,
    detail_tokens,
    logger,
)
from .errors import (
    AuaError,
    DeviceError,
    ElementNotFoundError,
    SelectorAmbiguousError,
    SelectorNotFoundError,
    StaleElementIdError,
    UsageError,
)
from .memory import RouteStep, _id_tail, recorded_selector
from .platforms.runtime import TargetRuntime
from .schema import (
    ActionResult,
    AnalyzeResult,
    AppContext,
    Element,
    ElementId,
    MatchMode,
    ResolveResult,
    center_of,
)
from .scroll_geom import Box, Sample, _contains, region_probe, scroll_movement, scrollable_boxes
from .selectors import (
    _MAX_CANDIDATES,
    app_elements,
    drop_redundant_ocr,
    element_digest,
    match_selector,
    nearest_elements,
    normalize_selector_prefix,
    selector_label,
)

if TYPE_CHECKING:
    from .engine import Engine


def _input_runtime(self: Engine) -> TargetRuntime:
    return self.platform.runtime_capability("ui.input", self.device)


def _touch_runtime(self: Engine) -> TargetRuntime:
    return self.platform.runtime_capability("device.touch", self.device)


# A run of text one line tall measures roughly two to three average character widths
# in height; a wrapped paragraph measures many. Used to decide whether aiming at a
# phrase inside an element can only move horizontally.
_SINGLE_LINE_HEIGHT_RATIO = 3.5


def _action_mark(verb: str, el: Element) -> str:
    """Compact capture timeline label — verb + best human/id token."""
    label = el.text or el.content_desc or _id_tail(el.resource_id) or el.id
    # Keep marks short for timeline readability.
    text = str(label).replace("\n", " ").strip()
    if len(text) > 40:
        text = text[:37] + "…"
    return f"{verb}:{text}"


def _published_id(el: Element | None) -> ElementId | None:
    """The id an action reports for *el* — the same stable id its observation publishes.

    Reporting the frame ordinal here made one payload speak two languages: `id: 34` at the top
    and `"id": "rid:characterCard_Teacher"` in the elements beneath it, for the same control.
    A caller echoing the top-level value back would be sending a number that appears nowhere in
    what it was given.
    """
    if el is None:
        return None
    from .identity import stable_key as _sk

    return el.stable_key or _sk(el)


def _action_site(self: Engine, element: Element | None) -> _ActionSite | None:
    """Where this action is happening, or None when screen or control is unknown.

        The control key prefers ``stable_key`` — the cross-frame fingerprint that exists exactly
        because element ids churn between analyses — and falls back to the resource-id tail. A
        node with neither is not keyed at all: a timing filed under a key that means something
        different next run is worse than no timing, because it would be spent as a deadline.

        The package rides along from this one cache read rather than being looked up later: by
        the time the settle path records the measurement, the interaction has already deleted
        the cache the package would have come from.
        """
    if element is None:
        return None
    cached = self._read_cache()
    screen = cached.meta.known_screen if cached and cached.meta else None
    control = element.stable_key or _id_tail(element.resource_id)
    if not (screen and control):
        return None
    return _ActionSite(str(screen), str(control), cached.screen.package if cached else None)


def _step(
    self: Engine,
    kind: str,
    element: Element | None = None,
    *,
    arg: str | None = None,
    submit: bool = False,
) -> RouteStep:
    """The structured record of one action (selector + redacted label, never a value)."""
    self._last_action_kind = kind
    # Where this action is happening, so its cost can be learned per (screen, control) and
    # not merely per action kind. Captured here because `_step` runs *before* the action,
    # while the cache still describes the screen we are acting from — afterwards it
    # describes the destination, which is not where the click was spent.
    self._last_action_site = self._action_site(element)
    selector = (
        recorded_selector(
            element,
            elements=(self._last_analyze_result.elements if self._last_analyze_result else ()),
        )
        if element
        else {
            "label": None,
            "content_desc": None,
            "resource_id": None,
            "by": None,
        }
    )
    return RouteStep(
        kind=kind,
        label=selector["label"],
        content_desc=selector["content_desc"],
        resource_id=selector["resource_id"],
        by=selector["by"],
        arg=arg,
        submit=submit,
        package=self._cached_package(),
    )


def resolve(
    self: Engine,
    target: str | int,
    *,
    fresh: bool = True,
) -> ResolveResult:
    """Remap a previous-frame id or ``stable_key`` onto the current screen.

        Integer ids die on every re-analyze; ``stable_key`` (and this remapper) survive.
        """
    from .identity import find_by_stable_key, remap_ids, stable_key

    cached = self._read_cache()
    current = self.analyze(source="auto", record=False) if fresh or cached is None else cached

    from_id: int | None = None
    key: str | None = None
    if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
        from_id = int(target)
        if cached is not None:
            prev = cached.element_by_id(from_id)
            if prev is not None:
                key = prev.stable_key or stable_key(prev)
                mapping = remap_ids(cached.elements, current.elements)
                if from_id in mapping:
                    to_id = mapping[from_id]
                    el = current.element_by_id(to_id)
                    return ResolveResult(
                        ok=True,
                        from_id=from_id,
                        to_id=to_id,
                        stable_key=key,
                        element=el,
                    )
        # Fall through: treat as missing and try key from fresh screen? No — id unknown.
        raise ElementNotFoundError(
            f"could not resolve id {from_id} onto the current screen",
            hint="Re-analyze after the screen changes, or pass a stable_key (rid:…).",
        )

    key = str(target).strip()
    hits = find_by_stable_key(current.elements, key)
    if len(hits) == 1:
        el = hits[0]
        return ResolveResult(ok=True, from_id=from_id, to_id=el.id, stable_key=key, element=el)
    if not hits:
        raise ElementNotFoundError(
            f"no element with stable_key {key!r} on the current screen",
            hint="Run `aua analyze` and use the element's stable_key, or a prior id.",
        )
    raise SelectorAmbiguousError(
        f"stable_key {key!r} matched {len(hits)} elements",
        hint="Disambiguate with --rid/--text or inspect the screen.",
    )


def resolve_selector(
    self: Engine,
    *,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
    index: int | None = None,
    first: bool = False,
    fresh: bool = True,
    vision_fallback: bool = True,
    prefer_clickable: bool = False,
) -> Element:
    """Resolve a one-shot selector to a single element, in this one call.

        Element ids die the moment the screen changes, so ``analyze`` → grep → ``tap <id>``
        is three round-trips whose middle step the caller has to hand-write. Resolving
        server-side collapses that to one, and the re-analyze also refreshes the id cache
        so ids in the returned observation are immediately usable.

        Raises rather than guessing: :class:`SelectorNotFoundError` on no match (with the
        nearest candidates), :class:`SelectorAmbiguousError` on several (with all of them).
        A silent pick is indistinguishable from "the app ignored a valid tap".
        """
    rid = normalize_selector_prefix("rid", rid)
    text = normalize_selector_prefix("text", text)
    desc = normalize_selector_prefix("desc", desc)
    selector = {"rid": rid, "text": text, "desc": desc}
    given = [field for field, value in selector.items() if value]
    if len(given) != 1:
        raise UsageError(
            "give exactly one selector: --rid <resource-id> | --text <label> | --desc <desc>",
            hint="e.g. `aua tap-and-analyze --rid notificationsButton` or `aua tap-and-analyze --text 'Create an app'`",
        )
    cached = None if fresh else self._read_cache()
    result = cached if cached is not None else self.analyze(source="hierarchy", record=False)
    elements = result.elements
    # Fused OCR readings of text the tree already reports would break tiering: see
    # drop_redundant_ocr. Must happen before matching, not after.
    matches = match_selector(drop_redundant_ocr(elements), rid=rid, text=text, desc=desc)
    label = selector_label(selector)
    if not matches and rid:
        # The element list prunes unlabeled, non-actionable containers, so a real
        # `containerDetail` is addressable without being listed. Ask the device
        # directly before giving up — the same lookup `has --by id` uses.
        container = self._resolve_container_rid(rid)
        if container is not None:
            return container
    if not matches and text and vision_fallback:
        matches, elements = self._match_by_vision(elements, text)
    if not matches:
        needle = rid or text or desc or ""
        near = nearest_elements(elements, needle)
        # Every row carries both an `id` (this analyze's ordinal) and a `rid` (the app's
        # resource-id), so `--rid 49` is the natural conflation of the two columns. Answering
        # it with "nearest: action_bar_root | System UI notification" sends the reader looking
        # for a spelling mistake that is not there, so name the actual mix-up first.
        if rid and rid.isdigit():
            hint = (
                f"{rid} is an element id, not a resource-id — ids are positional: "
                f"`aua tap-and-analyze {rid}`. Use --rid for the app's resource-id string "
                "(the `rid` column), and prefer it: ids are renumbered by every analyze."
            )
        elif near:
            hint = "nearest: " + " | ".join(element_digest(el) for el in near)
        else:
            hint = "Run `aua analyze` to see what is on screen."
        raise SelectorNotFoundError(
            f"no element matches {label} "
            f"({len(app_elements(elements))} app elements on screen)",
            hint=hint,
        )
    if len(matches) > 1:
        if index is not None:
            if not 0 <= index < len(matches):
                raise UsageError(
                    f"--index {index} out of range: {label} matches {len(matches)} elements",
                    hint="Indexes are 0-based and follow reading order (top-left first).",
                )
            return matches[index]
        if prefer_clickable:
            # Compose routinely renders a clickable icon tile and a caption beneath it
            # carrying the same text, as two separate accessibility nodes with no
            # overlap - measured on a real sheet: the tile at y 840-1000 and the caption
            # at y 1016-1051. So `tap --text "Attach Photos"` matches two elements and
            # both are genuine; no dedup rule can help, because neither is a duplicate
            # reading of the other. But only one of them can be tapped, and the caller
            # said tap. Choosing it is not guessing - it is the only interpretation that
            # can be carried out. Two tappable matches stay ambiguous, as they should.
            tappable = [el for el in matches if el.clickable]
            if len(tappable) == 1:
                logger.info(
                    "%s matched %d elements; taking the only clickable one (id=%s)",
                    label,
                    len(matches),
                    tappable[0].id,
                )
                return tappable[0]
        if not first:
            raise SelectorAmbiguousError(
                f"{label} matches {len(matches)} elements — "
                "disambiguate with --index <n> or take --first",
                hint="candidates: "
                + " | ".join(element_digest(el) for el in matches[:_MAX_CANDIDATES]),
            )
    return matches[0]


def _match_by_vision(
    self: Engine, elements: list[Element], text: str
) -> tuple[list[Element], list[Element]]:
    """Look again with vision when the hierarchy has no element carrying ``text``.

        Web content publishes almost nothing to the accessibility tree, so `--text Continue`
        fails on an OAuth consent page, a Google sign-in form or an in-app Terms screen where
        "Continue" is plainly on screen. Because vision element ids do not survive between CLI
        invocations, the only way through used to be a raw coordinate tap — the single thing
        this tool exists to remove.

        Only a label falls back. A resource-id is a property of the tree and pixels cannot
        supply one; a content-desc is likewise unobservable, so `--desc` stays strict.

        Returns the matches and the element list to describe the screen with, so that on a
        miss the "nearest" hint names what is *visible* rather than an empty WebView node.
        """
    try:
        seen = self.analyze(source="vision", record=False)
    except AuaError as exc:  # vision unavailable is not a selector error
        logger.debug("vision fallback for %r unavailable: %s", text, exc)
        return [], elements
    matches = match_selector(seen.elements, text=text)
    if matches:
        logger.info("resolved --text %r by vision; the hierarchy had no match", text)
        return matches, seen.elements
    return [], elements + [el for el in seen.elements if el not in elements]


def _resolve_container_rid(self: Engine, rid: str) -> Element | None:
    """A pruned container, addressed by resource-id via the device itself.

        Returns ``None`` when the device does not know it either, so the caller still
        raises :class:`SelectorNotFoundError` with its candidate list. ``id=-1`` marks an
        element that never came from an ``analyze``, so it is not in the id cache.
        """
    bounds = _input_runtime(self).find_text(rid, match=MatchMode.exact, by="id")
    if bounds is None:
        return None
    return Element(
        id=-1,
        type="Container",
        resource_id=rid,
        bounds=bounds,
        center=center_of(bounds),
    )


def _target(
    self: Engine, element_id: ElementId | None, selector: dict[str, Any] | None, *, verb: str = "tap"
) -> Element:
    """The element an action addresses: a freshly-bound prior id, or a selector.

        Integer ids are reading-order ordinals, not identities.  A network update can renumber a
        dynamic list between ``analyze`` and ``long-press`` without any AUA command in between;
        resolving the integer straight out of the old cache then acts on a different row.  Before
        every numeric action, re-read and remap the cached element by stable identity.  When the
        evidence no longer names the same labelled control, refuse before touching the device.
        """
    if selector:
        if selector.get("key"):
            return self._resolve_action_key(
                str(selector["key"]), bounds=selector.get("bounds"), verb=verb
            )
        named = {name: value for name, value in selector.items() if name != "bounds"}
        # The screen the caller is holding, read before `resolve_selector` refreshes the id
        # cache with its own fresh read (which is the point of that read, and also why this
        # cannot be sampled afterwards). The verdict is decided here rather than inside
        # `resolve_selector` because that method is also the one AUA calls to *look*, and
        # only an action asks "was my screen already gone".
        shown = self._read_cache()
        # A verb that needs something tappable may break a tie on clickability.
        found = self.resolve_selector(**named, prefer_clickable=verb in ("tap", "long-press"))
        self._note_screen_moved(shown, self._last_analyze_result)
        return found
    if element_id is None:
        raise UsageError(
            f"{verb} needs an element id or a selector",
            hint=f"`aua {verb} 9`, `aua {verb} --rid someId`, or `aua {verb} --text 'Label'`",
        )
    return self._resolve_action_id(element_id, verb=verb)


def _binding_label(element: Element) -> tuple[str, str]:
    """Normalised semantic label used to detect a resource-id that changed owners."""

    def normalise(value: str | None) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).casefold()

    return normalise(element.text), normalise(element.content_desc)


def _key_may_be_visual(key: str) -> bool:
    """Whether *key* could name an element that only OCR/vision can see.

        Label-derived keys (``tx:``/``cd:``) hash the element *type* alongside the label, and a
        vision reading's type is always ``Text`` rather than a widget class, so an OCR key can
        never collide with a hierarchy node. ``rid:``/``geo:``/``px:`` keys only ever come from
        accessibility data, so they never justify the extra pass.
        """
    return key.startswith(("tx:", "cd:"))


def _miss_observation(self: Engine, observation: AnalyzeResult) -> Any:
    """The screen attached to a miss, trimmed the way a successful action's is.

        The resolution read is the whole tree — status bar, wrappers and all — because it was
        taken to search, not to publish. Attaching it raw made a failure the most expensive
        payload the tool emits (147 rows against the ~20 an action returns). Same dials, so a
        caller reading a miss sees rows in the shape it already knows.

        One key survives that budget which a healthy action's does not: `capture_hint`. A miss
        has no `ActionResult` to hang it on, and "what happened in between" is precisely the
        question being asked here — the frame buffer is the only thing that can answer it. It
        was trimmed out of the shared preset because on a *successful* action it cost bytes to
        say nothing, which is an argument about the healthy path and not about this one.
        """
    from .projection import Projection
    from .schema import OutputFormat as _Fmt

    try:
        output = getattr(self.config, "output", None)
        view = Projection.for_observation(
            getattr(output, "observation_fields", None),
            meta=getattr(output, "observation_meta", None),
        )
        if view is None:
            return observation
        trimmed = view.apply(observation.as_dict(_Fmt.json))
        meta = trimmed.get("meta")
        if isinstance(meta, dict) and observation.meta.capture_hint:
            meta.setdefault("capture_hint", observation.meta.capture_hint)
        return trimmed
    except Exception:  # noqa: BLE001 - an attached screen is a bonus, never the failure
        return observation


def _resolve_action_key(
    self: Engine, key: str, *, bounds: Sequence[int] | None = None, verb: str = "tap"
) -> Element:
    """Address an element by cross-frame identity, independent of the numeric id cache.

        Integer ids are frame-local ordinals resolved through one cache file per device, which
        every caller of that device shares. A caller holding an observation produced by a
        *different* process — the dashboard, a second agent, a saved report — therefore cannot
        safely send a number: the file it would be validated against belongs to whoever wrote
        it last. ``stable_key`` is the only element name that outlives its frame, so it is what
        such a caller sends, and this resolves it against the live screen with no shared state
        in the path at all.
        """
    from .identity import closest_by_bounds, find_by_stable_key

    # The screen the caller is holding, read before this call's own read so the two can be
    # compared; see `_note_screen_moved`.
    shown = self._read_cache()
    # A resolution read is AUA's own evidence, never a published observation: recording
    # it would replace the caller's id space with a hierarchy-only view of it.
    current = self.analyze(
        source="hierarchy", with_ocr=False, record=False, record_ids=False
    )
    hits = find_by_stable_key(current.elements, key)
    if not hits and self._key_may_be_visual(key):
        current = self.analyze(
            source="hierarchy", with_ocr=True, record=False, record_ids=False
        )
        hits = find_by_stable_key(current.elements, key)
    moved = self._note_screen_moved(shown, current)
    if not hits:
        # The screen that proves the miss rides along: this read is how we know the key is
        # absent, so telling the caller to go and analyze would spend a round trip on a
        # payload already in hand — and when the screen moved underneath them (an
        # interstitial, a dialog), this observation is the answer they actually need.
        #
        # When that is what happened, say so as a fact rather than as a hedge. "It may have
        # changed under you" reads as boilerplate; "it did, and nothing you sent caused it"
        # is the difference between a recoverable answer and a caller re-sending the same id.
        if moved:
            current.meta.screen_moved = moved
            self._screen_moved = None
            why = f"is not on this screen, because {moved.rstrip('.')}"
        else:
            why = "is not on this screen — which may have changed under you"
        raise ElementNotFoundError(
            f"no element with stable_key {key!r} on the current screen for {verb}",
            hint=(
                f"No action was sent and {key!r} {why}. The current screen is attached as "
                "`observation`: use an id from it, or address the element with "
                "--rid/--text/--desc."
            ),
            observation=self._miss_observation(current),
        )
    if len(hits) == 1:
        return hits[0]
    chosen = closest_by_bounds(hits, bounds)
    if chosen is not None:
        return chosen
    where = ", ".join(f"id {hit.id} at {tuple(hit.bounds)}" for hit in hits[:8])
    raise SelectorAmbiguousError(
        f"stable_key {key!r} matches {len(hits)} elements for {verb}: {where}",
        hint=(
            "No action was sent. Send the bounds the element was published with so the "
            "right one can be chosen, or use a selector that names it uniquely."
        ),
    )


def _resolve_action_id(self: Engine, element_id: ElementId, *, verb: str) -> Element:
    """Remap a frame-local id onto a fresh hierarchy, or raise ``stale_element_id``.

        Resource ids are normally the strongest binding, but reusable row layouts give every
        item the same rid.  The label check is therefore intentional even after ``remap_ids``:
        if cached id 8 said "Draft Orion" and the same row now says "Suggested reply", touching
        the overlapping node would silently perform the requested action on the wrong content.
        """
    from .identity import remap_ids, stable_key

    cached = self._read_cache()
    if cached is None:
        raise ElementNotFoundError(
            "no cached analyze result", hint="Run `aua analyze` first to assign element ids."
        )
    previous = cached.element_by_id(element_id)
    if previous is None:
        valid = ", ".join(str(e.id) for e in cached.elements[:20]) or "(none)"
        raise ElementNotFoundError(
            f"element id {element_id} is not in the last analyze (valid: {valid})",
            hint="Re-run `aua analyze`; ids change when the screen changes.",
        )

    # Freshness validation compares hierarchy bindings, which is the cheap read and the
    # right one for almost every id.
    def rebind(with_ocr: bool) -> Element | None:
        fresh = self.analyze(
            source="hierarchy", with_ocr=with_ocr, record=False, record_ids=False
        )
        # Same pre-action read, same question: was the caller's screen already gone before
        # this call touched the device (see `_note_screen_moved`)?
        self._note_screen_moved(cached, fresh)
        mapped = remap_ids(cached.elements, fresh.elements).get(element_id)
        return fresh.element_by_id(mapped) if mapped is not None else None

    def bound(candidate: Element | None) -> bool:
        return candidate is not None and self._binding_label(
            previous
        ) == self._binding_label(candidate)

    candidate = rebind(False)
    if not bound(candidate) and (previous.source or "hierarchy") != "hierarchy":
        # An element the hierarchy cannot describe is missing from a hierarchy-only read
        # by construction, so its absence there is evidence of nothing. Without this
        # retry a canvas label read by OCR reports a changed binding 100% of the time and
        # is simply unreachable by number. It stays a retry rather than the first read:
        # the cheap path resolves most vision ids too, and goto's hierarchy-first budget
        # must not pay for a provider call it does not need.
        candidate = rebind(True)
    previous_key = previous.stable_key or stable_key(previous)
    if candidate is None or not bound(candidate):
        selector_hint = "a stable --rid/--text/--desc selector"
        if previous.resource_id:
            selector_hint = f"--rid {(previous.resource_id or '').rsplit('/', 1)[-1]}"
        elif previous.content_desc:
            selector_hint = f"--desc {previous.content_desc!r}"
        elif previous.text:
            selector_hint = f"--text {previous.text!r}"
        current_label = None
        if candidate is not None:
            current_label = candidate.text or candidate.content_desc
        detail = f"; it now resolves to {current_label!r}" if current_label else ""
        raise StaleElementIdError(
            f"element id {element_id} is stale for {verb}: binding {previous_key!r} changed{detail}",
            hint=(
                f"No action was sent. Use {selector_hint}, or re-run `aua analyze` and use "
                "an id from that fresh observation."
            ),
        )
    return candidate


def target_report(
    self: Engine,
    *,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
    index: int | None = None,
    first: bool = False,
) -> dict[str, Any]:
    """Answer "what does this label actually address?" without touching the device.

        This is the half of the acting-node problem that cost a verdict. A lane did not tap
        and misread the result — it *read state*: the tile's title reported
        ``clickable:false, enabled:true`` and it concluded the control was broken. Nothing in
        the output said that node carries no click action, so its ``enabled`` flag described a
        caption rather than a control, and the pair looks **identical whether the real control
        is enabled or disabled**. So there has to be a way to ask before believing.

        Returns the node named, the node that acts, their relation, the state that actually
        belongs to the control, and the point a tap would use.
        """
    from .selectors import acting_node, acting_report

    named = self.resolve_selector(
        rid=rid, text=text, desc=desc, index=index, first=first, prefer_clickable=False
    )
    cached = self._read_cache()
    pool = list(cached.elements) if cached is not None else [named]
    if all(other.id != named.id for other in pool):
        pool.append(named)
    found = acting_node(pool, named)
    acted = found.element if found.redirected else named
    aim_x, _ = self._tap_point(acted, text if acted.id == named.id else None)
    _, aim_y = self._aim(acted)
    return {
        "ok": True,
        "action": "target",
        "named": named.compact(),
        "acts": found.relation == "self",
        "acting": acting_report(found),
        "control": {
            # The id the observation publishes: `acting` explains which node actually
            # received the action, and an ordinal here cannot be found in the elements
            # beside it.
            "id": _published_id(acted),
            "type": acted.type,
            "clickable": bool(acted.clickable),
            "enabled": bool(acted.enabled),
            "checkable": acted.checkable,
            "checked": acted.checked,
            "long_clickable": acted.long_clickable,
            "bounds": list(acted.bounds),
        },
        "tap_point": [aim_x, aim_y],
        "hint": (
            None
            if found.relation == "self"
            else "`enabled`/`clickable` on the node you named describe a caption, not the "
            "control — read them off `control` instead."
        ),
    }


def _acting_target(self: Engine, el: Element, *, verb: str = "tap") -> tuple[Element, dict[str, Any]]:
    """The node that will receive the interaction, plus the report of that choice.

        Retargets **only when the named node carries no interaction at all**, which is the
        deliberate boundary. A node that is itself clickable is left exactly as it was, so the
        overwhelming majority of existing taps are untouched — the danger a change to `tap`
        carries is silently wrong results, and this narrows the blast radius to the case that
        is already broken. Tapping a non-clickable node's centre today dispatches into
        whatever is under it and usually does nothing (the observed tile returned ok:true and
        did not act), so redirecting there can only improve on a coin flip.

        Also refuses to move when the choice would be a guess: several candidate controls, or
        none, leave the target alone and say so in the report.
        """
    from .selectors import acting_node, acting_report

    cached = self._read_cache()
    pool = list(cached.elements) if cached is not None else [el]
    if all(other.id != el.id for other in pool):
        pool.append(el)
    found = acting_node(pool, el)
    if verb == "long-press" and found.relation == "sibling-subtree":
        raise UsageError(
            "long-press will not retarget a label into a sibling control subtree",
            hint=(
                "No gesture was sent. Inspect the acting candidates with `aua target`, then "
                "long-press the actual control by --rid or a fresh numeric id."
            ),
            code="unsafe_action_target",
        )
    report = acting_report(found)
    return (found.element if found.redirected else el), report


def _tap_point(self: Engine, el: Element, needle: str | None) -> tuple[int, int]:
    """Where to tap for *el*, aiming at *needle* when it is only part of one line.

        Two links can share a single line: "Terms of use and Privacy policy" is one
        underlined run where each phrase is its own tappable span. Android does not publish
        a ClickableSpan as a separate node, and OCR groups by visual line, so both the tree
        and the pixels describe that line as **one** element. Tapping its centre lands
        between the two links - on neither - and the only workaround left was a measured
        coordinate tap, which is exactly what this tool exists to remove.

        So when the matched element's text merely *contains* the phrase asked for, and the
        element is a single line, aim proportionally: estimate the phrase's horizontal share
        of the line and tap the middle of that. Character widths vary, so this is an
        estimate - but an estimate inside the right phrase beats the exact centre of the
        wrong one.

        Falls back to the element centre whenever the assumption does not hold: no needle,
        an exact match, a multi-line box, or a phrase not found in the text.
        """
    cx, cy = el.center
    text = (el.text or "").strip()
    want = (needle or "").strip()
    if not want or not text or len(want) >= len(text):
        return cx, cy
    start = text.casefold().find(want.casefold())
    if start < 0:
        return cx, cy
    x1, y1, x2, y2 = el.bounds
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return cx, cy
    # Is this one line? Aspect ratio does not answer that - a 640x200 paragraph is still
    # wider than it is tall. Comparing the height to the average character width does,
    # and is independent of screen density: one line of text is roughly two to three
    # character-widths tall, a paragraph is many. Guessing wrong here would aim at the
    # right horizontal offset on the wrong line, which is worse than the centre.
    avg_char_width = width / len(text)
    if height > _SINGLE_LINE_HEIGHT_RATIO * avg_char_width:
        return cx, cy
    mid_char = start + len(want) / 2.0
    x = int(x1 + width * (mid_char / len(text)))
    x = max(x1 + 1, min(x, x2 - 1))
    logger.info(
        "aiming at %r inside %r: x=%d instead of the line centre %d", want, text[:40], x, cx
    )
    return x, cy


def tap(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    observe: bool = True,
    with_image: bool | str | None = None,
    _hierarchy_settle: bool = False,
) -> ActionResult:
    named = self._target(element_id, selector)
    # The label a caller names is often not the node that acts — see `_acting_target`.
    # Resolve that first, then aim, so both corrections apply to the control rather than
    # to the caption sitting below it.
    el, acting = self._acting_target(named)
    if element_id is not None and acting.get("relation") == "sibling-subtree":
        raise UsageError(
            f"numeric id {element_id} names a label, not the sibling control that would act",
            hint=(
                "No gesture was sent. Numeric ids are exact frame bindings and are never "
                "retargeted to a sibling. Use the acting control's fresh id/rid from `aua "
                "target`, or use the visible-text selector deliberately."
            ),
            code="unsafe_action_target",
        )
    # Two independent corrections to the naive `el.center`, composed on separate axes.
    # `_tap_point` aims x at a named phrase inside a single line, so two links on one line
    # are separately reachable. `_aim` moves y out of the system navigation bar, whose
    # docstring states it never second-guesses x for exactly this reason. Take x from the
    # first and y from the second; `_aim` still raises when nothing of the element is
    # visible, which must survive rather than being swallowed here.
    # The phrase-aiming needle belongs to the *named* node's text, so it is only
    # meaningful when the named node is the one being tapped.
    needle = (selector or {}).get("text") if el.id == named.id else None
    cx, _ = self._tap_point(el, needle)
    _, cy = self._aim(el)
    step = self._step("tap", el)  # built pre-action: needs the cached package
    with self._acting(_action_mark("tap", el), capture_pre_action=not _hierarchy_settle):
        _input_runtime(self).click(cx, cy)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(
            ok=True, action="tap", id=_published_id(el), target=[cx, cy], acting=acting
        ),
        observe,
        with_image,
    )


def tap_point(
    self: Engine,
    x: int,
    y: int,
    *,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    """Tap an exact screen coordinate, and record it like any other action.

        For a canvas, or a grid that publishes one accessibility node per row with nothing
        per cell, a coordinate is the *correct* address rather than a workaround — there is
        no element to name. The alternative a runner is left with is `adb shell input tap`,
        which cannot be recorded, so any journey crossing such a surface becomes uncapturable.

        Both of `tap`'s aim corrections are deliberately bypassed: `_tap_point` moves x
        toward a named phrase and `_aim` moves y out of the system navigation bar, and each
        exists to guess better than `el.center`. Here the caller has already said exactly
        where, so guessing on either axis would be wrong — including the nav-bar lift, since
        tapping the bar itself is a legitimate thing to ask for.
        """
    step = self._step("tap-point", arg=f"{x},{y}")
    with self._acting(f"tap-point:{x},{y}"):
        _input_runtime(self).click(x, y)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(ok=True, action="tap-point", target=[x, y]), observe, with_image
    )


def long_press(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    ms: int = 600,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    named = self._target(element_id, selector, verb="long-press")
    el, acting = self._acting_target(named, verb="long-press")
    cx, cy = self._aim(el)
    step = self._step("long-press", el)
    with self._acting(_action_mark("long-press", el)):
        _input_runtime(self).long_click(cx, cy, ms)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(
            ok=True,
            action="long-press",
            id=_published_id(el),
            target=[cx, cy],
            acting=acting,
        ),
        observe,
        with_image,
    )


def mic_inject(
    self: Engine,
    wav_path: str | Path,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    control_mode: str = "hold",
    pre_roll_ms: int = 250,
    post_roll_ms: int = 250,
    observe: bool = True,
    with_image: bool | str | None = None,
    _action: str = "mic-inject",
    _source: str | None = None,
) -> ActionResult:
    """Inject PCM WAV audio with an optional hold or tap-to-toggle control."""

    if pre_roll_ms < 0 or post_roll_ms < 0:
        raise UsageError(
            "microphone pre-roll and post-roll must be zero or greater",
            code="mic_roll_invalid",
        )
    mic_mod = self.platform.capability("microphone")

    has_control = element_id is not None or selector is not None
    control_mode = mic_mod.validate_control_mode(control_mode, has_target=has_control)
    prepared = mic_mod.prepare_injection(self.device.serial, wav_path)
    el: Element | None = None
    acting: dict[str, Any] | None = None
    target: list[int] | None = None
    toggle_owner: str | None = None
    tree_runtime: TargetRuntime | None = None
    if has_control:
        tree_runtime = self.platform.runtime_capability("ui.tree", self.device)
        verb = "tap" if control_mode == "toggle" else "long-press"
        if control_mode == "toggle":
            owner_before_target = AppContext.coerce(tree_runtime.current_app())
            toggle_owner = str(owner_before_target.app_id or "")
            if not toggle_owner:
                raise DeviceError(
                    "could not prove which foreground app owns the toggle control",
                    code="mic_toggle_owner_unknown",
                    hint="No gesture or audio was sent. Observe the app again before retrying.",
                )
        named = self._target(element_id, selector, verb=verb)
        el, acting = self._acting_target(named, verb=verb)
        if control_mode == "toggle":
            if element_id is not None and acting.get("relation") == "sibling-subtree":
                raise UsageError(
                    f"numeric id {element_id} names a label, not the sibling toggle control",
                    hint=(
                        "No gesture was sent. Use the acting control's fresh id/rid from "
                        "`aua target`, or use the visible-text selector deliberately."
                    ),
                    code="unsafe_action_target",
                )
            if not el.enabled or not el.clickable:
                raise UsageError(
                    "toggle microphone control must be enabled and clickable",
                    code="mic_toggle_target_inactive",
                    hint="No gesture or audio was sent. Choose the active tap-to-start control.",
                )
            if el.checked is True or el.selected is True:
                raise UsageError(
                    "toggle microphone control is already active",
                    code="mic_toggle_already_active",
                    hint=(
                        "No gesture or audio was sent. Stop the active recording first, then "
                        "retry only after the control visibly returns to its off state."
                    ),
                )
            needle = (selector or {}).get("text") if el.id == named.id else None
            cx, _ = self._tap_point(el, needle)
            _, cy = self._aim(el)
            owner = AppContext.coerce(tree_runtime.current_app())
            current_owner = str(owner.app_id or "")
            if not current_owner or current_owner != toggle_owner:
                raise DeviceError(
                    "the foreground app changed after resolving the toggle control",
                    code="mic_toggle_owner_changed",
                    hint=(
                        "No gesture or audio was sent. Observe the current app and resolve "
                        "the control again."
                    ),
                )
        else:
            cx, cy = self._aim(el)
        target = [cx, cy]
    # Claim the affected-build one-attempt guard before either opening gesture. Burning an
    # attempt if control start subsequently fails is conservative; allowing another stream
    # can crash the emulator process on 36.4.10.
    prepared = mic_mod.claim_injection_attempt(prepared)

    # Settle timing still needs to know which public action ran, but microphone samples
    # (like typed values) must not silently become a replayable navigation route.
    self._step(_action, el)
    mark = _action if el is None else _action_mark("mic", el)
    control_started = False
    down_attempted = False
    toggle_stop_allowed = True
    action_error: BaseException | None = None
    terminal_mic_error: Any | None = None
    injection_attempted = False
    injection_completed = False

    def toggle_owner_failure(stage: str) -> DeviceError | None:
        try:
            if tree_runtime is None:
                raise DeviceError("microphone toggle has no foreground-app observation runtime")
            current = AppContext.coerce(tree_runtime.current_app())
        except BaseException as exc:
            return DeviceError(
                f"could not prove toggle control ownership {stage}: {type(exc).__name__}",
                code="mic_toggle_owner_unknown",
                hint="Do not tap or retry blindly; recording state may be unknown.",
            )
        current_package = str(current.app_id or "")
        if not current_package or current_package != toggle_owner:
            return DeviceError(
                f"the foreground app changed {stage}",
                code="mic_toggle_owner_changed",
                hint="AUA refused to tap the snapshotted point in a different app.",
            )
        return None

    try:
        with self._acting(mark):
            try:
                if target is not None:
                    if control_mode == "hold":
                        # DOWN can land before its response is lost. Mark the attempt first
                        # so cleanup sends the harmless matching UP even when this raises.
                        down_attempted = True
                        _touch_runtime(self).touch_down(*target)
                        control_started = True
                    else:
                        owner_error = toggle_owner_failure("immediately before toggle START")
                        if owner_error is not None:
                            action_error = owner_error
                        else:
                            try:
                                _touch_runtime(self).click_once(*target)
                            except BaseException as exc:
                                # Never compensate for an ambiguous START: a second tap could
                                # either stop a delivered first tap or start recording itself.
                                terminal_mic_error = mic_mod.MicToggleStartUncertainError()
                                logger.warning(
                                    "toggle START was not confirmed: %s", type(exc).__name__
                                )
                            else:
                                control_started = True
                    if control_started and pre_roll_ms:
                        time.sleep(pre_roll_ms / 1000.0)

                if control_mode == "toggle" and control_started and terminal_mic_error is None:
                    owner_error = toggle_owner_failure("before audio injection")
                    if owner_error is not None:
                        toggle_stop_allowed = False
                        terminal_mic_error = mic_mod.MicToggleStopUncertainError(
                            "toggle START was confirmed, but the original app no longer "
                            "provably owned the screen before audio injection"
                        ).note_followup_failure("ownership_before_audio", owner_error)

                if terminal_mic_error is None and action_error is None:
                    injection_attempted = True
                    try:
                        mic_mod.inject_prepared(prepared)
                        injection_completed = True
                    except mic_mod.MicDeliveryUncertainError as exc:
                        # INTERNAL can arrive after every sample was delivered. Finish the
                        # control and force one observation, but retain the typed outcome.
                        terminal_mic_error = exc
                    except BaseException as exc:
                        action_error = exc

                if (
                    control_started
                    and action_error is None
                    and injection_attempted
                    and (injection_completed or terminal_mic_error is not None)
                    and post_roll_ms
                ):
                    try:
                        time.sleep(post_roll_ms / 1000.0)
                    except BaseException as exc:
                        if terminal_mic_error is not None:
                            terminal_mic_error.note_followup_failure("post_roll", exc)
                        else:
                            action_error = exc
            except BaseException as exc:
                if terminal_mic_error is not None:
                    terminal_mic_error.note_followup_failure("control_action", exc)
                else:
                    action_error = exc
            finally:
                should_finish_hold = (
                    control_mode == "hold" and down_attempted and target is not None
                )
                should_finish_toggle = (
                    control_mode == "toggle"
                    and control_started
                    and toggle_stop_allowed
                    and target is not None
                )
                if should_finish_hold or should_finish_toggle:
                    assert target is not None
                    try:
                        if should_finish_hold:
                            _touch_runtime(self).touch_up(*target)
                        else:
                            owner_error = toggle_owner_failure("before toggle STOP")
                            if owner_error is not None:
                                raise owner_error
                            # Exact snapshotted point, exactly once; never re-resolve a
                            # label whose meaning may have changed from Start to Stop.
                            _touch_runtime(self).click_once(*target)
                    except BaseException as exc:
                        finish_stage = (
                            "touch_release" if control_mode == "hold" else "toggle_stop"
                        )
                        if terminal_mic_error is not None:
                            terminal_mic_error.note_followup_failure(finish_stage, exc)
                            logger.warning(
                                "control cleanup also failed after ambiguous microphone action"
                            )
                        elif injection_completed and action_error is None:
                            error_type = (
                                mic_mod.MicDeliveredReleaseError
                                if control_mode == "hold"
                                else mic_mod.MicToggleStopUncertainError
                            )
                            terminal_mic_error = error_type().note_followup_failure(
                                finish_stage, exc
                            )
                            logger.warning("control cleanup failed after audio delivery")
                        elif action_error is None:
                            if control_mode == "toggle":
                                terminal_mic_error = (
                                    mic_mod.MicToggleStopUncertainError().note_followup_failure(
                                        finish_stage, exc
                                    )
                                )
                            else:
                                action_error = exc
                        else:
                            if control_mode == "toggle":
                                terminal_mic_error = mic_mod.MicToggleStopUncertainError()
                                terminal_mic_error.note_followup_failure(
                                    "audio_action", action_error
                                )
                                terminal_mic_error.note_followup_failure(finish_stage, exc)
                                action_error = None
                            else:
                                # Preserve the known injection error for hold mode, whose
                                # UP cleanup is idempotent and does not toggle recording on.
                                logger.warning(
                                    "touch-up also failed after microphone injection failed"
                                )
    except BaseException as exc:
        if terminal_mic_error is not None:
            terminal_mic_error.note_followup_failure("action_cleanup", exc)
        elif action_error is None:
            action_error = exc

    if terminal_mic_error is None and action_error is not None:
        raise action_error

    wav = prepared.wav
    channel_label = "mono" if wav.channels == 1 else "stereo"
    media_detail = (
        f"{wav.duration_s:.3f}s PCM {wav.sample_format} {channel_label} at {wav.sample_rate} Hz"
    )
    if injection_completed:
        detail = f"injected {media_detail}"
    elif injection_attempted:
        detail = f"audio delivery did not complete cleanly for {media_detail}"
    else:
        detail = f"audio was not injected ({media_detail})"
    if _source:
        detail = f"{_source}; {detail}"
    if target is not None:
        control_label = "push-to-talk hold" if control_mode == "hold" else "toggle control"
        detail += (
            f" with {control_label} ({pre_roll_ms}ms pre-roll, {post_roll_ms}ms post-roll)"
        )
    action_result = ActionResult(
        ok=terminal_mic_error is None,
        action=_action,
        id=_published_id(el),
        target=target,
        detail=detail,
        acting=acting,
    )
    if terminal_mic_error is not None:
        try:
            # A late transport/gesture failure is the one outcome where a fresh screen is
            # mandatory: without it, `--no-observe` would leave callers with no safe way
            # to decide whether the already-sent samples took effect.
            result = self._observe(action_result, True, with_image)
            result_payload = result.model_dump(mode="json")
        except BaseException as exc:
            raise terminal_mic_error.note_followup_failure("observation", exc) from exc
        raise terminal_mic_error.with_result(result_payload)
    return self._observe(action_result, observe, with_image)


def mic_speak(
    self: Engine,
    text: str,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    control_mode: str = "hold",
    voice: str | None = None,
    rate: int | None = None,
    pre_roll_ms: int = 250,
    post_roll_ms: int = 250,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    """Synthesize *text* with macOS ``say`` and inject the resulting temporary WAV."""

    mic_mod = self.platform.capability("microphone")

    has_control = element_id is not None or selector is not None
    control_mode = mic_mod.validate_control_mode(control_mode, has_target=has_control)
    if pre_roll_ms < 0 or post_roll_ms < 0:
        raise UsageError(
            "microphone pre-roll and post-roll must be zero or greater",
            code="mic_roll_invalid",
        )

    import tempfile

    with tempfile.TemporaryDirectory(prefix="aua-mic-") as temp_dir:
        wav_path = Path(temp_dir) / "speech.wav"
        mic_mod.synthesize_speech(text, wav_path, voice=voice, rate=rate)
        return self.mic_inject(
            wav_path,
            element_id,
            selector=selector,
            control_mode=control_mode,
            pre_roll_ms=pre_roll_ms,
            post_roll_ms=post_roll_ms,
            observe=observe,
            with_image=with_image,
            _action="mic-speak",
            _source="generated with macOS say",
        )


def double_tap(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    el = self._target(element_id, selector, verb="double-tap")
    cx, cy = self._aim(el)
    step = self._step("double-tap", el)
    with self._acting(_action_mark("double-tap", el)):
        _input_runtime(self).double_click(cx, cy)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(ok=True, action="double-tap", id=_published_id(el), target=[cx, cy]),
        observe,
        with_image,
    )


def input_text(
    self: Engine,
    element_id: ElementId | None = None,
    text: str = "",
    *,
    selector: dict[str, Any] | None = None,
    submit: bool = False,
    send_key: str | None = None,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    if submit and send_key:
        raise UsageError(
            "input accepts either submit or send_key, not both",
            hint="Use submit for the IME action, or send_key for an explicit app control.",
        )
    if send_key and not observe:
        raise UsageError(
            "send_key requires the post-action observation",
            hint="Drop --no-observe so AUA can verify whether the composer cleared.",
        )
    el = self._target(element_id, selector, verb="input")
    cx, cy = self._aim(el)
    # The step records the field's SHAPE only — the typed value is never persisted
    # (PRD §6b privacy; observe_action strips `text` defensively too).
    step = self._step("input", el, submit=submit or bool(send_key))
    before = el.text
    with self._acting(_action_mark("input", el)):
        _input_runtime(self).input_text(cx, cy, text, clear=True, submit=submit)
    self._record_action_safe(step)
    verified = self._typed_text_landed(before, text, submit=submit)
    result = self._observe(
        ActionResult(
            ok=verified is not False,
            action="input",
            id=_published_id(el),
            detail=text,
            verified=verified,
        ),
        observe,
        with_image,
    )
    if send_key:
        if verified is False:
            return result
        sent = self.tap(
            selector={"key": send_key},
            observe=True,
            with_image=with_image,
        )
        sent.action = "input-send"
        sent.detail = "typed text and tapped the explicit semantic send control"
        sent.submitted = self._submission_status(el, text, sent.observation)
        if sent.submitted is False:
            sent.note = (
                "The explicit send control was tapped, but the same input still contains "
                "the typed text. Do not type it again; inspect the returned observation."
            )
        return sent
    if submit:
        result.submitted = self._submission_status(el, text, result.observation)
        if result.submitted is False:
            result.note = (
                "The IME action did not submit: the same input still contains the typed "
                "text. Do not type it again; use recommended_call if present."
            )
            result.recommended_call = self._semantic_send_recommendation(
                result.observation,
                field=el,
            )
    return result


def _submission_status(
    field: Element,
    text: str,
    observation: AnalyzeResult | None,
) -> bool | None:
    """Whether the same composer visibly retained or cleared the submitted value."""
    if observation is None or field.password or not text:
        return None
    field_id = _published_id(field)
    matches = [
        element
        for element in observation.elements
        if _published_id(element) == field_id
        or (
            field.resource_id is not None
            and element.resource_id == field.resource_id
        )
    ]
    if len(matches) != 1:
        return None
    after = (matches[0].text or "").strip()
    typed = text.strip()
    if typed and typed in after:
        return False
    before = (field.text or "").strip()
    if not after or after == before:
        return True
    return None


def _semantic_send_recommendation(
    observation: AnalyzeResult | None,
    *,
    field: Element,
) -> dict[str, Any] | None:
    """Return one unique visible send/submit/confirm control, never a guessed coordinate."""
    if observation is None:
        return None
    field_id = _published_id(field)
    ranked: list[tuple[int, Element]] = []
    for element in observation.elements:
        if not element.clickable or _published_id(element) == field_id:
            continue
        rid = (_id_tail(element.resource_id) or "").casefold()
        label = " ".join(
            value for value in (element.text, element.content_desc) if value
        ).casefold()
        score = 0
        if re.search(r"(?:^|[_-])(send|submit|confirm)(?:$|[_-])", rid):
            score += 4
        elif any(token in rid for token in ("send", "submit", "confirm")):
            score += 3
        if re.search(r"\b(send|submit|confirm)\b", label):
            score += 2
        if score:
            ranked.append((score, element))
    if not ranked:
        return None
    best = max(score for score, _element in ranked)
    winners = [element for score, element in ranked if score == best]
    if len(winners) != 1:
        return None
    control = winners[0]
    control_rid = _id_tail(control.resource_id)
    if control_rid:
        cli = f"aua tap-and-analyze --rid {shlex.quote(control_rid)}"
        mcp_args: dict[str, Any] = {"rid": control_rid}
    else:
        key = str(_published_id(control))
        cli = f"aua tap-and-analyze {shlex.quote(key)}"
        mcp_args = {"id": key}
    return {
        "kind": "semantic_send",
        "cli": cli,
        "mcp": {"tool": "tap", "arguments": mcp_args},
        "reason": (
            "The IME action left the text in the composer and this is the unique visible "
            "send/submit/confirm control. Tap it without typing again."
        ),
        "executes": True,
    }


def _system_bar_top(self: Engine) -> int | None:
    """Top edge of the bottom system bar, or None when it cannot be established.

        Read from the elements of the last analyze, so this costs no device call. The nav
        bar's *background* node carries no label and does not survive parsing, but its
        buttons do — Back / Home / Overview are clickable nodes with content-descriptions
        on the ``system`` window layer, sitting flush against the bottom edge. The
        shallowest of those is the bar's top.

        Returns None on gesture navigation (no buttons to find) and whenever the analyze
        cache has no system elements — for instance after ``analyze --no-system``. None
        means "do not adjust anything", so the worst case is the old behaviour, never a
        differently-wrong aim point.
        """
    cached = self._read_cache()
    if cached is None or not cached.elements:
        return None
    try:
        _w, height = self.device.window_size()
    except Exception:  # pragma: no cover - best effort
        return None
    if height <= 0:
        return None
    # Flush with the bottom edge, and starting inside the bottom band. The band matters:
    # a full-screen system overlay (the expanded shade) or a systemui dialog must not be
    # mistaken for the bar, or we would "clamp" taps in the middle of the screen.
    floor = height * _SYSTEM_BAR_BAND
    tops = [
        el.bounds[1]
        for el in cached.elements
        if el.window == "system" and el.bounds[3] >= height - 2 and el.bounds[1] >= floor
    ]
    return min(tops) if tops else None


def _aim(self: Engine, el: Element) -> tuple[int, int]:
    """Where to touch *el*: its centre, unless the centre is under the system nav bar.

        11 controls across 6 apps in one sweep published accessibility bounds extending
        *below* the nav bar window, which starts at y=1184 on this pool. A touch at the
        element's centre then lands on the bar, and does one of two things, neither of them
        the caller's intent:

        - it is delivered to **Home**, which backgrounds the app under test and loses any
          unsaved input — a state-corrupting side effect, reported as success;
        - it is swallowed while the app stays foregrounded, so "the app is still in the
          foreground" is not evidence that the touch landed.

        Both were silent, which is what makes them expensive: a run loses its journey and
        then reports that the *product* ignored it.

        So aim at the middle of the part of the element that is actually visible above the
        bar. Only ``y`` moves — the horizontal aim is never second-guessed here, so this
        composes with phrase-aiming, which only moves ``x``. When nothing of the element is
        visible, there is no honest aim point and this raises rather than touching the bar:
        an error the caller can act on beats a success that backgrounded the app.

        The element's own layer is checked first. A caller that resolved the system Back or
        Home button *means* the bar, and clamping that would break the one case where the
        centre is right.
        """
    cx, cy = el.center
    if el.window == "system":
        return cx, cy
    bar_top = self._system_bar_top()
    if bar_top is None or cy < bar_top:
        return cx, cy
    top = el.bounds[1]
    if top >= bar_top - 1:
        raise ElementNotFoundError(
            f"element {el.id} lies entirely under the system navigation bar "
            f"(element starts at y={top}, the bar starts at y={bar_top})",
            hint=(
                "Nothing of it is touchable: a tap there goes to the navigation bar and "
                "can background the app. Scroll it into view, dismiss what is covering "
                "it, or act on a visible element instead."
            ),
        )
    aimed = (top + bar_top - 1) // 2
    logger.info(
        "element %s extends under the system bar (y=%d); aiming at y=%d instead of %d",
        el.id,
        bar_top,
        aimed,
        cy,
    )
    return cx, aimed


def _typed_text_landed(self: Engine, before: str | None, text: str, *, submit: bool) -> bool | None:
    """Did the text actually go in? ``True`` / ``False`` / ``None`` for "cannot tell".

        `input` was the last command that could report `ok:true` having done nothing at all —
        the family the rest of this tool has already closed off (`tap` re-analyzes, `record`
        consults `ps`, `emulator stop` checks the running list). It was seen repeatedly across
        a sweep: a field that never took the text, reported as success, and the lane read the
        empty result as the product ignoring it.

        Only one situation is unambiguous enough to *fail*: the field is readable, it still
        holds exactly what it held before, and it does not contain what we typed. Everything
        else stays `ok` with an honest ``None``, because plenty of fields legitimately do not
        read back what you typed:

        - ``submit=True`` sends the value and leaves the field empty — the common case for a
          chat composer, and failing it would be far worse than the bug;
        - password fields report a mask, or nothing at all;
        - fields that reformat as you type (phone numbers, card numbers, dates) or truncate
          at ``maxLength`` hold a *transformed* value, which is still a successful input.

        An unreadable field is ``None``, never "unchanged" — the same rule as everywhere else
        here: never turn a failed observation into a verdict.
        """
    if not text or submit:
        return None
    after = self.device.focused_text()
    if after is None:
        return None
    if text in after:
        return True
    if after == (before or ""):
        logger.warning(
            "input did not change the field (still %d chars); reporting the truth",
            len(after),
        )
        return False
    return None


def clear(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    el = self._target(element_id, selector, verb="clear")
    cx, cy = el.center
    step = self._step("clear", el)
    with self._acting(_action_mark("clear", el)):
        runtime = _input_runtime(self)
        runtime.click(cx, cy)
        runtime.clear_text()
    self._record_action_safe(step)
    return self._observe(ActionResult(ok=True, action="clear", id=_published_id(el)), observe, with_image)


def _dump(self: Engine) -> str:
    runtime = self.platform.runtime_capability("ui.tree", self.device)
    return self.platform.dump_tree(runtime)


def _scroll_elements(self: Engine, raw_tree: str = "") -> list[Element]:
    """Normalize one native hierarchy before shared scroll geometry inspects it."""

    runtime = self.platform.runtime_capability("ui.tree", self.device)
    size = runtime.window_size()
    normalized = self.platform.normalize_tree(
        raw_tree or self.platform.dump_tree(runtime),
        size,
        geometry=runtime.display_geometry(),
        ignored_app_ids=self.config.memory.ignore_packages,
    )
    return normalized.elements


def _scroll_box(
    self: Engine,
    *,
    from_id: int | None = None,
    selector: dict[str, Any] | None = None,
    raw_tree: str = "",
) -> tuple[Box, bool]:
    """``(box, is_real_container)`` — where a directional swipe should actually happen.

        Swiping the middle of the *screen* is why scrolling "did nothing": on a screen whose
        list occupies a sub-rectangle (a sheet, a tab body, a pane under a fixed header) the
        gesture lands outside the scrollable and gets thrown away. So aim at the scrollable
        container: the one under the anchor element when given, else the biggest on screen.
        """
    device = self.device
    w, h = device.window_size()
    screen: Box = (0, 0, w, h)
    boxes = scrollable_boxes(self._scroll_elements(raw_tree), (w, h))
    if not boxes:
        return screen, False
    anchor = None
    if from_id is not None or selector:
        anchor = self._target(from_id, selector, verb="swipe").center
    if anchor is not None:
        inside = [b for b in boxes if _contains(b, anchor)]
        if inside:  # innermost container under the anchor wins (nested scrollables)
            return min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), True
        return (anchor[0], anchor[1], anchor[0], anchor[1]), False
    return boxes[0], True


def _swipe_path(self: Engine, box: Box, direction: str, percent: int) -> tuple[int, int, int, int]:
    """Swipe endpoints spanning *percent* of *box*, inset from its edges.

        The inset matters: a gesture that starts on the very edge of a list is grabbed by
        the system's back/notification gestures instead of the list.
        """
    w, h = self.device.window_size()
    x1b, y1b, x2b, y2b = box
    cx, cy = (x1b + x2b) // 2, (y1b + y2b) // 2
    span_x = max(1, int((x2b - x1b) * min(percent, 90) / 200))
    span_y = max(1, int((y2b - y1b) * min(percent, 90) / 200))
    d = direction.lower()
    if d == "up":
        path = (cx, cy + span_y, cx, cy - span_y)
    elif d == "down":
        path = (cx, cy - span_y, cx, cy + span_y)
    elif d == "left":
        path = (cx + span_x, cy, cx - span_x, cy)
    elif d == "right":
        path = (cx - span_x, cy, cx + span_x, cy)
    else:
        raise UsageError(f"unknown swipe direction '{direction}'", hint="up|down|left|right")
    x1, y1, x2, y2 = path
    clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
    return (
        clamp(x1, 1, w - 2),
        clamp(y1, 1, h - 2),
        clamp(x2, 1, w - 2),
        clamp(y2, 1, h - 2),
    )


def _settle_after_swipe(self: Engine) -> None:
    """Let a fling finish before probing, or every scroll reads as "barely moved"."""
    from . import imaging

    device = self.device
    gs = imaging.GridSettle(streak=2)
    deadline = time.monotonic() + 0.9
    while time.monotonic() < deadline:
        try:
            frame = self.platform.adapter_capability("ui.screenshot").capture_screenshot(device)
            if gs.feed(frame):
                return
        except Exception:
            break
        time.sleep(0.035)
    time.sleep(0.05)


def _probe(self: Engine, box: Box) -> Sample:
    return region_probe(self._scroll_elements(), box)


def _swipe_once(
    self: Engine,
    box: Box,
    direction: str,
    percent: int,
    *,
    allow_content_turnover: bool = False,
) -> tuple[int, bool, str | None]:
    """One verified swipe: ``(distance_along_axis, scrolled, evidence)``.

        ``travel``'s own ``moved`` is "the sample differs at all" — a repaint, a ripple, or an
        element appearing all set it, none of which mean the content scrolled. Reporting that
        as movement is how `scroll --direction down` came to answer ``moved steps=1`` with no
        distance at the very top of a list, where scrolling further is impossible. A measurable
        shift along the requested axis remains the primary verdict. For hierarchy-declared
        scrollable containers, substantial removal *and* addition of labels also proves a
        virtualized grid turned over even when sticky labels keep the median shift at zero.
        """
    before = self._probe(box)
    x1, y1, x2, y2 = self._swipe_path(box, direction, percent)
    _input_runtime(self).swipe(x1, y1, x2, y2)
    self._settle_after_swipe()
    return scroll_movement(
        before,
        self._probe(box),
        direction,
        allow_content_turnover=allow_content_turnover,
    )


def swipe(
    self: Engine,
    direction: str | None = None,
    *,
    from_id: int | None = None,
    selector: dict[str, Any] | None = None,
    percent: int = 70,
    coords: tuple[int, int, int, int] | None = None,
    observe: bool = True,
    verify: bool = False,
    with_image: bool | str | None = None,
) -> ActionResult:
    device = _input_runtime(self)
    if coords is not None:
        x1, y1, x2, y2 = coords
        step = self._step("swipe", arg="coords")
        with self._acting(f"swipe:{direction or 'coords'}"):
            device.swipe(x1, y1, x2, y2)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]),
            observe,
            with_image,
        )
    if direction is None:
        raise UsageError(
            "swipe needs a direction or --coords", hint="e.g. `aua swipe-and-analyze up`"
        )
    d = direction.lower()
    box, real = self._scroll_box(from_id=from_id, selector=selector)
    x1, y1, x2, y2 = self._swipe_path(box, d, percent)
    step = self._step("swipe", arg=d)
    if not verify:
        with self._acting(f"swipe:{d}"):
            device.swipe(x1, y1, x2, y2)
            # Only settle here when the caller skipped observe — otherwise
            # ``_observe`` already does change+idle and a second settle doubles latency.
            if not observe:
                self._settle_after_swipe()
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]), observe, with_image
        )
    with self._acting(f"swipe:{d}"):
        distance, moved, evidence = self._swipe_once(
            box,
            d,
            percent,
            allow_content_turnover=real,
        )
    self._record_action_safe(step)
    # ok stays True — the gesture WAS performed, and a swipe is also used to dismiss or
    # page things where "the screen did not move" is the expected outcome. The verdict
    # is reported instead of swallowed; `aua scroll-and-analyze` is the strict-exit-code variant.
    return self._observe(
        ActionResult(
            ok=True,
            action="swipe",
            target=[x1, y1, x2, y2],
            detail=detail_tokens(
                "moved" if moved else "no-change",
                dy=abs(distance) if moved and distance else None,
                scrollable=str(real).lower(),
                evidence=evidence,
            ),
        ),
        observe,
        with_image,
    )


def scroll(
    self: Engine,
    direction: str | None = None,
    *,
    pages: int = 1,
    to_end: bool = False,
    to_start: bool = False,
    from_id: int | None = None,
    selector: dict[str, Any] | None = None,
    percent: int = 70,
    max_steps: int = 25,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    """Scroll a container and report what actually happened.

        Outcomes in ``detail`` (first token): ``moved`` · ``reached-end`` (moved, then ran
        out of content) · ``already-at-end`` (the very first swipe changed nothing). The
        first two are ``ok``; ``already-at-end`` is ``ok`` only for ``--to-end/--to-start``,
        where being at the end IS the postcondition. Everywhere else a scroll that moved
        nothing is a failure, because "nothing left to scroll" and "my swipe missed the
        list" must not look the same to a caller looping until something appears.
        """
    if to_end and to_start:
        raise UsageError("--to-end and --to-start are mutually exclusive")
    if direction is None:
        direction = "down" if to_start else "up"
    d = direction.lower()
    if d not in ("up", "down", "left", "right"):
        raise UsageError(f"unknown scroll direction '{direction}'", hint="up|down|left|right")
    limit = max_steps if (to_end or to_start) else max(1, pages)
    box, real = self._scroll_box(from_id=from_id, selector=selector)
    step = self._step("scroll", arg=d)
    travelled = 0
    steps = 0
    evidence: str | None = None
    with self._acting():
        for _ in range(limit):
            dy, moved, swipe_evidence = self._swipe_once(
                box,
                d,
                percent,
                allow_content_turnover=real,
            )
            if not moved:
                break
            steps += 1
            travelled += abs(dy)
            if swipe_evidence == "content-turnover" or evidence is None:
                evidence = swipe_evidence
    self._record_action_safe(step)
    at_end = steps < limit
    if steps == 0:
        outcome = "already-at-end"
    elif at_end:
        outcome = "reached-end"
    else:
        outcome = "moved"
    ok = steps > 0 or to_end or to_start
    return self._observe(
        ActionResult(
            ok=ok,
            action="scroll",
            target=list(box),
            detail=detail_tokens(
                outcome,
                steps=steps,
                dy=travelled or None,
                direction=d,
                scrollable=str(real).lower(),
                evidence=evidence,
            ),
        ),
        observe,
        with_image,
    )


def scroll_to(
    self: Engine,
    query: str,
    *,
    match: str = "contains",
    ignore_case: bool = False,
    observe: bool = True,
    by: str = "text",
    direction: str = "up",
    max_swipes: int = 10,
    percent: int = 70,
    with_image: bool | str | None = None,
) -> ActionResult:
    """Scroll until *query* is on screen, verifying every swipe actually moved.

        Outcomes in ``detail`` (first token): ``already-visible`` · ``moved`` (found it,
        with ``dy``) · ``already-at-end`` (nothing scrolled, so the target is simply not on
        this screen) · ``target-not-found`` (scrolled the whole way and never saw it).
        Only the first two are ``ok`` — the old version returned ``ok:false`` with exit 0,
        which is the same as saying nothing at all to an automated caller.
        """
    query_field = "rid" if _is_resource_id_lookup(by) else "desc" if by == "desc" else "text"
    query = normalize_selector_prefix(query_field, query) or query
    mode = MatchMode(match)
    step = self._step("scroll-to", arg=query)
    candidates = self._locale_candidates(self.device, query, by)
    matched_via: tuple[str, str, str] | None = None

    def locate() -> tuple[int, int, int, int] | None:
        nonlocal matched_via
        runtime = _input_runtime(self)
        b = runtime.find_text(query, match=mode, ignore_case=ignore_case, by=by)
        if b is not None:
            matched_via = None
            return b
        for cand in candidates:
            b = runtime.find_text(cand[0], match=mode, ignore_case=ignore_case, by=by)
            if b is not None:
                matched_via = cand
                return b
        return None

    def locate_hint() -> str | None:
        if matched_via is not None:
            return self._translated_hint(matched_via[0], matched_via[1], matched_via[2], query)
        return None

    found = locate()
    if found is not None:
        return self._observe(
            ActionResult(
                ok=True,
                action="scroll-to",
                detail=detail_tokens("already-visible", target=query),
                target=list(found),
                hint=locate_hint(),
            ),
            observe,
            with_image,
        )
    box, real = self._scroll_box()
    travelled = 0
    steps = 0
    evidence: str | None = None
    exhausted = True
    with self._acting():
        for _ in range(max(1, max_swipes)):
            dy, moved, swipe_evidence = self._swipe_once(
                box,
                direction,
                percent,
                allow_content_turnover=real,
            )
            if moved:
                steps += 1
                travelled += abs(dy)
                if swipe_evidence == "content-turnover" or evidence is None:
                    evidence = swipe_evidence
            found = locate()
            if found is not None or not moved:
                exhausted = False
                break
    self._record_action_safe(step)
    if found is not None:
        outcome = "moved"
    elif steps == 0:
        outcome = "already-at-end"
    else:
        outcome = "target-not-found"
    return self._observe(
        ActionResult(
            ok=found is not None,
            action="scroll-to",
            detail=detail_tokens(
                outcome,
                target=query,
                steps=steps,
                dy=travelled or None,
                scrollable=str(real).lower(),
                evidence=evidence,
                exhausted="true" if (found is None and exhausted) else None,
            ),
            target=list(found) if found else None,
            hint=locate_hint()
            if found is not None
            else self._text_miss_hint(self.device, by, query, tried_translations=True),
        ),
        observe,
        with_image,
    )


def key(
    self: Engine,
    name: str,
    *,
    observe: bool = True,
    with_image: bool | str | None = None,
    _hierarchy_settle: bool = False,
) -> ActionResult:
    candidate = self.platform.normalize_key(name)
    step = self._step("key", arg=name)
    with self._acting(f"key:{name}", capture_pre_action=not _hierarchy_settle):
        _input_runtime(self).press(candidate)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(ok=True, action="key", detail=candidate), observe, with_image
    )


def hide_keyboard(
    self: Engine, *, observe: bool = True, with_image: bool | str | None = None
) -> ActionResult:
    """Dismiss the soft keyboard (Maestro ``hideKeyboard``).

        Prefer this over ``key back`` when the IME is covering the tree — back can
        leave the screen; hide-keyboard aims to only dismiss the keyboard.
        """
    step = self._step("hide-keyboard")
    with self._acting("hide-keyboard"):
        self.platform.runtime_capability("device.keyboard", self.device).hide_keyboard()
    self._record_action_safe(step)
    # Verify, don't assume. This returned ok=True unconditionally, and an IME that stays
    # up is not a cosmetic miss: it covers the bottom of the screen, so the button the
    # caller is trying to reach is hidden while the command says the keyboard is gone.
    shown = self._ime_shown()
    detail = "hidden" if shown is False else ("still-shown" if shown else "unknown")
    return self._observe(
        ActionResult(ok=shown is not True, action="hide-keyboard", detail=detail),
        observe,
        with_image,
    )


def _ime_shown(self: Engine) -> bool | None:
    """Is the soft keyboard up? ``None`` when the device will not say.

        Tri-state on purpose: "cannot tell" must not read as "hidden", or this check would
        recreate the very false-success it exists to catch.
        """
    return self.platform.runtime_capability(
        "device.keyboard", self.device
    ).keyboard_visible()


def paste(self: Engine, *, observe: bool = True, with_image: bool | str | None = None) -> ActionResult:
    with self._acting():
        self.platform.runtime_capability("device.clipboard", self.device).paste()
    # The clipboard value is deliberately not captured. Keep a lossy journal marker so a
    # recorded-flow preview refuses to pretend the resulting journey is self-contained.
    self._record_action_safe(RouteStep(kind="paste"))
    return self._observe(ActionResult(ok=True, action="paste"), observe, with_image)


def copy_text(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
) -> ActionResult:
    el = self._target(element_id, selector, verb="copy")
    text = (el.text or el.content_desc or "").strip()
    if not text:
        raise UsageError(
            "element has no text or content-desc to copy",
            hint="Pick a labelled element, or use `clipboard set` for a literal.",
        )
    self.platform.runtime_capability("device.clipboard", self.device).set_clipboard(text)
    return ActionResult(ok=True, action="copy", id=_published_id(el), detail=text)


def a11y_scroll(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    direction: str = "forward",
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    el = self._target(element_id, selector, verb="a11y scroll")
    d = (direction or "forward").lower()
    action = "SCROLL_FORWARD" if d in ("forward", "fwd", "down") else "SCROLL_BACKWARD"
    if d not in ("forward", "fwd", "down", "backward", "back", "up"):
        raise UsageError(
            f"unknown scroll direction {direction!r}",
            hint="Use --forward or --backward.",
        )
    cx, cy = el.center
    step = self._step("a11y-scroll", el, arg=d)
    with self._acting(f"a11y-scroll:{d}"):
        self.platform.runtime_capability(
            "device.accessibility", self.device
        ).a11y_action(cx, cy, action)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(ok=True, action="a11y-scroll", detail=f"{d} @{el.id}"),
        observe,
        with_image,
    )


def a11y_action(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    action: str = "CLICK",
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    el = self._target(element_id, selector, verb="a11y action")
    cx, cy = el.center
    act = (action or "CLICK").strip().upper()
    step = self._step("a11y-action", el, arg=act)
    with self._acting(f"a11y:{act}"):
        self.platform.runtime_capability(
            "device.accessibility", self.device
        ).a11y_action(cx, cy, act)
    self._record_action_safe(step)
    return self._observe(
        ActionResult(ok=True, action="a11y-action", detail=f"{act} @{el.id}"),
        observe,
        with_image,
    )


def erase(
    self: Engine,
    element_id: ElementId | None = None,
    *,
    selector: dict[str, Any] | None = None,
    chars: int | None = None,
    observe: bool = True,
    with_image: bool | str | None = None,
) -> ActionResult:
    """Erase text in a field (Maestro ``eraseText``): focus + delete *chars* or clear all."""
    el = (
        self._target(element_id, selector, verb="erase")
        if (element_id is not None or selector)
        else None
    )
    with self._acting():
        if el is not None:
            cx, cy = el.center
            _input_runtime(self).click(cx, cy)
        if chars is None or chars <= 0:
            _input_runtime(self).clear_text()
        else:
            self.platform.runtime_capability("device.keyboard", self.device).erase_chars(chars)
    detail = "all" if not chars or chars <= 0 else str(chars)
    return self._observe(
        ActionResult(ok=True, action="erase", id=_published_id(el), detail=detail),
        observe,
        with_image,
    )
