"""Element selectors — match by resource-id / text / content-desc (and route-step helpers).

Public entry points used by the engine and re-exported from :mod:`engine` for
``from android_ui_analyser.engine import match_selector`` compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .memory import REDACT_TOKENS, RouteStep, _id_tail
from .schema import Element

_SELECTOR_FIELDS = ("rid", "text", "desc")
_MAX_CANDIDATES = 8  # candidate elements echoed back in an ambiguous/not-found error
_NEAREST_FLOOR = 0.3  # similarity below which a "did you mean" suggestion is noise


def selector_label(selector: dict[str, Any]) -> str:
    """``rid:appsHubTabEXPLORE`` — how a selector is echoed back in errors."""
    for field in _SELECTOR_FIELDS:
        value = selector.get(field)
        if value:
            return f"{field}:{value}"
    return "<empty>"


def element_digest(el: Element) -> str:
    """One-line candidate description carrying everything needed to pick the right id."""
    parts = [f"id={el.id}"]
    if el.text:
        parts.append(f"text={el.text!r}")
    if el.content_desc and el.content_desc != el.text:
        parts.append(f"desc={el.content_desc!r}")
    if el.resource_id:
        parts.append(f"rid={_id_tail(el.resource_id)!r}")
    parts.append(f"type={el.type}")
    return " ".join(parts)


def _rid_tier(resource_id: str | None, needle: str) -> int | None:
    """Match rank for a resource-id (0 = best), or ``None`` for no match.

    Both forms of the id must work: the fully-qualified ``pkg:id/name`` a dump reports and
    the bare ``name`` an agent reads off the app's source.
    """
    if not resource_id:
        return None
    tail = _id_tail(resource_id) or ""
    if resource_id == needle:
        return 0
    if tail == needle:
        return 1
    if tail.lower() == needle.lower():
        return 2
    if needle.lower() in resource_id.lower():
        return 3
    return None


def _text_tier(el: Element, needle: str, *, desc_only: bool = False) -> int | None:
    """Match rank for a label (0 = exact), or ``None``. Exact always beats substring."""
    hays = [el.content_desc] if desc_only else [el.text, el.content_desc]
    values = [h.strip() for h in hays if h]
    if any(v == needle for v in values):
        return 0
    if any(v.lower() == needle.lower() for v in values):
        return 1
    if any(needle.lower() in v.lower() for v in values):
        return 2
    return None


def match_selector(
    elements: Sequence[Element],
    *,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
) -> list[Element]:
    """Every element matching the selector, at its single best tier.

    Tiering is what makes ``--text Explore`` usable on a screen that also has an
    "Explore more apps" row: an exact hit never gets drowned in substring hits.
    """
    tiers: dict[int, list[Element]] = {}
    for el in elements:
        if rid:
            tier = _rid_tier(el.resource_id, rid)
        elif text:
            tier = _text_tier(el, text)
        else:
            tier = _text_tier(el, desc or "", desc_only=True)
        if tier is not None:
            tiers.setdefault(tier, []).append(el)
    return tiers[min(tiers)] if tiers else []


def nearest_elements(
    elements: Sequence[Element], needle: str, limit: int = 5
) -> list[Element]:
    """Best "did you mean" candidates for a selector that matched nothing."""
    import difflib

    def score(el: Element) -> float:
        hays = [h for h in (el.text, el.content_desc, _id_tail(el.resource_id)) if h]
        return max(
            (difflib.SequenceMatcher(None, needle.lower(), h.lower()).ratio() for h in hays),
            default=0.0,
        )

    ranked = sorted(((score(el), el) for el in elements), key=lambda pair: pair[0], reverse=True)
    near = [el for value, el in ranked if value >= _NEAREST_FLOOR][:limit]
    return near or [el for _v, el in ranked[:limit]]


def match_step(elements: list[Element], step: RouteStep) -> Element | None:
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


# Private alias kept for engine call sites that used ``_match_step``.
_match_step = match_step
