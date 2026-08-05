"""Element selectors — match by resource-id / text / content-desc (and route-step helpers).

Public entry points used by the engine and re-exported from :mod:`engine` for
``from android_ui_analyser.engine import match_selector`` compatibility.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import Any

from .memory import REDACT_TOKENS, RouteStep, _id_tail
from .schema import Element

_SELECTOR_FIELDS = ("rid", "text", "desc")
_MAX_CANDIDATES = 8  # candidate elements echoed back in an ambiguous/not-found error
_NEAREST_FLOOR = 0.3  # similarity below which a "did you mean" suggestion is noise


def selector_label(selector: dict[str, Any]) -> str:
    """``rid:homeTabBROWSE`` — how a selector is echoed back in errors."""
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

    Tiering is what makes ``--text Browse`` usable on a screen that also has a
    "Browse more apps" row: an exact hit never gets drowned in substring hits.
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


_NEAR = 0.88  # similarity above which an OCR reading is the same text, misread
_NEAR_MIN_LEN = 6  # below this, a near match is more likely a different word
_GLYPH_MAX_LEN = 2  # an edge token this short is an icon read as letters, not a word


def _strip_glyph_tokens(text: str) -> str:
    """*text* without one- or two-character tokens at either end.

    OCR reads an icon sitting next to a label as a letter, so a button captioned
    "Continue with Google" with the Google mark beside it comes back as "G Continue with
    Google". That reading is *longer* than the tree's text, so a containment test - even a
    fuzzy one - keeps it, and the screen carries a duplicate that differs from the truth
    only by a glyph.

    Only the ends are trimmed, and only tokens too short to be words, so a reading that
    adds real content keeps it: "Save all changes" against a tree that says "Save" still
    survives, because "all" and "changes" are words.
    """
    parts = text.split()
    while parts and len(parts[0]) <= _GLYPH_MAX_LEN:
        parts.pop(0)
    while parts and len(parts[-1]) <= _GLYPH_MAX_LEN:
        parts.pop()
    return " ".join(parts)


def _near_contained(needle: str, haystack: str) -> bool:
    """Is *needle* a slightly-misread copy of some window of *haystack*?

    Both arguments are already casefolded. Compares only equal-length windows, because a
    misreading changes characters rather than length; that keeps this a same-text test
    rather than a general fuzzy search.
    """
    n = len(needle)
    if n < _NEAR_MIN_LEN or n > len(haystack):
        return False
    matcher = difflib.SequenceMatcher(a=needle, autojunk=False)
    for start in range(len(haystack) - n + 1):
        matcher.set_seq2(haystack[start : start + n])
        if matcher.real_quick_ratio() < _NEAR or matcher.quick_ratio() < _NEAR:
            continue
        if matcher.ratio() >= _NEAR:
            return True
    return False


def drop_redundant_ocr(elements: Sequence[Element]) -> list[Element]:
    """*elements* without OCR readings of text the hierarchy already reports.

    On macOS every hierarchy observation is fused with an Apple Vision pass, so text the
    tree already describes is described a second time as recognised pixels. For ``analyze``
    both are wanted. For a one-shot selector the second copy is destructive, in two ways:

    * A plain duplicate makes ``tap --text Settings`` raise "matches 2 elements" on a screen
      with one Settings tab.
    * Worse, OCR reads by visual line, so a composite label splits into words. A card
      labelled "Settings and privacy options" yields a fragment reading exactly
      ``Settings``, which is an *exact* match and therefore ties with the real tab - even
      though the hierarchy would have ranked that card below it as a mere substring. Tiering
      is what makes short labels usable, and an OCR fragment silently defeats it.

    Observed on a real bottom-bar layout: 7 of one screen's labels affected, every tab among
    them. So this runs before matching, not after - by the time tiers are computed the
    fragment has already been promoted and the damage is done.

    A reading is redundant when some non-OCR element both encloses it and already carries
    that text. Both conditions matter: enclosure alone would discard a genuine second
    occurrence elsewhere on screen, and text alone would discard web content that happens
    to repeat a word from the surrounding tree. Where the hierarchy is silent - a Chrome
    Custom Tab, a Flutter surface - nothing is enclosing and describing the text, so OCR
    survives untouched and remains the only witness there is.

    **Near matches count as redundant too**, above ``_NEAR`` similarity. OCR confuses
    letterforms - a capital I read as a lowercase L turns "Talk to AI personalities" into
    "Talk to Al personalities" - and an exact-substring test keeps precisely those, so the
    fused screen ends up carrying a subtly *wrong* label for text the tree had right. A
    misreading that survives because it is wrong is the worst of both worlds: an agent
    quoting screen copy, or a scenario asserting exact wording, reports it as fact. The
    threshold is high and short strings are exempt, so genuinely different labels
    ("Save" against "Sale") stay separate.

    **Never drops a repair of lossy text.** When the tree could not represent a character it
    emits U+FFFD, and then the OCR reading is the only correct account of that text - the
    opposite of redundant. Such a neighbour is skipped rather than matched against.

    **Overlap, not centre-containment.** An OCR box rarely lines up with the node that
    produced it - recognised glyphs sit lower and wider than the text view's bounds. Measured
    on a real screen: the tree had "Apps" at y 64-88 and OCR returned it at y 76-122, so the
    box's *centre* landed in the card underneath and the reading was compared against the
    wrong element's text. It then survived as a duplicate. Any overlapping node is a
    candidate now; the text test still does the deciding, so a neighbour that merely touches
    the box cannot cause a drop unless it already says the same thing.
    """
    solid = [el for el in elements if el.source != "ocr"]
    if not solid:
        return list(elements)

    def redundant(el: Element) -> bool:
        seen = (el.text or "").strip().casefold()
        if not seen:
            return False
        box = el.bounds
        for h in solid:
            b = h.bounds
            if b[2] < box[0] or b[0] > box[2] or b[3] < box[1] or b[1] > box[3]:
                continue  # no overlap at all
            known = f"{h.text or ''} {h.content_desc or ''}"
            if "�" in known:
                continue  # the tree lost characters here; OCR is the repair, not a copy
            known = known.casefold()
            for candidate in (seen, _strip_glyph_tokens(seen)):
                if candidate and (candidate in known or _near_contained(candidate, known)):
                    return True
        return False

    return [el for el in elements if el.source != "ocr" or not redundant(el)]


def ocr_added_app_content(elements: Sequence[Element]) -> bool:
    """Did the OCR pass contribute anything about the *app*, rather than system chrome?

    This is the evidence behind the experience-based OCR skip, so it has to mean what it
    says. Asking "were any OCR elements kept?" does not: the status-bar clock is read on
    every single screen, its digits never match the tree's, and it therefore survives every
    redundancy test. One clock reading is enough to score every visit as "OCR helped", which
    leaves `hierarchy_only_ok` at zero forever and quietly prevents the skip from ever
    engaging - the optimisation is then dead code that still pays for itself on every call.

    A reading counts as system chrome when a system element encloses it, which is exactly
    how the clock, the battery and the signal icons present. Anything in app territory, or
    over no tree element at all, counts as real.
    """
    from .projection import is_system_rid

    chrome = [
        el
        for el in elements
        if el.source != "ocr" and (is_system_rid(el.resource_id) or el.window == "system")
    ]
    for el in elements:
        if el.source != "ocr" or not (el.text or "").strip():
            continue
        cx, cy = el.center
        inside_chrome = any(
            c.bounds[0] <= cx <= c.bounds[2] and c.bounds[1] <= cy <= c.bounds[3]
            for c in chrome
        )
        if not inside_chrome:
            return True
    return False


def app_elements(elements: Sequence[Element]) -> list[Element]:
    """*elements* minus system chrome (``com.android.systemui`` & friends).

    A miss on an app selector is never explained by the status bar, so the diagnostic
    counts and candidate lists are computed over the app's own elements.
    """
    from .projection import is_system_rid

    return [el for el in elements if not is_system_rid(el.resource_id)]


def nearest_elements(
    elements: Sequence[Element], needle: str, limit: int = 5
) -> list[Element]:
    """Best "did you mean" candidates for a selector that matched nothing.

    App elements are ranked alone; system chrome is only offered when the app contributed
    nothing at all, because four status-bar ids ranked above nothing is worse than an empty
    hint — it reads as a real answer and sends the caller looking in the wrong place.
    """
    import difflib

    def score(el: Element) -> float:
        hays = [h for h in (el.text, el.content_desc, _id_tail(el.resource_id)) if h]
        return max(
            (difflib.SequenceMatcher(None, needle.lower(), h.lower()).ratio() for h in hays),
            default=0.0,
        )

    pool = app_elements(elements) or list(elements)
    ranked = sorted(((score(el), el) for el in pool), key=lambda pair: pair[0], reverse=True)
    near = [el for value, el in ranked if value >= _NEAREST_FLOOR][:limit]
    return near or [el for _v, el in ranked[:limit]]


def _pick(candidates: list[Element], index: int | None) -> Element | None:
    """The nth candidate when the step asked for one, else the best/first.

    Out of range returns None rather than falling back to the first match: a flow that
    said "the second See all" and got the first would tap the wrong thing and carry on,
    which is the failure mode `index:` exists to prevent. Diverging is the safe direction.
    """
    if not candidates:
        return None
    if index is None:
        return candidates[0]
    return candidates[index] if 0 <= index < len(candidates) else None


def match_step(elements: list[Element], step: RouteStep) -> Element | None:
    """Resolve a step's target element: resource-id tail first, then label.

    Redacted labels never match — a step whose only identity was PII hands off rather
    than guessing. Label matching keeps the legacy tolerance (exact, then
    prefix/substring for truncation drift).

    ``step.index`` disambiguates a selector that legitimately matches several elements,
    matching the CLI's ``--index``. It applies within whichever tier produced the matches,
    so the rid/exact/substring precedence is unchanged when no index is given.
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
            return _pick(matches, step.index)
    label = (step.label or "").strip()
    if not label or label in REDACT_TOKENS:
        return None
    exact = [e for e in elements if (e.text or e.content_desc or "") == label]
    if exact:
        return _pick(exact, step.index)
    low = label.lower()
    loose = [
        e
        for e in elements
        if (t := (e.text or e.content_desc or "").lower()) and (t.startswith(low) or low in t)
    ]
    return _pick(loose, step.index)


# Private alias kept for engine call sites that used ``_match_step``.
_match_step = match_step
