"""When a label matches two genuine nodes and only one is tappable, tap that one.

Found by a sweep lane and diagnosed by it precisely, which is why this fix is narrow.
`tap --text "Attach Photos"` raised `selector_ambiguous` on a Compose bottom sheet, and the
lane established what the candidates actually were:

    id=25  View      clickable=True   bounds=[32,840,352,1000]    text="Attach Photos"
    id=27  TextView  clickable=False  bounds=[98,1016,287,1051]   text="Attach Photos"

Both are **hierarchy** nodes - OCR elements in the same capture carried `source="ocr"` and
these carried none - and their boxes do not overlap at all, 16px apart. So this is not the
fused-OCR duplicate this project fixed twice today, and no dedup rule can address it: neither
element is a second reading of the other. They are a clickable icon tile and the caption
beneath it, both legitimately carrying the same label because `mergeDescendants` did not
collapse them.

What resolves it is the verb. The caller said *tap*, and only one candidate can be tapped, so
picking it is the only interpretation that can be carried out - not a guess. Two tappable
matches stay ambiguous, because then the ambiguity is real.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import SelectorAmbiguousError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen

TILE = Element(
    id=25, type="View", text="Attach Photos", clickable=True,
    bounds=[32, 840, 352, 1000], center=[192, 920],
)
CAPTION = Element(
    id=27, type="TextView", text="Attach Photos", clickable=False,
    bounds=[98, 1016, 287, 1051], center=[192, 1033],
)


def _result(*elements: Element) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=720, height=1280, package="com.example", source="hierarchy"),
        elements=list(elements),
        meta=Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy"),
    )


class _FakeEngine:
    _last_analyze_result = None

    def __init__(self, tree):
        self._tree = tree

    def analyze(self, *, source="auto", record=True, **_kw):
        return self._tree

    def _read_cache(self):
        return None

    def _note_screen_moved(self, _shown, _fresh):
        """Not what this double is for: it exercises the tie-break, not the caller's turn."""
        return None

    def _resolve_container_rid(self, _rid):
        return None


def _bind(tree):
    from android_ui_analyser.engine import Engine

    e = _FakeEngine(tree)
    e.resolve_selector = Engine.resolve_selector.__get__(e)
    e._match_by_vision = Engine._match_by_vision.__get__(e)
    e._target = Engine._target.__get__(e)
    return e


def test_tap_takes_the_clickable_tile():
    engine = _bind(_result(TILE, CAPTION))
    assert engine.resolve_selector(text="Attach Photos", prefer_clickable=True).id == 25


def test_a_tap_verb_enables_the_tie_break_without_being_asked():
    """`aua tap --text ...` must work with no extra flag - that is the point."""
    engine = _bind(_result(TILE, CAPTION))
    assert engine._target(None, {"text": "Attach Photos"}, verb="tap").id == 25


def test_a_non_tap_verb_still_reports_the_ambiguity():
    """`has`/`wait` are not asking to act, so the ambiguity is information, not an obstacle."""
    engine = _bind(_result(TILE, CAPTION))
    with pytest.raises(SelectorAmbiguousError):
        engine._target(None, {"text": "Attach Photos"}, verb="wait")


def test_two_clickable_matches_stay_ambiguous():
    """Real ambiguity must keep raising: silently picking one hides a wrong tap."""
    other = Element(
        id=26, type="View", text="Attach Photos", clickable=True,
        bounds=[368, 840, 688, 1000], center=[528, 920],
    )
    engine = _bind(_result(TILE, other))
    with pytest.raises(SelectorAmbiguousError):
        engine.resolve_selector(text="Attach Photos", prefer_clickable=True)


def test_no_clickable_match_stays_ambiguous():
    engine = _bind(_result(CAPTION, Element(
        id=28, type="TextView", text="Attach Photos", clickable=False,
        bounds=[454, 1016, 602, 1051], center=[528, 1033],
    )))
    with pytest.raises(SelectorAmbiguousError):
        engine.resolve_selector(text="Attach Photos", prefer_clickable=True)


def test_explicit_first_still_wins():
    engine = _bind(_result(CAPTION, TILE))
    assert engine.resolve_selector(text="Attach Photos", first=True).id == 27


def test_a_single_match_is_unaffected():
    engine = _bind(_result(TILE))
    assert engine.resolve_selector(text="Attach Photos", prefer_clickable=True).id == 25
