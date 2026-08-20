"""Text the hierarchy already reports must not be counted twice by a selector.

The bug this guards: on macOS every hierarchy observation is fused with an Apple Vision pass,
so `tap --text Catalog` raised "matches 2 elements - disambiguate with --index" on a synthetic
screen with exactly one Catalog tab.

Two distinct failures hide behind that one message:

* a plain duplicate - the same control read once as a node and once as pixels;
* an OCR *fragment* - OCR reads by visual line, so the card "Catalog Browse available products"
  yields a box reading exactly `Catalog`. That is an exact match, so it ties with the
  real tab, where the hierarchy would have ranked the card below it as a mere substring. An
  OCR fragment silently defeats the tiering that makes short labels usable at all, which is
  why the fix has to run before matching rather than after.

Ambiguity still has to mean something: two genuinely distinct controls sharing a label must
keep raising, because picking one silently is indistinguishable from the app ignoring a valid
tap.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import SelectorAmbiguousError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen
from android_ui_analyser.selectors import drop_redundant_ocr, match_selector


def _node(
    eid: int, text: str, bounds: list[int], *, source: str = "hierarchy", rid: str | None = None
) -> Element:
    return Element(
        id=eid,
        type="View" if source == "hierarchy" else "Text",
        text=text,
        resource_id=rid,
        bounds=bounds,
        center=[(bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2],
        clickable=source == "hierarchy",
        source=source,
    )


# Synthetic geometry for a 1080x2400 device.
TAB_CATALOG = _node(4, "Catalog", [780, 2100, 1040, 2240], rid="catalogTab")
OCR_ON_TAB = _node(51, "Catalog", [840, 2180, 990, 2230], source="ocr")
CARD_CATALOG = _node(9, "Catalog Browse available products", [40, 1100, 1040, 1260])
OCR_CARD_FRAGMENT = _node(52, "Catalog", [150, 1150, 310, 1210], source="ocr")


def test_reading_of_the_same_control_is_dropped():
    out = drop_redundant_ocr([TAB_CATALOG, OCR_ON_TAB])
    assert [el.id for el in out] == [4]


def test_fragment_of_a_longer_label_is_dropped():
    """The card's own text contains "Catalog", so the fragment is a reading of the card."""
    out = drop_redundant_ocr([CARD_CATALOG, OCR_CARD_FRAGMENT])
    assert [el.id for el in out] == [9]


def test_the_screen_resolves_to_exactly_one_catalog_tab():
    """The end-to-end shape of the bug: tab, card, and an OCR reading of each."""
    screen = [TAB_CATALOG, CARD_CATALOG, OCR_ON_TAB, OCR_CARD_FRAGMENT]

    matches = match_selector(drop_redundant_ocr(screen), text="Catalog")

    assert [el.id for el in matches] == [4], "the tab wins by exact match; the card is a substring"


def test_content_desc_also_counts_as_already_reported():
    icon = _node(3, "", [874, 2193, 1069, 2319])
    icon.content_desc = "Catalog"
    assert drop_redundant_ocr([icon, OCR_ON_TAB]) == [icon]


def test_two_real_controls_sharing_a_label_stay_ambiguous():
    a = _node(1, "Continue", [60, 1550, 520, 1660])
    b = _node(2, "Continue", [550, 1550, 1020, 1660])
    assert len(match_selector(drop_redundant_ocr([a, b]), text="Continue")) == 2


def test_web_content_survives_untouched():
    """A Custom Tab's WebView reports a title and nothing else; OCR is the only witness."""
    webview = _node(30, "Sign in - Google Accounts", [0, 142, 1080, 2175])
    button = _node(68, "Continue", [700, 2020, 870, 2060], source="ocr")

    out = drop_redundant_ocr([webview, button])

    assert 68 in [el.id for el in out], "dropping this makes an OAuth page untappable again"


def test_a_second_occurrence_elsewhere_is_not_dropped():
    """Enclosure matters: the same word further down the screen is a real candidate."""
    node = _node(1, "Save", [60, 300, 300, 380])
    far = _node(9, "Save", [700, 1800, 860, 1850], source="ocr")
    assert len(drop_redundant_ocr([node, far])) == 2


def test_only_ocr_on_screen_is_left_alone():
    only = [_node(7, "Continue", [700, 2020, 870, 2060], source="ocr")]
    assert drop_redundant_ocr(only) == only


# ---- through resolve_selector, which is where it actually bit ----


def _result(*elements: Element) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2424, package="com.example", source="hierarchy"),
        elements=list(elements),
        meta=Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy"),
    )


class _FakeEngine:
    def __init__(self, tree):
        self._tree = tree

    def analyze(self, *, source="auto", record=True, **_kw):
        return self._tree

    def _read_cache(self):
        return None

    def _resolve_container_rid(self, _rid):
        return None


def _bind(engine):
    from android_ui_analyser.engine import Engine

    engine.resolve_selector = Engine.resolve_selector.__get__(engine)
    engine._match_by_vision = Engine._match_by_vision.__get__(engine)
    return engine


def test_resolve_selector_taps_the_hierarchy_tab():
    engine = _bind(_FakeEngine(_result(TAB_CATALOG, CARD_CATALOG, OCR_ON_TAB, OCR_CARD_FRAGMENT)))
    assert engine.resolve_selector(text="Catalog").id == 4


def test_resolve_selector_still_raises_on_real_ambiguity():
    engine = _bind(
        _FakeEngine(
            _result(
                _node(1, "Continue", [60, 1550, 520, 1660]),
                _node(2, "Continue", [550, 1550, 1020, 1660]),
            )
        )
    )
    with pytest.raises(SelectorAmbiguousError):
        engine.resolve_selector(text="Continue")


# ---- near matches: an OCR misreading is the same text, not a new claim ----


def test_letterform_misread_is_still_redundant():
    """OCR read the capital I in "AI" as a lowercase L.

    An exact-substring test keeps precisely the readings that are *wrong*, so the fused
    screen ends up carrying a subtly incorrect label for text the tree had right.
    """
    card = _node(9, "Catalog Browse available products", [42, 863, 1038, 1010])
    misread = _node(60, "Browse avallable products", [186, 950, 560, 990], source="ocr")

    assert drop_redundant_ocr([card, misread]) == [card]


def test_different_labels_are_not_folded_together():
    """The near test must not become a fuzzy search. "Sale" is not "Save"."""
    node = _node(1, "Save changes now", [60, 300, 500, 380])
    other = _node(9, "Sale changes now", [700, 1800, 1000, 1850], source="ocr")

    assert len(drop_redundant_ocr([node, other])) == 2, "different text, and not enclosed"


def test_short_strings_do_not_near_match():
    """Below the length floor a one-character difference is usually a different word."""
    node = _node(1, "Maps", [60, 300, 300, 380])
    ocr = _node(9, "Caps", [100, 320, 200, 360], source="ocr")

    assert len(drop_redundant_ocr([node, ocr])) == 2


def test_lossy_text_repair_is_never_dropped():
    """Where the tree emitted U+FFFD, the OCR reading is the only correct account."""
    lossy = _node(5, "Divide both sides by 2 to solve for �: �", [42, 500, 1038, 600])
    repair = _node(
        70, "Divide both sides by 2 to solve for x: 5", [50, 520, 900, 580], source="ocr"
    )

    out = drop_redundant_ocr([lossy, repair])

    assert 70 in [el.id for el in out], "dropping this re-breaks the lossy-text repair"


# ---- glyph artifacts, and what "OCR helped" is allowed to mean ----

from android_ui_analyser.selectors import ocr_added_app_content  # noqa: E402


def test_icon_read_as_a_letter_is_still_redundant():
    """The Google mark beside a button comes back as a leading "G".

    The reading is *longer* than the tree's text, so containment - even fuzzy - keeps it,
    and the screen carries a duplicate differing only by a glyph. Worse than cosmetic: it
    keeps `ocr_helped` true forever, which pins `hierarchy_only_ok` at zero and disables
    the experience-based OCR skip on that screen for good.
    """
    button = _node(9, "Continue with Example ID", [60, 700, 1020, 840])
    reading = _node(60, "E Continue with Example ID", [80, 750, 900, 790], source="ocr")

    assert drop_redundant_ocr([button, reading]) == [button]


def test_a_reading_that_adds_real_words_survives():
    """Only short edge tokens are trimmed, so added content is never discarded."""
    node = _node(1, "Save", [60, 300, 600, 420])
    more = _node(9, "Save all changes", [80, 330, 500, 380], source="ocr")

    assert len(drop_redundant_ocr([node, more])) == 2


def test_clock_alone_does_not_count_as_ocr_helping():
    """The status-bar clock is read on every screen and never matches the tree's digits.

    Counting it would score every visit as "OCR helped", which is how an optimisation ends
    up dead code that still costs a pass on every call.
    """
    clock = _node(0, "1:18", [11, 49, 115, 92], rid="com.android.systemui:id/clock")
    clock.window = "system"
    reading = _node(50, "10:19 A", [38, 56, 154, 91], source="ocr")

    assert ocr_added_app_content([clock, reading]) is False


def test_real_app_text_does_count_as_ocr_helping():
    webview = _node(30, "Sign in - Google Accounts", [0, 142, 1080, 2175])
    button = _node(68, "Continue", [700, 2020, 870, 2060], source="ocr")

    assert ocr_added_app_content([webview, button]) is True


def test_offset_reading_matches_the_node_it_came_from():
    """Recognised glyphs sit lower and wider than the text view that produced them.

    Synthetic 720x1280 geometry: the tree has "Catalog" at y 64-92 and OCR returns it at
    y 76-122. Centre-containment puts the box in the card underneath, compares the reading
    against that card's text, found no match, and kept a duplicate. Overlap is the honest
    test - the box plainly covers the header.
    """
    header = _node(9, "Catalog", [32, 64, 180, 92])
    card = _node(10, "Sport Your personalised exercise routine", [32, 88, 352, 374])
    reading = _node(60, "Catalog", [30, 76, 184, 122], source="ocr")

    out = drop_redundant_ocr([header, card, reading])

    assert [el.id for el in out] == [9, 10], "the offset reading is still a duplicate"


def test_overlap_alone_never_drops_a_reading():
    """A neighbour that merely touches the box must not cause a drop.

    Overlap widens the candidate set, so the text test has to carry the decision. If a box
    grazing an unrelated card were enough, genuinely new content would vanish.
    """
    card = _node(10, "Sport Your personalised exercise routine", [32, 88, 352, 374])
    reading = _node(61, "Beat maker", [30, 76, 144, 122], source="ocr")

    assert 61 in [el.id for el in drop_redundant_ocr([card, reading])]
