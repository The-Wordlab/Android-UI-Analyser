"""Text the hierarchy already reports must not be counted twice by a selector.

The bug this guards: on macOS every hierarchy observation is fused with an Apple Vision pass,
so `tap --text Apps` raised "matches 2 elements - disambiguate with --index" on a screen with
exactly one Apps tab. Observed on a real app's home screen: 7 labels affected, every
bottom-bar tab among them - which is how a test suite navigates.

Two distinct failures hide behind that one message:

* a plain duplicate - the same control read once as a node and once as pixels;
* an OCR *fragment* - OCR reads by visual line, so the card "Apps Tools to make your life
  easier" yields a box reading exactly `Apps`. That is an exact match, so it ties with the
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


# Real geometry, captured from an app's home screen on a 1080x2424 device.
TAB_APPS = _node(4, "Apps", [874, 2193, 1069, 2319], rid="bottomBarTools")
OCR_ON_TAB = _node(51, "Apps", [922, 2280, 1017, 2318], source="ocr")
CARD_APPS = _node(9, "Apps Tools to make your life easier", [32, 1180, 1048, 1320])
OCR_CARD_FRAGMENT = _node(52, "Apps", [186, 1230, 281, 1268], source="ocr")


def test_reading_of_the_same_control_is_dropped():
    out = drop_redundant_ocr([TAB_APPS, OCR_ON_TAB])
    assert [el.id for el in out] == [4]


def test_fragment_of_a_longer_label_is_dropped():
    """The card's own text contains "Apps", so the fragment is a reading of the card."""
    out = drop_redundant_ocr([CARD_APPS, OCR_CARD_FRAGMENT])
    assert [el.id for el in out] == [9]


def test_the_hub_resolves_to_exactly_one_apps_tab():
    """The end-to-end shape of the bug: tab, card, and an OCR reading of each."""
    screen = [TAB_APPS, CARD_APPS, OCR_ON_TAB, OCR_CARD_FRAGMENT]

    matches = match_selector(drop_redundant_ocr(screen), text="Apps")

    assert [el.id for el in matches] == [4], "the tab wins by exact match; the card is a substring"


def test_content_desc_also_counts_as_already_reported():
    icon = _node(3, "", [874, 2193, 1069, 2319])
    icon.content_desc = "Apps"
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


def test_resolve_selector_taps_the_real_tab():
    engine = _bind(_FakeEngine(_result(TAB_APPS, CARD_APPS, OCR_ON_TAB, OCR_CARD_FRAGMENT)))
    assert engine.resolve_selector(text="Apps").id == 4


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
    card = _node(9, "Chat Talk to AI personalities", [42, 863, 1038, 1010])
    misread = _node(60, "Talk to Al personalities", [186, 950, 560, 990], source="ocr")

    assert drop_redundant_ocr([card, misread]) == [card]


def test_different_labels_are_not_folded_together():
    """The near test must not become a fuzzy search. "Sale" is not "Save"."""
    node = _node(1, "Save changes now", [60, 300, 500, 380])
    other = _node(9, "Sale changes now", [700, 1800, 1000, 1850], source="ocr")

    assert len(drop_redundant_ocr([node, other])) == 2, "different text, and not enclosed"


def test_short_strings_do_not_near_match():
    """Below the length floor a one-character difference is usually a different word."""
    node = _node(1, "Maps", [60, 300, 300, 380])
    ocr = _node(9, "Apps", [100, 320, 200, 360], source="ocr")

    assert len(drop_redundant_ocr([node, ocr])) == 2


def test_lossy_text_repair_is_never_dropped():
    """Where the tree emitted U+FFFD, the OCR reading is the only correct account."""
    lossy = _node(5, "Divide both sides by 2 to solve for �: �", [42, 500, 1038, 600])
    repair = _node(70, "Divide both sides by 2 to solve for x: 5", [50, 520, 900, 580], source="ocr")

    out = drop_redundant_ocr([lossy, repair])

    assert 70 in [el.id for el in out], "dropping this re-breaks the lossy-text repair"
