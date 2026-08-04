"""Two links on one line must be separately tappable.

Found by a sweep lane, not by imagination: the auth landing shows "Terms of use and Privacy
policy" as a single underlined run where each phrase is its own tappable span. Android does not
publish a ClickableSpan as a separate accessibility node, and OCR groups by visual line, so the
tree and the pixels both describe that line as ONE element spanning both links.

`tap --text "Terms of use"` therefore matched the whole line and tapped its centre - which is
between the two links, hitting neither. The lane's only way through was
`adb shell input tap <x> <y>` with coordinates measured off a screenshot, which is precisely
the coordinate-guessing this tool exists to remove.

Real geometry from that run: bounds [152,1087]-[570,1122] on a 720x1280 screen.
"""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import Element

LINE = Element(
    id=7,
    type="TextView",
    text="Terms of use and Privacy policy",
    bounds=[152, 1087, 570, 1122],
    center=[361, 1104],
)


def _point(el: Element, needle: str | None):
    return Engine._tap_point(object.__new__(Engine), el, needle)


def test_first_phrase_lands_left_of_centre():
    x, y = _point(LINE, "Terms of use")
    assert y == 1104, "vertical position is unchanged - it is one line"
    assert 152 < x < 361, f"should aim into the first phrase, got {x}"


def test_second_phrase_lands_right_of_centre():
    x, _ = _point(LINE, "Privacy policy")
    assert 361 < x < 570, f"should aim into the second phrase, got {x}"


def test_the_two_phrases_do_not_collide():
    assert _point(LINE, "Terms of use")[0] < _point(LINE, "Privacy policy")[0]


def test_exact_match_uses_the_centre():
    """When the element IS the phrase, the centre is already correct."""
    el = Element(id=1, type="Button", text="Continue", bounds=[0, 0, 200, 80], center=[100, 40])
    assert _point(el, "Continue") == (100, 40)


def test_no_needle_uses_the_centre():
    assert _point(LINE, None) == (361, 1104)


def test_phrase_absent_from_the_text_uses_the_centre():
    assert _point(LINE, "Cookie settings") == (361, 1104)


def test_a_multiline_block_is_left_alone():
    """Proportional aiming assumes one line; on a paragraph it would wander onto another."""
    block = Element(
        id=2,
        type="TextView",
        text="By continuing you agree to the Terms of use and the Privacy policy",
        bounds=[40, 900, 680, 1100],  # 640x200 - not a single line
        center=[360, 1000],
    )
    assert _point(block, "Privacy policy") == (360, 1000)


def test_the_point_stays_inside_the_element():
    for needle in ("Terms", "Terms of use", "and", "Privacy", "Privacy policy", "policy"):
        x, _ = _point(LINE, needle)
        assert 152 <= x <= 570, f"{needle!r} aimed outside the element at {x}"
