"""An OCR reading joins the screen only when it could be text the app shows.

On macOS every hierarchy observation is fused with an Apple Vision pass. Measured on one
real session of 16 observations, every landing screen carried the same non-text: the
status-bar clock and signal icons read as "| g", the logo read as the digit "2", the add
button as "+", and a label the tree had right came back as a misread with an icon glued to
its front ("if. Plana trip" for "Plan a trip"). A screen with the keyboard open added its
key rows ("ASDFGH"). None of these is something the app says; each costs tokens on every
observation and can be quoted back as fact by whatever reads the screen.

The rules below are deliberately narrow. OCR is kept *for* text the tree cannot see, so a
reading in app territory that matches no node is the case it exists for and must survive.
"""

from __future__ import annotations

from android_ui_analyser.schema import Element
from android_ui_analyser.selectors import drop_ocr_noise, drop_redundant_ocr


def _node(
    eid: int,
    text: str | None,
    bounds: list[int],
    *,
    source: str = "hierarchy",
    window: str | None = "app",
    rid: str | None = None,
    clickable: bool | None = None,
) -> Element:
    return Element(
        id=eid,
        type="View" if source == "hierarchy" else "Text",
        text=text,
        resource_id=rid,
        bounds=bounds,
        center=[(bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2],
        clickable=(source == "hierarchy") if clickable is None else clickable,
        source=source,
        window=None if source == "ocr" else window,
    )


# Geometry of a 1080x2400 device, taken from the measured screens.
STATUS_BAR = _node(
    1, None, [0, 0, 1080, 136], window="system", rid="com.android.systemui:id/status_bar"
)
CLOCK_READING = _node(50, "| g", [805, 45, 934, 95], source="ocr")
KEY_D = _node(2, "D", [272, 1808, 379, 1963], window="ime")
KEY_ROW_READING = _node(51, "ASDFGH", [5, 1808, 593, 1963], source="ocr")
LOGO_READING = _node(52, "2", [495, 436, 596, 548], source="ocr")
PLUS_READING = _node(53, "+", [52, 2170, 111, 2226], source="ocr")
CHEVRON_READING = _node(54, ">", [990, 700, 1040, 760], source="ocr")
ROW = _node(3, "Plan a trip", [42, 1023, 407, 1149])
ROW_MISREAD = _node(55, "if. Plana trip", [77, 1056, 369, 1120], source="ocr")
CANVAS_SCORE = _node(56, "Score 12", [400, 900, 680, 960], source="ocr")
UNNAMED_BUTTON = _node(4, None, [63, 1834, 1017, 1960], clickable=True)
BUTTON_CAPTION = _node(57, "Continue", [440, 1870, 640, 1925], source="ocr")
OK_READING = _node(58, "OK", [480, 1200, 600, 1250], source="ocr")


def _ocr_texts(elements: list[Element]) -> list[str]:
    return [el.text or "" for el in elements if el.source == "ocr"]


def test_a_reading_inside_the_status_bar_is_not_an_element():
    out = drop_ocr_noise([STATUS_BAR, CLOCK_READING])
    assert _ocr_texts(out) == []
    assert [el.id for el in out] == [1], "the tree's own status-bar node is untouched"


def test_a_reading_inside_the_keyboard_is_not_an_element():
    out = drop_ocr_noise([KEY_D, KEY_ROW_READING])
    assert _ocr_texts(out) == []


def test_one_recognised_glyph_is_not_text():
    """A logo, an add button and a chevron each read as the character they most resemble."""
    out = drop_ocr_noise([LOGO_READING, PLUS_READING, CHEVRON_READING])
    assert out == []


def test_a_misread_with_an_icon_glued_on_and_a_moved_space_is_a_copy_of_the_label():
    """ "if." is an icon read as letters, and OCR moves spaces; both still say "Plan a trip"."""
    out = drop_redundant_ocr([ROW, ROW_MISREAD])
    assert [el.id for el in out] == [3]


def test_text_the_tree_cannot_see_survives_both_filters():
    """The reason OCR runs at all: a canvas score, and the caption of a button the tree left unnamed."""
    screen = [STATUS_BAR, UNNAMED_BUTTON, CANVAS_SCORE, BUTTON_CAPTION, OK_READING]
    out = drop_ocr_noise(drop_redundant_ocr(screen))
    assert _ocr_texts(out) == ["Score 12", "Continue", "OK"]


def test_the_measured_landing_screen_keeps_no_noise():
    """All four readings the real screen carried, together, against the tree that produced them."""
    screen = [
        STATUS_BAR,
        ROW,
        UNNAMED_BUTTON,
        CLOCK_READING,
        LOGO_READING,
        ROW_MISREAD,
        PLUS_READING,
    ]
    out = drop_ocr_noise(drop_redundant_ocr(screen))
    assert _ocr_texts(out) == []
    assert [el.id for el in out] == [1, 3, 4]
