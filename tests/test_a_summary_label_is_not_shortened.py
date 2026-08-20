"""Summary labels are normalised, never cut.

`next_actions[].label` used to be sliced at 60 characters. Measured on 2026-08-10, that bought
nothing: the same string is already in `elements[].text` at full length in the same response, and
on the densest screen observed every label together came to 149 characters inside a 9,915-character
payload — with no row even reaching the limit.

It did cost something, though. A heading longer than the limit came back as a sentence that simply
stops, which reads as complete, and two agent runs took it at face value and spent an extra
`analyze` recovering text they had already been handed. Truncating one copy of a string the
response carries in full elsewhere is all downside.
"""

from __future__ import annotations

from android_ui_analyser.engine import _label


def test_a_long_label_survives_whole() -> None:
    heading = (
        "Create your account To enjoy the product you need to create an account first. "
        "Continue with Example ID Maybe later By continuing,"
    )
    assert _label(heading) == heading, "the same text is already in elements[].text in full"


def test_a_short_label_is_unchanged() -> None:
    assert _label("Continue with Example ID") == "Continue with Example ID"


def test_newlines_become_spaces() -> None:
    """A row has to stay one line; that is the only shaping a label needs."""
    assert _label("first\nsecond") == "first second"


def test_surrounding_whitespace_goes() -> None:
    assert _label("  Maybe later \n") == "Maybe later"


def test_nothing_is_marked_as_clipped_any_more() -> None:
    assert "…" not in _label("z" * 500)
