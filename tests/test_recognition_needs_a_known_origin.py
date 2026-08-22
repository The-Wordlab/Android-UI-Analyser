"""`destination_confirmed` requires a known origin, everywhere it is trusted.

Found while building Phase 0 and fixed globally here: the flag meant "recognised name differs
from the before name", and on a cold session (async memory, first action after the first
analyze) the before name is still unstamped — so recognising *anything*, including the
origin's own map entry, counted as a confirmed destination. Recognition with no known origin
has no differential power; it must not clear a stale caveat, and it must not suppress the
unready-destination verdict.
"""

from __future__ import annotations

from android_ui_analyser.engine import Engine


def test_recognition_without_a_known_origin_confirms_nothing() -> None:
    assert Engine._destination_confirmed("auth screen", None) is False


def test_recognition_of_a_different_screen_from_a_known_origin_confirms() -> None:
    assert Engine._destination_confirmed("auth screen", "entry screen") is True


def test_recognising_the_screen_we_left_confirms_nothing() -> None:
    assert Engine._destination_confirmed("entry screen", "entry screen") is False


def test_no_recognition_confirms_nothing() -> None:
    assert Engine._destination_confirmed(None, "entry screen") is False
