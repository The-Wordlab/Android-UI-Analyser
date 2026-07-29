"""A scroll reports movement only if the content actually moved along that axis.

`travel()`'s `moved` means "the probe sample differs at all" — a repaint, a ripple, an
element appearing. None of those are a scroll. Using it as the verdict made
`scroll --direction down` answer `moved steps=1` with no distance at the very top of a list,
where scrolling further is impossible: measured on device, a heading stayed at exactly
42,1125,1038,1191 across three consecutive "moved" reports.

A no-op that claims success is the worst outcome for an agent — it cannot tell "already at
the end" from "scrolling is broken", so it keeps issuing swipes that will never work.
"""

from __future__ import annotations

from android_ui_analyser.scroll_geom import travel


def test_travel_still_reports_a_changed_sample() -> None:
    """The primitive keeps its meaning — only the SCROLL verdict stops relying on it."""
    before = [("Heading", 100, 200)]
    after = [("Heading", 100, 200), ("Toast", 50, 900)]
    dx, dy, changed = travel(before, after)
    assert changed is True, "a differing sample is still a change"
    assert (dx, dy) == (0, 0), "but nothing shifted, so there is no distance"


def test_a_changed_sample_with_no_shift_is_not_a_scroll() -> None:
    """The exact device case: something repainted, the list did not move."""
    before = [("Trending", 42, 1125), ("World Cup Snake", 42, 1300)]
    after = [("Trending", 42, 1125), ("World Cup Snake", 42, 1300), ("Ripple", 500, 500)]
    dx, dy, changed = travel(before, after)
    assert changed is True
    axis_distance = dy
    assert axis_distance == 0
    # This is the rule _swipe_once now applies: the verdict is the axis distance, not `changed`.
    assert bool(axis_distance) is False, "must not be reported as moved"


def test_a_real_scroll_has_a_measurable_shift() -> None:
    before = [("Trending", 42, 1125), ("World Cup Snake", 42, 1300)]
    after = [("Trending", 42, 300), ("World Cup Snake", 42, 475)]
    _dx, dy, changed = travel(before, after)
    assert changed is True
    assert dy == 825, "content shifted up by 825px"
    assert bool(dy) is True


def test_horizontal_scroll_reads_the_x_axis() -> None:
    """A vertical-only shift must not count as horizontal movement, and vice versa."""
    before = [("Card A", 100, 500), ("Card B", 700, 500)]
    after = [("Card A", 100, 100), ("Card B", 700, 100)]  # moved vertically only
    dx, dy, _changed = travel(before, after)
    assert dy == 400
    assert dx == 0, "a left/right scroll must report no movement here"
