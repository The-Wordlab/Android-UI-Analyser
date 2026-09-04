"""A scroll reports movement only if the content actually moved along that axis.

`travel()`'s `moved` means "the probe sample differs at all" — a repaint, a ripple, an
element appearing. None of those are a scroll. Using it as the verdict made
`scroll --direction down` answer `moved steps=1` with no distance at the very top of a list,
where scrolling further is impossible. The synthetic samples below keep their list coordinates
fixed while an unrelated transient element appears.

A no-op that claims success is the worst outcome for an agent — it cannot tell "already at
the end" from "scrolling is broken", so it keeps issuing swipes that will never work.
"""

from __future__ import annotations

from android_ui_analyser.scroll_geom import scroll_movement, travel


def test_travel_still_reports_a_changed_sample() -> None:
    """The primitive keeps its meaning — only the SCROLL verdict stops relying on it."""
    before = [("Heading", 100, 200)]
    after = [("Heading", 100, 200), ("Toast", 50, 900)]
    dx, dy, changed = travel(before, after)
    assert changed is True, "a differing sample is still a change"
    assert (dx, dy) == (0, 0), "but nothing shifted, so there is no distance"


def test_a_changed_sample_with_no_shift_is_not_a_scroll() -> None:
    """Something repainted, but the list did not move."""
    before = [("Featured", 42, 1125), ("Blue notebook", 42, 1300)]
    after = [("Featured", 42, 1125), ("Blue notebook", 42, 1300), ("Toast", 500, 500)]
    dx, dy, changed = travel(before, after)
    assert changed is True
    axis_distance = dy
    assert axis_distance == 0
    # This is the rule _swipe_once now applies: the verdict is the axis distance, not `changed`.
    assert bool(axis_distance) is False, "must not be reported as moved"


def test_virtualized_grid_turnover_is_scroll_evidence_in_a_real_container() -> None:
    """Sticky filters can stay fixed while the visible card window changes completely."""
    before = [
        ("All", 20, 180),
        ("Fun", 160, 180),
        ("First card", 20, 420),
        ("Second card", 540, 420),
        ("Third card", 20, 980),
        ("Fourth card", 540, 980),
    ]
    after = [
        ("All", 20, 180),
        ("Fun", 160, 180),
        ("Later card one", 20, 390),
        ("Later card two", 540, 390),
        ("Later card three", 20, 950),
        ("Later card four", 540, 950),
    ]

    distance, moved, evidence = scroll_movement(
        before,
        after,
        "up",
        allow_content_turnover=True,
    )

    assert distance == 0, "the sticky labels still make the shared-label median zero"
    assert moved is True
    assert evidence == "content-turnover"


def test_content_turnover_is_not_trusted_without_a_declared_scroll_container() -> None:
    before = [("Old card one", 20, 420), ("Old card two", 540, 420)]
    after = [("New card one", 20, 420), ("New card two", 540, 420)]

    assert scroll_movement(before, after, "up", allow_content_turnover=False) == (
        0,
        False,
        None,
    )


def test_one_transient_label_is_not_content_turnover() -> None:
    before = [("Featured", 42, 1125), ("Blue notebook", 42, 1300)]
    after = [*before, ("Toast", 500, 500)]

    assert scroll_movement(before, after, "up", allow_content_turnover=True) == (
        0,
        False,
        None,
    )


def test_a_real_scroll_has_a_measurable_shift() -> None:
    before = [("Featured", 42, 1125), ("Blue notebook", 42, 1300)]
    after = [("Featured", 42, 300), ("Blue notebook", 42, 475)]
    _dx, dy, changed = travel(before, after)
    assert changed is True
    assert dy == 825, "content shifted up by 825px"
    assert bool(dy) is True


def test_turnover_with_identical_fallback_labels_is_still_scroll_evidence() -> None:
    """Unlabelled rows (icons, thumbnails) fall back to a shared class-name label in
    `region_probe`, so every item in the sample can carry the identical label. Diffing by
    label counts alone sees the same tally before and after and calls this `already-at-end`
    even though every item's position changed — exactly the false negative reported for
    horizontal attachment strips and image grids.
    """
    before = (
        ("android.widget.ImageView", 20, 400),
        ("android.widget.ImageView", 220, 400),
        ("android.widget.ImageView", 420, 400),
    )
    after = (
        ("android.widget.ImageView", -130, 400),
        ("android.widget.ImageView", 70, 400),
        ("android.widget.ImageView", 270, 400),
    )

    distance, moved, evidence = scroll_movement(
        before,
        after,
        "left",
        allow_content_turnover=True,
    )

    assert distance == 150
    assert moved is True
    assert evidence == "axis-shift"


def test_identical_fallback_labels_with_layout_jitter_are_not_scroll_evidence() -> None:
    before = (
        ("android.widget.ImageView", 20, 400),
        ("android.widget.ImageView", 220, 400),
        ("android.widget.ImageView", 420, 400),
    )
    after = (
        ("android.widget.ImageView", 21, 399),
        ("android.widget.ImageView", 219, 401),
        ("android.widget.ImageView", 422, 400),
    )

    assert scroll_movement(before, after, "left", allow_content_turnover=True) == (
        0,
        False,
        None,
    )


def test_identical_fallback_labels_moving_only_cross_axis_are_not_scroll_evidence() -> None:
    before = (
        ("android.widget.ImageView", 20, 400),
        ("android.widget.ImageView", 220, 400),
        ("android.widget.ImageView", 420, 400),
    )
    after = (
        ("android.widget.ImageView", 20, 350),
        ("android.widget.ImageView", 220, 350),
        ("android.widget.ImageView", 420, 350),
    )

    assert scroll_movement(before, after, "left", allow_content_turnover=True) == (
        0,
        False,
        None,
    )


def test_horizontal_scroll_reads_the_x_axis() -> None:
    """A vertical-only shift must not count as horizontal movement, and vice versa."""
    before = [("Card A", 100, 500), ("Card B", 700, 500)]
    after = [("Card A", 100, 100), ("Card B", 700, 100)]  # moved vertically only
    dx, dy, _changed = travel(before, after)
    assert dy == 400
    assert dx == 0, "a left/right scroll must report no movement here"
