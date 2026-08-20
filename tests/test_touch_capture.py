"""Reading the raw kernel touch stream, which is the only source that sees every press.

Accessibility reports a tap when the view announces one, and plenty do not — on one Compose
app, none did. The kernel has no such opinion. That makes this the source of record for "did a
finger land here", and it is why the parser gets its own tests rather than being covered
incidentally by the recorder's.

There is a second, sharper reason: this is the one path in AUA that cannot be exercised by
driving the device. ``adb shell input tap`` is injected above the kernel and produces nothing
here — only a real finger or a synthetic ``sendevent`` does. So a captured log, parsed offline,
is how the rules stay pinned.

The fixtures below are shaped exactly like real ``getevent -lt`` output, SYN_REPORT included.
That is not decoration: SYN is where the kernel commits a complete state, and leaving it out
of these fixtures once hid a bug that silently dropped presses.
"""

from __future__ import annotations

from android_ui_analyser.device_agent import parse_touch_log

AXES = {"/dev/input/event3": (32767, 32767)}
SCREEN = (1080, 2400)
DEV = "/dev/input/event3"

# ``#clock <uptime_seconds> <wall_ms>`` is written into the log before getevent starts, so the
# monotonic event stamps can be turned back into wall time by whichever process reads it.
CLOCK = "#clock 1000.000 1700000000000\n"


def _line(ts: float, code: str, value: int) -> str:
    kind = "EV_SYN" if code.startswith("SYN") else "EV_ABS"
    return f"[{ts:>15.6f}] {DEV}: {kind}       {code}   {value:08x}\n"


def _tap(ts: float, x: int | None, y: int | None, tracking: int = 1) -> str:
    """One press.

    ``x``/``y`` may be None, which is what the kernel does when an axis has not changed since
    the last report: it simply does not send it again.
    """

    out = _line(ts, "ABS_MT_TRACKING_ID", tracking)
    if x is not None:
        out += _line(ts + 0.001, "ABS_MT_POSITION_X", x)
    if y is not None:
        out += _line(ts + 0.002, "ABS_MT_POSITION_Y", y)
    out += _line(ts + 0.003, "SYN_REPORT", 0)
    out += _line(ts + 0.060, "ABS_MT_TRACKING_ID", 0xFFFFFFFF)
    out += _line(ts + 0.061, "SYN_REPORT", 0)
    return out


def test_a_press_becomes_a_point_in_screen_pixels() -> None:
    """Positions arrive on the device's own axis scale, not in pixels."""

    touches = parse_touch_log(CLOCK + _tap(1010.0, 16383, 9734), axis_maxima=AXES, screen=SCREEN)

    assert len(touches) == 1
    assert touches[0].x == 540, "half of 32767 should be half of the screen width"
    assert touches[0].y == 713
    assert touches[0].is_tap is True


def test_the_monotonic_stamp_is_turned_back_into_wall_time() -> None:
    """The recorder timestamps its own events with the wall clock, so these must agree.

    Without the conversion a press and the event it produced sit hours apart, every press looks
    unexplained, and the recorder redoes work it had already done.
    """

    touches = parse_touch_log(CLOCK + _tap(1010.0, 16383, 9734), axis_maxima=AXES, screen=SCREEN)

    # uptime 1000s == wall 1700000000000ms, so uptime 1010s is ten seconds later.
    assert touches[0].down_ms == 1700000010000


def test_each_press_reports_where_it_landed_not_where_the_last_one_did() -> None:
    """The position events arrive AFTER the tracking id that opens the gesture.

    Reading the position at tracking-id time instead gives whatever the previous finger left
    behind — wrong, and wrong in the most plausible-looking way, because every press still
    comes back with coordinates that were genuinely touched at some point.
    """

    log = CLOCK + _tap(1010.0, 9000, 5000, tracking=1) + _tap(1012.0, 20000, 25000, tracking=2)

    touches = parse_touch_log(log, axis_maxima=AXES, screen=SCREEN)

    assert [(t.x, t.y) for t in touches] == [(297, 366), (659, 1831)]


def test_two_presses_at_the_same_spot_are_both_recorded() -> None:
    """The kernel does not re-send an unchanged axis, and silence means "still there".

    Requiring both axes to report before a press counted looked reasonable and dropped the
    second of two taps in the same place — a bottom-bar tab pressed twice, a stepper, a
    keyboard key. Found by replaying a two-tap journey that came back with one step. The press
    is real and its position is known; the kernel simply had nothing new to say about it.
    """

    log = CLOCK + _tap(1010.0, 12258, 30473, tracking=1) + _tap(1012.0, None, None, tracking=2)

    touches = parse_touch_log(log, axis_maxima=AXES, screen=SCREEN)

    assert len(touches) == 2, "the second press at an unchanged position was dropped"
    assert (touches[0].x, touches[0].y) == (touches[1].x, touches[1].y)


def test_only_the_changed_axis_needs_resending() -> None:
    """The common real case: a finger moves along one axis, so only that one is re-reported."""

    log = CLOCK + _tap(1010.0, 9000, 5000, tracking=1) + _tap(1012.0, 20000, None, tracking=2)

    touches = parse_touch_log(log, axis_maxima=AXES, screen=SCREEN)

    assert [(t.x, t.y) for t in touches] == [(297, 366), (659, 366)]


def test_a_drag_is_reported_as_travelling() -> None:
    """A scroll must not replay as a press on whatever it happened to start over."""

    log = CLOCK + (
        _line(1010.000, "ABS_MT_TRACKING_ID", 1)
        + _line(1010.001, "ABS_MT_POSITION_X", 16383)
        + _line(1010.002, "ABS_MT_POSITION_Y", 20000)
        + _line(1010.003, "SYN_REPORT", 0)
        + _line(1010.100, "ABS_MT_POSITION_Y", 5000)
        + _line(1010.101, "SYN_REPORT", 0)
        + _line(1010.200, "ABS_MT_TRACKING_ID", 0xFFFFFFFF)
        + _line(1010.201, "SYN_REPORT", 0)
    )

    touches = parse_touch_log(log, axis_maxima=AXES, screen=SCREEN)

    assert len(touches) == 1
    assert touches[0].is_tap is False
    assert touches[0].travel_px > 40


def test_a_device_with_no_known_axes_is_skipped_rather_than_guessed() -> None:
    """Unnormalizing against the wrong maximum puts the press somewhere it never happened."""

    assert parse_touch_log(CLOCK + _tap(1010.0, 16383, 9734), axis_maxima={}, screen=SCREEN) == []


def test_a_log_with_no_clock_line_yields_nothing() -> None:
    """Every stamp would otherwise be monotonic seconds pretending to be wall time."""

    assert parse_touch_log(_tap(1010.0, 16383, 9734), axis_maxima=AXES, screen=SCREEN) == []


def test_a_gesture_that_never_reported_a_position_is_dropped() -> None:
    """Nothing has ever been touched, so there is no last position to fall back on."""

    log = CLOCK + (
        _line(1010.000, "ABS_MT_TRACKING_ID", 1)
        + _line(1010.200, "ABS_MT_TRACKING_ID", 0xFFFFFFFF)
        + _line(1010.201, "SYN_REPORT", 0)
    )

    assert parse_touch_log(log, axis_maxima=AXES, screen=SCREEN) == []


def test_noise_and_truncation_do_not_raise() -> None:
    """The log is whatever the device wrote before it was killed, mid-line included."""

    log = CLOCK + 'add device 1: /dev/input/event3\n  name: "touch"\n' + _tap(1010.0, 100, 200)
    log += "[    1011.0000"  # cut off exactly where pkill landed

    assert len(parse_touch_log(log, axis_maxima=AXES, screen=SCREEN)) == 1


def test_a_press_before_either_axis_reported_is_dropped_not_placed_at_the_origin() -> None:
    """The dangerous failure, and the reason a press is ever thrown away.

    An axis that has not changed is not re-sent, so the first press of a capture can arrive
    carrying no position at all — it inherits state the capture never saw. Reporting it at
    (0, 0) does not fail loudly; it resolves against whatever sits at the top of the screen and
    names the wrong control with complete confidence. A dropped press leaves an honest hole
    instead, which the recorder already knows how to report.

    (`getevent -pl` does expose a current value per axis, which would solve this properly, but
    emulators report 0 there regardless of where the last touch was.)
    """

    log = CLOCK + _tap(1010.0, 12258, None, tracking=1)

    assert parse_touch_log(log, axis_maxima=AXES, screen=SCREEN) == []


def test_the_axis_state_carries_across_presses_once_it_is_known() -> None:
    """Dropping is only for a position that was never established, not merely unchanged."""

    log = CLOCK + _tap(1010.0, 12258, 30473, tracking=1) + _tap(1012.0, None, None, tracking=2)

    touches = parse_touch_log(log, axis_maxima=AXES, screen=SCREEN)

    assert len(touches) == 2
    assert (touches[1].x, touches[1].y) == (touches[0].x, touches[0].y)


def test_the_device_listing_gives_up_its_axis_maxima() -> None:
    """Read from the capture, so it describes the device as it was when recording started."""

    from android_ui_analyser.device_agent import parse_device_axes

    listing = (
        "add device 1: /dev/input/event12\n"
        '  name:     "qwerty2"\n'
        "add device 2: /dev/input/event3\n"
        '  name:     "touch"\n'
        "    ABS_MT_POSITION_X     : value 0, min 0, max 32767, fuzz 0, flat 0, resolution 0\n"
        "    ABS_MT_POSITION_Y     : value 0, min 0, max 32767, fuzz 0, flat 0, resolution 0\n"
    )

    axes = parse_device_axes(listing)

    assert axes == {"/dev/input/event3": (32767, 32767)}, "a non-touch device slipped in"
