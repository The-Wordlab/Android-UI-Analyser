"""A recording is suspect when it is SHORT, not when the device's media clock ran ahead.

The old check compared the recorder process's wall time against the encoded media length with
``abs()`` and a two-second tolerance. Those are two different clocks on an emulator, and two
real runs of the same scenario failed it in opposite directions:

- 2026-09-13: recorder ran 173.32s, media 130.95s -> 42.37s SHORT.
- 2026-09-14: recorder ran 176.66s, media 214.06s -> 37.40s LONG.

The second one cannot have lost footage: there was more of it than there was time to lose. But
both were marked ``duration_check: failed``, cleanup failed closed, and every controller-harness
run came back BLOCKED regardless of what the product did.

So coverage is now judged one way. A shortfall still fails, because that is the case that can
actually cost evidence. Skew is reported under ``media_clock_skew`` instead of hidden, so a
caller can still see a device whose clocks disagree.
"""

from __future__ import annotations

from android_ui_analyser.platforms.android_recording import (
    _COVERAGE_TOLERANCE_S,
    _coverage_shortfall,
)

# The two runs that exposed this, as (process_seconds, media_seconds).
SHORT_RUN = (173.32, 130.9548)
LONG_RUN = (176.66, 214.0629)


def test_missing_footage_is_still_a_shortfall() -> None:
    shortfall = _coverage_shortfall(*SHORT_RUN)

    assert shortfall > _COVERAGE_TOLERANCE_S
    assert round(shortfall, 2) == 42.37


def test_a_media_clock_running_ahead_is_not_a_shortfall() -> None:
    """The whole bug in one assertion: 37 seconds of extra footage used to fail the run."""
    assert _coverage_shortfall(*LONG_RUN) == 0.0

    # And it would have failed under the old rule.
    process, media = LONG_RUN
    assert abs(process - media) > _COVERAGE_TOLERANCE_S


def test_a_matching_pair_is_clean_in_both_directions() -> None:
    assert _coverage_shortfall(100.0, 100.0) == 0.0
    assert _coverage_shortfall(100.0, 99.5) < _COVERAGE_TOLERANCE_S
    assert _coverage_shortfall(100.0, 100.5) == 0.0


def test_the_tolerance_still_bites_just_past_its_edge() -> None:
    assert _coverage_shortfall(100.0, 100.0 - _COVERAGE_TOLERANCE_S) == _COVERAGE_TOLERANCE_S
    assert _coverage_shortfall(100.0, 97.0) > _COVERAGE_TOLERANCE_S


def test_a_shortfall_is_never_negative() -> None:
    """Callers compare it against a tolerance; a negative would silently pass every check."""
    for process, media in ((0.0, 0.0), (0.0, 50.0), (10.0, 1000.0)):
        assert _coverage_shortfall(process, media) >= 0.0
