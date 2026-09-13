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

from pathlib import Path

import pytest

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

# --- 2026-09-14, second correction -------------------------------------------------------
#
# Judging a shortfall as loss was still wrong, because it assumes a constant frame rate.
# `screenrecord` emits a frame when the screen CHANGES. Run A1 of the model comparison:
#
#   wall 168.17s, media 130.00s, 126 frames, last frame at 128.92s
#   largest inter-frame gaps: 38.9s @ t=88.9, 21.1s @ t=67.8, 20.0s @ t=28.9, 18.9s @ t=48.9
#
# Every one of those lands where the controller was waiting on the model with the app idle.
# Nothing was lost; nothing happened. The scenario's own invariant is that the app ends up
# idle at home, so the check was guaranteed to fail exactly the runs it was meant to protect.
#
# Only a recording that captured nothing can still fail on coverage.

A1_RUN = {"wall_s": 168.17, "media_s": 130.001, "frames": 126}


def _timeline(*, media_s, wall_s=A1_RUN["wall_s"], exit_code=0, monkeypatch=None):
    """Drive timeline() with one segment of a given media length."""
    from android_ui_analyser.platforms import android_recording as rec

    events = f"begin 0 1000.0\nend 0 {1000.0 + wall_s} {exit_code}\nfinish {1000.0 + wall_s} stopped\n"
    monkeypatch.setattr(rec, "media_duration", lambda path: media_s)
    return rec.timeline(events, [Path("segment-0.mp4")], stop_uptime_s=1000.0 + wall_s,
                        start_uptime_s=1000.0)


def test_a_static_screen_no_longer_fails_the_run(monkeypatch):
    report = _timeline(media_s=A1_RUN["media_s"], monkeypatch=monkeypatch)
    assert report["duration_check"] == "passed"


def test_the_shortfall_is_still_reported_rather_than_hidden(monkeypatch):
    report = _timeline(media_s=A1_RUN["media_s"], monkeypatch=monkeypatch)
    assert report["coverage_shortfall_s"] == pytest.approx(38.17, abs=0.05)
    reasons = {gap["reason"] for gap in report["gaps"]}
    assert "static_screen_no_frames" in reasons


def test_capturing_nothing_at_all_still_fails(monkeypatch):
    # The one coverage result a screen that did not change cannot explain.
    report = _timeline(media_s=0.0, monkeypatch=monkeypatch)
    assert report["duration_check"] == "failed"


def test_a_recorder_that_died_still_fails(monkeypatch):
    report = _timeline(media_s=A1_RUN["media_s"], exit_code=1, monkeypatch=monkeypatch)
    assert report["duration_check"] == "failed"
    assert any(g["reason"] == "unverified_segment_coverage" for g in report["gaps"])


def test_a_recording_longer_than_its_wall_time_is_still_only_skew(monkeypatch):
    report = _timeline(media_s=A1_RUN["wall_s"] + 40, monkeypatch=monkeypatch)
    assert report["duration_check"] == "passed"
    assert report["media_clock_skew"], "a device clock running ahead must stay visible"


def test_an_encoder_that_was_not_running_still_fails(monkeypatch):
    """The distinction that matters: not running is a gap, running-but-static is not.

    Run A1's encoder was alive for the whole window and simply had nothing to encode. An
    encoder that stopped 220 seconds before the requested stop was not recording at all, and
    no amount of screen stillness accounts for that.
    """
    from android_ui_analyser.platforms import android_recording as rec

    monkeypatch.setattr(rec, "media_duration", lambda path: 20.0)
    report = rec.timeline("begin 0 100\nend 0 280 0\nfinish 280 stopped\n",
                          [Path("segment-0.mp4")], stop_uptime_s=500)
    assert report["duration_check"] == "failed"
    assert report["encoder_idle_gaps"], "the dark stretch must be named, not just counted"
    assert report["encoder_idle_gaps"][0]["reason"] == "recording_ended_before_stop"
