"""The redundant-analyze lint must not fire when the previous response invited the re-read.

Journalled 2026-08-22: a tap returned a transitional frame as the post-action screen, the agent
tapped a control that no longer existed, and its recovery ``analyze`` was answered with
"redundant analyze right after app-launch — prefer using the previous observation". A lint that
scolds the caller for recovering from the tool's *own admitted uncertainty* teaches the caller
to ignore the lint.

Two admissions license a follow-up ``analyze`` (both survive the journal's slim record):

* ``stale_risk`` on the previous result — the engine itself said the observation may predate
  the action, be mid-transition, or be an unrendered/loading destination;
* an empty previous observation — whose note literally says "Re-read with `analyze`".
"""

from __future__ import annotations

from android_ui_analyser.cli import _previous_response_invited_a_reread


def test_a_previous_stale_risk_licenses_the_reread() -> None:
    prev = {
        "ok": True,
        "action": "tap",
        "stale_risk": "the action left the previous screen but the new one has rendered nothing",
        "observation": {"elements_count": 7, "meta": None},
    }

    assert _previous_response_invited_a_reread(prev)


def test_an_empty_previous_observation_licenses_the_reread() -> None:
    prev = {"ok": True, "action": "app-launch", "observation": {"elements_count": 0, "meta": None}}

    assert _previous_response_invited_a_reread(prev)


def test_a_confident_previous_observation_keeps_the_lint() -> None:
    prev = {"ok": True, "action": "tap", "observation": {"elements_count": 7, "meta": None}}

    assert not _previous_response_invited_a_reread(prev)


def test_a_full_result_shape_without_slimming_also_counts() -> None:
    """In-process paths hand the un-slimmed dict; elements as a list must read the same."""
    prev = {"ok": True, "action": "tap", "observation": {"elements": [], "meta": {}}}

    assert _previous_response_invited_a_reread(prev)
