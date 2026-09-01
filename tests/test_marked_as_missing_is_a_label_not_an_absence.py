""" "Marked as missing" describes an ingredient, not an empty screen.

Measured 2026-09-01 (`mini-apps-cook-assistant`, session
`0b8753df7e774f40963aa8ffb8f9ea0f`). A session goal took its third phase verbatim from a
scenario title:

    Cook Assistant excludes ingredients the user marked as missing

`_ABSENCE_SUFFIX` matched the sentence's trailing word and swallowed everything before it,
so the phase became `subject="Cook Assistant excludes ingredients the user marked as",
expected="absent"` — it wanted its own subject terms to be *off* the screen. Phases 1 and 2
were acknowledged first try; phase 3 rejected two attempts, one describing the feature
working and one deliberately phrased as an absence claim, and stayed `active`. There is no
observable fact about the feature succeeding that can close it, because succeeding means
those words *are* on screen. `session finish` then refused with `session_incomplete` and the
only exit was `--allow-incomplete`, recording `verdict: "incomplete"` on work that was
finished and independently validated.

The trigger is not the verb. `hides`, `omits` and `excludes` produce no requirement at all —
measured. It is the `as` in front of the state word: `marked as missing`, `flagged as
missing`, `shown as unavailable` all name a label something *carries*, which is the opposite
of that thing being gone. Nothing here touches a real absence claim: "the banner is absent",
"the spinner is not visible" and "no error banner is shown" must keep parsing exactly as they
did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser import session  # noqa: E402


def _expectations(objective: str) -> list[tuple[str, str]]:
    return [(r.expected, r.subject) for r in session._phase_requirements(objective)]


@pytest.mark.parametrize(
    "objective",
    [
        "Cook Assistant excludes ingredients the user marked as missing",
        "The row is flagged as missing in the pantry list",
        "Items the user labelled as missing are struck through",
        "An ingredient shown as missing keeps its badge after a restart",
    ],
)
def test_a_label_spelled_as_a_state_is_not_an_absence_requirement(objective: str) -> None:
    """The measured shape: a phase that can never be acknowledged by real evidence."""
    absent = [subject for expected, subject in _expectations(objective) if expected == "absent"]
    assert absent == [], f"{objective!r} produced an unsatisfiable absence: {absent}"


@pytest.mark.parametrize(
    ("objective", "subject"),
    [
        ("The banner is absent after dismissal", "The banner"),
        ("Confirm the spinner is not visible once the list loads", "Confirm the spinner"),
        ("The upsell is missing from the grid", "The upsell"),
        ("without a network error the list loads", "a network error the list loads"),
    ],
)
def test_a_real_absence_claim_is_untouched(objective: str, subject: str) -> None:
    """Nothing here loosens the negative assertions the parser exists for."""
    absent = [s for expected, s in _expectations(objective) if expected == "absent"]
    assert subject in absent, f"{objective!r} lost its absence requirement: {absent}"


def test_the_measured_goal_yields_a_phase_that_evidence_can_close() -> None:
    """The point of the fix: the phase must be acknowledgeable by a fact about the feature.

    Not "it produces no requirement" — that would also be true if the parser had simply given
    up. What matters is that whatever it does produce can be satisfied by describing the
    product working, which is the only evidence a runner has.
    """
    objective = "Cook Assistant excludes ingredients the user marked as missing"
    for expected, _subject in _expectations(objective):
        assert expected != "absent", expected
