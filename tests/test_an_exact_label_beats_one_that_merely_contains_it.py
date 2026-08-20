"""A control whose label IS the destination must outrank one that merely contains it.

`_match_score` awards +40 whenever the goal phrase appears anywhere in a control's label. For a
goal of `Beacon`, both `Beacon` and `Beacon board` contain the phrase, match every goal term, and
therefore scored identically. The ranking tie then fell through to `-element.id`, so the winner was
whichever of the two happened to come first in the frame.

Measured 2026-08-18 on containment-shaped candidate sets — one distractor strictly containing the
target, four rotations each: the deterministic recommendation took the longer label in 6 of 24
rotations, exactly one per target, and only in the ordering that placed it first. The same relation
had already beaten every FunctionGemma version on a live screen.

Only the VISIBLE label may count toward over-specification. Folding the resource id in punished a
control for having a descriptive id and inverted the ranking in favour of unaddressable duplicates,
which `test_compiler_withholds_...` catches.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser.session import _match_score
from test_functiongemma_engine_policy import _element, _engine, _observation

# target, a label that strictly contains it, two unrelated controls
_TARGET = "Beacon"
_CONTAINS = "Beacon board"
_OTHERS = ("Atlas archive", "Delta desk")
_RIDS = {
    "Beacon": "com.example.catalog:id/openBeacon",
    "Beacon board": "com.example.catalog:id/openBeaconBoard",
    "Atlas archive": "com.example.catalog:id/openAtlas",
    "Delta desk": "com.example.catalog:id/openDelta",
}


def test_the_exact_label_outscores_the_one_that_contains_it() -> None:
    exact = _match_score(_TARGET, _TARGET, exactness=_TARGET)
    longer = _match_score(_TARGET, _CONTAINS, exactness=_CONTAINS)

    assert exact > longer, f"{exact} must beat {longer}"


def test_a_descriptive_resource_id_is_not_counted_as_over_specification() -> None:
    """The id carries no visible words; scoring it would punish addressability."""
    with_rid = _match_score(_TARGET, f"{_TARGET} openBeacon", exactness=_TARGET)
    bare = _match_score(_TARGET, _TARGET, exactness=_TARGET)

    assert with_rid == bare


def test_the_penalty_never_outweighs_a_matched_term() -> None:
    """It may only break a tie, never beat real evidence."""
    one_term_verbose = _match_score(
        "Beacon board", "Beacon board archive of everything", exactness="Beacon board archive"
    )
    wrong_control = _match_score("Beacon board", "Delta desk", exactness="Delta desk")

    assert one_term_verbose > wrong_control


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_the_deterministic_recommendation_ignores_frame_order(
    tmp_path: Path, rotation: int
) -> None:
    labels = [_TARGET, _CONTAINS, *_OTHERS]
    rotated = labels[rotation:] + labels[:rotation]
    engine, _factory = _engine(tmp_path, "off", None)
    observation = _observation(
        engine.device.serial,
        [_element(index, label, rid=_RIDS[label]) for index, label in enumerate(rotated, start=1)],
    )
    phase = SimpleNamespace(
        id="phase-containment",
        objective=f"Open {_TARGET}",
        kind="verify",
        constraints=[],
        recommended_call=None,
    )

    recommended = engine._phase_recommended_call(  # noqa: SLF001
        SimpleNamespace(session_id="containment", serial=engine.device.serial),
        phase,
        observation,
    )

    assert recommended is not None, f"rotation {rotation} produced no call"
    arguments = recommended["mcp"]["arguments"]
    assert arguments.get("rid") == "openBeacon", (
        f"rotation {rotation} chose {arguments!r} instead of the exact label"
    )
