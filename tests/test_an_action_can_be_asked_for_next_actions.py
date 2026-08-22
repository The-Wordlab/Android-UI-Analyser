"""The pre-filtered actionable list is still available — on request, and now honest.

It was on every action response. The argument for it was a *reasoning* cost, not a call: the
post-action screen already came back inline, but an agent had to scan ~50 `observation.elements`
to find which nodes it could act on, and that scan was expensive enough that agents preferred
`--no-observe` plus a filtered `analyze` — two calls to avoid one read. Measured on a 5-scenario
run, 37 taps produced 73 `analyze` calls.

That scan is gone. The folded observation is trimmed to the app's own ~20 rows with `clickable`
on each, so `[e for e in observation.elements if e["clickable"]]` is the same answer for free —
while the list cost 1384 bytes / 346 tokens on one real journalled response, 25% of the whole
response and more than the entire 1301-byte `elements` list it was a filtered subset of. So it
moved behind `output.next_actions`, and `tests/test_next_actions_does_not_ride_on_every_action.py`
owns that default.

What survives here is the contract for a caller that asks for it, plus the two invariants that
never depended on the default: no observation means no guidance, and every entry is named the
way the observation names it.
"""

from __future__ import annotations

from pathlib import Path

from conftest import FakeDevice
from test_memory import APPS, P, _engine

_OPT_IN = {"next_actions": True}


def test_an_action_hands_back_actionable_ids_when_asked(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next")
    eng = _engine(tmp_path, dev, output=_OPT_IN)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Reports"))

    r = eng.tap(target, observe=True)

    assert r.next_actions, "the opt-in must say what can be done next"
    assert all("id" in row for row in r.next_actions), "every entry must be directly actionable"
    assert any(row.get("label") for row in r.next_actions), "and carry a human-recognisable label"


def test_opting_out_of_the_observation_offers_no_next_actions(tmp_path: Path) -> None:
    """No observation means no knowledge of the new screen — so it must not invent guidance."""
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next2")
    eng = _engine(tmp_path, dev, output=_OPT_IN)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Reports"))

    r = eng.tap(target, observe=False)
    assert r.next_actions is None


def test_the_list_is_complete_rather_than_quietly_truncated(tmp_path: Path) -> None:
    """It used to cap at 12.

    Measured on one real screen: 15 elements were `clickable`, so three real controls
    (two content cards and an unlabelled view) were absent from a list an agent reads as
    "what I can do here". A shorter list that looks authoritative is worse than a long one,
    and a caller that wants a bound now passes `limit` and knows it asked for one.
    """
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next3")
    eng = _engine(tmp_path, dev, output=_OPT_IN)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Reports"))

    r = eng.tap(target, observe=True)
    observation = r.observation
    assert observation is not None
    actionable = [
        e
        for e in observation.elements
        if e.clickable or e.checkable or e.long_clickable or e.scrollable
    ]
    assert len(r.next_actions or []) == len(actionable), (
        "the list must name every control the observation says can be acted on"
    )
    assert len(eng._next_actions(observation, limit=2) or []) == 2, "an asked-for bound applies"


def test_state_flags_ride_only_on_the_rows_that_mean_something_by_them(tmp_path: Path) -> None:
    """`selected: false` on 12 of 12 rows is the default-flag leak this pins closed.

    The rows go through `schema.drop_default_flags`, the same rule `elements` uses, so a flag
    at its default is dropped and a *checkable* node's `checked: false` — which is the reading
    of an off switch, not a default — is kept.
    """
    from android_ui_analyser.identity import stable_key

    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next-selected")
    eng = _engine(tmp_path, dev, output=_OPT_IN)
    observation = eng.analyze(source="hierarchy")
    controls = [element for element in observation.elements if element.clickable][:3]
    assert len(controls) == 3
    controls[0].selected = True
    controls[1].selected = False
    controls[2].checkable = True
    controls[2].checked = False

    # `next_actions` names elements the way the payload does — by stable id.
    by_id = {row["id"]: row for row in eng._next_actions(observation) or []}

    assert by_id[stable_key(controls[0])]["selected"] is True
    assert "selected" not in by_id[stable_key(controls[1])], "a false default says nothing"
    assert by_id[stable_key(controls[2])]["checked"] is False, "an off switch IS the reading"
    assert by_id[stable_key(controls[2])]["checkable"] is True


def test_the_learned_cost_is_not_duplicated_onto_the_row(tmp_path: Path) -> None:
    """It used to be here, and it was the list's whole justification.

    Zero rows ever carried it on a fresh screen, because cost is learned per (screen, control).
    It now rides on `observation.elements[].cost`, which is one list rather than two — see
    `tests/test_learned_action_cost.py`.
    """
    from test_memory import _elements

    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next4")
    eng = _engine(tmp_path, dev, output=_OPT_IN)
    # The screen has to be known before a per-screen timing has anywhere to live.
    eng._memory.record_screen(package=P, elements=_elements(APPS), name_hint="apps")  # type: ignore[union-attr]
    first = eng.analyze(source="hierarchy")
    screen = first.meta.known_screen or "apps"

    el = next(e for e in first.elements if (e.text or "").startswith("Reports"))
    control = el.stable_key or (el.resource_id or "").rsplit("/", 1)[-1]
    eng._memory.record_action_timing(  # type: ignore[union-attr]
        P, screen=screen, control=control, ms=4800.0, outcome="changed"
    )

    r = eng.tap(el.id, observe=True)

    assert all("avg_ms" not in row for row in r.next_actions or []), (
        "the cost belongs to the element, and two copies of it is what this change removed"
    )
    observation = r.observation
    assert observation is not None
    assert any(e.cost for e in observation.elements), "and it must still reach the caller"
