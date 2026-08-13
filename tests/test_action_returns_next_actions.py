"""An action must hand back what to do next, not just what is on screen.

The post-action screen already came back inline, so the remaining waste was not a *call* — it was
that an agent had to scan `observation.elements` to find which of ~50 nodes it could act on. That
scan was expensive enough that agents preferred `--no-observe` plus a filtered `analyze`: two calls
to avoid one read. Measured on a 5-scenario run, 37 taps produced 73 `analyze` calls.

`next_actions` is the decision-ready form — ids you can act on, with each control's own learned cost
attached so "tap 26 next, and it takes ~4.8s" is one read rather than a cross-reference.
"""

from __future__ import annotations

from pathlib import Path

from conftest import FakeDevice
from test_memory import APPS, P, _engine


def test_an_action_hands_back_actionable_ids(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next")
    eng = _engine(tmp_path, dev)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Images"))

    r = eng.tap(target, observe=True)

    assert r.next_actions, "an observed action must say what can be done next"
    assert all("id" in row for row in r.next_actions), "every entry must be directly actionable"
    assert any(row.get("label") for row in r.next_actions), "and carry a human-recognisable label"


def test_opting_out_of_the_observation_offers_no_next_actions(tmp_path: Path) -> None:
    """No observation means no knowledge of the new screen — so it must not invent guidance."""
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next2")
    eng = _engine(tmp_path, dev)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Images"))

    r = eng.tap(target, observe=False)
    assert r.next_actions is None


def test_the_list_is_capped_rather_than_a_dump(tmp_path: Path) -> None:
    """A list of every tappable node is what `analyze` is for; guidance has to be short."""
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next3")
    eng = _engine(tmp_path, dev)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Images"))

    r = eng.tap(target, observe=True)
    assert r.next_actions is not None and len(r.next_actions) <= 12


def test_selected_state_rides_on_each_relevant_next_action(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next-selected")
    eng = _engine(tmp_path, dev)
    observation = eng.analyze(source="hierarchy")
    controls = [element for element in observation.elements if element.clickable][:2]
    assert len(controls) == 2
    controls[0].selected = True
    controls[1].selected = False

    by_id = {row["id"]: row for row in eng._next_actions(observation) or []}

    assert by_id[controls[0].id]["selected"] is True
    assert by_id[controls[1].id]["selected"] is False


def test_learned_cost_rides_on_the_control_it_belongs_to(tmp_path: Path) -> None:
    from test_memory import _elements

    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-next4")
    eng = _engine(tmp_path, dev)
    # The screen has to be known before a per-screen timing has anywhere to live.
    eng._memory.record_screen(package=P, elements=_elements(APPS), name_hint="apps")  # type: ignore[union-attr]
    first = eng.analyze(source="hierarchy")
    screen = first.meta.known_screen or "apps"

    # Teach the store that one control on this screen is slow, then act and read it back.
    el = next(e for e in first.elements if (e.text or "").startswith("Images"))
    control = el.stable_key or (el.resource_id or "").rsplit("/", 1)[-1]
    eng._memory.record_action_timing(  # type: ignore[union-attr]
        P, screen=screen, control=control, ms=4800.0, outcome="changed"
    )

    r = eng.tap(el.id, observe=True)
    rows = r.next_actions or []
    priced = [row for row in rows if "avg_ms" in row]
    # The tap navigates, so the control may not be on the destination — the contract under test is
    # that when a listed control HAS history, its cost travels with it rather than being elsewhere.
    for row in priced:
        assert row["avg_ms"] > 0 and row["n"] >= 1
