"""An unchanged screen must still report what AUA learned while it sat still.

`analyze` reuses the previous payload when the accessibility tree hashes the same. Reusing the
*tree* is sound — it really is identical — but `meta` also carries facts that come from app
memory rather than from the tree: the map's routes, saved flows, suggested deeplinks, research
tasks, the screen question, and learned control costs. Those keep changing while the screen does
not, so carrying the previous copies over reported yesterday's memory for as long as the caller
stood still — and standing still is exactly when it is deciding what to do next.

The reuse path already calls `_record_screen_safe` for `known_screen` and threw away the hints it
returns, so this costs no extra work.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.flows import Flow, FlowStore
from android_ui_analyser.memory import RouteStep
from conftest import FakeDevice, make_config
from test_memory import APPS, P, _elements, _engine


def _engine_on_a_still_screen(tmp_path: Path, serial: str):
    engine = _engine(tmp_path, FakeDevice(hierarchy_xml=APPS, package=P, serial=serial))
    assert engine._memory is not None
    engine._memory.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    first = engine.analyze(source="hierarchy")
    assert first.meta.flows == [], "nothing saved yet"
    return engine


def test_a_flow_saved_while_the_screen_sat_still_is_reported(tmp_path: Path) -> None:
    engine = _engine_on_a_still_screen(tmp_path, "emu-unchanged-flows")
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(name="saved_mid_session", app=P, steps=[RouteStep(kind="key", arg="back")])
    )

    again = engine.analyze(source="hierarchy")

    assert again.meta.unchanged is True, "an identical tree must take the reuse path"
    assert again.meta.flows == ["saved_mid_session"], (
        "a journey saved since the last analyze is replayable now, so it has to be offered now"
    )


def test_reuse_never_blanks_memory_fields_it_cannot_refresh(tmp_path: Path) -> None:
    """A non-recording observe snapshot has no hints to read, so it must keep what it had.

    Refreshing must not turn into erasing: an observation taken mid-transition deliberately
    skips recording, and answering "no routes, no flows" there would read as the map having
    forgotten them.
    """
    engine = _engine_on_a_still_screen(tmp_path, "emu-unchanged-keeps")
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(name="saved_mid_session", app=P, steps=[RouteStep(kind="key", arg="back")])
    )
    recorded = engine.analyze(source="hierarchy")
    assert recorded.meta.flows == ["saved_mid_session"]

    snapshot = engine.analyze(source="hierarchy", record=False)

    assert snapshot.meta.unchanged is True
    assert snapshot.meta.flows == ["saved_mid_session"], "reuse must not erase known memory"
