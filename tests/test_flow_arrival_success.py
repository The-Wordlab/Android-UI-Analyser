"""Positive arrival-proof regressions for recorded flow replay."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.flows import Flow, FlowStore
from android_ui_analyser.memory import RouteStep
from conftest import make_config
from test_memory import HOME, P, _elements, _engine, _store
from test_navigation import ScriptedDevice


def test_mapped_arrival_succeeds_with_fresh_same_context_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(
            name="verified_home",
            app=P,
            arrival_screen="home",
            arrival_status="mapped",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    engine = _engine(
        tmp_path,
        ScriptedDevice([HOME], package=P, serial="emu-arrival-success"),
    )

    result = engine.flow_run("verified_home")

    assert result["ok"] is True
    assert result["arrival_verified"] is True
    assert result["arrival_status"] == "verified"
    assert result["arrival_screen"] == {
        "expected": "home",
        "recognized": "home",
        "verified": True,
    }


def test_predicate_and_mapped_arrival_succeed_on_the_same_terminal_frame(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(
            name="verified_home_twice",
            app=P,
            arrival="text:Home,!text:Loading",
            arrival_screen="home",
            arrival_status="mapped",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    engine = _engine(
        tmp_path,
        ScriptedDevice(
            [HOME],
            package=P,
            serial="emu-arrival-combined-success",
            # The predicate is polled through `device.find_text`, and the fake's text index is
            # empty unless a test supplies one — so "Home" was absent from the selector while
            # plainly present in the hierarchy this same fake serves. The await then polled its
            # full 30s budget and only satisfied on the deadline's rich re-check: 30s of wall
            # clock for an assertion about the *first* frame. A double that disagrees with the
            # screen it is serving hides which path actually passed.
            text_index={"Home": (40, 120, 1040, 210)},
        ),
    )

    result = engine.flow_run("verified_home_twice")

    assert result["ok"] is True
    assert result["arrival_verified"] is True
    assert result["arrival"]["await_outcome"] == "satisfied"
    assert result["arrival_screen"]["verified"] is True
