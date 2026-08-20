"""Goal phases must represent alternatives and terminated work honestly."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import goal_phases
from conftest import FakeDevice, make_config


def _observation(serial: str, *, screen: str, title: str, controls: list[str]) -> AnalyzeResult:
    elements = [
        Element(
            id=0,
            type="android.widget.TextView",
            text=title,
            bounds=(0, 0, 800, 120),
            center=(400, 60),
            clickable=False,
            source=Source.hierarchy,
        )
    ]
    elements.extend(
        Element(
            id=index,
            type="android.widget.TextView",
            text=label,
            bounds=(0, index * 130, 800, index * 130 + 120),
            center=(400, index * 130 + 60),
            clickable=True,
            source=Source.hierarchy,
        )
        for index, label in enumerate(controls, start=1)
    )
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="com.example.settings",
            source="hierarchy",
        ),
        elements=elements,
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen=screen,
            device_serial=serial,
        ),
    )


def _engine(tmp_path: Path, serial: str) -> Engine:
    return Engine(
        make_config(
            cache={"dir": str(tmp_path / "cache")},
            memory={"enabled": False, "dir": str(tmp_path / "memory")},
        ),
        device=FakeDevice(serial=serial),
    )


def test_if_otherwise_is_one_alternative_checkpoint() -> None:
    phases = goal_phases(
        "Open catalog; then if mapped proof exists then replay it; otherwise record the "
        "explicit unverified result; finally restore network"
    )

    assert [phase.kind for phase in phases] == ["verify", "verify", "cleanup"]
    assert phases[1].objective == (
        "if mapped proof exists then replay it; otherwise record the explicit unverified result"
    )
    assert sum("otherwise" in phase.objective for phase in phases) == 1


def test_terminated_session_keeps_unfinished_phases_without_an_active_next_call(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "phase-terminated")
    observed = _observation(
        engine.device.serial,
        screen="catalog",
        title="Catalog",
        controls=[],
    )
    started = engine.session_start(
        "Inspect catalog. Then verify details",
        observation=observed,
    )

    finished = engine.session_finish(started["session_id"])

    assert finished["ok"] is True
    assert finished["terminated"] is True
    assert finished["finished"] is False
    assert finished["goal_progress"]["done"] is False
    assert finished["goal_progress"]["terminated"] is True
    assert finished["goal_progress"]["status"] == "terminated_incomplete"
    assert finished["goal_progress"]["current"]["status"] == "active"
    assert finished["goal_progress"]["next_call"] is None
    assert finished["goal_progress"]["checkpoint"] is None


def test_arrived_title_prefers_requested_flow_preview_over_weak_child_control(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "phase-arrival")
    main = _observation(
        engine.device.serial,
        screen="settings_home",
        title="Settings",
        controls=["Network & internet Mobile, Wi-Fi, hotspot"],
    )
    started = engine.session_start(
        "Navigate once to Network & internet and preview eval_arrival --last 1",
        observation=main,
    )
    destination = _observation(
        engine.device.serial,
        screen="network_and_internet",
        title="Network & internet",
        controls=["Internet AndroidWifi"],
    )

    progress = engine.session_progress(started["session_id"], observation=destination)[
        "goal_progress"
    ]

    assert progress["next_call"] == {
        "kind": "flow_save_preview",
        "cli": "aua flow save eval_arrival --last 1",
        "mcp": {
            "tool": "flow_save",
            "arguments": {"name": "eval_arrival", "last": 1},
        },
        "reason": (
            "The current mapped screen visibly matches 'Network & internet'; continue with "
            "the requested non-writing flow preview instead of navigating into a weaker "
            "one-word match."
        ),
        "executes": False,
        "arrival": {
            "status": "observed",
            "known_screen": "network_and_internet",
            "visible_title": "Network & internet",
        },
    }
    assert "tap" not in progress["next_call"]["cli"]


def test_origin_title_in_compound_phase_does_not_claim_destination_arrival(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "phase-origin-title")
    shelf = _observation(
        engine.device.serial,
        screen="tool_shelf",
        title="Tool Shelf",
        controls=["Vocabulary"],
    )

    started = engine.session_start(
        "From Tool Shelf, open Vocabulary's threaded recent",
        observation=shelf,
    )

    call = started["goal_progress"]["next_call"]
    assert call["kind"] == "manual_action"
    assert call["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"text": "Vocabulary"},
    }


def test_container_title_in_compound_phase_does_not_claim_child_arrival(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "phase-container-title")
    catalog = _observation(
        engine.device.serial,
        screen="example_catalog",
        title="Example Catalog",
        controls=["Saved Items"],
    )

    started = engine.session_start(
        "In Example Catalog, open Saved Items and verify the details",
        observation=catalog,
    )

    assert started["goal_progress"]["next_call"]["kind"] != "arrived"


def test_single_network_observation_has_one_consistent_top_level_call(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "phase-network-status")
    observed = _observation(
        engine.device.serial,
        screen="catalog",
        title="Catalog",
        controls=[],
    )

    started = engine.session_start(
        "Record the verified current network transport",
        observation=observed,
    )

    assert started["recommended_call"] == started["goal_progress"]["next_call"]
    assert started["recommended_call"] == {
        "kind": "network_status",
        "cli": "aua network status --verify",
        "mcp": {"tool": "network_status", "arguments": {"verify": True}},
        "reason": (
            "This phase records the verified current network transport before any "
            "reversible environment change."
        ),
        "executes": False,
    }
