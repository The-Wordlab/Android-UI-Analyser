"""Goal sessions correlate traces, coach waste, and clean up only their own state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser import journal, network, network_profiles
from android_ui_analyser.cli import _apply_phases_done, app
from android_ui_analyser.coaching import decorate_result
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import (
    complete_environment_phase,
    goal_phases,
    mark_phase_complete,
    phase_progress,
)
from conftest import FakeDevice, make_config

runner = CliRunner()


def _observation(serial: str) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package="com.example.catalog", source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=10,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="home",
            device_serial=serial,
        ),
    )


def _apps_observation(serial: str) -> AnalyzeResult:
    observed = _observation(serial)
    observed.elements = [
        Element(
            id=36,
            type="android.widget.TextView",
            text="Grammar 1 hr. ago",
            bounds=(0, 300, 900, 420),
            center=(450, 360),
            clickable=True,
            source=Source.hierarchy,
        ),
        Element(
            id=37,
            type="android.widget.TextView",
            text="Mathematics 12 hr. ago",
            bounds=(0, 430, 900, 550),
            center=(450, 490),
            clickable=True,
            source=Source.hierarchy,
        ),
    ]
    return observed


def test_goal_phases_preserve_order_without_splitting_ordinary_and() -> None:
    phases = goal_phases(
        "Verify the visible Grammar conversation opens offline with cached content and no "
        "loading. Compare the Mathematics item; finally restore connectivity before finishing."
    )

    assert [(phase.kind, phase.status) for phase in phases] == [
        ("environment", "active"),
        ("verify", "pending"),
        ("verify", "pending"),
        ("cleanup", "pending"),
    ]
    assert "cached content and no loading" in phases[1].objective
    assert "Mathematics" in phases[2].objective


def test_goal_phases_keep_online_setup_before_later_offline_transition() -> None:
    phases = goal_phases(
        "Establish the Grammar thread online; then switch offline; then verify Grammar; "
        "finally restore network"
    )

    assert [phase.kind for phase in phases] == ["verify", "environment", "verify", "cleanup"]
    assert phases[0].objective == "Establish the Grammar thread online"
    assert phases[0].status == "active"
    assert phases[1].status == "pending"


def test_verified_environment_phase_advances_and_evidence_checkpoints_stay_ordered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-phases")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _apps_observation(engine.device.serial))
    started = engine.session_start(
        "Verify Grammar opens offline. Compare Mathematics; restore connectivity."
    )
    state = engine._session_state(started["session_id"])

    state = complete_environment_phase(
        engine.config.cache.dir,
        state,
        command="network_offline",
        result={"ok": True, "verified": True, "state": {"offline": True}},
    )
    progress = phase_progress(state)
    assert progress["current"]["objective"] == "Verify Grammar opens offline"
    assert progress["next_call"]["cli"] == "aua tap-and-analyze 36"

    with pytest.raises(ValueError, match="complete 'phase_2' first"):
        mark_phase_complete(
            engine.config.cache.dir,
            state,
            phase_id="phase_3",
            evidence="Mathematics visible",
        )
    state = mark_phase_complete(
        engine.config.cache.dir,
        state,
        phase_id="phase_2",
        evidence="cached Grammar thread visible and Loading absent",
    )
    assert "Mathematics" in phase_progress(state)["current"]["objective"]


def test_cli_phase_checkpoint_advances_without_a_device_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-phase-cli")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _apps_observation(engine.device.serial))
    started = engine.session_start("Verify Grammar. Compare Mathematics")
    before_calls = list(engine.device.calls)

    _apply_phases_done(engine, ("phase_1=Grammar thread content visible",))

    progress = engine.session_progress(started["session_id"])["goal_progress"]
    assert progress["current"]["id"] == "phase_2"
    assert engine.device.calls == before_calls


def test_stale_deeplink_is_not_recommended_again_for_active_phase(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-stale-deeplink")
    observed = _apps_observation(engine.device.serial)
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)
    engine.session_start("Establish Grammar online; then switch offline")

    decorated = decorate_result(
        engine,
        "open_link",
        {
            "ok": True,
            "stale_risk": "the delivered intent did not move",
            "observation": observed.model_dump(mode="json"),
        },
        current_recorded=False,
    )

    call = decorated["goal_progress"]["next_call"]
    assert call["kind"] == "manual_action"
    assert "open-and-analyze" not in call["cli"]


def _engine(tmp_path: Path, serial: str = "goal-life") -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    return Engine(cfg, device=FakeDevice(serial=serial))


def _start(engine: Engine, monkeypatch: Any) -> dict[str, Any]:
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation(engine.device.serial))
    return engine.session_start("verify cached results while offline")


def test_cli_headed_accepts_an_already_attached_emulator(monkeypatch: Any) -> None:
    device = FakeDevice(serial="goal-headed-attached")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setattr(Engine, "analyze", lambda self, **_kwargs: _observation(device.serial))

    result = runner.invoke(
        app,
        [
            "--serial",
            device.serial,
            "session",
            "start",
            "--goal",
            "inspect the visible screen",
            "--headed",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"session_id"' in result.stdout


def test_journal_automatically_correlates_active_session_and_review_finds_waste(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path)
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "ok": True,
        "owner": None,
    }
    journal.record(
        **common,
        cmd="tap",
        result={"ok": True, "action": "tap", "observation": {"elements": []}},
    )
    journal.record(**common, cmd="analyze", result={"ok": True, "elements": []})
    journal.record(**common, cmd="has", args={"text": "Ready"}, result={"found": True})
    journal.record(**common, cmd="has", args={"text": "Loading"}, result={"found": False})

    rows = journal.read_since(engine.config.cache.dir, engine.device.serial)
    assert all(row["session_id"] == started["session_id"] for row in rows)
    review = engine.session_review(started["session_id"])
    assert review["patterns"]["redundant_analyze"]
    assert review["patterns"]["consecutive_has"]
    assert review["avoidable_calls"] == 2
    assert {item["id"] for item in review["advice"]} >= {
        "reuse_observation",
        "combine_assertions",
    }


def test_review_counts_folded_until_as_one_call_and_reports_its_timeout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-folded-until")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
        "extra": {"invocation_id": "one-agent-call"},
    }
    journal.record(
        **common,
        cmd="tap",
        args={"selector": {"rid": "next"}},
        ok=True,
        duration_ms=500,
        result={"ok": True, "action": "tap", "observation": {"elements": []}},
    )
    journal.record(
        **common,
        cmd="await_predicate",
        args={"predicate": "text:Destination", "adopt_action": True},
        ok=False,
        duration_ms=5_000,
        result={"ok": False, "action": "await", "detail": "timeout after 5000ms"},
    )

    review = engine.session_review(started["session_id"])

    assert review["calls"] == 1
    assert review["engine_events"] == 2
    assert review["duration_ms"] == 5_500
    assert "wait_after_observed_action" not in review["patterns"]
    assert review["patterns"]["predicate_timeout"][0]["predicate"] == "text:Destination"
    assert review["advice"][-1]["id"] == "exact_arrival_predicate"


def test_review_succeeds_while_reporting_run_failures_without_poisoning_itself(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-review-verdict")
    started = _start(engine, monkeypatch)
    journal.record(
        cache_dir=engine.config.cache.dir,
        serial=engine.device.serial,
        source="cli",
        owner=None,
        cmd="tap",
        ok=False,
        error={"code": "selector_not_found", "message": "moved"},
    )

    review = engine.session_review(started["session_id"])

    assert review["ok"] is True
    assert review["run_ok"] is False
    assert review["failures"] == 1

    journal.record(
        cache_dir=engine.config.cache.dir,
        serial=engine.device.serial,
        source="cli",
        owner=None,
        cmd="session_review",
        ok=False,
        result={"ok": False, "run_ok": False},
    )
    reviewed_again = engine.session_review(started["session_id"])
    assert reviewed_again["failures"] == 1, "a prior review must not become a new run failure"


def test_review_marks_historical_duplicate_invocation_unknown_without_guessing_visibility(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-recovered-invocation")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "daemon",
        "owner": None,
        "cmd": "tap",
        "extra": {"invocation_id": "same-agent-call"},
    }
    journal.record(
        **common,
        ok=False,
        error={"code": "selector_not_found", "message": "transient"},
    )
    journal.record(
        **common,
        ok=True,
        result={"ok": True, "action": "tap", "observation": {"elements": []}},
    )
    journal.record(
        cache_dir=engine.config.cache.dir,
        serial=engine.device.serial,
        source="cli",
        owner=None,
        cmd="analyze",
        ok=True,
        result={"ok": True, "elements": []},
    )

    review = engine.session_review(started["session_id"])

    assert review["ok"] is True
    assert review["run_ok"] is None
    assert review["failures"] == 0
    assert review["calls"] == 2
    assert "redundant_analyze" not in review["patterns"]
    assert review["patterns"]["ambiguous_invocation"] == [
        {
            "invocation_id": "same-agent-call",
            "cmd": "tap",
            "outcomes": [False, True],
        }
    ]
    assert review["advice"][-1]["id"] == "daemon_outcome_unknown"


def test_analyze_after_session_start_receives_immediate_reuse_advice(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-reuse-advice")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
    }
    journal.record(
        **common,
        cmd="session_start",
        ok=True,
        result={**started, "observation": {"elements": [], "meta": {"known_screen": "home"}}},
    )
    journal.record(
        **common,
        cmd="analyze",
        args={"source": "auto", "with_ocr": None},
        ok=True,
        result={"ok": True, "elements": []},
    )

    decorated = decorate_result(engine, "analyze", {"ok": True, "elements": []})

    assert decorated["advice"][0]["id"] == "reuse_observation"


def test_review_recommends_one_bounded_call_for_repeated_semantic_back(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-back-review")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
        "ok": True,
    }
    journal.record(
        **common,
        cmd="key",
        args={"name": "back"},
        result={"ok": True, "action": "key", "observation": {"elements": []}},
    )
    journal.record(
        **common,
        cmd="tap",
        args={"selector": {"rid": "buttonNavBack"}},
        result={"ok": True, "action": "tap", "observation": {"elements": []}},
    )

    review = engine.session_review(started["session_id"])

    assert review["patterns"]["repeated_back"][0]["calls"] == 2
    assert review["advice"][-1]["id"] == "bounded_back_navigation"


def test_review_flags_same_numeric_id_reused_across_changed_frames(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-frame-id-review")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
        "ok": True,
        "cmd": "tap",
        "args": {"element_id": 22, "selector": None},
    }
    journal.record(
        **common,
        result={
            "ok": True,
            "action": "tap",
            "observation": {"elements": [], "meta": {"known_screen": "nested-two"}},
        },
    )
    journal.record(
        **common,
        result={
            "ok": True,
            "action": "tap",
            "observation": {"elements": [], "meta": {"known_screen": "nested-one"}},
        },
    )

    review = engine.session_review(started["session_id"])

    assert review["patterns"]["reused_numeric_id"][0]["element_id"] == 22
    assert review["advice"][-1]["id"] == "do_not_reuse_frame_id"
    assert review["avoidable_calls"] == 0


def test_mcp_style_unrecorded_action_gets_same_numeric_id_reuse_coaching(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-frame-id-mcp")
    _start(engine, monkeypatch)
    journal.record(
        cache_dir=engine.config.cache.dir,
        serial=engine.device.serial,
        source="mcp",
        owner=None,
        ok=True,
        cmd="tap_and_analyze",
        args={"element_id": 22},
        result={
            "ok": True,
            "action": "tap",
            "observation": {"elements": [], "meta": {"known_screen": "nested-two"}},
        },
    )

    decorated = decorate_result(
        engine,
        "tap_and_analyze",
        {
            "ok": True,
            "action": "tap",
            "observation": {"elements": [], "meta": {"known_screen": "nested-one"}},
        },
        args={"element_id": 22},
        current_recorded=False,
    )

    assert decorated["advice"][0]["id"] == "do_not_reuse_frame_id"


def test_playback_resource_id_is_not_classified_as_back_navigation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-playback")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
        "ok": True,
        "cmd": "tap",
        "args": {"selector": {"rid": "mediaPlayback"}},
        "result": {"ok": True, "action": "tap", "observation": {"elements": []}},
    }
    journal.record(**common)
    journal.record(**common)

    review = engine.session_review(started["session_id"])

    assert "repeated_back" not in review["patterns"]


def test_finish_restores_network_state_created_by_session(tmp_path: Path, monkeypatch: Any) -> None:
    engine = _engine(tmp_path, "goal-owned")
    started = _start(engine, monkeypatch)
    offline = network.backup_path(engine.config.cache.dir, engine.device.serial)
    profile = network_profiles.profile_path(engine.config.cache.dir, engine.device.serial)
    offline.parent.mkdir(parents=True, exist_ok=True)
    profile.parent.mkdir(parents=True, exist_ok=True)
    offline.write_text("session-owned", encoding="utf-8")
    profile.write_text("session-owned", encoding="utf-8")
    restored: list[str] = []
    monkeypatch.setattr(
        engine,
        "network_restore",
        lambda: restored.append("offline") or {"ok": True, "action": "network-restore"},
    )
    monkeypatch.setattr(
        engine,
        "network_profile_restore",
        lambda: restored.append("profile") or {"ok": True, "action": "network-profile-restore"},
    )

    finished = engine.session_finish(started["session_id"])

    assert finished["ok"] is True
    assert restored == ["profile", "offline"]
    assert engine.session_review(started["session_id"])["finished_ms"] is not None


def test_finish_preserves_restore_points_that_predated_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-preexisting")
    offline = network.backup_path(engine.config.cache.dir, engine.device.serial)
    profile = network_profiles.profile_path(engine.config.cache.dir, engine.device.serial)
    offline.parent.mkdir(parents=True, exist_ok=True)
    profile.parent.mkdir(parents=True, exist_ok=True)
    offline.write_text("preexisting", encoding="utf-8")
    profile.write_text("preexisting", encoding="utf-8")
    started = _start(engine, monkeypatch)
    monkeypatch.setattr(
        engine,
        "network_restore",
        lambda: (_ for _ in ()).throw(AssertionError("must preserve prior state")),
    )
    monkeypatch.setattr(
        engine,
        "network_profile_restore",
        lambda: (_ for _ in ()).throw(AssertionError("must preserve prior state")),
    )

    finished = engine.session_finish(started["session_id"])

    assert finished["ok"] is True
    assert finished["cleanup"] == []
    assert offline.is_file() and profile.is_file()


def test_daemon_exposes_the_same_goal_session_lifecycle() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeEngine:
        def session_start(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("start", kwargs))
            return {"ok": True, "session_id": "s1"}

        def session_review(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("review", kwargs))
            return {"ok": True, "session_id": "s1"}

        def session_finish(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("finish", kwargs))
            return {"ok": True, "session_id": "s1"}

        def reach(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("reach", kwargs))
            return {"ok": True, "strategy": "goto"}

    engine = FakeEngine()
    assert dispatch(engine, {"cmd": "session_start", "args": {"goal": "open saved"}})["ok"]
    assert dispatch(engine, {"cmd": "session_review", "args": {"session_id": "s1"}})["ok"]
    assert dispatch(engine, {"cmd": "session_finish", "args": {"session_id": "s1"}})["ok"]
    assert dispatch(engine, {"cmd": "reach", "args": {"goal": "open saved"}})["ok"]
    assert [name for name, _args in calls] == ["start", "review", "finish", "reach"]


def test_explicit_session_emulator_start_is_owned_and_finished(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser import emulator

    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    engine = Engine(cfg)
    monkeypatch.setattr(engine, "list_devices", lambda: [])
    starts: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    monkeypatch.setattr(
        emulator,
        "start",
        lambda avd, **kwargs: (
            starts.append({"avd": avd, **kwargs}) or {"ok": True, "serial": "emulator-5590"}
        ),
    )
    monkeypatch.setattr(
        emulator,
        "stop",
        lambda **kwargs: stops.append(kwargs) or {"ok": True, "stopped": ["emulator-5590"]},
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation("emulator-5590"))

    started = engine.session_start(
        "inspect the current theme",
        start_emulator=True,
        headed=True,
        avd="Small_Phone",
    )
    finished = engine.session_finish(started["session_id"])

    assert started["emulator_started"] is True
    assert starts[0]["headless"] is False
    assert stops[0]["serial"] == "emulator-5590"
    assert finished["ok"] is True
