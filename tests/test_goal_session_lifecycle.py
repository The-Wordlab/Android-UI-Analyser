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
from android_ui_analyser.schema import AnalyzeResult, DeviceInfo, Element, Meta, Screen, Source
from android_ui_analyser.session import (
    complete_environment_phase,
    create_session_state,
    goal_phases,
    load_session_state,
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


def _control_observation(serial: str, label: str) -> AnalyzeResult:
    observed = _observation(serial)
    observed.elements = [
        Element(
            id=41,
            type="android.widget.Button",
            text=label,
            bounds=(40, 300, 800, 420),
            center=(420, 360),
            clickable=True,
            source=Source.hierarchy,
        )
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
    monkeypatch.setattr(
        engine, "analyze", lambda **_kwargs: _apps_observation(engine.device.serial)
    )
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
    assert progress["next_call"] is None
    handoff = engine.session_progress(started["session_id"])["goal_progress"]["next_call"]
    assert handoff == {
        "kind": "refresh_observation",
        "cli": "aua analyze --source hierarchy",
        "mcp": {"tool": "analyze_screen", "arguments": {"source": "hierarchy"}},
        "reason": (
            "The active UI phase began after a non-UI transition. Read one fresh hierarchy "
            "frame; its goal_progress will contain the exact next action."
        ),
        "executes": False,
    }
    refreshed = engine.session_progress(
        started["session_id"], observation=_apps_observation(engine.device.serial)
    )["goal_progress"]
    assert refreshed["next_call"]["cli"] == ("aua tap-and-analyze --text 'Grammar 1 hr. ago'")

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


def test_engine_network_goal_runs_status_offline_then_finish_with_exact_calls(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "goal-network-sequence")
    started = engine.session_start(
        "Record the verified active network transport; make the Example Emulator verifiably "
        "offline; restore the original connectivity on finish.",
        observation=_observation(engine.device.serial),
    )

    assert started["recommended_call"]["kind"] == "network_status"
    assert started["recommended_call"]["mcp"] == {
        "tool": "network_status",
        "arguments": {"verify": True},
    }

    status = decorate_result(
        engine,
        "network_status",
        {
            "ok": True,
            "verified": True,
            "state": {
                "active_network": True,
                "active_transports": ["wifi"],
                "internet_validated": True,
                "offline": False,
            },
        },
    )
    assert status["goal_progress"]["next_call"] == {
        "kind": "network_offline",
        "cli": "aua network offline --verify",
        "mcp": {"tool": "network_offline", "arguments": {"verify": True}},
        "reason": "This phase requires verified reversible network isolation.",
        "executes": True,
    }

    offline = decorate_result(
        engine,
        "network_offline",
        {"ok": True, "verified": True, "state": {"offline": True}},
    )
    assert offline["goal_progress"]["next_call"]["kind"] == "session_finish"
    assert offline["goal_progress"]["next_call"]["cli"] == "aua session finish"
    assert offline["goal_progress"]["next_call"]["mcp"] == {
        "tool": "session_finish",
        "arguments": {"session_id": started["session_id"]},
    }

    finished = engine.session_finish(started["session_id"])

    assert finished["finished"] is True
    assert finished["terminated"] is True
    assert finished["goal_progress"]["completed"] == 3
    assert finished["goal_progress"]["total"] == 3
    assert finished["goal_progress"]["done"] is True
    assert finished["goal_progress"]["next_call"] is None


def test_cli_phase_checkpoint_advances_without_a_device_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-phase-cli")
    monkeypatch.setattr(
        engine, "analyze", lambda **_kwargs: _apps_observation(engine.device.serial)
    )
    started = engine.session_start("Verify Grammar. Compare Mathematics")
    before_calls = list(engine.device.calls)

    _apply_phases_done(engine, ("phase_1=Grammar thread content visible",))

    progress = engine.session_progress(started["session_id"])["goal_progress"]
    assert progress["current"]["id"] == "phase_2"
    assert engine.device.calls == before_calls


def test_fresh_cli_resolves_active_owner_before_phase_annotation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser import cli, leases

    serial = "configured-emulator"
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
        device={"serial": serial},
        daemon={"enabled": False},
    )
    monkeypatch.setenv("AUA_OWNER", "fresh-phase-agent")
    state = create_session_state(
        cfg.cache.dir,
        goal="Verify the first checkpoint; then verify the second checkpoint",
        serial=serial,
        owner=leases.resolve_owner(None),
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    connections: list[str | None] = []

    def unexpected_connect(_engine: Engine, device_serial: str | None = None) -> FakeDevice:
        connections.append(device_serial)
        raise AssertionError("phase annotation must not connect to a device")

    monkeypatch.setattr(Engine, "_connect_target", unexpected_connect)
    monkeypatch.setattr(cli.GlobalOpts, "load", lambda _self: cfg)

    result = runner.invoke(
        app,
        [
            "--serial",
            serial,
            "--phase-done",
            "phase_1=first checkpoint is visible",
            "flow",
            "delete",
            "already-absent",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.stdout)
    assert payload["goal_progress"]["current"]["id"] == "phase_2"
    assert "annotation_warnings" not in payload
    updated = load_session_state(cfg.cache.dir, session_id=state.session_id)
    assert updated is not None
    assert updated.phases[0].status == "completed"
    assert updated.phases[0].evidence == "first checkpoint is visible"
    assert connections == []


def test_bad_inline_phase_annotation_warns_but_does_not_cancel_the_action(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser import cli

    engine = _engine(tmp_path, "annotation-device")
    actions: list[str] = []
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)

    def route(_engine: Engine, method: str, **_kwargs: Any) -> dict[str, Any]:
        actions.append(method)
        return {"ok": True, "action": "key", "detail": "back"}

    monkeypatch.setattr(cli, "_route", route)
    result = runner.invoke(
        app,
        ["--phase-done", "not-a-checkpoint", "key-and-analyze", "back", "--no-observe"],
    )

    assert result.exit_code == 0, result.output
    assert actions == ["key"]
    payload = __import__("json").loads(result.stdout)
    assert payload["annotation_warnings"][0]["annotation"] == "phase_done"


def test_cli_finish_exits_nonzero_for_an_incomplete_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser import cli

    engine = _engine(tmp_path, "incomplete-finish-cli")
    routed: list[dict[str, Any]] = []
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda _self: engine)

    def finish_route(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        routed.append(kwargs)
        return {
            "ok": False,
            "code": "session_incomplete",
            "finished": False,
            "terminated": False,
            "verdict": "incomplete",
        }

    monkeypatch.setattr(
        cli,
        "_route",
        finish_route,
    )

    result = runner.invoke(app, ["session", "finish"])

    assert result.exit_code == 1
    assert __import__("json").loads(result.stdout)["verdict"] == "incomplete"
    assert routed[-1]["summary"] is True

    full = runner.invoke(app, ["session", "finish", "--full"])

    assert full.exit_code == 1
    assert routed[-1]["summary"] is False


def test_generic_end_to_end_goal_accepts_two_concrete_observable_facts(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, "generic-flow-proof")
    started = engine.session_start(
        "Explore the Android app and complete one meaningful non-destructive end-to-end flow",
        observation=_observation(engine.device.serial),
    )

    completed = engine.session_mark_phase(
        "phase_1",
        "Conversation opened; assistant reply appeared; thread persisted after returning",
        session_id=started["session_id"],
    )

    assert completed["goal_progress"]["done"] is True


def test_phase_annotation_accepts_the_exact_reusable_observation_evidence_id(
    tmp_path: Path,
) -> None:
    from android_ui_analyser.session_artifacts import observation_evidence_id

    engine = _engine(tmp_path, "evidence-frame-proof")
    started = engine.session_start(
        "Verify Grammar Mathematics visible",
        observation=_observation(engine.device.serial),
    )
    proof = _control_observation(engine.device.serial, "Grammar Mathematics")
    proof.meta.fingerprint = "grammar-mathematics-proof"
    engine._write_cache(proof)
    evidence_id = observation_evidence_id(
        started["session_id"], proof.model_dump(mode="json")
    )

    completed = engine.session_mark_phase(
        "phase_1",
        evidence_id,
        session_id=started["session_id"],
    )

    assert completed["goal_progress"]["done"] is True


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
    assert "phases" not in decorated["goal_progress"]
    assert decorated["goal_progress"]["upcoming"] == [
        {
            "id": "phase_2",
            "objective": "Establish and verify the requested offline network state",
            "kind": "environment",
        }
    ]
    assert "phases" in engine.session_progress()["goal_progress"]


def test_goal_session_does_not_recommend_configured_destructive_manual_control(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={
            "enabled": False,
            "dir": str(tmp_path / "memory"),
            "destructive_labels": ["archive"],
        },
    )
    engine = Engine(cfg, device=FakeDevice(serial="goal-destructive-control"))
    observed = _control_observation(engine.device.serial, "Archive")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)

    started = engine.session_start("Archive catalog. Then inspect details")

    assert started["recommended_call"]["kind"] == "manual_observation"
    assert started["recommended_call"]["executes"] is False
    assert "tap-and-analyze" not in started["recommended_call"]["cli"]


def test_goal_session_does_not_execute_weak_incidental_one_token_control(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-weak-control")
    observed = _control_observation(engine.device.serial, "Online")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)

    started = engine.session_start(
        "Establish cached catalog online. Then verify the catalog details"
    )

    assert started["recommended_call"]["kind"] == "manual_observation"
    assert started["recommended_call"]["executes"] is False
    assert "tap-and-analyze" not in started["recommended_call"]["cli"]


def test_goal_session_ignores_generic_one_term_overlap_in_multiword_control(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-generic-control")
    observed = _control_observation(engine.device.serial, "Search Settings")
    observed.elements.append(
        Element(
            id=42,
            type="android.widget.TextView",
            text="Network & internet",
            bounds=(40, 440, 800, 560),
            center=(420, 500),
            clickable=True,
            source=Source.hierarchy,
        )
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)

    started = engine.session_start("On Android Settings, perform one harmless saveable UI action")

    next_call = started["goal_progress"]["next_call"]
    assert next_call["kind"] == "manual_observation"
    assert next_call["executes"] is False
    assert "tap-and-analyze" not in next_call["cli"]


def test_goal_session_keeps_specific_multi_term_manual_control_match(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-specific-control")
    observed = _control_observation(engine.device.serial, "Search Settings")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)

    started = engine.session_start("Search Settings")

    next_call = started["goal_progress"]["next_call"]
    assert next_call["kind"] == "manual_action"
    assert next_call["cli"] == "aua tap-and-analyze --text 'Search Settings'"
    assert next_call["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"text": "Search Settings"},
    }
    assert next_call["executes"] is True


def test_multiphase_recommendations_are_compact_and_planned_from_the_active_frame(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-lazy-phase-planning")
    vocabulary = _control_observation(engine.device.serial, "Open Vocabulary")
    equation = _control_observation(engine.device.serial, "Open Equation")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: vocabulary)

    started = engine.session_start("Open Vocabulary; then open Equation")

    assert "phases" not in started["goal_progress"]
    assert started["goal_progress"]["next_call"]["mcp"]["arguments"] == {"text": "Open Vocabulary"}
    persisted = load_session_state(tmp_path / "cache", session_id=started["session_id"])
    assert persisted is not None
    assert persisted.phases[1].recommended_call is None

    engine.session_mark_phase("phase_1", "Open Vocabulary control opened")
    progressed = engine.session_progress(observation=equation)

    assert progressed["goal_progress"]["current"]["id"] == "phase_2"
    assert progressed["goal_progress"]["next_call"]["cli"] == (
        "aua tap-and-analyze --text 'Open Equation'"
    )
    assert progressed["goal_progress"]["next_call"]["mcp"]["arguments"] == {"text": "Open Equation"}
    reloaded = load_session_state(tmp_path / "cache", session_id=started["session_id"])
    assert reloaded is not None
    assert reloaded.phases[1].recommended_call == progressed["goal_progress"]["next_call"]


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
    monkeypatch.setattr(
        "android_ui_analyser.platforms.android.AndroidPlatform.connect",
        lambda _platform, _target_id=None: device,
    )
    # Attached means listed: selection stays behind the adapter, and a headed request verifies
    # the candidate's actual window/audio facts via the adapter probe.
    monkeypatch.setattr(
        "android_ui_analyser.platforms.android.AndroidPlatform.list_targets",
        lambda _platform: [
            DeviceInfo(serial=device.serial, model="fake", android_version="14")
        ],
    )
    monkeypatch.setattr(
        "android_ui_analyser.platforms.android.probe_android_capabilities",
        lambda _cache, _serial: {"headed": True, "audio": True},
    )
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


def test_animation_goal_enables_scales_and_session_finish_restores_them(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from android_ui_analyser import devopts

    engine = _engine(tmp_path, "goal-animation")
    setup_backup = tmp_path / "setup-animation.json"
    devopts.anim_off(engine.device.shell, setup_backup)
    monkeypatch.setattr(
        engine, "analyze", lambda **_kwargs: _observation(engine.device.serial)
    )

    started = engine.session_start("verify the transition animation and easing")

    assert started["animations"] == {"requested": True, "enabled": True, "source": "goal"}
    assert devopts.read_state(engine.device.shell)["anim"]["window_animation_scale"] == "1"

    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

    assert finished["ok"] is True
    assert finished["cleanup"][0]["action"] == "animation_restore"
    assert devopts.read_state(engine.device.shell)["anim"]["window_animation_scale"] == "0"


@pytest.mark.parametrize(
    ("session_kwargs", "source"),
    [({"animations": True}, "flag"), ({"needs": ["animations"]}, "needs")],
)
def test_explicit_animation_session_controls_enable_and_restore(
    session_kwargs: dict[str, Any], source: str, monkeypatch: Any, tmp_path: Path
) -> None:
    from android_ui_analyser import devopts

    engine = _engine(tmp_path / source, f"goal-animation-{source}")
    devopts.anim_off(engine.device.shell, tmp_path / f"setup-{source}.json")
    monkeypatch.setattr(
        engine, "analyze", lambda **_kwargs: _observation(engine.device.serial)
    )

    started = engine.session_start("inspect the final visual state", **session_kwargs)
    assert started["animations"]["source"] == source
    assert started["animations"]["enabled"] is True

    engine.session_finish(started["session_id"], allow_incomplete=True)
    assert devopts.read_state(engine.device.shell)["anim"]["window_animation_scale"] == "0"


def test_session_start_app_alias_launches_and_reuses_that_observation(
    monkeypatch: Any, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, "goal-app-context")
    observed = _observation(engine.device.serial)
    launches: list[tuple[str, str | None]] = []

    def app_launch(_action: str, **kwargs: Any) -> Any:
        launches.append((kwargs["package"], kwargs.get("activity")))
        return engine_mod.ActionResult(
            ok=True,
            action="app-launch",
            observation=observed,
            observation_present=True,
        )

    monkeypatch.setattr(engine, "app", app_launch)
    monkeypatch.setattr(
        engine,
        "analyze",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("duplicate analyze")),
    )

    started = engine.session_start(
        "inspect catalog",
        package="com.example.catalog",
        activity=".MainActivity",
    )

    assert launches == [("com.example.catalog", ".MainActivity")]
    assert started["package"] == "com.example.catalog"


def test_session_start_refreshes_an_explicitly_unstable_launch_readback(
    monkeypatch: Any, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, "goal-app-unstable")
    weak = _observation(engine.device.serial)
    fresh = _control_observation(engine.device.serial, "History archive")
    fresh.elements[0].resource_id = "com.example.catalog:id/historyArchive"
    refreshes: list[str] = []

    monkeypatch.setattr(
        engine,
        "app",
        lambda *_args, **_kwargs: engine_mod.ActionResult(
            ok=True,
            action="app-launch",
            observation=weak,
            observation_present=True,
            next_actions=None,
            note=(
                "The app is foreground, but its launch screen has not produced a stable "
                "readback yet, so numeric next actions are withheld. Run `aua analyze` once "
                "before acting on an id."
            ),
        ),
    )

    def refresh(package: str) -> AnalyzeResult:
        refreshes.append(package)
        return fresh

    monkeypatch.setattr(engine, "_await_launch_hierarchy", refresh)

    started = engine.session_start(
        "Open History archive",
        package="com.example.catalog",
    )

    assert refreshes == ["com.example.catalog"]
    assert started["recommended_call"]["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"rid": "historyArchive"},
    }


def test_session_start_discards_a_launch_observation_from_the_previous_package(
    monkeypatch: Any, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, "goal-launch-authoritative")
    stale = _observation(engine.device.serial)
    stale.screen.package = "com.example.previous"
    fresh = _observation(engine.device.serial)
    analyzes: list[dict[str, Any]] = []

    monkeypatch.setattr(
        engine,
        "app",
        lambda *_args, **_kwargs: engine_mod.ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
        ),
    )

    def analyze(**kwargs: Any) -> AnalyzeResult:
        analyzes.append(kwargs)
        return fresh

    monkeypatch.setattr(engine, "analyze", analyze)

    started = engine.session_start("inspect catalog", package="com.example.catalog")

    assert started["package"] == "com.example.catalog"
    assert analyzes == [{"source": "hierarchy", "with_ocr": False, "no_cache": True}]


def test_session_start_refuses_a_persistently_mixed_launch_frame(
    monkeypatch: Any, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, "goal-launch-mismatch")
    stale = _observation(engine.device.serial)
    stale.screen.package = "com.example.previous"
    monkeypatch.setattr(
        engine,
        "app",
        lambda *_args, **_kwargs: engine_mod.ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
        ),
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: stale)

    with pytest.raises(engine_mod.DeviceError) as raised:
        engine.session_start("inspect catalog", package="com.example.catalog")

    assert raised.value.code == "launch_observation_mismatch"
    assert not list((tmp_path / "cache" / "sessions").glob("*.json"))


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


def test_review_does_not_penalize_wait_after_transitional_launch_observation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-launch-transition")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
    }
    risk = "launch produced only framework shell nodes; observation is transitional"
    journal.record(
        **common,
        cmd="app_launch",
        ok=True,
        result={
            "ok": True,
            "action": "app-launch",
            "stale_risk": risk,
            "observation": {"elements": [], "meta": {"stale_risk": risk}},
        },
    )
    journal.record(
        **common,
        cmd="wait",
        ok=True,
        result={"ok": True, "action": "wait", "observation": {"elements": []}},
    )

    review = engine.session_review(started["session_id"])

    assert "wait_after_observed_action" not in review["patterns"]
    assert not any(item["id"] == "fold_until" for item in review["advice"])


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


def test_review_accounts_for_matching_expected_error_without_poisoning_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-expected-error")
    started = _start(engine, monkeypatch)
    common = {
        "cache_dir": engine.config.cache.dir,
        "serial": engine.device.serial,
        "source": "cli",
        "owner": None,
    }
    journal.record(
        **common,
        cmd="flow_save",
        ok=False,
        error={"code": "usage", "message": "intentional probe"},
        extra={"expected_error_code": "usage", "expected_error_matched": True},
    )
    journal.record(**common, cmd="flow_delete", ok=True, result={"ok": True})

    review = engine.session_review(started["session_id"])

    assert review["run_ok"] is True
    assert review["failures"] == 0
    assert review["accounting"] == {
        "journal_events": 2,
        "top_level_calls": 2,
        "folded_internal_events": 0,
        "lifecycle_calls": 0,
        "task_calls": 2,
        "reporting_call_included": False,
        "top_level_calls_including_reporting_call": 3,
        "expected_error_probes": 1,
        "expected_error_matches": 1,
        "unexpected_failures": 0,
    }


def test_review_fails_when_declared_expected_error_does_not_happen(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, "goal-missing-expected-error")
    started = _start(engine, monkeypatch)
    journal.record(
        cache_dir=engine.config.cache.dir,
        serial=engine.device.serial,
        source="cli",
        owner=None,
        cmd="flow_save",
        ok=True,
        result={"ok": True},
        extra={"expected_error_code": "usage", "expected_error_matched": False},
    )

    review = engine.session_review(started["session_id"])

    assert review["run_ok"] is False
    assert review["failures"] == 1
    assert review["accounting"]["expected_error_probes"] == 1
    assert review["accounting"]["expected_error_matches"] == 0


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

    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

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

    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

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


def test_explicit_session_emulator_start_is_handed_to_the_warm_pool(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser import emulator

    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    engine = Engine(cfg)
    # Keep live host targets out of this deterministic provisioning test.
    online: list[DeviceInfo] = []
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(
        engine.platform,
        "probe_target_capabilities",
        lambda _serial: {"headed": True, "audio": True},
    )
    starts: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    monkeypatch.setattr(emulator, "select_avd_for_session", lambda avd, **_kwargs: avd)

    def fake_start(avd: str, **kwargs: Any) -> dict[str, Any]:
        starts.append({"avd": avd, **kwargs})
        online.append(
            DeviceInfo(serial="emulator-5590", model="fake", android_version="14")
        )
        return {
            "ok": True,
            "serial": "emulator-5590",
            "avd": avd,
            "instance": "fake.p5590",
            "pid": 4242,
        }

    monkeypatch.setattr(emulator, "start", fake_start)
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
        audio=True,
        avd="Small_Phone",
    )
    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

    assert started["emulator_started"] is True
    assert starts[0]["headless"] is False
    assert starts[0]["audio"] is True
    assert stops == []
    assert finished["ok"] is True
    assert [item["action"] for item in finished["cleanup"]] == [
        "lease_release",
        "owned_emulator_handoff",
    ]
    handoff = finished["cleanup"][-1]["result"]
    assert handoff["serial"] == "emulator-5590"
    assert handoff["retained"] is True
    assert handoff["leased"] is False
    assert handoff["idle_stop_s"] == 1200.0


def test_restore_error_keeps_owned_emulator_cached_and_leased_for_retry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser import emulator

    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    engine = Engine(cfg)
    # A live host target must never satisfy this deterministic provisioning test.
    online: list[DeviceInfo] = []
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(emulator, "select_avd_for_session", lambda avd, **_kwargs: avd or "fake")

    def fake_start(_avd: str, **_kwargs: Any) -> dict[str, Any]:
        online.append(
            DeviceInfo(serial="emulator-5592", model="fake", android_version="14")
        )
        return {
            "ok": True,
            "serial": "emulator-5592",
            "avd": "fake",
            "instance": "fake.p5592",
            "pid": 4242,
        }

    monkeypatch.setattr(emulator, "start", fake_start)
    monkeypatch.setattr(
        emulator,
        "stop",
        lambda **kwargs: pytest.fail(f"session finish must not stop the warm emulator: {kwargs}"),
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation("emulator-5592"))

    started = engine.session_start("inspect offline state", start_emulator=True)
    engine._device = FakeDevice(serial="emulator-5592")
    backup = network.backup_path(engine.config.cache.dir, "emulator-5592")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("session-owned", encoding="utf-8")
    monkeypatch.setattr(
        engine,
        "network_restore",
        lambda: {"ok": False, "detail": "network could not be restored"},
    )
    closed: list[str] = []
    monkeypatch.setattr(engine, "close", lambda: closed.append("emulator-5592"))

    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

    assert finished["ok"] is False
    assert closed == []
    assert all(item["action"] != "owned_emulator_handoff" for item in finished["cleanup"])
    assert engine._device is not None
    assert engine._device.serial == "emulator-5592"
    assert engine._lease_owner_resolved is not None
