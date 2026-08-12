"""Goal sessions correlate traces, coach waste, and clean up only their own state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser import journal, network, network_profiles
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from conftest import FakeDevice, make_config


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


def _engine(tmp_path: Path, serial: str = "goal-life") -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    return Engine(cfg, device=FakeDevice(serial=serial))


def _start(engine: Engine, monkeypatch: Any) -> dict[str, Any]:
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation(engine.device.serial))
    return engine.session_start("verify cached results while offline")


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
