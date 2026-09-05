"""Idle watchdog + activity touch for orphaned headless AVDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import emulator as emu
from android_ui_analyser import emulator_watchdog as wd
from android_ui_analyser import journal as journal_mod
from android_ui_analyser import target_activity


def test_touch_activity_bumps_matching_serial(tmp_path: Path) -> None:
    rec = tmp_path / "emulator"
    rec.mkdir()
    path = rec / "Pixel.json"
    path.write_text(
        json.dumps(
            {
                "avd": "Pixel",
                "instance": "Pixel",
                "serial": "emulator-5554",
                "started_by_aua": True,
                "last_activity": 1.0,
            }
        ),
        encoding="utf-8",
    )
    emu.touch_activity(tmp_path, "emulator-5554")
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["last_activity"] > 1.0


def test_journal_record_touches_activity(tmp_path: Path) -> None:
    rec = tmp_path / "emulator"
    rec.mkdir()
    path = rec / "Pixel.json"
    path.write_text(
        json.dumps(
            {
                "avd": "Pixel",
                "instance": "Pixel",
                "serial": "emulator-5554",
                "started_by_aua": True,
                "last_activity": 1.0,
            }
        ),
        encoding="utf-8",
    )
    journal_mod.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="cli",
        cmd="analyze",
        ok=True,
    )
    heartbeat = target_activity.read(tmp_path, "emulator-5554", platform="android")
    assert heartbeat is not None
    assert heartbeat["last_activity"] > 1.0


def test_watchdog_reads_platform_neutral_target_heartbeat(tmp_path: Path) -> None:
    target_activity.touch(tmp_path, "emulator-5554", platform="android", at=42.0)

    assert wd._last_activity({"last_activity": 1.0}, tmp_path, "emulator-5554") == 42.0


def test_watchdog_stops_when_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rec = tmp_path / "emulator"
    rec.mkdir()
    path = rec / "Pixel.json"
    path.write_text(
        json.dumps(
            {
                "avd": "Pixel",
                "instance": "Pixel",
                "serial": "emulator-5554",
                "pid": 4242,
                "started_by_aua": True,
                "idle_timeout_s": 10,
                "last_activity": 0.0,
                "started_at": 0.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        emu,
        "running_emulators",
        lambda: [
            {
                "serial": "emulator-5554",
                "state": "device",
                "model": "x",
                "android_version": "14",
            }
        ],
    )
    stopped: list[dict[str, Any]] = []

    def fake_stop(**kwargs: Any) -> dict[str, Any]:
        stopped.append(kwargs)
        path.unlink(missing_ok=True)
        return {"ok": True}

    monkeypatch.setattr(emu, "stop_spawned_instance", fake_stop)
    monkeypatch.setattr(wd.time, "sleep", lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    # First loop should fire idle stop before sleep.
    monkeypatch.setattr(wd.time, "time", lambda: 100.0)
    code = wd.run_watchdog(cache_dir=str(tmp_path), instance="Pixel")
    assert code == 0
    assert stopped and stopped[0].get("instance") == "Pixel"
    assert stopped[0].get("pid") == 4242


def test_watchdog_exits_when_idle_disabled(tmp_path: Path) -> None:
    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "Pixel.json").write_text(
        json.dumps(
            {
                "avd": "Pixel",
                "serial": "emulator-5554",
                "started_by_aua": True,
                "idle_timeout_s": 0,
            }
        ),
        encoding="utf-8",
    )
    assert wd.run_watchdog(cache_dir=str(tmp_path), instance="Pixel") == 0


def test_stop_mine_kills_watchdog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "only.json").write_text(
        json.dumps(
            {
                "avd": "only",
                "instance": "only",
                "serial": "emulator-5554",
                "pid": 99,
                "watchdog_pid": 8888,
                "started_by_aua": True,
            }
        ),
        encoding="utf-8",
    )
    killed_emu: list[str] = []
    killed_pids: list[int] = []
    monkeypatch.setattr(emu, "_adb_emu_kill", lambda s: killed_emu.append(s))
    monkeypatch.setattr(emu, "running_emulators", lambda: [])
    monkeypatch.setattr(
        emu.os,
        "kill",
        lambda pid, sig: killed_pids.append(pid),
    )
    monkeypatch.setattr(
        emu.os,
        "killpg",
        lambda pid, sig: None,
    )
    out = emu.stop(mine=True, cache_dir=tmp_path)
    assert killed_emu == [], "owned teardown must not involve the shared adb server"
    assert 8888 in killed_pids
    assert out["stopped"] == ["emulator-5554"]
    assert not (rec / "only.json").exists()
