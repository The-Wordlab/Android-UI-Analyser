"""Tests for the sneak-peek dashboard helpers (no real device required)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError


def test_latest_frame_picks_newest(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    root = tmp_path / "captures" / "emulator-5554"
    old = root / "sess-old" / "frames"
    new = root / "sess-new" / "frames"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"old")
    newer = new / "b.jpg"
    newer.write_bytes(b"new")
    os.utime(old / "a.jpg", (time.time() - 10, time.time() - 10))
    os.utime(newer, None)
    got = dash.latest_frame(tmp_path, "emulator-5554")
    assert got == newer


def test_recent_marks_reads_index(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    sess = tmp_path / "captures" / "emulator-5554" / "s1"
    (sess / "frames").mkdir(parents=True)
    idx = sess / "index.jsonl"
    idx.write_text(
        json.dumps({"t_ms": 1, "path": "frames/1.jpg", "hash": "a"})
        + "\n"
        + json.dumps({"t_ms": 2, "path": "frames/2.jpg", "hash": "b", "action": "tap:4"})
        + "\n",
        encoding="utf-8",
    )
    marks = dash.recent_marks(tmp_path, "emulator-5554")
    assert len(marks) == 1
    assert marks[0]["action"] == "tap:4"


def test_resolve_serial_requires_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import dashboard as dash

    class D:
        def __init__(self, serial: str) -> None:
            self.serial = serial
            self.state = "device"

    monkeypatch.setattr(
        "android_ui_analyser.device.list_devices",
        lambda: [D("emulator-5554"), D("emulator-5556")],
    )
    with pytest.raises(DeviceError, match="multiple"):
        dash.resolve_serial(None)
    assert dash.resolve_serial("emulator-5554") == "emulator-5554"


def test_resolve_dashboard_targets_grid_when_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import dashboard as dash

    monkeypatch.setattr(
        dash,
        "list_online_serials",
        lambda: ["emulator-5554", "emulator-5556"],
    )
    out = dash.resolve_dashboard_targets(None)
    assert out["mode"] == "grid"
    assert out["serials"] == ["emulator-5554", "emulator-5556"]
    assert out["focus"] is None
    forced = dash.resolve_dashboard_targets("emulator-5554")
    assert forced["mode"] == "detail"
    assert forced["focus"] == "emulator-5554"
    grid = dash.resolve_dashboard_targets(None, grid=True)
    assert grid["mode"] == "grid"


def test_owner_for_serial(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "a.p5554.json").write_text(
        json.dumps(
            {
                "avd": "a",
                "serial": "emulator-5554",
                "owner": "agent-a",
                "started_by_aua": True,
            }
        ),
        encoding="utf-8",
    )
    assert dash.owner_for_serial(tmp_path, "emulator-5554") == "agent-a"
    assert dash.owner_for_serial(tmp_path, "emulator-5556") is None


def test_ensure_capture_falls_back_to_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    cfg.daemon.socket = str(tmp_path / "no-daemon.sock")

    import android_ui_analyser.capture_sidecar as cs

    monkeypatch.setattr(
        cs,
        "start",
        lambda **k: {
            "ok": True,
            "action": "capture-sidecar-start",
            "status": "started",
            "socket": "x",
        },
    )
    out = dash.ensure_capture(serial="emulator-5554", config=cfg)
    assert out["via"] == "sidecar"
    assert out["ok"] is True
