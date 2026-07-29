"""Tests for headless AVD helpers (mocked — no real emulator required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.errors import UsageError
from conftest import make_config

runner = CliRunner()


def test_list_avds_parses_emulator_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu, "emulator_bin", lambda: "/fake/emulator")

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        class R:
            stdout = "Pixel_7\nPixel_8\n"
            returncode = 0

        assert cmd[0] == "/fake/emulator"
        assert "-list-avds" in cmd
        return R()

    monkeypatch.setattr(emu.subprocess, "run", fake_run)
    out = emu.list_avds()
    assert out["avds"] == ["Pixel_7", "Pixel_8"]
    assert out["count"] == 2


def test_start_requires_avd_when_multiple(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(
        emu, "list_avds", lambda: {"ok": True, "avds": ["a", "b"], "count": 2, "emulator": "x"}
    )
    monkeypatch.setattr(emu, "running_emulators", lambda: [])
    with pytest.raises(UsageError, match="multiple AVDs"):
        emu.start(cache_dir=tmp_path)


def test_start_headless_waits_for_serial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(
        emu, "list_avds", lambda: {"ok": True, "avds": ["only"], "count": 1, "emulator": "x"}
    )
    calls = {"n": 0}

    def running() -> list[dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] < 2:
            return []
        return [{"serial": "emulator-5554", "state": "device", "model": "x", "android_version": "14"}]

    monkeypatch.setattr(emu, "running_emulators", running)

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(emu.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(emu.time, "sleep", lambda *_: None)

    out = emu.start(headless=True, wait_s=5, cache_dir=tmp_path)
    assert out["serial"] == "emulator-5554"
    assert out["headless"] is True
    assert out["avd"] == "only"
    meta = json.loads((tmp_path / "emulator" / "only.json").read_text(encoding="utf-8"))
    assert meta["pid"] == 4242
    assert "-no-window" in meta["cmd"]


def test_cli_emulator_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(
        emu,
        "list_avds",
        lambda: {
            "ok": True,
            "action": "emulator-list",
            "emulator": "/e",
            "avds": ["Pixel_7"],
            "count": 1,
            "hint": None,
        },
    )
    result = runner.invoke(app, ["emulator", "list"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["avds"] == ["Pixel_7"]


def test_doctor_hints_headless_when_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu
    from android_ui_analyser.cli import _build_doctor_report
    from android_ui_analyser.engine import Engine

    cfg = make_config()
    eng = Engine(cfg)

    monkeypatch.setattr(Engine, "list_devices", lambda self: [])
    monkeypatch.setattr(
        emu,
        "status",
        lambda **k: {
            "ok": True,
            "emulator_ok": True,
            "emulator": "/e",
            "avds": ["Pixel_7"],
            "running": [],
        },
    )
    report = _build_doctor_report(eng)
    devices = report["checks"]["devices"]
    assert devices["count"] == 0
    assert "emulator start" in (devices.get("hint") or "")
    assert report["checks"]["emulator"]["ok"] is True


def test_status_includes_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(emu, "list_avds", lambda: {"avds": ["Pixel_7"]})
    monkeypatch.setattr(
        emu,
        "running_emulators",
        lambda: [
            {
                "serial": "emulator-5554",
                "model": "sdk",
                "android_version": "14",
                "state": "device",
            }
        ],
    )
    out = emu.status(cache_dir=tmp_path)
    assert out["running"][0]["serial"] == "emulator-5554"
