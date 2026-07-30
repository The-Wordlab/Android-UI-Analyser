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


def test_inspect_avd_marks_playstore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu

    avd = tmp_path / "Small_Phone.avd"
    avd.mkdir()
    (avd / "config.ini").write_text(
        "PlayStore.enabled=true\n"
        "tag.id=google_apis_playstore\n"
        "image.sysdir.1=system-images/android-34/google_apis_playstore/arm64-v8a/\n"
        "hw.lcd.width=720\n"
        "hw.lcd.height=1280\n"
        "hw.lcd.density=320\n"
        "hw.ramSize=1024\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(emu, "avd_dir", lambda name: tmp_path / f"{name}.avd")
    info = emu.inspect_avd("Small_Phone")
    assert info["play_store"] is True
    assert info["rootable"] is False
    assert info["api"] == 34


def test_inspect_avd_marks_google_apis_rootable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import emulator as emu

    avd = tmp_path / "aua_proxy.avd"
    avd.mkdir()
    (avd / "config.ini").write_text(
        "PlayStore.enabled=false\n"
        "tag.id=google_apis\n"
        "image.sysdir.1=system-images/android-30/google_apis/arm64-v8a/\n"
        "hw.ramSize=1536\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(emu, "avd_dir", lambda name: tmp_path / f"{name}.avd")
    info = emu.inspect_avd("aua_proxy")
    assert info["rootable"] is True
    assert info["play_store"] is False
    assert info["api"] == 30


def test_recommend_proxy_avd(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu, "preferred_abi", lambda: "arm64-v8a")
    monkeypatch.setattr(emu, "_list_avd_names", lambda: ["Small_Phone"])
    monkeypatch.setattr(
        emu,
        "inspect_avd",
        lambda n: {
            "name": n,
            "rootable": False,
            "play_store": True,
            "tag": "google_apis_playstore",
            "api": 34,
        },
    )
    out = emu.recommend_proxy_avd(api=30)
    assert out["package"] == "system-images;android-30;google_apis;arm64-v8a"
    assert out["name"] == "aua_proxy"
    assert "ensure-proxy" in out["create"]
    assert out["existing_rootable"] == []


def test_ensure_proxy_avd_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import emulator as emu

    avd = tmp_path / "aua_proxy.avd"
    avd.mkdir()
    (avd / "config.ini").write_text(
        "tag.id=google_apis\n"
        "image.sysdir.1=system-images/android-30/google_apis/arm64-v8a/\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(emu, "avd_dir", lambda name: tmp_path / f"{name}.avd")
    monkeypatch.setattr(emu, "preferred_abi", lambda: "arm64-v8a")
    monkeypatch.setattr(emu, "_list_avd_names", lambda: ["aua_proxy"])

    called = {"sdk": 0}

    def boom(*_a: Any, **_k: Any) -> Any:
        called["sdk"] += 1
        raise AssertionError("should not install when already rootable")

    monkeypatch.setattr(emu, "sdkmanager_bin", boom)
    out = emu.ensure_proxy_avd(name="aua_proxy", api=30)
    assert out["created"] is False
    assert out["ok"] is True
    assert called["sdk"] == 0


def test_ensure_proxy_avd_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu, "avd_dir", lambda name: tmp_path / f"{name}.avd")
    monkeypatch.setattr(emu, "preferred_abi", lambda: "arm64-v8a")
    monkeypatch.setattr(emu, "_list_avd_names", lambda: [])
    monkeypatch.setattr(emu, "sdk_root", lambda: tmp_path)
    monkeypatch.setattr(emu, "sdkmanager_bin", lambda: "/fake/sdkmanager")
    monkeypatch.setattr(emu, "avdmanager_bin", lambda: "/fake/avdmanager")
    monkeypatch.setattr(emu, "_image_installed", lambda _p: False)

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        if "create" in cmd:
            avd = tmp_path / "aua_proxy.avd"
            avd.mkdir(exist_ok=True)
            (avd / "config.ini").write_text(
                "tag.id=google_apis\n"
                "image.sysdir.1=system-images/android-30/google_apis/arm64-v8a/\n"
                "hw.ramSize=2048\n"
                "hw.lcd.width=1080\n"
                "hw.lcd.height=2220\n",
                encoding="utf-8",
            )
        return R()

    monkeypatch.setattr(emu, "_run_sdk", fake_run)
    out = emu.ensure_proxy_avd(name="aua_proxy", api=30, accept_licenses=False)
    assert out["created"] is True
    cfg = (tmp_path / "aua_proxy.avd" / "config.ini").read_text(encoding="utf-8")
    assert "hw.lcd.width=720" in cfg
    assert "hw.ramSize=1536" in cfg


def test_cli_recommend_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(
        emu,
        "recommend_proxy_avd",
        lambda **k: {
            "ok": True,
            "action": "emulator-recommend-proxy",
            "package": "system-images;android-30;google_apis;arm64-v8a",
            "create": "aua emulator ensure-proxy",
        },
    )
    result = runner.invoke(app, ["emulator", "recommend-proxy"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert "google_apis" in data["package"]
