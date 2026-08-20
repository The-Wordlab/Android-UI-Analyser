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


def test_started_record_scan_tolerates_parallel_cleanup(tmp_path: Path, monkeypatch) -> None:
    from android_ui_analyser import emulator as emu

    record_dir = tmp_path / "emulator"
    record_dir.mkdir(parents=True)
    disappearing = record_dir / "race.json"
    disappearing.write_text("{}", encoding="utf-8")
    original = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == disappearing:
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert emu._aua_started_records(tmp_path) == []


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
    monkeypatch.setattr(emu, "_wait_for_boot", lambda *_a, **_k: True)
    monkeypatch.setattr(emu, "_spawn_idle_watchdog", lambda **k: 7777)
    cleanup_calls: list[tuple[str, Path]] = []
    cleanup_results = iter(
        [
            {"ok": True, "checked": True, "cleared": False, "state_before": "unproxied"},
            {
                "ok": True,
                "checked": True,
                "cleared": True,
                "state_before": "blackholed",
                "state_after": "unproxied",
            },
        ]
    )
    monkeypatch.setattr(
        emu,
        "_clear_inherited_blackholed_proxy",
        lambda serial, *, cache_dir: (
            cleanup_calls.append((serial, Path(cache_dir)))
            or next(cleanup_results)
        ),
    )

    out = emu.start(headless=True, wait_s=5, cache_dir=tmp_path)
    assert out["serial"] == "emulator-5554"
    assert out["headless"] is True
    assert out["avd"] == "only"
    assert out["gpu"] == "host" or out["gpu"] == "swiftshader" or out["gpu"] == "auto"
    assert out["idle_timeout_s"] == 1200.0
    assert out["watchdog_pid"] == 7777
    assert "aua session start --goal <goal>" in out["hint"]
    assert "Standalone start only provisions" in out["hint"]
    assert "omit `--serial` from ordinary commands" in out["hint"]
    assert "emulator stop --mine" in out["hint"]
    assert "--serial emulator-5554" in out["hint"]
    assert "Pin with" not in out["hint"]
    meta = json.loads((tmp_path / "emulator" / "only.json").read_text(encoding="utf-8"))
    assert meta["pid"] == 4242
    assert meta["watchdog_pid"] == 7777
    assert meta["idle_timeout_s"] == 1200.0
    assert "-no-window" in meta["cmd"]
    assert "-gpu" in meta["cmd"]
    assert "swiftshader_indirect" not in meta["cmd"]
    assert meta.get("started_by_aua") is True
    assert cleanup_calls == [
        ("emulator-5554", tmp_path),
        ("emulator-5554", tmp_path),
    ]
    assert out["proxy_cleanup"]["cleared"] is True


def test_emulator_start_help_explains_provisioning_vs_lease_selection() -> None:
    result = runner.invoke(app, ["emulator", "start", "--help"])

    assert result.exit_code == 0
    help_text = " ".join(result.stdout.split())
    assert "owns pool selection" in help_text
    assert "process-bound leasing" in help_text
    assert "Pin later commands" not in help_text


def test_start_idle_stop_zero_skips_watchdog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        return [
            {
                "serial": "emulator-5554",
                "state": "device",
                "model": "x",
                "android_version": "14",
            }
        ]

    monkeypatch.setattr(emu, "running_emulators", running)

    class FakeProc:
        pid = 4242

    spawned: list[Any] = []
    monkeypatch.setattr(emu.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(emu.time, "sleep", lambda *_: None)
    monkeypatch.setattr(emu, "_wait_for_boot", lambda *_a, **_k: True)
    monkeypatch.setattr(
        emu, "_spawn_idle_watchdog", lambda **k: spawned.append(k) or 1
    )
    monkeypatch.setattr(
        emu,
        "_clear_inherited_blackholed_proxy",
        lambda *_a, **_k: {"ok": True, "checked": True, "cleared": False},
    )

    out = emu.start(headless=True, wait_s=5, cache_dir=tmp_path, idle_timeout_s=0)
    assert out["idle_timeout_s"] == 0.0
    assert out["watchdog_pid"] is None
    assert spawned == []


def test_start_parallel_allocates_port_and_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
        return [
            {
                "serial": "emulator-5554",
                "state": "device",
                "model": "x",
                "android_version": "14",
            }
        ]

    monkeypatch.setattr(emu, "running_emulators", running)

    class FakeProc:
        pid = 4242

    cmds: list[list[str]] = []

    def fake_popen(cmd: list[str], **_k: Any) -> FakeProc:
        cmds.append(list(cmd))
        return FakeProc()

    monkeypatch.setattr(emu.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(emu.time, "sleep", lambda *_: None)
    monkeypatch.setattr(emu, "_wait_for_boot", lambda *_a, **_k: True)
    monkeypatch.setattr(emu, "_spawn_idle_watchdog", lambda **k: 7777)
    monkeypatch.setattr(
        emu,
        "_clear_inherited_blackholed_proxy",
        lambda *_a, **_k: {"ok": True, "checked": True, "cleared": False},
    )

    out = emu.start(
        headless=True,
        wait_s=5,
        cache_dir=tmp_path,
        parallel=True,
        owner="agent-a",
        idle_timeout_s=0,
    )
    assert out["port"] == 5554
    assert out["serial"] == "emulator-5554"
    assert out["owner"] == "agent-a"
    assert out["read_only"] is True
    assert out["instance"] == "only.p5554"
    assert "-port" in cmds[0] and "5554" in cmds[0]
    assert "-read-only" in cmds[0]
    meta = json.loads((tmp_path / "emulator" / "only.p5554.json").read_text(encoding="utf-8"))
    assert meta["owner"] == "agent-a"
    assert meta["port"] == 5554


def test_parallel_second_instance_gets_next_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import emulator as emu

    # First instance already recorded on 5554.
    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "only.p5554.json").write_text(
        json.dumps(
            {
                "avd": "only",
                "instance": "only.p5554",
                "port": 5554,
                "serial": "emulator-5554",
                "owner": "agent-a",
                "started_by_aua": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(emu, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(
        emu, "list_avds", lambda: {"ok": True, "avds": ["only"], "count": 1, "emulator": "x"}
    )
    calls = {"n": 0}

    def running() -> list[dict[str, Any]]:
        calls["n"] += 1
        base = [
            {
                "serial": "emulator-5554",
                "state": "device",
                "model": "x",
                "android_version": "14",
            }
        ]
        if calls["n"] < 3:
            return base
        return base + [
            {
                "serial": "emulator-5556",
                "state": "device",
                "model": "x",
                "android_version": "14",
            }
        ]

    monkeypatch.setattr(emu, "running_emulators", running)

    class FakeProc:
        pid = 99

    monkeypatch.setattr(emu.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(emu.time, "sleep", lambda *_: None)
    monkeypatch.setattr(emu, "_wait_for_boot", lambda *_a, **_k: True)
    monkeypatch.setattr(emu, "_spawn_idle_watchdog", lambda **k: None)
    monkeypatch.setattr(
        emu,
        "_clear_inherited_blackholed_proxy",
        lambda *_a, **_k: {"ok": True, "checked": True, "cleared": False},
    )

    out = emu.start(
        headless=True,
        wait_s=5,
        cache_dir=tmp_path,
        parallel=True,
        owner="agent-b",
        idle_timeout_s=0,
    )
    assert out["port"] == 5556
    assert out["serial"] == "emulator-5556"
    assert out["owner"] == "agent-b"
    assert (tmp_path / "emulator" / "only.p5556.json").is_file()


def test_startup_clears_a_confirmed_unowned_blackholed_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import emulator as emu
    from android_ui_analyser import proxy_mock

    reports = iter(
        [
            {
                "ok": False,
                "state": "blackholed",
                "owned": False,
                "target": {"raw": "127.0.0.1:49097"},
            },
            {
                "ok": True,
                "state": "unproxied",
                "owned": False,
                "target": None,
            },
        ]
    )
    monkeypatch.setattr(proxy_mock, "proxy_health", lambda *_a, **_k: next(reports))
    cleared_states: list[str] = []
    monkeypatch.setattr(proxy_mock, "clear_state", lambda serial: cleared_states.append(serial))
    shell_calls: list[str] = []
    monkeypatch.setattr(emu, "_serial_shell", lambda _serial: lambda cmd: shell_calls.append(cmd) or "")

    out = emu._clear_inherited_blackholed_proxy("emulator-5554", cache_dir=tmp_path)

    assert out == {
        "ok": True,
        "checked": True,
        "cleared": True,
        "state_before": "blackholed",
        "state_after": "unproxied",
        "detail": "cleared an unowned blackholed proxy inherited from the AVD",
    }
    assert shell_calls == [
        "settings put global http_proxy :0",
        "settings delete global http_proxy",
    ]
    assert cleared_states == ["emulator-5554"]


@pytest.mark.parametrize(
    ("state", "owned", "ok"),
    [
        ("healthy", True, True),
        ("degraded", True, False),
        ("foreign", False, True),
        ("unknown", False, False),
        ("unproxied", False, True),
    ],
)
def test_startup_preserves_every_proxy_that_is_not_unowned_and_blackholed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: str,
    owned: bool,
    ok: bool,
) -> None:
    from android_ui_analyser import emulator as emu
    from android_ui_analyser import proxy_mock

    monkeypatch.setattr(
        proxy_mock,
        "proxy_health",
        lambda *_a, **_k: {"ok": ok, "state": state, "owned": owned},
    )
    monkeypatch.setattr(
        emu,
        "_serial_shell",
        lambda _serial: pytest.fail("a preserved proxy must not be changed"),
    )
    monkeypatch.setattr(
        proxy_mock,
        "clear_state",
        lambda _serial: pytest.fail("a preserved proxy must keep its ownership record"),
    )

    out = emu._clear_inherited_blackholed_proxy("emulator-5554", cache_dir=tmp_path)

    assert out["cleared"] is False
    assert out["state_before"] == state
    assert out["state_after"] == state


def test_stop_mine_scoped_by_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import emulator as emu

    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "a.p5554.json").write_text(
        json.dumps(
            {
                "avd": "a",
                "serial": "emulator-5554",
                "pid": 1,
                "owner": "agent-a",
                "started_by_aua": True,
            }
        ),
        encoding="utf-8",
    )
    (rec / "a.p5556.json").write_text(
        json.dumps(
            {
                "avd": "a",
                "serial": "emulator-5556",
                "pid": 2,
                "owner": "agent-b",
                "started_by_aua": True,
            }
        ),
        encoding="utf-8",
    )
    killed: list[str] = []
    monkeypatch.setattr(emu, "_adb_emu_kill", lambda s: killed.append(s))
    monkeypatch.setattr(emu, "running_emulators", lambda: [])
    monkeypatch.setattr(emu.os, "killpg", lambda *a, **k: None)
    out = emu.stop(mine=True, owner="agent-a", cache_dir=tmp_path)
    assert killed == ["emulator-5554"]
    assert out["stopped"] == ["emulator-5554"]
    assert not (rec / "a.p5554.json").exists()
    assert (rec / "a.p5556.json").exists()


def test_default_gpu_mode_mac_uses_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import emulator as emu

    monkeypatch.setattr(emu.sys, "platform", "darwin")
    assert emu.default_gpu_mode(headless=True) == "host"
    monkeypatch.setattr(emu.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert emu.default_gpu_mode(headless=True) == "swiftshader"


def test_stop_mine_kills_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import emulator as emu

    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "only.json").write_text(
        json.dumps({"avd": "only", "serial": "emulator-5554", "pid": 99}),
        encoding="utf-8",
    )
    killed: list[str] = []
    monkeypatch.setattr(emu, "_adb_emu_kill", lambda s: killed.append(s))
    monkeypatch.setattr(emu, "running_emulators", lambda: [])
    out = emu.stop(mine=True, cache_dir=tmp_path)
    assert killed == ["emulator-5554"]
    assert out["stopped"] == ["emulator-5554"]
    assert not (rec / "only.json").exists()


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
