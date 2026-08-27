"""Developer-option profiles — save/restore via FakeDevice shell."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser import devopts
from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config


def test_anim_off_restore_roundtrip(tmp_path: Path) -> None:
    device = FakeDevice()
    backup = tmp_path / "dev.json"
    before = devopts.read_state(device.shell)
    assert before["anim"]["window_animation_scale"] == "1"

    after = devopts.anim_off(device.shell, backup)
    assert after["anim"]["window_animation_scale"] == "0"
    assert backup.is_file()

    restored = devopts.anim_restore(device.shell, backup)
    assert restored["anim"]["window_animation_scale"] == "1"
    assert not backup.exists()


def test_anim_on_restores_disabled_scales(tmp_path: Path) -> None:
    device = FakeDevice()
    backup = tmp_path / "dev.json"
    devopts.anim_off(device.shell, tmp_path / "setup.json")

    enabled = devopts.anim_on(device.shell, backup)
    assert set(enabled["anim"].values()) == {"1"}

    restored = devopts.anim_restore(device.shell, backup)
    assert set(restored["anim"].values()) == {"0"}


def test_profile_ac_and_default(tmp_path: Path) -> None:
    device = FakeDevice()
    backup = tmp_path / "dev.json"
    ac = devopts.profile_ac(device.shell, backup)
    assert ac["anim"]["animator_duration_scale"] == "0"
    assert ac["crashes_visible"] is True
    assert ac["anr_show_background"] == "1"

    default = devopts.profile_default(device.shell, backup)
    assert default["anim"]["window_animation_scale"] == "1"
    assert not backup.is_file()


def test_engine_dev_profile(tmp_path: Path) -> None:
    device = FakeDevice()
    cfg = make_config(cache={"dir": str(tmp_path)})
    engine = Engine(cfg, device=device)
    out = engine.dev_profile("ac")
    assert out["ok"] is True
    assert out["action"] == "dev-profile-ac"
    assert any(c[0] == "shell" and "settings put" in c[1][0] for c in device.calls)
    out2 = engine.dev_profile("default")
    assert out2["action"] == "dev-profile-default"


def test_engine_dev_anim_on_and_restore(tmp_path: Path) -> None:
    device = FakeDevice()
    cfg = make_config(cache={"dir": str(tmp_path)})
    engine = Engine(cfg, device=device)
    devopts.anim_off(device.shell, tmp_path / "setup.json")

    assert engine.dev_anim("on")["anim"]["window_animation_scale"] == "1"
    assert engine.dev_anim("restore")["anim"]["window_animation_scale"] == "0"
