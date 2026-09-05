"""Developer-option profiles — save/restore via FakeDevice shell."""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser import device_ledger, devopts
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
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


class _WriteAheadDevDevice(FakeDevice):
    def shell(self, command: str) -> str:
        if command.startswith("settings put "):
            entries = device_ledger.read_ledger(self.serial, platform="android")
            assert any(
                entry.kind == "developer_settings"
                and entry.op == "restore_developer_settings"
                for entry in entries
            ), "developer settings reached the target before their restore record"
        return super().shell(command)


def test_crash_and_profile_mutations_record_before_the_first_settings_write(
    tmp_path: Path,
) -> None:
    device = _WriteAheadDevDevice(serial="developer-settings-runtime")
    cfg = make_config(cache={"dir": str(tmp_path / "cache")})
    engine = Engine(cfg, device=device)

    engine.dev_crashes(False)
    pending = device_ledger.read_ledger(device.serial, platform=engine.platform.name)
    entry = next(item for item in pending if item.key == "developer_settings")
    assert entry.args["backup_path"] == str(engine._dev_backup_path())

    engine.dev_profile("ac")
    restored = engine.dev_profile("default")

    assert restored["action"] == "dev-profile-default"
    assert device_ledger.read_ledger(device.serial, platform=engine.platform.name) == []


def test_unverified_developer_restore_keeps_backup_and_ledger(tmp_path: Path) -> None:
    class IgnoredRestoreDevice(FakeDevice):
        ignore_settings_writes = False

        def shell(self, command: str) -> str:
            if self.ignore_settings_writes and command.startswith("settings put "):
                return ""
            return super().shell(command)

    device = IgnoredRestoreDevice(serial="ignored-developer-restore")
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}),
        device=device,
    )
    engine.dev_profile("ac")
    device.ignore_settings_writes = True

    with pytest.raises(DeviceError) as raised:
        engine.dev_profile("default")

    assert raised.value.code == "developer_settings_restore_unverified"
    assert engine._dev_backup_path().is_file()
    assert any(
        entry.key == "developer_settings"
        for entry in device_ledger.read_ledger(device.serial, platform=engine.platform.name)
    )


def test_repeated_developer_changes_keep_the_first_cross_cache_restore_point(
    tmp_path: Path,
) -> None:
    device = FakeDevice(serial="shared-developer-runtime")
    first = Engine(
        make_config(cache={"dir": str(tmp_path / "first-cache")}),
        device=device,
    )
    first.dev_crashes(False)
    original_entry = device_ledger.read_ledger(
        device.serial, platform=first.platform.name
    )[0]

    second = Engine(
        make_config(cache={"dir": str(tmp_path / "second-cache")}),
        device=device,
    )
    second.dev_profile("ac")
    retained_entry = device_ledger.read_ledger(
        device.serial, platform=second.platform.name
    )[0]

    assert retained_entry.args == original_entry.args
    assert retained_entry.recorded == original_entry.recorded
    second.dev_profile("default")
    assert device_ledger.read_ledger(device.serial, platform=second.platform.name) == []
