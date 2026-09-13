"""SDK selection and prompt, attributable launcher failures (#11)."""

from __future__ import annotations

import sys
import time

import pytest

from android_ui_analyser import emulator as em
from android_ui_analyser.errors import DeviceError


@pytest.mark.parametrize("key", ["ANDROID_HOME", "ANDROID_SDK_ROOT"])
def test_explicit_sdk_launcher_takes_precedence_over_path(tmp_path, monkeypatch, key):
    sdk = tmp_path / "sdk"
    launcher = sdk / "emulator" / "emulator"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nexit 0\n")
    launcher.chmod(0o755)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setenv(key, str(sdk))
    monkeypatch.setattr(em.shutil, "which", lambda _name: "/old-sdk/emulator")
    assert em.emulator_bin() == str(launcher)


@pytest.mark.parametrize("configured", [False, True])
def test_path_launcher_still_works_without_a_usable_explicit_sdk(tmp_path, monkeypatch, configured):
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    if configured:
        monkeypatch.setenv("ANDROID_HOME", str(tmp_path / "missing-sdk"))
    monkeypatch.setattr(em.shutil, "which", lambda _name: "/path/emulator")
    assert em.emulator_bin() == "/path/emulator"


@pytest.mark.parametrize("explicit_port", [False, True])
def test_exited_launcher_reports_this_attempt_without_waiting_for_timeout(
    tmp_path, monkeypatch, explicit_port
):
    launcher = tmp_path / "fake-emulator"
    launcher.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('Could not launch qemu/darwin-x86_64/qemu-system-aarch64: "
        "No such file or directory', flush=True)\nsys.exit(2)\n"
    )
    launcher.chmod(0o755)
    monkeypatch.setattr(em, "emulator_bin", lambda: str(launcher))
    monkeypatch.setattr(em, "list_avds", lambda: {"avds": ["example"]})
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    monkeypatch.setattr(em, "_POLL_S", 0.01)
    instance = "example.p5556" if explicit_port else "example"
    log = em._pid_dir(tmp_path) / f"{instance}.log"
    log.write_text("OLD FAILURE FROM ANOTHER ATTEMPT\n")
    started = time.monotonic()
    with pytest.raises(DeviceError) as raised:
        em.start(
            "example", port=5556 if explicit_port else None,
            parallel=explicit_port, read_only=False, wait_s=1.5,
            cache_dir=tmp_path, animations=True, idle_timeout_s=0,
        )
    assert time.monotonic() - started < 1, "waited for timeout after launcher had exited"
    assert "exited" in str(raised.value)
    assert str(launcher) in raised.value.hint
    assert "darwin-x86_64/qemu-system-aarch64" in raised.value.hint
    assert "OLD FAILURE" not in raised.value.hint
    assert not (em._pid_dir(tmp_path) / f"{instance}.json").exists()
    assert not (em._reservation_dir() / "5556.port").exists()
