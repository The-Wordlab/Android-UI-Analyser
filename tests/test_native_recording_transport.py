"""Transport grammar only and fictional process state: no Android commands execute."""

import os
import subprocess
from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_recording
from test_native_recording_timeline import NativeTarget
from test_recording_lifecycle_recovery import NEW_BOOT, setup_recording
from test_recording_state_is_not_the_authority import _device


def test_native_launch_parses_after_real_uiautomator_adbutils_suffix(tmp_path, monkeypatch):
    from adbutils._device_base import BaseDevice
    from uiautomator2.base import _BaseClient

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    emitted = []
    adb = object.__new__(BaseDevice)

    def syntax_only(command, **kwargs):
        emitted.append(command)
        # -n parses stdin without executing ANY part of the emitted device command.
        checked = subprocess.run(["/bin/sh", "-n"], input=command, text=True,
                                 capture_output=True, timeout=5)
        assert checked.returncode == 0, checked.stderr
        assert command.endswith("; echo X4EXIT:$?")
        original = command.removesuffix("; echo X4EXIT:$?")
        # Only after real grammar validation, simulate transport output/start confirmation.
        return target.shell(original).encode() + b"X4EXIT:0"

    monkeypatch.setattr(adb, "shell", syntax_only)
    client = object.__new__(_BaseClient)
    client._dev = adb
    client._debug = False
    dev = _device(client)
    root = dev.recording_destination("/sdcard/fictional.mp4")
    assert dev.start_recording(root) == root
    launch = next(c for c in emitted if "nohup setsid sh " in c)
    assert f"nohup setsid sh {root}/supervisor.sh {root} 1800" in launch
    assert f"> {root}/supervisor.log 2>&1 < /dev/null &" in launch
    assert "; echo X4EXIT:$?" in launch
    assert dev.recording_metadata()["state"] == "recording"


@pytest.mark.parametrize("state", ["Z", "X"])
def test_empty_cmdline_is_ignored_only_with_authoritative_exited_stat(state):
    commands = []

    class Target:
        def shell(self, command):
            commands.append(command)
            if command.startswith("ps "):
                return "PID ARGS\n777 [screenrecord]\n"
            assert "/proc/$pid/stat" in command
            stat = f"777 (fictional (encoder)) {state} " + " ".join(["0"] * 49)
            return f"777 AUA_EMPTY {stat}\nAUA_CMDLINES_COMPLETE"

    assert android_recording._process_table(Target()) == []
    assert len(commands) == 2


@pytest.mark.parametrize("payload", [
    "", "AUA_EMPTY", "AUA_EMPTY unreadable", "AUA_EMPTY 777 (encoder) Z",
    "AUA_EMPTY 778 (encoder) Z " + "0 " * 49,
    "AUA_EMPTY 777 (encoder) R " + "0 " * 49,
    "AUA_EMPTY 777 (encoder) S " + "0 " * 49,
    "AUA_EMPTY 777 (encoder) Z " + "0 " * 48 + "invalid",
])
def test_empty_live_or_unknown_cmdline_remains_fail_closed(payload):
    class Target:
        def shell(self, command):
            if command.startswith("ps "):
                return "PID ARGS\n777 [sh]\n"
            return f"777 {payload}\nAUA_CMDLINES_COMPLETE"

    with pytest.raises(DeviceError) as error:
        android_recording._process_table(Target())
    assert error.value.code == "recording_status_unknown"


def test_legacy_pending_entry_keeps_cleanup_hint_after_boot_change(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.running = False
    dev._recording_state_path().unlink()
    target.boot = NEW_BOOT
    engine.record_device_change(key="screen_recording", kind="screen_recording", op="discard_recording",
                                args={"remote_path": "/sdcard/fictional-legacy.mp4"})
    # The ledger must actually belong to the preceding boot.
    from android_ui_analyser import device_ledger
    from test_recording_lifecycle_recovery import OLD_BOOT

    entry = engine._pending_device_change("screen_recording")
    device_ledger.record(dev.target_id, key=entry.key, kind=entry.kind, op=entry.op, args=entry.args,
                        instance_token=OLD_BOOT, platform=engine.platform.name,
                        platform_options_fingerprint=entry.platform_options_fingerprint)
    with pytest.raises(DeviceError) as error:
        engine.record_start()
    assert error.value.code == "recording_cleanup_pending"
    assert "teardown discard" in error.value.hint
    assert engine._pending_device_change("screen_recording").args == entry.args


@pytest.mark.skipif(os.environ.get('AUA_TEST_SYNTHETIC_LIFETIME') != '1',
                    reason='opt in to bounded synthetic host shell lifecycle test')
def test_launch_inherits_hup_protection_and_waits_for_actual_child(tmp_path, monkeypatch):
    import sys
    import time

    monkeypatch.setenv('AUA_CACHE__DIR', str(tmp_path / 'cache'))
    target = NativeTarget()
    dev = _device(target)
    root = dev.recording_destination(str(tmp_path / 'fictional'))
    dev.start_recording(root)
    launch = next(c for c in target.commands if 'nohup setsid sh ' in c)
    directory = Path(root)
    directory.mkdir()
    # Only these synthetic children execute. Model HUP arriving BEFORE nohup has
    # installed its own handler; pre-fork inherited protection must already hold.
    shim = tmp_path / 'nohup'
    shim.write_text('#!/bin/sh\nsleep 0.1\nkill -HUP $$\nexec "$@"\n')
    shim.chmod(0o700)
    # Portable host-only stand-in for Toybox setsid: perform the REAL session syscall.
    session = tmp_path / 'setsid'
    session.write_text(f'#!{sys.executable}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n')
    session.chmod(0o700)
    (directory / 'supervisor.sh').write_text(
        '#!/bin/sh\nsleep 0.2\necho $$ > "$1/supervisor.pid"\n'
        f"{sys.executable} -c 'import os; print(os.getsid(0))' > \"$1/session\"\n"
        'sleep 0.4\necho survived > "$1/heartbeat"\n'
    )
    launcher = tmp_path / 'lifetime-launcher.sh'
    launcher.write_text(launch + '; echo X4EXIT:$?\n')
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'])
    result = subprocess.run(['/bin/sh', str(launcher)], env=env, capture_output=True,
                            text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert (directory / 'supervisor.pid').is_file(), result.stdout
    assert 'AUA_LAUNCHED' in result.stdout
    assert not (directory / 'heartbeat').exists(), 'launcher waited for child completion'
    deadline = time.monotonic() + 2
    while not (directory / 'heartbeat').exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (directory / 'heartbeat').read_text().strip() == 'survived'
    assert int((directory / 'session').read_text()) != os.getsid(0)


def test_missing_parent_readiness_ack_keeps_owned_state(tmp_path, monkeypatch):
    monkeypatch.setenv('AUA_CACHE__DIR', str(tmp_path / 'cache'))
    target = NativeTarget()
    original = target.shell

    def no_ack(command):
        if 'nohup setsid sh ' in command:
            return 'AUA_LAUNCH_TIMEOUT'
        return original(command)

    target.shell = no_ack
    dev = _device(target)
    root = dev.recording_destination('/sdcard/fictional.mp4')
    with pytest.raises(DeviceError) as error:
        dev.start_recording(root)
    assert error.value.code == 'recording_start_unverified'
    assert 'launch readiness' in str(error.value)
    assert dev._recording_state_path().exists()
    assert not any(command.startswith('rm ') for command in target.commands)


def test_missing_native_session_detachment_fails_before_directory_creation(tmp_path, monkeypatch):
    monkeypatch.setenv('AUA_CACHE__DIR', str(tmp_path / 'cache'))
    target = NativeTarget()
    original = target.shell

    def no_setsid(command):
        if 'command -v setsid' in command:
            return ''
        return original(command)

    target.shell = no_setsid
    dev = _device(target)
    with pytest.raises(DeviceError) as error:
        dev.start_recording(dev.recording_destination('/sdcard/fictional.mp4'))
    assert error.value.code == 'recording_launch_unsupported'
    assert not any(command.startswith('umask') for command in target.commands)
