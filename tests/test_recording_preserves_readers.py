"""Readers of owned footage are not recorder processes or evidence of a live recorder."""

import pytest

from android_ui_analyser.platforms import android_recording
from test_native_recording_timeline import NativeTarget
from test_recording_state_is_not_the_authority import _device


def _keep_readers_running(monkeypatch, target):
    original_shell = target.shell
    readers = {
        201: f"/system/bin/cat {target.root}/segment-2.mp4",
        202: f"/system/bin/tail -f {target.root}/segment-2.mp4",
        203: f"sh -c cat {target.root}/segment-2.mp4",
        204: f"/system/bin/sh -c cat {target.root}/supervisor.sh",
    }

    def shell(command):
        result = original_shell(command)
        if command.startswith("ps "):
            result += "".join(f"{pid} {argv}\n" for pid, argv in readers.items())
        return result

    monkeypatch.setattr(target, "shell", shell)
    return readers


def test_discard_stops_encoders_without_signalling_or_waiting_for_readers(tmp_path, monkeypatch):
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    remote = device.start_recording(device.recording_destination())
    readers = _keep_readers_running(monkeypatch, target)
    monkeypatch.setattr(android_recording, "_FINALIZE_S", 0.01)

    device.discard_recording(remote)

    assert any("kill -2 102" in command for command in target.commands)
    assert not any(f"kill -2 {pid}" in command for pid in readers for command in target.commands)
    assert not device._recording_state_path().exists()
    assert any(command.startswith("rm -rf ") for command in target.commands)


@pytest.mark.parametrize("supervisor", [False, True])
def test_cache_miss_recovers_only_a_real_supervisor(tmp_path, monkeypatch, supervisor):
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.start_recording(device.recording_destination())
    device._recording_state_path().unlink()
    target.running = False
    _keep_readers_running(monkeypatch, target)
    readers_shell = target.shell

    def shell(command):
        result = readers_shell(command)
        if supervisor and command.startswith("ps "):
            result += f"101 /system/bin/sh {root}/supervisor.sh {root} 1800\n"
        return result

    monkeypatch.setattr(target, "shell", shell)
    recovered = _device(target).active_recording()

    assert recovered == (root if supervisor else None)
    assert device._recording_state_path().exists() is supervisor
