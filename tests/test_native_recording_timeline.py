"""Native segments retain elapsed media time and disclose every unverified interval."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from android_ui_analyser.platforms import android_recording


def _box(kind, payload=b""):
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def mp4(seconds):
    # Minimal container metadata only: this fixture is not playable native footage.
    mvhd = bytes(12) + (1000).to_bytes(4, "big") + int(seconds * 1000).to_bytes(4, "big")
    return _box(b"ftyp", b"isom") + _box(b"mdat", b"fictional") + _box(b"moov", _box(b"mvhd", mvhd))


def test_long_recording_rotates_and_validates_original_segment_durations(tmp_path):
    paths = []
    for i, seconds in enumerate([180, 180, 59]):
        path = tmp_path / f"segment-{i}.mp4"
        path.write_bytes(mp4(seconds))
        paths.append(path)
    events = "begin 0 100\nend 0 280 0\nbegin 1 280.4\nend 1 460.4 0\nbegin 2 460.9\nend 2 519.9 0\nfinish 519.9 stopped\n"
    report = android_recording.timeline(events, paths, stop_uptime_s=520)
    assert report["media_duration_s"] == 419
    assert report["requested_duration_s"] == 420
    assert len(report["segments"]) == 3
    assert len(report["gaps"]) >= 2
    assert report["continuous_coverage_verified"] is False
    assert report["gapless_guaranteed"] is False


@pytest.mark.parametrize("duration, finish", [(20, "finish 280 stopped\n"), (179, "")])
def test_short_media_and_supervisor_death_cannot_claim_complete_coverage(tmp_path, duration, finish):
    path = tmp_path / "segment-0.mp4"
    path.write_bytes(mp4(duration))
    report = android_recording.timeline("begin 0 100\nend 0 280 0\n" + finish, [path], stop_uptime_s=500)
    assert report["duration_check"] == "failed"
    assert report["continuous_coverage_verified"] is False
    assert report["gaps"]


def test_native_supervisor_uses_bounded_segments_without_sampling():
    script = android_recording.supervisor_script()
    assert "while" in script and "screenrecord --time-limit" in script
    assert "180" in script and "finish" in script and "begin" in script and "end" in script
    assert "screencap" not in script and "ffmpeg" not in script


def test_actual_supervisor_rotates_for_400_simulated_seconds(tmp_path):
    """Execute the production shell loop with a synthetic encoder and a fake boot clock.

    PATH contains only our fixture and host core utilities. No Android tools or device are
    involved; this verifies shell lifecycle behavior, not native encoding or visual timing.
    """
    root = tmp_path / "owned"
    root.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uptime = tmp_path / "uptime"
    uptime.write_text("100.0 0\n")
    media = tmp_path / "fixture.mp4"
    media.write_bytes(mp4(180))
    encoder = binaries / "screenrecord"
    encoder.write_text('''#!/bin/sh
set -eu
limit=$2
dest=$3
cp "$AUA_FIXTURE_MEDIA" "$dest"
read stamp rest < "$AUA_FIXTURE_UPTIME"
echo "$((${stamp%.*} + limit)).0 0" > "$AUA_FIXTURE_UPTIME"
echo "$limit" >> "$AUA_FIXTURE_LIMITS"
''')
    encoder.chmod(0o700)
    script = root / "supervisor.sh"
    script.write_text(android_recording.supervisor_script().replace("/proc/uptime", str(uptime)))
    env = {**os.environ, "PATH": f"{binaries}:/usr/bin:/bin",
           "AUA_FIXTURE_MEDIA": str(media), "AUA_FIXTURE_UPTIME": str(uptime),
           "AUA_FIXTURE_LIMITS": str(tmp_path / "limits")}
    result = subprocess.run(["/bin/sh", str(script), str(root), "400"],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "limits").read_text().splitlines() == ["180", "180", "40"]
    assert (root / "events").read_text().splitlines() == [
        "begin 0 100.0", "end 0 280.0 0", "begin 1 280.0", "end 1 460.0 0",
        "begin 2 460.0", "end 2 500.0 0", "finish 500.0 duration_limit",
    ]


class NativeTarget:
    """Only the Android transport is fake; exercise the actual runtime and supervisor seam."""
    def __init__(self):
        self.root = ""
        self.running = False
        self.stopped = False
        self.clock = 100.0
        self.commands = []
        self.corrupt = False
        self.identity = {}

    def shell(self, command):
        self.commands.append(command)
        if command == "cat /proc/sys/kernel/random/boot_id":
            return "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        if "command -v setsid" in command:
            return "AUA_SETSID"
        if command == "cat /proc/uptime":
            return f"{self.clock} 0"
        if command.startswith("# AUA_RECORDING_CMDLINES"):
            requested = command.split("for pid in ", 1)[1].split(";", 1)[0].split()
            table = dict(line.split(None, 1) for line in self.shell("ps -A -w -o PID,ARGS").splitlines()[1:] if line.strip())
            return "\n".join(f"{pid} {table.get(pid, 'AUA_GONE')}" for pid in requested) + "\nAUA_CMDLINES_COMPLETE"
        if command.startswith("ps "):
            head = "PID ARGS\n" if "PID" in command else "ARGS\n"
            if self.running:
                return head + f"101 sh {self.root}/supervisor.sh {self.root} 1800\n102 screenrecord --time-limit 180 {self.root}/segment-2.mp4\n"
            return head
        if command.startswith("umask"):
            self.root = command.split("mkdir ", 1)[1].split(" &&", 1)[0]
            tokens = shlex.split(command)
            self.identity = json.loads(tokens[tokens.index("printf") + 2])
            return "AUA_CREATED"
        if command.startswith(("nohup", "(nohup", "(trap")):
            self.running = True
            return "AUA_LAUNCHED"
        if command.startswith("printf %s") and command.endswith("/identity.json"):
            self.identity = json.loads(shlex.split(command)[2])
        if command.startswith("cat ") and command.endswith("/identity.json"):
            return json.dumps(self.identity)
        if command.startswith("cat ") and "/events" in command:
            if not self.stopped:
                return "begin 0 100\n"
            return "begin 0 100\nend 0 280 0\nbegin 1 280.4\nend 1 460.4 0\nbegin 2 460.9\nend 2 519.9 0\nfinish 519.9 stopped\n"
        if command.startswith("ls -l"):
            return command.split()[2]
        if "kill -2 102" in command:
            self.running = False
            self.stopped = True
        if command.endswith("|| echo AUA_REMOVED"):
            return "AUA_REMOVED"
        return ""

    def pull(self, remote, local):
        index = int(Path(remote).stem.removeprefix("segment-"))
        Path(local).write_bytes(b"corrupt" if self.corrupt else mp4([180, 180, 59][index]))


def test_actual_runtime_continues_beyond_native_limit_and_exports_original_segments(tmp_path, monkeypatch):
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    remote = device.recording_destination("/sdcard/fictional.mp4")
    assert device.start_recording(remote) == remote
    assert device.recording_metadata()["max_duration_s"] == 1800
    assert any("nohup setsid sh" in command for command in target.commands)
    # The next CLI invocation has no in-memory recording handle.
    device = _device(target)
    target.clock = 520
    monkeypatch.setattr(android_recording, "export_mp4", _fake_export, raising=False)
    saved = device.stop_recording(str(tmp_path / "journey.mp4"))
    assert saved == str(tmp_path / "journey.mp4")
    assert Path(saved).read_bytes() == mp4(419)
    report = json.loads(Path(saved + ".recording.json").read_text())
    assert len(report["segments"]) == 3
    assert report["requested_duration_s"] == 420
    assert report["media_duration_s"] == 419
    assert not device._recording_state_path().exists()
    for i, seconds in enumerate([180, 180, 59]):
        assert (Path(report["segment_directory"]) / f"segment-{i}.mp4").read_bytes() == mp4(seconds)
    assert any(command.startswith("rm -rf " + remote) for command in target.commands)


def test_corrupt_segment_retains_remote_and_stop_time_for_retry(tmp_path, monkeypatch):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    device.start_recording(device.recording_destination())
    target.clock = 520
    target.corrupt = True
    with pytest.raises(DeviceError, match="not a finalized MP4"):
        device.stop_recording(str(tmp_path / "journey.mp4"))
    assert not any(command.startswith("rm -rf") for command in target.commands)
    assert device._recording_state_path().exists()
    target.clock = 600
    target.corrupt = False
    monkeypatch.setattr(android_recording, "export_mp4", _fake_export, raising=False)
    saved = device.stop_recording(str(tmp_path / "journey.mp4"))
    assert json.loads(Path(saved + ".recording.json").read_text())["requested_duration_s"] == 420


def test_changed_boot_refuses_recovery_before_stop_or_delete(tmp_path, monkeypatch):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    device.start_recording(device.recording_destination())
    target.commands.clear()
    monkeypatch.setattr(device, "instance_token", lambda: "different-boot")
    with pytest.raises(DeviceError, match="another boot"):
        device.stop_recording(str(tmp_path / "journey.mp4"))
    assert not target.commands


def test_unfinished_supervisor_keeps_ownership_on_cleanup_timeout(tmp_path, monkeypatch):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    remote = device.start_recording(device.recording_destination())
    original_shell = target.shell

    def stubborn(command):
        if "kill -2 " in command:
            return ""  # signal never takes effect
        return original_shell(command)

    monkeypatch.setattr(target, "shell", stubborn)
    monkeypatch.setattr(android_recording, "_FINALIZE_S", 0.01)
    with pytest.raises(DeviceError, match="did not finalize"):
        device.discard_recording(remote)
    assert device._recording_state_path().exists()
    assert not any(command.startswith("rm -rf") for command in target.commands)


def test_dead_supervisor_with_live_encoder_is_cleaned_without_foreign_signals(tmp_path, monkeypatch):
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    remote = device.start_recording(device.recording_destination())
    original_shell = target.shell

    def dead_supervisor(command):
        result = original_shell(command)
        if command.startswith("ps "):
            result = "\n".join(line for line in result.splitlines() if "supervisor.sh" not in line)
            result += "\n777 screenrecord --time-limit 180 /sdcard/foreign.mp4\n"
        return result

    monkeypatch.setattr(target, "shell", dead_supervisor)
    device.discard_recording(remote)
    assert not device._recording_state_path().exists()
    assert not any("kill -2 777" in command for command in target.commands)


@pytest.mark.parametrize("events", [
    "begin 0 nan\n", "begin 0 10\nbegin 0 20\n", "begin 0 10\nend 0 9 0\n",
    "begin 0 10\nbegin 1 11\n", "begin 0 10\nend 0 20 0\nfinish 19 stopped\n",
])
def test_invalid_lifecycle_cannot_become_coverage(events):
    from android_ui_analyser.errors import DeviceError

    with pytest.raises(DeviceError, match="lifecycle log"):
        android_recording.timeline(events, [], stop_uptime_s=30)


def test_another_cache_recovers_a_surviving_encoder_after_supervisor_death(tmp_path, monkeypatch):
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "first-cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.start_recording(device.recording_destination())
    original_shell = target.shell

    def only_encoder(command):
        result = original_shell(command)
        if command.startswith("ps "):
            return "\n".join(line for line in result.splitlines() if "supervisor.sh" not in line)
        return result

    monkeypatch.setattr(target, "shell", only_encoder)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "second-cache"))
    recovered = _device(target)
    assert recovered.active_recording() == root
    assert recovered.recording_metadata()["mode"] == "native_segments"
    recovered.discard_recording(root)
    assert not recovered._recording_state_path().exists()


@pytest.mark.parametrize("foreign", [False, True])
def test_failed_mkdir_never_claims_existing_directory_and_absent_undo_is_safe(tmp_path, monkeypatch, foreign):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.recording_destination()
    original = target.shell

    def failed_mkdir(command):
        if command.startswith("umask"):
            return "AUA_NOT_CREATED"
        if "echo AUA_ABSENT" in command:
            return "" if foreign else "AUA_ABSENT"
        return original(command)

    monkeypatch.setattr(target, "shell", failed_mkdir)
    with pytest.raises(DeviceError, match="cannot be created"):
        device.start_recording(root)
    target.commands.clear()
    if foreign:
        with pytest.raises(DeviceError, match="ownership"):
            device.discard_recording(root)
        assert device._recording_state_path().exists()
    else:
        device.discard_recording(root)
        assert not device._recording_state_path().exists()
    assert not any(c.startswith(("touch ", "rm -rf")) or "kill -2" in c for c in target.commands)


def _fake_export(paths, destination):
    """Transport fixtures only contain MP4 headers; real codec export is tested separately."""
    destination.write_bytes(mp4(sum(android_recording.media_duration(p) for p in paths)))
    return "bitstream_copy_concat"


@pytest.mark.parametrize("part", ["", ".segments", ".recording.json"])
def test_stop_existing_output_is_rejected_before_device_side_effects(tmp_path, monkeypatch, part):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.start_recording(device.recording_destination())
    existing = tmp_path / ("journey.mp4" + part)
    existing.write_bytes(b"existing evidence")
    target.commands.clear()
    with pytest.raises(DeviceError, match="already exists"):
        device.stop_recording(str(tmp_path / "journey.mp4"))
    assert existing.read_bytes() == b"existing evidence"
    assert not any(c.startswith(("touch ", "rm -rf")) or "kill -2" in c for c in target.commands)
    assert device.active_recording() == root


def test_failed_export_preserves_remote_evidence_and_can_retry_same_path(tmp_path, monkeypatch):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.start_recording(device.recording_destination())
    target.clock = 520
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(DeviceError) as exc:
        device.stop_recording(str(tmp_path / "journey.mp4"))
    assert exc.value.code == "recording_export_unsupported"
    assert device.active_recording() == root
    assert not any(c.startswith("rm -rf") for c in target.commands)
    assert not (tmp_path / "journey.mp4").exists()
    monkeypatch.setattr(android_recording, "export_mp4", _fake_export)
    target.clock = 650
    saved = device.stop_recording(str(tmp_path / "journey.mp4"))
    assert saved == str(tmp_path / "journey.mp4")
    report = json.loads(Path(saved + ".recording.json").read_text())
    assert report["requested_duration_s"] == 420
    assert report["export"]["covers_wall_clock_gaps"] is False
    assert report["continuous_coverage_verified"] is False


def test_output_created_during_export_is_not_overwritten(tmp_path, monkeypatch):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.start_recording(device.recording_destination())
    target.clock = 520
    destination = tmp_path / "journey.mp4"

    def racing_export(paths, staged):
        destination.write_bytes(b"other writer's evidence")
        return _fake_export(paths, staged)

    monkeypatch.setattr(android_recording, "export_mp4", racing_export)
    with pytest.raises(DeviceError, match="publication failed"):
        device.stop_recording(str(destination))
    assert destination.read_bytes() == b"other writer's evidence"
    assert device.active_recording() == root
    assert not any(c.startswith("rm -rf") for c in target.commands)
    assert len(list((tmp_path / "journey.mp4.segments").glob("*.mp4"))) == 3


def test_cleanup_failure_keeps_export_and_remote_retryable_at_new_destination(tmp_path, monkeypatch):
    from android_ui_analyser.errors import DeviceError
    from test_recording_state_is_not_the_authority import _device

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = NativeTarget()
    device = _device(target)
    root = device.start_recording(device.recording_destination())
    target.clock = 520
    original = target.shell

    def failed_remove(command):
        if command.endswith("|| echo AUA_REMOVED"):
            return ""
        return original(command)

    monkeypatch.setattr(target, "shell", failed_remove)
    monkeypatch.setattr(android_recording, "export_mp4", _fake_export)
    with pytest.raises(DeviceError, match="artifacts remain"):
        device.stop_recording(str(tmp_path / "journey.mp4"))
    assert device.active_recording() == root
    assert (tmp_path / "journey.mp4").read_bytes() == mp4(419)
    monkeypatch.setattr(target, "shell", original)
    target.clock = 650
    saved = device.stop_recording(str(tmp_path / "retry.mp4"))
    assert json.loads(Path(saved + ".recording.json").read_text())["requested_duration_s"] == 420
    assert (tmp_path / "journey.mp4").read_bytes() == mp4(419)
    assert not device._recording_state_path().exists()
