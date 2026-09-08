"""Actual runtime/engine seams with fictional transports; never connect to Android."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli, device_ledger
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.mcp_server import _dispatch
from android_ui_analyser.platforms import android_recording
from conftest import make_config
from test_native_recording_timeline import NativeTarget, mp4
from test_recording_state_is_not_the_authority import _device
from test_recording_uses_selected_platform import Platform

OLD_BOOT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
NEW_BOOT = "11111111-2222-3333-4444-555555555555"


class LifecycleTarget(NativeTarget):
    def __init__(self):
        super().__init__()
        self.boot = OLD_BOOT
        self.identities = {}
        self.scan_error = False
        self.proc_error = False
        self.crashed = False
        self.tail = "corrupt"
        self.foreign = None

    def process_lines(self):
        if self.running:
            return {101: f"sh {self.root}/supervisor.sh {self.root} 1800",
                    102: f"screenrecord --time-limit 180 {self.root}/segment-2.mp4"}
        if self.foreign:
            return {777: f"screenrecord --time-limit 180 {self.foreign}/segment-0.mp4"}
        return {}

    def shell(self, command):
        if command == "cat /proc/sys/kernel/random/boot_id":
            return self.boot
        if command.startswith("ps "):
            self.commands.append(command)
            if self.scan_error:
                return "permission denied"
            lines = self.process_lines()
            return "PID ARGS\n" + "\n".join(f"{p} {c}" for p, c in lines.items())
        if command.startswith("# AUA_RECORDING_CMDLINES"):
            self.commands.append(command)
            if self.proc_error:
                return "AUA_UNKNOWN\n"
            return "\n".join(f"{p} {c}" for p, c in self.process_lines().items()) + "\nAUA_CMDLINES_COMPLETE\n"
        if "echo AUA_ABSENT" in command:
            self.commands.append(command)
            return "" if any(root in command for root in self.identities) else "AUA_ABSENT"
        if command.startswith("cat ") and command.endswith("/identity.json"):
            self.commands.append(command)
            root = command[4:].removesuffix("/identity.json")
            return json.dumps(self.identities.get(root, {}))
        if "AUA_FILE_PRESENT" in command:
            self.commands.append(command)
            if self.tail == "permission" and "segment-1.mp4" in command:
                return "AUA_FILE_UNKNOWN"
            if self.tail == "missing" and "segment-1.mp4" in command:
                return "AUA_FILE_MISSING"
            return "AUA_FILE_PRESENT"
        if command.startswith("cat ") and "/events" in command and self.crashed:
            return "begin 0 100\nend 0 280 0\nbegin 1 280.4\nend 1 281 1\nfinish 281 encoder_failed\n"
        result = super().shell(command)
        if command.startswith("umask"):
            self.identities[self.root] = dict(self.identity)
        return result

    def pull(self, remote, local):
        if self.crashed and remote.endswith("segment-1.mp4"):
            if self.tail in {"missing", "permission", "transport"}:
                raise RuntimeError("pull failed")
            Path(local).write_bytes(b"partial original bytes")
        elif self.crashed and self.tail == "all_corrupt":
            Path(local).write_bytes(b"partial first bytes")
        else:
            super().pull(remote, local)


def setup_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = LifecycleTarget()
    dev = _device(target)
    cfg = make_config(lease={"enabled": False}, memory={"enabled": False}, teardown={"watchdog": False})
    platform = Platform(cfg)
    platform.capabilities = platform.capabilities | {"device.recording.recovery"}
    engine = Engine(cfg, platform=platform, device=dev)
    root = engine.record_start().detail
    return target, dev, engine, root


@pytest.mark.parametrize("recycled", [False, True])
@pytest.mark.parametrize("cache_missing", [False, True])
def test_engine_start_after_new_boot_preserves_old_evidence_and_undo(tmp_path, monkeypatch, recycled, cache_missing):
    target, dev, engine, old = setup_recording(tmp_path, monkeypatch)
    original = json.loads(dev._recording_state_path().read_text())
    identity = dict(target.identities[old])
    if cache_missing:
        dev._recording_state_path().unlink()
    target.running = False
    target.boot = NEW_BOOT
    if recycled:
        target.identities.clear()
    target.commands.clear()
    fresh = engine.record_start()
    assert fresh.ok and fresh.detail != old
    archives = list((dev._recording_state_path().parent / "quarantine").glob("*.json"))
    assert any(json.loads(p.read_text())["recording"]["remote"] == original["remote"] for p in archives)
    entries = device_ledger.read_ledger(dev.target_id, platform="strict-fake")
    assert any(e.key != "screen_recording" and e.args["remote_path"] == old and e.instance_token == OLD_BOOT for e in entries)
    assert any(e.key == "screen_recording" and e.instance_token == NEW_BOOT for e in entries)
    assert not any(c.startswith(("rm ", "touch ", "actual=")) for c in target.commands)
    if not recycled:
        assert target.identities[old] == identity
    target.commands.clear()
    retained = [e for e in entries if e.instance_token == OLD_BOOT]
    undo = device_ledger.replay(
        dev.target_id, platform="strict-fake", entries=retained,
        context=device_ledger.UndoContext(
            serial=dev.target_id, platform="strict-fake", device=dev, instance_token=NEW_BOOT,
            runtime_capability=engine.platform.runtime_capability,
        ),
    )
    assert undo["failed"] and undo["remaining"] == 2
    assert not target.commands  # old-boot undo cannot delete persistent prior footage


@pytest.mark.parametrize("fault", ["boot_unknown", "identity", "scan", "proc", "same_root_live", "foreign"])
def test_new_boot_recovery_fails_closed_without_touching_current_recorders(tmp_path, monkeypatch, fault):
    target, dev, engine, old = setup_recording(tmp_path, monkeypatch)
    target.boot = NEW_BOOT
    target.running = False
    if fault == "boot_unknown":
        target.boot = ""
    elif fault == "identity":
        target.identities[old] = {"mode": "native_segments", "remote": old, "boot_id": "invalid"}
    elif fault == "scan":
        target.scan_error = True
    elif fault == "proc":
        target.foreign = android_recording.destination(None)
        target.proc_error = True
    elif fault == "same_root_live":
        target.running = True
    elif fault == "foreign":
        target.foreign = android_recording.destination(None)
        target.identities[target.foreign] = {"mode": "native_segments", "remote": target.foreign, "boot_id": NEW_BOOT}
    target.commands.clear()
    with pytest.raises(DeviceError):
        engine.record_start()
    assert not any(c.startswith(("rm ", "touch ", "umask", "nohup")) or "kill -2" in c for c in target.commands)
    assert any(e.args["remote_path"] == old for e in device_ledger.read_ledger(dev.target_id, platform="strict-fake"))


def test_scan_uses_wide_ps_and_authoritative_long_cmdlines(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    full = dict(target.process_lines())
    original = target.shell

    def truncated(command):
        if command.startswith("ps "):
            target.commands.append(command)
            return "PID ARGS\n" + "\n".join(f"{p} {c}"[:80] for p, c in full.items())
        return original(command)

    monkeypatch.setattr(target, "shell", truncated)
    target.commands.clear()
    assert len(android_recording._processes(dev, root)) == 2
    assert "ps -A -w -o PID,ARGS" in target.commands
    assert sum(c.startswith("# AUA_RECORDING_CMDLINES") for c in target.commands) == 1
    target.proc_error = True
    with pytest.raises(DeviceError):
        dev.discard_recording(root)
    assert not any(c.startswith("rm ") for c in target.commands)


def test_cross_cache_invalid_remote_json_is_structured(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "other-cache"))
    original = target.shell
    monkeypatch.setattr(target, "shell", lambda c: "not-json" if c.endswith("/identity.json") else original(c))
    with pytest.raises(DeviceError) as error:
        _device(target).active_recording()
    assert error.value.code == "recording_identity_mismatch"


@pytest.mark.parametrize("transport", ["engine", "cli", "mcp"])
@pytest.mark.parametrize("tail", ["corrupt", "missing"])
def test_crashed_encoder_salvages_mp4_and_reports_failed_coverage(tmp_path, monkeypatch, transport, tail):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.running = False
    target.crashed = True
    target.tail = tail
    target.clock = 420
    destination = tmp_path / "journey.mp4"
    if transport == "engine":
        result = engine.record_stop(str(destination)).model_dump()
    elif transport == "mcp":
        result = _dispatch(engine, "screen_record_stop", {"path": str(destination)})
    else:
        monkeypatch.setattr(cli, "_route", lambda _e, operation, **kw: getattr(engine, operation)(**kw))
        response = CliRunner().invoke(cli.app, ["record", "stop", str(destination)])
        assert response.exit_code != 0
        result = json.loads(response.stdout)
    assert result["detail"] == str(destination) and result["ok"] is False
    assert destination.read_bytes() == mp4(180)
    report = json.loads(Path(str(destination) + ".recording.json").read_text())
    assert report["duration_check"] == "failed" and report["continuous_coverage_verified"] is False
    assert report["segments"][1]["status"] == tail
    assert report["segments"][1]["exported"] is False
    assert report["cleanup_pending"] is True
    assert device_ledger.read_ledger(dev.target_id, platform="strict-fake")
    if tail == "corrupt":
        assert (Path(str(destination) + ".segments") / "segment-1.mp4").read_bytes() == b"partial original bytes"
    assert not any(c.startswith("rm ") for c in target.commands)


@pytest.mark.parametrize("tail", ["permission", "transport", "all_corrupt"])
def test_no_playable_or_unreadable_evidence_never_becomes_success(tmp_path, monkeypatch, tail):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.running = False
    target.crashed = True
    target.tail = tail
    target.clock = 420
    destination = tmp_path / "journey.mp4"
    with pytest.raises(DeviceError):
        engine.record_stop(str(destination))
    assert not destination.exists()
    assert device_ledger.read_ledger(dev.target_id, platform="strict-fake")
    assert dev._recording_state_path().exists()
    assert not any(c.startswith("rm ") for c in target.commands)
    if tail == "all_corrupt":
        report = json.loads(Path(str(destination) + ".recording.json").read_text())
        assert report["duration_check"] == "failed"
        assert (Path(str(destination) + ".segments") / "segment-0.mp4").read_bytes() == b"partial first bytes"


def test_salvage_retry_and_deliberate_cleanup_keep_partial_originals(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.running = False
    target.crashed = True
    target.clock = 420
    first = engine.record_stop(str(tmp_path / "first.mp4"))
    assert not first.ok
    target.clock = 600
    second = engine.record_stop(str(tmp_path / "second.mp4"))
    assert second.recording["requested_duration_s"] == first.recording["requested_duration_s"]
    assert not second.ok and second.recording["cleanup_pending"]
    cleanup = device_ledger.replay(
        dev.target_id, platform="strict-fake", context=device_ledger.UndoContext(
            serial=dev.target_id, platform="strict-fake", device=dev, instance_token=target.boot,
            runtime_capability=engine.platform.runtime_capability,
        ),
    )
    assert not cleanup["failed"] and cleanup["remaining"] == 0
    assert not dev._recording_state_path().exists()
    assert (tmp_path / "first.mp4.segments" / "segment-1.mp4").read_bytes() == b"partial original bytes"
    assert (tmp_path / "second.mp4.segments" / "segment-1.mp4").read_bytes() == b"partial original bytes"


@pytest.mark.parametrize("operation", ["start", "discard"])
def test_matching_local_boot_does_not_authorize_foreign_remote_identity(tmp_path, monkeypatch, operation):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.identities[root]["boot_id"] = NEW_BOOT
    target.commands.clear()
    with pytest.raises(DeviceError):
        engine.record_start() if operation == "start" else dev.discard_recording(root)
    assert not any(c.startswith(("rm ", "touch ", "actual=", "umask")) for c in target.commands)
    assert dev._recording_state_path().exists()


def test_changed_ledger_entry_is_not_rekeyed_after_recovery(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    expected = engine._pending_device_change("screen_recording")
    engine.record_device_change(key="screen_recording", kind="screen_recording", op="discard_recording",
                                args={"remote_path": android_recording.destination(None)})
    replacement = engine._pending_device_change("screen_recording")
    with pytest.raises(Exception, match="changed during recovery"):
        device_ledger.retain_stale_recording(
            dev.target_id, expected, archive_path=str(tmp_path / "quarantine.json"),
            current_instance=NEW_BOOT, registry_dir=engine.config.lease.registry_dir, platform="strict-fake",
        )
    assert engine._pending_device_change("screen_recording") == replacement


def test_absence_probe_requires_enoent_not_just_failed_file_test(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    original = target.shell

    def check_probe(command):
        if "AUA_FILE_PRESENT" in command:
            assert "LC_ALL=C ls -ld" in command and "No such file or directory" in command
            return "AUA_FILE_UNKNOWN"  # permission/transport ambiguity, not ENOENT
        return original(command)

    monkeypatch.setattr(target, "shell", check_probe)
    with pytest.raises(DeviceError, match="inspection failed"):
        android_recording._file_status(dev, root + "/segment-1.mp4")


def test_dead_supervisor_salvages_partial_last_segment_without_inventing_completion(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.running = False
    target.crashed = True
    target.clock = 420
    original = target.shell

    def unfinished(command):
        if command.startswith("cat ") and "/events" in command:
            return "begin 0 100\nend 0 280 0\nbegin 1 280.4\n"
        return original(command)

    monkeypatch.setattr(target, "shell", unfinished)
    result = engine.record_stop(str(tmp_path / "journey.mp4"))
    assert not result.ok and result.recording["finish"] is None
    assert result.recording["cleanup_pending"]
    assert (tmp_path / "journey.mp4.segments" / "segment-1.mp4").read_bytes() == b"partial original bytes"


def test_invalid_boot_id_cannot_start_a_new_native_recorder(tmp_path, monkeypatch):
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    target = LifecycleTarget()
    target.boot = "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz"
    dev = _device(target)
    with pytest.raises(DeviceError):
        dev.start_recording(dev.recording_destination())
    assert not any(c.startswith(("umask", "nohup", "printf")) for c in target.commands)
    assert not dev._recording_state_path().exists()


@pytest.mark.parametrize("recycled", [False, True])
def test_stale_boot_start_under_real_cli_command_lock(tmp_path, monkeypatch, recycled):
    from android_ui_analyser import leases
    from android_ui_analyser.platforms.identity import TargetRef

    target, dev, engine, old = setup_recording(tmp_path, monkeypatch)
    ref = TargetRef("strict-fake", dev.target_id)
    original = device_ledger.read_ledger(ref)[0]
    target.running = False
    target.boot = NEW_BOOT
    if recycled:
        target.identities.clear()
    with leases.device_command(engine.config.lease.registry_dir, ref):
        fresh = engine.record_start()
        # Metadata recovery must not weaken the ownership-transition prohibition.
        with (
            pytest.raises(RuntimeError, match="cannot be upgraded"),
            leases.device_transaction(engine.config.lease.registry_dir, ref),
        ):
            pytest.fail("shared command lock was upgraded")
    assert fresh.ok and fresh.detail != old
    retained = next(e for e in device_ledger.read_ledger(ref) if e.key.startswith("screen_recording:prior:"))
    assert retained.args == original.args
    assert retained.instance_token == original.instance_token == OLD_BOOT
    assert (retained.owner, retained.owner_pid, retained.owner_started) == (
        original.owner, original.owner_pid, original.owner_started,
    )
