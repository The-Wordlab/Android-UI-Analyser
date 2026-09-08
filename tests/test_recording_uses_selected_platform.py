"""CLI and MCP share the engine's optional recording timeline contract and ledger."""

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UnsupportedPlatformCapabilityError
from android_ui_analyser.mcp_server import _dispatch
from android_ui_analyser.platforms.runtime import TargetRuntime
from conftest import make_config
from test_virtual_targets import _FakePlatform


class Runtime(TargetRuntime):
    target_id = "fictional-video-target"

    def instance_token(self):
        return "fictional-boot"

    def recording_destination(self, requested=None):
        return "owned-native-recording"

    def active_recording(self):
        return None

    def start_recording(self, remote_path):
        entries = device_ledger.read_ledger(self.target_id, platform="strict-fake")
        assert any(e.kind == "screen_recording" for e in entries), "start must have a durable undo"
        return remote_path

    def stop_recording(self, local_path):
        return local_path

    def discard_recording(self, remote_path):
        pass

    def recording_metadata(self):
        return {"duration_check": "failed", "continuous_coverage_verified": False,
                "segments": [], "gaps": [{"reason": "encoder_exited"}]}


class Platform(_FakePlatform):
    capabilities = frozenset({"device.recording", "device.recording.timeline"})


@pytest.mark.parametrize("transport", ["engine", "mcp"])
def test_recording_reports_failed_coverage_and_uses_no_android_tools(tmp_path, monkeypatch, transport):
    from android_ui_analyser.platforms import android_recording
    monkeypatch.setattr(android_recording, "start", lambda *a, **kw: pytest.fail("Android bypass"))
    cfg = make_config(lease={"enabled": False}, memory={"enabled": False})
    runtime = Runtime()
    engine = Engine(cfg, platform=Platform(cfg), device=runtime)
    if transport == "engine":
        start = engine.record_start().model_dump()
        stop = engine.record_stop(str(tmp_path / "journey.mp4")).model_dump()
    else:
        start = _dispatch(engine, "screen_record_start", {})
        stop = _dispatch(engine, "screen_record_stop", {"path": str(tmp_path / "journey.mp4")})
    assert start["ok"] is True
    assert stop["ok"] is False
    assert stop["detail"] == str(tmp_path / "journey.mp4")
    assert stop["recording"]["duration_check"] == "failed"
    assert not device_ledger.read_ledger(runtime.target_id, platform="strict-fake")


def test_unsupported_platform_fails_explicitly_before_recording(tmp_path):
    cfg = make_config(lease={"enabled": False})
    engine = Engine(cfg, platform=_FakePlatform(cfg), device=Runtime())
    with pytest.raises(UnsupportedPlatformCapabilityError):
        engine.record_start()


def test_stale_recording_ledger_recovery_uses_selected_adapter(tmp_path, monkeypatch):
    from android_ui_analyser.platforms import android_recording

    class RecoveringRuntime(Runtime):
        boot = "old-fictional-boot"

        def instance_token(self):
            return self.boot

        def archive_stale_recording(self, remote_path, instance_token):
            assert remote_path == "owned-native-recording"
            assert instance_token == "old-fictional-boot"
            path = tmp_path / "retained-metadata.json"
            path.write_text("fictional retained metadata")
            return str(path)

    class RecoveringPlatform(Platform):
        capabilities = Platform.capabilities | {"device.recording.recovery"}

    monkeypatch.setattr(android_recording, "archive_stale", lambda *a: pytest.fail("Android bypass"))
    cfg = make_config(lease={"enabled": False}, memory={"enabled": False})
    runtime = RecoveringRuntime()
    engine = Engine(cfg, platform=RecoveringPlatform(cfg), device=runtime)
    engine.record_start()
    runtime.boot = "new-fictional-boot"
    assert engine.record_start().ok
    entries = device_ledger.read_ledger(runtime.target_id, platform="strict-fake")
    assert len(entries) == 2
    assert {e.instance_token for e in entries} == {"old-fictional-boot", "new-fictional-boot"}
