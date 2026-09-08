"""Only a current consuming input can admit an emulator microphone stream."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import mic
from android_ui_analyser.cli import _daemon_error
from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.services import CAPABILITY_METHODS
from android_ui_analyser.schema import ShellResult
from conftest import FakeDevice, make_config
from test_mic import HOLD_XML, _prepared
from test_platforms import _InjectedPlatform

ACTIVE = """Input thread 0x100, name AudioIn_Example, tid 123, type 3 (RECORD):
  Standby: no
  Input device: 0x80000004 (AUDIO_DEVICE_IN_BUILTIN_MIC)
  Frames read: 2048
  Hw silenced: no
  1 Tracks of which 1 are active
    Active Id Client(pid/uid)
       yes 1 123/10001
  Local log:
"""


class Clock:
    now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(mic, "time", value)
    return value


class DiagnosticsDevice(FakeDevice):
    def __init__(self, snapshots: list[str], **kwargs: Any) -> None:
        super().__init__(hierarchy_xml=HOLD_XML, serial="emulator-5554", **kwargs)
        self.snapshots = snapshots
        self.diagnostic_timeouts: list[float] = []
        self.events: list[str] = []

    def run_read_only_shell(self, argv: list[str], *, timeout_s: float = 30.0) -> ShellResult:
        assert argv == ["dumpsys", "media.audio_flinger"]
        self.events.append("readiness")
        self.diagnostic_timeouts.append(timeout_s)
        output = self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]
        return ShellResult(
            ok=True, serial=self.serial, argv=argv, stdout=output, exit_code=0, duration_ms=0
        )

    def touch_down(self, x: int, y: int) -> None:
        self.events.append("down")

    def touch_up(self, x: int, y: int) -> None:
        self.events.append("up")


@pytest.mark.parametrize(
    "dump",
    [
        "",
        ACTIVE.replace("Standby: no", "Standby: yes"),
        ACTIVE.replace("Frames read: 2048", "Frames read: 0"),
        ACTIVE.replace("0x80000004", "0"),
        ACTIVE.replace("Hw silenced: no", "Hw silenced: yes"),
        ACTIVE.replace("1 Tracks of which 1 are active", "0 Tracks"),
        ACTIVE.replace("1 Tracks of which 1 are active", "1 Tracks of which 0 are active"),
        "Historical Thread Log yesterday\n" + ACTIVE,
        "\n".join("- " + line for line in ACTIVE.splitlines()),
        ACTIVE.replace("Input thread", "Output thread"),
    ],
)
def test_inactive_and_historical_threads_do_not_admit_injection(dump: str) -> None:
    assert mic._active_recording_inputs(dump) == 0


def test_current_input_is_not_confused_with_historical_tracks() -> None:
    assert mic._active_recording_inputs(ACTIVE + "Historical Thread Log\n" + ACTIVE) == 1
    inactive = ACTIVE.replace("1 Tracks of which 1 are active", "0 Tracks")
    assert mic._active_recording_inputs(inactive + "Historical Thread Log\n" + ACTIVE) == 0
    assert mic._active_recording_inputs(inactive + ACTIVE) == 1


def test_poll_waits_for_current_recording_and_reports_only_bounded_provenance(clock: Clock) -> None:
    device = DiagnosticsDevice(["", ACTIVE])
    result = mic.wait_for_recording(device, timeout_ms=500, poll_ms=100)
    assert result == {
        "ready": True,
        "source": "android.audio_flinger",
        "phase": "before_injection",
        "status": "active",
        "active_inputs": 1,
        "checks": 2,
        "elapsed_ms": 100,
        "timeout_ms": 500,
        "input_sent": False,
    }
    assert device.diagnostic_timeouts == [0.5, 0.4]
    assert "10001" not in str(result)


def test_poll_stops_at_deadline_without_injecting(clock: Clock) -> None:
    device = DiagnosticsDevice([""])
    with pytest.raises(mic.MicRecordingNotReadyError) as caught:
        mic.wait_for_recording(device, timeout_ms=250, poll_ms=100)
    result = caught.value.to_dict()["error"]["recording_readiness"]
    assert result["ready"] is False
    assert result["checks"] == 3
    assert result["elapsed_ms"] == 250
    assert result["input_sent"] is False
    assert clock.now == 0.25


def test_unavailable_or_truncated_diagnostic_fails_closed(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = DiagnosticsDevice([ACTIVE])

    def truncated(*_args: Any, **_kwargs: Any) -> ShellResult:
        return ShellResult(
            ok=True,
            serial=device.serial,
            argv=[],
            stdout=ACTIVE,
            stdout_truncated=True,
            exit_code=0,
            duration_ms=0,
        )

    monkeypatch.setattr(device, "run_read_only_shell", truncated)
    with pytest.raises(mic.MicRecordingNotReadyError) as caught:
        mic.wait_for_recording(device)
    assert caught.value.readiness["status"] == "unavailable"
    assert caught.value.readiness["checks"] == 1


def test_active_diagnostic_returning_after_deadline_cannot_admit_audio(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = DiagnosticsDevice([ACTIVE])
    original_read = device.run_read_only_shell

    def delayed(*args: Any, **kwargs: Any) -> ShellResult:
        clock.sleep(0.6)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(device, "run_read_only_shell", delayed)
    with pytest.raises(mic.MicRecordingNotReadyError) as caught:
        mic.wait_for_recording(device, timeout_ms=500)
    assert caught.value.readiness["status"] == "deadline_exceeded"
    assert caught.value.readiness["ready"] is False
    assert caught.value.readiness["active_inputs"] == 0
    assert caught.value.readiness["checks"] == 1


@pytest.mark.parametrize("has_control", [False, True])
def test_no_recording_never_opens_stream_and_returns_fresh_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Clock, has_control: bool
) -> None:
    device = DiagnosticsDevice([""], width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_a, **_kw: prepared)
    monkeypatch.setattr(mic, "inject_prepared", lambda _p: pytest.fail("stream must not open"))
    with pytest.raises(mic.MicRecordingNotReadyError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"} if has_control else None,
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )
    error = caught.value.to_dict()["error"]
    assert error["code"] == "mic_recording_not_ready"
    assert "audio was not injected" in error["result"]["detail"]
    assert error["result"]["observation_present"] is True
    assert error["result"]["recording_readiness"]["ready"] is False
    assert device.events[0] == ("down" if has_control else "readiness")
    assert device.events[-1] == ("up" if has_control else "readiness")
    rebuilt = _daemon_error(error).to_dict()["error"]
    assert rebuilt["recording_readiness"] == error["recording_readiness"]
    assert rebuilt["result"] == error["result"]


def test_readiness_runs_after_hold_and_before_the_only_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    device = DiagnosticsDevice(["", ACTIVE], width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_a, **_kw: prepared)
    monkeypatch.setattr(mic, "inject_prepared", lambda _p: device.events.append("inject"))
    result = engine.mic_inject(
        prepared.wav.path,
        selector={"rid": "hold_to_talk"},
        pre_roll_ms=0,
        post_roll_ms=0,
        observe=False,
    )
    assert device.events == ["down", "readiness", "readiness", "inject", "up"]
    assert result.recording_readiness["ready"] is True


def test_engine_uses_selected_microphone_service_without_android_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    prepared = _prepared(tmp_path / "voice.wav")

    def forbidden(*_a: Any, **_kw: Any) -> None:
        pytest.fail("core bypassed selected microphone service")

    service = SimpleNamespace(**dict.fromkeys(CAPABILITY_METHODS["microphone"], forbidden))
    service.validate_control_mode = lambda *_a, **_kw: "hold"
    service.prepare_injection = lambda *_a, **_kw: prepared
    service.claim_injection_attempt = lambda p: p
    service.wait_for_recording = lambda _d: events.append("native-ready") or {"ready": True}
    service.inject_prepared = lambda _p: events.append("native-inject")

    class Platform(_InjectedPlatform):
        capabilities = _InjectedPlatform.capabilities | {"microphone"}

        def load_capability(self, capability: str) -> object | None:
            return service if capability == "microphone" else None

    config = make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False})
    device = FakeDevice()
    monkeypatch.setattr(device, "run_read_only_shell", forbidden)
    monkeypatch.setattr(mic, "wait_for_recording", forbidden)
    engine = Engine(config, device=device, platform=Platform(config))
    result = engine.mic_inject(prepared.wav.path, observe=False)
    assert result.ok is True
    assert events == ["native-ready", "native-inject"]


def test_toggle_owner_change_during_readiness_sends_no_audio_or_blind_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps: list[tuple[int, int]] = []
    monkeypatch.setattr(mic, "prepare_injection", lambda *_a, **_kw: prepared)
    monkeypatch.setattr(device, "click_once", lambda x, y: taps.append((x, y)))
    monkeypatch.setattr(mic, "inject_prepared", lambda _p: pytest.fail("owner changed"))

    def ready_after_owner_change(_device: Any) -> dict[str, Any]:
        device._pkg = "org.example.other"
        return {"ready": True}

    monkeypatch.setattr(mic, "wait_for_recording", ready_after_owner_change)
    with pytest.raises(mic.MicToggleStopUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path, selector={"rid": "hold_to_talk"}, control_mode="toggle",
            pre_roll_ms=0, post_roll_ms=0, observe=False,
        )
    error = caught.value.to_dict()["error"]
    assert len(taps) == 1
    assert error["followup_errors"][0]["code"] == "mic_toggle_owner_changed"
    assert "audio was not injected" in error["result"]["detail"]
