"""Focused emulator microphone injection contracts (no emulator required)."""

from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.device as device_mod
import android_ui_analyser.emulator as emulator_mod
import android_ui_analyser.engine as engine_mod
import android_ui_analyser.mic as mic
from android_ui_analyser import journal
from android_ui_analyser.capabilities import capability_manifest
from android_ui_analyser.cli import _daemon_error, app
from android_ui_analyser.daemon import _mic_request_timeout, dispatch
from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError, UsageError
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from android_ui_analyser.mcp_server import _tool_definitions
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

runner = CliRunner()

HOLD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.Button" text="Hold to talk"
        resource-id="org.example:id/hold_to_talk" clickable="true" long-clickable="true"
        enabled="true" package="com.test.app" bounds="[20,100][300,220]"/>
</hierarchy>"""

SIBLING_TOGGLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.view.View" package="com.test.app" enabled="true"
        resource-id="com.test.app:id/tile" bounds="[0,80][400,260]">
    <node class="android.view.View" package="com.test.app" clickable="true" enabled="true"
          resource-id="com.test.app:id/toggle" bounds="[20,100][180,180]"/>
    <node class="android.widget.TextView" package="com.test.app" enabled="true"
          text="Voice input" bounds="[20,200][300,240]"/>
  </node>
</hierarchy>"""

PHRASE_TOGGLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" package="com.test.app" clickable="true" enabled="true"
        text="Terms of use and Privacy policy" bounds="[20,100][380,140]"/>
</hierarchy>"""


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 8_000,
    frames: int = 800,
) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(bytes([0x81]) * frames * channels * sample_width)
    return path


def _endpoint_record(
    directory: Path,
    *,
    serial_port: int = 5554,
    grpc_port: int = 8554,
    token: str = "private-test-token",
    cmdline: str = "/sdk/emulator @Example -grpc-use-token",
    emulator_version: str = "36.4.9.0",
    pid: int = 4242,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"pid_{pid}.ini"
    record.write_text(
        "\n".join(
            (
                f"port.serial={serial_port}",
                f"grpc.port={grpc_port}",
                f"grpc.token={token}",
                f"cmdline={cmdline}",
                f"emulator.version={emulator_version}",
            )
        ),
        encoding="utf-8",
    )
    return record


def _prepared(path: Path) -> mic.PreparedInjection:
    return mic.PreparedInjection(
        wav=mic.WavInfo(path, 8_000, 1, 2, 800),
        endpoint=mic.EmulatorEndpoint("emulator-5554", 8554, "hidden"),
        grpc=object(),
    )


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


class _RpcError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__("server details deliberately ignored")
        self._name = name

    def code(self) -> _Status:
        return _Status(self._name)


class _FakeChannel:
    def __init__(self, *, failure: str | None = None) -> None:
        self.failure = failure
        self.closed = False
        self.method: str | None = None
        self.packets: list[bytes] = []
        self.metadata: tuple[tuple[str, str], ...] = ()
        self.timeout: float | None = None
        self.call_count = 0

    def stream_unary(self, method: str, **_kwargs: Any) -> Any:
        self.method = method

        def invoke(
            packets: Any,
            *,
            metadata: tuple[tuple[str, str], ...],
            timeout: float,
        ) -> None:
            self.call_count += 1
            # Pull the iterator here, like a backpressuring client-streaming server.
            self.packets = list(packets)
            self.metadata = metadata
            self.timeout = timeout
            if self.failure:
                raise _RpcError(self.failure)

        return invoke

    def close(self) -> None:
        self.closed = True


class _FakeGrpc:
    RpcError = _RpcError

    def __init__(self, *, failure: str | None = None) -> None:
        self.channel = _FakeChannel(failure=failure)
        self.channel_options: tuple[tuple[str, int], ...] | None = None

    def insecure_channel(
        self,
        target: str,
        *,
        options: tuple[tuple[str, int], ...],
    ) -> _FakeChannel:
        assert target == "127.0.0.1:8554"
        self.channel_options = options
        return self.channel


def test_single_attempt_tap_uses_one_adb_process_and_never_uiautomator_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = object.__new__(Uiautomator2Device)
    device.serial = "emulator-5554"

    class RetryTrap:
        def click(self, *_target: int) -> None:
            raise AssertionError("uiautomator2 click can retry internally and must not be called")

    device._d = RetryTrap()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(device_mod.subprocess, "run", fake_run)

    device.click_once(123, 456)

    assert len(calls) == 1
    assert calls[0][0] == [
        "adb",
        "-s",
        "emulator-5554",
        "shell",
        "input",
        "tap",
        "123",
        "456",
    ]
    assert calls[0][1]["check"] is False


def test_single_attempt_tap_failure_is_typed_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = object.__new__(Uiautomator2Device)
    device.serial = "emulator-5554"
    calls = 0

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "transport lost")

    monkeypatch.setattr(device_mod.subprocess, "run", fake_run)

    with pytest.raises(DeviceError) as caught:
        device.click_once(123, 456)

    assert calls == 1
    assert caught.value.code == "tap_delivery_uncertain"
    assert "Do not repeat" in str(caught.value.hint)


def test_discovers_matching_emulator_endpoint_without_exposing_token(tmp_path: Path) -> None:
    running = tmp_path / "avd/running"
    _endpoint_record(running, token="top-secret-value")
    _endpoint_record(
        running,
        serial_port=5556,
        grpc_port=8556,
        token="other-emulator-secret",
        pid=4243,
    )

    endpoint = mic.discover_emulator_endpoint(
        "emulator-5554",
        running_dirs=[running],
        pid_is_live=lambda _pid: True,
    )

    assert endpoint.port == 8554 and endpoint.pid == 4242
    assert endpoint.token == "top-secret-value"
    assert "top-secret-value" not in repr(endpoint)
    assert "top-secret-value" not in str(endpoint)


def test_discovery_reports_physical_missing_and_no_audio_endpoints(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as physical:
        mic.discover_emulator_endpoint("R58M123456")
    assert physical.value.code == "mic_requires_emulator"

    running = tmp_path / "running"
    with pytest.raises(DeviceError) as missing:
        mic.discover_emulator_endpoint("emulator-5554", running_dirs=[running])
    assert missing.value.code == "mic_endpoint_missing"

    _endpoint_record(
        running,
        cmdline='"/sdk/emulator" "@Example" "-no-audio" "-grpc-use-token"',
    )
    with pytest.raises(UsageError) as disabled:
        mic.discover_emulator_endpoint(
            "emulator-5554",
            running_dirs=[running],
            pid_is_live=lambda _pid: True,
        )
    assert disabled.value.code == "mic_audio_disabled"


def test_discovery_ignores_a_dead_higher_pid_record(tmp_path: Path) -> None:
    running = tmp_path / "running"
    _endpoint_record(running, grpc_port=8554, pid=123)
    _endpoint_record(running, grpc_port=9999, token="stale-secret", pid=999_999)

    endpoint = mic.discover_emulator_endpoint(
        "emulator-5554",
        running_dirs=[running],
        pid_is_live=lambda pid: pid == 123,
    )

    assert endpoint.pid == 123
    assert endpoint.port == 8554


def test_windows_pid_probe_never_calls_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mic, "_windows_pid_is_live", lambda _pid: True)

    def destructive_probe(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("os.kill must never be used as a Windows liveness probe")

    monkeypatch.setattr(mic.os, "kill", destructive_probe)

    assert mic._pid_is_live(4242, system="Windows") is True


def test_windows_pid_probe_uses_pointer_safe_handles_and_closes_them() -> None:
    closed: list[int] = []

    class FakeCall:
        def __init__(self, callback: Any) -> None:
            self.callback = callback
            self.argtypes: list[Any] | None = None
            self.restype: Any = None

        def __call__(self, *args: Any) -> Any:
            return self.callback(*args)

    def set_active(_handle: int, pointer: Any) -> bool:
        pointer._obj.value = 259
        return True

    class FakeKernel32:
        OpenProcess = FakeCall(lambda _access, _inherit, pid: 12_345 if pid == 4242 else 0)
        GetExitCodeProcess = FakeCall(set_active)
        CloseHandle = FakeCall(lambda handle: closed.append(handle) or True)

    kernel32 = FakeKernel32()

    assert mic._windows_pid_is_live(4242, kernel32=kernel32) is True
    assert kernel32.OpenProcess.argtypes is not None
    assert kernel32.OpenProcess.restype is not None
    assert kernel32.GetExitCodeProcess.argtypes is not None
    assert kernel32.CloseHandle.argtypes is not None
    assert closed == [12_345]


def test_validates_wav_and_encodes_small_server_backpressured_packets(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "stereo.wav",
        channels=2,
        sample_width=2,
        sample_rate=48_000,
        frames=4_801,
    )
    info = mic.inspect_pcm_wav(path)

    assert info.sample_format == "S16"
    assert info.duration_s == pytest.approx(4_801 / 48_000)
    assert mic.audio_format_message(info) == b"\x08\x80\xf7\x02\x10\x01\x18\x01"
    packets = list(mic.iter_audio_packets(info))
    assert len(packets) == 2  # 100 ms at 48 kHz, then the final frame
    assert packets[0].startswith(b"\x0a\x08\x08\x80\xf7\x02\x10\x01\x18\x01\x1a")
    # DeliveryMode field 4 is absent: MODE_UNSPECIFIED lets the server apply backpressure.
    assert b"\x20\x01" not in mic.audio_format_message(info)


@pytest.mark.parametrize(
    "channels,sample_width,sample_rate",
    [(3, 2, 8_000), (1, 3, 8_000), (1, 2, 48_001)],
)
def test_rejects_unsupported_pcm_wav_shapes(
    tmp_path: Path, channels: int, sample_width: int, sample_rate: int
) -> None:
    path = _write_wav(
        tmp_path / f"bad-{channels}-{sample_width}-{sample_rate}.wav",
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
    )
    with pytest.raises(UsageError) as caught:
        mic.inspect_pcm_wav(path)
    assert caught.value.code == "mic_wav_invalid"


def test_rejects_malformed_and_empty_wav(tmp_path: Path) -> None:
    malformed = tmp_path / "not-a-wave.wav"
    malformed.write_bytes(b"not wave data")
    with pytest.raises(UsageError) as bad:
        mic.inspect_pcm_wav(malformed)
    assert bad.value.code == "mic_wav_invalid"
    assert str(malformed) not in str(bad.value)

    empty = _write_wav(tmp_path / "empty.wav", frames=0)
    with pytest.raises(UsageError) as no_audio:
        mic.inspect_pcm_wav(empty)
    assert no_audio.value.code == "mic_wav_invalid"


def test_rejects_wav_longer_than_the_daemon_safe_limit(tmp_path: Path) -> None:
    too_long = _write_wav(
        tmp_path / "too-long.wav",
        sample_rate=1,
        frames=int(mic.MAX_WAV_DURATION_S) + 1,
    )

    with pytest.raises(UsageError) as caught:
        mic.inspect_pcm_wav(too_long)

    assert caught.value.code == "mic_wav_invalid"
    assert "at most 300 seconds" in str(caught.value)


def test_packet_stream_refuses_changed_wav_metadata_and_validated_frame_overrun(
    tmp_path: Path,
) -> None:
    path = _write_wav(tmp_path / "voice.wav", frames=800)
    inspected = mic.inspect_pcm_wav(path)
    _write_wav(path, frames=1_600)

    with pytest.raises(UsageError) as changed:
        list(mic.iter_audio_packets(inspected))

    assert changed.value.code == "mic_wav_invalid"
    assert "changed after validation" in str(changed.value)


def test_packet_stream_snapshots_audio_before_yielding_first_packet(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "voice.wav", frames=1_600)
    inspected = mic.inspect_pcm_wav(path)
    packets = mic.iter_audio_packets(inspected)

    first = next(packets)
    with wave.open(str(path), "wb") as replacement:
        replacement.setnchannels(1)
        replacement.setsampwidth(2)
        replacement.setframerate(8_000)
        replacement.writeframes(bytes([0x22]) * 1_600 * 2)

    assert first == mic.audio_packet_message(inspected, bytes([0x81]) * 800 * 2)
    assert list(packets) == [mic.audio_packet_message(inspected, bytes([0x81]) * 800 * 2)]


def test_authenticated_grpc_success_waits_for_response_and_closes_channel(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "voice.wav")
    grpc = _FakeGrpc()
    prepared = mic.PreparedInjection(
        mic.inspect_pcm_wav(path),
        mic.EmulatorEndpoint("emulator-5554", 8554, "one-use-secret"),
        grpc,
    )

    result = mic.inject_prepared(prepared)

    assert result.path == path
    assert grpc.channel.method == "/android.emulation.control.EmulatorController/injectAudio"
    assert grpc.channel.packets
    assert grpc.channel.metadata == (("authorization", "Bearer one-use-secret"),)
    assert grpc.channel_options == (("grpc.enable_http_proxy", 0),)
    assert grpc.channel.closed is True
    assert "one-use-secret" not in repr(prepared)


def test_affected_emulator_allows_only_one_persisted_stream_attempt_per_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_wav(tmp_path / "voice.wav")
    running = tmp_path / "avd/running"
    _endpoint_record(
        running,
        token="one-use-secret",
        pid=os.getpid(),
        emulator_version="36.4.10.0",
    )
    first_grpc = _FakeGrpc()
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "worker-a"))
    first = mic.prepare_injection(
        "emulator-5554",
        path,
        running_dirs=[running],
        grpc_module=first_grpc,
    )
    second_grpc = _FakeGrpc()
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "worker-b"))
    second = mic.prepare_injection(
        "emulator-5554",
        path,
        running_dirs=[running],
        grpc_module=second_grpc,
    )

    mic.inject_prepared(first)

    assert first_grpc.channel.call_count == 1
    assert first.attempt_guard == second.attempt_guard
    assert first.attempt_guard is not None
    assert first.attempt_guard.parent == running.resolve()
    assert first.attempt_guard.is_file()
    assert "one-use-secret" not in str(first.attempt_guard)
    assert "one-use-secret" not in first.attempt_guard.read_text(encoding="utf-8")

    with pytest.raises(DeviceError) as repeated:
        mic.inject_prepared(second)

    assert repeated.value.code == "mic_repeat_unsafe"
    assert "Do not retry" in str(repeated.value.hint)
    assert second_grpc.channel.call_count == 0


@pytest.mark.parametrize(
    "status,code",
    [
        ("FAILED_PRECONDITION", "mic_injection_precondition"),
        ("INVALID_ARGUMENT", "mic_injection_rejected"),
        ("UNAUTHENTICATED", "mic_endpoint_auth_failed"),
        ("DEADLINE_EXCEEDED", "mic_injection_timeout"),
        ("UNAVAILABLE", "mic_emulator_unavailable"),
        ("INTERNAL", "mic_delivery_uncertain"),
        ("UNKNOWN", "mic_injection_failed"),
    ],
)
def test_grpc_failures_are_focused_never_retried_and_always_close(
    tmp_path: Path, status: str, code: str
) -> None:
    path = _write_wav(tmp_path / f"{status}.wav")
    grpc = _FakeGrpc(failure=status)
    prepared = mic.PreparedInjection(
        mic.inspect_pcm_wav(path),
        mic.EmulatorEndpoint("emulator-5554", 8554, "never-print-this"),
        grpc,
    )

    with pytest.raises(DeviceError) as caught:
        mic.inject_prepared(prepared)

    assert caught.value.code == code
    assert "never-print-this" not in str(caught.value)
    assert grpc.channel.closed is True
    # A single stream call proves UNAVAILABLE is not blindly retried.
    assert grpc.channel.call_count == 1
    if status in {"DEADLINE_EXCEEDED", "UNAVAILABLE", "INTERNAL", "UNKNOWN"}:
        assert "Do not retry" in str(caught.value.hint)


@pytest.mark.parametrize(
    "code",
    [
        "mic_repeat_unsafe",
        "mic_endpoint_missing",
        "mic_injection_timeout",
        "mic_emulator_unavailable",
        "mic_speech_failed",
    ],
)
def test_daemon_reconstructs_mic_device_errors_with_device_exit_semantics(code: str) -> None:
    rebuilt = _daemon_error({"code": code, "message": "mic failed", "hint": "recover"})

    assert isinstance(rebuilt, DeviceError)
    assert rebuilt.code == code
    assert int(rebuilt.exit_code) == 3


def test_daemon_keeps_mic_validation_errors_as_usage_failures() -> None:
    rebuilt = _daemon_error({"code": "mic_wav_invalid", "message": "bad wav", "hint": "use PCM"})

    assert not isinstance(rebuilt, DeviceError)
    assert rebuilt.code == "mic_wav_invalid"
    assert int(rebuilt.exit_code) == 2


def test_hold_selector_stays_down_for_pre_audio_post_and_releases_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=HOLD_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    events: list[tuple[str, Any]] = []
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: events.append(("inject", None)),
    )
    monkeypatch.setattr(device, "touch_down", lambda x, y: events.append(("down", (x, y))))
    monkeypatch.setattr(device, "touch_up", lambda x, y: events.append(("up", (x, y))))
    monkeypatch.setattr(engine_mod.time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    result = engine.mic_inject(
        prepared.wav.path,
        selector={"rid": "hold_to_talk"},
        pre_roll_ms=100,
        post_roll_ms=200,
        observe=False,
    )

    assert result.action == "mic-inject" and result.id == 0
    assert [name for name, _value in events] == ["down", "sleep", "inject", "sleep", "up"]
    assert events[1][1] == pytest.approx(0.1)
    assert events[3][1] == pytest.approx(0.2)

    events.clear()

    def fail(_prepared: mic.PreparedInjection) -> None:
        events.append(("inject", None))
        raise DeviceError("stream failed", code="mic_injection_failed")

    monkeypatch.setattr(mic, "inject_prepared", fail)
    with pytest.raises(DeviceError):
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )
    assert [name for name, _value in events] == ["down", "inject", "up"]


def test_targetless_default_remains_audio_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(serial="emulator-5554")
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    injected: list[mic.PreparedInjection] = []
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", lambda value: injected.append(value))

    result = engine.mic_inject(
        prepared.wav.path,
        pre_roll_ms=0,
        post_roll_ms=0,
        observe=False,
    )

    assert result.ok is True
    assert injected == [prepared]
    assert not any(
        name in {"click", "click_once", "touch_down", "touch_up"} for name, _ in device.calls
    )


def test_toggle_selector_taps_start_and_stop_around_audio_with_cache_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=HOLD_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(
            cache={"dir": str(tmp_path / "cache"), "enabled": False},
            memory={"enabled": False},
        ),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    events: list[tuple[str, Any]] = []
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: events.append(("inject", None)),
    )
    monkeypatch.setattr(
        device,
        "click_once",
        lambda x, y: events.append(("tap", (x, y))),
    )
    monkeypatch.setattr(engine_mod.time, "sleep", lambda value: events.append(("sleep", value)))

    result = engine.mic_inject(
        prepared.wav.path,
        selector={"rid": "hold_to_talk"},
        control_mode="toggle",
        pre_roll_ms=100,
        post_roll_ms=200,
        observe=False,
    )

    assert result.ok is True
    assert [name for name, _ in events] == ["tap", "sleep", "inject", "sleep", "tap"]
    assert events[0][1] == events[-1][1]
    assert events[1][1] == pytest.approx(0.1)
    assert events[3][1] == pytest.approx(0.2)
    assert "toggle control" in str(result.detail)


def test_toggle_numeric_id_refuses_sibling_retarget_before_any_tap_or_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=SIBLING_TOGGLE_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    label = next(el for el in engine.analyze().elements if el.text == "Voice input")
    prepared = _prepared(tmp_path / "voice.wav")
    injections = 0
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)

    def inject(_prepared: mic.PreparedInjection) -> None:
        nonlocal injections
        injections += 1

    monkeypatch.setattr(mic, "inject_prepared", inject)

    with pytest.raises(UsageError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            label.id,
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    assert caught.value.code == "unsafe_action_target"
    assert injections == 0
    assert not any(name == "click_once" for name, _ in device.calls)


def test_toggle_text_selector_uses_phrase_aim_for_both_taps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=PHRASE_TOGGLE_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(
            cache={"dir": str(tmp_path / "cache"), "enabled": False},
            memory={"enabled": False},
        ),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps: list[tuple[int, int]] = []
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", lambda _prepared: None)
    monkeypatch.setattr(device, "click_once", lambda x, y: taps.append((x, y)))

    engine.mic_inject(
        prepared.wav.path,
        selector={"text": "Privacy policy"},
        control_mode="toggle",
        pre_roll_ms=0,
        post_roll_ms=0,
        observe=False,
    )

    assert len(taps) == 2 and taps[0] == taps[1]
    assert taps[0][0] > 200, "the phrase is right of the full line's centre"


def test_toggle_known_injection_failure_still_stops_once_and_preserves_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    events: list[str] = []
    original = DeviceError("stream failed", code="mic_injection_failed")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        device,
        "click_once",
        lambda *_target: events.append("tap"),
    )

    def fail(_prepared: mic.PreparedInjection) -> None:
        events.append("inject")
        raise original

    monkeypatch.setattr(mic, "inject_prepared", fail)

    with pytest.raises(DeviceError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    assert caught.value is original
    assert events == ["tap", "inject", "tap"]


def test_ambiguous_toggle_start_injects_nothing_sends_no_stop_and_forces_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps = 0
    injections = 0

    def ambiguous_start(*_target: int) -> None:
        nonlocal taps
        taps += 1
        raise DeviceError("tap response lost", code="tap_delivery_uncertain")

    def inject(_prepared: mic.PreparedInjection) -> None:
        nonlocal injections
        injections += 1

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", inject)
    monkeypatch.setattr(device, "click_once", ambiguous_start)
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "via": "hierarchy", "ms": 1},
    )

    with pytest.raises(mic.MicToggleStartUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    error = caught.value.to_dict()["error"]
    assert taps == 1 and injections == 0
    assert error["code"] == "mic_toggle_start_uncertain"
    assert "Recording may be active" in error["hint"]
    assert error["result"]["observation_present"] is True
    assert "audio was not injected" in error["result"]["detail"]
    rebuilt = _daemon_error(error)
    assert isinstance(rebuilt, mic.MicToggleStartUncertainError)
    assert rebuilt.to_dict()["error"]["result"]["observation_present"] is True


def test_toggle_stop_uncertainty_wraps_known_audio_error_and_forces_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps = 0
    original = DeviceError("stream failed", code="mic_injection_failed")

    def tap(*_target: int) -> None:
        nonlocal taps
        taps += 1
        if taps == 2:
            raise DeviceError("stop response lost", code="tap_delivery_uncertain")

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: (_ for _ in ()).throw(original),
    )
    monkeypatch.setattr(device, "click_once", tap)
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "via": "hierarchy", "ms": 1},
    )

    with pytest.raises(mic.MicToggleStopUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    error = caught.value.to_dict()["error"]
    assert taps == 2
    assert error["code"] == "mic_toggle_stop_uncertain"
    assert [item["code"] for item in error["followup_errors"]] == [
        "mic_injection_failed",
        "tap_delivery_uncertain",
    ]
    assert error["result"]["observation_present"] is True


def test_clean_toggle_delivery_with_uncertain_stop_is_nonretryable_and_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps = 0

    def tap(*_target: int) -> None:
        nonlocal taps
        taps += 1
        if taps == 2:
            raise DeviceError("stop response lost", code="tap_delivery_uncertain")

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", lambda _prepared: None)
    monkeypatch.setattr(device, "click_once", tap)
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "via": "hierarchy", "ms": 1},
    )

    with pytest.raises(mic.MicToggleStopUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    error = caught.value.to_dict()["error"]
    assert taps == 2
    assert error["code"] == "mic_toggle_stop_uncertain"
    assert error["followup_errors"][0]["code"] == "tap_delivery_uncertain"
    assert error["result"]["observation_present"] is True
    assert "Do not repeat" in error["hint"]
    rebuilt = _daemon_error(error)
    assert isinstance(rebuilt, mic.MicToggleStopUncertainError)
    assert int(rebuilt.exit_code) == 3
    assert rebuilt.to_dict()["error"]["followup_errors"] == error["followup_errors"]


def test_internal_delivery_remains_primary_when_toggle_stop_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps = 0

    def tap(*_target: int) -> None:
        nonlocal taps
        taps += 1
        if taps == 2:
            raise DeviceError("stop response lost", code="tap_delivery_uncertain")

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: (_ for _ in ()).throw(mic.MicDeliveryUncertainError()),
    )
    monkeypatch.setattr(device, "click_once", tap)
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "via": "hierarchy", "ms": 1},
    )

    with pytest.raises(mic.MicDeliveryUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    error = caught.value.to_dict()["error"]
    assert type(caught.value) is mic.MicDeliveryUncertainError
    assert taps == 2
    assert error["code"] == "mic_delivery_uncertain"
    assert error["followup_errors"][0]["stage"] == "toggle_stop"
    assert error["result"]["observation_present"] is True


@pytest.mark.parametrize(
    ("pre_roll_ms", "post_roll_ms", "fail_on_sleep", "injections"),
    [
        (100, 0, 1, 0),
        (0, 100, 1, 1),
    ],
)
def test_toggle_roll_failure_still_attempts_one_stop_and_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pre_roll_ms: int,
    post_roll_ms: int,
    fail_on_sleep: int,
    injections: int,
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    events: list[str] = []
    original = KeyboardInterrupt("cancelled during roll")
    sleep_calls = 0

    def tap(*_target: int) -> None:
        events.append("tap")

    def inject(_prepared: mic.PreparedInjection) -> None:
        events.append("inject")

    def sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        events.append("sleep")
        if sleep_calls == fail_on_sleep:
            raise original

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", inject)
    monkeypatch.setattr(device, "click_once", tap)
    monkeypatch.setattr(engine_mod.time, "sleep", sleep)

    with pytest.raises(KeyboardInterrupt) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=pre_roll_ms,
            post_roll_ms=post_roll_ms,
            observe=False,
        )

    assert caught.value is original
    assert events.count("tap") == 2
    assert events.count("inject") == injections


def test_guard_and_prestart_owner_refusals_happen_before_any_toggle_tap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps = 0
    injections = 0
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)

    def tap(*_target: int) -> None:
        nonlocal taps
        taps += 1

    def inject(_prepared: mic.PreparedInjection) -> None:
        nonlocal injections
        injections += 1

    monkeypatch.setattr(device, "click_once", tap)
    monkeypatch.setattr(mic, "inject_prepared", inject)
    monkeypatch.setattr(
        mic,
        "claim_injection_attempt",
        lambda _prepared: (_ for _ in ()).throw(
            DeviceError("one attempt per boot", code="mic_repeat_unsafe")
        ),
    )

    with pytest.raises(DeviceError) as guarded:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
        )
    assert guarded.value.code == "mic_repeat_unsafe"
    assert taps == injections == 0

    monkeypatch.setattr(
        mic,
        "claim_injection_attempt",
        lambda value: setattr(device, "_pkg", "com.other.app") or value,
    )
    with pytest.raises(DeviceError) as moved:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
        )
    assert moved.value.code == "mic_toggle_owner_changed"
    assert taps == injections == 0


def test_toggle_owner_change_after_start_sends_no_audio_or_blind_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    taps = 0
    injections = 0

    def start_then_navigate(*_target: int) -> None:
        nonlocal taps
        taps += 1
        device._pkg = "com.other.app"

    def inject(_prepared: mic.PreparedInjection) -> None:
        nonlocal injections
        injections += 1

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", inject)
    monkeypatch.setattr(device, "click_once", start_then_navigate)
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "via": "hierarchy", "ms": 1},
    )

    with pytest.raises(mic.MicToggleStopUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    error = caught.value.to_dict()["error"]
    assert taps == 1 and injections == 0
    assert error["code"] == "mic_toggle_stop_uncertain"
    assert error["followup_errors"][0]["code"] == "mic_toggle_owner_changed"
    assert error["result"]["observation_present"] is True


@pytest.mark.parametrize(
    ("xml", "code"),
    [
        (HOLD_XML.replace('enabled="true"', 'enabled="false"'), "mic_toggle_target_inactive"),
        (
            HOLD_XML.replace('enabled="true"', 'enabled="true" selected="true"'),
            "mic_toggle_already_active",
        ),
        (
            HOLD_XML.replace('enabled="true"', 'enabled="true" checked="true"'),
            "mic_toggle_already_active",
        ),
    ],
)
def test_toggle_rejects_inactive_or_already_on_control_before_gesture_or_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    xml: str,
    code: str,
) -> None:
    device = FakeDevice(hierarchy_xml=xml, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    injected = False
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)

    def inject(_prepared: mic.PreparedInjection) -> None:
        nonlocal injected
        injected = True

    monkeypatch.setattr(mic, "inject_prepared", inject)

    with pytest.raises(UsageError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            control_mode="toggle",
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    assert caught.value.code == code
    assert injected is False
    assert not any(
        name in {"click", "click_once", "touch_down", "touch_up"} for name, _ in device.calls
    )


def test_touch_down_lost_response_still_attempts_release_and_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=HOLD_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    events: list[str] = []
    original = DeviceError("touch-down response was lost", code="device")

    def delivered_down_then_failed(*_target: int) -> None:
        events.append("down")
        raise original

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(device, "touch_down", delivered_down_then_failed)
    monkeypatch.setattr(device, "touch_up", lambda *_target: events.append("up"))

    with pytest.raises(DeviceError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    assert caught.value is original
    assert events == ["down", "up"]


def test_internal_stream_error_releases_hold_and_carries_post_action_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=HOLD_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: (_ for _ in ()).throw(mic.MicDeliveryUncertainError()),
    )
    monkeypatch.setattr(
        engine,
        "_await_post_action_ready",
        lambda **_kwargs: {"changed": True, "via": "hierarchy", "ms": 1},
    )

    with pytest.raises(mic.MicDeliveryUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    assert [call[0] for call in device.calls[-2:]] == ["touch_down", "touch_up"]
    error = caught.value.to_dict()["error"]
    assert error["code"] == "mic_delivery_uncertain"
    assert "Do not retry blindly" in error["hint"]
    assert error["result"]["ok"] is False
    assert error["result"]["observation_present"] is True
    assert error["result"]["observation"]["meta"]["device_serial"] == "emulator-5554"

    daemon = dispatch(
        engine,
        {
            "cmd": "mic_inject",
            "args": {
                "wav_path": str(prepared.wav.path),
                "selector": {"rid": "hold_to_talk"},
                "pre_roll_ms": 0,
                "post_roll_ms": 0,
                "observe": True,
            },
        },
    )
    assert daemon["ok"] is False
    assert daemon["error"]["result"]["observation_present"] is True
    rebuilt = _daemon_error(daemon["error"])
    assert rebuilt.to_dict()["error"]["result"]["observation_present"] is True

    with pytest.raises(mic.MicDeliveryUncertainError) as mcp_error:
        mcp_dispatch(
            engine,
            "mic_inject_and_analyze",
            {
                "path": str(prepared.wav.path),
                "rid": "hold_to_talk",
                "pre_roll_ms": 0,
                "post_roll_ms": 0,
            },
        )
    assert mcp_error.value.to_dict()["error"]["result"]["observation_present"] is True


def test_internal_uncertainty_remains_primary_when_touch_release_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=HOLD_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: (_ for _ in ()).throw(mic.MicDeliveryUncertainError()),
    )
    monkeypatch.setattr(
        device,
        "touch_up",
        lambda *_args: (_ for _ in ()).throw(DeviceError("release failed")),
    )

    with pytest.raises(mic.MicDeliveryUncertainError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=True,
        )

    error = caught.value.to_dict()["error"]
    assert error["code"] == "mic_delivery_uncertain"
    assert error["result"]["observation_present"] is True
    assert error["followup_errors"][0]["stage"] == "touch_release"
    assert "touch release also failed" in error["hint"]
    rebuilt = _daemon_error(error)
    rebuilt_error = rebuilt.to_dict()["error"]
    assert rebuilt_error["followup_errors"][0]["stage"] == "touch_release"


def test_clean_delivery_with_release_failure_is_non_retryable_and_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        hierarchy_xml=HOLD_XML,
        serial="emulator-5554",
        width=400,
        height=800,
    )
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    calls = 0

    def delivered(_prepared: mic.PreparedInjection) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", delivered)
    monkeypatch.setattr(
        device,
        "touch_up",
        lambda *_args: (_ for _ in ()).throw(DeviceError("release failed")),
    )

    with pytest.raises(mic.MicDeliveredReleaseError) as caught:
        engine.mic_inject(
            prepared.wav.path,
            selector={"rid": "hold_to_talk"},
            pre_roll_ms=0,
            post_roll_ms=0,
            observe=False,
        )

    error = caught.value.to_dict()["error"]
    assert calls == 1
    assert error["code"] == "mic_delivered_release_failed"
    assert "Do not repeat" in error["hint"]
    assert error["result"]["observation_present"] is True
    assert error["followup_errors"][0]["stage"] == "touch_release"
    rebuilt = _daemon_error(error)
    assert isinstance(rebuilt, mic.MicDeliveredReleaseError)
    assert rebuilt.to_dict()["error"]["result"]["observation_present"] is True


def test_internal_uncertainty_remains_primary_when_observation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=FakeDevice(serial="emulator-5554"),
    )
    prepared = _prepared(tmp_path / "voice.wav")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        mic,
        "inject_prepared",
        lambda _prepared: (_ for _ in ()).throw(mic.MicDeliveryUncertainError()),
    )
    monkeypatch.setattr(
        engine,
        "_observe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DeviceError("observe failed")),
    )

    with pytest.raises(mic.MicDeliveryUncertainError) as caught:
        engine.mic_inject(prepared.wav.path, pre_roll_ms=0, post_roll_ms=0)

    error = caught.value.to_dict()["error"]
    assert error["code"] == "mic_delivery_uncertain"
    assert "result" not in error
    assert error["followup_errors"][0]["stage"] == "observation"
    assert "observation also failed" in error["hint"]


def test_hold_accepts_a_fresh_numeric_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=device,
    )
    observed = engine.analyze(source="hierarchy", with_ocr=False)
    prepared = _prepared(tmp_path / "voice.wav")
    monkeypatch.setattr(mic, "prepare_injection", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(mic, "inject_prepared", lambda _prepared: None)

    result = engine.mic_inject(
        prepared.wav.path,
        observed.elements[0].id,
        pre_roll_ms=0,
        post_roll_ms=0,
        observe=False,
    )

    assert result.id == observed.elements[0].id
    assert [call[0] for call in device.calls[-2:]] == ["touch_down", "touch_up"]


def test_speak_has_a_clear_unsupported_host_error(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as caught:
        mic.synthesize_speech("hello", tmp_path / "speech.wav", system="Linux")
    assert caught.value.code == "mic_speech_unsupported_host"


@pytest.mark.skipif(
    not Path("/usr/bin/say").exists(),
    reason="`mic.synthesize_speech` refuses outright without macOS /usr/bin/say, before any "
    "mock of subprocess.run is reached",
)
def test_speak_uses_mac_say_to_make_valid_pcm(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    inputs: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        inputs.append(str(kwargs["input"]))
        destination = Path(command[command.index("-o") + 1])
        _write_wav(destination, sample_rate=44_100)
        return subprocess.CompletedProcess(command, 0, "", "")

    output = mic.synthesize_speech(
        "testing one two",
        tmp_path / "speech.wav",
        voice="Samantha",
        rate=175,
        system="Darwin",
        run=fake_run,
    )

    assert output.is_file()
    assert "--data-format=LEI16@44100" in calls[0]
    assert calls[0][-4:] == ["--voice", "Samantha", "--rate", "175"]
    assert "testing one two" not in calls[0]
    assert inputs == ["testing one two"]


def test_cli_daemon_and_mcp_expose_equivalent_mic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_inject(
        self: Engine,
        wav_path: str | Path,
        element_id: int | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        calls.append(
            {
                "path": str(wav_path),
                "id": element_id,
                "selector": kwargs.get("selector"),
                "control_mode": kwargs.get("control_mode"),
                "pre": kwargs.get("pre_roll_ms"),
                "post": kwargs.get("post_roll_ms"),
                "observe": kwargs.get("observe"),
            }
        )
        return ActionResult(ok=True, action="mic-inject")

    monkeypatch.setattr(Engine, "mic_inject", fake_inject)
    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda serial=None: FakeDevice(serial=serial or "emulator-5554"),
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cli-cache"))
    wav_path = str(tmp_path / "voice.wav")
    _write_wav(Path(wav_path))

    cli = runner.invoke(
        app,
        [
            "--no-lease",
            "mic",
            "inject",
            wav_path,
            "--rid",
            "hold_to_talk",
            "--control-mode",
            "toggle",
            "--pre-roll-ms",
            "10",
            "--post-roll-ms",
            "20",
        ],
    )
    assert cli.exit_code == 0, cli.stderr
    assert json.loads(cli.stdout)["action"] == "mic-inject"

    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    daemon = dispatch(
        engine,
        {
            "cmd": "mic_inject",
            "args": {
                "wav_path": wav_path,
                "selector": {"rid": "hold_to_talk"},
                "control_mode": "toggle",
                "pre_roll_ms": 10,
                "post_roll_ms": 20,
                "observe": True,
            },
        },
    )
    assert daemon["ok"] is True

    mcp = mcp_dispatch(
        engine,
        "mic_inject_and_analyze",
        {
            "path": wav_path,
            "rid": "hold_to_talk",
            "control_mode": "toggle",
            "pre_roll_ms": 10,
            "post_roll_ms": 20,
        },
    )
    assert mcp["action"] == "mic-inject"
    assert calls == [
        {
            "path": wav_path,
            "id": None,
            "selector": {
                "rid": "hold_to_talk",
                "text": None,
                "desc": None,
                "index": None,
                "first": False,
            },
            "control_mode": "toggle",
            "pre": 10,
            "post": 20,
            "observe": True,
        },
        {
            "path": wav_path,
            "id": None,
            "selector": {"rid": "hold_to_talk"},
            "control_mode": "toggle",
            "pre": 10,
            "post": 20,
            "observe": True,
        },
        {
            "path": wav_path,
            "id": None,
            "selector": {"rid": "hold_to_talk", "text": None, "desc": None},
            "control_mode": "toggle",
            "pre": 10,
            "post": 20,
            "observe": True,
        },
    ]


def test_cli_resolves_relative_wav_before_routing_and_rejects_orphan_selector_modifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_inject(
        self: Engine,
        wav_path: str | Path,
        element_id: int | None = None,
        **_kwargs: Any,
    ) -> ActionResult:
        calls.append(str(wav_path))
        return ActionResult(ok=True, action="mic-inject", id=element_id)

    monkeypatch.setattr(Engine, "mic_inject", fake_inject)
    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda serial=None: FakeDevice(serial=serial or "emulator-5554"),
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cli-cache"))
    wav = _write_wav(tmp_path / "relative.wav")
    monkeypatch.chdir(tmp_path)

    relative = runner.invoke(app, ["--no-lease", "mic", "inject", wav.name])

    assert relative.exit_code == 0, relative.stderr
    assert calls == [str(wav.resolve())]

    no_inject_target = runner.invoke(
        app,
        ["--no-lease", "mic", "inject", wav.name, "--first"],
    )
    assert no_inject_target.exit_code == 2
    assert json.loads(no_inject_target.stderr)["error"]["code"] == "usage"

    no_speak_target = runner.invoke(
        app,
        ["--no-lease", "mic", "speak", "private words", "--index", "0"],
    )
    assert no_speak_target.exit_code == 2
    assert json.loads(no_speak_target.stderr)["error"]["code"] == "usage"

    ambiguous_id_and_selector = runner.invoke(
        app,
        ["--no-lease", "mic", "inject", wav.name, "12", "--rid", "hold_to_talk"],
    )
    assert ambiguous_id_and_selector.exit_code == 2
    assert json.loads(ambiguous_id_and_selector.stderr)["error"]["code"] == "usage"

    indexed_numeric_id = runner.invoke(
        app,
        ["--no-lease", "mic", "inject", wav.name, "12", "--index", "0"],
    )
    assert indexed_numeric_id.exit_code == 2
    assert json.loads(indexed_numeric_id.stderr)["error"]["code"] == "usage"

    first_numeric_id = runner.invoke(
        app,
        ["--no-lease", "mic", "inject", wav.name, "12", "--first"],
    )
    assert first_numeric_id.exit_code == 2
    assert json.loads(first_numeric_id.stderr)["error"]["code"] == "usage"

    ambiguous_speak_target = runner.invoke(
        app,
        [
            "--no-lease",
            "mic",
            "speak",
            "private words",
            "12",
            "--text",
            "hold to talk",
        ],
    )
    assert ambiguous_speak_target.exit_code == 2
    assert json.loads(ambiguous_speak_target.stderr)["error"]["code"] == "usage"

    assert calls == [str(wav.resolve())]


def test_invalid_control_modes_fail_before_wav_prepare_or_speech_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = Engine(
        make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False}),
        device=FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554"),
    )
    prepared = 0
    synthesized = 0

    def prepare(*_args: Any, **_kwargs: Any) -> mic.PreparedInjection:
        nonlocal prepared
        prepared += 1
        raise AssertionError("invalid control input must fail before WAV/endpoint preparation")

    def synthesize(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal synthesized
        synthesized += 1
        raise AssertionError("invalid control input must fail before private speech synthesis")

    monkeypatch.setattr(mic, "prepare_injection", prepare)
    monkeypatch.setattr(mic, "synthesize_speech", synthesize)

    with pytest.raises(UsageError) as invalid:
        engine.mic_inject(
            "unused.wav",
            selector={"rid": "hold_to_talk"},
            control_mode="press",
        )
    assert invalid.value.code == "mic_control_mode_invalid"

    with pytest.raises(UsageError) as missing:
        engine.mic_inject("unused.wav", control_mode="toggle")
    assert missing.value.code == "mic_toggle_target_required"

    with pytest.raises(UsageError) as private_invalid:
        engine.mic_speak(
            "private words",
            selector={"rid": "hold_to_talk"},
            control_mode="press",
        )
    assert private_invalid.value.code == "mic_control_mode_invalid"

    with pytest.raises(UsageError) as private_missing:
        engine.mic_speak("private words", control_mode="toggle")
    assert private_missing.value.code == "mic_toggle_target_required"
    assert prepared == synthesized == 0

    daemon = dispatch(
        engine,
        {"cmd": "mic_inject", "args": {"wav_path": "unused.wav", "control_mode": "toggle"}},
    )
    assert daemon["error"]["code"] == "mic_toggle_target_required"

    with pytest.raises(UsageError) as mcp_error:
        mcp_dispatch(
            engine,
            "mic_speak_and_analyze",
            {"speech": "private words", "control_mode": "toggle"},
        )
    assert mcp_error.value.code == "mic_toggle_target_required"

    cli_missing = runner.invoke(
        app,
        ["--no-lease", "mic", "inject", "unused.wav", "--control-mode", "toggle"],
    )
    assert cli_missing.exit_code == 2
    assert json.loads(cli_missing.stderr)["error"]["code"] == "mic_toggle_target_required"

    cli_invalid = runner.invoke(
        app,
        [
            "--no-lease",
            "mic",
            "speak",
            "private words",
            "--rid",
            "hold_to_talk",
            "--control-mode",
            "press",
        ],
    )
    assert cli_invalid.exit_code == 2
    assert json.loads(cli_invalid.stderr)["error"]["code"] == "mic_control_mode_invalid"


def test_mcp_and_cli_publish_control_mode_contract() -> None:
    tools = {tool.name: tool for tool in _tool_definitions()}
    for name in ("mic_inject_and_analyze", "mic_speak_and_analyze"):
        mode = tools[name].inputSchema["properties"]["control_mode"]
        assert mode["enum"] == ["hold", "toggle"]
        assert mode["default"] == "hold"

    inject_help = runner.invoke(app, ["mic", "inject", "--help"])
    speak_help = runner.invoke(app, ["mic", "speak", "--help"])
    assert inject_help.exit_code == speak_help.exit_code == 0
    assert "--control-mode" in inject_help.stdout
    assert "--control-mode" in speak_help.stdout


def test_mic_speak_cli_daemon_mcp_and_capability_names_are_in_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_speak(
        self: Engine,
        text: str,
        element_id: int | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        kwargs["element_id"] = element_id
        calls.append((text, kwargs))
        return ActionResult(ok=True, action="mic-speak")

    monkeypatch.setattr(Engine, "mic_speak", fake_speak)
    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda serial=None: FakeDevice(serial=serial or "emulator-5554"),
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cli-cache"))

    cli = runner.invoke(
        app,
        [
            "--no-lease",
            "mic",
            "speak",
            "hello world",
            "--rid",
            "hold_to_talk",
            "--control-mode",
            "toggle",
            "--voice",
            "Alex",
            "--rate",
            "160",
        ],
    )
    assert cli.exit_code == 0, cli.stderr

    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    assert dispatch(
        engine,
        {
            "cmd": "mic_speak",
            "args": {
                "text": "hello world",
                "selector": {"rid": "hold_to_talk"},
                "control_mode": "toggle",
                "voice": "Alex",
                "rate": 160,
            },
        },
    )["ok"]
    assert (
        mcp_dispatch(
            engine,
            "mic_speak_and_analyze",
            {
                "speech": "hello world",
                "rid": "hold_to_talk",
                "control_mode": "toggle",
                "voice": "Alex",
                "rate": 160,
            },
        )["action"]
        == "mic-speak"
    )

    names = {tool.name for tool in _tool_definitions()}
    assert {"mic_inject_and_analyze", "mic_speak_and_analyze"} <= names
    capability = next(item for item in capability_manifest() if item["id"] == "microphone")
    assert capability["mcp"] == "mic_inject_and_analyze"
    assert len(calls) == 3
    assert all(kwargs["control_mode"] == "toggle" for _text, kwargs in calls)


@pytest.mark.parametrize("modifier", [{"index": 0}, {"first": True}])
def test_mcp_rejects_selector_modifiers_on_numeric_mic_hold_id(
    modifier: dict[str, Any],
) -> None:
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())

    with pytest.raises(UsageError, match="cannot modify a numeric microphone control id"):
        mcp_dispatch(
            engine,
            "mic_inject_and_analyze",
            {"path": "unused.wav", "id": 12, **modifier},
        )


def test_mcp_direct_emulator_start_forwards_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_start(_avd: str | None, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True, "serial": "emulator-5554"}

    monkeypatch.setattr(emulator_mod, "start", fake_start)
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())

    result = mcp_dispatch(engine, "emulator_start", {"audio": True, "headless": True})

    schema = next(tool for tool in _tool_definitions() if tool.name == "emulator_start")
    assert schema.inputSchema["properties"]["audio"]["default"] is False
    assert result["serial"] == "emulator-5554"
    assert calls[0]["audio"] is True


def test_daemon_audio_timeout_covers_bounded_media_synthesis_and_hold(tmp_path: Path) -> None:
    longest = _write_wav(
        tmp_path / "longest.wav",
        sample_rate=1,
        frames=int(mic.MAX_WAV_DURATION_S),
    )

    inject_timeout = _mic_request_timeout(
        "mic_inject",
        {
            "wav_path": str(longest),
            "pre_roll_ms": 1_000,
            "post_roll_ms": 2_000,
        },
    )
    speak_timeout = _mic_request_timeout(
        "mic_speak",
        {"pre_roll_ms": 1_000, "post_roll_ms": 2_000},
    )

    assert inject_timeout == mic.MAX_WAV_DURATION_S + 3 + 60
    assert speak_timeout == (mic.SPEECH_SYNTHESIS_TIMEOUT_S + mic.MAX_WAV_DURATION_S + 3 + 60)
    assert inject_timeout > 300
    assert speak_timeout > inject_timeout


def test_mic_journal_redacts_speech_paths_and_summarizes_uncertain_observation(
    tmp_path: Path,
) -> None:
    private_speech = "a short private phrase"
    private_path = "/private/customer-name/voice-sample.wav"
    private_screen_text = "private transcript from the resulting screen"
    error = {
        "code": "mic_delivery_uncertain",
        "message": "samples may have arrived",
        "result": {
            "ok": False,
            "action": "mic-speak",
            "observation": {
                "elements": [{"id": 0, "text": private_screen_text}],
                "meta": {"known_screen": "voice-result"},
            },
        },
    }

    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="mcp",
        cmd="mic_speak",
        args={"speech": private_speech},
        ok=False,
        error=error,
    )
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="daemon",
        cmd="mic_inject",
        args={"wav_path": private_path},
        ok=True,
    )
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="daemon",
        cmd="mic_inject",
        args={"wav_path": private_path},
        ok=False,
        error={
            "code": "mic_wav_invalid",
            "message": f"cannot use WAV file '{private_path}': malformed",
        },
    )

    events = journal.read_since(tmp_path, "emulator-5554", limit=5)
    serialized = json.dumps(events)
    details = [
        journal.read_detail(tmp_path, "emulator-5554", event["detail_id"])
        for event in events
    ]
    serialized_details = json.dumps(details)
    assert private_speech not in serialized
    assert private_path not in serialized
    assert private_screen_text not in serialized
    assert private_speech not in serialized_details
    assert private_path not in serialized_details
    assert private_screen_text not in serialized_details
    assert events[0]["args"]["speech"] == f"<redacted speech: {len(private_speech)} chars>"
    assert journal.redact_args({"text": private_speech}, cmd="mic_speak")["text"] == (
        f"<redacted speech: {len(private_speech)} chars>"
    )
    assert events[0]["error"]["result"]["observation"]["elements_count"] == 1
    assert events[1]["args"]["wav_path"] == "<redacted audio path>"
    assert events[2]["error"]["message"] == "<redacted post-microphone text>"


def test_public_mic_implementation_and_docs_stay_app_agnostic() -> None:
    root = Path(__file__).parents[1]
    public_files = [
        root / "src/android_ui_analyser/mic.py",
        root / "src/android_ui_analyser/cli.py",
        root / "src/android_ui_analyser/guide.py",
        root / "README.md",
    ]
    forbidden = ("LanguageLearningApp", "the-wordlab.language", "com.wordlab")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_files)
    assert not any(term.casefold() in combined.casefold() for term in forbidden)
