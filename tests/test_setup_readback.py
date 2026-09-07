"""Setup must fail before app work when the native change or audio readiness is unproven."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import device_ledger, emulator, mic
from android_ui_analyser.engine import Engine
from android_ui_analyser.engine_apps import _recover_unpinned_launch
from android_ui_analyser.errors import DeviceError, UnsupportedPlatformCapabilityError, UsageError
from android_ui_analyser.platforms.android_device import Uiautomator2Device
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.runtime import TargetRuntime
from conftest import make_config
from test_mic import _endpoint_record


class _NeutralClock(TargetRuntime):
    target_id = "clock-target"

    def __init__(self) -> None:
        self.now = 1_700_000_000_000
        self.accept = True
        self.writes: list[int] = []

    def get_clock_ms(self) -> int:
        return self.now

    def utc_offset_minutes(self) -> int:
        return 0

    def set_clock(self, timestamp_ms: int) -> None:
        self.writes.append(timestamp_ms)
        if self.accept:
            self.now = timestamp_ms


class _NeutralPlatform(PlatformAdapter):
    name = "example"
    capabilities = frozenset({"device.clock"})

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        raise AssertionError("injected runtime must be used")

    def list_targets(self) -> list[Any]:
        raise AssertionError("setup test must not discover targets")

    def normalize_tree(self, *args: Any, **kwargs: Any) -> NormalizedTree:
        raise AssertionError("clock must not read a hierarchy")


def _clock_engine(tmp_path: Path) -> tuple[Engine, _NeutralClock]:
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, memory={"enabled": False})
    runtime = _NeutralClock()
    return Engine(cfg, device=runtime, platform=_NeutralPlatform(cfg)), runtime


def test_clock_noop_is_an_error_with_write_ahead_undo_retained(tmp_path: Path) -> None:
    engine, runtime = _clock_engine(tmp_path)
    runtime.accept = False
    with pytest.raises(DeviceError) as raised:
        engine.clock_set(timestamp_ms=runtime.now + 86_400_000)
    assert raised.value.code == "clock_write_unverified"
    assert engine._clock_backup_path().is_file()
    assert engine._pending_device_change("wall_clock", serial=runtime.target_id) is not None


@pytest.mark.parametrize("mode", ["explicit", "teardown"])
def test_denied_noop_clock_write_restores_by_readback_without_another_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    engine, runtime = _clock_engine(tmp_path)
    original = runtime.now

    def deny(timestamp_ms: int) -> None:
        runtime.writes.append(timestamp_ms)
        raise DeviceError("clock write denied", code="clock_write_failed")

    monkeypatch.setattr(runtime, "set_clock", deny)
    with pytest.raises(DeviceError, match="denied"):
        engine.clock_set(timestamp_ms=original + 86_400_000)
    pending = engine._pending_device_change("wall_clock", serial=runtime.target_id)
    assert pending is not None
    monkeypatch.setattr(time, "time", lambda: float(pending.args["saved_at"]) + 300)
    runtime.now += 300_000
    if mode == "explicit":
        assert engine.clock_set(restore=True).ok
        assert not engine._clock_backup_path().exists()
    else:
        result = device_ledger.replay(
            runtime.target_id,
            platform=engine.platform.name,
            context=device_ledger.UndoContext(
                serial=runtime.target_id,
                platform=engine.platform.name,
                device=runtime,
                runtime_capability=engine.platform.runtime_capability,
            ),
        )
        assert result["failed"] == [] and result["remaining"] == 0
    assert runtime.writes == [original + 86_400_000]
    assert engine._pending_device_change("wall_clock", serial=runtime.target_id) is None


@pytest.mark.parametrize("mode", ["explicit", "teardown"])
def test_unknown_clock_readback_keeps_restore_pending_after_an_unverifiable_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    engine, runtime = _clock_engine(tmp_path)
    engine.clock_set(timestamp_ms=runtime.now + 86_400_000)
    monkeypatch.setattr(runtime, "get_clock_ms", lambda: None)
    if mode == "explicit":
        with pytest.raises(DeviceError, match="readback"):
            engine.clock_set(restore=True)
        assert engine._clock_backup_path().exists()
    else:
        result = device_ledger.replay(
            runtime.target_id,
            platform=engine.platform.name,
            context=device_ledger.UndoContext(
                serial=runtime.target_id,
                platform=engine.platform.name,
                device=runtime,
                runtime_capability=engine.platform.runtime_capability,
            ),
        )
        assert result["failed"] and result["remaining"] == 1
    assert engine._pending_device_change("wall_clock", serial=runtime.target_id) is not None


def test_clock_restore_advances_original_time_and_only_forgets_after_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, runtime = _clock_engine(tmp_path)
    original = runtime.now
    engine.clock_set(timestamp_ms=original + 86_400_000)
    runtime.accept = False
    with pytest.raises(DeviceError, match="readback"):
        engine.clock_set(restore=True)
    assert engine._clock_backup_path().is_file()
    assert engine._pending_device_change("wall_clock", serial=runtime.target_id) is not None
    runtime.accept = True
    pending = engine._pending_device_change("wall_clock", serial=runtime.target_id)
    assert pending is not None
    monkeypatch.setattr(time, "time", lambda: float(pending.args["saved_at"]) + 300)
    engine.clock_set(restore=True)
    assert original + 300_000 <= runtime.now <= original + 302_000
    assert not engine._clock_backup_path().exists()
    assert engine._pending_device_change("wall_clock", serial=runtime.target_id) is None


@pytest.mark.parametrize("backup", ["missing", "stale"])
def test_clock_restore_uses_the_pending_ledger_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backup: str
) -> None:
    engine, runtime = _clock_engine(tmp_path)
    original = runtime.now
    engine.clock_set(timestamp_ms=original + 86_400_000)
    path = engine._clock_backup_path()
    if backup == "missing":
        path.unlink()
    else:
        path.write_text("1000", encoding="utf-8")
    pending = engine._pending_device_change("wall_clock", serial=runtime.target_id)
    assert pending is not None
    monkeypatch.setattr(time, "time", lambda: float(pending.args["saved_at"]) + 60)
    assert engine.clock_set(restore=True).ok
    assert original + 60_000 <= runtime.now <= original + 62_000


def test_a_new_clock_change_replaces_a_stale_backup_file(tmp_path: Path) -> None:
    engine, runtime = _clock_engine(tmp_path)
    path = engine._clock_backup_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1000", encoding="utf-8")
    original = runtime.now
    engine.clock_set(timestamp_ms=original + 86_400_000)
    assert path.read_text(encoding="utf-8") == str(original)


def test_clock_resolves_the_target_before_selecting_its_backup_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, runtime = _clock_engine(tmp_path)
    engine._device = None
    engine.config.device.serial = None
    selected: list[bool] = []

    def resolve(engine: Engine, capability: str) -> TargetRuntime:
        assert capability == "device.clock"
        engine._device = runtime
        selected.append(True)
        return runtime

    monkeypatch.setattr("android_ui_analyser.engine_environment._runtime_capability", resolve)
    engine.clock_set(timestamp_ms=runtime.now + 86_400_000)
    assert selected == [True]
    assert runtime.target_id in engine._clock_backup_path().name
    assert engine._clock_backup_path().is_file()
    assert not list(Path(engine.config.cache.dir).glob("*default*.txt"))


def test_teardown_noop_clock_restore_remains_a_failure() -> None:
    runtime = _NeutralClock()
    runtime.accept = False
    ctx = SimpleNamespace(require_device=lambda name: runtime)
    with pytest.raises(DeviceError, match="readback"):
        device_ledger._undo_set_clock(ctx, {"timestamp_ms": runtime.now + 86_400_000})


@pytest.mark.parametrize("exit_code,readback,expected", [(1, None, "clock_write_failed"), (0, 1_700_000_000_000, "clock_write_unverified")])
def test_android_clock_rejects_shell_refusal_and_false_success(
    exit_code: int, readback: int | None, expected: str
) -> None:
    runtime = object.__new__(Uiautomator2Device)
    runtime.serial = "emulator-fictional"
    commands: list[str] = []
    runtime._d = SimpleNamespace(
        shell=lambda command: commands.append(command)
        or SimpleNamespace(exit_code=exit_code, output=str(readback or "permission denied"))
    )
    with pytest.raises(DeviceError) as raised:
        runtime.set_clock(1_800_000_000_000)
    assert raised.value.code == expected
    assert commands[0].startswith("date -u ")


def test_audio_preflight_authenticates_without_opening_a_microphone_stream(tmp_path: Path) -> None:
    _endpoint_record(tmp_path, pid=os.getpid())
    calls: list[Any] = []
    channel = SimpleNamespace(
        unary_unary=lambda method, **kwargs: (
            calls.append(method) or (lambda payload, **options: calls.append((payload, options)))
        ),
        close=lambda: calls.append("closed"),
    )
    grpc = SimpleNamespace(insecure_channel=lambda *args, **kwargs: channel)
    result = mic.preflight("emulator-5554", running_dirs=[tmp_path], grpc_module=grpc)
    assert result["ok"] and result["authenticated"] and not result["input_sent"]
    assert calls[0].endswith("/getStatus")
    assert calls[1][0] == b""
    assert calls[-1] == "closed"
    assert "private-test-token" not in json.dumps(result)
    assert not list(tmp_path.glob("*.guard"))


def test_audio_preflight_auth_failure_is_typed_and_closes_without_sending_audio(
    tmp_path: Path,
) -> None:
    _endpoint_record(tmp_path, pid=os.getpid())
    closed: list[bool] = []

    class Unauthorized(Exception):
        def code(self) -> Any:
            return SimpleNamespace(name="UNAUTHENTICATED")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise Unauthorized("private-test-token")

    channel = SimpleNamespace(
        unary_unary=lambda *args, **kwargs: fail,
        close=lambda: closed.append(True),
    )
    with pytest.raises(DeviceError) as raised:
        mic.preflight(
            "emulator-5554",
            running_dirs=[tmp_path],
            grpc_module=SimpleNamespace(insecure_channel=lambda *args, **kwargs: channel),
        )
    assert raised.value.code == "mic_endpoint_auth_failed"
    assert "private-test-token" not in str(raised.value)
    assert closed == [True]


def test_launch_recovery_uses_only_the_selected_platform_runtime() -> None:
    operations: list[Any] = []
    runtime = SimpleNamespace(
        current_app=lambda: {"app_id": "example.app", "surface_id": ".Tools"},
        launcher_activities=lambda app_id: [".Tools", ".Main"],
        launch_app=lambda app_id, **kwargs: operations.append((app_id, kwargs)),
    )
    requested: list[str] = []
    adapter = SimpleNamespace(
        name="example",
        runtime_capability=lambda name, device: requested.append(name) or runtime,
    )
    fresh = object()
    engine = SimpleNamespace(
        platform=adapter,
        device=object(),
        _acting=nullcontext,
        _app_process_replaced=lambda app_id: None,
        _await_foreground=lambda device, app_id: True,
        _await_launch_hierarchy=lambda app_id: fresh,
    )
    assert _recover_unpinned_launch(engine, "example.app") == (fresh, ".Main")
    assert requested == ["app.lifecycle", "ui.tree"]
    assert operations == [("example.app", {"activity": ".Main"})]


def test_audio_preflight_missing_endpoint_fails_without_app_setup(tmp_path: Path) -> None:
    engine, _runtime = _clock_engine(tmp_path)
    # No target discovery/provisioning can occur, even if this test later changes to a fresh
    # engine. The only platform operation is the explicit fake microphone preflight.
    engine._prepare_session_target = lambda **kwargs: {"serial": "fake-audio"}  # type: ignore[method-assign]
    calls: list[str] = []

    def fail(target_id: str) -> dict[str, Any]:
        calls.append(target_id)
        raise DeviceError("endpoint missing", code="mic_endpoint_missing")

    engine.platform.capability = lambda name: SimpleNamespace(preflight=fail)  # type: ignore[method-assign]
    engine.install_app = lambda *args, **kwargs: pytest.fail("must preflight before install")  # type: ignore[method-assign]
    with pytest.raises(DeviceError) as raised:
        engine.session_start("check voice input", audio=True, apk="fictional.apk")
    assert raised.value.code == "mic_endpoint_missing"
    assert calls == ["fake-audio"]


def test_audio_start_on_an_unsupported_platform_is_typed_and_never_installs(tmp_path: Path) -> None:
    engine, _runtime = _clock_engine(tmp_path)
    engine._prepare_session_target = lambda **kwargs: {"serial": "fake-audio"}  # type: ignore[method-assign]
    engine.install_app = lambda *args, **kwargs: pytest.fail("must refuse before install")  # type: ignore[method-assign]
    with pytest.raises(UnsupportedPlatformCapabilityError) as raised:
        engine.session_start("check voice input", audio=True, apk="fictional.bundle")
    assert raised.value.code == "platform_capability_unsupported"


def test_audio_start_supplies_token_endpoint_and_preserves_explicit_port() -> None:
    args = emulator._audio_endpoint_args(None)
    assert "-grpc-use-token" in args
    assert 0 < int(args[args.index("-grpc") + 1]) < 65_536
    assert emulator._audio_endpoint_args(["-grpc", "18001"]) == [
        "-grpc", "18001", "-grpc-use-token"
    ]
    with pytest.raises(UsageError, match="conflicts"):
        emulator._audio_endpoint_args(["-no-audio"])
