"""The connected-target contract is neutral while Android imports remain compatible."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import android_ui_analyser.device as legacy_device
from android_ui_analyser.config import Config
from android_ui_analyser.errors import ConfigError, DeviceError, InvalidPlatformCapabilityError
from android_ui_analyser.platforms.android_device import (
    AndroidRuntimeBase,
    Uiautomator2Device,
    parse_runtime_permissions,
)
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.runtime import TargetRuntime
from android_ui_analyser.providers.base import Bounds, ScreenImage
from android_ui_analyser.schema import AppContext, DeviceInfo, MatchMode


class _NeutralRuntime(TargetRuntime):
    target_id = "neutral-target"

    def window_size(self) -> tuple[int, int]:
        return (300, 600)

    def dump_hierarchy(self, compressed: bool = False) -> str:
        return "<tree/>"

    def screenshot(self) -> ScreenImage:  # pragma: no cover - geometry does not capture
        raise AssertionError("display_geometry must not capture a frame")

    def current_app(self) -> AppContext:
        return AppContext(app_id="example.app", surface_id="main")

    def click(self, x: int, y: int) -> None:
        return None

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        return None

    def touch_down(self, x: int, y: int) -> None:
        return None

    def touch_up(self, x: int, y: int) -> None:
        return None

    def send_text(self, text: str, *, clear: bool = True) -> None:
        return None

    def clear_text(self) -> None:
        return None

    def send_ime_action(self, action: str = "search") -> None:
        return None

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        return None

    def press(self, key: str) -> None:
        return None

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        return None


class _NeutralAdapter(PlatformAdapter):
    name = "neutral"

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        return _NeutralRuntime()

    def list_targets(self) -> list[DeviceInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: tuple[str, ...] = (),
    ) -> NormalizedTree:
        return NormalizedTree([])


def test_neutral_runtime_defaults_to_identity_screenshot_geometry() -> None:
    geometry = _NeutralRuntime().display_geometry()

    assert geometry.canonical_size == (300, 600)
    assert geometry.native_size == (300, 600)
    assert geometry.to_native((41, 82)) == (41, 82)


def test_partial_runtime_does_not_implement_unclaimed_capabilities() -> None:
    class DiscoveryOnlyRuntime(TargetRuntime):
        target_id = "discovery-only"

    runtime = DiscoveryOnlyRuntime()
    adapter = _NeutralAdapter(Config())

    assert adapter.validate_runtime(runtime) is runtime
    with pytest.raises(DeviceError, match="tap input is unsupported"):
        runtime.click(1, 2)


def test_target_id_is_required_and_serial_is_only_a_compatibility_projection() -> None:
    class MissingIdentity(TargetRuntime):
        pass

    adapter = _NeutralAdapter(Config())

    with pytest.raises(InvalidPlatformCapabilityError, match="target_id"):
        adapter.validate_runtime(MissingIdentity())

    runtime = _NeutralRuntime()

    assert runtime.target_id == "neutral-target"
    assert runtime.serial == runtime.target_id
    assert "serial" not in type(runtime).__dict__

    class DynamicIdentity(TargetRuntime):
        def __init__(self, target_id: str) -> None:
            self.target_id = target_id

    dynamic = DynamicIdentity("runtime-selected-target")

    assert adapter.validate_runtime(dynamic) is dynamic
    assert dynamic.serial == "runtime-selected-target"


def test_runtime_identity_must_be_a_non_empty_string() -> None:
    class EmptyIdentity(TargetRuntime):
        target_id = ""

    adapter = _NeutralAdapter(Config())

    with pytest.raises(InvalidPlatformCapabilityError) as exc:
        adapter.validate_runtime(EmptyIdentity())

    assert exc.value.code == "platform_capability_invalid"
    assert "non-empty string target_id" in exc.value.message


def test_neutral_runtime_does_not_offer_android_transport_fallbacks() -> None:
    runtime = _NeutralRuntime()

    assert not hasattr(runtime, "adb_reverse")
    assert not hasattr(runtime, "logcat")
    assert not hasattr(runtime, "shell")
    with pytest.raises(DeviceError, match="keyboard dismissal is unsupported"):
        runtime.hide_keyboard()


def test_historical_device_imports_are_compatibility_reexports() -> None:
    assert legacy_device.Device is AndroidRuntimeBase
    assert legacy_device.Uiautomator2Device is Uiautomator2Device
    assert issubclass(Uiautomator2Device, TargetRuntime)
    assert legacy_device._KEYCODE_NAMES["back"] == "KEYCODE_BACK"


def test_adapter_options_are_closed_by_default() -> None:
    adapter = _NeutralAdapter(Config())

    assert adapter.validate_options({}) == {}
    with pytest.raises(ConfigError, match="does not accept configuration options"):
        adapter.validate_options({"misspelled": True})


def test_claimed_runtime_capability_must_override_optional_stubs() -> None:
    class ClaimsLifecycle(_NeutralAdapter):
        capabilities = frozenset({"app.lifecycle"})

    adapter = ClaimsLifecycle(Config())

    with pytest.raises(InvalidPlatformCapabilityError) as exc:
        adapter.validate_runtime(_NeutralRuntime())

    assert exc.value.code == "platform_capability_invalid"
    assert "launch_app" in exc.value.message


def test_claimed_adapter_capability_must_override_optional_stubs() -> None:
    class ClaimsScreenshot(_NeutralAdapter):
        capabilities = frozenset({"ui.screenshot"})

    adapter = ClaimsScreenshot(Config())

    with pytest.raises(InvalidPlatformCapabilityError) as exc:
        adapter.validate_declared_capabilities()

    assert exc.value.code == "platform_capability_invalid"
    assert "capture_screenshot" in exc.value.message


def test_unknown_capability_declaration_is_rejected() -> None:
    class ClaimsUnknown(_NeutralAdapter):
        capabilities = frozenset({"device.telepathy"})

    with pytest.raises(InvalidPlatformCapabilityError) as exc:
        ClaimsUnknown(Config()).validate_declared_capabilities()

    assert exc.value.code == "platform_capability_invalid"
    assert "device.telepathy" in exc.value.message


def test_android_permission_snapshot_reads_only_the_runtime_permission_section() -> None:
    state = parse_runtime_permissions(
        """
    install permissions:
      android.permission.INTERNET: granted=true
      android.permission.FOREGROUND_SERVICE: granted=true
    runtime permissions:
      android.permission.CAMERA: granted=true, flags=[ USER_SET]
      android.permission.RECORD_AUDIO: granted=false, flags=[ USER_SENSITIVE_WHEN_GRANTED]
    Queries:
      system apps queryable: false
"""
    )

    assert state == {
        "android.permission.CAMERA": True,
        "android.permission.RECORD_AUDIO": False,
    }


def test_android_media_undo_refuses_to_complete_while_the_file_remains() -> None:
    runtime = object.__new__(Uiautomator2Device)
    runtime.serial = "android-runtime"
    runtime._d = SimpleNamespace(shell=lambda _command: "")
    runtime.shell = (  # type: ignore[method-assign]
        lambda command: "AUA_MEDIA_REMAINS" if "if [ -e" in command else ""
    )

    with pytest.raises(DeviceError) as raised:
        runtime.remove_added_media("/tmp/photo.png", remote_dir="/gallery")

    assert raised.value.code == "media_cleanup_unverified"


def test_android_recording_undo_refuses_to_complete_while_the_file_remains() -> None:
    runtime = object.__new__(Uiautomator2Device)
    runtime.serial = "android-runtime"
    runtime._live_recording = lambda: (True, None)  # type: ignore[method-assign]
    runtime.shell = (  # type: ignore[method-assign]
        lambda command: "AUA_RECORDING_REMAINS" if "if [ -e" in command else ""
    )

    with pytest.raises(DeviceError) as raised:
        runtime.discard_recording("/sdcard/recording.mp4")

    assert raised.value.code == "recording_cleanup_unverified"
