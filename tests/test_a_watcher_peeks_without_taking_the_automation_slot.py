"""The dashboard is a watcher: it may look at a device, never drive it.

Before ``ui.peek`` existed the grid answered "which app is in front?" and "what does the
screen look like?" by opening a fresh uiautomator2 session per tile per poll - installing and
starting the on-device server on a cold emulator (up to 30 s, inside the host-wide adb lock)
and taking the UiAutomation slot from whichever agent held the device. Plain ``adb`` answers
both questions in tens of milliseconds and touches no automation session at all.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.errors import DeviceError, UnsupportedPlatformCapabilityError
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms import android as android_mod
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.contracts import ADAPTER_CAPABILITIES
from android_ui_analyser.platforms.geometry import DisplayGeometry

_PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"

_WINDOW_DUMP = """WINDOW MANAGER DISPLAY CONTENTS (dumpsys window displays)
  mCurrentFocus=Window{4a3c1b2 u0 com.example.notes/com.example.notes.MainActivity}
  mFocusedApp=ActivityRecord{9f0 u0 com.example.notes/.MainActivity t42}
"""


class _NoPeekPlatform(PlatformAdapter):
    """A plugin that never declared ``ui.peek`` - the iOS/web shape."""

    name = "no-peek"
    capabilities = frozenset()

    def connect(self, target_id: str | None = None):  # pragma: no cover - never called
        raise AssertionError("a peek must not connect")

    def list_targets(self):  # pragma: no cover - never called
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:  # pragma: no cover - never called
        raise AssertionError("not used by this test")


def _android(monkeypatch: pytest.MonkeyPatch) -> tuple[AndroidPlatform, list[list[str]]]:
    platform = AndroidPlatform(Config())
    monkeypatch.setattr(AndroidPlatform, "prepare_host", lambda self: None)

    def refuse(self: AndroidPlatform, target_id: str | None = None) -> Any:
        raise AssertionError("a peek must never open an automation session")

    monkeypatch.setattr(AndroidPlatform, "connect", refuse)
    argv_seen: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        argv_seen.append(list(argv))
        if argv[-2:] == ["dumpsys", "window"]:
            return subprocess.CompletedProcess(argv, 0, stdout=_WINDOW_DUMP, stderr="")
        if argv[-3:] == ["exec-out", "screencap", "-p"]:
            return subprocess.CompletedProcess(argv, 0, stdout=_PNG, stderr=b"")
        raise AssertionError(f"unexpected native call {argv}")

    monkeypatch.setattr(android_mod.subprocess, "run", fake_run)
    return platform, argv_seen


def test_ui_peek_is_an_adapter_capability_android_declares() -> None:
    assert set(ADAPTER_CAPABILITIES["ui.peek"].members) == {
        "peek_foreground_app",
        "peek_screenshot",
    }
    assert "ui.peek" in AndroidPlatform.capabilities
    AndroidPlatform(Config()).validate_declared_capabilities()


def test_a_platform_without_ui_peek_refuses_at_the_gate() -> None:
    platform = _NoPeekPlatform(Config())
    with pytest.raises(UnsupportedPlatformCapabilityError):
        platform.adapter_capability("ui.peek")
    with pytest.raises(UnsupportedPlatformCapabilityError):
        platform.peek_foreground_app("simulator-1")
    with pytest.raises(UnsupportedPlatformCapabilityError):
        platform.peek_screenshot("simulator-1")


def test_android_peeks_the_foreground_app_through_adb_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, argv_seen = _android(monkeypatch)

    context = platform.adapter_capability("ui.peek").peek_foreground_app("emulator-5554")

    assert context is not None
    assert context.app_id == "com.example.notes"
    assert argv_seen == [["adb", "-s", "emulator-5554", "shell", "dumpsys", "window"]]


def test_android_peeks_the_screen_through_adb_screencap_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, argv_seen = _android(monkeypatch)

    image = platform.adapter_capability("ui.peek").peek_screenshot("emulator-5554")

    assert image.png_bytes == _PNG
    assert argv_seen == [["adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"]]


def test_android_peek_reports_a_missing_or_garbled_screencap_as_a_device_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, _argv = _android(monkeypatch)

    def broken(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(argv, 1, stdout=b"error: device offline", stderr=b"")

    monkeypatch.setattr(android_mod.subprocess, "run", broken)
    with pytest.raises(DeviceError):
        platform.peek_screenshot("emulator-5554")

    def slow(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))

    monkeypatch.setattr(android_mod.subprocess, "run", slow)
    with pytest.raises(DeviceError):
        platform.peek_screenshot("emulator-5554")
    assert platform.peek_foreground_app("emulator-5554") is None
