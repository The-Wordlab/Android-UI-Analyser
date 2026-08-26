"""AndroidPlatform recovers one stale UiAutomation registration without core Android calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.android_runtime import (
    AndroidDeviceRuntime,
    stale_uiautomation_error,
)


class _StaleClient:
    def __init__(self) -> None:
        self.stop_calls = 0

    def dump_hierarchy(self, **_kwargs: Any) -> str:
        raise RuntimeError("java.lang.IllegalStateException: UiAutomation not connected")

    def stop_uiautomator(self) -> None:
        self.stop_calls += 1


class _HealthyClient:
    @staticmethod
    def dump_hierarchy(**_kwargs: Any) -> str:
        return "<hierarchy/>"


@pytest.mark.parametrize(
    "detail",
    [
        "java.lang.IllegalStateException: UiAutomation not connected",
        "UiAutomationService already registered",
        "UiAutomation already connected",
    ],
)
def test_only_known_stale_uiautomation_failures_trigger_reset(detail: str) -> None:
    assert stale_uiautomation_error(RuntimeError(detail)) is True
    assert stale_uiautomation_error(RuntimeError("device offline")) is False


def test_stale_runtime_is_killed_on_only_its_serial_then_retried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AndroidDeviceRuntime.__new__(AndroidDeviceRuntime)
    runtime.serial = "emulator-7777"
    stale = _StaleClient()
    runtime._d = stale
    runs: list[list[str]] = []
    reconnects: list[str] = []

    monkeypatch.setattr(
        "android_ui_analyser.platforms.android_runtime.subprocess.run",
        lambda command, **_kwargs: runs.append(command) or SimpleNamespace(returncode=0),
    )

    def reconnect() -> None:
        reconnects.append(runtime.serial)
        runtime._d = _HealthyClient()

    monkeypatch.setattr(runtime, "_connect", reconnect)

    result = runtime._call("dump_hierarchy", compressed=True)

    assert result == "<hierarchy/>"
    assert reconnects == ["emulator-7777"]
    assert stale.stop_calls == 1
    assert runs == [
        [
            "adb",
            "-s",
            "emulator-7777",
            "shell",
            "pkill",
            "-f",
            "com.wetest.uia2.Main",
        ]
    ]


def test_android_platform_constructs_the_android_owned_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = AndroidPlatform(Config())
    sentinel = object()
    monkeypatch.setattr(platform, "prepare_host", lambda: None)
    monkeypatch.setattr(
        "android_ui_analyser.device.resolve_serial",
        lambda target: "emulator-8888" if target is None else target,
    )
    monkeypatch.setattr(
        "android_ui_analyser.platforms.android_runtime.AndroidDeviceRuntime",
        lambda serial: sentinel if serial == "emulator-8888" else None,
    )

    assert platform.connect() is sentinel
