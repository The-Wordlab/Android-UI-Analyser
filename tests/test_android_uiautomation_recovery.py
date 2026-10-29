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
    sentinel = SimpleNamespace()
    monkeypatch.setattr(platform, "prepare_host", lambda: None)
    monkeypatch.setattr(
        "android_ui_analyser.device.resolve_serial",
        lambda target: "emulator-8888" if target is None else target,
    )
    monkeypatch.setattr(
        "android_ui_analyser.platforms.android_runtime.AndroidDeviceRuntime",
        lambda serial, **kwargs: (
            (setattr(sentinel, "_development_tools", kwargs["development_tools"]) or sentinel)
            if serial == "emulator-8888"
            else None
        ),
    )

    assert platform.connect() is sentinel
    assert sentinel._development_tools is platform.capability("development_tools")


def test_connection_releases_only_durably_owned_developer_helper_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser.device_ledger import Entry
    from android_ui_analyser.platforms.android_device import Uiautomator2Device

    runtime = AndroidDeviceRuntime.__new__(AndroidDeviceRuntime)
    runtime.serial = "fixture-target"
    events = []
    state = {"remove_installed_helper": True}
    from android_ui_analyser.device_ledger import ledger_dir
    from android_ui_analyser.platforms.options_transport import platform_options_fingerprint

    fingerprint = platform_options_fingerprint({}, key_dir=ledger_dir())
    entries = [
        Entry(key="unrelated", kind="other", op="restore_development_ui", args={"state": state}),
        Entry(
            key="development-tool-ui",
            platform_options_fingerprint=fingerprint,
            kind="development_tool_ui",
            op="restore_development_ui",
            args={"state": state},
        ),
    ]
    monkeypatch.setattr(
        "android_ui_analyser.device_ledger.read_ledger",
        lambda serial: entries if serial == runtime.serial else [],
    )
    monkeypatch.setattr(
        runtime,
        "restore_development_ui",
        lambda value: events.append(("restore", value)) or {"ok": True},
    )
    monkeypatch.setattr(
        Uiautomator2Device, "_connect", lambda self: events.append(("connect", self.serial))
    )
    runtime._connect()
    assert events == [("restore", state), ("connect", "fixture-target")]


def test_connection_never_stops_developer_helper_without_its_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser.platforms.android_device import Uiautomator2Device

    runtime = AndroidDeviceRuntime.__new__(AndroidDeviceRuntime)
    runtime.serial = "fixture-target"
    events = []
    monkeypatch.setattr("android_ui_analyser.device_ledger.read_ledger", lambda _: [])
    monkeypatch.setattr(
        runtime, "restore_development_ui", lambda _: pytest.fail("unowned helper touched")
    )
    monkeypatch.setattr(Uiautomator2Device, "_connect", lambda self: events.append(self.serial))
    runtime._connect()
    assert events == ["fixture-target"]


def test_connection_never_replays_helper_with_different_platform_options(monkeypatch):
    from android_ui_analyser.device_ledger import Entry
    from android_ui_analyser.errors import ConfigError

    runtime = AndroidDeviceRuntime.__new__(AndroidDeviceRuntime)
    runtime.serial = "fixture-target"
    entry = Entry(
        key="development-tool-ui",
        kind="development_tool_ui",
        op="restore_development_ui",
        args={"state": {"remove_installed_helper": True, "instance_token": "fixture-boot"}},
        platform_options_fingerprint="different-options",
    )
    monkeypatch.setattr("android_ui_analyser.device_ledger.read_ledger", lambda _: [entry])
    monkeypatch.setattr(
        runtime, "restore_development_ui", lambda _: pytest.fail("wrong options replayed")
    )
    with pytest.raises(ConfigError, match="different platform options"):
        runtime._connect()


def test_development_service_is_present_before_initial_connection(monkeypatch):
    service = object()
    seen = []
    monkeypatch.setattr(
        AndroidDeviceRuntime, "_connect", lambda self: seen.append(self._development_tools)
    )
    AndroidDeviceRuntime("fixture-target", development_tools=service)
    assert seen == [service]


def test_connection_does_not_proceed_after_unsuccessful_helper_restoration(monkeypatch):
    from android_ui_analyser.device_ledger import Entry, ledger_dir
    from android_ui_analyser.errors import ConfigError
    from android_ui_analyser.platforms.android_device import Uiautomator2Device
    from android_ui_analyser.platforms.options_transport import platform_options_fingerprint

    runtime = AndroidDeviceRuntime.__new__(AndroidDeviceRuntime)
    runtime.serial = "fixture-target"
    entry = Entry(
        key="development-tool-ui",
        kind="development_tool_ui",
        op="restore_development_ui",
        args={"state": {}},
        platform_options_fingerprint=platform_options_fingerprint({}, key_dir=ledger_dir()),
    )
    monkeypatch.setattr("android_ui_analyser.device_ledger.read_ledger", lambda _: [entry])
    monkeypatch.setattr(runtime, "restore_development_ui", lambda _: {"ok": False})
    monkeypatch.setattr(
        Uiautomator2Device, "_connect", lambda _: pytest.fail("connected after failed cleanup")
    )
    with pytest.raises(ConfigError, match="unsuccessful"):
        runtime._connect()
