"""A cleared app must not be launched inside Android's deferred removed-task kill window."""

from __future__ import annotations

import subprocess

import pytest

from android_ui_analyser import device as device_mod
from android_ui_analyser.cli import _daemon_error
from android_ui_analyser.device import Uiautomator2Device, _activity_dump_mentions_package
from android_ui_analyser.errors import DeviceError

PKG = "com.example.lesson"


class _ClearRaceDevice:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str]] = []

    def app_clear(self, package: str) -> None:
        self.commands.append(("app_clear", package))

    def shell(self, command: str) -> str:
        self.commands.append(("shell", command))
        return "Starting: Intent"


def _wrapper(raw: _ClearRaceDevice) -> Uiautomator2Device:
    wrapper = object.__new__(Uiautomator2Device)
    wrapper._d = raw
    wrapper.serial = "emulator-fictional"
    return wrapper


def _activity_dump_result(output: str, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["adb"],
        returncode=returncode,
        stdout=output,
        stderr="",
    )


def _install_activity_dumps(
    monkeypatch: pytest.MonkeyPatch,
    raw: _ClearRaceDevice,
    outputs: list[str],
) -> list[tuple[list[str], float]]:
    pending = list(outputs)
    calls: list[tuple[list[str], float]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, float)
        calls.append((args, timeout))
        raw.commands.append(("activity_dump", str(len(calls))))
        output = pending.pop(0) if len(pending) > 1 else pending[0]
        return _activity_dump_result(output)

    monkeypatch.setattr(device_mod.subprocess, "run", run)
    return calls


def test_clear_waits_for_removed_task_before_a_following_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_task = (
        "ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities)\n"
        "* Task{a1b2c3 #80 type=standard A=10141:com.example.lesson U=0 visible=true}\n"
        "  topActivity=ActivityRecord{d4e5f6 com.example.browser/.BrowserActivity}"
    )
    settled = (
        "ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities)\n"
        "RootTask #1: com.android.launcher3/.Launcher"
    )
    raw = _ClearRaceDevice()
    wrapper = _wrapper(raw)
    subprocess_calls = _install_activity_dumps(monkeypatch, raw, [old_task, settled])
    monkeypatch.setattr(device_mod.time, "sleep", lambda _seconds: None)

    wrapper.clear_app(PKG)
    wrapper.launch_app(PKG, activity=".MainActivity")

    assert raw.commands == [
        ("app_clear", PKG),
        ("activity_dump", "1"),
        ("activity_dump", "2"),
        ("shell", f"am start -n {PKG}/.MainActivity"),
    ]
    assert [args for args, _timeout in subprocess_calls] == [
        [
            "adb",
            "-s",
            "emulator-fictional",
            "shell",
            "dumpsys",
            "activity",
            "activities",
        ],
    ] * 2
    assert all(0 < timeout <= 12.0 for _args, timeout in subprocess_calls)


def test_activity_dump_package_match_is_exact() -> None:
    assert _activity_dump_mentions_package(
        "Task{#80 A=10141:com.example.lesson U=0}",
        PKG,
    )
    assert not _activity_dump_mentions_package(
        "Task{#81 A=10142:com.example.lessons U=0}",
        PKG,
    )
    assert not _activity_dump_mentions_package(
        "Task{#82 A=10143:prefix.com.example.lesson U=0}",
        PKG,
    )


def test_clear_quiescence_timeout_says_the_wipe_already_happened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _ClearRaceDevice()
    wrapper = _wrapper(raw)
    active = (
        f"ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities)\nTask{{#80 A=10141:{PKG} U=0}}"
    )
    _install_activity_dumps(monkeypatch, raw, [active])
    times = iter((10.0, 10.0, 10.1))
    monkeypatch.setattr(device_mod.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(device_mod, "_APP_CLEAR_SETTLE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(device_mod.time, "sleep", lambda _seconds: None)

    with pytest.raises(DeviceError) as raised:
        wrapper.clear_app(PKG)

    assert raised.value.code == "app_clear_unsettled"
    assert "wipe already happened" in (raised.value.hint or "")
    assert "without --clear" in (raised.value.hint or "")
    assert raw.commands == [
        ("app_clear", PKG),
        ("activity_dump", "1"),
    ]


@pytest.mark.parametrize("output", ["", "Permission Denial: can't dump ActivityManager"])
def test_clear_rejects_unrecognizable_activity_dump(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
) -> None:
    raw = _ClearRaceDevice()
    wrapper = _wrapper(raw)
    _install_activity_dumps(monkeypatch, raw, [output])

    with pytest.raises(DeviceError) as raised:
        wrapper.clear_app(PKG)

    assert raised.value.code == "app_clear_unsettled"
    assert "recognizable successful dump" in raised.value.message
    assert "wipe already happened" in (raised.value.hint or "")


def test_clear_bounds_a_hung_activity_dump_by_the_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _ClearRaceDevice()
    wrapper = _wrapper(raw)
    seen: dict[str, object] = {}

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, float)
        seen["args"] = args
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(device_mod.subprocess, "run", run)
    monkeypatch.setattr(device_mod, "_APP_CLEAR_SETTLE_TIMEOUT_S", 0.25)

    with pytest.raises(DeviceError) as raised:
        wrapper.clear_app(PKG)

    assert raised.value.code == "app_clear_unsettled"
    assert "bounded activity-task read timed out" in raised.value.message
    assert seen["args"] == [
        "adb",
        "-s",
        "emulator-fictional",
        "shell",
        "dumpsys",
        "activity",
        "activities",
    ]
    timeout = seen["timeout"]
    assert isinstance(timeout, float) and 0 < timeout <= 0.25


def test_clear_rejects_a_nonzero_activity_dump_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _ClearRaceDevice()
    wrapper = _wrapper(raw)

    def run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _activity_dump_result("adb transport failed", returncode=17)

    monkeypatch.setattr(device_mod.subprocess, "run", run)

    with pytest.raises(DeviceError) as raised:
        wrapper.clear_app(PKG)

    assert raised.value.code == "app_clear_unsettled"
    assert "exit status 17" in raised.value.message


def test_daemon_reconstructs_clear_unsettled_as_a_device_failure() -> None:
    rebuilt = _daemon_error(
        {
            "code": "app_clear_unsettled",
            "message": "clear barrier failed",
            "hint": "launch without --clear",
        }
    )

    assert isinstance(rebuilt, DeviceError)
    assert rebuilt.code == "app_clear_unsettled"
    assert int(rebuilt.exit_code) == 3
