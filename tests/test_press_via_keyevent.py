"""`key` must not pay uiautomator2's press overhead.

Measured on one u2 connection to a headless emulator: `shell("input keyevent KEYCODE_BACK")`
~103 ms, `press("back")` ~1125 ms — a 10x difference for the identical keystroke. A `key back`
ends most navigation steps, so that was ~1 s of pure tax on every step of an agent loop.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.device import _KEYCODE_NAMES, _PRESS_ALIASES, Uiautomator2Device
from android_ui_analyser.errors import DeviceError


class RecordingDevice(Uiautomator2Device):
    """Records which path a press took, without touching a device."""

    def __init__(self, *, shell_fails: bool = False) -> None:
        self.shells: list[str] = []
        self.u2_calls: list[tuple[str, tuple]] = []
        self._shell_fails = shell_fails

    def shell(self, command: str) -> str:  # type: ignore[override]
        if self._shell_fails:
            raise DeviceError("no shell here")
        self.shells.append(command)
        return ""

    def _call(self, method: str, *args: object) -> None:  # type: ignore[override]
        self.u2_calls.append((method, args))


@pytest.mark.parametrize("alias,keycode", sorted(_KEYCODE_NAMES.items()))
def test_every_named_key_goes_through_input_keyevent(alias: str, keycode: str) -> None:
    dev = RecordingDevice()
    dev.press(alias)
    assert dev.shells == [f"input keyevent {keycode}"]
    assert dev.u2_calls == [], "the slow path must not also run"


def test_raw_keycode_is_passed_straight_through() -> None:
    dev = RecordingDevice()
    dev.press("keycode_dpad_down")
    assert dev.shells == ["input keyevent KEYCODE_DPAD_DOWN"]


def test_an_unmapped_key_still_reaches_the_device() -> None:
    """Speed must not cost coverage: no keycode name means the old path, not an error."""
    dev = RecordingDevice()
    dev.press("wakeup")
    assert dev.shells == []
    assert dev.u2_calls == [("press", ("wakeup",))]


def test_a_device_without_shell_falls_back_rather_than_dropping_the_key() -> None:
    dev = RecordingDevice(shell_fails=True)
    dev.press("back")
    assert dev.u2_calls == [("press", ("back",))], "the keystroke must still be delivered"


def test_the_two_alias_tables_cover_the_same_keys() -> None:
    """Drift here would silently route some keys back onto the 1.1 s path."""
    assert set(_KEYCODE_NAMES) == set(_PRESS_ALIASES)
