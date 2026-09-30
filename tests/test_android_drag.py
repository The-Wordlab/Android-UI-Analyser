"""Android drags with one held uiautomator2 touch: down, optional hold, moves, up."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_device
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.android_device import Uiautomator2Device
from conftest import make_config


def test_android_drags_with_one_held_touch_and_honours_the_hold(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    events: list[tuple] = []
    sleeps: list[float] = []
    touch = SimpleNamespace(
        down=lambda x, y: events.append(("down", x, y)),
        move=lambda x, y: events.append(("move", x, y)),
        up=lambda x, y: events.append(("up", x, y)),
    )
    device = object.__new__(Uiautomator2Device)
    device._d = SimpleNamespace(touch=touch)
    monkeypatch.setattr(android_device.time, "sleep", sleeps.append)

    device.drag(100, 900, 400, 300, 320, 700)

    assert events[0] == ("down", 100, 900)
    assert events[-1] == ("up", 400, 300) and events[-2] == ("move", 400, 300)
    assert len(events) > 6 and all(kind == "move" for kind, *_ in events[1:-1])
    assert sleeps[0] == pytest.approx(0.7)  # hold before the first move
    assert AndroidPlatform(make_config()).supports("device.drag")


def test_android_releases_the_pointer_when_a_move_fails() -> None:
    events: list[str] = []

    def move(x: int, y: int) -> None:
        raise RuntimeError("rpc lost")

    touch = SimpleNamespace(
        down=lambda x, y: events.append("down"),
        move=move,
        up=lambda x, y: events.append("up"),
    )
    device = object.__new__(Uiautomator2Device)
    device._d = SimpleNamespace(touch=touch)

    with pytest.raises(DeviceError):
        device.drag(0, 0, 10, 10, 100, 0)
    assert events == ["down", "up"]
