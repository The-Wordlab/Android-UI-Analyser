"""Foreground activity reads stay fast while retaining the u2 compatibility fallback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from android_ui_analyser import device as device_mod
from android_ui_analyser.device import Uiautomator2Device, _foreground_from_window_dump
from android_ui_analyser.schema import AppContext


def _device() -> Uiautomator2Device:
    device = object.__new__(Uiautomator2Device)
    device.serial = "fictional-5554"
    device._d = None
    device._winsize = None
    device._recording_remote = None
    device._recording_proc = None
    return device


def test_focused_component_parser_supports_current_focus_and_relative_activity() -> None:
    output = (
        "mCurrentFocus=Window{abc u0 com.example.notes/.MainActivity}\n"
        "mFocusedApp=ActivityRecord{def u0 com.example.other/.OtherActivity t42}\n"
    )

    assert _foreground_from_window_dump(output) == {
        "package": "com.example.notes",
        "activity": "com.example.notes.MainActivity",
    }


def test_current_app_uses_fast_window_query_without_calling_u2(monkeypatch: Any) -> None:
    dev = _device()
    monkeypatch.setattr(
        device_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "mCurrentFocus=Window{abc u0 "
                "com.example.notes/com.example.notes.MainActivity}\n"
            ),
        ),
    )
    dev._call = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("u2 fallback must not run")
    )

    assert dev.current_app() == AppContext(
        app_id="com.example.notes",
        surface_id="com.example.notes.MainActivity",
    )


def test_current_app_falls_back_to_u2_when_window_has_no_component(monkeypatch: Any) -> None:
    dev = _device()
    monkeypatch.setattr(
        device_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="mCurrentFocus=null\n"),
    )
    dev._call = lambda name, *_args, **_kwargs: {  # type: ignore[method-assign]
        "package": "com.example.fallback",
        "activity": ".FallbackActivity",
    }

    assert dev.current_app() == AppContext(
        app_id="com.example.fallback",
        surface_id=".FallbackActivity",
    )
