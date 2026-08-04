"""Fast text entry: set_text → clipboard paste → send_keys fallback."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from android_ui_analyser.device import Uiautomator2Device


def _bare_device(u2: Any) -> Uiautomator2Device:
    """Uiautomator2Device without connecting to a real emulator."""
    dev = object.__new__(Uiautomator2Device)
    dev.serial = "emulator-5554"
    dev._d = u2
    dev._winsize = (1080, 2400)
    return dev


def test_send_text_prefers_set_text() -> None:
    focused = MagicMock()
    u2 = MagicMock(return_value=focused)
    calls: list[str] = []

    dev = _bare_device(u2)

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        raise AssertionError(f"should not fall back to {name}")

    dev._call = _call  # type: ignore[method-assign]
    dev.send_text("hello world", clear=True)
    focused.set_text.assert_called_once_with("hello world")
    assert calls == []


def test_send_text_clipboard_paste_when_set_text_fails() -> None:
    focused = MagicMock()
    focused.set_text.side_effect = RuntimeError("no a11y SET_TEXT")
    u2 = MagicMock(return_value=focused)
    u2.clipboard = "previous-value"
    shells: list[str] = []
    clip_writes: list[str] = []

    def shell(cmd: str) -> str:
        shells.append(cmd)
        return ""

    u2.shell = shell

    dev = _bare_device(u2)

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "set_clipboard":
            clip_writes.append(args[0])
            u2.clipboard = args[0]
            return None
        if name == "clear_text":
            return None
        if name == "send_keys":
            raise AssertionError("send_keys must not run when paste works")
        raise AssertionError(name)

    dev._call = _call  # type: ignore[method-assign]
    # clear_text tries set_text("") first — that also raises; force clear via _call path
    focused.set_text.side_effect = RuntimeError("no a11y SET_TEXT")

    dev.send_text("fast paste me", clear=True)

    assert "input keyevent 279" in shells
    assert "fast paste me" in clip_writes
    # Restored after paste
    assert clip_writes[-1] == "previous-value"
    assert u2.clipboard == "previous-value"


def test_send_text_append_skips_set_text_uses_paste() -> None:
    focused = MagicMock()
    u2 = MagicMock(return_value=focused)
    u2.clipboard = ""
    shells: list[str] = []
    u2.shell = lambda cmd: shells.append(cmd) or ""

    dev = _bare_device(u2)

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "set_clipboard":
            u2.clipboard = args[0]
            return None
        raise AssertionError(f"unexpected {name}")

    dev._call = _call  # type: ignore[method-assign]
    dev.send_text(" more", clear=False)

    focused.set_text.assert_not_called()
    assert "input keyevent 279" in shells


def test_send_text_falls_back_to_send_keys_when_paste_fails() -> None:
    focused = MagicMock()
    focused.set_text.side_effect = RuntimeError("no set_text")
    u2 = MagicMock(return_value=focused)
    u2.clipboard = ""

    def shell(_cmd: str) -> str:
        raise RuntimeError("paste keyevent failed")

    u2.shell = shell
    keys: list[tuple[Any, ...]] = []

    dev = _bare_device(u2)

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "set_clipboard":
            raise RuntimeError("no clipboard")
        if name == "clear_text":
            return None
        if name == "send_keys":
            keys.append(args)
            return None
        raise AssertionError(name)

    dev._call = _call  # type: ignore[method-assign]
    # clear_text: set_text("") also fails → _call clear_text
    focused.set_text.side_effect = RuntimeError("no set_text")

    dev.send_text("slow path", clear=True)
    assert keys == [("slow path",)]
