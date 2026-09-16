"""Fast text entry: set_text → clipboard paste → send_keys fallback."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.errors import DeviceError


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
    field = {"text": "old field content"}
    def set_text(value, **_kwargs):
        if value:
            raise RuntimeError("replacement SET_TEXT unavailable")
        field["text"] = ""

    focused.set_text.side_effect = set_text
    focused.get_text.side_effect = lambda: field["text"]
    u2 = MagicMock(return_value=focused)
    u2.clipboard = "previous-value"
    shells: list[str] = []
    clip_writes: list[str] = []

    def shell(cmd: str) -> str:
        shells.append(cmd)
        if cmd == "input keyevent 279":
            field["text"] = u2.clipboard
        return ""

    u2.shell = shell

    dev = _bare_device(u2)
    clip_reads: list[str] = []

    def get_clipboard() -> str:
        clip_reads.append(u2.clipboard)
        return str(u2.clipboard)

    dev.get_clipboard = get_clipboard  # type: ignore[method-assign]

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "set_clipboard":
            clip_writes.append(args[0])
            u2.clipboard = args[0]
            return None
        if name == "send_keys":
            raise AssertionError("send_keys must not run when paste works")
        raise AssertionError(name)

    dev._call = _call  # type: ignore[method-assign]
    dev.send_text("fast paste me", clear=True)

    assert "input keyevent 279" in shells
    assert "fast paste me" in clip_writes
    # The value that predated this lease is never restored into a paste overlay.  The transient
    # clipboard is emptied after the verified field update.
    assert "previous-value" not in clip_writes
    assert clip_reads == ["fast paste me"], "the pre-lease clipboard must never be read"
    assert clip_writes[-1] == ""
    assert u2.clipboard == ""
    assert field["text"] == "fast paste me"


def test_send_text_append_skips_set_text_uses_paste() -> None:
    focused = MagicMock()
    field = {"text": "hello"}
    focused.get_text.side_effect = lambda: field["text"]
    u2 = MagicMock(return_value=focused)
    u2.clipboard = ""
    shells: list[str] = []

    def shell(cmd: str) -> str:
        shells.append(cmd)
        if cmd == "input keyevent 279":
            field["text"] += u2.clipboard
        return ""

    u2.shell = shell

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
    assert field["text"] == "hello more"
    assert u2.clipboard == ""


def test_send_text_falls_back_to_send_keys_when_paste_fails() -> None:
    focused = MagicMock()
    def set_text(value, **_kwargs):
        if value:
            raise RuntimeError("replacement SET_TEXT unavailable")

    focused.set_text.side_effect = set_text
    focused.get_text.return_value = ""
    u2 = MagicMock(return_value=focused)
    u2.clipboard = ""

    def shell(_cmd: str) -> str:
        raise RuntimeError("paste keyevent failed")

    u2.shell = shell
    keys: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    dev = _bare_device(u2)

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "set_clipboard":
            raise RuntimeError("no clipboard")
        if name == "send_keys":
            keys.append((args, kwargs))
            return None
        raise AssertionError(name)

    dev._call = _call  # type: ignore[method-assign]
    dev.send_text("slow path", clear=True)
    assert keys == [(("slow path",), {"clear": False})]
    assert focused.set_text.call_args_list == [(('slow path',), {}), (('',), {"timeout": 1.0})]


def test_a_dispatched_but_unverified_paste_fails_closed_and_clears_clipboard() -> None:
    focused = MagicMock()
    field = {"text": ""}
    def set_text(value, **_kwargs):
        if value:
            raise RuntimeError("replacement SET_TEXT unavailable")
        field["text"] = ""

    focused.set_text.side_effect = set_text
    focused.get_text.side_effect = lambda: field["text"]
    u2 = MagicMock(return_value=focused)
    u2.clipboard = "unrelated"
    u2.shell = lambda _cmd: ""  # accepts KEYCODE_PASTE, but the field never changes
    calls: list[str] = []

    dev = _bare_device(u2)

    def _call(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        if name == "set_clipboard":
            u2.clipboard = args[0]
            return None
        if name == "send_keys":
            raise AssertionError("must not risk duplicating a paste that may have landed")
        raise AssertionError(name)

    dev._call = _call  # type: ignore[method-assign]

    with pytest.raises(DeviceError):
        dev.send_text("transient phrase", clear=True)

    assert "send_keys" not in calls
    assert u2.clipboard == ""


def _editable_info():
    return {"className": "android.widget.EditText", "resourceName": "example:id/composer",
            "packageName": "example", "enabled": True,
            "bounds": {"left": 0, "top": 20, "right": 100, "bottom": 60}}


def test_clear_refocuses_verified_editable_once_without_ime_broadcast_or_reconnect():
    focused = MagicMock()
    focused.info = _editable_info()
    focused.set_text.side_effect = [RuntimeError("ExtractedText.text null"), None]
    focused.get_text.return_value = ""
    dev = _bare_device(MagicMock(return_value=focused))
    dev._call = MagicMock(side_effect=AssertionError("no retrying IME fallback"))
    dev.clear_text()
    assert focused.set_text.call_args_list == [(('',), {"timeout": 1.0}), (('',), {"timeout": 1.0})]
    focused.click.assert_called_once_with(timeout=1.0)
    focused.get_text.assert_called_once_with(timeout=1.0)
    dev._call.assert_not_called()


@pytest.mark.parametrize("failure", ["noneditable", "changed", "second_clear", "nonempty", "unreadable"])
def test_clear_recovery_fails_closed_without_repeated_clear_or_input(failure):
    focused = MagicMock()
    info = _editable_info()
    focused.info = info
    focused.set_text.side_effect = [RuntimeError("clear unavailable"), None]
    focused.get_text.return_value = ""
    if failure == "noneditable":
        focused.info = {**info, "className": "android.widget.Button"}
    elif failure == "changed":
        focused.click.side_effect = lambda **_: setattr(focused, "info", {**info, "resourceName": "different"})
    elif failure == "second_clear":
        focused.set_text.side_effect = RuntimeError("ExtractedText.text null")
    elif failure == "nonempty":
        focused.get_text.return_value = "old value"
    else:
        focused.get_text.return_value = None
    dev = _bare_device(MagicMock(return_value=focused))
    dev._call = MagicMock(side_effect=AssertionError("no reconnect or broadcast"))
    with pytest.raises(DeviceError, match="after one semantic refocus"):
        dev.clear_text()
    assert focused.set_text.call_count == (1 if failure in {"noneditable", "changed"} else 2)
    assert focused.click.call_count == (0 if failure == "noneditable" else 1)
    dev._call.assert_not_called()


def test_replace_input_stops_after_one_failed_refocus_clear():
    focused = MagicMock()
    focused.info = _editable_info()
    focused.set_text.side_effect = RuntimeError("ExtractedText.text null")
    dev = _bare_device(MagicMock(return_value=focused))
    dev._call = MagicMock(side_effect=AssertionError("must not broadcast, reconnect or type again"))
    with pytest.raises(DeviceError, match="after one semantic refocus"):
        dev.send_text("new synthetic prompt", clear=True)
    assert focused.set_text.call_args_list == [
        (('new synthetic prompt',), {}), (('',), {"timeout": 1.0}), (('',), {"timeout": 1.0}),
    ]
    focused.click.assert_called_once_with(timeout=1.0)
    dev._call.assert_not_called()


def test_replace_input_refuses_fallback_if_empty_field_cannot_be_reconfirmed():
    focused = MagicMock()
    focused.set_text.side_effect = RuntimeError("replacement unavailable")
    focused.get_text.return_value = "new intervening content"
    dev = _bare_device(MagicMock(return_value=focused))
    dev._paste_via_clipboard = MagicMock(return_value=False)
    dev._call = MagicMock(side_effect=AssertionError("must not overwrite intervening content"))
    with pytest.raises(DeviceError, match="empty focused field"):
        dev.send_text("new synthetic prompt", clear=True)
    dev._call.assert_not_called()
