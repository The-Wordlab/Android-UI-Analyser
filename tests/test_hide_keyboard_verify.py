"""`hide-keyboard` must report whether the keyboard actually went away.

It returned ok=True unconditionally. An IME that stays up is not a cosmetic miss: it covers
the bottom of the screen, so the button the caller is reaching for is hidden while the command
claims the keyboard is gone. Observed on device — `hide-keyboard` said ok while
`dumpsys input_method` still reported `mInputShown=true`.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config


class ImeDevice(FakeDevice):
    """A device whose IME state is scripted, and which may ignore the dismiss."""

    def __init__(self, *, shown: bool | None, obeys: bool = True, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._shown = shown
        self._obeys = obeys

    def hide_keyboard(self) -> None:  # type: ignore[override]
        self.calls.append(("hide_keyboard", ()))
        if self._obeys and self._shown is not None:
            self._shown = False

    def shell(self, cmd: str) -> str:  # type: ignore[override]
        if "mInputShown" in cmd:
            if self._shown is None:
                return ""  # device declines to say
            return f"    mInputShown={'true' if self._shown else 'false'}"
        return ""


def _engine(device: FakeDevice) -> Engine:
    return Engine(make_config(daemon={"enabled": False}), device=device)


def test_reports_hidden_when_the_ime_goes_away() -> None:
    eng = _engine(ImeDevice(shown=True, obeys=True))
    res = eng.hide_keyboard(observe=False)
    assert res.ok is True
    assert res.detail == "hidden"


def test_reports_still_shown_when_the_ime_ignores_the_dismiss() -> None:
    """The bug: this used to answer ok=True with the keyboard still covering the screen."""
    eng = _engine(ImeDevice(shown=True, obeys=False))
    res = eng.hide_keyboard(observe=False)
    assert res.detail == "still-shown"
    assert res.ok is False, "a keyboard that is still up is not a success"


def test_unknown_is_not_reported_as_hidden() -> None:
    """Tri-state: "cannot tell" must not read as success, or the check recreates the bug."""
    eng = _engine(ImeDevice(shown=None))
    res = eng.hide_keyboard(observe=False)
    assert res.detail == "unknown"


@pytest.mark.parametrize("shown", [True, False, None])
def test_the_dismiss_is_always_attempted(shown: bool | None) -> None:
    dev = ImeDevice(shown=shown)
    _engine(dev).hide_keyboard(observe=False)
    assert ("hide_keyboard", ()) in dev.calls, "verification must not replace the action"
