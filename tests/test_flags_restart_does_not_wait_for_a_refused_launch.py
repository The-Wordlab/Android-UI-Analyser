"""A restart must not wait for an Activity the platform refused to start.

`flags set` pins an entry Activity so the app comes back where it was, and by default that
pin is *whatever was in the foreground* — which for a single-Activity app is an Activity that
is not exported. `am start -n` then refuses, `launch_app` raises, and the old code swallowed
the exception and still polled `_FLAGS_ENTRY_TIMEOUT_S` for a process that was never launched.

Every scenario sets flags, so that wait sat on the critical path of a whole sweep. The waste is
measured here as *device polls*, not wall clock: a timing assertion would be flaky on a loaded
host, and the number of `current_app()` calls is the thing that actually changed.
"""

from __future__ import annotations

import contextlib

import pytest

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.device import DeviceError


class _RestartDevice:
    """Minimal stand-in: refuses a pinned Activity, accepts the default launcher."""

    serial = "emu-restart"

    def __init__(self, *, refuse_pinned: bool = True, refuse_default: bool = False) -> None:
        self.refuse_pinned = refuse_pinned
        self.refuse_default = refuse_default
        self.foreground_polls = 0
        self.launches: list[str | None] = []
        self.stopped = 0
        self._up = False

    def stop_app(self, package: str) -> None:
        self.stopped += 1
        self._up = False

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        self.launches.append(activity)
        if activity is not None:
            if self.refuse_pinned:
                raise DeviceError(f"am start refused {package}/{activity}: Permission Denial")
            self._up = True
            return
        if self.refuse_default:
            raise DeviceError(f"am start refused {package}: Permission Denial")
        self._up = True

    def current_app(self) -> dict[str, str]:
        self.foreground_polls += 1
        return {"package": "com.example.app" if self._up else "com.android.launcher"}

    def wait_idle(self, ms: int) -> None:  # noqa: D102 - part of the surface _restart_app uses
        return None


def _restart(engine: engine_mod.Engine, device: _RestartDevice, monkeypatch):
    # `device` is a property on Engine, so patch the class rather than the instance.
    monkeypatch.setattr(engine_mod.Engine, "device", property(lambda self: device))
    return engine._restart_app("com.example.app", "com.example.app.MainActivity")


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch) -> engine_mod.Engine:
    eng = engine_mod.Engine.__new__(engine_mod.Engine)
    monkeypatch.setattr(
        engine_mod.Engine, "_acting", lambda self: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        engine_mod.Engine, "_foreground_activity", lambda self, pkg: "com.example.app.MainActivity"
    )
    return eng


def test_a_refused_pinned_activity_is_not_waited_for(engine: engine_mod.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: zero foreground polls are spent on the launch that never happened."""
    device = _RestartDevice(refuse_pinned=True)
    result = _restart(engine, device, monkeypatch)

    assert result.ok, result.error
    assert device.launches == ["com.example.app.MainActivity", None], device.launches
    # Before the fix this polled ~10 times (3.0s / 0.3s) waiting for the refused Activity,
    # then polled again for the fallback. Only the fallback's polls are legitimate.
    assert device.foreground_polls == 1, (
        f"{device.foreground_polls} foreground polls — a refused launch is still being waited for"
    )


def test_a_pinned_activity_that_starts_is_still_waited_for(engine: engine_mod.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard the guard: the fix must not skip the wait when the launch actually succeeded."""
    device = _RestartDevice(refuse_pinned=False)
    result = _restart(engine, device, monkeypatch)

    assert result.ok, result.error
    # Pinned entry came up, so the default launcher is never used.
    assert device.launches == ["com.example.app.MainActivity"], device.launches
    assert device.foreground_polls >= 1


def test_a_refused_fallback_reports_instead_of_waiting(engine: engine_mod.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """If even the default launcher refuses, say so — do not poll for six seconds first."""
    device = _RestartDevice(refuse_pinned=True, refuse_default=True)
    result = _restart(engine, device, monkeypatch)

    assert not result.ok
    assert "could not be relaunched" in (result.error or "")
    assert device.foreground_polls == 0, "nothing was launched, so nothing should be awaited"
