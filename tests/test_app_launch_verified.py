"""`app launch` must not claim success for an app that never came up.

uiautomator2's `app_start` runs `am start` and ignores its result, so a launch denied by the
platform still answered ok=True. Observed on a dev build: `am start` failed with
`SecurityException: Permission Denial ... not exported`, `aua app launch` returned
`{"ok":true,"action":"app-launch","detail":"<pkg>/<activity>"}`, and the launcher stayed in
front — after which every selector failed with an unrelated "no element matches", sending the
caller looking for a UI bug that did not exist.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from conftest import FakeDevice, make_config

PKG = "com.example.app"


class LaunchDevice(FakeDevice):
    """A device whose launch may or may not actually front the app."""

    def __init__(self, *, arrives: bool, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._arrives = arrives
        self._fronted = False

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        self.calls.append(("launch_app", (package, activity)))
        if self._arrives:
            self._fronted = True

    def current_app(self) -> dict[str, str]:  # type: ignore[override]
        return {"package": PKG if self._fronted else "com.android.launcher", "activity": ""}


def _engine(device: FakeDevice) -> Engine:
    return Engine(make_config(daemon={"enabled": False}), device=device)


def test_a_launch_that_arrives_is_reported_ok() -> None:
    dev = LaunchDevice(arrives=True)
    res = _engine(dev).app("launch", package=PKG)
    assert res.ok is True
    assert res.detail == PKG


def test_a_launch_that_never_fronts_the_app_is_an_error() -> None:
    """The bug: this used to answer ok=True with the launcher still in front."""
    eng = _engine(LaunchDevice(arrives=False))
    with pytest.raises(DeviceError) as err:
        eng.app("launch", package=PKG)
    assert "never reached the foreground" in str(err.value)


def test_a_pinned_activity_failure_names_the_likely_cause() -> None:
    """`--activity` is the common way to hit this, so the hint has to say so."""
    eng = _engine(LaunchDevice(arrives=False))
    with pytest.raises(DeviceError) as err:
        eng.app("launch", package=PKG, activity=".NotExported")
    assert "--activity" in (err.value.hint or "")


def test_the_launch_is_still_attempted_before_verifying() -> None:
    dev = LaunchDevice(arrives=False)
    with pytest.raises(DeviceError):
        _engine(dev).app("launch", package=PKG, activity=".Main")
    assert ("launch_app", (PKG, ".Main")) in dev.calls, "verification must not replace the action"


def test_arrival_polling_stops_as_soon_as_the_app_is_there() -> None:
    """A healthy launch must not pay the failure budget."""
    dev = LaunchDevice(arrives=True)
    seen = 0

    original = dev.current_app

    def counting() -> dict[str, str]:
        nonlocal seen
        seen += 1
        return original()

    dev.current_app = counting  # type: ignore[method-assign]
    _engine(dev).app("launch", package=PKG)
    assert seen == 1
