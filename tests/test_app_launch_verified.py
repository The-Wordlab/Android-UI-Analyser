"""`app launch` must not claim success for an app that never came up.

uiautomator2's `app_start` runs `am start` and ignores its result, so a launch denied by the
platform still answered ok=True. Observed on a dev build: `am start` failed with
`SecurityException: Permission Denial ... not exported`, `aua app launch` returned
`{"ok":true,"action":"app-launch","detail":"<pkg>/<activity>"}`, and the launcher stayed in
front — after which every selector failed with an unrelated "no element matches", sending the
caller looking for a UI bug that did not exist.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from conftest import FakeDevice, make_config

PKG = "com.example.app"


@pytest.fixture
def quick_foreground_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a launch that never arrives be refused without waiting out the real budget.

    `_await_foreground` polls for a generous 20s on purpose, and the test below pins that
    default. The refusal tests only assert *that* it refuses and what it says, so paying 20s of
    real wall clock each made three tests 60s of the suite's runtime. The poll loop itself still
    runs — only its deadline moves.
    """
    monkeypatch.setattr(
        Engine,
        "_await_foreground",
        staticmethod(functools.partial(Engine._await_foreground, timeout_ms=200)),
    )


def test_the_production_foreground_budget_stays_generous() -> None:
    """A cold start behind a long splash must not be mistaken for a launch that never happened.

    The refusal tests shrink this deadline to keep the suite quick, so the shipped value is
    asserted here instead — otherwise shrinking it in production would go unnoticed.
    """
    default = inspect.signature(Engine._await_foreground).parameters["timeout_ms"].default
    assert default == 20_000


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


def test_a_launch_that_never_fronts_the_app_is_an_error(quick_foreground_budget: None) -> None:
    """The bug: this used to answer ok=True with the launcher still in front."""
    eng = _engine(LaunchDevice(arrives=False))
    with pytest.raises(DeviceError) as err:
        eng.app("launch", package=PKG)
    assert "never reached the foreground" in str(err.value)


def test_a_pinned_activity_failure_names_the_likely_cause(quick_foreground_budget: None) -> None:
    """`--activity` is the common way to hit this, so the hint has to say so."""
    eng = _engine(LaunchDevice(arrives=False))
    with pytest.raises(DeviceError) as err:
        eng.app("launch", package=PKG, activity=".NotExported")
    assert "--activity" in (err.value.hint or "")


def test_the_launch_is_still_attempted_before_verifying(quick_foreground_budget: None) -> None:
    dev = LaunchDevice(arrives=False)
    with pytest.raises(DeviceError):
        _engine(dev).app("launch", package=PKG, activity=".Main")
    assert ("launch_app", (PKG, ".Main")) in dev.calls, "verification must not replace the action"


def _counting_current_app(dev: LaunchDevice) -> Callable[[], int]:
    """Count reads of the foreground app, so a polling loop is visible as a call count."""
    seen = 0
    original = dev.current_app

    def counting() -> dict[str, str]:
        nonlocal seen
        seen += 1
        return original()

    dev.current_app = counting  # type: ignore[method-assign]
    return lambda: seen


def test_arrival_polling_stops_as_soon_as_the_app_is_there() -> None:
    """A healthy launch must not pay the failure budget.

    Counted with ``observe=False`` so the number means only "how many times did arrival polling
    ask". A launch now also folds in the screen it landed on, and that observation legitimately
    reads the foreground once more — conflating the two would make this assert a total that says
    nothing about whether polling looped.
    """
    dev = LaunchDevice(arrives=True)
    count = _counting_current_app(dev)
    _engine(dev).app("launch", package=PKG, observe=False)
    assert count() == 1


def test_observing_the_landing_screen_does_not_reintroduce_polling() -> None:
    """The observation may cost reads; an unattributed tree needs one ownership proof."""
    dev = LaunchDevice(arrives=True)
    count = _counting_current_app(dev)
    _engine(dev).app("launch", package=PKG, observe=True)
    # One arrival check, the observation's own read, and at most one foreground recheck before an
    # otherwise useful hierarchy with no package attribution can be bound to the requested app.
    # A polling regression shows up here as a number that grows with the failure budget rather
    # than staying a small constant.
    assert count() <= 3, f"a healthy observed launch read the foreground {count()} times"
