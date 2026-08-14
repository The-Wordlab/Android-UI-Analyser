"""`app launch` is the first action of nearly every journey, so it must report the screen.

`_await_foreground` already proves the package reached the foreground, which is why a launch that
never happened cannot answer ok=True. What it did *not* do was say where the launch landed: the
response carried only `ok`/`detail`, so a caller spent a separate `analyze` to learn the screen and
had nothing structured to attach as evidence for the step. Every other action folds the post-action
screen in; this pins that launch does too, and that `--no-observe` still opts out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice
from test_memory import APPS, P, _engine


def test_launch_folds_in_the_screen_it_landed_on(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch")
    eng = _engine(tmp_path, dev)

    r = eng.app("launch", package=P)
    assert r.ok and r.action == "app-launch"
    assert r.observation_present is True, "launch must say whether it observed"
    assert r.observation is not None and r.observation.elements, "and return the screen"


def test_launch_can_opt_out_of_observing(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch2")
    eng = _engine(tmp_path, dev)

    r = eng.app("launch", package=P, observe=False)
    assert r.ok and r.observation is None
    # Opting out still answers the question rather than going silent.
    assert r.observation_present is False


def test_a_non_action_app_subcommand_stays_a_bare_answer(tmp_path: Path) -> None:
    # `current` is a query, not an action: it has no post-action screen to report, so it must not
    # claim one. Guards the boundary from the other side.
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch3")
    eng = _engine(tmp_path, dev)

    r = eng.app("current")
    assert r.observation_present is None
    assert r.observation is None


def test_detail_stays_the_launched_component_and_timing_is_structured(tmp_path: Path) -> None:
    """`detail` is a value, not a log line.

    Two markers used to be appended to it by the observe step — a `stale_risk` flag and a
    `settle=295ms via=pixels` timing tag — so an observed launch answered
    ``detail: "<pkg>/<activity> settle=295ms via=pixels"``. Anything parsing `detail` to learn what
    was launched got the timing glued on. Both now have their own fields.
    """
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch4")
    eng = _engine(tmp_path, dev)

    r = eng.app("launch", package=P)

    assert r.detail == P, f"detail must be the launched component alone, got {r.detail!r}"
    assert "settle=" not in (r.detail or ""), "timing must not be appended to detail"
    assert "stale_risk" not in (r.detail or ""), "caveats must not be appended to detail"
    if r.settle is not None:
        assert isinstance(r.settle, dict) and "ms" in r.settle, "settle is structured, not prose"


def test_launch_replaces_a_previous_package_hierarchy_with_one_authoritative_read(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch-authoritative")
    eng = _engine(tmp_path, dev)
    fresh = eng.analyze(source="hierarchy", with_ocr=False)
    stale = fresh.model_copy(deep=True)
    stale.screen.package = "com.example.previous"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        eng,
        "_observe",
        lambda *_args, **_kwargs: ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
        ),
    )
    monkeypatch.setattr(
        eng,
        "analyze",
        lambda **kwargs: calls.append(kwargs) or fresh,
    )

    result = eng.app("launch", package=P)

    assert result.observation is fresh
    assert calls == [{"source": "hierarchy", "with_ocr": False, "no_cache": True}]


def test_launch_refuses_a_persistently_mixed_package_hierarchy(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch-mismatch")
    eng = _engine(tmp_path, dev)
    stale = eng.analyze(source="hierarchy", with_ocr=False).model_copy(deep=True)
    stale.screen.package = "com.example.previous"
    monkeypatch.setattr(
        eng,
        "_observe",
        lambda *_args, **_kwargs: ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
        ),
    )
    monkeypatch.setattr(eng, "analyze", lambda **_kwargs: stale)

    with pytest.raises(DeviceError) as raised:
        eng.app("launch", package=P)

    assert raised.value.code == "launch_observation_mismatch"
