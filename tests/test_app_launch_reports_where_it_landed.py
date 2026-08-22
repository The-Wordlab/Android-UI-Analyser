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

from android_ui_analyser.cli import _daemon_error
from android_ui_analyser.errors import DeviceError, ExitCode
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice
from test_memory import APPS, P, _engine

SHELL_ONLY = (
    '<hierarchy rotation="0">'
    f'<node class="android.widget.LinearLayout" package="{P}" text="" '
    f'resource-id="{P}:id/action_bar_root" clickable="false" enabled="true" '
    'bounds="[0,0][1080,1920]">'
    f'<node class="android.view.View" package="{P}" text="" '
    'resource-id="android:id/content" clickable="false" enabled="true" '
    'bounds="[0,0][1080,1920]"/>'
    "</node></hierarchy>"
)


def test_launch_folds_in_the_screen_it_landed_on(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch")
    eng = _engine(tmp_path, dev)

    r = eng.app("launch", package=P)
    assert r.ok and r.action == "app-launch"
    assert r.observation_present is True, "launch must say whether it observed"
    assert r.observation is not None and r.observation.elements, "and return the screen"


def test_launch_waits_inside_the_same_call_for_content_after_a_shell_only_frame(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=SHELL_ONLY, package=P, serial="emu-launch-shell-heals")
    # The derived list is opt-in (`output.next_actions`); asked for here because "it was NOT
    # withheld" is half of what this test proves, and an absent field cannot say that.
    eng = _engine(tmp_path, dev, output={"next_actions": True})
    shell = eng.analyze(source="hierarchy", with_ocr=False)
    meaningful = shell.model_copy(deep=True)
    meaningful.elements[1].resource_id = f"{P}:id/catalog"
    meaningful.elements[1].text = "Catalog"
    meaningful.elements[1].clickable = True
    waited: list[Any] = []
    monkeypatch.setattr(eng, "_analyze_post_action", lambda *_args, **_kwargs: shell)
    monkeypatch.setattr(
        eng,
        "_await_meaningful_launch_observation",
        lambda initial: waited.append(initial) or (meaningful, 240),
    )

    result = eng.app("launch", package=P)

    assert waited == [shell]
    assert result.observation is meaningful
    assert result.stale_risk is None
    assert result.next_actions is not None
    assert result.note == "No separate analyze needed; state is in observation."
    assert result.settle is not None and result.settle["content_ms"] == 240


def test_launch_shell_timeout_is_explicitly_non_reusable(monkeypatch: Any, tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=SHELL_ONLY, package=P, serial="emu-launch-shell-stays")
    # Opted in, so `next_actions is None` below means "withheld" rather than "never emitted".
    eng = _engine(tmp_path, dev, output={"next_actions": True})
    shell = eng.analyze(source="hierarchy", with_ocr=False)
    monkeypatch.setattr(eng, "_analyze_post_action", lambda *_args, **_kwargs: shell)
    monkeypatch.setattr(
        eng,
        "_await_meaningful_launch_observation",
        lambda initial: (initial, 2_000),
    )

    result = eng.app("launch", package=P)

    assert result.observation is shell
    assert result.stale_risk and "transitional" in result.stale_risk
    assert result.observation.meta.stale_risk == result.stale_risk
    assert result.next_actions is None
    assert result.note and "wait-and-analyze --after-change" in result.note
    assert result.settle is not None and result.settle["content_ms"] == 2_000


def test_unattributed_launch_shell_is_bound_then_waited_in_the_same_call(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=SHELL_ONLY, package=P, serial="emu-launch-unattributed")
    eng = _engine(tmp_path, dev)
    shell = eng.analyze(source="hierarchy", with_ocr=False)
    shell.screen.package = None
    meaningful = shell.model_copy(deep=True)
    meaningful.screen.package = P
    meaningful.elements[1].resource_id = f"{P}:id/library"
    meaningful.elements[1].text = "Library"
    waited_packages: list[str | None] = []
    monkeypatch.setattr(eng, "_analyze_post_action", lambda *_args, **_kwargs: shell)
    monkeypatch.setattr(
        eng,
        "_await_meaningful_launch_observation",
        lambda initial: waited_packages.append(initial.screen.package) or (meaningful, 360),
    )

    result = eng.app("launch", package=P)

    assert waited_packages == [P], "foreground ownership must be bound before readiness polling"
    assert result.observation is meaningful
    assert result.stale_risk is None
    assert result.settle is not None and result.settle["content_ms"] == 360
    assert result.wall_ms is not None


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
    assert calls == [{"source": "hierarchy", "with_ocr": False, "no_cache": True, "record": False}]


def test_launch_retries_a_transient_systemui_hierarchy_while_app_stays_foreground(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch-attaching")
    # Opted in: "re-derived from the fresh frame, not the rejected one" needs a value to compare.
    eng = _engine(tmp_path, dev, output={"next_actions": True})
    target = eng.analyze(source="hierarchy", with_ocr=False)
    target.meta.known_screen = "target-login"
    target.meta.known_routes = ["target-route"]
    target.meta.suggested_gotos = ["target-goto"]
    target.meta.element_diff = {
        "added": ["target"],
        "removed": [],
        "changed": [],
        "prev_count": 0,
        "curr_count": len(target.elements),
    }
    systemui = target.model_copy(deep=True)
    systemui.screen.package = "com.android.systemui"
    systemui.meta.known_screen = "systemui-shade"
    systemui.meta.known_routes = ["systemui-route"]
    systemui.meta.suggested_gotos = ["systemui-goto"]
    systemui.meta.stale_risk = "systemui tree may be stale"
    for element in systemui.elements:
        element.id += 10_000
    samples = iter((systemui, target))
    stale = systemui.model_copy(deep=True)
    calls: list[dict[str, Any]] = []

    def observe_stale(*_args: Any, **_kwargs: Any) -> ActionResult:
        # Reproduce production `_observe`: the transient hierarchy became the authoritative id
        # cache before launch package validation rejected it.
        eng._write_cache(stale)
        return ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
            next_actions=[{"id": stale.elements[0].id, "label": "System UI"}],
            routes=["systemui-route", "systemui-goto"],
            known_screen="systemui-shade",
            action_diff_summary={"added": 99, "removed": 0, "changed": 0},
            change={"activity": {"after": "SystemUI"}},
            note="systemui note",
            stale_risk="systemui stale risk",
        )

    monkeypatch.setattr(eng, "_observe", observe_stale)
    monkeypatch.setattr(
        eng,
        "analyze",
        lambda **kwargs: calls.append(kwargs) or next(samples),
    )
    monkeypatch.setattr("android_ui_analyser.engine.time.sleep", lambda _seconds: None)

    result = eng.app("launch", package=P)

    assert result.observation is target
    cached = eng._read_cache()
    assert cached is not None
    assert cached.model_dump() == result.observation.model_dump()
    assert [element.id for element in cached.elements] == [
        element.id for element in result.observation.elements
    ]
    assert result.next_actions == eng._next_actions(target)
    assert result.routes == ["target-route", "target-goto"]
    assert result.known_screen == "target-login"
    assert result.action_diff_summary == eng._compact_action_diff(target.meta.element_diff)
    assert result.change is None
    assert result.stale_risk is None
    assert result.note == "No separate analyze needed; state is in observation."
    systemui_ids = {element.id for element in stale.elements}
    observed = result.observation.as_dict("json") if result.observation else {"elements": []}
    assert not systemui_ids.intersection(e["id"] for e in observed["elements"])
    assert "systemui-route" not in (result.routes or [])
    assert result.known_screen != "systemui-shade"
    assert calls == [
        {"source": "hierarchy", "with_ocr": False, "no_cache": True, "record": False},
        {"source": "hierarchy", "with_ocr": False, "no_cache": True, "record": False},
    ]


def test_launch_refuses_a_persistently_mixed_package_hierarchy(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch-mismatch")
    eng = _engine(tmp_path, dev)
    stale = eng.analyze(source="hierarchy", with_ocr=False).model_copy(deep=True)
    stale.screen.package = "com.example.previous"

    def observe_stale(*_args: Any, **_kwargs: Any) -> ActionResult:
        eng._write_cache(stale)
        return ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
        )

    monkeypatch.setattr(eng, "_observe", observe_stale)
    analyzes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        eng,
        "analyze",
        lambda **kwargs: analyzes.append(kwargs) or stale,
    )
    clock = [10.0]

    def tick() -> float:
        current = clock[0]
        clock[0] += 0.03
        return current

    monkeypatch.setattr("android_ui_analyser.engine.time.monotonic", tick)
    monkeypatch.setattr("android_ui_analyser.engine.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("android_ui_analyser.engine._LAUNCH_HIERARCHY_SETTLE_S", 0.05)

    with pytest.raises(DeviceError) as raised:
        eng.app("launch", package=P)

    assert raised.value.code == "launch_observation_mismatch"
    assert len(analyzes) == 2
    assert eng._read_cache() is None
    assert eng._last_analyze_elements is None
    assert eng._last_analyze_result is None


def test_launch_does_not_retry_mismatch_after_foreground_ownership_changes(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch-left")
    eng = _engine(tmp_path, dev)
    stale = eng.analyze(source="hierarchy", with_ocr=False).model_copy(deep=True)
    stale.screen.package = "com.android.systemui"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(eng, "_await_foreground", lambda *_args, **_kwargs: True)
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
        lambda **kwargs: calls.append(kwargs) or stale,
    )
    monkeypatch.setattr(
        dev,
        "current_app",
        lambda: {"package": "com.example.other", "activity": ".OtherActivity"},
    )

    with pytest.raises(DeviceError) as raised:
        eng.app("launch", package=P)

    assert raised.value.code == "launch_observation_mismatch"
    assert "ownership changed" in raised.value.message
    assert len(calls) == 1


def test_launch_does_not_attribute_an_empty_recovery_after_ownership_changes(
    monkeypatch: Any, tmp_path: Path
) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-launch-empty-left")
    eng = _engine(tmp_path, dev)
    stale = eng.analyze(source="hierarchy", with_ocr=False).model_copy(deep=True)
    stale.screen.package = "com.android.systemui"
    anonymous = stale.model_copy(deep=True)
    anonymous.screen.package = None
    monkeypatch.setattr(eng, "_await_foreground", lambda *_args, **_kwargs: True)

    def observe_stale(*_args: Any, **_kwargs: Any) -> ActionResult:
        eng._write_cache(stale)
        return ActionResult(
            ok=True,
            action="app-launch",
            observation=stale,
            observation_present=True,
        )

    monkeypatch.setattr(eng, "_observe", observe_stale)
    monkeypatch.setattr(eng, "analyze", lambda **_kwargs: anonymous)
    monkeypatch.setattr(
        dev,
        "current_app",
        lambda: {"package": "com.example.other", "activity": ".OtherActivity"},
    )

    with pytest.raises(DeviceError) as raised:
        eng.app("launch", package=P)

    assert raised.value.code == "launch_observation_mismatch"
    assert "no package attribution" in raised.value.message
    assert eng._read_cache() is None
    assert eng._last_analyze_elements is None
    assert eng._last_analyze_result is None


def test_daemon_reconstructs_launch_observation_mismatch_as_device_error() -> None:
    rebuilt = _daemon_error(
        {
            "code": "launch_observation_mismatch",
            "message": "launch hierarchy belonged to SystemUI",
            "hint": "inspect one fresh hierarchy",
        }
    )

    assert isinstance(rebuilt, DeviceError)
    assert rebuilt.code == "launch_observation_mismatch"
    assert rebuilt.exit_code == ExitCode.DEVICE
