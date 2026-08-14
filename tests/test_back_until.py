"""Bounded semantic Back navigation replaces repeated frame-local Back taps."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.schema import ActionResult, AnalyzeResult, Element, Meta, Screen
from conftest import FakeDevice, make_config

runner = CliRunner()


def _observation(
    serial: str = "back-until",
    *,
    elements: list[Element] | None = None,
    package: str = "com.example.app",
    known_screen: str = "destination",
) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=package, source="hierarchy"),
        elements=elements or [],
        meta=Meta(
            duration_ms=10,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen=known_screen,
            device_serial=serial,
        ),
    )


def _await(*, ok: bool, outcome: str, observation: AnalyzeResult | None = None) -> ActionResult:
    return ActionResult(
        ok=ok,
        action="await",
        detail=outcome,
        await_outcome=outcome,
        observation=observation or _observation(),
        observation_present=True,
    )


def test_back_until_stops_at_first_satisfied_destination(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    results = [
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="thread")),
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="detail")),
        _await(ok=True, outcome="satisfied", observation=_observation(known_screen="home")),
    ]
    keys: list[str] = []
    awaits: list[dict[str, Any]] = []

    def awaited(*_args: Any, **kwargs: Any) -> ActionResult:
        awaits.append(kwargs)
        return results.pop(0)

    monkeypatch.setattr(engine, "await_predicate", awaited)
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("rid:bottomNav", max_steps=5, step_timeout_ms=0)

    assert result.ok is True
    assert result.action == "back-until"
    assert result.detail == "satisfied after 2 back-navigation step(s)"
    assert result.observation is not None
    assert keys == ["back", "back"]
    assert awaits
    assert all(call["rich_ui"] is False for call in awaits)
    assert all(call["hierarchy_only"] is True for call in awaits)


def test_back_until_accepts_bare_mapped_screen_without_a_failed_probe(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    results = [
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="thread")),
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="detail")),
        _await(ok=True, outcome="satisfied", observation=_observation(known_screen="home")),
    ]
    targets: list[str] = []
    keys: list[str] = []

    def known(target: str, **_kwargs: Any) -> ActionResult:
        targets.append(target)
        return results.pop(0)

    monkeypatch.setattr(engine, "_await_known_screen", known)
    monkeypatch.setattr(
        engine,
        "await_predicate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bare mapped screen must not enter generic predicate parsing")
        ),
    )
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("home", max_steps=2, step_timeout_ms=0)

    assert result.ok is True
    assert targets == ["home", "home", "home"]
    assert keys == ["back", "back"]


def test_known_screen_wait_uses_read_only_map_recognition(monkeypatch: Any) -> None:
    engine = Engine(make_config(), device=FakeDevice(package="com.example.app"))
    observations = [
        _observation(known_screen=""),
        _observation(known_screen=""),
    ]
    recognized = iter(["detail", "home"])
    analyze_calls: list[dict[str, Any]] = []

    class Memory:
        def load(self, package: str) -> Any:
            assert package == "com.example.app"
            return type("MappedApp", (), {"screens": {"home": object()}})()

        def recognize_screen(self, *_args: Any, **_kwargs: Any) -> str:
            return next(recognized)

    engine._mem = Memory()  # type: ignore[assignment]
    monkeypatch.setattr(
        engine,
        "analyze",
        lambda **kwargs: analyze_calls.append(kwargs) or observations.pop(0),
    )
    monkeypatch.setattr(engine_mod.time, "sleep", lambda _seconds: None)

    result = engine._await_known_screen("home", timeout_ms=100, poll_ms=10)

    assert result.ok is True
    assert result.observation is not None
    assert result.observation.meta.known_screen == "home"
    assert analyze_calls == [
        {"source": "hierarchy", "with_ocr": False, "record": False},
        {"source": "hierarchy", "with_ocr": False, "record": False},
    ]


def test_back_until_stops_on_unrecognized_mapped_frame_instead_of_overshooting(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    results = [
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="detail")),
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="")),
    ]
    keys: list[str] = []
    monkeypatch.setattr(engine, "_await_known_screen", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("home", max_steps=4, step_timeout_ms=0)

    assert result.ok is False
    assert result.stop_reason == "screen_unrecognized"
    assert keys == ["back"]
    assert result.steps_run and len(result.steps_run) == 1


def test_back_until_rejects_unknown_mapped_screen_before_navigation(monkeypatch: Any) -> None:
    engine = Engine(make_config(), device=FakeDevice(package="com.example.app"))

    class Memory:
        def load(self, package: str) -> Any:
            assert package == "com.example.app"
            return type("MappedApp", (), {"screens": {"home": object()}})()

    engine._mem = Memory()  # type: ignore[assignment]
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation(known_screen=""))
    monkeypatch.setattr(
        engine,
        "key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not navigate")),
    )

    with pytest.raises(UsageError, match="not a mapped screen"):
        engine.back_until("typo_destination")


def test_back_until_requires_map_for_bare_screen_before_navigation(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation(known_screen="home"))
    monkeypatch.setattr(
        engine,
        "key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not navigate")),
    )

    with pytest.raises(UsageError, match="not a mapped screen"):
        engine.back_until("home")


def test_back_until_stops_on_mapped_loading_frame_before_another_back(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    results = [
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="detail")),
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="loading")),
    ]
    keys: list[str] = []
    monkeypatch.setattr(engine, "_await_known_screen", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(engine, "_mapped_screen_state", lambda _observation: "loading")
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("home", max_steps=4, step_timeout_ms=0)

    assert result.ok is False
    assert result.stop_reason == "screen_unstable"
    assert keys == ["back"]
    assert result.steps_run and len(result.steps_run) == 1


def test_back_until_re_resolves_toolbar_back_on_each_fresh_frame(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    back = Element(
        id=22,
        type="Button",
        resource_id="com.example.app:id/buttonNavBack",
        clickable=True,
        enabled=True,
        bounds=(0, 0, 100, 100),
        center=(50, 50),
    )
    thread_label = Element(
        id=30,
        type="TextView",
        text="Thread",
        bounds=(100, 200, 400, 300),
        center=(250, 250),
    )
    detail_label = thread_label.model_copy(update={"id": 31, "text": "Detail"})
    results = [
        ActionResult(
            ok=False,
            action="await",
            observation=_observation(elements=[back, thread_label]),
            observation_present=True,
        ),
        ActionResult(
            ok=False,
            action="await",
            observation=_observation(elements=[back.model_copy(update={"id": 7}), detail_label]),
            observation_present=True,
        ),
        _await(ok=True, outcome="satisfied"),
    ]
    selectors: list[dict[str, Any]] = []
    fast: list[bool] = []
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "tap",
        lambda *args, **kwargs: (
            selectors.append(kwargs["selector"])
            or fast.append(kwargs["_hierarchy_settle"])
            or ActionResult(ok=True, action="tap")
        ),
    )
    monkeypatch.setattr(
        engine,
        "key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("toolbar Back exists")),
    )

    result = engine.back_until("text:Home", max_steps=4, step_timeout_ms=0)

    assert result.ok is True
    assert selectors == [
        {"rid": "com.example.app:id/buttonNavBack"},
        {"rid": "com.example.app:id/buttonNavBack"},
    ]
    assert fast == [True, True]


def test_back_until_uses_explicit_fresh_unlabeled_id_then_semantic_controls(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    back = Element(
        id=22,
        type="View",
        clickable=True,
        bounds=(21, 147, 147, 273),
        center=(84, 210),
        parent=0,
        stable_key="geo:View:q10:fictional-back",
        window="app",
    )
    thread_label = Element(
        id=30,
        type="TextView",
        text="Thread",
        bounds=(100, 300, 400, 400),
        center=(250, 350),
    )
    results = [
        _await(
            ok=False,
            outcome="timeout",
            observation=_observation(elements=[back, thread_label]),
        ),
        _await(
            ok=False,
            outcome="timeout",
            observation=_observation(
                elements=[
                    back.model_copy(update={"id": 7}),
                    thread_label.model_copy(update={"id": 31, "text": "Detail"}),
                ]
            ),
        ),
        _await(ok=True, outcome="satisfied", observation=_observation(known_screen="home")),
    ]
    # Only the first unlabeled Back is authorized by fresh id. The next frame must resolve
    # independently from a semantic selector.
    results[1].observation.elements[0] = back.model_copy(
        update={"id": 7, "resource_id": "com.example.app:id/buttonNavBack"}
    )
    calls: list[tuple[int | None, dict[str, str] | None]] = []
    monkeypatch.setattr(
        engine,
        "_read_cache",
        lambda: _observation(elements=[back, thread_label]),
    )
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "tap",
        lambda element_id=None, selector=None, **_kwargs: (
            calls.append((element_id, selector)) or ActionResult(ok=True, action="tap")
        ),
    )
    monkeypatch.setattr(
        engine,
        "key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("app Back exists")),
    )

    result = engine.back_until("text:Home", back_id=22)

    assert result.ok is True
    assert calls == [(22, None), (None, {"rid": "com.example.app:id/buttonNavBack"})]
    assert result.steps_run
    assert [step["selector"] for step in result.steps_run] == [
        {"frame_id": "22"},
        {"rid": "com.example.app:id/buttonNavBack"},
    ]


def test_back_until_rechecks_predicate_after_activity_change_before_next_action(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    results = [
        _await(ok=False, outcome="timeout"),
        _await(ok=False, outcome="screen-changed"),
        _await(ok=True, outcome="satisfied"),
    ]
    keys: list[str] = []
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("text:Home", max_steps=4, step_timeout_ms=0)

    assert result.ok is True
    assert keys == ["back"]
    assert results == []


def test_back_until_rechecks_consecutive_activity_changes_without_overshooting(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    results = [
        _await(ok=False, outcome="timeout", observation=_observation(known_screen="thread")),
        _await(
            ok=False,
            outcome="screen-changed",
            observation=_observation(known_screen="transition-a"),
        ),
        _await(
            ok=False,
            outcome="screen-changed",
            observation=_observation(known_screen="transition-b"),
        ),
        _await(ok=True, outcome="satisfied", observation=_observation(known_screen="home")),
    ]
    keys: list[str] = []
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("text:Home", step_timeout_ms=1_000)

    assert result.ok is True
    assert keys == ["back"]
    assert result.steps_run and len(result.steps_run) == 1


def test_back_until_stops_on_delayed_package_change(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    changed = ActionResult(
        ok=False,
        action="await",
        await_outcome="screen-changed",
        observation=_observation(package="com.android.launcher", known_screen="launcher"),
        observation_present=True,
    )
    results = [_await(ok=False, outcome="timeout"), changed]
    keys: list[str] = []
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("text:Home", max_steps=4, step_timeout_ms=0)

    assert result.ok is False
    assert "foreground left" in str(result.detail)
    assert keys == ["back"]
    assert result.stop_reason == "package_changed"
    assert result.steps_run and len(result.steps_run) == 1
    assert str(result.steps_run[0]["to_screen"]).startswith("launcher:")
    assert result.steps_run[0]["changed"] is True


def test_back_until_validates_before_pressing_back(monkeypatch: Any) -> None:
    device = FakeDevice()
    engine = Engine(make_config(memory={"enabled": False}), device=device)

    with pytest.raises(UsageError, match="screen evidence"):
        engine.back_until("net:GET /destination")

    assert not any(name == "press" for name, _args in device.calls)


def test_back_until_rejects_negative_only_destination_before_action() -> None:
    device = FakeDevice()
    engine = Engine(make_config(memory={"enabled": False}), device=device)

    with pytest.raises(UsageError, match="positive destination"):
        engine.back_until("!text:Loading")

    assert not any(name == "press" for name, _args in device.calls)


def test_back_until_returns_without_action_when_destination_is_already_present(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    monkeypatch.setattr(
        engine,
        "await_predicate",
        lambda *_args, **_kwargs: _await(ok=True, outcome="satisfied"),
    )
    monkeypatch.setattr(
        engine,
        "key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not navigate")),
    )

    result = engine.back_until("text:Home")

    assert result.ok is True
    assert result.stop_reason == "already_satisfied"
    assert result.steps_run == []


def test_back_until_refuses_ambiguous_back_affordances(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    back_by_id = Element(
        id=4,
        type="Button",
        resource_id="com.example.app:id/navBack",
        clickable=True,
        bounds=(0, 0, 100, 100),
        center=(50, 50),
    )
    back_by_desc = back_by_id.model_copy(
        update={"id": 5, "resource_id": None, "content_desc": "Navigate up"}
    )
    monkeypatch.setattr(
        engine,
        "await_predicate",
        lambda *_args, **_kwargs: _await(
            ok=False,
            outcome="timeout",
            observation=_observation(elements=[back_by_id, back_by_desc]),
        ),
    )
    monkeypatch.setattr(
        engine,
        "tap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not tap")),
    )

    result = engine.back_until("text:Home")

    assert result.ok is False
    assert result.stop_reason == "ambiguous_back_affordance"
    assert result.steps_run == []


def test_back_until_does_not_mistake_playback_or_system_controls_for_back() -> None:
    playback = Element(
        id=6,
        type="Button",
        resource_id="com.example.app:id/playback",
        clickable=True,
        bounds=(0, 0, 100, 100),
        center=(50, 50),
    )
    system_back = playback.model_copy(
        update={
            "id": 7,
            "resource_id": None,
            "content_desc": "Back",
            "window": "system",
        }
    )

    status, selector, frame_id = Engine._semantic_back_selector(
        _observation(elements=[playback, system_back])
    )

    assert status == "none"
    assert selector is None
    assert frame_id is None


def test_back_until_does_not_auto_tap_unlabeled_top_left_control() -> None:
    hamburger = Element(
        id=22,
        type="View",
        clickable=True,
        bounds=(21, 147, 147, 273),
        center=(84, 210),
        parent=0,
        stable_key="geo:View:q10:fictional-hamburger",
        window="app",
    )

    status, selector, frame_id = Engine._semantic_back_selector(_observation(elements=[hamburger]))

    assert status == "none"
    assert selector is None
    assert frame_id is None


def test_back_until_invalid_explicit_id_refuses_before_navigation(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    monkeypatch.setattr(
        engine,
        "await_predicate",
        lambda *_args, **_kwargs: _await(
            ok=False,
            outcome="timeout",
            observation=_observation(elements=[]),
        ),
    )
    monkeypatch.setattr(
        engine,
        "key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not navigate")),
    )

    result = engine.back_until("text:Home", back_id=99)

    assert result.ok is False
    assert result.stop_reason == "no_back_affordance"
    assert result.steps_run == []


def test_back_until_remaps_explicit_id_from_callers_frame_before_tapping(
    monkeypatch: Any,
) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    original_back = Element(
        id=22,
        type="View",
        clickable=True,
        bounds=(21, 147, 147, 273),
        center=(84, 210),
        stable_key="geo:View:q10:fictional-back",
        window="app",
    )
    cached = _observation(elements=[original_back], known_screen="thread")
    wrong_new_22 = original_back.model_copy(
        update={
            "id": 22,
            "bounds": (300, 147, 426, 273),
            "center": (363, 210),
            "stable_key": "geo:View:q11:different-control",
        }
    )
    remapped_back = original_back.model_copy(update={"id": 23})
    results = [
        _await(
            ok=False,
            outcome="timeout",
            observation=_observation(elements=[wrong_new_22, remapped_back], known_screen="thread"),
        ),
        _await(ok=True, outcome="satisfied", observation=_observation(known_screen="home")),
    ]
    tapped: list[int | None] = []
    monkeypatch.setattr(engine, "_read_cache", lambda: cached)
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "tap",
        lambda element_id=None, **_kwargs: (
            tapped.append(element_id) or ActionResult(ok=True, action="tap")
        ),
    )

    result = engine.back_until("text:Home", back_id=22)

    assert result.ok is True
    assert tapped == [23]
    assert 22 not in tapped


def test_back_until_stops_after_one_no_progress_hardware_back(monkeypatch: Any) -> None:
    engine = Engine(
        make_config(memory={"enabled": False}), device=FakeDevice(package="com.example.app")
    )
    unchanged = _await(
        ok=False,
        outcome="timeout",
        observation=_observation(known_screen="nested"),
    )
    results = [unchanged.model_copy(deep=True), unchanged.model_copy(deep=True)]
    keys: list[str] = []
    monkeypatch.setattr(engine, "await_predicate", lambda *_args, **_kwargs: results.pop(0))
    monkeypatch.setattr(
        engine,
        "key",
        lambda name, **_kwargs: keys.append(name) or ActionResult(ok=True, action="key"),
    )

    result = engine.back_until("text:Home")

    assert result.ok is False
    assert result.stop_reason == "no_progress"
    assert "reuse this returned observation" in result.detail
    assert "--back-id <fresh-id>" in result.detail
    assert keys == ["back"]
    assert result.steps_run and result.steps_run[0]["changed"] is False


def test_cli_back_until_returns_final_observation(monkeypatch: Any) -> None:
    device = FakeDevice(serial="back-cli")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setattr(
        Engine,
        "back_until",
        lambda self, predicate, **kwargs: ActionResult(
            ok=True,
            action="back-until",
            detail=f"satisfied: {predicate}; max={kwargs['max_steps']}",
            observation=_observation(device.serial),
            observation_present=True,
        ),
    )

    result = runner.invoke(
        app,
        ["--serial", device.serial, "back-until-and-analyze", "rid:bottomNav", "--max-steps", "3"],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["action"] == "back-until"
    assert payload["observation"]["meta"]["known_screen"] == "destination"


def test_cli_back_until_deprecated_max_back_alias_maps_to_max_steps(
    monkeypatch: Any,
) -> None:
    device = FakeDevice(serial="back-cli-max-back")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        Engine,
        "back_until",
        lambda self, predicate, **kwargs: (
            calls.append({"predicate": predicate, **kwargs})
            or ActionResult(ok=True, action="back-until")
        ),
    )

    result = runner.invoke(
        app,
        [
            "--serial",
            device.serial,
            "back-until-and-analyze",
            "text:Home",
            "--max-back",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "--max-back is deprecated; use --max-steps" in result.stderr
    assert calls[0]["max_steps"] == 3


def test_cli_back_until_max_back_help_and_conflict_are_explicit(monkeypatch: Any) -> None:
    help_result = runner.invoke(app, ["back-until-and-analyze", "--help"])
    assert help_result.exit_code == 0
    assert "--max-back" in help_result.stdout
    normalized_help = " ".join(help_result.stdout.split())
    assert "Deprecated alias for" in normalized_help
    assert normalized_help.count("--max-steps") >= 2

    device = FakeDevice(serial="back-cli-max-conflict")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    conflict = runner.invoke(
        app,
        [
            "--serial",
            device.serial,
            "back-until-and-analyze",
            "text:Home",
            "--max-steps",
            "2",
            "--max-back",
            "3",
        ],
    )
    assert conflict.exit_code == 2
    assert "pass only one" in conflict.stderr
    assert "--max-steps N" in conflict.stderr


def test_cli_back_until_rejects_global_until_and_multiple_back_selectors(
    monkeypatch: Any,
) -> None:
    device = FakeDevice(serial="back-cli-invalid")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        Engine,
        "back_until",
        lambda self, predicate, **kwargs: (
            calls.append(kwargs) or ActionResult(ok=True, action="back-until")
        ),
    )

    folded = runner.invoke(
        app,
        [
            "--serial",
            device.serial,
            "--until",
            "text:Other",
            "back-until-and-analyze",
            "text:Home",
        ],
    )
    ambiguous = runner.invoke(
        app,
        [
            "--serial",
            device.serial,
            "back-until-and-analyze",
            "text:Home",
            "--back-rid",
            "navBack",
            "--back-desc",
            "Back",
        ],
    )

    assert folded.exit_code == 2
    assert "owns its destination predicate" in folded.stderr
    assert ambiguous.exit_code == 2
    assert "choose only one" in ambiguous.stderr
    assert calls == []


def test_daemon_dispatches_back_until() -> None:
    calls: list[dict[str, Any]] = []

    class FakeEngine:
        def back_until(self, **kwargs: Any) -> ActionResult:
            calls.append(kwargs)
            return ActionResult(ok=True, action="back-until")

    response = dispatch(
        FakeEngine(),
        {"cmd": "back_until", "args": {"predicate": "text:Home", "max_steps": 4}},
    )

    assert response["ok"] is True
    assert calls == [{"predicate": "text:Home", "max_steps": 4}]
