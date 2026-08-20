"""Focused regressions for flow execution and planner-capture trust boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import Flow, FlowStore
from android_ui_analyser.memory import AppMemoryStore, RouteStep, SessionState
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config
from test_memory import HOME, P, _elements, _hier, _node, _store

TRANSIT = "com.example.auth"
TRANSIT_SCREEN = _hier(
    _node(
        "android.widget.Button",
        text="Approve",
        rid=f"{TRANSIT}:id/approve",
        clk=True,
        pkg=TRANSIT,
    )
)
LAUNCHER = _hier(
    _node(
        "android.widget.TextView",
        text="Launcher",
        rid="com.example.launcher:id/title",
        pkg="com.example.launcher",
    )
)


def test_mapped_arrival_reobserves_after_non_observing_stop_app(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    FlowStore(cfg.memory).save(
        Flow(
            name="stop_is_not_home",
            app=P,
            arrival_screen="home",
            arrival_status="mapped",
            steps=[RouteStep(kind="stop-app", arg=P)],
        )
    )

    class StopsToLauncher(FakeDevice):
        def stop_app(self, package: str) -> None:
            super().stop_app(package)
            self._pkg = "com.example.launcher"
            self._xml = LAUNCHER

    device = StopsToLauncher(hierarchy_xml=HOME, package=P, serial="flow-stop-arrival")

    result = Engine(cfg, device=device).flow_run("stop_is_not_home")

    assert result["ok"] is False
    assert result["code"] == "arrival_screen_unverified"
    assert result["arrival_screen"]["recognized"] is None
    assert device.hierarchy_calls >= 2


class _FinalizingThread:
    def __init__(self, finalize: Callable[[], None]) -> None:
        self._finalize = finalize
        self.joined = False

    def is_alive(self) -> bool:
        return not self.joined

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self._finalize()
        self.joined = True


def _planner_engine(tmp_path: Path, *, serial: str) -> tuple[Engine, FakeDevice]:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        planner={"enabled": True},
        daemon={"enabled": False},
    )
    device = FakeDevice(hierarchy_xml=HOME, package=P, serial=serial)
    return Engine(cfg, device=device, factory=ProviderFactory(cfg)), device


def _record_back(memory: AppMemoryStore, serial: str) -> None:
    memory.observe_action(serial, RouteStep(kind="key", arg="back", package=P))


def test_navigate_save_flow_joins_async_provenance_before_selecting_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, device = _planner_engine(tmp_path, serial="navigate-finalize")
    finalizer: _FinalizingThread | None = None

    def drive(_goal: str, *, res: Any, **_kwargs: Any) -> tuple[bool, Any]:
        nonlocal finalizer
        memory = AppMemoryStore(engine.config.memory)

        def finalize() -> None:
            _record_back(memory, device.serial)

        finalizer = _FinalizingThread(finalize)
        engine._mem_thread = finalizer  # type: ignore[assignment]
        return True, res

    monkeypatch.setattr(engine, "_drive_with_planner", drive)

    result = engine.navigate("return home", save_flow="finalized_path")

    assert finalizer is not None and finalizer.joined
    assert result["ok"] is True
    assert FlowStore(engine.config.memory).load("finalized_path").steps[0].kind == "key"


def test_navigate_save_flow_returns_unsaved_refusal_for_finalized_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, device = _planner_engine(tmp_path, serial="navigate-boundary")

    def drive(_goal: str, *, res: Any, **_kwargs: Any) -> tuple[bool, Any]:
        memory = AppMemoryStore(engine.config.memory)
        _record_back(memory, device.serial)

        def finalize() -> None:
            finalized = memory.load_session(device.serial)
            finalized.capture_segment += 1
            finalized.capture_boundary_reason = "foreground app changed"
            memory.save_session(device.serial, finalized)

        engine._mem_thread = _FinalizingThread(finalize)  # type: ignore[assignment]
        return True, res

    monkeypatch.setattr(engine, "_drive_with_planner", drive)

    result = engine.navigate("leave app", save_flow="stale_segment")

    assert result["ok"] is False
    assert result["arrived"] is True
    assert result["code"] == "flow_capture_boundary"
    assert result["flow_save"]["saved"] is False
    assert FlowStore(engine.config.memory).find("stale_segment") == []


def test_navigate_save_flow_finds_new_action_after_rolling_journal_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, device = _planner_engine(tmp_path, serial="navigate-full-journal")
    memory = AppMemoryStore(engine.config.memory)
    session = memory.load_session(device.serial)
    session.package = P
    memory.save_session(device.serial, session)
    for index in range(40):
        memory.observe_action(
            device.serial,
            RouteStep(kind="key", arg=f"old-{index}", package=P),
        )
    assert len(memory.load_session(device.serial).recent) == 40

    def drive(_goal: str, *, res: Any, **_kwargs: Any) -> tuple[bool, Any]:
        _record_back(memory, device.serial)
        return True, res

    monkeypatch.setattr(engine, "_drive_with_planner", drive)

    result = engine.navigate("return home", save_flow="after_cap")

    assert result["ok"] is True
    saved = FlowStore(engine.config.memory).load("after_cap")
    assert [(step.kind, step.arg) for step in saved.steps] == [("key", "back")]


def test_navigate_save_flow_never_overwrites_existing_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, device = _planner_engine(tmp_path, serial="navigate-existing-flow")
    flow_store = FlowStore(engine.config.memory)
    flow_store.save(
        Flow(name="kept", app=P, steps=[RouteStep(kind="key", arg="home")])
    )
    memory = AppMemoryStore(engine.config.memory)

    def drive(_goal: str, *, res: Any, **_kwargs: Any) -> tuple[bool, Any]:
        _record_back(memory, device.serial)
        return True, res

    monkeypatch.setattr(engine, "_drive_with_planner", drive)

    result = engine.navigate("return home", save_flow="kept")

    assert result["ok"] is False
    assert result["code"] == "flow_capture_exists"
    assert result["flow_save"]["saved"] is False
    assert [(step.kind, step.arg) for step in flow_store.load("kept").steps] == [
        ("key", "home")
    ]


def _transit_resume_engine(
    tmp_path: Path,
    *,
    session_package: str,
    resumed_package: str | None,
) -> tuple[Engine, FakeDevice]:
    cfg = make_config(
        memory={
            "dir": str(tmp_path / "home"),
            "transit_packages": [TRANSIT],
        },
        daemon={"enabled": False},
    )
    FlowStore(cfg.memory).save(
        Flow(
            name="resume_auth",
            app=P,
            context_id="default",
            steps=[
                RouteStep(kind="open-link", arg="fiction://auth"),
                RouteStep(kind="key", arg="back", package=resumed_package),
            ],
        )
    )
    device = FakeDevice(
        hierarchy_xml=TRANSIT_SCREEN,
        package=TRANSIT,
        serial="flow-transit-resume",
    )
    AppMemoryStore(cfg.memory).save_session(
        device.serial,
        SessionState(package=session_package, active_context_id="default"),
    )
    return Engine(cfg, device=device), device


def test_from_step_may_resume_owned_explicit_transit_step(tmp_path: Path) -> None:
    engine, device = _transit_resume_engine(
        tmp_path,
        session_package=P,
        resumed_package=TRANSIT,
    )

    result = engine.flow_run("resume_auth", from_step=1)

    assert result["ok"] is True
    assert ("press", ("back",)) in device.calls


@pytest.mark.parametrize(
    ("session_package", "resumed_package"),
    [
        ("com.example.other", TRANSIT),
        (P, None),
    ],
    ids=["foreign-session", "implicit-step-package"],
)
def test_from_step_rejects_unproven_transit_resume(
    tmp_path: Path,
    session_package: str,
    resumed_package: str | None,
) -> None:
    engine, device = _transit_resume_engine(
        tmp_path,
        session_package=session_package,
        resumed_package=resumed_package,
    )

    with pytest.raises(UsageError, match="foreground package"):
        engine.flow_run("resume_auth", from_step=1)
    assert not any(call[0] == "press" for call in device.calls)
