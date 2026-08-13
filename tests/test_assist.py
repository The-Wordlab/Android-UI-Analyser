"""Opt-in planner assist (PRD §7.3): gating, recovery, destructive guard, suggestion.

The planner is off by default and never touches the happy path. These tests use a
`StubPlanner` (no network) wired into the engine's factory, and assert: it fires only
when `planner.enabled` AND `--assist`; it recovers a diverged `goto`/`flow` by dismissing
a blocker; it honors the destructive guard; and an un-assisted divergence suggests
`--assist` in its hint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import FlowStore
from android_ui_analyser.memory import AppMemoryStore, RouteStep
from android_ui_analyser.providers.base import PlannerDecision
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import StubPlanner, make_chain, make_config
from test_memory import APPS, HOME, P, _elements, _hier, _node, _store
from test_navigation import ScriptedDevice

# A screen recognised as `home` (shares home's anchors) but with `Apps` replaced by a
# blocking "Continue" dialog button — so the recorded `tap 'Apps'` edge can't match.
BLOCKED = _hier(
    _node("android.widget.TextView", text="Home", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button", text="Chat", rid="x:id/nav_chat", clk=True, b="[40,440][1040,540]"
    ),
    _node(
        "android.widget.Button",
        text="Continue",
        rid="x:id/dialog_ok",
        clk=True,
        b="[40,700][1040,800]",
    ),
)
DANGER = _hier(
    _node("android.widget.TextView", text="Confirm", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Delete my account",
        rid="x:id/del",
        clk=True,
        b="[40,300][1040,400]",
    ),
)
RESOURCE_ONLY_DANGER = _hier(
    _node("android.widget.TextView", text="Confirm", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        rid="x:id/deleteAccount",
        clk=True,
        b="[40,300][1040,400]",
    ),
)


def _engine(tmp_path: Path, device: ScriptedDevice, *, planner_enabled: bool) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        planner={"enabled": planner_enabled},
        daemon={"enabled": False},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _wire(monkeypatch, eng: Engine, stub: StubPlanner) -> None:
    orig = eng.factory.build_chain
    monkeypatch.setattr(
        eng.factory,
        "build_chain",
        lambda kind: make_chain("planner", [stub]) if kind == "planner" else orig(kind),
    )


def _tap_label(target: str):
    """A decider: tap the element whose label == target, else declare done."""

    def fn(objective: str, elements: list[dict]) -> PlannerDecision:
        for e in elements:
            if e.get("label") == target:
                return PlannerDecision(action="tap", target_id=e["id"], reason=f"dismiss {target}")
        return PlannerDecision(action="done", reason="nothing to dismiss")

    return fn


def _seed_home_apps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    from android_ui_analyser.memory import RouteStep

    store.record_route(
        P, "home", "apps", steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")]
    )


# --------------------------------------------------------------- _drive_with_planner unit


def test_drive_is_noop_when_planner_disabled(tmp_path, monkeypatch) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-a1")
    eng = _engine(tmp_path, dev, planner_enabled=False)
    stub = StubPlanner(decide_fn=_tap_label("nope"))
    _wire(monkeypatch, eng, stub)
    res = eng.analyze(source="hierarchy")
    ok, _ = eng._drive_with_planner("reach x", res=res, max_steps=3, allow_destructive=False)
    assert ok is False and stub.calls == 0  # disabled → never consulted


def test_drive_respects_destructive_guard(tmp_path, monkeypatch) -> None:
    dev = ScriptedDevice([DANGER, APPS], package=P, serial="emu-a2")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    stub = StubPlanner(decide_fn=_tap_label("Delete my account"))
    _wire(monkeypatch, eng, stub)
    res = eng.analyze(source="hierarchy")
    ok, _ = eng._drive_with_planner("wipe it", res=res, max_steps=3, allow_destructive=False)
    assert ok is False
    assert not any(c[0] == "click" for c in dev.calls)  # the destructive tap was refused

    dev2 = ScriptedDevice([DANGER, APPS], package=P, serial="emu-a2b")
    eng2 = _engine(tmp_path, dev2, planner_enabled=True)
    _wire(monkeypatch, eng2, StubPlanner(decide_fn=_tap_label("Delete my account")))
    res2 = eng2.analyze(source="hierarchy")
    eng2._drive_with_planner("wipe it", res=res2, max_steps=1, allow_destructive=True)
    assert any(c[0] == "click" for c in dev2.calls)  # allowed with the flag


def test_drive_guard_uses_resource_id_when_destructive_control_has_no_label(
    tmp_path, monkeypatch
) -> None:
    dev = ScriptedDevice([RESOURCE_ONLY_DANGER], package=P, serial="emu-a2-resource")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    res = eng.analyze(source="hierarchy")
    target = next(element for element in res.elements if element.resource_id.endswith("deleteAccount"))
    _wire(
        monkeypatch,
        eng,
        StubPlanner(decisions=[PlannerDecision(action="tap", target_id=target.id)]),
    )

    ok, _ = eng._drive_with_planner(
        "wipe it",
        res=res,
        max_steps=1,
        allow_destructive=False,
    )

    assert ok is False
    assert not any(call[0] == "click" for call in dev.calls)


def test_drive_gives_up_on_invalid_target(tmp_path, monkeypatch) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-a3")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    stub = StubPlanner(decisions=[PlannerDecision(action="tap", target_id=9999)])
    _wire(monkeypatch, eng, stub)
    res = eng.analyze(source="hierarchy")
    ok, _ = eng._drive_with_planner("x", res=res, max_steps=3, allow_destructive=False)
    assert ok is False  # off-screen id → hand off, never guess
    assert not any(c[0] == "click" for c in dev.calls)


# --------------------------------------------------------------- goto --assist


def test_goto_without_assist_suggests_it(tmp_path) -> None:
    _seed_home_apps(tmp_path)
    dev = ScriptedDevice([BLOCKED], package=P, serial="emu-a4")
    eng = _engine(tmp_path, dev, planner_enabled=True)  # enabled, but no --assist
    out = eng.goto("apps")  # diverges: no 'Apps' on the blocked screen
    assert out["ok"] is False and out["code"] == "element_not_found"
    assert "--assist" in out["hint"]  # the tool suggests it


def test_goto_assist_recovers_interstitial(tmp_path, monkeypatch) -> None:
    _seed_home_apps(tmp_path)
    dev = ScriptedDevice([BLOCKED, APPS], package=P, serial="emu-a5")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    stub = StubPlanner(decide_fn=_tap_label("Continue"))  # dismiss the blocker → APPS
    _wire(monkeypatch, eng, stub)
    out = eng.goto("apps", assist=True)
    assert out["ok"] is True and out["arrived"] is True, out
    assert stub.calls >= 1
    assert any(c[0] == "click" for c in dev.calls)  # the planner dismissed the dialog


def test_goto_assist_flag_without_config_does_nothing(tmp_path, monkeypatch) -> None:
    _seed_home_apps(tmp_path)
    dev = ScriptedDevice([BLOCKED, APPS], package=P, serial="emu-a6")
    eng = _engine(tmp_path, dev, planner_enabled=False)  # config off
    stub = StubPlanner(decide_fn=_tap_label("Continue"))
    _wire(monkeypatch, eng, stub)
    out = eng.goto("apps", assist=True)  # flag alone can't call a disabled provider
    assert out["ok"] is False and stub.calls == 0


# --------------------------------------------------------------- flow run --assist


def test_flow_assist_clears_blocker_and_resumes(tmp_path, monkeypatch) -> None:
    # A flow whose 2nd step needs 'Apps', but the screen is blocked until dismissed.
    from android_ui_analyser.flows import Flow, FlowStore
    from android_ui_analyser.memory import RouteStep

    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        planner={"enabled": True},
        daemon={"enabled": False},
    )
    FlowStore(cfg.memory).save(
        Flow(
            name="blocked",
            app=P,
            steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")],
        )
    )
    dev = ScriptedDevice([BLOCKED, APPS], package=P, serial="emu-a7")
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    stub = StubPlanner(decide_fn=_tap_label("Continue"))
    _wire(monkeypatch, eng, stub)
    out = eng.flow_run("blocked", assist=True)
    assert out["ok"] is True, out  # blocker dismissed → 'Apps' now tappable → resumed
    assert stub.calls >= 1


def test_flow_without_assist_suggests_it(tmp_path) -> None:
    from android_ui_analyser.flows import Flow, FlowStore
    from android_ui_analyser.memory import RouteStep

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    FlowStore(cfg.memory).save(
        Flow(name="b2", app=P, steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")])
    )
    dev = ScriptedDevice([BLOCKED], package=P, serial="emu-a8")
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    out = eng.flow_run("b2")
    assert out["ok"] is False and "--assist" in out["hint"]


# --------------------------------------------------------------- navigate (flywheel)

IMAGES = _hier(
    _node("android.widget.TextView", text="Create image", rid="x:id/h", b="[40,120][1040,210]"),
    _node("android.widget.Button", text="Generate", rid="x:id/go", clk=True, b="[40,640][400,740]"),
)


def _to_images_decider():
    """Tap 'Apps', then 'Images', then declare done — a scripted 2-hop journey."""

    def fn(objective: str, elements: list[dict]) -> PlannerDecision:
        for target in ("Apps", "Images"):
            for e in elements:
                if e.get("label") == target and e.get("clickable"):
                    return PlannerDecision(action="tap", target_id=e["id"])
        return PlannerDecision(action="done", reason="on the image screen")

    return fn


def test_navigate_requires_planner_enabled(tmp_path) -> None:
    from android_ui_analyser.errors import UsageError

    dev = ScriptedDevice([HOME], package=P, serial="emu-n0")
    eng = _engine(tmp_path, dev, planner_enabled=False)
    try:
        eng.navigate("open images")
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "planner" in str(exc).lower()


def test_navigate_drives_and_records_the_path(tmp_path, monkeypatch) -> None:
    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-n1")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decide_fn=_to_images_decider()))
    out = eng.navigate("open the image generator")
    assert out["ok"] is True and out["arrived"] is True, out
    assert sum(1 for c in dev.calls if c[0] == "click") == 2  # Apps + Images

    # The flywheel: the journey is now in memory as replayable edges.
    app_map = _store(tmp_path).load(P)
    assert app_map is not None and len(app_map.routes) >= 2
    assert all(e.steps for e in app_map.routes)  # every recorded edge is replayable


def test_navigate_until_stops_early(tmp_path, monkeypatch) -> None:
    # 'Generate' is on the IMAGES screen; --until should stop as soon as it's visible.
    dev = ScriptedDevice(
        [HOME, APPS, IMAGES],
        package=P,
        serial="emu-n2",
        text_index={"Generate": (40, 640, 400, 740)},
    )
    # text_index is only IMAGES' button; `has` finds it only once there.
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decide_fn=_to_images_decider()))
    out = eng.navigate("reach image creation", until="Generate")
    assert out["ok"] is True


def test_navigate_save_flow_writes_reusable_yaml(tmp_path, monkeypatch) -> None:
    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-n3")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decide_fn=_to_images_decider()))
    out = eng.navigate("open images", save_flow="to_images")
    assert out["ok"] and "flow_saved" in out
    flow = FlowStore(eng.config.memory).load("to_images")
    assert flow.app == P
    labels = [s.label for s in flow.steps if s.kind == "tap"]
    assert "Apps" in labels and "Images" in labels


def test_navigate_saved_flow_refreshes_same_engine_discovery_cache(
    tmp_path, monkeypatch
) -> None:
    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-n3-cache")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decide_fn=_to_images_decider()))
    assert eng._flows_for(P) == []

    out = eng.navigate("open images", save_flow="to_images_cache")

    assert out["ok"] is True
    assert "to_images_cache" in eng._flows_for(P)


def test_navigate_save_flow_refuses_lossy_recorded_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-navigate-lossy")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decisions=[PlannerDecision(action="done")]))
    original = eng._drive_with_planner

    def record_lossy(*args, **kwargs):
        store = AppMemoryStore(eng.config.memory)
        store.observe_action(
            dev.serial,
            RouteStep(
                kind="swipe",
                arg="up",
                package=P,
            ),
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(eng, "_drive_with_planner", record_lossy)

    result = eng.navigate("inspect catalog", save_flow="lossy_capture")

    assert result["ok"] is False
    assert result["code"] == "flow_capture_lossy"
    assert result["flow_save"]["saved"] is False
    assert not FlowStore(eng.config.memory).path("lossy_capture").exists()


def test_navigate_save_flow_refuses_mixed_capture_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-navigate-mixed")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decisions=[PlannerDecision(action="done")]))
    original = eng._drive_with_planner

    def record_mixed(*args, **kwargs):
        store = AppMemoryStore(eng.config.memory)
        session = store.load_session(dev.serial)
        start_order = session.next_capture_order
        session.recent.extend(
            [
                RouteStep(
                    kind="key",
                    arg="back",
                    package=P,
                    origin_package=P,
                    context_id="default",
                    capture_segment=1,
                    capture_order=start_order,
                ),
                RouteStep(
                    kind="key",
                    arg="back",
                    package="com.example.other",
                    origin_package="com.example.other",
                    context_id="default",
                    capture_segment=2,
                    capture_order=start_order + 1,
                ),
            ]
        )
        session.next_capture_order = start_order + 2
        store.save_session(dev.serial, session)
        return original(*args, **kwargs)

    monkeypatch.setattr(eng, "_drive_with_planner", record_mixed)

    result = eng.navigate("inspect catalog", save_flow="mixed_capture")

    assert result["ok"] is False
    assert result["code"] == "flow_capture_mixed"
    assert result["flow_save"]["saved"] is False
    assert not FlowStore(eng.config.memory).path("mixed_capture").exists()


def test_navigate_save_flow_refuses_a_journey_truncated_by_the_rolling_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-navigate-overflow")
    eng = _engine(tmp_path, dev, planner_enabled=True)
    _wire(monkeypatch, eng, StubPlanner(decisions=[PlannerDecision(action="done")]))

    def record_overflow(*_args, **_kwargs):
        store = AppMemoryStore(eng.config.memory)
        for _ in range(41):
            store.observe_action(dev.serial, RouteStep(kind="key", arg="back", package=P))
        return True, eng.analyze(source="hierarchy")

    monkeypatch.setattr(eng, "_drive_with_planner", record_overflow)

    result = eng.navigate("inspect catalog", max_steps=50, save_flow="overflow_capture")

    assert result["ok"] is False
    assert result["code"] == "flow_capture_overflow"
    assert result["flow_save"]["saved"] is False
    assert not FlowStore(eng.config.memory).path("overflow_capture").exists()


def test_daemon_dispatch_navigate() -> None:
    from android_ui_analyser.daemon import dispatch

    class FakeEng:
        def navigate(self, **kw: object) -> dict[str, object]:
            return {"ok": True, "goal": kw.get("goal")}

    r = dispatch(FakeEng(), {"cmd": "navigate", "args": {"goal": "x", "max_steps": 5}})
    assert r["ok"] and r["result"]["goal"] == "x"
