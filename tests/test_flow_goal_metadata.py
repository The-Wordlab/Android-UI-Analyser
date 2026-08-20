"""Reusable flows expose goal aliases, arrival evidence, and reversible network steps."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import Flow, FlowStore, parse_flow_yaml, render_flow_yaml
from android_ui_analyser.memory import (
    DEFAULT_CONTEXT_ID,
    AppMap,
    AppMemoryStore,
    RouteStep,
    ScreenRecord,
    SessionState,
)
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen
from android_ui_analyser.session import plan_goal_session
from conftest import FakeDevice, make_config

_NOW = "2026-01-02T03:04:05+00:00"


def _observation(serial: str = "flow-goal", *, with_recent: bool = False) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package="com.example.catalog", source="hierarchy"),
        elements=(
            [
                Element(
                    id=1,
                    type="android.widget.Button",
                    text="Recent item",
                    resource_id="com.example.catalog:id/recentItem",
                    bounds=(10, 10, 200, 100),
                    center=(105, 55),
                    clickable=True,
                )
            ]
            if with_recent
            else []
        ),
        meta=Meta(
            duration_ms=5,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="home",
            device_serial=serial,
        ),
    )


def test_flow_goal_metadata_round_trips_and_alias_matches() -> None:
    flow = parse_flow_yaml(
        """
name: cached_item
app: com.example.catalog
description: Open a locally cached item
aliases:
  - offline recent item
  - cached history
arrival: rid:cachedContent,!text:Loading
arrival_screen: cached_item
arrival_status: mapped
steps:
  - tap: {id: recentItem}
"""
    )

    reparsed = parse_flow_yaml(render_flow_yaml(flow))
    assert reparsed.aliases == ["offline recent item", "cached history"]
    assert reparsed.arrival == "rid:cachedContent,!text:Loading"
    assert reparsed.arrival_screen == "cached_item"
    assert reparsed.arrival_status == "mapped"
    app = AppMap(
        package="com.example.catalog",
        screens={
            "cached_item": ScreenRecord(
                name="cached_item",
                signature="sig-cached-item",
                first_seen=_NOW,
                last_seen=_NOW,
                last_verified=_NOW,
            )
        },
    )
    plan = plan_goal_session("offline recent item", _observation(), app=app, flows=[reparsed])
    assert plan.candidates[0].kind == "flow"
    assert plan.candidates[0].evidence["arrival"] == reparsed.arrival
    assert plan.candidates[0].evidence["arrival_screen"] == "cached_item"
    assert plan.candidates[0].safe is True
    assert plan.candidates[0].call.executes is True
    assert plan.selected_candidate == "flow:cached_item"


def test_predicate_verified_status_round_trips_while_legacy_predicates_still_load() -> None:
    verified = parse_flow_yaml(
        """
name: verified_predicate
arrival: rid:detailPanel,!text:Loading
arrival_status: predicate_verified
steps:
  - tap: {id: openDetails, by: id}
"""
    )
    legacy = parse_flow_yaml(
        """
name: legacy_predicate
arrival: rid:detailPanel
steps:
  - tap: {id: openDetails}
"""
    )

    assert parse_flow_yaml(render_flow_yaml(verified)).arrival_status == "predicate_verified"
    assert legacy.arrival_status is None


def test_network_flow_steps_are_risk_previewed_and_execute_when_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "offline.yaml"
    path.write_text(
        """
name: offline_cached
app: com.example.catalog
steps:
  - network_offline: {timeout_ms: 12000}
  - tap: {id: recentItem}
  - network_restore
""",
        encoding="utf-8",
    )
    flow = parse_flow_yaml(path.read_text(encoding="utf-8"))
    plan = plan_goal_session("offline cached", _observation(), flows=[flow])
    assert plan.candidates[0].safe is False
    assert {risk["code"] for risk in plan.candidates[0].risks} == {
        "environment_mutation",
        "arrival_unverified",
    }

    engine = Engine(
        make_config(memory={"enabled": False}),
        device=FakeDevice(serial="flow-goal"),
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation(with_recent=True))
    calls: list[str] = []
    monkeypatch.setattr(
        engine,
        "network_offline",
        lambda **_kwargs: calls.append("offline") or SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        engine,
        "network_restore",
        lambda **_kwargs: calls.append("restore") or SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        engine,
        "tap",
        lambda *_args, **_kwargs: calls.append("tap") or SimpleNamespace(ok=True),
    )
    # An authored flow is the explicit execution surface; goal-based automatic selection
    # remains blocked by the environment risk above.
    result = engine.flow_run(file=str(path), allow_unsafe=True)
    assert result["ok"] is True
    assert calls == ["offline", "tap", "restore"]


def test_invalid_flow_arrival_is_rejected_before_any_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bad-arrival.yaml"
    path.write_text(
        "name: bad\narrival: unknown:state\nsteps:\n  - tap: {id: recentItem}\n",
        encoding="utf-8",
    )
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    monkeypatch.setattr(
        engine,
        "analyze",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must preflight first")),
    )
    with pytest.raises(UsageError):
        engine.flow_run(file=str(path))


def test_absence_only_flow_arrival_is_not_proof_and_cannot_execute(
    tmp_path: Path,
) -> None:
    flow = Flow(
        name="absence_only",
        app="com.example.catalog",
        description="Open absence only",
        arrival="!text:Loading",
        steps=[RouteStep(kind="key", arg="back")],
    )
    plan = plan_goal_session("absence only", _observation(), flows=[flow])
    candidate = next(item for item in plan.candidates if item.id == "flow:absence_only")
    assert candidate.safe is False and candidate.call.executes is False
    assert {risk["code"] for risk in candidate.risks} == {"arrival_invalid"}
    assert plan.selected_candidate is None

    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    with pytest.raises(UsageError, match="positive arrival"):
        FlowStore(cfg.memory).save(flow)
    path = tmp_path / "absence-only.yaml"
    path.write_text(render_flow_yaml(flow), encoding="utf-8")
    device = FakeDevice(serial="absence-only", package="com.example.catalog")
    with pytest.raises(UsageError, match="positive arrival"):
        Engine(cfg, device=device).flow_run(file=str(path))
    assert device.calls == [] and device.hierarchy_calls == 0


def test_recorded_flow_context_is_filtered_and_execution_is_preflighted(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    flow = parse_flow_yaml(
        """
name: variant_item
app: com.example.catalog
context_id: flags-catalog-v2
arrival_status: unverified
steps:
  - tap: {id: recentItem}
"""
    )
    plan = plan_goal_session(
        "variant item",
        _observation(),
        context_id="default",
        flows=[flow],
    )
    assert all(candidate.id != "flow:variant_item" for candidate in plan.candidates)

    store = FlowStore(cfg.memory)
    store.save(flow)
    device = FakeDevice(serial="flow-context", package="com.example.catalog")
    AppMemoryStore(cfg.memory).save_session(
        device.serial,
        SessionState(package="com.example.catalog", active_context_id="default"),
    )
    engine = Engine(cfg, device=device)
    dry = engine.flow_run("variant_item", dry_run=True)
    assert dry["would_execute"] is False
    assert dry["context_compatible"] is False
    with pytest.raises(UsageError, match="recorded for context"):
        engine.flow_run("variant_item")


def test_goal_plan_ignores_a_foreign_session_cursor_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    device = FakeDevice(serial="goal-foreign-cursor", package="com.example.catalog")
    AppMemoryStore(cfg.memory).save_session(
        device.serial,
        SessionState(
            package="com.example.other",
            current_screen="other_detail",
            active_context_id="flags-other",
        ),
    )
    engine = Engine(cfg, device=device)
    observed = _observation(device.serial)
    observed.meta.known_screen = None
    captured: dict[str, Any] = {}

    import android_ui_analyser.session as session_mod

    def capture_plan(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(session_mod, "plan_goal_session", capture_plan)
    engine._goal_session_plan("open catalog", observed)

    assert captured["current_screen"] is None
    assert captured["context_id"] == "default"


def test_flow_list_does_not_borrow_context_from_another_foreground(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    FlowStore(cfg.memory).save(
        parse_flow_yaml(
            """
name: variant_catalog
app: com.example.catalog
context_id: flags-catalog-v2
steps:
  - key: back
"""
        )
    )
    device = FakeDevice(serial="list-foreign-cursor", package="com.example.catalog")
    AppMemoryStore(cfg.memory).save_session(
        device.serial,
        SessionState(package="com.example.other", active_context_id="flags-catalog-v2"),
    )

    listed = Engine(cfg, device=device).flow_list()

    assert listed["active_package"] == "com.example.catalog"
    assert listed["active_context_id"] is None
    assert listed["flows"][0]["context_compatible"] is None


def test_goal_planning_uses_runnable_storage_names_and_isolates_bad_files(
    tmp_path: Path,
) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    store = FlowStore(cfg.memory)
    store.flows_dir().mkdir(parents=True)
    (store.flows_dir() / "aaa_broken.yaml").write_text("steps: [", encoding="utf-8")
    (store.flows_dir() / "open_cached.yaml").write_text(
        "name: Friendly cached title\n"
        "app: com.example.catalog\n"
        "description: Open cached catalog item\n"
        "aliases: [offline recent item]\n"
        "steps:\n  - tap: {id: recentItem}\n",
        encoding="utf-8",
    )
    engine = Engine(
        cfg,
        device=FakeDevice(serial="flow-storage-name", package="com.example.catalog"),
    )

    plan = engine._goal_session_plan("Friendly cached title", _observation("flow-storage-name"))

    candidate = next(item for item in plan.candidates if item.kind == "flow")
    assert candidate.id == "flow:open_cached"
    assert "flow run open_cached" in candidate.call.cli


def test_long_lived_flow_hints_refresh_after_external_delete(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    store = FlowStore(cfg.memory)
    store.save(
        Flow(name="temporary", app="com.example.catalog", steps=[RouteStep(kind="key", arg="back")])
    )
    engine = Engine(
        cfg,
        device=FakeDevice(serial="flow-cache-refresh", package="com.example.catalog"),
    )

    assert engine._flows_for("com.example.catalog") == ["temporary"]
    assert store.delete("temporary") is True
    assert engine._flows_for("com.example.catalog") == []


def test_flow_hints_ignore_foreign_context_and_malformed_rows(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    store = FlowStore(cfg.memory)
    store.flows_dir().mkdir(parents=True)
    (store.flows_dir() / "broken.yaml").write_text("steps: [", encoding="utf-8")
    store.save(
        Flow(
            name="catalog_default",
            app="com.example.catalog",
            context_id=DEFAULT_CONTEXT_ID,
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    store.save(
        Flow(
            name="catalog_variant",
            app="com.example.catalog",
            context_id="flags-catalog-v2",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    device = FakeDevice(serial="hint-foreign-context", package="com.example.catalog")
    AppMemoryStore(cfg.memory).save_session(
        device.serial,
        SessionState(package="com.example.other", active_context_id="flags-catalog-v2"),
    )

    hints = Engine(cfg, device=device)._flows_for("com.example.catalog")

    assert hints == ["catalog_default"]
