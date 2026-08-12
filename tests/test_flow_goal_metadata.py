"""Reusable flows expose goal aliases, arrival evidence, and reversible network steps."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml, render_flow_yaml
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen
from android_ui_analyser.session import plan_goal_session
from conftest import FakeDevice, make_config


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
steps:
  - tap: {id: recentItem}
"""
    )

    reparsed = parse_flow_yaml(render_flow_yaml(flow))
    assert reparsed.aliases == ["offline recent item", "cached history"]
    assert reparsed.arrival == "rid:cachedContent,!text:Loading"
    plan = plan_goal_session("offline recent item", _observation(), flows=[reparsed])
    assert plan.candidates[0].kind == "flow"
    assert plan.candidates[0].evidence["arrival"] == reparsed.arrival


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
    assert {risk["code"] for risk in plan.candidates[0].risks} == {"environment_mutation"}

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
