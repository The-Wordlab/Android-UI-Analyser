"""Goal bootstrap chooses trusted knowledge before shortcuts, without duplicate reads."""

from __future__ import annotations

from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import Flow
from android_ui_analyser.memory import AppMap, Deeplink, RouteEdge, RouteStep, ScreenRecord
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from android_ui_analyser.session import GoalCall, GoalCandidate, GoalSessionPlan, plan_goal_session
from conftest import FakeDevice, make_config

_NOW = "2026-01-02T03:04:05+00:00"
_PKG = "com.example.catalog"


def _record(name: str, *, context: str = "default") -> ScreenRecord:
    return ScreenRecord(
        name=name,
        context_id=context,
        signature=f"sig-{name}-{context}",
        first_seen=_NOW,
        last_seen=_NOW,
        last_verified=_NOW,
    )


def _observation(screen: str = "home") -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=_PKG, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=12,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen=screen,
            device_serial="goal-emulator",
        ),
    )


def test_plan_ranks_active_verified_goto_then_matching_flow_then_deeplink() -> None:
    app = AppMap(
        package=_PKG,
        screens={
            "home": _record("home"),
            "saved_items": _record("saved_items"),
            "other_home": _record("other_home", context="flags-other"),
            "other_saved_items": _record("other_saved_items", context="flags-other"),
        },
        routes=[
            RouteEdge(
                id="route-default",
                from_screen="home",
                to_screen="saved_items",
                action="tap 'Saved items'",
                context_id="default",
                steps=[RouteStep(kind="tap", resource_id="savedItems")],
                status="verified",
                last_seen=_NOW,
            ),
            # A route from another flag context must not leak into the active plan.
            RouteEdge(
                id="route-other",
                from_screen="other_home",
                to_screen="other_saved_items",
                action="open catalog://settings?flag=other",
                context_id="flags-other",
                steps=[RouteStep(kind="open-link", arg="catalog://settings?flag=other")],
                status="verified",
                last_seen=_NOW,
            ),
        ],
        deeplinks=[
            Deeplink(
                uri="catalog://saved-items",
                note="Open saved items",
                probed=True,
                landed="saved_items",
            )
        ],
    )
    flow = Flow(
        name="open_saved_items",
        app=_PKG,
        description="Open the saved items list",
        steps=[RouteStep(kind="tap", resource_id="savedItems")],
    )

    plan = plan_goal_session(
        "open saved items",
        _observation(),
        app=app,
        current_screen="home",
        flows=[flow],
    )

    assert [candidate.kind for candidate in plan.candidates] == ["goto", "flow", "deeplink"]
    assert plan.selected_candidate == "goto:saved_items"
    assert plan.recommended_call.cli == "aua goto 'open saved items'"
    assert plan.recommended_call.mcp == {
        "tool": "goto",
        "arguments": {"goal": "open saved items"},
    }
    assert plan.candidates[0].evidence["context_id"] == "default"
    assert plan.candidates[0].safe is True
    assert plan.candidates[-1].safe is False
    assert plan.candidates[-1].status == "probed"


def test_risky_flow_is_previewed_and_never_selected_automatically() -> None:
    flow = Flow(
        name="offline_saved_items",
        app=_PKG,
        description="Set up the offline saved items scenario",
        steps=[
            RouteStep(kind="launch-app", arg=_PKG),
            RouteStep(kind="tap", resource_id="savedItems"),
        ],
    )

    plan = plan_goal_session("offline saved items", _observation(), flows=[flow])

    candidate = plan.candidates[0]
    assert candidate.kind == "flow"
    assert candidate.safe is False
    assert candidate.call.cli == "aua flow run offline_saved_items --dry-run"
    assert candidate.call.executes is False
    assert {risk["code"] for risk in candidate.risks} == {"app_lifecycle"}
    assert plan.selected_candidate is None
    assert plan.recommended_call.kind == "network_offline"
    assert plan.recommended_call.cli == "aua network offline --verify"
    assert "network offline --verify" in plan.warnings[0]


def test_goal_matching_ignores_conjunctions_and_requires_whole_words() -> None:
    unrelated = Flow(
        name="reset_account_google_login",
        app=_PKG,
        description="Reset the account and continue with login",
        steps=[RouteStep(kind="tap", resource_id="resetAccount")],
    )

    plan = plan_goal_session(
        "compare Grammar and Mathematics while offline",
        _observation(),
        flows=[unrelated],
    )

    assert not any(candidate.kind == "flow" for candidate in plan.candidates)
    assert plan.recommended_call.kind == "network_offline"
    assert plan.observation_note.startswith("This is the current settled screen")


def test_session_start_uses_exactly_one_analyze(monkeypatch, tmp_path) -> None:
    engine = Engine(
        make_config(
            memory={"enabled": False},
            cache={"dir": str(tmp_path / "cache")},
        ),
        device=FakeDevice(),
    )
    observed = _observation()
    calls = 0

    def analyze(**_kwargs: Any) -> AnalyzeResult:
        nonlocal calls
        calls += 1
        return observed

    monkeypatch.setattr(engine, "analyze", analyze)

    result = engine.session_start("inspect saved items")

    assert calls == 1
    assert result["observation"]["meta"]["known_screen"] == "home"
    assert result["recommended_call"]["kind"] == "map_find"
    assert result["relevant_capabilities"]


def test_reach_reuses_bootstrap_observation_for_goto(monkeypatch) -> None:
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    observed = _observation()
    analyze_calls = 0
    goto_observation: AnalyzeResult | None = None

    def analyze(**_kwargs: Any) -> AnalyzeResult:
        nonlocal analyze_calls
        analyze_calls += 1
        return observed

    candidate = GoalCandidate(
        id="goto:saved_items",
        kind="goto",
        name="saved_items",
        target="saved_items",
        safe=True,
        status="verified",
        call=GoalCall(
            kind="goto",
            cli="aua goto saved_items",
            mcp={"tool": "goto", "arguments": {"goal": "saved_items"}},
            reason="verified route",
        ),
    )
    plan = GoalSessionPlan(
        goal="saved_items",
        package=_PKG,
        current_screen="home",
        observation=observed,
        candidates=[candidate],
        selected_candidate=candidate.id,
        recommended_call=candidate.call,
    )

    def goto(_goal: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal goto_observation
        goto_observation = kwargs["_observation"]
        return {"ok": True, "arrived": True, "final_screen": "saved_items"}

    monkeypatch.setattr(engine, "analyze", analyze)
    monkeypatch.setattr(engine, "_goal_session_plan", lambda _goal, _observation: plan)
    monkeypatch.setattr(engine, "goto", goto)

    result = engine.reach("saved_items")

    assert result["ok"] is True
    assert result["strategy"] == "goto"
    assert analyze_calls == 1
    assert goto_observation is observed
