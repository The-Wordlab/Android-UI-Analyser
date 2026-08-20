"""Goal bootstrap chooses trusted knowledge before shortcuts, without duplicate reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import Flow
from android_ui_analyser.memory import AppMap, Deeplink, RouteEdge, RouteStep, ScreenRecord
from android_ui_analyser.schema import ActionResult, AnalyzeResult, Meta, Screen
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


def test_plan_recommends_longer_safe_goto_instead_of_one_hop_unsafe_route() -> None:
    app = AppMap(
        package=_PKG,
        screens={
            "home": _record("home"),
            "catalog": _record("catalog"),
            "saved_items": _record("saved_items"),
        },
        routes=[
            RouteEdge(
                id="unsafe-shortcut",
                from_screen="home",
                to_screen="saved_items",
                action="open catalog://saved-items",
                steps=[RouteStep(kind="open-link", arg="catalog://saved-items")],
                count=50,
                status="verified",
                last_seen=_NOW,
            ),
            RouteEdge(
                id="safe-catalog",
                from_screen="home",
                to_screen="catalog",
                action="tap 'Catalog'",
                steps=[RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog")],
                status="verified",
                last_seen=_NOW,
            ),
            RouteEdge(
                id="safe-saved",
                from_screen="catalog",
                to_screen="saved_items",
                action="tap 'Saved items'",
                steps=[RouteStep(kind="tap", label="Saved items", resource_id="saved_items")],
                status="verified",
                last_seen=_NOW,
            ),
        ],
    )

    plan = plan_goal_session(
        "open saved items",
        _observation(),
        app=app,
        current_screen="home",
    )

    candidate = plan.candidates[0]
    assert candidate.safe is True and candidate.call.kind == "goto"
    assert [edge["to"] for edge in candidate.evidence["route"]] == [
        "catalog",
        "saved_items",
    ]
    assert plan.recommended_call.cli == "aua goto 'open saved items'"


def test_plan_keeps_unsafe_only_goto_discoverable_as_review_not_execution() -> None:
    app = AppMap(
        package=_PKG,
        screens={"home": _record("home"), "saved_items": _record("saved_items")},
        routes=[
            RouteEdge(
                id="unsafe-only",
                from_screen="home",
                to_screen="saved_items",
                action="open catalog://saved-items",
                steps=[RouteStep(kind="open-link", arg="catalog://saved-items")],
                status="verified",
                last_seen=_NOW,
            )
        ],
    )

    plan = plan_goal_session(
        "open saved items",
        _observation(),
        app=app,
        current_screen="home",
    )

    candidate = plan.candidates[0]
    assert candidate.safe is False and candidate.status == "requires_review"
    assert candidate.call.kind == "goto_plan" and candidate.call.executes is False
    assert plan.selected_candidate is None
    assert plan.recommended_call.cli == "aua goto 'open saved items' --plan"
    assert plan.recommended_call.executes is False


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
    assert {risk["code"] for risk in candidate.risks} == {
        "app_lifecycle",
        "arrival_unverified",
    }
    assert plan.selected_candidate is None
    assert plan.recommended_call.kind == "network_offline"
    assert plan.recommended_call.cli == "aua network offline --verify"
    assert "network offline --verify" in plan.warnings[0]


def test_unverified_flow_needs_positive_reach_evidence(monkeypatch: Any, tmp_path: Path) -> None:
    """A one-call recipe may execute explicitly, but absence of proof is not arrival."""
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    observed = _observation()
    candidate = GoalCandidate(
        id="flow:open_catalog",
        kind="flow",
        name="open_catalog",
        safe=False,
        status="requires_review",
        risks=[
            {
                "code": "arrival_unverified",
                "reason": "no declared arrival",
                "path": "arrival",
            }
        ],
        call=GoalCall(
            kind="flow_preview",
            cli="aua flow run open_catalog --dry-run",
            mcp={"tool": "flow_run", "arguments": {"name": "open_catalog", "dry_run": True}},
            reason="preview unverified recipe",
            executes=False,
        ),
    )
    plan = GoalSessionPlan(
        goal="open catalog",
        package=_PKG,
        current_screen="home",
        observation=observed,
        candidates=[candidate],
        recommended_call=candidate.call,
    )
    flow_calls: list[AnalyzeResult] = []

    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)
    monkeypatch.setattr(engine, "_goal_session_plan", lambda *_args: plan)
    monkeypatch.setattr(
        engine,
        "flow_run",
        lambda _name, **kwargs: (
            flow_calls.append(kwargs["_observation"])
            or {"ok": True, "arrival_verified": False, "arrival_status": "unverified"}
        ),
    )

    refused = engine.reach("open catalog")
    assert refused["ok"] is False and refused["code"] == "navigation_unavailable"
    assert flow_calls == []
    still_refused = engine.reach("open catalog", allow_unsafe=True)
    assert still_refused["ok"] is False
    assert flow_calls == []

    monkeypatch.setattr(
        engine,
        "await_predicate",
        lambda *_args, **_kwargs: ActionResult(
            ok=True,
            action="await",
            await_outcome="satisfied",
        ),
    )
    proven = engine.reach("open catalog", until="text:Catalog ready")
    assert proven["ok"] is True and proven["strategy"] == "flow"
    assert flow_calls == [observed]


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


def test_reach_never_authorizes_nested_execution_or_invalid_authored_arrival(
    monkeypatch,
) -> None:
    observed = _observation()
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    calls: list[str] = []
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: observed)
    monkeypatch.setattr(
        engine,
        "flow_run",
        lambda name, **_kwargs: calls.append(name) or {"ok": True},
    )

    for code in ("nested_execution", "arrival_invalid", "arrival_screen_invalid"):
        candidate = GoalCandidate(
            id=f"flow:{code}",
            kind="flow",
            name=code,
            safe=False,
            status="requires_review",
            risks=[{"code": code, "reason": "must be explicit", "path": "steps[0]"}],
            call=GoalCall(
                kind="flow_preview",
                cli=f"aua flow run {code} --dry-run",
                mcp={"tool": "flow_run", "arguments": {"name": code, "dry_run": True}},
                reason="preview",
                executes=False,
            ),
        )
        plan = GoalSessionPlan(
            goal="open catalog",
            package=_PKG,
            current_screen="home",
            observation=observed,
            candidates=[candidate],
            recommended_call=candidate.call,
        )
        monkeypatch.setattr(engine, "_goal_session_plan", lambda *_args, p=plan: p)
        result = engine.reach(
            "open catalog",
            until="text:Catalog ready",
            allow_unsafe=True,
            allow_destructive=True,
        )
        assert result["ok"] is False and result["code"] == "navigation_unavailable"

    assert calls == []


def test_planner_rejects_a_stale_or_missing_mapped_arrival() -> None:
    app = AppMap(package=_PKG, screens={"home": _record("home")})
    flow = Flow(
        name="missing_destination",
        app=_PKG,
        description="Open missing destination",
        arrival_screen="not_in_map",
        arrival_status="mapped",
        steps=[RouteStep(kind="tap", resource_id="savedItems")],
    )

    plan = plan_goal_session(
        "open missing destination",
        _observation(),
        app=app,
        flows=[flow],
    )

    candidate = next(item for item in plan.candidates if item.id == "flow:missing_destination")
    assert candidate.safe is False and candidate.call.executes is False
    assert {risk["code"] for risk in candidate.risks} == {"arrival_screen_invalid"}
    assert plan.selected_candidate is None

    without_map = plan_goal_session(
        "open missing destination",
        _observation(),
        flows=[flow],
    )
    candidate_without_map = next(
        item for item in without_map.candidates if item.id == "flow:missing_destination"
    )
    assert candidate_without_map.safe is False
    assert {risk["code"] for risk in candidate_without_map.risks} == {"arrival_screen_invalid"}


def test_goto_never_authorizes_nested_execution_before_an_earlier_route_step(
    monkeypatch,
    tmp_path,
) -> None:
    app = AppMap(
        package=_PKG,
        screens={"home": _record("home"), "saved_items": _record("saved_items")},
        routes=[
            RouteEdge(
                id="nested-route",
                from_screen="home",
                to_screen="saved_items",
                action="compound route",
                context_id="default",
                steps=[
                    RouteStep(kind="key", arg="back"),
                    RouteStep(kind="flow", arg="child"),
                ],
                status="verified",
                last_seen=_NOW,
            )
        ],
    )
    engine = Engine(
        make_config(memory={"enabled": True, "dir": str(tmp_path / "memory")}),
        device=FakeDevice(),
    )
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation())
    mem = type("M", (), {})()
    engine._mem = mem
    mem.load = lambda _package: app
    mem.set_last_goal = lambda *_args, **_kwargs: None
    mem.load_session = lambda _serial: type(
        "S",
        (),
        {
            "active_context_id": "default",
            "package": _PKG,
            "current_screen": "home",
            "last_goal": None,
        },
    )()
    calls: list[str] = []
    monkeypatch.setattr(engine, "key", lambda *_args, **_kwargs: calls.append("key"))

    result = engine.goto("saved items", allow_unsafe=True, allow_destructive=True)

    assert result["ok"] is False and result["code"] == "unsafe_route"
    assert {risk["code"] for risk in result["risks"]} == {"nested_execution"}
    assert calls == []


def test_one_incidental_goal_word_cannot_recommend_a_risky_flow() -> None:
    unrelated = Flow(
        name="reset_account_google_login",
        app=_PKG,
        description="Reset the account while online and land on onboarding",
        steps=[RouteStep(kind="launch-app", arg=_PKG)],
    )

    plan = plan_goal_session(
        "Establish a cached Grammar thread online",
        _observation(),
        flows=[unrelated],
    )

    assert not any(candidate.kind == "flow" for candidate in plan.candidates)
    assert plan.recommended_call.kind == "map_find"


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
    # The bootstrap observation is already the freshest available evidence. With no map,
    # route, or unambiguous control, do not spend another call re-querying capabilities.
    assert result["recommended_call"]["kind"] == "manual_observation"
    assert result["recommended_call"]["mcp"] is None
    assert result["relevant_capabilities"]
    assert result["cleanup_call"] == {
        "cli": "aua session finish",
        "mcp": {
            "tool": "session_finish",
            "arguments": {"session_id": result["session_id"]},
        },
        "reason": (
            "Run this once when finished. It restores only session-owned reversible state "
            "and returns the efficiency review; do not restore the network separately first."
        ),
    }


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
