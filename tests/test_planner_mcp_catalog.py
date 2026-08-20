"""Pure contract checks for planner calls advertised to MCP agents."""

from __future__ import annotations

from android_ui_analyser.flows import Flow
from android_ui_analyser.mcp_server import _tool_definitions
from android_ui_analyser.memory import AppMap, Deeplink, RouteEdge, RouteStep, ScreenRecord
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from android_ui_analyser.session import GoalSessionPlan, plan_goal_session

_NOW = "2026-01-02T03:04:05+00:00"
_PACKAGE = "com.example.catalog"


def _record(name: str) -> ScreenRecord:
    return ScreenRecord(
        name=name,
        signature=f"sig-{name}",
        first_seen=_NOW,
        last_seen=_NOW,
        last_verified=_NOW,
    )


def _observation(known_screen: str = "home") -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=_PACKAGE, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen=known_screen,
            device_serial="catalog-contract",
        ),
    )


def _advertised_calls(label: str, plan: GoalSessionPlan) -> list[tuple[str, str | None]]:
    calls = [(f"{label}.recommended_call", plan.recommended_call.mcp.get("tool"))]
    calls.extend(
        (f"{label}.candidate[{candidate.id}]", candidate.call.mcp.get("tool"))
        for candidate in plan.candidates
    )
    return calls


def test_every_goal_planner_mcp_call_uses_a_public_tool_name() -> None:
    rich_app = AppMap(
        package=_PACKAGE,
        screens={"home": _record("home"), "saved_items": _record("saved_items")},
        routes=[
            RouteEdge(
                id="saved-items-route",
                from_screen="home",
                to_screen="saved_items",
                action="tap Saved items",
                steps=[RouteStep(kind="tap", resource_id="saved_items")],
                status="verified",
                last_seen=_NOW,
            )
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
        app=_PACKAGE,
        description="Open saved items",
        arrival_screen="saved_items",
        arrival_status="mapped",
        steps=[RouteStep(kind="tap", resource_id="saved_items")],
    )
    deeplink_only_app = AppMap(
        package=_PACKAGE,
        screens={"home": _record("home")},
        deeplinks=[
            Deeplink(
                uri="catalog://saved-items",
                note="Open saved items",
                probed=True,
                landed="saved_items",
            )
        ],
    )

    plans = [
        (
            "ranked_navigation",
            plan_goal_session(
                "open saved items",
                _observation(),
                app=rich_app,
                current_screen="home",
                flows=[flow],
            ),
        ),
        (
            "deeplink_fallback",
            plan_goal_session(
                "open saved items",
                _observation(),
                app=deeplink_only_app,
                current_screen="home",
            ),
        ),
        (
            "already_arrived",
            plan_goal_session(
                "home",
                _observation(),
                app=rich_app,
                current_screen="home",
            ),
        ),
        ("offline", plan_goal_session("verify catalog offline", _observation())),
        ("unknown", plan_goal_session("open recommendations", _observation())),
    ]
    advertised = [call for label, plan in plans for call in _advertised_calls(label, plan)]
    public_names = {tool.name for tool in _tool_definitions()}
    unpublished = [(source, name) for source, name in advertised if name not in public_names]

    assert unpublished == [], (
        "Goal planning exposed MCP calls that agents cannot invoke through the public catalog: "
        f"{unpublished}"
    )
