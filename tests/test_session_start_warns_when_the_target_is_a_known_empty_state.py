"""``session start`` accepted a goal its own map already knew could not be satisfied.

A live run was given "open the Drafts tab and show draft cards with different states". AUA had
mapped that exact screen ninety seconds earlier, and the record it kept included the anchor
``tx:no drafts`` — the screen's own words for "there is nothing here". Bootstrap said nothing.
The agent navigated there correctly, found nothing, and spent twelve minutes assuming it had
navigated wrong: seven relaunches, two invented resource ids, and a lost login.

The map already held the answer. So when the screen a plan is about to navigate to was last seen
showing an empty state, the plan says so. The goal may still be right — a draft may need
creating first — but that is a decision for the agent to make in the first ten seconds, not the
twelfth minute.

The signal is the screen's own copy, not a guess: an anchor reading "no <something>", "nothing
here/yet", "empty", or a resource id that says ``emptyState``.
"""

from __future__ import annotations

from android_ui_analyser.memory import AppMap, RouteEdge, RouteStep, ScreenRecord
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from android_ui_analyser.session import empty_state_anchor, plan_goal_session

_PKG = "com.example.app"
_NOW = "2026-08-21T10:00:00+00:00"


def _record(name: str, anchors: list[str] | None = None) -> ScreenRecord:
    return ScreenRecord(
        name=name,
        signature=f"sig-{name}",
        first_seen=_NOW,
        last_seen=_NOW,
        last_verified=_NOW,
        anchors=anchors or [],
    )


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=_PKG, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=12,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="home",
            device_serial="goal-emulator",
        ),
    )


# ----------------------------------------------------------------- reading the screen's words


def test_the_screens_own_empty_copy_is_recognised() -> None:
    assert empty_state_anchor(_record("drafts", ["tx:drafts", "tx:no drafts"])) == "tx:no drafts"


def test_an_empty_state_resource_id_counts() -> None:
    assert empty_state_anchor(_record("apps", ["id:emptystatecardapps"])) == "id:emptystatecardapps"


def test_nothing_here_counts() -> None:
    assert empty_state_anchor(_record("inbox", ["tx:nothing here yet"])) == "tx:nothing here yet"


def test_an_ordinary_populated_screen_is_not_flagged() -> None:
    record = _record("drafts", ["tx:drafts", "tx:my first app", "id:draftlist"])
    assert empty_state_anchor(record) is None


def test_a_word_merely_starting_with_no_is_not_an_empty_state() -> None:
    # "notifications" and "notes" start with "no" — matching those would cry wolf on half an app.
    assert empty_state_anchor(_record("s", ["tx:notifications", "tx:notes", "cd:november"])) is None


def test_a_screen_with_no_anchors_makes_no_claim() -> None:
    assert empty_state_anchor(_record("unknown")) is None


# ----------------------------------------------------------------- surfaced by the planner


def _app_with_empty_drafts() -> AppMap:
    app = AppMap(package=_PKG)
    app.screens["home"] = _record("home", ["id:home"])
    app.screens["drafts"] = _record("drafts", ["tx:drafts", "tx:no drafts"])
    app.routes.append(
        RouteEdge(
            id="route-drafts",
            from_screen="home",
            to_screen="drafts",
            action="tap 'Drafts'",
            context_id="default",
            steps=[RouteStep(kind="tap", label="Drafts")],
            status="verified",
            last_seen=_NOW,
        )
    )
    return app


def test_the_plan_warns_that_the_drafts_target_was_last_seen_empty() -> None:
    plan = plan_goal_session(
        "open the drafts tab", _observation(), app=_app_with_empty_drafts(), current_screen="home"
    )
    warning = next((w for w in plan.warnings if "empty" in w.lower()), None)
    assert warning is not None, f"no empty-state warning in {plan.warnings}"
    assert "drafts" in warning.lower()
    # Quote the screen's own words so the agent can trust the claim without re-navigating.
    assert "no drafts" in warning.lower()


def test_a_populated_target_produces_no_such_warning() -> None:
    app = _app_with_empty_drafts()
    app.screens["drafts"] = _record("drafts", ["tx:drafts", "tx:my first app"])
    plan = plan_goal_session(
        "open the drafts tab", _observation(), app=app, current_screen="home"
    )
    assert not [w for w in plan.warnings if "empty" in w.lower()]
