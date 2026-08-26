"""A destination name and current-screen arrival evidence are different facts."""

from __future__ import annotations

import pytest

from android_ui_analyser.memory import (
    DEFAULT_CONTEXT_ID,
    AppMap,
    RouteEdge,
    ScreenRecord,
    arrival_destination_terms,
    same_screen_family,
    screen_is_root,
    target_arrival_evidence,
)
from android_ui_analyser.schema import Element

_WHEN = "2026-08-14T00:00:00Z"


def _screen(name: str, **updates: object) -> ScreenRecord:
    base = ScreenRecord(
        name=name,
        signature=name,
        first_seen=_WHEN,
        last_seen=_WHEN,
        last_verified=_WHEN,
        anchors=[],
    )
    return base.model_copy(update=updates)


def _element(
    *,
    text: str | None = None,
    desc: str | None = None,
    rid: str | None = None,
    clickable: bool = False,
    y: int = 120,
) -> Element:
    return Element(
        id=0,
        type="TextView",
        text=text,
        content_desc=desc,
        resource_id=rid,
        clickable=clickable,
        bounds=(20, y, 900, y + 80),
        center=(460, y + 40),
    )


@pytest.mark.parametrize(
    ("updates", "goal", "value"),
    [
        ({}, "open saved items and verify the result count", "saved_items"),
        ({"canonical_name": "saved_items"}, "saved items", "saved_items"),
        ({"logical_name": "saved_items"}, "saved items", "saved_items"),
        ({"aliases": ["saved_items"]}, "saved items", "saved_items"),
    ],
)
def test_fresh_mapped_identity_is_target_specific_proof(
    updates: dict[str, object], goal: str, value: str
) -> None:
    name = "saved_items" if not updates else "generic_panel"
    app = AppMap(package="com.example.app", screens={name: _screen(name, **updates)})

    evidence = target_arrival_evidence(app, name, goal, [])

    assert evidence == {"source": "mapped_identity", "target": name, "value": value}


def test_clickable_destination_row_does_not_prove_current_screen_arrival() -> None:
    app = AppMap(
        package="com.example.app",
        screens={"catalog": _screen("catalog", anchors=["tx:display preferences"])},
    )
    row = _element(text="Display preferences", clickable=True)

    assert target_arrival_evidence(app, "catalog", "Display preferences", [row]) is None


def test_bottom_navigation_does_not_cancel_mapped_identity_proof() -> None:
    app = AppMap(
        package="com.example.app",
        screens={"apps_hub": _screen("apps_hub", logical_name="apps_hub")},
    )
    bottom_tab = _element(text="Apps Hub", clickable=True, y=2300)

    evidence = target_arrival_evidence(
        app,
        "apps_hub",
        "Open Apps Hub",
        [bottom_tab],
        screen_height=2400,
    )

    assert evidence == {
        "source": "mapped_identity",
        "target": "apps_hub",
        "value": "apps_hub",
    }


def test_context_variants_share_a_family_but_distinct_states_do_not() -> None:
    app = AppMap(
        package="com.example.app",
        screens={
            "apps_hub": _screen("apps_hub", logical_name="apps_hub"),
            "apps_hub__experiment_a": _screen(
                "apps_hub__experiment_a",
                logical_name="apps_hub",
            ),
            "apps_hub__loading": _screen(
                "apps_hub__loading",
                logical_name="apps_hub",
                state="loading",
            ),
        },
    )

    assert same_screen_family(app, "apps_hub", "apps_hub__experiment_a") is True
    assert same_screen_family(app, "apps_hub", "apps_hub__loading") is False


def test_container_title_does_not_prove_a_clickable_child_destination() -> None:
    app = AppMap(
        package="com.example.app",
        screens={"settings": _screen("settings")},
    )
    title = _element(text="Settings")
    row = _element(text="Saved items", clickable=True, y=300)

    assert (
        target_arrival_evidence(
            app,
            "settings",
            "Open saved items in Settings",
            [title, row],
            screen_height=2400,
        )
        is None
    )


def test_current_non_clickable_title_can_prove_arrival() -> None:
    app = AppMap(
        package="com.example.app",
        screens={"generic_panel": _screen("generic_panel", stale=True)},
    )
    title = _element(text="Display preferences")

    evidence = target_arrival_evidence(
        app,
        "generic_panel",
        "Reach Display preferences and verify the brightness control",
        [title],
        screen_height=2400,
    )

    assert evidence == {
        "source": "visible_title",
        "target": "generic_panel",
        "value": "Display preferences",
    }


def test_current_non_clickable_remembered_anchor_can_prove_arrival() -> None:
    app = AppMap(
        package="com.example.app",
        screens={
            "generic_panel": _screen(
                "generic_panel",
                stale=True,
                anchors=["id:display_preferences"],
            )
        },
    )
    anchor = _element(rid="com.example.app:id/display_preferences", y=900)

    evidence = target_arrival_evidence(
        app,
        "generic_panel",
        "Display preferences",
        [anchor],
        screen_height=2400,
    )

    assert evidence == {
        "source": "visible_anchor",
        "target": "generic_panel",
        "value": "id:display_preferences",
    }


def test_stale_alias_without_current_visible_proof_is_not_arrival() -> None:
    app = AppMap(
        package="com.example.app",
        screens={
            "generic_panel": _screen(
                "generic_panel",
                aliases=["display_preferences"],
                stale=True,
            )
        },
    )

    assert target_arrival_evidence(app, "generic_panel", "Display preferences", []) is None


def test_unknown_target_and_empty_goal_have_no_arrival_evidence() -> None:
    app = AppMap(package="com.example.app", screens={"catalog": _screen("catalog")})

    assert target_arrival_evidence(app, "missing", "Catalog", []) is None
    assert target_arrival_evidence(app, "catalog", "   ", []) is None


@pytest.mark.parametrize(
    "goal",
    [
        "Tap History archive from these choices: Grammar, History archive, Physics",
        "Press History archive among the visible destinations",
        "Click History archive and verify the result count",
        "Choose History archive among Grammar, History archive, and Physics",
    ],
)
def test_action_verbs_extract_only_the_requested_destination(goal: str) -> None:
    assert arrival_destination_terms(goal) == ["history", "archive"]


def test_screen_is_root_respects_routes_and_capture_context() -> None:
    app = AppMap(
        package="com.example.app",
        screens={
            "catalog": _screen("catalog"),
            "details": _screen("details"),
            "other_context": _screen("other_context", context_id="flags-fictional"),
        },
        routes=[
            RouteEdge(
                from_screen="catalog",
                to_screen="details",
                action="tap 'Details'",
                context_id=DEFAULT_CONTEXT_ID,
                last_seen=_WHEN,
            )
        ],
    )

    assert screen_is_root(app, "catalog", DEFAULT_CONTEXT_ID) is True
    assert screen_is_root(app, "details", DEFAULT_CONTEXT_ID) is False
    assert screen_is_root(app, "other_context", DEFAULT_CONTEXT_ID) is False
    assert screen_is_root(app, "missing", DEFAULT_CONTEXT_ID) is False
