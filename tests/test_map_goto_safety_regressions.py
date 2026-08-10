"""Regressions from an app-agnostic exploratory navigation audit.

Fixtures deliberately use fictional screens and selectors. The failures are properties of
screen identity and route replay, not knowledge about any tested product.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.memory import AppMemoryStore, RouteStep, _shortest_path
from test_memory import APPS, HOME, P, _elements, _engine, _hier, _node, _store
from test_navigation import IMAGES, ScriptedDevice


def _sparse_surface(title: str, *, renderer: str | None = None) -> str:
    nodes = [
        _node(
            "android.view.View",
            rid="x:id/action_bar_root",
            b="[0,80][1080,2200]",
        ),
        _node(
            "android.widget.Button",
            rid="x:id/buttonNavBack",
            desc="Back",
            clk=True,
            b="[20,100][120,200]",
        ),
        _node(
            "android.widget.TextView",
            text=title,
            rid="x:id/surfaceTitle",
            b="[140,110][900,200]",
        ),
    ]
    if renderer:
        nodes.append(
            _node(
                "android.view.View",
                rid=f"x:id/{renderer}",
                b="[40,280][1040,1800]",
            )
        )
    return _hier(*nodes)


def test_sparse_library_does_not_match_mailbox_from_generic_chrome(tmp_path: Path) -> None:
    store = _store(tmp_path)
    library = store.record_screen(
        package=P,
        elements=_elements(_sparse_surface("Specimen Library")),
        activity=".SingleHost",
        name_hint="specimen_library",
    )

    mailbox = store.record_screen(
        package=P,
        elements=_elements(_sparse_surface("Courier Mailbox")),
        activity=".SingleHost",
        name_hint="courier_mailbox",
    )

    assert mailbox.created is True
    assert mailbox.name != library.name


def test_unrelated_formula_renderers_need_a_discriminative_anchor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    preview = store.record_screen(
        package=P,
        elements=_elements(_sparse_surface("Formula Preview", renderer="latexView")),
        activity=".SingleHost",
        name_hint="formula_preview",
    )

    tutor = store.record_screen(
        package=P,
        elements=_elements(_sparse_surface("Equation Tutor", renderer="latexView")),
        activity=".SingleHost",
        name_hint="equation_tutor",
    )

    assert tutor.created is True
    assert tutor.name != preview.name


def test_one_known_destination_observation_does_not_self_verify_a_route(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    serial = "single-observation"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)

    store.observe_action(
        serial, RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog")
    )
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)

    app = store.load(P)
    edge = next(route for route in app.routes if route.from_screen == "home")
    assert edge.status == "provisional"
    assert _shortest_path(app, "catalog", start="home") == []


def test_conflicting_verified_origin_action_edges_are_both_quarantined(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_screen(package=P, elements=_elements(IMAGES), name_hint="studio")
    step = [RouteStep(kind="tap", label="Open", resource_id="open_destination")]

    store.record_route(P, "home", "catalog", steps=step)
    store.record_route(P, "home", "studio", steps=step)

    app = store.load(P)
    conflicts = [route for route in app.routes if route.from_screen == "home"]
    assert {route.status for route in conflicts} == {"provisional"}
    assert all("conflicting destination" in (route.rejection_reason or "") for route in conflicts)
    assert _shortest_path(app, "catalog", start="home") == []
    assert _shortest_path(app, "studio", start="home") == []


def test_pathfinding_quarantines_conflicts_already_marked_verified(tmp_path: Path) -> None:
    """Old maps may contain conflicts written before record-time demotion existed."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_screen(package=P, elements=_elements(IMAGES), name_hint="studio")
    step = [RouteStep(kind="tap", label="Open", resource_id="open_destination")]
    store.record_route(P, "home", "catalog", steps=step)
    store.record_route(P, "home", "studio", steps=step)
    app = store.load(P)
    for route in app.routes:
        route.status = "verified"  # simulate a pre-fix persisted map
    store.save(app)

    legacy_conflict = store.load(P)
    assert {route.status for route in legacy_conflict.routes} == {"verified"}
    assert _shortest_path(legacy_conflict, "catalog", start="home") == []
    assert _shortest_path(legacy_conflict, "studio", start="home") == []


def test_loading_snapshot_retains_action_for_the_settled_destination(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    serial = "loading-shell"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial, RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog")
    )
    loading = _hier(
        _node(
            "android.widget.TextView",
            text="Loading content",
            rid="x:id/loadingLabel",
            b="[40,300][1040,400]",
        )
    )

    assert (
        store.observe_screen_passive(
            serial,
            package=P,
            elements=_elements(loading),
            activity=".SingleHost",
            screen_height=800,
        )
        is None
    )
    pending = store.load_session(serial)
    assert pending.current_screen == "home" and len(pending.pending) == 1

    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)
    app = store.load(P)
    assert set(app.screens) == {"home", "catalog"}
    edge = next(route for route in app.routes if route.from_screen == "home")
    assert edge.to_screen == "catalog"
    assert [step.resource_id for step in edge.steps] == ["nav_catalog"]


def test_new_top_level_intent_supersedes_unresolved_pending_actions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    serial = "fresh-top-level-intent"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial, RouteStep(kind="tap", label="Open panel", resource_id="buttonPanel")
    )
    store.observe_action(
        serial, RouteStep(kind="tap", label="Catalog", resource_id="bottomNavCatalog")
    )

    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)

    edge = next(route for route in store.load(P).routes if route.from_screen == "home")
    assert [step.resource_id for step in edge.steps] == ["bottomNavCatalog"]


def test_same_package_monster_route_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    serial = "monster-route"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    for index in range(5):
        store.observe_action(
            serial,
            RouteStep(
                kind="tap",
                label=f"Step {index}",
                resource_id=f"wizardStep{index}",
            ),
        )

    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)

    edge = next(route for route in store.load(P).routes if route.from_screen == "home")
    assert edge.status == "rejected"
    assert "too many destination actions" in (edge.rejection_reason or "")


def _route_map(tmp_path: Path) -> AppMemoryStore:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_screen(package=P, elements=_elements(IMAGES), name_hint="studio")
    return store


def test_goto_accepts_the_final_target_reached_before_an_intermediate(tmp_path: Path) -> None:
    store = _route_map(tmp_path)
    store.record_route(
        P,
        "home",
        "catalog",
        steps=[RouteStep(kind="tap", label="Catalog", resource_id="nav_apps")],
    )
    store.record_route(
        P,
        "catalog",
        "studio",
        steps=[RouteStep(kind="tap", label="Studio", resource_id="tool_images")],
    )
    device = ScriptedDevice([HOME, IMAGES], package=P, serial="early-target")
    engine = _engine(tmp_path, device)

    result = engine.goto("studio")

    assert result["ok"] is True and result["arrived"] is True
    assert result["early_arrival"] is True
    assert result["final_screen"] == "studio"
    assert sum(call[0] == "click" for call in device.calls) == 1


def test_goto_reports_partial_edge_and_replans_without_repeating_it(tmp_path: Path) -> None:
    store = _route_map(tmp_path)
    store.record_route(
        P,
        "home",
        "studio",
        steps=[
            RouteStep(kind="tap", label="Catalog", resource_id="nav_apps"),
            RouteStep(kind="tap", label="Missing control", resource_id="missing_control"),
        ],
    )
    store.record_route(
        P,
        "catalog",
        "studio",
        steps=[RouteStep(kind="tap", label="Studio", resource_id="tool_images")],
    )
    device = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="partial-replan")
    engine = _engine(tmp_path, device)

    result = engine.goto("studio")

    assert result["ok"] is True and result["replanned_from"] == "catalog"
    assert result["hops"][0]["partial"] is True
    assert [step["step"] for step in result["hops"][0]["executed_steps"]] == [
        "tap 'Catalog'"
    ]
    assert result["hops"][-1]["known_screen"] == "studio"
    assert sum(call[0] == "click" for call in device.calls) == 2
