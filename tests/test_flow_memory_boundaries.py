"""Recorded-flow segments and persistent map routes keep separate provenance."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.memory import RouteStep
from test_memory import APPS, HOME, P, _elements, _store


def test_passive_foreign_non_transit_observation_segments_a_b_a_actions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-boundary"
    foreign = "com.example.reader"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=P))

    store.observe_screen_passive(serial, package=foreign, elements=[])
    store.observe_action(serial, RouteStep(kind="key", arg="menu", package=foreign))
    store.observe_screen_passive(serial, package=P, elements=_elements(HOME))
    store.observe_action(serial, RouteStep(kind="key", arg="back", package=P))

    recent = store.load_session(serial).recent
    assert [(step.origin_package, step.capture_segment) for step in recent] == [
        (P, 0),
        (foreign, 1),
        (P, 2),
    ]
    newest = [step for step in recent if step.capture_segment == recent[-1].capture_segment]
    assert [step.arg for step in newest] == ["back"]


def test_passive_configured_transit_preserves_origin_segment(tmp_path: Path) -> None:
    transit = "com.example.auth"
    store = _store(tmp_path, transit_packages=[transit])
    serial = "fictional-transit"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=P))
    store.observe_screen_passive(serial, package=transit, elements=[])
    store.observe_action(serial, RouteStep(kind="key", arg="accept", package=transit))

    session = store.load_session(serial)
    assert session.package == P and session.capture_segment == 0
    assert [(step.origin_package, step.capture_segment) for step in session.recent] == [
        (P, 0),
        (P, 0),
    ]


def test_flag_context_change_separates_recorded_actions_before_flow_save(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-context-boundary"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial,
        RouteStep(kind="tap", resource_id="oldVariant", by="id", package=P),
    )

    context_b = store.activate_flag_context(
        serial,
        P,
        {"catalog_variant": "b"},
        verified=True,
    )
    store.observe_action(
        serial,
        RouteStep(kind="tap", resource_id="newVariant", by="id", package=P),
    )

    session = store.load_session(serial)
    assert session.capture_segment == 1
    assert session.active_context_id == context_b
    assert [step.capture_segment for step in session.recent] == [0, 1]
    assert [step.context_id for step in session.recent] == ["default", context_b]
    newest_suffix = [
        step
        for step in session.recent
        if step.capture_segment == session.capture_segment
    ]
    assert [step.resource_id for step in newest_suffix] == ["newVariant"]


def test_default_context_app_data_clear_separates_recorded_actions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-default-clear-boundary"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=P))

    store.clear_context(serial, P)

    cleared = store.load_session(serial)
    assert cleared.capture_segment == 1
    assert cleared.capture_boundary_reason == f"app data cleared for {P}"
    assert cleared.current_screen is None
    assert cleared.pending == []
    assert cleared.active_context_id == "default"

    store.observe_action(serial, RouteStep(kind="key", arg="back", package=P))

    session = store.load_session(serial)
    assert [step.capture_segment for step in session.recent] == [0, 1]
    assert [step.arg for step in session.recent] == ["enter", "back"]


def test_app_data_clear_for_foreign_package_keeps_owned_session_unchanged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-foreign-clear-noop"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=P))
    before = store.load_session(serial)

    store.clear_context(serial, "com.example.reader")

    assert store.load_session(serial) == before


def test_transit_lifecycle_mutation_separates_the_owner_journey(tmp_path: Path) -> None:
    transit = "com.example.auth"
    store = _store(tmp_path, transit_packages=[transit])
    serial = "fictional-transit-lifecycle"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=P))
    store.observe_screen_passive(serial, package=transit, elements=[])

    store.clear_context(serial, transit)
    store.observe_action(serial, RouteStep(kind="key", arg="accept", package=transit))

    session = store.load_session(serial)
    assert session.package == P
    assert session.capture_segment == 1
    assert [step.capture_segment for step in session.recent] == [0, 1]
    assert [step.origin_package for step in session.recent] == [P, P]


def test_same_app_stop_and_launch_are_hard_capture_boundaries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-lifecycle-boundary"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=P))

    store.mark_capture_boundary(serial, P, f"app process stopped for {P}")
    store.mark_capture_boundary(serial, P, f"app process launched for {P}")
    store.observe_action(serial, RouteStep(kind="key", arg="back", package=P))

    session = store.load_session(serial)
    assert session.capture_segment == 2
    assert [step.capture_segment for step in session.recent] == [0, 2]
    assert session.current_screen is None


def test_foreign_pending_flags_start_a_new_capture_segment(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-flag-deeplink-boundary"
    foreign = "com.example.reader"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial,
        RouteStep(kind="tap", resource_id="oldVariant", by="id", package=P),
    )
    store.set_pending_flags(serial, P, {"old_variant": "a"})

    store.set_pending_flags(serial, foreign, {"new_variant": "b"})
    store.observe_action(
        serial,
        RouteStep(kind="tap", resource_id="newVariant", by="id", package=foreign),
    )

    session = store.load_session(serial)
    assert session.package == foreign
    assert session.capture_segment == 1
    assert session.current_screen is None
    assert session.pending_flags == {"new_variant": "b"}
    assert [step.capture_segment for step in session.recent] == [0, 1]
    assert [step.origin_package for step in session.recent] == [P, foreign]


def test_full_observation_route_strips_capture_provenance_but_keeps_transit_package(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    serial = "fictional-route"
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial, RouteStep(kind="tap", resource_id="nav_apps", by="id", package=P)
    )
    transit = "com.example.auth"
    store.observe_action(serial, RouteStep(kind="key", arg="enter", package=transit))
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)

    route = store.load(P).routes[-1]
    assert route.steps[0].package is None
    assert route.steps[1].package == transit
    assert all(
        step.origin_package is None
        and step.context_id is None
        and step.capture_segment is None
        for step in route.steps
    )


def test_passive_route_strips_capture_provenance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    serial = "fictional-passive-route"
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial, RouteStep(kind="tap", resource_id="nav_apps", by="id", package=P)
    )
    assert store.observe_screen_passive(
        serial, package=P, elements=_elements(APPS), screen_height=800
    ) == "apps"
    route = store.load(P).routes[-1]
    assert route.steps[0].model_dump()["origin_package"] is None
    assert route.steps[0].model_dump()["context_id"] is None
    assert route.steps[0].model_dump()["capture_segment"] is None
