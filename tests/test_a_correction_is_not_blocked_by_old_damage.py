"""A correction must leave the map no worse than it found it — not find it perfect.

`reconcile submit` validated the *candidate* map absolutely and refused the whole transaction on
any violation, including ones already sitting in the map it started from. Corrections are the only
mechanism that can repair a map, so this locked the repair shut with the damage it exists to
repair.

Measured 2026-08-10: one dangling route in 623 — a `from_screen` naming a screen that no longer
existed — vetoed every correction. 210 open research tasks had accumulated over a week against a
map that could not accept an answer, and a rename of an entirely unrelated screen came back as
`correction rejected: route route_… has missing source …`. Nothing said the reason was pre-existing
or that the caller's own report was fine.

Errors the correction introduces still reject it. Errors it inherits are carried onto the event
record instead, so inherited damage stays visible rather than becoming invisible.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.memory import AppMap, RouteEdge, ScreenRecord
from android_ui_analyser.reconcile import validate_map

_WHEN = "2026-08-10T00:00:00Z"


def _screen(name: str, sid: str) -> ScreenRecord:
    return ScreenRecord(
        name=name, id=sid, signature=name, first_seen=_WHEN, last_seen=_WHEN, last_verified=_WHEN
    )


def _map(*, dangling: bool) -> AppMap:
    app = AppMap(package="com.example.app")
    app.screens["home"] = _screen("home", "screen_home")
    app.screens["settings"] = _screen("settings", "screen_settings")
    app.routes.append(
        RouteEdge(
            id="route_ok",
            from_screen="home",
            to_screen="settings",
            action="tap 'Settings'",
            last_seen=_WHEN,
        )
    )
    if dangling:
        app.routes.append(
            RouteEdge(
                id="route_orphan",
                from_screen="deleted_screen",
                to_screen="home",
                action="tap 'Back'",
                last_seen=_WHEN,
            )
        )
    return app


def test_a_healthy_map_validates_clean() -> None:
    assert validate_map(_map(dangling=False)) == []


def test_the_orphan_is_still_reported() -> None:
    """The fix must not stop detecting it — only stop letting it veto other work."""
    errors = validate_map(_map(dangling=True))

    assert any("missing source deleted_screen" in e for e in errors)


def test_an_inherited_error_is_not_attributed_to_the_correction() -> None:
    """This is the whole bug: identical error, present before and after, so nobody introduced it."""
    before = _map(dangling=True)
    after = _map(dangling=True)
    after.screens["settings"].name = "preferences"
    after.screens["preferences"] = after.screens.pop("settings")
    after.routes[0].to_screen = "preferences"

    introduced = [e for e in validate_map(after) if e not in set(validate_map(before))]

    assert introduced == [], "a rename elsewhere did not create the orphan"


def test_a_correction_that_really_breaks_the_map_is_still_caught() -> None:
    before = _map(dangling=False)
    after = _map(dangling=False)
    after.screens.pop("settings")  # a route still points at it

    introduced = [e for e in validate_map(after) if e not in set(validate_map(before))]

    assert introduced, "deleting a screen a route needs must still reject"
    assert any("missing target settings" in e for e in introduced)


def test_a_correction_that_repairs_inherited_damage_is_allowed() -> None:
    before = _map(dangling=True)
    after = _map(dangling=False)

    introduced = [e for e in validate_map(after) if e not in set(validate_map(before))]

    assert introduced == []
    assert validate_map(after) == [], "and the map comes out clean"


@pytest.mark.parametrize("dangling", [True, False])
def test_validation_is_a_pure_read(dangling: bool) -> None:
    app = _map(dangling=dangling)
    routes_before = len(app.routes)

    validate_map(app)

    assert len(app.routes) == routes_before, "validating must never mutate the map"
