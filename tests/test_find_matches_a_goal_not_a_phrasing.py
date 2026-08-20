"""`--find` took the query as one literal substring, so only a caller who knew the phrasing won.

A goal is a phrase — "catalog search", "create account" — and the map holds those words scattered
across a screen's name, its anchors, its key-element labels and the routes that reach it. Matching
the query whole meant `--find "search"` and `--find "catalog"` each answered while `--find "catalog
search"` reported "no matching screen in memory" about a map that held the route.

The synthetic map demonstrates the failure without relying on any tested application's names or
route counts.

Terms now match independently, weighted by where they hit: what the screen is called, then what is
on it, then how you reach it. The route is the weakest evidence because every route string names
the screens it passes through, so without that split a detail screen reached *via* a Catalog tab
outranks the Catalog screen itself.

This fixes "returns nothing"; it does not claim the top hit is always the best one. Ordering
within a match is still crude, and the deeper problem is untouched: an auto-named screen like
`screen__add38371` tells a caller nothing about whether it is what they wanted.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.memory import (
    AppMap,
    KeyElement,
    RouteEdge,
    ScreenRecord,
    _find_targets,
    resolve_goal,
)

_WHEN = "2026-08-10T00:00:00Z"


def _screen(name: str, anchors: list[str]) -> ScreenRecord:
    return ScreenRecord(
        name=name,
        signature=name,
        first_seen=_WHEN,
        last_seen=_WHEN,
        last_verified=_WHEN,
        anchors=anchors,
    )


def _app() -> AppMap:
    app = AppMap(package="com.example.app")
    app.screens["search_results"] = _screen(
        "search_results", ["tx:no catalog products found", "tx:create product"]
    )
    app.screens["signup_sheet"] = _screen(
        "signup_sheet", ["tx:create your account", "tx:continue with example id"]
    )
    app.screens["unrelated"] = _screen("unrelated", ["tx:hello"])
    app.screens["elsewhere"] = _screen(
        "elsewhere", ["tx:hello"]
    )  # only reachable *via* a Catalog tap
    app.routes.append(
        RouteEdge(
            from_screen="home",
            to_screen="elsewhere",
            action="tap 'Catalog'",
            last_seen=_WHEN,
        )
    )
    return app


def test_a_single_word_still_works() -> None:
    assert "search_results" in _find_targets(_app(), "search")


def test_a_goal_phrase_finds_something_at_all() -> None:
    """This is the whole bug: two words that never appear adjacent returned nothing."""
    assert _find_targets(_app(), "no products") != []


def test_the_terms_do_not_have_to_be_adjacent() -> None:
    found = _find_targets(_app(), "products create")

    assert "search_results" in found, (
        "'create product' and 'no products found' are both on that screen"
    )


def test_a_term_that_is_nowhere_still_matches_nothing() -> None:
    assert _find_targets(_app(), "products zzzqqqxyz") == []


def test_the_name_outranks_the_contents() -> None:
    found = _find_targets(_app(), "signup")

    assert found[0] == "signup_sheet"


def test_what_is_on_the_screen_outranks_how_you_reach_it() -> None:
    """Every route names the screens it passes through, so route text is the weakest signal."""
    found = _find_targets(_app(), "catalog")

    assert found.index("search_results") < found.index("elsewhere")


def test_incoming_route_outranks_clickable_row_that_only_opens_destination() -> None:
    app = AppMap(package="com.example.app")
    app.screens["catalog"] = _screen("catalog", ["id:catalog"])
    app.screens["catalog"].key_elements = [
        KeyElement(type="Button", label="Display preferences", clickable=True)
    ]
    app.screens["generic_panel"] = _screen("generic_panel", ["id:panel"])
    app.routes.append(
        RouteEdge(
            from_screen="catalog",
            to_screen="generic_panel",
            action="tap 'Display preferences'",
            last_seen=_WHEN,
        )
    )

    found = _find_targets(app, "Display preferences")

    assert found == ["generic_panel", "catalog"]
    assert resolve_goal(app, "Display preferences", start="catalog") == "generic_panel"


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_does_not_match_everything(query: str) -> None:
    assert _find_targets(_app(), query) == []
