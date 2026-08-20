"""`--find` and `goto` required every word of the query to appear on the screen, "open" included.

That is right for a keyword query and fatal for a goal. A tester types "open the settings screen
and check the theme toggle"; the words *settings* and *theme* are on the map, and the words *open*,
*the*, *screen*, *check* are on no screen anywhere, so the all-terms-must-match test could never
pass. `resolve_goal` did not return a worse answer — it returned `None`.

The parameterized synthetic corpus wraps screen names in sentence shapes a tester actually types.
The keyword control is unchanged by construction, not by luck: the query is searched exactly as typed
first, and the words describing the asking are only stripped out on a second pass, reached solely
when the first found nothing. Loosening is therefore purely additive.

What is still missing is a synonym: a goal saying "premium upsell" about a screen labelled "Go
Premium" needs *upsell* to be recognised as vocabulary the app never uses, and no amount of
stopword removal gets there. That gap is real and is not addressed here.
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
    search_terms,
)

_WHEN = "2026-08-18T00:00:00Z"


def _screen(name: str, passive: list[str], controls: list[str] | None = None) -> ScreenRecord:
    return ScreenRecord(
        name=name,
        signature=f"sig_{name}",
        first_seen=_WHEN,
        last_seen=_WHEN,
        last_verified=_WHEN,
        key_elements=[KeyElement(type="TextView", label=p, clickable=False) for p in passive]
        + [KeyElement(type="Button", label=c, clickable=True) for c in (controls or [])],
    )


def _app() -> AppMap:
    app = AppMap(package="com.example.app")
    app.screens["settings"] = _screen("settings", ["Settings"], ["Theme", "Notifications"])
    app.screens["theme_picker"] = _screen("theme_picker", ["Theme"], ["Dark", "Light"])
    app.screens["notifications"] = _screen("notifications", ["Notifications"], ["Push"])
    # Links to settings but is not settings; must never win a goal naming settings.
    app.screens["home"] = _screen("home", ["Home"], ["Settings", "Search"])
    app.routes.append(
        RouteEdge(
            from_screen="home", to_screen="settings", action="tap 'Settings'", last_seen=_WHEN
        )
    )
    return app


@pytest.mark.parametrize(
    "goal",
    [
        "open the settings screen",
        "go to settings and check it loads",
        "navigate to the settings page",
        "verify the settings screen is shown",
        "tap through to settings",
        "open settings and verify the theme toggle",
    ],
)
def test_a_sentence_shaped_goal_resolves(goal: str) -> None:
    """This is the whole bug: every one of these used to return None."""
    assert resolve_goal(_app(), goal) == "settings"


def test_the_screen_outranks_the_row_that_merely_opens_it() -> None:
    """A clickable 'Settings' row on home describes how to leave home, not where you are."""
    assert resolve_goal(_app(), "open the settings screen") == "settings"


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("settings", "settings"),
        ("theme", "theme_picker"),  # the screen named for it beats the one holding the control
        ("notifications", "notifications"),
        ("theme picker", "theme_picker"),
    ],
)
def test_the_keyword_form_is_untouched(goal: str, expected: str) -> None:
    """The query is tried exactly as typed before anything is stripped from it."""
    assert resolve_goal(_app(), goal) == expected


def test_a_clause_break_lets_a_two_part_goal_still_find_its_destination() -> None:
    """A clause break marks where the destination ends, so the trailing clause is ignored."""
    assert resolve_goal(_app(), "open settings and check the theme toggle state") == "settings"


def test_a_goal_with_no_clause_break_is_the_honest_boundary() -> None:
    """Pinned so the limit is visible rather than discovered later as a regression.

    Nothing marks where the destination ends, so "toggle" -- a real content word on no screen --
    stays a requirement and every-term-must-match refuses. Answering needs either partial credit,
    which re-opens the sieve `test_a_word_that_is_nowhere_still_matches_nothing` forbids, or
    knowing "toggle" names the same thing as "Theme", which is semantics rather than stopwords.
    """
    goal = "please open settings so we can verify the theme toggle works correctly"

    assert resolve_goal(_app(), goal) is None


def test_a_synonym_is_still_out_of_reach() -> None:
    """The remaining gap, stated plainly: the app says "Go Premium", the tester says "upsell"."""
    app = AppMap(package="com.example.app")
    app.screens["paywall"] = _screen("paywall", ["Go Premium", "Yearly plan"], ["Subscribe"])

    assert resolve_goal(app, "verify the yearly plan is shown") == "paywall"
    assert resolve_goal(app, "verify the upsell is shown") is None


def test_a_word_that_is_nowhere_still_matches_nothing() -> None:
    """Loosening must not turn a specific query into a sieve. This is the precision guarantee."""
    assert _find_targets(_app(), "settings zzzqqqxyz") == []


def test_a_query_of_nothing_but_noise_still_searches_for_those_words() -> None:
    """`--find "open"` must look for the word, not fall back to matching every screen."""
    assert search_terms("open") == ["open"]
    assert _find_targets(_app(), "open") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_does_not_match_everything(query: str) -> None:
    assert _find_targets(_app(), query) == []


def test_search_terms_keeps_the_target_and_drops_the_asking() -> None:
    assert search_terms("verify the theme picker screen is shown") == ["theme", "picker"]
    assert search_terms("go to notifications and confirm it loads") == ["notifications"]
