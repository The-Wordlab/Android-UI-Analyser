"""aua asks about the screen you are standing on, and your next command carries the answer.

`meta.research_tasks` listed whatever was open, in map order, so it offered questions about
screens the caller had never seen and could not answer. Measured 2026-08-10: 970 open questions
had accumulated at roughly 130 per day of use, and not one had ever been answered — a backlog no
one-off cleanup fixes, because it regrows in a day.

The agent standing on a screen is the only one who knows what it is, and it is about to issue
another command anyway. So the question is scoped to the current screen and the answer rides on
whatever runs next: use the map, improve the map, in one call. It goes through `submit`, so an
inline answer gets the same transaction, validation and rollback id as a researched one.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.memory import AppMap, ScreenRecord, ask_about_current_screen
from android_ui_analyser.reconcile import _resolve_task

_WHEN = "2026-08-10T00:00:00Z"


def _screen(name: str, sid: str) -> ScreenRecord:
    return ScreenRecord(
        name=name, id=sid, signature=name, first_seen=_WHEN, last_seen=_WHEN, last_verified=_WHEN
    )


def _task(tid: str, sid: str, *, status: str = "open", kind: str = "poor_name") -> dict:
    return {
        "id": tid,
        "status": status,
        "issue_type": kind,
        "affected_ids": [sid],
        "questions": ["Which app destination or UI state does this screen represent?"],
    }


def _app() -> AppMap:
    app = AppMap(package="com.example.app")
    app.screens["screen_1"] = _screen("screen_1", "sid_1")
    app.screens["checkout"] = _screen("checkout", "sid_2")
    app.research_tasks = [
        _task("research_aaaa1111", "sid_1"),
        _task("research_bbbb2222", "sid_2"),
    ]
    return app


def test_it_asks_about_the_screen_you_are_on() -> None:
    ask = ask_about_current_screen(_app(), "screen_1")

    assert ask is not None
    assert ask["id"] == "research_aaaa1111"
    assert ask["about"] == "screen_1"


def test_it_never_asks_about_a_screen_you_cannot_see() -> None:
    """The whole reason 970 questions went unanswered: they were about somewhere else."""
    ask = ask_about_current_screen(_app(), "screen_1")

    assert ask is not None
    assert "research_bbbb2222" not in ask["id"]


def test_it_says_exactly_how_to_answer() -> None:
    ask = ask_about_current_screen(_app(), "screen_1")

    assert ask is not None
    assert "--answers research_aaaa1111=" in ask["how"]


def test_an_already_answered_question_is_not_asked_again() -> None:
    app = _app()
    app.research_tasks[0]["status"] = "applied"

    assert ask_about_current_screen(app, "screen_1") is None


def test_a_screen_off_the_map_raises_nothing() -> None:
    assert ask_about_current_screen(_app(), "never_seen") is None
    assert ask_about_current_screen(_app(), None) is None


def test_only_naming_questions_are_asked_inline() -> None:
    """A route conflict cannot be settled by looking at the screen; it needs research."""
    app = _app()
    app.research_tasks[0]["issue_type"] = "route_conflict"

    assert ask_about_current_screen(app, "screen_1") is None


def test_the_id_can_be_shortened_to_a_unique_tail() -> None:
    """The caller is retyping it into an unrelated command; the tail is enough."""
    assert _resolve_task(_app(), "aaaa1111")["id"] == "research_aaaa1111"


def test_an_ambiguous_tail_is_refused_rather_than_guessed() -> None:
    app = _app()
    app.research_tasks.append(_task("research_cccc1111", "sid_2"))

    with pytest.raises(ValueError, match="matches 2 open tasks"):
        _resolve_task(app, "1111")


def test_an_unknown_id_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown research task"):
        _resolve_task(_app(), "research_nope")
