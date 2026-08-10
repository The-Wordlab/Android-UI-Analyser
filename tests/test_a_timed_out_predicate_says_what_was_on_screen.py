"""`await_outcome: timeout` alone cannot tell a wrong predicate from a broken app.

Run 8 of the fresh-agent series (2026-08-10) waited on `text:No results` against a screen reading
"No apps found", and on `text:Sign` against one headed "Create your account". Both actions had
landed. Both burned the full 30s default and then reported a timeout the agent had to diagnose by
eye, writing afterwards that it "made it seem like something was wrong when actually the UI had
settled" and asking for exactly this: a line saying the action succeeded but the predicate did not
match. That was 60s of a 165s run — the slowest of the series.

`nearest_elements` already answers "you asked for X, here is what is actually there" when a
selector matches nothing. A timed-out predicate is the same question, so it gets the same answer.
"""

from __future__ import annotations

from types import SimpleNamespace

from android_ui_analyser.cli import _await_timeout_note, _predicate_needle
from android_ui_analyser.schema import Element


def _screen(*texts: str) -> SimpleNamespace:
    elements = [
        Element(
            id=i,
            type="TextView",
            bounds=[0, i * 10, 100, i * 10 + 10],
            center=[50, i * 10 + 5],
            text=t,
        )
        for i, t in enumerate(texts)
    ]
    return SimpleNamespace(observation=SimpleNamespace(elements=elements))


def test_the_needle_is_the_literal_the_caller_hoped_to_see() -> None:
    assert _predicate_needle("text:No results") == "No results"
    assert _predicate_needle("!text:Loading") == "Loading"
    assert _predicate_needle("rid:createButton") == "createButton"
    assert _predicate_needle("text:Done,rid:card") == "Done", "the first term is the subject"


def test_it_says_the_action_landed_and_the_predicate_did_not() -> None:
    note = _await_timeout_note("text:No results", 30000, _screen("No apps found"))

    assert "the action landed" in note, note
    assert "the predicate, not the app" in note, note
    assert "30000ms" in note, note


def test_it_shows_the_label_that_was_actually_there() -> None:
    note = _await_timeout_note("text:No results", 30000, _screen("No apps found", "Cancel"))

    assert "No apps found" in note, "the whole point is to hand over the real label"


def test_it_names_the_right_budget_flag() -> None:
    """Run 8 passed `--timeout 10000` and still waited 30s; the knob is `--until-timeout`."""
    note = _await_timeout_note("text:Sign", 30000, _screen("Create your account"))

    assert "--until-timeout" in note, note


def test_it_points_at_the_cheaper_predicate_shape() -> None:
    note = _await_timeout_note("text:Sign", 30000, _screen("Create your account"))

    assert "rid:<target>" in note, note


def test_an_empty_screen_still_gets_an_answer() -> None:
    note = _await_timeout_note("text:Whatever", 5000, SimpleNamespace(observation=None))

    assert "the action landed" in note, note
    assert "Re-read the screen" in note, note
