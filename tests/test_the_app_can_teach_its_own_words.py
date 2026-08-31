"""One app's vocabulary, learned once and reused, instead of re-derived per scenario.

The case that motivated this, from a real run: a scenario spec says **Feed**, the app's control is
labelled **Ideas**, and its resource id is `navBarSecondary`. `aua drive "open the feed tab"` scores
zero against every control on screen and reports `target_absent` with the target plainly visible.
Nothing is wrong with the rule — the words genuinely do not meet — and nothing was wrong with the
scenario either. The two vocabularies are simply different, and until now there was nowhere in AUA to
write that down: no glossary concept, nothing in goal resolution reading `app.knowledge`, and the
helper's word tables are `private static final` with no way to seed them.

**Why the goal is expanded rather than the rule taught.** The helper cannot read the app map, so
teaching `score()` would give the host lane a vocabulary the device lane does not have — two lanes
disagreeing about the same goal, which a test in `test_the_host_can_drive_without_the_helper.py`
exists to prevent. Folding the app's own words into the goal before it is sent anywhere fixes both
lanes at once, needs no protocol change, and needs no second implementation in Java.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.drive_rule import decide, expand_goal


def _projection(nodes: list[dict]) -> dict:
    return {"nodes": nodes, "more": False, "keys": [n["n"] for n in nodes]}


NAV = [
    {"n": "n1", "text": "Chat", "tap": True},
    {"n": "n2", "text": "Ideas", "tap": True},
    {"n": "n3", "text": "Apps", "tap": True},
]


# --------------------------------------------------------------------------- the case it exists for


def test_a_taught_word_reaches_a_control_the_goal_never_named() -> None:
    """The live failure, and the fix, in one assertion."""

    goal = "open the feed tab"
    assert decide(goal, _projection(NAV))["call"] == "handoff", "premise: this used to fail"

    taught = expand_goal(goal, {"feed": ["Ideas"]})
    got = decide(taught, _projection(NAV))
    assert got["call"] == "tap"
    assert got["n"] == "n2"


def test_an_untaught_goal_is_returned_unchanged() -> None:
    """No vocabulary, no behaviour change. This has to be free when nothing was taught."""

    assert expand_goal("open the apps section", {}) == "open the apps section"
    assert expand_goal("open the apps section", None) == "open the apps section"


def test_a_goal_that_already_used_the_apps_words_still_works() -> None:
    """Expansion adds; it never replaces. A goal that was already right must stay right."""

    got = decide(expand_goal("open the apps section", {"feed": ["Ideas"]}), _projection(NAV))
    assert got["call"] == "tap"
    assert got["n"] == "n3"


# --------------------------------------------------------------------------- how it expands


def test_only_terms_actually_present_in_the_goal_are_expanded() -> None:
    """A vocabulary is per app, not per goal, so most of it is irrelevant to any one goal."""

    vocab = {"feed": ["Ideas"], "profile": ["You"], "billing": ["Plans"]}
    assert expand_goal("open the feed", vocab) == "open the feed Ideas"


def test_a_word_the_goal_already_contains_is_not_added_twice() -> None:
    """`"open the feed Ideas Ideas"` scores the same but is noise in every log and report."""

    assert expand_goal("open the Ideas feed", {"feed": ["Ideas"]}) == "open the Ideas feed"


def test_a_term_may_teach_more_than_one_word() -> None:
    """One concept can surface under several labels across an app's screens."""

    got = expand_goal("open the feed", {"feed": ["Ideas", "Discover"]})
    assert "Ideas" in got and "Discover" in got


def test_matching_is_case_insensitive_and_the_apps_spelling_is_kept() -> None:
    """A scenario writes prose; the app's label is the thing that must match on screen."""

    assert expand_goal("Open The FEED", {"feed": ["Ideas"]}).endswith("Ideas")


def test_a_multi_word_term_is_matched_as_a_phrase() -> None:
    """"lock screen" is one concept, and expanding on the bare word "screen" would fire on almost
    every goal — that word appears in more than half of real harvested screens."""

    vocab = {"lock screen": ["Security"]}
    assert expand_goal("open the lock screen settings", vocab).endswith("Security")
    assert expand_goal("open the display screen", vocab) == "open the display screen"


def test_a_stopword_key_is_refused_rather_than_silently_matching_everything() -> None:
    """The one entry that could ruin a whole app's vocabulary.

    `{"the": ["Ideas"]}` would append "Ideas" to nearly every goal, so every screen would look like it
    contained the target. Refused at write time, where a human can see it, rather than degrading
    every run afterwards.
    """

    with pytest.raises(ValueError, match="stopword|too common"):
        expand_goal("open the feed", {"the": ["Ideas"]})
