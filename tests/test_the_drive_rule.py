"""The baseline driver's contract, and the one gap it is still honest about.

This is the only component in the experiment that has ever driven a real device successfully —
17 of 19 reachable destinations against 0 of 19 for every trained checkpoint. So its behaviour is
worth pinning precisely, including the parts that are weak.

Two of the three gaps it used to declare are closed here, and neither needed a model:

* a goal naming a host capability is refused up front (`HOST_TERMS`), and the tests below pin the
  admission rule for that list as tightly as the list itself — a word common on real screens is a
  false refusal, which is the expensive direction;
* a goal and a label that differ only in *spelling* now meet (`variants`). "Internet AndroidWifi"
  is the Wi-Fi row; a capital is a word boundary.

What is left is a genuine synonym, and `test_the_remaining_weakness_is_a_synonym_and_nothing_else`
is where that is written down.

The sweep at the end runs it over every harvested real screen when they are present, because the
property that matters most is structural: it chooses from the list it was shown, so it cannot name
an element that does not exist. Trained checkpoints authored strings instead and grounded at 0%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser.drive_projection import project  # noqa: E402
from android_ui_analyser.drive_rule import (  # noqa: E402
    HOST_PAIRS,
    HOST_TERMS,
    MAX_SCROLLS,
    STOPWORDS,
    content_words,
    decide,
    score,
    stalled,
    variants,
    words,
)

SCREENS = Path(__file__).resolve().parents[1] / "runs/functiongemma/screens"


def _projection(nodes: list[dict[str, object]], more: bool = False) -> dict[str, object]:
    return {"nodes": nodes, "more": more, "keys": [f"key-{n['n']}" for n in nodes]}


# --------------------------------------------------------------------------- tokenising


def test_words_needs_no_regex_because_this_becomes_java() -> None:
    assert words("Network & internet Mobile, Wi-Fi, hotspot") == [
        "network",
        "internet",
        "mobile",
        "wi",
        "fi",
        "hotspot",
    ]
    assert words(None) == []
    assert words("") == []


def test_content_words_drops_the_verbs_every_goal_shares() -> None:
    # "open", "the", "settings" appear in half of all goals and name nothing.
    assert content_words("open the Display settings") == ["display"]
    assert content_words("go to Storage") == ["storage"]


# --------------------------------------------------------------------------- the chrome trap


def test_normalized_system_and_ime_windows_never_reach_the_driver() -> None:
    projection = project(
        [
            {"id": "sys", "text": "Power off", "clickable": True, "window": "system"},
            {"id": "ime", "text": "Search", "clickable": True, "window": "ime"},
            {"id": "app", "text": "Continue", "clickable": True, "window": "app"},
        ]
    )

    assert [node["text"] for node in projection["nodes"]] == ["Continue"]


def test_chrome_never_wins_which_is_the_whole_9_to_17_difference() -> None:
    """A flat overlap score taps "Search settings" for any goal mentioning settings: 9/19.

    Scoring chrome at zero plus the first-token bonus is what took the same algorithm to 17/19.
    """

    nodes = [
        {"n": "n1", "text": "Search settings", "tap": True},
        {"n": "n2", "text": "Display & touch Dark theme, font size, touch", "tap": True},
        {"n": "n3", "text": "Network & internet Mobile, Wi-Fi, hotspot", "tap": True},
    ]
    got = decide("open the Display settings", _projection(nodes))
    assert got["call"] == "tap"
    assert got["n"] == "n2"

    assert score(content_words("open the Display settings"), nodes[0]) == 0.0


def test_the_head_word_outranks_a_summary_hit() -> None:
    """Android rows read "Title  Summary", so a match on the head is stronger evidence."""

    head = {"n": "n1", "text": "Storage 84 GB used", "tap": True}
    summary = {"n": "n2", "text": "Apps Assistant, default apps, storage", "tap": True}
    terms = content_words("open Storage")
    assert score(terms, head) > score(terms, summary)

    got = decide("open Storage", _projection([summary, head]))
    assert got["n"] == "n1"


def test_a_summary_only_match_still_counts() -> None:
    """ "turn on the hotspot" has to reach a row whose head word is "Network"."""

    nodes = [{"n": "n1", "text": "Network & internet Mobile, Wi-Fi, hotspot", "tap": True}]
    got = decide("turn on the hotspot", _projection(nodes))
    assert got["call"] == "tap"
    assert got["n"] == "n1"


# --------------------------------------------------------------------------- grounding


def test_it_can_only_ever_return_a_node_it_was_shown() -> None:
    """The structural property. Every trained checkpoint failed exactly here, at 0/496."""

    nodes = [{"n": "n1", "text": "Display", "tap": True}, {"n": "n2", "text": "Sound", "tap": True}]
    projection = _projection(nodes)
    got = decide("open Display", projection)
    assert got["n"] in {n["n"] for n in nodes}
    assert got["stable_key"] == "key-n1"


def test_a_non_tappable_node_is_never_chosen() -> None:
    nodes = [
        {"n": "n1", "text": "Display"},  # a heading, not actionable
        {"n": "n2", "text": "Sound", "tap": True},
    ]
    got = decide("open Display", _projection(nodes))
    assert got["call"] != "tap" or got["n"] == "n2"


# --------------------------------------------------------------------------- scroll vs refuse


def test_a_weak_match_scrolls_rather_than_guessing() -> None:
    nodes = [{"n": "n1", "text": "Sound", "tap": True}, {"n": "n2", "scroll": True}]
    got = decide("open Bluetooth", _projection(nodes, more=True))
    assert got["call"] == "scroll"


def test_scrolling_gives_up_eventually() -> None:
    nodes = [{"n": "n1", "text": "Sound", "tap": True}, {"n": "n2", "scroll": True}]
    got = decide("open Bluetooth", _projection(nodes, more=True), scrolls_used=MAX_SCROLLS)
    assert got["call"] == "handoff"


def test_nothing_to_scroll_means_refuse_now() -> None:
    nodes = [{"n": "n1", "text": "Sound", "tap": True}]
    got = decide("open Bluetooth", _projection(nodes, more=False))
    assert got["call"] == "handoff"
    assert got["reason"] == "target_absent"


def test_the_remaining_weakness_is_a_synonym_and_nothing_else() -> None:
    """What is left after spelling is handled, stated so nobody mistakes the gap for a bug.

    This test used to pin "Internet AndroidWifi" as permanently unreachable. It was not: that was
    the *tokeniser* failing to read a capital as a word boundary, and
    `test_a_capital_is_a_word_boundary_so_the_wifi_row_is_reachable` now covers it.

    The real remaining gap shares no string with its target at all — only a concept. "Display" is
    where you make text bigger and nothing about either phrase says so. No tokeniser reaches that,
    and AUA has nowhere to keep the mapping: `memory.KnowledgeItem` is free text and the goal matcher
    never reads it (`tests/test_a_goal_is_a_sentence_not_a_keyword.py` pins the same gap host-side).
    """

    nodes = [{"n": "n1", "text": "Display & touch Dark theme, font size", "tap": True}]
    got = decide("make the text bigger", _projection(nodes, more=False))
    assert got["call"] == "handoff"
    assert got["reason"] == "target_absent"


# --------------------------------------------------------------------------- real screens


@pytest.mark.parametrize("name", ["aosp", "apps", "locale"])
def test_it_grounds_on_every_harvested_real_screen(name: str) -> None:
    """Sweep the harvest: whatever it decides, a tap must name an element that exists."""

    path = SCREENS / f"{name}.jsonl"
    if not path.is_file():
        pytest.skip(f"{name}.jsonl not harvested in this checkout")

    checked = taps = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 400:
                break
            row = json.loads(line)
            elements = row.get("elements") or []
            # `locale.jsonl` names these fields differently; normalise or the projection sees none.
            normalised = [
                {
                    **element,
                    "rid": element.get("rid", element.get("resource_id")),
                    "desc": element.get("desc", element.get("content_desc")),
                }
                for element in elements
            ]
            projection = project(normalised)
            if not projection["nodes"]:
                continue
            checked += 1
            for goal in ("open the Display settings", "turn on wifi", "find storage"):
                got = decide(goal, projection)
                assert got["call"] in {"tap", "scroll", "handoff"}
                if got["call"] == "tap":
                    taps += 1
                    assert got["n"] in {n["n"] for n in projection["nodes"]}
                    assert got["stable_key"] in projection["keys"]
    assert checked, "no projectable screens found"


# --------------------------------------------------------------------------- spelling, not meaning


def test_a_capital_is_a_word_boundary_so_the_wifi_row_is_reachable() -> None:
    """The live miss, closed. "Internet AndroidWifi" IS the Wi-Fi row and a goal saying "wifi" now
    reaches it — not by learning a synonym, but by reading the capital as the separator it is.

    This is the case the old `test_the_known_weakness_is_reproducible_not_hidden` pinned as
    permanently out of reach. It was out of reach of the *tokeniser*, not of the rule.
    """

    # No join here: both parts of a join must be short, and "internet"/"androidwifi" are not. The
    # split alone is what reaches this label.
    assert variants("Internet AndroidWifi") == ["internet", "androidwifi", "android", "wifi"]
    nodes = [{"n": "n1", "text": "Internet AndroidWifi", "tap": True}]
    got = decide("turn on wifi", _projection(nodes, more=False))
    assert got["call"] == "tap"
    assert got["n"] == "n1"


def test_a_hyphen_is_a_word_boundary_too_so_the_join_is_needed_in_both_directions() -> None:
    """The other real harvested spelling. "Reset Bluetooth & Wi-Fi" splits to ...wi, fi."""

    assert "wifi" in variants("Reset Bluetooth & Wi-Fi")
    nodes = [{"n": "n1", "text": "Network & internet Mobile, Wi-Fi, hotspot", "tap": True}]
    got = decide("turn on wifi", _projection(nodes, more=False))
    assert got["call"] == "tap"


def test_an_already_correct_spelling_survives_being_split() -> None:
    """ "WiFi" matched before this change and must still match: the split gives wi + fi, and only
    keeping the unsplit word as well saves it."""

    assert "wifi" in variants("WiFi")
    assert decide("turn on wifi", _projection([{"n": "n1", "text": "WiFi", "tap": True}]))["n"] == "n1"


def test_it_is_spelling_and_not_substring_matching() -> None:
    """Substring containment would reach the same two labels and 664 more pairs in the harvest.

    "lock" must not reach "Clock", and "phone" must not reach "Microphone" — those are different
    words that happen to nest, and matching them is how a rule starts inventing destinations.
    """

    assert score(["lock"], {"text": "Clock", "tap": True}) == 0.0
    assert score(["phone"], {"text": "Microphone", "tap": True}) == 0.0


def test_a_join_deep_in_the_summary_does_not_collect_the_head_bonus() -> None:
    """The head bonus names the row; membership may look anywhere. Those must stay separate."""

    row = {"n": "n1", "text": "Connections Mobile data, Wi-Fi", "tap": True}
    assert score(["wifi"], row) == 1.0  # matched, but as a summary hit


# --------------------------------------------------------------------------- refusing the host


def test_a_host_capability_goal_is_refused_before_any_node_is_scored() -> None:
    """ "nothing scored well" used to read the same as "keep scrolling": 4 of 7 host goals stuck.

    The refusal has to come first. Scoring first only decides how much budget is spent proving that
    a goal no screen can satisfy is not on this screen.
    """

    nodes = [{"n": "n1", "text": "Screenshots", "tap": True}, {"n": "n2", "scroll": True}]
    got = decide("take a screenshot of this screen", _projection(nodes, more=True))
    assert got["call"] == "handoff"
    assert got["reason"] == "needs_host"

    for goal in ("read the logcat output", "query the app's database", "start the recording proxy"):
        assert decide(goal, _projection(nodes, more=True))["reason"] == "needs_host"


def test_the_refusal_list_holds_only_words_real_screens_do_not_use() -> None:
    """The admission rule, pinned. A word that names a host capability but is also ordinary screen
    vocabulary cannot be in here — that is a false refusal on a reachable destination.

    "system" is in 83% of the 638 harvested screens, "screen" in 53%, "network" in 3.4%. So
    "change the system time" and "turn the network off" stay unrefusable, on purpose.
    """

    for common in ("system", "screen", "network", "clock", "time", "device", "app", "new", "off"):
        assert common not in HOST_TERMS

    assert decide("go to network settings", _projection([{"n": "n1", "text": "Network & internet", "tap": True}]))["call"] == "tap"
    assert decide("open the system settings", _projection([{"n": "n1", "text": "System", "tap": True}]))["call"] == "tap"


def test_a_host_word_inside_a_real_destination_still_taps() -> None:
    """The refusal reads the goal, never the screen. A goal that names a destination is a goal even
    when a host row is on screen next to it."""

    nodes = [
        {"n": "n1", "text": "Storage 84 GB used", "tap": True},
        {"n": "n2", "text": "Take screenshot", "tap": True},
    ]
    assert decide("open Storage", _projection(nodes))["n"] == "n1"


# --------------------------------------------------------------------------- refusing a consent gate


def test_a_consent_dialog_renames_the_refusal_it_does_not_cause_one() -> None:
    """A dialog being up must never be grounds to stop — pressing "Don't allow" is legitimate, which
    is why `v12_corpus`'s `decline` family exists. So this only names a refusal already happening.
    """

    dialog = [
        {"n": "n1", "text": "Allow this app to send you notifications?"},
        {"n": "n2", "text": "Allow", "tap": True},
        {"n": "n3", "text": "Don't allow", "tap": True},
    ]
    got = decide("open Display", _projection(dialog, more=False))
    assert got["call"] == "handoff"
    assert got["reason"] == "needs_auth"


def test_reaching_the_decline_control_by_overlap_still_taps_it() -> None:
    """21 of the 733 dialog rows in data-v12/test.jsonl are reachable by word overlap. Those must
    keep tapping: the rename lives in the refusal branch and nowhere else."""

    dialog = [
        {"n": "n1", "text": "Allow", "tap": True},
        {"n": "n2", "text": "Deny", "tap": True},
    ]
    got = decide("deny it", _projection(dialog, more=False))
    assert got["call"] == "tap"
    assert got["n"] == "n2"


def test_only_a_consent_control_counts_and_not_every_dismissal() -> None:
    """ "No thanks" and "Not now" are the buttons on rating prompts and update nags, which gate
    nothing. `v12_corpus._is_auth` admits them; that is over-broad and this does not copy it."""

    for label in ("No thanks", "Not now", "Cancel"):
        nodes = [{"n": "n1", "text": label, "tap": True}]
        got = decide("open Display", _projection(nodes, more=False))
        assert got["reason"] == "target_absent", label


# --------------------------------------------------------------------------- tried, and stalled


def test_progress_is_read_off_the_node_and_never_joined() -> None:
    """Two fields on the node, exactly as `v12_progress` writes them, and nothing else.

    A node with no `tried` has not been touched — the field is omitted at zero rather than sent as
    zero, because a runtime that omitted it against a corpus that sent it is the train/serve gap
    that cost V10 its whole run.
    """

    assert not stalled({"n": "n1", "text": "Wi-Fi"})
    assert not stalled({"n": "n1", "text": "Wi-Fi", "tried": 0})
    assert not stalled({"n": "n1", "text": "Wi-Fi", "tried": 2, "last": "changed"})
    assert stalled({"n": "n1", "text": "Wi-Fi", "tried": 1, "last": "unchanged"})
    assert stalled({"n": "n1", "text": "Wi-Fi", "tried": 3, "last": "blocked"})


def test_the_goals_own_target_stalling_is_a_refusal() -> None:
    """The repeat-tap loop, closed. The screen has already settled, so a second identical tap gets
    an identical non-result — on-device that spent the whole budget re-pressing one row."""

    nodes = [{"n": "n1", "text": "Display", "tap": True, "tried": 2, "last": "unchanged"}]
    got = decide("open Display", _projection(nodes, more=False))
    assert got["call"] == "handoff"
    assert got["reason"] == "no_progress"
    assert got["n"] == "n1"  # says which node stalled, so the host can act on it


def test_a_blocked_target_refuses_the_same_way() -> None:
    nodes = [{"n": "n1", "text": "Display", "tap": True, "tried": 1, "last": "blocked"}]
    assert decide("open Display", _projection(nodes))["reason"] == "no_progress"


def test_a_stalled_node_that_is_not_the_target_changes_nothing() -> None:
    """The `tap_despite_stalls` distinction, which is the whole reason this reads only the winner.

    2,035 rows of data-v12/test.jsonl have this exact shape — a stalled node elsewhere plus an
    untried target — and they answer `tap`. Refusing because "something on this screen stalled"
    would lose all of them, and is how a rule learns to give up on a screen it never tried.
    """

    nodes = [
        {"n": "n1", "text": "Sound", "tap": True, "tried": 3, "last": "unchanged"},
        {"n": "n2", "text": "Display", "tap": True},
    ]
    got = decide("open Display", _projection(nodes))
    assert got["call"] == "tap"
    assert got["n"] == "n2"


def test_a_stalled_node_still_competes_and_is_not_scored_down() -> None:
    """Stalled nodes are scored normally and stay eligible to win. Zeroing them out instead would
    scroll away from the right answer and then report the wrong reason for giving up."""

    nodes = [{"n": "n1", "text": "Display", "tap": True, "tried": 1, "last": "unchanged"}]
    assert score(content_words("open Display"), nodes[0]) == 1.5
    # It wins, and because it wins the refusal can name it. With its score suppressed this would
    # have been an unexplained `target_absent`.
    got = decide("open Display", _projection(nodes, more=True))
    assert got["reason"] == "no_progress"


def test_an_untried_target_is_reached_even_after_a_stall_elsewhere_and_scrolls_spent() -> None:
    """Progress must not become a second, quieter way of exhausting the budget."""

    nodes = [
        {"n": "n1", "text": "Sound", "tap": True, "tried": 4, "last": "blocked"},
        {"n": "n2", "text": "Storage 84 GB used", "tap": True},
    ]
    got = decide("open Storage", _projection(nodes, more=True), scrolls_used=MAX_SCROLLS)
    assert got["call"] == "tap"
    assert got["n"] == "n2"


# --------------------------------------------------------------------------- it stops when it is done


def test_it_stops_once_it_has_pressed_the_control_the_goal_named() -> None:
    """The budget burn, measured live: "Apps" tapped three times, each one reporting `changed`.

        step 0   tap n19 'Apps'  -> CHANGED
        step 1   tap n23 'Apps'  -> CHANGED
        step 2   tap n20 'Apps'  -> CHANGED

    `no_progress` did not catch it because nothing stalled — every tap really did change the screen
    (a tab re-selects, a list scrolls back to top). The driver simply had no notion of having
    finished, so it re-pressed the best match until the budget ran out.

    This needs no vocabulary and no arrival judgement, which is why it is the fix worth having:
    "the control the goal names was pressed and the screen moved" is a fact about what this driver
    did, not a claim about what the screen now shows. It says the *action* is complete and nothing
    more — the caller still has to verify the outcome, and no wording here lets it skip that.
    """

    nodes = [
        {"n": "n1", "text": "Chat", "tap": True},
        {"n": "n2", "text": "Ideas", "tap": True},
        {"n": "n3", "text": "Widgets", "tap": True, "tried": 1, "last": "changed"},
    ]
    got = decide("navigate to the widgets section", _projection(nodes))
    assert got["call"] == "done", f"re-pressed a control it had already pressed: {got}"


def test_a_stall_on_the_goals_control_still_reports_no_progress() -> None:
    """The two outcomes on the same control mean opposite things, and both must survive.

    `changed` says the goal was carried out. `unchanged` says it was attempted and went nowhere. A
    single "already tried" rule would collapse them and lose the distinction the corpus was built to
    teach.
    """

    nodes = [{"n": "n1", "text": "Widgets", "tap": True, "tried": 2, "last": "unchanged"}]
    got = decide("go to widgets", _projection(nodes))
    assert got["call"] == "handoff"
    assert got["reason"] == "no_progress"


def test_a_control_someone_else_pressed_does_not_end_the_run() -> None:
    """Only the goal's own control counts. A stray `changed` elsewhere means nothing.

    Same shape as the `no_progress` rule: consult the winner, never the screen at large. Otherwise
    any run that had already acted once would declare itself finished.
    """

    nodes = [
        {"n": "n1", "text": "Something Else", "tap": True, "tried": 1, "last": "changed"},
        {"n": "n2", "text": "Widgets", "tap": True},
    ]
    got = decide("go to widgets", _projection(nodes))
    assert got["call"] == "tap"
    assert got["n"] == "n2"


def test_it_does_not_claim_done_before_it_has_done_anything() -> None:
    """An untouched control is work to do, not work completed."""

    nodes = [{"n": "n1", "text": "Widgets", "tap": True}]
    got = decide("go to widgets", _projection(nodes))
    assert got["call"] == "tap"


# --------------------------------------------------------------------------- a goal that only looks


def test_a_goal_that_only_asks_to_look_does_not_tap() -> None:
    """Measured hole, and the largest one left: `done` was right 6.9% of the time.

    Stratified over `data-v12/test.jsonl`, the rule scored `tap` 85%, `handoff` 93% and `scroll`
    100% — and `done` 7%. Every one of those 419 rows is a goal that asks a *question* about the
    screen rather than giving it an instruction:

        confirm taskslast is showing        is layers on screen
        check clock468 is here             can you see start voice

    The rule read them as navigation, found the best word match, and pressed it. For a QA suite
    that is the worst possible answer: the assertion was "is this visible", and the driver changed
    the screen before anything could be asserted about it.
    """

    nodes = [
        {"n": "n1", "text": "Widgets", "tap": True},
        {"n": "n2", "text": "Doodads"},
    ]
    got = decide("confirm doodads is showing", _projection(nodes))
    assert got["call"] == "done", f"acted on a screen it was only asked to read: {got}"
    assert got["n"] == "n2"


def test_looking_beats_tapping_even_when_the_subject_is_tappable() -> None:
    """Most things worth asserting about are also pressable. Pressable is not permission."""

    nodes = [{"n": "n1", "text": "Doodads", "tap": True}]
    got = decide("is doodads on screen", _projection(nodes))
    assert got["call"] == "done"
    assert got["n"] == "n1"


def test_a_look_goal_whose_subject_is_absent_scrolls_first() -> None:
    """Not found is not the same as not there — 47.7% of the look-shaped rows are `target_absent`,
    and the corpus reaches them through the same scroll-then-refuse path every other goal uses."""

    nodes = [{"n": "n1", "text": "Widgets", "tap": True}]
    got = decide("is doodads on screen", _projection(nodes, more=True))
    assert got["call"] == "scroll"


def test_a_look_goal_with_nothing_left_to_reveal_says_the_target_is_absent() -> None:
    """The honest QA answer to "is it visible" when it is not: absent, never `done`. A false `done`
    here is a false pass, which is the one error a test suite must not make."""

    nodes = [{"n": "n1", "text": "Widgets", "tap": True}]
    got = decide("is doodads on screen", _projection(nodes))
    assert got["call"] == "handoff"
    assert got["reason"] == "target_absent"


def test_an_instruction_that_merely_starts_the_same_way_still_acts() -> None:
    """`can you see X` asks. `can you open X` instructs. In the corpus both start "can you", 515
    look-goals against 2,645 action goals, so the opening words cannot be the test — the visibility
    phrase is."""

    nodes = [{"n": "n1", "text": "Doodads", "tap": True}]
    got = decide("can you open doodads", _projection(nodes))
    assert got["call"] == "tap"
    assert got["n"] == "n1"


def test_the_question_words_do_not_pollute_the_scoring() -> None:
    """"check ... is here" put `check` and `here` into the goal terms, so a row whose label happens
    to start with one of them outscored the row actually being asked about."""

    nodes = [
        {"n": "n1", "text": "Check for updates", "tap": True},
        {"n": "n2", "text": "Doodads"},
    ]
    got = decide("check doodads is here", _projection(nodes))
    assert got["call"] == "done"
    assert got["n"] == "n2", f"the frame word won the scoring: {got}"


def test_a_host_goal_is_still_refused_before_the_screen_is_read() -> None:
    """Ordering, pinned. A host capability is unreachable by looking as well as by tapping, and no
    look-shaped phrasing may turn that refusal into a `done`."""

    nodes = [{"n": "n1", "text": "Screenshot", "tap": True}]
    got = decide("is the screenshot on screen", _projection(nodes))
    assert got["call"] == "handoff"
    assert got["reason"] == "needs_host"


def test_a_stalled_control_does_not_make_a_look_goal_report_no_progress() -> None:
    """Nothing was being attempted, so nothing can have stalled. A look goal never taps, so the
    progress fields on its subject say only that some earlier step touched it."""

    nodes = [{"n": "n1", "text": "Doodads", "tap": True, "tried": 2, "last": "unchanged"}]
    got = decide("is doodads visible", _projection(nodes))
    assert got["call"] == "done"


def test_a_visibility_phrase_must_start_at_a_word_boundary() -> None:
    """The whole of this fix's cost, measured: 6 regressions on 5,741 held-out rows, 5 of them this.

    "reach the solution screen" ends with the letters of "on screen" and is an instruction. Without
    a boundary check the goal was read as a question, its subject came out as "reach the soluti",
    and a reachable destination was refused as absent — the expensive direction of error, because no
    later step recovers it.
    """

    nodes = [{"n": "n1", "text": "Solution", "tap": True}]
    got = decide("reach the solution screen", _projection(nodes))
    assert got["call"] == "tap", f"matched a phrase mid-word: {got}"
    assert got["n"] == "n1"


# --------------------------------------------------------------------------- refusing by a pair


def test_a_host_capability_named_by_two_ordinary_words_is_still_refused() -> None:
    """`needs_host` was the weakest class left at 68.1%, and every one of the 238 misses named a
    capability whose words are individually too common to refuse.

    "copy a photo into the gallery" is `aua media add`. No word in it can join `HOST_TERMS`:
    measured over the 638 harvested screens, "photo" is on 4.4% and "gallery" on 4.5%, and a false
    refusal on a reachable destination is the error no later step recovers. The *pair* is on 0.0%.

    That is the whole idea — the admission rule was only ever applied to single words, and a pair of
    ordinary words can be rare when neither word is.
    """

    nodes = [{"n": "n1", "text": "Photos", "tap": True}]
    got = decide("copy a photo into the gallery", _projection(nodes))
    assert got["call"] == "handoff", f"tried to reach a host capability on screen: {got}"
    assert got["reason"] == "needs_host"


def test_each_half_of_a_pair_on_its_own_still_navigates() -> None:
    """The pair is the refusal, never either word. Both halves have to stay usable in a goal that
    names a real destination, or this trades a vague handoff for a broken run."""

    for goal, label in (
        ("go to the photo settings", "Photo & video"),
        ("open the gallery", "Gallery"),
        ("go to connected devices", "Connected devices"),
        ("open verbose logs", "Verbose logs"),
        ("record a memo", "Record a memo"),
        ("list my alarms", "Alarms"),
    ):
        nodes = [{"n": "n1", "text": label, "tap": True}]
        got = decide(goal, _projection(nodes))
        assert got["call"] == "tap", f"{goal!r} was refused: {got}"


def test_the_pair_is_what_makes_the_logs_word_usable_at_all() -> None:
    """`HOST_TERMS` records dropping ``logs`` after measurement: it rescued "check the app logs" but
    refused "open verbose logs" on a real developer screen. Requiring a partner word is what settles
    that — the capability is refused and the destination is still reachable, which neither the word
    on its own nor its absence could manage."""

    logs_row = [{"n": "n1", "text": "Verbose logs", "tap": True}]
    assert decide("open verbose logs", _projection(logs_row))["call"] == "tap"

    refused = decide("check the app logs for errors", _projection(logs_row))
    assert refused["call"] == "handoff"
    assert refused["reason"] == "needs_host"


def test_a_pair_is_admitted_on_the_same_evidence_as_a_single_word() -> None:
    """The bar does not get lower because there are two words. Every pair must be measured on the
    harvest and clear the same 1% ceiling, and the ones that cannot are named in `HOST_PAIRS`."""

    screens = sorted(SCREENS.glob("*.jsonl")) if SCREENS.exists() else []
    if not screens:
        pytest.skip("harvested screens are not present")

    bags = []
    for path in screens:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            bag: set[str] = set()
            for element in json.loads(line).get("elements") or []:
                for field in ("text", "desc", "rid"):
                    value = element.get(field)
                    if value:
                        bag |= set(variants(value)) | set(words(value))
            bags.append(bag)

    ceiling = len(bags) // 100
    for pair in HOST_PAIRS:
        seen = sum(1 for bag in bags if all(term in bag for term in pair))
        assert seen <= ceiling, (
            f"{sorted(pair)} is on {seen}/{len(bags)} real screens, above the {ceiling} ceiling"
        )


def test_the_clock_stays_unrefusable_and_that_is_recorded_not_forgotten() -> None:
    """Two of the six failing shapes are not fixable this way and must not be forced.

    "change the system time" and "set the device clock to midnight" are `aua clock set`. Their pairs
    are ordinary Settings vocabulary — "system"+"time" is on 6.3% of harvested screens and
    "set"+"clock" on 48.9% — because Date & time is a real destination one screen away. Refusing
    them would break navigation to it, so they stay `target_absent`, which still hands the run back.
    """

    nodes = [{"n": "n1", "text": "Date & time", "tap": True}]
    assert decide("go to date and time", _projection(nodes))["call"] == "tap"
    for unrefusable in (frozenset({"system", "time"}), frozenset({"set", "clock"})):
        assert unrefusable not in HOST_PAIRS


def test_no_pair_is_made_of_a_word_the_goal_never_keeps() -> None:
    """A pair holding a stopword is not weak, it is unreachable — `content_words` strips the word
    before any pair is tested, so the entry can never fire and reads as coverage that is not there.

    Two such pairs were written and removed: "record"+"screen" and "recording"+"screen" both cleared
    the harvest measurement, and "screen" is a stopword.
    """

    for pair in HOST_PAIRS:
        assert not (pair & STOPWORDS), f"{sorted(pair)} can never fire: {sorted(pair & STOPWORDS)}"
