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
    HOST_TERMS,
    MAX_SCROLLS,
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
