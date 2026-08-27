"""The baseline driver — word overlap with a first-token bonus. It beat every trained checkpoint.

On 26 goals against a live Android 16 device, hierarchy-only perception, 8-step budget:

| | ckpt576 | ckpt9216 | this, naive | **this, ranked** |
|---|---:|---:|---:|---:|
| reachable destinations reached | 0/19 | 0/19 | 9/19 | **17/19** |
| goals correct overall | 7/26 | 0/26 | 14/26 | **20/26** |
| selectors that named a real node | never authored one | 0/71 | 100% | **100%** |

Two things about that table are worth more than the win.

**The gap between the two columns of this same file is one scoring rule.** A flat word-overlap
score taps "Search settings" for any goal containing the word "settings" and reaches 9 of 19. Adding
a first-token bonus and a score floor reaches 17. So most of this task is decided by tie-breaking,
not by understanding — which is the strongest argument available for *not* pointing a language model
at the navigation problem, and pointing it instead at the two things below that no amount of
tie-breaking solves.

**Grounding is 100% by construction, not by skill.** This selects an element out of the list it was
shown, so it cannot name a node that does not exist. Every trained checkpoint authored selector
strings instead and grounded at 0% — 0 of 496 answers across all 16 checkpoints. Choosing from a
list is a different problem from spelling a string, and only one of them is solvable at 350M.

Both gaps that table left open turned out not to need a model, and neither needed semantics.

**"It cannot say impossible"** was 4 of 7 host-capability goals ending stuck. Answered by
:data:`HOST_TERMS`: a goal naming a capability that lives on the host is refused before a single
node is scored. See that constant for why a word is allowed in and why the list stops where it does.

**"It cannot bridge a semantic gap"** was two misses where "the words simply do not meet". They do
meet — the tokeniser was throwing away the joint. ``"Internet AndroidWifi"`` reduces to
``["internet", "androidwifi"]``, so a goal saying ``wifi`` scored zero against the row that *is* the
Wi-Fi row; ``"Reset Bluetooth & Wi-Fi"`` reduces to ``[..., "wi", "fi"]``, so the same goal scored
zero there too. Both are capitalisation and punctuation, not meaning. :func:`variants` reads the
joint in each direction and both labels now match at full strength. Measured on
``runs/functiongemma/data-v12/test.jsonl``: 630 rows the rule provably could not answer became
answerable, with zero regressions — because the corpus generates exactly this phrasing on purpose
(see ``v12_goals._compound``) while the scorer had no way to resolve it.

A third gap, unchanged and named here so nobody mistakes it for solved: a goal and a label that
share no *string*, only a *concept* — "make the text bigger" for "Display", "upsell" for
"Go Premium". That is a synonym, it is not derivable from either string, and AUA has nowhere to keep
one (``memory.KnowledgeItem`` is free text and the goal matcher never reads it;
``tests/test_a_goal_is_a_sentence_not_a_keyword.py`` pins the gap). It stays open.

Kept deliberately portable — plain string and list operations, no regex, no dependencies — because
this has to be reimplemented in Java inside the helper APK, where there is no Python and no model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Words that carry no target information. Kept short on purpose: a long stop list starts encoding
#: assumptions about phrasing, and the goal vocabulary is not ours to control.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "to",
        "of",
        "on",
        "in",
        "at",
        "for",
        "with",
        "from",
        "me",
        "my",
        "i",
        "you",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "be",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "please",
        "just",
        "then",
        "now",
        "up",
        "open",
        "go",
        "goto",
        "show",
        "find",
        "get",
        "reach",
        "take",
        "bring",
        "land",
        "navigate",
        "screen",
        "page",
        "section",
        "settings",
        "setting",
        "view",
        "tab",
        "prove",
        "confirm",
        "need",
        "want",
        "let",
        "make",
    }
)

#: Chrome that appears on nearly every screen and is almost never the destination. Scoring it
#: normally is exactly how the naive variant lost 8 of its 19: "Search settings" shares a word with
#: half of all goals.
CHROME = frozenset(
    {
        "search settings",
        "search",
        "navigate up",
        "back",
        "more options",
        "options",
        "home",
        "close",
        "cancel",
        "dismiss",
        "open features menu",
        "overflow",
    }
)

#: Words that name something only the host can do. A goal containing one is refused before any node
#: is scored, because no amount of tapping reaches `aua screenshot`.
#:
#: **Both halves of the admission rule matter, and the second one is why this list is short.**
#:
#: 1. The word must name a capability the ``aua`` CLI actually has — ``screenshot``, ``logcat``,
#:    ``db query``, ``install``, ``proxy start``, ``network offline``, ``orient``, ``emulator start``.
#: 2. The word must almost never appear in an on-screen label, measured over the 638 real screens in
#:    ``runs/functiongemma/screens``. Every word here appears in **at most 1% of them**; most appear
#:    in none.
#:
#: The second test is the one doing the work, and it has to be made against screens somebody else's
#: apps produced. Justifying a refusal list against a generated list of host goals proves only that
#: the generator and the list were written by the same hand. Against the harvest, the words that a
#: *complete* list would need are exactly the ones that cannot be allowed: ``system`` appears in 83%
#: of real screens, ``screen`` in 53%, ``network`` in 3.4%, ``clock`` and ``time`` in 4-7%. So
#: "change the system time" and "turn the network off" are **not** refusable from goal text, and this
#: list does not try. That is not a hole to be patched later — AUA's host capabilities are named
#: after the same device concepts Settings exposes as destinations, so the collision is structural.
#:
#: **The trade is deliberately lopsided towards precision, because the two errors are not
#: comparable.** A missed refusal costs a handoff with a vaguer reason: the goal burns some of its
#: budget, nothing on screen matches, and the run hands off as ``target_absent`` instead of
#: ``needs_host``. The host gets control either way. A *false* refusal stops a run that would have
#: succeeded, on a destination that was reachable, and no later step recovers it. So a word that
#: rescues many host goals is still rejected if it fires once on a real navigation goal.
#:
#: Two words were dropped for exactly that reason after measurement, and are recorded here so the
#: reasoning is auditable rather than repeated: ``export`` (redundant with ``capture``, and the one
#: harvested screen carrying it reads "Import export", a plausible destination) and ``logs``
#: (rescues "check the app logs", but refuses "open verbose logs" on a real developer screen).
HOST_TERMS = frozenset(
    {
        "adb",
        "apk",
        "capture",
        "database",
        "dump",
        "emulator",
        "host",
        "install",
        "landscape",
        "logcat",
        "offline",
        "proxy",
        "query",
        "recording",
        "restore",
        "rotate",
        "screenshot",
        "sqlite",
        "traffic",
    }
)

#: The decline control of a consent dialog. Its presence is the only screen evidence available that
#: going further would mean granting something, and it is used for **nothing but naming a refusal
#: that was already happening** — see :func:`decide`.
#:
#: Deliberately three labels and not six. "Cancel" is already chrome, and "No thanks" / "Not now"
#: are the buttons on rating prompts, update nags and onboarding cards, none of which gate access.
#: Adding "cancel" alone took the precision of a dialog-present test from 0.45 to 0.37 on
#: ``data-v12/test.jsonl``. ``v12_corpus._is_auth`` does admit "no thanks" and "not now"; that is
#: over-broad, and only harmless there because the harvest contains no such screen.
DECLINE_LABELS = frozenset({"deny", "don't allow", "dont allow"})

#: A node must beat this to be worth tapping. Below it, scrolling for more screen is the better
#: move than acting on a weak match — this floor plus the first-token bonus is the entire
#: difference between 9/19 and 17/19.
SCORE_FLOOR = 1.0

#: Extra credit when the goal's first meaningful word is also the node's first word. Android rows
#: read "Title  Summary", so the head word is where the destination's name lives.
FIRST_TOKEN_BONUS = 1.5

#: Partial credit for a goal word appearing anywhere else in the node's text.
WORD_MATCH = 1.0

#: Credit for a goal word matching inside the resource id, which survives locale changes where the
#: label does not. Weighted below text because ids are also full of layout noise.
RID_MATCH = 0.5

#: How many times to scroll one screen before concluding the target is not there.
MAX_SCROLLS = 3

#: ``last`` values meaning the run is not advancing. Mirrors :data:`v12_progress.STALLED`; spelled
#: out here rather than imported because this file has to port to Java, where there is no import.
STALLED = frozenset({"blocked", "unchanged"})


def words(value: Any) -> list[str]:
    """Lowercased content words, punctuation stripped. No regex — this has to port to Java."""

    if not value:
        return []
    out: list[str] = []
    token: list[str] = []
    for char in str(value).lower():
        if char.isalnum():
            token.append(char)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def content_words(value: Any) -> list[str]:
    return [w for w in words(value) if w not in STOPWORDS and len(w) > 1]


def _case_parts(token: str) -> list[str]:
    """``AndroidWifi`` as ``["android", "wifi"]``. Empty when the token has no internal capital.

    Read off the original string, before :func:`words` lowercases it, because the capital *is* the
    separator. A boundary is a lowercase letter or digit followed by an uppercase one, which is how
    Android names a compound label it has no room to space out.
    """

    out: list[str] = []
    current: list[str] = []
    for char in token:
        previous = current[-1] if current else ""
        if char.isupper() and previous and (previous.islower() or previous.isdigit()):
            out.append("".join(current))
            current = [char]
        else:
            current.append(char)
    if current:
        out.append("".join(current))
    if len(out) < 2:
        return []
    return [part.lower() for part in out if len(part) > 1]


def _runs(value: Any) -> list[str]:
    """The alphanumeric runs of *value*, original case kept. :func:`words` with the lowercasing off."""

    if not value:
        return []
    out: list[str] = []
    token: list[str] = []
    for char in str(value):
        if char.isalnum():
            token.append(char)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def variants(value: Any) -> list[str]:
    """Every form of *value*'s words that a goal might spell, :func:`words` included.

    Three forms, all mechanical, none of them a synonym:

    * the words themselves;
    * each word split at its internal capitals — ``AndroidWifi`` gives ``android``, ``wifi``;
    * each adjacent pair of short words joined — ``Wi-Fi`` gives ``wifi``.

    Both directions are needed and each covers a real harvested label the other cannot: splitting
    reaches ``"Internet AndroidWifi"``, joining reaches ``"Reset Bluetooth & Wi-Fi"``. Keeping the
    unsplit word as well is what stops ``"WiFi"`` — which already read correctly — from breaking when
    the split turns it into ``wi`` + ``fi``.

    **Deliberately not substring containment**, which would reach the same two labels and much more
    besides: 664 of the 2,993 label words in the harvest are a proper substring of another one, so
    a goal about the ``lock`` screen would score against ``Clock`` and one about the ``phone``
    against ``microphone``. The joins are bounded the way ``v12_goals._compound`` already bounds
    them — two short parts, a plausible word length — for the same reason.
    """

    base = words(value)
    out = list(base)
    for run in _runs(value):
        out.extend(_case_parts(run))
    for index in range(len(base) - 1):
        left, right = base[index], base[index + 1]
        if len(left) <= 6 and len(right) <= 6:
            joined = left + right
            if 4 <= len(joined) <= 14:
                out.append(joined)
    return out


def _node_text(node: Mapping[str, Any]) -> str:
    return " ".join(str(node.get(key) or "") for key in ("text", "desc")).strip()


def stalled(node: Mapping[str, Any]) -> bool:
    """This node was acted on and the screen did not move. The whole of ``no_progress``, per node.

    Reads the two fields :mod:`v12_progress` writes onto a node — ``tried`` and ``last`` — and
    nothing else. There is deliberately no history list to join against: the joinable version of
    this was the source of every ground-truth bug in the V12 corpus, and the helper already holds
    every ``AccessibilityNodeInfo`` it taps, so it can count per node for free.
    """

    return int(node.get("tried") or 0) > 0 and str(node.get("last") or "") in STALLED


def score(goal_terms: Sequence[str], node: Mapping[str, Any]) -> float:
    """How well *node* answers a goal reduced to *goal_terms*.

    Zero for chrome, so the ever-present "Search settings" row cannot win by sharing one word.
    """

    label = _node_text(node)
    if label.strip().lower() in CHROME:
        return 0.0
    node_words = words(label)
    if not node_words and not node.get("rid"):
        return 0.0
    # Membership is tested against every spelling of the label; the head bonus is not. "First word"
    # has to keep meaning the row's actual first word, or a join two thirds down the summary would
    # collect the bonus meant for the row's own name.
    spellings = variants(label)

    total = 0.0
    for index, term in enumerate(goal_terms):
        if term in spellings:
            # The head word is where an Android row names itself; a hit in the summary is weaker.
            first = node_words and node_words[0] == term
            total += FIRST_TOKEN_BONUS if (first and index == 0) else WORD_MATCH
        elif term in words(node.get("rid")):
            total += RID_MATCH
    return total


def decide(
    goal: str,
    projection: Mapping[str, Any],
    *,
    scrolls_used: int = 0,
) -> dict[str, Any]:
    """One decision: tap a node, scroll for more, or hand off.

    ``projection`` is the output of :func:`drive_projection.project` — ``nodes``, ``more``, ``keys``.
    The returned ``n`` indexes straight back into ``keys``, so the caller always has a real element
    and never a string to resolve. That is the whole reason this grounds at 100%.
    """

    nodes = list(projection.get("nodes") or [])
    keys = list(projection.get("keys") or [])
    terms = content_words(goal)

    # Before the screen, not after it. A host goal is unreachable on every screen, so scoring first
    # only decides how much budget gets spent proving it: on data-v12/test.jsonl the unguarded rule
    # answered 264 of these with a scroll and 44 with a tap — acting on a screen that could never
    # satisfy the goal.
    host = [term for term in terms if term in HOST_TERMS]
    if host:
        return {
            "call": "handoff",
            "reason": "needs_host",
            "why": f"{host} names a host capability, not a destination on screen",
        }

    best_index = -1
    best_score = 0.0
    for index, node in enumerate(nodes):
        if not node.get("tap"):
            continue
        value = score(terms, node)
        if value > best_score:
            best_index, best_score = index, value

    if best_index >= 0 and best_score >= SCORE_FLOOR:
        # The goal's own target is the thing that stalled. Tapping it again is the loop this
        # closes: the screen has already settled, so a second identical tap gets an identical
        # non-result, and on-device that spent the whole budget re-pressing one row.
        #
        # Only the *winner* is consulted, and that is the entire distinction the V12 corpus was
        # built to teach. A stalled node somewhere on the screen means nothing — those rows answer
        # `tap` 49.6% of the time, because the node the goal is about has never been touched. So
        # stalled nodes are scored normally and stay eligible to win; what changes is only what
        # winning means when the winner is the one that already failed.
        if stalled(nodes[best_index]):
            return {
                "call": "handoff",
                "reason": "no_progress",
                "n": nodes[best_index].get("n"),
                "stable_key": keys[best_index] if best_index < len(keys) else None,
                "score": round(best_score, 2),
                "why": (
                    f"best match was tried {nodes[best_index].get('tried')} time(s) and last "
                    f"{nodes[best_index].get('last')}"
                ),
            }
        return {
            "call": "tap",
            "n": nodes[best_index].get("n"),
            "stable_key": keys[best_index] if best_index < len(keys) else None,
            "score": round(best_score, 2),
            "why": f"best word overlap with {terms}",
        }

    # Nothing convincing on screen. More screen is the cheaper hypothesis than refusal — but only
    # while there is reason to believe more screen exists.
    can_scroll = bool(projection.get("more")) or any(node.get("scroll") for node in nodes)
    if can_scroll and scrolls_used < MAX_SCROLLS:
        return {
            "call": "scroll",
            "why": f"best score {best_score:.2f} below floor {SCORE_FLOOR}; more content available",
        }

    # Refusing. The only question left is what to call it.
    #
    # A consent dialog on screen makes `needs_auth` the better name, and this is the only place the
    # dialog is ever consulted — which is what makes the check safe. "A dialog is up" must never be
    # grounds to *stop*, because pressing "Don't allow" is a legitimate move and `v12_corpus`'s
    # `decline` family exists to say so. Here nothing is being stopped: the rule had already given
    # up. Whether the goal was "about declining" therefore never has to be decided from its text —
    # if word overlap can reach the decline control, the tap branch above fired and this line was
    # never reached.
    #
    # Measured on data-v12/test.jsonl, over the 733 rows that show a decline control: the rule was
    # already answering `target_absent` on 709 of them, including all 332 whose truth is
    # `needs_auth`. So this renames a refusal that was happening anyway, and it never converts a tap
    # into a refusal. The 21 rows where the rule does reach the decline control by overlap still tap
    # it. And `needs_auth` is the more actionable of the two names regardless of which the corpus
    # would score: it tells the host there is a consent gate to answer, where `target_absent` tells
    # it to give up.
    if any(_node_text(node).strip().lower() in DECLINE_LABELS for node in nodes):
        return {
            "call": "handoff",
            "reason": "needs_auth",
            "why": f"best score {best_score:.2f} below floor behind a consent dialog",
        }

    # What is left is honestly weak: "nothing matched and nothing to scroll" cannot tell a target
    # that is absent from one present under a name sharing no *concept* with the goal. Spelling is
    # handled — see `variants` — but a true synonym is not, and nothing on the screen distinguishes
    # the two cases.
    return {
        "call": "handoff",
        "reason": "target_absent",
        "why": f"best score {best_score:.2f} below floor and nothing further to reveal",
    }
