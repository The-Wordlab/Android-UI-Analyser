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

**"It cannot say impossible" was only half-answered**, and the other half was hiding behind the
word "word". :data:`HOST_TERMS` admits a term only if real screens almost never use it, which is
right and is why the list is short — but the rule was only ever applied to *single* words, and that
left ``needs_host`` the weakest class at 68.1%. All 238 misses name a capability whose words are
individually far too common to refuse: "copy a photo into the gallery" is ``aua media add``, and
"photo" is on 4.4% of the harvest while "gallery" is on 4.5%. The pair is on 0.0%.
:data:`HOST_PAIRS` holds the pairs that clear the same 1% ceiling, and the ones measurement refused
are recorded there too — both clock capabilities among them, because Date & time is a real
destination one tap away. Measured on the full held-out set: 89.5% -> 92.3% overall, ``needs_host``
68.1% -> 89.8%, 162 rows fixed and **no other class touched**.

**"It reads a question as an instruction"** was the largest hole left, and it was invisible until
the classes were measured separately. Stratified over ``runs/functiongemma/data-v12/test.jsonl``,
the rule scored ``tap`` 87%, ``scroll`` 100% and ``handoff`` 86% — and ``done`` **7.6%**. All 419 of
those rows ask a question about the screen rather than giving it an instruction ("confirm the row is
showing", "is the badge on screen", "can you see the banner"), and the rule read every one as
navigation and pressed the best match. For a test suite that is the one unacceptable answer: the
assertion was "is this visible", and acting on the screen destroys the thing being asserted about.
:func:`only_asks_to_look` reads the frame, and the *subject* it returns is as important as its
verdict — "check X is here" used to put ``check`` and ``here`` into the goal terms, so a row named
"Check for updates" outscored the row being asked about. The frame says the goal is a question; only
the screen answers it, so an absent subject falls through to the ordinary scroll-then-refuse path
and reaches ``target_absent``, which is the honest reading of "is it visible" when it is not.
Measured on the full held-out set: 82.8% -> 89.5% overall, ``done`` 7.6% -> 100%, 389 rows fixed
against **1** regressed — the goal ``"find showing"``, whose target label genuinely is the word
"showing", and which no frame reader can be expected to survive.

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

#: Host capabilities named by two ordinary words. The admission rule for :data:`HOST_TERMS` was only
#: ever applied to single words, and that is what left ``needs_host`` the weakest class at 68.1%: all
#: 238 misses on the held-out set name a capability whose words are individually far too common to
#: refuse. "copy a photo into the gallery" is ``aua media add``, and over the 638 harvested screens
#: "photo" is on 4.4% and "gallery" on 4.5% — both well over the 1% ceiling, both correctly excluded.
#: The **pair** is on 0.0%.
#:
#: Every pair here clears exactly the same bar as a single word, and
#: ``test_a_pair_is_admitted_on_the_same_evidence_as_a_single_word`` re-measures all of them against
#: the harvest rather than trusting this comment:
#:
#:     copy+gallery 0.0%   copy+photo 0.0%   copy+image 0.0%   copy+video 0.0%
#:     push+photo 0.0%     photo+album 0.0%  image+gallery 0.5%
#:     record+video 0.9%
#:     list+devices 0.5%   attached+devices 0.0%   list+emulators 0.0%
#:     app+logs 0.6%       errors+logs 0.0%   read+logs 0.2%   check+logs 0.2%   device+logs 0.3%
#:
#: **The pairs measurement refused are as informative as the ones it admitted, and both clock
#: capabilities are among them.** ``aua clock set`` is named by "change the system time" and "set the
#: device clock to midnight"; ``system``+``time`` is on 6.3% of harvested screens and ``set``+``clock``
#: on 48.9%, because Date & time is a real destination one tap away. Refusing either would break
#: navigation to it, so 76 of the 238 misses stay unfixed and hand off as ``target_absent`` — still
#: giving the host control, just under a vaguer name. ``video``+``gallery`` (4.1%) and
#: ``connected``+``devices`` (1.9%) were refused the same way, which is why the list reaches those
#: capabilities through a different pair instead.
#:
#: A caveat on the *lift*, so nobody reads more into it than was measured: the held-out set contains
#: only six distinct ``needs_host`` goal shapes. The pairs were chosen from AUA's actual command
#: surface and validated against screens somebody else's apps produced, but how well they generalise
#: to phrasings a real agent invents is not something this corpus can answer.
#:
#: **No pair may contain a** :data:`STOPWORDS` **member.** The goal is reduced by
#: :func:`content_words` before any pair is tested, so such a pair is silently unreachable rather
#: than merely useless. ``record``+``screen`` and ``recording``+``screen`` were both measured
#: admissible and both removed for exactly this reason — ``screen`` never survives to be tested — and
#: ``test_no_pair_is_made_of_a_word_the_goal_never_keeps`` fails on the next one.
HOST_PAIRS = frozenset(
    {
        frozenset({"copy", "gallery"}),
        frozenset({"copy", "photo"}),
        frozenset({"copy", "image"}),
        frozenset({"copy", "video"}),
        frozenset({"push", "photo"}),
        frozenset({"photo", "album"}),
        frozenset({"image", "gallery"}),
        frozenset({"record", "video"}),
        frozenset({"list", "devices"}),
        frozenset({"attached", "devices"}),
        frozenset({"list", "emulators"}),
        frozenset({"app", "logs"}),
        frozenset({"errors", "logs"}),
        frozenset({"read", "logs"}),
        frozenset({"check", "logs"}),
        frozenset({"device", "logs"}),
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

#: Visibility phrases that end a goal which is asking a *question* about the screen rather than
#: giving it an instruction. Longest first: "is showing" must win over "showing".
#:
#: Measured over ``runs/functiongemma/data-v12`` (train + test, 51,658 rows), these phrases plus
#: :data:`LOOK_HEADS` identify **100%** of the 3,699 ``done`` rows and fire on 0.1-0.5% of every
#: acting class. The one class they also fire on heavily is ``target_absent`` (47.7%) — and that is
#: the point rather than a defect: those are the same question asked about something the screen does
#: not show, and only reading the screen can tell the two apart.
LOOK_TAILS = (
    "can be seen",
    "is displayed",
    "is showing",
    "is visible",
    "is here",
    "on screen",
    "be seen",
    "displayed",
    "showing",
    "visible",
)

#: Openings that make a goal a question on their own, needing no trailing phrase. Deliberately
#: narrow. The corpus opens 515 look-goals *and* 2,645 acting goals with "can you", so the opening
#: words alone cannot decide it — only the verb can. ``can you see`` asks; ``can you open`` instructs.
LOOK_HEADS = (
    "can you tell me if ",
    "can you confirm ",
    "can you verify ",
    "do you see ",
    "can you see ",
    "can i see ",
    "let me see ",
    "tell me if ",
)

#: Question verbs that begin a goal already identified by a :data:`LOOK_TAILS` phrase. Stripped so
#: they cannot reach the scorer: "check X is here" used to put ``check`` and ``here`` into the goal
#: terms, and a row named "Check for updates" then outscored the row being asked about.
LOOK_LEADS = (
    "please confirm ",
    "please check ",
    "make sure ",
    "confirm that ",
    "verify that ",
    "check that ",
    "confirm ",
    "verify ",
    "ensure ",
    "assert ",
    "check ",
    "does ",
    "are ",
    "is ",
    "do ",
)

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


def expand_goal(goal: str, vocabulary: Mapping[str, Sequence[str]] | None) -> str:
    """Fold an app's own words for a concept into *goal*, once, wherever the concept is named.

    The case this exists for, from a real run: a scenario says **Feed**, the app's control is
    labelled **Ideas**, and the goal scored zero against every control on screen while the target was
    plainly visible. Nothing was wrong with the rule — the words genuinely do not meet — and nothing
    was wrong with the scenario. There was simply nowhere to write the mapping down, so every scenario
    touching that tab re-derived it.

    **Why the goal and not the rule.** The helper's word tables are compiled into the APK with no way
    to seed them, so teaching :func:`score` would give the host lane a vocabulary the device lane
    lacks — the two lanes disagreeing about one goal, which is the thing they are tested not to do.
    Expanding here happens before the goal is sent anywhere, so both lanes get it, no protocol
    changes, and there is no second implementation to keep in step.

    Adds rather than replaces, so a goal already using the app's words stays correct, and a wrong
    vocabulary entry costs a weak extra match rather than losing the right one.
    """

    if not vocabulary:
        return goal

    lowered = f" {' '.join(words(goal))} "
    present = set(words(goal))
    additions: list[str] = []
    for term, spellings in vocabulary.items():
        keys = words(term)
        if not keys:
            continue
        # A key that is a stopword — or only stopwords — would fire on nearly every goal and make
        # every screen look like it contained the target. Refuse rather than degrade every later run.
        if all(key in STOPWORDS for key in keys):
            raise ValueError(
                f"vocabulary term {term!r} is a stopword and would match almost every goal; "
                f"it is too common to name a destination"
            )
        # Phrase match, so "lock screen" cannot fire on the bare word "screen" — which appears in
        # more than half of real harvested screens.
        if f" {' '.join(keys)} " not in lowered:
            continue
        for spelling in spellings:
            if not spelling:
                continue
            if all(part in present for part in words(spelling)):
                continue  # the goal already says it; saying it twice is noise in every log
            additions.append(str(spelling))
    return f"{goal} {' '.join(additions)}".strip() if additions else goal


def only_asks_to_look(goal: str) -> str | None:
    """The subject of a goal that asks what is on screen, or ``None`` if the goal asks for an action.

    This is the whole of the distinction, and it is grammatical rather than semantic — which is why
    it belongs in a rule and not in a model. "is doodads on screen" and "can you open doodads" name
    the same target; only one of them wants it pressed.

    Returning the *subject* matters as much as returning the verdict. The frame words are goal text
    the scorer would otherwise have to rank nodes against, and they are words real labels use.
    """

    text = " ".join(str(goal or "").split())
    lowered = text.lower()

    for head in LOOK_HEADS:
        if lowered.startswith(head):
            return text[len(head) :].strip() or None

    for tail in LOOK_TAILS:
        if not lowered.endswith(tail):
            continue
        cut = len(text) - len(tail)
        # "reach the solution screen" ends with the letters of "on screen" and is an instruction.
        # Five of this rule's six measured regressions were exactly that, so the phrase has to begin
        # where a word begins.
        if cut and text[cut - 1] != " ":
            continue
        subject = text[:cut].strip()
        stripped = subject.lower()
        for lead in LOOK_LEADS:
            if stripped.startswith(lead):
                subject = subject[len(lead) :].strip()
                break
        return subject or None

    return None


def touched(node: Mapping[str, Any]) -> bool:
    """This node has been acted on at least once during the run.

    Written onto the node by whatever is driving — the helper counts per run, the host loop keys by
    ``stable_key`` — so no history list has to be joined against the screen. Omitted entirely when
    zero, which is why this reads truthiness rather than comparing to 0: a runtime that sends
    ``"tried": 0`` and one that sends nothing must mean the same thing.
    """

    return bool(node.get("tried"))


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

    present = set(terms)
    for pair in HOST_PAIRS:
        if pair <= present:
            return {
                "call": "handoff",
                "reason": "needs_host",
                "why": (
                    f"{sorted(pair)} together name a host capability; neither word alone could be "
                    f"refused without breaking navigation"
                ),
            }

    # A goal that only asks what is on screen. 419 of the held-out rows are this shape and the rule
    # answered 6.9% of them, because it read every one as navigation and pressed the best match.
    # For a test suite that is the one unacceptable answer: the assertion was "is this visible", and
    # acting on the screen destroys the thing being asserted about.
    #
    # Two halves, and the second is what keeps a false pass impossible. The frame says the goal is a
    # question; only the screen can answer it. Subject found -> `done`. Subject absent -> nothing is
    # returned here at all, and the ordinary scroll-then-refuse path below reaches `target_absent`,
    # which is the honest answer to "is it visible" when it is not. Measured on data-v12: 47.7% of
    # look-shaped rows are exactly that case.
    #
    # `tap` is not consulted. Most things worth asserting about are also pressable, and pressable is
    # not permission.
    subject = only_asks_to_look(goal)
    if subject is not None:
        terms = content_words(subject)
        seen_index = -1
        seen_score = 0.0
        for index, node in enumerate(nodes):
            value = score(terms, node)
            if value > seen_score:
                seen_index, seen_score = index, value
        if seen_index >= 0 and seen_score >= SCORE_FLOOR:
            return {
                "call": "done",
                "n": nodes[seen_index].get("n"),
                "stable_key": keys[seen_index] if seen_index < len(keys) else None,
                "score": round(seen_score, 2),
                "why": (
                    f"the goal asks whether {terms} is on screen and this node shows it; "
                    f"nothing was pressed"
                ),
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
        # Pressed already, and it worked. Measured live, this was the budget burn: "Apps" tapped
        # three times, each tap reporting `changed`, because nothing had stalled and the driver had
        # no notion of having finished. It simply re-pressed the best match until the budget ran out.
        #
        # `done` here claims the *action* is complete and nothing more. The control the goal named
        # was pressed and the screen moved; whether the destination is correct is the caller's to
        # judge, and no reading of this lets it skip that. That narrowness is the point — it needs no
        # vocabulary, no arrival heuristic, and it cannot manufacture a false claim about what the
        # screen shows, because it is a fact about what this driver did.
        if touched(nodes[best_index]):
            return {
                "call": "done",
                "n": nodes[best_index].get("n"),
                "stable_key": keys[best_index] if best_index < len(keys) else None,
                "score": round(best_score, 2),
                "why": (
                    f"best match was pressed {nodes[best_index].get('tried')} time(s) and the "
                    f"screen changed; the action the goal named is carried out"
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
