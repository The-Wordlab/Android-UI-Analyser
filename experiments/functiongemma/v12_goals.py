"""Goals built from a real node, by mechanisms that cannot produce a wrong answer.

The first version of this used a hand-written concept map — head word ``security`` maps to goals about
``lock screen``, ``pin``, ``fingerprint`` — matched against whatever real label contained the head
word. On the harvest that paired *"go to the lock screen settings"* with the row **"App security No
info yet"**, which is not the lock screen and not a settings destination at all. It produced 998 such
pairs and looked like a large bridging corpus.

Wrong ground truth is worse than no ground truth: the model cannot tell it from the real thing, and
it trains against it just as hard. So the mechanisms here are restricted to ones where the answer
follows from the label itself, plus one property enforced on every row:

**Uniqueness.** A goal is only usable if exactly one node on its screen is the best match for it. The
harvest is full of screens where a heading repeats a row's words — 25% of clickable texts are
duplicated on their own screen — and a goal that fits two nodes has no single correct answer. Those
are dropped rather than guessed at, which is why :func:`goal_for` returns ``None`` so often.

What is deliberately *not* here: real semantic bridging, the kind where "make the text bigger" has to
reach "Display". It needs verified pairs, the harvest is too shallow on Settings to supply them
(top-level destinations appear once or twice each), and inventing them is exactly the mistake above.
That family stays out until the pairs are curated, and it is the one thing in this task that argues
for a larger model rather than better data.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from android_ui_analyser.drive_rule import CHROME, content_words, score, words

#: Frames a goal without naming a destination. Deliberately varied and deliberately noisy: the V11
#: corpus phrased every goal the same way, and the model learned the frame rather than the target.
FRAMES = (
    "open {t}",
    "go to {t}",
    "get to {t}",
    "find {t}",
    "take me to {t}",
    "I need {t}",
    "navigate to {t}",
    "show me {t}",
    "land on {t}",
    "{t}",
    "can you open {t}",
    "reach the {t} screen",
    "get me into {t}",
)


#: Phrasings that ask whether something is *already* here rather than asking to go there. They are a
#: separate set because they carry a different question, and they are used by more than one answer on
#: purpose: the first version generated them only when the target was present, which made
#: ``goal_first_word == "is"`` predict ``done`` at precision 1.000 without reading the screen. An
#: arrival question about something absent is a refusal, and the corpus needs both.
ARRIVAL_FRAMES = (
    "is {t} on screen",
    "confirm {t} is showing",
    "check {t} is here",
    "can you see {t}",
    "is {t} visible",
    "verify {t} is displayed",
)


def label_of(node: Mapping[str, Any]) -> str:
    return " ".join(str(node.get(key) or "") for key in ("text", "desc")).strip()


def _unique_best(goal: str, nodes: Sequence[Mapping[str, Any]], want: str) -> bool:
    """True when *want* is the single highest-scoring tappable node for *goal*.

    Scored with the baseline's own function on purpose. It is the thing this corpus has to beat, so
    a row it cannot even resolve unambiguously is not a fair test of anything — and a row where two
    nodes tie has no correct answer to learn.
    """

    terms = content_words(goal)
    if not terms:
        return False
    ranked = [(score(terms, node), node.get("n")) for node in nodes if node.get("tap")]
    ranked = [pair for pair in ranked if pair[0] > 0]
    if not ranked:
        return False
    ranked.sort(reverse=True)
    if ranked[0][1] != want:
        return False
    return len(ranked) == 1 or ranked[0][0] > ranked[1][0]


def _compound(term: str) -> str | None:
    """``Wi-Fi`` as ``wifi``: the same word, tokenised the way a person types it.

    This is the live miss, exactly. The Wi-Fi row reported itself as "Internet AndroidWifi", the goal
    said "wifi", and word overlap scored zero because the label tokenises to ``wi`` + ``fi``. No
    semantics are involved and nothing is invented — it is one string with the separator removed.
    """

    parts = [part for part in words(term) if part]
    if len(parts) < 2 or any(len(part) > 6 for part in parts):
        return None
    joined = "".join(parts)
    return joined if 4 <= len(joined) <= 14 else None


def goal_for(
    node: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    rng: random.Random,
    *,
    frames: Sequence[str] = FRAMES,
) -> tuple[str, str] | None:
    """A goal whose single correct answer is *node*, plus the mechanism that produced it.

    Returns ``None`` when no mechanism yields an unambiguous goal — a blank label, pure chrome, or a
    phrasing that fits a second node just as well. Callers are expected to try another node.
    """

    label = label_of(node)
    if not label or label.strip().lower() in CHROME:
        return None
    tokens = words(label)
    if not tokens:
        return None

    styles = ["head", "subset", "summary", "compound"]
    rng.shuffle(styles)
    # Head phrasing is the near-verbatim one, so it goes last: any other mechanism that works is
    # preferred. V11 had the target sitting verbatim in 73.6% of goals and learned to copy it.
    styles.sort(key=lambda style: style == "head")

    for style in styles:
        target: str | None = None
        if style == "head":
            keep = [word for word in tokens[:3] if len(word) > 1]
            target = " ".join(keep) if keep else None
        elif style == "subset":
            content = [word for word in content_words(label) if len(word) > 2]
            if len(content) >= 2:
                pick = rng.sample(content, 2)
                target = " ".join(pick)
        elif style == "summary":
            tail = [word for word in content_words(" ".join(tokens[2:])) if len(word) > 3]
            if tail:
                target = rng.choice(tail)
        elif style == "compound":
            for span in range(len(tokens) - 1):
                joined = _compound(" ".join(tokens[span : span + 2]))
                if joined:
                    target = joined
                    break

        if not target:
            continue
        goal = rng.choice(frames).format(t=target)
        if style == "compound":
            # This mechanism scores zero against its own label by design, so the baseline cannot be
            # asked to resolve it. Uniqueness is checked structurally instead: no *other* node on the
            # screen may yield the same compound, or the row has two defensible answers.
            if _compound_is_unique(target, nodes, node.get("n")):
                return goal, style
            continue
        if _unique_best(goal, nodes, node.get("n")):
            return goal, style
    return None


def _compound_is_unique(joined: str, nodes: Sequence[Mapping[str, Any]], want: str) -> bool:
    """No node other than *want* produces this same compound from its own label."""

    for other in nodes:
        if other.get("n") == want or not other.get("tap"):
            continue
        tokens = words(label_of(other))
        for span in range(max(0, len(tokens) - 1)):
            if _compound(" ".join(tokens[span : span + 2])) == joined:
                return False
    return True


def absent_goal(
    nodes: Sequence[Mapping[str, Any]],
    pool: Sequence[str],
    rng: random.Random,
    *,
    frames: Sequence[str] = FRAMES,
) -> str | None:
    """A goal naming something no node on this screen answers — the ground truth for refusal.

    Drawn from labels harvested off *other* screens, so the vocabulary is real rather than invented,
    and then checked against this screen: any overlap at all and it is discarded. A near-miss row
    would teach refusal on a screen where tapping was defensible, which is the more expensive
    mistake of the two.
    """

    for _ in range(24):
        label = pool[rng.randrange(len(pool))]
        if label.strip().lower() in CHROME:
            continue
        terms = content_words(label)
        if not terms:
            continue
        if any(score(terms, node) > 0 for node in nodes):
            continue
        keep = [word for word in words(label)[:3] if len(word) > 1]
        if not keep:
            continue
        return rng.choice(frames).format(t=" ".join(keep))
    return None
