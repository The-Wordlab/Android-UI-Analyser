"""What has already been tried, written onto the node it was tried on.

The previous design sent a separate history list — ``n3 Wi-Fi -> unchanged`` — and asked the model to
join it against the screen. Every bug in this experiment came out of that join:

* history node ids were invented from ``rng.randrange(1, 9)`` while the real target id could be
  ``n9``..``n14``, so "a history entry names a node above n8" predicted ``no_progress`` at precision
  **1.000** over 1,004 rows. The shortcut gate has no feature that reads the number in an id, so it
  reported the corpus clean.
* labels in history were truncated to 40 characters, so the join was against a prefix.
* the two families meant to share an evidence shape did not: repeated stalls on one real node scored
  ``no_progress`` 377 times against ``tap`` 3 times, so the probe testing that distinction was asking
  the model to contradict its own training data.

None of that is fixable by patching the join, because the join is the problem. The helper already
holds every ``AccessibilityNodeInfo`` — ``DriveFeature`` taps them directly — so the device can count
attempts per node for free and hand the model an answer instead of a puzzle:

    {"n": "n3", "text": "Wi-Fi", "tap": true, "tried": 2, "last": "unchanged"}

Now ``no_progress`` is "the node the goal names has ``tried`` above zero and a stalled ``last``",
which is one field on the node the model is already attending to. A 350M model doing set-against-set
comparison in one forward pass with no scratchpad was the hard part, and it was self-inflicted.

Two fields, plus one counter that genuinely is global:

``tried``   how many times this node has been acted on. Omitted when zero, because most nodes on most
            screens have never been touched and a field repeated across fourteen nodes is the single
            most expensive thing in the prompt.
``last``    what happened the last time it was acted on. Omitted with ``tried``.
``scrolls`` how many scrolls have been spent on this screen. Belongs to the screen rather than any
            node, and it is what makes "stop scrolling and refuse" decidable.

There is no free-text history at all. An id cannot be invented into a field that does not exist, which
is the same reasoning that made the pointing contract work where selector strings did not.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

#: An action landed and the screen became a different screen.
CHANGED = "changed"
#: An action landed and nothing happened. The evidence for ``no_progress``.
UNCHANGED = "unchanged"
#: The node refused the click outright. Also evidence.
BLOCKED = "blocked"

OUTCOMES = (CHANGED, UNCHANGED, BLOCKED)

#: Outcomes meaning the run is not advancing. What the model must learn to read — on the right node.
STALLED = (UNCHANGED, BLOCKED)

#: How many scrolls a screen may report. Above this, refusing is the only sound answer, which is what
#: makes the counter worth sending at all.
MAX_SCROLLS = 3


def annotate(
    node: MutableMapping[str, Any],
    *,
    tried: int,
    last: str | None = None,
) -> MutableMapping[str, Any]:
    """Write progress onto *node*, or leave it untouched when nothing has been tried.

    Omitting the fields at zero is not only about tokens. A node carrying ``"tried": 0`` teaches that
    the field is always present, and a runtime that omitted it would then be off-distribution — the
    same train/serve gap that cost V10 its whole run.
    """

    if tried <= 0:
        node.pop("tried", None)
        node.pop("last", None)
        return node
    node["tried"] = tried
    node["last"] = last or CHANGED
    return node


def touched(node: Mapping[str, Any]) -> bool:
    return int(node.get("tried") or 0) > 0


def stalled(node: Mapping[str, Any]) -> bool:
    """This node was tried and did not move the screen. The whole of ``no_progress``, per node."""

    return touched(node) and str(node.get("last") or "") in STALLED


def sample_tried(rng: random.Random) -> int:
    """How many times an already-touched node was touched. Shared by every family.

    One distribution for all of them, so the *count* cannot name the answer. When ``target_absent``
    alone used three stalls and ``no_progress`` used two, the count predicted the family at precision
    1.000 and the gate said so — this is the same failure prevented structurally rather than caught.
    """

    return rng.choice((1, 1, 2, 2, 2, 3, 3, 4))


def scatter(
    nodes: Sequence[MutableMapping[str, Any]],
    rng: random.Random,
    *,
    skip: str | None = None,
    count: int | None = None,
    outcomes: Sequence[str] = OUTCOMES,
) -> list[str]:
    """Mark some nodes as already tried, skipping *skip*, and return the ids marked.

    Applied to every family, including the ones whose answer is ``tap``. Without that, "some node on
    this screen has been tried" is enough to refuse, and the model never has to work out *which* node
    the goal is asking about — which is the entire behaviour being bought.
    """

    candidates = [node for node in nodes if node.get("tap") and node.get("n") != skip]
    if not candidates:
        return []
    if count is None:
        count = rng.randrange(0, min(4, len(candidates) + 1))
    rng.shuffle(candidates)
    marked: list[str] = []
    for node in candidates[:count]:
        annotate(
            node,
            tried=sample_tried(rng),
            last=outcomes[rng.randrange(len(outcomes))],
        )
        marked.append(str(node.get("n")))
    return marked
