"""V12's cheap features, fed to the V11 search.

The search is imported rather than copied: it is the part worth having exactly once. What changes
between contracts is the feature list, because "cheap" is defined relative to the task — anything
computable *without* comparing the goal's words to the screen's words is a way to be right for the
wrong reason.

This list is the second version. The first reported the corpus **clean** while a rule predicted
``no_progress`` at precision **1.000** over 1,004 rows: history entries invented node ids from
``randrange(1, 9)`` while the real target id could reach ``n14``, so "a history entry names a node
above n8" was a perfect tell. That rule passed every one of the search's firing conditions — precision
1.000, recall 0.178, support 1004 — and was invisible purely because no feature here read the *number*
in a node id. A gate with an incomplete feature list does not report uncertainty; it reports clean.

So the features below deliberately include the ones that were missing, and the design they now guard
is the one where progress lives on the node (:mod:`v12_progress`) instead of in a joinable list:

``any_node_stalled``
    The shortcut the whole ``tap_despite_stalls`` family exists to close. ``no_progress`` is correct
    when *the node the goal names* was tried and did not move. A model can approximate that with
    "some node on this screen has stalled", which needs no goal at all. If this predicts
    ``no_progress`` precisely, those rows are not doing their job.

``stalled_node_index_band`` and ``max_tried_index_band``
    The numeric leak, named directly. If the id of the stalled node predicts anything, some family is
    marking nodes from a different range than the others.

``touched_count_band``, ``stalled_count_band``, ``max_tried_band``, ``scrolls_band``, ``step_band``
    All the ways "how much has happened" can be counted. V11's model chose its refusal reason by
    counting, so every counter available to it is measured here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.functiongemma.v11_shortcut_gate import (
    ACCEPTED_RESIDUALS,
    PRECISION_LIMIT,
    RECALL_FLOOR,
    search,
)
from experiments.functiongemma.v12_corpus import answer_of
from experiments.functiongemma.v12_progress import STALLED


def label(row: Mapping[str, Any]) -> str:
    """The decision class, reason included.

    Reasons are kept because dropping them is how V11's real failure survived a rebuild: the corpus
    balanced its calls and not its reasons, so ``needs_host`` sat at 51% of refusals when nothing had
    been tried and ``no_progress`` at 40% once several had, under a reported spread of 0.0001.
    """

    return answer_of(row)


def _index(node_id: Any) -> int:
    text = str(node_id or "")
    return int(text[1:]) if text[1:].isdigit() else 0


def features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Everything decidable without matching goal words against screen words."""

    context = json.loads(row["messages"][1]["content"])
    screen = context["screen"]
    nodes = screen["nodes"]
    goal_words = str(context["goal"]).split()

    tappable = [node for node in nodes if node.get("tap")]
    touched = [node for node in nodes if int(node.get("tried") or 0) > 0]
    stalled = [node for node in touched if str(node.get("last") or "") in STALLED]

    return {
        # --- screen shape
        "node_count_band": min(len(nodes) // 3, 4),
        "tappable_count_band": min(len(tappable) // 3, 4),
        "one_tappable_only": len(tappable) == 1,
        "has_scrollable": any(node.get("scroll") for node in nodes),
        "more_flag": bool(screen.get("more")),
        "has_rid_only_node": any(
            node.get("rid") and not node.get("text") and not node.get("desc") for node in nodes
        ),
        # A permission dialog must not be sufficient grounds for `needs_auth` on its own; that is
        # what the `decline` family is for, and this is how it gets checked.
        "is_auth_screen": any(
            str(node.get("text") or node.get("desc") or "").strip().lower()
            in {"don't allow", "dont allow", "deny"}
            for node in nodes
        ),
        # --- progress, every way it can be counted
        "any_node_stalled": bool(stalled),
        "touched_count_band": min(len(touched), 3),
        "stalled_count_band": min(len(stalled), 3),
        "max_tried_band": min(max((int(n.get("tried") or 0) for n in touched), default=0), 4),
        "scrolls_band": min(int(screen.get("scrolls") or 0), 3),
        "step_band": min(int(context.get("step") or 0), 5),
        # --- the numeric leak that went undetected once already
        "stalled_node_index_band": min(max((_index(n.get("n")) for n in stalled), default=0) // 4, 3),
        "max_tried_index_band": min(max((_index(n.get("n")) for n in touched), default=0) // 4, 3),
        "stalled_node_is_last_listed": bool(stalled)
        and max(_index(n.get("n")) for n in stalled) == max(_index(n.get("n")) for n in nodes),
        # --- goal surface only, never goal against screen
        "goal_first_word": goal_words[0].casefold() if goal_words else "",
        "goal_word_count_band": min(len(goal_words) // 3, 4),
    }


def check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Run the search and decide whether the corpus may ship."""

    report = search(rows, feature_fn=features, label_fn=label)
    accepted = {entry["rule"] for entry in ACCEPTED_RESIDUALS}
    blocking = [f for f in report["shortcuts"] if f["rule"] not in accepted]
    report["blocking"] = blocking
    report["clean"] = not blocking
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a .jsonl of training rows")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    rows = []
    with args.path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if args.limit and index >= args.limit:
                break
            rows.append(json.loads(line))

    report = check(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["clean"]:
        print(
            f"\nBLOCKED: {len(report['blocking'])} cheap rule(s) predict a class at "
            f"precision >= {PRECISION_LIMIT} and recall >= {RECALL_FLOOR}."
        )
        return 1
    print("\nclean: no cheap rule predicts any minority class this precisely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
