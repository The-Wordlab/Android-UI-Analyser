"""Generic shortcut detector: can the label be predicted without reading the goal?

`v11_curriculum.check_confounds` gated the two confounds already known from V9 and V10 — node count
and visible destructiveness — and reported reassuring numbers, 0.0996 and 0.033, while **seven**
other rules predicted the answer at precision 1.000. Training then confirmed all of it: ten of
seventeen families scored exactly 0.000 while the shortcut-decidable ones scored 1.000.

The lesson is not "add scrollable to the gate list". It is that a gate enumerating known failures
certifies the next unknown one. This module inverts that: instead of asking about specific
suspicions, it enumerates every **cheap feature** and searches exhaustively for any rule that
predicts the label too well.

A feature is *cheap* when it can be computed without comparing the goal's words to the screen's
words. That comparison is the entire task, so anything decidable without it is, by construction, a
way to be right for the wrong reason.

The test is on **precision, not correlation**. A cheap feature is allowed to be informative — the
IME really is up when the keyboard needs dismissing, and a scrollable really is present when
scrolling is needed. What is not allowed is for it to be *sufficient*. Precision 1.000 means the
model never has to look at anything else, and a model given that option will take it every time.

No sklearn: an exhaustive search over single features and pairs is both dependency-free and more
useful than a fitted classifier, because it reports the offending rule in words a human can act on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: A rule is a shortcut when it is this precise about a class that is not already the majority.
PRECISION_LIMIT = 0.95
#: ...and covers at least this much of that class. A precise rule for three rows is a coincidence.
#:
#: Lowered from 0.25 after a calibration failure. The withdrawn host-lane residual measured recall
#: 0.257 on a 6,000-row sample and **below 0.25 on the full 38,460** — so with the exemption removed
#: the gate reported a clean corpus, while the trained model was demonstrably using that very
#: correlation to route device goals to the host lane. A rule too weak to trip the gate was still
#: strong enough to shape gradients.
#:
#: The deeper lesson is that this is a *threshold* patch on a *coverage* problem: the real signal
#: was "lowercase destination plus host-lane-style filler", and no feature here names it. A clean
#: gate is necessary and not sufficient — the trained model plus a targeted probe
#: (`v11_stress_probe`) is the only thing that settles exploitation.
RECALL_FLOOR = 0.15
#: Classes this common are the majority answer; predicting them well is not a shortcut.
BASE_RATE_CEILING = 0.60
#: Below this support a rule is not measured at all.
MIN_SUPPORT = 40

_DESTRUCTIVE_WORDS = ("delete", "erase", "reset", "sign out", "remove", "clear", "wipe", "revoke")


def _label(row: Mapping[str, Any]) -> str:
    """The decision class: the call, plus the argument that distinguishes it."""

    meta = row["metadata"]
    call = meta["call"]
    if call == "next_step":
        return f"next_step:{meta['kind']}"
    if call == "handoff":
        return f"handoff:{meta['reason']}"
    return call


def features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Every feature computable without comparing the goal's words to the screen's words."""

    context = json.loads(row["messages"][1]["content"])
    screen = context["screen"]
    nodes = screen["nodes"]
    goal = str(context["goal"])
    words = goal.split()
    history = context.get("history") or []

    def destructive(node: Mapping[str, Any]) -> bool:
        blob = " ".join(str(node.get(k, "")) for k in ("text", "desc", "rid")).casefold()
        return any(word in blob for word in _DESTRUCTIVE_WORDS)

    rids = [str(node["rid"]) for node in nodes if node.get("rid")]
    return {
        # --- screen shape
        "has_scrollable": any(node.get("scrollable") for node in nodes),
        "ime_up": bool(screen.get("ime")),
        "settling": bool(screen.get("settling")),
        "has_screen_name": bool(screen.get("name")),
        "node_count_band": min(len(nodes) // 3, 4),
        "clickable_count_band": min(sum(1 for n in nodes if n.get("clickable")) // 3, 4),
        "any_destructive_node": any(destructive(node) for node in nodes),
        "all_rids_share_prefix": bool(rids) and all(r.startswith(rids[0][:6]) for r in rids),
        "has_rid_only_node": any(
            n.get("rid") and not n.get("text") and not n.get("desc") for n in nodes
        ),
        # --- history shape
        "history_len_band": min(len(history), 5),
        "history_tail_kind": (history[-1].split()[0] if history else "<none>"),
        "history_has_back": any(line.startswith("key back") for line in history),
        "history_repeats_a_line": len(history) != len(set(history)),
        # --- goal surface, never goal-vs-screen
        "goal_first_word": words[0].casefold() if words else "",
        "goal_word_count_band": min(len(words) // 3, 4),
        "goal_has_inner_capital": any(w[:1].isupper() for w in words[1:]),
        "goal_inner_capital_count": min(sum(1 for w in words[1:] if w[:1].isupper()), 3),
    }


def _rules(table: Sequence[tuple[dict[str, Any], str]]) -> Iterable[tuple[str, Any, str]]:
    """Every single-feature equality rule, and every pair conjunction."""

    names = sorted(table[0][0])
    values: dict[str, set[Any]] = {name: set() for name in names}
    for row, _ in table:
        for name in names:
            values[name].add(row[name])
    for name in names:
        for value in sorted(values[name], key=repr):
            yield (name, value, "single")
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            for a in sorted(values[first], key=repr):
                for b in sorted(values[second], key=repr):
                    yield (f"{first}&{second}", (a, b), "pair")


def search(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_fn: Any = None,
    label_fn: Any = None,
) -> dict[str, Any]:
    """Find every cheap rule that predicts a minority class too precisely.

    *feature_fn* and *label_fn* default to this module's V11 pair. They are parameters so a later
    contract can bring its own cheap features to the same search rather than copying it — the search
    is the part worth having exactly once, and a second copy would drift out of step with this one.
    """

    feature_fn = feature_fn or features
    label_fn = label_fn or _label
    table = [(feature_fn(row), label_fn(row)) for row in rows]
    total = len(table)
    counts: dict[str, int] = {}
    for _, label in table:
        counts[label] = counts.get(label, 0) + 1
    base = {label: count / total for label, count in counts.items()}

    findings: list[dict[str, Any]] = []
    for name, value, arity in _rules(table):
        if arity == "single":
            hits = [label for feats, label in table if feats[name] == value]
        else:
            first, second = name.split("&")
            hits = [label for feats, label in table if (feats[first], feats[second]) == value]
        if len(hits) < MIN_SUPPORT:
            continue
        best_label = max(set(hits), key=hits.count)
        if base[best_label] > BASE_RATE_CEILING:
            continue
        precision = hits.count(best_label) / len(hits)
        recall = hits.count(best_label) / counts[best_label]
        if precision >= PRECISION_LIMIT and recall >= RECALL_FLOOR:
            findings.append(
                {
                    "rule": f"{name} == {value!r}",
                    "predicts": best_label,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "support": len(hits),
                    "base_rate": round(base[best_label], 4),
                    "lift": round(precision / base[best_label], 1),
                }
            )

    # Deduplicate: a pair rule that merely repeats a single-feature rule is noise, not a second
    # finding. Keep the shortest rule for each (predicted class, recall) pair.
    findings.sort(key=lambda f: (-f["precision"], -f["recall"], len(f["rule"])))
    seen: set[tuple[str, float]] = set()
    unique: list[dict[str, Any]] = []
    for finding in findings:
        key = (finding["predicts"], finding["recall"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)

    return {
        "rows": total,
        "classes": {label: round(rate, 4) for label, rate in sorted(base.items())},
        "thresholds": {
            "precision": PRECISION_LIMIT,
            "recall": RECALL_FLOOR,
            "base_rate_ceiling": BASE_RATE_CEILING,
            "min_support": MIN_SUPPORT,
        },
        "shortcuts": unique,
    }


#: Residuals that are *accepted*, each with the reason. A rule listed here still gets measured and
#: still appears in the report; it simply does not fail the build.
#:
#: The bar for adding an entry is high, and "I could not get rid of it" is not the bar. The
#: justification has to be that the cheap feature is a proxy for the *correct* inference rather than
#: a way around it. Adding one for any other reason recreates exactly the failure this module
#: exists to catch, so each entry names the class and carries a recall ceiling: if the correlation
#: grows beyond what was reviewed, the build fails again.
ACCEPTED_RESIDUALS: tuple[dict[str, Any], ...] = (
    # Deliberately empty.
    #
    # It held one entry — `goal_inner_capital_count & goal_word_count_band -> needs_host_lane`,
    # recall 0.257 — accepted on the argument that host-lane goals are genuinely different
    # sentences, so a surface proxy for "this asks for a capability the device lacks" is a proxy
    # for the *right* inference rather than a bypass.
    #
    # That argument did not survive contact with the trained model. Given
    # "Could you open colonnade for me — this is the whole task" — a lowercase destination plus a
    # length filler the host-lane generator also uses — it emitted `handoff(needs_host_lane)`. The
    # model was using the residual as a *reverse router*, sending device goals to the host lane.
    # A residual the model demonstrably exploits is not a residual; it is a shortcut with
    # paperwork.
    #
    # The mechanism stays because the argument for *declaring* residuals stays: if one is ever
    # genuinely warranted it must be named here with a reason and a recall ceiling, so a new leak
    # cannot hide behind an old exemption. The bar is that the cheap feature proxies the correct
    # inference **and** that a trained model is shown not to exploit it in reverse. The second half
    # is what was missing.
)


def _accepted(finding: Mapping[str, Any]) -> dict[str, Any] | None:
    """The declared residual covering *finding*, if any."""

    rule_features = frozenset(finding["rule"].split(" == ")[0].split("&"))
    for residual in ACCEPTED_RESIDUALS:
        if (
            finding["predicts"] == residual["predicts"]
            and rule_features <= residual["features"]
            and finding["recall"] <= residual["max_recall"]
        ):
            return dict(residual)
    return None


def check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Raise when a cheap rule is sufficient for a minority class and is not a declared residual."""

    report = search(rows)
    blocking: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for finding in report["shortcuts"]:
        residual = _accepted(finding)
        if residual is None:
            blocking.append(finding)
        else:
            accepted.append({**finding, "accepted_because": residual["reason"]})
    report["shortcuts"] = blocking
    report["accepted_residuals"] = accepted
    if blocking:
        lines = [
            f"  - {item['rule']}  ->  {item['predicts']}"
            f"  (precision {item['precision']:.3f}, recall {item['recall']:.3f},"
            f" lift {item['lift']}x)"
            for item in blocking[:12]
        ]
        raise ValueError(
            f"{len(blocking)} cheap rule(s) predict the label without reading the goal:\n"
            + "\n".join(lines)
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="a *.jsonl split")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    if args.limit and len(rows) > args.limit:
        stride = max(1, len(rows) // args.limit)
        rows = rows[::stride][: args.limit]
    report = search(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    return 1 if report["shortcuts"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
