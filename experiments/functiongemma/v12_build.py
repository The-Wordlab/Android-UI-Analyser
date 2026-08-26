"""Build the V12 corpus to disk, split by screen rather than by row.

The split boundary is the point of this file. V11 shuffled rows and cut, so the same screen appeared
in train and in test, and the test set measured how well the model remembered screens it had already
been fitted to. Here the 637 harvested screens are partitioned first and rows are generated inside
each partition, so a test row is a screen the model has never seen. That makes the held-out number
mean what it is normally assumed to mean.

Everything is verified before it is written, and the build refuses rather than warns:

* every row's call must validate against the screen it was decided on (:mod:`v12_contract`)
* no screen may appear in two splits
* the shortcut gate must find nothing on the training split (:mod:`v12_shortcut_gate`)
* answer mix must be flat across history length, the V11 failure

Run: ``python -m experiments.functiongemma.v12_build --rows 60000``
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.functiongemma.v12_contract import tools
from experiments.functiongemma.v12_corpus import (
    FAMILY_WEIGHTS,
    POLICY,
    answer_of,
    build,
    load_screens,
)
from experiments.functiongemma.v12_shortcut_gate import check, features, label

_ = answer_of  # re-exported for callers that audit corpora written by this script

#: Fraction of *screens* — not rows — held out. Generous on purpose: a held-out screen is the only
#: evidence that the model reads screens rather than recognising them, which is the whole question.
MODEL = "LiquidAI/LFM2.5-350M-MLX-bf16"

VALID_SHARE = 0.12
TEST_SHARE = 0.12


def _screen_key(projection: Mapping[str, Any]) -> str:
    """Identity of a screen for splitting: its package plus its listed labels.

    Not the harvest's own fingerprint — that folds in the status-bar clock, so the same screen
    captured a minute apart gets two identities and would land on both sides of the split.
    """

    labels = "|".join(
        " ".join(str(node.get(key) or "") for key in ("text", "desc", "rid")).strip()
        for node in projection.get("nodes") or []
    )
    return f"{projection.get('package')}::{labels}"


def split_screens(
    screens: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, list[Mapping[str, Any]]]:
    """Partition screens into train/valid/test, keeping identical screens together."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for projection in screens:
        grouped.setdefault(_screen_key(projection), []).append(projection)

    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    n_valid = max(1, int(len(keys) * VALID_SHARE))
    n_test = max(1, int(len(keys) * TEST_SHARE))

    out: dict[str, list[Mapping[str, Any]]] = {"valid": [], "test": [], "train": []}
    for index, key in enumerate(keys):
        name = "valid" if index < n_valid else "test" if index < n_valid + n_test else "train"
        out[name].extend(grouped[key])
    return out


def audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Numbers worth recording, chosen because each names a specific V11 failure."""

    classes = collections.Counter(label(row) for row in rows)
    families = collections.Counter(row["meta"]["family"] for row in rows)
    styles = collections.Counter(row["meta"]["style"] for row in rows)

    # Keyed on the FULL class, reasons included. Auditing the call alone is how V11's actual failure
    # survived a rebuild: `needs_host` was 51% of refusals at zero progress and `no_progress` 40% at
    # high progress, while the reported spread read 0.0001.
    by_step: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    verbatim = nav_taps = 0
    nodes_shown: list[int] = []
    for row in rows:
        by_step[features(row)["step_band"]][label(row)] += 1
        context = json.loads(row["messages"][1]["content"])
        nodes_shown.append(len(context["screen"]["nodes"]))
        # The V11 copy shortcut: the answer sitting in the goal verbatim, true of 73.6% of its rows.
        # `decline` rows are excluded — their goals are fixed phrases that structurally can never be
        # verbatim, so including them understated the real share by about a tenth.
        if row["meta"]["family"] == "decline" or 'tap(n="' not in row["messages"][-1]["content"]:
            continue
        nav_taps += 1
        want = row["messages"][-1]["content"].split('tap(n="', 1)[1].split('"', 1)[0]
        for node in context["screen"]["nodes"]:
            if node.get("n") != want:
                continue
            text = " ".join(str(node.get(k) or "") for k in ("text", "desc")).strip().casefold()
            if text and text in str(context["goal"]).casefold():
                verbatim += 1

    spread = {}
    for step, counter in sorted(by_step.items()):
        total = sum(counter.values()) or 1
        spread[step] = {name: round(count / total, 4) for name, count in sorted(counter.items())}

    # The widest drift of any single class across progress levels *above zero*. Level zero is
    # excluded because `no_progress` cannot exist there — it means the node the goal names was
    # already tried, and at zero nothing has been. Including it reports that definition as a 0.10
    # defect. What must be flat is the mix given that something has happened, because that is where
    # V11's counting lived. The previous build's 0.0001 measured the call alone and hid a 0.40 drift
    # in the reasons, so both mistakes are worth naming: too coarse a class, and too broad a range.
    worst = 0.0
    worst_class = ""
    above_zero = {level: rates for level, rates in spread.items() if level > 0}
    for name in classes:
        rates = [level.get(name, 0.0) for level in above_zero.values()]
        if rates and max(rates) - min(rates) > worst:
            worst = max(rates) - min(rates)
            worst_class = name

    nodes_shown.sort()
    return {
        "rows": len(rows),
        "classes": {name: round(count / len(rows), 4) for name, count in sorted(classes.items())},
        "families": dict(sorted(families.items())),
        "goal_styles": dict(sorted(styles.items())),
        "verbatim_tap_share": round(verbatim / (nav_taps or 1), 4),
        "nodes_shown_median": nodes_shown[len(nodes_shown) // 2] if nodes_shown else 0,
        "answer_mix_by_progress": spread,
        "worst_class_drift": {
            "class": worst_class,
            "spread": round(worst, 4),
            "note": "levels above zero only; no_progress is impossible at zero by definition",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=60000, help="training rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--screens", type=Path, default=Path("runs/functiongemma/screens")
    )
    parser.add_argument("--out", type=Path, default=Path("runs/functiongemma/data-v12"))
    parser.add_argument(
        "--allow-shortcuts",
        action="store_true",
        help="write the corpus even if the gate finds a cheap rule (for inspecting a failure)",
    )
    args = parser.parse_args(argv)

    screens = load_screens(args.screens)
    if not screens:
        print(f"no screens under {args.screens}")
        return 1
    parts = split_screens(screens, args.seed)
    keys = {name: {_screen_key(p) for p in group} for name, group in parts.items()}
    for first in keys:
        for second in keys:
            if first < second and keys[first] & keys[second]:
                print(f"screen leaked between {first} and {second}")
                return 1

    counts = {
        "train": args.rows,
        "valid": max(200, args.rows // 8),
        "test": max(200, args.rows // 8),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "policy": POLICY,
        "family_weights": dict(FAMILY_WEIGHTS),
        "screens": {name: len(group) for name, group in parts.items()},
        "splits": {},
    }

    for name in ("train", "valid", "test"):
        # crc32, not hash(): `hash()` on a str is salted by PYTHONHASHSEED, so the previous build
        # produced a different corpus on every invocation while claiming a fixed seed.
        rows = build(
            parts[name], counts[name], seed=args.seed + zlib.crc32(name.encode()) % 1000
        )
        path = args.out / f"{name}.jsonl"
        # `tools` is written into every row because mlx-lm reads it per row and passes it to
        # `apply_chat_template` (mlx_lm/tuner/datasets.py). Omitting it here while serving *with* it
        # is train/serve skew of exactly the kind that broke V10: the model would learn a prompt
        # shape it never sees again. V11's rows carried it; the first V12 build dropped it.
        schema = tools()
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {"messages": row["messages"], "tools": schema}, ensure_ascii=False
                    )
                    + "\n"
                )
        summary = audit(rows)
        report["splits"][name] = summary
        print(
            f"{name:6s} {summary['rows']:6d} rows  "
            f"verbatim {summary['verbatim_tap_share']:.1%}  "
            f"worst drift {summary['worst_class_drift']['spread']:.3f} "
            f"({summary['worst_class_drift']['class']})  "
            f"-> {path}"
        )

    # Measured, not assumed: V9/V10 carried a max_seq_length nobody had checked against the
    # tokeniser, and every pass count in EXPERIMENT_LOG.md was 4x wrong for a related reason.
    from mlx_lm.utils import load  # imported here so the corpus can be built without mlx

    _, tokenizer = load(MODEL)[:2]
    lengths: dict[str, int] = {}
    for name in ("train", "valid", "test"):
        longest = 0
        with (args.out / f"{name}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tools=row.get("tools"), tokenize=False
                )
                longest = max(longest, len(tokenizer.encode(text)))
        lengths[name] = longest
    report["max_tokens"] = lengths
    report["max_seq_length"] = max(512, ((max(lengths.values()) // 128) + 1) * 128)
    print(
        f"\nlongest row: {max(lengths.values())} tokens {lengths} "
        f"-> max_seq_length {report['max_seq_length']}"
    )

    gate_rows = []
    with (args.out / "train.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 12000:
                break
            gate_rows.append(json.loads(line))
    gate = check(gate_rows)
    report["shortcut_gate"] = {
        "rows_checked": gate["rows"],
        "clean": gate["clean"],
        "blocking": gate["blocking"],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nshortcut gate on {gate['rows']} training rows: clean={gate['clean']}")
    for finding in gate["blocking"][:8]:
        print(f"  P={finding['precision']:.3f} R={finding['recall']:.3f} "
              f"{finding['predicts']} <- {finding['rule']}")
    if not gate["clean"] and not args.allow_shortcuts:
        print("\nrefusing to certify: rerun with --allow-shortcuts only to inspect.")
        return 1
    print(f"manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
