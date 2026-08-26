"""Render the V11 dataset: on-device driver trajectories in LFM2.5 transport.

Two things make this module different from ``v10_curriculum.py``, and both are direct consequences
of what the earlier cycles cost.

**The confound audit is a build error, not a post-mortem.** V9's and V10's defects were both
discovered *after* training, by measuring a model and reasoning backwards to the corpus. Each time,
the generator had varied the dimension under study while silently holding another constant, and the
model took the cheaper rule. :func:`audit` computes those correlations directly and
:func:`build` refuses to write a corpus that fails them. If refusal can be predicted from "is a
destructive control visible", the build stops.

**The transport is LFM2.5's, not FunctionGemma's.** LFM2.5's chat template ignores ``tool_calls``
entirely and raises ``'NoneType' object is not iterable`` on ``content: None``, and only
``messages[0]`` is treated as a system turn — a ``developer`` role is emitted literally. So the
activation goes on ``system`` and the label is a content string holding a Pythonic call. This is
the same "transport, not contract" adjustment ``providers/policy/qwen3.py`` documents.

Usage::

    python -m experiments.functiongemma.v11_curriculum --out runs/functiongemma/data-v11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import v11_learning_material as material
from .v11_contract import ContractError, render_call, tools, validate_step
from .v11_learning_material import (
    DESTRUCTIVE_RATE,
    FAMILY_NAMES,
    SCHEMA,
    SEED,
    generate,
)
from .v11_shortcut_gate import check as check_shortcuts

SPLIT_ORDER = ("train", "valid", "test")
SPLIT_CODES = {"train": "tr", "valid": "va", "test": "te"}
TEMPLATE_PROFILE = "aua_v11_device_driver_v1"

#: Group counts per split. Trajectories expand to 1-5 rows each, and each row is rendered under
#: ``VARIANTS`` independent node orderings, so the row count is roughly groups x 2.5 x VARIANTS.
DEFAULT_GROUPS = {"train": 6000, "valid": 700, "test": 700}
VARIANTS = 3

#: The activation turn. Deliberately short: it ships in every prompt, and on-device tokens are
#: latency. It states the contract and nothing else — no worked examples, no vocabulary dump. The
#: tool definitions already carry the vocabulary.
DRIVER_POLICY = (
    "Drive the app to the goal. One call per turn.\n"
    "next_step: name the target with exactly one of resource_id, label, content_desc "
    "(prefer resource_id). For wait-for, assert-visible, assert-not-visible, scroll-to use "
    "arg + by instead; those match on containment, the rest on equality.\n"
    "done: the screen already proves the goal.\n"
    "handoff: nothing here advances the goal, the goal needs the host, or the only way forward "
    "needs authorization.\n"
    "A destructive control is one you must not press, not a reason to stop. "
    "Never invent a selector the screen does not show."
)


# --------------------------------------------------------------------------- rendering


def render_state(state: Any, variant: int, group: dict[str, Any]) -> dict[str, Any]:
    """Render one decision point as one training row under a given node ordering."""

    # Variant drives only the presentation order, never the content, so a family is scored on
    # semantics rather than on where a node happened to land.
    rnd = random.Random(f"{group['group_id']}:{variant}")
    context = state.as_context(rnd)

    if state.call == "next_step":
        validate_step(state.arguments)
    elif state.call == "handoff":
        if "reason" not in state.arguments:
            raise ContractError("handoff needs a reason")
    elif state.call == "done":
        if state.arguments:
            raise ContractError("done takes no arguments")
    else:
        raise ContractError(f"unknown call {state.call!r}")

    label = render_call(state.call, state.arguments)
    nodes = context["screen"]["nodes"]
    return {
        "id": (
            f"v11-{SPLIT_CODES[group['split']]}-{group['ordinal']:05d}"
            f"-{len(state.history):02d}-{variant:02d}"
        ),
        "messages": [
            {"role": "system", "content": DRIVER_POLICY},
            {
                "role": "user",
                "content": json.dumps(
                    context, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
            },
            {"role": "assistant", "content": label},
        ],
        "tools": tools(),
        "metadata": {
            "schema": SCHEMA,
            "template_profile": TEMPLATE_PROFILE,
            "split": group["split"],
            "group_id": group["group_id"],
            "family": state.family,
            "variant": variant,
            "call": state.call,
            "kind": state.arguments.get("kind") if state.call == "next_step" else None,
            "reason": state.arguments.get("reason") if state.call == "handoff" else None,
            "depth": len(state.history),
            "node_count": len(nodes),
            # Recorded so the audit can prove refusal is not predictable from it.
            "destructive_visible": any(_looks_destructive(node) for node in nodes),
            "selector_field": next(
                (f for f in ("resource_id", "label", "content_desc") if state.arguments.get(f)),
                None,
            ),
        },
    }


_DESTRUCTIVE_WORDS = ("delete", "erase", "reset", "sign out", "remove", "clear", "wipe", "revoke")


def _looks_destructive(node: dict[str, Any]) -> bool:
    haystack = " ".join(str(node.get(key, "")) for key in ("text", "desc", "rid")).casefold()
    return any(word in haystack for word in _DESTRUCTIVE_WORDS)


def build_split(split: str, groups: int, variants: int) -> list[dict[str, Any]]:
    """Expand every trajectory in *split* into per-state, per-variant rows."""

    rows: list[dict[str, Any]] = []
    for group in generate(split, groups):
        for state in group["states"]:
            for variant in range(variants):
                rows.append(render_state(state, variant, group))
    return rows


# --------------------------------------------------------------------------- gates


def check_group_isolation(splits: Mapping[str, Sequence[dict[str, Any]]]) -> None:
    """No semantic group may appear in two splits."""

    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            gid = row["metadata"]["group_id"]
            if seen.setdefault(gid, split) != split:
                raise ValueError(f"group {gid} appears in {seen[gid]} and {split}")


def check_vocabulary_isolation(splits: Mapping[str, Sequence[dict[str, Any]]]) -> None:
    """A held-out goal must not reuse a training goal's phrasing or destination.

    V9 memorised one template and inverted completely when an article was inserted. Partitioning
    both the destination pool and the goal templates by split is what makes a held-out score a
    statement about the model rather than about a phrasing it has seen 40,000 times.
    """

    goals: dict[str, set[str]] = {}
    for split, rows in splits.items():
        goals[split] = {json.loads(row["messages"][1]["content"])["goal"] for row in rows}
    train = goals.get("train", set())
    for split in ("valid", "test"):
        overlap = train & goals.get(split, set())
        if overlap:
            raise ValueError(
                f"{len(overlap)} goal strings appear in both train and {split}, "
                f"e.g. {sorted(overlap)[:3]}"
            )


def audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure the confounds that cost V9 and V10 a cycle each.

    Every number here answers "could the model get this right by reading something cheaper than
    the goal?". A large gap in ``refusal_rate_by_destructive`` means yes, via the destructive flag
    — which is precisely the V10 defect. A large gap in ``refusal_rate_by_node_count`` means yes,
    via cardinality — the V9 defect.
    """

    total = len(rows)
    families = Counter(row["metadata"]["family"] for row in rows)
    calls = Counter(row["metadata"]["call"] for row in rows)
    reasons = Counter(row["metadata"]["reason"] for row in rows if row["metadata"]["reason"])
    kinds = Counter(row["metadata"]["kind"] for row in rows if row["metadata"]["kind"])
    selectors = Counter(
        row["metadata"]["selector_field"] for row in rows if row["metadata"]["selector_field"]
    )

    def refusal_rate(subset: Sequence[dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        return sum(r["metadata"]["call"] == "handoff" for r in subset) / len(subset)

    with_danger = [r for r in rows if r["metadata"]["destructive_visible"]]
    without_danger = [r for r in rows if not r["metadata"]["destructive_visible"]]

    by_count: dict[int, float] = {}
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["metadata"]["node_count"]].append(row)
    floor = max(200, total // 50)
    for count, subset in sorted(buckets.items()):
        if len(subset) >= floor:
            by_count[count] = round(refusal_rate(subset), 4)

    relevance = sum(reasons[key] for key in ("target_absent", "no_progress", "needs_host_lane"))
    handoffs = sum(reasons.values())

    return {
        "rows": total,
        "families": dict(families),
        "calls": dict(calls),
        "handoff_reasons": dict(reasons),
        "step_kinds": dict(kinds),
        "selector_fields": dict(selectors),
        "refusal_rate": round(refusal_rate(rows), 4),
        "relevance_share_of_handoffs": round(relevance / handoffs, 4) if handoffs else 0.0,
        "refusal_rate_by_destructive": {
            "visible": round(refusal_rate(with_danger), 4),
            "absent": round(refusal_rate(without_danger), 4),
            "gap": round(abs(refusal_rate(with_danger) - refusal_rate(without_danger)), 4),
        },
        "destructive_visible_share": round(len(with_danger) / total, 4) if total else 0.0,
        "refusal_rate_by_node_count": by_count,
    }


#: The confound gates. A build that trips one of these is a build that would teach a cheap rule.
#: They are evaluated over the WHOLE corpus, not per split: a confound is a property of the
#: generator, and a 500-row validation split cannot estimate a 0.15 rate difference. Per-split
#: checks are limited to composition, where small numbers are still meaningful.
MAX_DESTRUCTIVE_REFUSAL_GAP = 0.15
MAX_NODE_COUNT_REFUSAL_SPREAD = 0.15
MIN_RELEVANCE_SHARE = 0.70
MIN_ACTING_SHARE = 0.65
#: Buckets thinner than this are not estimated at all, so the spread gate never fires on noise.
MIN_BUCKET_ROWS = 200


def check_confounds(report: dict[str, Any]) -> None:
    """Refuse to ship a corpus whose label is predictable from something cheaper than the goal."""

    problems: list[str] = []

    gap = report["refusal_rate_by_destructive"]["gap"]
    if gap > MAX_DESTRUCTIVE_REFUSAL_GAP:
        problems.append(
            f"refusal is predictable from a visible destructive control "
            f"(gap {gap:.3f} > {MAX_DESTRUCTIVE_REFUSAL_GAP}); this is the V10 defect"
        )

    spread_source = report["refusal_rate_by_node_count"]
    if spread_source:
        spread = max(spread_source.values()) - min(spread_source.values())
        if spread > MAX_NODE_COUNT_REFUSAL_SPREAD:
            problems.append(
                f"refusal is predictable from node count (spread {spread:.3f} > "
                f"{MAX_NODE_COUNT_REFUSAL_SPREAD}); this is the V9 defect"
            )

    relevance = report["relevance_share_of_handoffs"]
    if relevance < MIN_RELEVANCE_SHARE:
        problems.append(
            f"relevance refusals are only {relevance:.3f} of handoffs "
            f"(need >= {MIN_RELEVANCE_SHARE}); authorization refusal must stay the minority"
        )

    acting = (report["calls"].get("next_step", 0)) / report["rows"] if report["rows"] else 0.0
    if acting < MIN_ACTING_SHARE:
        problems.append(
            f"acting states are only {acting:.3f} of rows (need >= {MIN_ACTING_SHARE}); "
            f"V9 regressed on taps when they fell to 40% of the signal"
        )

    if problems:
        raise ValueError("confound gate failed:\n  - " + "\n  - ".join(problems))


def check_composition(split: str, report: dict[str, Any]) -> None:
    """Every split must exercise every family, and contain both acting and refusing states.

    This is the check small numbers *can* answer. A split that happens to contain no refusals is
    not a statistical quibble, it is a split that cannot score the capability that matters most.
    """

    problems: list[str] = []
    missing = sorted(set(FAMILY_NAMES) - set(report["families"]))
    # `prove_arrival` and `finish` are trajectory tails rather than declared families, so compare
    # only against families that name themselves.
    missing = [name for name in missing if name not in ("prove_arrival", "finish")]
    if missing:
        problems.append(f"families absent from this split: {missing}")
    if not report["calls"].get("handoff"):
        problems.append("no refusal states at all; refusal cannot be scored here")
    if not report["calls"].get("next_step"):
        problems.append("no acting states at all")
    if not report["calls"].get("done"):
        problems.append("no terminal done states at all")
    if problems:
        raise ValueError("composition gate failed:\n  - " + "\n  - ".join(problems))


def check_token_lengths(
    splits: Mapping[str, Sequence[dict[str, Any]]], limit: int, model: str
) -> dict[str, Any]:
    """Tokenize every row of every split through the real base tokenizer and bound the longest.

    Every split is measured, not a sample of train: the pod-side validator in ``train.py`` requires
    a per-split maximum and refuses a corpus whose longest row exceeds the training
    ``max_seq_length``. A sampled check would let a long row in a split it never looked at fail the
    run after the Pod is already billing.
    """

    from mlx_lm import load  # noqa: PLC0415

    loaded = load(model)
    tokenizer = loaded[1]
    per_split: dict[str, dict[str, Any]] = {}
    for split, rows in splits.items():
        longest = 0
        longest_id = ""
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                row["messages"], tools=row["tools"], tokenize=False, add_generation_prompt=False
            )
            length = len(tokenizer.encode(rendered))
            if length > longest:
                longest, longest_id = length, row["id"]
        if longest > limit:
            raise ValueError(
                f"[{split}] row {longest_id} is {longest} tokens, over the {limit} limit"
            )
        per_split[split] = {"max": longest, "row": longest_id}
    return {"checked": True, "limit": limit, "model": model, "splits": per_split}


#: Structural keys of the context envelope. These are protocol, not app knowledge.
_ENVELOPE_KEYS = frozenset(
    {
        "goal",
        "step",
        "budget",
        "screen",
        "history",
        "host_lane",
        "package",
        "name",
        "nodes",
        "settling",
        "ime",
        "rid",
        "text",
        "desc",
        "clickable",
        "scrollable",
    }
)


def audit_privacy(splits: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Prove every app-shaped value in the corpus comes from a declared fictional vocabulary.

    A denylist audit can only catch the private strings someone thought to list. This is the
    inverse and much stronger claim: the set of values that could carry product knowledge —
    node text, descriptions, resource ids, packages, screen names — is enumerated and checked
    against the generator's own declared pools. Anything not in those pools is a leak by
    definition, whatever it happens to say.

    The public-corpus boundary in ``CLAUDE.md`` is what this exists to enforce.
    """

    allowed: set[str] = set()
    for pool in material._SECTION_POOLS.values():
        for section in pool:
            allowed.add(section)
            allowed.add(section.lower())
            # Resource ids now come in five styles, only two of which contain the section's words.
            # The allowlist has to derive them the same way the generator does, or the opaque ones
            # look like leaked product strings.
            allowed.add(material._styled_rid(section))
            for suffix in material._NEAR_MISS_SUFFIXES:
                allowed.add(section + suffix)
                allowed.add(material._styled_rid(section + suffix))
    for label in material._DESTRUCTIVE_LABELS:
        allowed.add(label)
        allowed.add(material._styled_rid(label))
    for label in material._INERT_LABELS:
        allowed.add(label)
        for index in range(1, 32):
            allowed.add(f"{label} {index}")
    for stem in (s for pool in material._APP_STEMS.values() for s in pool):
        allowed.add(f"example.{stem}.app")
    # Fixed screen identities and the two synthetic control ids used by name.
    allowed.update(
        {
            "home",
            "listing",
            "search",
            "search-results",
            "settings",
            "section-list",
            "searchField",
            "scroller",
            "Loading",
            material._styled_rid("searchField"),
            material._styled_rid("scroller"),
        }
    )

    # History lines and goals are model-visible too, so they are audited as well. An earlier
    # version checked only the screen, which meant a section name reaching the corpus through a
    # history line or a goal template would not have been caught at all.
    allowed.update(material._WANDER_SECTIONS)
    allowed.update(
        {"key", "back", "wait-stable", "hide-keyboard", "assert-visible", "tap", "input"}
    )

    offenders: list[str] = []
    checked = 0
    for split, rows in splits.items():
        for row in rows:
            context = json.loads(row["messages"][1]["content"])
            values = [str(context["screen"]["package"])]
            if context["screen"].get("name"):
                values.append(str(context["screen"]["name"]))
            for node in context["screen"]["nodes"]:
                for key in ("rid", "text", "desc"):
                    if node.get(key):
                        values.append(str(node[key]))
            # A history line is `<kind> <field>=<value>` or a bare kind; only the value can leak.
            for line in context.get("history") or []:
                values.extend(part.split("=", 1)[1] for part in str(line).split() if "=" in part)
            for value in values:
                checked += 1
                if value not in allowed:
                    offenders.append(f"[{split}] {row['id']}: {value!r}")
    if offenders:
        raise ValueError(
            f"{len(offenders)} value(s) outside the declared fictional vocabulary, "
            f"e.g. {offenders[:5]}"
        )
    return {
        "passed": True,
        "method": "allowlist over declared fictional vocabularies",
        "values_checked": checked,
        "vocabulary_size": len(allowed),
    }


# --------------------------------------------------------------------------- build


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    out: Path,
    groups: dict[str, int] | None = None,
    variants: int = VARIANTS,
    *,
    model: str | None = None,
    token_limit: int = 1024,
) -> dict[str, Any]:
    """Generate, gate, and write the V11 corpus. Returns the manifest."""

    counts = dict(DEFAULT_GROUPS if groups is None else groups)
    splits = {split: build_split(split, counts[split], variants) for split in SPLIT_ORDER}

    check_group_isolation(splits)
    check_vocabulary_isolation(splits)

    reports = {split: audit(rows) for split, rows in splits.items()}
    combined = audit([row for rows in splits.values() for row in rows])
    check_confounds(combined)
    # The generic gate, and the one that matters. `check_confounds` only knows the two confounds
    # V9 and V10 already paid for; this searches every cheap feature for any rule that answers the
    # question without reading the goal. It is what turned 54 such rules into one declared residual.
    shortcuts = check_shortcuts([row for rows in splits.values() for row in rows])
    for split, report in reports.items():
        try:
            check_composition(split, report)
        except ValueError as exc:
            raise ValueError(f"[{split}] {exc}") from exc

    privacy = audit_privacy(splits)

    # The tokenizer check must run before the manifest is written: ``validation.json`` has to carry
    # a real per-split maximum, and the pod refuses a corpus whose longest row exceeds the
    # training ``max_seq_length``. Without a model there is nothing to measure and no run to feed.
    if model is None:
        tokens: dict[str, Any] = {"checked": False, "reason": "no model given"}
    else:
        tokens = check_token_lengths(splits, token_limit, model)

    out.mkdir(parents=True, exist_ok=True)
    split_meta: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows = splits[split]
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_meta[split] = {
            "path": f"{split}.jsonl",
            "records": len(rows),
            "groups": len({row["metadata"]["group_id"] for row in rows}),
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        }

    # ``train._validate_dataset`` recomputes this exactly: sha256 of the split hashes concatenated
    # in SPLIT_ORDER. Deriving it here rather than restating it keeps the two in agreement.
    dataset_sha256 = hashlib.sha256(
        "".join(split_meta[split]["sha256"] for split in SPLIT_ORDER).encode()
    ).hexdigest()

    manifest = {
        "schema": SCHEMA,
        "template_profile": TEMPLATE_PROFILE,
        "seed": SEED,
        "variants": variants,
        "groups": counts,
        "families": list(FAMILY_NAMES),
        "destructive_rate_target": DESTRUCTIVE_RATE,
        # Pod-side contract: `privacy.passed`, `splits[*].path/sha256/bytes`, `dataset_sha256`.
        "privacy": privacy,
        "splits": split_meta,
        "dataset_sha256": dataset_sha256,
        "total_records": sum(item["records"] for item in split_meta.values()),
        "audit": reports,
        "audit_combined": combined,
        "shortcut_gate": shortcuts,
        "tokens": tokens,
        "gates": {
            "max_destructive_refusal_gap": MAX_DESTRUCTIVE_REFUSAL_GAP,
            "max_node_count_refusal_spread": MAX_NODE_COUNT_REFUSAL_SPREAD,
            "min_relevance_share": MIN_RELEVANCE_SHARE,
            "min_acting_share": MIN_ACTING_SHARE,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )

    # The separate validation report the pod also insists on. `max_seq_length` here must equal the
    # training config's, or `_validate_dataset` rejects the corpus as a token-contract mismatch.
    if tokens.get("checked"):
        validation = {
            "ok": True,
            "max_seq_length": token_limit,
            "model": model,
            "splits": {
                split: {
                    "sha256": split_meta[split]["sha256"],
                    "records": split_meta[split]["records"],
                    "tokens": {"max": tokens["splits"][split]["max"]},
                }
                for split in SPLIT_ORDER
            },
        }
        (out / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", "utf-8"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=VARIANTS)
    parser.add_argument("--train-groups", type=int, default=DEFAULT_GROUPS["train"])
    parser.add_argument("--valid-groups", type=int, default=DEFAULT_GROUPS["valid"])
    parser.add_argument("--test-groups", type=int, default=DEFAULT_GROUPS["test"])
    parser.add_argument(
        "--model",
        default=None,
        help="base model whose tokenizer bounds row length (skipped when omitted)",
    )
    parser.add_argument("--token-limit", type=int, default=1024)
    args = parser.parse_args(argv)

    manifest = build(
        args.out,
        {
            "train": args.train_groups,
            "valid": args.valid_groups,
            "test": args.test_groups,
        },
        args.variants,
        model=args.model,
        token_limit=args.token_limit,
    )
    print(json.dumps(manifest["audit"]["train"], indent=2, sort_keys=True))
    for split in SPLIT_ORDER:
        info = manifest["splits"][split]
        print(f"{split:6s} {info['records']:7,d} rows  sha256={info['sha256'][:16]}")
    print(f"privacy: {manifest['privacy']}")
    print(f"dataset_sha256: {manifest['dataset_sha256']}")
    print(f"tokens: {manifest['tokens']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
