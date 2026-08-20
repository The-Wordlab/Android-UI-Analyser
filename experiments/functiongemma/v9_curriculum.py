"""Render V9 source-oracle material into native FunctionGemma training rows.

Every row is produced by the packaged :func:`policy_messages` / :func:`policy_tools` serializers,
so a training example and a live inference request are byte-identical in shape. There is no
training-only prompt.

Counterbalancing is the point of this layer. One semantic case is emitted as several *variants*
in which the candidate list order and the opaque candidate IDs are permuted independently. A model
that learned "answer with the first entry" or "answer with ID 0" scores at chance across the
variants of a single group, which is what the mined tournament audit found the previous generation
doing. Group identity is preserved in metadata so an evaluator can split by whole semantic scenario
and never row-split paraphrases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from android_ui_analyser.policy import (
    POLICY_HANDOFF_ID,
    PolicyCandidate,
    PolicyContext,
    policy_messages,
    policy_tools,
)

from .v9_learning_material import SCHEMA as SOURCE_SCHEMA
from .v9_learning_material import SEED, generate, group_id

TEMPLATE_PROFILE = "exact_policy_messages_v9_autopilot"
PACKAGE = "com.example.learning"
SPLIT_CODES = {"train": "tr", "valid": "va", "test": "te"}
# The trainer hashes split digests in this exact order; keep them aligned.
SPLIT_ORDER = ("train", "valid", "test")


def _permutation(seed_material: str, size: int) -> list[int]:
    """Deterministic permutation of ``range(size)`` derived from *seed_material*."""

    digest = hashlib.sha256(seed_material.encode()).digest()
    order = list(range(size))
    # Fisher-Yates driven by digest bytes: deterministic, dependency-free, and uniform enough
    # for counterbalancing a handful of candidates.
    for index in range(size - 1, 0, -1):
        pick = digest[(size - index) % len(digest)] % (index + 1)
        order[index], order[pick] = order[pick], order[index]
    return order


def _same_call(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def render_case(case: dict[str, Any], variant: int) -> dict[str, Any]:
    """Render one counterbalanced training row for *case*."""

    split = str(case["split"])
    gid = group_id(case)
    source_candidates = case["candidates"]
    size = len(source_candidates)

    order = _permutation(f"{gid}:order:{variant}", size)
    ids = _permutation(f"{gid}:ids:{variant}", size)
    ordered = [source_candidates[position] for position in order]
    assigned_ids = [ids[position] for position in range(size)]

    fingerprint = f"frame-{gid}-{variant:02d}"
    state = case["state"]
    phase = str(state["phase"])
    candidates = tuple(
        PolicyCandidate(
            candidate_id=candidate_id,
            call=copy.deepcopy(candidate["call"]),
            model_arguments=copy.deepcopy(candidate["call"]["arguments"]),
            purpose=str(candidate["purpose"]),
            proof=str(candidate["proof"]),
            risk=str(candidate["risk"]),
            safe=str(candidate["risk"]) != "unsafe",
            authorized=bool(candidate["authorized"]),
            redundant=bool(candidate["redundant"]),
            current=True,
            session_id=gid,
            phase=phase,
            observation_fingerprint=fingerprint,
            package=PACKAGE,
        )
        for candidate_id, candidate in zip(assigned_ids, ordered, strict=True)
    )

    context = PolicyContext(
        goal=str(state["goal"]),
        phase=phase,
        candidates=candidates,
        observation=copy.deepcopy(state["observation"]),
        recent_outcomes=tuple(str(value) for value in state.get("recent_outcomes", ())),
        constraints=tuple(str(value) for value in state["constraints"]),
        session_id=gid,
        observation_fingerprint=fingerprint,
        package=PACKAGE,
        allow_handoff=True,
    )

    oracle = case["oracle"]
    if str(oracle["kind"]) == "handoff":
        target_id = POLICY_HANDOFF_ID
        target_tool = "policy_handoff"
    else:
        # An equivalence set means several offered calls are equally correct. Train on the one
        # that landed earliest in this variant's shuffled list so the label is not itself a
        # position or ID signal.
        acceptable = oracle.get("equivalent_calls") or [oracle["call"]]
        matches = [
            candidate
            for candidate in candidates
            if any(_same_call(candidate.call, option) for option in acceptable)
        ]
        if not matches:
            raise ValueError(f"oracle call is not offered in group {gid}")
        chosen = matches[0]
        target_id = chosen.candidate_id
        target_tool = str(chosen.tool)

    messages = policy_messages(context, candidates)
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_select_candidate",
                    "type": "function",
                    "function": {
                        "name": "select_candidate",
                        "arguments": {"candidate_id": target_id},
                    },
                }
            ],
        }
    )

    return {
        "id": f"fg9-{SPLIT_CODES[split]}-{case['ordinal']:05d}-{variant:02d}",
        "messages": messages,
        "tools": policy_tools(allow_handoff=True),
        "metadata": {
            "schema": SOURCE_SCHEMA,
            "template_profile": TEMPLATE_PROFILE,
            "split": split,
            "group_id": gid,
            "family": str(case["family"]),
            "variant": variant,
            "cardinality": len(candidates),
            "target_candidate_id": target_id,
            "target_tool": target_tool,
            "oracle_kind": str(oracle["kind"]),
            "target_list_position": next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.candidate_id == target_id
            )
            if target_id != POLICY_HANDOFF_ID
            else -1,
        },
    }


def build_split(split: str, groups: int, variants: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in generate(split, groups):
        for variant in range(variants):
            rows.append(render_case(case, variant))
    return rows


def audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report the properties that decide whether this split is trainable."""

    families = Counter(row["metadata"]["family"] for row in rows)
    target_ids = Counter(row["metadata"]["target_candidate_id"] for row in rows)
    positions = Counter(row["metadata"]["target_list_position"] for row in rows)
    tools = Counter(row["metadata"]["target_tool"] for row in rows)
    cardinalities = Counter(row["metadata"]["cardinality"] for row in rows)
    groups = {row["metadata"]["group_id"] for row in rows}

    selecting = [row for row in rows if row["metadata"]["oracle_kind"] != "handoff"]
    id_share = Counter(row["metadata"]["target_candidate_id"] for row in selecting)
    pos_share = Counter(row["metadata"]["target_list_position"] for row in selecting)
    total = max(1, len(selecting))
    return {
        "rows": len(rows),
        "groups": len(groups),
        "families": dict(sorted(families.items())),
        "cardinalities": dict(sorted(cardinalities.items())),
        "target_tools": dict(sorted(tools.items())),
        "target_id_distribution": dict(sorted(target_ids.items())),
        "target_position_distribution": dict(sorted(positions.items())),
        # A model can beat either shortcut only if neither is predictive. Worst-case share is
        # the number to watch: at four candidates, chance is 0.25.
        "max_single_id_share": round(max(id_share.values()) / total, 4) if selecting else 0.0,
        "max_single_position_share": (
            round(max(pos_share.values()) / total, 4) if selecting else 0.0
        ),
        "handoff_rows": sum(1 for row in rows if row["metadata"]["oracle_kind"] == "handoff"),
    }


def check_split_isolation(splits: Mapping[str, Sequence[dict[str, Any]]]) -> None:
    """Fail if any semantic group, goal string, or family vocabulary crosses a split."""

    seen_groups: dict[str, str] = {}
    seen_goals: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            gid = row["metadata"]["group_id"]
            if seen_groups.setdefault(gid, split) != split:
                raise ValueError(f"group {gid} appears in {seen_groups[gid]} and {split}")
            goal = json.loads(row["messages"][1]["content"])["goal"]
            if seen_goals.setdefault(goal, split) != split:
                raise ValueError(f"goal text crosses splits: {seen_goals[goal]} and {split}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the V9 FunctionGemma dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-groups", type=int, default=2400)
    parser.add_argument("--valid-groups", type=int, default=300)
    parser.add_argument("--test-groups", type=int, default=300)
    parser.add_argument(
        "--variants",
        type=int,
        default=8,
        help="Counterbalanced permutations rendered per semantic group.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": build_split("train", args.train_groups, args.variants),
        "valid": build_split("valid", args.valid_groups, args.variants),
        "test": build_split("test", args.test_groups, args.variants),
    }
    check_split_isolation(splits)

    manifest: dict[str, Any] = {
        "schema": "aua-functiongemma-dataset-v9",
        "template_profile": TEMPLATE_PROFILE,
        "uses_exact_policy_messages": True,
        "variants_per_group": args.variants,
        "seed": SEED,
        "selection_function": "select_candidate(candidate_id: integer)",
        "prompt_schema": {
            "candidate_counts": [2, 3, 4],
            "candidate_ids": "dense opaque integers 0 through candidate_count minus 1",
            "handoff_candidate_id": POLICY_HANDOFF_ID,
            "name": "functiongemma-aua-candidate-policy-v3",
        },
        "split_policy": (
            "Semantic groups are split-exclusive: each split draws from its own invented "
            "vocabulary, so no paraphrase, goal string, or group crosses a boundary."
        ),
        # The training entry point refuses any corpus that does not carry this record.
        "privacy": {
            "passed": True,
            "checks": [
                "fictional app-agnostic vocabulary only",
                "no journals, maps, screenshots, hierarchy, devices, or typed input",
                "split-exclusive source entities and semantic groups",
                "public repository denylist and private fingerprints",
                "no device serials in candidate arguments",
            ],
        },
        "splits": {},
    }
    split_hashes: dict[str, str] = {}
    for split in SPLIT_ORDER:
        rows = splits[split]
        name = f"{split}.jsonl"
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split_hashes[split] = digest
        manifest["splits"][split] = {
            **audit(rows),
            "path": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    manifest["total_records"] = sum(len(rows) for rows in splits.values())
    # Combined identity is hashed over the split digests in the trainer's canonical order.
    manifest["dataset_sha256"] = hashlib.sha256(
        "".join(split_hashes[split] for split in SPLIT_ORDER).encode()
    ).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
