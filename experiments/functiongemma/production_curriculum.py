"""Leak-audited, production-shaped semantic-choice augmentation for v4.

The v1-v3 curriculum in :mod:`experiments.functiongemma.curriculum` remains frozen.
This module adds a narrow family that mirrors the packaged ``PolicyContext``
serializer: every offered call is a safe, current ``tap_and_analyze`` candidate and
the correct choice is the control named by the goal.  It never reads a journal,
device, emulator, ADB state, or AUA memory.

The reusable production smoke is a final gate, not training data.  Its four labels
are an explicit reserved vocabulary.  Generated train/validation/test semantic
labels use split-specific vocabularies, and generation fails closed if a literal
payload, semantic candidate set, entity label, or reserved smoke label leaks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.functiongemma.curriculum import (
    DEFAULT_DENYLIST,
    DEFAULT_SEED,
    SELECT_CANDIDATE_TOOL,
    _atomic_write,
    _canonical_json,
    _stable_seed,
    build_dataset,
    dataset_statistics,
    privacy_violations,
)

PRODUCTION_SEED = 20_260_815
PRODUCTION_CARDINALITIES = (2, 3, 4)
PRODUCTION_GROUP_COUNTS: dict[str, dict[int, int]] = {
    # Four-candidate rows dominate because the packaged v3 rollout contract is
    # exact-four.  Two and three candidates retain future cardinality coverage.
    "train": {2: 128, 3: 128, 4: 384},
    "valid": {2: 16, 3: 16, 4: 32},
    "test": {2: 16, 3: 16, 4: 32},
}
PRODUCTION_FIXTURE_REF = "aua-live-policy-v1"
PRODUCTION_REQUEST = "Choose the next action that advances the goal with direct proof."
PRODUCTION_DEVELOPER = (
    "You are a model that can do function calling with the following functions. "
    "You are an AUA policy selector for Android UI testing. Select exactly one "
    "supplied candidate. Candidate IDs are opaque and their order is arbitrary. "
    "Prefer direct semantic proof, current observations, bounded waits, and "
    "required cleanup. Never invent or rewrite a call."
)

# This public, fictional vocabulary is reserved exclusively for the final reusable
# smoke.  It is intentionally present only in this generator guard and the smoke
# source, never in emitted learning JSONL.
RESERVED_SMOKE_LABELS = frozenset({"Grammar", "Mathematics", "History", "Physics"})

_SPLIT_WORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "train": (
        (
            "Amber",
            "Brisk",
            "Calm",
            "Daring",
            "Eager",
            "Faint",
            "Golden",
            "Humble",
            "Ivory",
            "Jolly",
            "Keen",
            "Lively",
            "Mellow",
            "Nimble",
            "Opal",
            "Proud",
            "Quiet",
            "Rapid",
            "Silver",
            "Tender",
            "Umber",
            "Vivid",
            "Warm",
            "Young",
            "Zesty",
            "Copper",
            "Dusky",
            "Frosty",
            "Glowing",
            "Hardy",
            "Lucid",
            "Merry",
        ),
        (
            "Acorn",
            "Beacon",
            "Cedar",
            "Dune",
            "Elm",
            "Finch",
            "Grove",
            "Harbor",
            "Iris",
            "Juniper",
            "Kestrel",
            "Lantern",
            "Meadow",
            "Nectar",
            "Orchid",
            "Pebble",
            "Quartz",
            "Robin",
            "Summit",
            "Thistle",
            "Urchin",
            "Valley",
            "Walnut",
            "Yarrow",
            "Zephyr",
            "Brook",
            "Comet",
            "Drift",
            "Fern",
            "Island",
            "Jasper",
            "Lagoon",
        ),
    ),
    "valid": (
        (
            "Azure",
            "Bold",
            "Crisp",
            "Deep",
            "Earnest",
            "Floral",
            "Grand",
            "Honest",
            "Indigo",
            "Jaunty",
            "Kind",
            "Light",
        ),
        (
            "Almond",
            "Birch",
            "Coral",
            "Daisy",
            "Ember",
            "Falcon",
            "Garnet",
            "Hazel",
            "Inlet",
            "Lotus",
            "Maple",
            "Olive",
        ),
    ),
    "test": (
        (
            "Agile",
            "Blush",
            "Clear",
            "Delicate",
            "Even",
            "Glossy",
            "Hushed",
            "Icy",
            "Joyful",
            "Level",
            "Minty",
            "Neat",
        ),
        (
            "Apricot",
            "Bamboo",
            "Clover",
            "Delta",
            "Eagle",
            "Fjord",
            "Glacier",
            "Heather",
            "Lilac",
            "Marigold",
            "Otter",
            "Pine",
        ),
    ),
}

_RID_LABEL = re.compile(r"^open([A-Z][A-Za-z]+)$")


def _label_pool(split: str) -> tuple[str, ...]:
    modifiers, nouns = _SPLIT_WORDS[split]
    return tuple(f"{modifier} {noun}" for modifier in modifiers for noun in nouns)


def _rid(label: str) -> str:
    return "open" + label.replace(" ", "")


def _label_from_candidate(candidate: Mapping[str, Any]) -> str:
    rid = str(candidate["call"]["arguments"]["rid"])
    match = _RID_LABEL.fullmatch(rid)
    if match is None:
        raise ValueError(f"production candidate has a noncanonical rid: {rid!r}")
    compact = match.group(1)
    # Generated labels always contain two title-cased words.  The purpose carries
    # the spaced form, avoiding an ambiguous CamelCase reverse conversion.
    purpose = str(candidate["purpose"])
    prefix = "Tap the current-frame '"
    suffix = "' control and observe the result."
    if not purpose.startswith(prefix) or not purpose.endswith(suffix):
        raise ValueError("production candidate purpose does not expose its semantic label")
    label = purpose[len(prefix) : -len(suffix)]
    if _rid(label) != f"open{compact}":
        raise ValueError("production candidate rid and purpose label disagree")
    return label


def _candidate(label: str, candidate_id: int) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "call": {
            "tool": "tap_and_analyze",
            "arguments": {"rid": _rid(label)},
        },
        "purpose": f"Tap the current-frame '{label}' control and observe the result.",
        "risk": "safe",
        "authorized": True,
        "redundant": False,
        "proof": "The exact call returns a folded post-action observation.",
        "cleanup": "none",
    }


def _semantic_groups(
    split: str,
    group_counts: Mapping[int, int],
    *,
    seed: int,
) -> list[tuple[int, int, tuple[str, ...], str]]:
    """Return unique ``(cardinality, ordinal, labels, target)`` group specifications."""

    pool = _label_pool(split)
    rng = random.Random(_stable_seed("production-semantic-groups-v4", seed, split))
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[tuple[int, int, tuple[str, ...], str]] = []
    for cardinality in PRODUCTION_CARDINALITIES:
        count = int(group_counts.get(cardinality, 0))
        if count < 0:
            raise ValueError("production group counts cannot be negative")
        for ordinal in range(count):
            for _attempt in range(10_000):
                labels = tuple(rng.sample(pool, cardinality))
                target = labels[rng.randrange(cardinality)]
                signature = (target, tuple(sorted(labels)))
                if signature not in seen:
                    seen.add(signature)
                    result.append((cardinality, ordinal, labels, target))
                    break
            else:  # pragma: no cover - impossible with the configured combinatorial pools
                raise ValueError(f"could not construct a unique {split} semantic group")
    return result


def _ordered_candidates(
    labels: Sequence[str],
    target: str,
    variant: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Counterbalance target ID and target list position for one semantic state."""

    cardinality = len(labels)
    if variant not in range(cardinality * cardinality):
        raise ValueError("production variant is outside its cardinality square")
    target_id, target_position = divmod(variant, cardinality)
    distractors = [label for label in labels if label != target]
    rotation = (target_id + target_position) % len(distractors) if distractors else 0
    distractors = distractors[rotation:] + distractors[:rotation]
    ordered_labels = list(distractors)
    ordered_labels.insert(target_position, target)

    remaining_ids = [value for value in range(cardinality) if value != target_id]
    if remaining_ids:
        id_rotation = target_position % len(remaining_ids)
        remaining_ids = remaining_ids[id_rotation:] + remaining_ids[:id_rotation]
    id_cursor = iter(remaining_ids)
    candidates = [
        _candidate(label, target_id if label == target else next(id_cursor))
        for label in ordered_labels
    ]
    return candidates, target_id, target_position


def _build_production_record(
    *,
    split: str,
    cardinality: int,
    ordinal: int,
    labels: Sequence[str],
    target: str,
    variant: int,
) -> dict[str, Any]:
    candidates, target_id, target_position = _ordered_candidates(labels, target, variant)
    group_id = f"production-{split}-c{cardinality}-{ordinal:04d}"
    goal = f"Tap {target} and prove the settled result."
    state = {
        "fixture_ref": PRODUCTION_FIXTURE_REF,
        "request": PRODUCTION_REQUEST,
        "goal": goal,
        "phase": f"choice_step_{ordinal % 3 + 1}",
        "observation": {
            "fresh": True,
            "known_screen": f"fixture_{split}_choice_panel",
            "outcome": "known",
            "element_count": cardinality,
        },
        "recent_outcomes": [
            "session_started=true",
            "fresh_observation=true",
            "goal_checkpoint_reached=false",
        ],
        "constraints": [
            "Use fresh semantic evidence",
            "Select exactly one supplied candidate",
            "Require direct proof",
        ],
        "candidates": candidates,
    }
    target_call = {
        "tool": "tap_and_analyze",
        "arguments": {"rid": _rid(target)},
    }
    return {
        "id": f"fg-v4-{split}-c{cardinality}-{ordinal:04d}-v{variant:02d}",
        "messages": [
            {"role": "developer", "content": PRODUCTION_DEVELOPER},
            {"role": "user", "content": _canonical_json(state)},
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
            },
        ],
        "tools": [copy.deepcopy(SELECT_CANDIDATE_TOOL)],
        "metadata": {
            "split": split,
            "group_id": group_id,
            "episode_id": f"episode-{group_id}",
            "step": 0,
            "variant": variant,
            "intent": goal,
            "family": f"production_semantic_tap_{cardinality}",
            "label": f"production_semantic_tap_{cardinality}",
            "scenario_kind": "semantic_choice",
            "criticality": "normal",
            "template_profile": "production_policy_context_v4",
            "target_candidate_id": target_id,
            "target_position": target_position,
            "target_call": target_call,
            "tool_name": "tap_and_analyze",
            "curriculum_version": "v4",
        },
    }


def build_production_rows(
    group_counts: Mapping[str, Mapping[int, int]] | None = None,
    *,
    seed: int = PRODUCTION_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Build only the production-shaped v4 augmentation."""

    counts = PRODUCTION_GROUP_COUNTS if group_counts is None else group_counts
    if tuple(counts) != ("train", "valid", "test"):
        raise ValueError("group_counts must contain train, valid, and test in that order")
    result: dict[str, list[dict[str, Any]]] = {}
    for split in counts:
        rows: list[dict[str, Any]] = []
        for cardinality, ordinal, labels, target in _semantic_groups(
            split, counts[split], seed=seed
        ):
            for variant in range(cardinality * cardinality):
                record = _build_production_record(
                    split=split,
                    cardinality=cardinality,
                    ordinal=ordinal,
                    labels=labels,
                    target=target,
                    variant=variant,
                )
                findings = privacy_violations(record, denylist=denylist)
                if findings:
                    raise ValueError(
                        f"privacy audit failed for {record['id']}: " + "; ".join(findings)
                    )
                rows.append(record)
        random.Random(_stable_seed("production-row-order-v4", seed, split)).shuffle(rows)
        result[split] = rows
    audit_production_boundaries(result)
    return result


def _row_semantics(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    state = json.loads(str(row["messages"][1]["content"]))
    candidates = state["candidates"]
    labels = tuple(sorted(_label_from_candidate(candidate) for candidate in candidates))
    target_id = int(row["metadata"]["target_candidate_id"])
    target = _label_from_candidate(next(item for item in candidates if item["id"] == target_id))
    return target, labels


def audit_production_boundaries(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed on literal, semantic, entity, or smoke-fixture leakage."""

    payload_owner: dict[str, str] = {}
    semantic_owner: dict[tuple[str, tuple[str, ...]], str] = {}
    entity_owner: dict[str, str] = {}
    reserved = {label.casefold() for label in RESERVED_SMOKE_LABELS}
    counts: dict[str, dict[str, Any]] = {}

    for split in ("train", "valid", "test"):
        rows = dataset.get(split)
        if rows is None:
            raise ValueError(f"production dataset is missing split {split}")
        cardinalities: Counter[int] = Counter()
        groups: set[str] = set()
        for row in rows:
            messages = row["messages"]
            prompt_payload = _canonical_json({"messages": messages, "tools": row["tools"]})
            prior_payload_split = payload_owner.setdefault(prompt_payload, split)
            if prior_payload_split != split:
                raise ValueError("literal learning payload overlaps production splits")

            target, labels = _row_semantics(row)
            semantic = (target, labels)
            prior_semantic_split = semantic_owner.setdefault(semantic, split)
            if prior_semantic_split != split:
                raise ValueError("semantic target/candidate group overlaps production splits")
            for label in labels:
                folded = label.casefold()
                if folded in reserved:
                    raise ValueError(f"reserved production-smoke label leaked into {split}")
                prior_entity_split = entity_owner.setdefault(folded, split)
                if prior_entity_split != split:
                    raise ValueError("semantic entity label overlaps production splits")

            state = json.loads(str(messages[1]["content"]))
            cardinality = len(state["candidates"])
            if cardinality not in PRODUCTION_CARDINALITIES:
                raise ValueError(f"unsupported production cardinality: {cardinality}")
            expected_keys = {
                "fixture_ref",
                "request",
                "goal",
                "phase",
                "observation",
                "recent_outcomes",
                "constraints",
                "candidates",
            }
            if set(state) != expected_keys or state["fixture_ref"] != PRODUCTION_FIXTURE_REF:
                raise ValueError("production row does not match the packaged context shape")
            ids = {candidate["id"] for candidate in state["candidates"]}
            if ids != set(range(cardinality)):
                raise ValueError("production candidate IDs must be dense and zero-based")
            candidate_keys = {
                "id",
                "call",
                "purpose",
                "risk",
                "authorized",
                "redundant",
                "proof",
                "cleanup",
            }
            if any(set(candidate) != candidate_keys for candidate in state["candidates"]):
                raise ValueError("production candidate does not match PolicyCandidate model shape")
            cardinalities[cardinality] += 1
            groups.add(str(row["metadata"]["group_id"]))
        counts[split] = {
            "rows": len(rows),
            "groups": len(groups),
            "cardinalities": dict(sorted(cardinalities.items())),
        }

    return {
        "passed": True,
        "literal_payloads_disjoint": True,
        "semantic_groups_disjoint": True,
        "entity_labels_disjoint": True,
        "reserved_smoke_labels_absent_from_learning_rows": True,
        "reserved_smoke_semantic_case_absent": True,
        "splits": counts,
    }


def build_v4_dataset(
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    production_group_counts: Mapping[str, Mapping[int, int]] | None = None,
    seed: int = DEFAULT_SEED,
    production_seed: int = PRODUCTION_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Return frozen v3 rows plus the leak-audited v4 production augmentation."""

    legacy = build_dataset(base_split_sizes, seed=seed, denylist=denylist)
    production = build_production_rows(
        production_group_counts,
        seed=production_seed,
        denylist=denylist,
    )
    dataset: dict[str, list[dict[str, Any]]] = {}
    seen_payloads: set[str] = set()
    seen_groups: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        rows = [*legacy[split], *production[split]]
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            if payload in seen_payloads:
                raise ValueError(f"duplicate v4 learning payload: {row['id']}")
            seen_payloads.add(payload)
            group_id = str(row["metadata"]["group_id"])
            owner = seen_groups.setdefault(group_id, split)
            if owner != split:
                raise ValueError(f"semantic group {group_id} crosses {owner} and {split}")
        random.Random(_stable_seed("combined-row-order-v4", production_seed, split)).shuffle(rows)
        dataset[split] = rows
    return dataset


def _split_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode("utf-8")


def write_v4_dataset(
    output_dir: str | Path,
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    production_group_counts: Mapping[str, Mapping[int, int]] | None = None,
    seed: int = DEFAULT_SEED,
    production_seed: int = PRODUCTION_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, Any]:
    """Write the combined v4 corpus and its deterministic provenance manifest."""

    output = Path(output_dir)
    legacy = build_dataset(base_split_sizes, seed=seed, denylist=denylist)
    production = build_production_rows(
        production_group_counts,
        seed=production_seed,
        denylist=denylist,
    )
    dataset = build_v4_dataset(
        base_split_sizes,
        production_group_counts=production_group_counts,
        seed=seed,
        production_seed=production_seed,
        denylist=denylist,
    )
    statistics = dataset_statistics(dataset)
    leakage = audit_production_boundaries(production)
    entries: dict[str, dict[str, Any]] = {}
    legacy_hashes: dict[str, str] = {}

    for split in ("train", "valid", "test"):
        payload = _split_payload(dataset[split])
        filename = f"{split}.jsonl"
        _atomic_write(output / filename, payload)
        legacy_payload = _split_payload(legacy[split])
        legacy_hashes[split] = hashlib.sha256(legacy_payload).hexdigest()
        split_stats = statistics["splits"][split]
        entries[split] = {
            "path": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "legacy_v3_records": len(legacy[split]),
            "production_v4_records": len(production[split]),
            **split_stats,
        }

    combined_hash = hashlib.sha256(
        "".join(entries[split]["sha256"] for split in ("train", "valid", "test")).encode()
    ).hexdigest()
    manifest = {
        "format": "functiongemma-aua-candidate-policy-v4",
        "seed": seed,
        "production_seed": production_seed,
        "selection_function": "select_candidate(candidate_id: integer)",
        "split_policy": (
            "frozen v3 semantic groups plus split-exclusive production semantic groups; "
            "all assignment precedes ID/order expansion"
        ),
        "base_v3": {
            "format": "functiongemma-aua-candidate-policy-v3",
            "split_sha256": legacy_hashes,
            "records": {split: len(legacy[split]) for split in legacy},
        },
        "production_v4": {
            "template_profile": "production_policy_context_v4",
            "cardinalities": list(PRODUCTION_CARDINALITIES),
            "variants_per_group": {
                str(cardinality): cardinality * cardinality
                for cardinality in PRODUCTION_CARDINALITIES
            },
            "leakage_audit": leakage,
        },
        "total_records": statistics["total_records"],
        "ratios": statistics["ratios"],
        "privacy": {
            "passed": True,
            "checks": [
                "denylist",
                "host paths and direct identifiers",
                "fictional package namespace",
                "high-entropy opaque tokens",
            ],
        },
        "dataset_sha256": combined_hash,
        "splits": entries,
    }
    _atomic_write(output / "manifest.json", f"{_canonical_json(manifest)}\n".encode())
    return manifest
