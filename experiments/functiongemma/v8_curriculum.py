"""Native FunctionGemma v8 curriculum with an authenticated handoff outcome.

The frozen v7 corpus remains unchanged. This module appends fictional, split-exclusive
corrections derived from the no-map five-model audit, rendered through the exact packaged
``policy_messages`` serializer. Candidate ID ``-1`` is a non-executing return of control;
it is never an AUA call and is accepted only when the prompt explicitly authenticates it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.functiongemma.curriculum import (
    DEFAULT_DENYLIST,
    DEFAULT_SEED,
    _atomic_write,
    _canonical_json,
    dataset_statistics,
    privacy_violations,
)
from experiments.functiongemma.semantic_context_curriculum import build_v7_dataset
from experiments.functiongemma.v8_learning_material import FAMILIES, SEED, build_v8_source

from android_ui_analyser.policy import (
    POLICY_HANDOFF_ID,
    PolicyCandidate,
    PolicyContext,
    policy_messages,
    policy_tools,
)

V8_FORMAT = "functiongemma-aua-candidate-policy-v8"
V8_TEMPLATE_PROFILE = "exact_policy_messages_v8_handoff"
V8_HANDOFF_REASON = "no_supplied_candidate_advances_goal"


def _canonical_candidate(value: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "call": value["call"],
            "purpose": value["purpose"],
            "proof": value["proof"],
            "risk": value["risk"],
            "authorized": value["authorized"],
            "redundant": value["redundant"],
        }
    )


def _group_sources(split: str) -> list[Mapping[str, Any]]:
    rows = build_v8_source(seed=SEED)[split]
    by_group: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group_id = str(row["group_id"])
        current = by_group.get(group_id)
        if current is None or int(row["metadata"]["variant"]) < int(current["metadata"]["variant"]):
            by_group[group_id] = row
    return [by_group[group_id] for group_id in sorted(by_group)]


def _semantic_candidates(source: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in source["candidates"]:
        value = copy.deepcopy(dict(candidate))
        value.pop("id", None)
        value.pop("position", None)
        unique[_canonical_candidate(value)] = value
    return tuple(unique[key] for key in sorted(unique))


def _variant_assignment(
    semantic: Sequence[Mapping[str, Any]],
    oracle: Mapping[str, Any],
    variant: int,
) -> tuple[list[tuple[int, Mapping[str, Any]]], int, int]:
    cardinality = len(semantic)
    if variant not in range(cardinality * cardinality):
        raise ValueError("v8 native variant is outside its cardinality square")
    id_lane, position_lane = divmod(variant, cardinality)
    if oracle["kind"] == "handoff":
        order = list(range(cardinality))
        order = order[position_lane:] + order[:position_lane]
        assigned = [
            ((semantic_index + id_lane) % cardinality, semantic[semantic_index])
            for semantic_index in order
        ]
        return assigned, POLICY_HANDOFF_ID, POLICY_HANDOFF_ID

    target_key = _canonical_json(oracle["call"])
    target_index = next(
        (
            index
            for index, candidate in enumerate(semantic)
            if _canonical_json(candidate["call"]) == target_key
        ),
        None,
    )
    if target_index is None:
        raise ValueError("v8 select oracle is absent from its semantic candidates")
    distractors = [index for index in range(cardinality) if index != target_index]
    rotation = (id_lane + position_lane) % len(distractors) if distractors else 0
    distractors = distractors[rotation:] + distractors[:rotation]
    order = list(distractors)
    order.insert(position_lane, target_index)
    remaining_ids = [value for value in range(cardinality) if value != id_lane]
    if remaining_ids:
        shift = position_lane % len(remaining_ids)
        remaining_ids = remaining_ids[shift:] + remaining_ids[:shift]
    id_cursor = iter(remaining_ids)
    assigned = [
        (id_lane if semantic_index == target_index else next(id_cursor), semantic[semantic_index])
        for semantic_index in order
    ]
    return assigned, id_lane, position_lane


def _record(
    *,
    split: str,
    source: Mapping[str, Any],
    variant: int,
) -> dict[str, Any]:
    semantic = _semantic_candidates(source)
    assigned, target_id, target_position = _variant_assignment(
        semantic,
        source["oracle"],
        variant,
    )
    source_group = str(source["group_id"])
    source_ordinal = source_group.rsplit("-", 1)[-1]
    split_code = {"train": "tr", "valid": "va", "test": "te"}[split]
    group_id = f"v8n-{split_code}-{source_ordinal}"
    state = source["state"]
    phase = str(state["phase"])
    fingerprint = f"frame-{group_id}"
    package = "com.example.learning"
    candidates = tuple(
        PolicyCandidate(
            candidate_id=candidate_id,
            call=copy.deepcopy(candidate["call"]),
            model_arguments=copy.deepcopy(candidate["call"]["arguments"]),
            purpose=str(candidate["purpose"]),
            proof=str(candidate["proof"]),
            risk=str(candidate["risk"]),
            safe=True,
            authorized=bool(candidate["authorized"]),
            redundant=bool(candidate["redundant"]),
            current=True,
            session_id=group_id,
            phase=phase,
            observation_fingerprint=fingerprint,
            package=package,
        )
        for candidate_id, candidate in assigned
    )
    context = PolicyContext(
        goal=str(state["goal"]),
        phase=phase,
        candidates=candidates,
        observation=copy.deepcopy(state["observation"]),
        recent_outcomes=(
            "session_active=true",
            "outcome=known",
            "goal_checkpoint_reached=false",
        ),
        constraints=tuple(str(value) for value in state["constraints"]),
        session_id=group_id,
        observation_fingerprint=fingerprint,
        package=package,
        allow_handoff=True,
    )
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
    oracle_kind = str(source["oracle"]["kind"])
    target_call = None if oracle_kind == "handoff" else copy.deepcopy(source["oracle"]["call"])
    target_tool = (
        "policy_handoff"
        if oracle_kind == "handoff"
        else str(target_call["tool"] if isinstance(target_call, Mapping) else "unknown")
    )
    cardinality = len(candidates)
    return {
        "id": f"fg8-{split_code}-{source_ordinal}-{variant:02d}",
        "messages": messages,
        "tools": policy_tools(allow_handoff=True),
        "metadata": {
            "split": split,
            "group_id": group_id,
            "episode_id": f"ep-{group_id}",
            "step": 0,
            "case_id": f"{group_id}-v{variant:02d}",
            "variant": variant,
            "intent": str(state["goal"]),
            "family": str(source["family"]),
            "label": str(source["family"]),
            "scenario_kind": "v8_live_failure_correction",
            "criticality": (
                "critical"
                if source["family"] in {"target_absent_handoff", "proof_cleanup_recovery"}
                else "normal"
            ),
            "template_profile": V8_TEMPLATE_PROFILE,
            "target_candidate_id": target_id,
            "target_position": target_position,
            "target_call": target_call,
            "target_outcome": oracle_kind,
            "tool_name": target_tool,
            "cardinality": cardinality,
            "permutation_group": True,
            "permutations_total": cardinality * cardinality,
            "curriculum_version": "v8",
        },
    }


def build_v8_native_rows(
    *,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Render the v8 source-oracle layer into exact FunctionGemma rows."""

    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        rows: list[dict[str, Any]] = []
        for source in _group_sources(split):
            cardinality = len(_semantic_candidates(source))
            for variant in range(cardinality * cardinality):
                row = _record(split=split, source=source, variant=variant)
                findings = privacy_violations(row, denylist=denylist)
                if findings:
                    raise ValueError(f"privacy audit failed for {row['id']}: {'; '.join(findings)}")
                rows.append(row)
        random.Random(f"v8-native:{SEED}:{split}").shuffle(rows)
        result[split] = rows
    audit_v8_native_rows(result)
    return result


def audit_v8_native_rows(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed on schema drift, leakage, and opaque-ID/order shortcuts."""

    payload_owner: dict[str, str] = {}
    group_owner: dict[str, str] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    stats: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        rows = dataset.get(split)
        if rows is None:
            raise ValueError(f"v8 native dataset is missing split {split}")
        families: Counter[str] = Counter()
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            if payload_owner.setdefault(payload, split) != split:
                raise ValueError("v8 literal learning payload crosses splits")
            metadata = row["metadata"]
            group_id = str(metadata["group_id"])
            if group_owner.setdefault(group_id, split) != split:
                raise ValueError("v8 semantic group crosses splits")
            groups[group_id].append(row)
            families[str(metadata["family"])] += 1
            state = json.loads(str(row["messages"][1]["content"]))
            candidates = state["candidates"]
            cardinality = len(candidates)
            if cardinality not in {3, 4}:
                raise ValueError("v8 correction row has an unsupported cardinality")
            if {candidate["id"] for candidate in candidates} != set(range(cardinality)):
                raise ValueError("v8 candidate IDs are not dense")
            if state.get("handoff") != {
                "allowed": True,
                "candidate_id": POLICY_HANDOFF_ID,
                "reason": V8_HANDOFF_REASON,
            }:
                raise ValueError("v8 row lacks the authenticated handoff contract")
            target_id = int(metadata["target_candidate_id"])
            if metadata["target_outcome"] == "handoff":
                if target_id != POLICY_HANDOFF_ID:
                    raise ValueError("v8 handoff target did not use the reserved ID")
            elif target_id not in {candidate["id"] for candidate in candidates}:
                raise ValueError("v8 action target is not offered")
            reconstructed = tuple(
                PolicyCandidate(
                    candidate_id=int(candidate["id"]),
                    call=copy.deepcopy(candidate["call"]),
                    model_arguments=copy.deepcopy(candidate["call"]["arguments"]),
                    purpose=str(candidate["purpose"]),
                    proof=str(candidate["proof"]),
                    cleanup=str(candidate["cleanup"]),
                    risk=str(candidate["risk"]),
                    safe=True,
                    authorized=bool(candidate["authorized"]),
                    redundant=bool(candidate["redundant"]),
                )
                for candidate in candidates
            )
            reconstructed_context = PolicyContext(
                goal=str(state["goal"]),
                phase=str(state["phase"]),
                candidates=reconstructed,
                observation=copy.deepcopy(state["observation"]),
                recent_outcomes=tuple(str(value) for value in state["recent_outcomes"]),
                constraints=tuple(str(value) for value in state["constraints"][:-3]),
                allow_handoff=True,
            )
            if row["messages"][:2] != policy_messages(reconstructed_context, reconstructed):
                raise ValueError("v8 row drifted from packaged policy_messages")
            if row["tools"] != policy_tools(allow_handoff=True):
                raise ValueError("v8 row drifted from packaged policy_tools")
        stats[split] = {
            "rows": len(rows),
            "groups": len({str(row["metadata"]["group_id"]) for row in rows}),
            "families": dict(sorted(families.items())),
        }

    for group_id, rows in groups.items():
        cardinality = int(rows[0]["metadata"]["cardinality"])
        if len(rows) != cardinality * cardinality:
            raise ValueError(f"v8 group {group_id} lacks its counterbalanced variants")
        if rows[0]["metadata"]["target_outcome"] == "select":
            expected = Counter(dict.fromkeys(range(cardinality), cardinality))
            if Counter(int(row["metadata"]["target_candidate_id"]) for row in rows) != expected:
                raise ValueError(f"v8 group {group_id} leaks target ID")
            if Counter(int(row["metadata"]["target_position"]) for row in rows) != expected:
                raise ValueError(f"v8 group {group_id} leaks target position")
        else:
            semantic_positions: Counter[tuple[str, int]] = Counter()
            semantic_ids: Counter[tuple[str, int]] = Counter()
            for row in rows:
                state = json.loads(str(row["messages"][1]["content"]))
                for position, candidate in enumerate(state["candidates"]):
                    key = _canonical_json(candidate["call"])
                    semantic_positions[(key, position)] += 1
                    semantic_ids[(key, int(candidate["id"]))] += 1
            if set(semantic_positions.values()) != {cardinality}:
                raise ValueError(f"v8 handoff group {group_id} leaks candidate position")
            if set(semantic_ids.values()) != {cardinality}:
                raise ValueError(f"v8 handoff group {group_id} leaks candidate ID")
    return {
        "passed": True,
        "uses_exact_policy_messages": True,
        "handoff_candidate_id": POLICY_HANDOFF_ID,
        "literal_payloads_disjoint": True,
        "semantic_groups_disjoint": True,
        "splits": stats,
    }


def build_v8_dataset(
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Return frozen v7 foundations plus native handoff/correction rows."""

    base = build_v7_dataset(base_split_sizes, seed=seed, denylist=denylist)
    native = build_v8_native_rows(denylist=denylist)
    result: dict[str, list[dict[str, Any]]] = {}
    payloads: set[str] = set()
    groups: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        rows = [*base[split], *native[split]]
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            if payload in payloads:
                raise ValueError(f"duplicate v8 learning payload: {row['id']}")
            payloads.add(payload)
            group_id = str(row["metadata"]["group_id"])
            if groups.setdefault(group_id, split) != split:
                raise ValueError(f"v8 semantic group {group_id} crosses splits")
        random.Random(f"combined-v8:{SEED}:{split}").shuffle(rows)
        result[split] = rows
    return result


def _split_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode()


def write_v8_dataset(
    output_dir: str | Path,
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, Any]:
    """Write deterministic v8 JSONL and its leakage/protocol manifest."""

    output = Path(output_dir)
    base = build_v7_dataset(base_split_sizes, seed=seed, denylist=denylist)
    native = build_v8_native_rows(denylist=denylist)
    dataset = build_v8_dataset(base_split_sizes, seed=seed, denylist=denylist)
    statistics = dataset_statistics(dataset)
    audit = audit_v8_native_rows(native)
    entries: dict[str, dict[str, Any]] = {}
    base_hashes: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        payload = _split_payload(dataset[split])
        filename = f"{split}.jsonl"
        _atomic_write(output / filename, payload)
        base_hashes[split] = hashlib.sha256(_split_payload(base[split])).hexdigest()
        entries[split] = {
            "path": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "base_v7_records": len(base[split]),
            "native_v8_records": len(native[split]),
            **statistics["splits"][split],
        }
    combined_hash = hashlib.sha256(
        "".join(entries[split]["sha256"] for split in ("train", "valid", "test")).encode()
    ).hexdigest()
    manifest = {
        "format": V8_FORMAT,
        "seed": seed,
        "source_seed": SEED,
        "selection_function": "select_candidate(candidate_id: integer; -1 means handoff)",
        "prompt_schema": {
            "candidate_ids": "dense opaque integers 0 through candidate_count minus 1",
            "candidate_counts": [2, 3, 4],
            "handoff_candidate_id": POLICY_HANDOFF_ID,
        },
        "split_policy": (
            "frozen v7 foundations plus split-exclusive fictional v8 corrections; "
            "semantic assignment precedes dense ID and position counterbalancing"
        ),
        "base_v7": {
            "format": "functiongemma-aua-candidate-policy-v7",
            "split_sha256": base_hashes,
            "records": {split: len(base[split]) for split in base},
        },
        "native_v8": {
            "families": list(FAMILIES),
            "source_records": 1000,
            "rendered_records": sum(len(rows) for rows in native.values()),
            "techniques": [
                "exact packaged handoff-aware policy serializer",
                "non-executing authenticated -1 handoff outcome",
                "counterbalanced opaque IDs and list positions",
                "meta-control hard negatives and shared-token destinations",
                "two-hop, target-absent, unknown-outcome, and cleanup corrections",
            ],
            "leakage_and_invariance_audit": audit,
        },
        "total_records": statistics["total_records"],
        "ratios": statistics["ratios"],
        "privacy": {
            "passed": True,
            "checks": [
                "fictional app-agnostic vocabulary only",
                "no journals, maps, screenshots, hierarchy, devices, or typed input",
                "split-exclusive source entities and semantic groups",
                "public repository denylist and private fingerprints",
            ],
        },
        "dataset_sha256": combined_hash,
        "splits": entries,
    }
    _atomic_write(output / "manifest.json", f"{_canonical_json(manifest)}\n".encode())
    return manifest
