"""Exact-serializer, permutation-complete FunctionGemma curriculum for v6.

V5 learned recovery semantics but failed a live AUA invariance check: the same
four semantic controls and goal produced different answers when a fresh session
changed opaque IDs and list order.  This augmentation uses the packaged
``policy_messages`` serializer directly and expands every fictional semantic
state across all 24 candidate orders and all 24 dense-ID assignments.

The vocabulary is split-exclusive and fictional.  This module never reads an
AUA journal, map, device, emulator, screenshot, hierarchy, or private app trace.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Make the repository's src layout importable from a clean source archive.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.functiongemma.curriculum import (
    DEFAULT_DENYLIST,
    DEFAULT_SEED,
    _atomic_write,
    _canonical_json,
    _stable_seed,
    dataset_statistics,
    privacy_violations,
)
from experiments.functiongemma.production_curriculum import _label_pool
from experiments.functiongemma.recovery_curriculum import build_v5_dataset

from android_ui_analyser.policy import (  # noqa: E402
    PolicyCandidate,
    PolicyContext,
    policy_messages,
    policy_tools,
)

LIVE_CONTEXT_SEED = 20_260_818
LIVE_CONTEXT_GROUP_COUNTS = {"train": 40, "valid": 8, "test": 8}
LIVE_CONTEXT_CARDINALITY = 4
LIVE_CONTEXT_VARIANTS = 24 * 24
LIVE_CONTEXT_FAMILY = "live_ambiguous_semantic_tap"

# These labels exist only in the independent live-context smoke.  Keeping them
# out of learning rows makes the final serializer gate lexically held out.
RESERVED_LIVE_SMOKE_LABELS = frozenset(
    {"Aster Almanac", "Beryl Ledger", "Cinder Checklist", "Dovetail Toolkit"}
)

_SCREENS = {
    "train": (
        "Example Shelf",
        "Sample Console",
        "Fixture Library",
        "Demo Workspace",
        "Practice Board",
    ),
    "valid": ("Validation Gallery", "Reference Desk", "Trial Cabinet"),
    "test": ("Evaluation Atrium", "Verification Rack", "Testing Alcove"),
}

_SUBTITLES = {
    "train": (
        "tools, recent items, defaults",
        "collections, saved entries, preferences",
        "shortcuts, activity, configuration",
        "catalogs, history, options",
    ),
    "valid": (
        "utilities, latest entries, choices",
        "groups, stored items, controls",
        "links, recent work, setup",
        "indexes, activity, preferences",
    ),
    "test": (
        "features, newest items, settings",
        "folders, retained entries, options",
        "routes, prior work, configuration",
        "directories, history, controls",
    ),
}

_GOAL_TEMPLATES = (
    "Open {target} from {screen} among the visible {choices} choices.",
    "From {screen}, choose {target} rather than {alternatives}.",
    "Use the {target} destination on {screen}; the alternatives are {alternatives}.",
    "On {screen}, the available destinations are {choices}. Open {target}.",
)


def _choices(labels: Sequence[str]) -> str:
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _goal(labels: Sequence[str], target: str, screen: str, template_index: int) -> str:
    alternatives = [label for label in labels if label != target]
    return _GOAL_TEMPLATES[template_index % len(_GOAL_TEMPLATES)].format(
        target=target,
        screen=screen,
        choices=_choices(labels),
        alternatives=_choices(alternatives),
    )


def _selector(primary: str, full_label: str, selector_lane: int) -> dict[str, str]:
    if selector_lane in {0, 1}:
        return {"text": full_label}
    if selector_lane == 2:
        return {"rid": "open" + primary.replace(" ", "")}
    return {"desc": f"Open {primary}"}


def _group_specs(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    if split not in _SCREENS:
        raise ValueError(f"unsupported live-context split: {split}")
    if count <= 0:
        raise ValueError("live-context group counts must be positive")
    pool = _label_pool(split)
    rng = random.Random(_stable_seed("live-context-groups-v6", seed, split))
    seen: set[tuple[str, ...]] = set()
    groups: list[dict[str, Any]] = []
    for ordinal in range(count):
        for _attempt in range(10_000):
            labels = tuple(rng.sample(pool, LIVE_CONTEXT_CARDINALITY))
            signature = tuple(sorted(labels))
            if signature not in seen:
                seen.add(signature)
                break
        else:  # pragma: no cover - the configured pools are much larger than the request
            raise ValueError(f"could not construct unique live-context group for {split}")
        target = labels[ordinal % LIVE_CONTEXT_CARDINALITY]
        groups.append(
            {
                "ordinal": ordinal,
                "labels": labels,
                "target": target,
                "screen": _SCREENS[split][ordinal % len(_SCREENS[split])],
                "template_index": ordinal % len(_GOAL_TEMPLATES),
                "selector_lane": ordinal % 4,
            }
        )
    return groups


def _candidate(
    *,
    candidate_id: int,
    primary: str,
    full_label: str,
    selector_lane: int,
    group_id: str,
) -> PolicyCandidate:
    arguments = _selector(primary, full_label, selector_lane)
    return PolicyCandidate(
        candidate_id=candidate_id,
        call={"tool": "tap_and_analyze", "arguments": arguments},
        model_arguments=arguments,
        purpose=f"Tap the current-frame {full_label!r} control and observe the result.",
        proof="The exact call returns a folded post-action observation.",
        safe=True,
        authorized=True,
        redundant=False,
        current=True,
        session_id=group_id,
        phase="phase_1",
        observation_fingerprint=f"frame-{group_id}",
        package="com.example.learning",
    )


def _record(
    *,
    split: str,
    group: Mapping[str, Any],
    order: Sequence[int],
    id_assignment: Sequence[int],
    variant: int,
) -> dict[str, Any]:
    ordinal = int(group["ordinal"])
    labels = tuple(str(value) for value in group["labels"])
    target = str(group["target"])
    screen = str(group["screen"])
    group_id = f"live-{split}-{ordinal:03d}"
    subtitles = _SUBTITLES[split]
    full_labels = tuple(
        f"{label} {subtitles[(ordinal + index) % len(subtitles)]}"
        for index, label in enumerate(labels)
    )
    candidates_by_index = {
        index: _candidate(
            candidate_id=int(id_assignment[index]),
            primary=labels[index],
            full_label=full_labels[index],
            selector_lane=int(group["selector_lane"]),
            group_id=group_id,
        )
        for index in range(LIVE_CONTEXT_CARDINALITY)
    }
    candidates = tuple(candidates_by_index[index] for index in order)
    target_index = labels.index(target)
    target_candidate = candidates_by_index[target_index]
    goal = _goal(labels, target, screen, int(group["template_index"]))
    context = PolicyContext(
        goal=goal,
        phase="phase_1",
        candidates=candidates,
        observation={
            "fresh": True,
            "known_screen": f"fixture_{split}_choice_panel",
        },
        recent_outcomes=(
            "session_active=true",
            "outcome=known",
            "goal_checkpoint_reached=false",
        ),
        constraints=(
            "Select only a supplied guard-approved candidate.",
            "Do not invent or execute a call.",
        ),
        session_id=group_id,
        observation_fingerprint=f"frame-{group_id}",
        package="com.example.learning",
    )
    target_id = target_candidate.candidate_id
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
        "id": f"fg-v6-{split}-{ordinal:03d}-v{variant:03d}",
        "messages": messages,
        "tools": policy_tools(),
        "metadata": {
            "split": split,
            "group_id": group_id,
            "episode_id": f"episode-{group_id}",
            "step": 0,
            "case_id": f"{group_id}-v{variant:03d}",
            "variant": variant,
            "intent": goal,
            "family": LIVE_CONTEXT_FAMILY,
            "label": LIVE_CONTEXT_FAMILY,
            "scenario_kind": "production_live_context_permutation",
            "criticality": "normal",
            "template_profile": "exact_policy_messages_v6",
            "target_candidate_id": target_id,
            "target_position": tuple(order).index(target_index),
            "target_call": target_candidate.trusted_call(),
            "tool_name": "tap_and_analyze",
            "cardinality": LIVE_CONTEXT_CARDINALITY,
            "permutation_group": True,
            "permutations_total": LIVE_CONTEXT_VARIANTS,
            "candidate_order": list(order),
            "id_assignment": list(id_assignment),
            "selector_lane": int(group["selector_lane"]),
            "curriculum_version": "v6",
        },
    }


def build_live_context_rows(
    group_counts: Mapping[str, int] | None = None,
    *,
    seed: int = LIVE_CONTEXT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Build the v6-only exact-production-context augmentation."""

    counts = dict(LIVE_CONTEXT_GROUP_COUNTS if group_counts is None else group_counts)
    if tuple(counts) != ("train", "valid", "test"):
        raise ValueError("group_counts must contain train, valid, and test in order")
    orders = tuple(itertools.permutations(range(LIVE_CONTEXT_CARDINALITY)))
    assignments = tuple(itertools.permutations(range(LIVE_CONTEXT_CARDINALITY)))
    result: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        rows: list[dict[str, Any]] = []
        for group in _group_specs(split, int(count), seed):
            variant = 0
            for order in orders:
                for assignment in assignments:
                    row = _record(
                        split=split,
                        group=group,
                        order=order,
                        id_assignment=assignment,
                        variant=variant,
                    )
                    findings = privacy_violations(row, denylist=denylist)
                    if findings:
                        raise ValueError(
                            f"privacy audit failed for {row['id']}: {'; '.join(findings)}"
                        )
                    rows.append(row)
                    variant += 1
        random.Random(_stable_seed("live-context-row-order-v6", seed, split)).shuffle(rows)
        result[split] = rows
    audit_live_context_boundaries(result)
    return result


def audit_live_context_boundaries(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed on leakage, serializer drift, or incomplete permutations."""

    payload_owner: dict[str, str] = {}
    entity_owner: dict[str, str] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    stats: dict[str, Any] = {}
    reserved = {value.casefold() for value in RESERVED_LIVE_SMOKE_LABELS}
    expected_state_keys = {
        "fixture_ref",
        "request",
        "goal",
        "phase",
        "observation",
        "recent_outcomes",
        "constraints",
        "candidates",
    }
    for split in ("train", "valid", "test"):
        rows = dataset.get(split)
        if rows is None:
            raise ValueError(f"live-context dataset is missing split {split}")
        split_groups: set[str] = set()
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            prior = payload_owner.setdefault(payload, split)
            if prior != split:
                raise ValueError("live-context learning payload crosses splits")
            metadata = row["metadata"]
            group_id = str(metadata["group_id"])
            split_groups.add(group_id)
            groups[group_id].append(row)
            if metadata.get("permutation_group") is not True:
                raise ValueError("v6 live-context row is not marked as a permutation group")
            state = json.loads(str(row["messages"][1]["content"]))
            if set(state) != expected_state_keys or state["phase"] != "phase_1":
                raise ValueError("v6 row drifted from the production policy context shape")
            candidates = state["candidates"]
            if len(candidates) != LIVE_CONTEXT_CARDINALITY:
                raise ValueError("v6 live context must contain exactly four candidates")
            if {candidate["id"] for candidate in candidates} != set(range(4)):
                raise ValueError("v6 candidate IDs must be dense 0..3")
            target_id = int(metadata["target_candidate_id"])
            target = next(candidate for candidate in candidates if candidate["id"] == target_id)
            if target.get("authorized") is not True or target.get("redundant") is not False:
                raise ValueError("v6 oracle selected an ineligible candidate")
            if target.get("call") != metadata.get("target_call"):
                raise ValueError("v6 target call metadata drifted from the candidate")
            for candidate in candidates:
                purpose = str(candidate["purpose"])
                if not purpose.startswith("Tap the current-frame '"):
                    raise ValueError("v6 candidate purpose drifted from Engine output")
                call = candidate["call"]
                arguments = call.get("arguments") if isinstance(call, Mapping) else None
                if call.get("tool") != "tap_and_analyze" or not isinstance(arguments, Mapping):
                    raise ValueError("v6 candidate is not a production tap_and_analyze call")
                visible_values = [str(value) for value in arguments.values()]
                for value in visible_values:
                    folded = value.casefold()
                    if any(reserved_label in folded for reserved_label in reserved):
                        raise ValueError("reserved live-smoke label leaked into v6 learning rows")
                    owner = entity_owner.setdefault(folded, split)
                    if owner != split:
                        raise ValueError("v6 selector vocabulary crosses splits")
            reconstructed_candidates = tuple(
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
            reconstructed = PolicyContext(
                goal=str(state["goal"]),
                phase=str(state["phase"]),
                candidates=reconstructed_candidates,
                observation=copy.deepcopy(state["observation"]),
                recent_outcomes=tuple(str(value) for value in state["recent_outcomes"]),
                constraints=tuple(str(value) for value in state["constraints"][:-2]),
            )
            if row["messages"][:2] != policy_messages(reconstructed, reconstructed_candidates):
                raise ValueError("v6 row is not byte-shaped by production policy_messages")
        stats[split] = {
            "rows": len(rows),
            "groups": len(split_groups),
            "variants_per_group": LIVE_CONTEXT_VARIANTS,
        }

    expected_orders = set(itertools.permutations(range(4)))
    expected_assignments = set(itertools.permutations(range(4)))
    for group_id, variants in groups.items():
        if len(variants) != LIVE_CONTEXT_VARIANTS:
            raise ValueError(f"live-context group {group_id} lacks 576 variants")
        orders = Counter(tuple(row["metadata"]["candidate_order"]) for row in variants)
        assignments = Counter(tuple(row["metadata"]["id_assignment"]) for row in variants)
        if set(orders) != expected_orders or set(assignments) != expected_assignments:
            raise ValueError(f"live-context group {group_id} lacks full permutations")
        if set(orders.values()) != {24} or set(assignments.values()) != {24}:
            raise ValueError(f"live-context group {group_id} permutations are not counterbalanced")
        target_ids = Counter(int(row["metadata"]["target_candidate_id"]) for row in variants)
        target_positions = Counter(int(row["metadata"]["target_position"]) for row in variants)
        if target_ids != Counter({0: 144, 1: 144, 2: 144, 3: 144}):
            raise ValueError(f"live-context group {group_id} leaks target IDs")
        if target_positions != Counter({0: 144, 1: 144, 2: 144, 3: 144}):
            raise ValueError(f"live-context group {group_id} leaks target positions")

    return {
        "passed": True,
        "uses_exact_policy_messages": True,
        "literal_payloads_disjoint": True,
        "selector_vocabularies_disjoint": True,
        "reserved_live_smoke_labels_absent": True,
        "all_candidate_orders": 24,
        "all_dense_id_assignments": 24,
        "variants_per_semantic_group": LIVE_CONTEXT_VARIANTS,
        "splits": stats,
    }


def build_v6_dataset(
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    live_group_counts: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    live_seed: int = LIVE_CONTEXT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Return frozen v5 rows plus exact-production-context v6 rows."""

    base = build_v5_dataset(base_split_sizes, seed=seed, denylist=denylist)
    live = build_live_context_rows(live_group_counts, seed=live_seed, denylist=denylist)
    result: dict[str, list[dict[str, Any]]] = {}
    seen_payloads: set[str] = set()
    group_owner: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        rows = [*base[split], *live[split]]
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            if payload in seen_payloads:
                raise ValueError(f"duplicate v6 learning payload: {row['id']}")
            seen_payloads.add(payload)
            group_id = str(row["metadata"]["group_id"])
            owner = group_owner.setdefault(group_id, split)
            if owner != split:
                raise ValueError(f"semantic group {group_id} crosses {owner} and {split}")
        random.Random(_stable_seed("combined-row-order-v6", live_seed, split)).shuffle(rows)
        result[split] = rows
    return result


def _split_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode("utf-8")


def write_v6_dataset(
    output_dir: str | Path,
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    live_group_counts: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    live_seed: int = LIVE_CONTEXT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, Any]:
    """Write v6 and a manifest pinning its frozen v5 foundation."""

    output = Path(output_dir)
    base = build_v5_dataset(base_split_sizes, seed=seed, denylist=denylist)
    live = build_live_context_rows(live_group_counts, seed=live_seed, denylist=denylist)
    dataset = build_v6_dataset(
        base_split_sizes,
        live_group_counts=live_group_counts,
        seed=seed,
        live_seed=live_seed,
        denylist=denylist,
    )
    statistics = dataset_statistics(dataset)
    audit = audit_live_context_boundaries(live)
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
            "base_v5_records": len(base[split]),
            "live_context_v6_records": len(live[split]),
            **statistics["splits"][split],
        }
    combined_hash = hashlib.sha256(
        "".join(entries[split]["sha256"] for split in ("train", "valid", "test")).encode()
    ).hexdigest()
    manifest = {
        "format": "functiongemma-aua-candidate-policy-v6",
        "seed": seed,
        "live_context_seed": live_seed,
        "selection_function": "select_candidate(candidate_id: integer)",
        "split_policy": (
            "frozen v5 groups plus split-exclusive exact-policy serializer states; "
            "every semantic state expands to all candidate-order and dense-ID permutations"
        ),
        "base_v5": {
            "format": "functiongemma-aua-candidate-policy-v5",
            "split_sha256": base_hashes,
            "records": {split: len(base[split]) for split in base},
        },
        "live_context_v6": {
            "family": LIVE_CONTEXT_FAMILY,
            "group_counts": dict(
                LIVE_CONTEXT_GROUP_COUNTS if live_group_counts is None else live_group_counts
            ),
            "techniques": [
                "exact packaged policy_messages serializer",
                "all 24 candidate orders",
                "all 24 dense ID assignments",
                "realistic fictional text, rid, and description selectors",
                "split-exclusive goals, labels, screens, and selectors",
            ],
            "leakage_and_invariance_audit": audit,
        },
        "total_records": statistics["total_records"],
        "ratios": statistics["ratios"],
        "privacy": {
            "passed": True,
            "checks": [
                "denylist",
                "fictional public vocabulary",
                "no journals, maps, screenshots, hierarchy, devices, or private app traces",
                "reserved independent-smoke labels absent",
            ],
        },
        "dataset_sha256": combined_hash,
        "splits": entries,
    }
    _atomic_write(output / "manifest.json", f"{_canonical_json(manifest)}\n".encode())
    return manifest
