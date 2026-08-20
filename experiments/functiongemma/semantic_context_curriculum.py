"""Broad exact-runtime semantic disambiguation curriculum for FunctionGemma v7.

V6 proved opaque-ID invariance but spent 23,040 training rows on every permutation
of only forty semantic states.  V7 instead creates many independent fictional
states that mirror the actual post-compiler ambiguity: two to four safe controls
share the requested destination and differ by a privacy-screened summary.  The
model sees the exact packaged ``policy_messages`` serialization and selects the
summary named by AUA's candidate-backed goal projection.

The module is host-only.  It never reads Android, journals, maps, screenshots,
hierarchies, typed input, or private application data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.functiongemma.curriculum import (  # noqa: E402
    DEFAULT_DENYLIST,
    DEFAULT_SEED,
    _atomic_write,
    _canonical_json,
    _stable_seed,
    dataset_statistics,
    privacy_violations,
)
from experiments.functiongemma.production_curriculum import _label_pool  # noqa: E402
from experiments.functiongemma.recovery_curriculum import build_v5_dataset  # noqa: E402

from android_ui_analyser.policy import (  # noqa: E402
    PolicyCandidate,
    PolicyContext,
    policy_messages,
    policy_tools,
)

SEMANTIC_CONTEXT_SEED = 20_260_819
SEMANTIC_CARDINALITIES = (2, 3, 4)
SEMANTIC_GROUP_COUNTS: dict[str, dict[int, int]] = {
    "train": {2: 512, 3: 512, 4: 512},
    "valid": {2: 96, 3: 96, 4: 96},
    "test": {2: 96, 3: 96, 4: 96},
}
SEMANTIC_FAMILY = "semantic_destination_choice"

# Kept only for the independent v7 smoke and rejected from emitted rows.
RESERVED_V7_SMOKE_TERMS = frozenset(
    {
        "Chronicle",
        "archived notebooks",
        "guided lessons",
        "recent timelines",
        "saved exhibits",
    }
)

_SCREENS = {
    "train": ("Example Shelf", "Fixture Cabinet", "Practice Console", "Demo Gallery"),
    "valid": ("Validation Alcove", "Reference Rack", "Trial Workspace"),
    "test": ("Evaluation Atrium", "Verification Desk", "Testing Board"),
}

_OBJECTIVE_TEMPLATES = (
    "Open {target} using the row whose summary mentions {qualifier}.",
    "Open {target}; choose the current row described by {qualifier}.",
    "Navigate to {target}; select the option whose details say {qualifier}.",
    "Reach {target} through the entry marked {qualifier}.",
)


def _selector(full_label: str, lane: int) -> dict[str, str]:
    if lane == 0:
        return {"text": full_label}
    if lane == 1:
        return {"desc": f"Open {full_label}"}
    return {"rid": "open" + "".join(part.title() for part in full_label.split())}


def _group_specs(
    split: str,
    counts: Mapping[int, int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    pool = _label_pool(split)
    rng = random.Random(_stable_seed("semantic-context-groups-v7", seed, split))
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[dict[str, Any]] = []
    global_ordinal = 0
    for cardinality in SEMANTIC_CARDINALITIES:
        count = int(counts.get(cardinality, 0))
        if count <= 0:
            raise ValueError("every v7 cardinality needs a positive group count")
        for ordinal in range(count):
            for _attempt in range(20_000):
                target = rng.choice(pool)
                available = list(pool)
                rng.shuffle(available)
                used_terms = set(target.casefold().split())
                selected_qualifiers: list[str] = []
                for value in available:
                    terms = set(value.casefold().split())
                    if terms & used_terms:
                        continue
                    selected_qualifiers.append(value)
                    used_terms.update(terms)
                    if len(selected_qualifiers) == cardinality:
                        break
                if len(selected_qualifiers) != cardinality:  # pragma: no cover
                    continue
                qualifiers = tuple(selected_qualifiers)
                target_qualifier = qualifiers[rng.randrange(cardinality)]
                signature = (target, tuple(sorted(qualifiers)))
                if signature not in seen:
                    seen.add(signature)
                    break
            else:  # pragma: no cover - configured pools have ample combinations
                raise ValueError(f"could not construct unique v7 group for {split}")
            result.append(
                {
                    "cardinality": cardinality,
                    "ordinal": ordinal,
                    "global_ordinal": global_ordinal,
                    "target": target,
                    "qualifiers": qualifiers,
                    "target_qualifier": target_qualifier,
                }
            )
            global_ordinal += 1
    return result


def _candidate(
    *,
    candidate_id: int,
    target: str,
    qualifier: str,
    lane: int,
    group_id: str,
) -> PolicyCandidate:
    full_label = f"{target} {qualifier.lower()}"
    arguments = _selector(full_label, lane)
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


def _variant_candidates(
    *,
    target: str,
    qualifiers: Sequence[str],
    target_qualifier: str,
    variant: int,
    group_id: str,
    lane_offset: int,
) -> tuple[tuple[PolicyCandidate, ...], PolicyCandidate, int]:
    cardinality = len(qualifiers)
    if variant not in range(cardinality * cardinality):
        raise ValueError("v7 variant is outside its cardinality square")
    target_id, target_position = divmod(variant, cardinality)
    distractors = [value for value in qualifiers if value != target_qualifier]
    rotation = (target_id + target_position) % len(distractors) if distractors else 0
    distractors = distractors[rotation:] + distractors[:rotation]
    ordered = list(distractors)
    ordered.insert(target_position, target_qualifier)
    remaining_ids = [value for value in range(cardinality) if value != target_id]
    if remaining_ids:
        shift = target_position % len(remaining_ids)
        remaining_ids = remaining_ids[shift:] + remaining_ids[:shift]
    id_cursor = iter(remaining_ids)
    candidates: list[PolicyCandidate] = []
    selected: PolicyCandidate | None = None
    for position, qualifier in enumerate(ordered):
        candidate = _candidate(
            candidate_id=target_id if qualifier == target_qualifier else next(id_cursor),
            target=target,
            qualifier=qualifier,
            lane=(lane_offset + position) % 3,
            group_id=group_id,
        )
        candidates.append(candidate)
        if qualifier == target_qualifier:
            selected = candidate
    assert selected is not None
    return tuple(candidates), selected, target_position


def _record(
    *,
    split: str,
    group: Mapping[str, Any],
    variant: int,
) -> dict[str, Any]:
    cardinality = int(group["cardinality"])
    ordinal = int(group["ordinal"])
    global_ordinal = int(group["global_ordinal"])
    target = str(group["target"])
    qualifiers = tuple(str(value) for value in group["qualifiers"])
    target_qualifier = str(group["target_qualifier"])
    group_id = f"semantic-{split}-c{cardinality}-{ordinal:04d}"
    candidates, selected, target_position = _variant_candidates(
        target=target,
        qualifiers=qualifiers,
        target_qualifier=target_qualifier,
        variant=variant,
        group_id=group_id,
        lane_offset=global_ordinal,
    )
    objective = _OBJECTIVE_TEMPLATES[global_ordinal % len(_OBJECTIVE_TEMPLATES)].format(
        target=target,
        qualifier=target_qualifier.lower(),
    )
    # This is the exact result of Engine._policy_selection_goal for the
    # generated objective: both values already came from privacy-screened
    # candidate prose, so no raw objective copy crosses the model boundary.
    # Each term keeps the spelling the objective used, matching the runtime
    # goal builder — candidate labels are cased, and a case-folded rare label
    # stops binding to the label it names.
    model_goal = (
        f"Requested destination: {target}. "
        f"Matching evidence: {target_qualifier.lower()}."
    )
    context = PolicyContext(
        goal=model_goal,
        phase="phase_1",
        candidates=candidates,
        observation={
            "fresh": True,
            "known_screen": _SCREENS[split][global_ordinal % len(_SCREENS[split])],
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
                        "arguments": {"candidate_id": selected.candidate_id},
                    },
                }
            ],
        }
    )
    return {
        "id": f"fg-v7-{split}-c{cardinality}-{ordinal:04d}-v{variant:02d}",
        "messages": messages,
        "tools": policy_tools(),
        "metadata": {
            "split": split,
            "group_id": group_id,
            "episode_id": f"episode-{group_id}",
            "step": 0,
            "case_id": f"{group_id}-v{variant:02d}",
            "variant": variant,
            "intent": objective,
            "model_goal": model_goal,
            "family": SEMANTIC_FAMILY,
            "label": SEMANTIC_FAMILY,
            "scenario_kind": "exact_runtime_semantic_disambiguation",
            "criticality": "normal",
            "template_profile": "exact_policy_messages_v7",
            "target_candidate_id": selected.candidate_id,
            "target_position": target_position,
            "target_call": selected.trusted_call(),
            "tool_name": "tap_and_analyze",
            "cardinality": cardinality,
            "variants_total": cardinality * cardinality,
            "selector_lane": global_ordinal % 3,
            "curriculum_version": "v7",
        },
    }


def build_semantic_context_rows(
    group_counts: Mapping[str, Mapping[int, int]] | None = None,
    *,
    seed: int = SEMANTIC_CONTEXT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Build the v7-only exact-runtime semantic augmentation."""

    counts = SEMANTIC_GROUP_COUNTS if group_counts is None else group_counts
    if tuple(counts) != ("train", "valid", "test"):
        raise ValueError("group_counts must contain train, valid, and test in order")
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        rows: list[dict[str, Any]] = []
        for group in _group_specs(split, counts[split], seed=seed):
            cardinality = int(group["cardinality"])
            for variant in range(cardinality * cardinality):
                row = _record(split=split, group=group, variant=variant)
                findings = privacy_violations(row, denylist=denylist)
                if findings:
                    raise ValueError(f"privacy audit failed for {row['id']}: {'; '.join(findings)}")
                rows.append(row)
        random.Random(_stable_seed("semantic-context-order-v7", seed, split)).shuffle(rows)
        result[split] = rows
    audit_semantic_context_boundaries(result)
    return result


def audit_semantic_context_boundaries(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed on split leakage, schema drift, and ID/position shortcuts."""

    payload_owner: dict[str, str] = {}
    group_owner: dict[str, str] = {}
    entity_owner: dict[str, str] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    stats: dict[str, Any] = {}
    reserved = tuple(value.casefold() for value in RESERVED_V7_SMOKE_TERMS)
    for split in ("train", "valid", "test"):
        rows = dataset.get(split)
        if rows is None:
            raise ValueError(f"v7 semantic dataset is missing split {split}")
        cardinalities: Counter[int] = Counter()
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            owner = payload_owner.setdefault(payload, split)
            if owner != split:
                raise ValueError("v7 literal learning payload crosses splits")
            if any(value in payload.casefold() for value in reserved):
                raise ValueError("reserved v7 smoke vocabulary leaked into learning rows")
            metadata = row["metadata"]
            group_id = str(metadata["group_id"])
            if group_owner.setdefault(group_id, split) != split:
                raise ValueError("v7 semantic group crosses splits")
            groups[group_id].append(row)
            state = json.loads(str(row["messages"][1]["content"]))
            candidates = state["candidates"]
            cardinality = len(candidates)
            cardinalities[cardinality] += 1
            if cardinality not in SEMANTIC_CARDINALITIES:
                raise ValueError("v7 row has an unsupported cardinality")
            if {candidate["id"] for candidate in candidates} != set(range(cardinality)):
                raise ValueError("v7 candidate IDs are not dense")
            target_id = int(metadata["target_candidate_id"])
            target = next(candidate for candidate in candidates if candidate["id"] == target_id)
            if target["call"] != metadata["target_call"]:
                raise ValueError("v7 target metadata drifted from the exact call")
            if not str(state["goal"]).startswith("Requested destination: "):
                raise ValueError("v7 goal did not use the production candidate-backed projection")
            for candidate in candidates:
                purpose = str(candidate["purpose"])
                entity = purpose.removeprefix("Tap the current-frame '").removesuffix(
                    "' control and observe the result."
                )
                folded = entity.casefold()
                if entity_owner.setdefault(folded, split) != split:
                    raise ValueError("v7 visible entity vocabulary crosses splits")
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
                constraints=tuple(str(value) for value in state["constraints"][:-2]),
            )
            if row["messages"][:2] != policy_messages(reconstructed_context, reconstructed):
                raise ValueError("v7 row drifted from packaged policy_messages")
        stats[split] = {
            "rows": len(rows),
            "groups": len({str(row["metadata"]["group_id"]) for row in rows}),
            "cardinalities": dict(sorted(cardinalities.items())),
        }

    for group_id, rows in groups.items():
        cardinality = int(rows[0]["metadata"]["cardinality"])
        expected = cardinality * cardinality
        if len(rows) != expected:
            raise ValueError(f"v7 group {group_id} lacks its counterbalanced variants")
        target_ids = Counter(int(row["metadata"]["target_candidate_id"]) for row in rows)
        positions = Counter(int(row["metadata"]["target_position"]) for row in rows)
        expected_counts = Counter(dict.fromkeys(range(cardinality), cardinality))
        if target_ids != expected_counts:
            raise ValueError(f"v7 group {group_id} leaks target ID")
        if positions != expected_counts:
            raise ValueError(f"v7 group {group_id} leaks target position")
    return {
        "passed": True,
        "uses_exact_policy_messages": True,
        "literal_payloads_disjoint": True,
        "semantic_groups_disjoint": True,
        "entity_vocabularies_disjoint": True,
        "reserved_v7_smoke_terms_absent": True,
        "splits": stats,
    }


def build_v7_dataset(
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    semantic_group_counts: Mapping[str, Mapping[int, int]] | None = None,
    seed: int = DEFAULT_SEED,
    semantic_seed: int = SEMANTIC_CONTEXT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Return frozen v5 recovery foundations plus broad v7 runtime states."""

    base = build_v5_dataset(base_split_sizes, seed=seed, denylist=denylist)
    semantic = build_semantic_context_rows(
        semantic_group_counts,
        seed=semantic_seed,
        denylist=denylist,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    seen_payloads: set[str] = set()
    group_owner: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        rows = [*base[split], *semantic[split]]
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            if payload in seen_payloads:
                raise ValueError(f"duplicate v7 learning payload: {row['id']}")
            seen_payloads.add(payload)
            group_id = str(row["metadata"]["group_id"])
            if group_owner.setdefault(group_id, split) != split:
                raise ValueError(f"semantic group {group_id} crosses splits")
        random.Random(_stable_seed("combined-v7-row-order", semantic_seed, split)).shuffle(rows)
        result[split] = rows
    return result


def _split_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode()


def write_v7_dataset(
    output_dir: str | Path,
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    semantic_group_counts: Mapping[str, Mapping[int, int]] | None = None,
    seed: int = DEFAULT_SEED,
    semantic_seed: int = SEMANTIC_CONTEXT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, Any]:
    """Write v7 JSONL plus deterministic provenance and leakage evidence."""

    output = Path(output_dir)
    base = build_v5_dataset(base_split_sizes, seed=seed, denylist=denylist)
    semantic = build_semantic_context_rows(
        semantic_group_counts,
        seed=semantic_seed,
        denylist=denylist,
    )
    dataset = build_v7_dataset(
        base_split_sizes,
        semantic_group_counts=semantic_group_counts,
        seed=seed,
        semantic_seed=semantic_seed,
        denylist=denylist,
    )
    statistics = dataset_statistics(dataset)
    audit = audit_semantic_context_boundaries(semantic)
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
            "semantic_v7_records": len(semantic[split]),
            **statistics["splits"][split],
        }
    combined_hash = hashlib.sha256(
        "".join(entries[split]["sha256"] for split in ("train", "valid", "test")).encode()
    ).hexdigest()
    manifest = {
        "format": "functiongemma-aua-candidate-policy-v7",
        "seed": seed,
        "semantic_context_seed": semantic_seed,
        "selection_function": "select_candidate(candidate_id: integer)",
        "split_policy": (
            "frozen v5 foundations plus split-exclusive exact-runtime semantic groups; "
            "semantic assignment precedes counterbalanced dense IDs and positions"
        ),
        "base_v5": {
            "format": "functiongemma-aua-candidate-policy-v5",
            "split_sha256": base_hashes,
            "records": {split: len(base[split]) for split in base},
        },
        "semantic_context_v7": {
            "family": SEMANTIC_FAMILY,
            "cardinalities": list(SEMANTIC_CARDINALITIES),
            "group_counts": (
                SEMANTIC_GROUP_COUNTS if semantic_group_counts is None else semantic_group_counts
            ),
            "variants_per_group": {str(value): value * value for value in SEMANTIC_CARDINALITIES},
            "techniques": [
                "exact packaged policy_messages serializer",
                "candidate-backed disambiguating goal projection",
                "many independent same-destination semantic states",
                "counterbalanced dense IDs and target positions",
                "resource-id, text, and description selectors",
            ],
            "leakage_and_invariance_audit": audit,
        },
        "total_records": statistics["total_records"],
        "ratios": statistics["ratios"],
        "privacy": {
            "passed": True,
            "checks": [
                "denylist and public one-way fingerprints",
                "fictional com.example vocabulary",
                "no journals, maps, screenshots, hierarchy, devices, or typed input",
                "reserved independent-smoke vocabulary absent",
            ],
        },
        "dataset_sha256": combined_hash,
        "splits": entries,
    }
    _atomic_write(output / "manifest.json", f"{_canonical_json(manifest)}\n".encode())
    return manifest
