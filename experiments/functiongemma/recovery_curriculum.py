"""Recovery-focused, counterfactual FunctionGemma curriculum for v5.

The frozen v3 and v4 rows remain byte-for-byte unchanged.  This module adds
split-isolated minimal pairs for the decisions that matter after an Android
action: observe an unknown outcome, restore owned state, wait for loading,
refresh stale evidence, refuse an irrelevant target, and finish only after
proof and cleanup are complete.

Half of the new semantic groups mask candidate function and argument names.
The descriptions, state, proof, authorization, and cleanup metadata remain
meaningful, following Hammer's function-masking idea without changing
FunctionGemma's sole public ``select_candidate`` output schema.

All examples are deterministic and fictional.  This module never reads a
journal, device, emulator, ADB state, AUA memory, or ignored training artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.functiongemma.curriculum import (
    DEFAULT_DENYLIST,
    DEFAULT_SEED,
    SELECT_CANDIDATE_TOOL,
    _atomic_write,
    _canonical_json,
    _stable_seed,
    dataset_statistics,
    privacy_violations,
)
from experiments.functiongemma.production_curriculum import (
    PRODUCTION_DEVELOPER,
    PRODUCTION_FIXTURE_REF,
    PRODUCTION_REQUEST,
    RESERVED_SMOKE_LABELS,
    build_v4_dataset,
)

RECOVERY_SEED = 20_260_817
RECOVERY_CARDINALITIES = (2, 3, 4)
RECOVERY_GROUPS_PER_FAMILY = {"train": 96, "valid": 18, "test": 18}

RECOVERY_FAMILIES = (
    "unknown_outcome",
    "terminal_ready",
    "cleanup_pending",
    "cleanup_complete",
    "loading_transition",
    "settled_transition",
    "stale_observation",
    "fresh_observation",
    "irrelevant_target",
    "relevant_target",
)

_PAIR_BY_FAMILY = {
    "unknown_outcome": "outcome_vs_finish",
    "terminal_ready": "outcome_vs_finish",
    "cleanup_pending": "cleanup_vs_finish",
    "cleanup_complete": "cleanup_vs_finish",
    "loading_transition": "loading_vs_action",
    "settled_transition": "loading_vs_action",
    "stale_observation": "stale_vs_action",
    "fresh_observation": "stale_vs_action",
    "irrelevant_target": "irrelevance_vs_action",
    "relevant_target": "irrelevance_vs_action",
}

_PAIR_FAMILIES = {
    pair: tuple(family for family in RECOVERY_FAMILIES if _PAIR_BY_FAMILY[family] == pair)
    for pair in dict.fromkeys(_PAIR_BY_FAMILY.values())
}

_SPLIT_ENTITIES = {
    "train": (
        "Amber Ledger",
        "Brisk Meadow",
        "Copper Lantern",
        "Daring Pebble",
        "Eager Willow",
        "Frosty Harbor",
        "Golden Kestrel",
        "Humble Orchid",
        "Ivory Summit",
        "Jolly Thistle",
        "Keen Valley",
        "Lucid Walnut",
    ),
    "valid": (
        "Azure Almond",
        "Bold Birch",
        "Crisp Coral",
        "Deep Daisy",
        "Earnest Ember",
        "Floral Falcon",
    ),
    "test": (
        "Agile Apricot",
        "Blush Bamboo",
        "Clear Clover",
        "Delicate Delta",
        "Even Eagle",
        "Glossy Fjord",
    ),
}

_SPLIT_OUTCOME_WORDING = {
    "train": (
        "The last mutating call has no trustworthy completion result.",
        "The current UI may reflect either side of the attempted transition.",
    ),
    "valid": (
        "Whether the preceding mutation landed is not established.",
        "No replay is allowed until current screen truth is recovered.",
    ),
    "test": (
        "The preceding state-changing operation has an unresolved outcome.",
        "A fresh read is required before another mutation can be justified.",
    ),
}

_SPLIT_WORDING_REPLACEMENTS = {
    "train": (
        ("Read ", "Inspect "),
        ("Close ", "Conclude "),
        ("Restore ", "Reinstate "),
        ("Repeat ", "Reissue "),
        ("Request ", "Obtain "),
        ("current", "live"),
        ("fresh", "newly captured"),
        ("proof", "evidence"),
    ),
    "valid": (
        ("Read ", "Capture "),
        ("Close ", "Finalize "),
        ("Restore ", "Recover "),
        ("Repeat ", "Reapply "),
        ("Request ", "Collect "),
        ("current", "present"),
        ("fresh", "latest"),
        ("proof", "verification"),
    ),
    "test": (
        ("Read ", "Establish "),
        ("Close ", "End "),
        ("Restore ", "Return "),
        ("Repeat ", "Perform again "),
        ("Request ", "Acquire "),
        ("current", "up-to-date"),
        ("fresh", "authoritative"),
        ("proof", "substantiation"),
    ),
}


@dataclass(frozen=True)
class CandidateSpec:
    semantic_tool: str
    arguments: Mapping[str, Any]
    purpose: str
    proof: str
    cleanup: str = "none"
    risk: str = "safe"
    authorized: bool = True
    redundant: bool = False


def _compact_entity(entity: str) -> str:
    return "".join(part for part in entity.split() if part)


def _candidate(
    semantic_tool: str,
    arguments: Mapping[str, Any],
    *,
    purpose: str,
    proof: str,
    cleanup: str = "none",
    risk: str = "safe",
    authorized: bool = True,
    redundant: bool = False,
) -> CandidateSpec:
    if risk not in {"safe", "unsafe", "destructive"}:
        raise ValueError(f"invalid recovery candidate risk: {risk}")
    return CandidateSpec(
        semantic_tool=semantic_tool,
        arguments=dict(arguments),
        purpose=purpose,
        proof=proof,
        cleanup=cleanup,
        risk=risk,
        authorized=authorized,
        redundant=redundant,
    )


def _pair_specs(pair: str, entity: str, family: str) -> tuple[list[CandidateSpec], int]:
    rid = f"open{_compact_entity(entity)}"
    if pair == "outcome_vs_finish":
        unknown = family == "unknown_outcome"
        return [
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Read current UI truth without replaying the unresolved mutation.",
                proof="A fresh observation distinguishes the origin from the landed destination.",
                redundant=not unknown,
            ),
            _candidate(
                "session_finish",
                {},
                purpose="Close the goal session after every phase and cleanup obligation is proved.",
                proof="Terminal accounting closes lifecycle ownership only when no proof is missing.",
                risk="unsafe" if unknown else "safe",
                authorized=not unknown,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": rid},
                purpose=f"Repeat the mutation that attempts to open {entity}.",
                proof="A second mutation cannot establish which attempt caused the resulting state.",
                risk="unsafe" if unknown else "safe",
                authorized=not unknown,
                redundant=True,
            ),
            _candidate(
                "session_review",
                {},
                purpose="Read lifecycle accounting without observing or closing the UI state.",
                proof="Accounting alone is not arrival evidence.",
                redundant=True,
            ),
        ], 0 if unknown else 1

    if pair == "cleanup_vs_finish":
        pending = family == "cleanup_pending"
        return [
            _candidate(
                "network_restore",
                {"timeout_ms": 15_000},
                purpose="Restore the session-owned connectivity snapshot and verify its read-back.",
                proof="The reversible network obligation is discharged before lifecycle closure.",
                cleanup="completes required network cleanup" if pending else "none",
                redundant=not pending,
            ),
            _candidate(
                "session_finish",
                {},
                purpose="Close the session only after all proof and cleanup obligations are complete.",
                proof="Terminal accounting confirms that no owned state remains.",
                risk="unsafe" if pending else "safe",
                authorized=not pending,
            ),
            _candidate(
                "network_status",
                {"verify": True},
                purpose="Inspect connectivity without changing the saved restoration obligation.",
                proof="Status evidence alone does not restore session-owned state.",
                cleanup="network_restore remains required" if pending else "none",
                redundant=True,
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Read the UI even though the pending decision concerns environment cleanup.",
                proof="A hierarchy refresh does not prove connectivity restoration.",
                redundant=True,
            ),
        ], 0 if pending else 1

    if pair == "loading_vs_action":
        loading = family == "loading_transition"
        return [
            _candidate(
                "await_and_analyze",
                {"predicate": "rid:detailPanel,!text:Loading", "timeout_ms": 8_000},
                purpose="Bound the existing transition and require settled destination evidence.",
                proof="The result contains the detail anchor and excludes the loading marker.",
                redundant=not loading,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": rid},
                purpose=f"Open {entity} once from the settled, fresh list screen.",
                proof="The action returns a folded destination observation.",
                risk="unsafe" if loading else "safe",
                authorized=not loading,
                redundant=loading,
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Request an unbounded extra observation instead of the phase-specific wait.",
                proof="A generic refresh has no required positive and negative arrival predicate.",
                redundant=True,
            ),
            _candidate(
                "key_and_analyze",
                {"name": "back"},
                purpose="Leave the current screen instead of completing the requested transition.",
                proof="Back navigation cannot prove the requested destination.",
                risk="unsafe",
                authorized=False,
            ),
        ], 0 if loading else 1

    if pair == "stale_vs_action":
        stale = family == "stale_observation"
        return [
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Refresh the frame before using a selector whose observation is stale.",
                proof="A new hierarchy establishes current selector identity and package state.",
                redundant=not stale,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": rid},
                purpose=f"Open {entity} from a fresh current-package observation.",
                proof="The stable selector and returned observation directly advance the phase.",
                risk="unsafe" if stale else "safe",
                authorized=not stale,
            ),
            _candidate(
                "resolve",
                {"target": _compact_entity(entity).casefold()},
                purpose="Resolve a semantic target without first refreshing stale screen truth.",
                proof="Resolution cannot bind an expired frame to a current element.",
                redundant=not stale,
            ),
            _candidate(
                "session_finish",
                {},
                purpose="End the session while the active UI checkpoint is incomplete.",
                proof="Termination cannot replace missing current-frame evidence.",
                risk="unsafe",
                authorized=False,
            ),
        ], 0 if stale else 1

    if pair == "irrelevance_vs_action":
        irrelevant = family == "irrelevant_target"
        return [
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Recover current screen truth because none of the visible actions matches the goal.",
                proof="A fresh observation can expose a relevant control or justify escalation.",
                redundant=not irrelevant,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": rid},
                purpose=f"Open the goal-matching {entity} control when it is present and current.",
                proof="The stable selector returns direct post-action evidence.",
                risk="unsafe" if irrelevant else "safe",
                authorized=not irrelevant,
            ),
            _candidate(
                "reach",
                {"goal": "example_unrelated_panel"},
                purpose="Navigate toward a different visible label that does not satisfy the goal.",
                proof="Arrival at an unrelated panel is not phase evidence.",
                redundant=True,
            ),
            _candidate(
                "session_finish",
                {},
                purpose="Stop because the current screen lacks a relevant action.",
                proof="Missing a candidate requires recovery or escalation, not false completion.",
                risk="unsafe",
                authorized=False,
            ),
        ], 0 if irrelevant else 1

    raise ValueError(f"unknown recovery pair: {pair}")


def _state(split: str, family: str, entity: str, ordinal: int) -> dict[str, Any]:
    pair = _PAIR_BY_FAMILY[family]
    base = {
        "fixture_ref": PRODUCTION_FIXTURE_REF,
        "request": PRODUCTION_REQUEST,
        "goal": f"Open {entity}, prove the destination, restore owned state, and finish safely.",
        "phase": family,
        "observation": {
            "fresh": True,
            "known_screen": f"fixture_{split}_{pair}_{ordinal:03d}",
            "outcome": "known",
            "goal_checkpoint_reached": False,
            "element_count": 4,
            "source": "hierarchy",
        },
        "recent_outcomes": [
            "A goal session owns the active test lifecycle.",
            "No unverified mutation may be replayed.",
        ],
        "constraints": [
            "Select exactly one supplied candidate",
            "Require direct proof",
            "Never finish with missing proof or cleanup",
        ],
    }
    observation = base["observation"]
    outcomes = base["recent_outcomes"]
    constraints = base["constraints"]

    if family == "unknown_outcome":
        observation.update(fresh=False, known_screen="unknown", outcome="unknown", element_count=0)
        outcomes[:] = list(_SPLIT_OUTCOME_WORDING[split])
        constraints.append("Observe current truth before any replay or finish")
    elif family == "terminal_ready":
        observation["goal_checkpoint_reached"] = True
        outcomes[:] = ["Every goal phase has direct proof.", "No cleanup obligation remains."]
    elif family == "cleanup_pending":
        observation["goal_checkpoint_reached"] = True
        outcomes[:] = ["Goal proof is complete.", "The saved network snapshot is still offline."]
        constraints.append("Restore session-owned connectivity before finish")
    elif family == "cleanup_complete":
        observation["goal_checkpoint_reached"] = True
        outcomes[:] = [
            "Goal proof is complete.",
            "Connectivity restoration has verified read-back.",
        ]
    elif family == "loading_transition":
        observation.update(known_screen="transition", outcome="loading", element_count=1)
        outcomes[:] = ["The authorized tap was dispatched once.", "Loading is still visible."]
        constraints.append("Wait for positive arrival and absence of Loading")
    elif family == "settled_transition":
        observation.update(known_screen="fixture_list", outcome="known", element_count=4)
        outcomes[:] = ["The list screen is settled.", f"The {entity} selector is current."]
    elif family == "stale_observation":
        observation.update(fresh=False, outcome="stale", element_count=0)
        outcomes[:] = ["The previous frame changed.", "No current selector binding is available."]
        constraints.append("Refresh before any frame-bound action")
    elif family == "fresh_observation":
        outcomes[:] = [
            "The latest result returned a fresh frame.",
            f"The {entity} selector is current.",
        ]
    elif family == "irrelevant_target":
        observation.update(known_screen="unrelated_panel", element_count=3)
        outcomes[:] = [
            "The requested target is absent.",
            "Visible controls belong to another panel.",
        ]
        constraints.append("Recover or escalate instead of choosing an irrelevant action")
    elif family == "relevant_target":
        outcomes[:] = ["The requested target is visible.", f"The {entity} selector is current."]
    else:  # pragma: no cover - guarded by RECOVERY_FAMILIES
        raise ValueError(f"unknown recovery family: {family}")
    return base


def _masked_call(
    spec: CandidateSpec,
    *,
    split: str,
    pair: str,
    ordinal: int,
    index: int,
    masked: bool,
) -> dict[str, Any]:
    if not masked:
        return {"tool": spec.semantic_tool, "arguments": copy.deepcopy(dict(spec.arguments))}
    salt = hashlib.sha256(f"{split}:{pair}:{ordinal}:{index}".encode()).hexdigest()[:10]
    arguments = {
        f"field_{position}_{salt[:4]}": value
        for position, value in enumerate(spec.arguments.values())
    }
    return {"tool": f"operation_{salt}", "arguments": arguments}


def _split_wording(value: str, split: str) -> str:
    result = value
    for source, replacement in _SPLIT_WORDING_REPLACEMENTS[split]:
        result = result.replace(source, replacement)
    suffix = {
        "train": " Treat this as live-state guidance.",
        "valid": " Treat this as present-state guidance.",
        "test": " Treat this as up-to-date state guidance.",
    }[split]
    return result + suffix


def _emitted_spec(
    spec: CandidateSpec,
    *,
    split: str,
    pair: str,
    ordinal: int,
    index: int,
    masked: bool,
) -> dict[str, Any]:
    return {
        "call": _masked_call(
            spec,
            split=split,
            pair=pair,
            ordinal=ordinal,
            index=index,
            masked=masked,
        ),
        "purpose": _split_wording(spec.purpose, split),
        "risk": spec.risk,
        "authorized": spec.authorized,
        "redundant": spec.redundant,
        "proof": _split_wording(spec.proof, split),
        "cleanup": spec.cleanup,
    }


def _ordered_candidates(
    specs: Sequence[tuple[int, dict[str, Any]]],
    *,
    target_index: int,
    variant: int,
) -> tuple[list[dict[str, Any]], int, int, dict[int, str]]:
    cardinality = len(specs)
    target_id, target_position = divmod(variant, cardinality)
    target = next(value for original, value in specs if original == target_index)
    distractors = [value for original, value in specs if original != target_index]
    rotation = (target_id + target_position) % len(distractors) if distractors else 0
    distractors = distractors[rotation:] + distractors[:rotation]
    ordered = list(distractors)
    ordered.insert(target_position, target)
    remaining_ids = [
        candidate_id for candidate_id in range(cardinality) if candidate_id != target_id
    ]
    if remaining_ids:
        id_rotation = target_position % len(remaining_ids)
        remaining_ids = remaining_ids[id_rotation:] + remaining_ids[:id_rotation]
    id_cursor = iter(remaining_ids)
    candidates: list[dict[str, Any]] = []
    semantic_tools: dict[int, str] = {}
    for value in ordered:
        candidate_id = target_id if value is target else next(id_cursor)
        semantic_tools[candidate_id] = str(value.pop("_semantic_tool"))
        candidates.append({"id": candidate_id, **value})
        value["_semantic_tool"] = semantic_tools[candidate_id]
    for candidate in candidates:
        candidate.pop("_semantic_tool", None)
    return candidates, target_id, target_position, semantic_tools


def _selected_indices(target_index: int, cardinality: int) -> tuple[int, ...]:
    if cardinality not in RECOVERY_CARDINALITIES:
        raise ValueError(f"unsupported recovery cardinality: {cardinality}")
    pair_target = 1 - target_index if target_index in {0, 1} else 0
    selected = [target_index, pair_target]
    for index in range(4):
        if index not in selected and len(selected) < cardinality:
            selected.append(index)
    return tuple(selected)


def _build_record(
    *,
    split: str,
    family: str,
    ordinal: int,
    variant: int,
) -> dict[str, Any]:
    pair = _PAIR_BY_FAMILY[family]
    entity_pool = _SPLIT_ENTITIES[split]
    pair_index = tuple(_PAIR_FAMILIES).index(pair)
    entity = f"{entity_pool[(ordinal + pair_index * 3) % len(entity_pool)]} Card {ordinal:03d}"
    cardinality = RECOVERY_CARDINALITIES[ordinal % len(RECOVERY_CARDINALITIES)]
    masked = ordinal % 2 == 1
    specs, target_index = _pair_specs(pair, entity, family)
    selected = _selected_indices(target_index, cardinality)
    emitted: list[tuple[int, dict[str, Any]]] = []
    for index in selected:
        value = _emitted_spec(
            specs[index],
            split=split,
            pair=pair,
            ordinal=ordinal,
            index=index,
            masked=masked,
        )
        value["_semantic_tool"] = specs[index].semantic_tool
        emitted.append((index, value))
    candidates, target_id, target_position, semantic_tools = _ordered_candidates(
        emitted,
        target_index=target_index,
        variant=variant,
    )
    state = _state(split, family, entity, ordinal)
    state["observation"]["element_count"] = cardinality
    state["candidates"] = candidates
    group_id = f"recovery-{split}-{family}-{ordinal:03d}"
    # Keep synthetic identifiers below the fail-closed validator's 40-character
    # opaque-token threshold. Human-readable family names remain in metadata.
    pair_id = f"r5-{split[0]}-p{pair_index}-{ordinal:03d}"
    episode_id = f"ep-r5-{split[0]}-{tuple(_PAIR_FAMILIES).index(pair)}-{ordinal:03d}"
    return {
        "id": f"fg-v5-{split}-{family}-{ordinal:03d}-v{variant:02d}",
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
            "episode_id": episode_id,
            "counterfactual_pair_id": pair_id,
            "step": _PAIR_FAMILIES[pair].index(family),
            "steps_total": 2,
            "variant": variant,
            "intent": state["goal"],
            "family": family,
            "label": family,
            "scenario_kind": "recovery_counterfactual",
            "criticality": "critical"
            if family not in {"fresh_observation", "relevant_target"}
            else "normal",
            "template_profile": "function_masked_v5" if masked else "recovery_counterfactual_v5",
            "target_candidate_id": target_id,
            "target_position": target_position,
            "target_call": {
                "tool": specs[target_index].semantic_tool,
                "arguments": copy.deepcopy(dict(specs[target_index].arguments)),
            },
            "tool_name": specs[target_index].semantic_tool,
            "candidate_semantic_tools": {str(key): value for key, value in semantic_tools.items()},
            "cardinality": cardinality,
            "function_masked": masked,
            "curriculum_version": "v5",
        },
    }


def build_recovery_rows(
    groups_per_family: Mapping[str, int] | None = None,
    *,
    seed: int = RECOVERY_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Build the v5-only recovery augmentation."""

    counts = dict(RECOVERY_GROUPS_PER_FAMILY if groups_per_family is None else groups_per_family)
    if tuple(counts) != ("train", "valid", "test"):
        raise ValueError("groups_per_family must contain train, valid, and test in that order")
    result: dict[str, list[dict[str, Any]]] = {}
    for split, group_count in counts.items():
        if group_count <= 0 or group_count % 6:
            raise ValueError(
                "every recovery split needs a positive multiple of six groups per family"
            )
        rows: list[dict[str, Any]] = []
        for family in RECOVERY_FAMILIES:
            for ordinal in range(group_count):
                cardinality = RECOVERY_CARDINALITIES[ordinal % len(RECOVERY_CARDINALITIES)]
                for variant in range(cardinality * cardinality):
                    row = _build_record(
                        split=split,
                        family=family,
                        ordinal=ordinal,
                        variant=variant,
                    )
                    findings = privacy_violations(row, denylist=denylist)
                    if findings:
                        raise ValueError(
                            f"privacy audit failed for {row['id']}: {'; '.join(findings)}"
                        )
                    rows.append(row)
        random.Random(_stable_seed("recovery-row-order-v5", seed, split)).shuffle(rows)
        result[split] = rows
    audit_recovery_boundaries(result)
    return result


def audit_recovery_boundaries(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed on split leakage, shortcut IDs, masking, and early-finish labels."""

    payload_owner: dict[str, str] = {}
    group_owner: dict[str, str] = {}
    entity_owner: dict[str, str] = {}
    wording_owner: dict[str, str] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    split_stats: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        rows = dataset.get(split)
        if rows is None:
            raise ValueError(f"recovery dataset is missing split {split}")
        masked_rows = 0
        family_counts: Counter[str] = Counter()
        cardinalities: Counter[int] = Counter()
        for row in rows:
            metadata = row["metadata"]
            group_id = str(metadata["group_id"])
            owner = group_owner.setdefault(group_id, split)
            if owner != split:
                raise ValueError("recovery semantic group crosses splits")
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            payload_split = payload_owner.setdefault(payload, split)
            if payload_split != split:
                raise ValueError("recovery learning payload crosses splits")
            state = json.loads(str(row["messages"][1]["content"]))
            entity = str(state["goal"]).removeprefix("Open ").split(",", 1)[0].casefold()
            entity_split = entity_owner.setdefault(entity, split)
            if entity_split != split:
                raise ValueError("recovery entity vocabulary crosses splits")
            if any(label.casefold() in payload.casefold() for label in RESERVED_SMOKE_LABELS):
                raise ValueError("reserved production-smoke label leaked into v5 recovery rows")
            target_id = int(metadata["target_candidate_id"])
            target = next(
                candidate for candidate in state["candidates"] if candidate["id"] == target_id
            )
            if target.get("authorized") is not True or target.get("redundant") is True:
                raise ValueError("v5 recovery oracle selected an invalid candidate")
            family = str(metadata["family"])
            for candidate in state["candidates"]:
                wording = _canonical_json(
                    {"purpose": candidate["purpose"], "proof": candidate["proof"]}
                )
                wording_split = wording_owner.setdefault(wording, split)
                if wording_split != split:
                    raise ValueError("recovery purpose/proof wording crosses splits")
            semantic_target = str(metadata["tool_name"])
            if semantic_target == "session_finish":
                observation = state["observation"]
                if (
                    observation.get("outcome") != "known"
                    or observation.get("goal_checkpoint_reached") is not True
                    or any(
                        "required" in str(candidate.get("cleanup"))
                        for candidate in state["candidates"]
                    )
                ):
                    raise ValueError("session_finish is an oracle before proof and cleanup")
            masked_rows += bool(metadata["function_masked"])
            family_counts[family] += 1
            cardinalities[int(metadata["cardinality"])] += 1
            groups[group_id].append(row)

        if masked_rows * 2 != len(rows):
            raise ValueError("v5 function masking must cover exactly half of every split")
        split_stats[split] = {
            "rows": len(rows),
            "families": dict(sorted(family_counts.items())),
            "cardinalities": dict(sorted(cardinalities.items())),
            "masked_rows": masked_rows,
        }

    for group_id, variants in groups.items():
        cardinality = int(variants[0]["metadata"]["cardinality"])
        if len(variants) != cardinality * cardinality:
            raise ValueError(f"recovery group {group_id} lacks its full ID/order square")
        target_ids = Counter(int(row["metadata"]["target_candidate_id"]) for row in variants)
        target_positions = Counter(int(row["metadata"]["target_position"]) for row in variants)
        expected = Counter(dict.fromkeys(range(cardinality), cardinality))
        if target_ids != expected or target_positions != expected:
            raise ValueError(f"recovery group {group_id} leaks target ID or position")

    return {
        "passed": True,
        "literal_payloads_disjoint": True,
        "semantic_groups_disjoint": True,
        "entity_labels_disjoint": True,
        "purpose_and_proof_wording_disjoint": True,
        "function_masking_ratio": 0.5,
        "irrelevance_family_ratio": 1 / len(RECOVERY_FAMILIES),
        "no_premature_finish_oracle": True,
        "splits": split_stats,
    }


def build_v5_dataset(
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    production_group_counts: Mapping[str, Mapping[int, int]] | None = None,
    recovery_groups_per_family: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    recovery_seed: int = RECOVERY_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Return frozen v4 rows plus the v5 recovery augmentation."""

    base = build_v4_dataset(
        base_split_sizes,
        production_group_counts=production_group_counts,
        seed=seed,
        denylist=denylist,
    )
    recovery = build_recovery_rows(
        recovery_groups_per_family,
        seed=recovery_seed,
        denylist=denylist,
    )
    dataset: dict[str, list[dict[str, Any]]] = {}
    seen_payloads: set[str] = set()
    seen_groups: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        rows = [*base[split], *recovery[split]]
        for row in rows:
            payload = _canonical_json({"messages": row["messages"], "tools": row["tools"]})
            if payload in seen_payloads:
                raise ValueError(f"duplicate v5 learning payload: {row['id']}")
            seen_payloads.add(payload)
            group_id = str(row["metadata"]["group_id"])
            owner = seen_groups.setdefault(group_id, split)
            if owner != split:
                raise ValueError(f"semantic group {group_id} crosses {owner} and {split}")
        random.Random(_stable_seed("combined-row-order-v5", recovery_seed, split)).shuffle(rows)
        dataset[split] = rows
    return dataset


def _split_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode("utf-8")


def write_v5_dataset(
    output_dir: str | Path,
    base_split_sizes: Mapping[str, int] | None = None,
    *,
    production_group_counts: Mapping[str, Mapping[int, int]] | None = None,
    recovery_groups_per_family: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    recovery_seed: int = RECOVERY_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, Any]:
    """Write v5 and a manifest that pins the unchanged v4 foundation."""

    output = Path(output_dir)
    base = build_v4_dataset(
        base_split_sizes,
        production_group_counts=production_group_counts,
        seed=seed,
        denylist=denylist,
    )
    recovery = build_recovery_rows(
        recovery_groups_per_family,
        seed=recovery_seed,
        denylist=denylist,
    )
    dataset = build_v5_dataset(
        base_split_sizes,
        production_group_counts=production_group_counts,
        recovery_groups_per_family=recovery_groups_per_family,
        seed=seed,
        recovery_seed=recovery_seed,
        denylist=denylist,
    )
    statistics = dataset_statistics(dataset)
    recovery_audit = audit_recovery_boundaries(recovery)
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
            "base_v4_records": len(base[split]),
            "recovery_v5_records": len(recovery[split]),
            **statistics["splits"][split],
        }
    combined_hash = hashlib.sha256(
        "".join(entries[split]["sha256"] for split in ("train", "valid", "test")).encode()
    ).hexdigest()
    manifest = {
        "format": "functiongemma-aua-candidate-policy-v5",
        "seed": seed,
        "recovery_seed": recovery_seed,
        "selection_function": "select_candidate(candidate_id: integer)",
        "split_policy": (
            "frozen v4 groups plus split-exclusive recovery counterfactual pairs; "
            "assignment precedes dense ID and order expansion"
        ),
        "base_v4": {
            "format": "functiongemma-aua-candidate-policy-v4",
            "split_sha256": base_hashes,
            "records": {split: len(base[split]) for split in base},
        },
        "recovery_v5": {
            "families": list(RECOVERY_FAMILIES),
            "cardinalities": list(RECOVERY_CARDINALITIES),
            "groups_per_family": dict(
                RECOVERY_GROUPS_PER_FAMILY
                if recovery_groups_per_family is None
                else recovery_groups_per_family
            ),
            "techniques": [
                "minimal counterfactual recovery pairs",
                "50 percent deterministic function and argument masking",
                "10 percent irrelevant-target recovery family",
                "external proof and cleanup oracle",
            ],
            "leakage_and_oracle_audit": recovery_audit,
        },
        "total_records": statistics["total_records"],
        "ratios": statistics["ratios"],
        "privacy": {
            "passed": True,
            "checks": [
                "denylist",
                "host paths and direct identifiers",
                "fictional public vocabulary",
                "high-entropy opaque tokens",
            ],
        },
        "dataset_sha256": combined_hash,
        "splits": entries,
    }
    _atomic_write(output / "manifest.json", f"{_canonical_json(manifest)}\n".encode())
    return manifest
