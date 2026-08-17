from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.curriculum import (
    DEFAULT_SEED,
    build_dataset,
    write_dataset,
)
from experiments.functiongemma.production_curriculum import (
    PRODUCTION_CARDINALITIES,
    PRODUCTION_FIXTURE_REF,
    RESERVED_SMOKE_LABELS,
    _label_from_candidate,
    audit_production_boundaries,
    build_production_rows,
    build_v4_dataset,
    write_v4_dataset,
)
from experiments.functiongemma.run_production_smoke import SUBJECTS, TARGET_SUBJECT

COMPACT_BASE = {"train": 128, "valid": 128, "test": 128}
COMPACT_GROUPS = {
    "train": {2: 2, 3: 2, 4: 2},
    "valid": {2: 1, 3: 1, 4: 1},
    "test": {2: 1, 3: 1, 4: 1},
}


def _payload(row: dict) -> str:
    return json.dumps(
        {"messages": row["messages"], "tools": row["tools"]},
        sort_keys=True,
        separators=(",", ":"),
    )


def _state(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def test_v4_is_additive_without_changing_any_v3_row_or_split_membership() -> None:
    v3 = build_dataset(COMPACT_BASE, seed=DEFAULT_SEED)
    v4 = build_v4_dataset(
        COMPACT_BASE,
        production_group_counts=COMPACT_GROUPS,
        seed=DEFAULT_SEED,
    )

    for split in v3:
        v3_by_id = {row["id"]: row for row in v3[split]}
        v4_legacy_by_id = {
            row["id"]: row for row in v4[split] if row["metadata"].get("curriculum_version") != "v4"
        }
        assert v4_legacy_by_id == v3_by_id
        assert {row["metadata"]["group_id"] for row in v4_legacy_by_id.values()} == {
            row["metadata"]["group_id"] for row in v3[split]
        }


def test_production_rows_match_policy_context_shape_and_counterbalance_ids_and_order() -> None:
    rows = build_production_rows(COMPACT_GROUPS)
    groups: dict[str, list[dict]] = defaultdict(list)
    for split_rows in rows.values():
        for row in split_rows:
            state = _state(row)
            cardinality = len(state["candidates"])
            assert cardinality in PRODUCTION_CARDINALITIES
            assert state["fixture_ref"] == PRODUCTION_FIXTURE_REF
            assert set(state) == {
                "fixture_ref",
                "request",
                "goal",
                "phase",
                "observation",
                "recent_outcomes",
                "constraints",
                "candidates",
            }
            assert {candidate["id"] for candidate in state["candidates"]} == set(range(cardinality))
            assert {candidate["call"]["tool"] for candidate in state["candidates"]} == {
                "tap_and_analyze"
            }
            assert all(candidate["risk"] == "safe" for candidate in state["candidates"])
            assert all(candidate["authorized"] for candidate in state["candidates"])
            assert not any(candidate["redundant"] for candidate in state["candidates"])
            groups[row["metadata"]["group_id"]].append(row)

    for variants in groups.values():
        cardinality = len(_state(variants[0])["candidates"])
        assert len(variants) == cardinality * cardinality
        target_ids = Counter(row["metadata"]["target_candidate_id"] for row in variants)
        target_positions = Counter(row["metadata"]["target_position"] for row in variants)
        expected_distribution = Counter(dict.fromkeys(range(cardinality), cardinality))
        assert target_ids == expected_distribution
        assert target_positions == expected_distribution

        semantic_states = []
        candidate_sets = []
        for row in variants:
            state = _state(row)
            candidates = state.pop("candidates")
            semantic_states.append(state)
            normalized = []
            for candidate in candidates:
                value = deepcopy(candidate)
                value.pop("id")
                normalized.append(json.dumps(value, sort_keys=True))
            candidate_sets.append(sorted(normalized))
        assert all(state == semantic_states[0] for state in semantic_states)
        assert all(candidate_set == candidate_sets[0] for candidate_set in candidate_sets)


def test_learning_splits_and_reserved_smoke_semantics_are_strictly_held_out() -> None:
    rows = build_production_rows(COMPACT_GROUPS)
    audit = audit_production_boundaries(rows)
    assert audit["passed"] is True
    assert audit["literal_payloads_disjoint"] is True
    assert audit["semantic_groups_disjoint"] is True
    assert audit["entity_labels_disjoint"] is True
    assert audit["reserved_smoke_labels_absent_from_learning_rows"] is True

    payloads: dict[str, set[str]] = {}
    semantic_groups: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    entities: dict[str, set[str]] = {}
    serialized_learning = ""
    for split, split_rows in rows.items():
        payloads[split] = {_payload(row) for row in split_rows}
        semantic_groups[split] = set()
        entities[split] = set()
        for row in split_rows:
            state = _state(row)
            labels = tuple(
                sorted(_label_from_candidate(candidate) for candidate in state["candidates"])
            )
            target_id = row["metadata"]["target_candidate_id"]
            target = _label_from_candidate(
                next(candidate for candidate in state["candidates"] if candidate["id"] == target_id)
            )
            semantic_groups[split].add((target, labels))
            entities[split].update(labels)
            serialized_learning += row["messages"][1]["content"].casefold()

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert payloads[left].isdisjoint(payloads[right])
        assert semantic_groups[left].isdisjoint(semantic_groups[right])
        assert entities[left].isdisjoint(entities[right])

    assert frozenset(SUBJECTS) == RESERVED_SMOKE_LABELS
    assert TARGET_SUBJECT in RESERVED_SMOKE_LABELS
    assert all(label.casefold() not in serialized_learning for label in SUBJECTS)
    smoke_signature = (TARGET_SUBJECT, tuple(sorted(SUBJECTS)))
    assert all(smoke_signature not in values for values in semantic_groups.values())


def test_v4_manifest_is_deterministic_and_pins_frozen_v3_bytes(tmp_path: Path) -> None:
    first = write_v4_dataset(
        tmp_path / "first",
        COMPACT_BASE,
        production_group_counts=COMPACT_GROUPS,
    )
    second = write_v4_dataset(
        tmp_path / "second",
        COMPACT_BASE,
        production_group_counts=COMPACT_GROUPS,
    )
    legacy = write_dataset(tmp_path / "legacy", COMPACT_BASE)

    assert first == second
    assert first["format"] == "functiongemma-aua-candidate-policy-v4"
    assert first["production_v4"]["leakage_audit"]["passed"] is True
    for split in ("train", "valid", "test"):
        assert (tmp_path / "first" / f"{split}.jsonl").read_bytes() == (
            tmp_path / "second" / f"{split}.jsonl"
        ).read_bytes()
        assert first["base_v3"]["split_sha256"][split] == legacy["splits"][split]["sha256"]
        assert (
            first["splits"][split]["sha256"]
            == hashlib.sha256((tmp_path / "first" / f"{split}.jsonl").read_bytes()).hexdigest()
        )
