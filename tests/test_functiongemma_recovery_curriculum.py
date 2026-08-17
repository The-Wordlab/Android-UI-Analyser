from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.production_curriculum import build_v4_dataset
from experiments.functiongemma.recovery_curriculum import (
    RECOVERY_CARDINALITIES,
    RECOVERY_FAMILIES,
    audit_recovery_boundaries,
    build_recovery_rows,
    build_v5_dataset,
    write_v5_dataset,
)

COMPACT_BASE = {"train": 128, "valid": 128, "test": 128}
COMPACT_PRODUCTION = {
    "train": {2: 2, 3: 2, 4: 2},
    "valid": {2: 1, 3: 1, 4: 1},
    "test": {2: 1, 3: 1, 4: 1},
}
COMPACT_RECOVERY = {"train": 6, "valid": 6, "test": 6}


def _state(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def _payload(row: dict) -> str:
    return json.dumps(
        {"messages": row["messages"], "tools": row["tools"]},
        sort_keys=True,
        separators=(",", ":"),
    )


def test_v5_is_additive_and_preserves_every_v4_learning_row() -> None:
    v4 = build_v4_dataset(
        COMPACT_BASE,
        production_group_counts=COMPACT_PRODUCTION,
    )
    v5 = build_v5_dataset(
        COMPACT_BASE,
        production_group_counts=COMPACT_PRODUCTION,
        recovery_groups_per_family=COMPACT_RECOVERY,
    )

    for split in v4:
        v4_by_id = {row["id"]: row for row in v4[split]}
        inherited = {
            row["id"]: row for row in v5[split] if row["metadata"].get("curriculum_version") != "v5"
        }
        assert inherited == v4_by_id


def test_recovery_rows_are_counterbalanced_and_production_shaped() -> None:
    dataset = build_recovery_rows(COMPACT_RECOVERY)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rows in dataset.values():
        for row in rows:
            state = _state(row)
            cardinality = row["metadata"]["cardinality"]
            assert cardinality in RECOVERY_CARDINALITIES
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
            assert all(
                set(candidate)
                == {
                    "id",
                    "call",
                    "purpose",
                    "risk",
                    "authorized",
                    "redundant",
                    "proof",
                    "cleanup",
                }
                for candidate in state["candidates"]
            )
            groups[row["metadata"]["group_id"]].append(row)

    for variants in groups.values():
        cardinality = variants[0]["metadata"]["cardinality"]
        assert len(variants) == cardinality * cardinality
        expected = Counter(dict.fromkeys(range(cardinality), cardinality))
        assert Counter(row["metadata"]["target_candidate_id"] for row in variants) == expected
        assert Counter(row["metadata"]["target_position"] for row in variants) == expected

        semantic_states = []
        candidate_sets = []
        target_tools = []
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
            target_tools.append(row["metadata"]["tool_name"])
        assert all(state == semantic_states[0] for state in semantic_states)
        assert all(candidate_set == candidate_sets[0] for candidate_set in candidate_sets)
        assert all(tool == target_tools[0] for tool in target_tools)


def test_counterfactual_pairs_reverse_only_after_relevant_state_changes() -> None:
    dataset = build_recovery_rows(COMPACT_RECOVERY)
    representatives: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for split, rows in dataset.items():
        if split != "test":
            continue
        for row in rows:
            if row["metadata"]["variant"] != 0:
                continue
            pair = row["metadata"]["counterfactual_pair_id"]
            ordinal = int(pair.rsplit("-", 1)[1])
            representatives[(pair.rsplit("-", 1)[0], ordinal)][row["metadata"]["family"]] = row

    assert representatives
    for paired in representatives.values():
        assert len(paired) == 2
        rows = list(paired.values())
        assert rows[0]["metadata"]["tool_name"] != rows[1]["metadata"]["tool_name"]
        assert rows[0]["metadata"]["cardinality"] == rows[1]["metadata"]["cardinality"]
        assert rows[0]["metadata"]["function_masked"] == rows[1]["metadata"]["function_masked"]


def test_finish_is_never_an_oracle_until_proof_and_cleanup_are_complete() -> None:
    dataset = build_recovery_rows(COMPACT_RECOVERY)
    finish_families: set[str] = set()
    for rows in dataset.values():
        for row in rows:
            if row["metadata"]["tool_name"] != "session_finish":
                continue
            state = _state(row)
            assert state["observation"]["outcome"] == "known"
            assert state["observation"]["goal_checkpoint_reached"] is True
            assert not any(
                "remains required" in candidate["cleanup"] for candidate in state["candidates"]
            )
            finish_families.add(row["metadata"]["family"])
    assert finish_families == {"terminal_ready", "cleanup_complete"}


def test_function_masking_and_irrelevance_match_the_declared_v5_mix() -> None:
    dataset = build_recovery_rows(COMPACT_RECOVERY)
    audit = audit_recovery_boundaries(dataset)
    assert audit["passed"] is True
    assert audit["function_masking_ratio"] == 0.5
    assert audit["irrelevance_family_ratio"] == 0.1
    assert audit["no_premature_finish_oracle"] is True
    assert audit["purpose_and_proof_wording_disjoint"] is True

    for rows in dataset.values():
        assert sum(row["metadata"]["function_masked"] for row in rows) * 2 == len(rows)
        assert {row["metadata"]["family"] for row in rows} == set(RECOVERY_FAMILIES)
        for row in rows:
            tools = {candidate["call"]["tool"] for candidate in _state(row)["candidates"]}
            if row["metadata"]["function_masked"]:
                assert all(tool.startswith("operation_") for tool in tools)
            else:
                assert not any(tool.startswith("operation_") for tool in tools)


def test_v5_manifest_is_deterministic_and_pins_v4_bytes(tmp_path: Path) -> None:
    first = write_v5_dataset(
        tmp_path / "first",
        COMPACT_BASE,
        production_group_counts=COMPACT_PRODUCTION,
        recovery_groups_per_family=COMPACT_RECOVERY,
    )
    second = write_v5_dataset(
        tmp_path / "second",
        COMPACT_BASE,
        production_group_counts=COMPACT_PRODUCTION,
        recovery_groups_per_family=COMPACT_RECOVERY,
    )
    v4 = build_v4_dataset(
        COMPACT_BASE,
        production_group_counts=COMPACT_PRODUCTION,
    )

    assert first == second
    assert first["format"] == "functiongemma-aua-candidate-policy-v5"
    assert first["recovery_v5"]["leakage_and_oracle_audit"]["passed"] is True
    for split in ("train", "valid", "test"):
        payload = (tmp_path / "first" / f"{split}.jsonl").read_bytes()
        assert payload == (tmp_path / "second" / f"{split}.jsonl").read_bytes()
        assert first["splits"][split]["sha256"] == hashlib.sha256(payload).hexdigest()
        v4_payload = "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in v4[split]
        ).encode()
        assert first["base_v4"]["split_sha256"][split] == hashlib.sha256(v4_payload).hexdigest()


def test_v5_splits_have_no_literal_learning_overlap() -> None:
    dataset = build_recovery_rows(COMPACT_RECOVERY)
    payloads = {split: {_payload(row) for row in rows} for split, rows in dataset.items()}
    assert payloads["train"].isdisjoint(payloads["valid"])
    assert payloads["train"].isdisjoint(payloads["test"])
    assert payloads["valid"].isdisjoint(payloads["test"])


def test_v5_rows_do_not_emit_secret_shaped_identifiers() -> None:
    high_entropy = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_+/=-]{40,}(?![A-Za-z0-9])")
    dataset = build_recovery_rows(COMPACT_RECOVERY)

    for split, rows in dataset.items():
        for row in rows:
            compact = json.dumps(row, sort_keys=True, separators=(",", ":"))
            assert high_entropy.search(compact) is None, (split, row["id"])
