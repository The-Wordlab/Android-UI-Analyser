from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.semantic_context_curriculum import (
    RESERVED_V7_SMOKE_TERMS,
    audit_semantic_context_boundaries,
    build_semantic_context_rows,
)

from android_ui_analyser.engine import Engine

COMPACT_COUNTS = {
    "train": {2: 1, 3: 1, 4: 1},
    "valid": {2: 1, 3: 1, 4: 1},
    "test": {2: 1, 3: 1, 4: 1},
}


def _state(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def test_v7_rows_use_exact_runtime_goal_and_variable_cardinality() -> None:
    dataset = build_semantic_context_rows(COMPACT_COUNTS)
    audit = audit_semantic_context_boundaries(dataset)

    assert audit["passed"] is True
    assert audit["uses_exact_policy_messages"] is True
    assert audit["reserved_v7_smoke_terms_absent"] is True
    for rows in dataset.values():
        assert len(rows) == 4 + 9 + 16
        assert Counter(len(_state(row)["candidates"]) for row in rows) == Counter(
            {2: 4, 3: 9, 4: 16}
        )
        for row in rows:
            state = _state(row)
            assert state["goal"].startswith("Requested destination: ")
            assert ". Matching evidence: " in state["goal"]
            candidates = [
                SimpleNamespace(purpose=candidate["purpose"]) for candidate in state["candidates"]
            ]
            assert state["goal"] == Engine._policy_selection_goal(  # noqa: SLF001
                row["metadata"]["intent"],
                candidates,
            )
            assert {candidate["id"] for candidate in state["candidates"]} == set(
                range(len(state["candidates"]))
            )


def test_v7_groups_counterbalance_target_id_and_position_without_full_permutation_bloat() -> None:
    dataset = build_semantic_context_rows(COMPACT_COUNTS)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rows in dataset.values():
        for row in rows:
            groups[row["metadata"]["group_id"]].append(row)

    assert len(groups) == 9
    for rows in groups.values():
        cardinality = rows[0]["metadata"]["cardinality"]
        assert len(rows) == cardinality * cardinality
        expected = Counter(dict.fromkeys(range(cardinality), cardinality))
        assert Counter(row["metadata"]["target_candidate_id"] for row in rows) == expected
        assert Counter(row["metadata"]["target_position"] for row in rows) == expected


def test_v7_splits_are_disjoint_and_reserved_smoke_vocabulary_is_absent() -> None:
    dataset = build_semantic_context_rows(COMPACT_COUNTS)
    payload = json.dumps(dataset, sort_keys=True).casefold()
    for term in RESERVED_V7_SMOKE_TERMS:
        assert term.casefold() not in payload

    entities: dict[str, set[str]] = {}
    for split, rows in dataset.items():
        entities[split] = {
            candidate["purpose"].casefold()
            for row in rows
            for candidate in _state(row)["candidates"]
        }
    assert entities["train"].isdisjoint(entities["valid"])
    assert entities["train"].isdisjoint(entities["test"])
    assert entities["valid"].isdisjoint(entities["test"])
