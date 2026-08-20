from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.live_context_curriculum import (
    LIVE_CONTEXT_VARIANTS,
    RESERVED_LIVE_SMOKE_LABELS,
    audit_live_context_boundaries,
    build_live_context_rows,
)

from android_ui_analyser.policy import SELECTOR_POLICY

COMPACT_GROUPS = {"train": 1, "valid": 1, "test": 1}


def _state(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def test_live_context_rows_use_exact_production_shape_and_all_permutations() -> None:
    dataset = build_live_context_rows(COMPACT_GROUPS)
    audit = audit_live_context_boundaries(dataset)

    assert audit["passed"] is True
    assert audit["uses_exact_policy_messages"] is True
    assert audit["reserved_live_smoke_labels_absent"] is True
    assert {split: len(rows) for split, rows in dataset.items()} == {
        "train": LIVE_CONTEXT_VARIANTS,
        "valid": LIVE_CONTEXT_VARIANTS,
        "test": LIVE_CONTEXT_VARIANTS,
    }
    for split, rows in dataset.items():
        assert all(
            row["messages"][0] == {"role": "developer", "content": SELECTOR_POLICY} for row in rows
        )
        assert all(row["metadata"]["split"] == split for row in rows)
        assert all(row["metadata"]["permutations_total"] == LIVE_CONTEXT_VARIANTS for row in rows)
        assert all(_state(row)["fixture_ref"] == "aua-live-policy-v1" for row in rows)


def test_each_semantic_group_is_id_and_position_counterbalanced() -> None:
    dataset = build_live_context_rows(COMPACT_GROUPS)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rows in dataset.values():
        for row in rows:
            groups[row["metadata"]["group_id"]].append(row)

    assert len(groups) == 3
    for rows in groups.values():
        assert len({tuple(row["metadata"]["candidate_order"]) for row in rows}) == 24
        assert len({tuple(row["metadata"]["id_assignment"]) for row in rows}) == 24
        assert Counter(row["metadata"]["target_candidate_id"] for row in rows) == Counter(
            {0: 144, 1: 144, 2: 144, 3: 144}
        )
        assert Counter(row["metadata"]["target_position"] for row in rows) == Counter(
            {0: 144, 1: 144, 2: 144, 3: 144}
        )


def test_split_selectors_are_disjoint_and_smoke_labels_are_reserved() -> None:
    dataset = build_live_context_rows(COMPACT_GROUPS)
    values: dict[str, set[str]] = {}
    compact = json.dumps(dataset, sort_keys=True).casefold()

    for label in RESERVED_LIVE_SMOKE_LABELS:
        assert label.casefold() not in compact
    for split, rows in dataset.items():
        values[split] = {
            str(value).casefold()
            for row in rows
            for candidate in _state(row)["candidates"]
            for value in candidate["call"]["arguments"].values()
        }
    assert values["train"].isdisjoint(values["valid"])
    assert values["train"].isdisjoint(values["test"])
    assert values["valid"].isdisjoint(values["test"])
