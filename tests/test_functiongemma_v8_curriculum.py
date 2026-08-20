from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.v8_curriculum import (
    FAMILIES,
    V8_HANDOFF_REASON,
    audit_v8_native_rows,
    build_v8_native_rows,
)

from android_ui_analyser.policy import POLICY_HANDOFF_ID, policy_tools


def _state(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def test_v8_native_rows_are_exact_handoff_aware_runtime_examples() -> None:
    dataset = build_v8_native_rows()
    audit = audit_v8_native_rows(dataset)

    assert audit["passed"] is True
    assert audit["uses_exact_policy_messages"] is True
    assert audit["handoff_candidate_id"] == POLICY_HANDOFF_ID
    assert {split: len(rows) for split, rows in dataset.items()} == {
        "train": 2360,
        "valid": 295,
        "test": 295,
    }
    for rows in dataset.values():
        assert {row["metadata"]["family"] for row in rows} == set(FAMILIES)
        for row in rows:
            state = _state(row)
            assert state["handoff"] == {
                "allowed": True,
                "candidate_id": -1,
                "reason": V8_HANDOFF_REASON,
            }
            assert row["tools"] == policy_tools(allow_handoff=True)
            assert len(state["candidates"]) in {3, 4}
            assert {candidate["id"] for candidate in state["candidates"]} == set(
                range(len(state["candidates"]))
            )


def test_v8_select_and_handoff_groups_are_counterbalanced() -> None:
    dataset = build_v8_native_rows()
    groups: dict[str, list[dict]] = defaultdict(list)
    for rows in dataset.values():
        for row in rows:
            groups[row["metadata"]["group_id"]].append(row)

    assert len(groups) == 250
    assert Counter(rows[0]["metadata"]["family"] for rows in groups.values()) == Counter(
        dict.fromkeys(FAMILIES, 50)
    )
    for rows in groups.values():
        cardinality = rows[0]["metadata"]["cardinality"]
        assert len(rows) == cardinality * cardinality
        if rows[0]["metadata"]["target_outcome"] == "handoff":
            assert {row["metadata"]["target_candidate_id"] for row in rows} == {-1}
            assert {row["metadata"]["target_position"] for row in rows} == {-1}
        else:
            expected = Counter(dict.fromkeys(range(cardinality), cardinality))
            assert Counter(row["metadata"]["target_candidate_id"] for row in rows) == expected
            assert Counter(row["metadata"]["target_position"] for row in rows) == expected


def test_v8_native_rows_keep_live_evaluation_vocabulary_held_out() -> None:
    payload = json.dumps(build_v8_native_rows(), sort_keys=True).casefold()
    for value in (
        "notification history",
        "notification cooldown",
        "sound & vibration",
        "battery usage",
        "com.android.settings",
        "clear text",
        "screen locking sound",
    ):
        assert value.casefold() not in payload
