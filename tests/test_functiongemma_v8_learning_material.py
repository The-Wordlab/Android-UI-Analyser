from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.v8_learning_material import (
    FAMILIES,
    build_v8_source,
    write_v8_source,
)


def test_v8_source_is_balanced_fictional_and_split_clean() -> None:
    dataset = build_v8_source()

    assert {split: len(rows) for split, rows in dataset.items()} == {
        "train": 800,
        "valid": 100,
        "test": 100,
    }
    groups = {split: {row["group_id"] for row in rows} for split, rows in dataset.items()}
    assert not (groups["train"] & groups["valid"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["valid"] & groups["test"])
    for rows in dataset.values():
        counts = Counter(row["family"] for row in rows)
        assert set(counts) == set(FAMILIES)
        assert len(set(counts.values())) == 1
        payload = json.dumps(rows).casefold()
        for heldout in (
            "notification history",
            "notification cooldown",
            "sound & vibration",
            "battery usage",
            "com.android.settings",
        ):
            assert heldout not in payload


def test_v8_oracles_are_unambiguous_and_handoffs_have_no_target() -> None:
    for rows in build_v8_source().values():
        for row in rows:
            calls = [candidate["call"] for candidate in row["candidates"]]
            oracle = row["oracle"]
            if oracle["kind"] == "select":
                assert calls.count(oracle["call"]) == 1
                assert row["metadata"]["render_status"] == "ready_for_model_renderer"
            else:
                assert oracle == {"kind": "handoff", "reason": "target_absent"}
                assert row["metadata"]["render_status"] == "blocked_until_handoff_protocol"
            ids = sorted(candidate["id"] for candidate in row["candidates"])
            assert ids == list(range(len(ids)))


def test_v8_writer_is_deterministic(tmp_path: Path) -> None:
    first = write_v8_source(tmp_path / "first")
    second = write_v8_source(tmp_path / "second")

    assert first == second
