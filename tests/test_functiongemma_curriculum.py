from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.curriculum import (
    DEFAULT_SEED,
    DEFAULT_SPLIT_SIZES,
    LABELS,
    PRESERVED_V1_LABELS,
    PRESERVED_V2_LABELS,
    PUBLIC_AUA_ARGUMENTS,
    SEQUENTIAL_CURRICULUM_SEEDS,
    SEQUENTIAL_LABELS,
    VARIANTS_PER_GROUP,
    build_dataset,
    build_group_assignments,
    dataset_statistics,
    iter_candidate_calls,
    privacy_violations,
    write_dataset,
)

from android_ui_analyser.mcp_server import _tool_definitions


@pytest.fixture(scope="module")
def compact_dataset():
    return build_dataset({"train": 128, "valid": 128, "test": 128})


def test_default_curriculum_contract_is_balanced_and_grouped() -> None:
    assert DEFAULT_SPLIT_SIZES == {"train": 12_288, "valid": 2_048, "test": 2_048}
    dataset = build_dataset()
    stats = dataset_statistics(dataset)

    assert stats["total_records"] == 16_384
    assert stats["ratios"]["recovery_or_counterfactual"] >= 0.25
    assert stats["ratios"]["cleanup_or_terminal"] >= 0.15

    seen_groups: set[str] = set()
    seen_payloads: set[str] = set()
    for split, expected_size in DEFAULT_SPLIT_SIZES.items():
        rows = dataset[split]
        assert len(rows) == expected_size
        label_counts = Counter(row["metadata"]["label"] for row in rows)
        assert set(label_counts) == set(LABELS)
        assert len(set(label_counts.values())) == 1

        groups = {row["metadata"]["group_id"] for row in rows}
        assert not (seen_groups & groups)
        seen_groups |= groups
        assert all(
            sum(row["metadata"]["group_id"] == group_id for row in rows) == VARIANTS_PER_GROUP
            for group_id in groups
        )

        for row in rows:
            payload = json.dumps(
                {"messages": row["messages"], "tools": row["tools"]},
                sort_keys=True,
                separators=(",", ":"),
            )
            assert payload not in seen_payloads
            seen_payloads.add(payload)


def test_rows_declare_only_the_candidate_selector(compact_dataset) -> None:
    target_positions: dict[str, set[int]] = defaultdict(set)
    target_ids: dict[str, set[int]] = defaultdict(set)

    for rows in compact_dataset.values():
        for row in rows:
            assert row["messages"][0]["content"].startswith(
                "You are a model that can do function calling with the following functions."
            )
            assert len(row["tools"]) == 1
            function = row["tools"][0]["function"]
            assert function["name"] == "select_candidate"
            assert function["parameters"]["required"] == ["candidate_id"]

            assistant = row["messages"][-1]
            assert assistant["role"] == "assistant"
            assert len(assistant["tool_calls"]) == 1
            tool_call = assistant["tool_calls"][0]["function"]
            assert tool_call["name"] == "select_candidate"
            selected = tool_call["arguments"]["candidate_id"]

            state = json.loads(row["messages"][1]["content"])
            ids = [candidate["id"] for candidate in state["candidates"]]
            assert len(ids) == len(set(ids))
            assert 4 <= len(ids) <= 8
            assert selected in ids
            assert selected == row["metadata"]["target_candidate_id"]
            group = row["metadata"]["group_id"]
            target_positions[group].add(ids.index(selected))
            target_ids[group].add(selected)

    assert all(len(positions) > 1 for positions in target_positions.values())
    assert all(len(ids) > 1 for ids in target_ids.values())


def test_same_state_id_permutations_force_shortcuts_to_chance(compact_dataset) -> None:
    """Visible state without candidate contents must contain no target-ID signal."""
    invariant_sets: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for rows in compact_dataset.values():
        for row in rows:
            state = json.loads(row["messages"][1]["content"])
            candidates = state.pop("candidates")
            invariant_sets[(row["metadata"]["group_id"], json.dumps(state, sort_keys=True))].append(
                (row, candidates)
            )

    assert invariant_sets
    for variants in invariant_sets.values():
        assert len(variants) == 4
        targets = [row["metadata"]["target_candidate_id"] for row, _ in variants]
        assert Counter(targets) == Counter({0: 1, 1: 1, 2: 1, 3: 1})

        normalized_sets = []
        target_calls = []
        permutations = []
        for row, candidates in variants:
            normalized = []
            permutation = []
            for candidate in candidates:
                candidate_without_id = deepcopy(candidate)
                candidate_id = candidate_without_id.pop("id")
                normalized.append(candidate_without_id)
                permutation.append((candidate_id, candidate_without_id["call"]["tool"]))
            normalized_sets.append(sorted(json.dumps(item, sort_keys=True) for item in normalized))
            permutations.append(tuple(permutation))
            target_calls.append(row["metadata"]["target_call"])

        assert all(candidate_set == normalized_sets[0] for candidate_set in normalized_sets)
        assert all(target_call == target_calls[0] for target_call in target_calls)
        assert len(set(permutations)) == 4


def test_v3_preserves_every_v1_group_split_membership() -> None:
    expected_hashes = {
        "train": "0d6fda91ff9d7284f08f4a349d042317588ca2d450d8c0b81b25614558c54a78",
        "valid": "53c0db80ad0c77b733bdbc6ea52a8c51bd4a99f468a9acbd8597d8457d52e86b",
        "test": "2b435ecb7c5f96baa59bf702e873b0b7d2b4c5293d8cd401cdc3bbbdbd9609aa",
    }
    assignments = build_group_assignments()
    for split, groups in assignments.items():
        preserved = sorted(group.group_id for group in groups if group.label in PRESERVED_V1_LABELS)
        payload = ("\n".join(preserved) + "\n").encode()
        assert hashlib.sha256(payload).hexdigest() == expected_hashes[split]


def test_v3_preserves_every_v2_group_split_membership() -> None:
    expected_hashes = {
        "train": "809879bc3e1e9807cb636b1ba9a2755ee08c04e03c465ef2c3cdec160ea29ec8",
        "valid": "87851e87d131177090706ed2e35cb20f7938966f05c5c472eb5e8e149f9d33d9",
        "test": "993006092481e15f02c090eabecd33a140ea700b9809609dbbc86896952b86ab",
    }
    assignments = build_group_assignments()
    for split, groups in assignments.items():
        preserved = sorted(group.group_id for group in groups if group.label in PRESERVED_V2_LABELS)
        payload = ("\n".join(preserved) + "\n").encode()
        assert hashlib.sha256(payload).hexdigest() == expected_hashes[split]


def test_v2_adds_verified_offline_and_ambiguous_recovery_oracles(compact_dataset) -> None:
    rows = [row for split_rows in compact_dataset.values() for row in split_rows]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["metadata"]["label"]].append(row)

    expected = {
        "enter_verified_offline_state": "network_offline",
        "recover_ambiguous_mutation": "analyze_screen",
    }
    for label, tool in expected.items():
        assert by_label[label]
        assert {row["metadata"]["tool_name"] for row in by_label[label]} == {tool}
        assert {row["metadata"]["criticality"] for row in by_label[label]} == {"critical"}

    offline = by_label["enter_verified_offline_state"][0]
    offline_state = json.loads(offline["messages"][1]["content"])
    assert offline_state["session"]["active"] is True
    assert offline_state["session"]["owns_reversible_state"] is True
    offline_candidates = offline_state["candidates"]
    offline_target = next(
        candidate
        for candidate in offline_candidates
        if candidate["id"] == offline["metadata"]["target_candidate_id"]
    )
    assert offline_target["call"] == {
        "tool": "network_offline",
        "arguments": {"verify": True, "timeout_ms": 10_000},
    }
    assert "network_restore" in offline_target["cleanup"]
    assert {candidate["call"]["tool"] for candidate in offline_candidates} == {
        "network_offline",
        "network_status",
        "session_finish",
    }
    assert any(candidate["redundant"] for candidate in offline_candidates)

    ambiguous = by_label["recover_ambiguous_mutation"][0]
    ambiguous_state = json.loads(ambiguous["messages"][1]["content"])
    assert ambiguous_state["observation"]["fresh"] is False
    ambiguous_candidates = ambiguous_state["candidates"]
    assert {candidate["call"]["tool"] for candidate in ambiguous_candidates} == {
        "analyze_screen",
        "tap_and_analyze",
        "session_progress",
        "session_finish",
    }
    replay = next(
        candidate
        for candidate in ambiguous_candidates
        if candidate["call"]["tool"] == "tap_and_analyze"
    )
    assert replay["authorized"] is False


def test_v3_covers_all_sequential_runtime_phases_with_source_oracles(compact_dataset) -> None:
    expected = {
        "sequence_start": (
            "not_started",
            "session_start",
            Counter(
                {
                    "session_start": 1,
                    "analyze_screen": 1,
                    "network_offline": 1,
                    "session_finish": 1,
                }
            ),
        ),
        "sequence_prepare_offline": (
            "prepare_offline",
            "network_offline",
            Counter(
                {
                    "network_offline": 1,
                    "network_status": 1,
                    "analyze_screen": 1,
                    "session_finish": 1,
                }
            ),
        ),
        "sequence_open_item": (
            "open_record",
            "tap_and_analyze",
            Counter({"tap_and_analyze": 2, "analyze_screen": 1, "session_finish": 1}),
        ),
        "sequence_recover_unknown": (
            "recover_unknown",
            "analyze_screen",
            Counter(
                {
                    "analyze_screen": 1,
                    "tap_and_analyze": 1,
                    "session_progress": 1,
                    "session_finish": 1,
                }
            ),
        ),
        "sequence_restore": (
            "restore_environment",
            "network_restore",
            Counter(
                {
                    "network_restore": 1,
                    "network_status": 1,
                    "network_offline": 1,
                    "session_finish": 1,
                }
            ),
        ),
        "sequence_finish": (
            "finish",
            "session_finish",
            Counter(
                {
                    "session_finish": 1,
                    "session_review": 1,
                    "analyze_screen": 1,
                    "network_restore": 1,
                }
            ),
        ),
    }
    assert DEFAULT_SEED not in SEQUENTIAL_CURRICULUM_SEEDS
    assert len(set(SEQUENTIAL_CURRICULUM_SEEDS)) == len(SEQUENTIAL_CURRICULUM_SEEDS)
    for _split, rows in compact_dataset.items():
        sequential = [row for row in rows if row["metadata"]["label"] in SEQUENTIAL_LABELS]
        assert len(sequential) == len(SEQUENTIAL_LABELS) * VARIANTS_PER_GROUP
        for label, (phase, tool, bundle) in expected.items():
            label_rows = [row for row in sequential if row["metadata"]["label"] == label]
            assert len(label_rows) == VARIANTS_PER_GROUP
            assert {row["metadata"]["tool_name"] for row in label_rows} == {tool}
            assert {json.loads(row["messages"][1]["content"])["phase"] for row in label_rows} == {
                phase
            }

            for row in label_rows:
                state = json.loads(row["messages"][1]["content"])
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
                assert (
                    Counter(candidate["call"]["tool"] for candidate in state["candidates"])
                    == bundle
                )
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
                target = next(
                    candidate
                    for candidate in state["candidates"]
                    if candidate["id"] == row["metadata"]["target_candidate_id"]
                )
                assert target["authorized"] is True
                assert target["redundant"] is False
                if label in {
                    "sequence_open_item",
                    "sequence_recover_unknown",
                    "sequence_restore",
                }:
                    early_finish = next(
                        candidate
                        for candidate in state["candidates"]
                        if candidate["call"]["tool"] == "session_finish"
                    )
                    assert early_finish["authorized"] is False
                    assert "network_restore" in early_finish["cleanup"]

        prepare = json.loads(
            next(
                row for row in sequential if row["metadata"]["label"] == "sequence_prepare_offline"
            )["messages"][1]["content"]
        )
        analyze = next(
            candidate
            for candidate in prepare["candidates"]
            if candidate["call"]["tool"] == "analyze_screen"
        )
        finish = next(
            candidate
            for candidate in prepare["candidates"]
            if candidate["call"]["tool"] == "session_finish"
        )
        assert analyze["redundant"] is True
        assert finish["authorized"] is False


def test_sequential_episodes_are_split_whole_and_test_is_lexically_held_out(
    compact_dataset,
) -> None:
    episode_splits: dict[str, set[str]] = defaultdict(set)
    sequential_requests: dict[str, set[str]] = defaultdict(set)
    for split, rows in compact_dataset.items():
        sequential = [row for row in rows if row["metadata"]["label"] in SEQUENTIAL_LABELS]
        for row in sequential:
            episode_splits[row["metadata"]["episode_id"]].add(split)
            state = json.loads(row["messages"][1]["content"])
            sequential_requests[split].add(state["request"])
            expected_profile = "heldout_lexical_v3" if split == "test" else "source_oracle_v3"
            assert row["metadata"]["template_profile"] == expected_profile
            if split == "test":
                assert all("=" not in outcome for outcome in state["recent_outcomes"])
            else:
                assert all("=" in outcome for outcome in state["recent_outcomes"])

    assert all(len(splits) == 1 for splits in episode_splits.values())
    assert sequential_requests["test"].isdisjoint(sequential_requests["train"])
    assert sequential_requests["test"].isdisjoint(sequential_requests["valid"])


def test_every_candidate_is_a_current_public_mcp_call_with_flat_args(compact_dataset) -> None:
    public = {tool.name: tool.inputSchema for tool in _tool_definitions()}
    assert set(PUBLIC_AUA_ARGUMENTS) <= set(public)
    for name, expected_keys in PUBLIC_AUA_ARGUMENTS.items():
        actual_keys = set(public[name].get("properties", {}))
        assert expected_keys == actual_keys, f"curriculum schema drift for {name}"

    for rows in compact_dataset.values():
        for row in rows:
            candidates = list(iter_candidate_calls(row))
            for name, arguments in candidates:
                assert name in public
                assert "args" not in arguments
                assert "arguments" not in arguments
                assert set(arguments) <= set(public[name]["properties"])
            assert row["metadata"]["target_call"] in [
                {"tool": name, "arguments": dict(arguments)} for name, arguments in candidates
            ]


def test_generation_and_manifest_hashes_are_deterministic(tmp_path) -> None:
    sizes = {"train": 128, "valid": 128, "test": 128}
    first = write_dataset(tmp_path / "first", sizes, seed=DEFAULT_SEED)
    second = write_dataset(tmp_path / "second", sizes, seed=DEFAULT_SEED)
    assert first == second

    for split in sizes:
        first_bytes = (tmp_path / "first" / f"{split}.jsonl").read_bytes()
        second_bytes = (tmp_path / "second" / f"{split}.jsonl").read_bytes()
        assert first_bytes == second_bytes
        assert hashlib.sha256(first_bytes).hexdigest() == first["splits"][split]["sha256"]
        assert len(first_bytes.splitlines()) == sizes[split]

    persisted = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert persisted == first
    assert persisted["format"] == "functiongemma-aua-candidate-policy-v3"
    assert persisted["privacy"]["passed"] is True
    assert persisted["splits"]["test"]["template_profiles"]["heldout_lexical_v3"] > 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("copied from /Users/example/private/run.json", "local host path"),
        ("contact person@example.org", "email address"),
        ("backend 203.0.113.4", "IPv4 address"),
        ("package com.vendor.actual", "non-fictional package"),
        ("Bearer abc", "denylisted term"),
        ("key QWxhZGRpbjpvcGVuIHNlc2FtZTEyMzQ1Njc4OTA=", "high-entropy opaque token"),
    ],
)
def test_privacy_audit_rejects_non_synthetic_material(value: str, expected: str) -> None:
    assert any(expected in finding for finding in privacy_violations(value))


def test_generated_material_passes_privacy_audit(compact_dataset) -> None:
    assert not [
        (row["id"], findings)
        for rows in compact_dataset.values()
        for row in rows
        if (findings := privacy_violations(row))
    ]


def test_invalid_split_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive multiple"):
        build_dataset({"train": 127, "valid": 128, "test": 128})
