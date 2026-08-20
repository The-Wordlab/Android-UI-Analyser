"""Contract tests for the V9 source-oracle material and its dataset renderer.

These pin the properties that decide whether V9 is trainable at all: every family must build a
case whose oracle is actually offered, no semantic group or goal string may cross a split, and
neither the opaque candidate ID nor the list position may predict the answer.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.v9_curriculum import (
    build_split,
    check_split_isolation,
    render_case,
)
from experiments.functiongemma.v9_learning_material import (
    _HANDOFF_FAMILIES,
    FAMILIES,
    _case,
    generate,
    group_id,
)

from android_ui_analyser.policy import (
    POLICY_HANDOFF_ID,
    PolicyCandidate,
    PolicyContext,
    policy_messages,
    policy_tools,
)


def test_every_family_builds_and_offers_its_oracle() -> None:
    for ordinal, family in enumerate(FAMILIES):
        case = _case(family, "train", ordinal)
        assert case["family"] == family
        assert 2 <= len(case["candidates"]) <= 4, family
        oracle = case["oracle"]
        if family in _HANDOFF_FAMILIES:
            assert oracle["kind"] == "handoff", family
            continue
        assert oracle["kind"] == "select", family
        offered = [json.dumps(c["call"], sort_keys=True) for c in case["candidates"]]
        assert json.dumps(oracle["call"], sort_keys=True) in offered, family


def test_handoff_families_never_offer_an_authorized_advancing_call() -> None:
    """A handoff case must not secretly contain a correct answer."""

    for ordinal, family in enumerate(sorted(_HANDOFF_FAMILIES)):
        case = _case(family, "train", ordinal)
        assert case["oracle"] == {"kind": "handoff"}
        if family == "destructive_requires_authorization":
            assert all(c["authorized"] is False for c in case["candidates"]), family


def test_every_family_appears_in_a_short_split() -> None:
    """The family cycle is interleaved, so a small split still covers every family."""

    families = {case["family"] for case in generate("valid", len(FAMILIES))}
    assert families == set(FAMILIES)


def test_rows_use_the_packaged_serializers_verbatim() -> None:
    case = _case("destination_versus_breadcrumb_leaf", "train", 3)
    row = render_case(case, variant=0)

    assert row["tools"] == policy_tools(allow_handoff=True)
    assert [message["role"] for message in row["messages"]] == ["developer", "user", "assistant"]

    state = json.loads(row["messages"][1]["content"])
    candidates = tuple(
        PolicyCandidate(
            candidate_id=int(entry["id"]),
            call={"tool": entry["call"]["tool"], "arguments": entry["call"]["arguments"]},
            model_arguments=entry["call"]["arguments"],
            purpose=entry["purpose"],
            proof=entry["proof"],
            risk=entry["risk"],
            authorized=entry["authorized"],
            redundant=entry["redundant"],
            phase=state["phase"],
        )
        for entry in state["candidates"]
    )
    context = PolicyContext(
        goal=state["goal"],
        phase=state["phase"],
        candidates=candidates,
        observation=state["observation"],
        recent_outcomes=tuple(state["recent_outcomes"]),
        constraints=tuple(state["constraints"][: -len(_APPENDED_CONSTRAINTS)]),
        allow_handoff=True,
    )
    assert policy_messages(context, candidates)[0] == row["messages"][0]


_APPENDED_CONSTRAINTS = (
    "Select exactly one supplied candidate",
    "Require direct proof",
    "Select candidate ID -1 only when no supplied action directly advances the goal",
)


def test_handoff_rows_target_the_reserved_id() -> None:
    case = _case("target_absent_handoff", "train", 5)
    row = render_case(case, variant=0)
    call = row["messages"][2]["tool_calls"][0]["function"]
    assert call["name"] == "select_candidate"
    assert call["arguments"]["candidate_id"] == POLICY_HANDOFF_ID
    assert row["metadata"]["oracle_kind"] == "handoff"


def test_an_equivalent_entrypoint_tie_is_decided_not_refused() -> None:
    """Two controls reaching one destination must produce a selection, never a handoff."""

    case = _case("equivalent_entrypoint_tie", "train", 0)
    assert len(case["oracle"]["equivalent_calls"]) == 2
    for variant in range(6):
        row = render_case(case, variant)
        chosen = row["messages"][2]["tool_calls"][0]["function"]["arguments"]["candidate_id"]
        assert chosen != POLICY_HANDOFF_ID
        offered = {entry["id"] for entry in json.loads(row["messages"][1]["content"])["candidates"]}
        assert chosen in offered


def test_neither_candidate_id_nor_list_position_predicts_the_answer() -> None:
    rows = build_split("train", groups=len(FAMILIES) * 6, variants=12)
    selecting = [row for row in rows if row["metadata"]["oracle_kind"] != "handoff"]
    ids = Counter(row["metadata"]["target_candidate_id"] for row in selecting)
    positions = Counter(row["metadata"]["target_list_position"] for row in selecting)

    # Four candidates means chance is 0.25. A shortcut learner needs a dominant bucket; allow
    # sampling slack but nothing close to a usable signal.
    assert max(ids.values()) / len(selecting) < 0.32
    assert max(positions.values()) / len(selecting) < 0.32


def test_splits_share_no_group_or_goal() -> None:
    splits = {
        "train": build_split("train", 40, 2),
        "valid": build_split("valid", 20, 2),
        "test": build_split("test", 20, 2),
    }
    check_split_isolation(splits)

    goals = {
        split: {json.loads(row["messages"][1]["content"])["goal"] for row in rows}
        for split, rows in splits.items()
    }
    assert not goals["train"] & goals["valid"]
    assert not goals["train"] & goals["test"]
    assert not goals["valid"] & goals["test"]


def test_split_isolation_check_actually_fails_on_a_leak() -> None:
    train = build_split("train", 4, 1)
    with pytest.raises(ValueError):
        check_split_isolation({"train": train, "valid": train})


def test_variants_of_one_group_share_identity_but_differ_in_rendering() -> None:
    case = _case("shared_token_destination", "train", 7)
    rendered = [render_case(case, variant) for variant in range(6)]
    assert len({row["metadata"]["group_id"] for row in rendered}) == 1
    assert group_id(case) == rendered[0]["metadata"]["group_id"]
    orderings = {
        tuple(entry["purpose"] for entry in json.loads(row["messages"][1]["content"])["candidates"])
        for row in rendered
    }
    assert len(orderings) > 1, "counterbalancing produced identical candidate orderings"


def test_material_names_no_real_application() -> None:
    """The corpus is fictional: it must survive the repository's own privacy scanner.

    The banned names are deliberately not written here — this file is itself scanned. The
    repo guard holds them as one-way fingerprints, so reuse it rather than restating them.
    """

    from test_no_app_specific_refs import _BANNED_FINGERPRINTS, _TOKEN, _fingerprint

    rows = build_split("train", len(FAMILIES) * 2, 2)
    blob = json.dumps(rows)
    by_length: dict[int, set[str]] = {}
    for length, digest in _BANNED_FINGERPRINTS:
        by_length.setdefault(length, set()).add(digest)
    for token in _TOKEN.findall(blob):
        folded = token.casefold()
        for length, digests in by_length.items():
            for start in range(len(folded) - length + 1):
                assert _fingerprint(folded[start : start + length])[1] not in digests, token
    # A device serial is host state and must never become training content. The word itself
    # is fine — ``emulator_recommend_proxy`` is a real AUA command the model should know.
    assert not re.search(r"emulator-\w", blob, re.IGNORECASE)
    # Only the obviously fictional example package may appear.
    assert "com.example." in blob
