"""Privacy-safe source-oracle material derived from the live v8 A/B failures.

This is deliberately one layer before model chat-template rendering.  It can represent
``handoff`` without pretending the current v7 ``select_candidate`` protocol supports it.
After the production handoff protocol lands, a renderer can turn every row into native
FunctionGemma/Qwen training examples without revisiting private or live-app traces.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "aua-policy-source-oracle-v8"
SEED = 20260818
SPLIT_GROUPS = {"train": 200, "valid": 25, "test": 25}
FAMILIES = (
    "meta_control_negative",
    "shared_token_destination",
    "two_hop_navigation",
    "target_absent_handoff",
    "proof_cleanup_recovery",
)
_WORDS = {
    "train": ("Archive", "Canvas", "Garden", "Ledger", "Notebook", "Workshop"),
    "valid": ("Compendium", "Gallery", "Observatory", "Registry", "Studio"),
    "test": ("Almanac", "Foundry", "Library", "Portfolio", "Repository"),
}


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


def _candidate(call: dict[str, Any], purpose: str, proof: str) -> dict[str, Any]:
    return {
        "call": call,
        "purpose": purpose,
        "proof": proof,
        "risk": "safe",
        "authorized": True,
        "redundant": False,
    }


def _semantic_case(
    split: str, ordinal: int
) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    family = FAMILIES[ordinal % len(FAMILIES)]
    words = _WORDS[split]
    topic = f"{words[ordinal % len(words)]} {ordinal:03d}"
    other = f"{words[(ordinal + 1) % len(words)]} {ordinal:03d}"
    state: dict[str, Any] = {
        "goal": f"Open {topic} and prove the requested destination without changing state.",
        "phase": "open_requested_destination",
        "observation": {"fresh": True, "source": "hierarchy", "element_count": 12},
        "recent_outcomes": [],
        "constraints": ["read_only", "fresh_observation_required"],
    }

    if family == "meta_control_negative":
        target = _candidate(
            _call("tap_and_analyze", text=topic),
            f"Open the exact {topic} row and observe its destination.",
            "The folded observation can prove the requested destination.",
        )
        candidates = [
            target,
            _candidate(
                _call("tap_and_analyze", rid="clear_query"),
                "Clear the current search query.",
                "The folded observation only proves that search text changed.",
            ),
            _candidate(
                _call("tap_and_analyze", text=f"{other} results"),
                f"Open the different {other} result.",
                "The folded observation proves a different destination.",
            ),
            _candidate(
                _call("key_and_analyze", name="back"),
                "Leave the current result list.",
                "The folded observation proves navigation away from the requested row.",
            ),
        ]
        oracle = {"kind": "select", "call": target["call"]}
    elif family == "shared_token_destination":
        exact = f"{topic} details"
        state["goal"] = f"Open {exact}; similarly named detail controls are distractors."
        target = _candidate(
            _call("tap_and_analyze", text=exact),
            f"Open the exact {exact} destination row.",
            "The post-action observation proves the details destination.",
        )
        candidates = [
            target,
            *[
                _candidate(
                    _call("tap_and_analyze", text=f"{topic} {suffix}"),
                    f"Open the {topic} {suffix} control.",
                    "The observation proves a related but different setting.",
                )
                for suffix in ("locking detail", "click details", "detail alerts")
            ],
        ]
        oracle = {"kind": "select", "call": target["call"]}
    elif family == "two_hop_navigation":
        second_hop = ordinal % 2 == 1
        label = f"{topic} record" if second_hop else f"{topic} collection"
        state["phase"] = "open_record" if second_hop else "open_collection"
        state["goal"] = (
            f"Open {topic}; this is {'the intermediate' if second_hop else 'the initial'} navigation frame."
        )
        target = _candidate(
            _call("tap_and_analyze", text=label),
            f"Open the exact {label} row and consume its observation.",
            "The folded observation proves progress to the next typed phase.",
        )
        candidates = [
            target,
            _candidate(
                _call("tap_and_analyze", text=f"{other} collection"),
                f"Open the unrelated {other} collection.",
                "The observation proves the wrong branch.",
            ),
            _candidate(
                _call("analyze_screen", source="hierarchy", no_cache=True),
                "Observe the unchanged fresh frame again.",
                "This repeats evidence without advancing the typed phase.",
            ),
        ]
        oracle = {"kind": "select", "call": target["call"]}
    elif family == "target_absent_handoff":
        state["goal"] = f"From this current list only, open {topic} workspace; hand off if absent."
        candidates = [
            _candidate(
                _call("tap_and_analyze", text=f"{topic} alerts"),
                f"Open {topic} alerts, which is not the requested workspace.",
                "The observation would prove a different destination.",
            ),
            _candidate(
                _call("tap_and_analyze", text=f"{other} workspace"),
                f"Open the different {other} workspace.",
                "The observation would prove the wrong entity.",
            ),
            _candidate(
                _call("scroll_and_analyze", direction="up", percent=70),
                "Navigate beyond the bounded current list.",
                "This violates the current-list-only constraint.",
            ),
        ]
        oracle = {"kind": "handoff", "reason": "target_absent"}
    else:
        cleanup_pending = ordinal % 2 == 1
        state["phase"] = "cleanup" if cleanup_pending else "prove_arrival"
        state["observation"]["outcome"] = "unknown" if not cleanup_pending else "satisfied"
        state["constraints"].append("cleanup_required")
        if cleanup_pending:
            target_call = _call("network_restore", verify=True)
            purpose = "Restore the session-owned network state before finish."
        else:
            target_call = _call("analyze_screen", source="hierarchy", no_cache=True)
            purpose = "Resolve the unknown outcome with one fresh read before any replay."
        target = _candidate(
            target_call,
            purpose,
            "Deterministic proof or cleanup state authorizes the next typed phase.",
        )
        candidates = [
            target,
            _candidate(
                _call("session_finish"),
                "Finish immediately despite unresolved proof or cleanup.",
                "This would terminate before the required state is proven.",
            ),
            _candidate(
                _call("tap_and_analyze", text=topic),
                "Replay the last mutation before resolving its outcome.",
                "This risks a duplicate mutation.",
            ),
        ]
        oracle = {"kind": "select", "call": target["call"]}
    return family, state, candidates, oracle


def build_v8_source(*, seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    dataset: dict[str, list[dict[str, Any]]] = {}
    for split, groups in SPLIT_GROUPS.items():
        rows: list[dict[str, Any]] = []
        for ordinal in range(groups):
            family, state, semantic_candidates, oracle = _semantic_case(split, ordinal)
            group_id = f"v8-{split}-{family}-{ordinal:04d}"
            for variant in range(4):
                candidates = [dict(item) for item in semantic_candidates]
                cardinality = len(candidates)
                order = list(range(cardinality))
                random.Random(f"{seed}:{group_id}:order:{variant}").shuffle(order)
                ids = list(range(cardinality))
                random.Random(f"{seed}:{group_id}:ids:{variant}").shuffle(ids)
                materialized = []
                for position, semantic_index in enumerate(order):
                    item = dict(candidates[semantic_index])
                    item["id"] = ids[semantic_index]
                    item["position"] = position
                    materialized.append(item)
                rows.append(
                    {
                        "schema": SCHEMA,
                        "id": f"{group_id}-v{variant}",
                        "split": split,
                        "group_id": group_id,
                        "family": family,
                        "state": state,
                        "candidates": materialized,
                        "oracle": oracle,
                        "metadata": {
                            "variant": variant,
                            "privacy": "fictional_app_agnostic",
                            "render_status": (
                                "blocked_until_handoff_protocol"
                                if oracle["kind"] == "handoff"
                                else "ready_for_model_renderer"
                            ),
                        },
                    }
                )
        random.Random(f"{seed}:{split}:rows").shuffle(rows)
        dataset[split] = rows
    return dataset


def write_v8_source(output_dir: Path, *, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_v8_source(seed=seed)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "seed": seed,
        "protocol_gate": "explicit handoff outcome must land before rendering target-absent rows",
        "splits": {},
    }
    for split, rows in dataset.items():
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path = output_dir / f"{split}.jsonl"
        path.write_text(payload)
        manifest["splits"][split] = {
            "rows": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "families": dict(sorted(Counter(row["family"] for row in rows).items())),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    write_v8_source(Path("runs/functiongemma/data-v8-source"))
