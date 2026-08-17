#!/usr/bin/env python3
"""Fail-closed structural, privacy, split, and token-length validation for SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mlx_lm.utils import load_tokenizer

SPLITS = ("train", "valid", "test")
ACTIVATION = "You are a model that can do function calling with the following functions."
FORBIDDEN_MARKERS = ("emulator-", "/users/", "/private/", "@")
HIGH_ENTROPY = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_+/=-]{40,}(?![A-Za-z0-9])")
TOKEN = re.compile(r"[A-Za-z0-9_@.:/+%-]+")


def _private_fingerprints() -> set[tuple[int, str]]:
    """Reuse the public repository's one-way denylist without publishing its cleartext."""
    guard = Path(__file__).resolve().parents[2] / "tests" / "test_no_app_specific_refs.py"
    if not guard.is_file():
        raise ValueError(f"repository privacy guard is missing: {guard}")
    values = runpy.run_path(str(guard)).get("_BANNED_FINGERPRINTS")
    if not isinstance(values, set) or not values:
        raise ValueError("repository privacy fingerprints could not be loaded")
    return values


def _contains_private_fingerprint(text: str, fingerprints: set[tuple[int, str]]) -> bool:
    by_length: dict[int, set[str]] = defaultdict(set)
    for length, digest in fingerprints:
        by_length[length].add(digest)
    for token in TOKEN.findall(text):
        folded = token.casefold()
        for length, digests in by_length.items():
            for start in range(len(folded) - length + 1):
                value = folded[start : start + length]
                if hashlib.sha256(value.encode()).hexdigest() in digests:
                    return True
    return False


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            result.append(row)
    if not result:
        raise ValueError(f"{path}: empty split")
    return result


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _validate_row(
    row: dict[str, Any],
    *,
    split: str,
    tokenizer: Any,
    max_seq_length: int,
    private_fingerprints: set[tuple[int, str]],
) -> dict[str, Any]:
    compact = json.dumps(row, sort_keys=True, separators=(",", ":"))
    lowered = compact.lower()
    for needle in FORBIDDEN_MARKERS:
        if needle in lowered:
            raise ValueError(f"{split}: forbidden private/host marker {needle!r}")
    if _contains_private_fingerprint(compact, private_fingerprints):
        raise ValueError(f"{split}: repository-private fingerprint found")
    if HIGH_ENTROPY.search(compact):
        raise ValueError(f"{split}: possible secret or high-entropy identifier")

    messages = row.get("messages")
    tools = row.get("tools")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{split}: expected exactly three messages")
    if [message.get("role") for message in messages] != ["developer", "user", "assistant"]:
        raise ValueError(f"{split}: roles must be developer, user, assistant")
    developer = messages[0].get("content")
    if not isinstance(developer, str) or not developer.startswith(ACTIVATION):
        raise ValueError(f"{split}: FunctionGemma developer activation is missing")
    if not isinstance(tools, list) or len(tools) != 1:
        raise ValueError(f"{split}: exactly one function declaration is required")
    function_schema = tools[0].get("function", {})
    if function_schema.get("name") != "select_candidate":
        raise ValueError(f"{split}: only select_candidate may be declared")
    if not isinstance(metadata, dict) or metadata.get("split") != split:
        raise ValueError(f"{split}: missing or mismatched metadata.split")

    try:
        state = json.loads(messages[1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{split}: user state is not valid JSON") from exc
    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 8:
        raise ValueError(f"{split}: expected two to eight candidates")
    ids = [candidate.get("id") for candidate in candidates]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
        raise ValueError(f"{split}: every candidate id must be an integer")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split}: candidate ids are not unique")
    for candidate in candidates:
        call = candidate.get("call")
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
            raise ValueError(f"{split}: every candidate needs an exact AUA call")
        if not isinstance(call.get("arguments", {}), dict):
            raise ValueError(f"{split}: call arguments must be an object")
        if candidate.get("risk") not in {"safe", "unsafe", "destructive"}:
            raise ValueError(f"{split}: candidate risk must be explicit")
        if not isinstance(candidate.get("authorized"), bool):
            raise ValueError(f"{split}: candidate authorization must be explicit")

    assistant = messages[2]
    calls = assistant.get("tool_calls")
    if assistant.get("content") is not None or not isinstance(calls, list) or len(calls) != 1:
        raise ValueError(f"{split}: assistant target must be exactly one tool call")
    target_function = calls[0].get("function", {})
    if target_function.get("name") != "select_candidate":
        raise ValueError(f"{split}: target function must be select_candidate")
    target_id = target_function.get("arguments", {}).get("candidate_id")
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if target_id not in by_id:
        raise ValueError(f"{split}: target candidate is not offered")
    target = by_id[target_id]
    if not target["authorized"]:
        raise ValueError(f"{split}: oracle target is unauthorized")
    if target.get("redundant", False):
        raise ValueError(f"{split}: oracle target is marked redundant")

    full_tokens = tokenizer.apply_chat_template(messages, tools=tools, return_dict=False)
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, add_generation_prompt=True, return_dict=False
    )
    if len(full_tokens) > max_seq_length:
        raise ValueError(
            f"{split}: {len(full_tokens)} tokens exceeds max_seq_length={max_seq_length}"
        )
    if len(full_tokens) <= len(prompt_tokens):
        raise ValueError(f"{split}: target completion has no trainable tokens")

    return {
        "case_id": str(metadata.get("case_id") or metadata.get("group_id")),
        "group_id": str(metadata.get("group_id")),
        "family": str(metadata.get("family") or metadata.get("intent")),
        "target_id": target_id,
        "target_tool": target["call"]["tool"],
        "criticality": str(metadata.get("criticality") or "normal"),
        "scenario_kind": str(metadata.get("scenario_kind") or "unknown"),
        "tokens": len(full_tokens),
        "prompt_tokens": len(prompt_tokens),
        "completion_tokens": len(full_tokens) - len(prompt_tokens),
        "digest": hashlib.sha256(compact.encode()).hexdigest(),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model)
    private_fingerprints = _private_fingerprints()
    split_stats: dict[str, Any] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    all_digests: set[str] = set()
    all_records: list[dict[str, Any]] = []

    for split in SPLITS:
        path = args.data_dir / f"{split}.jsonl"
        rows = _rows(path)
        records = [
            _validate_row(
                row,
                split=split,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                private_fingerprints=private_fingerprints,
            )
            for row in rows
        ]
        for record in records:
            if record["digest"] in all_digests:
                raise ValueError(f"duplicate row across splits: {record['case_id']}")
            all_digests.add(record["digest"])
            group_splits[record["group_id"]].add(split)
        all_records.extend(records)
        lengths = [record["tokens"] for record in records]
        split_stats[split] = {
            "rows": len(records),
            "groups": len({record["group_id"] for record in records}),
            "families": dict(sorted(Counter(record["family"] for record in records).items())),
            "target_tools": dict(
                sorted(Counter(record["target_tool"] for record in records).items())
            ),
            "critical_cases": sum(record["criticality"] == "critical" for record in records),
            "tokens": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "p95": _percentile(lengths, 0.95),
                "max": max(lengths),
            },
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    leaked_groups = {group: splits for group, splits in group_splits.items() if len(splits) != 1}
    if leaked_groups:
        sample = next(iter(leaked_groups.items()))
        raise ValueError(f"scenario group leaked across splits: {sample}")

    target_positions = Counter(record["target_id"] for record in all_records)
    if len(target_positions) < 4:
        raise ValueError("target positions are not sufficiently randomized")
    maximum = max(target_positions.values())
    minimum = min(target_positions.values())
    if maximum > minimum * 1.75:
        raise ValueError(f"target-position distribution is too imbalanced: {target_positions}")

    result = {
        "ok": True,
        "model": str(args.model),
        "max_seq_length": args.max_seq_length,
        "total_rows": len(all_records),
        "total_groups": len(group_splits),
        "target_positions": dict(sorted(target_positions.items())),
        "splits": split_stats,
    }
    output = args.output or args.data_dir / "validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_seq_length < 64:
        raise SystemExit("max sequence length is unexpectedly small")
    print(json.dumps(validate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
