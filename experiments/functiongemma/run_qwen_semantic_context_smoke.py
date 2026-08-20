#!/usr/bin/env python3
"""Run the untouched v7 semantic matrix through a Qwen3 MLX adapter."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.functiongemma import evaluate_qwen
from experiments.functiongemma.run_semantic_context_smoke import (
    CARDINALITIES,
    QUALIFIERS,
    _context,
)

from android_ui_analyser.policy import policy_messages, policy_tools


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cardinality in CARDINALITIES:
        qualifiers = QUALIFIERS[:cardinality]
        for qualifier_index, target_qualifier in enumerate(qualifiers):
            for target_id in range(cardinality):
                for target_position in range(cardinality):
                    case_id = (
                        f"qwen-semantic-c{cardinality}-q{qualifier_index}-"
                        f"i{target_id}-p{target_position}"
                    )
                    context = _context(
                        qualifiers,
                        target_qualifier,
                        target_id,
                        target_position,
                        case_id,
                    )
                    rows.append(
                        {
                            "messages": [
                                *policy_messages(context),
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "select_candidate",
                                                "arguments": {"candidate_id": target_id},
                                            },
                                        }
                                    ],
                                },
                            ],
                            "tools": policy_tools(),
                            "metadata": {
                                "case_id": case_id,
                                "group_id": case_id,
                                "family": f"semantic_context_c{cardinality}",
                                "criticality": "critical",
                                "target_qualifier": target_qualifier,
                            },
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _rows()
    with tempfile.TemporaryDirectory(prefix="aua-qwen-smoke-") as temporary:
        data = Path(temporary) / "smoke.jsonl"
        data.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = evaluate_qwen.evaluate(
            argparse.Namespace(
                model=args.model,
                adapter=args.adapter,
                data=data,
                output=args.output,
                limit=None,
                batch_size=32,
                prefill_batch_size=8,
                max_tokens=64,
            )
        )
    metrics = report["metrics"]
    passed = (
        metrics["cases"] == 99
        and metrics["parse_success"] == 1.0
        and metrics["candidate_exists"] == 1.0
        and metrics["candidate_accuracy"] == 1.0
    )
    report["format"] = "qwen3-aua-semantic-context-smoke-v1"
    report["passed"] = passed
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "cases": metrics["cases"],
                "accuracy": metrics["candidate_accuracy"],
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
