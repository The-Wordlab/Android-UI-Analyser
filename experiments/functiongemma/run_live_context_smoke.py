#!/usr/bin/env python3
"""Run a held-out exact-live-policy permutation gate without Android.

The four fictional controls are absent from every v6 learning split.  Each is
used as the target across all 24 candidate orders and four balanced dense-ID
assignments, matching the shape that exposed v5's chance-level live behavior.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make the repository's src layout importable from a clean source archive.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.functiongemma.live_context_curriculum import RESERVED_LIVE_SMOKE_LABELS
from experiments.functiongemma.run_production_smoke import (
    BUNDLED_ADAPTER,
    RecordingGenerator,
    _parse_record,
    build_selector,
)

from android_ui_analyser.policy import PolicyCandidate, PolicyContext, evaluate_policy  # noqa: E402

LABELS = tuple(sorted(RESERVED_LIVE_SMOKE_LABELS))
SUBTITLES = (
    "tools, recent entries, defaults",
    "collections, saved items, preferences",
    "shortcuts, activity, configuration",
    "catalogs, history, options",
)
ID_ASSIGNMENTS = tuple(tuple((index + offset) % 4 for index in range(4)) for offset in range(4))
CASE_COUNT = 4 * 24 * 4


@dataclass(frozen=True)
class LiveSmokeCase:
    target: str
    order_index: int
    id_assignment_index: int
    target_candidate_id: int
    target_position: int
    selected_candidate_id: int | None
    selected_label: str | None
    parsed: bool
    offered: bool
    provider_agreed: bool
    semantic_correct: bool
    error: str | None


def _candidate(label: str, index: int, candidate_id: int, case_id: str) -> PolicyCandidate:
    full_label = f"{label} {SUBTITLES[index]}"
    arguments = {"text": full_label}
    return PolicyCandidate(
        candidate_id=candidate_id,
        call={"tool": "tap_and_analyze", "arguments": arguments},
        model_arguments=arguments,
        purpose=f"Tap the current-frame {full_label!r} control and observe the result.",
        proof="The exact call returns a folded post-action observation.",
        safe=True,
        authorized=True,
        redundant=False,
        current=True,
        session_id=case_id,
        phase="phase_1",
        observation_fingerprint=f"frame-{case_id}",
        package="com.example.learning",
    )


def _context(
    target: str,
    order: Sequence[int],
    id_assignment: Sequence[int],
    case_id: str,
) -> PolicyContext:
    candidates_by_index = {
        index: _candidate(label, index, int(id_assignment[index]), case_id)
        for index, label in enumerate(LABELS)
    }
    candidates = tuple(candidates_by_index[index] for index in order)
    choices = ", ".join(LABELS[:-1]) + f", and {LABELS[-1]}"
    return PolicyContext(
        goal=(f"Open {target} from Example Settings among the visible {choices} choices."),
        phase="phase_1",
        candidates=candidates,
        observation={"fresh": True, "known_screen": "fixture_live_choice_panel"},
        recent_outcomes=(
            "session_active=true",
            "outcome=known",
            "goal_checkpoint_reached=false",
        ),
        constraints=(
            "Select only a supplied guard-approved candidate.",
            "Do not invent or execute a call.",
        ),
        session_id=case_id,
        observation_fingerprint=f"frame-{case_id}",
        package="com.example.learning",
    )


def run_smoke(
    selector: Any,
    recorder: RecordingGenerator,
    *,
    model: str,
    adapter: str,
) -> dict[str, Any]:
    cases: list[LiveSmokeCase] = []
    orders = tuple(itertools.permutations(range(4)))
    for target in LABELS:
        target_index = LABELS.index(target)
        for order_index, order in enumerate(orders):
            for assignment_index, assignment in enumerate(ID_ASSIGNMENTS):
                case_id = f"live-{target_index}-{order_index}-{assignment_index}"
                context = _context(target, order, assignment, case_id)
                before = len(recorder.records)
                decision = evaluate_policy(context, selector, mode="shadow", max_candidates=4)
                records = recorder.records[before:]
                record = records[0] if len(records) == 1 else None
                selected_id, parsed, offered, parse_error = _parse_record(record, {0, 1, 2, 3})
                selected = decision.selected_candidate
                selected_label = None
                if selected is not None:
                    selected_label = next(
                        label
                        for index, label in enumerate(LABELS)
                        if int(assignment[index]) == selected.candidate_id
                    )
                provider_agreed = bool(
                    parsed
                    and offered
                    and selected is not None
                    and selected_id == decision.selected_candidate_id
                )
                semantic_correct = provider_agreed and selected_label == target
                error = (
                    f"expected one generation, got {len(records)}"
                    if len(records) != 1
                    else parse_error or selector.last_error
                )
                cases.append(
                    LiveSmokeCase(
                        target=target,
                        order_index=order_index,
                        id_assignment_index=assignment_index,
                        target_candidate_id=int(assignment[target_index]),
                        target_position=tuple(order).index(target_index),
                        selected_candidate_id=selected_id,
                        selected_label=selected_label,
                        parsed=parsed,
                        offered=offered,
                        provider_agreed=provider_agreed,
                        semantic_correct=semantic_correct,
                        error=error,
                    )
                )
    semantic_correct = sum(case.semantic_correct for case in cases)
    per_target = {
        target: {
            "cases": sum(case.target == target for case in cases),
            "correct": sum(case.target == target and case.semantic_correct for case in cases),
        }
        for target in LABELS
    }
    gates = {
        "complete_384_case_matrix": len(cases) == CASE_COUNT,
        "protocol_parse_100_percent": all(case.parsed for case in cases),
        "offered_id_100_percent": all(case.offered for case in cases),
        "provider_agreement_100_percent": all(case.provider_agreed for case in cases),
        "semantic_accuracy_100_percent": semantic_correct == len(cases),
        "every_target_96_of_96": all(value["correct"] == 96 for value in per_target.values()),
    }
    return {
        "format": "functiongemma-aua-live-context-smoke-v1",
        "host_only": True,
        "fictional": True,
        "model": model,
        "adapter": adapter,
        "matrix": {
            "labels": list(LABELS),
            "targets": len(LABELS),
            "candidate_orders": len(orders),
            "dense_id_assignments": [list(value) for value in ID_ASSIGNMENTS],
            "cases": len(cases),
        },
        "metrics": {
            "semantic_correct": semantic_correct,
            "semantic_accuracy": semantic_correct / len(cases),
            "per_target": per_target,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "cases": [asdict(case) for case in cases],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=BUNDLED_ADAPTER)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selector, recorder = build_selector(args.model, args.adapter)
        report = run_smoke(
            selector,
            recorder,
            model=str(Path(args.model).resolve()),
            adapter=str(args.adapter),
        )
    except Exception as exc:
        report = {
            "format": "functiongemma-aua-live-context-smoke-v1",
            "host_only": True,
            "fictional": True,
            "passed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": report.get("matrix", {}).get("cases", 0),
                "semantic_accuracy": report.get("metrics", {}).get("semantic_accuracy"),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
