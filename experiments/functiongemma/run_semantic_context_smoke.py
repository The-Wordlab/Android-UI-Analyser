#!/usr/bin/env python3
"""Run the untouched v7 variable-cardinality semantic gate without Android."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.functiongemma.run_production_smoke import (  # noqa: E402
    BUNDLED_ADAPTER,
    RecordingGenerator,
    _parse_record,
    build_selector,
)
from experiments.functiongemma.semantic_context_curriculum import (  # noqa: E402
    RESERVED_V7_SMOKE_TERMS,
)

from android_ui_analyser.policy import PolicyCandidate, PolicyContext, evaluate_policy  # noqa: E402

TARGET = "Chronicle"
QUALIFIERS = (
    "archived notebooks",
    "guided lessons",
    "recent timelines",
    "saved exhibits",
)
CARDINALITIES = (2, 3, 4)
CASE_COUNT = sum(value**3 for value in CARDINALITIES)


@dataclass(frozen=True)
class SemanticSmokeCase:
    cardinality: int
    target_qualifier: str
    target_candidate_id: int
    target_position: int
    selected_candidate_id: int | None
    selected_qualifier: str | None
    parsed: bool
    offered: bool
    provider_agreed: bool
    semantic_correct: bool
    error: str | None


def _candidate(qualifier: str, candidate_id: int, case_id: str) -> PolicyCandidate:
    full_label = f"{TARGET} {qualifier}"
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
    qualifiers: Sequence[str],
    target_qualifier: str,
    target_id: int,
    target_position: int,
    case_id: str,
) -> PolicyContext:
    distractors = [value for value in qualifiers if value != target_qualifier]
    rotation = (target_id + target_position) % len(distractors) if distractors else 0
    distractors = distractors[rotation:] + distractors[:rotation]
    ordered = list(distractors)
    ordered.insert(target_position, target_qualifier)
    remaining_ids = [value for value in range(len(qualifiers)) if value != target_id]
    if remaining_ids:
        shift = target_position % len(remaining_ids)
        remaining_ids = remaining_ids[shift:] + remaining_ids[:shift]
    id_cursor = iter(remaining_ids)
    candidates = tuple(
        _candidate(
            qualifier,
            target_id if qualifier == target_qualifier else next(id_cursor),
            case_id,
        )
        for qualifier in ordered
    )
    return PolicyContext(
        goal=(
            f"Requested destination: {TARGET.casefold()}. "
            f"Matching evidence: {target_qualifier.casefold()}."
        ),
        phase="phase_1",
        candidates=candidates,
        observation={"fresh": True, "known_screen": "fixture_semantic_choice_panel"},
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
    cases: list[SemanticSmokeCase] = []
    for cardinality in CARDINALITIES:
        qualifiers = QUALIFIERS[:cardinality]
        for qualifier_index, target_qualifier in enumerate(qualifiers):
            for target_id in range(cardinality):
                for target_position in range(cardinality):
                    case_id = (
                        f"semantic-c{cardinality}-q{qualifier_index}-"
                        f"i{target_id}-p{target_position}"
                    )
                    context = _context(
                        qualifiers,
                        target_qualifier,
                        target_id,
                        target_position,
                        case_id,
                    )
                    before = len(recorder.records)
                    decision = evaluate_policy(
                        context,
                        selector,
                        mode="shadow",
                        max_candidates=4,
                    )
                    records = recorder.records[before:]
                    record = records[0] if len(records) == 1 else None
                    selected_id, parsed, offered, parse_error = _parse_record(
                        record,
                        set(range(cardinality)),
                    )
                    selected = decision.selected_candidate
                    selected_qualifier = None
                    if selected is not None:
                        for candidate in context.candidates:
                            if candidate.candidate_id == selected.candidate_id:
                                selected_qualifier = str(
                                    candidate.model_arguments["text"]
                                ).removeprefix(f"{TARGET} ")
                                break
                    provider_agreed = bool(
                        parsed
                        and offered
                        and selected is not None
                        and selected_id == decision.selected_candidate_id
                    )
                    cases.append(
                        SemanticSmokeCase(
                            cardinality=cardinality,
                            target_qualifier=target_qualifier,
                            target_candidate_id=target_id,
                            target_position=target_position,
                            selected_candidate_id=selected_id,
                            selected_qualifier=selected_qualifier,
                            parsed=parsed,
                            offered=offered,
                            provider_agreed=provider_agreed,
                            semantic_correct=(
                                provider_agreed and selected_qualifier == target_qualifier
                            ),
                            error=(
                                f"expected one generation, got {len(records)}"
                                if len(records) != 1
                                else parse_error or selector.last_error
                            ),
                        )
                    )
    by_cardinality = {
        str(cardinality): {
            "cases": sum(case.cardinality == cardinality for case in cases),
            "correct": sum(
                case.cardinality == cardinality and case.semantic_correct for case in cases
            ),
        }
        for cardinality in CARDINALITIES
    }
    target_ids = Counter(case.target_candidate_id for case in cases)
    target_positions = Counter(case.target_position for case in cases)
    gates = {
        "complete_matrix": len(cases) == CASE_COUNT,
        "protocol_parse_100_percent": all(case.parsed for case in cases),
        "offered_id_100_percent": all(case.offered for case in cases),
        "provider_agreement_100_percent": all(case.provider_agreed for case in cases),
        "semantic_accuracy_100_percent": all(case.semantic_correct for case in cases),
        "every_cardinality_100_percent": all(
            value["correct"] == value["cases"] for value in by_cardinality.values()
        ),
    }
    return {
        "format": "functiongemma-aua-semantic-context-smoke-v1",
        "host_only": True,
        "fictional": True,
        "reserved_terms": sorted(RESERVED_V7_SMOKE_TERMS),
        "model": model,
        "adapter": adapter,
        "matrix": {
            "cases": len(cases),
            "cardinalities": list(CARDINALITIES),
            "target_id_histogram": dict(sorted(target_ids.items())),
            "target_position_histogram": dict(sorted(target_positions.items())),
        },
        "metrics": {
            "semantic_correct": sum(case.semantic_correct for case in cases),
            "semantic_accuracy": sum(case.semantic_correct for case in cases) / len(cases),
            "by_cardinality": by_cardinality,
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
            "format": "functiongemma-aua-semantic-context-smoke-v1",
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
