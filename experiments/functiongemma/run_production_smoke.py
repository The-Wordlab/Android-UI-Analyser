#!/usr/bin/env python3
"""Exercise the packaged FunctionGemma policy serializer without Android.

This is a host-only invariance smoke test, not an Android test.  It crosses all
24 orders of four fictional controls with four independent dense-ID mappings.
The production policy guard authors every exact call; FunctionGemma may only
select an offered opaque ID.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make the repository's src layout importable when this file is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from android_ui_analyser.policy import (  # noqa: E402
    PolicyCandidate,
    PolicyContext,
    evaluate_policy,
    policy_tools,
)
from android_ui_analyser.providers.policy.functiongemma import (  # noqa: E402
    BUNDLED_ADAPTER,
    FunctionGemmaPolicySelector,
    SelectionProtocolError,
    parse_candidate_id,
)

SUBJECTS = ("Grammar", "Mathematics", "History", "Physics")
TARGET_SUBJECT = "Mathematics"
PHASE = "phase_1"
CASE_COUNT = 96
MIN_SEMANTIC_ACCURACY = 0.95
MAX_BIAS_GAP = 0.10

# Each subject occupies each opaque ID exactly once.  These assignments are
# crossed independently with every candidate-list order, so ID and position do
# not leak the target.
DENSE_ID_PERMUTATIONS = tuple(
    tuple((subject_index + offset) % len(SUBJECTS) for subject_index in range(len(SUBJECTS)))
    for offset in range(len(SUBJECTS))
)


@dataclass(frozen=True)
class GenerationRecord:
    """One raw generation plus the tokenizer needed for production parsing."""

    output: Any
    tokenizer: Any


class RecordingGenerator:
    """Transparent wrapper that records outputs from the provider's generator."""

    def __init__(self, generate: Callable[..., Any]) -> None:
        self._generate = generate
        self.records: list[GenerationRecord] = []

    def __call__(self, model: Any, tokenizer: Any, prompt: Any, **kwargs: Any) -> Any:
        output = self._generate(model, tokenizer, prompt, **kwargs)
        self.records.append(GenerationRecord(output=output, tokenizer=tokenizer))
        return output


@dataclass(frozen=True)
class SmokeCase:
    """Auditable result for one order and dense-ID assignment."""

    case_id: str
    order_index: int
    id_permutation_index: int
    subject_order: tuple[str, ...]
    dense_ids_by_subject: tuple[int, ...]
    target_candidate_id: int
    target_position: int
    selected_candidate_id: int | None
    selected_position: int | None
    protocol_parsed: bool
    offered_id: bool
    provider_agreed: bool
    semantic_correct: bool
    decision_status: str
    raw_output: str | None
    error: str | None

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["subject_order"] = list(self.subject_order)
        value["dense_ids_by_subject"] = list(self.dense_ids_by_subject)
        return value


def _existing_absolute_directory(value: str | Path, label: str) -> Path:
    authored = Path(value)
    if not authored.is_absolute():
        raise ValueError(f"{label} must be an absolute local path")
    resolved = authored.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing local directory")
    return resolved


def _adapter_setting(value: str | Path | None) -> str | None:
    """Return the provider setting; ``None`` means the packaged adapter."""

    if value is None or str(value).strip().lower() in {"", "bundled", "default"}:
        return None
    return str(_existing_absolute_directory(value, "adapter"))


def build_selector(
    model: str | Path,
    adapter: str | Path | None = BUNDLED_ADAPTER,
    *,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    generator: Callable[..., Any] | None = None,
    sampler_factory: Callable[..., Any] | None = None,
) -> tuple[FunctionGemmaPolicySelector, RecordingGenerator]:
    """Build the production selector with a capture-only generator wrapper.

    If runtime functions are omitted, MLX is imported locally.  The model and
    any non-bundled adapter must already exist as absolute directories, so this
    function can never turn a repository ID into an implicit download.
    """

    model_path = _existing_absolute_directory(model, "model")
    adapter_path = _adapter_setting(adapter)
    supplied = (model_loader, generator, sampler_factory)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError("model_loader, generator, and sampler_factory must be supplied together")
    if model_loader is None:
        from mlx_lm import load
        from mlx_lm.generate import generate
        from mlx_lm.sample_utils import make_sampler

        model_loader = load
        generator = generate
        sampler_factory = make_sampler
    assert generator is not None
    assert model_loader is not None
    assert sampler_factory is not None
    recorder = RecordingGenerator(generator)
    settings: dict[str, Any] = {
        "model_path": str(model_path),
        "adapter_path": adapter_path,
        "max_tokens": 24,
    }
    if adapter_path is not None:
        manifest_path = Path(adapter_path) / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base_model = manifest.get("base_model") if isinstance(manifest, dict) else None
            adapter_value = manifest.get("adapter") if isinstance(manifest, dict) else None
            if isinstance(base_model, dict) and isinstance(adapter_value, dict):
                settings.update(
                    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    model_sha256=base_model.get("sha256"),
                    adapter_sha256=adapter_value.get("sha256"),
                )
    selector = FunctionGemmaPolicySelector(
        settings,
        model_loader=model_loader,
        generator=recorder,
        sampler_factory=sampler_factory,
    )
    return selector, recorder


def _trusted_call(subject: str) -> dict[str, Any]:
    return {
        "tool": "tap_and_analyze",
        "arguments": {"rid": f"open{subject}"},
    }


def _context(
    subject_order: Sequence[str],
    dense_ids_by_subject: Sequence[int],
    *,
    case_id: str,
) -> PolicyContext:
    id_by_subject = dict(zip(SUBJECTS, dense_ids_by_subject, strict=True))
    candidates = tuple(
        PolicyCandidate(
            candidate_id=id_by_subject[subject],
            call=_trusted_call(subject),
            model_arguments={"rid": f"open{subject}"},
            purpose=f"Tap the current-frame '{subject}' control and observe the result.",
            proof="The exact call returns a folded post-action observation.",
            safe=True,
            authorized=True,
            redundant=False,
            current=True,
            session_id=case_id,
            phase=PHASE,
            observation_fingerprint=f"frame-{case_id}",
            package="com.example.learning",
        )
        for subject in subject_order
    )
    return PolicyContext(
        goal="Tap Mathematics and prove the settled result.",
        phase=PHASE,
        candidates=candidates,
        observation={
            "fresh": True,
            "known_screen": "fixture_subject_picker",
            "outcome": "known",
            "element_count": len(SUBJECTS),
        },
        recent_outcomes=(
            "session_started=true",
            "fresh_observation=true",
            "goal_checkpoint_reached=false",
        ),
        constraints=("Use the current observation",),
        session_id=case_id,
        observation_fingerprint=f"frame-{case_id}",
        package="com.example.learning",
    )


def _parse_record(
    record: GenerationRecord | None,
    offered_ids: set[int],
) -> tuple[int | None, bool, bool, str | None]:
    if record is None:
        return None, False, False, "provider produced no generation"
    try:
        selected_id = parse_candidate_id(record.output, record.tokenizer, policy_tools())
    except SelectionProtocolError as exc:
        return None, False, False, f"{type(exc).__name__}: {exc}"
    offered = selected_id in offered_ids
    return (
        selected_id,
        True,
        offered,
        None if offered else f"candidate {selected_id} was not offered",
    )


def _rate(cases: Sequence[SmokeCase], field: str) -> float:
    return sum(bool(getattr(case, field)) for case in cases) / len(cases) if cases else 0.0


def _accuracy_by(
    cases: Sequence[SmokeCase],
    field: str,
    values: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], float]:
    groups: dict[str, dict[str, Any]] = {}
    accuracies: list[float] = []
    for value in values:
        selected = [case for case in cases if getattr(case, field) == value]
        correct = sum(case.semantic_correct for case in selected)
        accuracy = correct / len(selected) if selected else 0.0
        groups[str(value)] = {
            "cases": len(selected),
            "semantic_correct": correct,
            "semantic_accuracy": accuracy,
        }
        accuracies.append(accuracy)
    return groups, max(accuracies) - min(accuracies)


def run_smoke(
    selector: FunctionGemmaPolicySelector,
    recorder: RecordingGenerator,
    *,
    model: str,
    adapter: str,
) -> dict[str, Any]:
    """Run the balanced 96-case matrix through production policy code."""

    cases: list[SmokeCase] = []
    orders = tuple(itertools.permutations(SUBJECTS))
    target_call = _trusted_call(TARGET_SUBJECT)
    for id_index, dense_ids in enumerate(DENSE_ID_PERMUTATIONS):
        id_by_subject = dict(zip(SUBJECTS, dense_ids, strict=True))
        for order_index, subject_order in enumerate(orders):
            case_id = f"id-{id_index:02d}-order-{order_index:02d}"
            context = _context(subject_order, dense_ids, case_id=case_id)
            offered_ids = {candidate.candidate_id for candidate in context.candidates}
            generation_count = len(recorder.records)
            decision = evaluate_policy(
                context,
                selector,
                mode="shadow",
                max_candidates=len(SUBJECTS),
            )
            new_records = recorder.records[generation_count:]
            record = new_records[0] if len(new_records) == 1 else None
            parsed_id, parsed, offered, parse_error = _parse_record(record, offered_ids)
            selected = decision.selected_candidate
            selected_position = next(
                (
                    index
                    for index, candidate in enumerate(context.candidates)
                    if candidate.candidate_id == decision.selected_candidate_id
                ),
                None,
            )
            provider_agreed = bool(
                parsed
                and offered
                and selected is not None
                and parsed_id == decision.selected_candidate_id
            )
            semantic_correct = bool(
                provider_agreed and selected is not None and selected.trusted_call() == target_call
            )
            provider_error = selector.last_error
            error = (
                f"expected one generation, got {len(new_records)}"
                if len(new_records) != 1
                else parse_error or provider_error
            )
            cases.append(
                SmokeCase(
                    case_id=case_id,
                    order_index=order_index,
                    id_permutation_index=id_index,
                    subject_order=tuple(subject_order),
                    dense_ids_by_subject=tuple(dense_ids),
                    target_candidate_id=id_by_subject[TARGET_SUBJECT],
                    target_position=subject_order.index(TARGET_SUBJECT),
                    selected_candidate_id=parsed_id,
                    selected_position=selected_position,
                    protocol_parsed=parsed,
                    offered_id=offered,
                    provider_agreed=provider_agreed,
                    semantic_correct=semantic_correct,
                    decision_status=decision.status,
                    raw_output=str(record.output) if record is not None else None,
                    error=error,
                )
            )

    by_target_id, id_gap = _accuracy_by(cases, "target_candidate_id", range(len(SUBJECTS)))
    by_target_position, position_gap = _accuracy_by(cases, "target_position", range(len(SUBJECTS)))
    protocol_rate = _rate(cases, "protocol_parsed")
    offered_rate = _rate(cases, "offered_id")
    agreement_rate = _rate(cases, "provider_agreed")
    semantic_accuracy = _rate(cases, "semantic_correct")
    selected_id_counts = Counter(
        case.selected_candidate_id for case in cases if case.selected_candidate_id is not None
    )
    selected_position_counts = Counter(
        case.selected_position for case in cases if case.selected_position is not None
    )
    gates = {
        "complete_96_case_matrix": len(cases) == CASE_COUNT,
        "protocol_parse_100_percent": protocol_rate == 1.0,
        "offered_id_100_percent": offered_rate == 1.0,
        "provider_protocol_agreement_100_percent": agreement_rate == 1.0,
        "semantic_accuracy_at_least_95_percent": (semantic_accuracy >= MIN_SEMANTIC_ACCURACY),
        "no_meaningful_target_id_bias": id_gap <= MAX_BIAS_GAP,
        "no_meaningful_target_position_bias": position_gap <= MAX_BIAS_GAP,
    }
    status = selector.status()
    return {
        "format": "functiongemma-aua-production-smoke-v1",
        "host_only": True,
        "fictional": True,
        "model": model,
        "adapter": adapter,
        "matrix": {
            "subjects": list(SUBJECTS),
            "target_subject": TARGET_SUBJECT,
            "phase": PHASE,
            "candidate_orders": len(orders),
            "dense_id_permutations": [list(value) for value in DENSE_ID_PERMUTATIONS],
            "cases": len(cases),
        },
        "thresholds": {
            "protocol_parse_rate": 1.0,
            "offered_id_rate": 1.0,
            "provider_protocol_agreement_rate": 1.0,
            "minimum_semantic_accuracy": MIN_SEMANTIC_ACCURACY,
            "maximum_accuracy_gap_by_target_id_or_position": MAX_BIAS_GAP,
        },
        "metrics": {
            "protocol_parse_rate": protocol_rate,
            "offered_id_rate": offered_rate,
            "provider_protocol_agreement_rate": agreement_rate,
            "semantic_accuracy": semantic_accuracy,
            "semantic_correct": sum(case.semantic_correct for case in cases),
            "by_target_candidate_id": by_target_id,
            "by_target_position": by_target_position,
            "target_id_accuracy_gap": id_gap,
            "target_position_accuracy_gap": position_gap,
            "selected_candidate_id_counts": {
                str(key): selected_id_counts[key] for key in sorted(selected_id_counts)
            },
            "selected_position_counts": {
                str(key): selected_position_counts[key] for key in sorted(selected_position_counts)
            },
        },
        "provider": {
            "name": selector.name,
            "loaded": status["loaded"],
            "provenance": status["provenance"],
        },
        "gates": gates,
        "passed": all(gates.values()),
        "cases": [case.as_json() for case in cases],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Absolute path to an already-downloaded local MLX FunctionGemma model",
    )
    parser.add_argument(
        "--adapter",
        default=BUNDLED_ADAPTER,
        help="'bundled'/'default' (the packaged adapter) or an absolute adapter directory",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    adapter_display = (
        BUNDLED_ADAPTER
        if str(args.adapter).strip().lower() in {"", "bundled", "default"}
        else str(args.adapter)
    )
    try:
        selector, recorder = build_selector(args.model, args.adapter)
        report = run_smoke(
            selector,
            recorder,
            model=str(Path(args.model).resolve()),
            adapter=adapter_display,
        )
    except Exception as exc:
        report = {
            "format": "functiongemma-aua-production-smoke-v1",
            "host_only": True,
            "fictional": True,
            "model": args.model,
            "adapter": adapter_display,
            "passed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "passed": report["passed"],
        "cases": report.get("matrix", {}).get("cases", 0),
        "semantic_accuracy": report.get("metrics", {}).get("semantic_accuracy"),
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
