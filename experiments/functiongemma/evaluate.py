#!/usr/bin/env python3
"""Evaluate a FunctionGemma AUA candidate-selection policy without a device.

The model never authors an Android mutation.  AUA supplies a bounded list of
fully specified candidate calls and the model selects one integer id.  This
keeps schema validation, authorization, and cleanup in deterministic code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STRICT_CALL = re.compile(
    r"\s*<start_function_call>call:select_candidate\{candidate_id:(-?[0-9]+)\}"
    r"(?:<end_function_call>)?\s*"
)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    family: str
    criticality: str
    expected_candidate_id: int
    predicted_candidate_id: int | None
    expected_tool: str
    predicted_tool: str | None
    parsed: bool
    exactly_one_call: bool
    candidate_exists: bool
    authorized: bool | None
    redundant: bool | None
    raw_output: str
    error: str | None = None

    @property
    def correct(self) -> bool:
        return (
            self.parsed
            and self.exactly_one_call
            and self.candidate_exists
            and self.predicted_candidate_id == self.expected_candidate_id
        )


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def _target(row: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("messages must contain developer, user, and assistant turns")
    assistant = messages[-1]
    calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("assistant must contain exactly one target tool call")
    function = calls[0].get("function", {})
    if function.get("name") != "select_candidate":
        raise ValueError("target function must be select_candidate")
    arguments = function.get("arguments", {})
    target_id = arguments.get("candidate_id")
    if not isinstance(target_id, int) or isinstance(target_id, bool):
        raise ValueError("target candidate_id must be an integer")

    user = messages[-2]
    if not isinstance(user, dict) or not isinstance(user.get("content"), str):
        raise ValueError("user content must be a JSON string")
    state = json.loads(user["content"])
    candidates = state.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("user state must contain candidates")
    by_id = {candidate.get("id"): candidate for candidate in candidates}
    if target_id not in by_id:
        raise ValueError(f"target candidate {target_id} is absent")
    return target_id, by_id[target_id], state


def _prompt(row: dict[str, Any], tokenizer: Any) -> list[int]:
    messages = row["messages"][:-1]
    return tokenizer.apply_chat_template(
        messages,
        tools=row["tools"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _parse_prediction(row: dict[str, Any], output: str, tokenizer: Any) -> Prediction:
    target_id, target_candidate, state = _target(row)
    metadata = row.get("metadata") or {}
    candidates = {
        candidate["id"]: candidate
        for candidate in state["candidates"]
        if isinstance(candidate, dict) and isinstance(candidate.get("id"), int)
    }
    parsed = False
    strict_match = STRICT_CALL.fullmatch(output)
    exactly_one_call = strict_match is not None
    predicted_id: int | None = None
    error: str | None = None
    try:
        if strict_match is None:
            raise ValueError("output is not exactly one canonical FunctionGemma call")
        parsed_call = tokenizer.tool_parser(output, row["tools"])
        if parsed_call.get("name") != "select_candidate":
            raise ValueError(f"unexpected function {parsed_call.get('name')!r}")
        value = parsed_call.get("arguments", {}).get("candidate_id")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"candidate_id is not an integer: {value!r}")
        if value != int(strict_match.group(1)):
            raise ValueError("protocol parser and strict call disagree")
        predicted_id = value
        parsed = True
    except Exception as exc:  # model output is intentionally treated as untrusted input
        error = f"{type(exc).__name__}: {exc}"

    predicted_candidate = candidates.get(predicted_id)
    predicted_call = (
        predicted_candidate.get("call") if isinstance(predicted_candidate, dict) else None
    )
    target_call = target_candidate.get("call") or {}
    return Prediction(
        case_id=str(metadata.get("case_id") or metadata.get("group_id") or "unknown"),
        family=str(metadata.get("family") or metadata.get("intent") or "unknown"),
        criticality=str(metadata.get("criticality") or "normal"),
        expected_candidate_id=target_id,
        predicted_candidate_id=predicted_id,
        expected_tool=str(target_call.get("tool") or "unknown"),
        predicted_tool=(
            str(predicted_call.get("tool"))
            if isinstance(predicted_call, dict) and predicted_call.get("tool")
            else None
        ),
        parsed=parsed,
        exactly_one_call=exactly_one_call,
        candidate_exists=predicted_candidate is not None,
        authorized=(
            bool(predicted_candidate.get("authorized", True))
            if isinstance(predicted_candidate, dict)
            else None
        ),
        redundant=(
            bool(predicted_candidate.get("redundant", False))
            if isinstance(predicted_candidate, dict)
            else None
        ),
        raw_output=output,
        error=error,
    )


def _macro_f1(predictions: list[Prediction]) -> float:
    labels = sorted({prediction.expected_tool for prediction in predictions})
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            prediction.expected_tool == label and prediction.predicted_tool == label
            for prediction in predictions
        )
        false_positive = sum(
            prediction.expected_tool != label and prediction.predicted_tool == label
            for prediction in predictions
        )
        false_negative = sum(
            prediction.expected_tool == label and prediction.predicted_tool != label
            for prediction in predictions
        )
        precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision and recall else 0.0
        )
    return statistics.fmean(scores) if scores else 0.0


def _sequence_metrics(rows: list[dict[str, Any]], predictions: list[Prediction]) -> dict[str, Any]:
    """Report only structurally valid static sequences; never imply closed-loop execution."""
    declared_groups: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        metadata = row.get("metadata") or {}
        episode_id = metadata.get("episode_id")
        if episode_id:
            declared_groups[str(episode_id)].append(
                (
                    int(metadata.get("step", -1)),
                    int(metadata.get("steps_total", -1)),
                    prediction.correct,
                )
            )
    well_formed = {
        episode_id: steps
        for episode_id, steps in declared_groups.items()
        if steps
        and all(total == steps[0][1] and total > 0 for _, total, _ in steps)
        and sorted(step for step, _, _ in steps) == list(range(steps[0][1]))
    }
    completed = sum(all(correct for _, _, correct in steps) for steps in well_formed.values())
    return {
        "declared_groups": len(declared_groups),
        "well_formed_sequences": len(well_formed),
        "all_steps_correct": completed,
        "static_completion_rate": completed / len(well_formed) if well_formed else None,
        "row_coverage": sum(len(steps) for steps in declared_groups.values()) / len(rows),
        "note": (
            "Only structurally valid static sequences are scored. Dataset group IDs are not "
            "closed-loop episodes; runtime simulation is reported separately."
        ),
    }


def _permutation_metrics(
    rows: list[dict[str, Any]], predictions: list[Prediction]
) -> dict[str, Any]:
    """Measure complete opaque-ID/order invariance groups independently of row accuracy."""

    declared: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        metadata = row.get("metadata") or {}
        if metadata.get("permutation_group") is not True:
            continue
        group_id = metadata.get("group_id")
        if group_id is None:
            continue
        declared[str(group_id)].append(
            (
                int(metadata.get("variant", -1)),
                int(metadata.get("permutations_total", -1)),
                prediction.correct,
            )
        )
    well_formed = {
        group_id: variants
        for group_id, variants in declared.items()
        if variants
        and all(total == variants[0][1] and total > 0 for _, total, _ in variants)
        and sorted(variant for variant, _, _ in variants) == list(range(variants[0][1]))
    }
    all_correct = sum(
        all(correct for _, _, correct in variants) for variants in well_formed.values()
    )
    covered_rows = sum(len(variants) for variants in declared.values())
    correct_rows = sum(correct for variants in declared.values() for _, _, correct in variants)
    return {
        "declared_groups": len(declared),
        "well_formed_groups": len(well_formed),
        "all_variants_correct": all_correct,
        "group_accuracy": all_correct / len(well_formed) if well_formed else None,
        "row_accuracy": correct_rows / covered_rows if covered_rows else None,
        "row_coverage": covered_rows / len(rows),
        "note": (
            "A group passes only when every declared candidate-order and opaque-ID variant "
            "selects the same semantic target."
        ),
    }


def _metrics(rows: list[dict[str, Any]], predictions: list[Prediction]) -> dict[str, Any]:
    total = len(predictions)
    by_tool: dict[str, dict[str, Any]] = {}
    for label in sorted({prediction.expected_tool for prediction in predictions}):
        selected = [prediction for prediction in predictions if prediction.expected_tool == label]
        by_tool[label] = {
            "cases": len(selected),
            "accuracy": sum(prediction.correct for prediction in selected) / len(selected),
        }
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({prediction.family for prediction in predictions}):
        selected = [prediction for prediction in predictions if prediction.family == family]
        by_family[family] = {
            "cases": len(selected),
            "accuracy": sum(prediction.correct for prediction in selected) / len(selected),
            "unauthorized_selections": sum(
                prediction.authorized is False for prediction in selected
            ),
            "redundant_selections": sum(prediction.redundant is True for prediction in selected),
        }
    critical = [prediction for prediction in predictions if prediction.criticality == "critical"]
    unauthorized = [prediction for prediction in predictions if prediction.authorized is False]
    redundant = [prediction for prediction in predictions if prediction.redundant is True]
    rows_with_unauthorized = 0
    rows_with_redundant = 0
    unique_eligible_cases = 0
    first_eligible_correct = 0
    lowest_eligible_id_correct = 0
    target_positions: Counter[int] = Counter()
    for row in rows:
        target_id, _, state = _target(row)
        target_positions[target_id] += 1
        candidates = state["candidates"]
        eligible = [
            candidate
            for candidate in candidates
            if candidate.get("authorized") is True
            and candidate.get("redundant") is False
            and candidate.get("risk") == "safe"
        ]
        unique_eligible_cases += len(eligible) == 1
        first_eligible_correct += bool(eligible and eligible[0].get("id") == target_id)
        lowest_eligible_id_correct += bool(
            eligible and min(candidate["id"] for candidate in eligible) == target_id
        )
        rows_with_unauthorized += any(
            candidate.get("authorized") is False for candidate in candidates
        )
        rows_with_redundant += any(candidate.get("redundant") is True for candidate in candidates)
    return {
        "cases": total,
        "parse_success": sum(prediction.parsed for prediction in predictions) / total,
        "exactly_one_call": sum(prediction.exactly_one_call for prediction in predictions) / total,
        "candidate_exists": sum(prediction.candidate_exists for prediction in predictions) / total,
        "candidate_accuracy": sum(prediction.correct for prediction in predictions) / total,
        "tool_accuracy": sum(
            prediction.expected_tool == prediction.predicted_tool for prediction in predictions
        )
        / total,
        "macro_tool_f1": _macro_f1(predictions),
        "critical_cases": len(critical),
        "critical_accuracy": (
            sum(prediction.correct for prediction in critical) / len(critical) if critical else None
        ),
        "unauthorized_selections": len(unauthorized),
        "unauthorized_selection_rate": len(unauthorized) / total,
        "cases_exposing_unauthorized_candidate": rows_with_unauthorized,
        "unauthorized_rate_when_exposed": (
            len(unauthorized) / rows_with_unauthorized if rows_with_unauthorized else None
        ),
        "redundant_selections": len(redundant),
        "redundant_selection_rate": len(redundant) / total,
        "cases_exposing_redundant_candidate": rows_with_redundant,
        "redundant_rate_when_exposed": (
            len(redundant) / rows_with_redundant if rows_with_redundant else None
        ),
        "by_expected_tool": by_tool,
        "by_family": by_family,
        "sequences": _sequence_metrics(rows, predictions),
        "permutation_groups": _permutation_metrics(rows, predictions),
        "predicted_candidate_histogram": dict(
            sorted(Counter(str(item.predicted_candidate_id) for item in predictions).items())
        ),
        "target_candidate_histogram": dict(sorted(target_positions.items())),
        "majority_candidate_id_baseline": max(target_positions.values()) / total,
        "deterministic_flag_baselines": {
            "unique_eligible_cases": unique_eligible_cases,
            "first_eligible_accuracy": first_eligible_correct / total,
            "lowest_eligible_id_accuracy": lowest_eligible_id_correct / total,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_adapter(model: str, adapter: Path | None) -> dict[str, Any] | None:
    if adapter is None:
        return None
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise ValueError(f"incomplete adapter directory: {adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_model = str(config.get("model") or "")
    if not configured_model:
        raise ValueError("adapter config does not identify its base model")
    requested = Path(model)
    configured = Path(configured_model)
    if requested.exists() and configured.exists():
        matches = requested.resolve() == configured.resolve()
    else:
        matches = configured_model == model
    if not matches:
        raise ValueError(
            f"adapter base model mismatch: configured={configured_model!r}, requested={model!r}"
        )
    if config.get("fine_tune_type") not in {"lora", "dora", "full"}:
        raise ValueError(f"unsupported adapter fine_tune_type: {config.get('fine_tune_type')!r}")
    return {
        "config": config,
        "config_sha256": _sha256(config_path),
        "weights_sha256": _sha256(weights_path),
        "weights_bytes": weights_path.stat().st_size,
    }


def _validate_tokenizer(tokenizer: Any) -> None:
    if not tokenizer.has_chat_template or not callable(tokenizer.tool_parser):
        raise ValueError("model tokenizer lacks FunctionGemma chat/tool support")
    if not tokenizer.tool_call_start or not tokenizer.tool_call_end:
        raise ValueError("model tokenizer lacks FunctionGemma tool boundaries")
    smoke = "<start_function_call>call:select_candidate{candidate_id:3}"
    parsed = tokenizer.tool_parser(smoke, [])
    if parsed != {"name": "select_candidate", "arguments": {"candidate_id": 3}}:
        raise ValueError(f"FunctionGemma parser smoke failed: {parsed!r}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # Keep policy parsing/metrics importable in the repository's ordinary test
    # environment; MLX is an experiment-only dependency loaded at execution time.
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    rows = _read_jsonl(args.data, args.limit)
    adapter_provenance = _validate_adapter(args.model, args.adapter)
    started = time.perf_counter()
    model, tokenizer = load(args.model, adapter_path=str(args.adapter) if args.adapter else None)
    # The converted tokenizer exposes FunctionGemma's tool boundary but does not list it as
    # EOS.  Stop there explicitly so one policy decision cannot spill into a second call.
    if tokenizer.tool_call_end:
        tokenizer.add_eos_token(tokenizer.tool_call_end)
    _validate_tokenizer(tokenizer)
    load_seconds = time.perf_counter() - started

    tokenized = [_prompt(row, tokenizer) for row in rows]
    predictions: list[Prediction] = []
    generation_started = time.perf_counter()
    sampler = make_sampler(temp=0.0)
    for row_chunk, prompt_chunk in zip(
        _chunks(rows, args.batch_size), _chunks(tokenized, args.batch_size), strict=True
    ):
        response = batch_generate(
            model,
            tokenizer,
            prompt_chunk,
            max_tokens=args.max_tokens,
            sampler=sampler,
            prefill_batch_size=min(args.prefill_batch_size, len(prompt_chunk)),
            completion_batch_size=min(args.batch_size, len(prompt_chunk)),
            verbose=False,
        )
        predictions.extend(
            _parse_prediction(row, output, tokenizer)
            for row, output in zip(row_chunk, response.texts, strict=True)
        )
    generation_seconds = time.perf_counter() - generation_started

    result = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_provenance": adapter_provenance,
        "dataset": str(args.data),
        "dataset_sha256": _sha256(args.data),
        "limit": args.limit,
        "greedy": True,
        "max_tokens": args.max_tokens,
        "timing": {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "cases_per_second": len(rows) / generation_seconds,
        },
        "metrics": _metrics(rows, predictions),
        "predictions": [
            asdict(prediction) | {"correct": prediction.correct} for prediction in predictions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="MLX model directory or Hugging Face id")
    parser.add_argument("--adapter", type=Path, help="Optional MLX LoRA/full adapter directory")
    parser.add_argument("--data", type=Path, required=True, help="Held-out JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Metrics and predictions JSON")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--min-candidate-accuracy", type=float, default=0.0)
    parser.add_argument("--min-critical-accuracy", type=float, default=0.0)
    parser.add_argument("--min-parse-success", type=float, default=0.0)
    parser.add_argument("--max-unauthorized-selections", type=int)
    parser.add_argument("--max-redundant-selections", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.prefill_batch_size < 1 or args.max_tokens < 1:
        raise SystemExit("batch sizes and max tokens must be positive")
    result = evaluate(args)
    metrics = result["metrics"]
    print(
        json.dumps(
            {
                "cases": metrics["cases"],
                "candidate_accuracy": metrics["candidate_accuracy"],
                "critical_accuracy": metrics["critical_accuracy"],
                "parse_success": metrics["parse_success"],
                "exactly_one_call": metrics["exactly_one_call"],
                "unauthorized_selections": metrics["unauthorized_selections"],
                "redundant_selections": metrics["redundant_selections"],
                "cases_per_second": result["timing"]["cases_per_second"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    failures: list[str] = []
    if metrics["candidate_accuracy"] < args.min_candidate_accuracy:
        failures.append("candidate_accuracy")
    critical_accuracy = metrics["critical_accuracy"]
    if args.min_critical_accuracy > 0 and critical_accuracy is None:
        failures.append("critical_cases_missing")
    elif critical_accuracy is not None and critical_accuracy < args.min_critical_accuracy:
        failures.append("critical_accuracy")
    if metrics["parse_success"] < args.min_parse_success:
        failures.append("parse_success")
    if (
        args.max_unauthorized_selections is not None
        and metrics["unauthorized_selections"] > args.max_unauthorized_selections
    ):
        failures.append("unauthorized_selections")
    if (
        args.max_redundant_selections is not None
        and metrics["redundant_selections"] > args.max_redundant_selections
    ):
        failures.append("redundant_selections")
    if failures:
        raise SystemExit("evaluation gates failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
