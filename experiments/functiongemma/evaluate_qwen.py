#!/usr/bin/env python3
"""Evaluate a Qwen3 MLX LoRA on the frozen AUA candidate-policy contract."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from experiments.functiongemma import evaluate as common

STRICT_QWEN_CALL = re.compile(
    r"\s*(?:<think>.*?</think>\s*)?<tool_call>\s*(\{.*\})\s*</tool_call>\s*",
    re.DOTALL,
)
STOP_STRIPPED_QWEN_CALL = re.compile(
    r"\s*(?:<think>.*?</think>\s*)?<tool_call>\s*(\{.*\})\s*",
    re.DOTALL,
)
QWEN_TOOL_CALL_END = "</tool_call>"


def _prompt(row: dict[str, Any], tokenizer: Any) -> list[int]:
    return tokenizer.apply_chat_template(
        row["messages"][:-1],
        tools=row["tools"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


def _parse_prediction(row: dict[str, Any], output: str) -> common.Prediction:
    target_id, target_candidate, state = common._target(row)  # noqa: SLF001
    metadata = row.get("metadata") or {}
    candidates = common._candidate_map(state)  # noqa: SLF001
    match = STRICT_QWEN_CALL.fullmatch(output)
    predicted_id: int | None = None
    parsed = False
    error: str | None = None
    try:
        if match is None:
            raise ValueError("output is not exactly one canonical Qwen tool call")
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict) or payload.get("name") != "select_candidate":
            raise ValueError("output calls an unexpected function")
        arguments = payload.get("arguments")
        value = arguments.get("candidate_id") if isinstance(arguments, dict) else None
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("candidate_id is not an integer")
        predicted_id = value
        parsed = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    predicted_candidate = candidates.get(predicted_id) if predicted_id is not None else None
    predicted_call = (
        predicted_candidate.get("call") if isinstance(predicted_candidate, dict) else None
    )
    target_call = target_candidate.get("call") or {}
    return common.Prediction(
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
        exactly_one_call=match is not None,
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


def _restore_generation_stop(output: str) -> tuple[str, bool]:
    """Restore only the configured EOS token that MLX omits from generated text."""
    if STRICT_QWEN_CALL.fullmatch(output) is not None:
        return output, False
    if STOP_STRIPPED_QWEN_CALL.fullmatch(output) is None:
        return output, False
    return f"{output.rstrip()}\n{QWEN_TOOL_CALL_END}", True


def _parse_generated_prediction(row: dict[str, Any], output: str) -> tuple[common.Prediction, bool]:
    parseable, restored = _restore_generation_stop(output)
    prediction = _parse_prediction(row, parseable)
    return replace(prediction, raw_output=output), restored


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_lm import load  # noqa: PLC0415
    from mlx_lm.generate import batch_generate  # noqa: PLC0415
    from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

    rows = common._read_jsonl(args.data, args.limit)  # noqa: SLF001
    provenance = common._validate_adapter(args.model, args.adapter)  # noqa: SLF001
    started = time.perf_counter()
    model, tokenizer = cast(
        tuple[Any, Any],
        load(args.model, adapter_path=str(args.adapter) if args.adapter else None),
    )
    tokenizer.add_eos_token("</tool_call>")
    load_seconds = time.perf_counter() - started
    tokenized = [_prompt(row, tokenizer) for row in rows]
    predictions: list[common.Prediction] = []
    restored_stop_tokens = 0
    generation_started = time.perf_counter()
    sampler = make_sampler(temp=0.0)
    for row_chunk, prompt_chunk in zip(
        common._chunks(rows, args.batch_size),  # noqa: SLF001
        common._chunks(tokenized, args.batch_size),  # noqa: SLF001
        strict=True,
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
        for row, output in zip(row_chunk, response.texts, strict=True):
            prediction, restored = _parse_generated_prediction(row, output)
            predictions.append(prediction)
            restored_stop_tokens += int(restored)
    generation_seconds = time.perf_counter() - generation_started
    result = {
        "protocol": "qwen3-native-tool-call-v1",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_provenance": provenance,
        "dataset": str(args.data),
        "dataset_sha256": common._sha256(args.data),  # noqa: SLF001
        "limit": args.limit,
        "greedy": True,
        "max_tokens": args.max_tokens,
        "generation_stop": {
            "token": QWEN_TOOL_CALL_END,
            "omitted_by_generator": True,
            "restored_before_strict_parse": restored_stop_tokens,
        },
        "timing": {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "cases_per_second": len(rows) / generation_seconds,
        },
        "metrics": common._metrics(rows, predictions),  # noqa: SLF001
        "predictions": [
            asdict(prediction) | {"correct": prediction.correct} for prediction in predictions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate(args)
    metrics = result["metrics"]
    print(
        json.dumps(
            {
                "cases": metrics["cases"],
                "candidate_accuracy": metrics["candidate_accuracy"],
                "critical_accuracy": metrics["critical_accuracy"],
                "parse_success": metrics["parse_success"],
                "unauthorized_selections": metrics["unauthorized_selections"],
                "redundant_selections": metrics["redundant_selections"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
