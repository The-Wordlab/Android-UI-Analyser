#!/usr/bin/env python3
"""Select a FunctionGemma checkpoint by strict safety and worst-family accuracy.

This host-only utility evaluates every saved MLX LoRA checkpoint on validation
data.  A checkpoint is eligible only when parsing and offered-ID checks are
perfect and it makes zero unauthorized or redundant selections.  Among eligible
checkpoints, the winner maximizes the weakest family before critical and overall
accuracy.  The untouched test split is evaluated exactly once, after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return value


def checkpoint_score(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("evaluation report has no metrics object")
    by_family = metrics.get("by_family")
    if not isinstance(by_family, Mapping) or not by_family:
        raise ValueError("evaluation report has no by_family metrics")
    family_accuracies = {
        str(name): float(value["accuracy"])
        for name, value in by_family.items()
        if isinstance(value, Mapping) and "accuracy" in value
    }
    if len(family_accuracies) != len(by_family):
        raise ValueError("evaluation report has an invalid family accuracy")
    permutation_groups = metrics.get("permutation_groups")
    permutation_accuracy: float | None = None
    permutation_well_formed = True
    if isinstance(permutation_groups, Mapping):
        declared = int(permutation_groups.get("declared_groups", 0))
        well_formed = int(permutation_groups.get("well_formed_groups", 0))
        value = permutation_groups.get("group_accuracy")
        if declared > 0:
            permutation_accuracy = float(value) if value is not None else 0.0
            permutation_well_formed = well_formed == declared
    strict_safety = (
        float(metrics.get("parse_success", 0.0)) == 1.0
        and float(metrics.get("exactly_one_call", 0.0)) == 1.0
        and float(metrics.get("candidate_exists", 0.0)) == 1.0
        and int(metrics.get("unauthorized_selections", -1)) == 0
        and int(metrics.get("redundant_selections", -1)) == 0
        and permutation_well_formed
        and permutation_accuracy in {None, 1.0}
    )
    critical = metrics.get("critical_accuracy")
    return {
        "strict_safety_passed": strict_safety,
        "worst_family_accuracy": min(family_accuracies.values()),
        "worst_families": sorted(
            name
            for name, accuracy in family_accuracies.items()
            if accuracy == min(family_accuracies.values())
        ),
        "critical_accuracy": float(critical) if critical is not None else 0.0,
        "candidate_accuracy": float(metrics.get("candidate_accuracy", 0.0)),
        "permutation_group_accuracy": permutation_accuracy,
        "permutation_groups_well_formed": permutation_well_formed,
        "parse_success": float(metrics.get("parse_success", 0.0)),
        "unauthorized_selections": int(metrics.get("unauthorized_selections", -1)),
        "redundant_selections": int(metrics.get("redundant_selections", -1)),
    }


def _checkpoint_iteration(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else 2**31 - 1


def select_checkpoint(reports: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    """Return the strict validation winner and an auditable ranking."""

    ranked: list[dict[str, Any]] = []
    for checkpoint, report in reports:
        score = checkpoint_score(report)
        ranked.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                **score,
            }
        )
    ranked.sort(
        key=lambda item: (
            bool(item["strict_safety_passed"]),
            float(item["worst_family_accuracy"]),
            float(item["permutation_group_accuracy"] or 0.0),
            float(item["critical_accuracy"]),
            float(item["candidate_accuracy"]),
            -_checkpoint_iteration(Path(str(item["checkpoint"]))),
        ),
        reverse=True,
    )
    eligible = [item for item in ranked if item["strict_safety_passed"]]
    return {
        "selection_policy": (
            "strict parse/offered ID and zero unauthorized/redundant; then max worst-family, "
            "critical, overall accuracy; earlier checkpoint wins an exact tie"
        ),
        "eligible_checkpoints": len(eligible),
        "selected": eligible[0] if eligible else None,
        "ranked": ranked,
    }


def _checkpoint_files(adapter_dir: Path) -> list[Path]:
    checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
    if not checkpoints:
        final = adapter_dir / "adapters.safetensors"
        if final.is_file():
            checkpoints = [final]
    if not checkpoints or any(path.stat().st_size == 0 for path in checkpoints):
        raise ValueError(f"adapter directory has no complete checkpoints: {adapter_dir}")
    return checkpoints


def _run_evaluation(
    *,
    model: Path,
    adapter_dir: Path,
    checkpoint: Path,
    data: Path,
    output: Path,
    evaluator_module: str = "experiments.functiongemma.evaluate",
    batch_size: int = 32,
    prefill_batch_size: int = 8,
) -> dict[str, Any]:
    staging = output.parent / f".{checkpoint.stem}-adapter"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(adapter_dir / "adapter_config.json", staging / "adapter_config.json")
    os.symlink(checkpoint.resolve(), staging / "adapters.safetensors")
    try:
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                evaluator_module,
                "--model",
                str(model),
                "--adapter",
                str(staging),
                "--data",
                str(data),
                "--output",
                str(output),
                "--batch-size",
                str(batch_size),
                "--prefill-batch-size",
                str(prefill_batch_size),
                "--max-tokens",
                "48",
            ],
            check=True,
        )
        return _load_json(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.resolve()
    adapter_dir = args.adapter_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not model.is_dir() or not adapter_dir.is_dir() or not data_dir.is_dir():
        raise ValueError("model, adapter-dir, and data-dir must be existing local directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_module = getattr(args, "evaluator_module", "experiments.functiongemma.evaluate")
    batch_size = int(getattr(args, "batch_size", 32))
    prefill_batch_size = int(getattr(args, "prefill_batch_size", 8))
    if batch_size < 1 or prefill_batch_size < 1 or prefill_batch_size > batch_size:
        raise ValueError("evaluation batch sizes must satisfy 1 <= prefill <= batch")
    reports: list[tuple[Path, Mapping[str, Any]]] = []
    for checkpoint in _checkpoint_files(adapter_dir):
        report_path = output_dir / f"validation-{checkpoint.stem}.json"
        report = _run_evaluation(
            model=model,
            adapter_dir=adapter_dir,
            checkpoint=checkpoint,
            data=data_dir / "valid.jsonl",
            output=report_path,
            evaluator_module=evaluator_module,
            batch_size=batch_size,
            prefill_batch_size=prefill_batch_size,
        )
        reports.append((checkpoint, report))
    selection = select_checkpoint(reports)
    selected = selection["selected"]
    if selected is None:
        summary = {**selection, "test_evaluated": False, "strict_test_passed": False}
    else:
        selected_checkpoint = Path(str(selected["checkpoint"]))
        selected_dir = output_dir / "selected-adapter"
        selected_dir.mkdir(exist_ok=False)
        shutil.copy2(adapter_dir / "adapter_config.json", selected_dir / "adapter_config.json")
        shutil.copy2(selected_checkpoint, selected_dir / "adapters.safetensors")
        test_report = _run_evaluation(
            model=model,
            adapter_dir=selected_dir,
            checkpoint=selected_dir / "adapters.safetensors",
            data=data_dir / "test.jsonl",
            output=output_dir / "test-selected.json",
            evaluator_module=evaluator_module,
            batch_size=batch_size,
            prefill_batch_size=prefill_batch_size,
        )
        test_score = checkpoint_score(test_report)
        summary = {
            **selection,
            "selected_adapter": str(selected_dir),
            "selected_adapter_sha256": _sha256(selected_dir / "adapters.safetensors"),
            "test_evaluated": True,
            "test_score": test_score,
            "strict_test_passed": test_score["strict_safety_passed"],
        }
    summary_path = output_dir / "checkpoint-selection.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluator-module",
        default="experiments.functiongemma.evaluate",
        choices=(
            "experiments.functiongemma.evaluate",
            "experiments.functiongemma.evaluate_qwen",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    return parser


def main() -> None:
    result = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "eligible_checkpoints": result["eligible_checkpoints"],
                "selected": result["selected"],
                "strict_test_passed": result["strict_test_passed"],
            },
            sort_keys=True,
        )
    )
    if result["selected"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
