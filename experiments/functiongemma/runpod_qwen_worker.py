#!/usr/bin/env python3
"""Host-only Qwen3-0.6B AUA-policy training worker for one RunPod Pod."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from experiments.functiongemma import runpod_worker as base

DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-bf16"
DEFAULT_MODEL_REVISION = "9cd6692855d3e06772228e9a962b2606359b2d24"
TOKENIZER_MODEL = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_CONFIG = "experiments/functiongemma/train-lora-qwen3-1.7b-v10.yaml"
EVAL_BATCH_SIZE = 32
EVAL_PREFILL_BATCH_SIZE = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--curriculum-version", choices=("v7", "v10"), default="v7")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--mlx-package", default=base.DEFAULT_MLX_PACKAGE)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--max-runtime-seconds", type=int, required=True)
    parser.add_argument("--input-adapter-dir", type=Path)
    parser.add_argument("--input-adapter-archive-sha256")
    parser.add_argument("--input-data-dir", type=Path)
    parser.add_argument("--input-data-archive-sha256")
    return parser


def run(args: argparse.Namespace, *, token_input: TextIO = sys.stdin) -> dict[str, Any]:
    mlx_cuda_runtime = base._configure_mlx_cuda_runtime()  # noqa: SLF001
    output_dir = args.output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    run_root = output_dir.parent
    metadata_path = output_dir / "worker-metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "qwen3-0.6b-aua-policy-v7",
        "run_id": args.run_id,
        "status": "running",
        "phase": "initializing",
        "started_at": base._utc_now(),  # noqa: SLF001
        "completed_at": None,
        "mode": "evaluation_only" if args.input_adapter_dir else "full",
        "curriculum_version": "v7",
        "model": {
            "repository": args.model_id,
            "revision": args.model_revision,
            "tokenizer_repository": TOKENIZER_MODEL,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_config_sha256": None,
        },
        "mlx_package": args.mlx_package,
        "source": {
            "base_revision": args.source_revision,
            "archive_sha256": args.source_archive_sha256,
            "includes_reviewed_worktree_overrides": True,
        },
        "environment": {},
        "durations_seconds": {},
        "dataset_manifest_sha256": None,
        "training_metadata": None,
        "input_adapter": (
            {
                "directory": str(args.input_adapter_dir),
                "archive_sha256": args.input_adapter_archive_sha256,
            }
            if args.input_adapter_dir
            else None
        ),
        "input_dataset": (
            {
                "directory": str(args.input_data_dir),
                "archive_sha256": args.input_data_archive_sha256,
            }
            if args.input_data_dir
            else None
        ),
        "checkpoint_selection": None,
        "evaluation_contract": {
            "batch_size": EVAL_BATCH_SIZE,
            "prefill_batch_size": EVAL_PREFILL_BATCH_SIZE,
            "greedy": True,
        },
        "quality_gates": None,
        "error": None,
    }
    base._atomic_json(metadata_path, metadata)  # noqa: SLF001
    secret = token_input.readline().strip()
    if not secret:
        raise ValueError("HF_TOKEN was not provided on stdin")

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Qwen RunPod worker exceeded its runtime limit")

    previous_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.max_runtime_seconds)
    phase_started = time.monotonic()
    try:
        metadata["phase"] = "installing_dependencies"
        base._atomic_json(metadata_path, metadata)  # noqa: SLF001
        base._install_dependencies(args.mlx_package)  # noqa: SLF001
        metadata["durations_seconds"]["dependency_install"] = round(
            time.monotonic() - phase_started, 3
        )
        metadata["environment"] = {
            "python": sys.version.split()[0],
            "packages": base._package_versions(),  # noqa: SLF001
            "gpus": base._gpu_details(),  # noqa: SLF001
            "mlx_cuda": mlx_cuda_runtime,
        }
        if not metadata["environment"]["gpus"]:
            raise RuntimeError("nvidia-smi did not report a CUDA GPU")

        metadata["phase"] = "downloading_model"
        base._atomic_json(metadata_path, metadata)  # noqa: SLF001
        phase_started = time.monotonic()
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        model_path = Path(
            snapshot_download(
                repo_id=args.model_id,
                revision=args.model_revision,
                cache_dir=run_root / "hf-cache",
                token=secret,
            )
        ).resolve()
        tokenizer_path = Path(
            snapshot_download(
                repo_id=TOKENIZER_MODEL,
                revision=TOKENIZER_REVISION,
                allow_patterns=["tokenizer_config.json"],
                cache_dir=run_root / "hf-cache",
                token=secret,
            )
        ).resolve()
        source_tokenizer_config = tokenizer_path / "tokenizer_config.json"
        destination_tokenizer_config = model_path / "tokenizer_config.json"
        destination_tokenizer_config.unlink()
        shutil.copy2(source_tokenizer_config, destination_tokenizer_config)
        metadata["model"]["tokenizer_config_sha256"] = base._sha256(  # noqa: SLF001
            destination_tokenizer_config
        )
        metadata["durations_seconds"]["model_download"] = round(time.monotonic() - phase_started, 3)
        secret = ""

        from experiments.functiongemma import train  # noqa: PLC0415

        if args.input_data_dir is None:
            data_dir = run_root / "data"
            metadata["phase"] = "generating_dataset"
            base._atomic_json(metadata_path, metadata)  # noqa: SLF001
            phase_started = time.monotonic()
            base._run(  # noqa: SLF001
                [
                    sys.executable,
                    "-m",
                    "experiments.functiongemma.generate_dataset",
                    "--output-dir",
                    str(data_dir),
                    "--curriculum-version",
                    args.curriculum_version,
                ]
            )
            base._run(  # noqa: SLF001
                [
                    sys.executable,
                    "-m",
                    "experiments.functiongemma.validate_dataset",
                    "--model",
                    str(model_path),
                    "--data-dir",
                    str(data_dir),
                    "--max-seq-length",
                    "1024",
                ]
            )
            dataset_identity = train._validate_dataset(  # noqa: SLF001
                data_dir,
                max_seq_length=1024,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            metadata["durations_seconds"]["dataset_generation_and_validation"] = round(
                time.monotonic() - phase_started, 3
            )
        else:
            data_dir = args.input_data_dir.resolve()
            metadata["phase"] = "verifying_input_dataset"
            base._atomic_json(metadata_path, metadata)  # noqa: SLF001
            phase_started = time.monotonic()
            dataset_identity = train._validate_dataset(  # noqa: SLF001
                data_dir,
                max_seq_length=1024,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            metadata["durations_seconds"]["dataset_hash_verification"] = round(
                time.monotonic() - phase_started, 3
            )
        metadata["dataset_manifest_sha256"] = dataset_identity["manifest_sha256"]

        if args.input_adapter_dir is None:
            metadata["phase"] = "training"
            base._atomic_json(metadata_path, metadata)  # noqa: SLF001
            phase_started = time.monotonic()
            adapter_dir = output_dir / "adapter"
            training = train.run(
                mode="full",
                config_path=base._config_path(args.config),  # noqa: SLF001
                model=model_path,
                data_dir=data_dir,
                adapter_path=adapter_dir,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            metadata["durations_seconds"]["training"] = round(time.monotonic() - phase_started, 3)
        else:
            metadata["phase"] = "validating_input_adapter"
            base._atomic_json(metadata_path, metadata)  # noqa: SLF001
            adapter_dir = args.input_adapter_dir.resolve()
            training_path = adapter_dir / "run-metadata.json"
            training = json.loads(training_path.read_text(encoding="utf-8"))
            if training.get("status") != "completed" or training.get("mode") != "full":
                raise ValueError("input adapter metadata is not a completed full training run")
            training_model = training.get("model") or {}
            training_dataset = training.get("dataset") or {}
            if training_model.get("revision") != args.model_revision:
                raise ValueError("input adapter model revision differs from the pinned Qwen base")
            if training_dataset.get("manifest_sha256") != args.expected_manifest_sha256:
                raise ValueError("input adapter dataset differs from the pinned v7 corpus")
            hashes = training.get("final_adapter_hashes")
            if not isinstance(hashes, dict):
                raise ValueError("input adapter metadata has no completed file hashes")
            for name, expected_hash in hashes.items():
                candidate = adapter_dir / name
                if not candidate.is_file() or base._sha256(candidate) != expected_hash:  # noqa: SLF001
                    raise ValueError(f"input adapter file hash mismatch: {name}")
            metadata["durations_seconds"]["training"] = 0.0
        metadata["training_metadata"] = {
            "status": training["status"],
            "completed_checkpoint": training["completed_checkpoint"],
            "final_adapter_hashes": training["final_adapter_hashes"],
        }

        metadata["phase"] = "selecting_checkpoint"
        base._atomic_json(metadata_path, metadata)  # noqa: SLF001
        phase_started = time.monotonic()
        from experiments.functiongemma import select_checkpoint  # noqa: PLC0415

        evaluation_dir = output_dir / "evaluation"
        selection = select_checkpoint.run(
            argparse.Namespace(
                model=model_path,
                adapter_dir=adapter_dir,
                data_dir=data_dir,
                output_dir=evaluation_dir,
                evaluator_module="experiments.functiongemma.evaluate_qwen",
                batch_size=EVAL_BATCH_SIZE,
                prefill_batch_size=EVAL_PREFILL_BATCH_SIZE,
            )
        )
        metadata["durations_seconds"]["checkpoint_selection_and_static_eval"] = round(
            time.monotonic() - phase_started, 3
        )
        metadata["checkpoint_selection"] = {
            "eligible_checkpoints": selection["eligible_checkpoints"],
            "selected": selection["selected"],
            "selected_adapter_sha256": selection.get("selected_adapter_sha256"),
            "strict_test_passed": selection["strict_test_passed"],
        }
        smoke_passed = False
        selected_adapter = selection.get("selected_adapter")
        if selected_adapter:
            metadata["phase"] = "running_quality_gates"
            base._atomic_json(metadata_path, metadata)  # noqa: SLF001
            smoke = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "experiments.functiongemma.run_qwen_semantic_context_smoke",
                    "--model",
                    str(model_path),
                    "--adapter",
                    str(selected_adapter),
                    "--output",
                    str(evaluation_dir / "semantic-context-smoke.json"),
                ],
                cwd=base.REPO_ROOT,
                check=False,
            )
            smoke_passed = smoke.returncode == 0
        metadata["quality_gates"] = {
            "strict_static_test": bool(selection["strict_test_passed"]),
            "semantic_context_smoke": smoke_passed,
            "passed": bool(selection["strict_test_passed"] and smoke_passed),
        }
        if args.input_adapter_dir is not None:
            exported_adapter = output_dir / "adapter"
            exported_adapter.mkdir(exist_ok=False)
            for name in ("run-metadata.json", "adapter_config.json", "adapters.safetensors"):
                shutil.copy2(adapter_dir / name, exported_adapter / name)
        base._copy_evidence(  # noqa: SLF001
            data_dir,
            base._config_path(args.config),
            output_dir,  # noqa: SLF001
        )
        metadata.update(
            {"status": "completed", "phase": "completed", "completed_at": base._utc_now()}
        )
        base._atomic_json(metadata_path, metadata)  # noqa: SLF001
        return metadata
    except BaseException as exc:
        metadata.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "completed_at": base._utc_now(),  # noqa: SLF001
                "error": {"type": type(exc).__name__, "message": base._redact(str(exc), secret)},  # noqa: SLF001
            }
        )
        base._atomic_json(metadata_path, metadata)  # noqa: SLF001
        raise RuntimeError(
            f"Qwen worker failed during {metadata['phase']}: {type(exc).__name__}: "
            f"{base._redact(str(exc), secret)}"  # noqa: SLF001
        ) from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        secret = ""


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except BaseException as exc:
        print(str(exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "training_seconds": result["durations_seconds"].get("training"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
