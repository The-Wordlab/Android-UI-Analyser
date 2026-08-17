#!/usr/bin/env python3
"""Host-only FunctionGemma training worker executed inside one RunPod Pod.

The Hugging Face token is read once from stdin. It is never accepted as an
argument, written to disk, placed in the Pod environment, or included in run
metadata. This module does not import AUA's Android runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "mlx-community/functiongemma-270m-it-bf16"
DEFAULT_MODEL_REVISION = "bb327a9ad61044e1496a2bee2365a6b6a6684c72"
DEFAULT_MLX_PACKAGE = "mlx[cuda12]==0.32.0"
DEFAULT_CONFIG = "experiments/functiongemma/train-lora.yaml"
REQUIREMENTS = REPO_ROOT / "experiments" / "functiongemma" / "requirements.txt"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(value: str, secret: str) -> str:
    return value.replace(secret, "<redacted>") if secret else value


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def _install_dependencies(mlx_package: str) -> None:
    common = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--break-system-packages",
    ]
    _run([*common, mlx_package])
    _run([*common, "-r", str(REQUIREMENTS)])


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "mlx",
        "mlx-cuda-12",
        "mlx-lm",
        "huggingface-hub",
        "transformers",
        "tokenizers",
        "datasets",
        "safetensors",
        "numpy",
        "PyYAML",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "missing"
    return result


def _gpu_details() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            result.append({"name": fields[0], "memory_mib": fields[1], "driver_version": fields[2]})
    return result


def _config_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("training config escapes the repository") from exc
    if not path.is_file():
        raise ValueError(f"training config is missing: {relative}")
    return path


def _copy_evidence(data_dir: Path, config_path: Path, output_dir: Path) -> None:
    evidence = output_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for source in (data_dir / "manifest.json", data_dir / "validation.json", config_path):
        shutil.copy2(source, evidence / source.name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "benchmark", "full"), default="benchmark")
    parser.add_argument("--curriculum-version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--mlx-package", default=DEFAULT_MLX_PACKAGE)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--max-runtime-seconds", type=int, required=True)
    return parser


def run(args: argparse.Namespace, *, token_input: TextIO = sys.stdin) -> dict[str, Any]:
    output_dir = args.output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    run_root = output_dir.parent
    metadata_path = output_dir / "worker-metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "phase": "initializing",
        "started_at": _utc_now(),
        "completed_at": None,
        "mode": args.mode,
        "curriculum_version": args.curriculum_version,
        "model": {"repository": args.model_id, "revision": args.model_revision},
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
        "error": None,
    }
    _atomic_json(metadata_path, metadata)

    secret = token_input.readline().strip()
    if not secret:
        raise ValueError("HF_TOKEN was not provided on stdin")

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("RunPod worker exceeded its runtime limit")

    previous_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.max_runtime_seconds)
    phase_started = time.monotonic()
    try:
        metadata["phase"] = "installing_dependencies"
        _atomic_json(metadata_path, metadata)
        _install_dependencies(args.mlx_package)
        metadata["durations_seconds"]["dependency_install"] = round(
            time.monotonic() - phase_started, 3
        )

        metadata["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
            "gpus": _gpu_details(),
        }
        if not metadata["environment"]["gpus"]:
            raise RuntimeError("nvidia-smi did not report a CUDA GPU")

        metadata["phase"] = "downloading_model"
        _atomic_json(metadata_path, metadata)
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
        metadata["durations_seconds"]["model_download"] = round(time.monotonic() - phase_started, 3)
        secret = ""

        data_dir = run_root / "data"
        metadata["phase"] = "generating_dataset"
        _atomic_json(metadata_path, metadata)
        phase_started = time.monotonic()
        _run(
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
        _run(
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
        manifest_hash = _sha256(data_dir / "manifest.json")
        if args.expected_manifest_sha256 and manifest_hash != args.expected_manifest_sha256:
            raise ValueError(
                "generated manifest SHA256 does not match the launcher pin: "
                f"expected {args.expected_manifest_sha256}, found {manifest_hash}"
            )
        metadata["dataset_manifest_sha256"] = manifest_hash
        metadata["durations_seconds"]["dataset_generation_and_validation"] = round(
            time.monotonic() - phase_started, 3
        )

        metadata["phase"] = "training"
        _atomic_json(metadata_path, metadata)
        phase_started = time.monotonic()
        from experiments.functiongemma import train  # noqa: PLC0415

        adapter_dir = output_dir / "adapter"
        training = train.run(
            mode=args.mode,
            config_path=_config_path(args.config),
            model=model_path,
            data_dir=data_dir,
            adapter_path=adapter_dir,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        metadata["durations_seconds"]["training"] = round(time.monotonic() - phase_started, 3)
        metadata["training_metadata"] = {
            "status": training["status"],
            "completed_checkpoint": training["completed_checkpoint"],
            "final_adapter_hashes": training["final_adapter_hashes"],
        }
        _copy_evidence(data_dir, _config_path(args.config), output_dir)
        metadata.update(
            {
                "status": "completed",
                "phase": "completed",
                "completed_at": _utc_now(),
            }
        )
        _atomic_json(metadata_path, metadata)
        return metadata
    except BaseException as exc:
        metadata.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "completed_at": _utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": _redact(str(exc), secret),
                },
            }
        )
        _atomic_json(metadata_path, metadata)
        raise RuntimeError(
            f"FunctionGemma worker failed during {metadata['phase']}: "
            f"{type(exc).__name__}: {_redact(str(exc), secret)}"
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
