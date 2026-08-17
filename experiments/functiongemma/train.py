#!/usr/bin/env python3
"""Reproducible, fail-closed MLX training runner for the FunctionGemma experiment.

This is deliberately host-only.  It never imports or invokes AUA's Android runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import types
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SPLITS = ("train", "valid", "test")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "functiongemma" / "train-lora.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "runs" / "functiongemma" / "data"
METADATA_NAME = "run-metadata.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash file names and contents so the model artifact has one stable identity."""
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(_sha256(candidate).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object: {path}")
    return value


def _resolve_local(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _validate_dataset(
    data_dir: Path,
    *,
    max_seq_length: int,
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Verify the frozen manifest, split bytes, and prior tokenizer contract."""
    manifest_path = data_dir / "manifest.json"
    validation_path = data_dir / "validation.json"
    manifest = _load_json(manifest_path, "dataset manifest")
    validation = _load_json(validation_path, "dataset validation report")
    manifest_file_hash = _sha256(manifest_path)
    if expected_manifest_sha256 and manifest_file_hash != expected_manifest_sha256:
        raise ValueError(
            "manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, found {manifest_file_hash}"
        )
    if manifest.get("privacy", {}).get("passed") is not True:
        raise ValueError("dataset manifest does not record a passing privacy audit")
    if validation.get("ok") is not True:
        raise ValueError("dataset validation report is not passing")
    if validation.get("max_seq_length") != max_seq_length:
        raise ValueError(
            "token contract mismatch: validation used "
            f"{validation.get('max_seq_length')!r}, training uses {max_seq_length}"
        )

    split_hashes: dict[str, str] = {}
    split_manifest = manifest.get("splits")
    split_validation = validation.get("splits")
    if not isinstance(split_manifest, dict) or not isinstance(split_validation, dict):
        raise ValueError("manifest or validation report has no split mapping")
    for split in SPLITS:
        manifest_entry = split_manifest.get(split)
        validation_entry = split_validation.get(split)
        if not isinstance(manifest_entry, dict) or not isinstance(validation_entry, dict):
            raise ValueError(f"missing {split} split metadata")
        relative = manifest_entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"manifest has no path for {split}")
        split_path = (data_dir / relative).resolve()
        try:
            split_path.relative_to(data_dir)
        except ValueError as exc:
            raise ValueError(f"manifest path escapes data directory: {relative}") from exc
        if not split_path.is_file():
            raise ValueError(f"dataset split is missing: {split_path}")
        actual_hash = _sha256(split_path)
        declared_hash = manifest_entry.get("sha256")
        if actual_hash != declared_hash:
            raise ValueError(
                f"{split} SHA256 mismatch: expected {declared_hash}, found {actual_hash}"
            )
        if validation_entry.get("sha256") != actual_hash:
            raise ValueError(f"{split} validation hash does not match the frozen split")
        declared_bytes = manifest_entry.get("bytes")
        if declared_bytes is not None and declared_bytes != split_path.stat().st_size:
            raise ValueError(f"{split} byte count does not match the manifest")
        token_stats = validation_entry.get("tokens")
        token_max = token_stats.get("max") if isinstance(token_stats, dict) else None
        if not isinstance(token_max, int) or isinstance(token_max, bool):
            raise ValueError(f"{split} validation report has no integer maximum token count")
        if token_max > max_seq_length:
            raise ValueError(f"{split} maximum token count {token_max} exceeds {max_seq_length}")
        split_hashes[split] = actual_hash

    combined_hash = hashlib.sha256(
        "".join(split_hashes[split] for split in SPLITS).encode()
    ).hexdigest()
    if manifest.get("dataset_sha256") != combined_hash:
        raise ValueError("combined dataset SHA256 does not match the manifest")
    return {
        "directory": str(data_dir),
        "manifest_sha256": manifest_file_hash,
        "dataset_sha256": combined_hash,
        "split_sha256": split_hashes,
        "validation_sha256": _sha256(validation_path),
        "max_seq_length": max_seq_length,
    }


def _model_identity(model: str | Path) -> dict[str, Any]:
    resolved = _resolve_local(model)
    if not resolved.is_dir():
        raise ValueError(
            f"training requires an already-downloaded local model directory; not found: {resolved}"
        )
    revision = resolved.name if resolved.parent.name == "snapshots" else None
    config_path = resolved / "config.json"
    config = _load_json(config_path, "model config") if config_path.is_file() else {}
    return {
        "requested": str(model),
        "resolved_path": str(resolved),
        "sha256": _tree_sha256(resolved),
        "revision": revision,
        "declared_source": config.get("_name_or_path"),
    }


def _load_backend() -> tuple[Any, Any]:
    """Import the optional MLX stack only when a validated run is ready to start."""
    import mlx.core as mx  # noqa: PLC0415
    import mlx_lm.lora as lora  # noqa: PLC0415

    return mx, lora


def _effective_args(
    config: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    mode: str,
    config_path: Path,
    model_path: Path,
    data_dir: Path,
    adapter_path: Path,
    resume_adapter_file: Path | None,
) -> dict[str, Any]:
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"unknown MLX-LoRA configuration keys: {sorted(unknown)}")
    effective = {**defaults, **config}
    effective.update(
        {
            "config": str(config_path),
            "model": str(model_path),
            "data": str(data_dir),
            "adapter_path": str(adapter_path),
            "train": True,
            "test": False,
            "resume_adapter_file": (
                str(resume_adapter_file) if resume_adapter_file is not None else None
            ),
        }
    )
    if mode == "smoke":
        effective.update(
            {
                "batch_size": min(int(effective["batch_size"]), 2),
                "grad_accumulation_steps": 1,
                "iters": 12,
                "save_every": 12,
                "steps_per_eval": 6,
                "steps_per_report": 1,
                "val_batches": min(int(effective["val_batches"]), 4),
            }
        )
    return effective


def _adapter_hashes(adapter_path: Path) -> dict[str, str]:
    if not adapter_path.is_dir():
        return {}
    return {
        candidate.relative_to(adapter_path).as_posix(): _sha256(candidate)
        for candidate in sorted(adapter_path.rglob("*"))
        if candidate.is_file() and candidate.name != METADATA_NAME
    }


def _validate_target(
    adapter_path: Path, *, mode: str, resume_adapter_file: Path | None
) -> str | None:
    if mode == "resume":
        if resume_adapter_file is None or not resume_adapter_file.is_file():
            raise ValueError("resume mode requires an existing --resume-adapter-file")
        return _sha256(resume_adapter_file)
    if resume_adapter_file is not None:
        raise ValueError("--resume-adapter-file is only valid in resume mode")
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise ValueError(f"adapter target is not empty: {adapter_path}; use explicit resume mode")
    return None


def run(
    *,
    mode: str,
    config_path: Path,
    model: str | Path,
    data_dir: Path,
    adapter_path: Path | None = None,
    resume_adapter_file: Path | None = None,
    expected_manifest_sha256: str | None = None,
    backend_loader: Callable[[], tuple[Any, Any]] = _load_backend,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Validate, seed, train, and persist an auditable run record."""
    if mode not in {"smoke", "full", "resume"}:
        raise ValueError(f"unsupported mode: {mode}")
    config_path = _resolve_local(config_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"training config is not a mapping: {config_path}")

    model_identity = _model_identity(model)
    model_path = Path(model_identity["resolved_path"])
    data_dir = _resolve_local(data_dir)
    configured_adapter = adapter_path or Path(str(raw_config.get("adapter_path", "adapters")))
    resolved_adapter = _resolve_local(configured_adapter)
    if mode == "smoke" and adapter_path is None:
        resolved_adapter = resolved_adapter.with_name(f"{resolved_adapter.name}-smoke")
    resolved_resume = _resolve_local(resume_adapter_file) if resume_adapter_file else None
    parent_adapter_hash = _validate_target(
        resolved_adapter, mode=mode, resume_adapter_file=resolved_resume
    )

    mx, lora = backend_loader()
    effective = _effective_args(
        raw_config,
        lora.CONFIG_DEFAULTS,
        mode=mode,
        config_path=config_path,
        model_path=model_path,
        data_dir=data_dir,
        adapter_path=resolved_adapter,
        resume_adapter_file=resolved_resume,
    )
    max_seq_length = int(effective["max_seq_length"])
    dataset_identity = _validate_dataset(
        data_dir,
        max_seq_length=max_seq_length,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    seed = int(effective["seed"])
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    metadata_path = resolved_adapter / METADATA_NAME
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "status": "running",
        "started_at": now(),
        "completed_at": None,
        "model": model_identity,
        "dataset": dataset_identity,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "exact_args": effective,
        "parent_adapter_sha256": parent_adapter_hash,
        "completed_checkpoint": None,
        "final_adapter_hashes": {},
        "error": None,
    }
    _atomic_json(metadata_path, metadata)

    try:
        lora.run(types.SimpleNamespace(**effective))
        completed_checkpoint = resolved_adapter / "adapters.safetensors"
        if not completed_checkpoint.is_file():
            raise RuntimeError("MLX-LoRA returned without writing adapters.safetensors")
        metadata.update(
            {
                "status": "completed",
                "completed_at": now(),
                "completed_checkpoint": str(completed_checkpoint.resolve()),
                "final_adapter_hashes": _adapter_hashes(resolved_adapter),
            }
        )
        _atomic_json(metadata_path, metadata)
        return metadata
    except BaseException as exc:
        metadata.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "completed_at": now(),
                "final_adapter_hashes": _adapter_hashes(resolved_adapter),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _atomic_json(metadata_path, metadata)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full", "resume"))
    parser.add_argument("--model", required=True, help="Downloaded local model directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--resume-adapter-file", type=Path)
    parser.add_argument(
        "--expected-manifest-sha256",
        help="Optional pinned SHA256 of manifest.json; split hashes are always mandatory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        mode=args.mode,
        config_path=args.config,
        model=args.model,
        data_dir=args.data_dir,
        adapter_path=args.adapter_path,
        resume_adapter_file=args.resume_adapter_file,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
