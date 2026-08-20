"""Package a trained checkpoint into a portable, hash-pinned adapter directory.

A checkpoint trained on a rented GPU records that machine's absolute base-model path in
``adapter_config.json``. Once the Pod is deleted that path resolves nowhere, and the provider
correctly refuses to load it — the exact failure the V8 evaluation hit, which silently reported
0% for both provider smokes because the model had never been loaded at all.

Portability is therefore not "copy the weights". It is: rewrite the declared base-model path to a
local snapshot, then pin the base model, adapter weights, and rollout manifest by SHA-256 so the
provider can verify identity instead of trusting a path. This tool does that in one place.

``--max-mode advisory`` is an explicit local operator decision and is what the autopilot execution
lane requires; the default stays ``shadow``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROMPT_SCHEMA = {
    "candidate_counts": [2, 3, 4],
    "candidate_ids": "dense opaque integers 0 through candidate_count minus 1",
    "handoff_candidate_id": -1,
    "name": "functiongemma-aua-candidate-policy-v3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(directory: Path) -> str:
    """Hash a model directory the way the provider's directory-tree check does."""

    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def package(
    *,
    weights: Path,
    adapter_config: Path,
    model_dir: Path,
    destination: Path,
    max_mode: str,
    model_sha256: str | None = None,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, destination / "adapters.safetensors")

    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    # The trained-on path is gone; bind the adapter to the local snapshot instead.
    config["model"] = str(model_dir)
    config["adapter_path"] = str(destination)
    (destination / "adapter_config.json").write_text(
        json.dumps(config, indent=4) + "\n", encoding="utf-8"
    )

    adapter_hash = _sha256(destination / "adapters.safetensors")
    config_hash = _sha256(destination / "adapter_config.json")
    base_hash = model_sha256 or _directory_sha256(model_dir)
    manifest = {
        "schema_version": 1,
        "adapter": {
            "bytes": (destination / "adapters.safetensors").stat().st_size,
            "config": "adapter_config.json",
            "config_sha256": config_hash,
            "dropout": config.get("lora_parameters", {}).get("dropout"),
            "fine_tune_type": config.get("fine_tune_type", "lora"),
            "rank": config.get("lora_parameters", {}).get("rank"),
            "scale": config.get("lora_parameters", {}).get("scale"),
            "sha256": adapter_hash,
            "weights": "adapters.safetensors",
        },
        "base_model": {"sha256": base_hash},
        "prompt_schema": PROMPT_SCHEMA,
        "rollout": {"max_mode": max_mode},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "destination": str(destination),
        "model_path": str(model_dir),
        "model_sha256": base_hash,
        "adapter_sha256": adapter_hash,
        "manifest_sha256": _sha256(manifest_path),
        "max_mode": max_mode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make a trained adapter portable and pinned.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--max-mode", choices=("shadow", "advisory"), default="shadow")
    parser.add_argument(
        "--model-sha256",
        help="Reuse a known base-model digest instead of rehashing the snapshot.",
    )
    args = parser.parse_args(argv)

    result = package(
        weights=args.weights.resolve(),
        adapter_config=args.adapter_config.resolve(),
        model_dir=args.model_dir.resolve(),
        destination=args.destination.resolve(),
        max_mode=args.max_mode,
        model_sha256=args.model_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
