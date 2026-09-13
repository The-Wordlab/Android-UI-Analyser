#!/usr/bin/env python3
"""Historical pinned vLLM launch plans; current inference policy is local-only.

The current comparison manifest disables cloud inference execution. Use
serve_local.py for the Mac; retained vLLM plans document the earlier pilot.

No local model loading or downloads occur in the default dry run. Execution
records Hub-declared weight identities and the exact chat template before vLLM
downloads/loads weights. Those identities are provenance, not a GPU accuracy
check. Serve one model at a time, stop it, then launch the next model.

On the pod, inside its isolated vLLM 0.29.0 environment:
  python serve_model.py --model gemma4-e4b --execute
For an already downloaded snapshot, add --snapshot-dir PATH --identity-file JSON.
The identity must be a saved Hub API response with blobs=true, or a launcher
identity containing hub_declared_files. Offline startup verifies every required
file without network access. A legacy identity with only weight hashes is not
sufficient to verify the tokenizer. Default dry runs remain read-only.
Use an SSH tunnel to 127.0.0.1:8000; the API never binds a public interface.
The diagnostic default is 16,384 context tokens. Pass --max-model-len 32768
for the common 32K lane and set the controller's --max-tokens 4096 separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import socket
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

VLLM_VERSION = "0.29.0"
DEFAULT_MAX_MODEL_LEN = 16384
# Common supported range for these four pinned snapshots. Liquid 8B-A1B declares
# 128000 positions; the other three checkpoints declare at least 131072.
MIN_MODEL_LEN = 1024
MAX_MODEL_LEN = 128000
VLLM_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"
IMAGES = {
    "cuda13": "vllm/vllm-openai@sha256:"
    "c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1",
    "cuda12_9": "vllm/vllm-openai@sha256:"
    "7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2",
}
# Actual chat_template.jinja bytes, read from each pinned Hub revision. The
# generic vLLM Gemma template is deliberately not substituted for Unified 12B.
TEMPLATE_HASHES = {
    "gemma4-e4b": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    "gemma4-12b": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    "lfm25-2p6b": "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef",
    "lfm25-8b-a1b": "c5b67247c9f736f6c40eaa91ae1ef27dc67f5b041eedaadb97a1bea5030fb0ff",
}
REQUIRED_SNAPSHOT_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}


def declared_files(metadata: dict) -> list[dict]:
    """Retain hashes for loadable artifacts from a saved Hub blobs=true response."""
    return [
        {
            "filename": item["rfilename"],
            "size": item.get("size"),
            "sha256": (item.get("lfs") or {}).get("sha256"),
            "git_blob_sha1": item.get("blobId") if not item.get("lfs") else None,
        }
        for item in metadata["siblings"]
        if item["rfilename"].endswith((".safetensors", ".json", ".jinja", ".model", ".tiktoken"))
        or PurePosixPath(item["rfilename"]).name in {"vocab.txt", "merges.txt"}
    ]


def _verify_file(path: Path, declaration: dict) -> None:
    size = declaration.get("size")
    if type(size) is not int or size < 0 or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Missing or wrong-size snapshot artifact: {declaration['filename']}")
    expected = declaration.get("sha256") or declaration.get("git_blob_sha1")
    is_sha256 = bool(declaration.get("sha256"))
    if not isinstance(expected, str) or not re.fullmatch(
        r"[0-9a-f]{64}" if is_sha256 else r"[0-9a-f]{40}", expected
    ):
        raise ValueError(f"Missing valid file hash: {declaration['filename']}")
    digest = hashlib.sha256() if is_sha256 else hashlib.sha1()
    if not is_sha256:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Snapshot artifact hash mismatch: {declaration['filename']}")


def offline_plan(plan: dict, snapshot_dir: Path, identity_file: Path) -> dict:
    """Validate trusted saved provenance and local bytes; never call the Hub."""
    identity = json.loads(identity_file.read_text())
    expected = plan["model"]
    if "siblings" in identity:
        repo, revision = identity.get("id"), identity.get("sha")
        files = declared_files(identity)
    else:
        repo = identity.get("model", {}).get("repository")
        revision = identity.get("model", {}).get("revision")
        files = identity.get("hub_declared_files")
    if (repo, revision) != (expected["repository"], expected["revision"]):
        raise ValueError("Saved identity does not match the expected repository and revision")
    if not isinstance(files, list) or not files:
        raise ValueError(
            "Offline identity needs complete hub_declared_files, including tokenizer hashes"
        )
    root = snapshot_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot-dir must be a complete model directory")
    by_name = {}
    for item in files:
        filename = item.get("filename") if isinstance(item, dict) else None
        if not isinstance(filename, str) or not filename or "\\" in filename:
            raise ValueError("Invalid snapshot artifact filename")
        parts = PurePosixPath(filename)
        if parts.is_absolute() or ".." in parts.parts or parts.as_posix() != filename:
            raise ValueError("Snapshot artifact filename must be a relative normalized path")
        if filename in by_name:
            raise ValueError("Duplicate snapshot artifact declaration")
        by_name[filename] = item
    missing = REQUIRED_SNAPSHOT_FILES - by_name.keys()
    if missing:
        raise ValueError("Identity lacks required snapshot files: " + ", ".join(sorted(missing)))
    weights = {name for name in by_name if name.endswith(".safetensors")}
    if not weights:
        raise ValueError("Identity contains no safetensors weights")
    actual_weights = {p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")}
    if actual_weights != weights:
        raise ValueError("Snapshot weight files do not match the complete declared shard set")
    index_name = "model.safetensors.index.json"
    if len(weights) > 1 and index_name not in by_name:
        raise ValueError("A sharded snapshot requires a hashed weight index")
    for filename, item in by_name.items():
        # Hub snapshot symlinks may point outside the directory; hash their contents.
        _verify_file(root / filename, item)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != expected["model_type"]:
        raise ValueError("Snapshot architecture differs from the comparison manifest")
    template_hash = hashlib.sha256((root / "chat_template.jinja").read_bytes()).hexdigest()
    if template_hash != plan["chat_template_sha256"]:
        raise ValueError("Snapshot chat template differs from the pinned template")
    if index_name in by_name:
        weight_map = json.loads((root / index_name).read_text()).get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(value, str) for value in weight_map.values())
            or set(weight_map.values()) != weights
        ):
            raise ValueError("Weight index does not reference exactly the verified shards")
    argv = list(plan["argv"])
    for flag in ("--model", "--tokenizer"):
        argv[argv.index(flag) + 1] = str(root)
    argv[argv.index("--chat-template") + 1] = str(root / "chat_template.jinja")
    # The bytes were verified locally; there is no remote revision to resolve.
    for flag in ("--revision", "--tokenizer-revision"):
        index = argv.index(flag)
        del argv[index : index + 2]
    return {
        **plan,
        "argv": argv,
        "shell_command": "HF_HUB_OFFLINE=1 " + shlex.join(argv),
        "environment": {"HF_HUB_OFFLINE": "1"},
        "offline": True,
        "identity_file": str(identity_file.resolve()),
        "snapshot_dir": str(root),
        "hub_declared_files": files,
        "identity_evidence": "all declared snapshot artifacts verified against saved hashes; no Hub requests",
    }


def fetch(url: str) -> bytes:
    headers = {"User-Agent": "aua-controller-provenance/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def launch_plan(
    model: dict, runtime_dir: Path, eager: bool, *, max_model_len: int = DEFAULT_MAX_MODEL_LEN
) -> dict:
    if type(max_model_len) is not int or not MIN_MODEL_LEN <= max_model_len <= MAX_MODEL_LEN:
        raise ValueError(f"max-model-len must be an integer between {MIN_MODEL_LEN} and {MAX_MODEL_LEN}")
    repo, revision = model["repository"], model["revision"]
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo):
        raise ValueError("Expected a Hugging Face owner/repository model ID")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable 40-character model revision is required")
    template_hash = TEMPLATE_HASHES[model["id"]]
    artifact_dir = runtime_dir / "identities" / model["id"] / revision
    template_path = artifact_dir / "chat_template.jinja"
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        repo,
        "--revision",
        revision,
        "--tokenizer",
        repo,
        "--tokenizer-revision",
        revision,
        "--served-model-name",
        repo,
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(max_model_len),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        model["tool_call_parser"],
        "--reasoning-parser",
        model["reasoning_parser"],
        "--chat-template",
        str(template_path),
        "--chat-template-content-format",
        "string",
        "--download-dir",
        str(runtime_dir / "hf-cache"),
    ]
    if model["model_type"] in {"gemma4", "gemma4_unified"}:
        argv += ["--limit-mm-per-prompt", '{"image":0,"audio":0,"video":0}']
    if model.get("chat_template_kwargs"):
        argv += ["--default-chat-template-kwargs", json.dumps(model["chat_template_kwargs"])]
    if eager:
        argv += ["--enforce-eager"]
    return {
        "model": model,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "vllm_version": VLLM_VERSION,
        "vllm_source_revision": VLLM_COMMIT,
        "official_image_manifest_digests": IMAGES,
        "image_registry_checked_utc": "2026-09-11",
        "chat_template_sha256": template_hash,
        "chat_template_source": f"https://huggingface.co/{repo}/resolve/{revision}/chat_template.jinja",
        "artifact_dir": str(artifact_dir),
        "api_base_url": "http://127.0.0.1:8000/v1",
        "identity_evidence": "pinned revision and Hub-declared SHA256s; loaded weights not yet checked",
        "argv": argv,
        "shell_command": shlex.join(argv),
    }


def prepare_identity(plan: dict) -> None:
    model = plan["model"]
    repo = urllib.parse.quote(model["repository"], safe="/")
    revision = model["revision"]
    metadata = json.loads(
        fetch(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    )
    if metadata.get("sha") != revision:
        raise ValueError("Hub resolved a different model revision")
    config_bytes = fetch(f"https://huggingface.co/{repo}/resolve/{revision}/config.json")
    config = json.loads(config_bytes)
    if config.get("model_type") != model["model_type"]:
        raise ValueError("The checkpoint architecture differs from the comparison manifest")
    template = fetch(plan["chat_template_source"])
    if hashlib.sha256(template).hexdigest() != plan["chat_template_sha256"]:
        raise ValueError("The pinned chat template has an unexpected SHA256")
    weights = [
        {
            "filename": item["rfilename"],
            "size": item.get("size"),
            "sha256": (item.get("lfs") or {}).get("sha256"),
        }
        for item in metadata["siblings"]
        if item["rfilename"].endswith(".safetensors")
    ]
    if not weights or any(not item["sha256"] for item in weights):
        raise ValueError("Expected SHA256 metadata for all safetensors weight files")
    plan["hub_declared_weight_files"] = weights
    plan["hub_declared_files"] = declared_files(metadata)
    plan["checkpoint_architectures"] = config.get("architectures")
    plan["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    plan["runtime_package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "torch", "transformers", "huggingface-hub")
    }
    artifact_dir = Path(plan["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "chat_template.jinja").write_bytes(template)
    (artifact_dir / "config.json").write_bytes(config_bytes)
    (artifact_dir / "launch_identity.json").write_text(json.dumps(plan, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(TEMPLATE_HASHES))
    parser.add_argument(
        "--comparison", type=Path, default=Path(__file__).with_name("comparison.json")
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=Path("/workspace/aua-controller-runtime")
    )
    parser.add_argument(
        "--execute", action="store_true", help="Load and serve on the current Runpod pod"
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, help="Complete local snapshot for offline serving"
    )
    parser.add_argument(
        "--identity-file", type=Path, help="Saved Hub blobs=true or complete launcher identity JSON"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Separate diagnostic lane; disables graph capture",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=(f"Shared context and batched-token budget, {MIN_MODEL_LEN}..{MAX_MODEL_LEN}; "
              f"default {DEFAULT_MAX_MODEL_LEN}. Use 32768 for the explicit 32K lane."),
    )
    args = parser.parse_args()
    if bool(args.snapshot_dir) != bool(args.identity_file):
        parser.error("--snapshot-dir and --identity-file must be supplied together")
    if not MIN_MODEL_LEN <= args.max_model_len <= MAX_MODEL_LEN:
        parser.error(f"--max-model-len must be between {MIN_MODEL_LEN} and {MAX_MODEL_LEN}")
    comparison = json.loads(args.comparison.read_text())
    if args.execute and comparison.get("inference_location") == "local_mac_only":
        parser.error("Inference is local-only; use serve_local.py. Runpod is reserved for training.")
    model = next(item for item in comparison["models"] if item["id"] == args.model)
    plan = launch_plan(
        model, args.runtime_dir.resolve(), args.enforce_eager, max_model_len=args.max_model_len
    )
    if args.snapshot_dir:
        plan = offline_plan(plan, args.snapshot_dir, args.identity_file)
    if not args.execute:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return 0
    if not os.environ.get("RUNPOD_POD_ID"):
        parser.error("--execute must run on an existing Runpod pod (RUNPOD_POD_ID is absent)")
    version = importlib.metadata.version("vllm")
    if version.split("+")[0] != VLLM_VERSION:
        parser.error(
            f"Expected vLLM {VLLM_VERSION}, found {version}; use an isolated inference environment"
        )
    # Refuse a second server rather than spending time loading a second model.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Port 8000 is occupied; stop the previous model process first")
    if plan.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        artifact_dir = Path(plan["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "offline_launch_identity.json").write_text(
            json.dumps(plan, indent=2) + "\n"
        )
    else:
        prepare_identity(plan)
    print(json.dumps(plan, indent=2), flush=True)
    os.execv(sys.executable, plan["argv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
