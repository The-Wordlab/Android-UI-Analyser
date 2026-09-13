"""Offline serving validates cached artifacts without contacting the Hub."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller import serve_model as serving


@pytest.fixture
def cached_model(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("offline serving attempted network access")

    monkeypatch.setattr(serving, "fetch", no_network)
    monkeypatch.setattr(serving.urllib.request, "urlopen", no_network)
    model = next(
        item
        for item in json.loads(Path(serving.__file__).with_name("comparison.json").read_text())[
            "models"
        ]
        if item["id"] == "lfm25-2p6b"
    )
    root = tmp_path / "snapshot"
    root.mkdir()
    contents = {
        "config.json": json.dumps({"model_type": "lfm2"}).encode(),
        "generation_config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "chat_template.jinja": b"test-pinned-template",
        "model-00001-of-00002.safetensors": b"first-test-weight",
        "model-00002-of-00002.safetensors": b"second-test-weight",
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        ).encode(),
    }
    monkeypatch.setitem(
        serving.TEMPLATE_HASHES,
        model["id"],
        hashlib.sha256(contents["chat_template.jinja"]).hexdigest(),
    )
    siblings = []
    for name, content in contents.items():
        (root / name).write_bytes(content)
        item = {"rfilename": name, "size": len(content)}
        if name.endswith(".safetensors"):
            item["lfs"] = {"sha256": hashlib.sha256(content).hexdigest()}
        else:
            item["blobId"] = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        siblings.append(item)
    metadata = {"id": model["repository"], "sha": model["revision"], "siblings": siblings}
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(metadata))
    plan = serving.launch_plan(model, tmp_path / "runtime", False)
    return root, identity, metadata, plan


def test_raw_hub_identity_is_verified_without_network(cached_model):
    root, identity, _, plan = cached_model
    result = serving.offline_plan(plan, root, identity)
    argv = result["argv"]
    assert argv[argv.index("--model") + 1] == str(root)
    assert argv[argv.index("--tokenizer") + 1] == str(root)
    assert argv[argv.index("--chat-template") + 1] == str(root / "chat_template.jinja")
    assert "--revision" not in argv and "--tokenizer-revision" not in argv
    assert result["environment"] == {"HF_HUB_OFFLINE": "1"}
    assert argv[argv.index("--served-model-name") + 1] == plan["model"]["repository"]


def test_saved_launcher_identity_and_cache_blob_symlink(cached_model, tmp_path):
    root, identity, metadata, plan = cached_model
    identity.write_text(
        json.dumps(
            {
                "model": plan["model"],
                "hub_declared_files": serving.declared_files(metadata),
            }
        )
    )
    original = root / "model-00001-of-00002.safetensors"
    blob = tmp_path / "verified-blob"
    original.rename(blob)
    original.symlink_to(blob)
    assert serving.offline_plan(plan, root, identity)["offline"] is True


@pytest.mark.parametrize("field", ["id", "sha"])
def test_wrong_repository_or_revision_is_rejected(cached_model, field):
    root, identity, metadata, plan = cached_model
    metadata[field] = "wrong"
    identity.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="repository and revision"):
        serving.offline_plan(plan, root, identity)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00002-of-00002.safetensors",
    ],
)
def test_missing_required_artifact_is_rejected(cached_model, filename):
    root, identity, _, plan = cached_model
    (root / filename).unlink()
    with pytest.raises(ValueError, match="artifact|shard set"):
        serving.offline_plan(plan, root, identity)


@pytest.mark.parametrize("filename", ["tokenizer.json", "model-00001-of-00002.safetensors"])
def test_same_size_corruption_is_rejected(cached_model, filename):
    root, identity, _, plan = cached_model
    path = root / filename
    path.write_bytes(b"X" * path.stat().st_size)
    with pytest.raises(ValueError, match="hash mismatch"):
        serving.offline_plan(plan, root, identity)


def test_extra_weight_file_is_rejected(cached_model):
    root, identity, _, plan = cached_model
    (root / "unverified.safetensors").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="complete declared shard set"):
        serving.offline_plan(plan, root, identity)


@pytest.mark.parametrize(
    "damage", ["missing_tokenizer", "missing_hash", "traversal", "duplicate", "missing_index"]
)
def test_incomplete_or_unsafe_identity_is_rejected(cached_model, damage):
    root, identity, metadata, plan = cached_model
    files = serving.declared_files(metadata)
    if damage == "missing_tokenizer":
        files = [item for item in files if item["filename"] != "tokenizer.json"]
    elif damage == "missing_hash":
        next(item for item in files if item["filename"] == "tokenizer.json")["git_blob_sha1"] = None
    elif damage == "traversal":
        files[0]["filename"] = "../config.json"
    elif damage == "duplicate":
        files.append(copy.deepcopy(files[0]))
    elif damage == "missing_index":
        files = [item for item in files if item["filename"] != "model.safetensors.index.json"]
    identity.write_text(json.dumps({"model": plan["model"], "hub_declared_files": files}))
    with pytest.raises(ValueError):
        serving.offline_plan(plan, root, identity)


def test_offline_main_exec_sets_environment_and_never_prepares_online(cached_model, monkeypatch):
    root, identity, _, plan = cached_model
    captured = {}
    historical_manifest = root.parent / "historical-comparison.json"
    historical_manifest.write_text(json.dumps({"models": [plan["model"]]}))

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def connect_ex(self, address):
            return 1

    def execute(executable, argv):
        captured.update(argv=argv, offline=serving.os.environ.get("HF_HUB_OFFLINE"))

    monkeypatch.setenv("RUNPOD_POD_ID", "test-only")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(serving.socket, "socket", Probe)
    monkeypatch.setattr(serving.importlib.metadata, "version", lambda name: "0.29.0")
    monkeypatch.setattr(serving.os, "execv", execute)
    monkeypatch.setattr(
        serving, "prepare_identity", lambda *args: pytest.fail("online startup was called")
    )
    monkeypatch.setattr(
        serving.sys,
        "argv",
        [
            "serve_model.py",
            "--comparison",
            str(historical_manifest),
            "--model",
            plan["model"]["id"],
            "--snapshot-dir",
            str(root),
            "--identity-file",
            str(identity),
            "--runtime-dir",
            str(root.parent / "runtime"),
            "--execute",
        ],
    )
    assert serving.main() == 0
    assert captured["offline"] == "1"
    assert captured["argv"][captured["argv"].index("--model") + 1] == str(root)
    assert list((root.parent / "runtime").rglob("offline_launch_identity.json"))


def test_current_policy_blocks_cloud_execution_before_setup(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_POD_ID", "test-only")
    monkeypatch.setattr(serving.sys, "argv", ["serve_model.py", "--model", "gemma4-e4b", "--execute"])
    monkeypatch.setattr(serving, "launch_plan", lambda *a, **kw: pytest.fail("cloud setup was reached"))
    with pytest.raises(SystemExit) as error:
        serving.main()
    assert error.value.code == 2
    assert "Inference is local-only" in capsys.readouterr().err


def test_default_dry_run_needs_no_snapshot_or_network(cached_model, monkeypatch, capsys):
    _, _, _, plan = cached_model
    monkeypatch.setattr(serving.sys, "argv", ["serve_model.py", "--model", plan["model"]["id"]])
    assert serving.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["argv"][result["argv"].index("--model") + 1] == plan["model"]["repository"]
    assert result["max_model_len"] == result["max_num_batched_tokens"] == 16384
    for flag in ("--max-model-len", "--max-num-batched-tokens"):
        assert result["argv"][result["argv"].index(flag) + 1] == "16384"


@pytest.mark.parametrize("model_id", sorted(serving.TEMPLATE_HASHES))
def test_32k_lane_uses_the_same_context_and_batch_budget_for_each_model(tmp_path, model_id):
    comparison = json.loads(Path(serving.__file__).with_name("comparison.json").read_text())
    model = next(item for item in comparison["models"] if item["id"] == model_id)
    plan = serving.launch_plan(model, tmp_path, False, max_model_len=32768)

    assert plan["max_model_len"] == plan["max_num_batched_tokens"] == 32768
    for flag in ("--max-model-len", "--max-num-batched-tokens"):
        assert plan["argv"][plan["argv"].index(flag) + 1] == "32768"
    assert plan["argv"][plan["argv"].index("--max-num-seqs") + 1] == "1"
    assert plan["argv"][plan["argv"].index("--dtype") + 1] == "bfloat16"


@pytest.mark.parametrize("context", [0, -1, 1023, 128001, True, 32768.0])
def test_launch_plan_rejects_invalid_or_unsupported_context(cached_model, context):
    _, _, _, plan = cached_model
    with pytest.raises(ValueError, match="max-model-len"):
        serving.launch_plan(plan["model"], Path("/unused"), False, max_model_len=context)


def test_offline_cli_preserves_explicit_32k_budget(cached_model, monkeypatch, capsys):
    root, identity, _, plan = cached_model
    monkeypatch.setattr(serving.sys, "argv", [
        "serve_model.py", "--model", plan["model"]["id"],
        "--snapshot-dir", str(root), "--identity-file", str(identity),
        "--max-model-len", "32768",
    ])

    assert serving.main() == 0
    result = json.loads(capsys.readouterr().out)

    assert result["offline"] is True and result["dry_run"] is True
    assert result["max_model_len"] == result["max_num_batched_tokens"] == 32768
    for flag in ("--max-model-len", "--max-num-batched-tokens"):
        assert result["argv"][result["argv"].index(flag) + 1] == "32768"


@pytest.mark.parametrize("context", ["0", "128001", "not-a-number"])
def test_cli_rejects_invalid_context_before_loading_or_accessing_the_hub(
    cached_model, monkeypatch, context,
):
    _, _, _, plan = cached_model
    monkeypatch.setattr(serving.sys, "argv", [
        "serve_model.py", "--model", plan["model"]["id"], "--max-model-len", context,
    ])
    with pytest.raises(SystemExit) as exc:
        serving.main()
    assert exc.value.code == 2
