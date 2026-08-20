from __future__ import annotations

import io
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma import runpod_benchmark as runpod
from experiments.functiongemma import runpod_worker

TEST_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKbGfAvWt/j3ZycGWfSPaQjaiav+b7nr0YKZEkCqoDpl "
    "aua-runpod-test"
)
TEST_FINGERPRINT = "SHA256:Y8pXZuLOG/CpcxYU125gEAXqj+vSUW3YNI3SW3Z9KEQ"
BASELINE_FINGERPRINT = "SHA256:" + "A" * 43


def _tar_gz(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _default_config(tmp_path: Path) -> runpod.Config:
    args = runpod._parser().parse_args(
        ["--revision", "a" * 40, "--output-dir", str(tmp_path / "result")]
    )
    return runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))


def test_real_defaults_are_a_pinned_fresh_base_v3_benchmark(tmp_path: Path) -> None:
    config = _default_config(tmp_path)

    assert config.mode == "benchmark"
    assert config.curriculum_version == "v3"
    assert config.config_path == "experiments/functiongemma/train-lora.yaml"
    assert config.expected_manifest_sha256 == runpod.V3_MANIFEST_SHA256
    assert config.mlx_package == "mlx[cuda12]==0.32.0"
    assert config.terminate_after == "2026-08-17T13:30:00Z"


def test_v4_throughput_selection_uses_its_distinct_config_and_manifest(tmp_path: Path) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
            "--curriculum-version",
            "v4",
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))

    assert config.config_path == "experiments/functiongemma/train-lora-v4.yaml"
    assert config.expected_manifest_sha256 == runpod.V4_MANIFEST_SHA256


def test_v5_selection_uses_recovery_config_and_manifest_pin(tmp_path: Path) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
            "--curriculum-version",
            "v5",
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))

    assert config.config_path == "experiments/functiongemma/train-lora-v5.yaml"
    assert config.expected_manifest_sha256 == runpod.V5_MANIFEST_SHA256


def test_v6_selection_uses_live_context_config_and_manifest_pin(tmp_path: Path) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
            "--curriculum-version",
            "v6",
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))

    assert config.config_path == "experiments/functiongemma/train-lora-v6.yaml"
    assert config.expected_manifest_sha256 == runpod.V6_MANIFEST_SHA256


def test_v7_selection_uses_semantic_context_config_and_manifest_pin(tmp_path: Path) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
            "--curriculum-version",
            "v7",
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))

    assert config.config_path == "experiments/functiongemma/train-lora-v7-seed61.yaml"
    assert config.expected_manifest_sha256 == runpod.V7_MANIFEST_SHA256


def test_v8_selection_uses_handoff_config_and_manifest_pin(tmp_path: Path) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
            "--curriculum-version",
            "v8",
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))

    assert config.config_path == "experiments/functiongemma/train-lora-v8-seed83.yaml"
    assert config.expected_manifest_sha256 == runpod.V8_MANIFEST_SHA256


def test_create_has_server_ttl_but_no_secret_and_worker_uses_module_form(tmp_path: Path) -> None:
    config = _default_config(tmp_path)
    command = runpod._create_command(config, "ssh-ed25519 PUBLIC")
    rendered = " ".join(command)

    assert "--terminate-after 2026-08-17T13:30:00Z" in rendered
    assert "--wait" in command
    assert command[command.index("--volume-in-gb") + 1] == "0"
    assert "--network-volume-id" not in command
    assert "RUNPOD_API_KEY" not in rendered
    assert "HF_TOKEN" not in rendered
    remote = runpod._remote_command(config, "b" * 64)
    assert "python3 -m experiments.functiongemma.runpod_worker" in remote
    assert "git clone" not in remote


def test_preflight_requires_current_account_ssh_key_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _default_config(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(runpod.shutil, "which", lambda _name: "/fictional/runpodctl")

    def runner(command, *, env=None, timeout=None):
        del env, timeout
        commands.append(command)
        if command == ["runpodctl", "pod", "create", "--help"]:
            return runpod.CommandResult(0, b"--terminate-after --wait", b"")
        if command == ["runpodctl", "ssh", "add-key", "--help"]:
            return runpod.CommandResult(0, b"--key-file", b"")
        if command == ["runpodctl", "ssh", "remove-key", "--help"]:
            return runpod.CommandResult(0, b"--fingerprint", b"")
        return runpod.CommandResult(0, b"", b"")

    runpod._local_preflight(config, runner, {})

    assert ["runpodctl", "ssh", "list-keys", "--help"] in commands
    assert ["runpodctl", "ssh", "add-key", "--help"] in commands
    assert ["runpodctl", "ssh", "remove-key", "--help"] in commands


def test_preflight_accepts_reviewed_v5_config_before_its_first_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
            "--curriculum-version",
            "v5",
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))
    commands: list[list[str]] = []
    monkeypatch.setattr(runpod.shutil, "which", lambda _name: "/fictional/runpodctl")

    def runner(command, *, env=None, timeout=None):
        del env, timeout
        commands.append(command)
        if command == ["runpodctl", "pod", "create", "--help"]:
            return runpod.CommandResult(0, b"--terminate-after --wait", b"")
        if command == ["runpodctl", "ssh", "add-key", "--help"]:
            return runpod.CommandResult(0, b"--key-file", b"")
        if command == ["runpodctl", "ssh", "remove-key", "--help"]:
            return runpod.CommandResult(0, b"--fingerprint", b"")
        if command[:3] == ["git", "cat-file", "-e"] and ":" in command[-1]:
            return runpod.CommandResult(1, b"", b"untracked")
        return runpod.CommandResult(0, b"", b"")

    runpod._local_preflight(config, runner, {})

    assert ["git", "cat-file", "-e", f"{'a' * 40}^{{commit}}"] in commands
    assert not any(
        command[:3] == ["git", "cat-file", "-e"] and ":" in command[-1] for command in commands
    )


def test_worker_uses_module_invocation_for_fresh_archive_commands() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "experiments" / "functiongemma" / "runpod_worker.py"
    ).read_text(encoding="utf-8")

    assert '"experiments.functiongemma.generate_dataset"' in source
    assert '"experiments.functiongemma.validate_dataset"' in source
    assert '"experiments/functiongemma/generate_dataset.py"' not in source
    assert '[*common, "pytest", "-r", str(REQUIREMENTS)]' in source


def test_worker_sizes_mlx_cuda_graph_cache_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLX_CUDA_GRAPH_CACHE_SIZE", raising=False)

    configured = runpod_worker._configure_mlx_cuda_runtime()

    assert configured == {"graph_cache_size": "2048"}
    assert runpod_worker.os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] == "2048"


def test_worker_authenticates_v7_variable_cardinality_adapter(tmp_path: Path) -> None:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    config = {
        "fine_tune_type": "lora",
        "model": str(model),
        "lora_parameters": {"rank": 32, "scale": 64.0, "dropout": 0.05},
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fictional-v7-adapter")

    manifest = runpod_worker._write_v7_rollout_manifest(model, adapter)

    assert manifest["rollout"] == {"max_mode": "shadow"}
    assert manifest["prompt_schema"]["candidate_counts"] == [2, 3, 4]
    assert manifest["adapter"]["rank"] == 32
    assert (adapter / "manifest.json").is_file()


def test_worker_authenticates_v8_handoff_protocol(tmp_path: Path) -> None:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "model": str(model),
                "lora_parameters": {"rank": 32, "scale": 64.0, "dropout": 0.05},
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapters.safetensors").write_bytes(b"fictional-v8-adapter")

    manifest = runpod_worker._write_v8_rollout_manifest(model, adapter)

    assert manifest["prompt_schema"]["candidate_counts"] == [2, 3, 4]
    assert manifest["prompt_schema"]["handoff_candidate_id"] == -1


def test_functiongemma_worker_accepts_hash_verified_evaluation_only_inputs(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    files = {
        "adapter_config.json": b"{}",
        "0000512_adapters.safetensors": b"checkpoint-512",
        "adapters.safetensors": b"checkpoint-final",
    }
    hashes = {}
    for name, payload in files.items():
        path = adapter / name
        path.write_bytes(payload)
        hashes[name] = runpod_worker._sha256(path)
    (adapter / "run-metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "mode": "full",
                "model": {"revision": "model-revision"},
                "dataset": {"manifest_sha256": "manifest-hash"},
                "final_adapter_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )

    metadata = runpod_worker._validate_input_adapter(
        adapter,
        model_revision="model-revision",
        manifest_sha256="manifest-hash",
    )

    assert metadata["final_adapter_hashes"] == hashes
    (adapter / "0000512_adapters.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        runpod_worker._validate_input_adapter(
            adapter,
            model_revision="model-revision",
            manifest_sha256="manifest-hash",
        )


def test_functiongemma_worker_parser_supports_evaluation_only_archives() -> None:
    args = runpod_worker._parser().parse_args(
        [
            "--run-id",
            "eval",
            "--source-revision",
            "a" * 40,
            "--source-archive-sha256",
            "b" * 64,
            "--output-root",
            "/tmp/output",
            "--max-runtime-seconds",
            "3600",
            "--input-adapter-dir",
            "/tmp/input/adapter",
            "--input-adapter-archive-sha256",
            "c" * 64,
            "--input-data-dir",
            "/tmp/input/data",
            "--input-data-archive-sha256",
            "d" * 64,
        ]
    )

    assert args.input_adapter_dir == Path("/tmp/input/adapter")
    assert args.input_data_dir == Path("/tmp/input/data")


def test_source_archive_is_git_only_plus_reviewed_functiongemma_overrides(tmp_path: Path) -> None:
    args = runpod._parser().parse_args(
        [
            "--revision",
            runpod.subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=runpod.REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    config = runpod._config(args, datetime(2026, 8, 17, 12, 0, tzinfo=UTC))
    payload, overrides = runpod._source_archive(config, runpod._run_command)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
        worker = archive.extractfile("experiments/functiongemma/runpod_worker.py")
        assert worker is not None
        assert b"token_input.readline" in worker.read()
        validator = archive.extractfile("experiments/functiongemma/validate_dataset.py")
        assert validator is not None
        assert b"POLICY_HANDOFF_ID" in validator.read()
    assert overrides == [
        "experiments/functiongemma/curriculum.py",
        "experiments/functiongemma/evaluate.py",
        "experiments/functiongemma/evaluate_qwen.py",
        "experiments/functiongemma/generate_dataset.py",
        "experiments/functiongemma/live_context_curriculum.py",
        "experiments/functiongemma/recovery_curriculum.py",
        "experiments/functiongemma/run_live_context_smoke.py",
        "experiments/functiongemma/run_production_smoke.py",
        "experiments/functiongemma/run_semantic_context_smoke.py",
        "experiments/functiongemma/run_qwen_semantic_context_smoke.py",
        "experiments/functiongemma/semantic_context_curriculum.py",
        "experiments/functiongemma/select_checkpoint.py",
        "experiments/functiongemma/train.py",
        "experiments/functiongemma/train-lora-v5.yaml",
        "experiments/functiongemma/train-lora-v6.yaml",
        "experiments/functiongemma/train-lora-v7-seed61.yaml",
        "experiments/functiongemma/train-lora-v7-seed67.yaml",
        "experiments/functiongemma/train-lora-v7-seed71.yaml",
        "experiments/functiongemma/train-lora-qwen3-0.6b-v7.yaml",
        "experiments/functiongemma/train-lora-qwen3.5-0.8b-v10.yaml",
        "experiments/functiongemma/train-lora-qwen3-1.7b-v10.yaml",
        "experiments/functiongemma/train-lora-v8-seed83.yaml",
        "experiments/functiongemma/train-lora-v9-seed91.yaml",
        "experiments/functiongemma/train-lora-v9-seed97.yaml",
        "experiments/functiongemma/train-lora-v9-seed101.yaml",
        "experiments/functiongemma/train-lora-v10-seed103.yaml",
        "experiments/functiongemma/train-lora-v10-seed103-long.yaml",
        "experiments/functiongemma/validate_dataset.py",
        "experiments/functiongemma/v8_curriculum.py",
        "experiments/functiongemma/v8_learning_material.py",
        "experiments/functiongemma/v9_curriculum.py",
        "experiments/functiongemma/v9_learning_material.py",
        "experiments/functiongemma/v10_command_catalog.py",
        "experiments/functiongemma/v10_curriculum.py",
        "experiments/functiongemma/v10_learning_material.py",
        "experiments/functiongemma/runpod_worker.py",
        "experiments/functiongemma/runpod_qwen_worker.py",
        "src/android_ui_analyser/policy.py",
        "src/android_ui_analyser/providers/policy/functiongemma.py",
    ]
    assert "experiments/functiongemma/recovery_curriculum.py" in names
    assert "experiments/functiongemma/runpod_worker.py" in names
    assert ".env" not in names
    assert not any(name.startswith("runs/") for name in names)


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    payload = _tar_gz({"../escaped": b"bad"})
    with pytest.raises(runpod.LaunchError, match="path traversal"):
        runpod._safe_extract(payload, tmp_path / "artifacts")


def test_input_dataset_archive_requires_exact_frozen_evaluation_files() -> None:
    required = {
        "manifest.json": b"{}",
        "validation.json": b"{}",
        "train.jsonl": b"{}\n",
        "valid.jsonl": b"{}\n",
        "test.jsonl": b"{}\n",
    }

    runpod._validate_input_dataset_archive(_tar_gz(required))

    with pytest.raises(runpod.LaunchError, match="unexpected path"):
        runpod._validate_input_dataset_archive(_tar_gz(required | {"secret.txt": b"no"}))
    with pytest.raises(runpod.LaunchError, match="incomplete"):
        runpod._validate_input_dataset_archive(
            _tar_gz({k: v for k, v in required.items() if k != "test.jsonl"})
        )


def test_remote_qwen_evaluation_reuses_pinned_adapter_and_dataset_archives(
    tmp_path: Path,
) -> None:
    config = _default_config(tmp_path)
    adapter_archive = tmp_path / "adapter.tar.gz"
    dataset_archive = tmp_path / "dataset.tar.gz"
    adapter_archive.write_bytes(b"adapter archive")
    dataset_archive.write_bytes(b"dataset archive")
    config = runpod.Config(
        **{
            **config.__dict__,
            "worker_module": "experiments.functiongemma.runpod_qwen_worker",
            "input_adapter_archive": adapter_archive,
            "input_dataset_archive": dataset_archive,
        }
    )

    remote = runpod._remote_command(config, "b" * 64)

    assert "--input-adapter-dir" in remote
    assert "--input-adapter-archive-sha256" in remote
    assert "--input-data-dir" in remote
    assert "--input-data-archive-sha256" in remote
    assert runpod._sha256_bytes(adapter_archive.read_bytes()) in remote
    assert runpod._sha256_bytes(dataset_archive.read_bytes()) in remote


def test_ephemeral_public_key_fingerprint_matches_openssh_sha256() -> None:
    assert runpod._ssh_public_key_fingerprint(TEST_PUBLIC_KEY) == TEST_FINGERPRINT


def test_null_key_list_from_current_runpodctl_means_empty_inventory() -> None:
    assert runpod._parse_ssh_key_inventory(b'{"keys":null}\n') == ()


def test_network_volume_inventory_uses_read_only_rest_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = runpod.RunpodAPI("secret")
    calls: list[tuple[str, str, set[int]]] = []

    def request(method: str, path: str, *, allowed: set[int]):
        calls.append((method, path, allowed))
        return [{"id": "volume-existing"}]

    monkeypatch.setattr(client, "_request", request)

    assert client.list_network_volumes() == [{"id": "volume-existing"}]
    assert calls == [("GET", "/networkvolumes", {200})]


def test_primary_and_deferred_cleanup_cover_every_available_termination_signal() -> None:
    expected_names = {
        name for name in ("SIGHUP", "SIGINT", "SIGQUIT", "SIGTERM") if hasattr(runpod.signal, name)
    }
    assert {runpod.signal.Signals(signum).name for signum in runpod.TERMINATION_SIGNALS} == (
        expected_names
    )

    with runpod._signal_cleanup():
        for signum in runpod.TERMINATION_SIGNALS:
            handler = runpod.signal.getsignal(signum)
            assert callable(handler)
            with pytest.raises(KeyboardInterrupt, match=f"received signal {signum}"):
                handler(signum, None)

    with runpod._defer_cleanup_signals() as pending:
        for signum in runpod.TERMINATION_SIGNALS:
            handler = runpod.signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)
    assert pending == list(runpod.TERMINATION_SIGNALS)


class FakeClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.created = False
        self.deleted = False
        self.volume_ids = {"volume-existing"}

    def list_pods(self) -> list[dict[str, object]]:
        self.events.append("list")
        if self.created and not self.deleted:
            return [{"id": "pod-exact", "name": "aua-fg-test", "desiredStatus": "RUNNING"}]
        return []

    def get_pod(self, pod_id: str) -> dict[str, object]:
        assert pod_id == "pod-exact"
        return {"id": pod_id, "costPerHr": "0.99"}

    def list_network_volumes(self) -> list[dict[str, object]]:
        self.events.append("list-volumes")
        return [{"id": volume_id} for volume_id in sorted(self.volume_ids)]

    def delete_pod(self, pod_id: str) -> None:
        self.events.append(f"delete:{pod_id}")
        assert pod_id == "pod-exact"
        self.deleted = True


class FakeSSHAccount:
    def __init__(
        self,
        events: list[str],
        *,
        add_mode: str = "success",
        remove_mode: str = "success",
        remove_signals: tuple[int, ...] = (),
    ) -> None:
        self.events = events
        self.add_mode = add_mode
        self.remove_mode = remove_mode
        self.remove_signals = remove_signals
        self.fingerprints = {BASELINE_FINGERPRINT}

    def run(self, command: list[str]) -> runpod.CommandResult:
        assert command[:4] == ["runpodctl", "--output", "json", "ssh"]
        action = command[4]
        if action == "list-keys":
            self.events.append("ssh-list")
            payload = {
                "keys": [{"fingerprint": fingerprint} for fingerprint in sorted(self.fingerprints)]
            }
            return runpod.CommandResult(0, json.dumps(payload).encode(), b"")
        if action == "add-key":
            self.events.append("ssh-add")
            key_path = Path(command[command.index("--key-file") + 1])
            assert key_path.read_text(encoding="utf-8") == TEST_PUBLIC_KEY
            if self.add_mode == "fail":
                return runpod.CommandResult(1, b"", b"fictional add failure")
            self.fingerprints.add(TEST_FINGERPRINT)
            if self.add_mode == "lost":
                raise runpod.subprocess.TimeoutExpired(command, 30)
            return runpod.CommandResult(0, b'{"added":true}\n', b"")
        if action == "remove-key":
            self.events.append("ssh-remove")
            assert command[command.index("--fingerprint") + 1] == TEST_FINGERPRINT
            for signum in self.remove_signals:
                handler = runpod.signal.getsignal(signum)
                assert callable(handler)
                handler(signum, None)
            if self.remove_mode == "stuck":
                return runpod.CommandResult(1, b"", b"fictional remove failure")
            self.fingerprints.discard(TEST_FINGERPRINT)
            if self.remove_mode == "lost":
                raise runpod.subprocess.TimeoutExpired(command, 30)
            return runpod.CommandResult(0, b'{"removed":true}\n', b"")
        raise AssertionError(f"unexpected SSH action: {action}")


def _write_fake_ephemeral_key(command: list[str]) -> None:
    key = Path(command[command.index("-f") + 1])
    key.write_text("fictional-private-key", encoding="utf-8")
    key.with_suffix(".pub").write_text(TEST_PUBLIC_KEY, encoding="utf-8")


def test_cleanup_recovers_a_lost_create_response_by_exact_unique_name() -> None:
    events: list[str] = []
    client = FakeClient(events)
    client.created = True

    result = runpod._terminate_and_audit(
        client,
        pod_name="aua-fg-test",
        pod_ids=set(),
        sleep=lambda _seconds: None,
    )

    assert result["verified_no_active_pod"] is True
    assert result["deleted_pod_ids"] == ["pod-exact"]


def _lifecycle_with_export_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_behavior,
    *,
    cleanup_signals: tuple[int, ...] = (),
    worker_result: runpod.CommandResult | None = None,
    key_add_mode: str = "success",
    key_remove_mode: str = "success",
):
    config = _default_config(tmp_path)
    config = runpod.Config(**{**config.__dict__, "run_id": "fg-test", "pod_name": "aua-fg-test"})
    secrets = runpod.Secrets("runpod-secret-value", "hf-secret-value")
    source = _tar_gz({"README.md": b"source"})
    events: list[str] = []
    client = FakeClient(events)
    account = FakeSSHAccount(
        events,
        add_mode=key_add_mode,
        remove_mode=key_remove_mode,
        remove_signals=cleanup_signals,
    )
    monkeypatch.setattr(runpod, "_source_archive", lambda _config, _runner, _env: (source, ["x"]))
    original_delete = client.delete_pod

    def delete_with_signals(pod_id: str) -> None:
        for signum in cleanup_signals:
            handler = runpod.signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)
        original_delete(pod_id)

    client.delete_pod = delete_with_signals  # type: ignore[method-assign]

    def runner(command, *, env=None, stdin=None, timeout=None):
        del env, timeout
        if command[0] == "ssh-keygen":
            _write_fake_ephemeral_key(command)
            return runpod.CommandResult(0, b"", b"")
        if command[0] == "runpodctl":
            if command[3] == "ssh":
                return account.run(command)
            assert command[3:5] == ["pod", "create"]
            events.append("pod-create")
            client.created = True
            return runpod.CommandResult(
                0,
                json.dumps({"id": "pod-exact", "ssh": {"ip": "192.0.2.1", "port": 22022}}).encode(),
                b"",
            )
        if stdin == source:
            return runpod.CommandResult(0, b"", b"")
        if stdin == b"hf-secret-value\n":
            return worker_result or runpod.CommandResult(0, b"trained\n", b"")
        events.append("artifact-export")
        return export_behavior()

    return config, secrets, client, account, events, runner


def test_export_timeout_cannot_skip_exact_id_delete_or_inventory_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout():
        raise runpod.subprocess.TimeoutExpired(["ssh", "artifact-export"], 180)

    config, secrets, client, _account, events, runner = _lifecycle_with_export_behavior(
        tmp_path, monkeypatch, timeout
    )

    with pytest.raises(runpod.subprocess.TimeoutExpired):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert events.index("artifact-export") < events.index("delete:pod-exact")
    assert metadata["cleanup"]["verified_no_active_pod"] is True
    assert metadata["cleanup"]["deleted_pod_ids"] == ["pod-exact"]
    assert metadata["network_volumes"]["verified_no_new_network_volume"] is True
    assert metadata["artifact_error"].startswith("TimeoutExpired:")
    assert metadata["ssh_key"]["cleanup"]["verified_baseline_restored"] is True
    assert _account.fingerprints == {BASELINE_FINGERPRINT}


def test_ssh_key_add_failure_prevents_pod_create_and_confirms_unchanged_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, secrets, client, account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        lambda: pytest.fail("artifact export must not run when key registration fails"),
        key_add_mode="fail",
    )

    with pytest.raises(runpod.LaunchError, match="registration did not converge"):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert "pod-create" not in events
    assert account.fingerprints == {BASELINE_FINGERPRINT}
    assert metadata["ssh_key"]["registration"]["attempted"] is True
    assert metadata["ssh_key"]["registration"]["verified_present"] is False
    assert metadata["ssh_key"]["cleanup"]["verified_baseline_restored"] is True
    assert metadata["cleanup"]["verified_no_active_pod"] is True


def test_lost_ssh_key_add_response_is_recovered_by_exact_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, secrets, client, account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        lambda: runpod.CommandResult(1, b"", b"artifact unavailable"),
        key_add_mode="lost",
    )

    with pytest.raises(runpod.LaunchError, match="artifact export failed"):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    registration = metadata["ssh_key"]["registration"]
    assert registration["verified_present"] is True
    assert registration["recovered_after_add_error"] is True
    assert registration["add_error"].startswith("TimeoutExpired:")
    assert events.index("ssh-add") < events.index("pod-create")
    assert events.index("delete:pod-exact") < events.index("ssh-remove")
    assert metadata["ssh_key"]["cleanup"]["verified_baseline_restored"] is True
    assert account.fingerprints == {BASELINE_FINGERPRINT}


def test_lost_ssh_key_remove_response_is_recovered_by_baseline_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, secrets, client, account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        lambda: runpod.CommandResult(1, b"", b"artifact unavailable"),
        key_remove_mode="lost",
    )

    with pytest.raises(runpod.LaunchError, match="artifact export failed"):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    key_cleanup = metadata["ssh_key"]["cleanup"]
    assert key_cleanup["verified_baseline_restored"] is True
    assert key_cleanup["remove_attempts"] == 1
    assert any("TimeoutExpired" in error for error in key_cleanup["errors"])
    assert events.index("delete:pod-exact") < events.index("ssh-remove")
    assert account.fingerprints == {BASELINE_FINGERPRINT}


def test_key_cleanup_unverified_is_primary_after_pod_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, secrets, client, account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        lambda: runpod.CommandResult(1, b"", b"artifact unavailable"),
        worker_result=runpod.CommandResult(1, b"", b"training failed"),
        key_remove_mode="stuck",
    )

    with pytest.raises(runpod.CleanupUnverifiedError, match="temporary RunPod account SSH key"):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert metadata["cleanup"]["verified_no_active_pod"] is True
    assert metadata["ssh_key"]["cleanup"]["verified_baseline_restored"] is False
    assert metadata["error"]["type"] == "CleanupUnverifiedError"
    assert "earlier error was LaunchError: RunPod worker failed" in metadata["error"]["message"]
    assert events.index("delete:pod-exact") < events.index("ssh-remove")
    assert TEST_FINGERPRINT in account.fingerprints


def test_signals_during_export_and_cleanup_are_deferred_until_after_exact_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted_export():
        for signum in runpod.TERMINATION_SIGNALS:
            handler = runpod.signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)
        return runpod.CommandResult(1, b"", b"interrupted export")

    config, secrets, client, _account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        interrupted_export,
        cleanup_signals=runpod.TERMINATION_SIGNALS,
    )

    with pytest.raises(runpod.LaunchError, match="artifact export failed"):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert events.index("artifact-export") < events.index("delete:pod-exact")
    assert metadata["cleanup"]["verified_no_active_pod"] is True
    assert metadata["cleanup"]["deleted_pod_ids"] == ["pod-exact"]
    signal_names = [runpod.signal.Signals(signum).name for signum in runpod.TERMINATION_SIGNALS]
    assert metadata["cleanup"]["deferred_signals"] == signal_names * 3
    assert metadata["network_volumes"]["verified_no_new_network_volume"] is True


def test_secret_bearing_artifact_archive_is_rejected_before_any_artifact_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _tar_gz({"leak.txt": b"prefix hf-secret-value suffix"})
    config, secrets, _client, _account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        lambda: runpod.CommandResult(0, artifact, b""),
    )

    with pytest.raises(runpod.LaunchError, match="archive contains a configured secret"):
        runpod.execute(
            config,
            secrets,
            client=_client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert not (config.output_dir / "artifacts.tar.gz").exists()
    assert not (config.output_dir / "artifacts").exists()
    assert events.index("artifact-export") < events.index("delete:pod-exact")
    assert metadata["cleanup"]["verified_no_active_pod"] is True
    assert metadata["cleanup"]["deleted_pod_ids"] == ["pod-exact"]
    assert metadata["network_volumes"]["verified_no_new_network_volume"] is True
    assert metadata["artifact_error"] == "LaunchError: archive contains a configured secret"


def test_cleanup_unverified_overrides_and_preserves_earlier_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, secrets, client, _account, events, runner = _lifecycle_with_export_behavior(
        tmp_path,
        monkeypatch,
        lambda: runpod.CommandResult(1, b"", b"artifact unavailable"),
        worker_result=runpod.CommandResult(1, b"", b"training failed"),
    )

    def leave_pod_active(pod_id: str) -> None:
        assert pod_id == "pod-exact"
        events.append(f"delete:{pod_id}")

    client.delete_pod = leave_pod_active  # type: ignore[method-assign]

    with pytest.raises(runpod.CleanupUnverifiedError, match="CLEANUP UNVERIFIED") as raised:
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    assert "earlier error was LaunchError: RunPod worker failed: training failed" in str(
        raised.value
    )
    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert metadata["cleanup"]["verified_no_active_pod"] is False
    assert metadata["error"]["type"] == "CleanupUnverifiedError"
    assert metadata["error"]["message"].startswith("CLEANUP UNVERIFIED:")
    assert (
        "earlier error was LaunchError: RunPod worker failed: training failed"
        in metadata["error"]["message"]
    )
    assert "delete:pod-exact" in events


def test_main_does_not_describe_unverified_cleanup_as_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runpod,
        "_secrets",
        lambda _path: runpod.Secrets("runpod-secret-value", "hf-secret-value"),
    )

    def fail_cleanup(_config, _secrets):
        raise runpod.CleanupUnverifiedError("CLEANUP UNVERIFIED: fictional audit failure")

    monkeypatch.setattr(runpod, "execute", fail_cleanup)

    result = runpod.main(
        [
            "--execute",
            "--revision",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "CLEANUP UNVERIFIED: fictional audit failure" in captured.err
    assert "Run failed safely" not in captured.err


def test_artifact_is_hashed_before_exact_id_delete_and_zero_resource_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _default_config(tmp_path)
    config = runpod.Config(
        **{
            **config.__dict__,
            "run_id": "fg-test",
            "pod_name": "aua-fg-test",
            "expected_manifest_sha256": None,
        }
    )
    secrets = runpod.Secrets("runpod-secret-value", "hf-secret-value")
    source = _tar_gz({"README.md": b"source"})
    source_sha256 = runpod._sha256_bytes(source)
    adapter_payload = b"trained-adapter"
    adapter_sha256 = runpod._sha256_bytes(adapter_payload)
    manifest_payload = b'{"fictional":"manifest"}\n'
    manifest_sha256 = runpod._sha256_bytes(manifest_payload)
    config_payload = b"fictional: config\n"
    config_sha256 = runpod._sha256_bytes(config_payload)
    final_hashes = {"adapters.safetensors": adapter_sha256}
    training_metadata = {
        "status": "completed",
        "mode": "benchmark",
        "model": {"revision": config.model_revision},
        "dataset": {"manifest_sha256": manifest_sha256},
        "config": {"sha256": config_sha256},
        "final_adapter_hashes": final_hashes,
        "exact_args": {"iters": 128},
    }
    artifact = _tar_gz(
        {
            "worker-metadata.json": json.dumps(
                {
                    "run_id": "fg-test",
                    "status": "completed",
                    "source": {
                        "base_revision": config.revision,
                        "archive_sha256": source_sha256,
                    },
                    "model": {
                        "repository": config.model_id,
                        "revision": config.model_revision,
                    },
                    "dataset_manifest_sha256": manifest_sha256,
                    "training_metadata": {
                        "status": "completed",
                        "final_adapter_hashes": final_hashes,
                    },
                    "durations_seconds": {"training": 12.5},
                    "environment": {"gpus": [{"name": "fictional gpu"}]},
                }
            ).encode(),
            "adapter/adapters.safetensors": adapter_payload,
            "adapter/run-metadata.json": json.dumps(training_metadata).encode(),
            "evidence/manifest.json": manifest_payload,
            "evidence/train-lora.yaml": config_payload,
        }
    )
    events: list[str] = []
    child_calls: list[tuple[list[str], dict[str, str], bytes | None]] = []
    client = FakeClient(events)
    account = FakeSSHAccount(events)
    monkeypatch.setattr(runpod, "_source_archive", lambda _config, _runner, _env: (source, ["x"]))

    def runner(command, *, env=None, stdin=None, timeout=None):
        del timeout
        child_calls.append((command, dict(env or {}), stdin))
        if command[0] == "ssh-keygen":
            _write_fake_ephemeral_key(command)
            return runpod.CommandResult(0, b"", b"")
        if command[0] == "runpodctl":
            if command[3] == "ssh":
                return account.run(command)
            assert command[3:5] == ["pod", "create"]
            events.append("pod-create")
            client.created = True
            return runpod.CommandResult(
                0,
                json.dumps({"id": "pod-exact", "ssh": {"ip": "192.0.2.1", "port": 22022}}).encode(),
                b"",
            )
        if stdin == source:
            events.append("source-upload")
            return runpod.CommandResult(0, b"", b"")
        if stdin == b"hf-secret-value\n":
            events.append("train")
            return runpod.CommandResult(0, b"trained\n", b"")
        events.append("artifact-export")
        return runpod.CommandResult(0, artifact, b"")

    result = runpod.execute(
        config,
        secrets,
        client=client,
        runner=runner,
        preflight=False,
        sleep=lambda _seconds: None,
    )

    assert result["status"] == "completed"
    assert result["artifact"]["archive_sha256"] == runpod._sha256_bytes(artifact)
    assert result["cleanup"]["verified_no_active_pod"] is True
    assert result["ssh_key"]["fingerprint"] == TEST_FINGERPRINT
    assert result["ssh_key"]["registration"]["verified_present"] is True
    assert result["ssh_key"]["cleanup"]["verified_baseline_restored"] is True
    assert account.fingerprints == {BASELINE_FINGERPRINT}
    assert result["network_volumes"] == {
        "creation_requested": False,
        "baseline_ids": ["volume-existing"],
        "after_ids": ["volume-existing"],
        "new_ids": [],
        "verified_no_new_network_volume": True,
        "error": None,
    }
    assert events.index("artifact-export") < events.index("delete:pod-exact")
    assert events.index("delete:pod-exact") < events.index("ssh-remove")
    assert (config.output_dir / "source.tar.gz").read_bytes() == source
    assert result["source"]["archive_sha256"] == source_sha256
    runpodctl_env = next(env for command, env, _stdin in child_calls if command[0] == "runpodctl")
    assert runpodctl_env["RUNPOD_API_KEY"] == "runpod-secret-value"
    assert "HF_TOKEN" not in runpodctl_env
    token_inputs = 0
    for command, environment, command_input in child_calls:
        rendered = " ".join(command)
        assert "runpod-secret-value" not in rendered
        assert "hf-secret-value" not in rendered
        assert "fictional-private-key" not in rendered
        assert b"fictional-private-key" not in (command_input or b"")
        if command_input == b"hf-secret-value\n":
            token_inputs += 1
        else:
            assert b"hf-secret-value" not in (command_input or b"")
        if command[0] != "runpodctl":
            assert "RUNPOD_API_KEY" not in environment
            assert "HF_TOKEN" not in environment
    assert token_inputs == 1
    serialized = (config.output_dir / "launcher-metadata.json").read_text(encoding="utf-8")
    assert "runpod-secret-value" not in serialized
    assert "hf-secret-value" not in serialized
    assert "fictional-private-key" not in serialized


def test_failure_still_deletes_exact_id_and_audits_no_active_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _default_config(tmp_path)
    config = runpod.Config(**{**config.__dict__, "run_id": "fg-test", "pod_name": "aua-fg-test"})
    secrets = runpod.Secrets("runpod-secret-value", "hf-secret-value")
    source = _tar_gz({"README.md": b"source"})
    events: list[str] = []
    client = FakeClient(events)
    account = FakeSSHAccount(events)
    monkeypatch.setattr(runpod, "_source_archive", lambda _config, _runner, _env: (source, ["x"]))

    def runner(command, *, env=None, stdin=None, timeout=None):
        del env, timeout
        if command[0] == "ssh-keygen":
            _write_fake_ephemeral_key(command)
            return runpod.CommandResult(0, b"", b"")
        if command[0] == "runpodctl":
            if command[3] == "ssh":
                return account.run(command)
            assert command[3:5] == ["pod", "create"]
            events.append("pod-create")
            client.created = True
            return runpod.CommandResult(
                0,
                json.dumps({"id": "pod-exact", "ssh": {"ip": "192.0.2.1", "port": 22022}}).encode(),
                b"",
            )
        if stdin == source:
            return runpod.CommandResult(0, b"", b"")
        if stdin == b"hf-secret-value\n":
            return runpod.CommandResult(1, b"", b"training failed")
        return runpod.CommandResult(1, b"", b"nothing to export")

    with pytest.raises(runpod.LaunchError, match="worker failed"):
        runpod.execute(
            config,
            secrets,
            client=client,
            runner=runner,
            preflight=False,
            sleep=lambda _seconds: None,
        )

    metadata = json.loads((config.output_dir / "launcher-metadata.json").read_text())
    assert metadata["cleanup"]["verified_no_active_pod"] is True
    assert metadata["ssh_key"]["cleanup"]["verified_baseline_restored"] is True
    assert account.fingerprints == {BASELINE_FINGERPRINT}
    assert "delete:pod-exact" in events
    assert events.index("delete:pod-exact") < events.index("ssh-remove")
    assert "runpod-secret-value" not in json.dumps(metadata)
    assert "hf-secret-value" not in json.dumps(metadata)
