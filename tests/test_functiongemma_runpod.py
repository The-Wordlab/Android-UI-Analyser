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


def test_worker_uses_module_invocation_for_fresh_archive_commands() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "experiments" / "functiongemma" / "runpod_worker.py"
    ).read_text(encoding="utf-8")

    assert '"experiments.functiongemma.generate_dataset"' in source
    assert '"experiments.functiongemma.validate_dataset"' in source
    assert '"experiments/functiongemma/generate_dataset.py"' not in source


def test_source_archive_is_git_only_plus_two_reviewed_overrides(tmp_path: Path) -> None:
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
    assert overrides == [
        "experiments/functiongemma/train.py",
        "experiments/functiongemma/runpod_worker.py",
    ]
    assert "experiments/functiongemma/runpod_worker.py" in names
    assert ".env" not in names
    assert not any(name.startswith("runs/") for name in names)


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    payload = _tar_gz({"../escaped": b"bad"})
    with pytest.raises(runpod.LaunchError, match="path traversal"):
        runpod._safe_extract(payload, tmp_path / "artifacts")


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
):
    config = _default_config(tmp_path)
    config = runpod.Config(**{**config.__dict__, "run_id": "fg-test", "pod_name": "aua-fg-test"})
    secrets = runpod.Secrets("runpod-secret-value", "hf-secret-value")
    source = _tar_gz({"README.md": b"source"})
    events: list[str] = []
    client = FakeClient(events)
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
            key = Path(command[command.index("-f") + 1])
            key.write_text("private", encoding="utf-8")
            key.with_suffix(".pub").write_text("ssh-ed25519 PUBLIC", encoding="utf-8")
            return runpod.CommandResult(0, b"", b"")
        if command[0] == "runpodctl":
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

    return config, secrets, client, events, runner


def test_export_timeout_cannot_skip_exact_id_delete_or_inventory_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout():
        raise runpod.subprocess.TimeoutExpired(["ssh", "artifact-export"], 180)

    config, secrets, client, events, runner = _lifecycle_with_export_behavior(
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


def test_signals_during_export_and_cleanup_are_deferred_until_after_exact_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted_export():
        for signum in runpod.TERMINATION_SIGNALS:
            handler = runpod.signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)
        return runpod.CommandResult(1, b"", b"interrupted export")

    config, secrets, client, events, runner = _lifecycle_with_export_behavior(
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
    assert metadata["cleanup"]["deferred_signals"] == signal_names * 2
    assert metadata["network_volumes"]["verified_no_new_network_volume"] is True


def test_secret_bearing_artifact_archive_is_rejected_before_any_artifact_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _tar_gz({"leak.txt": b"prefix hf-secret-value suffix"})
    config, secrets, _client, events, runner = _lifecycle_with_export_behavior(
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
    config, secrets, client, events, runner = _lifecycle_with_export_behavior(
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
    child_calls: list[tuple[list[str], dict[str, str]]] = []
    client = FakeClient(events)
    monkeypatch.setattr(runpod, "_source_archive", lambda _config, _runner, _env: (source, ["x"]))

    def runner(command, *, env=None, stdin=None, timeout=None):
        del timeout
        child_calls.append((command, dict(env or {})))
        if command[0] == "ssh-keygen":
            key = Path(command[command.index("-f") + 1])
            key.write_text("private", encoding="utf-8")
            key.with_suffix(".pub").write_text("ssh-ed25519 PUBLIC", encoding="utf-8")
            return runpod.CommandResult(0, b"", b"")
        if command[0] == "runpodctl":
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
    assert result["network_volumes"] == {
        "creation_requested": False,
        "baseline_ids": ["volume-existing"],
        "after_ids": ["volume-existing"],
        "new_ids": [],
        "verified_no_new_network_volume": True,
        "error": None,
    }
    assert events.index("artifact-export") < events.index("delete:pod-exact")
    assert (config.output_dir / "source.tar.gz").read_bytes() == source
    assert result["source"]["archive_sha256"] == source_sha256
    runpodctl_env = next(env for command, env in child_calls if command[0] == "runpodctl")
    assert runpodctl_env["RUNPOD_API_KEY"] == "runpod-secret-value"
    assert "HF_TOKEN" not in runpodctl_env
    for command, environment in child_calls:
        rendered = " ".join(command)
        assert "runpod-secret-value" not in rendered
        assert "hf-secret-value" not in rendered
        if command[0] != "runpodctl":
            assert "RUNPOD_API_KEY" not in environment
            assert "HF_TOKEN" not in environment
    serialized = (config.output_dir / "launcher-metadata.json").read_text(encoding="utf-8")
    assert "runpod-secret-value" not in serialized
    assert "hf-secret-value" not in serialized


def test_failure_still_deletes_exact_id_and_audits_no_active_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _default_config(tmp_path)
    config = runpod.Config(**{**config.__dict__, "run_id": "fg-test", "pod_name": "aua-fg-test"})
    secrets = runpod.Secrets("runpod-secret-value", "hf-secret-value")
    source = _tar_gz({"README.md": b"source"})
    events: list[str] = []
    client = FakeClient(events)
    monkeypatch.setattr(runpod, "_source_archive", lambda _config, _runner, _env: (source, ["x"]))

    def runner(command, *, env=None, stdin=None, timeout=None):
        del env, timeout
        if command[0] == "ssh-keygen":
            key = Path(command[command.index("-f") + 1])
            key.write_text("private", encoding="utf-8")
            key.with_suffix(".pub").write_text("ssh-ed25519 PUBLIC", encoding="utf-8")
            return runpod.CommandResult(0, b"", b"")
        if command[0] == "runpodctl":
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
    assert "delete:pod-exact" in events
    assert "runpod-secret-value" not in json.dumps(metadata)
    assert "hf-secret-value" not in json.dumps(metadata)
