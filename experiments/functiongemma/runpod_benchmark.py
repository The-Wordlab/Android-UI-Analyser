#!/usr/bin/env python3
"""Cost-bounded RunPod launcher for the host-only FunctionGemma benchmark.

Dry-run is the default. Spending requires ``--execute``. A server-side hard
termination deadline is set when the Pod is created; the launcher additionally
terminates the exact Pod in ``finally`` and audits that neither its ID nor its
unique name remains active.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
DEFAULT_GPU = "NVIDIA L40S"
DEFAULT_MODEL = "mlx-community/functiongemma-270m-it-bf16"
DEFAULT_MODEL_REVISION = "bb327a9ad61044e1496a2bee2365a6b6a6684c72"
DEFAULT_MLX_PACKAGE = "mlx[cuda12]==0.32.0"
V3_MANIFEST_SHA256 = "d96d69e7f25df0b10272d6e20027eea3f609a34a741ce50d60d75b7f983df60b"
V4_MANIFEST_SHA256 = "3a271e8ff153b9179997edbb9822962b383348405bc77b15259dc3a733b6a9b7"
SAFE_VALUE = re.compile(r"^[A-Za-z0-9._/@:+\-]+$")
MLX_PACKAGE = re.compile(r"^mlx\[cuda(?:12|13)\]==[0-9]+\.[0-9]+\.[0-9]+$")
SECRET_KEYS = ("RUNPOD_API_KEY", "HF_TOKEN")
TERMINATION_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGHUP", "SIGINT", "SIGQUIT", "SIGTERM")
    if hasattr(signal, name)
)


class LaunchError(RuntimeError):
    """A fail-closed launcher error with no secret-bearing response body."""


class CleanupUnverifiedError(LaunchError):
    """The Pod cleanup audit failed, irrespective of any earlier run error."""


@dataclass(frozen=True)
class Secrets:
    runpod_api_key: str
    hf_token: str


@dataclass(frozen=True)
class Config:
    run_id: str
    pod_name: str
    output_dir: Path
    gpu: str
    image: str
    revision: str
    mode: str
    curriculum_version: str
    config_path: str
    model_id: str
    model_revision: str
    mlx_package: str
    expected_manifest_sha256: str | None
    ttl_minutes: int
    wait_minutes: int
    max_hourly_usd: float
    max_total_usd: float
    terminate_after: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


CommandRunner = Callable[..., CommandResult]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _redact(value: str | bytes, secrets: Secrets) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    for secret in (secrets.runpod_api_key, secrets.hf_token):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def _load_env_file(path: Path, environment: dict[str, str]) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in SECRET_KEYS or key in environment:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            environment[key] = value


def _secrets(env_file: Path) -> Secrets:
    environment = dict(os.environ)
    _load_env_file(env_file, environment)
    missing = [key for key in SECRET_KEYS if not environment.get(key)]
    if missing:
        raise LaunchError(f"missing required secret environment variables: {', '.join(missing)}")
    return Secrets(environment["RUNPOD_API_KEY"], environment["HF_TOKEN"])


def _scrubbed_child_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in SECRET_KEYS}


def _run_command(
    command: list[str],
    *,
    env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
    timeout: float | None = None,
) -> CommandResult:
    completed = subprocess.run(  # noqa: S603
        command,
        env=dict(env) if env is not None else None,
        input=stdin,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class RunpodAPI:
    def __init__(self, token: str, *, timeout: float = 20.0) -> None:
        self._token = token
        self._timeout = timeout

    def _request(self, method: str, path: str, *, allowed: set[int]) -> Any:
        request = urllib.request.Request(
            f"{API_ROOT}{path}",
            method=method,
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                status = response.status
                payload = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            payload = b""
        except OSError as exc:
            raise LaunchError(
                f"RunPod API {method} {path} was unreachable: {type(exc).__name__}"
            ) from None
        if status not in allowed:
            raise LaunchError(f"RunPod API {method} {path} returned HTTP {status}")
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            raise LaunchError(f"RunPod API {method} {path} returned invalid JSON") from None

    def list_pods(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/pods", allowed={200})
        if not isinstance(payload, list):
            raise LaunchError("RunPod list-pods response was not a list")
        return [pod for pod in payload if isinstance(pod, dict)]

    def list_network_volumes(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/networkvolumes", allowed={200})
        if not isinstance(payload, list):
            raise LaunchError("RunPod network-volume response was not a list")
        return [volume for volume in payload if isinstance(volume, dict)]

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        payload = self._request(
            "GET", f"/pods/{urllib.parse.quote(pod_id, safe='')}", allowed={200}
        )
        if not isinstance(payload, dict):
            raise LaunchError("RunPod get-pod response was not an object")
        return payload

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{urllib.parse.quote(pod_id, safe='')}", allowed={204, 404})


def _matching_pods(
    pods: list[dict[str, Any]], *, pod_name: str, pod_ids: set[str]
) -> list[dict[str, Any]]:
    return [
        pod for pod in pods if pod.get("name") == pod_name or str(pod.get("id") or "") in pod_ids
    ]


def _network_volume_ids(volumes: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for volume in volumes:
        volume_id = str(volume.get("id") or "")
        if not volume_id:
            raise LaunchError("RunPod network-volume inventory contained an entry without an ID")
        result.add(volume_id)
    return result


def _terminate_and_audit(
    client: RunpodAPI,
    *,
    pod_name: str,
    pod_ids: set[str],
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 6,
) -> dict[str, Any]:
    deleted: set[str] = set()
    errors: list[str] = []
    for attempt in range(attempts):
        # Known IDs are deleted directly first. The exact-name lookup below is
        # only the recovery path for a create response lost after billing began.
        for pod_id in sorted(pod_ids - deleted):
            try:
                client.delete_pod(pod_id)
                deleted.add(pod_id)
            except LaunchError as exc:
                errors.append(str(exc))
        try:
            matches = _matching_pods(client.list_pods(), pod_name=pod_name, pod_ids=pod_ids)
            active = [pod for pod in matches if pod.get("desiredStatus") != "TERMINATED"]
            if not active:
                return {
                    "verified_no_active_pod": True,
                    "deleted_pod_ids": sorted(deleted),
                    "attempts": attempt + 1,
                    "errors": errors,
                }
            for pod in active:
                pod_id = str(pod.get("id") or "")
                if not pod_id:
                    errors.append("matching active Pod had no ID")
                    continue
                try:
                    client.delete_pod(pod_id)
                    deleted.add(pod_id)
                    pod_ids.add(pod_id)
                except LaunchError as exc:
                    errors.append(str(exc))
        except LaunchError as exc:
            errors.append(str(exc))
        if attempt + 1 < attempts:
            sleep(min(2**attempt, 8))
    return {
        "verified_no_active_pod": False,
        "deleted_pod_ids": sorted(deleted),
        "attempts": attempts,
        "errors": errors,
    }


def _validate_remote_value(label: str, value: str) -> str:
    if not SAFE_VALUE.fullmatch(value):
        raise LaunchError(f"{label} contains unsupported characters")
    return value


def _validate_config_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise LaunchError("--config must stay inside the cloned repository")
    return _validate_remote_value("--config", value)


def _validate_mlx_package(value: str) -> str:
    if not MLX_PACKAGE.fullmatch(value):
        raise LaunchError("--mlx-package must pin an MLX CUDA wheel like mlx[cuda12]==0.32.0")
    return value


def _run_id(now: datetime) -> str:
    return f"fg-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _config(args: argparse.Namespace, now: datetime) -> Config:
    run_id = args.run_id or _run_id(now)
    _validate_remote_value("--run-id", run_id)
    terminate_after = (now + timedelta(minutes=args.ttl_minutes)).replace(microsecond=0)
    output = args.output_dir or REPO_ROOT / "runs" / "functiongemma" / "runpod" / run_id
    expected = args.expected_manifest_sha256
    if expected is None:
        expected = V4_MANIFEST_SHA256 if args.curriculum_version == "v4" else V3_MANIFEST_SHA256
    config_path = args.config or (
        "experiments/functiongemma/train-lora-v4.yaml"
        if args.curriculum_version == "v4"
        else "experiments/functiongemma/train-lora.yaml"
    )
    if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise LaunchError("--expected-manifest-sha256 must be 64 lowercase hex characters")
    if args.ttl_minutes <= args.wait_minutes + 10:
        raise LaunchError("TTL must leave at least ten minutes after the Pod readiness window")
    if args.max_hourly_usd <= 0 or args.max_total_usd <= 0:
        raise LaunchError("cost limits must be positive")
    return Config(
        run_id=run_id,
        pod_name=f"aua-{run_id}",
        output_dir=output.resolve(),
        gpu=args.gpu,
        image=args.image,
        revision=_validate_remote_value("--revision", args.revision),
        mode=args.mode,
        curriculum_version=args.curriculum_version,
        config_path=_validate_config_path(config_path),
        model_id=_validate_remote_value("--model-id", args.model_id),
        model_revision=_validate_remote_value("--model-revision", args.model_revision),
        mlx_package=_validate_mlx_package(args.mlx_package),
        expected_manifest_sha256=expected,
        ttl_minutes=args.ttl_minutes,
        wait_minutes=args.wait_minutes,
        max_hourly_usd=args.max_hourly_usd,
        max_total_usd=args.max_total_usd,
        terminate_after=terminate_after.isoformat().replace("+00:00", "Z"),
    )


def _create_command(config: Config, public_key: str) -> list[str]:
    return [
        "runpodctl",
        "--output",
        "json",
        "pod",
        "create",
        "--name",
        config.pod_name,
        "--image",
        config.image,
        "--gpu-id",
        config.gpu,
        "--gpu-count",
        "1",
        "--container-disk-in-gb",
        "50",
        "--volume-in-gb",
        "0",
        "--cloud-type",
        "SECURE",
        "--min-cuda-version",
        "12.0",
        "--ports",
        "22/tcp",
        "--ssh",
        "--env",
        json.dumps({"SSH_PUBLIC_KEY": public_key}, separators=(",", ":")),
        "--terminate-after",
        config.terminate_after,
        "--wait",
        "--wait-timeout",
        f"{config.wait_minutes}m",
    ]


def _remote_command(config: Config, source_archive_sha256: str) -> str:
    remote_root = f"/workspace/aua-functiongemma/{config.run_id}"
    worker_runtime = config.ttl_minutes * 60 - 300
    arguments = [
        "python3",
        "-m",
        "experiments.functiongemma.runpod_worker",
        "--run-id",
        config.run_id,
        "--source-revision",
        config.revision,
        "--source-archive-sha256",
        source_archive_sha256,
        "--output-root",
        f"{remote_root}/output",
        "--mode",
        config.mode,
        "--curriculum-version",
        config.curriculum_version,
        "--config",
        config.config_path,
        "--model-id",
        config.model_id,
        "--model-revision",
        config.model_revision,
        "--mlx-package",
        config.mlx_package,
        "--max-runtime-seconds",
        str(worker_runtime),
    ]
    if config.expected_manifest_sha256:
        arguments.extend(["--expected-manifest-sha256", config.expected_manifest_sha256])
    script = [
        "set -euo pipefail",
        "umask 077",
        f"cd {shlex.quote(remote_root + '/repo')}",
        "exec " + " ".join(shlex.quote(value) for value in arguments),
    ]
    return "; ".join(script)


def _source_archive(
    config: Config, runner: CommandRunner, child_env: Mapping[str, str] | None = None
) -> tuple[bytes, list[str]]:
    """Archive the pinned Git tree plus the two reviewed RunPod runner overrides.

    This avoids requiring the new launcher commit to be published before its first
    use while still excluding ignored/untracked files such as .env, runs, models,
    and device journals.
    """
    archived = runner(
        ["git", "archive", "--format=tar", config.revision],
        env=child_env,
        timeout=120,
    )
    if archived.returncode != 0:
        raise LaunchError("could not archive the pinned Git revision")
    overrides = [
        "experiments/functiongemma/train.py",
        "experiments/functiongemma/runpod_worker.py",
    ]
    override_set = set(overrides)
    output = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as source,
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as target,
    ):
        for member in source.getmembers():
            if member.name in override_set:
                continue
            extracted = source.extractfile(member) if member.isfile() else None
            target.addfile(member, extracted)
        for relative in overrides:
            path = REPO_ROOT / relative
            if not path.is_file():
                raise LaunchError(f"reviewed source override is missing: {relative}")
            payload = path.read_bytes()
            member = tarfile.TarInfo(relative)
            member.size = len(payload)
            member.mode = path.stat().st_mode & 0o777
            member.mtime = 0
            target.addfile(member, io.BytesIO(payload))
    return output.getvalue(), overrides


def _ssh_command(ip: str, port: int, key: Path, known_hosts: Path, remote: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"root@{ip}",
        remote,
    ]


def _parse_json(payload: bytes, description: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        raise LaunchError(f"{description} did not return one JSON object") from None
    if not isinstance(parsed, dict):
        raise LaunchError(f"{description} did not return one JSON object")
    return parsed


def _price(pod: Mapping[str, Any]) -> float:
    values = []
    for key in ("costPerHr", "adjustedCostPerHr"):
        try:
            if pod.get(key) is not None:
                values.append(float(pod[key]))
        except (TypeError, ValueError):
            continue
    if not values:
        raise LaunchError("RunPod did not report an hourly Pod price")
    return max(values)


def _safe_extract(payload: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise LaunchError("artifact archive contains a path traversal") from exc
            if member.issym() or member.islnk() or member.isdev():
                raise LaunchError("artifact archive contains an unsupported link or device")
        archive.extractall(root, members=members)  # noqa: S202


def _assert_no_secrets(path: Path, secrets: Secrets) -> None:
    needles = [value.encode() for value in (secrets.runpod_api_key, secrets.hf_token) if value]
    for candidate in path.rglob("*"):
        if candidate.is_file():
            payload = candidate.read_bytes()
            if any(needle in payload for needle in needles):
                raise LaunchError("downloaded artifact contains a configured secret")


def _assert_archive_no_secrets(payload: bytes, secrets: Secrets) -> None:
    needles = [value.encode() for value in (secrets.runpod_api_key, secrets.hf_token) if value]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            content = handle.read() if handle is not None else b""
            if any(needle in content for needle in needles):
                raise LaunchError("archive contains a configured secret")


def _verify_artifacts(path: Path, config: Config, source_archive_sha256: str) -> dict[str, Any]:
    metadata_path = path / "worker-metadata.json"
    if not metadata_path.is_file():
        raise LaunchError("artifact bundle has no worker-metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != config.run_id or metadata.get("status") != "completed":
        raise LaunchError("RunPod worker did not record a completed matching run")
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("base_revision") != config.revision:
        raise LaunchError("worker source revision does not match the launcher")
    if source.get("archive_sha256") != source_archive_sha256:
        raise LaunchError("worker source archive hash does not match the launcher")
    model = metadata.get("model")
    if not isinstance(model, dict) or model != {
        "repository": config.model_id,
        "revision": config.model_revision,
    }:
        raise LaunchError("worker model identity does not match the launcher")
    if (
        config.expected_manifest_sha256
        and metadata.get("dataset_manifest_sha256") != config.expected_manifest_sha256
    ):
        raise LaunchError("worker dataset manifest hash does not match the launcher pin")

    adapter = path / "adapter" / "adapters.safetensors"
    if not adapter.is_file():
        raise LaunchError("artifact bundle has no completed adapters.safetensors")
    adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
    training_metadata_path = path / "adapter" / "run-metadata.json"
    if not training_metadata_path.is_file():
        raise LaunchError("artifact bundle has no adapter/run-metadata.json")
    training = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    if training.get("status") != "completed" or training.get("mode") != config.mode:
        raise LaunchError("adapter training metadata is not a completed matching mode")
    training_model = training.get("model")
    if (
        not isinstance(training_model, dict)
        or training_model.get("revision") != config.model_revision
    ):
        raise LaunchError("adapter base-model revision does not match the launcher")
    training_dataset = training.get("dataset")
    if not isinstance(training_dataset, dict):
        raise LaunchError("adapter metadata has no dataset identity")
    if training_dataset.get("manifest_sha256") != metadata.get("dataset_manifest_sha256"):
        raise LaunchError("adapter and worker dataset identities disagree")
    evidence_manifest = path / "evidence" / "manifest.json"
    if not evidence_manifest.is_file() or hashlib.sha256(
        evidence_manifest.read_bytes()
    ).hexdigest() != metadata.get("dataset_manifest_sha256"):
        raise LaunchError("exported dataset manifest does not match worker metadata")
    evidence_config = path / "evidence" / Path(config.config_path).name
    training_config = training.get("config")
    if (
        not evidence_config.is_file()
        or not isinstance(training_config, dict)
        or hashlib.sha256(evidence_config.read_bytes()).hexdigest() != training_config.get("sha256")
    ):
        raise LaunchError("exported training config does not match adapter metadata")
    hashes = training.get("final_adapter_hashes")
    if not isinstance(hashes, dict) or hashes.get("adapters.safetensors") != adapter_sha256:
        raise LaunchError("actual adapter SHA256 does not match adapter metadata")
    worker_training = metadata.get("training_metadata")
    if (
        not isinstance(worker_training, dict)
        or worker_training.get("status") != "completed"
        or worker_training.get("final_adapter_hashes") != hashes
    ):
        raise LaunchError("worker and adapter completion metadata disagree")
    exact_args = training.get("exact_args")
    if config.mode == "benchmark" and (
        not isinstance(exact_args, dict) or exact_args.get("iters") != 128
    ):
        raise LaunchError("benchmark artifact did not record exactly 128 MLX iterations")
    return {
        "worker_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "training_metadata_sha256": hashlib.sha256(training_metadata_path.read_bytes()).hexdigest(),
        "adapter_sha256": adapter_sha256,
        "training_seconds": metadata.get("durations_seconds", {}).get("training"),
        "gpu": metadata.get("environment", {}).get("gpus", []),
    }


@contextlib.contextmanager
def _signal_cleanup() -> Iterator[None]:
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in TERMINATION_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextlib.contextmanager
def _defer_cleanup_signals() -> Iterator[list[int]]:
    """Record termination signals without allowing them to skip cloud cleanup."""
    previous: dict[int, Any] = {}
    pending: list[int] = []

    def defer(signum: int, _frame: Any) -> None:
        pending.append(signum)

    for signum in TERMINATION_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, defer)
    try:
        yield pending
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _local_preflight(config: Config, runner: CommandRunner, child_env: Mapping[str, str]) -> None:
    if shutil.which("runpodctl") is None:
        raise LaunchError("runpodctl is required; install the current RunPod CLI first")
    if config.output_dir.exists():
        raise LaunchError(f"output directory already exists: {config.output_dir}")
    help_result = runner(["runpodctl", "pod", "create", "--help"], env=child_env, timeout=30)
    if help_result.returncode != 0:
        raise LaunchError("local preflight failed: runpodctl pod")
    help_text = help_result.stdout.decode("utf-8", errors="replace")
    if "--terminate-after" not in help_text or "--wait" not in help_text:
        raise LaunchError("installed runpodctl lacks required hard-TTL/readiness flags")

    checks = [
        ["git", "cat-file", "-e", f"{config.revision}:{config.config_path}"],
    ]
    for command in checks:
        result = runner(command, env=child_env, timeout=30)
        if result.returncode != 0:
            raise LaunchError(f"local preflight failed: {command[0]} {command[1]}")
    for relative in (
        "experiments/functiongemma/train.py",
        "experiments/functiongemma/runpod_worker.py",
    ):
        if not (REPO_ROOT / relative).is_file():
            raise LaunchError(f"reviewed source override is missing: {relative}")


def execute(
    config: Config,
    secrets: Secrets,
    *,
    client: RunpodAPI | None = None,
    runner: CommandRunner = _run_command,
    preflight: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    client = client or RunpodAPI(secrets.runpod_api_key)
    child_env = _scrubbed_child_env()
    if preflight:
        _local_preflight(config, runner, child_env)
    source_archive, source_overrides = _source_archive(config, runner, child_env)
    _assert_archive_no_secrets(source_archive, secrets)
    # Authentication and unique-name ownership are proven before any resource
    # can be created. This keeps the name-based lost-response recovery scoped.
    baseline_volume_ids = _network_volume_ids(client.list_network_volumes())
    existing = _matching_pods(client.list_pods(), pod_name=config.pod_name, pod_ids=set())
    if any(pod.get("desiredStatus") != "TERMINATED" for pod in existing):
        raise LaunchError(f"an active RunPod Pod already uses the unique name {config.pod_name}")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    source_archive_path = config.output_dir / "source.tar.gz"
    source_archive_path.write_bytes(source_archive)
    metadata_path = config.output_dir / "launcher-metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "status": "running",
        "started_at": _utc_now().isoformat(),
        "completed_at": None,
        "pod": {
            "id": None,
            "name": config.pod_name,
            "gpu": config.gpu,
            "image": config.image,
            "terminate_after": config.terminate_after,
            "hourly_usd": None,
        },
        "artifact": None,
        "artifact_error": None,
        "worker_log": None,
        "source": {
            "base_revision": config.revision,
            "archive": str(source_archive_path),
            "archive_sha256": _sha256_bytes(source_archive),
            "reviewed_worktree_overrides": source_overrides,
        },
        "network_volumes": {
            "creation_requested": False,
            "baseline_ids": sorted(baseline_volume_ids),
            "after_ids": None,
            "new_ids": None,
            "verified_no_new_network_volume": None,
            "error": None,
        },
        "cleanup": None,
        "error": None,
    }
    _atomic_json(metadata_path, metadata)
    pod_ids: set[str] = set()
    primary_error: BaseException | None = None
    ssh_endpoint: tuple[str, int] | None = None
    archive_payload: bytes | None = None

    with tempfile.TemporaryDirectory(prefix="aua-runpod-") as temporary:
        temporary_path = Path(temporary)
        private_key = temporary_path / "id_ed25519"
        known_hosts = temporary_path / "known_hosts"
        key_result = runner(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
            env=child_env,
            timeout=30,
        )
        if key_result.returncode != 0:
            raise LaunchError("could not generate an ephemeral SSH key")
        public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
        process_env = dict(child_env)
        process_env["RUNPOD_API_KEY"] = secrets.runpod_api_key

        try:
            with _signal_cleanup():
                created = runner(
                    _create_command(config, public_key),
                    env=process_env,
                    timeout=config.wait_minutes * 60 + 60,
                )
                if created.returncode != 0:
                    raise LaunchError(
                        "RunPod create/readiness failed: "
                        + _redact(created.stderr, secrets).strip()
                    )
                created_payload = _parse_json(created.stdout, "runpodctl create")
                pod_id = str(created_payload.get("id") or "")
                if not pod_id:
                    raise LaunchError("RunPod create response omitted the Pod ID")
                pod_ids.add(pod_id)
                metadata["pod"]["id"] = pod_id

                pod = client.get_pod(pod_id)
                hourly = _price(pod)
                metadata["pod"]["hourly_usd"] = hourly
                if hourly > config.max_hourly_usd:
                    raise LaunchError(
                        f"actual hourly price ${hourly:.4f} exceeds ${config.max_hourly_usd:.4f}"
                    )
                ceiling = hourly * config.ttl_minutes / 60
                if ceiling > config.max_total_usd:
                    raise LaunchError(
                        f"hard-TTL cost ceiling ${ceiling:.4f} exceeds ${config.max_total_usd:.4f}"
                    )

                ssh = created_payload.get("ssh")
                if not isinstance(ssh, dict):
                    raise LaunchError("RunPod create response omitted SSH connection details")
                ip = str(ssh.get("ip") or "")
                try:
                    port = int(ssh.get("port"))
                except (TypeError, ValueError):
                    port = 0
                if not ip or not 1 <= port <= 65535:
                    raise LaunchError("RunPod create response contained invalid SSH details")
                ssh_endpoint = (ip, port)

                remote_root = f"/workspace/aua-functiongemma/{config.run_id}"
                uploaded = runner(
                    _ssh_command(
                        ip,
                        port,
                        private_key,
                        known_hosts,
                        (
                            "set -euo pipefail; umask 077; "
                            f"mkdir -p {shlex.quote(remote_root + '/repo')}; "
                            f"tar -C {shlex.quote(remote_root + '/repo')} -xzf -"
                        ),
                    ),
                    env=child_env,
                    stdin=source_archive,
                    timeout=180,
                )
                if uploaded.returncode != 0:
                    raise LaunchError(
                        "source upload failed: " + _redact(uploaded.stderr, secrets).strip()
                    )

                remote = _remote_command(config, _sha256_bytes(source_archive))
                trained = runner(
                    _ssh_command(ip, port, private_key, known_hosts, remote),
                    env=child_env,
                    stdin=(secrets.hf_token + "\n").encode(),
                    timeout=config.ttl_minutes * 60 - 240,
                )
                stdout_text = _redact(trained.stdout, secrets)
                stderr_text = _redact(trained.stderr, secrets)
                log_text = f"[stdout]\n{stdout_text}\n[stderr]\n{stderr_text}\n"
                worker_log_path = config.output_dir / "worker.log"
                worker_log_path.write_text(log_text, encoding="utf-8")
                metadata["worker_log"] = {
                    "path": str(worker_log_path),
                    "sha256": hashlib.sha256(log_text.encode()).hexdigest(),
                }
                if stdout_text:
                    print(stdout_text, end="")
                if trained.returncode != 0:
                    raise LaunchError("RunPod worker failed: " + stderr_text.strip())
        except BaseException as exc:
            primary_error = exc
        finally:
            # Export is best-effort, but exact-ID deletion is structurally
            # unavoidable: it lives in the nested finally and termination
            # signals are deferred until both deletion and inventory audits end.
            with _defer_cleanup_signals() as deferred_signals:
                try:
                    try:
                        if ssh_endpoint is not None:
                            ip, port = ssh_endpoint
                            remote_output = f"/workspace/aua-functiongemma/{config.run_id}/output"
                            exported = runner(
                                _ssh_command(
                                    ip,
                                    port,
                                    private_key,
                                    known_hosts,
                                    f"tar -C {shlex.quote(remote_output)} -czf - .",
                                ),
                                env=child_env,
                                timeout=180,
                            )
                            if exported.returncode == 0 and exported.stdout:
                                archive_payload = exported.stdout
                                _assert_archive_no_secrets(archive_payload, secrets)
                                archive_path = config.output_dir / "artifacts.tar.gz"
                                archive_path.write_bytes(archive_payload)
                                artifact_dir = config.output_dir / "artifacts"
                                _safe_extract(archive_payload, artifact_dir)
                                _assert_no_secrets(config.output_dir, secrets)
                                verification = _verify_artifacts(
                                    artifact_dir, config, _sha256_bytes(source_archive)
                                )
                                metadata["artifact"] = {
                                    "archive": str(archive_path),
                                    "archive_sha256": _sha256_bytes(archive_payload),
                                    **verification,
                                }
                            elif primary_error is None:
                                artifact_error = LaunchError(
                                    "artifact export failed before Pod termination: "
                                    + _redact(exported.stderr, secrets).strip()
                                )
                                metadata["artifact_error"] = str(artifact_error)
                                primary_error = artifact_error
                            elif exported.returncode != 0 or not exported.stdout:
                                metadata["artifact_error"] = (
                                    "artifact export failed after an earlier run failure: "
                                    + _redact(exported.stderr, secrets).strip()
                                )
                    except BaseException as exc:
                        metadata["artifact_error"] = (
                            f"{type(exc).__name__}: {_redact(str(exc), secrets)}"
                        )
                        if primary_error is None:
                            primary_error = exc
                finally:
                    try:
                        cleanup = _terminate_and_audit(
                            client,
                            pod_name=config.pod_name,
                            pod_ids=pod_ids,
                            sleep=sleep,
                        )
                    except BaseException as exc:
                        cleanup = {
                            "verified_no_active_pod": False,
                            "deleted_pod_ids": [],
                            "attempts": 0,
                            "errors": [f"unexpected cleanup error: {type(exc).__name__}"],
                        }
                        if primary_error is None:
                            primary_error = exc
                    metadata["cleanup"] = cleanup
                    if not cleanup["verified_no_active_pod"]:
                        cleanup_message = (
                            "CLEANUP UNVERIFIED: could not verify that the RunPod Pod was "
                            "terminated; hard TTL remains set"
                        )
                        if primary_error is not None:
                            cleanup_message += (
                                f"; earlier error was {type(primary_error).__name__}: "
                                f"{_redact(str(primary_error), secrets)}"
                            )
                        primary_error = CleanupUnverifiedError(cleanup_message)

                    try:
                        after_volume_ids = _network_volume_ids(client.list_network_volumes())
                        new_volume_ids = after_volume_ids - baseline_volume_ids
                        metadata["network_volumes"].update(
                            {
                                "after_ids": sorted(after_volume_ids),
                                "new_ids": sorted(new_volume_ids),
                                "verified_no_new_network_volume": not new_volume_ids,
                            }
                        )
                        if new_volume_ids and primary_error is None:
                            primary_error = LaunchError(
                                "RunPod network-volume inventory gained unexpected resources"
                            )
                    except BaseException as exc:
                        metadata["network_volumes"].update(
                            {
                                "verified_no_new_network_volume": False,
                                "error": f"{type(exc).__name__}: {_redact(str(exc), secrets)}",
                            }
                        )
                        if primary_error is None:
                            primary_error = exc

            if deferred_signals:
                metadata["cleanup"]["deferred_signals"] = [
                    signal.Signals(signum).name for signum in deferred_signals
                ]
                if primary_error is None:
                    primary_error = KeyboardInterrupt(
                        "termination signal deferred until RunPod cleanup completed"
                    )

    metadata["completed_at"] = _utc_now().isoformat()
    metadata["status"] = "completed" if primary_error is None else "failed"
    if primary_error is not None:
        metadata["error"] = {
            "type": type(primary_error).__name__,
            "message": _redact(str(primary_error), secrets),
        }
    _atomic_json(metadata_path, metadata)
    _assert_no_secrets(config.output_dir, secrets)
    if primary_error is not None:
        raise primary_error
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually create a billable Pod")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--mode", choices=("smoke", "benchmark", "full"), default="benchmark")
    parser.add_argument("--curriculum-version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--config")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--mlx-package", default=DEFAULT_MLX_PACKAGE)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--ttl-minutes", type=int, default=90)
    parser.add_argument("--wait-minutes", type=int, default=10)
    parser.add_argument("--max-hourly-usd", type=float, default=1.25)
    parser.add_argument("--max-total-usd", type=float, default=2.00)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.revision is None:
        args.revision = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            env=_scrubbed_child_env(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    now = _utc_now()
    config = _config(args, now)
    plan = {
        "execute": args.execute,
        "run_id": config.run_id,
        "pod_name": config.pod_name,
        "gpu": config.gpu,
        "image": config.image,
        "revision": config.revision,
        "mode": config.mode,
        "curriculum_version": config.curriculum_version,
        "training_config": config.config_path,
        "dataset_manifest_sha256": config.expected_manifest_sha256,
        "model": {"repository": config.model_id, "revision": config.model_revision},
        "mlx_package": config.mlx_package,
        "hard_terminate_after": config.terminate_after,
        "max_hourly_usd": config.max_hourly_usd,
        "max_total_usd": config.max_total_usd,
        "output_dir": str(config.output_dir),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("Dry run only. Re-run with --execute to authorize a billable Pod.")
        return 0
    secrets = _secrets(args.env_file)
    try:
        result = execute(config, secrets)
    except KeyboardInterrupt:
        print(
            f"Interrupted; cleanup was attempted. Inspect {config.output_dir / 'launcher-metadata.json'}",
            file=sys.stderr,
        )
        return 130
    except BaseException as exc:
        message = _redact(str(exc), secrets)
        if isinstance(exc, CleanupUnverifiedError):
            failure = message
        else:
            failure = f"Run failed safely: {message}"
        print(f"{failure}. Inspect {config.output_dir / 'launcher-metadata.json'}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
