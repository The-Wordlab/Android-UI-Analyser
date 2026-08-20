from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma import train


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "hub" / "snapshots" / "revision-example"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "example/function-model"}), encoding="utf-8"
    )
    (model / "weights.safetensors").write_bytes(b"fictional model weights")

    data = tmp_path / "data"
    data.mkdir()
    split_hashes: dict[str, str] = {}
    splits: dict[str, dict] = {}
    validation_splits: dict[str, dict] = {}
    for split in train.SPLITS:
        split_path = data / f"{split}.jsonl"
        split_path.write_text(f'{{"split":"{split}"}}\n', encoding="utf-8")
        digest = _hash(split_path)
        split_hashes[split] = digest
        splits[split] = {
            "path": split_path.name,
            "sha256": digest,
            "bytes": split_path.stat().st_size,
        }
        validation_splits[split] = {"sha256": digest, "tokens": {"max": 93}}
    dataset_hash = hashlib.sha256(
        "".join(split_hashes[split] for split in train.SPLITS).encode()
    ).hexdigest()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "privacy": {"passed": True},
                "dataset_sha256": dataset_hash,
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    (data / "validation.json").write_text(
        json.dumps(
            {
                "ok": True,
                "max_seq_length": 128,
                "splits": validation_splits,
            }
        ),
        encoding="utf-8",
    )

    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "adapter_path": "ignored-by-explicit-test-path",
                "batch_size": 8,
                "grad_accumulation_steps": 4,
                "iters": 80,
                "max_seq_length": 128,
                "save_every": 20,
                "seed": 17,
                "steps_per_eval": 20,
                "steps_per_report": 5,
                "val_batches": 8,
            }
        ),
        encoding="utf-8",
    )
    return model, data, config


class FakeMX:
    class random:
        seeds: list[int] = []

        @classmethod
        def seed(cls, value: int) -> None:
            cls.seeds.append(value)


class FakeLora:
    CONFIG_DEFAULTS = {
        "adapter_path": "adapters",
        "batch_size": 4,
        "config": None,
        "data": "data",
        "grad_accumulation_steps": 1,
        "iters": 100,
        "max_seq_length": 128,
        "model": "model",
        "resume_adapter_file": None,
        "save_every": 20,
        "seed": 0,
        "steps_per_eval": 20,
        "steps_per_report": 5,
        "test": False,
        "train": True,
        "val_batches": 8,
    }

    def __init__(self, failure: BaseException | None = None) -> None:
        self.calls: list[SimpleNamespace] = []
        self.samples: list[tuple[float, float]] = []
        self.failure = failure

    def run(self, args: SimpleNamespace) -> None:
        self.calls.append(args)
        self.samples.append((random.random(), float(np.random.random())))
        adapter = Path(args.adapter_path)
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "partial.safetensors").write_bytes(b"partial")
        if self.failure is not None:
            raise self.failure
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter / "adapters.safetensors").write_bytes(b"trained adapter")


def _backend(fake: FakeLora):
    return lambda: (FakeMX, fake)


def _run(
    tmp_path: Path,
    *,
    fake: FakeLora | None = None,
    mode: str = "full",
    adapter_path: Path | None = None,
    resume_adapter_file: Path | None = None,
    now=None,
):
    model, data, config = _fixture_files(tmp_path)
    fake = fake or FakeLora()
    adapter_path = adapter_path or tmp_path / "adapter"
    return (
        train.run(
            mode=mode,
            config_path=config,
            model=model,
            data_dir=data,
            adapter_path=adapter_path,
            resume_adapter_file=resume_adapter_file,
            backend_loader=_backend(fake),
            now=now or (lambda: "2026-08-15T00:00:00+00:00"),
        ),
        fake,
        adapter_path,
        data,
    )


def test_success_seeds_every_rng_and_records_exact_provenance(tmp_path: Path) -> None:
    FakeMX.random.seeds.clear()
    times = iter(("start", "complete"))
    result, fake, adapter, data = _run(tmp_path, now=lambda: next(times))

    assert FakeMX.random.seeds == [17]
    assert fake.samples == [(random.Random(17).random(), np.random.RandomState(17).random())]
    assert result["status"] == "completed"
    assert result["started_at"] == "start"
    assert result["completed_at"] == "complete"
    assert result["model"]["resolved_path"] == str(
        (tmp_path / "hub" / "snapshots" / "revision-example").resolve()
    )
    assert result["model"]["revision"] == "revision-example"
    assert result["model"]["sha256"]
    assert result["dataset"]["manifest_sha256"] == _hash(data / "manifest.json")
    assert result["config"]["sha256"]
    assert result["exact_args"] == vars(fake.calls[0])
    assert result["completed_checkpoint"] == str((adapter / "adapters.safetensors").resolve())
    assert set(result["final_adapter_hashes"]) == {
        "adapter_config.json",
        "adapters.safetensors",
        "partial.safetensors",
    }
    assert json.loads((adapter / train.METADATA_NAME).read_text()) == result
    assert not list(adapter.glob(".*.tmp"))


def test_smoke_mode_applies_bounded_overrides(tmp_path: Path) -> None:
    result, fake, _, _ = _run(tmp_path, mode="smoke")

    args = result["exact_args"]
    assert args["iters"] == 12
    assert args["batch_size"] == 2
    assert args["grad_accumulation_steps"] == 1
    assert args["save_every"] == 12
    assert args["steps_per_eval"] == 6
    assert args["val_batches"] == 4
    assert len(fake.calls) == 1


def test_benchmark_mode_runs_exactly_128_mlx_iterations(tmp_path: Path) -> None:
    result, fake, _, _ = _run(tmp_path, mode="benchmark")

    args = result["exact_args"]
    assert args["iters"] == 128
    assert args["batch_size"] == 8
    assert args["grad_accumulation_steps"] == 4
    assert args["save_every"] == 128
    assert args["steps_per_eval"] == 64
    assert args["steps_per_report"] == 5
    assert len(fake.calls) == 1


def test_tampered_split_fails_before_training_or_metadata(tmp_path: Path) -> None:
    model, data, config = _fixture_files(tmp_path)
    (data / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    fake = FakeLora()

    with pytest.raises(ValueError, match="train SHA256 mismatch"):
        train.run(
            mode="full",
            config_path=config,
            model=model,
            data_dir=data,
            adapter_path=adapter,
            backend_loader=_backend(fake),
        )

    assert fake.calls == []
    assert not adapter.exists()


def test_manifest_pin_and_token_contract_are_fail_closed(tmp_path: Path) -> None:
    model, data, config = _fixture_files(tmp_path)
    fake = FakeLora()
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        train.run(
            mode="full",
            config_path=config,
            model=model,
            data_dir=data,
            adapter_path=tmp_path / "adapter-pin",
            expected_manifest_sha256="0" * 64,
            backend_loader=_backend(fake),
        )
    validation = json.loads((data / "validation.json").read_text())
    validation["max_seq_length"] = 127
    (data / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(ValueError, match="token contract mismatch"):
        train.run(
            mode="full",
            config_path=config,
            model=model,
            data_dir=data,
            adapter_path=tmp_path / "adapter-token",
            backend_loader=_backend(fake),
        )
    assert fake.calls == []


def test_nonempty_target_requires_explicit_resume(tmp_path: Path) -> None:
    model, data, config = _fixture_files(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    parent = adapter / "adapters.safetensors"
    parent.write_bytes(b"parent")
    fake = FakeLora()
    with pytest.raises(ValueError, match="not empty"):
        train.run(
            mode="full",
            config_path=config,
            model=model,
            data_dir=data,
            adapter_path=adapter,
            backend_loader=_backend(fake),
        )

    result = train.run(
        mode="resume",
        config_path=config,
        model=model,
        data_dir=data,
        adapter_path=adapter,
        resume_adapter_file=parent,
        backend_loader=_backend(fake),
    )
    assert result["parent_adapter_sha256"] == hashlib.sha256(b"parent").hexdigest()
    assert result["exact_args"]["resume_adapter_file"] == str(parent.resolve())


@pytest.mark.parametrize(
    ("failure", "status"),
    [(RuntimeError("training failed"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_failure_and_interruption_are_recorded_then_reraised(
    tmp_path: Path, failure: BaseException, status: str
) -> None:
    fake = FakeLora(failure)
    with pytest.raises(type(failure)):
        _run(tmp_path, fake=fake)

    metadata = json.loads((tmp_path / "adapter" / train.METADATA_NAME).read_text())
    assert metadata["status"] == status
    assert metadata["completed_checkpoint"] is None
    assert metadata["error"]["type"] == type(failure).__name__
    assert metadata["final_adapter_hashes"] == {
        "partial.safetensors": hashlib.sha256(b"partial").hexdigest()
    }


def test_backend_is_loaded_only_after_local_preflight(tmp_path: Path) -> None:
    loaded = False

    def loader():
        nonlocal loaded
        loaded = True
        raise AssertionError("backend should remain lazy")

    with pytest.raises(FileNotFoundError):
        train.run(
            mode="full",
            config_path=tmp_path / "missing.yaml",
            model=tmp_path / "missing-model",
            data_dir=tmp_path / "missing-data",
            backend_loader=loader,
        )
    assert loaded is False
