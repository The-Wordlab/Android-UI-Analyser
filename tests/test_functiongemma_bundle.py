from __future__ import annotations

import hashlib
import json
import struct
import tomllib
from pathlib import Path

from android_ui_analyser.providers.policy.functiongemma import (
    bundled_adapter_path,
)

EXPECTED_NOTICE = (
    "Gemma is provided under and subject to the Gemma Terms of Use found at "
    "ai.google.dev/gemma/terms\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_bundled_adapter_matches_manifest_and_has_no_machine_paths() -> None:
    root = bundled_adapter_path()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    adapter = manifest["adapter"]
    weights = root / adapter["weights"]
    config = root / adapter["config"]

    assert weights.stat().st_size == adapter["bytes"] == 15_215_272
    assert _sha256(weights) == adapter["sha256"]
    assert _sha256(config) == adapter["config_sha256"]
    assert adapter["training_output_sha256"] == (
        "f4d2f2ed67ea1b50cdc8db511900df789d8767961ffc6f7271fe40478718575b"
    )
    assert "/Users/" not in config.read_text(encoding="utf-8")
    assert manifest["rollout"] == {"max_mode": "shadow"}
    assert manifest["evaluation"]["production_smoke"] == {
        "cases": 96,
        "semantic_correct": 60,
        "semantic_accuracy": 0.625,
        "protocol_parse_success": 1.0,
        "offered_id_success": 1.0,
        "provider_protocol_agreement": 1.0,
        "passed": False,
    }
    assert manifest["evaluation"]["production_verdict"].startswith("shadow only")


def test_bundled_adapter_carries_prominent_derivative_metadata() -> None:
    weights = bundled_adapter_path() / "adapters.safetensors"
    with weights.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))

    metadata = header["__metadata__"]
    assert metadata["modified_by"] == "The Wordlab"
    assert metadata["modified_at"] == "2026-08-14"
    assert "LoRA fine-tuning" in metadata["modification"]
    assert metadata["upstream_model"] == "google/functiongemma-270m-it"
    assert metadata["license"].startswith("Gemma Terms of Use")


def test_bundled_adapter_includes_exact_notice_and_complete_terms() -> None:
    root = bundled_adapter_path()
    assert (root / "NOTICE").read_text(encoding="utf-8") == EXPECTED_NOTICE

    agreement = (root / "LICENSE").read_text(encoding="utf-8")
    prohibited = (root / "GEMMA_PROHIBITED_USE_POLICY.md").read_text(encoding="utf-8")
    assert all(f"Section {number}:" in agreement for number in range(1, 5))
    assert "FunctionGemma" in agreement
    assert "Gemma Prohibited Use Policy" in prohibited
    assert "Making automated decisions" in prohibited


def test_project_package_metadata_points_to_the_mixed_license_notice() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    mixed_license = (project_root / "LICENSE").read_text(encoding="utf-8")

    assert project["project"]["license"] == {"file": "LICENSE"}
    assert "does not apply to the modified FunctionGemma model derivative" in mixed_license
    assert "resources/functiongemma" in mixed_license
    plugin = json.loads(
        (project_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    assert plugin["license"] == "MIT AND LicenseRef-Gemma-Terms-of-Use"
