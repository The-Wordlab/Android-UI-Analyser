from __future__ import annotations

import hashlib
import json
import struct
import tomllib
from pathlib import Path

from android_ui_analyser.policy import POLICY_HANDOFF_ID
from android_ui_analyser.providers.policy.functiongemma import (
    PROMPT_CANDIDATE_IDS,
    PROMPT_HANDOFF_FIELD,
    PROMPT_SCHEMA_NAME,
    FunctionGemmaPolicySelector,
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

    assert weights.stat().st_size == adapter["bytes"] == 30_403_414
    assert _sha256(weights) == adapter["sha256"]
    assert _sha256(config) == adapter["config_sha256"]
    # Packaging adds the safetensors distribution metadata, so the shipped file differs from the
    # raw training output by those bytes; this pin preserves the original checkpoint identity.
    assert adapter["training_output_sha256"] == (
        "69b1a360af4aa75b070ec085d196ab2cae62db40317ee8dcd1367a0986204b8f"
    )
    config_text = config.read_text(encoding="utf-8")
    # A shipped artifact must not carry the packager's machine or the training pod's filesystem.
    assert "/Users/" not in config_text
    assert "/workspace/" not in config_text
    # Advisory is permitted here on purpose. The protection against unintended execution is
    # that policy.enabled defaults to false — nothing is loaded and no resources are spent until
    # an operator turns the policy on — not a ceiling inside the artifact. Shadow remains
    # available as a development setting for tracing decisions without acting on them.
    assert manifest["rollout"] == {"max_mode": "advisory"}
    evaluation = manifest["evaluation"]
    # Scores come from an independently authored probe. A probe that shares its generator's
    # phrasing scores the phrasing: an earlier in-house probe reported 6/6 on a refusal capability
    # that independent measurement put at 0/144.
    assert evaluation["probe_jobs"] == 150
    assert evaluation["probe_refusal_best_checkpoint"] == "18/38"
    assert evaluation["device"]["wrong_taps"] == 0
    # The artifact must state plainly that it is not promoted.
    assert "not promoted" in evaluation["verdict"]


def test_the_bundle_declares_the_capability_it_was_trained_and_measured_with() -> None:
    """A manifest that under-declares silently disables the adapter on most real screens.

    The prompt schema is what the provider trusts: it gates which candidate counts reach the model
    and whether a refusal is accepted at all. Packaging this adapter alongside the previous
    generation's four-only, no-handoff schema loaded and passed every other check while failing
    closed on any screen offering two or three safe candidates, and discarding the refusal the
    device runs actually exercised. Capability therefore has to be pinned to the artifact that was
    measured, not inherited from whatever shipped before it.
    """

    manifest = json.loads((bundled_adapter_path() / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest["prompt_schema"]

    # The wire protocol is unchanged and its name stays frozen; only the adapter generation moved.
    assert schema["name"] == PROMPT_SCHEMA_NAME
    assert schema["candidate_ids"] == PROMPT_CANDIDATE_IDS
    assert schema["candidate_counts"] == [2, 3, 4]
    assert schema[PROMPT_HANDOFF_FIELD] == POLICY_HANDOFF_ID
    # The two cardinality formats are mutually exclusive; the older scalar must be gone.
    assert "candidate_count" not in schema

    selector = FunctionGemmaPolicySelector({"model_path": None, "max_tokens": 24})
    assert [count for count in (2, 3, 4) if selector.supports_candidate_count(count)] == [2, 3, 4]
    assert selector.supports_handoff() is True


def test_bundled_adapter_carries_prominent_derivative_metadata() -> None:
    weights = bundled_adapter_path() / "adapters.safetensors"
    with weights.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))

    metadata = header["__metadata__"]
    assert metadata["modified_by"] == "The Wordlab"
    assert metadata["modified_at"] == "2026-08-19"
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
