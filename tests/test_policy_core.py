from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import android_ui_analyser.providers.policy.functiongemma as functiongemma_mod
from android_ui_analyser.config import Config, default_config_yaml
from android_ui_analyser.policy import (
    PolicyCandidate,
    PolicyContext,
    compile_policy_context,
    evaluate_policy,
    guard_candidates,
    policy_status,
)
from android_ui_analyser.providers.base import Availability
from android_ui_analyser.providers.policy.functiongemma import (
    FunctionGemmaPolicySelector,
    parse_candidate_id,
    validate_local_artifacts,
)
from android_ui_analyser.providers.registry import ProviderFactory, registered_names


def _candidate(
    candidate_id: int = 17,
    *,
    tool: str = "tap_and_analyze",
    arguments: dict[str, Any] | None = None,
    **values: Any,
) -> PolicyCandidate:
    defaults: dict[str, Any] = {
        "candidate_id": candidate_id,
        "call": {
            "tool": tool,
            "arguments": arguments
            if arguments is not None
            else {"rid": "com.example.fixture:id/openRecord"},
        },
        "purpose": "Open the visible fictional record.",
        "proof": "The settled detail screen is visible.",
        "session_id": "session-fixture",
        "phase": "open_record",
        "observation_fingerprint": "fresh-fingerprint",
        "package": "com.example.fixture",
    }
    defaults.update(values)
    return PolicyCandidate(**defaults)


def _context(*candidates: PolicyCandidate, **values: Any) -> PolicyContext:
    defaults: dict[str, Any] = {
        "goal": "Open the fictional record and prove arrival.",
        "phase": "open_record",
        "candidates": tuple(candidates),
        "observation": {"fresh": True, "known_screen": "fixture_home"},
        "session_id": "session-fixture",
        "observation_fingerprint": "fresh-fingerprint",
        "package": "com.example.fixture",
    }
    defaults.update(values)
    return PolicyContext(**defaults)


class _Selector:
    name = "fixture_policy"

    def __init__(
        self,
        selected: int | None,
        *,
        available: bool = True,
        raises: bool = False,
    ) -> None:
        self.selected = selected
        self.available = available
        self.raises = raises
        self.availability_calls = 0
        self.select_calls = 0
        self.context: PolicyContext | None = None

    def is_available(self) -> Availability:
        self.availability_calls += 1
        if self.raises:
            raise RuntimeError("fixture availability failure")
        return Availability(self.available, "fixture unavailable" if not self.available else "ok")

    def select(self, context: PolicyContext) -> int | None:
        self.select_calls += 1
        self.context = context
        if self.raises:
            raise RuntimeError("fixture inference failure")
        return self.selected


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"fictional model bytes")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "model": str(model)}),
        encoding="utf-8",
    )
    (adapter / "adapters.safetensors").write_bytes(b"fictional adapter bytes")
    return model, adapter


def _settings(model: Path, adapter: Path, **values: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "model_path": str(model),
        "adapter_path": str(adapter),
        "max_tokens": 24,
    }
    settings.update(values)
    return settings


def _required_file_manifest(model: Path, *relative_paths: str) -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "sha256": hashlib.sha256((model / relative).read_bytes()).hexdigest(),
            "bytes": (model / relative).stat().st_size,
        }
        for relative in relative_paths
    }


def _write_bundled_manifest(
    model: Path,
    adapter: Path,
    *,
    adapter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_files = _required_file_manifest(model, "config.json", "weights.safetensors")
    canonical_hash = functiongemma_mod._canonical_model_sha256(
        {relative: value["sha256"] for relative, value in required_files.items()}
    )
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapters.safetensors"
    adapter_identity: dict[str, Any] = {
        "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "bytes": weights_path.stat().st_size,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "fine_tune_type": "lora",
    }
    adapter_identity.update(adapter_overrides or {})
    manifest = {
        "schema_version": 1,
        "rollout": {"max_mode": "shadow"},
        "prompt_schema": {
            "name": functiongemma_mod.PROMPT_SCHEMA_NAME,
            "candidate_ids": functiongemma_mod.PROMPT_CANDIDATE_IDS,
            "candidate_count": functiongemma_mod.PROMPT_CANDIDATE_COUNT,
        },
        "base_model": {"sha256": canonical_hash, "files": required_files},
        "adapter": adapter_identity,
    }
    (adapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _explicit_advisory_settings(
    model: Path,
    adapter: Path,
    *,
    candidate_counts: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    model_hash = functiongemma_mod._tree_sha256(model)
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapters.safetensors"
    adapter_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "rollout": {"max_mode": "advisory"},
        "prompt_schema": {
            "name": functiongemma_mod.PROMPT_SCHEMA_NAME,
            "candidate_ids": functiongemma_mod.PROMPT_CANDIDATE_IDS,
            **(
                {"candidate_counts": list(candidate_counts)}
                if candidate_counts is not None
                else {"candidate_count": functiongemma_mod.PROMPT_CANDIDATE_COUNT}
            ),
        },
        "base_model": {"sha256": model_hash},
        "adapter": {
            "sha256": adapter_hash,
            "bytes": weights_path.stat().st_size,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "fine_tune_type": "lora",
        },
    }
    manifest_path = adapter / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return _settings(
        model,
        adapter,
        model_sha256=model_hash,
        adapter_sha256=adapter_hash,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


class _Tokenizer:
    has_chat_template = True
    tool_call_start = "<start_function_call>"
    tool_call_end = "<end_function_call>"

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None
        self.tools: list[dict[str, Any]] | None = None
        self.eos: list[str] = []

    def tool_parser(self, output: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        marker = "candidate_id:"
        value = int(output.split(marker, maxsplit=1)[1].split("}", maxsplit=1)[0])
        return {"name": "select_candidate", "arguments": {"candidate_id": value}}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        **_: Any,
    ) -> list[int]:
        self.messages = messages
        self.tools = tools
        return [1, 2, 3]

    def add_eos_token(self, token: str) -> None:
        self.eos.append(token)


def test_policy_config_is_disabled_and_weightless_by_default() -> None:
    cfg = Config()
    assert cfg.policy.enabled is False
    assert cfg.policy.mode == "off"
    assert cfg.policy.chain == ["functiongemma"]
    assert cfg.policy.max_candidates == 4
    assert cfg.models["functiongemma"] == {
        "model_path": None,
        "adapter_path": None,
        "max_tokens": 24,
        "model_sha256": None,
        "adapter_sha256": None,
        "manifest_sha256": None,
    }
    assert "functiongemma" in registered_names("policy")


def test_policy_config_round_trips_and_bounds_candidate_count() -> None:
    parsed = yaml.safe_load(default_config_yaml())
    cfg = Config.model_validate(parsed)
    assert cfg.policy.mode == "off"
    with pytest.raises(ValidationError):
        Config.model_validate({"policy": {"max_candidates": 0}})
    with pytest.raises(ValidationError):
        Config.model_validate({"policy": {"max_candidates": 5}})
    with pytest.raises(ValidationError):
        Config.model_validate({"policy": {"mode": "execute"}})


def test_guard_withholds_every_untrusted_or_stale_candidate() -> None:
    current = _candidate(
        11,
        session_id="session-fixture",
        phase="open_record",
        observation_fingerprint="fresh-fingerprint",
        package="com.example.fixture",
    )
    candidates = (
        current,
        _candidate(12, safe=False),
        _candidate(13, risk="unsafe"),
        _candidate(14, authorized=False),
        _candidate(15, redundant=True),
        _candidate(16, current=False),
        _candidate(17, observation_fingerprint="stale-fingerprint"),
        _candidate(18, arguments={"rid": "fixture", "allow_unsafe": True}),
        _candidate(19, call={"tool": "tap_and_analyze"}),
        _candidate(20),
        _candidate(20),
    )
    context = _context(
        *candidates,
        session_id="session-fixture",
        observation_fingerprint="fresh-fingerprint",
        package="com.example.fixture",
    )
    assert guard_candidates(context, max_candidates=4) == (current,)
    with pytest.raises(ValueError, match="1 to 4"):
        guard_candidates(context, max_candidates=5)


def test_compiler_keeps_training_shape_but_never_unscreened_call_values_or_identity() -> None:
    secret_selector = "com.example.fixture:id/verySensitiveRecord"
    candidate = _candidate(
        arguments={"rid": secret_selector, "until": "text:Private fixture copy"},
        session_id="session-secret",
        observation_fingerprint="fingerprint-secret",
        package="com.example.fixture",
    )
    context = _context(
        candidate,
        observation={
            "fresh": True,
            "known_screen": "fixture_home",
            "elements": [{"text": "private UI copy"}],
            "serial": "device-secret",
        },
        recent_outcomes=("screen_changed=true",),
        constraints=("Use a current observation",),
        session_id="session-secret",
        observation_fingerprint="fingerprint-secret",
        package="com.example.fixture",
    )
    compiled = compile_policy_context(context, guard_candidates(context))
    encoded = json.dumps(compiled)
    assert set(compiled) == {
        "fixture_ref",
        "request",
        "goal",
        "phase",
        "observation",
        "recent_outcomes",
        "constraints",
        "candidates",
    }
    assert compiled["candidates"][0]["call"] == {
        "tool": "tap_and_analyze",
        "arguments": {},
    }
    assert compiled["candidates"][0]["cleanup"] == "none"
    assert secret_selector not in encoded
    assert "Private fixture copy" not in encoded
    assert "session-secret" not in encoded
    assert "fingerprint-secret" not in encoded
    assert "device-secret" not in encoded
    assert "private UI copy" not in encoded


def test_production_prompt_contract_matches_frozen_v3_curriculum(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from experiments.functiongemma.curriculum import SELECT_CANDIDATE_TOOL as FROZEN_TOOL
    from experiments.functiongemma.runtime import SELECTOR_POLICY as FROZEN_POLICY

    from android_ui_analyser.policy import SELECTOR_POLICY, policy_tools

    candidate = _candidate(
        0,
        arguments={"rid": "com.example.fixture:id/openRecord", "until": "text:Detail"},
        model_arguments={"rid": "fixture-open-record", "until": "text:Fixture detail"},
    )
    compiled = compile_policy_context(_context(candidate))

    assert SELECTOR_POLICY == FROZEN_POLICY
    assert policy_tools() == [FROZEN_TOOL]
    assert set(compiled) == {
        "fixture_ref",
        "request",
        "goal",
        "phase",
        "observation",
        "recent_outcomes",
        "constraints",
        "candidates",
    }
    assert set(compiled["candidates"][0]) == {
        "id",
        "call",
        "purpose",
        "risk",
        "authorized",
        "redundant",
        "proof",
        "cleanup",
    }
    assert set(compiled["candidates"][0]["call"]) == {"tool", "arguments"}


def test_compiler_emits_only_explicit_privacy_screened_model_arguments() -> None:
    candidate = _candidate(
        2,
        arguments={"rid": "openRecord", "until": "text:Private fixture copy"},
        model_arguments={"rid": "openRecord"},
    )

    compiled = compile_policy_context(_context(candidate), guard_candidates(_context(candidate)))

    assert compiled["candidates"][0]["call"] == {
        "tool": "tap_and_analyze",
        "arguments": {"rid": "openRecord"},
    }
    assert "Private fixture copy" not in json.dumps(compiled)


def test_zero_and_one_candidates_skip_provider_and_model() -> None:
    selector = _Selector(17, raises=True)
    empty = evaluate_policy(_context(), selector, mode="shadow")
    assert empty.status == "no_candidate"
    one = evaluate_policy(_context(_candidate()), selector, mode="advisory")
    assert one.status == "deterministic"
    assert one.model_used is False
    assert "recommended_call" not in one.as_json()
    assert selector.availability_calls == selector.select_calls == 0


def test_shadow_records_opaque_choice_but_only_advisory_exposes_trusted_call() -> None:
    candidates = (_candidate(31), _candidate(73, tool="wait_for"))
    selector = _Selector(73)
    shadow = evaluate_policy(_context(*candidates), selector, mode="shadow")
    assert shadow.status == "selected"
    assert shadow.model_used is True
    assert shadow.selected_candidate_id == 73
    assert "recommended_call" not in shadow.as_json()
    advisory = evaluate_policy(_context(*candidates), selector, mode="advisory")
    assert advisory.as_json()["recommended_call"]["tool"] == "wait_for"
    assert selector.context is not None
    assert selector.context.candidates == candidates


@pytest.mark.parametrize("selected", [None, True, 999])
def test_invalid_model_selection_fails_closed(selected: Any) -> None:
    decision = evaluate_policy(
        _context(_candidate(31), _candidate(73)),
        _Selector(selected),
        mode="advisory",
    )
    assert decision.status == "invalid_selection"
    assert decision.selected_candidate is None
    assert "recommended_call" not in decision.as_json()


def test_unavailable_and_raising_provider_fail_closed() -> None:
    context = _context(_candidate(31), _candidate(73))
    unavailable = evaluate_policy(context, _Selector(31, available=False), mode="advisory")
    assert unavailable.status == "unavailable"
    failed = evaluate_policy(context, _Selector(31, raises=True), mode="advisory")
    assert failed.status == "unavailable"
    assert unavailable.selected_candidate is failed.selected_candidate is None


def test_local_artifacts_require_absolute_matching_lora_paths(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    provenance = validate_local_artifacts(_settings(model, adapter), include_model_hash=True)
    assert provenance["fine_tune_type"] == "lora"
    assert provenance["adapter_bytes"] > 0
    assert len(provenance["adapter_sha256"]) == 64
    assert len(provenance["model_sha256"]) == 64

    with pytest.raises(ValueError, match="absolute local path"):
        validate_local_artifacts(_settings(Path("relative-model"), adapter))
    config_path = adapter / "adapter_config.json"
    config_path.write_text(
        json.dumps({"fine_tune_type": "full", "model": str(model)}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="LoRA"):
        validate_local_artifacts(_settings(model, adapter))


def test_optional_hashes_are_verified_before_load(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    identity = validate_local_artifacts(_settings(model, adapter), include_model_hash=True)
    verified = validate_local_artifacts(
        _settings(
            model,
            adapter,
            model_sha256=identity["model_sha256"],
            adapter_sha256=identity["adapter_sha256"],
        )
    )
    assert verified["model_hash_verified"] is True
    assert verified["adapter_hash_verified"] is True
    with pytest.raises(ValueError, match="adapter_sha256"):
        validate_local_artifacts(_settings(model, adapter, adapter_sha256="0" * 64))


def test_bundled_adapter_uses_manifest_for_portable_base_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    identity = validate_local_artifacts(_settings(model, adapter))
    weights = adapter / "adapters.safetensors"
    required_files = _required_file_manifest(model, "config.json", "weights.safetensors")
    canonical_hash = functiongemma_mod._canonical_model_sha256(
        {relative: value["sha256"] for relative, value in required_files.items()}
    )
    # Unrelated snapshot files are deliberately outside the runtime identity.
    (model / "README.md").write_text("fictional extra documentation", encoding="utf-8")
    # The packaged adapter config may retain its training-machine path: the
    # manifest, not that non-portable path, binds a bundled adapter to its base.
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "model": "/old/training/snapshot"}),
        encoding="utf-8",
    )
    (adapter / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rollout": {"max_mode": "shadow"},
                "prompt_schema": {
                    "name": functiongemma_mod.PROMPT_SCHEMA_NAME,
                    "candidate_ids": functiongemma_mod.PROMPT_CANDIDATE_IDS,
                    "candidate_count": functiongemma_mod.PROMPT_CANDIDATE_COUNT,
                },
                "base_model": {"sha256": canonical_hash, "files": required_files},
                "adapter": {
                    "sha256": identity["adapter_sha256"],
                    "bytes": weights.stat().st_size,
                    "weights": "adapters.safetensors",
                    "config": "adapter_config.json",
                    "config_sha256": hashlib.sha256(
                        (adapter / "adapter_config.json").read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)

    provenance = validate_local_artifacts(
        {"model_path": str(model), "adapter_path": None, "max_tokens": 24}
    )

    assert provenance["adapter_source"] == "bundled"
    assert provenance["model_hash_verified"] is True
    assert provenance["adapter_hash_verified"] is True
    assert provenance["adapter_path"] == str(adapter.resolve())
    assert provenance["model_sha256"] == canonical_hash
    assert provenance["model_hash_kind"] == "required_runtime_files"
    assert provenance["model_required_files"] == ["config.json", "weights.safetensors"]
    assert provenance["model_required_bytes"] == sum(
        value["bytes"] for value in required_files.values()
    )


def test_bundled_manifest_rejects_wrong_external_base(tmp_path: Path, monkeypatch) -> None:
    model, adapter = _artifacts(tmp_path)
    weights = adapter / "adapters.safetensors"
    adapter_hash = validate_local_artifacts(_settings(model, adapter))["adapter_sha256"]
    required_files = _required_file_manifest(model, "config.json", "weights.safetensors")
    (adapter / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rollout": {"max_mode": "shadow"},
                "prompt_schema": {
                    "name": functiongemma_mod.PROMPT_SCHEMA_NAME,
                    "candidate_ids": functiongemma_mod.PROMPT_CANDIDATE_IDS,
                    "candidate_count": functiongemma_mod.PROMPT_CANDIDATE_COUNT,
                },
                "base_model": {"sha256": "0" * 64, "files": required_files},
                "adapter": {
                    "sha256": adapter_hash,
                    "bytes": weights.stat().st_size,
                    "config_sha256": hashlib.sha256(
                        (adapter / "adapter_config.json").read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)

    with pytest.raises(ValueError, match="model_sha256"):
        validate_local_artifacts({"model_path": str(model), "adapter_path": "bundled"})


def test_bundled_manifest_validates_each_required_file(tmp_path: Path, monkeypatch) -> None:
    model, adapter = _artifacts(tmp_path)
    weights = adapter / "adapters.safetensors"
    required_files = _required_file_manifest(model, "config.json", "weights.safetensors")
    required_files["config.json"]["bytes"] += 1
    canonical_hash = functiongemma_mod._canonical_model_sha256(
        {relative: value["sha256"] for relative, value in required_files.items()}
    )
    (adapter / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rollout": {"max_mode": "shadow"},
                "prompt_schema": {
                    "name": functiongemma_mod.PROMPT_SCHEMA_NAME,
                    "candidate_ids": functiongemma_mod.PROMPT_CANDIDATE_IDS,
                    "candidate_count": functiongemma_mod.PROMPT_CANDIDATE_COUNT,
                },
                "base_model": {"sha256": canonical_hash, "files": required_files},
                "adapter": {
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "bytes": weights.stat().st_size,
                    "config_sha256": hashlib.sha256(
                        (adapter / "adapter_config.json").read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)

    with pytest.raises(ValueError, match="byte size mismatch: config.json"):
        validate_local_artifacts({"model_path": str(model), "adapter_path": None})


def test_bundled_manifest_verifies_adapter_config_hash_and_lora_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    config_path = adapter / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "model": "/nonportable/training/path",
                "lora_parameters": {"rank": 16, "scale": 32.0, "dropout": 0.05},
            }
        ),
        encoding="utf-8",
    )
    manifest = _write_bundled_manifest(
        model,
        adapter,
        adapter_overrides={"rank": 16, "scale": 32.0, "dropout": 0.05},
    )
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)

    assert (
        validate_local_artifacts({"model_path": str(model), "adapter_path": "bundled"})[
            "adapter_config_sha256"
        ]
        == manifest["adapter"]["config_sha256"]
    )

    manifest["adapter"]["rank"] = 8
    (adapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="rank does not match"):
        validate_local_artifacts({"model_path": str(model), "adapter_path": None})

    manifest["adapter"]["rank"] = 16
    (adapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config SHA-256"):
        validate_local_artifacts({"model_path": str(model), "adapter_path": None})


def test_bundled_manifest_rejects_incompatible_prompt_schema(tmp_path: Path, monkeypatch) -> None:
    model, adapter = _artifacts(tmp_path)
    manifest = _write_bundled_manifest(model, adapter)
    manifest["prompt_schema"]["candidate_count"] = 5
    (adapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)

    with pytest.raises(ValueError, match="prompt_schema is incompatible"):
        validate_local_artifacts({"model_path": str(model), "adapter_path": None})


def test_bundled_provenance_cache_uses_required_file_stat_signature(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    _write_bundled_manifest(model, adapter)
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)
    original_hash = functiongemma_mod._file_sha256
    hashes: list[str] = []

    def counting_hash(path: Path) -> str:
        hashes.append(path.name)
        return original_hash(path)

    monkeypatch.setattr(functiongemma_mod, "_file_sha256", counting_hash)
    selector = FunctionGemmaPolicySelector(
        {"model_path": str(model), "adapter_path": None, "max_tokens": 24},
        runtime_availability=lambda: Availability(True, "fixture runtime"),
    )

    assert selector.is_available().ok is True
    first_hash_count = len(hashes)
    assert first_hash_count > 0
    assert selector.is_available().ok is True
    assert len(hashes) == first_hash_count

    config_path = model / "config.json"
    stat = config_path.stat()
    os.utime(config_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert selector.is_available().ok is True
    assert len(hashes) > first_hash_count


def test_bundled_adapter_forces_full_verification_immediately_before_first_load(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    _write_bundled_manifest(model, adapter)
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)
    original_hash = functiongemma_mod._file_sha256
    hashes: list[str] = []

    def counting_hash(path: Path) -> str:
        hashes.append(path.name)
        return original_hash(path)

    monkeypatch.setattr(functiongemma_mod, "_file_sha256", counting_hash)
    tokenizer = _Tokenizer()
    loads: list[str] = []

    def load(model_path: str, *, adapter_path: str) -> tuple[object, _Tokenizer]:
        loads.append(f"{model_path}:{adapter_path}")
        return object(), tokenizer

    selector = FunctionGemmaPolicySelector(
        {"model_path": str(model), "adapter_path": None, "max_tokens": 24},
        model_loader=load,
        generator=lambda *args, **kwargs: (
            "<start_function_call>call:select_candidate{candidate_id:1}"
        ),
        sampler_factory=lambda **kwargs: kwargs,
    )
    assert selector.is_available().ok is True
    availability_hashes = len(hashes)

    assert (
        selector.select(_context(_candidate(0), _candidate(1), _candidate(2), _candidate(3))) == 1
    )
    assert len(hashes) > availability_hashes
    assert len(loads) == 1


def test_runtime_file_canonical_hash_matches_published_identity() -> None:
    runtime_hashes = {
        "chat_template.jinja": "db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b",
        "config.json": "c60d3e4c9a04dcb53fae5e616d2d76a75f3eeecdc6e82f9b805904fc11724608",
        "generation_config.json": "f3f694d69c044000d84e777938fe314137af2ae06014c552332d1c82f5e1b13d",
        "model.safetensors": "fb64bf18b2911fcaa59d44c1b7d5842a011a874530be9dc5bc9d307e82b4edee",
        "model.safetensors.index.json": "419991d2afa817203a4941777d0511cc736c7c38912e33021366ba7b7a2bb83b",
        "tokenizer.json": "3b83627b470a2b3eeb6cbd480490191e50f21549cb3de1d0fcc1a001a48c6c04",
        "tokenizer_config.json": "9ffb9b4c29d60699a0c50b00abce43379640678852e442b1dec36a0421003a42",
    }

    assert (
        functiongemma_mod._canonical_model_sha256(runtime_hashes)
        == "76aabb2800b6b9e6da9160028dfb233bbfa723d8c33e21623022ca87a8fa9fd5"
    )


def test_functiongemma_is_lazy_local_and_protocol_strict(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    tokenizer = _Tokenizer()
    loads: list[tuple[str, str]] = []
    generations: list[dict[str, Any]] = []

    def load(model_path: str, *, adapter_path: str) -> tuple[object, _Tokenizer]:
        loads.append((model_path, adapter_path))
        return object(), tokenizer

    def generate(*args: Any, **kwargs: Any) -> str:
        generations.append(kwargs)
        return "<start_function_call>call:select_candidate{candidate_id:1}"

    selector = FunctionGemmaPolicySelector(
        _settings(model, adapter),
        model_loader=load,
        generator=generate,
        sampler_factory=lambda **kwargs: kwargs,
    )
    assert selector.is_available().ok is True
    assert loads == []
    context = _context(
        _candidate(0),
        _candidate(1, tool="wait_for"),
        _candidate(2, tool="analyze_screen"),
        _candidate(3, tool="has"),
    )
    assert selector.select(context) == 1
    assert selector.select(context) == 1
    assert loads == [(str(model.resolve()), str(adapter.resolve()))]
    assert generations[0]["max_tokens"] == 24
    assert generations[0]["sampler"] == {"temp": 0.0}
    assert tokenizer.eos == ["<end_function_call>"]
    assert tokenizer.messages is not None
    model_user_json = tokenizer.messages[-1]["content"]
    assert "com.example.fixture:id/openRecord" not in model_user_json


def test_functiongemma_rejects_malformed_and_off_list_calls(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    tokenizer = _Tokenizer()
    outputs = iter(
        [
            "explanation <start_function_call>call:select_candidate{candidate_id:31}",
            "<start_function_call>call:select_candidate{candidate_id:999}",
        ]
    )
    selector = FunctionGemmaPolicySelector(
        _settings(model, adapter),
        model_loader=lambda *args, **kwargs: (object(), tokenizer),
        generator=lambda *args, **kwargs: next(outputs),
        sampler_factory=lambda **kwargs: kwargs,
    )
    context = _context(_candidate(0), _candidate(1), _candidate(2), _candidate(3))
    assert selector.select(context) is None
    assert selector.select(context) is None
    assert selector.last_error is not None


def test_functiongemma_rejects_non_dense_ids_before_model_load() -> None:
    loads: list[str] = []
    selector = FunctionGemmaPolicySelector(
        {"max_tokens": 24},
        model_loader=lambda *args, **kwargs: loads.append("loaded"),  # type: ignore[arg-type]
        generator=lambda *args, **kwargs: "",
        sampler_factory=lambda **kwargs: kwargs,
    )

    assert (
        selector.select(_context(_candidate(0), _candidate(1), _candidate(2), _candidate(9)))
        is None
    )
    assert selector.last_error == (
        "the FunctionGemma adapter does not support this dense candidate cardinality"
    )
    assert loads == []


@pytest.mark.parametrize("count", [2, 3])
def test_functiongemma_rejects_candidate_counts_absent_from_training(count: int) -> None:
    loads: list[str] = []
    selector = FunctionGemmaPolicySelector(
        {"max_tokens": 24},
        model_loader=lambda *args, **kwargs: loads.append("loaded"),  # type: ignore[arg-type]
        generator=lambda *args, **kwargs: "",
        sampler_factory=lambda **kwargs: kwargs,
    )

    assert selector.select(_context(*(_candidate(index) for index in range(count)))) is None
    assert "does not support" in str(selector.last_error)
    assert loads == []


@pytest.mark.parametrize("count", [2, 3, 4])
def test_authenticated_adapter_supports_learned_variable_cardinality(
    tmp_path: Path,
    count: int,
) -> None:
    model, adapter = _artifacts(tmp_path)
    settings = _explicit_advisory_settings(
        model,
        adapter,
        candidate_counts=(2, 3, 4),
    )
    tokenizer = _Tokenizer()
    selector = FunctionGemmaPolicySelector(
        settings,
        model_loader=lambda *args, **kwargs: (object(), tokenizer),
        generator=lambda *args, **kwargs: (
            "<start_function_call>call:select_candidate{candidate_id:1}"
        ),
        sampler_factory=lambda **kwargs: kwargs,
    )

    assert selector.supports_candidate_count(count) is True
    assert selector.select(_context(*(_candidate(index) for index in range(count)))) == 1
    assert selector.status()["supported_candidate_counts"] == [2, 3, 4]


@pytest.mark.parametrize("count", [2, 3])
def test_core_skips_functiongemma_availability_for_unsupported_cardinality(count: int) -> None:
    selector = FunctionGemmaPolicySelector(
        {"max_tokens": 24},
        runtime_availability=lambda: (_ for _ in ()).throw(
            AssertionError("availability and artifact hashing must be skipped")
        ),
    )

    decision = evaluate_policy(
        _context(*(_candidate(index) for index in range(count))),
        selector,
        mode="shadow",
    )

    assert decision.status == "unsupported_cardinality"
    assert decision.model_used is False
    assert decision.selected_candidate is None


def test_shadow_only_bundled_capability_blocks_advisory_before_availability(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    _write_bundled_manifest(model, adapter)
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)
    selector = FunctionGemmaPolicySelector(
        {"model_path": str(model), "adapter_path": None, "max_tokens": 24},
        runtime_availability=lambda: (_ for _ in ()).throw(
            AssertionError("blocked advisory must skip availability and artifact hashes")
        ),
    )

    decision = evaluate_policy(
        _context(_candidate(0), _candidate(1), _candidate(2), _candidate(3)),
        selector,
        mode="advisory",
    )

    assert decision.status == "unsupported_mode"
    assert decision.model_used is False
    assert decision.selected_candidate is None
    assert "recommended_call" not in decision.as_json()
    assert decision.error == "bundled manifest limits rollout to shadow"


def test_shadow_only_bundled_capability_still_allows_shadow_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    _write_bundled_manifest(model, adapter)
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)
    tokenizer = _Tokenizer()
    selector = FunctionGemmaPolicySelector(
        {"model_path": str(model), "adapter_path": None, "max_tokens": 24},
        model_loader=lambda *args, **kwargs: (object(), tokenizer),
        generator=lambda *args, **kwargs: (
            "<start_function_call>call:select_candidate{candidate_id:1}"
        ),
        sampler_factory=lambda **kwargs: kwargs,
    )

    decision = evaluate_policy(
        _context(_candidate(0), _candidate(1), _candidate(2), _candidate(3)),
        selector,
        mode="shadow",
    )

    assert decision.status == "selected"
    assert decision.model_used is True
    assert decision.selected_candidate_id == 1
    assert "recommended_call" not in decision.as_json()


def test_explicit_advisory_requires_pinned_rollout_and_artifact_hashes(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    unpinned = FunctionGemmaPolicySelector(_settings(model, adapter))
    assert unpinned.supports_mode("advisory") is False
    assert unpinned.rollout_capability()["authenticated"] is False

    settings = _explicit_advisory_settings(model, adapter)
    selector = FunctionGemmaPolicySelector(settings)
    capability = selector.rollout_capability()
    provenance = validate_local_artifacts(settings)

    assert selector.supports_mode("advisory") is True
    assert capability["authenticated"] is True
    assert capability["max_mode"] == "advisory"
    assert provenance["rollout_authenticated"] is True
    assert provenance["rollout_max_mode"] == "advisory"


def test_policy_status_marks_shadow_only_provider_not_ready_for_advisory(
    tmp_path: Path, monkeypatch
) -> None:
    model, adapter = _artifacts(tmp_path)
    _write_bundled_manifest(model, adapter)
    monkeypatch.setattr(functiongemma_mod, "bundled_adapter_path", lambda: adapter)
    cfg = Config()
    cfg.policy.enabled = True
    cfg.policy.mode = "advisory"
    cfg.models["functiongemma"].update(
        {"model_path": str(model), "adapter_path": None, "max_tokens": 24}
    )

    status = policy_status(cfg)

    assert status["ready"] is False
    assert status["providers"][0]["configured_mode_supported"] is False
    assert status["providers"][0]["rollout"] == {
        "authenticated": True,
        "max_mode": "shadow",
        "supported_modes": ["shadow"],
        "source": "bundled_manifest",
        "reason": "bundled manifest limits rollout to shadow",
    }


def test_parser_requires_exactly_one_canonical_call() -> None:
    tokenizer = _Tokenizer()
    tools: list[dict[str, Any]] = []
    assert (
        parse_candidate_id(
            "<start_function_call>call:select_candidate{candidate_id:73}", tokenizer, tools
        )
        == 73
    )
    with pytest.raises(ValueError):
        parse_candidate_id(
            "<start_function_call>call:select_candidate{candidate_id:73} trailing",
            tokenizer,
            tools,
        )


def test_policy_status_is_host_only_and_reports_disabled_readiness(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    cfg = Config.model_validate(
        {
            "policy": {"enabled": False, "mode": "off", "chain": ["functiongemma"]},
            "models": {"functiongemma": _settings(model, adapter)},
        }
    )
    status = policy_status(cfg, factory=ProviderFactory(cfg))
    assert status["enabled"] is False
    assert status["ready"] is False
    assert status["providers"][0]["loaded"] is False


def test_provider_status_reports_artifacts_when_runtime_is_unavailable(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    selector = FunctionGemmaPolicySelector(
        _settings(model, adapter),
        runtime_availability=lambda: Availability(False, "fictional runtime absent"),
    )

    status = selector.status()

    assert status["available"] is False
    assert status["runtime"] == {"ready": False, "reason": "fictional runtime absent"}
    assert status["artifacts"]["ready"] is True
    assert len(status["provenance"]["adapter_sha256"]) == 64
    assert status["loaded"] is False
