from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import android_ui_analyser.providers.policy.functiongemma as functiongemma_mod
from android_ui_analyser.config import Config, default_config_yaml
from android_ui_analyser.policy import (
    POLICY_HANDOFF_ID,
    PolicyCandidate,
    PolicyContext,
    _candidate_semantic_terms,
    _semantic_terms,
    compile_policy_context,
    evaluate_policy,
    evaluate_selective_policy,
    guard_candidates,
    policy_messages,
    policy_status,
    policy_tools,
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


def _choose_purpose(term: str) -> Callable[[PolicyContext], int]:
    def choose(context: PolicyContext) -> int:
        return next(
            candidate.candidate_id for candidate in context.candidates if term in candidate.purpose
        )

    return choose


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


class _SemanticSelector:
    def __init__(self, name: str, choose: Callable[[PolicyContext], int]) -> None:
        self.name = name
        self.choose = choose
        self.availability_calls = 0
        self.select_calls = 0

    def is_available(self) -> Availability:
        self.availability_calls += 1
        return Availability(True, "ready")

    def select(self, context: PolicyContext) -> int:
        self.select_calls += 1
        return self.choose(context)

    def supports_candidate_count(self, count: int) -> bool:
        return 2 <= count <= 4

    def supports_mode(self, mode: str) -> bool:
        return mode in {"shadow", "advisory"}

    def supports_handoff(self) -> bool:
        return True


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
    handoff: bool = False,
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
            **({"handoff_candidate_id": POLICY_HANDOFF_ID} if handoff else {}),
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


def test_selective_hybrid_accepts_unique_primary_match_without_loading_reviewer() -> None:
    settings = _candidate(
        0,
        arguments={"rid": "openSettings"},
        model_arguments={"rid": "openSettings"},
        purpose="Tap the Settings control and observe the result.",
    )
    ideas = _candidate(
        1,
        arguments={"rid": "openIdeas"},
        model_arguments={"rid": "openIdeas"},
        purpose="Tap the Ideas control and observe the result.",
    )
    primary = _SemanticSelector("fast", _choose_purpose("Settings"))
    reviewer = _SemanticSelector("reviewer", _choose_purpose("Settings"))

    decision = evaluate_selective_policy(
        _context(settings, ideas, goal="Open Settings, then open Theme."),
        [primary, reviewer],
        mode="advisory",
    )

    assert decision.status == "selected"
    assert decision.provider == "fast"
    assert decision.selected_candidate is settings
    assert primary.select_calls == 2
    assert reviewer.availability_calls == reviewer.select_calls == 0
    assert decision.selection_trace[0]["semantic_reason"] == "unique_direct_semantic_match"


def test_selective_hybrid_reviewer_overrides_unanimous_but_unrelated_primary() -> None:
    settings = _candidate(
        0,
        arguments={"rid": "openSettings"},
        model_arguments={"rid": "openSettings"},
        purpose="Tap the Settings control and observe the result.",
    )
    ideas = _candidate(
        1,
        arguments={"rid": "openIdeas"},
        model_arguments={"rid": "openIdeas"},
        purpose="Tap the Ideas control and observe the result.",
    )
    primary = _SemanticSelector("fast", _choose_purpose("Ideas"))
    reviewer = _SemanticSelector("reviewer", _choose_purpose("Settings"))

    decision = evaluate_selective_policy(
        _context(settings, ideas, goal="Open Settings, then open Theme."),
        [primary, reviewer],
        mode="advisory",
    )

    assert decision.status == "selected"
    assert decision.provider == "reviewer"
    assert decision.selected_candidate is settings
    assert primary.select_calls == 2
    assert reviewer.select_calls == 3
    assert decision.selection_trace[0]["status"] == "review_required"
    assert decision.selection_trace[0]["semantic_reason"] == (
        "selected_candidate_has_no_goal_overlap"
    )
    assert decision.selection_trace[1]["status"] == "selected"


def test_selective_hybrid_hands_off_when_reviewer_has_no_consensus() -> None:
    first = _candidate(
        0,
        arguments={"rid": "firstHistory"},
        model_arguments={"rid": "firstHistory"},
        purpose="Tap the first History control and observe the result.",
    )
    second = _candidate(
        1,
        arguments={"rid": "secondHistory"},
        model_arguments={"rid": "secondHistory"},
        purpose="Tap the second History control and observe the result.",
    )

    def _alternating() -> object:
        calls = 0

        def choose(context: PolicyContext) -> int:
            nonlocal calls
            calls += 1
            term = "first History" if calls % 2 else "second History"
            return _choose_purpose(term)(context)

        return choose

    # Neither local model can settle on one control across its independently permuted
    # reviews. Only then does control return to the parent agent.
    primary = _SemanticSelector("fast", _alternating())
    reviewer = _SemanticSelector("reviewer", _alternating())

    decision = evaluate_selective_policy(
        _context(first, second, goal="Open History."),
        [primary, reviewer],
        mode="advisory",
    )

    assert decision.status == "handoff"
    assert decision.selected_candidate is None
    assert decision.selection_trace[0]["status"] == "no_consensus"
    assert decision.selection_trace[-1]["status"] == "no_consensus"
    # Consensus is abandoned as soon as two reviews disagree, so neither selector
    # spends its full review budget here.
    assert primary.select_calls == 2
    assert reviewer.select_calls == 2
    assert "recommended_call" not in decision.as_json()


def test_selective_hybrid_refuses_a_final_reviewer_choice_with_no_goal_overlap() -> None:
    settings = _candidate(
        0,
        arguments={"rid": "openSettings"},
        model_arguments={"rid": "openSettings"},
        purpose="Tap the Settings control and observe the result.",
    )
    ideas = _candidate(
        1,
        arguments={"rid": "openIdeas"},
        model_arguments={"rid": "openIdeas"},
        purpose="Tap the Ideas control and observe the result.",
    )
    primary = _SemanticSelector("fast", _choose_purpose("Ideas"))
    reviewer = _SemanticSelector("reviewer", _choose_purpose("Ideas"))

    decision = evaluate_selective_policy(
        _context(settings, ideas, goal="Open Settings."),
        [primary, reviewer],
        mode="advisory",
    )

    # Observed live: given a goal naming a destination the application does not contain, the
    # terminal reviewer unanimously selected an unrelated navigation control and the turn
    # executed the tap. Being the last authority permits a judgement call, not action on a
    # control that shares nothing with the goal.
    assert decision.status == "handoff"
    assert decision.selected_candidate is None
    assert decision.selection_trace[-1]["status"] == "rejected_semantic"
    assert decision.selection_trace[-1]["semantic_reason"] == (
        "selected_candidate_has_no_goal_overlap"
    )
    assert "recommended_call" not in decision.as_json()


def test_selective_hybrid_lets_the_primary_break_a_tie_without_the_reviewer() -> None:
    """A live screen may expose two controls that reach the same destination.

    A home screen commonly carries both a bottom tab and an empty-state card bearing the
    same destination label. Both are correct, so no term-overlap comparison can separate
    them. Breaking that tie is the judgement the model supplies, and the fast primary
    must be allowed to supply it alone.
    """

    tab = _candidate(
        0,
        arguments={"rid": "homeTabBROWSE"},
        model_arguments={"rid": "homeTabBROWSE"},
        purpose="Tap the Browse control and observe the result.",
    )
    card = _candidate(
        1,
        arguments={"rid": "emptyStateCardBrowse"},
        model_arguments={"rid": "emptyStateCardBrowse"},
        purpose="Tap the Browse card control and observe the result.",
    )
    primary = _SemanticSelector("fast", _choose_purpose("Browse control"))
    reviewer = _SemanticSelector("reviewer", _choose_purpose("Browse card"))

    decision = evaluate_selective_policy(
        _context(tab, card, goal="Open Browse, then open Catalog."),
        [primary, reviewer],
        mode="advisory",
    )

    assert decision.status == "selected"
    assert decision.provider == "fast"
    assert decision.selected_candidate is tab
    assert decision.selection_trace[0]["semantic_review_required"] is False
    assert decision.selection_trace[0]["semantic_reason"] == "tied_best_goal_overlap"
    assert reviewer.availability_calls == reviewer.select_calls == 0


def test_identifier_only_controls_still_expose_their_words_to_the_overlap_check() -> None:
    """Regression: a resource id is often the only place a control names itself.

    In the synthetic regression fixture, every candidate on a screen whose controls carry no
    visible text is scored ``selected_candidate_has_no_goal_overlap`` — including the correct one —
    because ``buttonSettings`` tokenised to the single opaque term ``buttonsettings``. With
    every candidate equally "unrelated", the comparison carried no information, and gating on
    it would have refused correct navigation on every such screen.
    """

    goal_terms = _semantic_terms("Settings")
    correct = _candidate(
        0,
        arguments={"rid": "buttonSettings"},
        model_arguments={"rid": "buttonSettings"},
        purpose="Tap the current-frame control and observe the result.",
    )
    unrelated = _candidate(
        1,
        arguments={"rid": "archiveTab"},
        model_arguments={"rid": "archiveTab"},
        purpose="Tap the current-frame control and observe the result.",
    )

    assert goal_terms & _candidate_semantic_terms(correct) == {"settings"}
    assert goal_terms & _candidate_semantic_terms(unrelated) == set()

    # The whole token survives alongside its parts, so single-word labels are unaffected.
    assert {"settings", "button", "buttonsettings"} <= _candidate_semantic_terms(correct)


def test_a_goal_naming_an_absent_destination_never_executes_a_tap() -> None:
    """Regression for the live absent-target failure.

    Goal named a destination the application does not contain. FunctionGemma failed
    consensus, Gemma 4 unanimously chose an unrelated navigation tab, and because it was the
    last configured reviewer the turn executed the tap and then reported the waypoint as
    completed. Both reviewers must now decline.
    """

    services = _candidate(
        0,
        arguments={"rid": "archiveTab"},
        model_arguments={"rid": "archiveTab"},
        purpose="Tap the current-frame control and observe the result.",
    )
    chats = _candidate(
        1,
        arguments={"rid": "dashboardTab"},
        model_arguments={"rid": "dashboardTab"},
        purpose="Tap the current-frame control and observe the result.",
    )
    primary = _SemanticSelector("fast", _choose_purpose("Tap the current-frame"))
    reviewer = _SemanticSelector("reviewer", _choose_purpose("Tap the current-frame"))

    decision = evaluate_selective_policy(
        _context(services, chats, goal="Bookkeeping Ledger"),
        [primary, reviewer],
        mode="advisory",
    )

    assert decision.status == "handoff"
    assert decision.selected_candidate is None
    assert "recommended_call" not in decision.as_json()


def test_handoff_protocol_is_explicit_and_fails_closed_when_not_authenticated() -> None:
    candidates = (_candidate(0), _candidate(1))
    allowed_context = _context(*candidates, allow_handoff=True)
    compiled = compile_policy_context(allowed_context, candidates)

    assert compiled["handoff"] == {
        "allowed": True,
        "candidate_id": POLICY_HANDOFF_ID,
        "reason": "no_supplied_candidate_advances_goal",
    }
    assert "candidate ID -1" in policy_messages(allowed_context)[0]["content"]
    assert "or select -1" in policy_tools(allow_handoff=True)[0]["function"]["description"]

    handoff = evaluate_policy(
        allowed_context,
        _Selector(POLICY_HANDOFF_ID),
        mode="advisory",
    )
    assert handoff.status == "handoff"
    assert handoff.model_used is True
    assert handoff.selected_candidate is None
    assert handoff.as_json()["handoff_reason"] == "no_supplied_candidate_advances_goal"
    assert "recommended_call" not in handoff.as_json()

    rejected = evaluate_policy(
        _context(*candidates),
        _Selector(POLICY_HANDOFF_ID),
        mode="advisory",
    )
    assert rejected.status == "invalid_selection"
    assert "without an authenticated handoff protocol" in str(rejected.error)


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


# The manifest is narrowed deliberately rather than relying on the bundled one. These counts were
# 2 and 3 against the bundled four-only v3 adapter, but v10 authenticates 2, 3 and 4 — and since the
# guard trims to at most four and a lone candidate is decided deterministically, no unsupported
# cardinality is reachable through the bundle at all. Asserting the rule therefore needs an adapter
# whose manifest genuinely authenticates less than the caller offers.
@pytest.mark.parametrize("count", [2, 3])
def test_functiongemma_rejects_candidate_counts_absent_from_training(
    tmp_path: Path,
    count: int,
) -> None:
    model, adapter = _artifacts(tmp_path)
    loads: list[str] = []
    selector = FunctionGemmaPolicySelector(
        _explicit_advisory_settings(model, adapter, candidate_counts=(4,)),
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


def test_authenticated_adapter_accepts_only_manifest_bound_handoff(tmp_path: Path) -> None:
    model, adapter = _artifacts(tmp_path)
    settings = _explicit_advisory_settings(
        model,
        adapter,
        candidate_counts=(2, 3, 4),
        handoff=True,
    )
    tokenizer = _Tokenizer()
    selector = FunctionGemmaPolicySelector(
        settings,
        model_loader=lambda *args, **kwargs: (object(), tokenizer),
        generator=lambda *args, **kwargs: (
            "<start_function_call>call:select_candidate{candidate_id:-1}"
        ),
        sampler_factory=lambda **kwargs: kwargs,
    )
    context = _context(_candidate(0), _candidate(1), allow_handoff=True)

    assert selector.supports_handoff() is True
    assert selector.status()["supports_handoff"] is True
    assert selector.select(context) == POLICY_HANDOFF_ID
    assert tokenizer.tools is not None
    assert "or select -1" in tokenizer.tools[0]["function"]["description"]

    legacy = FunctionGemmaPolicySelector(
        _explicit_advisory_settings(model, adapter, candidate_counts=(2, 3, 4))
    )
    assert legacy.supports_handoff() is False


@pytest.mark.parametrize("count", [2, 3])
def test_core_skips_functiongemma_availability_for_unsupported_cardinality(
    tmp_path: Path,
    count: int,
) -> None:
    model, adapter = _artifacts(tmp_path)
    selector = FunctionGemmaPolicySelector(
        _explicit_advisory_settings(model, adapter, candidate_counts=(4,)),
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


def test_authenticated_explicit_adapter_accepts_stale_training_host_model_path(
    tmp_path: Path,
) -> None:
    model, adapter = _artifacts(tmp_path)
    stale_model = tmp_path / "deleted-runpod-snapshot"
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "model": str(stale_model)}),
        encoding="utf-8",
    )
    settings = _explicit_advisory_settings(model, adapter)

    provenance = validate_local_artifacts(settings)

    assert provenance["model_hash_verified"] is True
    assert provenance["adapter_declared_model_path"] == str(stale_model)
    assert provenance["adapter_declared_model_path_exists"] is False

    unpinned = _settings(model, adapter)
    with pytest.raises(ValueError, match="existing absolute local path"):
        validate_local_artifacts(unpinned)


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
