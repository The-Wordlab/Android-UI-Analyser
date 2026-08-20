from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser.policy import PolicyCandidate, PolicyContext
from android_ui_analyser.providers.policy.gemma4 import (
    Gemma4PolicySelector,
    parse_candidate_id,
    verdict_failure_kind,
)


class _Processor:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: Any) -> str:
        self.messages = messages
        return "fictional prompt"


def _candidate(candidate_id: int, label: str) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=candidate_id,
        call={"tool": "tap_and_analyze", "arguments": {"rid": f"open{label}"}},
        model_arguments={"rid": f"open{label}"},
        purpose=f"Tap the {label} control and observe the result.",
        proof="The exact call returns a folded observation.",
        session_id="fixture-session",
        phase="open_destination",
        observation_fingerprint="fixture-frame",
        package="com.example.fixture",
    )


def _context() -> PolicyContext:
    return PolicyContext(
        goal="Open Settings.",
        phase="open_destination",
        candidates=(_candidate(0, "Ideas"), _candidate(1, "Settings")),
        observation={"fresh": True},
        session_id="fixture-session",
        observation_fingerprint="fixture-frame",
        package="com.example.fixture",
        allow_handoff=True,
    )


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "gemma4"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fictional weights")
    return model


def test_gemma4_parser_requires_one_exact_verdict() -> None:
    assert parse_candidate_id("FINAL_CANDIDATE_ID=3") == 3
    assert (
        parse_candidate_id("<|channel>thought\ncareful reasoning<channel|>FINAL_CANDIDATE_ID=-1")
        == -1
    )
    for invalid in ("candidate 3", "FINAL_CANDIDATE_ID=3 trailing", "FINAL_CANDIDATE_ID=true"):
        with pytest.raises(ValueError):
            parse_candidate_id(invalid)


def test_gemma4_parser_accepts_any_reasoning_shape_before_the_verdict() -> None:
    """Reasoning format is the model's business; only the verdict is contractual.

    Anchoring on the thinking channel's exact delimiters discarded well-formed verdicts
    whenever the wrapper varied, which silently dropped the reviewer out of the chain.
    """

    assert parse_candidate_id("I compared each control.\nFINAL_CANDIDATE_ID=2") == 2
    assert parse_candidate_id("<|channel>thought\nunterminated reasoningFINAL_CANDIDATE_ID=0") == 0
    assert parse_candidate_id("FINAL_CANDIDATE_ID=1\n\n") == 1


def test_gemma4_parser_still_refuses_more_than_one_verdict() -> None:
    with pytest.raises(ValueError):
        parse_candidate_id("FINAL_CANDIDATE_ID=1 then FINAL_CANDIDATE_ID=2")
    assert verdict_failure_kind("FINAL_CANDIDATE_ID=1 then FINAL_CANDIDATE_ID=2") == (
        "multiple_verdict_markers"
    )
    assert verdict_failure_kind("no verdict at all") == "no_verdict_marker"
    assert verdict_failure_kind("FINAL_CANDIDATE_ID=3 trailing") == (
        "malformed_or_trailing_verdict"
    )


def test_gemma4_records_why_a_verdict_was_unusable(tmp_path: Path) -> None:
    """A reviewer that silently returns None is undiagnosable in production.

    The recorded diagnosis describes only the *shape* of the output — never model text,
    candidate labels, or resource ids.
    """

    processor = _Processor()

    def load(path: str) -> tuple[object, _Processor]:
        return object(), processor

    def generate(*_: Any, **__: Any) -> SimpleNamespace:
        # Reasoning that ran out of budget before it could commit to a verdict.
        return SimpleNamespace(text="weighing the controls", generation_tokens=128)

    selector = Gemma4PolicySelector(
        {
            "model_path": str(_model_dir(tmp_path)),
            "max_tokens": 128,
            "max_mode": "advisory",
        },
        model_loader=load,
        generator=generate,
    )

    assert selector.select(_context()) is None
    diagnosis = selector.last_selection
    assert diagnosis["parsed"] is False
    assert diagnosis["failure"] == "no_verdict_marker"
    assert diagnosis["truncated"] is True
    assert diagnosis["generation_tokens"] == 128
    assert diagnosis["verdict_markers"] == 0
    assert selector.status()["last_selection"]["failure"] == "no_verdict_marker"
    # No model or app text is retained anywhere in the diagnosis.
    assert "weighing" not in json.dumps(diagnosis)


def test_gemma4_provider_is_local_lazy_and_returns_only_an_offered_id(tmp_path: Path) -> None:
    processor = _Processor()
    load_calls: list[str] = []

    def load(path: str) -> tuple[object, _Processor]:
        load_calls.append(path)
        return object(), processor

    def generate(*_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(text="FINAL_CANDIDATE_ID=1", generation_tokens=7)

    selector = Gemma4PolicySelector(
        {
            "model_path": str(_model_dir(tmp_path)),
            "max_tokens": 128,
            "max_mode": "advisory",
        },
        model_loader=load,
        generator=generate,
    )

    assert selector.is_available().ok is True
    assert selector.select(_context()) == 1
    assert selector.select(_context()) == 1
    assert len(load_calls) == 1
    assert selector.last_selection == {
        "parsed": True,
        "selected_id": 1,
        "generation_tokens": 7,
    }
    rendered = str(processor.messages)
    assert "Settings" in rendered
    assert "fixture-session" not in rendered
    assert "fixture-frame" not in rendered


def test_gemma4_provider_fails_closed_on_offered_id_violation(tmp_path: Path) -> None:
    selector = Gemma4PolicySelector(
        {
            "model_path": str(_model_dir(tmp_path)),
            "max_mode": "advisory",
        },
        model_loader=lambda _path: (object(), _Processor()),
        generator=lambda *_args, **_kwargs: SimpleNamespace(
            text="FINAL_CANDIDATE_ID=99",
            generation_tokens=3,
        ),
    )

    assert selector.select(_context()) is None
    assert "outside the guarded candidates" in str(selector.last_error)


def test_gemma4_advisory_requires_explicit_local_opt_in(tmp_path: Path) -> None:
    selector = Gemma4PolicySelector(
        {"model_path": str(_model_dir(tmp_path))},
        model_loader=lambda _path: (object(), _Processor()),
        generator=lambda *_args, **_kwargs: SimpleNamespace(text="FINAL_CANDIDATE_ID=0"),
    )

    assert selector.supports_mode("shadow") is True
    assert selector.supports_mode("advisory") is False
