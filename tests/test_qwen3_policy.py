"""Contract tests for the optional local Qwen3 policy selector."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser.policy import PolicyCandidate, PolicyContext
from android_ui_analyser.providers.policy.qwen3 import Qwen3PolicySelector, parse_tool_call


class _Tokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.tools: Any = None

    def apply_chat_template(self, messages, tools=None, **_: Any) -> str:
        self.messages = list(messages)
        self.tools = tools
        return "PROMPT"


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "qwen3"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fictional weights")
    return model


def _adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"fictional lora")
    return adapter


def _candidate(candidate_id: int, rid: str) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=candidate_id,
        call={"tool": "tap_and_analyze", "arguments": {"rid": rid}},
        model_arguments={"rid": rid},
        purpose=f"Open the {rid} control.",
        proof="A folded observation follows.",
        phase="verify",
        # The guard only accepts candidates it can prove were compiled from this frame.
        session_id="fixture-session",
        observation_fingerprint="fixture-frame",
        package="com.example.fixture",
    )


def _context(*candidates: PolicyCandidate, allow_handoff: bool = True) -> PolicyContext:
    return PolicyContext(
        goal="Open Registry and prove the page.",
        phase="verify",
        candidates=candidates,
        observation={"fresh": True},
        constraints=("read_only",),
        session_id="fixture-session",
        observation_fingerprint="fixture-frame",
        package="com.example.fixture",
        allow_handoff=allow_handoff,
    )


def test_parser_requires_one_exact_tool_call() -> None:
    assert (
        parse_tool_call(
            '<tool_call>{"name":"select_candidate","arguments":{"candidate_id":2}}</tool_call>'
        )
        == 2
    )
    # A thinking block before the verdict is part of the envelope, not prose.
    assert (
        parse_tool_call(
            '<think>weighing</think><tool_call>{"name":"select_candidate",'
            '"arguments":{"candidate_id":-1}}</tool_call>'
        )
        == -1
    )
    for invalid in (
        "candidate 2",
        '<tool_call>{"name":"other","arguments":{"candidate_id":1}}</tool_call>',
        '<tool_call>{"name":"select_candidate","arguments":{"candidate_id":"1"}}</tool_call>',
        '<tool_call>{"name":"select_candidate","arguments":{"candidate_id":1}}</tool_call> trailing',
    ):
        # Malformed JSON raises ValueError via json, everything else via our checks.
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_tool_call(invalid)


def test_activation_is_rendered_on_a_system_turn(tmp_path) -> None:
    """Qwen3's template has no developer role; training used system, so inference must too."""

    tokenizer = _Tokenizer()

    def load(path: str, **_: Any) -> tuple[object, _Tokenizer]:
        return object(), tokenizer

    def generate(*_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(
            text='<tool_call>{"name":"select_candidate","arguments":{"candidate_id":1}}</tool_call>'
        )

    selector = Qwen3PolicySelector(
        {
            "model_path": str(_model_dir(tmp_path)),
            "adapter_path": str(_adapter_dir(tmp_path)),
            "max_tokens": 48,
            "max_mode": "advisory",
        },
        model_loader=load,
        generator=generate,
    )
    assert (
        selector.select(_context(_candidate(0, "openRegistry"), _candidate(1, "openArchive"))) == 1
    )

    roles = [message["role"] for message in tokenizer.messages]
    assert roles == ["system", "user"], roles
    assert "developer" not in roles
    # The activation text itself is unchanged — only the turn it rides on.
    assert tokenizer.messages[0]["content"].startswith("You are a model that can do function")
    # Exactly one function is declared, as for every other provider.
    assert isinstance(tokenizer.tools, list) and len(tokenizer.tools) == 1
    assert tokenizer.tools[0]["function"]["name"] == "select_candidate"


def test_an_id_outside_the_guarded_set_is_refused(tmp_path) -> None:
    def load(path: str, **_: Any) -> tuple[object, _Tokenizer]:
        return object(), _Tokenizer()

    def generate(*_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(
            text='<tool_call>{"name":"select_candidate","arguments":{"candidate_id":7}}</tool_call>'
        )

    selector = Qwen3PolicySelector(
        {"model_path": str(_model_dir(tmp_path)), "max_tokens": 48, "max_mode": "advisory"},
        model_loader=load,
        generator=generate,
    )
    assert selector.select(_context(_candidate(0, "a"), _candidate(1, "b"))) is None
    assert selector.last_selection["parsed"] is False


def test_advisory_requires_an_explicit_operator_choice(tmp_path) -> None:
    settings = {"model_path": str(_model_dir(tmp_path)), "max_tokens": 48}
    shadow_only = Qwen3PolicySelector(settings)
    assert shadow_only.supports_mode("shadow") is True
    assert shadow_only.supports_mode("advisory") is False

    advisory = Qwen3PolicySelector({**settings, "max_mode": "advisory"})
    assert advisory.supports_mode("advisory") is True


def test_status_never_reveals_model_or_candidate_text(tmp_path) -> None:
    selector = Qwen3PolicySelector(
        {"model_path": str(_model_dir(tmp_path)), "max_tokens": 48, "max_mode": "advisory"}
    )
    status = selector.status()
    assert status["provider"] == "qwen3"
    assert status["activation_role"] == "system"
    assert status["supported_candidate_counts"] == [2, 3, 4]
    assert status["supports_handoff"] is True
    blob = json.dumps(status)
    assert "Open the" not in blob
    assert "Registry" not in blob
