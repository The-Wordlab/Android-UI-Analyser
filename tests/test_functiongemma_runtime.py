from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.closed_loop import Candidate, DecisionContext
from experiments.functiongemma.curriculum import SELECT_CANDIDATE_TOOL
from experiments.functiongemma.runtime import (
    ACTIVATION_PREFIX,
    INVALID_CANDIDATE_ID,
    FunctionGemmaChooser,
    SelectionProtocolError,
    parse_candidate_id,
    policy_messages,
    policy_tools,
    serialize_context,
)


def _context() -> DecisionContext:
    return DecisionContext(
        goal="Open the fictional record and restore state.",
        phase="recover_unknown",
        state={
            "session_active": True,
            "observed_screen": "fixture_home",
            "network": "offline",
            "outcome": "unknown",
            "cleanup_required": True,
            "goal_checkpoint_reached": False,
        },
        candidates=(
            Candidate(
                id=3,
                call={"tool": "analyze_screen", "arguments": {"source": "auto"}},
                purpose="Observe before retrying.",
                risk="safe",
                authorized=True,
                redundant=False,
                proof="Fresh observation.",
            ),
            Candidate(
                id=1,
                call={"tool": "tap_and_analyze", "arguments": {"id": 44}},
                purpose="Repeat an unknown mutation.",
                risk="unsafe",
                authorized=False,
                redundant=True,
                proof="No disambiguating evidence.",
            ),
        ),
    )


class _Tokenizer:
    has_chat_template = True
    tool_call_start = "<start_function_call>"
    tool_call_end = "<end_function_call>"

    def __init__(self) -> None:
        self.eos: list[str] = []
        self.prompt_args: dict[str, Any] | None = None

    def add_eos_token(self, value: str) -> None:
        self.eos.append(value)

    @staticmethod
    def tool_parser(output: str, _tools: list[dict[str, Any]]) -> dict[str, Any]:
        marker = "candidate_id:"
        candidate_id = int(output.split(marker, 1)[1].split("}", 1)[0])
        return {"name": "select_candidate", "arguments": {"candidate_id": candidate_id}}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.prompt_args = {"messages": messages, **kwargs}
        return [10, 20, 30]


def test_context_serialization_uses_training_vocabulary_without_oracle_labels() -> None:
    state = serialize_context(_context())

    assert list(state) == [
        "fixture_ref",
        "request",
        "goal",
        "phase",
        "observation",
        "recent_outcomes",
        "constraints",
        "candidates",
    ]
    assert state["goal"] == "Open the fictional record and restore state."
    assert state["phase"] == "recover_unknown"
    assert state["observation"] == {"fresh": False, "known_screen": "fixture_home"}
    assert "Observe before replaying" in state["constraints"][-1]
    assert state["candidates"][0]["id"] == 3
    assert state["candidates"][0]["call"]["tool"] == "analyze_screen"
    assert "correct" not in json.dumps(state)

    messages = policy_messages(_context())
    assert messages[0]["content"].startswith(ACTIVATION_PREFIX)
    assert json.loads(messages[1]["content"]) == state
    assert policy_tools() == [SELECT_CANDIDATE_TOOL]
    assert policy_tools()[0] is not SELECT_CANDIDATE_TOOL


def test_strict_parser_accepts_one_call_and_rejects_trailing_or_multiple_calls() -> None:
    tokenizer = _Tokenizer()
    tools = policy_tools()
    exact = "<start_function_call>call:select_candidate{candidate_id:3}<end_function_call>"

    assert parse_candidate_id(exact, tokenizer, tools) == 3

    invalid = (exact + " trailing", exact + exact, "select_candidate{candidate_id:3}")
    for output in invalid:
        try:
            parse_candidate_id(output, tokenizer, tools)
        except SelectionProtocolError:
            pass
        else:
            raise AssertionError(f"strict parser accepted invalid output: {output!r}")


def test_chooser_loads_lazily_adds_tool_eos_and_generates_greedily() -> None:
    tokenizer = _Tokenizer()
    calls: dict[str, Any] = {"loads": 0, "generations": 0}

    def loader(model: str, *, adapter_path: str | None) -> tuple[object, _Tokenizer]:
        calls["loads"] += 1
        calls["load_args"] = (model, adapter_path)
        return object(), tokenizer

    def sampler_factory(*, temp: float) -> dict[str, float]:
        calls["temperature"] = temp
        return {"temperature": temp}

    def generator(
        _model: object,
        _tokenizer: _Tokenizer,
        prompt: list[int],
        **kwargs: Any,
    ) -> str:
        calls["generations"] += 1
        calls["prompt"] = prompt
        calls["generation_args"] = kwargs
        return "<start_function_call>call:select_candidate{candidate_id:3}"

    chooser = FunctionGemmaChooser(
        "fictional-model",
        model_loader=loader,
        generator=generator,
        sampler_factory=sampler_factory,
    )
    assert calls["loads"] == 0

    assert chooser(_context()) == 3
    assert chooser(_context()) == 3

    assert calls["loads"] == 1
    assert calls["generations"] == 2
    assert calls["temperature"] == 0.0
    assert tokenizer.eos == ["<end_function_call>"]
    assert calls["prompt"] == [10, 20, 30]
    assert calls["generation_args"]["verbose"] is False
    assert tokenizer.prompt_args is not None
    assert tokenizer.prompt_args["add_generation_prompt"] is True
    assert tokenizer.prompt_args["tools"] == [SELECT_CANDIDATE_TOOL]


def test_invalid_protocol_becomes_an_invalid_simulator_selection() -> None:
    tokenizer = _Tokenizer()

    chooser = FunctionGemmaChooser(
        "fictional-model",
        model_loader=lambda *_args, **_kwargs: (object(), tokenizer),
        generator=lambda *_args, **_kwargs: "not a tool call",
        sampler_factory=lambda **_kwargs: object(),
    )

    assert chooser(_context()) == INVALID_CANDIDATE_ID
    assert chooser.decisions[-1].valid_protocol is False
    assert chooser.decisions[-1].candidate_exists is False
