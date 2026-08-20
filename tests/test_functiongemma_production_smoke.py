from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma import run_production_smoke as smoke


class _Tokenizer:
    has_chat_template = True
    tool_call_start = "<start_function_call>"
    tool_call_end = "<end_function_call>"

    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]]] = []
        self.eos: list[str] = []

    def add_eos_token(self, value: str) -> None:
        self.eos.append(value)

    @staticmethod
    def tool_parser(output: str, _tools: list[dict[str, Any]]) -> dict[str, Any]:
        match = re.fullmatch(
            r"<start_function_call>call:select_candidate\{candidate_id:(-?[0-9]+)\}"
            r"(?:<end_function_call>)?",
            output,
        )
        if match is None:
            raise ValueError("invalid fixture call")
        return {
            "name": "select_candidate",
            "arguments": {"candidate_id": int(match.group(1))},
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> list[int]:
        self.messages.append(messages)
        self.tools.append(tools)
        return [len(self.messages)]


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fictional model")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "model": str(model)}),
        encoding="utf-8",
    )
    (adapter / "adapters.safetensors").write_bytes(b"fictional adapter")
    return model, adapter


def _runtime(
    tmp_path: Path,
    generate: Any,
) -> tuple[smoke.FunctionGemmaPolicySelector, smoke.RecordingGenerator, _Tokenizer]:
    model, adapter = _artifacts(tmp_path)
    tokenizer = _Tokenizer()
    selector, recorder = smoke.build_selector(
        model,
        adapter,
        model_loader=lambda *_args, **_kwargs: (object(), tokenizer),
        generator=generate,
        sampler_factory=lambda **kwargs: kwargs,
    )
    return selector, recorder, tokenizer


def _semantic_generator(
    _model: Any,
    tokenizer: _Tokenizer,
    _prompt: Any,
    **_kwargs: Any,
) -> str:
    state = json.loads(tokenizer.messages[-1][-1]["content"])
    target = next(
        candidate
        for candidate in state["candidates"]
        if candidate["call"]["arguments"]["rid"] == "openMathematics"
    )
    return (
        "<start_function_call>call:select_candidate"
        f"{{candidate_id:{target['id']}}}<end_function_call>"
    )


def test_balanced_matrix_uses_the_production_serializer_and_passes(tmp_path: Path) -> None:
    selector, recorder, tokenizer = _runtime(tmp_path, _semantic_generator)

    report = smoke.run_smoke(
        selector,
        recorder,
        model=str(tmp_path / "model"),
        adapter=str(tmp_path / "adapter"),
    )

    assert report["passed"] is True
    assert report["host_only"] is True
    assert report["matrix"]["cases"] == 96
    assert report["matrix"]["candidate_orders"] == 24
    assert len(report["matrix"]["dense_id_permutations"]) == 4
    assert report["metrics"]["protocol_parse_rate"] == 1.0
    assert report["metrics"]["offered_id_rate"] == 1.0
    assert report["metrics"]["semantic_accuracy"] == 1.0
    assert {value["cases"] for value in report["metrics"]["by_target_candidate_id"].values()} == {
        24
    }
    assert {value["cases"] for value in report["metrics"]["by_target_position"].values()} == {24}
    assert report["metrics"]["selected_candidate_id_counts"] == {
        "0": 24,
        "1": 24,
        "2": 24,
        "3": 24,
    }
    assert len(tokenizer.messages) == 96

    first_state = json.loads(tokenizer.messages[0][-1]["content"])
    assert first_state["phase"] == "phase_1"
    assert first_state["recent_outcomes"] == [
        "session_started=true",
        "fresh_observation=true",
        "goal_checkpoint_reached=false",
    ]
    assert len(first_state["candidates"]) == 4
    assert all(candidate["purpose"] for candidate in first_state["candidates"])
    assert all(candidate["proof"] for candidate in first_state["candidates"])
    assert {candidate["call"]["arguments"]["rid"] for candidate in first_state["candidates"]} == {
        "openGrammar",
        "openMathematics",
        "openHistory",
        "openPhysics",
    }


def test_semantic_gate_and_id_bias_catch_a_valid_but_id_fixed_model(tmp_path: Path) -> None:
    selector, recorder, _tokenizer = _runtime(
        tmp_path,
        lambda *_args, **_kwargs: (
            "<start_function_call>call:select_candidate{candidate_id:0}<end_function_call>"
        ),
    )

    report = smoke.run_smoke(
        selector,
        recorder,
        model=str(tmp_path / "model"),
        adapter=str(tmp_path / "adapter"),
    )

    assert report["metrics"]["protocol_parse_rate"] == 1.0
    assert report["metrics"]["offered_id_rate"] == 1.0
    assert report["metrics"]["semantic_accuracy"] == 0.25
    assert report["metrics"]["target_id_accuracy_gap"] == 1.0
    assert report["gates"]["semantic_accuracy_at_least_95_percent"] is False
    assert report["gates"]["no_meaningful_target_id_bias"] is False
    assert report["passed"] is False


def test_offered_id_gate_fails_and_cli_writes_json_with_nonzero_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector, recorder, _tokenizer = _runtime(
        tmp_path,
        lambda *_args, **_kwargs: (
            "<start_function_call>call:select_candidate{candidate_id:9}<end_function_call>"
        ),
    )
    monkeypatch.setattr(smoke, "build_selector", lambda *_args, **_kwargs: (selector, recorder))
    output = tmp_path / "report.json"

    status = smoke.main(
        [
            "--model",
            str(tmp_path / "model"),
            "--adapter",
            str(tmp_path / "adapter"),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert status == 1
    assert report["passed"] is False
    assert report["metrics"]["protocol_parse_rate"] == 1.0
    assert report["metrics"]["offered_id_rate"] == 0.0
    assert report["gates"]["offered_id_100_percent"] is False


def test_model_and_nonbundled_adapter_must_be_absolute_local_directories(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def loader(*_args: Any, **_kwargs: Any) -> tuple[object, _Tokenizer]:
        calls.append("loaded")
        return object(), _Tokenizer()

    with pytest.raises(ValueError, match="model must be an absolute local path"):
        smoke.build_selector(
            "google/functiongemma-270m-it",
            "bundled",
            model_loader=loader,
            generator=lambda *_args, **_kwargs: "",
            sampler_factory=lambda **kwargs: kwargs,
        )
    model, _adapter = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="adapter must be an absolute local path"):
        smoke.build_selector(
            model,
            "relative-adapter",
            model_loader=loader,
            generator=lambda *_args, **_kwargs: "",
            sampler_factory=lambda **kwargs: kwargs,
        )
    assert calls == []
