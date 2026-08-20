from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma import run_live_context_smoke as smoke
from experiments.functiongemma.run_production_smoke import build_selector


class _Tokenizer:
    has_chat_template = True
    tool_call_start = "<start_function_call>"
    tool_call_end = "<end_function_call>"

    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []

    def add_eos_token(self, _value: str) -> None:
        return None

    @staticmethod
    def tool_parser(output: str, _tools: list[dict[str, Any]]) -> dict[str, Any]:
        match = re.fullmatch(
            r"<start_function_call>call:select_candidate\{candidate_id:(-?[0-9]+)\}"
            r"(?:<end_function_call>)?",
            output,
        )
        if match is None:
            raise ValueError("invalid fixture call")
        return {"name": "select_candidate", "arguments": {"candidate_id": int(match.group(1))}}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> list[int]:
        del tools
        self.messages.append(messages)
        return [len(self.messages)]


def _runtime(tmp_path: Path, generator: Any):
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "model": str(model)}), encoding="utf-8"
    )
    (adapter / "adapters.safetensors").write_bytes(b"adapter")
    tokenizer = _Tokenizer()
    selector, recorder = build_selector(
        model,
        adapter,
        model_loader=lambda *_args, **_kwargs: (object(), tokenizer),
        generator=generator,
        sampler_factory=lambda **kwargs: kwargs,
    )
    return selector, recorder, tokenizer


def _semantic_generator(_model: Any, tokenizer: _Tokenizer, _prompt: Any, **_kwargs: Any) -> str:
    state = json.loads(tokenizer.messages[-1][-1]["content"])
    target = state["goal"].split("Open ", 1)[1].split(" from ", 1)[0]
    selected = next(
        candidate
        for candidate in state["candidates"]
        if candidate["purpose"].startswith(f"Tap the current-frame '{target} ")
    )
    return (
        "<start_function_call>call:select_candidate"
        f"{{candidate_id:{selected['id']}}}<end_function_call>"
    )


def test_live_context_smoke_covers_every_target_order_and_balanced_id(tmp_path: Path) -> None:
    selector, recorder, tokenizer = _runtime(tmp_path, _semantic_generator)

    report = smoke.run_smoke(
        selector,
        recorder,
        model=str(tmp_path / "model"),
        adapter=str(tmp_path / "adapter"),
    )

    assert report["passed"] is True
    assert report["matrix"]["cases"] == 384
    assert report["matrix"]["candidate_orders"] == 24
    assert report["metrics"]["semantic_accuracy"] == 1.0
    assert {value["correct"] for value in report["metrics"]["per_target"].values()} == {96}
    assert len(tokenizer.messages) == 384


def test_live_context_smoke_rejects_an_opaque_id_shortcut(tmp_path: Path) -> None:
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

    assert report["metrics"]["semantic_accuracy"] == 0.25
    assert report["gates"]["semantic_accuracy_100_percent"] is False
    assert report["passed"] is False
