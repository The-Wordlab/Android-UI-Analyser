from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma import evaluate as common_evaluate
from experiments.functiongemma import evaluate_qwen
from experiments.functiongemma.run_qwen_semantic_context_smoke import _rows


def _row() -> dict[str, object]:
    return {
        "messages": [
            {"role": "developer", "content": "selector"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "id": 0,
                                "call": {"tool": "analyze_screen", "arguments": {}},
                                "authorized": True,
                                "redundant": False,
                            },
                            {
                                "id": 1,
                                "call": {"tool": "tap_and_analyze", "arguments": {}},
                                "authorized": True,
                                "redundant": False,
                            },
                        ]
                    }
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "select_candidate",
                            "arguments": {"candidate_id": 1},
                        },
                    }
                ],
            },
        ],
        "tools": [],
        "metadata": {"case_id": "qwen-fixture", "family": "fixture"},
    }


def test_qwen_parser_accepts_one_exact_native_tool_call() -> None:
    output = '<tool_call>\n{"name":"select_candidate","arguments":{"candidate_id":1}}\n</tool_call>'
    prediction = evaluate_qwen._parse_prediction(_row(), output)

    assert prediction.correct is True
    assert prediction.parsed is True
    assert prediction.predicted_candidate_id == 1


def test_qwen_parser_rejects_trailing_or_multiple_calls() -> None:
    exact = '<tool_call>\n{"name":"select_candidate","arguments":{"candidate_id":1}}\n</tool_call>'

    assert evaluate_qwen._parse_prediction(_row(), exact + " trailing").parsed is False
    assert evaluate_qwen._parse_prediction(_row(), exact + exact).parsed is False


def test_qwen_generation_boundary_restores_only_the_configured_eos_token() -> None:
    stopped = '<tool_call>\n{"name":"select_candidate","arguments":{"candidate_id":1}}\n'

    assert evaluate_qwen._parse_prediction(_row(), stopped).parsed is False
    prediction, restored = evaluate_qwen._parse_generated_prediction(_row(), stopped)

    assert restored is True
    assert prediction.correct is True
    assert prediction.raw_output == stopped


def test_qwen_generation_boundary_does_not_repair_trailing_or_multiple_output() -> None:
    stopped = '<tool_call>\n{"name":"select_candidate","arguments":{"candidate_id":1}}\n'
    exact = f"{stopped}</tool_call>"

    trailing, trailing_restored = evaluate_qwen._parse_generated_prediction(
        _row(), stopped + "trailing"
    )
    multiple, multiple_restored = evaluate_qwen._parse_generated_prediction(_row(), exact + exact)

    assert trailing.parsed is False
    assert trailing_restored is False
    assert multiple.parsed is False
    assert multiple_restored is False


def test_adapter_validation_accepts_same_huggingface_snapshot_on_new_pod(
    tmp_path: Path,
) -> None:
    revision = "bc82a1060abf25e90be9782b12c00fa55d9bf542"
    old_model = f"/workspace/old/hf-cache/models--Qwen--Qwen3-0.6B-MLX-bf16/snapshots/{revision}"
    new_model = tmp_path / "hf-cache" / "models--Qwen--Qwen3-0.6B-MLX-bf16" / "snapshots" / revision
    new_model.mkdir(parents=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"model": old_model, "fine_tune_type": "lora"}),
        encoding="utf-8",
    )
    (adapter / "adapters.safetensors").write_bytes(b"weights")

    result = common_evaluate._validate_adapter(str(new_model), adapter)  # noqa: SLF001

    assert result is not None
    assert result["weights_bytes"] == 7


def test_adapter_validation_rejects_different_huggingface_snapshot(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "model": "/cache/models--Qwen--Qwen3-0.6B-MLX-bf16/snapshots/revision-a",
                "fine_tune_type": "lora",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapters.safetensors").write_bytes(b"weights")
    requested = tmp_path / "models--Qwen--Qwen3-0.6B-MLX-bf16" / "snapshots" / "revision-b"
    requested.mkdir(parents=True)

    with pytest.raises(ValueError, match="adapter base model mismatch"):
        common_evaluate._validate_adapter(str(requested), adapter)  # noqa: SLF001


def test_qwen_smoke_is_the_full_balanced_99_case_matrix() -> None:
    rows = _rows()
    cardinalities = [
        len(json.loads(row["messages"][1]["content"])["candidates"])  # type: ignore[index]
        for row in rows
    ]

    assert len(rows) == 99
    assert {value: cardinalities.count(value) for value in {2, 3, 4}} == {
        2: 8,
        3: 27,
        4: 64,
    }
