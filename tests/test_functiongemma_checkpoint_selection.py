from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.select_checkpoint import checkpoint_score, select_checkpoint


def _report(
    *,
    accuracy: float,
    weak_family: float,
    unauthorized: int = 0,
    redundant: int = 0,
    parse: float = 1.0,
    permutation_accuracy: float | None = None,
) -> dict:
    metrics = {
        "parse_success": parse,
        "exactly_one_call": parse,
        "candidate_exists": 1.0,
        "candidate_accuracy": accuracy,
        "critical_accuracy": accuracy,
        "unauthorized_selections": unauthorized,
        "redundant_selections": redundant,
        "by_family": {
            "ordinary": {"accuracy": accuracy},
            "recover_unknown": {"accuracy": weak_family},
        },
    }
    if permutation_accuracy is not None:
        metrics["permutation_groups"] = {
            "declared_groups": 1,
            "well_formed_groups": 1,
            "group_accuracy": permutation_accuracy,
        }
    return {"metrics": metrics}


def test_checkpoint_score_is_fail_closed_and_reports_worst_family() -> None:
    score = checkpoint_score(_report(accuracy=0.999, weak_family=0.97, unauthorized=1))

    assert score["strict_safety_passed"] is False
    assert score["worst_family_accuracy"] == 0.97
    assert score["worst_families"] == ["recover_unknown"]


def test_selection_rejects_safer_looking_aggregate_with_one_safety_error(tmp_path: Path) -> None:
    unsafe = tmp_path / "0000256_adapters.safetensors"
    safe = tmp_path / "0000512_adapters.safetensors"
    unsafe.write_bytes(b"unsafe")
    safe.write_bytes(b"safe")

    result = select_checkpoint(
        [
            (unsafe, _report(accuracy=1.0, weak_family=1.0, unauthorized=1)),
            (safe, _report(accuracy=0.99, weak_family=0.98)),
        ]
    )

    assert result["eligible_checkpoints"] == 1
    assert result["selected"]["checkpoint"] == str(safe)


def test_selection_prefers_worst_family_then_critical_and_earlier_checkpoint(
    tmp_path: Path,
) -> None:
    first = tmp_path / "0000256_adapters.safetensors"
    second = tmp_path / "0000512_adapters.safetensors"
    third = tmp_path / "0000768_adapters.safetensors"
    for path in (first, second, third):
        path.write_bytes(path.name.encode())

    result = select_checkpoint(
        [
            (first, _report(accuracy=0.995, weak_family=0.98)),
            (second, _report(accuracy=0.999, weak_family=0.97)),
            (third, _report(accuracy=0.995, weak_family=0.98)),
        ]
    )

    assert result["selected"]["checkpoint"] == str(first)


def test_selection_rejects_any_incomplete_permutation_group(tmp_path: Path) -> None:
    almost = tmp_path / "0000256_adapters.safetensors"
    invariant = tmp_path / "0000512_adapters.safetensors"
    almost.write_bytes(b"almost")
    invariant.write_bytes(b"invariant")

    result = select_checkpoint(
        [
            (
                almost,
                _report(
                    accuracy=0.9999,
                    weak_family=1.0,
                    permutation_accuracy=0.99,
                ),
            ),
            (
                invariant,
                _report(accuracy=0.99, weak_family=0.99, permutation_accuracy=1.0),
            ),
        ]
    )

    assert result["eligible_checkpoints"] == 1
    assert result["selected"]["checkpoint"] == str(invariant)


def test_checkpoint_score_rejects_a_malformed_declared_permutation_group() -> None:
    report = _report(accuracy=1.0, weak_family=1.0, permutation_accuracy=1.0)
    report["metrics"]["permutation_groups"]["well_formed_groups"] = 0

    score = checkpoint_score(report)

    assert score["permutation_groups_well_formed"] is False
    assert score["strict_safety_passed"] is False
