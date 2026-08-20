from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.history_miner import (  # noqa: E402
    SCHEMA,
    build_history_corpus,
    write_history_corpus,
)

PRIVATE = "Private Product Destination"
PRIVATE_PACKAGE = "com.private.product"
PRIVATE_RID = "com.private.product:id/private_destination"


def _phase(*, status: str = "completed", source: str = "observation") -> dict:
    return {
        "id": "phase-secret-id",
        "objective": f"Open {PRIVATE} and prove its private toolbar.",
        "kind": "verify",
        "status": status,
        "requirements": [{"subject": PRIVATE, "expected": "present", "terms": [PRIVATE]}],
        "proof": {
            "source": source,
            "verified": True,
            "evidence": f"Observed {PRIVATE} in {PRIVATE_PACKAGE}",
        },
        "assertions": [{"text": PRIVATE, "resource_id": PRIVATE_RID}],
    }


def _session(*, serial: str = "emulator-5554", phase: dict | None = None) -> dict:
    return {
        "session_id": "private-session-id",
        "serial": serial,
        "owner": "private-owner",
        "goal": f"Test {PRIVATE}",
        "finished_ms": 1234,
        "phases": [phase or _phase()],
    }


def _events() -> list[dict]:
    return [
        {
            "ts_ms": 100,
            "source": "daemon",
            "cmd": "session_progress",
            "ok": True,
            "serial": "emulator-5554",
            "session_id": "private-session-id",
            "invocation_id": "invocation-one",
            "args": {"goal": f"Test {PRIVATE}", "package": PRIVATE_PACKAGE},
            "result": {
                "goal_progress": {
                    "completed": 0,
                    "total": 1,
                    "current": {"id": "phase-secret-id", "objective": PRIVATE},
                    "policy": {
                        "mode": "advisory",
                        "status": "selected",
                        "provider": "functiongemma",
                        "model_used": True,
                        "candidate_count": 2,
                        "eligible_candidate_ids": [0, 1],
                        "selected_candidate_id": 1,
                        "compiler": {
                            "target_term_count": 3,
                            "stages": {"offered": 2},
                            "recommended_call_offered": True,
                        },
                    },
                    "policy_suggestion": {
                        "candidate_id": 1,
                        "mcp": {
                            "tool": "tap_and_analyze",
                            "arguments": {"rid": PRIVATE_RID},
                        },
                        "cli": f"aua tap-and-analyze --rid {PRIVATE_RID}",
                    },
                }
            },
        },
        {
            "ts_ms": 200,
            "source": "daemon",
            "cmd": "tap",
            "ok": True,
            "serial": "emulator-5554",
            "session_id": "private-session-id",
            "invocation_id": "invocation-two",
            "args": {
                "selector": {"rid": PRIVATE_RID, "text": PRIVATE},
                "package": PRIVATE_PACKAGE,
            },
            "result": {
                "observation": {"elements_count": 12, "meta": PRIVATE},
                "goal_progress": {"completed": 1, "total": 1, "done": True},
            },
        },
        {
            "ts_ms": 300,
            "source": "daemon",
            "cmd": "session_finish",
            "ok": True,
            "serial": "emulator-5554",
            "session_id": "private-session-id",
            "invocation_id": "invocation-three",
            "result": {"finished": True},
        },
    ]


def test_history_corpus_joins_structured_success_without_source_copy() -> None:
    episodes, decisions, seeds, summary = build_history_corpus(
        _events(), {"private-session-id": _session()}
    )

    assert len(episodes) == 1
    assert episodes[0]["outcome"] == "completed"
    assert episodes[0]["all_completed_phases_structured"] is True
    assert decisions[0]["classification"] == "selected_followed_structured_progress"
    assert decisions[0]["training_use"] == "positive_template"
    assert {row["family"] for row in seeds} >= {
        "structured_sequence_success",
        "selected_followed_structured_progress",
    }
    assert summary["native_training_rows"] == 0
    assert summary["training_status"] == "requires_fictionalization"

    payload = json.dumps([episodes, decisions, seeds, summary], sort_keys=True)
    for private in (PRIVATE, PRIVATE_PACKAGE, PRIVATE_RID, "private-owner", "phase-secret-id"):
        assert private not in payload


def test_manual_proof_is_never_promoted_to_structured_training_truth() -> None:
    manual = _session(phase=_phase(source="manual_evidence"))
    episodes, decisions, seeds, _summary = build_history_corpus(
        _events(), {"private-session-id": manual}
    )

    assert episodes[0]["all_completed_phases_structured"] is False
    assert decisions[0]["classification"] == "selected_followed_unproven"
    assert all(row["family"] != "structured_sequence_success" for row in seeds)
    assert all(row["training_use"] != "positive_template" for row in seeds)


def test_physical_sessions_are_excluded_even_when_their_events_exist() -> None:
    episodes, decisions, seeds, summary = build_history_corpus(
        _events(), {"private-session-id": _session(serial="physical-private-device")}
    )

    assert episodes == []
    assert decisions == []
    assert seeds == []
    assert summary["sessions_excluded"] == {"non_emulator": 1}


def test_unknown_commands_and_source_values_cannot_enter_output() -> None:
    events = _events()
    events[1]["cmd"] = PRIVATE
    events[1]["source"] = PRIVATE_PACKAGE
    episodes, _decisions, _seeds, _summary = build_history_corpus(
        events, {"private-session-id": _session()}
    )

    assert episodes[0]["events"][1]["command"] == "other"
    assert episodes[0]["events"][1]["source"] == "other"


def test_writer_is_deterministic_private_safe_and_refuses_accidental_overwrite(
    tmp_path: Path,
) -> None:
    journal_dir = tmp_path / "journal"
    session_dir = tmp_path / "sessions" / "emulator-private"
    output = tmp_path / "output"
    journal_dir.mkdir()
    session_dir.mkdir(parents=True)
    (journal_dir / "private.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _events()), encoding="utf-8"
    )
    (session_dir / "private.json").write_text(json.dumps(_session()), encoding="utf-8")

    first = write_history_corpus(
        journal_dir=journal_dir,
        session_dir=session_dir.parent,
        output_dir=output,
    )
    with pytest.raises(FileExistsError):
        write_history_corpus(
            journal_dir=journal_dir,
            session_dir=session_dir.parent,
            output_dir=output,
        )
    second = write_history_corpus(
        journal_dir=journal_dir,
        session_dir=session_dir.parent,
        output_dir=output,
        overwrite=True,
    )

    assert first == second
    assert first["schema"] == SCHEMA
    assert first["privacy"] == {
        "packages_allowed": False,
        "physical_sessions_allowed": False,
        "raw_source_values_emitted": 0,
        "selector_values_allowed": False,
        "source_copy_allowed": False,
        "violations": 0,
    }
    combined = "".join(path.read_text() for path in output.iterdir() if path.is_file())
    for private in (PRIVATE, PRIVATE_PACKAGE, PRIVATE_RID, "private-owner", "emulator-private"):
        assert private not in combined
