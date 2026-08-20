"""Contract tests for the opt-in local policy training trace."""

from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser import policy_trace
from android_ui_analyser.policy import (
    PolicyCandidate,
    PolicyContext,
    evaluate_selective_policy,
)


class _Selector:
    name = "fixture"

    def __init__(self, choose) -> None:
        self._choose = choose
        self.select_calls = 0

    def is_available(self):
        from android_ui_analyser.providers.base import Availability

        return Availability(True, "fixture")

    def supports_candidate_count(self, count: int) -> bool:
        return 2 <= count <= 4

    def supports_handoff(self) -> bool:
        return True

    def supports_mode(self, mode: str) -> bool:
        return True

    def select(self, context: PolicyContext) -> int:
        self.select_calls += 1
        return self._choose(context)


def _candidate(candidate_id: int, rid: str, purpose: str) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=candidate_id,
        call={"tool": "tap_and_analyze", "arguments": {"rid": rid, "secret_path": "/private/x"}},
        model_arguments={"rid": rid},
        purpose=purpose,
        proof="A folded observation follows.",
        session_id="session-fixture",
        phase="verify",
        observation_fingerprint="frame-fixture",
        package="com.example.app",
    )


def _context(*candidates: PolicyCandidate) -> PolicyContext:
    return PolicyContext(
        goal="Open Registry and prove the page.",
        phase="verify",
        candidates=candidates,
        observation={"fresh": True},
        constraints=("read_only",),
        session_id="session-fixture",
        observation_fingerprint="frame-fixture",
        package="com.example.app",
        allow_handoff=True,
    )


def _records(directory: Path) -> list[dict]:
    path = directory / "decisions.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_tracing_is_off_unless_the_environment_names_a_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(policy_trace.ENV_VAR, raising=False)
    assert policy_trace.enabled() is False
    assert policy_trace.trace_directory() is None

    first = _candidate(0, "openRegistry", "Open the Registry row.")
    second = _candidate(1, "openArchive", "Open the Archive row.")
    evaluate_selective_policy(
        _context(first, second),
        [_Selector(lambda _: 0)],
        mode="advisory",
    )
    # Nothing anywhere: no directory is created, no file is written.
    assert list(tmp_path.iterdir()) == []


def test_a_decision_records_the_exact_model_facing_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(policy_trace.ENV_VAR, str(tmp_path))
    first = _candidate(0, "openRegistry", "Open the Registry row.")
    second = _candidate(1, "openArchive", "Open the Archive row.")

    decision = evaluate_selective_policy(
        _context(first, second),
        [_Selector(lambda _: 0)],
        mode="advisory",
    )
    assert decision.status == "selected"

    records = _records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "decision"
    assert record["schema"] == policy_trace.SCHEMA
    # The recorded prompt is what a training row must reconstruct: the candidate semantics.
    purposes = [candidate["purpose"] for candidate in record["prompt"]["candidates"]]
    assert "Open the Registry row." in purposes
    assert record["prompt"]["goal"] == "Open Registry and prove the page."
    # The model's own choice is recorded as state, never as a label.
    assert record["label"] is None


def test_the_trace_never_carries_trusted_call_arguments(tmp_path, monkeypatch) -> None:
    """Only the screened projection crosses the boundary — the same rule as inference."""

    monkeypatch.setenv(policy_trace.ENV_VAR, str(tmp_path))
    first = _candidate(0, "openRegistry", "Open the Registry row.")
    second = _candidate(1, "openArchive", "Open the Archive row.")
    evaluate_selective_policy(
        _context(first, second),
        [_Selector(lambda _: 0)],
        mode="advisory",
    )
    blob = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    # ``secret_path`` lives only in the trusted call, which the model never sees.
    assert "secret_path" not in blob
    assert "/private/x" not in blob


def test_an_outcome_joins_its_decision_and_records_frame_movement(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(policy_trace.ENV_VAR, str(tmp_path))
    first = _candidate(0, "openRegistry", "Open the Registry row.")
    second = _candidate(1, "openArchive", "Open the Archive row.")
    evaluate_selective_policy(
        _context(first, second),
        [_Selector(lambda _: 0)],
        mode="advisory",
    )
    identity = policy_trace.last_decision_id()
    assert identity is not None

    policy_trace.record_outcome(
        identity,
        executed=True,
        verdict="followed",
        action_ok=True,
        before_fingerprint="frame-a",
        after_fingerprint="frame-b",
        phase_progressed=True,
    )
    records = _records(tmp_path)
    outcome = records[-1]
    assert outcome["kind"] == "outcome"
    assert outcome["decision_id"] == records[0]["decision_id"]
    assert outcome["verdict"] == "followed"
    assert outcome["frame_changed"] is True
    assert outcome["phase_progressed"] is True


def test_an_uncorrelated_outcome_is_dropped_rather_than_guessed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(policy_trace.ENV_VAR, str(tmp_path))
    policy_trace.record_outcome(None, executed=True, verdict="followed")
    assert _records(tmp_path) == []


def test_status_reports_state_without_revealing_recorded_content(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(policy_trace.ENV_VAR, str(tmp_path))
    status = policy_trace.status()
    assert status["enabled"] is True
    assert status["directory"] == str(tmp_path)
    assert status["env_var"] == policy_trace.ENV_VAR
    assert "prompt" not in status


def test_policy_status_reports_whether_recording_is_actually_happening(
    tmp_path, monkeypatch
) -> None:
    """A trace nobody can confirm is a trace that silently records nothing.

    Recording is env-var-only by design, which also meant nothing reported it: a run whose variable
    was set in a different shell from the one that started the daemon looked exactly like a working
    one until the directory turned up empty days later. The record count is the cheap confirmation,
    and it must read no record content.
    """

    from android_ui_analyser.config import Config
    from android_ui_analyser.policy import policy_status

    monkeypatch.delenv(policy_trace.ENV_VAR, raising=False)
    off = policy_status(Config())["training_trace"]
    assert off["enabled"] is False
    assert off["env_var"] == policy_trace.ENV_VAR
    # Nothing to count when it is off, so the key is absent rather than a misleading zero.
    assert "records" not in off

    monkeypatch.setenv(policy_trace.ENV_VAR, str(tmp_path))
    empty = policy_status(Config())["training_trace"]
    assert empty["enabled"] is True and empty["records"] == 0

    (tmp_path / "decisions.jsonl").write_text(
        '{"kind":"decision","prompt":{"goal":"Open Registry"}}\n\n'
        '{"kind":"outcome","verdict":"followed"}\n',
        encoding="utf-8",
    )
    counted = policy_status(Config())["training_trace"]
    # Blank lines are not records.
    assert counted["records"] == 2
    # Reporting readiness must never quote what was recorded.
    assert "Registry" not in json.dumps(counted)
