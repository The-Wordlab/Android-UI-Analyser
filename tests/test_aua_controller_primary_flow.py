"""A harness exports its observed journal suffix, never guesses or saves a global flow."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.run_realapp import (
    ControllerJournalError,
    PrimaryFlowUnavailable,
    RunError,
    export_primary_flow,
)


def _journal(tmp_path, entries):
    (tmp_path / "controller").mkdir()
    (tmp_path / "controller/tool-calls.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries))


def test_exports_exact_preview_only_to_run_directory(tmp_path):
    _journal(tmp_path, [{"tool": "input_and_analyze", "arguments": {"text": "private-typed-value"},
                         "executed": True, "result": {"ok": True}}])
    calls = []

    async def call(name, args, actor):
        calls.append((name, args))
        return {"ok": True, "steps": 1, "scope": {"requested_last": 1, "selected": 1,
                "boundary_omitted": 0}, "preview": 'steps:\n  - input: {text: "${PARAM_1}"}\n'}

    result = asyncio.run(export_primary_flow(call, tmp_path))
    assert result == {"route_action_count": 1, "primary_flow": "flow.yaml"}
    assert (tmp_path / "flow.yaml").read_text() == 'steps:\n  - input: {text: "${PARAM_1}"}\n'
    assert "private-typed-value" not in (tmp_path / "flow.yaml").read_text()
    assert calls == [("flow_save", {"name": "controller-route", "last": 1, "save": False})]


@pytest.mark.parametrize("entry", [
    {"tool": "tap_and_analyze", "executed": True, "result": {"ok": False}},
    {"tool": "tap_and_analyze", "dispatch_started": True, "execution_outcome": "unknown"},
    {"tool": "unsupported_mutation", "executed": True, "result": {"ok": True}},
])
def test_incomplete_or_unsupported_journal_cannot_produce_a_primary_flow(tmp_path, entry):
    _journal(tmp_path, [entry])

    async def forbidden(*args):
        pytest.fail("cannot preview an unproved journal")

    with pytest.raises(RunError):
        asyncio.run(export_primary_flow(forbidden, tmp_path))
    assert not (tmp_path / "flow.yaml").exists()


def test_capture_boundary_omission_refuses_export(tmp_path):
    _journal(tmp_path, [{"tool": "tap_and_analyze", "executed": True, "result": {"ok": True}}])

    async def call(*args):
        return {"ok": True, "steps": 1, "scope": {"requested_last": 1, "selected": 1,
                "boundary_omitted": 1}, "preview": "untrusted suffix"}

    with pytest.raises(RunError, match="exact clean"):
        asyncio.run(export_primary_flow(call, tmp_path))
    assert not (tmp_path / "flow.yaml").exists()


def test_pure_observation_never_steals_setup_actions(tmp_path):
    _journal(tmp_path, [{"tool": "analyze_screen", "executed": True, "result": {"ok": True}}])

    async def forbidden(*args):
        pytest.fail("there are no controller actions to preview")

    result = asyncio.run(export_primary_flow(forbidden, tmp_path))
    assert result == {"route_action_count": 0, "flow_not_applicable_reason": "observation_only_no_actions"}


def test_malformed_journal_is_execution_error_not_optional_export_failure(tmp_path):
    _journal(tmp_path, [])
    (tmp_path / "controller/tool-calls.jsonl").write_text("not JSON\n")
    with pytest.raises(ControllerJournalError):
        asyncio.run(export_primary_flow(None, tmp_path))


def _recovered_journal(tmp_path):
    miss = {"step": 0, "tool": "tap_and_analyze", "executed": True, "dispatch_started": True,
            "result": {"error": {"code": "element_not_found", "hint": "No action was sent. Inspect the observation.",
                "observation_present": True, "observation": {"screen": {"package": "example.app"},
                "elements": [], "meta": {"fingerprint": "fresh-frame"}}}}}
    success = {"step": 1, "tool": "tap_and_analyze", "executed": True, "result": {"ok": True}}
    back = {"step": 2, "tool": "back_gesture_and_analyze", "executed": True, "result": {"ok": True}}
    terminal = {"step": 3, "tool": "session_finish", "executed": True, "evidence_ref": "E4",
                "result": {"ok": False, "finished": False, "claim_recorded": True}}
    report = {"stop_reason": "terminal_claimed", "unknown_tool_outcomes": 0,
              "terminal_submission": {"tool": "session_finish", "evidence_ref": "E4",
                                      "result": terminal["result"]}}
    entries = [miss, success, back, terminal]
    _journal(tmp_path, entries)
    (tmp_path / "controller/controller-result.json").write_text(json.dumps(report))
    return entries, report


def test_definitive_unsent_miss_then_successful_route_and_terminal_claim_only_omit_flow(tmp_path):
    _recovered_journal(tmp_path)

    async def forbidden(*args):
        pytest.fail("a recovered journal is not a clean replay suffix")

    with pytest.raises(PrimaryFlowUnavailable) as error:
        asyncio.run(export_primary_flow(forbidden, tmp_path))
    assert error.value.route_action_count == 2
    assert error.value.recovered_no_action == 1
    assert not (tmp_path / "flow.yaml").exists()


@pytest.mark.parametrize("corruption", ["unknown", "no_claim", "no_report", "unrecovered", "ambiguous",
                                       "no_observation", "possibly_sent", "cleanup", "entry_unknown", "sent_evidence"])
def test_recovered_selector_exception_refuses_any_unproved_execution(tmp_path, corruption):
    entries, report = _recovered_journal(tmp_path)
    if corruption == "unknown":
        report["unknown_tool_outcomes"] = 1
    elif corruption == "no_claim":
        entries.pop()
    elif corruption == "no_report":
        report = {}
    elif corruption == "unrecovered":
        entries = [entries[-1], entries[0]]
    elif corruption == "ambiguous":
        entries[0]["result"]["error"]["code"] = "device_error"
    elif corruption == "no_observation":
        entries[0]["result"]["error"].pop("observation")
    elif corruption == "possibly_sent":
        entries[0]["result"]["error"]["hint"] = "Action may have been sent."
    elif corruption == "cleanup":
        entries[-1]["result"] = {"ok": False, "finished": False, "error": "cleanup failed"}
    elif corruption == "sent_evidence":
        entries[0]["result"]["capture_evidence"] = {"action": "tap"}
    else:
        entries[0]["execution_outcome"] = "unknown"
    (tmp_path / "controller/tool-calls.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    (tmp_path / "controller/controller-result.json").write_text(json.dumps(report))
    with pytest.raises(ControllerJournalError):
        asyncio.run(export_primary_flow(None, tmp_path))


def test_confirmed_no_action_then_terminal_claim_has_exact_zero_action_count(tmp_path):
    entries, _ = _recovered_journal(tmp_path)
    (tmp_path / "controller/tool-calls.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in (entries[0], entries[-1])))
    with pytest.raises(PrimaryFlowUnavailable) as error:
        asyncio.run(export_primary_flow(None, tmp_path))
    assert error.value.route_action_count == 0 and error.value.recovered_no_action == 1
