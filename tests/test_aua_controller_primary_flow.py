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
