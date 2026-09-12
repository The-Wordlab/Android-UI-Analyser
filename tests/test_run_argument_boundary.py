"""The executable entrypoint preserves nested commands and literal arguments after --."""

from __future__ import annotations

import sys

import pytest

from android_ui_analyser import agent_run, cli


@pytest.mark.parametrize(
    "arguments",
    [
        ["analyze", "--fields", "id,text"],
        ["input-and-analyze", "rid:editor", "--", "--fields"],
    ],
)
def test_real_entrypoint_preserves_run_exec_arguments(tmp_path, monkeypatch, arguments):
    calls = []

    def execute(path, child_arguments):
        calls.append((path, child_arguments))
        return {"ok": True}, 0

    run_path = tmp_path / "run.json"
    monkeypatch.setattr(agent_run, "execute_run", execute)
    monkeypatch.setattr(sys, "argv", ["aua", "run", "exec", str(run_path), "--", *arguments])
    # CliRunner(app) bypasses run() and would miss an outer argv preprocessing regression.
    with pytest.raises(SystemExit) as finished:
        cli.run()
    assert finished.value.code == 0
    assert calls == [(run_path, arguments)]


def test_field_alias_rewrites_action_option_but_preserves_literal_text():
    arguments = ["input-and-analyze", "rid:editor", "--fields", "id,text", "--", "--fields"]
    assert cli.alias_fields_on_actions(arguments) == [
        "input-and-analyze",
        "rid:editor",
        "--observe-fields",
        "id,text",
        "--",
        "--fields",
    ]
    assert cli.alias_fields_on_actions(["analyze", "--fields", "id,text"]) == [
        "analyze",
        "--fields",
        "id,text",
    ]
    assert cli.alias_fields_on_actions(["input-and-analyze", "rid:editor", "--", "--fields"]) == [
        "input-and-analyze",
        "rid:editor",
        "--",
        "--fields",
    ]
