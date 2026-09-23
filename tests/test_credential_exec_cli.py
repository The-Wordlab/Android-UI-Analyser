"""The credential wrapper preserves argv/output and never launches on malformed setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from android_ui_analyser import cli, credential_exec


@pytest.fixture(autouse=True)
def no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_run", lambda *_a, **_kw: pytest.fail("credential exec used _run"))
    monkeypatch.setattr(
        cli.GlobalOpts, "engine", lambda *_a: pytest.fail("credential exec built engine")
    )


def test_child_argv_is_preserved_after_explicit_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> tuple[dict[str, Any], int]:
        calls.append((command, kwargs))
        typer.echo("child stdout")
        typer.echo("child stderr", err=True)
        return {"ok": True, "status": "completed", "started": True, "exit_code": 0}, 0

    monkeypatch.setattr(credential_exec, "run_with_credentials", run)
    destination = tmp_path / ".env"
    child = ["python", "runner.py", "--require", "CHILD_OPTION", "--timeout", "7", "--", "-x"]
    result = CliRunner().invoke(
        cli.app,
        [
            "config",
            "exec",
            "--env-file",
            str(destination),
            "--require",
            "EXAMPLE_KEY",
            "--require",
            "SECOND_KEY",
            "--timeout",
            "45",
            "--no-prompt",
            "--",
            *child,
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "child stdout\n"
    assert result.stderr == "child stderr\n"
    assert calls == [
        (
            child,
            {
                "required": ["EXAMPLE_KEY", "SECOND_KEY"],
                "env_file": destination,
                "timeout_s": 45,
                "prompt": False,
                "optional": [],
            },
        )
    ]


@pytest.mark.parametrize("exit_code", [0, 17, 130])
def test_default_controls_and_child_exit_do_not_append_json(
    monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> tuple[dict[str, Any], int]:
        calls.append((command, kwargs))
        return {
            "ok": exit_code == 0,
            "status": "completed",
            "started": True,
            "exit_code": exit_code,
        }, exit_code

    monkeypatch.setattr(credential_exec, "run_with_credentials", run)
    result = CliRunner().invoke(
        cli.app, ["config", "exec", "--require", "EXAMPLE_KEY", "--", "child"]
    )
    assert result.exit_code == exit_code
    assert result.stdout == result.stderr == ""
    assert calls == [
        (
            ["child"],
            {
                "required": ["EXAMPLE_KEY"],
                "env_file": Path(".env"),
                "timeout_s": 300,
                "prompt": True,
                "optional": [],
            },
        )
    ]


@pytest.mark.parametrize(
    ("status", "started", "exit_code"),
    [
        ("cancelled", False, 130),
        ("error", False, 1),
        ("error", True, 1),
    ],
)
def test_wrapper_failure_is_safe_json_on_stderr(
    monkeypatch: pytest.MonkeyPatch, status: str, started: bool, exit_code: int
) -> None:
    metadata: dict[str, Any] = {"ok": False, "status": status, "started": started}
    if status == "error":
        metadata["error"] = {
            "code": "credential_test_failure",
            "message": "Credential setup failed.",
        }
    monkeypatch.setattr(
        credential_exec, "run_with_credentials", lambda *_a, **_kw: (metadata, exit_code)
    )
    result = CliRunner().invoke(
        cli.app, ["config", "exec", "--require", "EXAMPLE_KEY", "--", "child"]
    )
    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert json.loads(result.stderr) == metadata


@pytest.mark.parametrize(
    "arguments",
    [
        ["--", "child"],
        ["--require", "EXAMPLE_KEY", "--"],
        ["--require", "EXAMPLE_KEY", "child"],
        ["--require", "EXAMPLE_KEY", "child", "--", "--child-flag"],
        ["--require", "EXAMPLE_KEY", "--unknown-wrapper-flag", "--", "child"],
    ],
)
def test_invalid_wrapper_invocations_never_call_helper(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(
        credential_exec,
        "run_with_credentials",
        lambda *_a, **_kw: pytest.fail("invalid invocation launched helper"),
    )
    result = CliRunner().invoke(cli.app, ["config", "exec", *arguments])
    assert result.exit_code == 2, result.output


def test_cli_global_rewriting_does_not_change_child_flags() -> None:
    arguments = [
        "config",
        "exec",
        "--env-file",
        ".env",
        "--require",
        "EXAMPLE_KEY",
        "--timeout",
        "25",
        "--",
        "aua",
        "run",
        "exec",
        "run.json",
        "--",
        "--timeout",
        "123",
        "analyze",
        "--fields",
        "id",
    ]
    assert cli.hoist_global_options(cli.alias_fields_on_actions(arguments)) == arguments


def test_help_needs_no_required_variable_or_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        credential_exec,
        "run_with_credentials",
        lambda *_a, **_kw: pytest.fail("help launched helper"),
    )
    result = CliRunner().invoke(cli.app, ["config", "exec", "--help"])
    assert result.exit_code == 0
    assert "--require" in result.stdout
    assert "--no-prompt" in result.stdout
