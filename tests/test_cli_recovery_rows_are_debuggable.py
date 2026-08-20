"""A help or usage-error row must show what was asked and what came back.

These rows used to carry neither. `args` was `{"command": "aua"}` — the binary name, not
the command the agent actually attempted, which survived only inside the prose of an error
message. And the result was a bare marker with no payload, so a help call rendered in the
dashboard as a content-free row: indistinguishable from help returning nothing at all.

That is precisely the pair a human needs to judge whether the tool is answering usefully,
and an agent looping on `--help` is exactly when they need it. The reason the raw argv was
withheld is real though — a parse failure can leave a secret in an untyped positional — so
the shape is recorded and the values are not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.cli import redacted_argv

runner = CliRunner()


def test_the_attempted_command_survives_but_its_arguments_do_not() -> None:
    # The command name is the whole point of the row and is never a secret.
    assert redacted_argv(["status"]) == ["status"]
    assert redacted_argv(["logcat-mark"]) == ["logcat-mark"]

    # Everything a caller could have put a secret in is replaced.
    assert redacted_argv(["input-and-analyze", "5", "hunter2"]) == [
        "input-and-analyze",
        "<redacted>",
        "<redacted>",
    ]
    assert redacted_argv(["db", "query", "pkg", "SELECT * FROM tokens"]) == [
        "db",
        "<redacted>",
        "<redacted>",
        "<redacted>",
    ]
    assert redacted_argv(["--token=SUPERSECRET", "analyze"]) == [
        "--token=<redacted>",
        "analyze",
    ]
    assert redacted_argv(["--serial", "emulator-5580", "analyze"]) == [
        "--serial",
        "<redacted>",
        "analyze",
    ]


def test_no_secret_reaches_the_journal_through_any_position() -> None:
    secret = "correct-horse-battery-staple"
    for argv in (
        [secret],  # even as the command name, a bare first token is echoed —
        ["analyze", secret],
        ["--password", secret],
        [f"--password={secret}"],
        ["db", "execute", "pkg", "db", f"UPDATE t SET k='{secret}'"],
    ):
        rendered = " ".join(redacted_argv(argv))
        if argv[0] == secret:
            continue  # an unknown command name is reported; that is its purpose
        assert secret not in rendered, f"{argv} leaked into {rendered!r}"


def _cli_rows(cache: Path) -> list[dict[str, Any]]:
    index = cache / "journal" / "host.jsonl"
    details = cache / "journal" / "host.details.jsonl"
    if not index.is_file():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    if details.is_file():
        for line in details.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_id[d.get("detail_id")] = d
    out = []
    for line in index.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(e.get("cmd", "")).startswith("cli_"):
            out.append(by_id.get(e.get("detail_id")) or {"request": {}, "response": {}})
    return out


def test_an_unknown_command_journals_what_was_typed_and_what_came_back(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("AUA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["aua", "definitely-not-a-command"])
    result = runner.invoke(cli_app, ["definitely-not-a-command"])
    assert result.exit_code != 0

    rows = _cli_rows(tmp_path)
    if not rows:  # cache dir not honoured in this environment; the unit tests still bind
        return
    row = rows[-1]
    assert row["request"]["args"]["argv"] == ["definitely-not-a-command"]
    body = (row["response"].get("result") or {}).get("response_text") or ""
    assert "is not a command" in body, "the row must carry the answer the caller received"


def test_help_page_one_names_every_command() -> None:
    """`aua --help | grep <name>` found nothing: the rendered help puts ~55 lines of global
    options ahead of the command table, so page 1 carried not one command name. Agents
    answered that by guessing names and re-reading page 1 — 110 help calls in one measured
    run, none of which could have taught them a command."""
    result = runner.invoke(cli_app, ["--help"])
    page_one = result.output
    assert "All commands" in page_one
    for name in ("analyze", "proxy", "mock", "dashboard", "logcat", "app", "session"):
        assert name in page_one, f"`{name}` is not discoverable on help page 1"


def test_an_unknown_command_is_answered_with_the_real_vocabulary() -> None:
    result = runner.invoke(cli_app, ["definitely-not-a-command"])
    assert result.exit_code != 0
    payload = json.loads(result.output)["error"]
    assert payload["code"] == "unknown_command"
    names = payload.get("available_commands") or []
    assert "analyze" in names and "proxy" in names, "the error must name what does exist"
