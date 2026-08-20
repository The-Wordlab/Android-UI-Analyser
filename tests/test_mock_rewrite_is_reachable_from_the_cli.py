"""`aua mock rewrite` — the CLI door onto response patching.

The rewrite half of the proxy (patch the real response instead of stubbing it) shipped
implemented and addon-tested but with no engine, CLI or MCP surface, so no caller could
reach it. These tests cover the option parsing that door depends on: a header pair, a
JSON field assignment, and a literal substitution all have their own little syntax, and
getting one wrong silently arms a rule that changes nothing.
"""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.engine import Engine

runner = CliRunner()


def _capture(monkeypatch: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake(self: Engine, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update({"method": method, "path": path, **kwargs})
        return {"ok": True, "action": "mock-rewrite", "rule": {"id": "r1"}, "count": 1}

    monkeypatch.setattr(Engine, "mock_rewrite", fake)
    return seen


def test_mock_rewrite_parses_headers_json_sets_and_replacements(monkeypatch: Any) -> None:
    seen = _capture(monkeypatch)

    result = runner.invoke(
        cli_app,
        [
            "mock", "rewrite", "GET", "/v1/feed",
            "--host", "api.example.test",
            "--status", "429",
            "--header", "Retry-After: 30",
            "--set", 'items[0].title="patched"',
            "--set", "quota.remaining=0",
            "--delete", "meta.cursor",
            "--replace", "premium=>free",
            "--times", "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/feed"
    assert seen["host"] == "api.example.test"
    assert seen["status"] == 429
    assert seen["headers"] == {"Retry-After": "30"}
    # A JSON value is decoded; `0` must arrive as an int, not the string "0".
    assert seen["set_json"] == {'items[0].title': "patched", "quota.remaining": 0}
    assert seen["delete_json"] == ["meta.cursor"]
    assert seen["replace"] == [("premium", "free")]
    assert seen["times"] == 2


def test_mock_rewrite_leaves_unused_options_unset(monkeypatch: Any) -> None:
    """Empty option lists must arrive as None, not as an empty dict the rule builder
    would read as "the caller asked for no headers"."""
    seen = _capture(monkeypatch)

    result = runner.invoke(cli_app, ["mock", "rewrite", "POST", "/v1/pay", "--status", "402"])

    assert result.exit_code == 0, result.output
    assert seen["status"] == 402
    assert seen["headers"] is None
    assert seen["set_json"] is None
    assert seen["delete_json"] is None
    assert seen["replace"] is None


def test_mock_rewrite_rejects_malformed_option_syntax(monkeypatch: Any) -> None:
    _capture(monkeypatch)

    for bad in (
        ["--header", "no-colon-here"],
        ["--set", "no-equals-sign"],
        ["--replace", "no-arrow"],
    ):
        result = runner.invoke(cli_app, ["mock", "rewrite", "GET", "/v1/feed", *bad])
        assert result.exit_code != 0, f"{bad} should have been refused: {result.output}"


def test_malformed_options_answer_with_the_json_envelope_not_a_traceback(
    monkeypatch: Any,
) -> None:
    """These were parsed in the command body, outside `_run`'s error wrapper, so a typo
    printed a forty-line Python traceback and exited 1 — while every other refusal in the
    tool emits one line of JSON and exits 2. An agent cannot branch on a traceback."""
    _capture(monkeypatch)

    for bad in (
        ["--header", "NoColon"],
        ["--set", "notanassignment"],
        ["--replace", "nofatarrow"],
    ):
        result = runner.invoke(
            cli_app, ["mock", "rewrite", "GET", "/v1/feed", "--host", "api.example.test", *bad]
        )
        assert result.exit_code == 2, f"{bad}: expected the usage exit code"
        assert "Traceback" not in result.output, f"{bad}: raw traceback reached the caller"
