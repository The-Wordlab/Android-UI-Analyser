"""`--until-timeout` with no `--until` waits for nothing, so it must not be accepted quietly.

Run 7 of the fresh-agent series (2026-08-10) tapped with `--until-timeout 3000` and recorded it
in its own report as a wait — "timeout was safety bound", "settled in 411ms". No wait happened.
The flag only bounds a `--until`, and without a predicate there is nothing to bound and no
`await_outcome` in the response. Having no assertion to point at, that run then spent four of its
ten commands proving the result by hand: a re-analyze piped through `jq`, two `has` checks, and a
`capture last --help`.

A flag that is silently ignored while the caller believes it guarantees something is the same
failure this codebase keeps removing: a value you cannot distinguish from its absence. One failed
call that names the predicate costs far less than a false belief carried through a whole session.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from android_ui_analyser.cli import app

runner = CliRunner()


def _error(*globals_first: str) -> dict:
    """Globals go before the subcommand: `run()` hoists them, a direct invoke does not."""
    result = runner.invoke(app, [*globals_first, "tap-and-analyze", "--rid", "x"])
    combined = result.output + str(result.stderr or "")
    for line in combined.splitlines():
        if line.startswith("{"):
            return {"exit_code": result.exit_code, **json.loads(line).get("error", {})}
    return {"exit_code": result.exit_code}


def test_a_dangling_until_timeout_is_refused() -> None:
    err = _error("--until-timeout", "3000")

    assert err["exit_code"] != 0, "silently ignoring it is what produced the false belief"
    assert "--until-timeout" in err.get("message", ""), err
    assert "no --until was given" in err.get("message", ""), err


def test_the_refusal_names_the_predicate_forms() -> None:
    hint = _error("--until-timeout", "3000").get("hint", "")

    assert "--until 'rid:<target>'" in hint, hint
    assert "await_outcome" in hint, "the reason to use a predicate is that it asserts for you"


def test_a_dangling_until_poll_is_refused_too() -> None:
    err = _error("--until-poll", "200")

    assert err["exit_code"] != 0
    assert "--until-poll" in err.get("message", ""), err


def test_both_are_named_when_both_dangle() -> None:
    message = _error("--until-timeout", "3000", "--until-poll", "200").get("message", "")

    assert "--until-timeout" in message and "--until-poll" in message, message


def test_the_equals_spelling_is_caught() -> None:
    err = _error("--until-timeout=3000")

    assert err["exit_code"] != 0, "`--opt=value` is the same request"
