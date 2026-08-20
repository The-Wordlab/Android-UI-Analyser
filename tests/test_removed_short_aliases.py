"""The short action aliases are gone, and saying so is the whole point.

They were hidden from ``--help`` first. That did not work, because **a hidden alias still
answers**: a caller who reaches for the obvious short name gets a weaker response and is never
corrected, so the habit survives in agent memory, in downstream docs and in prompts. One
downstream suite measured it — 2322 invocations, ``tap-and-analyze`` used *zero* times, bare
``tap`` followed by a separate ``analyze`` 255 times, 36% of every tap.

So the contract these tests pin is not "the command is absent" but "the command is absent **and
the error names its replacement**". A bare `No such command` would leave the caller guessing;
naming the replacement costs one failed call and fixes the caller for the rest of the session.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import _REMOVED_ACTION_ALIASES, app

runner = CliRunner()


@pytest.mark.parametrize("old", _REMOVED_ACTION_ALIASES)
def test_removed_alias_fails_and_names_its_replacement(old: str) -> None:
    result = runner.invoke(app, [old, "--rid", "whatever"])

    assert result.exit_code == 2, f"`aua {old}` should be a usage error, got {result.exit_code}"
    payload = json.loads(result.stderr)["error"]
    assert payload["code"] == "removed_command"
    # The replacement must be named, not merely implied - that is what makes the failure useful.
    assert f"{old}-and-analyze" in payload["message"]
    assert f"aua {old}" in payload["message"]


@pytest.mark.parametrize("old", _REMOVED_ACTION_ALIASES)
def test_replacement_still_exists(old: str) -> None:
    """Every name we point callers at must actually resolve, or the error is a dead end."""
    result = runner.invoke(app, [f"{old}-and-analyze", "--help"])
    assert result.exit_code == 0, f"aua {old}-and-analyze should exist"


def test_removed_aliases_are_not_advertised_in_help() -> None:
    """Gone from `--help` too: a listed-but-broken command is worse than an absent one."""
    listed = runner.invoke(app, ["--help"]).stdout
    for old in _REMOVED_ACTION_ALIASES:
        assert f"\n  {old} " not in listed, f"{old} should not be advertised"


def test_the_error_survives_arbitrary_arguments() -> None:
    """The message must be about the *name*, never a complaint about the caller's flags.

    A caller reaching for `aua tap` passes whatever options it always passed. If argument
    parsing rejected those first, the response would describe the wrong problem and the caller
    would 'fix' the flags and try the dead name again.
    """
    for argv in (
        ["tap", "17"],
        ["tap", "--rid", "submitButton", "--index", "2"],
        ["input", "--rid", "queryField", "hello world"],
        ["wait", "--for", "Loading", "--absent", "--timeout", "200"],
        ["scroll-to", "--by", "id", "someTag", "--nonsense-flag"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 2, argv
        assert json.loads(result.stderr)["error"]["code"] == "removed_command", argv


@pytest.mark.parametrize("old", _REMOVED_ACTION_ALIASES)
def test_removed_alias_has_no_plausible_help_page(old: str) -> None:
    """`--help` on a dead command must NOT render an empty page and exit 0.

    Regression for the 2026-08-08 probe, where three of four lanes independently reported that
    `aua tap --help` printed `Usage: aua tap [OPTIONS]` with an empty options box and exited 0 --
    so the careful caller who checks help *before* guessing was told the command exists and takes
    nothing, while the caller who just guessed got the correct error. Help must agree with runtime.
    """
    result = runner.invoke(app, [old, "--help"])
    assert result.exit_code == 2, f"`aua {old} --help` rendered a help page instead of refusing"
    assert json.loads(result.stderr)["error"]["code"] == "removed_command"
