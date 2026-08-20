"""The table of global options is read off the callback, because a hand-kept copy drifted.

`_first_subcommand` skips the *values* of value-taking globals so that `aua --serial X analyze
--format json` still gets its options hoisted. It knew which globals take a value from a dict
written out by hand, and that dict was missing three of them: `--owner`, `--needs`, `--page`.

A missing entry does not fail loudly. The scan stops one token early, on the option's own value,
and treats that value as the subcommand — so both callers of `_first_subcommand` then reason
about a command that does not exist. Measured 2026-08-10: `aua --owner <id> --format tsv analyze
--fields id,text` rewrote `--fields` to `--observe-fields` (the alias is only meant for commands
that have no `--fields` of their own; `analyze` has one) and Click answered "No such option
'--observe-fields'" for an option the caller never typed. The agent that hit it called it a "CLI
inconsistency", went to `analyze --help`, found `--fields` documented, and could not reconcile
the two — it spent the rest of the run working around a tool it had stopped trusting.

Deriving the table removes the class of bug rather than the three instances of it.
"""

from __future__ import annotations

import typer.main

from android_ui_analyser.cli import (
    _first_subcommand,
    _global_opts,
    alias_fields_on_actions,
    app,
    hoist_global_options,
)


def test_every_global_the_callback_declares_is_in_the_table() -> None:
    declared = {
        name
        for param in typer.main.get_command(app).params
        for name in [*param.opts, *param.secondary_opts]
        if name.startswith("--")
    }

    assert declared <= set(_global_opts()), "a global absent from the table breaks subcommand detection"


def test_the_globals_that_drifted_are_known_to_take_a_value() -> None:
    table = _global_opts()

    assert table["--owner"] is True
    assert table["--needs"] is True
    assert table["--page"] is True


def test_a_flag_is_not_recorded_as_taking_a_value() -> None:
    """`--no-cache analyze` must find `analyze`, not swallow it as `--no-cache`'s value."""
    table = _global_opts()

    assert table["--no-cache"] is False
    assert table["--with-image"] is False
    assert _first_subcommand(["--no-cache", "analyze"]) == 1


def test_the_subcommand_is_found_past_a_global_that_used_to_be_missing() -> None:
    assert _first_subcommand(["--owner", "someone:1", "analyze"]) == 2
    assert _first_subcommand(["--needs", "camera", "tap-and-analyze"]) == 2


def test_analyze_keeps_its_own_fields_option_behind_an_owner() -> None:
    argv = ["--owner", "someone:1", "--format", "tsv", "analyze", "--fields", "id,text"]

    assert alias_fields_on_actions(argv) == argv, "analyze declares --fields; nothing to alias"


def test_an_action_still_gets_the_alias_behind_an_owner() -> None:
    argv = ["--owner", "someone:1", "tap-and-analyze", "--rid", "x", "--fields", "id,text"]

    assert "--observe-fields" in alias_fields_on_actions(argv)


def test_hoisting_still_works_behind_a_global_that_used_to_be_missing() -> None:
    hoisted = hoist_global_options(["--owner", "someone:1", "analyze", "--format", "json"])

    assert hoisted.index("--format") < hoisted.index("analyze"), "--format belongs to the group"
