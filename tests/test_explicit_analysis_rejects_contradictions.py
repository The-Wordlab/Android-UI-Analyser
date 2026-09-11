"""Contradictory analysis options fail before the action callback can run."""

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from android_ui_analyser import cli


@pytest.mark.parametrize("args", [
    ["tap-and-analyze", "--rid", "fictional_button"],
    ["input-and-analyze", "--rid", "fictional_field", "hello"],
    ["await-and-analyze", "text:Ready"],
    ["key-and-analyze", "back"],
    ["a11y", "action-and-analyze", "--rid", "fictional_button", "CLICK"],
    ["a11y", "scroll-and-analyze", "--rid", "fictional_list"],
    ["mic", "inject", "fictional.wav"],
    ["mic", "speak", "hello"],
])
def test_cli_rejects_before_callback(monkeypatch, args):
    calls = []
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: calls.append("callback"))
    result = CliRunner().invoke(cli.app, [*args, "--no-observe"])
    assert result.exit_code == 2, result.output
    assert "--no-observe" in result.stderr
    assert calls == []


def _explicit_commands(group):
    for command in group.commands.values():
        if isinstance(command, click.Group):
            yield from _explicit_commands(command)
        elif isinstance(command, cli.AnalyzeCommand):
            yield command


def test_all_explicit_commands_keep_defaults_and_reject_false_overrides():
    commands = list(_explicit_commands(get_command(cli.app)))
    assert len(commands) > 10
    for command in commands:
        params = {p.name: p.default for p in command.params}
        seen = []
        command.callback = lambda seen=seen, **kw: seen.append(kw)
        # await's shared callback defaults to False. Explicit command parsing/help must
        # default to True without changing the callback used by other compositions.
        if "observe" in params:
            assert params["observe"] is True
        ctx = click.Context(command)
        ctx.params = dict(params)
        for name in params:
            ctx.set_parameter_source(name, click.core.ParameterSource.DEFAULT)
        command.invoke(ctx)
        assert seen and seen[0].get("observe", True) is True
        assert seen[0].get("no_observe", False) is False
        for source in (click.core.ParameterSource.COMMANDLINE,
                       click.core.ParameterSource.DEFAULT_MAP,
                       click.core.ParameterSource.ENVIRONMENT):
            name = "observe" if "observe" in params else "no_observe"
            if name not in params:
                continue
            ctx.params = dict(params, **{name: name == "no_observe"})
            ctx.set_parameter_source(name, source)
            with pytest.raises(click.UsageError, match="--no-observe"):
                command.invoke(ctx)
        assert len(seen) == 1


def test_legacy_action_still_accepts_no_observe(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: calls.append("callback"))
    result = CliRunner().invoke(cli.app, ["a11y", "action", "--rid", "fictional", "CLICK", "--no-observe"])
    assert result.exit_code == 0, result.output
    assert calls == ["callback"]
