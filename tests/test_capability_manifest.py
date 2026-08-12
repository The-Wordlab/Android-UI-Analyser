"""The canonical discovery catalogue must point only at real CLI and MCP surfaces."""

from __future__ import annotations

import click
from typer.main import get_command
from typer.testing import CliRunner

from android_ui_analyser.capabilities import capability_manifest, render_mcp_instructions
from android_ui_analyser.cli import app
from android_ui_analyser.mcp_server import _tool_definitions


def test_every_manifest_mcp_tool_exists() -> None:
    tools = {tool.name for tool in _tool_definitions()}
    missing = {
        capability["id"]: capability["mcp"]
        for capability in capability_manifest()
        if capability.get("mcp") and capability["mcp"] not in tools
    }
    assert missing == {}


def test_first_manifest_cli_namespace_exists_in_help() -> None:
    runner = CliRunner()
    command_group = get_command(app)
    known = set(command_group.list_commands(click.Context(command_group)))
    commands = {
        next(token for token in str(item["cli"]).split()[1:] if token in known)
        for item in capability_manifest()
    }
    for command in commands:
        help_result = runner.invoke(app, [command, "--help"])
        assert help_result.exit_code == 0, command


def test_mcp_initialization_teaches_the_same_priority_and_cleanup() -> None:
    instructions = render_mcp_instructions()
    assert instructions.index("goto") < instructions.index("matching saved flow")
    assert instructions.index("matching saved flow") < instructions.index("deeplink")
    assert "network_restore or session_finish" in instructions
