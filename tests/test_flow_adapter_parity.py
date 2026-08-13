"""CLI and MCP preserve the preview-first flow-save boundary."""

from __future__ import annotations

import json
from typing import Any

import anyio
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import cli as cli_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from conftest import FakeDevice, make_config

runner = CliRunner()


def _engine() -> Engine:
    return Engine(make_config(), device=FakeDevice())


def _first_text(result: Any) -> str:
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    raise AssertionError(f"no text content in {result!r}")


def test_cli_flow_save_routes_preview_by_default_and_commit_explicitly(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_route(_engine: Engine, command: str, **kwargs: object) -> dict[str, object]:
        assert command == "flow_save"
        calls.append(kwargs)
        saved = bool(kwargs["save"])
        return {
            "ok": True,
            "action": "flow-save" if saved else "flow-save-preview",
            "saved": saved,
        }

    monkeypatch.setattr(cli_mod, "_route", fake_route)

    preview = runner.invoke(app, ["flow", "save", "journey"])
    commit = runner.invoke(app, ["flow", "save", "journey", "--save"])

    assert preview.exit_code == 0 and json.loads(preview.stdout)["saved"] is False
    assert commit.exit_code == 0 and json.loads(commit.stdout)["saved"] is True
    assert [call["save"] for call in calls] == [False, True]
    assert all(call["dry_run"] is False and call["force"] is False for call in calls)


def test_cli_flow_save_rejects_nonpositive_last_before_routing(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(cli_mod, "_route", lambda *_args, **_kwargs: calls.append(True))

    result = runner.invoke(app, ["flow", "save", "journey", "--last", "0"])

    assert result.exit_code != 0
    assert calls == []


def test_mcp_flow_save_dispatches_preview_by_default_and_commit_explicitly(
    monkeypatch,
) -> None:
    engine = _engine()
    calls: list[dict[str, object]] = []

    def fake_save(name: str, **kwargs: object) -> dict[str, object]:
        calls.append({"name": name, **kwargs})
        saved = bool(kwargs["save"])
        return {
            "ok": True,
            "action": "flow-save" if saved else "flow-save-preview",
            "saved": saved,
        }

    monkeypatch.setattr(engine, "flow_save", fake_save)
    server = build_server(engine)

    async def run() -> list[dict[str, object]]:
        async with create_connected_server_and_client_session(server) as client:
            preview = await client.call_tool("flow_save", {"name": "journey"})
            commit = await client.call_tool(
                "flow_save", {"name": "journey", "save": True}
            )
            return [json.loads(_first_text(preview)), json.loads(_first_text(commit))]

    preview, commit = anyio.run(run)

    assert preview["saved"] is False and preview["action"] == "flow-save-preview"
    assert commit["saved"] is True and commit["action"] == "flow-save"
    assert [call["save"] for call in calls] == [False, True]
    assert all(call["dry_run"] is False and call["force"] is False for call in calls)
