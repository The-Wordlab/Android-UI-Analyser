"""CLI and MCP preserve the preview-first flow-save boundary."""

from __future__ import annotations

import json
from typing import Any

import anyio
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import cli as cli_mod
from android_ui_analyser import journal
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.session import create_session_state
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
            "arrival_status": "predicate_verified",
            "arrival_proof": {"status": "verified", "source": "satisfied_action_until"},
            "selector_resilience": [
                {"selector": "composite", "strength": "unknown", "cross_frame": False}
            ],
            "preview": "arrival_status: predicate_verified\n",
        }

    monkeypatch.setattr(cli_mod, "_route", fake_route)

    preview = runner.invoke(app, ["flow", "save", "journey"])
    commit = runner.invoke(app, ["flow", "save", "journey", "--save"])

    preview_payload = json.loads(preview.stdout)
    assert preview.exit_code == 0 and preview_payload["saved"] is False
    assert commit.exit_code == 0 and json.loads(commit.stdout)["saved"] is True
    assert preview_payload["arrival_status"] == "predicate_verified"
    assert preview_payload["arrival_proof"]["source"] == "satisfied_action_until"
    assert preview_payload["selector_resilience"][0] == {
        "selector": "composite",
        "strength": "unknown",
        "cross_frame": False,
    }
    assert "arrival_status: predicate_verified" in preview_payload["preview"]
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
            "arrival_status": "predicate_verified",
            "arrival_proof": {"status": "verified", "source": "satisfied_action_until"},
            "selector_resilience": [
                {"selector": "composite", "strength": "unknown", "cross_frame": False}
            ],
            "preview": "arrival_status: predicate_verified\n",
        }

    monkeypatch.setattr(engine, "flow_save", fake_save)
    server = build_server(engine)

    async def run() -> list[dict[str, object]]:
        async with create_connected_server_and_client_session(server) as client:
            preview = await client.call_tool("flow_save", {"name": "journey"})
            commit = await client.call_tool("flow_save", {"name": "journey", "save": True})
            return [json.loads(_first_text(preview)), json.loads(_first_text(commit))]

    preview, commit = anyio.run(run)

    assert preview["saved"] is False and preview["action"] == "flow-save-preview"
    assert commit["saved"] is True and commit["action"] == "flow-save"
    assert preview["arrival_status"] == "predicate_verified"
    assert preview["arrival_proof"]["source"] == "satisfied_action_until"
    assert preview["selector_resilience"][0] == {
        "selector": "composite",
        "strength": "unknown",
        "cross_frame": False,
    }
    assert "arrival_status: predicate_verified" in preview["preview"]
    assert [call["save"] for call in calls] == [False, True]
    assert all(call["dry_run"] is False and call["force"] is False for call in calls)


def test_cli_flow_delete_routes_idempotent_engine_result(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_route(_engine: Engine, command: str, **kwargs: object) -> dict[str, object]:
        calls.append((command, kwargs))
        return {
            "ok": True,
            "action": "flow-delete",
            "flow": kwargs["name"],
            "deleted": False,
            "status": "already_absent",
        }

    monkeypatch.setattr(cli_mod, "_route", fake_route)

    result = runner.invoke(app, ["flow", "delete", "gone"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "already_absent"
    assert calls == [("flow_delete", {"name": "gone"})]


def test_two_host_only_deletes_are_both_accounted_in_the_active_session(
    monkeypatch,
) -> None:
    cfg = make_config()
    owner = "flow-review-owner"
    state = create_session_state(
        cfg.cache.dir,
        goal="Delete an optional recorded flow twice",
        serial="emulator-5554",
        owner=owner,
        recommended_kind="flow",
        recommended_cli="aua flow delete optional",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    monkeypatch.setenv("AUA_OWNER", owner)

    first = runner.invoke(app, ["--serial", state.serial, "flow", "delete", "optional"])
    second = runner.invoke(app, ["--serial", state.serial, "flow", "delete", "optional"])

    assert first.exit_code == second.exit_code == 0
    rows = [
        event
        for event in journal.read_since(cfg.cache.dir, state.serial, since_ms=state.started_ms)
        if event.get("cmd") == "flow_delete"
    ]
    assert len(rows) == 2
    assert all(event.get("owner") == owner for event in rows)
    assert all(event.get("session_id") == state.session_id for event in rows)


def test_mcp_flow_delete_routes_idempotent_engine_result(monkeypatch) -> None:
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "flow_delete",
        lambda name: {
            "ok": True,
            "action": "flow-delete",
            "flow": name,
            "deleted": False,
            "status": "already_absent",
        },
    )
    server = build_server(engine)

    async def run() -> dict[str, object]:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("flow_delete", {"name": "gone"})
            return json.loads(_first_text(result))

    result = anyio.run(run)

    assert result["ok"] is True
    assert result["deleted"] is False
    assert result["status"] == "already_absent"
