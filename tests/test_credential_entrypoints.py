"""Credential entry is a host-only boundary, never a device action or a value argument."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.main import get_command
from typer.testing import CliRunner

from android_ui_analyser import cli, credentials, mcp_server
from android_ui_analyser.engine import Engine


class NoDeviceEngine:
    """Server construction is allowed; any request-time engine use is a failure."""

    def capture_service_start(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"credential request accessed engine.{name}")


def _server() -> Any:
    return mcp_server.build_server(cast(Engine, NoDeviceEngine()))


def _payload(result: Any) -> dict[str, Any]:
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    return cast(dict[str, Any], json.loads(result.content[0].text))


@pytest.mark.parametrize(
    ("status", "ok", "exit_code"),
    [("saved", True, 0), ("already_set", True, 0), ("cancelled", False, 130), ("error", False, 1)],
)
def test_cli_credential_result_and_exit_without_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str, ok: bool, exit_code: int
) -> None:
    destination = tmp_path / ".env"
    expected = {"ok": ok, "status": status, "name": "EXAMPLE_API_KEY", "env_file": str(destination)}
    calls = []

    def request(name: str, env_file: str | Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((name, env_file, kwargs))
        return expected

    monkeypatch.setattr(credentials, "request_secret", request)
    monkeypatch.setattr(cli, "_run", lambda *_a, **_kw: pytest.fail("credential used _run"))
    monkeypatch.setattr(
        cli.GlobalOpts, "engine", lambda *_a: pytest.fail("credential built engine")
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "config",
            "secret",
            "EXAMPLE_API_KEY",
            "--env-file",
            str(destination),
            "--replace",
            "--timeout",
            "45",
        ],
    )
    assert result.exit_code == exit_code, result.output
    assert json.loads(result.stdout) == expected
    assert calls == [("EXAMPLE_API_KEY", destination, {"replace": True, "timeout_s": 45})]


def test_cli_default_destination_and_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"ok": True, "status": "already_set"}

    monkeypatch.setattr(credentials, "request_secret", request)
    result = CliRunner().invoke(cli.app, ["config", "secret", "EXAMPLE_TOKEN"])
    assert result.exit_code == 0
    assert calls == [(("EXAMPLE_TOKEN", Path(".env")), {"replace": False, "timeout_s": 300})]


def test_credential_schemas_have_no_secret_value_or_device_annotation_arguments() -> None:
    schema = next(
        t.inputSchema for t in mcp_server._tool_definitions() if t.name == "credential_request"
    )
    assert set(schema["properties"]) == {"name", "env_file", "replace", "timeout_s"}
    assert schema["required"] == ["name"]
    assert schema["additionalProperties"] is False
    command = get_command(cli.app).commands["config"].commands["secret"]
    assert {p.name for p in command.params} == {"name", "env_file", "replace", "timeout"}


@pytest.mark.parametrize(
    "ok,status", [(True, "saved"), (True, "already_set"), (False, "cancelled"), (False, "error")]
)
def test_mcp_credential_bypasses_engine_journal_and_images(
    monkeypatch: pytest.MonkeyPatch, ok: bool, status: str
) -> None:
    from android_ui_analyser import journal

    calls = []
    expected = {"ok": ok, "status": status, "name": "EXAMPLE_API_KEY", "env_file": "/project/.env"}

    def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(credentials, "request_secret", request)
    monkeypatch.setattr(
        journal, "record", lambda **_kw: pytest.fail("credential entered device journal")
    )

    async def run() -> Any:
        async with create_connected_server_and_client_session(_server()) as client:
            return await client.call_tool(
                "credential_request",
                {
                    "name": "EXAMPLE_API_KEY",
                    "env_file": "/project/.env",
                    "replace": True,
                    "timeout_s": 45,
                },
            )

    result = anyio.run(run)
    assert _payload(result) == expected
    assert result.isError is not ok
    assert calls == [(("EXAMPLE_API_KEY", "/project/.env"), {"replace": True, "timeout_s": 45})]


def test_direct_mcp_dispatch_uses_same_helper_before_any_engine_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"ok": True, "status": "already_set"}

    monkeypatch.setattr(credentials, "request_secret", request)
    result = mcp_server._dispatch(
        cast(Engine, NoDeviceEngine()), "credential_request", {"name": "EXAMPLE_TOKEN"}
    )
    assert result["status"] == "already_set"
    assert calls == [(("EXAMPLE_TOKEN", ".env"), {"replace": False, "timeout_s": 300})]


def test_mcp_dialog_wait_does_not_block_other_protocol_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(3), "test did not release dialog"
        return {"ok": False, "status": "cancelled"}

    monkeypatch.setattr(credentials, "request_secret", request)

    async def run() -> None:
        async with (
            create_connected_server_and_client_session(_server()) as client,
            anyio.create_task_group() as tasks,
        ):
            tasks.start_soon(client.call_tool, "credential_request", {"name": "EXAMPLE_TOKEN"})
            try:
                assert await anyio.to_thread.run_sync(entered.wait, 1)
                with anyio.fail_after(1):
                    listed = await client.list_tools()
                assert any(tool.name == "credential_request" for tool in listed.tools)
            finally:
                release.set()

    anyio.run(run)
