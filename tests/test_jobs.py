from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.daemon import DaemonClient, dispatch, serve
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.jobs import JobManager, manager_for, reject_if_active
from android_ui_analyser.mcp_server import _tool_definitions, build_server
from conftest import FakeDevice, make_config


def _engine(*, text_index: dict[str, tuple[int, int, int, int]] | None = None) -> Engine:
    return Engine(make_config(), device=FakeDevice(text_index=text_index or {}))


def _wait_for_status(manager: JobManager, job_id: str, wanted: set[str]) -> dict[str, object]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = manager.status(job_id)
        if state["status"] in wanted:
            return state
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {wanted}: {manager.status(job_id)}")


def test_background_wait_can_be_reconnected_cancelled_and_persisted() -> None:
    engine = _engine()
    manager = manager_for(engine)
    started = manager.start(
        "await",
        {
            "predicate": "text:Never",
            "timeout_ms": 5_000,
            "poll_ms": 20,
            "observe": False,
        },
    )
    job_id = str(started["job_id"])

    running = _wait_for_status(manager, job_id, {"running"})
    assert running["terminal"] is False
    assert running["recommended_call"]["mcp"] == {
        "tool": "job_status",
        "arguments": {"job_id": job_id},
    }
    with pytest.raises(UsageError, match="cannot share") as blocked:
        reject_if_active(engine, "tap")
    assert blocked.value.code == "job_busy"

    requested = manager.cancel(job_id)
    assert requested["status"] == "cancel_requested"
    terminal = manager.wait(job_id, timeout_ms=1_000)
    assert terminal["status"] == "cancelled"
    assert terminal["terminal"] is True
    assert terminal["error"]["code"] == "job_cancelled"

    # The terminal result is durable, not only held by the worker thread.
    assert JobManager(engine).status(job_id)["status"] == "cancelled"


def test_background_wait_returns_terminal_evidence() -> None:
    manager = manager_for(_engine(text_index={"Ready": (0, 0, 100, 100)}))
    started = manager.start(
        "await",
        {
            "predicate": "text:Ready",
            "timeout_ms": 1_000,
            "poll_ms": 20,
            "observe": False,
        },
    )
    terminal = _wait_for_status(manager, str(started["job_id"]), {"succeeded"})
    assert terminal["ok"] is True
    assert terminal["run_ok"] is True
    assert terminal["result"]["await_outcome"] == "satisfied"
    assert terminal["recommended_call"] is None


def test_daemon_dispatch_serializes_device_calls_behind_job() -> None:
    engine = _engine()
    started = dispatch(
        engine,
        {
            "cmd": "job_start",
            "args": {
                "operation": "await",
                "predicate": "text:Never",
                "timeout_ms": 5_000,
                "poll_ms": 20,
                "observe": False,
            },
        },
    )
    assert started["ok"] is True
    job_id = started["result"]["job_id"]
    _wait_for_status(manager_for(engine), job_id, {"running"})

    blocked = dispatch(engine, {"cmd": "analyze", "args": {}})
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "job_busy"
    cancelled = dispatch(engine, {"cmd": "job_cancel", "args": {"job_id": job_id}})
    assert cancelled["ok"] is True
    assert manager_for(engine).wait(job_id, timeout_ms=1_000)["status"] == "cancelled"


def test_daemon_socket_remains_responsive_for_status_and_cancel() -> None:
    engine = _engine()
    ready = threading.Event()
    stop = threading.Event()
    with tempfile.TemporaryDirectory(prefix="aua_job_") as directory:
        socket = str(Path(directory) / "daemon.sock")
        thread = threading.Thread(
            target=serve,
            args=(engine, socket),
            kwargs={"ready_event": ready, "_stop_event": stop},
            daemon=True,
        )
        thread.start()
        assert ready.wait(timeout=2.0)
        try:
            client = DaemonClient(socket)
            started = client.call(
                "job_start",
                operation="await",
                predicate="text:Never",
                timeout_ms=5_000,
                poll_ms=20,
                observe=False,
            )["result"]
            job_id = started["job_id"]

            before = time.monotonic()
            status = client.call("job_status", job_id=job_id)["result"]
            assert time.monotonic() - before < 0.5
            assert status["status"] in {"queued", "running"}
            blocked = client.call("analyze")
            assert blocked["error"]["code"] == "job_busy"
            assert client.call("job_cancel", job_id=job_id)["ok"] is True
            terminal = client.call("job_wait", job_id=job_id, timeout_ms=1_000)["result"]
            assert terminal["status"] == "cancelled"
        finally:
            stop.set()
            thread.join(timeout=3.0)
            assert not thread.is_alive()


def test_mcp_exposes_job_lifecycle_and_guard() -> None:
    tools = {tool.name: tool for tool in _tool_definitions()}
    assert {"job_start", "job_status", "job_wait", "job_cancel", "job_list"} <= tools.keys()
    assert tools["job_wait"].inputSchema["properties"]["timeout_ms"]["maximum"] == 10_000

    server = build_server(_engine())

    async def run() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        async with create_connected_server_and_client_session(server) as client:
            started_result = await client.call_tool(
                "job_start",
                {
                    "operation": "await",
                    "predicate": "text:Never",
                    "timeout_ms": 5_000,
                    "poll_ms": 20,
                    "observe": False,
                },
            )
            started = json.loads(started_result.content[0].text)
            blocked_result = await client.call_tool("analyze_screen", {})
            blocked = json.loads(blocked_result.content[0].text)
            cancelled_result = await client.call_tool(
                "job_cancel", {"job_id": started["job_id"]}
            )
            cancelled = json.loads(cancelled_result.content[0].text)
            return started, blocked, cancelled

    started, blocked, cancelled = anyio.run(run)
    assert started["terminal"] is False
    assert blocked["error"]["code"] == "job_busy"
    assert cancelled["status"] in {"cancel_requested", "cancelled"}


def test_job_cli_help_teaches_reconnect_contract() -> None:
    group = CliRunner().invoke(app, ["job", "--help"])
    assert group.exit_code == 0
    assert "reconnectable status" in group.stdout
    start = CliRunner().invoke(app, ["job", "start", "--help"])
    assert start.exit_code == 0
    assert "await|wait-stable|wait-changed|wait-after-change" in start.stdout
