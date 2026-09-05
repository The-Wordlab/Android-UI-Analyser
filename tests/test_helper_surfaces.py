"""Helper lifecycle uses one Engine boundary with durable setup restoration."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser import device_ledger
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import _LONG_POLL_COMMANDS, dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from android_ui_analyser.mcp_server import _tool_definitions
from conftest import FakeDevice, make_config

runner = CliRunner()


class _Helper:
    def __init__(self) -> None:
        self.engine: Engine | None = None
        self.installed = False
        self.enabled = False
        self.restored: list[dict[str, Any]] = []

    def is_installed(self, serial: str) -> bool:
        return self.installed

    def is_enabled(self, serial: str) -> bool:
        return self.enabled

    def snapshot_state(self, serial: str) -> dict[str, Any]:
        return {
            "enabled_services": ["example.reader/.Service"],
            "accessibility_enabled": "0",
            "restricted_settings_appop": "default",
            "adbd_root": False,
        }

    def enable(self, serial: str) -> dict[str, Any]:
        assert self.engine is not None
        pending = device_ledger.read_ledger(serial, platform=self.engine.platform.name)
        assert {entry.key for entry in pending} >= {
            "automatic_device_agent_package",
            "device_agent_service",
        }, "helper setup reached the target before its package/service restore records"
        self.installed = True
        self.enabled = True
        return {"enabled": True, "bound": True}

    def status(self, serial: str) -> dict[str, Any]:
        return {"installed": self.installed, "enabled": self.enabled, "bound": self.enabled}

    def install(self, serial: str, *, reinstall: bool, force: bool) -> dict[str, Any]:
        self.installed = True
        return {"installed": True, "reinstall": reinstall, "force": force}

    def restore_state(self, serial: str, state: dict[str, Any]) -> dict[str, Any]:
        assert self.engine is not None
        assert self.engine._pending_device_change(
            "device_agent_service", serial=serial
        ) is not None
        self.restored.append(dict(state))
        self.enabled = False
        return {"enabled": False, "remaining": state["enabled_services"]}

    def disable(self, serial: str) -> dict[str, Any]:
        self.enabled = False
        return {"enabled": False, "remaining": []}

    def remove(self, serial: str) -> dict[str, Any]:
        self.installed = False
        self.enabled = False
        return {"installed": False, "enabled": False}


def test_helper_enable_records_full_setup_state_before_mutation_and_disable_forgets_it(
    tmp_path,
) -> None:
    runtime = FakeDevice(serial="helper-runtime")
    engine = Engine(make_config(cache={"dir": str(tmp_path / "cache")}), device=runtime)
    helper = _Helper()
    helper.engine = engine
    engine.platform.capability = lambda name: helper  # type: ignore[method-assign]

    enabled = engine.helper_enable()

    assert enabled["enabled"] is True
    pending = device_ledger.read_ledger(runtime.serial, platform=engine.platform.name)
    service = next(entry for entry in pending if entry.key == "device_agent_service")
    assert service.args == helper.snapshot_state(runtime.serial)

    disabled = engine.helper_disable()

    assert disabled["enabled"] is False
    assert helper.restored == [service.args]
    remaining = device_ledger.read_ledger(runtime.serial, platform=engine.platform.name)
    assert {entry.key for entry in remaining} == {"automatic_device_agent_package"}

    engine.helper_remove()
    assert device_ledger.read_ledger(runtime.serial, platform=engine.platform.name) == []


@pytest.fixture
def helper_method_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def result(name: str, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append((name, dict(kwargs or {})))
        return {"ok": True, "action": name.replace("_", "-")}

    monkeypatch.setattr(Engine, "helper_status", lambda self: result("helper_status"))
    monkeypatch.setattr(
        Engine,
        "helper_install",
        lambda self, **kwargs: result("helper_install", kwargs),
    )
    monkeypatch.setattr(Engine, "helper_enable", lambda self: result("helper_enable"))
    monkeypatch.setattr(Engine, "helper_disable", lambda self: result("helper_disable"))
    monkeypatch.setattr(Engine, "helper_remove", lambda self: result("helper_remove"))
    return calls


def test_cli_daemon_and_mcp_share_every_helper_engine_method(
    helper_method_spy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_mod.Engine,
        "_connect_target",
        lambda _engine, serial=None: FakeDevice(serial=serial or "helper-surface"),
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    for argv in (
        ["helper", "status"],
        ["helper", "install", "--reinstall", "--force"],
        ["helper", "enable"],
        ["helper", "disable"],
        ["helper", "remove"],
    ):
        result = runner.invoke(app, ["--no-lease", *argv])
        assert result.exit_code == 0, result.stderr

    engine = Engine(make_config(lease={"enabled": False}), device=FakeDevice())
    assert dispatch(engine, {"cmd": "helper_status", "args": {}})["ok"]
    assert dispatch(
        engine,
        {"cmd": "helper_install", "args": {"reinstall": True, "force": True}},
    )["ok"]
    for name in ("helper_enable", "helper_disable", "helper_remove"):
        assert dispatch(engine, {"cmd": name, "args": {}})["ok"]

    assert mcp_dispatch(engine, "helper_status", {})["ok"]
    assert mcp_dispatch(
        engine, "helper_install", {"reinstall": True, "force": True}
    )["ok"]
    for name in ("helper_enable", "helper_disable", "helper_remove"):
        assert mcp_dispatch(engine, name, {})["ok"]

    counts = Counter(name for name, _ in helper_method_spy)
    assert counts == {
        "helper_status": 3,
        "helper_install": 3,
        "helper_enable": 3,
        "helper_disable": 3,
        "helper_remove": 3,
    }
    assert "helper_install" in _LONG_POLL_COMMANDS
    assert "helper_enable" in _LONG_POLL_COMMANDS
    assert "helper_remove" in _LONG_POLL_COMMANDS
    assert {
        "helper_status",
        "helper_install",
        "helper_enable",
        "helper_disable",
        "helper_remove",
    } <= {tool.name for tool in _tool_definitions()}
