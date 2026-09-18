"""Uninstall is confirmed, platform-neutral, and identical through CLI and MCP."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli, mcp_server
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UnsupportedPlatformCapabilityError, UsageError
from android_ui_analyser.platforms import InstalledApp
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config
from test_platform_app_install import NoInstallPlatform


class Installer(NoInstallPlatform):
    name = "fixture-installer"
    capabilities = frozenset({"app.install"})

    def installed_app(self, runtime, app_id):
        return InstalledApp(app_id=app_id, installed=True)

    def inspect_app_bundle(self, path):
        raise AssertionError("uninstall must not inspect bundles")

    def install_app_bundle(self, runtime, path, *, grant_permissions=True):
        raise AssertionError("uninstall must not install bundles")

    def uninstall_app(self, runtime, app_id):
        runtime.calls.append(("uninstall", app_id))


def test_uninstall_requires_confirmation_and_uses_the_selected_adapter():
    config = make_config(lease={"enabled": False}, teardown={"enabled": False})
    device = FakeDevice()
    engine = Engine(config, device=device, platform=Installer(config))
    with pytest.raises(UsageError, match="pass --yes"):
        engine.app("uninstall", package="com.example.app")
    with pytest.raises(UsageError, match="package"):
        engine.app("uninstall", confirmed=True)
    assert not device.calls
    result = engine.app("uninstall", package="com.example.app", confirmed=True)
    assert result.ok and result.action == "app-uninstall"
    assert device.calls == [("uninstall", "com.example.app")]


def test_uninstall_is_explicitly_unsupported_without_install_capability():
    config = make_config(lease={"enabled": False}, teardown={"enabled": False})
    device = FakeDevice()
    engine = Engine(config, device=device, platform=NoInstallPlatform(config))
    with pytest.raises(UnsupportedPlatformCapabilityError) as exc:
        engine.app("uninstall", package="com.example.app", confirmed=True)
    assert exc.value.code == "platform_capability_unsupported"
    assert not device.calls


def test_cli_and_mcp_forward_the_same_confirmation(monkeypatch):
    calls = []

    def app(action, **kwargs):
        calls.append((action, kwargs))
        return ActionResult(ok=True, action=f"app-{action}")

    engine = Engine(make_config(), device=FakeDevice())
    monkeypatch.setattr(engine, "app", app)
    monkeypatch.setattr(cli, "_route", lambda _engine, method, **kw: app(kw.pop("action"), **kw))
    refused = CliRunner().invoke(cli.app, ["app", "uninstall", "com.example.app"])
    assert refused.exit_code != 0 and "pass --yes" in refused.output
    assert not calls
    result = CliRunner().invoke(cli.app, ["app", "uninstall", "com.example.app", "--yes"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["action"] == "app-uninstall"
    mcp_server._dispatch_tool(
        engine,
        "app",
        {
            "action": "uninstall",
            "package": "com.example.app",
            "confirmed": True,
        },
    )
    assert len(calls) == 2
    for action, arguments in calls:
        assert action == "uninstall"
        assert arguments["confirmed"] is True
        assert arguments["package"] == "com.example.app"
    schema = next(tool for tool in mcp_server._tool_definitions() if tool.name == "app")
    assert schema.inputSchema["properties"]["confirmed"]["default"] is False
