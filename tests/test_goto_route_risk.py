"""Safety boundaries for replaying automatically learned ``goto`` routes."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_module
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.memory import AppMemoryStore, RouteStep, route_step_risks
from conftest import FakeDevice, make_config
from test_memory import APPS, HOME, P, _elements, _engine, _store
from test_navigation import ScriptedDevice

runner = CliRunner()


class LinkDevice(ScriptedDevice):
    """A scripted device whose fictional deeplink reaches the next hierarchy."""

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        super().open_link(uri, package=package)
        self._advance()


def _route_engine(
    tmp_path: Path,
    steps: list[RouteStep],
    *,
    serial: str,
    device_type: type[ScriptedDevice] = ScriptedDevice,
) -> tuple[Engine, ScriptedDevice]:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_route(P, "home", "catalog", steps=steps)
    device = device_type([HOME, APPS], package=P, serial=serial)
    return _engine(tmp_path, device), device


def _mutating_calls(device: FakeDevice) -> list[tuple[str, object]]:
    names = {
        "a11y_action",
        "click",
        "input_text",
        "long_click",
        "open_link",
        "press",
        "set_clipboard",
        "swipe",
    }
    return [call for call in device.calls if call[0] in names]


def test_route_risk_classifier_is_generic_and_conservative() -> None:
    settings = route_step_risks(
        RouteStep(kind="open-link", arg="fiction://flags?mode=alternate"),
        origin_package="org.example.catalog",
        destructive_labels=[],
    )
    external = route_step_risks(
        RouteStep(kind="open-link", arg="https://example.invalid/help"),
        origin_package="org.example.catalog",
        destructive_labels=[],
    )
    tap = route_step_risks(
        RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog"),
        origin_package="org.example.catalog",
        destructive_labels=[],
    )

    assert [risk["code"] for risk in settings] == ["settings_mutation"]
    assert [risk["code"] for risk in external] == ["external_navigation"]
    assert tap == []


def test_goto_refuses_whole_route_before_safe_prefix_can_run(tmp_path: Path) -> None:
    engine, device = _route_engine(
        tmp_path,
        [
            RouteStep(kind="tap", label="Apps", resource_id="nav_apps"),
            RouteStep(kind="open-link", arg="fiction://flags?mode=alternate"),
        ],
        serial="risk-prefix",
    )

    result = engine.goto("catalog")

    assert result["ok"] is False and result["code"] == "unsafe_route"
    assert result["required_opt_in"] == ["--allow-unsafe"]
    assert result["route"][0]["risk"] == "requires_opt_in"
    assert result["risks"][0]["code"] == "settings_mutation"
    assert "configuration" in result["risks"][0]["reason"]
    assert "No route step was executed" in result["hint"]
    assert _mutating_calls(device) == []


def test_goto_refuses_state_configuration_step_before_dispatch(tmp_path: Path) -> None:
    engine, device = _route_engine(
        tmp_path,
        [
            RouteStep(kind="flags-apply", arg="fixtures/alternate.yaml"),
            RouteStep(kind="tap", label="Apps", resource_id="nav_apps"),
        ],
        serial="risk-config",
    )

    result = engine.goto("catalog")

    assert result["ok"] is False and result["code"] == "unsafe_route"
    assert result["risks"][0]["code"] == "settings_mutation"
    assert result["route"][0]["risks"][0]["reason"] == "flags-apply changes app configuration"
    assert _mutating_calls(device) == []


def test_goto_plan_discloses_external_link_without_acting(tmp_path: Path) -> None:
    engine, device = _route_engine(
        tmp_path,
        [RouteStep(kind="open-link", arg="https://example.invalid/help")],
        serial="risk-plan",
    )

    result = engine.goto("catalog", plan=True)

    assert result["ok"] is True and result["plan"] is True
    assert result["route"][0]["risk"] == "requires_opt_in"
    assert result["route"][0]["risks"][0]["code"] == "external_navigation"
    assert _mutating_calls(device) == []


def test_goto_runs_disclosed_deeplink_only_with_explicit_opt_in(tmp_path: Path) -> None:
    engine, device = _route_engine(
        tmp_path,
        [RouteStep(kind="open-link", arg="fiction://catalog")],
        serial="risk-opt-in",
        device_type=LinkDevice,
    )

    refused = engine.goto("catalog")
    assert refused["code"] == "unsafe_route"
    assert _mutating_calls(device) == []

    allowed = engine.goto("catalog", allow_unsafe=True)
    assert allowed["ok"] is True and allowed["arrived"] is True
    assert sum(call[0] == "open_link" for call in device.calls) == 1


def test_goto_still_executes_navigation_only_tap(tmp_path: Path) -> None:
    engine, device = _route_engine(
        tmp_path,
        [RouteStep(kind="tap", label="Apps", resource_id="nav_apps")],
        serial="risk-safe-tap",
    )

    result = engine.goto("catalog")

    assert result["ok"] is True and result["arrived"] is True
    assert result["route"][0]["risk"] == "safe_navigation"
    assert sum(call[0] == "click" for call in device.calls) == 1


def test_cli_goto_requires_the_explicit_unsafe_opt_in(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    store = AppMemoryStore(make_config().memory)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_route(
        P,
        "home",
        "catalog",
        steps=[RouteStep(kind="open-link", arg="fiction://catalog")],
    )
    device = LinkDevice([HOME, APPS], package=P, serial="risk-cli")
    monkeypatch.setattr(engine_module, "connect", lambda serial=None: device)

    refused = runner.invoke(app, ["--format", "compact", "goto", "catalog"])
    assert refused.exit_code == 1
    assert json.loads(refused.stdout)["code"] == "unsafe_route"
    assert _mutating_calls(device) == []

    allowed = runner.invoke(
        app,
        ["--format", "compact", "goto", "catalog", "--allow-unsafe"],
    )
    assert allowed.exit_code == 0, allowed.stderr
    assert json.loads(allowed.stdout)["arrived"] is True


def test_mcp_goto_schema_exposes_deliberate_unsafe_opt_in(tmp_path: Path) -> None:
    server = build_server(_engine(tmp_path, FakeDevice(hierarchy_xml=HOME, package=P)))

    async def run() -> dict[str, object]:
        async with create_connected_server_and_client_session(server) as client:
            tools = await client.list_tools()
            goto_tool = next(tool for tool in tools.tools if tool.name == "goto")
            return goto_tool.inputSchema

    schema = anyio.run(run)
    allow_unsafe = schema["properties"]["allow_unsafe"]  # type: ignore[index]
    assert allow_unsafe["default"] is False
    assert "reviewing the preview" in allow_unsafe["description"]
