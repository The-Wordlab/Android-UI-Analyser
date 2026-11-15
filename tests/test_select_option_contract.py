"""Transport, platform and validation boundaries for one semantic native choice."""
from __future__ import annotations

import asyncio
import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import cli
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    InvalidPlatformCapabilityError,
    UnsupportedPlatformCapabilityError,
    UsageError,
)
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree
from android_ui_analyser.platforms.ios import IOSPlatform
from android_ui_analyser.schema import Element
from conftest import make_config, make_screen_image
from test_platform_runtime import _NeutralAdapter, _NeutralRuntime


class ChoiceRuntime(_NeutralRuntime):
    def __init__(self):
        self.choices = []

    def screenshot(self):
        return make_screen_image(width=300, height=600)

    def select_option(self, element, *, value=None, label=None):
        self.choices.append((element, value, label))
        return "two" if value is None else value

    def click(self, *args):
        raise AssertionError("semantic select must not use a generic coordinate tap")

    def send_text(self, *args, **kwargs):
        raise AssertionError("semantic select must not type into global focus")


class ChoicePlatform(_NeutralAdapter):
    capabilities = frozenset({"ui.tree", "ui.screenshot", "ui.select"})

    def __init__(self, config, runtime):
        super().__init__(config)
        self.runtime = runtime

    def connect(self, target_id=None):
        return self.runtime

    def capture_screenshot(self, runtime):
        return runtime.screenshot()

    def normalize_tree(self, *args, **kwargs):
        return NormalizedTree([Element(
            id=0, type="ComboBox", resource_id="choice", text="Choose",
            bounds=(10, 10, 100, 50), center=(55, 30), clickable=True, enabled=True,
        )])


def _engine(tmp_path):
    config = make_config(device={"serial": "neutral-target"}, memory={"enabled": False},
                         lease={"enabled": False}, teardown={"enabled": False})
    config.cache.dir = str(tmp_path / "cache")
    config.daemon.enabled = False
    runtime = ChoiceRuntime()
    return Engine(config, platform=ChoicePlatform(config, runtime)), runtime


def test_select_routes_through_neutral_runtime_without_keyboard_or_android(tmp_path):
    engine, runtime = _engine(tmp_path)
    try:
        result = engine.select_option(selector={"rid": "choice"}, value="", observe=False)
        assert result.ok
        assert len(runtime.choices) == 1
        target, value, label = runtime.choices[0]
        assert target.resource_id == "choice"
        assert value == ""  # empty is a real supplied option value
        assert label is None
    finally:
        engine.close()


@pytest.mark.parametrize("choice", [{}, {"value": "two", "label": "Second"}])
def test_exactly_one_choice_is_required_before_platform_mutation(tmp_path, choice):
    engine, runtime = _engine(tmp_path)
    try:
        with pytest.raises(UsageError):
            engine.select_option(selector={"rid": "choice"}, **choice)
        assert runtime.choices == []
    finally:
        engine.close()


def test_declared_select_capability_requires_runtime_implementation(tmp_path):
    engine, _ = _engine(tmp_path)
    try:
        with pytest.raises(InvalidPlatformCapabilityError):
            engine.platform.validate_runtime(_NeutralRuntime())
    finally:
        engine.close()


@pytest.mark.parametrize("adapter", [AndroidPlatform, IOSPlatform, _NeutralAdapter])
def test_unclaimed_select_is_explicitly_unsupported_without_native_fallback(adapter):
    platform = adapter(make_config())
    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        platform.runtime_capability("ui.select", ChoiceRuntime())
    assert caught.value.code == "platform_capability_unsupported"


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize("choice", [{"value": "two"}, {"label": "Second"}])
def test_transports_reach_the_same_engine_selection(tmp_path, monkeypatch, surface, choice):
    engine, runtime = _engine(tmp_path)
    try:
        observation = engine.analyze(source="hierarchy", with_ocr=False)
        handle = observation.elements[0].handle
        assert handle and handle.startswith("el:")
        if surface == "cli":
            monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
            monkeypatch.setattr(cli.GlobalOpts, "load", lambda self: engine.config)
            key, value = next(iter(choice.items()))
            result = CliRunner().invoke(
                cli.app, ["select-option", handle, f"--{key}", value, "--no-observe"]
            )
            assert result.exit_code == 0, result.output
            payload = json.loads(result.stdout)
        else:
            async def invoke():
                async with create_connected_server_and_client_session(build_server(engine)) as client:
                    result = await client.call_tool(
                        "select_option", {"id": handle, "observe": False, **choice}
                    )
                    assert not result.isError, result
                    return json.loads(next(x.text for x in result.content if x.type == "text"))
            payload = asyncio.run(invoke())
        assert payload["ok"] is True
        assert len(runtime.choices) == 1
        assert runtime.choices[0][1:] == (choice.get("value"), choice.get("label"))
    finally:
        engine.close()
