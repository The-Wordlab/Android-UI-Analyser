"""Public key validation must admit the aliases the Android runtime implements."""

from __future__ import annotations

import json

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.android_device import (
    _KEYCODE_NAMES,
    _PRESS_ALIASES,
    Uiautomator2Device,
)
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from conftest import FakeDevice, make_config


class KeyDevice(FakeDevice):
    """Exercise real Android key dispatch with an in-memory transport boundary."""

    press = Uiautomator2Device.press

    def shell(self, cmd):
        self.calls.append(("shell", (cmd,)))
        return ""


def test_every_existing_runtime_key_alias_passes_android_normalization():
    platform = AndroidPlatform(make_config())
    device = KeyDevice()
    assert _KEYCODE_NAMES.keys() == _PRESS_ALIASES.keys()
    for alias, keycode in _KEYCODE_NAMES.items():
        candidate = platform.normalize_key(f"  {alias.upper()}  ")
        device.press(candidate)
        assert device.calls[-1] == ("shell", (f"input keyevent {keycode}",))
    assert len(device.calls) == len(_KEYCODE_NAMES)


@pytest.mark.parametrize("surface", ["engine", "cli", "mcp"])
@pytest.mark.parametrize(
    ("name", "keycode"),
    [("paste", "KEYCODE_PASTE"), ("backspace", "KEYCODE_DEL"), ("not_a_key", None)],
)
def test_key_aliases_use_shared_validation_dispatch_and_observation(
    surface, name, keycode, tmp_path, monkeypatch
):
    device = KeyDevice()
    cfg = make_config(
        lease={"enabled": False},
        memory={"enabled": False},
        capture={"enabled": False},
        daemon={"enabled": False},
        output={"with_image": False},
    )
    observation_calls = []
    observation = AnalyzeResult(
        screen=Screen(width=1080, height=2400, source="hierarchy"),
        elements=[],
        meta=Meta(duration_ms=0, tier_used="hierarchy", path="hierarchy"),
    )

    def observe(self, result, requested, with_image=None, **kwargs):
        observation_calls.append(requested)
        assert requested is True
        result.observation = observation
        result.observation_present = True
        return result

    monkeypatch.setattr(Engine, "_observe", observe)
    monkeypatch.setattr(Engine, "_connect_target", lambda self, serial=None: device)
    if surface == "cli":
        path = tmp_path / "aua.json"
        path.write_text(cfg.model_dump_json())
        monkeypatch.setenv("AUA_CONFIG", str(path))
        outcome = CliRunner().invoke(app, ["--format", "json", "key-and-analyze", name])
        assert outcome.exit_code == (0 if keycode else 2), outcome.output
        payload = json.loads(outcome.stdout if keycode else outcome.stderr)
    else:
        engine = Engine(cfg, device=device)
        try:
            if surface == "engine":
                if keycode:
                    payload = engine.key(name).model_dump(mode="json")
                else:
                    with pytest.raises(UsageError) as caught:
                        engine.key(name)
                    payload = caught.value.to_dict()
            else:
                server = build_server(engine)

                async def run():
                    async with create_connected_server_and_client_session(server) as client:
                        outcome = await client.call_tool("key_and_analyze", {"name": name})
                        assert len(outcome.content) == 1
                        return json.loads(outcome.content[0].text)

                payload = anyio.run(run)
        finally:
            engine.close()

    shells = [call for call in device.calls if call[0] == "shell"]
    if keycode:
        assert payload["ok"] is True and payload["detail"] == name
        assert payload["observation_present"] is True
        assert shells == [("shell", (f"input keyevent {keycode}",))]
        assert observation_calls == [True]
    else:
        assert payload["error"]["code"] == "usage"
        assert "unknown key" in payload["error"]["message"]
        assert shells == [] and observation_calls == []
        assert device.screenshot_calls == 0 and device.hierarchy_calls == 0
