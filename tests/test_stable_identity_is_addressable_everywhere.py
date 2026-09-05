"""Every caller-facing surface can address an element by the identity that outlives its frame.

An integer id is a reading-order ordinal resolved through one shared cache file per device.
Any caller holding an observation produced by *another* process — the dashboard, a second
agent, a saved report — can only address it safely by ``stable_key``, so the CLI and MCP
must both accept one, not just the engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.assertions import normalize_selector
from android_ui_analyser.cli import app
from android_ui_analyser.mcp_server import _SELECTOR_PROPS, _selector_from_args
from conftest import FakeDevice

runner = CliRunner()

_HIERARCHY = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.Button" text="Continue"
        resource-id="com.example.fiction:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


@pytest.fixture
def patched_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeDevice:
    device = FakeDevice(
        hierarchy_xml=_HIERARCHY,
        resource_index={"com.example.fiction:id/continue_btn": (40, 200, 1040, 320)},
    )
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: device)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_LEASE__REGISTRY_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    return device


def test_the_cli_taps_by_stable_key(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["tap-and-analyze", "--key", "rid:continue_btn"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["action"] == "tap"
    assert any(call[0] == "click" for call in patched_device.calls)


def test_rid_flag_accepts_the_published_rid_prefix(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["tap-and-analyze", "--rid", "rid:continue_btn"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["action"] == "tap"
    assert any(call[0] == "click" for call in patched_device.calls)


def test_has_accepts_the_same_redundant_rid_prefix(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["has", "--rid", "rid:continue_btn"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["found"] is True


def test_the_cli_needs_no_prior_analyze_to_use_a_key(patched_device: FakeDevice) -> None:
    """The point of a key: it is meaningful without this process's id cache."""
    result = runner.invoke(app, ["tap-and-analyze", "--key", "rid:not_on_screen"])

    assert result.exit_code != 0
    assert "stable_key" in (result.stdout + str(result.stderr))
    assert not any(call[0] == "click" for call in patched_device.calls)


def test_mcp_advertises_stable_key_on_every_selector_tool() -> None:
    assert "stable_key" in _SELECTOR_PROPS
    assert _SELECTOR_PROPS["stable_key"]["type"] == "string"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"stable_key": "rid:continue_btn"}, {"key": "rid:continue_btn"}),
        (
            {"stable_key": "tx:49e6d8ed09", "bounds": [10, 20, 30, 40]},
            {"key": "tx:49e6d8ed09", "bounds": [10, 20, 30, 40]},
        ),
        ({"rid": "continue_btn"}, {"rid": "continue_btn", "text": None, "desc": None}),
        ({"rid": "rid:continue_btn"}, {"rid": "continue_btn", "text": None, "desc": None}),
    ],
)
def test_mcp_turns_a_stable_key_into_an_identity_selector(
    args: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert _selector_from_args(args) == expected


def test_nested_selectors_accept_the_same_published_prefix() -> None:
    assert normalize_selector({"rid": "rid:continue_btn"}) == {"rid": "continue_btn"}
    assert normalize_selector({"id": "id:continue_btn"}) == {"rid": "continue_btn"}
