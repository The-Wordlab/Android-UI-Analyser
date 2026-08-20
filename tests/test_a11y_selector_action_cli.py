"""Accessibility actions accept ACTION as the only positional when a selector names the node."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from conftest import FakeDevice

XML = """<hierarchy rotation="0">
  <node class="android.widget.Button" text="Expand fiction"
        resource-id="com.example.fiction:id/expand_button" clickable="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


@pytest.mark.parametrize("command", ["action", "action-and-analyze"])
def test_selector_only_a11y_action_parses_and_dispatches(
    tmp_path, monkeypatch, command: str
) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_MEMORY__DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    result = CliRunner().invoke(
        app,
        ["a11y", command, "--rid", "expand_button", "CLICK", "--no-observe"],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["action"] == "a11y-action"
    assert any(name == "a11y_action" and args[-1] == "CLICK" for name, args in device.calls)


def test_legacy_id_then_action_order_still_works(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_MEMORY__DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    runner = CliRunner()
    observed = runner.invoke(app, ["analyze", "--source", "hierarchy"])
    element_id = json.loads(observed.stdout)["elements"][0]["id"]

    result = runner.invoke(
        app,
        ["a11y", "action", str(element_id), "CLICK", "--no-observe"],
    )

    assert result.exit_code == 0, result.stderr
    assert any(name == "a11y_action" and args[-1] == "CLICK" for name, args in device.calls)
