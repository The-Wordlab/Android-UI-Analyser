"""A malformed postcondition must never become a successful device mutation first."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import _parse_await_terms
from android_ui_analyser.errors import UsageError
from conftest import FakeDevice

BUTTON_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.Button" text="Launch demo"
        resource-id="com.example.fiction:id/launch_demo" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


@pytest.fixture
def cli_device(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=BUTTON_XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_MEMORY__DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    return device


def test_invalid_global_until_causes_zero_device_actions(cli_device: FakeDevice) -> None:
    result = CliRunner().invoke(
        app,
        ["--until", "not-a-field", "tap-and-analyze", "--rid", "launch_demo"],
    )

    assert result.exit_code == 2
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "usage"
    assert cli_device.calls == []
    assert cli_device.hierarchy_calls == 0, "preflight must happen before even reading the device"


def test_negative_only_global_until_causes_zero_device_actions(cli_device: FakeDevice) -> None:
    result = CliRunner().invoke(
        app,
        ["--until", "!text:Loading", "tap-and-analyze", "--rid", "launch_demo"],
    )

    assert result.exit_code == 2
    error = json.loads(result.stderr)["error"]
    assert "positive arrival" in error["message"]
    assert "standalone" in error["hint"]
    assert cli_device.calls == []
    assert cli_device.hierarchy_calls == 0


def test_negative_only_standalone_predicate_remains_valid() -> None:
    terms = _parse_await_terms("!text:Loading")

    assert len(terms) == 1 and terms[0].negated is True


def test_negative_only_global_until_is_accepted_by_standalone_wait(
    cli_device: FakeDevice,
) -> None:
    result = CliRunner().invoke(
        app,
        ["--until", "!text:Loading", "--until-timeout", "0", "wait-and-analyze"],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    # Accepted, and named for the evidence it actually has: every term was negated, so this
    # proves what the caller left is gone and nothing about what arrived. It used to report
    # `satisfied`, which a caller could reasonably read as arrival proof.
    assert payload["await_outcome"] == "absence-satisfied"
    assert payload["ok"] is True
    mutation_calls = {"click", "long_click", "double_click", "input_text", "press", "swipe"}
    assert not any(name in mutation_calls for name, _args in cli_device.calls)


def test_a_literal_comma_has_an_explicit_escape() -> None:
    terms = _parse_await_terms(r"text:Hello\, explorer,!text:Loading")

    assert [(term.by, term.value, term.negated) for term in terms] == [
        ("text", "Hello, explorer", False),
        ("text", "Loading", True),
    ]


def test_an_incomplete_predicate_escape_is_a_usage_error() -> None:
    with pytest.raises(UsageError) as caught:
        _parse_await_terms("text:unfinished\\")
    assert caught.value.code == "usage"
