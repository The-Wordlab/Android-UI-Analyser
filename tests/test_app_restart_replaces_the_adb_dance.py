"""`aua app restart` exists so no test setup has to shell out to adb.

Every harness in and around this project hand-rolled the same two lines to get an app back to a
known screen — `adb shell am force-stop <pkg>` then `adb shell am start -n <pkg>/<activity>` —
usually with a blind `sleep` after them. That is three tools (adb, the shell, a guess) to do one
thing aua already had both halves of, and the sleep is the same bad habit `--until` exists to
kill: `aua app restart-and-analyze <pkg> --activity <act> --until 'rid:<something>'` is the whole
sequence, and it reports `await_outcome` instead of hoping.

Data is deliberately preserved. A reset that also wiped feature-flag overrides and the login
session would destroy the preconditions it is being used to establish; `clear --yes` is the
explicit, confirmed way to ask for that.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser import cli as cli_mod
from android_ui_analyser.cli import app
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice

runner = CliRunner()

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record what the CLI asks the engine to do, without touching a device."""
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: FakeDevice(hierarchy_xml=_XML))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    original = engine_mod.Engine.app

    def spy(self: Any, action: str, **kwargs: Any) -> Any:
        recorded.append({"action": action, **kwargs})
        return original(self, action, **kwargs)

    monkeypatch.setattr(engine_mod.Engine, "app", spy)
    return recorded


def test_restart_is_a_stop_then_a_launch(calls: list[dict[str, Any]]) -> None:
    result = runner.invoke(app, ["app", "restart", "com.example.app"])

    assert result.exit_code == 0, result.output
    assert [c["action"] for c in calls] == ["stop", "launch"], calls
    assert calls[0]["package"] == "com.example.app"
    assert calls[1]["package"] == "com.example.app"


def test_restart_pins_the_activity_it_was_given(calls: list[dict[str, Any]]) -> None:
    """Dev builds ship more than one launcher, so a bare launch opens the wrong one."""
    runner.invoke(app, ["app", "restart", "com.example.app", "--activity", ".RealEntry"])

    launch = next(c for c in calls if c["action"] == "launch")
    assert launch["activity"] == ".RealEntry"


def test_restart_never_wipes_app_data(calls: list[dict[str, Any]]) -> None:
    runner.invoke(app, ["app", "restart", "com.example.app"])

    launch = next(c for c in calls if c["action"] == "launch")
    assert launch["clear_state"] is False, "a reset must not destroy flags or the session"


def test_the_and_analyze_spelling_returns_the_screen(calls: list[dict[str, Any]]) -> None:
    runner.invoke(app, ["app", "restart-and-analyze", "com.example.app"])

    launch = next(c for c in calls if c["action"] == "launch")
    assert launch["observe"] is True, "the whole point is not needing a follow-up analyze"


def test_restart_without_a_package_says_so(calls: list[dict[str, Any]]) -> None:
    result = runner.invoke(app, ["app", "restart"])

    assert result.exit_code != 0
    combined = result.output + str(result.stderr or "")
    assert "package" in combined, combined
    assert not calls, "nothing should have been done to the device"


def test_plain_app_lifecycle_commands_use_the_shared_routed_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI app calls must reach daemon journaling/coaching like MCP and analyzed actions do."""
    routed: list[dict[str, Any]] = []

    def route(_engine: Any, method: str, **kwargs: Any) -> ActionResult:
        routed.append({"method": method, **kwargs})
        return ActionResult(ok=True, action=f"app-{kwargs['action']}")

    monkeypatch.setattr(cli_mod, "_route", route)

    stopped = runner.invoke(app, ["app", "stop", "com.example.app"])
    launched = runner.invoke(app, ["app", "launch", "com.example.app"])

    assert stopped.exit_code == 0, stopped.output
    assert launched.exit_code == 0, launched.output
    assert [(item["method"], item["action"]) for item in routed] == [
        ("app", "stop"),
        ("app", "launch"),
    ]
