"""AC checklist suite runner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.errors import ExitCode
from android_ui_analyser.suite import parse_suite, run_suite
from conftest import FakeDevice, make_engine

runner = CliRunner()

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Notifications"
        resource-id="com.test.app:id/appsHubNotifications" clickable="false" enabled="true"
        bounds="[40,100][1040,180]"/>
  <node index="1" class="android.widget.TextView" text="Hi there"
        resource-id="com.test.app:id/greeting" clickable="false" enabled="true"
        bounds="[40,200][1040,280]"/>
  <node index="2" class="android.widget.TextView" text="Done"
        resource-id="com.test.app:id/done" clickable="true" enabled="true"
        bounds="[40,300][1040,380]"/>
</hierarchy>"""

_SUITE = """
name: notifications_ac
checks:
  - has: "Notifications"
  - expect:
      rid: appsHubNotifications
      exists: true
  - expect:
      text: "Hi"
      exists: true
      match: contains
  - wait_for: "Done"
"""


def test_parse_suite() -> None:
    suite = parse_suite(_SUITE)
    assert suite.name == "notifications_ac"
    assert len(suite.checks) == 4
    assert suite.checks[0].kind == "has"
    assert suite.checks[1].kind == "expect" and suite.checks[1].rid == "appsHubNotifications"
    assert suite.checks[2].match == "contains"
    assert suite.checks[3].kind == "wait_for"


def test_run_suite_pass() -> None:
    eng = make_engine(
        device=FakeDevice(
            hierarchy_xml=_XML,
            text_index={"Notifications": (40, 100, 1040, 180), "Done": (40, 300, 1040, 380)},
            resource_index={"com.test.app:id/appsHubNotifications": (40, 100, 1040, 180)},
        )
    )
    result = run_suite(eng, parse_suite(_SUITE))
    assert result.ok
    assert result.passed == 4
    assert result.failed == 0


def test_run_suite_stops_on_fail() -> None:
    eng = make_engine(device=FakeDevice(hierarchy_xml=_XML, text_index={}))
    suite = parse_suite(
        """
name: fail_early
checks:
  - has: "Missing"
  - has: "AlsoMissing"
"""
    )
    result = run_suite(eng, suite)
    assert not result.ok
    assert result.stopped_early
    assert len(result.results) == 1
    assert result.failed == 1

    result2 = run_suite(eng, suite, continue_on_fail=True)
    assert not result2.ok
    assert not result2.stopped_early
    assert len(result2.results) == 2


def test_run_suite_launches_app() -> None:
    dev = FakeDevice(
        hierarchy_xml=_XML,
        text_index={"Notifications": (40, 100, 1040, 180), "Done": (40, 300, 1040, 380)},
        resource_index={"com.test.app:id/appsHubNotifications": (40, 100, 1040, 180)},
    )
    eng = make_engine(device=dev)
    suite = parse_suite(
        """
name: with_app
app: co.example.app
checks:
  - has: "Notifications"
"""
    )
    result = run_suite(eng, suite)
    assert result.ok
    assert ("launch_app", ("co.example.app",)) in dev.calls


def test_cli_suite_run_pass_and_fail(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "ac.yaml"
    path.write_text(_SUITE, encoding="utf-8")
    dev = FakeDevice(
        hierarchy_xml=_XML,
        text_index={"Notifications": (40, 100, 1040, 180), "Done": (40, 300, 1040, 380)},
        resource_index={"com.test.app:id/appsHubNotifications": (40, 100, 1040, 180)},
    )
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    r = runner.invoke(app, ["suite", "run", str(path)])
    assert r.exit_code == 0, r.stderr
    assert "PASS" in r.stdout

    r2 = runner.invoke(app, ["suite", "run", str(path), "--json"])
    assert r2.exit_code == 0, r2.stderr
    body = json.loads(r2.stdout)
    assert body["ok"] is True
    assert body["passed"] == 4

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        'name: bad\nchecks:\n  - has: "Nope"\n  - has: "AlsoNope"\n',
        encoding="utf-8",
    )
    r3 = runner.invoke(app, ["suite", "run", str(bad)])
    assert r3.exit_code == int(ExitCode.ASSERTION)
    assert "FAIL" in r3.stdout

    r4 = runner.invoke(app, ["suite", "run", str(bad), "--continue", "--json"])
    assert r4.exit_code == int(ExitCode.ASSERTION)
    body4 = json.loads(r4.stdout)
    assert body4["failed"] == 2
    assert body4["stopped_early"] is False


def test_cli_suite_stdin(monkeypatch) -> None:
    dev = FakeDevice(
        hierarchy_xml=_XML,
        text_index={"Notifications": (40, 100, 1040, 180)},
    )
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    yaml_text = 'name: stdin_ac\nchecks:\n  - has: "Notifications"\n'
    r = runner.invoke(app, ["suite", "run", "-", "--json"], input=yaml_text)
    assert r.exit_code == 0, r.stderr
    assert json.loads(r.stdout)["ok"] is True
