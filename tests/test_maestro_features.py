"""Maestro-parity device controls: clipboard, location, flows repeat/retry, etc."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import parse_flow_yaml
from conftest import FakeDevice, make_config

runner = CliRunner()

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


def _engine(dev: FakeDevice) -> Engine:
    return Engine(make_config(), device=dev)


def test_clipboard_paste_copy_erase() -> None:
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev)
    eng.clipboard_set("hello")
    assert dev.get_clipboard() == "hello"
    analyzed = eng.analyze(source="hierarchy")
    btn = next(e for e in analyzed.elements if e.text == "Continue")
    copied = eng.copy_text(btn.id)
    assert copied.detail == "Continue"
    assert dev.get_clipboard() == "Continue"
    eng.paste(observe=False)
    assert ("paste", ()) in dev.calls
    analyzed = eng.analyze(source="hierarchy")
    btn = next(e for e in analyzed.elements if e.text == "Continue")
    eng.erase(btn.id, chars=3, observe=False)
    assert ("erase_chars", (3,)) in dev.calls


def test_device_extras_engine() -> None:
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev)
    eng.location_set(37.42, -122.08)
    assert dev._location == (37.42, -122.08)
    eng.orientation_set("landscape")
    assert dev._orientation == "landscape"
    eng.airplane_set(True)
    assert dev._airplane is True
    eng.airplane_toggle()
    assert dev._airplane is False
    remote = eng.media_add(__file__).detail
    assert remote.endswith(Path(__file__).name)
    eng.record_start()
    assert dev._recording is not None
    saved = eng.record_stop("/tmp/aua_test.mp4").detail
    assert saved == "/tmp/aua_test.mp4"
    eng.clock_set(timestamp_ms=1_700_000_000_000)
    assert ("set_clock", (1_700_000_000_000,)) in dev.calls


def test_app_launch_clear_and_kill() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    eng = _engine(dev)
    eng.app("launch", package="com.x", clear_state=True, confirmed=True)
    assert ("clear_app", ("com.x",)) in dev.calls
    assert any(c[0] == "launch_app" for c in dev.calls)
    eng.app("kill", package="com.x")
    assert ("stop_app", ("com.x",)) in dev.calls


def test_cli_maestro_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    assert runner.invoke(app, ["clipboard", "set", "hi"]).exit_code == 0
    assert runner.invoke(app, ["clipboard", "get"]).exit_code == 0
    assert runner.invoke(app, ["paste-and-analyze", "--no-observe"]).exit_code == 0
    assert runner.invoke(app, ["copy", "--text", "Continue"]).exit_code == 0
    assert runner.invoke(app, ["location", "set", "37.42,-122.08"]).exit_code == 0
    assert runner.invoke(app, ["orientation", "set", "portrait"]).exit_code == 0
    assert runner.invoke(app, ["airplane", "on"]).exit_code == 0
    assert runner.invoke(app, ["airplane", "toggle"]).exit_code == 0
    assert runner.invoke(app, ["media", "add", __file__]).exit_code == 0
    assert runner.invoke(app, ["record", "start"]).exit_code == 0
    assert runner.invoke(app, ["record", "stop", "/tmp/out.mp4"]).exit_code == 0
    assert runner.invoke(app, ["clock", "set", "--ms", "1700000000000"]).exit_code == 0
    assert runner.invoke(app, ["clock", "restore"]).exit_code == 0
    assert runner.invoke(app, ["app", "launch", "com.x", "--clear", "--yes"]).exit_code == 0
    # Without --yes, clear must refuse.
    assert runner.invoke(app, ["app", "clear", "com.x"]).exit_code == 2


def test_flow_repeat_and_retry_parse() -> None:
    flow = parse_flow_yaml(
        """
schema_version: 1
name: blocks
steps:
  - repeat:
      times: 2
      steps:
        - tap: Continue
  - retry:
      max_retries: 3
      steps:
        - key: back
"""
    )
    assert flow.steps[0].kind == "repeat"
    assert flow.steps[0].repeat == 2
    assert len(flow.steps[0].substeps) == 1
    assert flow.steps[1].kind == "retry"
    assert flow.steps[1].max_retries == 3


def test_flow_repeat_runs_substeps(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.test.app")
    eng = _engine(dev)
    flow = parse_flow_yaml(
        """
name: r
app: com.test.app
steps:
  - repeat:
      times: 2
      steps:
        - tap: Continue
"""
    )
    steps = flow.steps
    fail, _ = eng._run_steps(
        steps,
        origin_package="com.test.app",
        allow_destructive=True,
        allow_goto_steps=False,
    )
    assert fail is None
    assert sum(1 for c, _ in dev.calls if c == "click") == 2
