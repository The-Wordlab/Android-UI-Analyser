"""High-value device/action primitives: clear/grant, hide-keyboard, double-tap, ExitCode.INTERNAL."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ExitCode
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


def test_exit_code_internal_is_one() -> None:
    assert ExitCode.INTERNAL == 1
    # The crash we hit: ExitCode(1) must resolve (was ValueError before INTERNAL existed).
    assert ExitCode(1) is ExitCode.INTERNAL


def test_app_clear_and_grant() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    eng = _engine(dev)
    assert eng.app("clear", package="com.x", confirmed=True).ok
    assert ("clear_app", ("com.x",)) in dev.calls
    assert eng.app("grant", package="com.x").ok
    assert ("grant_permissions", ("com.x",)) in dev.calls
    # Aliases Maestro users will try.
    eng.app("clear-state", package="com.x", confirmed=True)
    eng.app("grant-permissions", package="com.x")
    assert sum(1 for c, _ in dev.calls if c == "clear_app") == 2


def test_hide_keyboard_and_double_tap() -> None:
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev)
    analyzed = eng.analyze(source="hierarchy")
    btn = next(e for e in analyzed.elements if e.text == "Continue")

    assert eng.double_tap(btn.id, observe=False).ok
    assert ("double_click", (btn.center[0], btn.center[1])) in dev.calls

    assert eng.hide_keyboard(observe=False).ok
    assert ("hide_keyboard", ()) in dev.calls


def test_cli_app_clear_and_hide_keyboard(monkeypatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: dev)

    r = runner.invoke(app, ["app", "clear", "com.x", "--yes"])
    assert r.exit_code == 0, r.stderr
    assert ("clear_app", ("com.x",)) in dev.calls

    r2 = runner.invoke(app, ["app", "grant", "com.x"])
    assert r2.exit_code == 0, r2.stderr

    r3 = runner.invoke(app, ["hide-keyboard-and-analyze", "--no-observe"])
    assert r3.exit_code == 0, r3.stderr
    assert ("hide_keyboard", ()) in dev.calls

    r4 = runner.invoke(app, ["double-tap-and-analyze", "--rid", "continue_btn", "--no-observe"])
    assert r4.exit_code == 0, r4.stderr
    assert any(c == "double_click" for c, _ in dev.calls)


def test_cli_internal_error_exits_one_with_structured_stderr(monkeypatch) -> None:
    """Regression: generic handler must not crash with ValueError: 1 is not a valid ExitCode."""

    def boom(serial: str | None = None):  # pragma: no cover - exercised via CLI
        raise RuntimeError("simulated daemon permission failure")

    monkeypatch.setattr(
        engine_mod.Engine,
        "_connect_target",
        lambda _engine, serial=None: boom(serial),
    )
    # `devices` builds an engine; force a path that connects.
    r = runner.invoke(app, ["app", "foreground"])
    assert r.exit_code == 1
    err = json.loads(r.stderr)
    assert err["error"]["code"] == "internal_error"
    assert "simulated daemon permission failure" in err["error"]["message"]
