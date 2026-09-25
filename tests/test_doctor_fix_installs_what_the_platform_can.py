"""``aua doctor --fix`` installs the host tooling a platform knows how to install.

The iOS adapter needs AXe, which lives in a Homebrew tap that a current Homebrew refuses to
load until it is trusted. Doctor used to say "axe missing" and point at a one-liner that
failed on that trust step; now it can do the three steps itself and report each one.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import android_ui_analyser.cli as cli
from android_ui_analyser.cli import app
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms.ios import IOSPlatform

runner = CliRunner()


@pytest.fixture
def quiet_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the rest of doctor off the host: this is about the fix section only."""
    monkeypatch.setattr(cli, "_build_doctor_report", lambda _engine: {"checks": {}, "providers": {}})


def test_doctor_fix_reports_each_install_step(quiet_doctor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        IOSPlatform,
        "doctor_fix",
        lambda _self: {
            "fixed": [
                {
                    "tool": "axe",
                    "steps": [
                        {"command": "brew tap cameroncooke/axe", "ok": True, "detail": None},
                        {"command": "brew trust cameroncooke/axe", "ok": True, "detail": None},
                        {"command": "brew install axe", "ok": True, "detail": None},
                    ],
                }
            ],
            "skipped": [],
        },
    )

    result = runner.invoke(app, ["--platform", "ios", "doctor", "--fix"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "[FIX ] installed axe" in result.stdout
    assert "brew trust cameroncooke/axe — ok" in result.stdout

    as_json = runner.invoke(app, ["--platform", "ios", "--format", "json", "doctor", "--fix"])
    assert json.loads(as_json.stdout)["fix"]["fixed"][0]["tool"] == "axe"


def test_doctor_without_fix_installs_nothing(quiet_doctor, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(IOSPlatform, "doctor_fix", lambda _self: calls.append("fix") or {})

    result = runner.invoke(app, ["--platform", "ios", "doctor"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls == []
    assert "[FIX ]" not in result.stdout


def test_a_failed_fix_is_the_one_thing_doctor_fails_on(
    quiet_doctor, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_self: IOSPlatform) -> dict:
        raise DeviceError("`brew install axe` failed: no formula", code="ios_tool_install_failed")

    monkeypatch.setattr(IOSPlatform, "doctor_fix", boom)

    result = runner.invoke(app, ["--platform", "ios", "doctor", "--fix"])

    assert result.exit_code != 0
    assert "ios_tool_install_failed" in result.stderr
