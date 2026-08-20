"""A slow emulator start must look alive, never like an empty completed command.

Observed with a windowed AVD: the execution shell yielded after 6.2 seconds with no output while
QEMU was booting normally. The runner mistook that non-terminal yield for failure, launched a
duplicate, and then closed both with ``emulator stop --mine``. AUA owns the command contract, so
it must explicitly say that startup is still running and that the same process should be polled.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli


class _SlowVirtualDevices:
    def start(self, avd: str | None, **kwargs: Any) -> dict[str, Any]:
        time.sleep(0.04)
        return {
            "ok": True,
            "action": "emulator-start",
            "avd": avd,
            "serial": "emulator-9998",
            "headless": kwargs["headless"],
            "proxy_cleanup": {
                "ok": True,
                "checked": True,
                "cleared": False,
                "state_before": "unproxied",
                "state_after": "unproxied",
            },
        }


def test_slow_cli_start_reports_progress_without_corrupting_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _SlowVirtualDevices()
    monkeypatch.setattr(cli, "_platform_capability", lambda _ctx, _name: service)
    monkeypatch.setattr(cli, "_EMULATOR_PROGRESS_INTERVAL_S", 0.01)

    result = CliRunner().invoke(
        cli.app,
        [
            "--format",
            "json",
            "emulator",
            "start",
            "--avd",
            "Example_API_34",
            "--windowed",
            "--wait",
            "180",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["serial"] == "emulator-9998"
    progress = [
        json.loads(line.removeprefix("AUA_PROGRESS "))
        for line in result.stderr.splitlines()
        if line.startswith("AUA_PROGRESS ")
    ]
    assert [event["stage"] for event in progress][0] == "starting"
    assert "waiting" in [event["stage"] for event in progress]
    assert [event["stage"] for event in progress][-1] == "completed"
    assert progress[0]["mode"] == "windowed"
    assert "poll the same process" in progress[0]["message"]
    assert "do not retry" in next(
        event["message"] for event in progress if event["stage"] == "waiting"
    )
    assert "AUA_PROGRESS" not in result.stdout


def test_interrupted_start_says_cancelled_instead_of_promising_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        pytest.raises(KeyboardInterrupt),
        cli._emulator_start_progress(
            avd="Example_API_34",
            headless=False,
            wait_s=180,
        ),
    ):
        raise KeyboardInterrupt

    progress = [
        json.loads(line.removeprefix("AUA_PROGRESS "))
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("AUA_PROGRESS ")
    ]
    assert progress[-1]["stage"] == "cancelled"
    assert "no final result" in progress[-1]["message"]
