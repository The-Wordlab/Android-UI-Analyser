"""Package status and read-only shell stay leased, structured, and platform-gated."""

from __future__ import annotations

import io
import json
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.cli as cli_mod
import android_ui_analyser.device as device_mod
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.device import (
    Uiautomator2Device,
    _android_shell_is_read_only,
)
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UnsupportedPlatformCapabilityError, UsageError
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from android_ui_analyser.mcp_server import _tool_definitions
from android_ui_analyser.platforms import InstalledApp
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.schema import ShellResult
from conftest import FakeDevice, make_config

runner = CliRunner()


class FakeShellProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class ReadOnlyShellDevice(FakeDevice):
    def run_read_only_shell(self, argv: list[str], *, timeout_s: float = 30.0) -> ShellResult:
        self.calls.append(("run_read_only_shell", (list(argv), timeout_s)))
        return ShellResult(
            ok=True,
            serial=self.serial,
            argv=list(argv),
            stdout="package:/data/app/example/base.apk\n",
            exit_code=0,
            duration_ms=4,
        )


class StatusShellPlatform(PlatformAdapter):
    name = "status-shell-test"
    capabilities = frozenset({"app.status", "device.shell"})

    def __init__(self, config: Any, *, installed: bool = True) -> None:
        super().__init__(config)
        self.installed = installed
        self.status_calls: list[tuple[FakeDevice, str]] = []

    def connect(self, target_id: str | None = None):  # type: ignore[no-untyped-def]
        raise AssertionError("the injected target should be used")

    def list_targets(self):  # type: ignore[no-untyped-def]
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree([])

    def installed_app(self, runtime: FakeDevice, app_id: str) -> InstalledApp:  # type: ignore[override]
        self.status_calls.append((runtime, app_id))
        return InstalledApp(
            app_id=app_id,
            installed=self.installed,
            version_name="2.1.0" if self.installed else None,
            version_code="17" if self.installed else None,
        )


class NoStatusShellPlatform(StatusShellPlatform):
    name = "no-status-shell"
    capabilities = frozenset()


def _engine(tmp_path: Path, *, installed: bool = True) -> tuple[Engine, StatusShellPlatform]:
    cfg = make_config(cache={"dir": str(tmp_path)}, memory={"enabled": False})
    target = ReadOnlyShellDevice(serial="leased-emulator-5560")
    platform = StatusShellPlatform(cfg, installed=installed)
    return Engine(cfg, device=target, platform=platform), platform


def test_package_status_and_shell_use_the_injected_selected_target(tmp_path: Path) -> None:
    engine, platform = _engine(tmp_path)

    status = engine.app_status("com.example.app")
    shell = engine.shell(["pm", "path", "com.example.app"], timeout_ms=12_000)

    assert status.model_dump(mode="json") == {
        "ok": True,
        "action": "app-status",
        "package": "com.example.app",
        "installed": True,
        "serial": "leased-emulator-5560",
        "version_name": "2.1.0",
        "version_code": "17",
        "mode": "read-only",
    }
    target = platform.status_calls[0][0]
    assert target.serial == "leased-emulator-5560"
    assert shell.serial == target.serial
    assert ("run_read_only_shell", (["pm", "path", "com.example.app"], 12.0)) in target.calls


@pytest.mark.parametrize("method", ["app_status", "shell"])
def test_unsupported_platform_refuses_before_touching_the_target(
    tmp_path: Path, method: str
) -> None:
    cfg = make_config(cache={"dir": str(tmp_path)}, memory={"enabled": False})
    target = ReadOnlyShellDevice()
    engine = Engine(cfg, device=target, platform=NoStatusShellPlatform(cfg))

    with pytest.raises(UnsupportedPlatformCapabilityError) as raised:
        if method == "app_status":
            engine.app_status("com.example.app")
        else:
            engine.shell(["getprop", "ro.build.version.sdk"])

    assert raised.value.code == "platform_capability_unsupported"
    assert target.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["pm", "path", "com.example.app"],
        ["cmd", "package", "query-activities", "-a", "android.intent.action.MAIN"],
        ["getprop", "ro.build.version.sdk"],
        ["dumpsys", "package", "com.example.app"],
        ["dumpsys", "activity", "top"],
        ["settings", "get", "global", "http_proxy"],
        ["wm", "size"],
        ["logcat", "-d", "-t", "20"],
        ["pidof", "com.example.app"],
    ],
)
def test_android_read_only_shell_allow_list_accepts_diagnostics(argv: list[str]) -> None:
    assert _android_shell_is_read_only(argv)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["pm", "clear", "com.example.app"],
        ["settings", "put", "global", "http_proxy", "127.0.0.1:8080"],
        ["dumpsys", "battery", "set", "level", "5"],
        ["wm", "size", "1080x1920"],
        ["logcat", "-c"],
        ["ip", "link", "set", "wlan0", "down"],
        ["cat", "/dev/zero"],
        ["dumpsys"],
        ["dumpsys", "package"],
        ["rm", "-rf", "/sdcard/example"],
        ["sh", "-c", "id > /sdcard/output"],
        ["am", "force-stop", "com.example.app"],
    ],
)
def test_android_read_only_shell_allow_list_rejects_mutation_and_unknowns(
    argv: list[str],
) -> None:
    assert not _android_shell_is_read_only(argv)


def test_android_shell_quotes_remote_argv_and_uses_the_runtime_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = object.__new__(Uiautomator2Device)
    target.serial = "emulator-5572"
    calls: list[tuple[list[str], dict[str, Any]]] = []

    monkeypatch.setattr("android_ui_analyser.emulator.adb_bin", lambda: "/sdk/adb")

    def popen(argv: list[str], **kwargs: Any) -> FakeShellProcess:
        calls.append((argv, kwargs))
        return FakeShellProcess(stdout=b"partial\n", stderr=b"denied\n", returncode=7)

    monkeypatch.setattr(device_mod.subprocess, "Popen", popen)

    result = target.run_read_only_shell(["pm", "path", "com.example.app"], timeout_s=9)

    assert calls[0][0] == [
        "/sdk/adb",
        "-s",
        "emulator-5572",
        "shell",
        "pm path com.example.app",
    ]
    assert calls[0][1] == {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    assert result.exit_code == 7
    assert result.ok is False
    assert result.stderr == "denied\n"
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False
    assert result.output_limit_bytes == 256 * 1024


@pytest.mark.parametrize(
    "suspicious",
    [
        "; setprop ro.example.pwned 1",
        "| id",
        "> /sdcard/out",
        "$(setprop ro.example.pwned 1)",
        "`setprop ro.example.pwned 1`",
        "line\nsetprop ro.example.pwned 1",
        "&& reboot",
    ],
)
def test_android_shell_quotes_remote_control_syntax_as_literal_argument(
    monkeypatch: pytest.MonkeyPatch,
    suspicious: str,
) -> None:
    target = object.__new__(Uiautomator2Device)
    target.serial = "emulator-5572"
    calls: list[list[str]] = []
    argv = ["getprop", suspicious]

    monkeypatch.setattr("android_ui_analyser.emulator.adb_bin", lambda: "/sdk/adb")

    def popen(command: list[str], **kwargs: Any) -> FakeShellProcess:
        del kwargs
        calls.append(command)
        return FakeShellProcess()

    monkeypatch.setattr(device_mod.subprocess, "Popen", popen)

    target.run_read_only_shell(argv)

    remote_command = calls[0][-1]
    assert calls[0][:-1] == ["/sdk/adb", "-s", "emulator-5572", "shell"]
    assert remote_command == shlex.join(argv)
    assert remote_command != " ".join(argv)
    assert shlex.split(remote_command) == argv


def test_android_shell_caps_each_output_stream_and_reports_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = object.__new__(Uiautomator2Device)
    target.serial = "emulator-5572"
    limit = 256 * 1024

    monkeypatch.setattr("android_ui_analyser.emulator.adb_bin", lambda: "/sdk/adb")
    monkeypatch.setattr(
        device_mod.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeShellProcess(
            stdout=b"x" * (limit + 101),
            stderr=b"y" * (limit + 51),
        ),
    )

    result = target.run_read_only_shell(["getprop"])

    assert len(result.stdout.encode()) == limit
    assert len(result.stderr.encode()) == limit
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert result.output_limit_bytes == limit


def test_android_shell_refuses_mutation_before_spawning_adb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = object.__new__(Uiautomator2Device)
    target.serial = "emulator-5572"
    monkeypatch.setattr(
        device_mod.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("adb must not run for a refused mutation"),
    )

    with pytest.raises(UsageError) as raised:
        target.run_read_only_shell(["pm", "clear", "com.example.app"])

    assert raised.value.code == "shell_mutation_refused"


def test_cli_app_exists_is_a_boolean_check_and_status_is_informational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"value": False}
    routed: list[tuple[str, dict[str, Any]]] = []

    def route(_engine: Engine, method: str, **kwargs: Any) -> dict[str, Any]:
        routed.append((method, kwargs))
        return {
            "ok": True,
            "action": "app-status",
            "package": kwargs["package"],
            "installed": installed["value"],
            "serial": "leased-emulator-5560",
            "mode": "read-only",
        }

    monkeypatch.setattr(cli_mod, "_route", route)
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    missing = runner.invoke(app, ["app", "exists", "com.example.app"])
    status = runner.invoke(app, ["app", "status", "com.example.app"])
    installed["value"] = True
    present = runner.invoke(app, ["app", "exists", "com.example.app"])

    assert missing.exit_code == 1
    assert json.loads(missing.stdout)["installed"] is False
    assert status.exit_code == 0
    assert json.loads(status.stdout)["installed"] is False
    assert present.exit_code == 0
    assert [method for method, _ in routed] == ["app_status", "app_status", "app_status"]


def test_cli_shell_routes_structured_argv_and_maps_remote_failure_to_exit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed: list[tuple[str, dict[str, Any]]] = []

    def route(_engine: Engine, method: str, **kwargs: Any) -> ShellResult:
        routed.append((method, kwargs))
        return ShellResult(
            ok=False,
            serial="leased-emulator-5560",
            argv=kwargs["argv"],
            stderr="not found\n",
            exit_code=1,
            duration_ms=2,
        )

    monkeypatch.setattr(cli_mod, "_route", route)
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    result = runner.invoke(
        app,
        ["shell", "--shell-timeout", "12", "pm", "path", "com.example.missing"],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["mode"] == "read-only"
    assert routed == [
        (
            "shell",
            {
                "argv": ["pm", "path", "com.example.missing"],
                "timeout_ms": 12_000,
            },
        )
    ]


def test_daemon_and_mcp_use_the_same_engine_methods(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path)

    daemon_status = dispatch(engine, {"cmd": "app_status", "args": {"package": "com.example.app"}})
    daemon_shell = dispatch(
        engine,
        {"cmd": "shell", "args": {"argv": ["pm", "path", "com.example.app"]}},
    )
    mcp_status = mcp_dispatch(engine, "app_status", {"package": "com.example.app"})
    mcp_shell = mcp_dispatch(engine, "shell_read_only", {"argv": ["pm", "path", "com.example.app"]})

    assert daemon_status["result"]["serial"] == "leased-emulator-5560"
    assert daemon_shell["result"]["mode"] == "read-only"
    assert mcp_status["installed"] is True
    assert mcp_shell["argv"] == ["pm", "path", "com.example.app"]

    tools = {tool.name: tool for tool in _tool_definitions()}
    assert tools["app_status"].inputSchema["required"] == ["package"]
    assert tools["shell_read_only"].inputSchema["required"] == ["argv"]
    assert tools["shell_read_only"].inputSchema["additionalProperties"] is False
