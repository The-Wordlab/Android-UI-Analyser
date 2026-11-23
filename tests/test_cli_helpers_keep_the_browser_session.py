"""Element helper reads must use the browser that owns the caller's observation."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli, daemon
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.platforms.web import WebPlatform
from test_web_platform import URL, FakeConnection, FakeLauncher, _config


def test_cli_helpers_read_the_warm_page_without_launching_a_temporary_browser(tmp_path, monkeypatch):
    connection = FakeConnection()
    launcher = FakeLauncher(connection)
    cfg = _config(tmp_path, url=URL, headless=False)
    cfg.daemon.enabled = True
    cfg.daemon.socket = str(tmp_path / "daemon.sock")
    warm = Engine(cfg, platform=WebPlatform(cfg, launcher=launcher))
    routed = []
    callers = []

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def call(self, command, **kwargs):
            routed.append(command)
            result = daemon.dispatch(warm, {"cmd": command, "args": kwargs})
            result["response_decorated"] = True
            return result

    def caller(_options):
        config = cfg.model_copy(deep=True)
        engine = Engine(config, platform=WebPlatform(config, launcher=FakeLauncher(FakeConnection())))
        monkeypatch.setattr(engine, "_connect_target", lambda *_a, **_kw: pytest.fail("temporary browser"))
        callers.append(engine)
        return engine

    monkeypatch.setattr(cli.GlobalOpts, "load", lambda _options: cfg.model_copy(deep=True))
    monkeypatch.setattr(cli.GlobalOpts, "engine", caller)
    monkeypatch.setattr(daemon, "is_running", lambda _cfg: True)
    monkeypatch.setattr(daemon, "running_version", lambda _cfg: daemon._aua_version())
    monkeypatch.setattr(daemon, "running_runtime_fingerprint", daemon.runtime_config_fingerprint)
    monkeypatch.setattr(daemon, "running_policy_fingerprint", daemon.policy_config_fingerprint)
    monkeypatch.setattr(daemon, "DaemonClient", Client)
    monkeypatch.setattr(daemon, "_adopt_client_owner", lambda *_a, **_kw: None)

    try:
        shown = warm.analyze(source="hierarchy", with_ocr=False)
        element = next(row for row in shown.elements if row.resource_id == "submit")
        assert element.handle
        runner = CliRunner()
        for arguments, expected_command in (
            (["inspect", element.handle], "inspect"),
            (["inspect", str(element.id)], "inspect"),
            (["resolve", element.handle], "resolve"),
            (["resolve", "rid:submit"], "resolve"),
        ):
            result = runner.invoke(cli.app, arguments)
            assert result.exit_code == 0, result.stderr
            payload = json.loads(result.stdout)
            inspected = payload if expected_command == "inspect" else payload["element"]
            assert inspected["resource_id"] == "submit"
            assert inspected["handle"] == element.handle
            assert routed[-1] == expected_command
        assert len(launcher.calls) == 1
        assert len(callers) == 4
        assert all(engine._device is None for engine in callers)
        assert not connection.closed
    finally:
        warm.close()


@pytest.mark.parametrize("method, arguments", [("inspect", {"element_id": "el:fixture"}),
                                               ("resolve", {"target": "rid:submit"})])
@pytest.mark.parametrize("enabled", [False, True])
def test_missing_session_daemon_refuses_instead_of_opening_an_empty_browser(
    tmp_path, monkeypatch, method, arguments, enabled,
):
    cfg = _config(tmp_path, url=URL)
    cfg.daemon.enabled = enabled
    engine = Engine(cfg, platform=WebPlatform(cfg, launcher=FakeLauncher(FakeConnection())))
    monkeypatch.setattr(engine, "_connect_target", lambda *_a, **_kw: pytest.fail("temporary browser"))
    monkeypatch.setattr(daemon, "is_running", lambda _cfg: False)
    monkeypatch.setattr(daemon, "running_version", lambda _cfg: False)
    try:
        with pytest.raises(UsageError, match="requires the warm daemon") as caught:
            cli._route(engine, method, **arguments)
        assert caught.value.code == "browser_daemon_required"
        assert engine._device is None
    finally:
        engine.close()
