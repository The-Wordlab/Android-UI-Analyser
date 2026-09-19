"""Browser controls stay platform-neutral and share one Engine path."""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli, daemon, engine_browser
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.runtime import TargetRuntime


class _Runtime(TargetRuntime):
    target_id = "neutral-browser-target"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []


def _operation(name: str):
    def call(self: _Runtime, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((name, args, kwargs))
        return {"ok": True, "operation": name, "events": [], "pages": []}

    call.__name__ = name
    return call


for _name in (
    "browser_storage",
    "browser_storage_export",
    "browser_storage_import",
    "browser_storage_clear",
    "browser_cache_clear",
    "browser_reset",
    "browser_network_status",
    "browser_set_offline",
    "browser_set_throttle",
    "browser_set_cors",
    "browser_clear_cors",
    "browser_set_proxy",
    "browser_clear_proxy",
    "browser_har_start",
    "browser_har_stop",
    "browser_har_replay",
    "browser_har_clear",
    "browser_mock_add",
    "browser_mock_clear",
    "browser_diagnostics",
    "browser_diagnostics_clear",
    "browser_diagnostics_mark",
    "browser_pages",
    "browser_page_select",
    "browser_page_close",
    "browser_trace_start",
    "browser_trace_stop",
    "session_state_begin",
    "session_state_finish",
):
    setattr(_Runtime, _name, _operation(_name))


class _Platform(PlatformAdapter):
    name = "neutral-browser"
    capabilities = frozenset(
        {
            "browser.storage",
            "browser.network",
            "browser.diagnostics",
            "browser.pages",
            "browser.trace",
            "session.state",
        }
    )

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        del target_id
        raise AssertionError("runtime is injected")

    def list_targets(self) -> list[Any]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        del raw_tree, screen_size, ignored_app_ids
        return NormalizedTree([])


def test_every_browser_engine_operation_uses_declared_runtime_capabilities() -> None:
    runtime = _Runtime()
    engine = Engine(Config(), device=runtime, platform=_Platform(Config()))

    engine.browser_storage(include_values=True)
    engine.browser_storage_export("state.json")
    engine.browser_storage_import("state.json")
    engine.browser_storage_clear(["local"])
    engine.browser_cache_clear()
    engine.browser_reset()
    engine.browser_network_status()
    engine.browser_offline(True)
    engine.browser_throttle(latency_ms=20)
    engine.browser_cors_add(origin="https://fixture.test", hosts=["api.fixture.test"])
    engine.browser_cors_clear()
    engine.browser_proxy_set("http://proxy.fixture.test:8080")
    engine.browser_proxy_clear()
    engine.browser_har_start("capture.har")
    engine.browser_har_stop()
    engine.browser_har_replay("capture.har")
    engine.browser_har_clear()
    engine.browser_mock_add("**/api/**")
    engine.browser_mock_clear()
    engine.browser_logs()
    engine.browser_logs_clear()
    engine.browser_pages()
    engine.browser_page_select("page-2")
    engine.browser_page_close("page-2")
    engine.browser_trace_start()
    engine.browser_trace_stop("trace.zip")

    called = {name for name, _args, _kwargs in runtime.calls}
    assert called == {
        name
        for name in vars(_Runtime)
        if name.startswith("browser_") and name != "browser_diagnostics_mark"
    }


def test_browser_engine_module_has_no_native_transport_dependency() -> None:
    source = inspect.getsource(engine_browser)
    assert "playwright" not in source.casefold()
    assert "android" not in source.casefold()
    assert "adb" not in source.casefold()


_BROWSER_COMMAND_CASES = [
    (["logs"], "browser_logs", {"limit": 100, "kinds": [], "since_ms": None}),
    (["clear-logs"], "browser_logs_clear", {}),
    (["storage", "--include-values"], "browser_storage", {"include_values": True}),
    (["storage-clear", "--kind", "local"], "browser_storage_clear", {"kinds": ["local"]}),
    (["cache-clear"], "browser_cache_clear", {}),
    (["reset"], "browser_reset", {}),
    (["network"], "browser_network_status", {}),
    (["offline"], "browser_offline", {"offline": True}),
    (["online"], "browser_offline", {"offline": False}),
    (
        ["throttle", "--latency-ms", "20"],
        "browser_throttle",
        {"latency_ms": 20, "download_kbps": 0, "upload_kbps": 0},
    ),
    (
        ["cors-add", "--origin", "https://fixture.test", "--host", "api.fixture.test"],
        "browser_cors_add",
        {
            "origin": "https://fixture.test",
            "hosts": ["api.fixture.test"],
            "methods": [],
            "headers": [],
            "credentials": False,
        },
    ),
    (["cors-clear"], "browser_cors_clear", {}),
    (
        ["proxy-set", "http://proxy.fixture.test:8080"],
        "browser_proxy_set",
        {
            "server": "http://proxy.fixture.test:8080",
            "bypass": None,
            "username": None,
            "password": None,
        },
    ),
    (["proxy-clear"], "browser_proxy_clear", {}),
    (["har-stop"], "browser_har_stop", {}),
    (["har-clear"], "browser_har_clear", {}),
    (
        ["mock-add", "**/api/**", "--body", "fixture"],
        "browser_mock_add",
        {"url": "**/api/**", "status": 200, "body": "fixture", "headers": {}, "abort": False},
    ),
    (["mock-clear", "--id", "mock-1"], "browser_mock_clear", {"rule_id": "mock-1"}),
    (["pages"], "browser_pages", {}),
    (["page-select", "page-2"], "browser_page_select", {"page_id": "page-2"}),
    (["page-close", "page-2"], "browser_page_close", {"page_id": "page-2"}),
    (["trace-start"], "browser_trace_start", {}),
]


@pytest.mark.parametrize("argv, method, kwargs", _BROWSER_COMMAND_CASES)
def test_browser_cli_routes_to_shared_daemon_engine(monkeypatch, argv, method, kwargs):
    runtime = _Runtime()
    engine = Engine(Config(), device=runtime, platform=_Platform(Config()))
    routed = []

    def route(actual_engine, actual_method, **actual_kwargs):
        assert actual_engine is engine
        routed.append((actual_method, actual_kwargs))
        response = daemon.dispatch(engine, {"cmd": actual_method, "args": actual_kwargs})
        assert response["ok"], response
        return response["result"]

    monkeypatch.setattr(cli, "_run", lambda ctx, go: go(engine, None))
    monkeypatch.setattr(cli, "_route", route)
    monkeypatch.setattr(daemon, "_adopt_client_owner", lambda *a, **kw: None)
    result = CliRunner().invoke(cli.app, ["--format", "compact", "browser", *argv])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"]
    assert routed == [(method, kwargs)]
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    "command", ["storage-export", "storage-import", "har-start", "har-replay", "trace-stop"]
)
def test_browser_artifact_paths_resolve_in_cli_working_directory(monkeypatch, tmp_path, command):
    monkeypatch.chdir(tmp_path)
    routed = []
    monkeypatch.setattr(cli, "_run", lambda ctx, go: go(object(), None))
    monkeypatch.setattr(
        cli, "_route", lambda engine, method, **kw: routed.append((method, kw)) or {"ok": True}
    )
    result = CliRunner().invoke(cli.app, ["--format", "compact", "browser", command, "artifact"])
    assert result.exit_code == 0, result.output
    assert routed[0][1]["path"] == str(tmp_path / "artifact")


def test_browser_controls_never_succeed_in_a_disposable_context(monkeypatch):
    from android_ui_analyser.errors import UsageError

    config = Config()
    config.daemon.enabled = False
    runtime = _Runtime()
    engine = Engine(config, device=runtime, platform=_Platform(config))
    monkeypatch.setattr(engine, "_lease_device", lambda: runtime.target_id)
    with pytest.raises(UsageError) as error:
        cli._route(engine, "browser_offline", offline=True)
    assert error.value.code == "browser_daemon_required"
    assert runtime.calls == []


def test_browser_daemon_keeps_android_unsupported_result(monkeypatch):
    from conftest import FakeDevice, make_engine

    engine = make_engine(device=FakeDevice())
    monkeypatch.setattr(daemon, "_adopt_client_owner", lambda *a, **kw: None)
    response = daemon.dispatch(engine, {"cmd": "browser_storage", "args": {}})
    assert not response["ok"]
    assert response["error"]["code"] == "platform_capability_unsupported"
