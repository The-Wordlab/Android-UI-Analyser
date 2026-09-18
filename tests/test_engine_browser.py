"""Browser controls stay platform-neutral and share one Engine path."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from android_ui_analyser import engine_browser
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
