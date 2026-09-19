from __future__ import annotations

import asyncio
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ConfigError, DeviceError, UnsupportedPlatformCapabilityError
from android_ui_analyser.platforms import registry
from android_ui_analyser.platforms.registry import PlatformFactory
from android_ui_analyser.platforms.web import WebPlatform
from android_ui_analyser.platforms.web_tools import PlaywrightConnection, WebLaunchOptions

URL = "https://example.test/app"


def _config(tmp_path: Path, **options) -> Config:
    return Config.model_validate(
        {
            "device": {"platform": "web", "serial": URL},
            "platforms": {"web": options},
            "cache": {"dir": str(tmp_path / "cache")},
            "memory": {"enabled": False},
            "lease": {"enabled": False},
            "teardown": {"enabled": False},
            "capture": {"enabled": False},
            "ocr": {"enabled": False},
            "perf": {
                "prefetch": False,
                "predictive_prefetch": False,
                "auto_daemon": False,
                "stable_delay_ms": {"default": 0, "tap": 0},
            },
        }
    )


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (360, 640), "white").save(stream, format="PNG")
    return stream.getvalue()


class FakeConnection:
    def __init__(self) -> None:
        self.url = URL
        self.calls: list[tuple] = []
        self.closed = False

    def viewport_size(self) -> tuple[int, int]:
        return (360, 640)

    def snapshot(self) -> str:
        return (
            '{"format":"aua-web-dom/1","url":"https://example.test/app","nodes":['
            '{"tag":"h1","text":"Web fixture","bounds":[20,20,220,60]},'
            '{"tag":"input","description":"Email","resource_id":"email",'
            '"bounds":[20,90,300,130],"clickable":true,"enabled":true},'
            '{"tag":"button","text":"Continue","resource_id":"submit",'
            '"bounds":[20,160,180,210],"clickable":true,"enabled":true}]}'
        )

    def screenshot_png(self) -> bytes:
        return _png()

    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", x, y))

    def long_click(self, x: int, y: int, duration_ms: int) -> None:
        self.calls.append(("long_click", x, y, duration_ms))

    def type_text(self, text: str) -> None:
        self.calls.append(("type", text))

    def clear_text(self) -> None:
        self.calls.append(("clear",))

    def press(self, key: str) -> None:
        self.calls.append(("press", key))

    def go_back(self) -> None:
        self.calls.append(("back",))

    def reload(self) -> None:
        self.calls.append(("reload",))

    def scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> None:
        self.calls.append(("scroll", x, y, delta_x, delta_y))

    def goto(self, url: str) -> None:
        self.url = url
        self.calls.append(("goto", url))

    def wait_idle(self, timeout_ms: int) -> None:
        self.calls.append(("idle", timeout_ms))

    def storage(self, *, include_values: bool = False) -> dict:
        self.calls.append(("storage", include_values))
        return {"ok": True, "include_values": include_values}

    def storage_export(self, path: str) -> dict:
        self.calls.append(("storage_export", path))
        return {"ok": True, "path": path}

    def storage_import(self, path: str) -> dict:
        self.calls.append(("storage_import", path))
        return {"ok": True, "path": path}

    def storage_clear(self, kinds) -> dict:
        self.calls.append(("storage_clear", tuple(kinds)))
        return {"ok": True, "cleared": list(kinds)}

    def cache_clear(self) -> dict:
        self.calls.append(("cache_clear",))
        return {"ok": True}

    def reset(self) -> dict:
        self.calls.append(("reset",))
        return {"ok": True}

    def network_status(self) -> dict:
        self.calls.append(("network_status",))
        return {"ok": True, "offline": False}

    def set_offline(self, offline: bool) -> dict:
        self.calls.append(("offline", offline))
        return {"ok": True, "offline": offline}

    def set_throttle(self, *, latency_ms: int, download_kbps: int, upload_kbps: int) -> dict:
        self.calls.append(("throttle", latency_ms, download_kbps, upload_kbps))
        return {"ok": True}

    def set_cors(self, *, origin, hosts, methods, headers, credentials) -> dict:
        self.calls.append(("cors", origin, tuple(hosts), tuple(methods), tuple(headers), credentials))
        return {"ok": True}

    def clear_cors(self) -> dict:
        self.calls.append(("cors_clear",))
        return {"ok": True}

    def set_proxy(self, server, *, bypass, username, password) -> dict:
        self.calls.append(("proxy", server, bypass, username, password))
        return {"ok": True, "password_configured": password is not None}

    def clear_proxy(self) -> dict:
        self.calls.append(("proxy_clear",))
        return {"ok": True}

    def har_start(self, path: str) -> dict:
        self.calls.append(("har_start", path))
        return {"ok": True}

    def har_stop(self) -> dict:
        self.calls.append(("har_stop",))
        return {"ok": True}

    def har_replay(self, path: str, *, url: str | None, not_found: str) -> dict:
        self.calls.append(("har_replay", path, url, not_found))
        return {"ok": True}

    def har_clear(self) -> dict:
        self.calls.append(("har_clear",))
        return {"ok": True}

    def mock_add(self, url: str, *, status: int, body: str, headers, abort: bool) -> dict:
        self.calls.append(("mock_add", url, status, body, headers, abort))
        return {"ok": True}

    def mock_clear(self, rule_id: str | None = None) -> dict:
        self.calls.append(("mock_clear", rule_id))
        return {"ok": True}

    def diagnostics(self, *, limit: int, kinds, since_ms: int | None) -> dict:
        self.calls.append(("diagnostics", limit, tuple(kinds), since_ms))
        return {
            "ok": True,
            "events": [
                {
                    "timestamp_ms": 123,
                    "kind": "console",
                    "level": "warning",
                    "message": "fixture warning",
                    "url": URL,
                }
            ],
        }

    def diagnostics_clear(self) -> dict:
        self.calls.append(("diagnostics_clear",))
        return {"ok": True}

    def mark_diagnostics(self, name: str, *, clear: bool = False) -> dict:
        self.calls.append(("diagnostics_mark", name, clear))
        return {"ok": True, "timestamp_ms": 123}

    def pages(self) -> dict:
        self.calls.append(("pages",))
        return {"ok": True, "pages": [{"id": "page-1", "active": True}]}

    def page_select(self, page_id: str) -> dict:
        self.calls.append(("page_select", page_id))
        return {"ok": True}

    def page_close(self, page_id: str) -> dict:
        self.calls.append(("page_close", page_id))
        return {"ok": True}

    def trace_start(self) -> dict:
        self.calls.append(("trace_start",))
        return {"ok": True}

    def trace_stop(self, path: str) -> dict:
        self.calls.append(("trace_stop", path))
        return {"ok": True}

    def session_begin(self, session_id: str) -> dict:
        self.calls.append(("session_begin", session_id))
        return {"ok": True}

    def session_finish(self, session_id: str) -> dict:
        self.calls.append(("session_finish", session_id))
        return {"ok": True}

    def close(self) -> None:
        self.closed = True


class FakeLauncher:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.calls: list[tuple[str, WebLaunchOptions]] = []

    def launch(self, url: str, options: WebLaunchOptions) -> FakeConnection:
        self.calls.append((url, options))
        return self.connection


def _adapter(tmp_path: Path, connection: FakeConnection, **options) -> WebPlatform:
    platform = WebPlatform(_config(tmp_path, **options), launcher=FakeLauncher(connection))
    platform.options = platform.validate_options(options)
    platform.validate_declared_capabilities()
    return platform


def test_web_is_a_built_in_platform_selected_without_loading_android(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    loaded: list[str] = []
    original = registry._load_builtin

    def spy(name: str) -> None:
        loaded.append(name)
        original(name)

    monkeypatch.setattr(registry, "_load_builtin", spy)

    assert "web" in registry.available_platforms()
    platform = PlatformFactory(_config(tmp_path)).create()

    assert isinstance(platform, WebPlatform)
    assert platform.name == "web"
    assert "android" not in loaded


def test_web_options_are_closed_typed_and_safe(tmp_path: Path) -> None:
    platform = WebPlatform(_config(tmp_path))
    options = platform.validate_options(
        {
            "url": "https://example.test/app",
            "browser": "chromium",
            "headless": False,
            "viewport_width": 390,
            "viewport_height": 844,
            "channel": "chrome",
        }
    )
    assert options["url"] == URL
    assert options["headless"] is False
    assert options["viewport_width"] == 390
    with pytest.raises(ConfigError, match="does not accept options"):
        platform.validate_options({"endpoint": "x"})
    with pytest.raises(ConfigError, match="must not contain credentials"):
        platform.validate_options({"url": "https://user:secret@example.test"})
    with pytest.raises(ConfigError, match="channel is supported only"):
        platform.validate_options({"browser": "firefox", "channel": "firefox"})


def test_web_engine_reuses_the_shared_analysis_and_action_path(tmp_path: Path) -> None:
    connection = FakeConnection()
    platform = _adapter(tmp_path, connection, headless=True)
    engine = Engine(_config(tmp_path, headless=True), platform=platform)

    result = engine.analyze(source="hierarchy", with_ocr=False)
    assert result.screen.package == "example.test"
    assert result.screen.activity == "/app"
    assert [element.text for element in result.elements] == ["Web fixture", None, "Continue"]
    assert result.elements[2].stable_key == "rid:submit"

    action = engine.tap(selector={"rid": "submit"}, observe=False)
    assert action.ok is True
    assert ("click", 100, 185) in connection.calls


def test_web_runtime_routes_input_scroll_keys_and_links(tmp_path: Path) -> None:
    connection = FakeConnection()
    runtime = _adapter(tmp_path, connection).connect(URL)

    runtime.input_text(30, 100, "me@example.test", submit=True)
    runtime.swipe(180, 500, 180, 200)
    runtime.press("back")
    runtime.press("refresh")
    runtime.press("home")
    runtime.open_link("https://example.test/next")

    assert connection.calls[:4] == [
        ("click", 30, 100),
        ("clear",),
        ("type", "me@example.test"),
        ("press", "Enter"),
    ]
    assert ("scroll", 180, 500, 0, 300) in connection.calls
    assert ("back",) in connection.calls
    assert ("reload",) in connection.calls
    assert connection.calls[-2:] == [("goto", URL), ("goto", "https://example.test/next")]


def test_web_refuses_android_only_services_without_loading_android(tmp_path: Path) -> None:
    platform = _adapter(tmp_path, FakeConnection())
    with pytest.raises(UnsupportedPlatformCapabilityError, match="network"):
        platform.capability("network")


def test_web_browser_lab_controls_share_the_engine_runtime_path(tmp_path: Path) -> None:
    connection = FakeConnection()
    engine = Engine(_config(tmp_path), platform=_adapter(tmp_path, connection))

    assert engine.browser_storage(include_values=True)["include_values"] is True
    assert engine.browser_offline(True)["offline"] is True
    engine.browser_throttle(latency_ms=40, download_kbps=512, upload_kbps=128)
    engine.browser_cors_add(origin="https://fixture.test", hosts=["api.fixture.test"])
    engine.browser_mock_add("**/api/**", body="fixture")
    assert engine.browser_pages()["pages"][0]["id"] == "page-1"
    engine.browser_trace_start()

    assert ("storage", True) in connection.calls
    assert ("offline", True) in connection.calls
    assert ("throttle", 40, 512, 128) in connection.calls
    assert ("trace_start",) in connection.calls


def test_web_diagnostics_are_normalized_for_existing_app_log_features(tmp_path: Path) -> None:
    connection = FakeConnection()
    platform = _adapter(tmp_path, connection)
    runtime = platform.connect(URL)

    mark = platform.mark_diagnostics(runtime, "before", clear=True)
    window = platform.diagnostic_window(runtime, since="before", app_id="example.test")

    assert mark["clock"] == "host"
    assert window.lines == ["console | fixture warning"]
    assert window.events[0].level.value == "warning"
    assert ("diagnostics_mark", "before", True) in connection.calls


def test_goal_session_registers_and_restores_browser_state(tmp_path: Path) -> None:
    connection = FakeConnection()
    engine = Engine(_config(tmp_path), platform=_adapter(tmp_path, connection))
    observation = engine.analyze(source="hierarchy", with_ocr=False)

    started = engine.session_start("verify the fictional checkout", observation=observation)
    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

    assert "browser_session_restore" in started["cleanup"]
    assert ("session_begin", started["session_id"]) in connection.calls
    assert ("session_finish", started["session_id"]) in connection.calls
    assert finished["terminated"] is True


def test_cli_goal_session_baseline_stays_in_the_daemon_that_mutates_it(tmp_path, monkeypatch):
    from android_ui_analyser import cli, daemon

    class StatefulConnection(FakeConnection):
        offline = False

        def __init__(self):
            super().__init__()
            self.baselines = {}

        def set_offline(self, offline):
            self.offline = offline
            return super().set_offline(offline)

        def session_begin(self, session_id):
            self.baselines[session_id] = self.offline
            return super().session_begin(session_id)

        def session_finish(self, session_id):
            self.offline = self.baselines.pop(session_id)
            return super().session_finish(session_id)

    connection = StatefulConnection()
    cfg = _config(tmp_path)
    warm_engine = Engine(cfg, platform=_adapter(tmp_path, connection))
    routed = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def call(self, method, **kwargs):
            routed.append(method)
            response = daemon.dispatch(warm_engine, {"cmd": method, "args": kwargs})
            response["response_decorated"] = True
            return response

    monkeypatch.setattr(daemon, "is_running", lambda config: True)
    monkeypatch.setattr(daemon, "running_version", lambda config: daemon._aua_version())
    monkeypatch.setattr(daemon, "running_policy_fingerprint", daemon.policy_config_fingerprint)
    monkeypatch.setattr(daemon, "running_runtime_fingerprint", daemon.runtime_config_fingerprint)
    monkeypatch.setattr(daemon, "DaemonClient", Client)
    monkeypatch.setattr(daemon, "_adopt_client_owner", lambda *a, **kw: None)

    def command(method, **kwargs):
        # Each CLI invocation creates a new Engine; none may connect a disposable page.
        caller = Engine(cfg, platform=_adapter(tmp_path, FakeConnection()))
        monkeypatch.setattr(caller, "_connect_target", lambda *a: pytest.fail("cold context"))
        return cli._route(caller, method, **kwargs)

    try:
        started = command("session_start", goal="verify the fictional checkout")
        command("browser_offline", offline=True)
        assert connection.offline
        finished = command("session_finish", session_id=started["session_id"], allow_incomplete=True)
        assert finished["ok"], finished
        assert not connection.offline
        assert routed == ["session_start", "browser_offline", "session_finish"]
        assert not connection.baselines
    finally:
        warm_engine.close()


def test_web_requires_a_url_and_reports_install_help(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.device.serial = None
    platform = WebPlatform(config)
    platform.options = platform.validate_options({})
    with pytest.raises(DeviceError, match="no web URL configured"):
        platform.connect()

    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    with pytest.raises(DeviceError, match="optional Playwright") as exc:
        platform.prepare_host()
    assert "playwright install chromium" in str(exc.value.hint)


def test_playwright_connection_keeps_sync_transport_off_the_async_mcp_thread() -> None:
    calls: list[tuple[str, int]] = []
    executor = ThreadPoolExecutor(max_workers=1)
    owner_thread = executor.submit(threading.get_ident).result()

    class Keyboard:
        def press(self, key: str) -> None:
            calls.append((f"press:{key}", threading.get_ident()))

    class Page:
        keyboard = Keyboard()

        def is_closed(self) -> bool:
            return False

        @property
        def url(self) -> str:
            calls.append(("url", threading.get_ident()))
            return URL

    class Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append((self.name, threading.get_ident()))

    class Playwright:
        def stop(self) -> None:
            calls.append(("playwright", threading.get_ident()))

    connection = PlaywrightConnection(
        executor,
        Playwright(),
        Closable("browser"),
        object(),
        WebLaunchOptions(),
        URL,
    )
    connection._context = Closable("context")
    connection._page = Page()

    async def invoke_like_mcp() -> None:
        assert connection.url == URL
        connection.press("Enter")

    asyncio.run(invoke_like_mcp())
    connection.close()

    assert [name for name, _thread in calls] == [
        "url",
        "press:Enter",
        "context",
        "browser",
        "playwright",
    ]
    assert {thread for _name, thread in calls} == {owner_thread}


def test_web_snapshot_pumps_intercepted_responses_before_reading_dom() -> None:
    calls: list[str] = []
    executor = ThreadPoolExecutor(max_workers=1)

    class Page:
        frames: tuple = ()
        url = URL

        def is_closed(self) -> bool:
            return False

        def wait_for_timeout(self, timeout_ms: int) -> None:
            assert timeout_ms == 0
            calls.append("event-loop")

        def title(self) -> str:
            calls.append("title")
            return "Fixture"

    class Closable:
        def close(self) -> None:
            pass

    class Playwright:
        def stop(self) -> None:
            pass

    connection = PlaywrightConnection(
        executor,
        Playwright(),
        Closable(),
        object(),
        WebLaunchOptions(),
        URL,
    )
    connection._context = Closable()
    connection._page = Page()

    payload = json.loads(connection.snapshot())
    connection.close()

    assert payload["url"] == URL
    assert calls == ["event-loop", "title"]
