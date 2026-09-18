from __future__ import annotations

import asyncio
import io
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
        Closable("context"),
        Page(),
    )

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
