"""Existing desktop sessions stay owned by the user through attach, actions and teardown."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ConfigError, DeviceError, UnsupportedPlatformCapabilityError
from android_ui_analyser.platforms.registry import PlatformFactory
from android_ui_analyser.platforms.web import WebPlatform
from android_ui_analyser.platforms.web_cdp import (
    CdpAttachOptions,
    CdpConnection,
    CdpLauncher,
    cdp_endpoint,
    cdp_page_url,
    cdp_target_id,
)
from android_ui_analyser.platforms.web_tools import WebLaunchOptions
from test_web_platform import FakeConnection, _config

ENDPOINT = "http://127.0.0.1:9222"
FILE_URL = "file:///tmp/desktop-fixture/index.html"


def config(tmp_path, **options):
    result = _config(tmp_path, connection="existing-cdp", cdp_endpoint=ENDPOINT, **options)
    result.device.serial = None
    return result


class Page:
    def __init__(self, url=FILE_URL):
        self.url = url
        self.closed = False
        self.handlers = {}
        self.frames = []
        self.main_frame = self
        self.context = SimpleNamespace(pages=[self], new_cdp_session=self.new_cdp_session)
        self.viewport_size = None
        self.input = []
        self.mouse = SimpleNamespace(click=lambda x, y: self.input.append((x, y)))
        self.pixel_ratio = 1
        self.captured_png = None
        self.detached_captures = 0

    def is_closed(self):
        return self.closed

    def opener(self):
        return None

    def on(self, event, callback):
        self.handlers[event] = callback

    def set_default_timeout(self, timeout):
        self.timeout = timeout

    def set_default_navigation_timeout(self, timeout):
        self.navigation_timeout = timeout

    def evaluate(self, script):
        return {"width": 800, "height": 600}

    def new_cdp_session(self, page):
        assert page is self
        def detach():
            self.detached_captures += 1
        return SimpleNamespace(send=self.capture, detach=detach)

    def capture(self, method, options):
        assert method == "Page.captureScreenshot"
        assert options == {"format": "png"}
        ratio = self.pixel_ratio
        image = Image.new("RGB", (800 * ratio, 600 * ratio), "white")
        ImageDraw.Draw(image).rectangle(
            (500 * ratio, 100 * ratio, 700 * ratio, 500 * ratio), fill="blue"
        )
        output = BytesIO()
        image.save(output, format="PNG")
        self.captured_png = output.getvalue()
        return {"data": base64.b64encode(self.captured_png).decode()}

    def title(self):
        return "Desktop fixture"


@pytest.fixture
def attached():
    page = Page()
    browser = SimpleNamespace(contexts=[page.context])
    stopped = []
    playwright = SimpleNamespace(stop=lambda: stopped.append(True))
    conn = CdpConnection(
        ThreadPoolExecutor(max_workers=1), playwright, browser, object(), WebLaunchOptions(), ""
    )
    try:
        yield conn, page, stopped
    finally:
        conn.close()


def test_attach_actions_and_finish_preserve_the_existing_context(attached):
    conn, page, stopped = attached
    conn.initialize_attached(FILE_URL)
    assert conn._context is page.context
    assert conn.viewport_size() == (800, 600)
    assert conn._call(conn._snapshot_viewport) == {"width": 800, "height": 600}
    assert conn.screenshot_png() == page.captured_png
    conn.click(42, 51)
    assert page.input == [(42, 51)]
    assert conn.session_begin("s1")["captured"] == []
    assert conn.session_finish("s1")["detached"]
    assert stopped == [True]
    assert not page.closed
    assert conn._context is page.context
    conn.close()
    assert stopped == [True]


@pytest.mark.parametrize("ratio", [1, 2, 3])
def test_attached_capture_preserves_content_at_css_coordinates(attached, ratio):
    conn, page, _ = attached
    conn.initialize_attached(FILE_URL)
    page.pixel_ratio = ratio
    png = conn.screenshot_png()
    with Image.open(BytesIO(png)) as image:
        assert image.size == (800, 600)
        assert image.getpixel((600, 300)) == (0, 0, 255)
        assert image.getpixel((400, 300)) == (255, 255, 255)
        assert image.getpixel((750, 300)) == (255, 255, 255)
    assert page.viewport_size is None
    assert not page.input
    assert page.detached_captures == 1
    if ratio == 1:
        assert png == page.captured_png


@pytest.mark.parametrize("payload", [None, {"data": "not base64"}, {"data": "aW52YWxpZA=="}])
def test_attached_capture_rejects_invalid_image(attached, payload):
    conn, page, _ = attached
    conn.initialize_attached(FILE_URL)
    page.capture = lambda *args: payload
    with pytest.raises(DeviceError) as error:
        conn.screenshot_png()
    assert error.value.code == "screencap_failed"
    assert page.detached_captures == 1


def test_attached_capture_detaches_on_transport_failure(attached):
    conn, page, _ = attached
    conn.initialize_attached(FILE_URL)
    def fail(*args):
        raise TimeoutError("capture expired")
    page.capture = fail
    with pytest.raises(DeviceError, match="capture expired"):
        conn.screenshot_png()
    assert page.detached_captures == 1


@pytest.mark.parametrize("use_mark", [True, False])
def test_diagnostics_report_buffer_loss_and_reset_it_when_cleared(attached, use_mark):
    conn, _, _ = attached
    conn.initialize_attached(None)
    for i in range(2001):
        conn._add_event("console", "log", f"event {i}")
    report = conn.diagnostics(limit=5, kinds=(), since_ms=None)
    assert report["buffer_overflow"] and report["truncated"]
    assert report["count"] == 5 and report["total_count"] == 2000
    if use_mark:
        conn.mark_diagnostics("fresh", clear=True)
    else:
        conn.diagnostics_clear()
    report = conn.diagnostics(limit=5, kinds=(), since_ms=None)
    assert not report["buffer_overflow"] and report["count"] == 0


def test_ambiguous_attachment_refuses_to_guess_and_can_be_disambiguated(attached):
    conn, page, _ = attached
    second = Page("https://fixture.test/other")
    page.context.pages.append(second)
    with pytest.raises(DeviceError, match="matched 2 pages") as error:
        conn.initialize_attached(None)
    assert error.value.code == "web_cdp_target_ambiguous"
    conn.initialize_attached(FILE_URL)
    assert len(conn.pages()["pages"]) == 1
    assert conn.pages()["pages"][0]["url"] == FILE_URL
    with pytest.raises(ConfigError, match="not selected"):
        conn.page_select("1")


def test_missing_window_does_not_create_or_navigate_one(attached):
    conn, page, _ = attached
    with pytest.raises(DeviceError) as error:
        conn.initialize_attached("https://fixture.test/missing")
    assert error.value.code == "web_cdp_target_missing"
    assert page.url == FILE_URL


def test_closed_window_never_switches_target_and_still_detaches(attached):
    conn, page, stopped = attached
    conn.initialize_attached(None)
    page.context.pages.append(Page("https://fixture.test/other"))
    page.closed = True
    page.handlers["close"]()
    with pytest.raises(DeviceError, match="attached page is closed"):
        conn.click(1, 1)
    conn.session_finish("s1")
    assert stopped == [True]
    assert page.input == []


@pytest.mark.parametrize(
    "method,args",
    [
        ("storage", ()),
        ("storage_clear", (["cookies"],)),
        ("reset", ()),
        ("set_offline", (True,)),
        ("trace_start", ()),
        ("page_close", ("0",)),
    ],
)
def test_attached_transport_refuses_destructive_browser_controls(attached, method, args):
    conn, _, _ = attached
    conn.initialize_attached(None)
    with pytest.raises(UnsupportedPlatformCapabilityError):
        getattr(conn, method)(*args)


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "https://127.0.0.1:9222",
        "http://example.test:9222",
        "http://127.0.0.1",
        "http://user:secret@localhost:9222",
        "http://localhost:9222/path",
        "http://localhost:9222?token=secret",
        "http://localhost:99999",
        "http://[",
    ],
)
def test_cdp_endpoint_validation(endpoint):
    with pytest.raises(ConfigError, match="loopback"):
        cdp_endpoint(endpoint)


def test_endpoint_identity_is_normalized():
    assert cdp_endpoint("http://localhost:9222/") == ENDPOINT
    assert cdp_endpoint("http://[::1]:9222") == "http://[::1]:9222"
    assert cdp_target_id(ENDPOINT) == cdp_target_id("http://[::1]:9222")
    assert cdp_target_id(ENDPOINT) != cdp_target_id("http://127.0.0.1:9223")
    assert cdp_page_url(FILE_URL) == FILE_URL
    with pytest.raises(ConfigError):
        cdp_page_url("file://remote-host/private")


def test_cdp_endpoint_is_exclusive_even_when_agents_select_different_pages(tmp_path):
    from android_ui_analyser.errors import DeviceLeasedError

    engines = []
    try:
        for owner, page in [("agent-a", FILE_URL), ("agent-b", "https://fixture.test/other")]:
            cfg = config(tmp_path, page_url=page)
            cfg.lease.enabled = True
            cfg.lease.registry_dir = str(tmp_path / "leases")
            engine = Engine(cfg)
            engine._lease_owner = owner
            engines.append(engine)
        assert engines[0]._lease_device() == cdp_target_id(ENDPOINT)
        with pytest.raises(DeviceLeasedError):
            engines[1]._lease_device()
    finally:
        for engine in engines:
            engine.close()


def test_factory_and_existing_engine_paths_support_cdp(tmp_path, monkeypatch):
    from android_ui_analyser.platforms import registry

    loaded = []
    original = registry._load_builtin

    def load(name):
        loaded.append(name)
        return original(name)

    monkeypatch.setattr(registry, "_load_builtin", load)
    cfg = config(tmp_path, page_url=FILE_URL)
    platform = PlatformFactory(cfg).create()
    assert "android" not in loaded
    assert platform.supports("ui.tree") and platform.supports("session.state")
    assert not platform.supports("browser.storage")
    assert not platform.supports("browser.network")
    assert not platform.supports("browser.trace")
    connection = FakeConnection()
    calls = []

    class Launcher:
        def launch(self, options):
            calls.append(options)
            return connection

    platform._cdp_launcher = Launcher()
    platform._uses_default_cdp_launcher = False
    engine = Engine(cfg, platform=platform)
    try:
        assert (
            engine.analyze(source="hierarchy", with_ocr=False).elements[2].stable_key
            == "rid:submit"
        )
        assert engine.tap(selector={"rid": "submit"}, observe=False).ok
        assert ("click", 100, 185) in connection.calls
        assert calls[0].page_url == FILE_URL
        assert platform.list_targets()[0].target_id == cdp_target_id(ENDPOINT)
        with pytest.raises(DeviceError, match="does not match"):
            platform.connect("another-target")
    finally:
        engine.close()


@pytest.mark.parametrize("options", [{"headless": False}, {"storage_state": "private.json"}])
def test_cdp_rejects_isolated_options(tmp_path, options):
    platform = WebPlatform(config(tmp_path, **options))
    with pytest.raises(ConfigError, match="isolated-browser options"):
        platform.validate_options(platform.config.platform_options("web"))


def test_cdp_options_do_not_leak_into_isolated_browser_mode(tmp_path):
    platform = WebPlatform(_config(tmp_path))
    with pytest.raises(ConfigError, match="require connection"):
        platform.validate_options({"cdp_endpoint": ENDPOINT})


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_disconnects_on_finish_and_failed_selection(monkeypatch, fail):
    import sys

    page = Page()
    browser = SimpleNamespace(contexts=[page.context])
    stopped = []
    calls = []

    def attach(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return browser

    playwright = SimpleNamespace(
        _impl_obj=SimpleNamespace(_connection=SimpleNamespace(_abort=lambda: None)),
        chromium=SimpleNamespace(connect_over_cdp=attach),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        SimpleNamespace(
            sync_playwright=lambda: SimpleNamespace(start=lambda: playwright),
        ),
    )
    options = CdpAttachOptions(
        endpoint=ENDPOINT, page_url="https://fixture.test/missing" if fail else None
    )
    if fail:
        with pytest.raises(DeviceError, match="matched 0"):
            CdpLauncher().launch(options)
    else:
        connection = CdpLauncher().launch(options)
        connection.session_finish("s1")
    assert calls == [(ENDPOINT, {"timeout": 30_000, "no_defaults": True})]
    assert stopped == [True]
    assert not page.closed


@pytest.mark.parametrize("has_owner", [False, True])
def test_snapshot_skips_a_swapped_guest_before_evaluating_its_dead_context(attached, has_owner):
    conn, page, _ = attached
    conn.initialize_attached(FILE_URL)
    evaluated_guest = []

    class SwappedGuest:
        url = "about:blank"

        def frame_element(self):
            if not has_owner:
                raise RuntimeError("Frame has been detached")
            return SimpleNamespace(
                evaluate=lambda _script: True,
                bounding_box=lambda: pytest.fail("unsupported guest must be skipped"),
                dispose=lambda: None,
            )

        def evaluate(self, _script):
            evaluated_guest.append(True)
            return {"nodes": []}

    page.wait_for_timeout = lambda _timeout: None
    page.frames = [page, SwappedGuest()]
    page.evaluate = lambda script: (
        {"width": 800, "height": 600} if script == "() => ({width: innerWidth, height: innerHeight})"
        else {"nodes": [{"text": "Ready", "bounds": [0, 0, 30, 20]}]}
    )

    payload = json.loads(conn.snapshot())

    assert not evaluated_guest, "a swapped guest has no execution context to evaluate"
    assert payload["nodes"][0]["text"] == "Ready"


def test_snapshot_still_reads_live_child_frames_at_their_page_offset(attached):
    conn, page, _ = attached
    conn.initialize_attached(FILE_URL)
    disposed = []
    handle = SimpleNamespace(
        evaluate=lambda _script: False,
        bounding_box=lambda: {"x": 10, "y": 20},
        dispose=lambda: disposed.append(True),
    )
    child = SimpleNamespace(
        url="https://fixture.test/frame",
        frame_element=lambda: handle,
        evaluate=lambda _script: {"nodes": [{"text": "Child", "bounds": [1, 2, 3, 4]}]},
    )
    page.wait_for_timeout = lambda _timeout: None
    page.frames = [page, child]
    page.evaluate = lambda script: (
        {"width": 800, "height": 600} if script == "() => ({width: innerWidth, height: innerHeight})" else {"nodes": []}
    )

    payload = json.loads(conn.snapshot())

    assert payload["nodes"][0]["text"] == "Child"
    assert payload["nodes"][0]["bounds"] == [11, 22, 13, 24]
    assert disposed == [True]
