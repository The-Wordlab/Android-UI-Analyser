"""Unpacked extensions stay in a driver-owned temporary Chromium profile."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.errors import ConfigError, DeviceError
from android_ui_analyser.platforms.web import WebPlatform
from android_ui_analyser.platforms.web_tools import PlaywrightLauncher, WebLaunchOptions


@pytest.fixture
def extension(tmp_path):
    path = tmp_path / "test extension"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 3,
                "name": "Fixture extension",
                "version": "1.0",
            }
        )
    )
    return path


def test_extension_options_resolve_directories_and_preserve_order(extension, tmp_path):
    second = tmp_path / "second"
    second.mkdir()
    (second / "manifest.json").write_text((extension / "manifest.json").read_text())
    platform = WebPlatform(Config())
    platform.options = platform.validate_options(
        {
            "extension_paths": [str(extension), str(second), str(extension)],
        }
    )
    assert platform._launch_options().extension_paths == (str(extension), str(second))


@pytest.mark.parametrize(
    "options",
    [
        {"browser": "firefox"},
        {"browser": "webkit"},
        {"connection": "existing-chrome"},
        {"connection": "existing-cdp", "cdp_endpoint": "http://127.0.0.1:9222"},
        {"channel": "chrome"},
        {"channel": "msedge"},
        {"executable_path": "/usr/bin/chromium"},
        {"service_workers": "block"},
    ],
)
def test_extension_options_refuse_unsupported_launch_modes(extension, options):
    with pytest.raises(ConfigError, match="extension_paths"):
        WebPlatform(Config()).validate_options({"extension_paths": [str(extension)], **options})


@pytest.mark.parametrize("value", [[], "extension", None, [None], [7], [""]])
def test_extension_paths_must_be_a_nonempty_list_of_strings(value):
    with pytest.raises(ConfigError, match="extension_paths"):
        WebPlatform(Config()).validate_options({"extension_paths": value})


@pytest.mark.parametrize("manifest", [None, "not json", "[]", '{"manifest_version": 2}'])
def test_extension_requires_readable_mv3_manifest(extension, manifest):
    path = extension / "manifest.json"
    if manifest is None:
        path.unlink()
    else:
        path.write_text(manifest)
    with pytest.raises(ConfigError, match="manifest|Manifest"):
        WebPlatform(Config()).validate_options({"extension_paths": [str(extension)]})


def test_comma_in_extension_path_is_rejected_before_launch(extension):
    path = extension.rename(extension.with_name("comma,extension"))
    with pytest.raises(ConfigError, match="commas"):
        WebPlatform(Config()).validate_options({"extension_paths": [str(path)]})


@pytest.mark.parametrize("option", ["args", "user_data_dir", "profile_path"])
def test_arbitrary_launch_flags_and_personal_profiles_remain_unsupported(option):
    with pytest.raises(ConfigError, match="does not accept"):
        WebPlatform(Config()).validate_options({option: "unused"})


class Page:
    def __init__(self):
        self.url = "about:blank"
        self.closed = False
        self.handlers = {}
        self.frames = []

    def opener(self):
        return None

    def on(self, event, callback):
        self.handlers[event] = callback

    def is_closed(self):
        return self.closed

    def goto(self, url, **kwargs):
        self.url = url

    def evaluate(self, script, *args):
        return {}

    def title(self):
        return "Fixture"


class Context:
    def __init__(self):
        self.pages = [Page()]
        self.closed = False
        self.storage = {"cookies": [], "origins": []}
        self.service_workers = [
            SimpleNamespace(url="chrome-extension://" + "a" * 32 + "/worker.js")
        ]

    def close(self):
        self.closed = True
        for page in self.pages:
            page.closed = True

    def set_default_timeout(self, timeout):
        pass

    def set_default_navigation_timeout(self, timeout):
        pass

    def on(self, *args):
        pass

    def route(self, *args):
        pass

    def set_offline(self, offline):
        pass

    def set_storage_state(self, state):
        self.storage = state

    def storage_state(self, **kwargs):
        return self.storage

    def new_cdp_session(self, page):
        return SimpleNamespace(send=lambda *args: None, detach=lambda: None)


@pytest.fixture
def driver(monkeypatch):
    launches = []
    contexts = []
    stopped = []

    def persistent(profile, **kwargs):
        launches.append((profile, kwargs))
        context = Context()
        contexts.append(context)
        return context

    browser_type = SimpleNamespace(
        launch=lambda **kw: pytest.fail("must not use ordinary Chrome launch or fallback"),
        launch_persistent_context=persistent,
    )
    runtime = SimpleNamespace(
        _impl_obj=SimpleNamespace(_connection=SimpleNamespace(_abort=lambda: None)),
        chromium=browser_type,
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        SimpleNamespace(
            sync_playwright=lambda: SimpleNamespace(start=lambda: runtime),
        ),
    )
    return SimpleNamespace(
        launches=launches, contexts=contexts, stopped=stopped, browser_type=browser_type
    )


def test_extension_launch_owns_temp_profile_and_uses_new_headless_chromium(driver, extension):
    options = WebLaunchOptions(extension_paths=(str(extension), "/other extension"))
    connection = PlaywrightLauncher().launch("https://example.test", options)
    try:
        profile, kwargs = driver.launches[0]
        assert profile == ""  # driver-managed temporary directory; no user path
        assert kwargs["channel"] == "chromium"
        assert kwargs["headless"] is True
        assert kwargs["args"] == [
            f"--disable-extensions-except={extension},/other extension",
            f"--load-extension={extension},/other extension",
        ]
        assert len(driver.contexts[0].pages) == 1
        assert connection.url == "https://example.test"
        pages = connection.pages()
        assert pages["pages"][0]["id"] == "page-1"
        assert pages["service_workers"][0]["url"].endswith("/worker.js")
        connection.goto("chrome-extension://" + "a" * 32 + "/popup.html")
        assert connection.url.endswith("/popup.html")
    finally:
        connection.close()
    assert driver.contexts[0].closed
    assert driver.stopped == [True]
    connection.close()
    assert driver.stopped == [True]


def test_extension_session_finish_discards_profile_and_restores_web_baseline(driver, extension):
    connection = PlaywrightLauncher().launch(
        "https://example.test",
        WebLaunchOptions(extension_paths=(str(extension),)),
    )
    try:
        baseline = {"cookies": [{"name": "fixture"}], "origins": []}
        driver.contexts[0].storage = baseline
        connection.session_begin("session-1")
        connection.goto("https://example.test/after")
        result = connection.session_finish("session-1")
        assert result["extension_state"] == "discarded"
        assert result["restored"] is True
        assert driver.contexts[0].closed
        assert driver.contexts[1].storage == baseline
        assert driver.launches[1][0] == ""
        assert connection.url == "https://example.test"
        connection.reset()
        assert driver.contexts[1].closed
        assert driver.contexts[2].storage == {"cookies": [], "origins": []}
    finally:
        connection.close()


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: c.storage_import("unused.json"),
        lambda c: c.cache_clear(),
        lambda c: c.set_proxy("http://localhost:1234", bypass=None, username=None, password=None),
        lambda c: c.clear_proxy(),
        lambda c: c.har_start("unused.har"),
        lambda c: c.har_stop(),
        lambda c: c.har_replay("unused.har", url=None, not_found="abort"),
        lambda c: c.har_clear(),
    ],
)
def test_lab_operations_do_not_silently_discard_extension_login(driver, extension, operation):
    connection = PlaywrightLauncher().launch(
        "https://example.test",
        WebLaunchOptions(extension_paths=(str(extension),)),
    )
    try:
        with pytest.raises(ConfigError, match="unavailable with web extension_paths"):
            operation(connection)
        assert len(driver.contexts) == 1
        assert not driver.contexts[0].closed
        assert connection._proxy is None
        assert connection._har_record_path is None
    finally:
        connection.close()


def test_failed_extension_launch_stops_driver_without_falling_back_to_chrome(driver, extension):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture launch failed")

    driver.browser_type.launch_persistent_context = fail
    with pytest.raises(DeviceError, match="fixture launch failed"):
        PlaywrightLauncher().launch(
            "https://example.test",
            WebLaunchOptions(extension_paths=(str(extension),)),
        )
    assert driver.stopped == [True]


def test_failed_startup_navigation_closes_only_owned_context(driver, extension, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture navigation failed")

    monkeypatch.setattr(Page, "goto", fail)
    with pytest.raises(DeviceError, match="fixture navigation failed"):
        PlaywrightLauncher().launch(
            "https://example.test",
            WebLaunchOptions(extension_paths=(str(extension),)),
        )
    assert driver.contexts[0].closed
    assert driver.stopped == [True]


def test_shutdown_does_not_reopen_a_closed_page(driver, extension):
    connection = PlaywrightLauncher().launch(
        "https://example.test",
        WebLaunchOptions(extension_paths=(str(extension),)),
    )
    driver.contexts[0].pages.clear()
    connection._page = None
    connection.close()
    assert driver.contexts[0].closed
    assert driver.stopped == [True]
