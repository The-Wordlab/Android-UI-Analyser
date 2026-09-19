"""Page-driven closes and CLI closes share a valid active-page lifecycle."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from android_ui_analyser.platforms.web_tools import PlaywrightConnection, WebLaunchOptions


class Page:
    def __init__(self, url, opener=None):
        self.url = url
        self.closed = False
        self.parent = opener
        self.handlers = {}

    def on(self, event, callback):
        self.handlers[event] = callback

    def opener(self):
        return self.parent

    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True
        self.handlers["close"]()

    def goto(self, url, **kwargs):
        assert not self.closed
        self.url = url

    def bring_to_front(self):
        assert not self.closed


@pytest.fixture
def connection():
    with ThreadPoolExecutor(max_workers=1) as executor:
        conn = PlaywrightConnection(
            executor,
            object(),
            object(),
            object(),
            WebLaunchOptions(browser="firefox"),
            "https://fixture.test/home",
        )

        class Context:
            pages = []

            def new_page(self, url="about:blank", opener=None):
                page = Page(url, opener)
                self.pages.append(page)
                conn._on_page(page)
                return page

        conn._context = Context()
        yield conn


def test_page_self_close_restores_live_opener_instead_of_unrelated_tab(connection):
    parent = connection._context.new_page("https://fixture.test/parent")
    connection._context.new_page("https://fixture.test/unrelated")
    popup = connection._context.new_page("https://fixture.test/popup", parent)
    connection._call(popup.close)
    assert connection.url == parent.url
    connection.goto("https://fixture.test/continued")
    assert parent.url == "https://fixture.test/continued"


def test_closing_background_page_does_not_change_active_page(connection):
    background = connection._context.new_page("https://fixture.test/background")
    foreground = connection._context.new_page("https://fixture.test/foreground")
    result = connection.page_close(connection._page_id(background))
    assert result["active"] == connection._page_id(foreground)
    assert connection.url == foreground.url


def test_closed_opener_falls_back_to_another_live_page(connection):
    parent = connection._context.new_page("https://fixture.test/parent")
    remaining = connection._context.new_page("https://fixture.test/remaining")
    popup = connection._context.new_page("https://fixture.test/popup", parent)
    connection._call(parent.close)
    connection._call(popup.close)
    assert connection.url == remaining.url


def test_last_page_close_recovers_lazily_without_reopening_during_shutdown(connection):
    page = connection._context.new_page("https://fixture.test/last")
    connection._call(page.close)
    assert len(connection._context.pages) == 1
    assert connection._page is None
    assert connection.url == "https://fixture.test/home"
    assert len(connection._context.pages) == 2


def test_stale_closed_handle_is_recovered_even_without_close_callback(connection):
    parent = connection._context.new_page("https://fixture.test/parent")
    stale = connection._context.new_page("https://fixture.test/stale")
    stale.closed = True
    assert connection.url == parent.url
