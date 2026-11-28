"""`scroll-to` walks down a web page that has a sideways-scrolling block on screen.

A `<pre>` with `overflow-x: auto` whose lines are wider than the viewport is a scroll container,
but only along x. The page itself scrolls along y and has no element of its own in the tree.
An un-anchored vertical scroll therefore has to move the page, not aim at the `<pre>`: aimed
at the block, the wheel moved the page by a fraction of the block's height, the block moved
with it, and the scroll ended as `movement-unverified` after one swipe. Reported 2026-11-28
from a long page at a 390 px viewport.

A sideways swipe is the other half of the rule and must keep its target: the block still
scrolls sideways when asked.

This runs the real DOM snapshot and a real wheel in headless Chromium behind the connection the
engine drives; it skips where no Playwright browser is installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms import web_tree
from android_ui_analyser.platforms.web_tools import _DOM_SNAPSHOT_SCRIPT
from test_web_platform import FakeConnection, _adapter, _config

sync_api = pytest.importorskip("playwright.sync_api")

_VIEWPORT = (390, 844)
_LONG_LINE = "plum --crate north-row --ripeness 7 --label windfall-batch-nineteen " * 3
_CHAPTER = """
<section>
  <h2>Chapter {n}</h2>
  <p>Filler prose for chapter {n}, written only so the page is long.</p>
  <pre>{line}</pre>
  <p>Closing words of chapter {n}.</p>
</section>"""
_PAGE = (
    "<!doctype html><html><head><style>"
    "body { margin: 0; padding: 16px; font: 16px/1.5 sans-serif; }"
    "section { min-height: 520px; }"
    "pre { overflow-x: auto; padding: 12px; }"
    "</style></head><body><h1>Sample long page</h1>"
    + "".join(_CHAPTER.format(n=n, line=_LONG_LINE) for n in range(1, 7))
    + "</body></html>"
)


class _BrowserPage(FakeConnection):
    """The real snapshot script and a real wheel, behind the connection the engine drives."""

    def __init__(self, page) -> None:
        super().__init__()
        self._page = page

    def viewport_size(self) -> tuple[int, int]:
        return _VIEWPORT

    def snapshot(self) -> str:
        return json.dumps(self._page.evaluate(_DOM_SNAPSHOT_SCRIPT))

    def screenshot_png(self) -> bytes:
        return self._page.screenshot()

    def scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> None:
        self._page.mouse.move(x, y)
        self._page.mouse.wheel(delta_x, delta_y)


@pytest.fixture
def page() -> Iterator[Any]:
    try:
        playwright = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Playwright unavailable: {exc}")
    try:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:  # pragma: no cover - no browser downloaded (CI)
            pytest.skip(f"no Playwright Chromium: {exc}")
        page = browser.new_page(viewport={"width": _VIEWPORT[0], "height": _VIEWPORT[1]})
        page.set_content(_PAGE)
        yield page
        browser.close()
    finally:
        playwright.stop()


@pytest.fixture
def engine(tmp_path: Path, page: Any) -> Engine:
    return Engine(_config(tmp_path), platform=_adapter(tmp_path, _BrowserPage(page)))


def test_the_web_tree_reports_the_block_as_sideways_only(page: Any) -> None:
    tree = web_tree.normalize(json.dumps(page.evaluate(_DOM_SNAPSHOT_SCRIPT)), _VIEWPORT)

    blocks = [el for el in tree.elements if el.scrollable]
    assert blocks and {el.type for el in blocks} == {"Pre"}, "the fixture needs a block on screen"
    assert [tree.scroll_axis.get(el.id) for el in blocks] == ["x"] * len(blocks)


def test_scroll_to_reaches_text_below_a_sideways_scrolling_block(engine: Engine) -> None:
    result = engine.scroll_to("Chapter 5", observe=False)

    assert result.ok, result.detail
    assert str(result.detail).startswith("moved "), result.detail


def test_a_sideways_swipe_still_scrolls_the_sideways_block(engine: Engine, page: Any) -> None:
    engine.analyze(source="hierarchy")

    result = engine.swipe("left", selector={"text": "plum --crate", "index": 0}, observe=False)

    assert result.ok, result.detail
    assert page.evaluate("document.querySelector('pre').scrollLeft") > 0, "the block moved"
    assert page.evaluate("window.scrollY") == 0, "and the page did not"
