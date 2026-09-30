"""A web control is published once, not once per icon and label inside it.

CSS `cursor` is inherited, so every `<span>`, `<svg>`, `<path>` and `<circle>` inside a
`cursor: pointer` button shows the pointer too. The DOM snapshot read that as "clickable" on each
of them, and being clickable is also what keeps an unlabeled node in the list at all. Reported
2026-09-30 from a hand-written mobile prototype: 86 clickable elements on 22 distinct centres, a
button, its inner span and its inner path all clickable at one point and the two children with no
text to choose between them.

Only the element the pointer cursor starts on is the control. This runs the real snapshot script
in headless Chromium, because the rule lives in the browser's computed styles; it skips where no
Playwright browser is installed.
"""

from __future__ import annotations

import json

import pytest

from android_ui_analyser.platforms import web_tree
from android_ui_analyser.platforms.web_tools import _DOM_SNAPSHOT_SCRIPT
from android_ui_analyser.schema import Element

sync_api = pytest.importorskip("playwright.sync_api")

_PAGE = """<!doctype html>
<html><head><style>
  button, .icon-button, .card { cursor: pointer; }
  header { display: flex; gap: 24px; padding: 8px; }
</style></head><body>
<header>
  <button aria-label="Back to Home"><span>&lsaquo;</span><svg width="16" height="16">
    <path d="M10 2 L4 8 L10 14" stroke="black" fill="none"/></svg></button>
  <h2>Sample Hub</h2>
  <button aria-label="Sample settings"><svg width="20" height="20">
    <circle cx="10" cy="10" r="8" fill="none" stroke="black"/></svg></button>
  <div class="icon-button"><svg width="20" height="20"><circle cx="10" cy="10" r="8"/></svg></div>
</header>
<main>
  <a href="#more">Read <b>more</b></a>
  <div class="card"><p>Card body</p><button>Add</button></div>
</main>
</body></html>"""

_VIEWPORT = (360, 800)


@pytest.fixture(scope="module")
def elements() -> list[Element]:
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
        snapshot = page.evaluate(_DOM_SNAPSHOT_SCRIPT)
        browser.close()
    finally:
        playwright.stop()
    return web_tree.normalize(json.dumps(snapshot), _VIEWPORT).elements


def _clickable(elements: list[Element]) -> list[tuple[str, str | None, str | None]]:
    return [(el.type, el.text, el.content_desc) for el in elements if el.clickable]


def test_each_control_is_clickable_once(elements: list[Element]) -> None:
    assert _clickable(elements) == [
        ("Button", "‹", "Back to Home"),
        ("Button", None, "Sample settings"),
        ("Div", None, None),
        ("Link", "Read more", None),
        # A clickable card keeps its own button: that one is a real control of its own.
        ("Div", None, None),
        ("Button", "Add", None),
    ]


def test_unlabeled_icon_parts_are_not_listed(elements: list[Element]) -> None:
    types = {el.type for el in elements}
    assert not types & {"Svg", "Path", "Circle"}, sorted(types)


def test_a_labeled_child_stays_as_text_but_not_as_a_target(elements: list[Element]) -> None:
    children = {el.text: el for el in elements if el.type in {"Span", "B"}}
    assert set(children) == {"‹", "more"}
    assert not any(el.clickable for el in children.values())
