"""A scroll is never aimed at a container that cannot move along its axis.

A platform that can tell reports the axis per scroll container on `NormalizedTree.scroll_axis`.
The engine reads only that neutral map: a container that scrolls sideways only is skipped for
an up/down scroll, and the scroll then goes to the screen exactly as it would with no scroll
container at all. A platform with a single `scrollable` flag reports no axis, and every
scroll container then counts for both directions, as it always did.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.base import NormalizedTree
from conftest import FakeDevice, make_config
from test_scroll_context import engine, node

SCREEN = (0, 0, 300, 600)
BLOCK = (10, 200, 290, 260)


def page(*, shift: int = 0, axis: str | None = "x") -> NormalizedTree:
    """A page of rows with one scroll container on it; `shift` moves the whole page up."""
    elements = [
        node(1, (10, 100 - shift, 290, 130 - shift), text="First row"),
        node(2, (10, 200 - shift, 290, 260 - shift), scrollable=True),
        node(3, (10, 210 - shift, 290, 250 - shift), parent=2, text="A very long line"),
        node(4, (10, 400 - shift, 290, 430 - shift), text="Second row"),
    ]
    return NormalizedTree(elements, app_id="example.app", scroll_axis={2: axis} if axis else {})


def test_a_sideways_container_is_skipped_for_a_vertical_scroll(monkeypatch) -> None:
    eng, _runtime = engine(monkeypatch, [page()])

    assert eng._scroll_box(direction="up") == (SCREEN, False)
    assert eng._scroll_box(direction="down") == (SCREEN, False)


def test_a_sideways_container_is_still_the_target_of_a_sideways_scroll(monkeypatch) -> None:
    eng, _runtime = engine(monkeypatch, [page()])

    assert eng._scroll_box(direction="left") == (BLOCK, True)
    assert eng._scroll_box(direction="right") == (BLOCK, True)


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_a_container_with_no_reported_axis_scrolls_either_way(monkeypatch, direction) -> None:
    eng, _runtime = engine(monkeypatch, [page(axis=None)])

    assert eng._scroll_box(direction=direction) == (BLOCK, True)


def test_a_vertical_scroll_moves_the_page_past_a_sideways_container(monkeypatch) -> None:
    eng, runtime = engine(monkeypatch, [page(), page(shift=80)])

    result = eng.scroll("up", observe=False)

    assert result.ok and str(result.detail).startswith("moved "), result.detail
    assert "dy=80" in str(result.detail) and "scrollable=false" in str(result.detail)
    x1, y1, x2, y2 = runtime.swipes[0][:4]
    assert y1 - y2 > BLOCK[3] - BLOCK[1], "the gesture spans the screen, not the small block"


_ANDROID_LIST = """\
<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node package="com.test.app" class="android.widget.HorizontalScrollView"
        resource-id="com.test.app:id/strip" scrollable="true"
        bounds="[0,400][1080,700]" clickable="false"/>
</hierarchy>
"""


@pytest.mark.parametrize("direction", ["up", "left"])
def test_android_reports_no_axis_and_keeps_its_container(tmp_path, direction) -> None:
    config = make_config(cache={"dir": str(tmp_path)})
    eng = Engine(config, device=FakeDevice(hierarchy_xml=_ANDROID_LIST))

    assert eng.platform.normalize_tree(_ANDROID_LIST, (1080, 2400)).scroll_axis == {}
    assert eng._scroll_box(direction=direction) == ((0, 400, 1080, 700), True)
