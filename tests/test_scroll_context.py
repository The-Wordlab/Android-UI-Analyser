"""A replacement surface cannot turn a performed swipe into verified list movement."""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree
from android_ui_analyser.schema import Element
from conftest import FakeDevice, make_config
from test_platform_runtime import _NeutralAdapter, _NeutralRuntime


def node(id, bounds, *, type="View", parent=None, **kwargs):
    return Element(
        id=id,
        type=type,
        bounds=bounds,
        center=((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2),
        parent=parent,
        window="app",
        **kwargs,
    )


def listing(*, later=False, shift=0, offset=0, rid="cards", app_id="example.app"):
    elements = [
        node(offset + 100, (0, 0, 300, 600), resource_id="main"),
        node(
            offset + 10,
            (0, 100, 300, 550),
            parent=offset + 100,
            resource_id=rid,
            scrollable=True,
        ),
        node(offset + 11, (10, 110, 100, 140), parent=offset + 10, text="All"),
        node(offset + 12, (150, 110, 250, 140), parent=offset + 10, text="Saved"),
        node(
            offset + 1,
            (10, 250 - shift, 100, 280 - shift),
            parent=offset + 10,
            text="Third card" if later else "First card",
        ),
        node(
            offset + 2,
            (10, 400 - shift, 100, 430 - shift),
            parent=offset + 10,
            text="Fourth card" if later else "Second card",
        ),
    ]
    return NormalizedTree(elements, app_id=app_id)


def popup(*, retained=False):
    elements = list(listing().elements) if retained else []
    elements.extend(
        [
            node(200, (100, 150, 290, 500), type="Menu", resource_id="context_menu"),
            node(201, (110, 180, 280, 230), parent=200, text="Open"),
            node(202, (110, 300, 280, 350), parent=200, text="Share"),
        ]
    )
    return NormalizedTree(elements, app_id="example.app")


class Runtime(_NeutralRuntime):
    def __init__(self):
        self.swipes = []
        self.tree_reads = 0
        self.lookups = 0

    def dump_hierarchy(self, compressed=False):
        self.tree_reads += 1
        return str(len(self.swipes))

    def swipe(self, *args):
        self.swipes.append(args)

    def find_text(self, text, **kwargs):
        self.lookups += 1
        return (110, 180, 280, 230) if self.swipes and text == "Open" else None


class Adapter(_NeutralAdapter):
    capabilities = frozenset({"ui.tree", "ui.input"})

    def __init__(self, config, frames):
        super().__init__(config)
        self.frames = frames

    def normalize_tree(self, raw_tree, screen_size, **kwargs):
        return self.frames[min(int(raw_tree), len(self.frames) - 1)]


def engine(monkeypatch, frames):
    config = make_config(memory={"enabled": False}, capture={"enabled": False})
    runtime = Runtime()
    eng = Engine(config, device=runtime, platform=Adapter(config, frames))
    monkeypatch.setattr(eng, "_settle_after_swipe", lambda: None)

    def native_forbidden(*args, **kwargs):
        raise AssertionError("neutral scrolling cannot enter the Android adapter")

    monkeypatch.setattr(AndroidPlatform, "normalize_tree", native_forbidden)
    return eng, runtime


@pytest.mark.parametrize("method", ["scroll", "to_end", "to_start", "swipe", "scroll_to"])
def test_popup_is_unverified_without_replaying_or_matching_menu_text(monkeypatch, method):
    eng, runtime = engine(monkeypatch, [listing(), popup()])
    if method == "swipe":
        result = eng.swipe("down", percent=3, verify=True, observe=False)
    elif method == "scroll_to":
        result = eng.scroll_to("Open", observe=False)
    else:
        options = {method: True} if method in {"to_end", "to_start"} else {"pages": 5}
        result = eng.scroll("down", observe=False, **options)
    assert not result.ok
    assert result.detail.startswith("movement-unverified ")
    assert "evidence=container-unverified" in result.detail
    assert len(runtime.swipes) == 1
    assert runtime.tree_reads == 3, "reuse selection/before/after trees; no context reread"
    if method == "scroll_to":
        assert runtime.lookups == 1, "menu text is not the requested list destination"


@pytest.mark.parametrize(
    "changed",
    [
        popup(retained=True),
        listing(later=True, rid="other_cards"),
        listing(later=True, app_id="other.app"),
    ],
)
def test_retained_background_or_replaced_context_does_not_prove_movement(monkeypatch, changed):
    eng, runtime = engine(monkeypatch, [listing(), changed])
    result = eng.scroll("down", to_end=True, observe=False)
    assert not result.ok and result.detail.startswith("movement-unverified ")
    assert len(runtime.swipes) == 1


@pytest.mark.parametrize("anonymous", [False, True])
def test_virtualized_turnover_survives_new_frame_ids_and_row_contents(monkeypatch, anonymous):
    rid = None if anonymous else "cards"
    eng, runtime = engine(monkeypatch, [listing(rid=rid), listing(later=True, offset=300, rid=rid)])
    result = eng.scroll("up", observe=False)
    assert result.ok and result.detail.startswith("moved ")
    assert "evidence=content-turnover" in result.detail
    assert len(runtime.swipes) == 1 and runtime.tree_reads == 3


def test_real_axis_movement_and_a_verified_end_remain_successful(monkeypatch):
    before = listing()
    after = listing(shift=80, offset=300)
    # Remove sticky labels so the remaining shared labels measure the row translation.
    before.elements[:] = [e for e in before.elements if e.text not in {"All", "Saved"}]
    after.elements[:] = [e for e in after.elements if e.text not in {"All", "Saved"}]
    eng, runtime = engine(monkeypatch, [before, after, after])
    result = eng.scroll("up", to_end=True, observe=False)
    assert result.ok and result.detail.startswith("reached-end ")
    assert "dy=80" in result.detail and "evidence=axis-shift" in result.detail
    assert len(runtime.swipes) == 2


def test_popup_after_one_valid_step_does_not_become_a_reached_end(monkeypatch):
    eng, runtime = engine(monkeypatch, [listing(), listing(later=True), popup()])
    result = eng.scroll("up", to_end=True, observe=False)
    assert not result.ok and result.detail.startswith("movement-unverified ")
    assert "steps=1" in result.detail
    assert len(runtime.swipes) == 2


@pytest.mark.parametrize("later,shift", [(True, 0), (False, 80), (False, 0)])
def test_flat_tree_without_row_ownership_cannot_prove_an_end(monkeypatch, later, shift):
    before, after = listing(), listing(later=later, shift=shift)
    for frame in [before, after]:
        frame.elements[:] = [e.model_copy(update={"parent": None}) for e in frame.elements]
    eng, runtime = engine(monkeypatch, [before, after])
    result = eng.scroll("up", to_end=True, observe=False)
    assert not result.ok and result.detail.startswith("movement-unverified ")
    assert len(runtime.swipes) == 1


def test_duplicate_anonymous_overlay_root_does_not_disappear_from_context(monkeypatch):
    before, after = listing(), listing()
    for frame in [before, after]:
        frame.elements[0] = frame.elements[0].model_copy(update={"resource_id": None})
    after.elements.append(after.elements[0].model_copy(update={"id": 999}))
    eng, runtime = engine(monkeypatch, [before, after])
    result = eng.scroll("up", to_end=True, observe=False)
    assert not result.ok and result.detail.startswith("movement-unverified ")
    assert len(runtime.swipes) == 1


def test_android_popup_hierarchy_cannot_replace_a_scrollable_list(monkeypatch):
    before = """<hierarchy><node package="example.app" class="android.widget.ScrollView"
      resource-id="example.app:id/cards" scrollable="true" bounds="[0,100][300,550]">
      <node class="android.widget.TextView" text="First card" bounds="[10,200][100,240]"/>
      <node class="android.widget.TextView" text="Second card" bounds="[10,350][100,390]"/>
    </node></hierarchy>"""
    after = """<hierarchy><node package="example.app" class="android.widget.FrameLayout"
      resource-id="example.app:id/menu" bounds="[100,150][290,500]">
      <node class="android.widget.TextView" text="Open" bounds="[110,200][280,240]"/>
      <node class="android.widget.TextView" text="Share" bounds="[110,350][280,390]"/>
    </node></hierarchy>"""

    class AndroidFixture(FakeDevice):
        def swipe(self, *args, **kwargs):
            super().swipe(*args, **kwargs)
            self._xml = after

    runtime = AndroidFixture(hierarchy_xml=before, width=300, height=600)
    config = make_config(memory={"enabled": False}, capture={"enabled": False})
    eng = Engine(config, device=runtime)
    monkeypatch.setattr(eng, "_settle_after_swipe", lambda: None)
    result = eng.scroll("down", to_end=True, observe=False)
    assert not result.ok and result.detail.startswith("movement-unverified ")
    assert len([call for call in runtime.calls if call[0] == "swipe"]) == 1
    assert runtime.hierarchy_calls == 3
