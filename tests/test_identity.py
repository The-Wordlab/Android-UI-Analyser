"""Cross-frame identity + resolve."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.hierarchy import parse_hierarchy
from android_ui_analyser.identity import (
    attach_visual_stable_keys,
    find_by_stable_key,
    remap_ids,
    stable_key,
)
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.schema import Element
from conftest import FakeDevice, make_config, make_png

HIERARCHY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Hello" bounds="[0,0][1080,120]"/>
  <node index="1" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
  <node index="2" class="android.widget.EditText" content-desc="Email"
        resource-id="com.test.app:id/email" clickable="true" enabled="true"
        bounds="[40,400][1040,500]"/>
</hierarchy>"""

REORDERED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.EditText" content-desc="Email"
        resource-id="com.test.app:id/email" clickable="true" enabled="true"
        bounds="[40,50][1040,150]"/>
  <node index="1" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
  <node index="2" class="android.widget.TextView" text="Hello" bounds="[0,400][1080,520]"/>
</hierarchy>"""

VISUAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.view.View" clickable="true" enabled="true"
        bounds="[20,20][100,100]"/>
</hierarchy>"""


def _visual_frame(*, horizontal: bool = False, offset: int = 0) -> bytes:
    shape = (
        ((30, 55 + offset, 90, 65 + offset), (0, 0, 0))
        if horizontal
        else ((55 + offset, 30, 65 + offset, 90), (0, 0, 0))
    )
    return make_png(width=200, height=200, color=(245, 245, 245), boxes=[shape])


def _element(
    element_id: int,
    *,
    bounds: tuple[int, int, int, int] = (20, 20, 100, 100),
    stable: str | None = None,
    text: str | None = None,
    rid: str | None = None,
    clickable: bool = True,
) -> Element:
    x1, y1, x2, y2 = bounds
    return Element(
        id=element_id,
        type="View",
        text=text,
        resource_id=rid,
        bounds=bounds,
        center=((x1 + x2) // 2, (y1 + y2) // 2),
        clickable=clickable,
        stable_key=stable,
    )


def test_stable_key_prefers_resource_id() -> None:
    els = parse_hierarchy(HIERARCHY_XML, (1080, 1920))
    by_rid = {e.resource_id: e for e in els if e.resource_id}
    assert stable_key(by_rid["com.test.app:id/continue_btn"]) == "rid:continue_btn"
    assert all(e.stable_key for e in els)


def test_remap_ids_across_reorder() -> None:
    a = parse_hierarchy(HIERARCHY_XML, (1080, 1920))
    b = parse_hierarchy(REORDERED_XML, (1080, 1920))
    mapping = remap_ids(a, b)
    cont_a = next(e for e in a if e.resource_id and e.resource_id.endswith("continue_btn"))
    cont_b = next(e for e in b if e.resource_id and e.resource_id.endswith("continue_btn"))
    assert mapping[cont_a.id] == cont_b.id


def test_engine_resolve_by_id_and_key() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=HIERARCHY_XML)
    engine = Engine(cfg, device=device)
    first = engine.analyze(source="hierarchy")
    cont = next(e for e in first.elements if (e.resource_id or "").endswith("continue_btn"))
    old_id = cont.id
    key = cont.stable_key
    assert key == "rid:continue_btn"

    # Keep prior cache, change next dump so remapping is meaningful.
    engine._write_cache(first)
    device._xml = REORDERED_XML
    resolved = engine.resolve(old_id, fresh=True)
    assert resolved.ok
    assert resolved.stable_key == key
    assert resolved.to_id is not None
    assert resolved.element is not None
    assert (resolved.element.resource_id or "").endswith("continue_btn")

    by_key = engine.resolve(key, fresh=True)
    assert by_key.ok and by_key.to_id == resolved.to_id


def test_visual_keys_only_replace_unlabeled_actionable_geometry() -> None:
    image = ScreenImage(_visual_frame(), width=200, height=200)
    unlabeled = _element(0)
    labeled = _element(1, bounds=(105, 20, 190, 100), text="Open archive")
    resource = _element(2, bounds=(20, 105, 100, 190), rid="app:id/back")
    passive = _element(3, bounds=(105, 105, 190, 190), clickable=False)

    attached = attach_visual_stable_keys(
        [unlabeled, labeled, resource, passive], image, screen_size=(200, 200)
    )

    assert attached[0].stable_key is not None and attached[0].stable_key.startswith("px:View:")
    assert attached[1].stable_key is not None and attached[1].stable_key.startswith("tx:")
    assert attached[2].stable_key == "rid:back"
    assert attached[3].stable_key is not None and attached[3].stable_key.startswith("geo:")


def test_pixel_keys_remap_by_visual_distance_and_keep_legacy_geo_compatible() -> None:
    previous = _element(7, stable="px:View:000000000000000f")
    close = _element(2, bounds=(24, 20, 104, 100), stable="px:View:000000000000000d")
    far = _element(3, stable="px:View:ffffffffffffffff")

    assert remap_ids([previous], [far, close]) == {7: 2}
    assert find_by_stable_key([far, close], previous.stable_key or "") == [close]

    legacy = _element(8)
    legacy = legacy.model_copy(update={"stable_key": stable_key(legacy)})
    current = _element(9, stable="px:View:1234567890abcdef")
    assert remap_ids([legacy], [current]) == {8: 9}
    assert find_by_stable_key([current], legacy.stable_key or "") == [current]


def test_real_crop_hash_tolerates_small_render_jitter_but_not_a_different_shape() -> None:
    element = _element(1)

    def attached(frame: bytes, element_id: int) -> Element:
        candidate = element.model_copy(update={"id": element_id, "stable_key": None})
        return attach_visual_stable_keys(
            [candidate], ScreenImage(frame, width=200, height=200), screen_size=(200, 200)
        )[0]

    original = attached(_visual_frame(), 1)
    shifted = attached(_visual_frame(offset=1), 2)
    different = attached(_visual_frame(horizontal=True), 3)

    assert original.stable_key is not None and original.stable_key.startswith("px:")
    assert remap_ids([original], [shifted]) == {1: 2}
    assert remap_ids([original], [different]) == {}


def test_duplicate_visual_keys_still_require_spatial_evidence() -> None:
    previous = _element(4, bounds=(20, 20, 80, 80), stable="px:View:0123456789abcdef")
    overlapping = _element(5, bounds=(22, 22, 82, 82), stable=previous.stable_key)
    distant = _element(6, bounds=(120, 120, 180, 180), stable=previous.stable_key)
    assert remap_ids([previous], [distant, overlapping]) == {4: 5}

    moved_a = _element(7, bounds=(100, 20, 160, 80), stable=previous.stable_key)
    moved_b = _element(8, bounds=(20, 100, 80, 160), stable=previous.stable_key)
    assert remap_ids([previous], [moved_a, moved_b]) == {}


def test_engine_uses_one_platform_neutral_screenshot_for_visual_identity() -> None:
    device = FakeDevice(
        hierarchy_xml=VISUAL_XML,
        width=200,
        height=200,
        screenshot_bytes=_visual_frame(),
    )
    engine = Engine(make_config(memory={"enabled": False}), device=device)

    result = engine.analyze(source="hierarchy", with_ocr=False, record=False)

    assert device.screenshot_calls == 1
    assert len(result.elements) == 1
    assert result.elements[0].stable_key is not None
    assert result.elements[0].stable_key.startswith("px:View:")


def test_unchanged_hierarchy_does_not_reuse_a_changed_visual_key() -> None:
    device = FakeDevice(
        hierarchy_xml=VISUAL_XML,
        width=200,
        height=200,
        screenshots=[_visual_frame(), _visual_frame(horizontal=True)],
    )
    engine = Engine(make_config(memory={"enabled": False}), device=device)

    first = engine.analyze(source="hierarchy", with_ocr=False, record=False)
    second = engine.analyze(source="hierarchy", with_ocr=False, record=False)

    assert device.screenshot_calls == 2
    assert first.elements[0].stable_key != second.elements[0].stable_key
    assert second.meta.unchanged is False


def test_with_image_saves_the_exact_frame_used_for_pixel_identity(tmp_path: Path) -> None:
    first_frame = _visual_frame()
    device = FakeDevice(
        hierarchy_xml=VISUAL_XML,
        width=200,
        height=200,
        screenshots=[first_frame, _visual_frame(horizontal=True)],
    )
    engine = Engine(make_config(memory={"enabled": False}), device=device)
    output = tmp_path / "inspection.png"

    result = engine.analyze(
        source="hierarchy", with_ocr=False, with_image=str(output), record=False
    )

    assert device.screenshot_calls == 1
    assert output.read_bytes() == first_frame
    assert result.meta.raw_image == str(output)
