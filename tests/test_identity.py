"""Cross-frame identity + resolve."""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.hierarchy import parse_hierarchy
from android_ui_analyser.identity import remap_ids, stable_key
from conftest import FakeDevice, make_config

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
