"""Window layer + --no-ime projection + a11y engine wiring."""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.hierarchy import classify_window, parse_hierarchy
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import OutputFormat
from conftest import FakeDevice, make_config

IME_XML = """\
<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node index="0" package="com.test.app" class="android.widget.FrameLayout"
        bounds="[0,0][1080,2400]" clickable="false">
    <node index="0" package="com.test.app" class="android.widget.TextView"
          text="Hello" resource-id="com.test.app:id/title"
          bounds="[0,100][200,160]" clickable="false"/>
    <node index="1" package="com.google.android.inputmethod.latin"
          class="android.widget.Button" text="q"
          resource-id="com.google.android.inputmethod.latin:id/key_q"
          bounds="[0,1800][100,1900]" clickable="true"/>
  </node>
</hierarchy>
"""


def test_classify_window() -> None:
    assert classify_window("com.test.app") == "app"
    assert classify_window("com.google.android.inputmethod.latin") == "ime"
    assert classify_window("com.android.systemui") == "system"
    assert classify_window("com.android.permissioncontroller") == "overlay"


def test_hierarchy_sets_window() -> None:
    els = parse_hierarchy(IME_XML, screen_size=(1080, 2400))
    by_text = {e.text: e for e in els if e.text}
    assert by_text["Hello"].window == "app"
    assert by_text["q"].window == "ime"


def test_no_ime_projection() -> None:
    els = parse_hierarchy(IME_XML, screen_size=(1080, 2400))
    payload = {
        "screen": {"width": 1080, "height": 2400},
        "elements": [e.model_dump(mode="json") for e in els],
        "meta": {},
    }
    view = Projection.parse(fmt=OutputFormat.json, no_ime=True)
    kept = view.select(payload)
    texts = {e.get("text") for e in kept}
    assert "Hello" in texts
    assert "q" not in texts


def test_a11y_scroll_records(tmp_path) -> None:  # noqa: ANN001
    xml = """\
<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node package="com.test.app" class="android.widget.ScrollView"
        resource-id="com.test.app:id/list" scrollable="true"
        bounds="[0,0][1080,800]" clickable="false"/>
</hierarchy>
"""
    device = FakeDevice(hierarchy_xml=xml)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)
    res = engine.analyze(source="hierarchy", record=False)
    assert res.elements
    eid = res.elements[0].id
    out = engine.a11y_scroll(eid, direction="forward", observe=False)
    assert out.ok
    assert ("a11y_action", (540, 400, "SCROLL_FORWARD")) in device.calls
