"""Deeplink chooser dismissal."""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config

CHOOSER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Open with" bounds="[0,0][1080,100]"/>
  <node index="1" class="android.widget.TextView" text="Luzia Dev" clickable="true"
        bounds="[40,200][1040,300]"/>
  <node index="2" class="android.widget.Button" text="Just once" clickable="true"
        bounds="[40,400][500,480]"/>
  <node index="3" class="android.widget.Button" text="Always" clickable="true"
        bounds="[540,400][1040,480]"/>
</hierarchy>"""

APP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Chats" bounds="[0,0][1080,100]"/>
</hierarchy>"""


def test_open_link_pins_package() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=APP_XML)
    engine = Engine(cfg, device=device)
    result = engine.open_link("luzia://chats", package="co.thewordlab.luzia.dev", observe=False)
    assert result.ok
    assert ("open_link", ("luzia://chats", "co.thewordlab.luzia.dev")) in device.calls


def test_dismiss_chooser_taps_prefer() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=CHOOSER_XML)
    device._pkg = "android"
    device._act = ".internal.app.ResolverActivity"
    engine = Engine(cfg, device=device)

    dumps = [CHOOSER_XML, CHOOSER_XML, APP_XML]

    def rotating() -> str:
        device.hierarchy_calls += 1
        if dumps:
            device._xml = dumps.pop(0)
        return device._xml

    device.dump_hierarchy = rotating  # type: ignore[method-assign]
    assert engine._dismiss_chooser(prefer="co.thewordlab.luzia.dev")
    assert any(c[0] == "click" for c in device.calls)
