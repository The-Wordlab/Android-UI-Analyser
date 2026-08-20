"""Deeplink package pin + chooser failure."""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from conftest import FakeDevice, make_config

CHOOSER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Open with" bounds="[0,0][1080,100]"/>
  <node index="1" class="android.widget.TextView" text="Example App Dev" clickable="true"
        bounds="[40,200][1040,300]"/>
  <node index="2" class="android.widget.TextView" text="Example App" clickable="true"
        bounds="[40,320][1040,400]"/>
  <node index="3" class="android.widget.Button" text="Just once" clickable="true"
        bounds="[40,500][500,580]"/>
  <node index="4" class="android.widget.Button" text="Always" clickable="true"
        bounds="[540,500][1040,580]"/>
</hierarchy>"""

APP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Chats" bounds="[0,0][1080,100]"/>
</hierarchy>"""


def test_open_link_defaults_to_foreground_package_pin() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=APP_XML, package="com.example.app.dev")
    engine = Engine(cfg, device=device)
    result = engine.open_link("myapp://orders", observe=False)
    assert result.ok
    assert ("open_link", ("myapp://orders", "com.example.app.dev")) in device.calls


def test_open_link_no_package_pin_skips_pin() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=APP_XML, package="com.example.app.dev")
    engine = Engine(cfg, device=device)
    engine.open_link("myapp://orders", pin_package=False, observe=False)
    assert ("open_link", ("myapp://orders",)) in device.calls


def test_open_link_errors_when_chooser_persists() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=CHOOSER_XML)
    device._pkg = "android"
    device._act = ".internal.app.ResolverActivity"
    engine = Engine(cfg, device=device)
    with pytest.raises(DeviceError) as ei:
        engine.open_link("myapp://orders", pin_package=False, observe=False)
    assert "Open with" in str(ei.value)
    assert "Example App Dev" in (ei.value.hint or "")


def test_dismiss_chooser_taps_prefer() -> None:
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=CHOOSER_XML)
    device._pkg = "android"
    device._act = ".internal.app.ResolverActivity"
    engine = Engine(cfg, device=device)

    dumps = [CHOOSER_XML, CHOOSER_XML, APP_XML]

    def rotating(*_args: object, **_kwargs: object) -> str:
        device.hierarchy_calls += 1
        if dumps:
            device._xml = dumps.pop(0)
        return device._xml

    device.dump_hierarchy = rotating  # type: ignore[method-assign]
    assert engine._dismiss_chooser(prefer="com.example.app.dev")
    assert any(c[0] == "click" for c in device.calls)
