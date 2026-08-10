"""Numeric ids are frame-local and must not silently move to different dynamic content."""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import StaleElementIdError, UsageError
from conftest import FakeDevice, make_config


def _row(label: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="{label}"
        resource-id="com.example.fiction:id/dynamic_row" clickable="true"
        long-clickable="true" enabled="true" bounds="[20,200][1060,340]"/>
</hierarchy>"""


def test_long_press_refuses_an_id_that_now_labels_a_different_row(tmp_path) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=_row("Draft Orion"))
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)
    observed = engine.analyze(source="hierarchy", record=False)
    old_id = next(element.id for element in observed.elements if element.text == "Draft Orion")

    # A live update reuses both the rid and the exact bounds, so a positional lookup (and a
    # rid-only remapper) would long-press the wrong content.
    device._xml = _row("Suggested reply")
    with pytest.raises(StaleElementIdError) as caught:
        engine.long_press(old_id, observe=False)

    assert caught.value.code == "stale_element_id"
    assert "No action was sent" in (caught.value.hint or "")
    assert not any(name == "long_click" for name, _args in device.calls)


def test_numeric_action_remaps_when_the_same_binding_is_renumbered(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first = """<hierarchy rotation="0">
      <node class="android.widget.Button" text="Archive nebula"
            resource-id="com.example.fiction:id/archive" clickable="true"
            bounds="[20,400][1060,520]"/>
    </hierarchy>"""
    reordered = """<hierarchy rotation="0">
      <node class="android.widget.TextView" text="New heading" bounds="[20,40][1060,120]"/>
      <node class="android.widget.Button" text="Archive nebula"
            resource-id="com.example.fiction:id/archive" clickable="true"
            bounds="[20,400][1060,520]"/>
    </hierarchy>"""
    device = FakeDevice(hierarchy_xml=first)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)
    old = engine.analyze(source="hierarchy", record=False).elements[0]

    device._xml = reordered
    result = engine.tap(old.id, observe=False)

    assert result.ok is True
    assert any(name == "click" for name, _args in device.calls)


def test_long_press_does_not_guess_through_a_sibling_control_subtree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    hierarchy = """<hierarchy rotation="0">
      <node class="android.view.ViewGroup" content-desc="Fictional tile group"
            bounds="[0,120][1080,520]">
        <node class="android.view.View" resource-id="com.example.fiction:id/tile_control"
              clickable="true" bounds="[40,160][1040,360]"/>
        <node class="android.widget.TextView" text="Nebula card"
              bounds="[40,380][1040,460]"/>
      </node>
    </hierarchy>"""
    device = FakeDevice(hierarchy_xml=hierarchy)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)

    with pytest.raises(UsageError) as caught:
        engine.long_press(selector={"text": "Nebula card"}, observe=False)

    assert caught.value.code == "unsafe_action_target"
    assert not any(name == "long_click" for name, _args in device.calls)
