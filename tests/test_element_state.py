"""Interaction-state fields: tri-state parsing (hierarchy) + what ``compact`` spends tokens on.

The contract these pin down is that a caller can READ a toggle instead of screenshotting it,
and can always tell *off* from *unknown*.
"""

from __future__ import annotations

import json

from android_ui_analyser.hierarchy import parse_hierarchy
from android_ui_analyser.schema import Element

SCREEN = (1080, 2400)

FULL_ATTRS = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node text="Push notifications" resource-id="com.x:id/settingsSwitch" class="android.widget.Switch"
        content-desc="" checkable="true" checked="true" clickable="true" enabled="true"
        focusable="true" focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false" bounds="[0,0][200,100]"/>
  <node text="" resource-id="com.x:id/pwd" class="android.widget.EditText" content-desc="Password"
        checkable="false" checked="false" clickable="true" enabled="false" focusable="true"
        focused="false" scrollable="false" long-clickable="true" password="true"
        selected="false" bounds="[0,120][200,220]"/>
  <node text="Browse" resource-id="com.x:id/tabBrowse" class="android.widget.TextView"
        content-desc="" checkable="false" checked="false" clickable="true" enabled="true"
        focusable="true" focused="false" scrollable="false" long-clickable="false"
        password="false" selected="true" bounds="[0,240][200,340]"/>
  <node text="" resource-id="com.x:id/list" class="android.widget.ScrollView" content-desc=""
        checkable="false" checked="false" clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="true" long-clickable="false" password="false"
        selected="false" bounds="[0,360][1080,2000]"/>
</hierarchy>"""

NO_ATTRS = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Bare" bounds="[0,0][200,100]"/>
</hierarchy>"""

STATE_FIELDS = ("checkable", "checked", "selected", "scrollable", "long_clickable", "password")


def _by_rid(xml: str) -> dict[str, Element]:
    return {(e.resource_id or "").rsplit("/", 1)[-1]: e for e in parse_hierarchy(xml, SCREEN)}


# --------------------------------------------------------------------------- parsing


def test_reported_attributes_become_real_booleans() -> None:
    els = _by_rid(FULL_ATTRS)
    assert (els["settingsSwitch"].checkable, els["settingsSwitch"].checked) == (True, True)
    assert els["pwd"].password is True
    assert els["pwd"].long_clickable is True
    assert els["tabBrowse"].selected is True
    assert els["list"].scrollable is True


def test_reported_false_stays_false_not_none() -> None:
    switch = _by_rid(FULL_ATTRS)["settingsSwitch"]
    assert switch.selected is False
    assert switch.password is False


def test_unreported_attributes_are_none() -> None:
    element = parse_hierarchy(NO_ATTRS, SCREEN)[0]
    for name in STATE_FIELDS:
        assert getattr(element, name) is None, name


def test_vision_style_elements_default_to_unknown_state() -> None:
    element = Element(id=0, type="Box", bounds=(0, 0, 10, 10), center=(5, 5))
    for name in STATE_FIELDS:
        assert getattr(element, name) is None, name


def test_disabled_element_keeps_its_long_standing_bool() -> None:
    assert _by_rid(FULL_ATTRS)["pwd"].enabled is False


# --------------------------------------------------------------------------- compact()


def test_compact_spends_tokens_only_on_state_worth_reporting() -> None:
    tab = _by_rid(FULL_ATTRS)["tabBrowse"].compact()
    assert tab["selected"] is True
    for name in ("checkable", "checked", "scrollable", "long_clickable", "password"):
        assert name not in tab, name


def test_compact_reports_a_checkable_nodes_off_state() -> None:
    off = Element(
        id=0, type="Switch", bounds=(0, 0, 10, 10), center=(5, 5), checkable=True, checked=False
    )
    assert off.compact()["checked"] is False
    on = off.model_copy(update={"checked": True})
    assert on.compact()["checked"] is True


def test_compact_marks_unknown_state_on_a_checkable_node_as_null() -> None:
    unknown = Element(
        id=0, type="Switch", bounds=(0, 0, 10, 10), center=(5, 5), checkable=True, checked=None
    )
    assert unknown.compact()["checked"] is None


def test_compact_stays_a_subset_of_the_full_dump() -> None:
    for element in parse_hierarchy(FULL_ATTRS, SCREEN):
        full = json.loads(element.model_dump_json())
        assert set(element.compact()) <= set(full)


def test_full_dump_always_carries_every_state_key() -> None:
    dumped = parse_hierarchy(NO_ATTRS, SCREEN)[0].model_dump(mode="json")
    assert all(name in dumped for name in STATE_FIELDS)
