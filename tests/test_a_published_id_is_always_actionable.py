"""Whatever names an element in an observation must still address it in the next call.

Three ways that broke, all of them reachable from the dashboard and two of them from the
CLI/MCP as well:

* the numeric id cache is one shared file per device, so a caller that publishes ids
  without recording them hands out numbers that are validated against a *different*
  caller's screen — observed on 2026-08-21 as ``element id 30 is stale for tap: binding
  'rid:bc_smartspace_view' changed`` when id 30 was in fact the app's own intro card;
* an element only OCR could see is absent from the hierarchy-only freshness read, so its
  id can never be confirmed and every tap on it reports a changed binding;
* ``stable_key`` — the one name that outlives the frame it came from — could not be used
  to act at all, so a caller holding an observation from another process had no safe way
  to address anything in it.
"""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import ChainSpec, TextBox
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, StubOcr, make_config

_SCREEN = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Nebula inbox" bounds="[40,80][1040,160]"/>
  <node class="android.widget.Button" text="Continue"
        resource-id="com.example.fiction:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,400][1040,520]"/>
</hierarchy>"""

_TWO_ROWS = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.Button" text="First"
        resource-id="com.example.fiction:id/row" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
  <node class="android.widget.Button" text="Second"
        resource-id="com.example.fiction:id/row" clickable="true" enabled="true"
        bounds="[40,400][1040,520]"/>
</hierarchy>"""


class _FusedOcr(StubOcr):
    """Named for the one provider hierarchy+OCR fusion accepts (see _start_hierarchy_ocr)."""

    name = "apple_vision"


class _OcrFactory(ProviderFactory):
    """A factory whose only enabled perception is one fixed OCR reading."""

    def __init__(self, config: Any, boxes: list[TextBox]) -> None:
        super().__init__(config)
        self._boxes = boxes

    def is_enabled(self, kind: str) -> bool:
        return kind == "ocr"

    def build_chain(self, kind: str) -> ChainSpec:
        if kind != "ocr":
            return ChainSpec(kind=kind, providers=[])
        return ChainSpec(kind="ocr", providers=[_FusedOcr(result=list(self._boxes))])


def _engine(tmp_path: Any, device: FakeDevice, boxes: list[TextBox] | None = None) -> Engine:
    # conftest disables hierarchy augmentation by default; an OCR-fusion test must opt in.
    config = make_config(
        cache={"dir": str(tmp_path)},
        ocr={"enabled": bool(boxes), "augment_hierarchy": bool(boxes)},
    )
    factory = _OcrFactory(config, boxes or [])
    return Engine(config, device=device, factory=factory)


def test_a_stable_key_taps_with_no_id_cache_at_all(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The dashboard's situation: it holds an observation, but not this device's id file."""
    device = FakeDevice(hierarchy_xml=_SCREEN)
    engine = _engine(tmp_path, device)

    result = engine.tap(selector={"key": "rid:continue_btn"}, observe=False)

    assert result.ok is True
    assert any(name == "click" for name, _args in device.calls)


def test_a_stable_key_that_left_the_screen_is_refused_not_guessed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=_SCREEN)
    engine = _engine(tmp_path, device)

    with pytest.raises(Exception) as caught:
        engine.tap(selector={"key": "rid:not_on_this_screen"}, observe=False)

    assert getattr(caught.value, "code", "") in {"element_not_found", "selector_not_found"}
    assert not any(name == "click" for name, _args in device.calls)


def test_a_repeated_key_is_disambiguated_by_the_bounds_it_was_published_with(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Reusable row layouts give every row the same rid; position is the only evidence left."""
    device = FakeDevice(hierarchy_xml=_TWO_ROWS)
    engine = _engine(tmp_path, device)

    result = engine.tap(
        selector={"key": "rid:row", "bounds": [40, 400, 1040, 520]}, observe=False
    )

    assert result.ok is True
    clicks = [args for name, args in device.calls if name == "click"]
    assert clicks, "no gesture was sent"
    assert clicks[0][1] > 320, f"tapped the wrong row: {clicks[0]}"


def test_a_repeated_key_with_no_position_is_ambiguous_rather_than_a_coin_flip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=_TWO_ROWS)
    engine = _engine(tmp_path, device)

    with pytest.raises(Exception) as caught:
        engine.tap(selector={"key": "rid:row"}, observe=False)

    assert getattr(caught.value, "code", "") == "selector_ambiguous"
    assert not any(name == "click" for name, _args in device.calls)


def test_an_id_that_only_ocr_could_see_is_not_reported_as_a_changed_binding(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The hierarchy cannot describe a canvas label, so a hierarchy-only recheck cannot either."""
    device = FakeDevice(hierarchy_xml=_SCREEN)
    engine = _engine(tmp_path, device, [TextBox(text="SEE MORE", bounds=(760, 900, 1030, 990))])
    observed = engine.analyze(source="hierarchy", with_ocr=True)
    ocr = [element for element in observed.elements if element.source == "ocr"]
    assert ocr, "the fixture no longer produces an OCR-only element"

    result = engine.tap(ocr[0].id, observe=False)

    assert result.ok is True
    assert any(name == "click" for name, _args in device.calls)


def test_a_key_that_only_ocr_could_see_still_addresses_its_element(tmp_path) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(hierarchy_xml=_SCREEN)
    engine = _engine(tmp_path, device, [TextBox(text="SEE MORE", bounds=(760, 900, 1030, 990))])
    observed = engine.analyze(source="hierarchy", with_ocr=True)
    ocr = next(element for element in observed.elements if element.source == "ocr")
    assert ocr.stable_key

    result = engine.tap(
        selector={"key": ocr.stable_key, "bounds": list(ocr.bounds)}, observe=False
    )

    assert result.ok is True
    assert any(name == "click" for name, _args in device.calls)


_OTHER_SCREEN = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Archive" bounds="[40,80][1040,160]"/>
  <node class="android.widget.Button" text="Send"
        resource-id="com.example.fiction:id/send_btn" clickable="true" enabled="true"
        bounds="[40,600][1040,720]"/>
</hierarchy>"""


def test_bypassing_the_cache_still_records_the_ids_it_handed_back(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``--no-cache`` means "do not reuse a stale result", never "publish unusable ids".

    The two were one flag, so a caller that asked for a fresh screen got numbers validated
    against the previous one — the same divergence the dashboard shipped, reachable from the
    CLI and from MCP.
    """
    device = FakeDevice(hierarchy_xml=_SCREEN)
    engine = _engine(tmp_path, device)
    engine.analyze(source="hierarchy")

    device._xml = _OTHER_SCREEN
    observed = engine.analyze(source="hierarchy", no_cache=True)
    send = next(
        element
        for element in observed.elements
        if (element.resource_id or "").endswith("send_btn")
    )

    result = engine.tap(send.id, observe=False)

    assert result.ok is True
    assert any(name == "click" for name, _args in device.calls)
