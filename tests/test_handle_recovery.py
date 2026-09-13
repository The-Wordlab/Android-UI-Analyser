"""Changing status text and identical controls must have a usable action path."""
from __future__ import annotations

import pytest

from android_ui_analyser.errors import ElementNotFoundError, UsageError
from android_ui_analyser.mcp_server import _selector_from_args
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import AnalyzeResult, Element
from test_element_handles import BootDevice, clicks, engine_for, rows


def panel(label: str, *, label_type: str = "TextView") -> str:
    return f'''<hierarchy>
      <node package="com.example.notes" class="android.widget.FrameLayout"
        resource-id="com.example.notes:id/content" bounds="[0,0][1080,2000]">
        <node package="com.example.notes" class="android.widget.{label_type}" text="{label}" bounds="[10,10][900,90]"/>
        <node package="com.example.notes" class="android.widget.Button" text="Open" clickable="true" enabled="true"
          resource-id="com.example.notes:id/open" bounds="[10,100][200,200]"/>
      </node></hierarchy>'''


@pytest.mark.parametrize(("before", "after", "kind"), [
    ("Timer: 10", "Timer: 11", "TextView"),
    ("Elapsed time: 00:10", "Elapsed time: 00:11", "TextView"),
    ("Progress: 10%", "Progress: 11%", "TextView"),
    ("00:10", "00:11", "Chronometer"),
    ("12:10", "12:11", "TextClock"),
])
def test_explicit_running_status_does_not_rename_sibling_button(before, after, kind):
    device = BootDevice(hierarchy_xml=panel(before, label_type=kind))
    engine = engine_for(device)
    observed = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    target = next(el["id"] for el in observed["elements"] if el.get("text") == "Open")
    device._xml = panel(after, label_type=kind)
    assert engine.tap(selector=_selector_from_args({"id": target}), observe=False).ok
    assert len(clicks(device)) == 1


@pytest.mark.parametrize(("before", "after"), [
    ("Delete Orion?", "Delete Vega?"),
    ("Delete item 10?", "Delete item 11?"),
    ("Timer: 10 for Orion", "Timer: 11 for Vega"),
])
def test_generic_dialog_subject_still_invalidates_its_button(before, after):
    device = BootDevice(hierarchy_xml=panel(before))
    engine = engine_for(device)
    observed = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    target = next(el["id"] for el in observed["elements"] if el.get("text") == "Open")
    device._xml = panel(after)
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)


def test_duplicates_offer_explicit_positions_without_weakening_handle_identity():
    device = BootDevice(hierarchy_xml=rows("Duplicate", "Duplicate"))
    engine = engine_for(device)
    observation = engine.analyze(source="hierarchy", with_ocr=False)
    payload = observation.as_dict("compact")
    first, second = payload["elements"]
    assert first["id_reusable"] is second["id_reusable"] is False
    assert first["selector"] == {"rid": "com.example.fiction:id/row", "index": 0}
    assert second["selector"] == {"rid": "com.example.fiction:id/row", "index": 1}
    assert AnalyzeResult.model_validate(payload).as_dict("compact")["elements"] == payload["elements"]

    with pytest.raises(ElementNotFoundError) as failure:
        engine.tap(first["id"], observe=False)
    assert not clicks(device)
    assert "refreshing its ID will not help" in failure.value.hint
    assert engine.tap(selector=_selector_from_args(second["selector"]), observe=False).ok
    assert len(clicks(device)) == 1 and clicks(device)[0][1] > 400


def test_filtering_keeps_original_selector_index_and_folded_recovery_fields():
    device = BootDevice(hierarchy_xml=rows("Duplicate", "Duplicate"))
    engine = engine_for(device)
    payload = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    view = Projection.parse(
        fields=engine.config.output.observation_fields,
        region=["0,400,1080,1000"],
    )
    filtered = view.apply(payload, fmt="compact")
    assert len(filtered["elements"]) == 1
    assert filtered["elements"][0]["id_reusable"] is False
    assert filtered["elements"][0]["selector"]["index"] == 1


def test_context_miss_returns_a_resource_selector_without_repeating_the_action():
    device = BootDevice(hierarchy_xml=panel("Subject Orion"))
    engine = engine_for(device)
    observed = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    target = next(el["id"] for el in observed["elements"] if el.get("text") == "Open")
    device._xml = panel("Subject Vega")
    with pytest.raises(ElementNotFoundError) as failure:
        engine.tap(target, observe=False)
    assert not clicks(device)
    evidence = failure.value.observation
    payload = evidence.as_dict("compact") if hasattr(evidence, "as_dict") else evidence
    button = next(el for el in payload["elements"] if el.get("text") == "Open")
    assert button["selector"] == {"rid": "com.example.notes:id/open"}


def test_explicit_second_position_is_refused_when_only_one_match_remains():
    device = BootDevice(hierarchy_xml=rows("Duplicate", "Duplicate"))
    engine = engine_for(device)
    observed = engine.analyze(source="hierarchy", with_ocr=False).as_dict("compact")
    selector = observed["elements"][1]["selector"]
    device._xml = rows("Duplicate")
    with pytest.raises(UsageError, match="out of range"):
        engine.tap(selector=_selector_from_args(selector), observe=False)
    assert not clicks(device)


def test_default_tsv_exposes_non_reusable_id_and_selector():
    engine = engine_for(BootDevice(hierarchy_xml=rows("Duplicate", "Duplicate")))
    observation = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    view = Projection.parse(fmt="tsv")
    rendered = view.render_tsv(view.apply(observation, fmt="json"))
    assert "id_reusable" in rendered and "selector" in rendered
    assert "false" in rendered and "'index': 1" in rendered


def test_selector_does_not_depend_on_a_vision_candidate_missing_from_fresh_hierarchy():
    from android_ui_analyser.element_handles import selector_alternative

    first = Element(id=1, type="Button", text="Open", bounds=(0, 0, 100, 100), center=(50, 50), source="hierarchy")
    detected = first.model_copy(update={"id": 2, "source": first.source.__class__("detection")})
    second = first.model_copy(update={"id": 3, "bounds": (0, 200, 100, 300)})
    assert selector_alternative(detected, [first, detected, second]) is None
    assert selector_alternative(second, [first, detected, second]) is None
    assert selector_alternative(second, [first, second]) == {"text": "Open", "index": 1}
