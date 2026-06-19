"""AC7 — output schema: validation, compact-is-a-subset, pretty round-trips (PRD §8)."""

from __future__ import annotations

import json

from android_ui_analyser.schema import (
    SCHEMA_VERSION,
    AnalyzeResult,
    Element,
    Meta,
    PathKind,
    Screen,
    ScreenSource,
    Source,
    Tier,
)


def _sample() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="com.x",
            activity=".Main",
            source=ScreenSource.hierarchy,
        ),
        elements=[
            Element(
                id=0,
                type="Button",
                text="Sign in",
                resource_id="com.x:id/in",
                bounds=(10, 20, 200, 80),
                center=(105, 50),
                clickable=True,
            ),
            Element(
                id=1,
                type="Text",
                text=None,
                bounds=(0, 0, 50, 20),
                center=(25, 10),
                source=Source.ocr,
                confidence=0.93,
            ),
        ],
        meta=Meta(
            duration_ms=42,
            tier_used=Tier.hierarchy,
            path=PathKind.hierarchy,
            providers_used=[],
            device_serial="emulator-5554",
        ),
    )


def test_schema_version_is_one() -> None:
    assert _sample().schema_version == SCHEMA_VERSION == 1


def test_json_is_single_line_and_validates_back() -> None:
    out = _sample().render("json")
    assert "\n" not in out
    AnalyzeResult.model_validate(json.loads(out))  # validates against the model


def test_pretty_round_trips() -> None:
    res = _sample()
    pretty = res.render("pretty")
    assert "\n" in pretty  # indented
    assert AnalyzeResult.model_validate(json.loads(pretty)) == res


def test_compact_is_strict_subset_and_drops_defaults() -> None:
    res = _sample()
    full = json.loads(res.render("json"))
    compact = json.loads(res.render("compact"))

    for ce, fe in zip(compact["elements"], full["elements"], strict=True):
        assert set(ce).issubset(set(fe))  # never invents keys
        if fe["enabled"] is True:
            assert "enabled" not in ce  # default dropped
        if fe["focused"] is False:
            assert "focused" not in ce
        if fe["confidence"] is None:
            assert "confidence" not in ce
        if fe["text"] is None:
            assert "text" not in ce
        if fe["source"] == "hierarchy":
            assert "source" not in ce  # default source dropped
    # id/type/bounds/center always present
    for ce in compact["elements"]:
        assert {"id", "type", "bounds", "center"} <= set(ce)


def test_compact_keeps_nondefault_fields() -> None:
    compact = json.loads(_sample().render("compact"))
    ocr_el = next(e for e in compact["elements"] if e["id"] == 1)
    assert ocr_el["source"] == "ocr"  # non-default source kept
    assert "confidence" in ocr_el  # non-null confidence kept
    btn = next(e for e in compact["elements"] if e["id"] == 0)
    assert btn["clickable"] is True  # non-default clickable kept


def test_element_compact_helper() -> None:
    el = Element(id=3, type="X", bounds=(0, 0, 1, 1), center=(0, 0))
    d = el.compact()
    assert d == {"id": 3, "type": "X", "bounds": [0, 0, 1, 1], "center": [0, 0]}
