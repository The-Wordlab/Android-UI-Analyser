"""Saved-flow selector quality is explicit rather than inferred from YAML spelling."""

from __future__ import annotations

from android_ui_analyser.flows import (
    describe_selector_resilience,
    recorded_selector_resilience,
)
from android_ui_analyser.memory import RouteStep


def test_selector_strength_ladder_and_localization_risk() -> None:
    rid = describe_selector_resilience(rid="openDetails")
    desc = describe_selector_resilience(desc="Open details")
    text = describe_selector_resilience(text="Details")
    current_id = describe_selector_resilience(element_id=17)

    assert (rid.selector, rid.strength, rid.localization_risk, rid.cross_frame) == (
        "rid",
        "strong",
        False,
        True,
    )
    assert (desc.selector, desc.strength, desc.localization_risk) == (
        "desc",
        "medium",
        True,
    )
    assert (text.selector, text.strength, text.localization_risk) == (
        "text",
        "weak",
        True,
    )
    assert (current_id.selector, current_id.strength, current_id.cross_frame) == (
        "id",
        "frame_only",
        False,
    )
    assert "observation" in current_id.reason


def test_recorded_selector_disclosure_is_value_free_and_recursive() -> None:
    steps = [
        RouteStep(kind="tap", resource_id="openDetails", by="id"),
        RouteStep(kind="input", label="Localized prompt", by="text", index=1),
        RouteStep(
            kind="retry",
            max_retries=2,
            substeps=[RouteStep(kind="clear", content_desc="Clear field", by="desc")],
        ),
    ]

    disclosure = recorded_selector_resilience(steps)
    payload = [item.model_dump(mode="json") for item in disclosure]

    assert [item["selector"] for item in payload] == ["rid", "text", "desc"]
    assert payload[1]["index_sensitive"] is True
    assert payload[2]["step"] == "step 3 > step 1"
    rendered = repr(payload)
    assert "openDetails" not in rendered
    assert "Localized prompt" not in rendered
    assert "Clear field" not in rendered


def test_legacy_composite_selector_does_not_claim_resource_id_strength() -> None:
    legacy = RouteStep(
        kind="tap",
        resource_id="openDetails",
        content_desc="Open details",
        label="Details",
        by=None,
    )
    explicit_rid = legacy.model_copy(update={"by": "rid"})

    legacy_disclosure, rid_disclosure = recorded_selector_resilience([legacy, explicit_rid])

    assert (
        legacy_disclosure.selector,
        legacy_disclosure.strength,
        legacy_disclosure.cross_frame,
        legacy_disclosure.localization_risk,
    ) == ("composite", "unknown", False, True)
    assert (rid_disclosure.selector, rid_disclosure.strength, rid_disclosure.cross_frame) == (
        "rid",
        "strong",
        True,
    )
    rendered = repr(
        [legacy_disclosure.model_dump(mode="json"), rid_disclosure.model_dump(mode="json")]
    )
    assert "openDetails" not in rendered
    assert "Open details" not in rendered
    assert "Details" not in rendered
