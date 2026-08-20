"""New recorded selectors are stable, private, and fail closed when ambiguous."""

from __future__ import annotations

import pytest

from android_ui_analyser.memory import RouteStep, is_destructive_step, recorded_selector
from android_ui_analyser.schema import Element
from android_ui_analyser.selectors import match_step


def _element(
    element_id: int,
    *,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
    kind: str = "android.widget.Button",
) -> Element:
    top = element_id * 20
    return Element(
        id=element_id,
        type=kind,
        text=text,
        content_desc=desc,
        resource_id=rid,
        bounds=(0, top, 100, top + 20),
        center=(50, top + 10),
        clickable=True,
    )


@pytest.mark.parametrize("field", ["text", "desc"])
def test_recorded_selector_checks_uniqueness_after_persisted_truncation(field: str) -> None:
    prefix = "A" * 60
    first = _element(1, **{field: prefix + " first"})
    second = _element(2, **{field: prefix + " second"})

    selector = recorded_selector(second, elements=[first, second])

    assert selector == {
        "resource_id": None,
        "content_desc": None,
        "label": None,
        "by": None,
    }


@pytest.mark.parametrize(
    ("step", "elements"),
    [
        (
            RouteStep(kind="tap", resource_id="row", by="id"),
            [_element(1, rid="x:id/row"), _element(2, rid="x:id/row")],
        ),
        (
            RouteStep(kind="tap", content_desc="Open", by="desc"),
            [_element(1, desc="Open"), _element(2, desc="Open")],
        ),
        (
            RouteStep(kind="tap", label="Open", by="text"),
            [_element(1, text="Open"), _element(2, text="Open")],
        ),
    ],
)
def test_strict_recorded_selector_refuses_duplicate_runtime_match(
    step: RouteStep, elements: list[Element]
) -> None:
    assert match_step(elements, step) is None
    assert match_step(elements, step.model_copy(update={"index": 1})) is elements[1]


@pytest.mark.parametrize(
    ("step", "expected_id"),
    [
        (
            RouteStep(
                kind="tap",
                resource_id="wrongButton",
                content_desc="Intended",
                by="desc",
            ),
            2,
        ),
        (
            RouteStep(
                kind="tap",
                resource_id="wrongButton",
                content_desc="Wrong",
                label="Intended",
                by="text",
            ),
            2,
        ),
    ],
)
def test_explicit_by_ignores_supplemental_selector_fields(
    step: RouteStep, expected_id: int
) -> None:
    elements = [
        _element(1, rid="x:id/wrongButton", text="Wrong", desc="Wrong"),
        _element(2, rid="x:id/intendedButton", text="Intended", desc="Intended"),
    ]

    matched = match_step(elements, step)

    assert matched is not None and matched.id == expected_id


def test_legacy_selector_still_keeps_first_match_compatibility() -> None:
    elements = [_element(1, text="Open"), _element(2, text="Open")]
    assert match_step(elements, RouteStep(kind="tap", label="Open")) is elements[0]


def test_secret_field_keeps_stable_id_but_never_persists_text_or_description() -> None:
    field = _element(
        1,
        rid="fiction:id/passwordField",
        text="private value",
        desc="Account password",
        kind="android.widget.EditText",
    )
    selector = recorded_selector(field, elements=[field])
    assert selector == {
        "resource_id": "passwordField",
        "content_desc": None,
        "label": None,
        "by": "id",
    }

    no_id = field.model_copy(update={"resource_id": None})
    assert recorded_selector(no_id, elements=[no_id])["by"] is None


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        (
            {"desc": "Open profile", "text": "person@example.test"},
            {
                "resource_id": None,
                "content_desc": "Open profile",
                "label": None,
                "by": "desc",
            },
        ),
        (
            {"text": "Open profile"},
            {
                "resource_id": None,
                "content_desc": None,
                "label": "Open profile",
                "by": "text",
            },
        ),
    ],
    ids=["description", "text"],
)
def test_pii_resource_id_falls_back_to_safe_semantic_selector(
    fallback: dict[str, str], expected: dict[str, str | None]
) -> None:
    element = _element(1, rid="fiction:id/profile_person@example.test", **fallback)

    assert recorded_selector(element, elements=[element]) == expected


def test_pii_resource_id_without_independently_safe_fallback_is_refused() -> None:
    element = _element(
        1,
        rid="fiction:id/profile_person@example.test",
        text="person@example.test",
    )
    assert recorded_selector(element, elements=[element]) == {
        "resource_id": None,
        "content_desc": None,
        "label": None,
        "by": None,
    }


@pytest.mark.parametrize("rid", ["passwordField", "pinEntry", "session_token"])
def test_secret_semantic_resource_id_remains_usable_without_supplemental_copy(rid: str) -> None:
    element = _element(
        1,
        rid=f"fiction:id/{rid}",
        text="private value",
        desc="Continue",
    )

    assert recorded_selector(element, elements=[element]) == {
        "resource_id": rid,
        "content_desc": None,
        "label": None,
        "by": "id",
    }


def test_stable_non_pii_resource_id_remains_preferred() -> None:
    element = _element(1, rid="fiction:id/profileSettings", text="Profile settings")

    assert recorded_selector(element, elements=[element]) == {
        "resource_id": "profileSettings",
        "content_desc": None,
        "label": "Profile settings",
        "by": "id",
    }


@pytest.mark.parametrize("rid", ["deleteAccount", "sign_out_button", "pay-now"])
def test_destructive_resource_id_is_tokenized(rid: str) -> None:
    assert is_destructive_step(
        RouteStep(kind="tap", resource_id=rid, by="id"),
        ["delete", "sign out", "pay"],
    )
