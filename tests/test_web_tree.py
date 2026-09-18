from __future__ import annotations

import json

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.platforms import web_tree


def _snapshot(*nodes: dict, url: str = "https://example.test/app") -> str:
    return json.dumps({"format": web_tree.TREE_FORMAT, "url": url, "nodes": list(nodes)})


def test_web_dom_normalizes_visible_semantics_and_stable_ids() -> None:
    tree = web_tree.normalize(
        _snapshot(
            {
                "tag": "h1",
                "text": "Welcome",
                "bounds": [20, 10, 220, 60],
                "enabled": True,
            },
            {
                "tag": "input",
                "input_type": "email",
                "description": "Email address",
                "resource_id": "email",
                "bounds": [20, 80, 300, 120],
                "clickable": True,
                "enabled": True,
                "focused": True,
                "parent": 0,
            },
            {
                "tag": "button",
                "text": "Continue",
                "resource_id": "submit",
                "bounds": [20, 140, 160, 185],
                "clickable": True,
                "enabled": True,
                "parent": 0,
            },
        ),
        (360, 640),
    )

    assert tree.app_id == "example.test"
    assert [element.text for element in tree.elements] == ["Welcome", None, "Continue"]
    assert tree.elements[1].content_desc == "Email address"
    assert tree.elements[1].type == "TextField"
    assert tree.elements[1].focused is True
    assert tree.elements[2].stable_key == "rid:submit"
    assert tree.elements[2].parent == 0


def test_web_tree_excludes_offscreen_nodes_and_clips_partial_nodes() -> None:
    tree = web_tree.normalize(
        _snapshot(
            {"tag": "button", "text": "Above", "bounds": [0, -80, 100, -20]},
            {"tag": "button", "text": "Partial", "bounds": [-10, 20, 100, 70]},
            {"tag": "button", "text": "Below", "bounds": [0, 700, 100, 740]},
        ),
        (360, 640),
    )

    assert [element.text for element in tree.elements] == ["Partial"]
    assert tree.elements[0].bounds == (0, 20, 100, 70)


@pytest.mark.parametrize(
    ("by", "query", "expected"),
    [
        ("text", "Continue", (20, 140, 160, 185)),
        ("rid", "submit", (20, 140, 160, 185)),
        ("id", "submit", (20, 140, 160, 185)),
        ("desc", "next step", (20, 140, 160, 185)),
    ],
)
def test_web_find_bounds_uses_the_same_viewport_contract(
    by: str, query: str, expected: tuple[int, int, int, int]
) -> None:
    raw = _snapshot(
        {
            "tag": "button",
            "text": "Continue",
            "description": "Go to next step",
            "resource_id": "submit",
            "bounds": [20, 140, 160, 185],
        },
        {
            "tag": "button",
            "text": "Continue",
            "resource_id": "hidden-submit",
            "bounds": [20, 800, 160, 845],
        },
    )

    assert web_tree.find_bounds(raw, screen_size=(360, 640), query=query, by=by) == expected


def test_web_find_bounds_rejects_unknown_selector_fields() -> None:
    with pytest.raises(UsageError, match="unknown selector field"):
        web_tree.find_bounds(_snapshot(), screen_size=(360, 640), query="x", by="css")


def test_web_app_identity_does_not_publish_a_full_url() -> None:
    assert web_tree.app_id("https://example.test/private?token=secret") == "example.test"
