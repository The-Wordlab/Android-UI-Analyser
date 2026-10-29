"""Preview checks prove artifact facts, never live app or touch behavior."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from android_ui_analyser import engine_previews as previews
from android_ui_analyser.errors import UsageError


@pytest.fixture
def component_tree(tmp_path):
    # Shape verified against Android CLI 1.1.16717390 output from the fictional fixture.
    data = {
        "previewId": "CatalogPreview", "methodFQN": "com.example.fixture.CatalogPreview",
        "width": 400, "height": 800, "densityDpi": 320,
        "components": [{
            "name": "Column", "bounds": {"left": 0, "top": 0, "right": 400, "bottom": 800},
            "children": [
                {"name": "Text", "bounds": {"left": 20, "top": 30, "right": 220, "bottom": 70},
                 "semantics": {"text": "Catalog"},
                 "sourceLocation": {"fileName": "Catalog.kt", "lineNumber": 12}},
                {"name": "Button", "bounds": {"left": 20, "top": 100, "right": 220, "bottom": 196},
                 "semantics": {"role": "Button", "testTag": "continue"}},
            ],
        }],
    }
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(data))
    return path


def test_render_requests_json_and_component_tree_through_neutral_engine(tmp_path):
    calls = []
    engine = SimpleNamespace(
        config=SimpleNamespace(cache=SimpleNamespace(dir=tmp_path)),
        development_tool=lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True, "data": {"success": True, "results": []}},
    )
    result = previews.preview_render(engine, ":app", fqns=["com.example.fixture.Preview"])
    assert calls[0][0] == ("preview.render",)
    assert calls[0][1]["options"]["component-tree"] is True
    assert calls[0][1]["options"]["format"] == "json"
    assert calls[0][1]["arguments"] == [":app"]
    assert result["runtime_verified"] is False
    assert result["interactive"] is False


def test_bad_render_request_cannot_build_or_connect():
    def forbidden(*args, **kwargs):
        pytest.fail("invalid request reached development tool")
    engine = SimpleNamespace(development_tool=forbidden)
    with pytest.raises(UsageError):
        previews.preview_render(engine, fqns=[])
    with pytest.raises(UsageError):
        previews.preview_render(engine, fqns=["Fixture.Preview"], component_tree=False, no_image=True)


def test_inspection_preserves_sources_semantics_and_dp(component_tree):
    result = previews.preview_inspect(None, str(component_tree), max_nodes=2)
    assert result["component_count"] == 3
    assert result["truncated"] is True
    text = result["components"][1]
    assert text["source"] == {"fileName": "Catalog.kt", "lineNumber": 12}
    assert text["text"] == "Catalog"
    assert text["width_dp"] == 100
    assert result["sha256"]
    assert "cannot be tapped" in result["id_scope"]


def test_checks_prove_exact_static_facts(component_tree):
    result = previews.preview_check(None, str(component_tree), [
        {"selector": {"text": "Catalog"}, "property": "exists", "expected": True},
        {"selector": {"name": "Text"}, "property": "count", "expected": 1},
        {"selector": {"test_tag": "continue"}, "property": "within_canvas", "expected": True},
        {"selector": {"test_tag": "continue"}, "property": "min_height_dp", "expected": 48},
        {"selector": {"text": "Missing"}, "property": "exists", "expected": False},
    ])
    assert result["ok"] and result["verified"]
    assert result["runtime_verified"] is False
    assert result["proof_kind"] == "static_component_tree_assertions"


def test_missing_or_ambiguous_nodes_cannot_pass_a_bounds_check(component_tree):
    result = previews.preview_check(None, str(component_tree), [
        {"selector": {"text": "Missing"}, "property": "within_canvas", "expected": False},
        {"selector": {"name": "Button"}, "property": "min_height_dp", "expected": 49},
    ])
    assert not result["ok"]
    assert not result["verified"]
    assert not any(item["ok"] for item in result["checks"])


@pytest.mark.parametrize("bad", [
    {"selector": {}, "property": "exists", "expected": True},
    {"selector": {"text": "Catalog"}, "property": "exists", "expected": "true"},
    {"selector": {"name": "Text"}, "property": "clickable", "expected": True},
    {"selector": {"name": "Text"}, "property": "count", "expected": -1},
    {"selector": {"name": "Text"}, "property": "count", "expected": float("nan")},
])
def test_invalid_check_does_not_become_a_vacuous_pass(component_tree, bad):
    with pytest.raises(UsageError):
        previews.preview_check(None, str(component_tree), [bad])


def test_malformed_tree_and_no_checks_are_rejected(tmp_path, component_tree):
    with pytest.raises(UsageError):
        previews.preview_check(None, str(component_tree), [])
    invalid = tmp_path / "unknown.json"
    invalid.write_text('{"components": [], "width": 0, "height": 800, "densityDpi": 160}')
    with pytest.raises(UsageError):
        previews.preview_inspect(None, str(invalid))


def test_mcp_render_inlines_only_its_produced_preview_image(tmp_path):
    from android_ui_analyser.mcp_server import _image_block

    path = tmp_path / "preview.png"
    contents = b"\x89PNG\r\n\x1a\nfictional-preview-fixture"
    path.write_bytes(contents)
    payload = {"ok": True, "previews": [{"imagePath": str(path)}]}
    image = _image_block("preview_render", payload)
    assert image is not None
    assert image.mimeType == "image/png"
    assert base64.b64decode(image.data) == contents
    assert _image_block("preview_render", {**payload, "ok": False}) is None
    assert _image_block("preview_render", {"ok": True, "previews": []}) is None
