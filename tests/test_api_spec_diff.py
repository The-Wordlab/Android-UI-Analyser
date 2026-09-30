"""Diffing two OpenAPI specs into the changes an already-shipped client can feel."""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.api_spec_diff import diff_specs, match_operations, normalize_path


def _op(fields: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": fields or {}}
    if required:
        schema["required"] = required
    return {"responses": {"200": {"content": {"application/json": {"schema": schema}}}}}


def _spec(
    paths: dict[str, dict[str, Any]], components: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec: dict[str, Any] = {"openapi": "3.1.0", "paths": paths}
    if components:
        spec["components"] = {"schemas": components}
    return spec


@pytest.mark.parametrize(
    ("raw", "base", "expected"),
    [
        ("items", "/api/v1/", "/api/v1/items"),
        ("/health", "/api/v1/", "/health"),  # Retrofit: a leading slash is host root
        ("items/{itemId}/parts", "/api/v1", "/api/v1/items/{}/parts"),
        ("items/\\(id)", "/api/v1/", "/api/v1/items/{}"),  # Swift interpolation
        ("/api/v1/items/${id}", "", "/api/v1/items/{}"),  # JS template literal
        ("items?page=2", "/api/v1/", "/api/v1/items"),
        ("https://api.example.test/api/v1/items/", "", "/api/v1/items"),
    ],
)
def test_every_codebase_spelling_of_a_route_meets_the_spec_spelling(raw, base, expected):
    assert normalize_path(raw, base) == expected


def test_a_client_reaches_the_literal_route_before_the_templated_one():
    known = {"GET /a/featured": {}, "GET /a/{}": {}, "POST /a/{}": {}}

    assert match_operations("GET", "/a/featured", known) == ["GET /a/featured"]
    assert match_operations("GET", "/a/42", known) == ["GET /a/{}"]
    # A scanner that cannot see the method reaches every method on the route.
    assert sorted(match_operations("*", "/a/{}", known)) == ["GET /a/{}", "POST /a/{}"]
    assert match_operations("GET", "/b", known) == []


def test_the_diff_reports_only_what_an_already_shipped_client_can_feel():
    node = {
        "type": "object",
        "properties": {"child": {"$ref": "#/components/schemas/Node"}, "id": {"type": "string"}},
    }
    base = _spec(
        {
            "/gone": {"get": _op()},
            "/items": {
                "get": _op(
                    {
                        "title": {"type": "string"},
                        "count": {"type": "integer"},
                        "owner": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                        },
                        "note": {"type": "string"},
                        "tree": {"$ref": "#/components/schemas/Node"},
                    },
                    required=["title", "note"],
                ),
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                }
                            }
                        }
                    },
                    "responses": {"201": {}},
                },
            },
        },
        {"Node": node},
    )
    head = _spec(
        {
            "/items": {
                "get": {
                    **_op(
                        {
                            "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "count": {"type": "string"},
                            "note": {"type": "string"},
                            "tree": {"$ref": "#/components/schemas/Node"},
                            "added": {"type": "string"},
                        },
                        required=["title"],
                    ),
                    "parameters": [{"in": "query", "name": "page", "required": True}],
                },
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "kind": {"type": "string"},
                                    },
                                    "required": ["kind"],
                                }
                            }
                        }
                    },
                    "responses": {"201": {}},
                },
            },
            "/new": {"get": _op()},
        },
        {"Node": node},
    )

    diff = diff_specs(base, head)
    found = {(c["operation"], c["change"], c.get("field")) for c in diff["changes"]}

    assert found == {
        ("GET /gone", "operation_removed", None),
        ("GET /items", "response_field_now_optional", "title"),  # became nullable
        ("GET /items", "response_field_type_changed", "count"),
        ("GET /items", "response_field_removed", "owner"),  # one finding, not one per child
        ("GET /items", "response_field_now_optional", "note"),  # no longer required
        ("GET /items", "request_parameter_now_required", "query:page"),
        ("POST /items", "request_field_now_required", "kind"),
    }
    assert diff["added_operations"] == ["GET /new"]


def test_a_required_field_inside_a_new_optional_object_demands_nothing_of_an_old_client():
    # Found on a real backend: a new optional `caps` object with a required child was reported
    # as a break for every client, though no old client can send `caps` at all.
    def post(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        body = {"content": {"application/json": {"schema": schema}}}
        return {"/send": {"post": {"requestBody": body, "responses": {"200": {}}}}}

    caps = {"type": "object", "properties": {"ids": {"type": "array"}}, "required": ["ids"]}
    known = {"type": "object", "properties": {"ids": {"type": "array"}}}
    opts_before = {"type": "object", "properties": {"mode": {"type": "string"}}}
    opts_after = {**opts_before, "required": ["mode"]}

    new_optional_parent = diff_specs(
        _spec(post({"text": {"type": "string"}})),
        _spec(post({"text": {"type": "string"}, "caps": {"anyOf": [caps, {"type": "null"}]}})),
    )
    old_parent_new_demand = diff_specs(
        _spec(post({"opts": opts_before, "known": known})),
        _spec(post({"opts": opts_after, "known": known})),
    )

    assert new_optional_parent["changes"] == []
    # An object an old client may already send is different: it now has to carry `mode`.
    assert [(c["change"], c["field"]) for c in old_parent_new_demand["changes"]] == [
        ("request_field_now_required", "opts.mode")
    ]


def test_a_field_that_only_changed_case_style_is_named_as_a_likely_rename():
    # The whole of one real backend's three-month history was this: snake_case → camelCase.
    def page(*names: str) -> dict[str, Any]:
        item = {"type": "object", "properties": {n: {"type": "string"} for n in names}}
        return _op({"items": {"type": "array", "items": item}, names[0]: {"type": "string"}})

    diff = diff_specs(
        _spec({"/feed": {"get": page("next_cursor", "created_at", "title")}}),
        _spec({"/feed": {"get": page("nextCursor", "createdAt", "title")}}),
    )

    assert {(c["field"], c.get("detail")) for c in diff["changes"]} == {
        ("next_cursor", "renamed to `nextCursor`?"),
        ("items[].next_cursor", "renamed to `items[].nextCursor`?"),
        ("items[].created_at", "renamed to `items[].createdAt`?"),
    }
