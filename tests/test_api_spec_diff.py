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
        ("GET /items", "response_field_now_nullable", "title"),
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
        ("items[].next_cursor", "renamed to `nextCursor`?"),
        ("items[].created_at", "renamed to `createdAt`?"),
    }


def _changes(base: dict[str, Any], head: dict[str, Any]) -> set[tuple[Any, ...]]:
    return {
        (c["operation"], c["change"], c.get("field"), c.get("detail"))
        for c in diff_specs(base, head)["changes"]
    }


def _enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def test_a_new_value_in_a_response_enum_is_a_change_though_it_is_an_addition():
    # A strict Codable or Moshi enum without a fallback fails on a value it has never seen.
    base = _spec({"/a": {"get": _op({"state": _enum("on", "off"), "kind": _enum("x")})}})
    head = _spec(
        {"/a": {"get": _op({"state": _enum("on", "off", "paused"), "kind": {"type": "string"}})}}
    )

    assert _changes(base, head) == {
        ("GET /a", "response_enum_value_added", "state", "adds `paused`"),
        ("GET /a", "response_enum_value_added", "kind", "no longer an enum"),
    }


def test_optional_and_nullable_are_different_promises_and_both_are_watched():
    base = _spec(
        {
            "/a": {
                "get": _op(
                    {"hint": {"type": "string"}, "name": {"type": "string"}}, required=["name"]
                )
            }
        }
    )
    head = _spec(
        {
            "/a": {
                "get": _op(
                    {
                        "hint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "name": {"type": "string"},
                    }
                )
            }
        }
    )

    assert _changes(base, head) == {
        # Optional before, but never null: a defaulted non-null property breaks on `null`.
        ("GET /a", "response_field_now_nullable", "hint", None),
        ("GET /a", "response_field_now_optional", "name", None),
    }


def test_array_elements_and_widened_response_types_are_compared():
    def tags(items: dict[str, Any], total: Any) -> dict[str, Any]:
        return _spec(
            {"/a": {"get": _op({"tags": {"type": "array", "items": items}, "total": total})}}
        )

    narrowed = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    changes = _changes(
        tags({"type": "string"}, narrowed), tags({"type": "integer"}, {"type": "integer"})
    )

    # A response that narrows (`integer|string` → `integer`) only sends what clients already take.
    assert changes == {
        ("GET /a", "response_field_type_changed", "tags", "array<string> → array<integer>")
    }


def test_a_request_that_narrows_what_it_accepts_is_a_change():
    def op(page: dict[str, Any], sort: dict[str, Any], note: dict[str, Any]) -> dict[str, Any]:
        body = {"type": "object", "properties": {"note": note}}
        return _spec(
            {
                "/a": {
                    "post": {
                        "parameters": [
                            {"in": "query", "name": "page", "schema": page},
                            {"in": "query", "name": "sort", "schema": sort},
                        ],
                        "requestBody": {"content": {"application/json": {"schema": body}}},
                        "responses": {"200": {}},
                    }
                }
            }
        )

    nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    base = op({"type": "string"}, _enum("new", "old", "top"), nullable)
    head = op({"type": "integer"}, _enum("new", "top"), {"type": "string"})

    assert _changes(base, head) == {
        ("POST /a", "request_parameter_type_changed", "query:page", "string → integer"),
        ("POST /a", "request_enum_value_removed", "query:sort", "rejects `old`"),
        ("POST /a", "request_field_no_longer_nullable", "note", None),
    }


def test_union_branches_merge_whatever_order_the_spec_lists_them_in():
    def union(*branches: dict[str, Any]) -> dict[str, Any]:
        return _spec(
            {
                "/a": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {"schema": {"anyOf": list(branches)}}
                                }
                            }
                        }
                    }
                }
            }
        )

    def branch(*kinds: str) -> dict[str, Any]:
        return {"type": "object", "properties": {"kind": _enum(*kinds)}}

    assert _changes(union(branch("a"), branch("b")), union(branch("b", "c"), branch("a"))) == {
        ("GET /a", "response_enum_value_added", "kind", "adds `c`")
    }


def test_every_success_response_is_compared_not_only_the_first():
    def op(created: dict[str, Any]) -> dict[str, Any]:
        def body(props: dict[str, Any]) -> dict[str, Any]:
            schema = {"type": "object", "properties": props}
            return {"content": {"application/json": {"schema": schema}}}

        return _spec(
            {
                "/a": {
                    "post": {
                        "responses": {"200": body({"id": {"type": "string"}}), "201": body(created)}
                    }
                }
            }
        )

    assert _changes(op({"url": {"type": "string"}}), op({})) == {
        ("POST /a", "response_field_removed", "url", None)
    }


def test_a_field_is_attributed_to_the_named_schema_that_declares_it():
    score = {"type": "object", "properties": {"points": {"type": "integer"}}}
    reply = {"type": "object", "properties": {"score": {"$ref": "#/components/schemas/Score"}}}
    base = _spec(
        {
            "/score": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Score"}
                                }
                            }
                        }
                    }
                }
            },
            "/chat": {
                "post": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Reply"}
                                }
                            }
                        }
                    }
                }
            },
        },
        {"Score": score, "Reply": reply},
    )
    head = _spec(base["paths"], {"Score": {"type": "object", "properties": {}}, "Reply": reply})

    changes = diff_specs(base, head)["changes"]

    assert {(c["operation"], c["field"], c["schema"], c["schema_field"]) for c in changes} == {
        ("GET /score", "points", "Score", "points"),
        ("POST /chat", "score.points", "Score", "points"),
    }
