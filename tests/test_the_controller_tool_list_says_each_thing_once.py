"""Every tool schema carried its description twice, on every call, all run long.

The conversion lifts `parameters.description` up to `function.description` -- which is the field
an OpenAI-shaped tool list actually shows the model -- and then sends the original as well.
Measured on a real run: 1,970 of the tool list's 8,870 characters, or 22%, were that duplicate,
and the whole list is resent on every call. Over one eight-call run it is ~16k characters of text
the model has already read on the line above.

`oneOf` is not the problem and is left alone: it is 186 characters, 2%, and it is the only thing
saying a tap names its target exactly one way -- id, rid, text or desc, never two at once.
"""

from __future__ import annotations

import json

from experiments.aua_controller.run_realapp import realapp_tools


def _tools() -> list[dict]:
    """The real MCP schemas the controller is handed, through the real conversion."""
    from test_aua_controller_realapp import MCP_SCHEMAS  # noqa: PLC0415

    return realapp_tools(MCP_SCHEMAS)


def test_no_tool_repeats_its_description_inside_its_parameters() -> None:
    for tool in _tools():
        function = tool["function"]
        assert "description" not in (function.get("parameters") or {}), function["name"]


def test_the_description_that_survives_is_the_one_the_model_is_shown() -> None:
    """Dropping the wrong copy would silently strip the guidance instead of deduplicating it."""
    from test_aua_controller_realapp import MCP_SCHEMAS  # noqa: PLC0415

    tap = next(t["function"] for t in _tools() if t["function"]["name"] == "tap_and_analyze")
    assert "selector" in tap["description"], "the selector guidance must survive on the function"
    for tool in _tools():
        name = tool["function"]["name"]
        if (MCP_SCHEMAS.get(name) or {}).get("description"):
            assert tool["function"]["description"], f"{name} lost its description entirely"


def test_the_selector_rule_is_not_what_gets_dropped() -> None:
    """`oneOf` says a tap names its target exactly one way -- id, rid, text or desc, never two.

    It is 186 characters of the tool list's 8,870, or 2%, and it is the only thing that says so.
    Deduplicating descriptions must not take it with them.
    """
    from experiments.aua_controller.run_realapp import realapp_compact_schema  # noqa: PLC0415

    rule = [{"required": [key]} for key in ("id", "rid", "text", "desc")]
    kept = realapp_compact_schema("tap_and_analyze", {
        "type": "object", "description": "Tap something.",
        "properties": {"id": {"type": "string"}, "rid": {"type": "string"},
                       "text": {"type": "string"}, "desc": {"type": "string"}},
        "oneOf": rule})
    assert kept.get("oneOf") == rule


def test_the_list_actually_got_smaller() -> None:
    tools = _tools()
    with_dupes = json.dumps([
        {**t, "function": {**t["function"], "parameters": {
            **t["function"]["parameters"], "description": t["function"]["description"]}}}
        for t in tools])
    assert len(json.dumps(tools)) < len(with_dupes)
