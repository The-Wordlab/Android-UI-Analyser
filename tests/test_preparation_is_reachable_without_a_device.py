"""An MCP-only agent must be able to hold the whole conversation, and hold it off a device.

Preparation is a conversation about an app, not a run on one. If it needed a lease, the cheapest
and most useful part of a session - agreeing what is actually being proven - would queue behind a
device that is not needed yet, and two agents preparing two claims could not do it at once.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _LEASE_FREE_TOOLS, _dispatch, _tool_definitions
from conftest import FakeDevice, make_config

_HIERARCHY = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<hierarchy rotation="0">'
    '<node index="0" class="android.widget.TextView" text="Hi" bounds="[0,0][1080,120]"/>'
    "</hierarchy>"
)
PACKAGE = "com.example.app"
PREPARE_TOOLS = (
    "prepare_start",
    "prepare_answer",
    "prepare_show",
    "prepare_discard",
    "prepare_list",
)


def _engine() -> Engine:
    return Engine(make_config(), device=FakeDevice(hierarchy_xml=_HIERARCHY))


def test_every_step_of_the_conversation_is_published_and_needs_no_lease() -> None:
    tools = {tool.name: tool for tool in _tool_definitions()}
    for name in PREPARE_TOOLS:
        assert name in tools, f"MCP has no {name}; preparation would be CLI-only"
        assert name in _LEASE_FREE_TOOLS, f"{name} must not queue behind a device it does not use"
    assert set(tools["prepare_answer"].inputSchema["required"]) == {
        "package",
        "prepare_id",
        "answers",
    }
    assert set(tools["prepare_discard"].inputSchema["required"]) == {"package", "prepare_id"}


def test_the_whole_interview_runs_over_mcp_and_ends_in_a_contract() -> None:
    engine = _engine()

    opened = _dispatch(
        engine,
        "prepare_start",
        {"package": PACKAGE, "goal": "the hub badge shows once on first open"},
    )
    assert opened["ok"] and not opened["ready"]
    prepare_id = opened["prepare_id"]

    partial = _dispatch(
        engine,
        "prepare_answer",
        {"package": PACKAGE, "prepare_id": prepare_id, "answers": {"scope": "ui"}},
    )
    assert partial["answers"]["scope"] == "ui"
    assert not partial["ready"]

    finished = _dispatch(
        engine,
        "prepare_answer",
        {
            "package": PACKAGE,
            "prepare_id": prepare_id,
            "agent": "test-agent",
            "answers": {
                "build": "/tmp/app-debug.apk",
                "signin": "guest",
                "precondition": "hubBadgeSeen absent",
                "seeding": "datastore",
                "success": "rid:hubBadge",
                "repeat": "!rid:hubBadge",
                "restore": "none",
            },
        },
    )
    assert finished["saved"]
    assert finished["remembered_ids"]
    assert finished["provenance"][0]["from_answer"] == "success"

    listed = _dispatch(engine, "prepare_list", {"package": PACKAGE})
    assert listed["scenarios"][0]["scenario"] == "the-hub-badge-shows-once-on-first-open"
    assert listed["in_progress"] == []

    abandoned = _dispatch(
        engine,
        "prepare_start",
        {"package": PACKAGE, "goal": "a different unfinished claim"},
    )
    dropped = _dispatch(
        engine,
        "prepare_discard",
        {"package": PACKAGE, "prepare_id": abandoned["prepare_id"]},
    )
    assert dropped == {"ok": True, "prepare_id": abandoned["prepare_id"], "discarded": True}


def test_an_answer_call_with_nothing_in_it_is_refused(monkeypatch: Any) -> None:
    engine = _engine()
    opened = _dispatch(engine, "prepare_start", {"package": PACKAGE, "goal": "badge shows once"})
    try:
        _dispatch(
            engine,
            "prepare_answer",
            {"package": PACKAGE, "prepare_id": opened["prepare_id"], "answers": {}},
        )
    except Exception as exc:  # AuaError, surfaced to the MCP client as a usage error
        assert "non-empty" in str(exc)
    else:  # pragma: no cover - the call must not silently succeed
        raise AssertionError("an empty answers object was accepted")
