"""A failed action still reports the screen; the model must be allowed to read it.

When AUA refuses an action it did not send -- a stale handle, an id that no longer resolves --
it answers with the error AND a full observation of what is actually on screen, nested under
``error.observation``. The failure is about the action, not about the screen: the screen is as
readable as any other. Reading only the top-level ``observation`` made every such step look
like a blank screen, so the navigator declined ``too_few_controls`` and the step fell through
to the chat model, at full price, on a frame it could have answered itself.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.compaction import compact_frame  # noqa: E402
from experiments.aua_controller.typesafe_navigator import (  # noqa: E402
    TAP_TOOL,
    TypeSafeNavigator,
    candidates,
    screen_for_model,
)

from test_the_system_one_navigator_only_answers_confident_taps import (  # noqa: E402
    SCREEN,
    FakeClient,
)

REFUSED = {
    "ok": False,
    "error": {
        "code": "stale_element_id",
        "message": "element id 4 is stale for tap: binding 'Privacy' changed",
        "hint": "No action was sent. Re-run `aua analyze`.",
        "observation_present": True,
        "observation": SCREEN["observation"],
    },
}


def test_a_refused_action_still_carries_a_readable_screen() -> None:
    frame = compact_frame(REFUSED, keep_ids=True)
    elements = (frame.get("observation") or {}).get("elements") or []
    assert [e.get("text") for e in elements] == ["Notifications", "Privacy", "Version 1.2.3"]


def test_the_refusal_itself_survives_so_the_model_knows_the_action_failed() -> None:
    frame = compact_frame(REFUSED, keep_ids=True)
    assert frame["ok"] is False
    assert frame["error"]["code"] == "stale_element_id"


def test_a_top_level_observation_still_wins_over_a_nested_one() -> None:
    both = dict(SCREEN, error={"code": "x", "observation": {"screen": {}, "elements": []}})
    frame = compact_frame(both, keep_ids=True)
    assert len((frame.get("observation") or {}).get("elements") or []) == 3


def test_a_frame_with_no_screen_anywhere_is_still_left_alone() -> None:
    bare = {"ok": False, "error": {"code": "device_offline", "message": "no device"}}
    frame = compact_frame(bare, keep_ids=True)
    assert frame["error"]["code"] == "device_offline"
    assert "observation" not in frame


def test_the_controls_of_a_refused_action_are_offered_as_choices() -> None:
    frame = compact_frame(REFUSED, keep_ids=True)
    assert len(candidates(frame.get("observation"))) == 2


def test_the_screen_shown_to_the_model_is_the_one_behind_the_error() -> None:
    frame = compact_frame(REFUSED, keep_ids=True)
    state = screen_for_model(frame)
    assert "Notifications" in str(state)


def test_the_navigator_answers_a_refused_action_instead_of_paying_the_chat_model() -> None:
    client = FakeClient()
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=[TAP_TOOL])
    proposal = asyncio.run(navigator(REFUSED))
    assert proposal is not None, "the screen was readable; this should not have declined"
    assert proposal["tool"] == TAP_TOOL
    assert client.calls == 1
