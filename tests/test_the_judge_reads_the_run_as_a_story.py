"""The judge reads the run as a story: what was done, then what appeared -- one line each.

It used to get two lists and a join key: an `action_log` of steps, and `intermediate_frames` of
raw element dicts, linked only by `after_step`. Deciding "did pressing App language show the
language list" meant finding step 3 in one list, then the frame whose evidence_position said 3
in the other, then reading forty dicts. The request was long, and its author could not read it
either. Now each observation is one entry: the action in plain words, the screen as one line of
its own labels, and only the extra facts that apply to it (network, unchanged, loading).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from experiments.aua_controller.judgement import annotate_judge_frames, judge_outcome, judge_story


def _raw(title: str, *rows: str, fp: str, selected: str | None = None,
         network: list[str] | None = None) -> dict:
    elements: list[dict[str, Any]] = [{"id": "el:title", "text": title, "bounds": [0, 0, 100, 10]}]
    for i, row in enumerate(rows):
        element = {"id": "el:" + row.lower().replace(" ", "_"), "text": row, "clickable": True,
                   "bounds": [0, 10 * (i + 1), 100, 10 * (i + 2)]}
        if row == selected:
            element["selected"] = True
        elements.append(element)
    meta: dict[str, Any] = {"fingerprint": fp}
    if network:
        meta["network_calls"] = network
    return {"ok": True, "observation": {"screen": {"package": "example.app", "width": 100, "height": 200},
                                        "meta": meta, "elements": elements}}


ENTRIES = [
    {"tool": "app_launch_and_analyze", "ref": "E0", "step": None,
     "raw": _raw("Welcome", "Sign in", "Browse as a guest", fp="f0")},
    {"tool": "tap_and_analyze", "ref": "E1", "step": 0,
     "raw": _raw("Home", "Chat", "Settings", fp="f1", network=["GET /v1/profile -> 200"])},
    {"tool": "tap_and_analyze", "ref": "E2", "step": 1,
     "raw": _raw("Settings", "Theme", "App language en", fp="f2")},
    {"tool": "scroll_and_analyze", "ref": "E3", "step": 2,
     "raw": _raw("Settings", "Theme", "App language en", fp="f2")},          # nothing moved
    {"tool": "tap_and_analyze", "ref": "E4", "step": 3,
     "raw": _raw("App language", "English", "Spanish", fp="f4", selected="English")},
    {"tool": "back_gesture_and_analyze", "ref": "E5", "step": 4,
     "raw": _raw("Settings", "Theme", "App language en", fp="f5")},
]
ACTIONS = [
    {"step": 0, "tool": "tap_and_analyze", "arguments": {"id": "el:sign_in"}},
    {"step": 1, "tool": "tap_and_analyze", "arguments": {"text": "Settings"}},
    {"step": 2, "tool": "scroll_and_analyze", "arguments": {"direction": "down"}},
    {"step": 3, "tool": "tap_and_analyze", "arguments": {"id": "el:gone"},
     "resolved_target": {"resource_id": "settingLanguage"}},
    {"step": 4, "tool": "back_gesture_and_analyze", "arguments": {}},
]
FRAMES = annotate_judge_frames(ENTRIES)


class _Decider:
    """Captures the payload instead of answering it."""

    max_tokens = 2048

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def decide(self, **kwargs: Any) -> Any:
        self.seen.append(kwargs)
        criteria = list(kwargs.get("criteria_order") or [])
        return {"result": {"verdict": "unverified", "confidence": 0.5, "reasons": [],
                           "criteria": [{"criterion": name, "result": "not_verified", "evidence": ""}
                                        for name in criteria]}}


def test_one_entry_per_observation_in_the_order_it_happened() -> None:
    story = judge_story(FRAMES, ACTIONS)
    assert [entry["ref"] for entry in story] == ["E0", "E1", "E2", "E3", "E4", "E5"]
    assert all({"ref", "step", "action", "screen"} <= set(entry) for entry in story)


def test_each_action_is_told_in_plain_words() -> None:
    # An id is resolved against the screen it was chosen on; a text selector is itself; an id
    # that no shown screen carries falls back to what AUA resolved it to.
    assert [entry["action"] for entry in judge_story(FRAMES, ACTIONS)] == [
        "open the app", "press 'Sign in'", "press 'Settings'", "scroll down",
        "press 'settingLanguage'", "back"]


def test_the_screen_is_one_line_of_its_own_labels() -> None:
    story = judge_story(FRAMES, ACTIONS)
    assert story[0]["screen"] == "Welcome · [Sign in] · [Browse as a guest]"
    assert story[4]["screen"] == "App language · [English] ✓ · [Spanish]"


def test_extra_facts_appear_only_where_they_apply() -> None:
    story = judge_story(FRAMES, ACTIONS)
    assert story[1]["network"] == ["GET /v1/profile -> 200"]
    assert "network" not in story[2]
    assert story[3]["changed"] is False, "a scroll that moved nothing must say so"
    assert "changed" not in story[2]


def test_steps_the_sampler_left_out_are_named_so_nothing_is_assumed() -> None:
    story = judge_story([FRAMES[0], FRAMES[2], FRAMES[5]], ACTIONS)
    assert story[1]["steps_not_shown"] == [{"step": 0, "action": "press 'Sign in'"}]
    assert story[2]["steps_not_shown"] == [{"step": 2, "action": "scroll down"},
                                           {"step": 3, "action": "press 'settingLanguage'"}]
    assert "steps_not_shown" not in story[0]


def test_the_judge_is_sent_the_story_and_nothing_raw() -> None:
    decider = _Decider()
    asyncio.run(judge_outcome(decider, goal="Switch the app language", final_frame=FRAMES[-1],
                              frames=FRAMES[:-1], actions=ACTIONS,
                              contract="- Settings offers an app-language entry."))
    context = decider.seen[0]["context"]
    assert "journey" in context and context["final"]["ref"] == "E5"
    for gone in ("intermediate_frames", "action_log", "final_frame", "layout_evidence_note"):
        assert gone not in context, gone
    dumped = json.dumps(context)
    for raw in ('"id"', '"bounds"', "center_pct", "el:", "evidence_position"):
        assert raw not in dumped, raw


def test_the_question_speaks_in_the_storys_own_terms() -> None:
    decider = _Decider()
    asyncio.run(judge_outcome(decider, goal="g", final_frame=FRAMES[-1], frames=FRAMES[:-1],
                              actions=ACTIONS, contract="- A bullet."))
    question = decider.seen[0]["question"]
    assert "journey" in question and "steps_not_shown" in question
    for stale in ("evidence_position", "intermediate_frames", "final frame is the current screen"):
        assert stale not in question, stale
