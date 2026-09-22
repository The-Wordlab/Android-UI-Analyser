"""The navigator's value is in what it refuses, so the refusals are what these pin.

Every path that is not a confident tap must return None, because None is what hands the step
back to the chat model. A navigator that answers a step it should not have is worse than one
that answers nothing.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.compaction import compact_frame  # noqa: E402
from experiments.aua_controller.typesafe_navigator import (  # noqa: E402
    TAP_TOOL,
    TypeSafeNavigator,
    build_questions,
    candidates,
    numbered,
    screen_for_model,
    what_happened,
)

SCREEN = {
    "ok": True,
    "observation": {
        "screen": {"package": "com.example.demo", "activity": ".Settings"},
        "meta": {"fingerprint": "fp-1"},
        "elements": [
            {"id": "el:aaa", "text": "Notifications", "clickable": True, "bounds": [0, 0, 7, 7]},
            {"id": "el:bbb", "text": "Privacy", "clickable": True, "bounds": [0, 8, 7, 15]},
            {"id": "el:ccc", "text": "Version 1.2.3", "bounds": [0, 16, 7, 23]},
        ],
    },
}


class FakeClient:
    def __init__(self, kind="tap", target="1", kind_conf=0.99, target_conf=0.95,
                 settled=0.02, error=None):
        self.kind, self.target = kind, target
        self.kind_conf, self.target_conf, self.settled = kind_conf, target_conf, settled
        self.error = error
        self.calls = 0

    async def system_one(self, *, state, questions, model, timeout=None):
        self.calls += 1
        if self.error:
            raise self.error
        # One question now: a press is the numbered control itself, so `kind="tap"` here means
        # "press whichever control `target` names" and every other kind answers as itself.
        choice = self.target if self.kind == "tap" else self.kind
        confidence = self.target_conf if self.kind == "tap" else self.kind_conf
        return SimpleNamespace(
            answers={"move": SimpleNamespace(choice=choice, confidence=confidence)},
            usage=SimpleNamespace(input_tokens=430),
        )


def propose(client, **kwargs):
    navigator = TypeSafeNavigator("Open notification settings", client=client,
                                  tools=[TAP_TOOL], **kwargs)
    return asyncio.run(navigator(SCREEN)), navigator


def test_a_confident_tap_is_proposed_as_a_bound_tool_call() -> None:
    action, navigator = propose(FakeClient())
    assert action == {"tool": TAP_TOOL, "arguments": {"id": "el:aaa"},
                      "reason": action["reason"]}
    assert "0.95" in action["reason"]
    assert navigator.report()["accepted"] == 1


@pytest.mark.parametrize("kind", ["achieved", "back", "scroll_down", "type"])
def test_every_action_that_is_not_a_tap_goes_back_to_the_chat_model(kind: str) -> None:
    # Ending, rewinding, scrolling and typing were the measured weak spots; none of them is
    # this navigator's to decide, however sure it sounds.
    action, navigator = propose(FakeClient(kind=kind, kind_conf=1.0, target_conf=1.0))
    assert action is None
    assert navigator.report()["declined"] == {f"kind:{kind}": 1}


def test_a_tap_below_the_gate_is_declined() -> None:
    action, navigator = propose(FakeClient(target_conf=0.62))
    assert action is None
    assert navigator.report()["declined"] == {"below_confidence": 1}


def test_one_answer_carries_one_confidence() -> None:
    # There is no second question about a press to disagree with, so the press's own number is
    # the whole gate. `min()` of an action and an operand was two numbers about different things.
    assert propose(FakeClient(target_conf=0.55))[0] is None
    assert propose(FakeClient(target_conf=0.99))[0] is not None


def test_a_target_that_is_not_on_this_screen_is_refused() -> None:
    action, navigator = propose(FakeClient(target="el:zzz"))
    assert action is None
    assert navigator.report()["declined"] == {"unknown_target": 1}


def test_a_request_failure_costs_a_step_not_a_run() -> None:
    action, navigator = propose(FakeClient(error=TimeoutError("slow")))
    assert action is None
    assert navigator.report()["declined"] == {"request_failed:TimeoutError": 1}


def test_shadow_mode_records_the_tap_it_would_have_taken_and_takes_nothing() -> None:
    action, navigator = propose(FakeClient(), shadow=True)
    assert action is None
    report = navigator.report()
    assert report["shadow"] is True and report["accepted"] == 0
    # `target` is the menu index the model answered; `target_id` is what it means.
    assert navigator.proposals[0]["target_id"] == "el:aaa"


def test_a_screen_without_a_real_choice_is_not_worth_a_request() -> None:
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    bare = {"ok": True, "observation": {"screen": {}, "meta": {},
                                        "elements": [{"id": "el:only", "text": "OK", "clickable": True}]}}
    assert asyncio.run(navigator(bare)) is None
    assert client.calls == 0, "one control is not a choice; do not pay for the question"


def test_a_run_that_was_not_offered_tap_never_proposes_one() -> None:
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=["swipe_and_analyze"])
    assert asyncio.run(navigator(SCREEN)) is None
    assert client.calls == 0


def test_only_interactive_controls_become_options() -> None:
    options = candidates(SCREEN["observation"])
    assert set(options) == {"el:aaa", "el:bbb"}, "static text is not a tap target"
    assert options["el:aaa"] == "Notifications"


def test_a_switch_reads_its_state_in_the_option_label() -> None:
    options = candidates({"elements": [
        {"id": "el:s1", "text": "Promotional messages", "checked": False, "clickable": True},
        {"id": "el:s2", "text": "Security alerts", "checked": True, "clickable": True},
    ]})
    assert options["el:s1"].endswith("[switch is OFF]")
    assert options["el:s2"].endswith("[switch is ON]")


def test_a_nonsense_gate_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        TypeSafeNavigator("g", client=FakeClient(), min_confidence=0.0)


def test_the_controllers_own_tool_shape_is_understood() -> None:
    # The controller offers OpenAI-shaped entries. Reading the top level finds no name, which
    # silently declined every step of a live shadow run before this was pinned.
    from experiments.aua_controller.typesafe_navigator import tool_names

    offered = [{"type": "function", "function": {"name": TAP_TOOL, "parameters": {}}},
               {"type": "function", "function": {"name": "session_finish", "parameters": {}}}]
    assert tool_names(offered) == {TAP_TOOL, "session_finish"}

    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=offered)
    assert asyncio.run(navigator(SCREEN)) is not None
    assert "tap_not_offered" not in navigator.report()["declined"]


def test_plain_names_still_work_and_an_unknown_shape_offers_nothing() -> None:
    from experiments.aua_controller.typesafe_navigator import tool_names

    assert tool_names([TAP_TOOL]) == {TAP_TOOL}
    assert tool_names([{"type": "function"}, 7, None]) == set()


def test_the_same_tap_is_not_proposed_twice_for_an_unchanged_screen() -> None:
    # Observed live: a System One model reads each screen from scratch, so on a screen its own
    # tap failed to change it confidently proposes that tap again, and the run loops until the
    # step budget ends it. The chat model holds the transcript and can see the repeat.
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    assert asyncio.run(navigator(SCREEN)) is not None
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}


def test_the_same_tap_is_allowed_again_once_the_screen_has_moved_on() -> None:
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    assert asyncio.run(navigator(SCREEN)) is not None
    moved = {"ok": True, "observation": {**SCREEN["observation"], "meta": {"fingerprint": "fp-2"}}}
    assert asyncio.run(navigator(moved)) is not None, "a new screen is a new decision"


class RecordingClient(FakeClient):
    """Keeps the state it was sent, so the journey can be read back."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states: list[dict] = []

    async def system_one(self, *, state, questions, model, timeout=None):
        self.states.append(state)
        return await super().system_one(state=state, questions=questions, model=model, timeout=timeout)


def test_the_whole_journey_is_sent_not_a_list_of_tool_names() -> None:
    # A System One model keeps nothing between calls, but the run fits in one request. Tool
    # names alone measured 33% target accuracy against 41% for the journey.
    client = RecordingClient()
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    moved = {"ok": True, "observation": {**SCREEN["observation"], "meta": {"fingerprint": "fp-2"}}}
    asyncio.run(navigator(moved))

    first, second = client.states
    assert first["journey_so_far"] == [], "nothing has happened yet"
    turn = second["journey_so_far"][0]
    assert turn["you_chose"] == "press 'Notifications'", "the model's words, not AUA's"
    assert turn["what_happened"].startswith("now showing:")
    assert "[Notifications]" in turn["what_happened"], "and the journey shows what was there"
    assert "screen_you_saw" not in turn, "a list of every label per turn is context rot"


def test_a_screen_that_did_not_move_is_said_so_in_the_journey() -> None:
    # This is the fact that stops the loop: the model can see its own tap changed nothing.
    client = RecordingClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    asyncio.run(navigator(SCREEN))
    said = client.states[1]["journey_so_far"][0]["what_happened"]
    assert said.startswith("the screen did not change")
    assert "Notifications" in said, "and it says which screen refused to move"


class WideClient(FakeClient):
    """Remembers the questions the widened space asked."""

    def __init__(self, *args, direction="down", **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction

    async def system_one(self, *, state, questions, model, timeout=None):
        response = await super().system_one(state=state, questions=questions, model=model,
                                            timeout=timeout)
        self.questions = questions
        return response


WIDE_TOOLS = [TAP_TOOL, "scroll_and_analyze", "back_gesture_and_analyze", "session_finish",
              "wait_and_analyze"]


def wide(client, **kwargs):
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=WIDE_TOOLS,
                                  action_space="full", **kwargs)
    return asyncio.run(navigator(SCREEN)), navigator


def test_scrolling_is_two_actions_and_carries_its_own_direction() -> None:
    # The direction is the action, not a second question about it: "which way should this be
    # scrolled" is a hop of indirection, and gating on min(action, direction) mixed two separate
    # questions. Replayed over 11 saved screens the merged form picked the same action 11/11.
    action, _ = wide(WideClient(kind="scroll_up"))
    assert action["tool"] == "scroll_and_analyze"
    assert action["arguments"] == {"direction": "up"}
    action, _ = wide(WideClient(kind="scroll_down"))
    assert action["arguments"] == {"direction": "down"}


def test_the_widened_space_goes_back_with_no_operand() -> None:
    action, _ = wide(WideClient(kind="back"))
    assert action == {"tool": "back_gesture_and_analyze", "arguments": {},
                      "reason": action["reason"]}


def test_the_widened_space_finishes_with_an_outcome_and_never_a_note() -> None:
    # The note is free text and would reach the judge as evidence, so a non-generative model
    # must not supply one. The enum it can answer is the whole claim.
    action, _ = wide(WideClient(kind="already_satisfied"))
    assert action["tool"] == "session_finish"
    assert action["arguments"] == {"outcome": "already_satisfied"}, "no fabricated note"


def test_typing_is_refused_even_in_the_widened_space() -> None:
    # Jev returns a choice, never a string. The public harnesses call a small generative model
    # here; this one hands the step back to the chat model, which is the same move.
    action, navigator = wide(WideClient(kind="type"))
    assert action is None
    assert navigator.report()["declined"] == {"kind:type": 1}


def test_a_widened_action_whose_tool_was_not_offered_is_refused() -> None:
    navigator = TypeSafeNavigator("g", client=WideClient(kind="scroll_down"), tools=[TAP_TOOL],
                                  action_space="full")
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"scroll_not_offered": 1}


def test_an_action_without_an_operand_is_gated_on_itself_alone() -> None:
    # A scroll carries its own direction and a back takes nothing, so there is no second answer
    # to gate on -- a certain tap target says nothing about either.
    action, navigator = wide(WideClient(kind="scroll_down", kind_conf=0.40, target_conf=1.0))
    assert action is None
    assert navigator.report()["declined"] == {"below_confidence": 1}
    assert navigator.proposals[0]["gate"] == 0.40


def test_the_same_scroll_is_not_repeated_on_a_screen_it_did_not_move() -> None:
    client = WideClient(kind="scroll_down")
    navigator = TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS, action_space="full")
    assert asyncio.run(navigator(SCREEN)) is not None
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}


def test_the_widened_space_is_still_one_question() -> None:
    # Finishing is a move like any other, so it sits in the same list as the presses and the
    # scrolls. The separate `outcome` question is gone: on a mid-run step none of the finished
    # outcomes was true, so a second question had to offer `in_progress` -- and that answer then
    # vetoed a confident `done` on the one row where nothing needed to change (measured: 17
    # vetoes on a 27-row run, all 10 premature ones already under the gate, the 2 right ones
    # above it and lost).
    client = WideClient()
    wide(client)
    assert client.calls == 1
    assert {"move"} == set(client.questions)
    offered = set(client.questions["move"].criteria)
    assert {"achieved", "already_satisfied"} <= offered
    assert "in_progress" not in offered and "done" not in offered


def test_the_narrow_default_asks_no_operand_questions() -> None:
    client = WideClient()
    TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS)
    asyncio.run(TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS)(SCREEN))
    assert set(client.questions) == {"move"}


def test_an_unknown_action_space_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        TypeSafeNavigator("g", client=FakeClient(), action_space="everything")


def test_the_controls_are_offered_as_a_numbered_menu_not_as_their_ids() -> None:
    # AUA ids are 32-character hex digests, and jev-1.13 is documented to do worse on opaque and
    # numeric representations than on semantic ones. The public browser harnesses hand it an
    # indexed table for the same reason, so the model is asked about "1" described as what a
    # person reads, and code maps the answer back to the id.
    criteria, by_index = numbered(candidates(SCREEN["observation"]))
    assert criteria == {"1": "Notifications", "2": "Privacy"}
    assert by_index == {"1": "el:aaa", "2": "el:bbb"}
    assert not any(key.startswith("el:") for key in criteria), "no digest reaches the model"


def test_the_numbered_answer_is_mapped_back_to_the_real_element() -> None:
    action, navigator = propose(FakeClient(target="2"))
    assert action["arguments"] == {"id": "el:bbb"}, "answer 2 is the second control"
    assert navigator.proposals[0]["target_id"] == "el:bbb", "the report names the element"


def test_an_index_that_is_not_on_the_menu_is_refused() -> None:
    action, navigator = propose(FakeClient(target="9"))
    assert action is None
    assert navigator.report()["declined"] == {"unknown_target": 1}


def test_being_blocked_is_offered_so_it_can_be_declined() -> None:
    # `blocked` exists so a stuck run has somewhere to put the truth, but acting on it means
    # ending the run, and ending a run early is this model's worst measured skill.
    action, navigator = propose(FakeClient(kind="blocked", kind_conf=1.0, target_conf=1.0))
    assert action is None
    assert navigator.report()["declined"] == {"kind:blocked": 1}


def test_the_boundary_options_are_offered_in_both_action_spaces() -> None:
    from experiments.aua_controller.typesafe_navigator import ACTION_KINDS, build_questions

    for space in ("taps", "full"):
        criteria = build_questions({"el:a": "A", "el:b": "B"}, action_space=space)["move"].criteria
        assert {"wait", "blocked"} <= set(criteria), space
    assert {"wait", "blocked"} <= set(ACTION_KINDS)


def test_every_call_is_written_to_the_transcript_with_its_cost(tmp_path) -> None:
    # The report keeps the decision; the transcript keeps the evidence -- what was sent, what came
    # back, and what the call cost. Without it a run can be summarised but never audited.
    path = tmp_path / "system-one-turns.jsonl"
    navigator = TypeSafeNavigator("Open notification settings", client=FakeClient(),
                                  tools=[TAP_TOOL], transcript_path=path)
    asyncio.run(navigator(SCREEN))

    entry = json.loads(path.read_text().strip())
    assert entry["call"] == 1
    assert entry["menu"] == {"1": "Notifications", "2": "Privacy"}
    # The request is kept in the shape it goes out in, not a summary of it.
    assert set(entry["request"]) == {"model", "state", "questions"}
    assert entry["request"]["state"]["this_is_the_new_screen"]["elements"]
    assert entry["verdict"]["accepted"] is True
    assert set(entry["request"]["questions"]) == {"move"}
    assert entry["response"]["answers"]["move"]["choice"] == "1"
    assert entry["response"]["answers"]["move"]["confidence"] == 0.95
    assert entry["input_tokens"] == 430
    assert entry["usd"] == pytest.approx(430 * 42 / 1e9)
    assert navigator.report()["usd"] == pytest.approx(430 * 42 / 1e9)


def test_a_run_without_a_transcript_path_writes_nothing_and_still_works(tmp_path, monkeypatch) -> None:
    # The first version of this asserted an empty tmp_path the navigator was never given, so it
    # could not fail. Watch the write instead.
    written: list[object] = []
    navigator = TypeSafeNavigator("g", client=FakeClient(), tools=[TAP_TOOL])
    monkeypatch.setattr(navigator, "_record", lambda entry: written.append(entry))
    assert asyncio.run(navigator(SCREEN)) is not None
    assert navigator.report()["transcript"] is None
    assert written, "the turn is still assembled; only the file is absent"


def test_a_mid_run_screen_is_never_forced_to_name_a_finish() -> None:
    # With finishing folded into the move list, a step in the middle of a run simply picks a
    # press or a scroll; nothing asks it to describe a run that has not stopped.
    from experiments.aua_controller.typesafe_navigator import build_questions

    questions = build_questions({"h1": "Settings"}, action_space="full")
    assert set(questions) == {"move"}
    assert "in_progress" not in questions["move"].criteria


@pytest.mark.parametrize("outcome", ["achieved", "already_satisfied"])
def test_finishing_is_a_move_that_carries_its_own_outcome(outcome: str) -> None:
    # The row that found this: an observe-only goal, screen already right, Jev sure it was done
    # at 0.84 -- and declined, because a second question said "in progress". One answer now.
    action, navigator = wide(WideClient(kind=outcome, kind_conf=0.84), min_confidence=0.80)
    assert action == {"tool": "session_finish", "arguments": {"outcome": outcome},
                      "reason": f"System One {outcome} at confidence 0.84: finish as {outcome}"}
    assert navigator.report()["declined"] == {}


def test_ending_a_run_as_hopeless_is_still_the_chat_models_call() -> None:
    # `blocked` and `not_achievable` are offered so the truth has somewhere to go, and refused
    # so a System One hunch never ends a run on a verdict the judge cannot check.
    for kind in ("blocked", "not_achievable"):
        action, navigator = wide(WideClient(kind=kind))
        assert action is None
        assert navigator.report()["declined"] == {f"kind:{kind}": 1}


def test_a_screen_with_no_readable_labels_still_says_it_moved() -> None:
    # A canvas, a game, a WebView that announces nothing: there is no sketch to draw, and the
    # one-bit answer is all there is. It must not come out blank.
    from experiments.aua_controller.typesafe_navigator import what_happened

    assert what_happened({"change": {"activity_changed": True}}, moved=True) == "the screen changed"


def test_a_screen_that_barely_moved_shows_it_is_the_same_screen() -> None:
    # Observed live: tapping sign-in left the activity alone and swapped 2 of 32 controls while
    # the login was in flight. Told "screen changed", the navigator pressed sign-in again. Told
    # "2 controls appeared, 2 went away", it had no way to know which screen that was. The words
    # say it: this is still the sign-in page.
    from experiments.aua_controller.typesafe_navigator import what_happened

    said = what_happened({"change": {"activity_changed": False},
                          "observation": {"elements": [{"text": "Sign in", "clickable": True},
                                                       {"text": "Forgot password?"}]}},
                         moved=True)
    assert "[Sign in]" in said and "Forgot password?" in said
    assert "still working" not in said, "nothing is inferred; the screen speaks for itself"


def test_a_relabel_shows_the_new_label() -> None:
    # A control that only changed its text used to be counted and never quoted, so "2 were
    # relabelled" left the one fact that mattered -- what it now says -- out of the journey.
    from experiments.aua_controller.typesafe_navigator import what_happened

    said = what_happened({"change": {"activity_changed": False},
                          "observation": {"elements": [{"text": "Signing in…"}]}}, moved=True)
    assert "Signing in…" in said


def test_nothing_about_the_meaning_of_a_change_is_ever_asserted() -> None:
    # One control changing IS the completed action on a settings toggle, and IS mid-flight work
    # on a login. No wording can be right for both, so the journey states and never interprets.
    from experiments.aua_controller.typesafe_navigator import what_happened

    said = what_happened({"change": {"activity_changed": False},
                          "observation": {"elements": [{"text": "Dark mode", "checked": True}]}},
                         moved=True)
    for guess in ("still working", "loading", "in progress", "finished", "succeeded"):
        assert guess not in said, guess


def test_a_pressable_control_is_marked_and_a_label_is_not() -> None:
    # "What could I have pressed on that screen" is the question a journey turn gets read for.
    from experiments.aua_controller.typesafe_navigator import what_happened

    said = what_happened({"observation": {"elements": [{"text": "Settings", "clickable": True},
                                                       {"text": "Version 1.2.3"}]}}, moved=True)
    assert "[Settings]" in said and "Version 1.2.3" in said and "[Version" not in said


def test_an_unmoved_screen_still_says_so_whatever_the_frame_carries() -> None:
    from experiments.aua_controller.typesafe_navigator import what_happened

    assert what_happened({"change": {"activity_changed": True}}, moved=False) == \
        "the screen did not change at all"


def test_a_frame_without_change_telemetry_falls_back_to_the_old_wording() -> None:
    from experiments.aua_controller.typesafe_navigator import what_happened

    assert what_happened({}, moved=True) == "the screen changed"


def test_finishing_is_gated_on_its_one_confidence() -> None:
    # One question, one answer, one number. Ending a run early is this model's worst measured
    # skill, so the gate applies to finishing exactly as it does to a press.
    action, navigator = wide(WideClient(kind="achieved", kind_conf=0.51))
    assert action is None, "0.51 is under the default gate"
    record = navigator.proposals[0]
    assert record["operand"] == "achieved"
    assert record["gate"] == 0.51
    assert "outcome_confidence" not in record


def test_a_declined_call_says_why_in_its_transcript_entry(tmp_path) -> None:
    # A step handed to the chat model without a recorded reason is unreadable afterwards: the
    # log shows DeepSeek acting and nothing about the refusal that put it there.
    path = tmp_path / "turns.jsonl"
    navigator = TypeSafeNavigator("g", client=FakeClient(target_conf=0.42), tools=[TAP_TOOL],
                                  transcript_path=path)
    assert asyncio.run(navigator(SCREEN)) is None

    entry = json.loads(path.read_text().strip())
    verdict = entry["verdict"]
    assert verdict["accepted"] is False
    assert verdict["declined_because"] == "below_confidence"
    assert verdict["gate"] == 0.42 and verdict["gate_needed"] == 0.85


def test_an_accepted_call_records_its_verdict_too(tmp_path) -> None:
    path = tmp_path / "turns.jsonl"
    navigator = TypeSafeNavigator("g", client=FakeClient(), tools=[TAP_TOOL], transcript_path=path)
    assert asyncio.run(navigator(SCREEN)) is not None

    verdict = json.loads(path.read_text().strip())["verdict"]
    assert verdict["accepted"] is True and verdict.get("declined_because") is None


def test_exactly_one_transcript_line_is_written_per_call(tmp_path) -> None:
    path = tmp_path / "turns.jsonl"
    navigator = TypeSafeNavigator("g", client=FakeClient(), tools=[TAP_TOOL], transcript_path=path)
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    moved = {"ok": True, "observation": {**SCREEN["observation"], "meta": {"fingerprint": "fp-9"}}}
    asyncio.run(navigator(moved))
    assert len(path.read_text().strip().splitlines()) == 2


def test_a_control_the_app_never_named_is_placed_not_hashed() -> None:
    # 11% of real options carried no text, desc or resource id, so the label fell back to the
    # element's own digest -- the opaque value the numbering exists to keep out of the request.
    options = candidates({
        "screen": {"width": 1000, "height": 2000},
        "elements": [{"id": "el:deadbeefdeadbeefdeadbeef", "clickable": True,
                      "bounds": [800, 100, 960, 220]},
                     {"id": "el:aaa", "text": "Settings", "clickable": True}],
    })
    assert options["el:deadbeefdeadbeefdeadbeef"] == "unlabelled control, top right of the screen"
    assert not any("deadbeef" in label for label in options.values())


def test_an_unnamed_control_with_no_bounds_still_reads_as_a_control() -> None:
    options = candidates({"elements": [{"id": "el:x", "clickable": True},
                                       {"id": "el:y", "clickable": True}]})
    assert options["el:x"] == "unlabelled control"


def test_a_named_control_is_untouched_by_the_fallback() -> None:
    options = candidates({"screen": {"width": 1000, "height": 2000},
                          "elements": [{"id": "el:a", "text": "Continue", "clickable": True,
                                        "bounds": [0, 0, 10, 10]},
                                       {"id": "el:b", "text": "Back", "clickable": True}]})
    assert options["el:a"] == "Continue"


def test_waiting_is_an_action_the_navigator_can_actually_take() -> None:
    # Asking "is this screen still loading?" and then paying a chat model to answer the same
    # question was an option that cost a round trip to say nothing. The tool already existed.
    action, _ = wide(WideClient(kind="wait"))
    assert action["tool"] == "wait_and_analyze"
    assert action["arguments"] == {"idle": True}


def test_a_second_wait_on_the_same_activity_escalates_even_as_the_screen_churns() -> None:
    # A loading screen re-fingerprints on every frame it redraws, so keying a wait on the
    # fingerprint would never repeat and never escalate -- it would wait until the step budget.
    # The activity is the thing that holds still while a screen loads.
    client = WideClient(kind="wait")
    navigator = TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS, action_space="full")
    def loading(fingerprint):
        return {"ok": True, "change": {"activity_after": ".Auth"},
                "observation": {**SCREEN["observation"], "meta": {"fingerprint": fingerprint}}}

    assert asyncio.run(navigator(loading("fp-1"))) is not None
    assert asyncio.run(navigator(loading("fp-2"))) is None, "same activity, still loading"
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}


def test_waiting_is_refused_when_the_run_was_never_offered_the_tool() -> None:
    navigator = TypeSafeNavigator("g", client=WideClient(kind="wait"), tools=[TAP_TOOL],
                                  action_space="full")
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"wait_not_offered": 1}


# --- what goes on the wire -------------------------------------------------------------------
# Every test above this line drives a fake client, so none of them could see what the request
# body actually contains. Four of the last five fixes were found by a human reading that body.

def sent(navigator_kwargs=None, screens=(SCREEN,)):
    """Run the navigator over some screens and hand back every state it sent."""
    client = RecordingClient()
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=[TAP_TOOL],
                                  **(navigator_kwargs or {}))
    for screen in screens:
        asyncio.run(navigator(screen))
        navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    return client.states, navigator


def test_no_element_digest_ever_reaches_the_model() -> None:
    # The numbered menu exists to keep 32-character handles out of the request. They went on
    # arriving anyway, on every element of the state body.
    states, _ = sent()
    body = json.dumps(states)
    assert not re.search(r"[0-9a-f]{32}", body), "a digest is in the request"
    assert "el:aaa" not in body


def test_no_fingerprint_or_pixel_bounds_reach_the_model() -> None:
    states, _ = sent()
    screen = states[0]["this_is_the_new_screen"]
    assert "meta" not in screen and "fingerprint" not in json.dumps(screen)
    assert "bounds" not in json.dumps(screen), "pixel numbers are not something it can use"


def test_every_journey_turn_says_what_was_chosen_and_what_followed() -> None:
    moved = {"ok": True, "observation": {**SCREEN["observation"], "meta": {"fingerprint": "fp-2"}}}
    states, _ = sent(screens=(SCREEN, moved, SCREEN))
    for state in states:
        for turn in state["journey_so_far"]:
            assert turn["you_chose"] and turn["you_chose"] != "(nothing yet)"
            assert turn["what_happened"]
            assert "_label" not in turn


def test_a_step_the_chat_model_took_still_appears_in_the_journey() -> None:
    # The journey only ever recorded steps this navigator won. In the default space that is a
    # small minority, so the model was told it was on step 6 of a run that was on step 12.
    client = RecordingClient(target_conf=0.10)          # every proposal falls under the gate
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    assert asyncio.run(navigator(SCREEN)) is None, "declined, so the chat model acts"
    navigator.observed("scroll_and_analyze", {})
    moved = {"ok": True, "observation": {**SCREEN["observation"], "meta": {"fingerprint": "fp-2"}}}
    asyncio.run(navigator(moved))

    journey = client.states[1]["journey_so_far"]
    assert len(journey) == 1
    assert journey[0]["you_chose"] == "scroll", "AUA's tool name said back as the model's word"


def test_a_turn_survives_a_screen_this_navigator_could_not_read() -> None:
    # `too_few_controls` returned before the turn was closed, so the turn was closed later
    # against a screen it never saw -- a tap on Notifications came back as a scroll on '?'.
    client = RecordingClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    bare = {"ok": True, "observation": {"meta": {"fingerprint": "fp-2"},
                                        "elements": [{"id": "el:z", "text": "OK", "clickable": True}]}}
    assert asyncio.run(navigator(bare)) is None
    navigator.observed("back_gesture_and_analyze", {})
    asyncio.run(navigator({"ok": True, "observation": {**SCREEN["observation"],
                                                       "meta": {"fingerprint": "fp-3"}}}))

    journey = client.states[-1]["journey_so_far"]
    assert [t["you_chose"] for t in journey] == ["press 'Notifications'", "back"]


def test_shadow_mode_declines_exactly_once() -> None:
    # `_decline("shadow")` was called on two separate lines, so one shadow proposal reported two.
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL], shadow=True)
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"shadow": 1}


def test_the_journey_is_trimmed_from_the_oldest_end() -> None:
    from experiments.aua_controller.typesafe_navigator import MAX_JOURNEY_CHARS

    client = RecordingClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    navigator._journey = [{"n": i, "you_chose": "tap on 'x'", "what_happened": "y" * 400}
                          for i in range(1, 401)]
    asyncio.run(navigator(SCREEN))
    journey = client.states[0]["journey_so_far"]
    assert len(json.dumps(journey)) <= MAX_JOURNEY_CHARS
    assert journey[-1]["n"] == 400, "the newest turns are the ones a loop is made of"


@pytest.mark.parametrize("bounds,expected", [
    ([0, 0, 100, 100], "top left"),
    ([450, 950, 550, 1050], "middle centre"),
    ([900, 1900, 1000, 2000], "bottom right"),
])
def test_an_unnamed_control_is_placed_on_the_right_third(bounds, expected) -> None:
    from experiments.aua_controller.typesafe_navigator import where

    said = where({"bounds": bounds}, {"width": 1000, "height": 2000})
    assert said == f"unlabelled control, {expected} of the screen"


def test_the_finish_outcomes_still_match_the_harness_they_are_sent_to() -> None:
    # This enum is a copy of the harness's. A harness change would not propagate, and the
    # mismatch would only show as a rejected call.
    from experiments.aua_controller.run_realapp import FINISH_OUTCOMES as HARNESS
    from experiments.aua_controller.typesafe_navigator import FINISH_OUTCOMES

    assert set(FINISH_OUTCOMES) == set(HARNESS)


# --------------------------------------------------- what the app is still waiting on


def test_the_calls_still_in_the_air_reach_the_model() -> None:
    """Without this the model cannot tell a loading screen from a finished one.

    On the real step it got wrong, being told the login POST had not answered moved it from
    `tap` at 0.92 to `wait` at 0.72 -- the same screen, the same menu, one extra line of state.
    """
    screen = screen_for_model({"observation": {
        "screen": {"package": "com.example.app"},
        "meta": {"network_calls": ["POST /v1/auth/login"]},
        "elements": [{"text": "Sign in", "id": "el:abc"}],
    }})
    assert screen["network"] == ["POST /v1/auth/login"]


def test_a_quiet_screen_carries_no_network_key_at_all() -> None:
    """Every token of state that is not about the decision costs accuracy on this model."""
    screen = screen_for_model({"observation": {
        "screen": {"package": "com.example.app"},
        "meta": {"fingerprint": "abc123"},
        "elements": [{"text": "Sign in", "id": "el:abc"}],
    }})
    assert "network" not in screen


def test_compaction_does_not_drop_the_calls_still_in_the_air() -> None:
    """`META_FIELDS` is an allowlist, so a new field is invisible until it is named there.

    This is exactly how the `checked` flag went missing: the engine reported it, compaction
    dropped it, and a contract bullet about a switch could never be verified.
    """
    compact = compact_frame({"observation": {
        "screen": {"package": "com.example.app"},
        "meta": {"fingerprint": "abc", "network_calls": ["POST /v1/auth/login"]},
        "elements": [{"text": "Sign in", "id": "el:abc", "clickable": True}],
    }}, keep_ids=True)
    assert compact["observation"]["meta"]["network_calls"] == ["POST /v1/auth/login"]


# ----------------------------------------- one question, each action carrying its own operand


def test_every_control_is_its_own_action() -> None:
    """Splitting "what kind of move" from "which control" made the model answer both, always.

    It picked `wait` and named a button in the same breath, and the harness threw the button
    away. Nothing was wrong with the answer -- the operand was speculative, and speculation is
    nearly free on a model that prices the state once -- but nobody reading the request should
    have to be told to ignore half of it. The API has no question conditional on another answer,
    so folding the controls into the action list is the only shape that removes the dependency.
    """
    questions = build_questions({"el:a": "Allow", "el:b": "Ask me later"})
    assert list(questions) == ["move"]
    criteria = questions["move"].criteria
    assert criteria["1"] == "Press 'Allow'"
    assert criteria["2"] == "Press 'Ask me later'"
    for kind in ("scroll_down", "scroll_up", "back", "achieved", "already_satisfied", "wait",
                 "blocked", "not_achievable", "type"):
        assert kind in criteria, kind
    assert "tap" not in criteria, "a bare `tap` names no control and is not an action"


def test_the_gate_reads_one_confidence_now() -> None:
    """`min()` of two questions about different things was never one number about one decision."""
    navigator = TypeSafeNavigator("open settings",
                                  client=FakeClient(target="1", target_conf=0.9),
                                  tools=[TAP_TOOL], min_confidence=0.85)
    proposal = asyncio.run(navigator(SCREEN))
    assert proposal is not None and proposal["tool"] == TAP_TOOL
    verdict = navigator.proposals[-1]
    assert verdict["gate"] == 0.9
    assert "target_confidence" not in verdict


def test_an_action_that_takes_no_operand_names_none() -> None:
    navigator = TypeSafeNavigator("go back", client=FakeClient(kind="back", kind_conf=0.95),
                                  tools=[TAP_TOOL, "back_gesture_and_analyze"],
                                  min_confidence=0.85, action_space="full")
    proposal = asyncio.run(navigator(SCREEN))
    assert proposal is not None and proposal["tool"] == "back_gesture_and_analyze"
    assert navigator.proposals[-1]["operand"] == "back"


def test_a_control_that_is_not_on_this_screen_is_refused() -> None:
    """The answer is an index into a menu built for this screen and nothing else."""
    navigator = TypeSafeNavigator("open settings", client=FakeClient(target="99"),
                                  tools=[TAP_TOOL], min_confidence=0.5)
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.declined.get("unknown_target") == 1


# ------------------------------- the journey in the model's own words, with the screen in it


def test_a_turn_shows_the_screen_it_landed_on() -> None:
    """Counts were facts about a screen the model never saw, which is not the same as evidence.

    "7 controls appeared, 2 went away, out of 32" cannot tell a login page from a settings list,
    and the model has to decide whether it is looping. The labels can. The current screen is in
    the state in full; what the journey was missing is what the *earlier* ones looked like.
    """
    landed = what_happened({"observation": {"elements": [
        {"text": "Welcome back"},
        {"text": "Sign in", "clickable": True},
        {"text": "Browse as a guest", "clickable": True},
    ]}}, moved=True)
    assert "Welcome back" in landed
    assert "[Sign in]" in landed, "a pressable control is marked as one"
    assert "Browse as a guest" in landed
    assert "controls appeared" not in landed


def test_a_screen_that_did_not_move_still_says_so_first() -> None:
    """The one-bit answer is what says "your tap did nothing"; the labels do not replace it."""
    landed = what_happened({"observation": {"elements": [{"text": "Welcome back"}]}}, moved=False)
    assert landed.startswith("the screen did not change")
    assert "Welcome back" in landed


def test_a_long_screen_is_cut_short() -> None:
    """A journey turn carrying forty labels buys context rot on a model documented to suffer it."""
    many = {"observation": {"elements": [{"text": f"Row number {n}"} for n in range(40)]}}
    landed = what_happened(many, moved=True)
    assert len(landed) < 400
    assert "Row number 0" in landed


def test_the_journey_says_what_the_model_chose_not_what_the_harness_called_it() -> None:
    """`tap_and_analyze` is AUA's function name. The model answered "press 'Privacy'".

    Showing it a word it never used, for a decision it did make, is a vocabulary it has to
    translate before it can read its own history.
    """
    navigator = TypeSafeNavigator("open settings", client=FakeClient(target="2"),
                                  tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:bbb"})
    assert navigator._pending["you_chose"] == "press 'Privacy'"


def test_a_step_the_chat_model_took_is_named_in_the_same_words() -> None:
    """The journey is one story; half of it in AUA's vocabulary makes it two."""
    navigator = TypeSafeNavigator("open settings", client=FakeClient(), tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    for tool, expected in (("back_gesture_and_analyze", "back"),
                           ("wait_and_analyze", "wait"),
                           ("session_finish", "done"),
                           ("input_and_analyze", "type")):
        navigator.observed(tool, {})
        assert navigator._pending["you_chose"] == expected, tool


def test_a_forgotten_turn_never_reaches_the_journey() -> None:
    # A press AUA refused as stale was never sent, so it is not part of the story the model
    # reads on the next step. The harness says `forget()`; the open turn is dropped.
    client = RecordingClient(kind="tap", target="1")
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=WIDE_TOOLS,
                                  action_space="full")
    action = asyncio.run(navigator(SCREEN))
    assert action is not None
    navigator.observed(action["tool"], action["arguments"])
    navigator.forget()
    asyncio.run(navigator(SCREEN))
    assert client.states[-1]["journey_so_far"] == []


def test_a_status_bar_item_with_a_checked_field_is_not_a_switch() -> None:
    """A raw hierarchy dump puts ``checked: false`` on every node, the status-bar clock included.

    Seen on the first frame of every row: 22 status-bar nodes became "Press '11:28 [switch is
    OFF]'" and friends, thirteen junk options that diluted the one real choice.
    """
    options = candidates({"elements": [
        {"id": "el:clock", "text": "11:28", "resource_id": "com.android.systemui:id/clock",
         "clickable": False, "checkable": False, "checked": False},
        {"id": "el:login", "text": "Log in", "clickable": True, "checkable": False, "checked": False},
        {"id": "el:dark", "text": "Dark mode", "clickable": True, "checkable": True, "checked": True},
    ]})
    assert "el:clock" not in options, "a non-interactive node is not an option because it carries a checked flag"
    assert options["el:login"] == "Log in", "a plain button is not a switch"
    assert options["el:dark"].endswith("[switch is ON]")


class SequenceClient(FakeClient):
    """Answers a fixed sequence of (choice, confidence) pairs, one per call."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    async def system_one(self, *, state, questions, model, timeout=None):
        self.calls += 1
        choice, confidence = self.answers.pop(0)
        return SimpleNamespace(answers={"move": SimpleNamespace(choice=choice, confidence=confidence)},
                               usage=SimpleNamespace(input_tokens=400))


LOOK_TOOLS = [TAP_TOOL, "wait_and_analyze", "session_finish"]


def test_a_near_miss_gets_one_more_look_before_the_chat_model() -> None:
    """Replayed eight times, the same screen scored 0.66-0.78 and never crossed the gate; a fresh
    read after a wait scored 0.85 every time. So a near miss buys one wait, not a retry."""
    client = SequenceClient([("achieved", 0.72), ("achieved", 0.72), ("achieved", 0.72)])
    navigator = TypeSafeNavigator("Look at the landing screen", client=client, tools=LOOK_TOOLS,
                                  action_space="full", min_confidence=0.80)

    first = asyncio.run(navigator(SCREEN))
    assert first["tool"] == "wait_and_analyze" and first["arguments"] == {"idle": True}
    assert "second look" in first["reason"] and "0.72" in first["reason"]
    assert navigator.proposals[-1]["second_look"] is True and navigator.proposals[-1]["accepted"] is False

    second = asyncio.run(navigator(SCREEN))
    assert second is None, "the same near miss on the same screen is handed to the chat model"
    third = asyncio.run(navigator(SCREEN))
    assert third is None, "one more look means one"
    assert navigator.declined == {"second_look": 1, "below_confidence": 2}


def test_a_confident_answer_after_the_second_look_is_taken() -> None:
    client = SequenceClient([("achieved", 0.7), ("achieved", 0.9)])
    navigator = TypeSafeNavigator("Look at the landing screen", client=client, tools=LOOK_TOOLS,
                                  action_space="full", min_confidence=0.80)
    assert asyncio.run(navigator(SCREEN))["tool"] == "wait_and_analyze"
    assert asyncio.run(navigator(SCREEN))["tool"] == "session_finish"


def test_a_clear_miss_is_handed_over_at_once() -> None:
    client = SequenceClient([("achieved", 0.55)])
    navigator = TypeSafeNavigator("Look at the landing screen", client=client, tools=LOOK_TOOLS,
                                  action_space="full", min_confidence=0.80)
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.declined == {"below_confidence": 1}


def test_a_near_miss_without_a_wait_tool_is_handed_over() -> None:
    client = SequenceClient([("1", 0.7)])
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=[TAP_TOOL],
                                  min_confidence=0.80)
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.declined == {"below_confidence": 1}


def test_the_second_look_floor_must_sit_under_the_gate() -> None:
    with pytest.raises(ValueError):
        TypeSafeNavigator("g", client=FakeClient(), min_confidence=0.8, second_look_floor=0.9)


def test_a_text_field_is_named_as_one_in_the_menu_and_in_the_state() -> None:
    # Live shape: the goal said "tap the composer and type"; the only place the word "composer"
    # appeared on screen was the resource id of the attachments button beside the field, and the
    # field itself was labelled only by its hint. Every option read "Press '...'", so the menu
    # gave the model no way to tell the field from the button, and it pressed the button twice
    # at 0.96 and 0.93 -- each time opening a sheet the chat model then had to close.
    observation = {"elements": [
        {"id": "el:field", "text": "Ask me anything", "editable": True, "clickable": True},
        {"id": "el:add", "resource_id": "buttonOpenComposerAttachments", "clickable": True},
        {"id": "el:hint", "text": "Ask me anything"},
    ]}
    options = candidates(observation)
    assert options["el:field"] == "Ask me anything (text field)"
    assert options["el:add"] == "buttonOpenComposerAttachments", "a button needs no role; Press says it"
    state = screen_for_model({"observation": observation})
    assert {"text": "Ask me anything", "editable": True} in state["elements"]
    assert {"text": "Ask me anything"} in state["elements"], "plain text stays plain"
