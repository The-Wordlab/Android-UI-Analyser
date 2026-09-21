"""The navigator's value is in what it refuses, so the refusals are what these pin.

Every path that is not a confident tap must return None, because None is what hands the step
back to the chat model. A navigator that answers a step it should not have is worse than one
that answers nothing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.typesafe_navigator import (  # noqa: E402
    TAP_TOOL,
    TypeSafeNavigator,
    candidates,
    numbered,
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
        return SimpleNamespace(
            answers={
                "action": SimpleNamespace(choice=self.kind, confidence=self.kind_conf),
                "target": SimpleNamespace(choice=self.target, confidence=self.target_conf),
                "settled": SimpleNamespace(noul=self.settled),
            },
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


@pytest.mark.parametrize("kind", ["done", "back", "scroll_down", "type", "wait"])
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


def test_the_weaker_of_the_two_choices_is_the_one_that_gates() -> None:
    # A certain target reached by an uncertain action is still an uncertain step.
    action, _ = propose(FakeClient(kind_conf=0.55, target_conf=1.0))
    assert action is None


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
    navigator.observed(TAP_TOOL)
    moved = {"ok": True, "observation": {**SCREEN["observation"], "meta": {"fingerprint": "fp-2"}}}
    asyncio.run(navigator(moved))

    first, second = client.states
    assert first["journey_so_far"] == [], "nothing has happened yet"
    turn = second["journey_so_far"][0]
    assert turn["you_chose"] == f"{TAP_TOOL} on 'Notifications'"
    assert turn["what_happened"] == "screen changed"
    assert "Notifications" in turn["screen_you_saw"]


def test_a_screen_that_did_not_move_is_said_so_in_the_journey() -> None:
    # This is the fact that stops the loop: the model can see its own tap changed nothing.
    client = RecordingClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL)
    asyncio.run(navigator(SCREEN))
    assert client.states[1]["journey_so_far"][0]["what_happened"] == "SCREEN DID NOT CHANGE"


class WideClient(FakeClient):
    """Answers the operand questions the widened space adds."""

    def __init__(self, *args, direction="down", outcome="achieved", operand_conf=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction, self.outcome, self.operand_conf = direction, outcome, operand_conf

    async def system_one(self, *, state, questions, model, timeout=None):
        response = await super().system_one(state=state, questions=questions, model=model,
                                            timeout=timeout)
        response.answers["direction"] = SimpleNamespace(choice=self.direction,
                                                        confidence=self.operand_conf)
        response.answers["outcome"] = SimpleNamespace(choice=self.outcome,
                                                      confidence=self.operand_conf)
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
    action, _ = wide(WideClient(kind="done", outcome="blocked"))
    assert action["tool"] == "session_finish"
    assert action["arguments"] == {"outcome": "blocked"}, "no fabricated note"


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


def test_the_operand_questions_are_asked_in_the_same_single_request() -> None:
    # One request prices the state once and answers in parallel, so the operands for actions
    # that lose are free. Asking them in a second call would give the saving away.
    client = WideClient()
    wide(client)
    assert client.calls == 1
    assert {"action", "target", "settled", "outcome"} == set(client.questions)


def test_the_narrow_default_asks_no_operand_questions() -> None:
    client = WideClient()
    TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS)
    asyncio.run(TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS)(SCREEN))
    assert set(client.questions) == {"action", "target", "settled"}


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
        criteria = build_questions({"el:a": "A", "el:b": "B"}, action_space=space)["action"].criteria
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
    assert set(entry["request"]["questions"]) == {"action", "target", "settled"}
    assert entry["response"]["answers"]["action"]["choice"] == "tap"
    assert entry["response"]["answers"]["target"]["confidence"] == 0.95
    assert entry["input_tokens"] == 430
    assert entry["usd"] == pytest.approx(430 * 42 / 1e9)
    assert navigator.report()["usd"] == pytest.approx(430 * 42 / 1e9)


def test_a_run_without_a_transcript_path_writes_nothing_and_still_works(tmp_path) -> None:
    navigator = TypeSafeNavigator("g", client=FakeClient(), tools=[TAP_TOOL])
    assert asyncio.run(navigator(SCREEN)) is not None
    assert navigator.report()["transcript"] is None
    assert not list(tmp_path.iterdir())


def test_finishing_while_the_goal_is_unfinished_is_refused() -> None:
    # The four finish outcomes all describe a run that has stopped, so on a step in the middle of
    # one none of them is true and the model answered `blocked` on 6 of 10 real screens with
    # nothing blocking anything. `in_progress` gives the truth somewhere to go -- and asking to
    # stop while reporting the goal unfinished is a contradiction, not a decision to act on.
    action, navigator = wide(WideClient(kind="done", outcome="in_progress"))
    assert action is None
    assert navigator.report()["declined"] == {"done_but_unfinished": 1}


def test_in_progress_is_offered_but_is_never_a_finish_argument() -> None:
    from experiments.aua_controller.typesafe_navigator import FINISH_OUTCOMES, UNFINISHED

    assert UNFINISHED in FINISH_OUTCOMES, "the model must be able to say it is mid-run"
    action, _ = wide(WideClient(kind="done", outcome="achieved"))
    assert action["arguments"]["outcome"] != UNFINISHED
    assert action["arguments"] == {"outcome": "achieved"}


def test_a_new_activity_is_reported_as_a_different_screen() -> None:
    from experiments.aua_controller.typesafe_navigator import what_happened

    assert what_happened({"change": {"activity_changed": True}}, moved=True) == \
        "a different screen opened"


def test_a_handful_of_redrawn_controls_is_not_reported_as_progress() -> None:
    # Observed live: tapping sign-in left the activity alone and swapped 2 of 32 controls while
    # the login was in flight. Told only "screen changed", the navigator pressed sign-in again.
    from experiments.aua_controller.typesafe_navigator import what_happened

    said = what_happened({"change": {"activity_changed": False},
                          "action_diff_summary": {"added": 2, "removed": 2, "curr_count": 32}},
                         moved=True)
    assert "SAME screen redrew" in said and "2 of 32" in said
    assert "still working" in said, "the model needs the reason, not just the count"


def test_a_wholesale_replacement_is_still_ordinary_progress() -> None:
    from experiments.aua_controller.typesafe_navigator import what_happened

    said = what_happened({"change": {"activity_changed": False},
                          "action_diff_summary": {"added": 33, "removed": 40, "curr_count": 52}},
                         moved=True)
    assert said == "screen changed"


def test_an_unmoved_screen_still_says_so_whatever_the_frame_carries() -> None:
    from experiments.aua_controller.typesafe_navigator import what_happened

    assert what_happened({"change": {"activity_changed": True}}, moved=False) == \
        "SCREEN DID NOT CHANGE"


def test_a_frame_without_change_telemetry_falls_back_to_the_old_wording() -> None:
    from experiments.aua_controller.typesafe_navigator import what_happened

    assert what_happened({}, moved=True) == "screen changed"


def test_the_record_names_the_operand_the_gate_actually_read() -> None:
    # A finish is gated on its outcome, not on the tap target it never used. Printing the target
    # beside that gate made the log contradict itself: target 0.86, gate 0.51, same row.
    action, navigator = wide(WideClient(kind="done", outcome="achieved", operand_conf=0.51))
    assert action is None, "0.51 is under the default gate"
    record = navigator.proposals[0]
    assert record["operand"] == "achieved"
    assert record["operand_confidence"] == 0.51
    assert record["gate"] == 0.51, "the gate is the action and its own operand, nothing else"


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
    navigator.observed(TAP_TOOL)
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


def test_a_second_wait_on_a_screen_that_never_moved_escalates() -> None:
    # No counter needed: waiting twice on an identical screen is the same (fingerprint, action)
    # pair the repeat guard already refuses, so one wait per screen and then the chat model.
    client = WideClient(kind="wait")
    navigator = TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS, action_space="full")
    assert asyncio.run(navigator(SCREEN)) is not None
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}


def test_waiting_is_refused_when_the_run_was_never_offered_the_tool() -> None:
    navigator = TypeSafeNavigator("g", client=WideClient(kind="wait"), tools=[TAP_TOOL],
                                  action_space="full")
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"wait_not_offered": 1}
