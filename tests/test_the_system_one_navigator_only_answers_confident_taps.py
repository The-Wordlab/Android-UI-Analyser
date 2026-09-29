"""The navigator's value is in what it refuses, so the refusals are what these pin.

Every path that is not a confident move must return None, because None is what hands the step
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
    sketch,
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
LIST_SCREEN = {"ok": True, "observation": {**SCREEN["observation"], "elements": [
    *SCREEN["observation"]["elements"], {"id": "el:list", "scrollable": True, "bounds": [0, 0, 7, 23]}]}}
FIELD_SCREEN = {
    "ok": True,
    "observation": {
        "screen": {"package": "com.example.demo", "activity": ".Chat"},
        "meta": {"fingerprint": "fp-chat"},
        "elements": [
            {"id": "el:menu", "desc": "menu button, opens the side drawer", "clickable": True, "bounds": [0, 0, 7, 7]},
            {"id": "el:field", "text": "Ask me anything", "editable": True, "clickable": True, "bounds": [0, 8, 7, 15]},
            {"id": "el:send", "text": "Send", "clickable": True, "bounds": [8, 8, 15, 15]},
        ],
    },
}
TWO_FIELDS_SCREEN = {
    "ok": True,
    "observation": {
        "screen": {"package": "com.example.demo", "activity": ".Login"},
        "meta": {"fingerprint": "fp-login"},
        "elements": [
            {"id": "el:user", "text": "Email", "editable": True, "clickable": True, "bounds": [0, 0, 7, 7]},
            {"id": "el:pass", "text": "Password", "editable": True, "clickable": True, "bounds": [0, 8, 7, 15]},
            {"id": "el:go", "text": "Sign in", "clickable": True, "bounds": [8, 8, 15, 15]},
        ],
    },
}
WIDE_TOOLS = [TAP_TOOL, "scroll_and_analyze", "back_gesture_and_analyze", "session_finish",
              "wait_and_analyze", "input_and_analyze"]
BRIEF = ("Open the side drawer. Then tap `New chat`. Then tap the message field, type exactly "
         "`Hello there` (submit=false), then tap `Send`. Then wait for the reply, then type exactly "
         "`Thanks` and tap `Send`.")


def answer(choice, confidence):
    return SimpleNamespace(choice=choice, confidence=confidence, probabilities={choice: confidence})


class FakeClient:
    """Answers the three questions with fixed choices, and keeps what it was sent."""

    def __init__(self, operation="press", target="1", op_conf=0.99, target_conf=0.95,
                 text="1", text_conf=0.95, error=None):
        self.operation, self.target, self.text = operation, target, text
        self.op_conf, self.target_conf, self.text_conf = op_conf, target_conf, text_conf
        self.error = error
        self.calls = 0
        self.states: list[dict] = []
        self.questions: list[dict] = []

    async def system_one(self, *, state, questions, model, timeout=None):
        self.calls += 1
        self.states.append(state)
        self.questions.append(questions)
        if self.error:
            raise self.error
        answers = {"operation": answer(self.operation, self.op_conf),
                   "target": answer(self.target, self.target_conf)}
        if "text" in questions:
            answers["text"] = answer(self.text, self.text_conf)
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=430))


def propose(client, goal="Open notification settings", screen=SCREEN, tools=(TAP_TOOL,), **kwargs):
    navigator = TypeSafeNavigator(goal, client=client, tools=list(tools), **kwargs)
    return asyncio.run(navigator(screen)), navigator


def moved(fingerprint="fp-2", screen=SCREEN):
    return {"ok": True, "observation": {**screen["observation"], "meta": {"fingerprint": fingerprint}}}


# ------------------------------------------------------------ the brief, whole, in three questions


def test_jev_reads_the_whole_brief_not_one_step_of_it() -> None:
    # A pointer into the brief was on the wrong step most of the time: of 90 presses the chat
    # model made on a listed control, the step Jev was shown named that control 17 times. Asked
    # with the whole brief, Jev matched 83% of those presses instead of 53%.
    client = FakeClient()
    propose(client, goal=BRIEF, screen=FIELD_SCREEN, tools=WIDE_TOOLS)
    state = client.states[0]
    assert state["goal"] == BRIEF
    assert "done_before_this" not in state and "step" not in json.dumps(state["journey_so_far"])
    assert client.calls == 1, "one request per screen; nothing is re-asked to move a pointer"


def test_three_narrow_questions_go_out_in_one_request() -> None:
    # One judgment per question, the operand questions asked beside the operation because the
    # state is priced once -- TypeSafe's own advice and the public browser harness's shape.
    client = FakeClient()
    propose(client, goal=BRIEF, screen=FIELD_SCREEN, tools=WIDE_TOOLS, action_space="full")
    questions = client.questions[0]
    assert set(questions) == {"operation", "target", "text"}
    assert set(questions["target"].criteria) == {"1", "2", "3"}, "controls only, nothing else"
    assert questions["target"].criteria["2"] == "Tap the text field 'Ask me anything' so text can be typed into it"
    assert set(questions["text"].criteria.values()) == {"Hello there", "Thanks"}


def test_no_boundary_move_competes_with_the_controls() -> None:
    # "This step is done" and "back" were the top pick on 66% of the screens the old merged
    # question handed over: they soaked up the probability whenever a step's words missed.
    operations = build_questions({"el:a": "Allow", "el:b": "Later"}, action_space="full")["operation"].criteria
    assert set(operations) == {"press", "scroll_down", "scroll_up", "back", "wait", "finished", "other"}
    for gone in ("achieved", "already_satisfied", "blocked", "not_achievable"):
        assert gone not in operations


def test_only_the_operations_this_screen_allows_are_offered() -> None:
    narrow = build_questions({"el:a": "Allow", "el:b": "Later"}, can_scroll=False)["operation"].criteria
    assert set(narrow) == {"press", "wait", "finished", "other"}, "nothing to type into, nothing to scroll"
    field = build_questions(candidates(FIELD_SCREEN["observation"]), action_space="full",
                            can_scroll=False)["operation"].criteria
    assert "type" in field and "scroll_down" not in field


def test_there_is_no_text_question_when_the_brief_quotes_nothing_to_type() -> None:
    client = FakeClient()
    propose(client, goal="Send a short message", screen=FIELD_SCREEN, tools=WIDE_TOOLS)
    assert "text" not in client.questions[0]


# ------------------------------------------------------------------------------------ pressing


def test_a_confident_press_is_proposed_as_a_bound_tool_call() -> None:
    action, navigator = propose(FakeClient())
    assert action == {"tool": TAP_TOOL, "arguments": {"id": "el:aaa"}, "reason": action["reason"]}
    assert "0.99" in action["reason"] and "Notifications" in action["reason"]
    assert navigator.report()["accepted"] == 1


def test_the_numbered_answer_is_mapped_back_to_the_real_element() -> None:
    action, navigator = propose(FakeClient(target="2"))
    assert action["arguments"] == {"id": "el:bbb"}, "answer 2 is the second control"
    assert navigator.proposals[0]["target_id"] == "el:bbb", "the report names the element"


def test_an_unsure_operation_is_declined() -> None:
    action, navigator = propose(FakeClient(op_conf=0.62))
    assert action is None
    assert navigator.report()["declined"] == {"below_confidence": 1}


def test_a_sure_press_on_an_unsure_control_is_declined() -> None:
    # A press is only as good as the control it lands on.
    action, navigator = propose(FakeClient(op_conf=0.99, target_conf=0.60))
    assert action is None
    assert navigator.report()["declined"] == {"below_confidence": 1}
    assert navigator.proposals[0]["operand_confidence"] == 0.6


def test_a_control_that_is_not_on_this_screen_is_refused() -> None:
    action, navigator = propose(FakeClient(target="9"))
    assert action is None
    assert navigator.report()["declined"] == {"unknown_target": 1}


@pytest.mark.parametrize("operation", ["finished", "other"])
def test_stopping_and_moves_it_cannot_make_go_to_the_chat_model(operation: str) -> None:
    # Offered so the truth has somewhere to go, and refused: ending a run early is this model's
    # worst measured skill, and a long-press or a relaunch is not a move it can make.
    action, navigator = propose(FakeClient(operation=operation, op_conf=1.0), tools=WIDE_TOOLS,
                                action_space="full")
    assert action is None
    assert navigator.report()["declined"] == {f"kind:{operation}": 1}


@pytest.mark.parametrize("operation", ["back", "scroll_down"])
def test_the_narrow_space_hands_back_and_scroll_to_the_chat_model(operation: str) -> None:
    action, navigator = propose(FakeClient(operation=operation, op_conf=1.0), tools=WIDE_TOOLS)
    assert action is None
    assert navigator.report()["declined"] == {f"kind:{operation}": 1}


def test_a_request_failure_costs_a_step_not_a_run() -> None:
    action, navigator = propose(FakeClient(error=TimeoutError("slow")))
    assert action is None
    assert navigator.report()["declined"] == {"request_failed:TimeoutError": 1}


def test_shadow_mode_records_the_press_it_would_have_taken_and_takes_nothing() -> None:
    action, navigator = propose(FakeClient(), shadow=True)
    assert action is None
    report = navigator.report()
    assert report["shadow"] is True and report["accepted"] == 0
    assert report["declined"] == {"shadow": 1}
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
    assert propose(client, tools=["swipe_and_analyze"])[0] is None
    assert client.calls == 0


def test_the_same_press_is_not_proposed_twice_for_an_unchanged_screen() -> None:
    # Observed live: a System One model reads each screen from scratch, so on a screen its own
    # tap failed to change it confidently proposes that tap again, and the run loops.
    navigator = TypeSafeNavigator("g", client=FakeClient(), tools=[TAP_TOOL])
    assert asyncio.run(navigator(SCREEN)) is not None
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}
    assert asyncio.run(navigator(moved())) is not None, "a new screen is a new decision"


def test_a_nonsense_gate_or_space_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        TypeSafeNavigator("g", client=FakeClient(), min_confidence=0.0)
    with pytest.raises(ValueError):
        TypeSafeNavigator("g", client=FakeClient(), action_space="everything")


# ----------------------------------------------------------------------- the other moves


def test_the_full_space_scrolls_and_goes_back() -> None:
    action, _ = propose(FakeClient(operation="scroll_up"), screen=LIST_SCREEN, tools=WIDE_TOOLS,
                        action_space="full")
    assert action["tool"] == "scroll_and_analyze" and action["arguments"] == {"direction": "up"}
    action, _ = propose(FakeClient(operation="back"), tools=WIDE_TOOLS, action_space="full")
    assert action["tool"] == "back_gesture_and_analyze" and action["arguments"] == {}


def test_a_move_whose_tool_was_not_offered_is_refused() -> None:
    action, navigator = propose(FakeClient(operation="scroll_down"), screen=LIST_SCREEN, action_space="full")
    assert action is None
    assert navigator.report()["declined"] == {"scroll_not_offered": 1}
    action, navigator = propose(FakeClient(operation="wait"))
    assert navigator.report()["declined"] == {"wait_not_offered": 1}


def test_the_same_scroll_is_not_repeated_on_a_screen_it_did_not_move() -> None:
    navigator = TypeSafeNavigator("g", client=FakeClient(operation="scroll_down"), tools=WIDE_TOOLS,
                                  action_space="full")
    assert asyncio.run(navigator(LIST_SCREEN)) is not None
    assert asyncio.run(navigator(LIST_SCREEN)) is None
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}


def test_waiting_is_a_move_in_either_space() -> None:
    for space in ("taps", "full"):
        action, _ = propose(FakeClient(operation="wait"), tools=WIDE_TOOLS, action_space=space)
        assert action["tool"] == "wait_and_analyze" and action["arguments"] == {"idle": True}


def test_a_second_wait_on_the_same_activity_escalates_even_as_the_screen_churns() -> None:
    # A loading screen re-fingerprints on every frame it redraws, so keying a wait on the
    # fingerprint would never repeat and never escalate. The activity holds still.
    navigator = TypeSafeNavigator("g", client=FakeClient(operation="wait"), tools=WIDE_TOOLS)

    def loading(fingerprint):
        return {"ok": True, "change": {"activity_after": ".Auth"},
                "observation": {**SCREEN["observation"], "meta": {"fingerprint": fingerprint}}}

    assert asyncio.run(navigator(loading("fp-1"))) is not None
    assert asyncio.run(navigator(loading("fp-2"))) is None, "same activity, still loading"
    assert navigator.report()["declined"] == {"repeat_on_unchanged_screen": 1}


# ------------------------------------------------------------------------------------ typing


def test_the_text_typed_is_one_the_brief_quotes_chosen_among_them() -> None:
    # Nothing is generated: the string is the author's, and Jev only says which one comes next.
    action, _ = propose(FakeClient(operation="type", text="2"), goal=BRIEF, screen=FIELD_SCREEN,
                        tools=WIDE_TOOLS)
    assert action == {"tool": "input_and_analyze",
                      "arguments": {"id": "el:field", "text": "Thanks", "submit": False},
                      "reason": action["reason"]}


def test_typing_without_a_quoted_text_goes_to_the_chat_model() -> None:
    action, navigator = propose(FakeClient(operation="type"), goal="Send a short message",
                                screen=FIELD_SCREEN, tools=WIDE_TOOLS)
    assert action is None
    assert navigator.report()["declined"] == {"kind:type": 1}


def test_an_unsure_text_is_not_typed() -> None:
    action, navigator = propose(FakeClient(operation="type", text_conf=0.5), goal=BRIEF,
                                screen=FIELD_SCREEN, tools=WIDE_TOOLS)
    assert action is None
    assert navigator.report()["declined"] == {"below_confidence": 1}


def test_with_two_fields_the_text_goes_into_the_one_the_target_names() -> None:
    brief = "Type exactly `someone@example.com` into Email, then tap `Sign in`."
    action, _ = propose(FakeClient(operation="type", target="1"), goal=brief,
                        screen=TWO_FIELDS_SCREEN, tools=WIDE_TOOLS)
    assert action["arguments"]["id"] == "el:user"
    action, navigator = propose(FakeClient(operation="type", target="3"), goal=brief,
                                screen=TWO_FIELDS_SCREEN, tools=WIDE_TOOLS)
    assert action is None, "the target is a button, so which field is unknown"
    assert navigator.report()["declined"] == {"kind:type": 1}


def test_the_quoted_texts_typed_so_far_are_told_and_a_secret_never_is() -> None:
    client = FakeClient(op_conf=0.1)
    navigator = TypeSafeNavigator(BRIEF, client=client, tools=WIDE_TOOLS)
    asyncio.run(navigator(FIELD_SCREEN))
    navigator.observed("input_and_analyze", {"id": "el:field", "text": "Hello there"})
    asyncio.run(navigator(moved("fp-3", FIELD_SCREEN)))
    navigator.observed("input_and_analyze", {"id": "el:field", "text": "hunter2", "submit": True})
    asyncio.run(navigator(moved("fp-4", FIELD_SCREEN)))

    state = client.states[-1]
    assert state["typed_so_far"] == ["Hello there"]
    assert [t["you_chose"] for t in state["journey_so_far"]] == [
        "type `Hello there` into the text field 'Ask me anything'",
        "type text into the text field 'Ask me anything' and send it",
    ]
    assert "hunter2" not in json.dumps(client.states)


# ----------------------------------------------------------------------------- the menu


def test_only_interactive_controls_become_options() -> None:
    options = candidates(SCREEN["observation"])
    assert set(options) == {"el:aaa", "el:bbb"}, "static text is not a tap target"
    assert options["el:aaa"] == "Notifications"


def test_the_controls_are_offered_as_a_numbered_menu_not_as_their_ids() -> None:
    # AUA ids are 32-character hex digests, and jev-1.13 is documented to do worse on opaque and
    # numeric representations than on semantic ones.
    criteria, by_index = numbered(candidates(SCREEN["observation"]))
    assert criteria == {"1": "Notifications", "2": "Privacy"}
    assert by_index == {"1": "el:aaa", "2": "el:bbb"}


def test_a_switch_reads_its_state_in_the_option_label() -> None:
    options = candidates({"elements": [
        {"id": "el:s1", "text": "Promotional messages", "checked": False, "clickable": True},
        {"id": "el:s2", "text": "Security alerts", "checked": True, "clickable": True},
    ]})
    assert options["el:s1"].endswith("[switch is OFF]")
    assert options["el:s2"].endswith("[switch is ON]")


def test_a_status_bar_item_with_a_checked_field_is_not_a_switch() -> None:
    # A raw hierarchy dump puts `checked: false` on every node, the status-bar clock included.
    options = candidates({"elements": [
        {"id": "el:clock", "text": "11:28", "clickable": False, "checkable": False, "checked": False},
        {"id": "el:login", "text": "Log in", "clickable": True, "checkable": False, "checked": False},
        {"id": "el:dark", "text": "Dark mode", "clickable": True, "checkable": True, "checked": True},
    ]})
    assert "el:clock" not in options
    assert options["el:login"] == "Log in", "a plain button is not a switch"
    assert options["el:dark"].endswith("[switch is ON]")


def test_a_text_field_is_named_as_one() -> None:
    # A field is labelled by its hint, and a goal that named the field matched the button beside
    # it whose id shared a word. The role is what tells them apart.
    options = candidates(FIELD_SCREEN["observation"])
    assert options["el:field"] == "Ask me anything (text field)"
    assert options["el:send"] == "Send"


def test_a_control_the_app_never_named_is_placed_not_hashed() -> None:
    options = candidates({
        "screen": {"width": 1000, "height": 2000},
        "elements": [{"id": "el:deadbeefdeadbeefdeadbeef", "clickable": True, "bounds": [800, 100, 960, 220]},
                     {"id": "el:aaa", "text": "Settings", "clickable": True}],
    })
    assert options["el:deadbeefdeadbeefdeadbeef"] == "unlabelled control, top right of the screen"
    assert candidates({"elements": [{"id": "el:x", "clickable": True}]})["el:x"] == "unlabelled control"


@pytest.mark.parametrize("bounds,expected", [
    ([0, 0, 100, 100], "top left"),
    ([450, 950, 550, 1050], "middle centre"),
    ([900, 1900, 1000, 2000], "bottom right"),
])
def test_an_unnamed_control_is_placed_on_the_right_third(bounds, expected) -> None:
    from experiments.aua_controller.typesafe_navigator import where

    assert where({"bounds": bounds}, {"width": 1000, "height": 2000}) == f"unlabelled control, {expected} of the screen"


def test_the_controllers_own_tool_shape_is_understood() -> None:
    # The controller offers OpenAI-shaped entries. Reading the top level finds no name, which
    # silently declined every step of a live shadow run before this was pinned.
    from experiments.aua_controller.typesafe_navigator import tool_names

    offered = [{"type": "function", "function": {"name": TAP_TOOL, "parameters": {}}},
               {"type": "function", "function": {"name": "session_finish", "parameters": {}}}]
    assert tool_names(offered) == {TAP_TOOL, "session_finish"}
    assert tool_names([TAP_TOOL]) == {TAP_TOOL}
    assert tool_names([{"type": "function"}, 7, None]) == set()
    assert propose(FakeClient(), tools=offered)[0] is not None


# ------------------------------------------------------------------------------ the journey


def test_the_whole_journey_is_sent_in_the_models_own_words() -> None:
    client = FakeClient()
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    asyncio.run(navigator(moved()))

    first, second = client.states
    assert first["journey_so_far"] == [] and "previous_screen" not in first
    assert second["journey_so_far"] == [{"n": 1, "you_chose": "press 'Notifications'"}]
    assert "[Notifications]" in second["previous_screen"]


def test_a_screen_that_did_not_move_is_said_so_in_the_journey() -> None:
    # This is the fact that stops the loop: the model can see its own tap changed nothing.
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    asyncio.run(navigator(SCREEN))
    assert client.states[1]["journey_so_far"][0]["screen_did_not_change"] is True


def test_a_step_the_chat_model_took_is_named_in_the_same_words() -> None:
    client = FakeClient(op_conf=0.1)
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    for tool, expected in (("scroll_and_analyze", "scroll"), ("back_gesture_and_analyze", "back"),
                           ("wait_and_analyze", "wait"), ("session_finish", "done")):
        asyncio.run(navigator(moved(f"fp-{tool}")))
        navigator.observed(tool, {})
        assert navigator._pending["you_chose"] == expected, tool


def test_a_turn_survives_a_screen_this_navigator_could_not_read() -> None:
    # `too_few_controls` returned before the turn was closed, so the turn was closed later
    # against a screen it never saw.
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    bare = {"ok": True, "observation": {"meta": {"fingerprint": "fp-2"},
                                        "elements": [{"id": "el:z", "text": "OK", "clickable": True}]}}
    assert asyncio.run(navigator(bare)) is None
    navigator.observed("back_gesture_and_analyze", {})
    asyncio.run(navigator(moved("fp-3")))
    assert [t["you_chose"] for t in client.states[-1]["journey_so_far"]] == ["press 'Notifications'", "back"]


def test_a_forgotten_turn_never_reaches_the_journey() -> None:
    # A press AUA refused as stale was never sent, so it is not part of the story.
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    action = asyncio.run(navigator(SCREEN))
    navigator.observed(action["tool"], action["arguments"])
    navigator.forget()
    asyncio.run(navigator(moved()))
    assert client.states[-1]["journey_so_far"] == []


def test_the_journey_is_trimmed_from_the_oldest_end() -> None:
    from experiments.aua_controller.typesafe_navigator import MAX_JOURNEY_CHARS

    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    navigator._journey = [{"n": i, "you_chose": "press 'x'", "pad": "y" * 400} for i in range(1, 401)]
    asyncio.run(navigator(SCREEN))
    journey = client.states[0]["journey_so_far"]
    assert len(json.dumps(journey)) <= MAX_JOURNEY_CHARS
    assert journey[-1]["n"] == 400, "the newest turns are the ones a loop is made of"


def test_a_screen_sketch_marks_what_could_be_pressed_and_is_cut_short() -> None:
    said = sketch({"observation": {"elements": [{"text": "Welcome back"},
                                                {"text": "Sign in", "clickable": True}]}})
    assert said == "Welcome back · [Sign in]"
    many = sketch({"observation": {"elements": [{"text": f"Row number {n}"} for n in range(40)]}})
    assert len(many) < 400 and "Row number 0" in many


# ----------------------------------------------------------------------------- the wire


def test_no_element_digest_fingerprint_or_bounds_reach_the_model() -> None:
    client = FakeClient()
    navigator = TypeSafeNavigator("g", client=client, tools=[TAP_TOOL])
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    asyncio.run(navigator(moved()))
    body = json.dumps(client.states)
    assert not re.search(r"[0-9a-f]{32}", body) and "el:aaa" not in body
    screen = client.states[0]["this_is_the_new_screen"]
    assert "fingerprint" not in json.dumps(screen) and "bounds" not in json.dumps(screen)


def test_the_calls_still_in_the_air_reach_the_model_and_a_quiet_screen_says_nothing() -> None:
    # Told the login POST had not answered, the model waited instead of pressing sign-in again.
    busy = screen_for_model({"observation": {"screen": {"package": "com.example.app"},
                                             "meta": {"network_calls": ["POST /v1/auth/login"]},
                                             "elements": [{"text": "Sign in", "id": "el:abc"}]}})
    assert busy["network"] == ["POST /v1/auth/login"]
    quiet = screen_for_model({"observation": {"screen": {}, "meta": {"fingerprint": "abc"},
                                              "elements": [{"text": "Sign in"}]}})
    assert "network" not in quiet


def test_compaction_does_not_drop_the_calls_still_in_the_air() -> None:
    compact = compact_frame({"observation": {
        "screen": {"package": "com.example.app"},
        "meta": {"fingerprint": "abc", "network_calls": ["POST /v1/auth/login"]},
        "elements": [{"text": "Sign in", "id": "el:abc", "clickable": True}],
    }}, keep_ids=True)
    assert compact["observation"]["meta"]["network_calls"] == ["POST /v1/auth/login"]


def test_every_call_is_written_to_the_transcript_with_its_verdict_and_cost(tmp_path) -> None:
    path = tmp_path / "system-one-turns.jsonl"
    navigator = TypeSafeNavigator("Open notification settings", client=FakeClient(op_conf=0.42),
                                  tools=[TAP_TOOL], transcript_path=path)
    asyncio.run(navigator(SCREEN))
    navigator.observed(TAP_TOOL, {"id": "el:aaa"})
    navigator.client.op_conf = 0.99
    asyncio.run(navigator(moved()))

    declined, accepted = [json.loads(line) for line in path.read_text().splitlines()]
    assert declined["menu"] == {"1": "Notifications", "2": "Privacy"}
    assert set(declined["request"]) == {"model", "state", "questions"}
    assert set(declined["request"]["questions"]) == {"operation", "target"}
    assert declined["verdict"]["declined_because"] == "below_confidence"
    assert declined["verdict"]["gate"] == 0.42 and declined["verdict"]["gate_needed"] == 0.85
    assert accepted["verdict"]["accepted"] is True and "declined_because" not in accepted["verdict"]
    assert accepted["input_tokens"] == 430 and accepted["usd"] == pytest.approx(430 * 42 / 1e9)
    assert navigator.report()["usd"] == pytest.approx(2 * 430 * 42 / 1e9)


def test_a_run_without_a_transcript_path_still_works(monkeypatch) -> None:
    written: list[object] = []
    navigator = TypeSafeNavigator("g", client=FakeClient(), tools=[TAP_TOOL])
    monkeypatch.setattr(navigator, "_record", lambda entry: written.append(entry))
    assert asyncio.run(navigator(SCREEN)) is not None
    assert navigator.report()["transcript"] is None
    assert written, "the turn is still assembled; only the file is absent"



def test_jev_does_not_read_past_a_step_only_the_chat_model_can_take() -> None:
    # Live, Jev read a whole cold-start brief and did its post-restart taps before the restart,
    # which only the chat model can do. It now reads up to that step until the run has taken it.
    brief = ("Tap `Chats`. Then force-close the app with app_force_stop. Then relaunch it with "
             "app_relaunch_and_analyze. Then tap the settings gear.")
    tools = [*WIDE_TOOLS, "app_force_stop", "app_relaunch_and_analyze"]
    client = FakeClient()
    navigator = TypeSafeNavigator(brief, client=client, tools=tools)
    asyncio.run(navigator(SCREEN))
    first = client.states[-1]["goal"]
    assert first.startswith("Tap `Chats`. Then force-close the app with app_force_stop.")
    assert "settings gear" not in first and "relaunch" not in first
    assert first.endswith("The brief goes on after this step.")
    navigator.observed("app_force_stop")
    asyncio.run(navigator(moved("fp-3")))
    assert "app_relaunch_and_analyze." in client.states[-1]["goal"]
    assert "settings gear" not in client.states[-1]["goal"]
    navigator.observed("app_relaunch_and_analyze")
    asyncio.run(navigator(moved("fp-4")))
    assert client.states[-1]["goal"] == brief, "once both are done it reads the whole brief again"
