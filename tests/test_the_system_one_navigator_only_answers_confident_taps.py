"""The navigator's value is in what it refuses, so the refusals are what these pin.

Every path that is not a confident tap must return None, because None is what hands the step
back to the chat model. A navigator that answers a step it should not have is worse than one
that answers nothing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.typesafe_navigator import (  # noqa: E402
    TAP_TOOL,
    TypeSafeNavigator,
    candidates,
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
    def __init__(self, kind="tap", target="el:aaa", kind_conf=0.99, target_conf=0.95,
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


@pytest.mark.parametrize("kind", ["done", "back", "scroll", "type"])
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
    assert navigator.proposals[0]["target"] == "el:aaa"


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


WIDE_TOOLS = [TAP_TOOL, "scroll_and_analyze", "back_gesture_and_analyze", "session_finish"]


def wide(client, **kwargs):
    navigator = TypeSafeNavigator("Open notification settings", client=client, tools=WIDE_TOOLS,
                                  action_space="full", **kwargs)
    return asyncio.run(navigator(SCREEN)), navigator


def test_the_widened_space_scrolls_with_the_direction_it_chose() -> None:
    action, _ = wide(WideClient(kind="scroll", direction="up"))
    assert action["tool"] == "scroll_and_analyze"
    assert action["arguments"] == {"direction": "up"}


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
    navigator = TypeSafeNavigator("g", client=WideClient(kind="scroll"), tools=[TAP_TOOL],
                                  action_space="full")
    assert asyncio.run(navigator(SCREEN)) is None
    assert navigator.report()["declined"] == {"scroll_not_offered": 1}


def test_the_operand_gates_the_step_not_the_target_of_an_action_without_one() -> None:
    # A scroll is gated on its direction; an uncertain direction is an uncertain scroll even
    # when the tap target it did not choose came back certain.
    action, navigator = wide(WideClient(kind="scroll", operand_conf=0.40))
    assert action is None
    assert navigator.report()["declined"] == {"below_confidence": 1}


def test_the_same_scroll_is_not_repeated_on_a_screen_it_did_not_move() -> None:
    client = WideClient(kind="scroll")
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
    assert {"action", "target", "settled", "direction", "outcome"} == set(client.questions)


def test_the_narrow_default_asks_no_operand_questions() -> None:
    client = WideClient()
    TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS)
    asyncio.run(TypeSafeNavigator("g", client=client, tools=WIDE_TOOLS)(SCREEN))
    assert set(client.questions) == {"action", "target", "settled"}


def test_an_unknown_action_space_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        TypeSafeNavigator("g", client=FakeClient(), action_space="everything")
