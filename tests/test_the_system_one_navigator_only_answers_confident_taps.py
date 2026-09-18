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
