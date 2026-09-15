"""A gesture the controller may use must be one the flow store will accept.

`flow_save` refuses a route containing a raw swipe -- "swipe capture omits
coordinates/container/percentage; author `scroll: up` by hand instead - a scroll names its container
and replays, where a raw swipe is positional". Offering swipe without scroll therefore leaves a
controller unable to move the screen and still produce a saveable flow: every such route is refused,
whatever the app did. Callers that require a saved flow as their definition of done cannot pass at
all.
"""

from experiments.aua_controller.run_realapp import CONTROLLER_TOOLS


def test_offering_swipe_requires_offering_scroll() -> None:
    if "swipe_and_analyze" in CONTROLLER_TOOLS:
        assert "scroll_and_analyze" in CONTROLLER_TOOLS, (
            "the controller can move the screen only with a gesture flow_save refuses, so no "
            "scrolling journey can leave a replayable flow"
        )


def test_the_controller_keeps_its_essential_verbs() -> None:
    for tool in ("analyze_screen", "tap_and_analyze", "input_and_analyze", "session_finish"):
        assert tool in CONTROLLER_TOOLS


def test_tool_names_are_unique() -> None:
    assert len(CONTROLLER_TOOLS) == len(set(CONTROLLER_TOOLS))
