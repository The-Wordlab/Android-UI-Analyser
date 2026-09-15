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


def test_a_contract_naming_a_deeplink_can_actually_follow_one():
    """A bullet about a deeplink is unjudgeable by a controller that cannot open one.

    Measured 2026-09-15: a tools scenario asserting that a tool's deeplink opens the same
    launchpad as the tap route left three criteria unverifiable, and the judge said exactly why --
    "there was no OS intent/URL-launch tool available, so the URI was never triggered". The engine
    has always exposed `open_link`; only the controller's list left it out, the same shape as the
    scroll gap this file already guards.
    """
    from experiments.aua_controller.run_realapp import (
        CONTROLLER_TOOLS,
        REALAPP_COMPACT_PROPERTIES,
    )

    assert "open_link_and_analyze" in CONTROLLER_TOOLS, (
        "a contract that names a deeplink needs a controller that can follow it"
    )
    kept = REALAPP_COMPACT_PROPERTIES.get("open_link_and_analyze", frozenset())
    assert "uri" in kept, "which link was followed is the evidence the bullet is judged on"
    assert "package" not in kept and "prefer" not in kept, (
        "routing detail describes this host, not the journey"
    )
