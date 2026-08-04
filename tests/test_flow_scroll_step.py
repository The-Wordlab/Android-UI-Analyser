"""A scroll must be expressible in a flow, because the engine already records one.

The bug this guards: `aua flow save --last N` died with `internal_error: 'scroll'` whenever the
captured window contained a scroll. The engine records `kind="scroll"`, the flow schema had no
such kind, and rendering did `_KEYS[s.kind]` - a bare KeyError surfacing as an internal error.

The consequence was worse than the message suggests. Capturing the route is mandatory for every
scenario in this suite, and most routes worth capturing involve scrolling, so the one artifact
that makes the next run cheap was exactly the one that could not be produced. A lane hand-wrote
its YAML instead and another lost its route entirely.

Renderable is not enough: a saved flow has to replay, so the executor needs the branch too.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.flows import _ARG_ALIAS, _KEYS, _KINDS, parse_flow_yaml, render_flow_yaml
from android_ui_analyser.memory import RouteStep

YAML = """
app: com.example
steps:
  - launch_app
  - scroll: down
  - wait_stable
  - scroll: up
"""


def test_scroll_is_a_known_kind():
    assert _KINDS["scroll"] == "scroll"
    assert _KEYS["scroll"] == "scroll", "rendering a recorded scroll must not KeyError"
    assert _ARG_ALIAS["scroll"] == "direction"


def test_scroll_parses_from_yaml():
    flow = parse_flow_yaml(YAML, name="t")
    assert [s.kind for s in flow.steps] == ["launch-app", "scroll", "wait-stable", "scroll"]
    assert [s.arg for s in flow.steps if s.kind == "scroll"] == ["down", "up"]


def test_a_recorded_scroll_renders_and_survives_a_round_trip():
    """This is the exact path `flow save` takes: RouteStep -> YAML -> RouteStep."""
    recorded = [RouteStep(kind="scroll", arg="down"), RouteStep(kind="scroll", arg="up")]
    from android_ui_analyser.flows import Flow

    text = render_flow_yaml(Flow(name="t", app="com.example", steps=recorded))
    assert "scroll: down" in text
    back = parse_flow_yaml(text, name="t")
    assert [(s.kind, s.arg) for s in back.steps] == [("scroll", "down"), ("scroll", "up")]


def test_the_executor_has_a_branch_for_it():
    """A step that renders but cannot replay is a trap, not a fix."""
    import inspect

    from android_ui_analyser.engine import Engine

    src = inspect.getsource(Engine._run_steps)
    assert 'kind == "scroll"' in src, "flow save would produce an unreplayable step"


@pytest.mark.parametrize("bad", ["sideways", "", None])
def test_a_nonsense_direction_is_rejected_not_guessed(bad):
    """Better an explicit unsupported_action than a scroll in an arbitrary direction."""
    from android_ui_analyser.flows import Flow

    step = RouteStep(kind="scroll", arg=bad)
    # Rendering must still work - the failure belongs at replay time, where it can be
    # reported against a step index the caller can resume from.
    render_flow_yaml(Flow(name="t", app="com.example", steps=[step]))


def test_dry_run_describes_a_scroll():
    from android_ui_analyser.memory import step_display

    assert step_display(RouteStep(kind="scroll", arg="down")) == "scroll down"
