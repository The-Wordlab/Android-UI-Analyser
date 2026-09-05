"""A canvas has no element to name, so a coordinate is the correct address.

Three sweep lanes hit the same wall independently. A step-sequencer grid published one
accessibility node per row - text truncated to "Beat grid Row 1, step " - with individual
cells carrying no id, no text and no content-desc. Nothing in the element vocabulary can
address a cell that does not exist as a node, so painting a beat meant dropping to
`adb shell input tap` with coordinates worked out from the grid's pixel pitch.

That drop is the real cost: an `adb` tap happens outside the engine, so it is not recorded,
and a journey containing one cannot be captured as a flow at all. The grid surfaces are
exactly the ones worth capturing, because they take many steps to drive.

The other half of this is what a point tap must NOT do. `tap` applies two corrections to
`el.center` on separate axes - `_tap_point` moves x toward a named phrase inside a wrapped
line, `_aim` moves y up out of the system navigation bar. Both exist to guess better than a
bounding box. A caller passing an explicit point has already said where, so both are
bypassed; the nav-bar lift in particular must not apply, since tapping the bar itself is a
legitimate request.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import _parse_point
from android_ui_analyser.flows import parse_flow_yaml, render_flow_yaml


class _Recorder:
    """Stands in for the device: remembers where it was told to click."""

    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("412,733", (412, 733)),
        ("412, 733", (412, 733)),  # a human types the space
        ("412.0,733.4", (412, 733)),  # a measured value off a screenshot
        ("0,0", (0, 0)),
    ],
)
def test_a_point_is_parsed(arg, expected):
    assert _parse_point(arg) == expected


@pytest.mark.parametrize("arg", [None, "", "412", "412,733,900", "left,733", "-1,733", "412,-1"])
def test_a_bad_point_is_rejected_rather_than_guessed(arg):
    assert _parse_point(arg) is None


def test_a_point_step_round_trips_through_yaml():
    """Recorded like any other action means it must survive save and reload."""
    flow = parse_flow_yaml(
        "schema_version: 1\nname: paint\nsteps:\n  - tap_point: '412,733'\n", name="paint"
    )
    assert [(s.kind, s.arg) for s in flow.steps] == [("tap-point", "412,733")]
    assert "tap_point: 412,733" in render_flow_yaml(flow)


def test_the_mapping_form_is_accepted_too():
    flow = parse_flow_yaml(
        "schema_version: 1\nname: paint\nsteps:\n  - tap_point: {point: '100,200'}\n", name="paint"
    )
    assert flow.steps[0].arg == "100,200"


def test_tap_point_clicks_exactly_where_it_was_told():
    """No aim correction on either axis - including a y that sits inside the nav bar.

    1208 is the coordinate from the nav-bar report: on this pool the bar window starts at
    y=1184, so `_aim` would lift a y of 1208 clear of it. An explicit point must not be lifted.
    """
    from android_ui_analyser.engine import Engine

    class _NoOpContext:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Stub:
        """Only the collaborators tap_point touches - `device` is a property on Engine."""

        def __init__(self) -> None:
            self.device = _Recorder()
            self.platform = type(
                "_Platform",
                (),
                {
                    "runtime_capability": staticmethod(
                        lambda capability, runtime: runtime
                    )
                },
            )()
            self.recorded: list[str | None] = []

        def _step(self, kind, element=None, *, arg=None, submit=False):
            self.recorded.append(arg)
            return arg

        def _acting(self, mark):
            return _NoOpContext()

        def _record_action_safe(self, step):
            self.recorded_step = step

        def _observe(self, result, observe, with_image):
            return result

    stub = _Stub()
    result = Engine.tap_point(stub, 360, 1208, observe=False)

    assert stub.device.clicks == [(360, 1208)], "the point must not be moved on either axis"
    assert result.target == [360, 1208]
    assert stub.recorded == ["360,1208"], "unrecorded means it can never become a flow"
    assert stub.recorded_step == "360,1208", "the step must reach the recorder"
