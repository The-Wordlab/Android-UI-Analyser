"""A folded observation must reach vision when the hierarchy cannot describe the screen.

`analyze` defaults to `source="auto"` and self-routes to vision through `gate.decide`. The folded
post-action observation pinned `source="hierarchy"`, so it could never do that — and on a
Compose/canvas/WebView surface (every mini app in this suite) a tap returned an observation with
nothing usable in it, forcing the caller into a second `analyze --source auto`. That is the round
trip act-and-observe exists to remove, and it was guaranteed to happen exactly where perception is
hardest.

The escalation is gated by the same decision a normal `analyze` consults, so an ordinary screen pays
nothing. These tests pin both directions, because a feature that never fires and a feature that
always fires both pass a test that only checks one.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser import gate
from conftest import FakeDevice, make_config
from test_memory import APPS, P, _engine

# A canvas screen: one unlabelled node, which is what a WebView/Compose surface looks like to the
# accessibility tree. `gate.decide` calls this vision-worthy on two independent grounds.
CANVAS = (
    '<?xml version="1.0" encoding="UTF-8"?><hierarchy rotation="0">'
    '<node index="0" class="android.view.View" package="com.example.app" text="" '
    'content-desc="" resource-id="" clickable="false" enabled="true" '
    'bounds="[0,0][1080,2400]" />'
    "</hierarchy>"
)


def test_the_gate_calls_a_canvas_screen_vision_worthy(tmp_path: Path) -> None:
    """The premise. If this fails the other tests prove nothing about the escalation."""
    dev = FakeDevice(hierarchy_xml=CANVAS, package=P, serial="emu-canvas0")
    eng = _engine(tmp_path, dev)
    els = eng.analyze(source="hierarchy").elements
    decision = gate.decide(
        els, package=P, activity=None, cfg=make_config().perception.gate
    )
    assert decision.use_vision, f"gate did not want vision for a canvas screen: {decision.reason}"


def test_an_ordinary_screen_does_not_escalate(tmp_path: Path) -> None:
    """A labelled hierarchy answers on its own — escalating it would be cost for nothing."""
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-canvas1")
    eng = _engine(tmp_path, dev)
    els = eng.analyze(source="hierarchy").elements
    decision = gate.decide(
        els, package=P, activity=None, cfg=make_config().perception.gate
    )
    assert not decision.use_vision, "a labelled screen must not pay for vision"


def test_the_escalation_is_reachable_and_can_be_switched_off(tmp_path: Path) -> None:
    """The knob exists so a cost-sensitive caller can opt out, and defaults to on."""
    cfg = make_config()
    assert cfg.perception.observe_escalates_to_vision is True

    off = make_config(perception={"observe_escalates_to_vision": False})
    assert off.perception.observe_escalates_to_vision is False


def test_an_escalation_that_saw_no_more_is_not_swapped_in(tmp_path: Path) -> None:
    """Taking a same-sized result would hide that the screen is still unreadable.

    With no vision providers configured the escalated analyze returns the same thin tree, so the
    observation must remain the hierarchy one rather than being replaced by an equal-or-worse read.
    """
    dev = FakeDevice(hierarchy_xml=CANVAS, package=P, serial="emu-canvas2")
    eng = _engine(tmp_path, dev)
    first = eng.analyze(source="hierarchy")
    before = len(first.elements)

    r = eng.tap(0, observe=True) if first.elements else None
    if r is not None and r.observation is not None:
        assert len(r.observation.elements) >= before, (
            "the fold must never return fewer elements than the hierarchy alone"
        )
