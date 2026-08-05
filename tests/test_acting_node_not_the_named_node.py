"""A selector resolved the node you named, which is frequently not the node that acts.

Observed 2026-08-04: a lane filed `FAIL_CRITICAL` against a product because a bottom-sheet tile
reported `clickable:false, enabled:true`. The control was **enabled and working**.

The design-system tile puts the click on an inner `Box`, and the visible title exists only as
*non-clickable* nodes **outside** that Box, because `Modifier.clickable` does not merge
descendants. So matching by visible text returns a node with no click action and no `disabled()`
semantics — and it returns that same node, with those same two flag values, **identically whether
the control is enabled or disabled**. There was nothing in the output to tell the two apart. The
label's centre also sits ~110px below the clickable bounds, so a tap there dispatches, returns
`ok:true`, and does nothing. It took two devices and a luminance measurement to retract.

Two consequences, and the second is the one that cost the verdict:

* a tap aimed at the caption misses the control;
* a *read* of the caption's `enabled` flag describes a caption, and reads as a product verdict.

Note what the geometry rules out. The label is **outside** the control's bounds, so bounds
containment cannot get from either node to the other in either direction — which is why
`Element.parent` had to exist. Only the tree relates them.

The deliberate boundary: a node that is itself clickable is never retargeted, so the vast majority
of existing taps are byte-for-byte unchanged. Redirection happens only where the named node carries
no interaction at all — which today dispatches into whatever lies under a caption and usually does
nothing. And where several nearby nodes could be the control, nothing is chosen: `ambiguous` with
the candidates listed is the honest answer, and guessing would recreate this bug with better
manners.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.hierarchy import parse_hierarchy
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.selectors import acting_node, match_selector
from conftest import FakeDevice, make_config

SCREEN = (1080, 2400)
PKG = "com.example.app"


def _tile(*, enabled: str = "true") -> str:
    """The reported shape: click on an inner Box, caption as a sibling *outside* its bounds.

    The caption is 50px below the clickable box, and its centre 195px below the box's centre —
    the same scale as the ~110px measured in the field.
    """
    return f"""<hierarchy rotation="0">
 <node class="android.widget.LinearLayout" package="{PKG}" enabled="true" bounds="[0,0][1080,2400]">
  <node class="android.view.View" package="{PKG}" enabled="true"
        resource-id="{PKG}:id/tileContainer" bounds="[40,900][1040,1300]">
   <node class="android.view.View" package="{PKG}" clickable="true" enabled="{enabled}"
         resource-id="{PKG}:id/tileHit" bounds="[60,920][1020,1150]"/>
   <node class="android.widget.TextView" package="{PKG}" enabled="true" text="Beat Painter"
         bounds="[60,1200][1020,1260]"/>
  </node>
 </node>
</hierarchy>"""


# The container carries an app resource-id so the parser keeps it as an element — an unlabelled,
# non-actionable, non-leaf wrapper is folded away, and then its children have no shared ancestor
# to search. That is a real property of the parser, not a fixture convenience.
BUSY = f"""<hierarchy rotation="0">
 <node class="android.webkit.WebView" package="{PKG}" enabled="true"
       resource-id="{PKG}:id/containerAccounts" bounds="[0,0][1080,2400]">
  <node class="android.view.View" package="{PKG}" enabled="true" text="Choose an account"
        bounds="[40,200][1040,280]"/>
  <node class="android.view.View" package="{PKG}" enabled="true" clickable="true" text="First option"
        bounds="[40,400][1040,500]"/>
  <node class="android.view.View" package="{PKG}" enabled="true" clickable="true" text="Second option"
        bounds="[40,520][1040,620]"/>
 </node>
</hierarchy>"""

LONELY = f"""<hierarchy rotation="0">
 <node class="android.widget.TextView" package="{PKG}" enabled="true" text="Version 1.2.3"
       bounds="[40,200][1040,260]"/>
</hierarchy>"""


def _engine(tmp_path: Path, xml: str) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    dev = FakeDevice(hierarchy_xml=xml, package=PKG, width=SCREEN[0], height=SCREEN[1])
    return Engine(cfg, device=dev, factory=ProviderFactory(cfg))


def _named(xml: str, label: str):
    els = parse_hierarchy(xml, SCREEN)
    return els, match_selector(els, text=label)[0]


# ------------------------------------------------- the tree is the only route between them


def test_the_caption_lies_outside_the_controls_bounds(tmp_path: Path) -> None:
    """Pins the geometry that makes containment useless, so the parent links are load-bearing."""
    els, caption = _named(_tile(), "Beat Painter")
    control = next(e for e in els if e.clickable)
    cx1, cy1, cx2, cy2 = control.bounds
    lx, ly = caption.center
    assert not (cx1 <= lx <= cx2 and cy1 <= ly <= cy2), "if it were inside, bounds would suffice"
    assert caption.parent == control.parent, "siblings under one container — only the tree links them"


def test_the_named_node_reports_the_misleading_pair(tmp_path: Path) -> None:
    """The exact reading that was filed as a critical product failure."""
    _els, caption = _named(_tile(), "Beat Painter")
    assert (caption.clickable, caption.enabled) == (False, True)


def test_the_control_is_found_through_the_shared_ancestor(tmp_path: Path) -> None:
    els, caption = _named(_tile(), "Beat Painter")
    found = acting_node(els, caption)
    assert found.relation == "sibling-subtree"
    assert found.element.clickable is True
    assert found.element.resource_id and found.element.resource_id.endswith("tileHit")


def test_enabled_and_disabled_no_longer_read_identically(tmp_path: Path) -> None:
    """The core defect: the named node's flags were the same either way, so nothing discriminated.

    Same selector, same caption, opposite control state — the resolved control must differ.
    """
    reports = {}
    for state in ("true", "false"):
        els, caption = _named(_tile(enabled=state), "Beat Painter")
        assert (caption.clickable, caption.enabled) == (False, True), "caption is identical"
        reports[state] = acting_node(els, caption).element.enabled
    assert reports == {"true": True, "false": False}


# ------------------------------------------------- what must NOT change, and what must refuse


def test_a_clickable_node_is_never_retargeted(tmp_path: Path) -> None:
    """The blast-radius boundary: ordinary taps must be untouched by this change."""
    els, button = _named(BUSY, "First option")
    found = acting_node(els, button)
    assert found.relation == "self" and found.element.id == button.id
    assert not found.redirected


def test_several_candidates_are_reported_rather_than_guessed(tmp_path: Path) -> None:
    """A heading in a busy container must not be redirected to whichever button sorted first."""
    els, heading = _named(BUSY, "Choose an account")
    found = acting_node(els, heading)
    assert found.relation == "ambiguous"
    assert found.element.id == heading.id, "target unchanged when the choice would be a guess"
    assert len(found.candidates) == 2


def test_nothing_nearby_acting_is_said_plainly(tmp_path: Path) -> None:
    els, label = _named(LONELY, "Version 1.2.3")
    found = acting_node(els, label)
    assert found.relation == "none"
    assert found.element.id == label.id


# ------------------------------------------------- the acting node in output


def test_tap_aims_at_the_control_and_says_so(tmp_path: Path) -> None:
    """A tap at the caption centre dispatched and did nothing; it must land on the control."""
    eng = _engine(tmp_path, _tile())
    eng.analyze(source="hierarchy")
    els, caption = _named(_tile(), "Beat Painter")
    control = next(e for e in els if e.clickable)

    out = eng.tap(selector={"text": "Beat Painter"}, observe=False)

    assert out.ok
    assert out.target is not None
    assert out.target[1] != caption.center[1], "aimed at the caption again"
    assert abs(out.target[1] - control.center[1]) <= 40, f"not on the control: {out.target}"
    assert out.acting is not None
    assert out.acting["relation"] == "sibling-subtree"
    assert out.acting["named_id"] == caption.id
    assert out.acting["id"] == control.id


def test_tap_on_a_real_control_reports_that_it_acted_on_what_was_named(tmp_path: Path) -> None:
    """`acting` is reported unconditionally: "the node you named is the one that acts" is a fact
    a reader needs, and a field that appears only sometimes gets read as "nothing to see"."""
    eng = _engine(tmp_path, BUSY)
    eng.analyze(source="hierarchy")

    out = eng.tap(selector={"text": "First option"}, observe=False)

    assert out.acting is not None
    assert out.acting["relation"] == "self"
    assert "named_id" not in out.acting


def test_target_report_separates_the_caption_from_the_control(tmp_path: Path) -> None:
    """The read-only half — the one that would have prevented the retracted verdict."""
    eng = _engine(tmp_path, _tile(enabled="false"))
    eng.analyze(source="hierarchy")

    out = eng.target_report(text="Beat Painter")

    assert out["acts"] is False, "the caption does not act"
    # `compact()` omits `clickable` when it is False, which is exactly the shape that made the
    # caption look like an ordinary element — so absence is the assertion.
    assert out["named"].get("clickable") is not True
    # The state that belongs to the control, which is the answer that was missing.
    assert out["control"]["clickable"] is True
    assert out["control"]["enabled"] is False
    assert out["acting"]["relation"] == "sibling-subtree"
    assert out["hint"] and "caption" in out["hint"]
    assert out["tap_point"][1] < out["named"]["center"][1], "would tap the control, not the caption"


def test_target_report_on_a_real_control_has_no_caveat(tmp_path: Path) -> None:
    eng = _engine(tmp_path, BUSY)
    eng.analyze(source="hierarchy")

    out = eng.target_report(text="First option")

    assert out["acts"] is True
    assert out["hint"] is None
    assert out["control"]["clickable"] is True


@pytest.mark.parametrize("bad", [{}, {"text": None}])
def test_target_needs_a_selector(tmp_path: Path, bad: dict) -> None:
    """A bare `target` must not silently describe some arbitrary node."""
    from android_ui_analyser.errors import UsageError

    eng = _engine(tmp_path, BUSY)
    with pytest.raises(UsageError):
        eng.target_report(**bad)
