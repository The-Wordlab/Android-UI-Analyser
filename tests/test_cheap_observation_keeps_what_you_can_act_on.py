"""The cheap observation may drop rows a caller cannot use — never rows it must act on.

Trimming the folded observation is right: measured on a 5-scenario run, 37 taps produced 73
separate `analyze` calls because one unfilterable read cost more than two targeted ones. But the
first trim filtered rows by *label*, and a design-system tile puts its click handler on an inner
container with no text, no content-desc and no resource-id. The view then showed the title with
`clickable: false` and no clickable node at all — indistinguishable from "the control is absent or
disabled", which is precisely the reading that produced a false FAIL_CRITICAL against the maths
composer, and precisely what `acting` exists to prevent.

The columns matter for the same reason: `checked`/`selected` are how "is that switch on" is
answered without a screenshot. Trimmed away, a caller cannot tell "off" from "not reported".

`enabled` answers the same question by the opposite route. It is a plain bool the a11y tree
always reports, so `true` is the overwhelming majority and says nothing, while `false` is the
whole message. The observation therefore omits `enabled: true` and **always** emits
`enabled: false` — absence means enabled, and a control you must not tap is never silent. That
is a stronger guarantee than presence was: the old view spent a key on every row to state the
default, and still had no test proving a disabled control was visible at all.
"""

from __future__ import annotations

from android_ui_analyser.config import Config
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, OutputFormat, Screen

_META = Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy")


def _screen() -> AnalyzeResult:
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package="com.example.app", source="hierarchy"),
        elements=[
            # The tile: a labelled non-clickable title, and the unlabelled container that acts.
            Element(id=1, type="TextView", bounds=[40, 700, 600, 760], center=[320, 730],
                    text="Photos", clickable=False, enabled=True, window="app"),
            Element(id=2, type="Box", bounds=[40, 600, 600, 700], center=[320, 650],
                    clickable=True, enabled=True, window="app"),
            Element(id=3, type="Switch", bounds=[40, 900, 600, 980], center=[320, 940],
                    resource_id="com.example.app:id/settingsSwitch", checkable=True,
                    checked=True, clickable=True, enabled=True, window="app"),
            Element(id=4, type="Tab", bounds=[40, 1000, 600, 1080], center=[320, 1040],
                    resource_id="com.example.app:id/catalogTab", selected=True,
                    clickable=True, enabled=True, window="app"),
            # The control a caller must not tap. Its `enabled: false` is the whole payload.
            Element(id=5, type="Button", bounds=[40, 1100, 600, 1180], center=[320, 1140],
                    resource_id="com.example.app:id/submitButton", text="Submit",
                    clickable=True, enabled=False, window="app"),
            # System chrome: still worth dropping — that is where the cost win comes from.
            Element(id=0, type="FrameLayout", bounds=[0, 0, 1080, 74], center=[540, 37],
                    resource_id="com.android.systemui:id/status_bar", window="system"),
        ],
        meta=_META,
    )


def _rows(spec: str) -> list[dict]:
    proj = Projection.for_observation(spec, fmt=OutputFormat.json)
    assert proj is not None
    return proj.apply(_screen().model_dump(mode="json"))["elements"]


def test_an_unlabelled_but_tappable_node_survives() -> None:
    ids = {r["id"] for r in _rows(Config().output.observation_fields)}
    assert 2 in ids, "the node that actually acts was dropped for having no label"
    assert 1 in ids, "and the label it belongs to is still there"


def test_system_chrome_is_still_dropped() -> None:
    # The trim has to keep paying for itself, or it is only a behaviour change.
    assert 0 not in {r["id"] for r in _rows(Config().output.observation_fields)}


def test_state_is_judgeable_from_the_observation() -> None:
    switch = next(r for r in _rows(Config().output.observation_fields) if r["id"] == 3)
    assert switch.get("checked") is True, "'is that switch on' must be answerable without a screenshot"
    tab = next(r for r in _rows(Config().output.observation_fields) if r["id"] == 4)
    assert tab.get("selected") is True, "the active tab must be visible in the default view"


def test_a_disabled_control_says_so_and_an_enabled_one_stays_silent() -> None:
    """`enabled` earns a key only when it is `false`; absence is the documented default."""
    rows = {r["id"]: r for r in _rows(Config().output.observation_fields)}
    assert rows[5]["enabled"] is False, "a control you must not tap is never silent about it"
    assert "enabled" not in rows[3], "'enabled: true' restates the default on every single row"


def test_all_restores_the_full_dump() -> None:
    assert Projection.for_observation("all", fmt=OutputFormat.json) is None


def test_analyze_nonempty_is_unchanged_by_this() -> None:
    """`analyze --nonempty` means "rows a human can read" — it must not gain the tile container."""
    proj = Projection.parse(fmt=OutputFormat.json, fields="id,text", nonempty=True)
    ids = {r["id"] for r in proj.apply(_screen().model_dump(mode="json"))["elements"]}
    assert 2 not in ids, "the label-based filter still applies to analyze --nonempty"
