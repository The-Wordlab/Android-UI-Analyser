"""A flow could not say which "See all" it meant, so two routes stopped short.

Found by two sweep lanes driving a scoreboard screen that shows the same "See all" label
once per section. The CLI has always handled it - `aua tap --text "See all" --index 1` - but
flow YAML had no way to express the same thing, so both routes ended at the ambiguity rather
than crossing it. One capability, two surfaces, and only one of them could use it.

Note this is the opposite of what the deferred-fix list recorded under its flow/CLI parity
item, which claimed `index:` was "available to a flow element step and absent from the
equivalent CLI call". The CLI is the surface that had it.

Out-of-range is deliberately a miss rather than a fallback to the first match: a flow that
asked for the second match and silently got the first would tap the wrong control and carry
on, which is exactly what asking for an index is meant to prevent.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml, render_flow_yaml
from android_ui_analyser.memory import RouteStep
from android_ui_analyser.schema import Element
from android_ui_analyser.selectors import match_step

# Three sections, each with its own "See all" - the shape that stopped the routes.
SCREEN = [
    Element(id=1, type="TextView", text="Groups", bounds=[0, 100, 720, 140], center=[360, 120]),
    Element(id=2, type="TextView", text="See all", bounds=[600, 100, 700, 140], center=[650, 120]),
    Element(id=3, type="TextView", text="Matches", bounds=[0, 300, 720, 340], center=[360, 320]),
    Element(id=4, type="TextView", text="See all", bounds=[600, 300, 700, 340], center=[650, 320]),
    Element(id=5, type="TextView", text="Ranking", bounds=[0, 500, 720, 540], center=[360, 520]),
    Element(id=6, type="TextView", text="See all", bounds=[600, 500, 700, 540], center=[650, 520]),
]


def test_a_step_without_an_index_still_takes_the_first():
    """The precedence that every existing flow depends on must not shift."""
    assert match_step(SCREEN, RouteStep(kind="tap", label="See all")).id == 2


@pytest.mark.parametrize(("nth", "expected"), [(0, 2), (1, 4), (2, 6)])
def test_an_index_selects_the_nth_match(nth, expected):
    assert match_step(SCREEN, RouteStep(kind="tap", label="See all", index=nth)).id == expected


def test_an_out_of_range_index_misses_rather_than_guessing():
    assert match_step(SCREEN, RouteStep(kind="tap", label="See all", index=3)) is None


def test_an_index_applies_to_a_resource_id_selector_too():
    screen = [
        Element(id=7, type="Button", resource_id="app:id/row", bounds=[0, 0, 100, 40], center=[50, 20]),
        Element(id=8, type="Button", resource_id="app:id/row", bounds=[0, 50, 100, 90], center=[50, 70]),
    ]
    assert match_step(screen, RouteStep(kind="tap", resource_id="row", index=1)).id == 8


def test_the_yaml_schema_accepts_index():
    flow = parse_flow_yaml(
        "schema_version: 1\nname: r\nsteps:\n  - tap: {text: See all, index: 1}\n", name="r"
    )
    assert flow.steps[0].index == 1


def test_index_survives_a_render_reparse_round_trip():
    """A captured or hand-authored flow must not lose the disambiguation on rewrite."""
    flow = parse_flow_yaml(
        "schema_version: 1\nname: r\nsteps:\n  - tap: {text: See all, index: 2}\n", name="r"
    )
    text = render_flow_yaml(flow)
    assert "index: 2" in text
    assert parse_flow_yaml(text, name="r").steps[0].index == 2


def test_a_step_without_an_index_renders_unchanged():
    """The bare-string shorthand must survive - most steps have no index."""
    flow = parse_flow_yaml("schema_version: 1\nname: r\nsteps:\n  - tap: Send\n", name="r")
    assert "tap: Send" in render_flow_yaml(flow)


@pytest.mark.parametrize("bad", ["two", "-1", "1.5"])
def test_a_nonsense_index_is_refused_at_parse_time(bad):
    with pytest.raises(UsageError, match="index"):
        parse_flow_yaml(
            f"schema_version: 1\nname: r\nsteps:\n  - tap: {{text: See all, index: {bad}}}\n",
            name="r",
        )
