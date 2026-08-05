"""A flow's `scroll_to` step could not say which way to search, and `scroll_to` only searches one.

Found by a lane at the cost of a flow-validation cycle. `scroll_to` defaults to swiping up, which
means "keep looking further down the list". On a fresh session the tool grid it targeted opens
*already scrolled past* the card — the target was above the fold, not below — so the search moved
away from it and the flow failed live validation despite being correct in shape.

The CLI has had `--direction` all along. The flow step had no `direction:` key, and `_parse_step`
rejects unknown keys, so the step could not express "look upwards" at all. The workaround was an
explicit `swipe: down` before `scroll_to`, which only works by luck of distance.

Why it matters past that one flow: the failure presents as a **missing element**, not as a search
that went the wrong way, so it invites the wrong diagnosis ("the card is gone") instead of the right
one ("I searched away from it"). Any list or grid that can open mid-scroll fails identically.

This is the same class as the `index:` and `--by rid` gaps: a capability that exists on one surface
and not the other. So the vocabulary is deliberately identical to `_swipe_path`'s and to the CLI's
`--direction`, and an unrecognised value is refused at parse time — before a device is touched —
rather than coerced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import check_saveable, parse_flow_yaml, render_flow_yaml
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

HEAD = "schema_version: 1\nname: reach_card\nsteps:\n"


def _one(step_yaml: str):
    return parse_flow_yaml(HEAD + f"  - {step_yaml}\n", name="reach_card").steps[0]


def test_a_scroll_to_step_can_say_look_upwards() -> None:
    """The step the grid needed and could not write."""
    step = _one("scroll_to: {text: Transcribe, direction: down}")
    assert step.kind == "scroll-to"
    assert step.arg == "Transcribe"
    assert step.direction == "down"


def test_the_default_is_unchanged_so_existing_flows_keep_their_behaviour() -> None:
    """Every committed flow omits `direction:`; none of them may start searching the other way."""
    assert _one("scroll_to: Transcribe").direction is None
    assert _one("scroll_to: {text: Transcribe}").direction is None


@pytest.mark.parametrize("way", ["up", "down", "left", "right"])
def test_the_vocabulary_matches_the_swipe_primitive(way: str) -> None:
    """One vocabulary across surfaces, or this becomes the gap it is fixing."""
    assert _one(f"scroll_to: {{text: Card, direction: {way}}}").direction == way


@pytest.mark.parametrize("way", ["upwards", "UPP", "north", "", 1, True])
def test_an_unrecognised_direction_is_refused_at_parse_time(way: object) -> None:
    """Before a device is touched — the `index:` precedent: refuse rather than coerce.

    A silently-ignored `direction: upwards` would search the default way and report the target
    missing, which is precisely the misdiagnosis this step exists to prevent.
    """
    with pytest.raises(UsageError) as err:
        _one(f"scroll_to: {{text: Card, direction: {way!r}}}")
    assert "direction" in str(err.value)


def test_direction_round_trips_through_yaml() -> None:
    """`check_saveable` re-parses its own rendering, so a key that renders away is a silent loss."""
    flow = parse_flow_yaml(
        HEAD + "  - scroll_to: {text: Transcribe, direction: down}\n", name="reach_card"
    )
    rendered = render_flow_yaml(flow)
    assert "direction: down" in rendered
    assert parse_flow_yaml(rendered, name="reach_card").steps[0].direction == "down"
    assert check_saveable(flow) == [], "a directional scroll_to must be saveable"


def test_a_direction_survives_alongside_the_other_step_keys() -> None:
    """Ordinary co-existence: the unknown-key guard must not fire on a legitimate combination."""
    step = _one("scroll_to: {id: cardTranscribe, direction: down, timeout_ms: 4000}")
    assert (step.arg, step.by, step.direction, step.timeout_ms) == (
        "cardTranscribe",
        "id",
        "down",
        4000,
    )


def _drive(tmp_path: Path, step_yaml: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run one step through the real flow executor, capturing the `scroll_to` call it makes."""
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=FakeDevice(package="com.test.app"), factory=ProviderFactory(cfg))
    seen: dict[str, Any] = {}

    def spy(self: Engine, query: str, **kw: Any) -> ActionResult:
        seen.update(kw, query=query)
        return ActionResult(ok=True, action="scroll-to", detail="moved")

    monkeypatch.setattr(Engine, "scroll_to", spy)
    steps = parse_flow_yaml(HEAD + f"  - {step_yaml}\n", name="reach_card").steps
    eng._run_steps(steps, origin_package="com.test.app", allow_destructive=False)
    return seen


def test_the_executor_passes_the_direction_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key that parses and renders but never reaches the device is a trap, not a fix."""
    seen = _drive(tmp_path, "scroll_to: {text: Transcribe, direction: down}", monkeypatch)
    assert seen["query"] == "Transcribe"
    assert seen["direction"] == "down"


def test_the_executor_defaults_to_the_cli_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _drive(tmp_path, "scroll_to: Transcribe", monkeypatch)
    assert seen["direction"] == "up", "an existing flow must behave exactly as before"


def test_direction_is_not_smuggled_onto_steps_that_do_not_take_it() -> None:
    """`swipe`/`scroll` already spell direction as their argument; adding a second spelling there
    would create two ways to say one thing. Other kinds must still reject the key loudly."""
    assert _one("swipe: down").arg == "down"
    assert _one("scroll: down").arg == "down"
    with pytest.raises(UsageError) as err:
        _one("wait_for: {text: Ready, direction: down}")
    assert "unknown keys" in str(err.value)
