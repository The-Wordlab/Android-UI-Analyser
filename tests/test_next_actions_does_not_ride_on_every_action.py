"""A filtered subset of `elements` must not cost more than `elements`.

Measured on one real journalled response
(`~/.cache/android-ui-analyser/journal/<serial>.details.jsonl`, 2026-08-22):

    elements       1301 B   325 tok   21 rows
    next_actions   1384 B   346 tok   12 rows     <- 25% of the whole response

Three separate faults sat in that one 12-row list:

* **Default-flag leak.** 12 of 12 rows carried `"checked": false, "selected": false`.
  :func:`schema.drop_default_flags` is the single rule for "a flag at its default says
  nothing", it was applied to `elements`, and it was never applied to the derived list — so
  the trimming work that shrank `elements` left its own copy of the same waste behind.
* **Silently lossy.** 15 elements were `clickable` and the cap was 12, so three real controls
  were missing from a list an agent reads as "what I can do here". A shorter list that looks
  authoritative is worse than a duplicated one.
* **Its headline justification was absent.** `avg_ms` is what made a row worth more than the
  element it copied — "tap this next, and it takes ~4.8s" in one read. Zero rows carried it:
  cost is learned per (screen, control), so on a fresh screen, which is exactly when guidance
  matters most, the column is empty.

And the scan it removed no longer exists. `next_actions` was written when the observation was a
~50-node dump; it is now trimmed to ~20 rows with `clickable` on each one, so filtering
`observation.elements` *is* the read the list was standing in for.

So: off by default behind `output.next_actions`, complete rather than capped when asked for,
default-trimmed by the same rule as `elements` — and the one thing `elements` could not express,
the learned per-control cost, now rides on the element it belongs to as `cost`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.identity import stable_key
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PACKAGE = "com.example.fiction"

# 15 clickable controls under one header — the shape of the measured screen, where a cap of 12
# hid three of them. A `Switch` and a selected tab are in there so the flag columns have
# something real to say on the rows that own them.
_ROWS = [
    ("android.widget.Button", f"Card {index}", f"{PACKAGE}:id/card_{index}")
    for index in range(1, 14)
]


def _node(cls: str, text: str, rid: str, *, top: int, extra: str = "") -> str:
    return (
        f'<node class="{cls}" package="{PACKAGE}" text="{text}" resource-id="{rid}" '
        f'clickable="true" enabled="true" {extra} bounds="[40,{top}][1040,{top + 90}]"/>'
    )


HIERARCHY = (
    '<hierarchy rotation="0">'
    f'<node class="android.widget.TextView" package="{PACKAGE}" text="Catalog" '
    f'resource-id="{PACKAGE}:id/header" clickable="false" enabled="true" '
    'bounds="[40,200][1040,290]"/>'
    + "".join(
        _node(cls, text, rid, top=300 + 100 * offset)
        for offset, (cls, text, rid) in enumerate(_ROWS)
    )
    + _node(
        "android.widget.Switch",
        "Notifications",
        f"{PACKAGE}:id/switchNotifications",
        top=1700,
        extra='checkable="true" checked="false"',
    )
    + _node(
        "android.widget.Button",
        "Browse",
        f"{PACKAGE}:id/tabBrowse",
        top=1800,
        extra='selected="true"',
    )
    + "</hierarchy>"
)


def _engine(tmp_path: Path, serial: str, **output: Any) -> Engine:
    overrides: dict[str, Any] = {
        "memory": {"dir": str(tmp_path / "home")},
        "daemon": {"enabled": False},
    }
    if output:
        overrides["output"] = output
    cfg = make_config(**overrides)
    device = FakeDevice(hierarchy_xml=HIERARCHY, package=PACKAGE, serial=serial)
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _acted(engine: Engine) -> Any:
    first = engine.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "") == "Card 1")
    return engine.tap(target, observe=True)


# ------------------------------------------------------------------ 1. it is not on by default


def test_an_observed_action_does_not_pay_for_next_actions(tmp_path: Path) -> None:
    """25% of a response, for a filtered copy of the list printed beside it."""
    result = _acted(_engine(tmp_path, "emu-na-default"))

    assert result.next_actions is None, (
        "the default action response must not carry a filtered duplicate of its own elements"
    )
    assert "next_actions" not in json.loads(result.render("json"))


def test_the_default_observation_still_says_what_can_be_acted_on(tmp_path: Path) -> None:
    """The replacement has to be present, or removal is just a loss.

    `clickable` rides on every row of the folded observation, so the filter that replaces
    `next_actions` is a comprehension over the list the caller already has.
    """
    result = _acted(_engine(tmp_path, "emu-na-replacement"))

    observation = result.observation
    assert observation is not None
    actionable = [e for e in observation.elements if e.clickable]
    assert len(actionable) >= 15, "the fixture must have more controls than the old cap"


# ---------------------------------------------------- 2. the learned cost rides on the element


def test_a_learned_cost_reaches_the_caller_on_the_element_it_belongs_to(tmp_path: Path) -> None:
    """`avg_ms` was the one field `elements` could not express. It is now an element field.

    Note where the cost is *not*: `meta.slow_controls` carries the same numbers, and it is
    absent from the `changed` meta preset that every folded observation is trimmed to — so
    before this, the only route from memory to an acting agent was `next_actions[].avg_ms`.
    """
    from android_ui_analyser import hierarchy

    engine = _engine(tmp_path, "emu-na-cost")
    assert engine._memory is not None
    parsed = hierarchy.parse_hierarchy(HIERARCHY, (1080, 2400))
    engine._memory.record_screen(package=PACKAGE, elements=parsed, name_hint="catalog")
    slow_key = next(e.stable_key for e in parsed if (e.text or "") == "Card 9")
    assert slow_key

    first = engine.analyze(source="hierarchy")
    screen = first.meta.known_screen
    assert screen, "a per-(screen, control) timing needs a recognised screen to live on"
    engine._memory.record_action_timing(
        PACKAGE, screen=screen, control=slow_key, ms=4800.0, outcome="changed"
    )

    target = next(e.id for e in first.elements if (e.text or "") == "Card 1")
    result = engine.tap(target, observe=True)

    observation = result.observation
    assert observation is not None
    priced = {
        stable_key(element): element.cost
        for element in observation.elements
        if element.cost is not None
    }
    assert priced.get(slow_key) == {"avg_ms": 4800, "max_ms": 4800, "n": 1}, (
        f"the learned cost must ride on its own control; priced rows were {priced}"
    )
    untouched = next(e.stable_key for e in parsed if (e.text or "") == "Card 5")
    assert untouched not in priced, "a control with no history must stay unpriced"
    # `rid:card_1` is priced too, and correctly: the tap under test measured itself.
    assert set(priced) == {slow_key, "rid:card_1"}


def test_the_learned_cost_survives_the_default_observation_view(tmp_path: Path) -> None:
    """A field the default projection drops has not reached the caller at all."""
    from android_ui_analyser.config import OutputCfg
    from android_ui_analyser.projection import FIELD_ALIASES, Projection

    assert "cost" in FIELD_ALIASES, "`--fields cost` must name the column"
    view = Projection.for_observation(OutputCfg().observation_fields, meta="changed")
    assert view is not None
    assert "cost" in view.columns(), (
        "the learned cost is only preserved if the default folded observation keeps the column"
    )


# ----------------------------------------------------- 3. the opt-in list is honest when asked


def test_the_opt_in_list_leaks_no_default_flags(tmp_path: Path) -> None:
    """12 of 12 rows said `checked: false, selected: false`, and meant nothing by it."""
    result = _acted(_engine(tmp_path, "emu-na-flags", next_actions=True))
    rows = result.next_actions or []
    assert rows, "the opt-in must actually emit the list"

    for row in rows:
        if not row.get("checkable"):
            assert "checked" not in row, f"a non-checkable row still reports checked: {row}"
        assert row.get("selected") is not False, f"`selected: false` says nothing: {row}"

    switches = [row for row in rows if row.get("checkable")]
    assert switches and all("checked" in row for row in switches), (
        "an off switch keeps `checked: false` — that one IS the reading, not a default"
    )
    assert [row for row in rows if row.get("selected") is True], (
        "and a genuinely selected row keeps its flag"
    )


def test_the_opt_in_list_names_every_control_it_can_act_on(tmp_path: Path) -> None:
    """The cap hid three real controls from a list that reads as authoritative."""
    result = _acted(_engine(tmp_path, "emu-na-complete", next_actions=True))
    observation = result.observation
    assert observation is not None
    listed = {row["id"] for row in result.next_actions or []}
    actionable = {
        stable_key(element)
        for element in observation.elements
        if element.clickable or element.checkable or element.long_clickable or element.scrollable
    }

    assert len(actionable) > 12, "the fixture must exceed the cap that used to truncate"
    assert actionable - listed == set(), (
        f"a list of what can be done here silently omitted {sorted(actionable - listed)}"
    )


# -------------------------------------------------------------- 4. the guidance is not a lie


def _string_constants(code: Any) -> list[str]:
    """Every string literal in *code*, nested comprehensions and closures included."""
    out: list[str] = []
    for const in code.co_consts:
        if isinstance(const, str):
            out.append(const)
        elif hasattr(const, "co_consts"):
            out += _string_constants(const)
    return out


def test_no_guidance_points_at_a_field_the_default_response_omits() -> None:
    """Guidance naming an absent field costs the agent a call to discover it is absent.

    `_phase_recommended_call`'s `manual_observation` fallback used to answer "inspect this
    result's next_actions and choose deliberately" — a call the caller cannot make once the
    field is not there.
    """
    guidance = " ".join(_string_constants(Engine._phase_recommended_call.__code__))
    assert "next_actions" not in guidance, (
        "the session planner still tells a caller to read a field the response omits"
    )
    assert "observation.elements" in guidance, "and it must name the list that replaced it"


def test_the_skill_teaches_the_filter_that_replaced_the_list() -> None:
    from android_ui_analyser import guide

    skill = guide.render_skill()
    assert "next_actions" not in skill
    assert "clickable" in skill, "the skill must teach the filter that replaces it"

    manual = guide.render_markdown(brief=False)
    assert "Continue through the returned `next_actions`" not in manual, (
        "the manual instructed agents to walk a list the default response no longer emits"
    )
