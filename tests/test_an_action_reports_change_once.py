"""An action answered "what changed" three times, and the dearest answer was the least usable.

Measured on one real tap: the response was 921 tokens, and 51% of it was change-reporting
split across three overlapping blocks — `meta.element_diff` (318 tokens), `change` (138), and
`action_diff_summary` (17). One question, three answers, at 473 tokens.

`element_diff` was the expensive one and the weakest. Its content on that tap:

    added      3 entries    12 tokens
    removed   40 entries   289 tokens   <- 91% of the block
    changed    0 entries     0 tokens

``removed`` is 91% of the cost and names elements that are **gone from the screen** — nothing
a caller can tap, read or assert on. And all three ``added`` ids were already present in
``elements`` with full detail, so that part is a pointer to data printed directly below it.
The one slice carrying something nothing else says — ``changed``, with its text from/to — was
empty, and is the smallest.

So it leaves the observation's default `meta`. What survives answers the same question for a
tenth of the price and is *actionable*: `unchanged` is the one-bit "did my tap do anything",
`action_diff_summary` carries the counts, and `change` carries `activity_changed` plus the
added/removed **text**, which reads without a second lookup.

Two things had to be checked before removing it, and this file pins both, because each is a
way the removal could have quietly broken something else:

* `action_diff_summary` is *derived* from `element_diff` — but in the engine, upstream of the
  projection that trims `meta`. Trimming the presented block therefore cannot empty the
  summary.
* `OutputFormat.delta` exists to emit "compact plus the diff", and keeps its own whitelist.
  A format whose entire purpose is the diff must not lose it because an action stopped
  showing it.
"""

from __future__ import annotations

import json

from android_ui_analyser.config import Config
from android_ui_analyser.projection import OBSERVATION_META_PRESETS, Projection
from android_ui_analyser.schema import (
    AnalyzeResult,
    Element,
    Meta,
    OutputFormat,
    Screen,
)

PACKAGE = "com.example.fiction"

# The shape that made this expensive: a few things arrived, a great many left.
_DIFF = {
    "added": ["tx:1111111111"],
    "removed": [f"rid:row#{n}" for n in range(1, 21)],
    "changed": [{"id": "tx:1111111111", "text": {"from": "Loading", "to": "Ready"}}],
    "prev_count": 26,
    "curr_count": 7,
}


def _screen() -> AnalyzeResult:
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[
            Element(
                id=1,
                type="Button",
                text="Ready",
                bounds=[0, 400, 1080, 520],
                center=[540, 460],
                clickable=True,
                enabled=True,
                window="app",
            )
        ],
        meta=Meta(
            duration_ms=10,
            tier_used="hierarchy",
            path="hierarchy",
            element_diff=dict(_DIFF),
            unchanged=False,
        ),
    )


def _observation(**kw: object) -> dict:
    view = Projection.for_observation(
        kw.pop("fields", Config().output.observation_fields),
        meta=kw.pop("meta", Config().output.observation_meta),
    )
    assert view is not None
    return view.apply(_screen().as_dict(OutputFormat.json))


# ------------------------------------------------------------------ out of the default meta


def test_the_default_observation_meta_omits_the_element_diff() -> None:
    assert "element_diff" not in _observation()["meta"]


def test_the_preset_does_not_list_it() -> None:
    assert "element_diff" not in OBSERVATION_META_PRESETS["changed"]


def test_the_one_bit_answer_survives() -> None:
    """`unchanged` is what a caller actually branches on, and it costs nothing."""
    assert "unchanged" in OBSERVATION_META_PRESETS["changed"]


def test_asking_for_it_returns_it() -> None:
    """"Unless the agent asks" is the whole contract, so asking has to work."""
    both = _observation(meta="all")["meta"]
    assert both["element_diff"]["removed"] == _DIFF["removed"]

    just_one = _observation(meta="element_diff")["meta"]
    assert set(just_one) == {"element_diff"}


def test_removing_it_is_a_real_saving() -> None:
    default = len(json.dumps(_observation(), separators=(",", ":")))
    with_diff = len(json.dumps(_observation(meta="all"), separators=(",", ":")))
    assert default < with_diff / 2, f"no material saving: {default}B vs {with_diff}B"


# --------------------------------------------------- the two things that must not break


def test_the_action_summary_is_still_computed_from_it() -> None:
    """The counts come from the engine, upstream of the trim — so they cannot be trimmed away."""
    from android_ui_analyser.engine import Engine

    summary = Engine._compact_action_diff(dict(_DIFF))

    assert summary is not None
    assert summary["added"] == 1 and summary["removed"] == 20
    assert summary["prev_count"] == 26 and summary["curr_count"] == 7


def test_the_delta_format_still_carries_the_diff() -> None:
    """`delta` exists to emit the diff; it must not lose it because actions stopped showing it."""
    rendered = json.loads(_screen().render(OutputFormat.delta))

    assert "element_diff" in rendered["meta"]


def test_the_readable_change_block_is_untouched() -> None:
    """`change.text_added`/`text_removed` is the channel that replaces the id lists.

    It sits at the top level of an action response rather than inside the observation's
    `meta`, so no `meta` preset can trim it — which is what makes it safe to drop the id-level
    diff. It is also the more useful half: "Search Settings" and "Network & internet …" left
    the screen says what happened; `rid:row#7` left the screen does not.
    """
    from android_ui_analyser.schema import ActionResult

    fields = set(ActionResult.model_fields)
    assert "change" in fields, "the readable change block must stay a first-class field"
    assert "element_diff" not in fields, "the id-level diff belongs to the observation's meta"
    assert "action_diff_summary" in fields, "the counts stay, at a tenth of the cost"
