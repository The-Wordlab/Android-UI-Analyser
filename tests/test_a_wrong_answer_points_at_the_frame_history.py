"""When something is wrong, name the one artefact that shows what happened in between.

The rolling deduped screencap buffer (`capture_sidecar.py`) is the only thing AUA keeps that
can show an interstitial sliding in, a dialog appearing, or a screen replaced twice —
`meta.capture_hint` names the exact command (`aua capture last --since last-action`). The
payload-trimming work dropped it from the `changed` observation-meta preset because on a
settled, successful action it was pure cost: measured on one real settings screen the full
`meta` block was 299 of the observation's 919 tokens, and `research_tasks`,
`suggested_deeplinks` and `capture_hint` alone accounted for over half of what was left.

Trimming by "is this usually empty" removed a field whose whole value is that it is usually
empty. So it comes back **only where something is wrong**:

* on the miss payloads that now carry the screen they already read
  (`errors._CarriesObservation` — `element_not_found`, `stale_element_id`), where "what
  happened in between" is the question the caller is actually asking; and
* on an arrival that did not settle, where `stale_risk` says the folded observation may not
  describe the action's effect.

Not on a healthy action. The top-level `ActionResult.capture_hint` was in fact still being
attached to *every* observed action whenever the buffer was live — a fixed per-action cost of
90 B / ~22 tok for a pointer to a buffer nothing was wrong with — so the gate is applied
there rather than duplicating the key into `meta` and paying twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ElementNotFoundError
from android_ui_analyser.schema import ActionResult, AnalyzeResult, Element, Meta, Screen
from conftest import FakeDevice, make_config

PACKAGE = "com.example.fiction"

_SCREEN = f"""<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.Button" text="Continue"
        resource-id="{PACKAGE}:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,400][1040,520]"/>
</hierarchy>"""

_HINT = "recent pixel change after last action — `aua capture last --since last-action`"


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    return Engine(make_config(cache={"dir": str(tmp_path)}), device=device)


def _observation(*, empty: bool = False) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=(
            []
            if empty
            else [
                Element(
                    id=1,
                    type="android.widget.Button",
                    text="Continue",
                    bounds=[40, 400, 1040, 520],
                    center=[540, 460],
                    clickable=True,
                )
            ]
        ),
        meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy"),
    )


# ------------------------------------------------------------------- the gate itself


def test_a_settled_action_raises_no_question_the_buffer_could_answer() -> None:
    """The case that got `capture_hint` trimmed: nothing is wrong, so nothing is offered."""
    healthy = ActionResult(ok=True, action="tap", observation=_observation())

    assert Engine._frame_history_matters(healthy) is False


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            ActionResult(
                ok=True,
                action="tap",
                observation=_observation(),
                stale_risk="post-action wait saw no confirmed screen change (via=unchanged)",
            ),
            id="the arrival never settled",
        ),
        pytest.param(
            ActionResult(
                ok=True, action="tap", observation=_observation(), settled_unmet=True
            ),
            id="the wait hit its ceiling with the screen still moving",
        ),
        pytest.param(
            ActionResult(ok=True, action="tap", observation=_observation(empty=True)),
            id="the screen came back empty",
        ),
        pytest.param(
            ActionResult(ok=False, action="tap", observation=_observation()),
            id="the action failed",
        ),
    ],
)
def test_a_verdict_the_frames_could_explain_names_them(result: ActionResult) -> None:
    assert Engine._frame_history_matters(result) is True


# --------------------------------------------------------------- through a real action


def _with_a_live_buffer(engine: Engine) -> None:
    """Pretend the rolling buffer is running and has seen a post-action pixel change.

    The buffer itself is exercised in `test_capture.py`; what is under test here is *when*
    its pointer is attached, so the readiness answer is pinned rather than raced.
    """
    engine._capture_hint = lambda: _HINT  # type: ignore[method-assign]


def test_a_healthy_action_does_not_name_the_frame_buffer(tmp_path: Path) -> None:
    """0 bytes on the path this whole trimming series exists to protect."""
    engine = _engine(tmp_path, FakeDevice(hierarchy_xml=_SCREEN, package=PACKAGE))
    _with_a_live_buffer(engine)

    result = engine.tap(selector={"key": "rid:continue_btn"}, observe=False)

    assert result.stale_risk is None, "this fixture must be the healthy case"
    assert result.capture_hint is None, result.capture_hint


def test_an_arrival_that_did_not_settle_names_the_frame_buffer(tmp_path: Path) -> None:
    """`stale_risk` means the folded screen may predate the action — the frames say which."""
    engine = _engine(tmp_path, FakeDevice(hierarchy_xml=_SCREEN, package=PACKAGE))
    _with_a_live_buffer(engine)

    result = engine.tap(selector={"key": "rid:continue_btn"})

    assert result.stale_risk, "this fixture must be the unsettled case"
    assert result.capture_hint == _HINT


def test_a_key_miss_carries_the_frame_history_with_the_screen_it_read(tmp_path: Path) -> None:
    """A miss has no `ActionResult` to hang a hint on, so it rides in the attached meta.

    The attached screen is trimmed with the same dials a successful action uses, and that
    budget is what dropped `capture_hint`. On a miss it is the answer, not the cost.
    """
    engine = _engine(tmp_path, FakeDevice(hierarchy_xml=_SCREEN, package=PACKAGE))
    _with_a_live_buffer(engine)

    with pytest.raises(ElementNotFoundError) as caught:
        engine.tap(selector={"key": "rid:not_on_this_screen"})

    error = caught.value.to_dict()["error"]
    assert isinstance(error, dict)
    assert error["observation_present"] is True
    assert error["observation"]["meta"]["capture_hint"] == _HINT


def test_the_daemon_boundary_does_not_drop_the_miss_hint() -> None:
    """The rebuild reconstructs the error from a dict; the nested meta has to survive it.

    The attached observation itself was dropped here three times before it stuck, so the
    field riding inside it gets its own pin rather than an assumption.
    """
    from android_ui_analyser.cli import _daemon_error

    rebuilt = _daemon_error(
        {
            "code": "element_not_found",
            "message": "no element with stable_key 'rid:gone' on the current screen for tap",
            "hint": "use an id from the attached observation",
            "observation_present": True,
            "observation": {
                "screen": {"width": 1080, "height": 2400, "package": PACKAGE},
                "elements": [],
                "meta": {"capture_hint": _HINT},
            },
        }
    )

    error = rebuilt.to_dict()["error"]
    assert isinstance(error, dict)
    assert json.dumps(error, ensure_ascii=False).count(_HINT) == 1
    assert error["observation"]["meta"]["capture_hint"] == _HINT
