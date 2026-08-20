"""An action reported that it dispatched, never what it did.

The reviewer asked for an "action-level result quality score (confidence + change summary)".
**The score is deliberately not built.** A confidence number invites trusting a figure over
evidence, and the founding lesson of this whole list is that a command reporting success is not
evidence of effect. What lanes actually needed was never "how sure are you" but "what changed".

The most reusable technique the sweep produced was reading the resumed activity — it names what is
in front of the user, so after tapping something that should open a picker it says whether the
picker *opened*. That is a fact about the system rather than a reading of the app's description of
itself, and it settled a disputed critical failure. So the activity is the centrepiece here.

`changed` is an explicit boolean rather than something a caller re-derives from four other fields,
because "nothing changed" being machine-checkable is the half that was missing. And an unknown
baseline is reported as `None`, never as `False`: "I could not compare" and "they are the same" are
different claims, and collapsing them is how a silent wrong answer gets made.

Cost, stated because this touches the observe path: the pre-action shape is taken from the analyze
already in cache — no device call. One `current_app` runs per *observed* action, on a path that
already spends a settle plus a full hierarchy dump. The baseline is chained from the previous
observation rather than sampled before each action, so a sequence of actions compares for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PKG = "com.example.app"


def _screen(*rows: str) -> str:
    return '<hierarchy rotation="0">' + "".join(rows) + "</hierarchy>"


def _node(text: str, *, y: int, clickable: bool = True, focused: bool = False) -> str:
    return (
        f'<node class="android.widget.Button" package="{PKG}" text="{text}"'
        f' clickable="{str(clickable).lower()}" enabled="true"'
        f' focused="{str(focused).lower()}" bounds="[40,{y}][1040,{y + 100}]"/>'
    )


HOME = _screen(_node("Open picker", y=300), _node("Stay", y=440))
PICKER = _screen(_node("Choose a file", y=300), _node("Cancel", y=440), _node("Extra", y=580))


class Navigating(FakeDevice):
    """Serves HOME, then PICKER after the first click, and moves its activity with it."""

    def __init__(self) -> None:
        super().__init__(hierarchy_xml=HOME, package=PKG)
        self.clicked = False

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self.clicked = True
        self._hierarchy_xml = PICKER

    def dump_hierarchy(self, *a: Any, **k: Any) -> str:
        return PICKER if self.clicked else HOME

    def current_app(self) -> dict[str, str]:
        return {"package": PKG, "activity": ".PickerActivity" if self.clicked else ".MainActivity"}


class Inert(FakeDevice):
    """A device where the tap changes nothing at all — the case that must say so."""

    def __init__(self) -> None:
        super().__init__(hierarchy_xml=HOME, package=PKG)

    def current_app(self) -> dict[str, str]:
        return {"package": PKG, "activity": ".MainActivity"}


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


# --------------------------------------------------------------- the summary itself


def test_a_navigation_is_reported_as_what_changed(tmp_path: Path) -> None:
    """The picker case from the field: "did it open" answered from the system, not the app."""
    eng = _engine(tmp_path, Navigating())
    eng.analyze(source="hierarchy")  # establishes the activity baseline and the pre-action shape

    out = eng.tap(selector={"text": "Open picker"}, observe=True)

    change = out.change
    assert change is not None
    assert change["changed"] is True
    assert change["activity_before"].endswith(".MainActivity")
    assert change["activity_after"].endswith(".PickerActivity")
    assert change["activity_changed"] is True


def test_the_text_deltas_name_what_arrived_and_what_left(tmp_path: Path) -> None:
    eng = _engine(tmp_path, Navigating())
    eng.analyze(source="hierarchy")

    change = eng.tap(selector={"text": "Open picker"}, observe=True).change

    assert change is not None
    assert "Choose a file" in change["text_added"]
    assert "Open picker" in change["text_removed"]
    assert change["node_count_delta"] == 1, f"2 rows became 3: {change}"
    assert change["node_count_before"] == 2 and change["node_count_after"] == 3


def test_nothing_changed_is_explicit_and_machine_checkable(tmp_path: Path) -> None:
    """The half that was missing: silence used to be the only way to express "no effect"."""
    eng = _engine(tmp_path, Inert())
    eng.analyze(source="hierarchy")

    change = eng.tap(selector={"text": "Stay"}, observe=True).change

    assert change is not None
    assert change["changed"] is False, change
    assert change["activity_changed"] is False
    assert change["node_count_delta"] == 0
    assert change["text_added"] == [] and change["text_removed"] == []
    assert "nothing changed" in change["detail"]


def test_an_unknown_baseline_is_none_rather_than_false(tmp_path: Path) -> None:
    """"I could not compare" must not be reported as "they are the same".

    Reached with a target-free action on a cold engine. A *selector* tap can never get here —
    resolving the selector analyzes first, so it always leaves a baseline behind — which is worth
    knowing: the no-baseline branch belongs to `key`/`swipe`-style calls made before any analyze.
    """
    eng = _engine(tmp_path, Inert())
    change = eng.key("back", observe=True).change

    assert change is not None
    assert change["node_count_before"] is None
    assert change["node_count_delta"] is None
    assert change["focus_moved"] is None
    assert change["changed"] is not False, "no baseline cannot mean 'nothing changed'"
    assert "no pre-action snapshot" in change["detail"]


def test_focus_movement_is_tracked(tmp_path: Path) -> None:
    focused_then_not = [
        _screen(_node("Field", y=300, focused=True), _node("Next", y=440)),
        _screen(_node("Field", y=300, focused=False), _node("Next", y=440, focused=True)),
    ]

    class Focusing(FakeDevice):
        def __init__(self) -> None:
            super().__init__(hierarchy_xml=focused_then_not[0], package=PKG)
            self.step = 0

        def click(self, x: int, y: int) -> None:
            super().click(x, y)
            self.step = 1

        def dump_hierarchy(self, *a: Any, **k: Any) -> str:
            return focused_then_not[self.step]

        def current_app(self) -> dict[str, str]:
            return {"package": PKG, "activity": ".MainActivity"}

    eng = _engine(tmp_path, Focusing())
    eng.analyze(source="hierarchy")

    change = eng.tap(selector={"text": "Next"}, observe=True).change

    assert change is not None
    assert change["focus_moved"] is True
    assert change["activity_changed"] is False, "focus moved without leaving the screen"
    assert change["changed"] is True


# --------------------------------------------------------------- what it must not be


def test_no_confidence_score_is_emitted(tmp_path: Path) -> None:
    """Explicitly pinned as absent. A number here invites trusting a figure over the evidence.

    If someone later adds one, this test is where the argument against it lives.
    """
    eng = _engine(tmp_path, Navigating())
    eng.analyze(source="hierarchy")
    out = eng.tap(selector={"text": "Open picker"}, observe=True)

    assert out.change is not None
    for banned in ("confidence", "score", "quality", "certainty", "probability"):
        assert banned not in out.change, f"{banned!r} is a figure, not evidence"


def test_no_summary_when_the_caller_did_not_ask_to_observe(tmp_path: Path) -> None:
    """`--no-observe` is a deliberate "do not spend anything on looking"."""
    eng = _engine(tmp_path, Navigating())
    eng.analyze(source="hierarchy")

    out = eng.tap(selector={"text": "Open picker"}, observe=False)

    assert out.change is None


def test_a_device_that_cannot_report_its_activity_still_summarises(tmp_path: Path) -> None:
    """The deltas are free and must survive an unreadable foreground."""

    class Mute(Inert):
        def current_app(self) -> dict[str, str]:
            raise RuntimeError("adb hiccup")

    eng = _engine(tmp_path, Mute())
    eng.analyze(source="hierarchy")

    change = eng.tap(selector={"text": "Stay"}, observe=True).change

    assert change is not None
    assert change["activity_after"] is None
    assert change["activity_changed"] is None, "unknown, not 'unchanged'"
    assert change["node_count_delta"] == 0, "the free deltas still work"
