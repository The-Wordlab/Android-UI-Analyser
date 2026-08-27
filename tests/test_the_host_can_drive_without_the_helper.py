"""The lane that works everywhere: the same rule, driven from the host.

Why this exists at all. The scoring rule scored 82.2% on held-out real screens and lived only inside
the helper APK — and Android will not bind a sideloaded accessibility service unless adbd can run as
root, which rules out every retail phone and every Play-image emulator. On such a device the only
other autopilot, ``session autopilot``, needs a local policy model that measured 17.5% correct node
selection against this rule's 82.2%. So the good rule was unreachable on the most ordinary targets
there are, and the reachable one was the weak one.

These tests pin the three things that make the host lane trustworthy rather than merely present:
it calls the *same* ``decide``, it keeps per-node progress across re-numbering, and it never reaches
for adb.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

from android_ui_analyser.drive_projection import project
from android_ui_analyser.drive_rule import decide
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen
from conftest import make_config


def _element(index: int, text: str, *, clickable: bool = True, key: str | None = None) -> Element:
    return Element(
        id=index,
        type="TextView",
        text=text,
        bounds=(0, index * 100, 1080, index * 100 + 90),
        center=(540, index * 100 + 45),
        clickable=clickable,
        stable_key=key or f"tx:{text.lower().replace(' ', '')}",
    )


def _observation(*elements: Element) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, source="hierarchy", package="com.example.app"),
        elements=list(elements),
        meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy"),
    )


class _Screens:
    """A device that changes screen when the right row is tapped, and not otherwise.

    Deliberately renumbers element ids on every read. Real ``analyze`` does — the id is a frame-local
    ordinal — and a loop that keyed progress by position would silently lose track of what it had
    already tried, which is the one thing ``no_progress`` depends on.
    """

    def __init__(self, *, moves_on: str, dead: bool = False) -> None:
        self.moves_on = moves_on
        self.dead = dead
        self.taps: list[int] = []
        self.scrolls = 0
        self.arrived = False
        self.reads = 0

    def analyze(self, **_kwargs: Any) -> AnalyzeResult:
        self.reads += 1
        offset = self.reads * 10  # ids churn between frames, exactly as the real thing does
        if self.arrived:
            return _observation(
                _element(offset + 1, "Destination reached"),
                _element(offset + 2, "Something else"),
            )
        return _observation(
            _element(offset + 1, "Unrelated row"),
            _element(offset + 2, self.moves_on),
            _element(offset + 3, "Another unrelated row"),
        )

    def tap(self, element_id: int, **_kwargs: Any) -> Any:
        self.taps.append(element_id)
        if not self.dead and (element_id % 10) == 2:
            self.arrived = True
        return type("R", (), {"ok": True})()

    def scroll(self, *_args: Any, **_kwargs: Any) -> Any:
        self.scrolls += 1
        return type("R", (), {"ok": False})()


def _engine(tmp_path: Any, screens: _Screens) -> Engine:
    cfg = make_config(cache={"dir": str(tmp_path / "cache")})
    engine = Engine.__new__(Engine)
    engine.config = cfg  # type: ignore[misc]
    engine.analyze = screens.analyze  # type: ignore[method-assign]
    engine.tap = screens.tap  # type: ignore[method-assign]
    engine.scroll = screens.scroll  # type: ignore[method-assign]
    return engine


# --------------------------------------------------------------------------- it works with no helper


def test_it_reaches_a_goal_with_no_helper_and_no_root(tmp_path: Any) -> None:
    screens = _Screens(moves_on="Widgets")
    got = Engine.drive_on_host(_engine(tmp_path, screens), "go to widgets", budget=4)

    assert got["ran_on"] == "host"
    assert screens.taps, "nothing was ever tapped"
    first = next(s for s in got["steps"] if s["decision"] == "tap")
    assert first["label"] == "Widgets"
    assert first["outcome"] == "changed"


def test_a_goal_needing_the_host_is_refused_before_anything_is_tapped(tmp_path: Any) -> None:
    """The rule checks host vocabulary before scoring, so no budget is spent proving the obvious."""

    screens = _Screens(moves_on="Widgets")
    got = Engine.drive_on_host(_engine(tmp_path, screens), "take a screenshot", budget=4)

    assert got["stop_reason"] == "handoff"
    assert got["steps"][0]["reason"] == "needs_host"
    assert screens.taps == [], "it acted on a screen that could never satisfy the goal"


def test_an_empty_goal_is_a_usage_error_not_a_wasted_run(tmp_path: Any) -> None:
    screens = _Screens(moves_on="Widgets")
    with pytest.raises(UsageError):
        Engine.drive_on_host(_engine(tmp_path, screens), "   ")
    assert screens.reads == 0


# --------------------------------------------------------------------------- progress survives churn


def test_it_stops_for_no_progress_instead_of_tapping_the_same_row_forever(tmp_path: Any) -> None:
    """The loop this closes. A tap that changes nothing, twice, is not worth a third.

    ``dead=True`` makes every tap land and move nothing, which is precisely the shape that spent an
    entire budget re-pressing one row before the rule learned to read its own outcomes.
    """

    screens = _Screens(moves_on="Widgets", dead=True)
    got = Engine.drive_on_host(_engine(tmp_path, screens), "go to widgets", budget=6)

    assert got["stop_reason"] == "handoff"
    assert got["steps"][-1]["reason"] == "no_progress"
    # One tap to learn it does nothing, then a refusal — not six.
    assert len(screens.taps) <= 2, f"tapped {len(screens.taps)} times before giving up"


def test_progress_is_keyed_by_stable_key_not_by_position(tmp_path: Any) -> None:
    """Element ids renumber on every analyze; the fake above renumbers deliberately.

    Keyed by position, the second read would look like a fresh screen and ``no_progress`` could never
    fire. This is the assertion that would have caught that.
    """

    screens = _Screens(moves_on="Widgets", dead=True)
    got = Engine.drive_on_host(_engine(tmp_path, screens), "go to widgets", budget=6)

    tapped = {i % 10 for i in screens.taps}
    assert tapped == {2}, "it tapped a different row each frame — ids were treated as identity"
    assert got["steps"][-1]["reason"] == "no_progress"


def test_the_projection_carries_progress_or_the_stall_branch_is_dead() -> None:
    """Found while wiring this: ``project`` rebuilt nodes and dropped ``tried``/``last``.

    ``decide`` reads both off the winning node, so silently dropping them left its ``no_progress``
    branch unreachable for every caller that projects — which is every real one. The bug was
    invisible because the corpus builds its nodes directly and never goes through ``project``.
    """

    nodes = [
        {"text": "Widgets", "clickable": True, "id": 7, "tried": 2, "last": "unchanged"},
        {"text": "Other", "clickable": True, "id": 8},
    ]
    projection = project(nodes)
    widget = next(n for n in projection["nodes"] if n.get("text") == "Widgets")
    assert widget.get("tried") == 2, "progress was dropped by the projection"
    assert widget.get("last") == "unchanged"

    got = decide("go to widgets", projection)
    assert got["call"] == "handoff" and got["reason"] == "no_progress"


# --------------------------------------------------------------------------- one rule, two lanes


def test_both_lanes_call_the_same_decide() -> None:
    """The device lane and the host lane must not be able to disagree about what to do.

    They can differ in speed — a round trip per step against none — and in nothing else. Two copies
    of the rule would drift, and the whole argument for the host lane is that it is the *same* rule
    reaching devices the helper cannot.
    """

    source = inspect.getsource(Engine.drive_on_host)
    assert "decide(" in source, "the host lane does not call the shared rule"
    assert "from .drive_rule import decide" in source


def test_the_host_lane_never_reaches_for_adb() -> None:
    """It goes through `analyze`/`tap`/`scroll`, which the selected adapter owns."""

    # The docstring is stripped first: it *explains* the adb-root constraint that makes this lane
    # necessary, and a scan that cannot tell prose from a call reports its own explanation as a
    # violation.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Engine.drive_on_host)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:] if ast.get_docstring(function) else function.body
    code = "\n".join(ast.unparse(node) for node in body)

    for banned in ("adb", "uiautomator", "run-as", "dumpsys", "subprocess", "shell"):
        assert banned not in code, f"{banned!r} bypasses the platform boundary"
    # And it does reach the device, through the seams the selected adapter owns.
    for seam in ("self.analyze()", "self.tap(", "self.scroll("):
        assert seam in code, f"{seam} missing — the loop does not drive anything"
