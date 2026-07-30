"""An action's inline observation must not be a half-drawn screen.

`_await_post_action_ready` had a fast path: if the hierarchy differed enough from the
PRE-action tree, return on the first sample (`via=hierarchy-fast`). But a big delta says we
*left* the old screen — not that the new one finished rendering. A header-only frame differs
from the screen we came from maximally, so it cleared the threshold on sight and the caller got
an observation with the list body missing.

Measured consequence: one navigation run in four read **zero** rows off a screen that has five,
because `analyze` ran against the header-only frame. Every action observed on device reported
`via=hierarchy-fast`, including `input` — this was the normal path, and it had no test.
"""

from __future__ import annotations

import io
from functools import cache

from PIL import Image, ImageDraw

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import ScreenImage
from conftest import FakeDevice, make_config
from test_memory import _hier, _node

PKG = "com.example.app"


@cache
def _frame(rows_drawn: int, *, w: int = 320, h: int = 640) -> bytes:
    """A list screen with *rows_drawn* rows painted — the rest of the screen is static.

    Realistic on purpose: a progressive render repaints only the list band. A fixture whose
    every pixel changes gets ALL grid cells masked as animation, and an all-masked grid reports
    idle vacuously — which would make this test pass for the wrong reason.

    Cached and rectangle-drawn because the settle loop is on a real clock: a slow frame factory
    spends the caller's deadline and the wait times out on render cost that is the test's own.
    """
    img = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, w - 20, 70), fill=(30, 60, 120))  # static header band
    for i in range(rows_drawn):
        top = 120 + i * 70
        draw.rectangle((20, top, w - 20, min(top + 50, h)), fill=(60, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rows(n: int) -> str:
    """A list screen with *n* rows — the body arrives progressively on a real device."""
    return _hier(
        _node("android.widget.TextView", text="App language", rid="x:id/title", b="[40,120][1040,210]"),
        *(
            _node(
                "android.widget.TextView",
                text=f"Language {i}",
                rid=f"x:id/lang{i}",
                clk=True,
                b=f"[40,{300 + i * 140}][1040,{400 + i * 140}]",
            )
            for i in range(n)
        ),
    )


PREVIOUS_SCREEN = _hier(
    *(
        _node(
            "android.widget.TextView",
            text=f"Setting {i}",
            rid=f"x:id/row{i}",
            clk=True,
            b=f"[40,{200 + i * 140}][1040,{300 + i * 140}]",
        )
        for i in range(8)
    )
)


class RenderingDevice(FakeDevice):
    """A screen painting its list over *paint_frames* frames, pixels and tree in step.

    Progress is driven by frames served, not wall-clock, so the fixture is deterministic under
    load. Coupling the two is the point: on a real device unpainted rows ARE pixel motion, so
    "pixels idle while the tree is still partial" is not a state that can occur — a fixture that
    allows it tests an impossible case and flakes.
    """

    def __init__(self, *, paint_frames: int = 0, rows: int = 5, **kw: object) -> None:
        super().__init__(hierarchy_xml=_rows(rows), **kw)  # type: ignore[arg-type]
        self._paint_frames = paint_frames
        self._rows = rows
        self._frames_served = 0

    def _drawn(self) -> int:
        """Rows painted so far — one per frame, so every frame differs until it is done.

        Quantising several frames onto the same row count made consecutive frames identical,
        which reads as pixel-idle: the fixture would then be modelling a render that visually
        stalls mid-paint and resumes, which is not a thing a device does.
        """
        if self._paint_frames <= 0:
            return self._rows
        return min(self._rows, max(0, self._frames_served - 1))

    def screenshot(self) -> ScreenImage:  # type: ignore[override]
        self.screenshot_calls += 1
        self._frames_served += 1
        return ScreenImage(_frame(self._drawn()), width=self._w, height=self._h)

    def dump_hierarchy(self, compressed: bool = False) -> str:  # type: ignore[override]
        self.hierarchy_calls += 1
        self.last_tree_served = _rows(self._rows if self._drawn() >= self._rows else 0)
        return self.last_tree_served


def _engine(device: FakeDevice) -> Engine:
    return Engine(make_config(daemon={"enabled": False}), device=device)


def _await(eng: Engine, pre_tree: tuple[str, ...]) -> dict:
    eng._pre_action_tree_fp = pre_tree
    eng._pre_action_sig = None
    # Production poll interval on purpose: a tighter one burns the whole screenshot stream
    # before the first hierarchy sample, so the fixture would read as "already settled".
    return eng._await_post_action_ready(settle_ms=45, total_timeout_ms=1100)


def _tree_of(xml: str, eng: Engine) -> tuple[str, ...]:
    from android_ui_analyser import hierarchy as hierarchy_mod

    els = hierarchy_mod.parse_hierarchy(xml, (320, 640))
    parts = [
        f"{(e.resource_id or '').split('/')[-1]}:{(e.text or e.content_desc or '')[:40]}"
        for e in els
        if getattr(e, "window", None) != "system" and (e.resource_id or e.text or e.content_desc)
    ]
    return tuple(parts[:60])


def test_the_wait_ends_on_the_full_tree_not_the_partial_one() -> None:
    """The regression, stated as what the caller cares about: are the rows there?

    Old behaviour returned on the first hierarchy sample — a header with no rows — because it
    differed from the screen we had left.
    """
    dev = RenderingDevice(paint_frames=6, package=PKG, width=320, height=640)
    eng = _engine(dev)
    ready = _await(eng, _tree_of(PREVIOUS_SCREEN, eng))
    assert ready["timeout"] is False, f"must not fall back to the deadline: {ready}"
    assert "Language 4" in dev.last_tree_served, "the wait ended on a tree with no rows"
    assert dev.hierarchy_calls >= 2, "a painting screen must cost a confirming dump"


def test_a_finished_screen_still_takes_the_fast_path() -> None:
    """Speed must survive: an already-drawn screen is accepted on the first sample."""
    dev = RenderingDevice(paint_frames=0, package=PKG, width=320, height=640)
    eng = _engine(dev)
    ready = _await(eng, _tree_of(PREVIOUS_SCREEN, eng))
    assert ready["via"] == "hierarchy-fast", ready
    assert ready["timeout"] is False
    assert dev.hierarchy_calls == 1, "a drawn screen must not pay a confirming dump"
