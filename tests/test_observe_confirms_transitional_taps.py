"""A tap observation must not hand an agent a pager frame that invites a second tap.

The Luna Apps Hub run caught a concrete unsafe sequence: tapping ``Explore`` moved the
current page sideways, AUA returned while the old labels were still present plus one transient
unlabelled node, and the agent tapped ``Explore`` again.  Dispatch succeeded both times; the
bad evidence was the first post-action observation.

Fast hierarchy/pixel settles get one longer stability confirmation before AUA reads the screen.
That rule is deliberately content-independent: the live UI also produced OCR-over-old-hierarchy
and mixed-old/new-hierarchy frames that looked semantically complete but were not stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

PKG = "com.example.app"


def _screen(*rows: str) -> str:
    return '<hierarchy rotation="0">' + "".join(rows) + "</hierarchy>"


def _node(
    text: str = "",
    *,
    rid: str = "",
    y: int,
    clickable: bool = False,
) -> str:
    return (
        f'<node class="android.view.View" package="{PKG}" text="{text}"'
        f' resource-id="{rid}" clickable="{str(clickable).lower()}" enabled="true"'
        f' bounds="[20,{y}][500,{y + 80}]"/>'
    )


HOME = _screen(
    _node("Explore", rid=f"{PKG}:id/explore", y=20, clickable=True),
    _node("My Apps", y=120),
    _node("No saved apps yet", y=220),
)
LAYOUT_ONLY_FRAME = _screen(
    _node("Explore", rid=f"{PKG}:id/explore", y=20, clickable=True),
    _node("My Apps", y=120),
    _node("No saved apps yet", y=220),
    _node(y=320),  # transient, unlabelled pager edge
)
EXPLORE = _screen(
    _node("Explore", rid=f"{PKG}:id/explore", y=20, clickable=True),
    _node("Search", rid=f"{PKG}:id/search", y=120, clickable=True),
    _node("Recommended", y=220),
)


class DelayedExplore(FakeDevice):
    def __init__(self) -> None:
        super().__init__(hierarchy_xml=HOME, package=PKG)

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._xml = LAYOUT_ONLY_FRAME


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perception={"observe_escalates_to_vision": False},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def test_layout_only_frame_is_confirmed_before_tap_returns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dev = DelayedExplore()
    eng = _engine(tmp_path, dev)
    eng.analyze(source="hierarchy")

    # Pin the real trace's first settle result: AUA thought a fast hierarchy transition had
    # completed, although the following analyze still described the old page sliding away.
    monkeypatch.setattr(
        eng,
        "_await_post_action_ready",
        lambda **_: {
            "changed": True,
            "masked": 0,
            "ms": 265,
            "timeout": False,
            "via": "hierarchy-fast",
        },
    )
    confirmations: list[dict[str, Any]] = []

    def finish_navigation(**kwargs: Any) -> ActionResult:
        confirmations.append(kwargs)
        dev._xml = EXPLORE
        return ActionResult(ok=True, action="wait-stable")

    monkeypatch.setattr(eng, "wait_stable", finish_navigation)

    out = eng.tap(selector={"text": "Explore"}, observe=True)

    assert len([call for call in dev.calls if call[0] == "click"]) == 1
    assert confirmations == [
        {"interval_ms": 80, "settle_ms": 350, "timeout_ms": 1400, "observe": False}
    ]
    assert out.observation is not None
    assert {e.text for e in out.observation.elements if e.text} >= {
        "Explore",
        "Search",
        "Recommended",
    }
    assert out.change is not None and out.change["changed"] is True
    assert out.stale_risk is None
    assert out.settle is not None and out.settle["confirmation_ms"] >= 0


def test_complete_destination_still_confirms_a_fast_settle(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class ImmediateExplore(DelayedExplore):
        def click(self, x: int, y: int) -> None:
            super().click(x, y)
            self._xml = EXPLORE

    dev = ImmediateExplore()
    eng = _engine(tmp_path, dev)
    eng.analyze(source="hierarchy")
    monkeypatch.setattr(
        eng,
        "_await_post_action_ready",
        lambda **_: {
            "changed": True,
            "masked": 0,
            "ms": 145,
            "timeout": False,
            "via": "hierarchy-fast",
        },
    )

    confirmations: list[dict[str, Any]] = []

    def confirm(**kwargs: Any) -> ActionResult:
        confirmations.append(kwargs)
        return ActionResult(ok=True, action="wait-stable")

    monkeypatch.setattr(eng, "wait_stable", confirm)

    out = eng.tap(selector={"text": "Explore"}, observe=True)

    assert out.observation is not None
    assert any(e.text == "Search" for e in out.observation.elements)
    assert confirmations == [
        {"interval_ms": 80, "settle_ms": 350, "timeout_ms": 1400, "observe": False}
    ]
    assert out.settle is not None and out.settle["confirmation_ms"] >= 0
