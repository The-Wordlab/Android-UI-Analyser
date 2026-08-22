"""The unready verdict must not need an Activity change, and system noise must not veto it.

Measured live while building Phase 1 (single-Activity app, fictional names here): re-entering
a content screen the instant the network returned handed back a destination with **no readable
app content at all** — a bare nav-back control and a status-bar notification — and the verdict
stayed silent, twice over:

* ``departure_without_arrival`` requires ``activity_changed``, and a single-Activity (Compose)
  app never changes Activity for in-app navigation — the dominant modern shape;
* the additive-arrival test read ``change.text_added``, which counts **every window** — the
  status-bar clock ticking over (``8:15``) and the Wi-Fi icon returning counted as "content
  arrived" and vetoed the verdict even where it applied.

So the classifier now measures additive arrival on the app's own elements only, and a strong
subtractive transition that lands on a content-bare tree (no labelled app node, at most one
unlabelled affordance) is unready regardless of Activity — the `shell_only_tree` detector the
design doc had queued for Phase 2, promoted because Phase 1 is nearly inert without it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PKG = "com.example.fiction"

JOURNALLED_SETTLE = {"changed": True, "masked": 0, "ms": 199, "timeout": False, "via": "hierarchy"}


def _screen(*rows: str) -> str:
    return '<hierarchy rotation="0">' + "".join(rows) + "</hierarchy>"


def _node(
    text: str = "",
    *,
    rid: str = "",
    y: int,
    clickable: bool = False,
    pkg: str = PKG,
) -> str:
    return (
        f'<node class="android.view.View" package="{pkg}" text="{text}"'
        f' resource-id="{rid}" clickable="{str(clickable).lower()}" enabled="true"'
        f' bounds="[20,{y}][500,{y + 80}]"/>'
    )


def _clock(text: str) -> str:
    return (
        f'<node class="android.widget.TextView" package="com.android.systemui" text="{text}"'
        f' resource-id="com.android.systemui:id/clock" clickable="false" enabled="true"'
        f' bounds="[0,0][100,40]"/>'
    )


def _home(clock: str) -> str:
    return _screen(
        _clock(clock),
        _node("Trending stickers", rid=f"{PKG}:id/homeTitle", y=60),
        _node("Sticker pack one", rid=f"{PKG}:id/packOne", y=160, clickable=True),
        _node("Sticker pack two", rid=f"{PKG}:id/packTwo", y=260, clickable=True),
        _node("Open gallery", rid=f"{PKG}:id/openGallery", y=360, clickable=True),
    )


def _bare(clock: str) -> str:
    """The gallery before its fetch completes: a back affordance, a page-indicator dot, and
    nothing readable — the dot is measured live shape, and punctuation is not content."""
    return _screen(
        _clock(clock),
        _node(rid=f"{PKG}:id/buttonNavBack", y=60, clickable=True),
        _node("\u2022", y=140),
        _node(y=220),
    )


def _rendered(clock: str) -> str:
    return _screen(
        _clock(clock),
        _node(rid=f"{PKG}:id/buttonNavBack", y=60, clickable=True),
        _node("Fresh gallery item", rid=f"{PKG}:id/galleryItem", y=160, clickable=True),
    )


class SingleActivityGallery(FakeDevice):
    """Same Activity throughout; the clock ticks over during the transition (system noise)."""

    def __init__(self, *, render_after_s: float) -> None:
        super().__init__(hierarchy_xml=_home("8:14"), package=PKG, activity=".MainActivity")
        self._render_after_s = render_after_s
        self._tapped_at: float | None = None

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._tapped_at = time.monotonic()

    def dump_hierarchy(self, compressed: bool = False) -> str:
        self.hierarchy_calls += 1
        if self._tapped_at is None:
            return _home("8:14")
        if time.monotonic() - self._tapped_at < self._render_after_s:
            return _bare("8:15")
        return _rendered("8:15")


def _engine(tmp_path: Path, device: FakeDevice, **perf: Any) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perception={"observe_escalates_to_vision": False},
        perf={"stable_delay_ms": {"default": 0}, **perf},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _tap_gallery(eng: Engine, monkeypatch: Any) -> Any:
    gallery = next(
        e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Open gallery"
    )
    monkeypatch.setattr(eng, "_await_post_action_ready", lambda **_: dict(JOURNALLED_SETTLE))
    return eng.tap(gallery, observe=True)


def test_a_bare_destination_without_an_activity_change_is_not_called_settled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dev = SingleActivityGallery(render_after_s=30.0)
    eng = _engine(tmp_path, dev, arrival_extension_ms=400)

    out = _tap_gallery(eng, monkeypatch)

    change = out.change or {}
    assert change.get("activity_changed") is False, f"fixture drifted: {change}"
    assert "8:15" in (change.get("text_added") or []), (
        f"fixture drifted: the system clock must tick to model the live noise: {change}"
    )
    assert out.stale_risk, "a content-bare destination must not be handed over silently"
    assert out.arrival is not None and out.arrival["state"] == "unconfirmed"
    assert "shell_only_tree" in out.arrival["evidence"]


def test_the_wait_returns_the_gallery_once_it_renders(tmp_path: Path, monkeypatch: Any) -> None:
    dev = SingleActivityGallery(render_after_s=0.45)
    eng = _engine(tmp_path, dev)

    out = _tap_gallery(eng, monkeypatch)

    assert out.stale_risk is None
    texts = {e.text for e in (out.observation.elements if out.observation else []) if e.text}
    assert "Fresh gallery item" in texts
    assert out.arrival is not None and out.arrival["state"] == "settled"
    assert "content_arrived" in out.arrival["evidence"]


def test_a_sparse_but_labelled_destination_is_left_alone(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A thin screen with readable app content is a destination, not a loading shell."""

    class LabelledSparse(SingleActivityGallery):
        def dump_hierarchy(self, compressed: bool = False) -> str:
            self.hierarchy_calls += 1
            if self._tapped_at is None:
                return _home("8:14")
            return _screen(
                _clock("8:15"),
                _node(rid=f"{PKG}:id/buttonNavBack", y=60, clickable=True),
                _node("Gallery is empty", rid=f"{PKG}:id/emptyState", y=160),
            )

    dev = LabelledSparse(render_after_s=99.0)
    eng = _engine(tmp_path, dev)

    started = time.monotonic()
    out = _tap_gallery(eng, monkeypatch)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert out.stale_risk is None, f"labelled content is arrival evidence: {out.stale_risk}"
    assert out.arrival is None
    assert elapsed_ms < 1500, f"no extension may be spent on a rendered screen: {elapsed_ms:.0f}ms"
