"""Rolling capture buffer — dedupe, diff summary, prune, engine hint."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from android_ui_analyser.capture import (
    CaptureBuffer,
    CaptureCfgView,
    FrameEntry,
    diff_summary,
    frame_hash,
)
from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import ScreenImage
from conftest import FakeDevice, make_config, make_png


def _png_color(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (w, h), rgb)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_frame_hash_stable_and_sensitive() -> None:
    a = np.zeros((64, 64), dtype=np.uint8)
    b = a.copy()
    b[10, 10] = 255
    assert frame_hash(a) == frame_hash(a.copy())
    assert frame_hash(a) != frame_hash(b)


def test_diff_summary_reports_center_change(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.new("RGB", (96, 96), (0, 0, 0)).save(p1, quality=90)
    changed = Image.new("RGB", (96, 96), (0, 0, 0))
    # paint center block white
    for y in range(32, 64):
        for x in range(32, 64):
            changed.putpixel((x, y), (255, 255, 255))
    changed.save(p2, quality=90)
    t0 = 1_000_000
    entries = [
        FrameEntry(t_ms=t0, path=str(p1), hash="a", bytes=10, w=96, h=96),
        FrameEntry(t_ms=t0 + 120, path=str(p2), hash="b", bytes=10, w=96, h=96, action="tap"),
    ]
    lines = diff_summary(entries, threshold=5.0)
    assert lines
    assert "center" in lines[0]
    assert "t+0" in lines[0]


def test_prune_ttl_and_max_mb(tmp_path: Path) -> None:
    shots = [
        ScreenImage(_png_color(40, 40, (i * 40, 0, 0)), width=40, height=40) for i in range(5)
    ]
    idx = {"i": 0}

    def shot() -> ScreenImage:
        i = min(idx["i"], len(shots) - 1)
        idx["i"] += 1
        return shots[i]

    cfg = CaptureCfgView(idle_fps=100, burst_fps=100, burst_ms=0, ttl_s=3600, max_mb=1, jpeg_quality=50)
    buf = CaptureBuffer(root=tmp_path, serial="fake", cfg=cfg, screenshot=shot)
    # Manually inject old + large entries
    frames = tmp_path / "fake" / buf.session_id / "frames"
    frames.mkdir(parents=True)
    old = frames / "old.jpg"
    Image.new("RGB", (10, 10), (1, 2, 3)).save(old)
    now = int(time.time() * 1000)
    with buf._lock:
        buf._entries = [
            FrameEntry(t_ms=now - 999_999_000, path=str(old), hash="old", bytes=100, w=10, h=10),
        ]
    removed = buf._prune()
    assert removed >= 1
    assert not old.exists()


def test_buffer_dedupes_identical_frames(tmp_path: Path) -> None:
    red = ScreenImage(_png_color(64, 64, (200, 0, 0)), width=64, height=64)
    blue = ScreenImage(_png_color(64, 64, (0, 0, 200)), width=64, height=64)
    stream = [red, red, red, blue, blue]
    idx = {"i": 0}

    def shot() -> ScreenImage:
        i = min(idx["i"], len(stream) - 1)
        idx["i"] += 1
        return stream[i]

    cfg = CaptureCfgView(idle_fps=50, burst_fps=50, burst_ms=500, ttl_s=60, max_mb=50)
    buf = CaptureBuffer(root=tmp_path, serial="emu", cfg=cfg, screenshot=shot)
    buf.start()
    buf.mark("tap")
    deadline = time.time() + 2.0
    while time.time() < deadline and len(buf._entries) < 2:
        time.sleep(0.05)
    buf.stop()
    # Only color changes should be kept (red once, blue once) — at least 2, not 5
    assert 2 <= len(buf._entries) <= 3
    out = buf.last(since_ms=0)
    assert out["count"] == len(buf._entries)
    assert isinstance(out["summary"], list)


def test_engine_capture_hint_after_burst(tmp_path: Path) -> None:
    red = make_png(80, 80, color=(180, 20, 20))
    blue = make_png(80, 80, color=(20, 20, 180))
    device = FakeDevice(
        screenshot_bytes=red,
        screenshots=[red, red, blue, blue, blue],
        width=80,
        height=80,
    )
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        capture={
            "enabled": True,
            "idle_fps": 30,
            "burst_fps": 30,
            "burst_ms": 800,
            "ttl_s": 60,
            "hint": True,
        },
    )
    engine = Engine(cfg, device=device)
    status = engine.capture_start()
    assert status["running"] is True
    engine._screen_changed()  # mark + burst
    deadline = time.time() + 2.5
    while time.time() < deadline and not engine._capture.hint_ready():  # type: ignore[union-attr]
        time.sleep(0.05)
    assert engine._capture_hint() is not None
    last = engine.capture_last(since="last-action")
    assert last["count"] >= 1
    engine.capture_stop()


def test_capture_status_without_buffer(tmp_path: Path) -> None:
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=FakeDevice())
    st = engine.capture_status()
    assert st["running"] is False
    assert "daemon" in (st.get("hint") or "").lower()
