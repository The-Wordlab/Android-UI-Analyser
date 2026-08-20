"""Rolling capture buffer — dedupe, diff summary, prune, engine hint."""

from __future__ import annotations

import threading
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
    with engine._acting():  # mark + burst, bracketing a (no-op) interaction
        pass
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


def test_diff_summary_region_filter(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.new("RGB", (96, 96), (0, 0, 0)).save(p1, quality=90)
    changed = Image.new("RGB", (96, 96), (0, 0, 0))
    for y in range(32, 64):
        for x in range(32, 64):
            changed.putpixel((x, y), (255, 255, 255))
    changed.save(p2, quality=90)
    t0 = 1_000_000
    entries = [
        FrameEntry(t_ms=t0, path=str(p1), hash="a", bytes=10, w=96, h=96),
        FrameEntry(t_ms=t0 + 80, path=str(p2), hash="b", bytes=10, w=96, h=96),
    ]
    all_lines = diff_summary(entries, threshold=5.0)
    assert all_lines and "center" in all_lines[0]
    filtered = diff_summary(entries, threshold=5.0, region="upper")
    assert filtered and "no 'upper' cell change" in filtered[0]
    center_only = diff_summary(entries, threshold=5.0, region="center")
    assert center_only and "center" in center_only[0]


def test_extend_burst_on_change(tmp_path: Path) -> None:
    colors = [(i * 20, 0, 255 - i * 20) for i in range(8)]
    shots = [ScreenImage(_png_color(48, 48, c), width=48, height=48) for c in colors]
    idx = {"i": 0}

    def shot() -> ScreenImage:
        i = min(idx["i"], len(shots) - 1)
        idx["i"] += 1
        return shots[i]

    cfg = CaptureCfgView(
        idle_fps=40,
        burst_fps=40,
        burst_ms=80,  # short base burst — extension should keep it open
        extend_burst_on_change=True,
        ttl_s=60,
        max_mb=50,
    )
    buf = CaptureBuffer(root=tmp_path, serial="burst", cfg=cfg, screenshot=shot)
    buf.start()
    buf.mark("tap:Go")
    deadline = time.time() + 2.0
    while time.time() < deadline and len(buf._entries) < 4:
        time.sleep(0.03)
    buf.stop()
    assert len(buf._entries) >= 3


def test_export_gif_and_explain(tmp_path: Path) -> None:
    from android_ui_analyser.capture import change_duration_ms, export_animation, local_narration

    p1 = tmp_path / "1.jpg"
    p2 = tmp_path / "2.jpg"
    Image.new("RGB", (40, 40), (10, 10, 10)).save(p1, quality=80)
    Image.new("RGB", (40, 40), (200, 20, 20)).save(p2, quality=80)
    t0 = 5_000_000
    entries = [
        FrameEntry(t_ms=t0, path=str(p1), hash="a", bytes=10, w=40, h=40, action="tap:X"),
        FrameEntry(t_ms=t0 + 150, path=str(p2), hash="b", bytes=10, w=40, h=40),
    ]
    out = tmp_path / "clip.gif"
    written = export_animation(entries, out, fmt="gif", fps=5)
    assert Path(written).is_file()
    assert change_duration_ms(entries) == 150
    payload = {
        "frames": [e.__dict__ for e in entries],
        "summary": ["t+0–t+150ms: center changed"],
        "change_duration_ms": 150,
    }
    narr = local_narration(payload)
    assert "tap:X" in narr
    assert "150ms" in narr

    red = ScreenImage(_png_color(40, 40, (200, 0, 0)), width=40, height=40)
    blue = ScreenImage(_png_color(40, 40, (0, 0, 200)), width=40, height=40)
    stream = [red, blue]
    idx = {"i": 0}

    def shot() -> ScreenImage:
        i = min(idx["i"], len(stream) - 1)
        idx["i"] += 1
        return stream[i]

    cfg = CaptureCfgView(idle_fps=50, burst_fps=50, burst_ms=400, ttl_s=60, max_mb=50)
    buf = CaptureBuffer(root=tmp_path / "buf", serial="exp", cfg=cfg, screenshot=shot)
    buf.start()
    buf.mark("tap")
    deadline = time.time() + 2.0
    while time.time() < deadline and len(buf._entries) < 2:
        time.sleep(0.04)
    exp = buf.export(tmp_path / "out.gif", fmt="gif", fps=6)
    assert Path(exp["path"]).is_file()
    explained = buf.explain_local()
    assert "narration" in explained
    buf.stop()


def test_suite_failure_attaches_capture(tmp_path: Path) -> None:
    from android_ui_analyser.suite import parse_suite, run_suite

    red = make_png(60, 60, color=(180, 20, 20))
    blue = make_png(60, 60, color=(20, 20, 180))
    device = FakeDevice(
        screenshot_bytes=red,
        screenshots=[red, blue, blue],
        width=60,
        height=60,
        hierarchy_xml=(
            '<?xml version="1.0"?><hierarchy rotation="0">'
            '<node class="android.widget.TextView" text="Hello" '
            'clickable="false" enabled="true" bounds="[0,0][60,60]"/></hierarchy>'
        ),
        text_index={"Hello": (0, 0, 60, 60)},
    )
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        capture={"enabled": True, "idle_fps": 40, "burst_fps": 40, "burst_ms": 500, "hint": True},
    )
    engine = Engine(cfg, device=device)
    engine.capture_start()
    with engine._acting("tap:Hello"):
        pass
    deadline = time.time() + 2.0
    while time.time() < deadline and len(engine._capture._entries) < 1:  # type: ignore[union-attr]
        time.sleep(0.04)
    suite = parse_suite(
        """
name: fail_cap
checks:
  - has: "Nope"
"""
    )
    result = run_suite(engine, suite)
    assert not result.ok
    assert result.capture is not None
    assert result.results[0].capture is not None
    engine.capture_stop()


def test_action_result_capture_hint(tmp_path: Path) -> None:
    red = make_png(50, 50, color=(100, 0, 0))
    blue = make_png(50, 50, color=(0, 0, 100))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><hierarchy rotation="0">'
        '<node index="0" class="android.widget.Button" text="Go" '
        'resource-id="com.test:id/go" clickable="true" enabled="true" '
        'bounds="[10,10][40,40]"/></hierarchy>'
    )
    device = FakeDevice(
        hierarchy_xml=xml,
        screenshot_bytes=red,
        screenshots=[red, blue, blue, blue],
        width=50,
        height=50,
        text_index={"Go": (10, 10, 40, 40)},
        resource_index={"com.test:id/go": (10, 10, 40, 40)},
    )
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        capture={"enabled": True, "idle_fps": 40, "burst_fps": 40, "burst_ms": 600, "hint": True},
    )
    engine = Engine(cfg, device=device)
    engine.capture_start()
    with engine._acting("tap:Go"):
        device.click(25, 25)
    deadline = time.time() + 2.0
    while time.time() < deadline and not engine._capture.hint_ready():  # type: ignore[union-attr]
        time.sleep(0.05)
    # Simulate post-action observe attaching the hint (same path as ActionResult).
    hint = engine._capture_hint()
    assert hint is not None
    engine.capture_stop()


def test_sidecar_disabled_raises(tmp_path: Path) -> None:
    engine = Engine(
        make_config(cache={"dir": str(tmp_path)}, capture={"enabled": False, "sidecar": False}),
        device=FakeDevice(),
    )
    from android_ui_analyser.errors import UsageError

    try:
        engine.capture_sidecar_start()
    except UsageError as exc:
        assert "sidecar" in str(exc).lower()
    else:
        raise AssertionError("expected UsageError when capture.sidecar is false")


def test_region_from_point() -> None:
    from android_ui_analyser.engine import _region_from_point

    assert _region_from_point(10, 10, 300, 300) == "upper-left"
    assert _region_from_point(150, 150, 300, 300) == "center"
    assert _region_from_point(290, 290, 300, 300) == "lower-right"


def test_pause_can_wait_for_a_frame_already_in_flight(tmp_path: Path) -> None:
    """Setting the flag is not enough — one stale frame is what broke the helper handover.

    ``pause()`` on its own only asks the sampling loop to stop *next* time round. A thread
    already past that check goes on to take one more screenshot, and during a handover to the
    on-device helper that single frame reconnects uiautomator2, which takes back the
    UiAutomation slot the helper was just handed and tears its accessibility service down
    mid-run. It showed up as the device stopping at a different step every time.

    So the waiting form has to be a real handshake: when it returns True, no frame grab is in
    flight, and none will start.
    """

    started = threading.Event()
    release = threading.Event()
    shots = {"n": 0}

    def shot() -> ScreenImage:
        shots["n"] += 1
        started.set()
        release.wait(2.0)  # hold the loop inside the grab, exactly like a slow screenshot
        return ScreenImage(_png_color(32, 32, (shots["n"], 0, 0)), width=32, height=32)

    cfg = CaptureCfgView(idle_fps=50, burst_fps=50, burst_ms=500, ttl_s=60, max_mb=50)
    buf = CaptureBuffer(root=tmp_path, serial="handover", cfg=cfg, screenshot=shot)
    buf.start()
    try:
        assert started.wait(2.0), "the sampling loop never took a frame"

        # Pause while a grab is mid-flight: it must not claim to be settled yet.
        assert buf.pause("handover", settle_s=0.2) is False
        release.set()
        assert buf.pause("handover", settle_s=2.0) is True

        # Nothing may start after a settled pause, however long we wait.
        taken = shots["n"]
        time.sleep(0.2)
        assert shots["n"] == taken, "a frame was grabbed after the buffer settled"
    finally:
        release.set()
        buf.stop()


def test_pause_without_settling_stays_non_blocking(tmp_path: Path) -> None:
    """Idle-pause and `capture off` must keep their old fire-and-forget behaviour."""

    cfg = CaptureCfgView(idle_fps=1, burst_fps=1, burst_ms=100, ttl_s=60, max_mb=50)
    buf = CaptureBuffer(
        root=tmp_path,
        serial="idle",
        cfg=cfg,
        screenshot=lambda: ScreenImage(_png_color(8, 8, (0, 0, 0)), width=8, height=8),
    )
    assert buf.pause("idle") is True
    assert buf.paused is True
