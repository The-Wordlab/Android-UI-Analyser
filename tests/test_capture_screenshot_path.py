"""The capture loop must default to the u2 screenshot, not ``adb exec-out screencap -p``.

``screencap_png`` reads as the "fast, raw" path, so the capture buffer was pointed at it by
default. Measured end-to-end it is the *slow* one — the device encodes a full-resolution PNG
rather than a JPEG, costing ~2.2x more on both an emulator (720x1280: 35ms vs 79ms) and a
physical device (1440x3120: 210ms vs 471ms). Nothing is bought for that: every frame is
re-encoded to JPEG on write, and OCR recall over four Settings screens was identical either
way (51/73 strings recovered from both). The flag survives for pixel-exact captures; the
default must stay on the cheap path.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config


def _capture_source(tmp_path: Path, **perf: bool) -> tuple[Engine, str]:
    """Start capture and report which device method one frame actually reaches.

    This used to read the *name* of the callable the buffer was handed. It cannot any more:
    the buffer is deliberately no longer given a device-bound method, because holding one kept
    uiautomator2 alive past every teardown and let a sampling tick reconnect the server mid
    handover. The engine picks the source per frame instead, so the question "which path does
    capture use" has to be answered by taking a frame rather than by reading a name — which is
    the better test regardless, since it survives the next refactor of the same indirection.
    """

    overrides: dict[str, object] = {
        "cache": {"dir": str(tmp_path / "cache")},
        "capture": {"enabled": True, "idle_fps": 1, "burst_fps": 1, "ttl_s": 60},
    }
    if perf:
        overrides["perf"] = perf
    cfg = make_config(**overrides)
    engine = Engine(cfg, device=FakeDevice())

    reached: list[str] = []
    device = engine._device
    assert device is not None
    for method in ("screenshot", "screencap_png"):
        original = getattr(device, method, None)
        if original is None:
            continue

        def spy(*args: object, _method: str = method, _original: object = original, **kw: object):
            reached.append(_method)
            return _original(*args, **kw)  # type: ignore[operator]

        setattr(device, method, spy)

    engine.capture_start()
    assert engine._capture is not None
    # Quiesce the sampling thread first, or its own ticks race the one frame being measured.
    engine._capture.pause("test", settle_s=2.0)
    reached.clear()
    engine._capture.screenshot()

    assert reached, "taking a frame reached no device method at all"
    # The *entry* point is the question. ``screencap_png`` delegates onward internally, so
    # the last name reached says nothing about which path capture chose.
    return engine, reached[0]


def test_capture_defaults_to_the_u2_screenshot(tmp_path: Path) -> None:
    engine, name = _capture_source(tmp_path)
    try:
        assert name == "screenshot", (
            "capture must default to device.screenshot — screencap_png makes the device "
            "encode a full-res PNG and measures ~2.2x slower for identical OCR recall"
        )
    finally:
        engine.capture_stop()


def test_capture_adb_screencap_flag_still_opts_into_lossless_png(tmp_path: Path) -> None:
    engine, name = _capture_source(tmp_path, capture_adb_screencap=True)
    try:
        assert name == "screencap_png", (
            "perf.capture_adb_screencap=true must still route the capture loop through "
            "adb exec-out screencap -p for callers that need pixel-exact frames"
        )
    finally:
        engine.capture_stop()


def test_the_shipped_default_is_off() -> None:
    assert make_config().perf.capture_adb_screencap is False
