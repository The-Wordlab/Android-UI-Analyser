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
    overrides: dict[str, object] = {
        "cache": {"dir": str(tmp_path / "cache")},
        "capture": {"enabled": True, "idle_fps": 1, "burst_fps": 1, "ttl_s": 60},
    }
    if perf:
        overrides["perf"] = perf
    cfg = make_config(**overrides)
    engine = Engine(cfg, device=FakeDevice())
    engine.capture_start()
    assert engine._capture is not None
    return engine, engine._capture.screenshot.__func__.__name__


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
