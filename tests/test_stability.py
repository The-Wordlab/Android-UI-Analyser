"""AC14 — ``wait --for-stable`` settles on a (stubbed) screenshot stream and times out
with a clear, structured error, WITHOUT running OCR or a hierarchy parse.

The settle check is a cheap perceptual-hash over screenshots only — we assert that by
spying on the device: ``dump_hierarchy`` is never called, and the OCR provider chain is
never built/invoked.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ExitCode, StabilityTimeout
from android_ui_analyser.imaging import dhash, hamming, is_stable
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config, make_png

runner = CliRunner()

# Two structurally-distinct frames (a moving solid box → different dHash gradients).
FRAME_A = make_png(
    width=200, height=400, color=(240, 240, 240), boxes=[((10, 40, 90, 360), (0, 0, 0))]
)
FRAME_B = make_png(
    width=200, height=400, color=(240, 240, 240), boxes=[((110, 40, 190, 360), (0, 0, 0))]
)


def _engine(device: FakeDevice) -> Engine:
    cfg = make_config(daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


# --------------------------------------------------------------------------- hash unit


def test_dhash_identical_is_stable_distinct_is_not() -> None:
    a = ScreenImage(FRAME_A, width=200, height=400)
    b = ScreenImage(FRAME_B, width=200, height=400)
    assert is_stable(dhash(a), dhash(a))  # a frame equals itself
    assert hamming(dhash(a), dhash(b)) > 8  # the two frames are clearly different


# --------------------------------------------------------------------------- settle


def test_wait_stable_settles_without_ocr_or_hierarchy() -> None:
    # changes for two frames, then holds steady → settles on the steady run.
    stream = [FRAME_A, FRAME_B] + [FRAME_A] * 20
    dev = FakeDevice(screenshots=stream)
    eng = _engine(dev)
    res = eng.wait_stable(interval_ms=1, settle_ms=3, timeout_ms=3000)
    assert res.ok and res.action == "wait-stable"
    assert dev.hierarchy_calls == 0  # NO hierarchy parse
    assert dev.screenshot_calls > 0  # it polled screenshots


def test_wait_stable_does_not_build_the_ocr_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(screenshots=[FRAME_A] * 10)
    eng = _engine(dev)

    built: list[str] = []
    real_build = eng.factory.build_chain
    monkeypatch.setattr(
        eng.factory, "build_chain", lambda kind: built.append(kind) or real_build(kind)
    )
    eng.wait_stable(interval_ms=1, settle_ms=2, timeout_ms=2000)
    assert built == []  # no provider chain (OCR/detection) is ever constructed


# --------------------------------------------------------------------------- timeout


def test_wait_stable_times_out_with_structured_error() -> None:
    # Alternating frames never hold steady → never settles.
    dev = FakeDevice(screenshots=[FRAME_A, FRAME_B] * 200)
    eng = _engine(dev)
    with pytest.raises(StabilityTimeout) as ei:
        eng.wait_stable(interval_ms=1, settle_ms=50, timeout_ms=25)
    assert ei.value.exit_code == ExitCode.DEVICE == 3
    assert ei.value.hint  # actionable hint
    assert dev.hierarchy_calls == 0


# --------------------------------------------------------------------------- CLI wiring


def test_cli_wait_for_stable_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(screenshots=[FRAME_A] * 12)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    res = runner.invoke(
        app,
        ["wait", "--for-stable", "--interval", "1", "--settle", "2", "--timeout", "3000"],
    )
    assert res.exit_code == 0, res.stderr
    assert "wait-stable" in res.stdout


def test_cli_wait_for_stable_timeout_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(screenshots=[FRAME_A, FRAME_B] * 200)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    res = runner.invoke(
        app,
        ["wait", "--for-stable", "--interval", "1", "--settle", "50", "--timeout", "25"],
    )
    assert res.exit_code == 3
    import json

    err = json.loads(res.stderr)
    assert err["error"]["code"] == "wait_timeout"
