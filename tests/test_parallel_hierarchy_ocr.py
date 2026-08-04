"""Parallel Apple Vision + hierarchy: keep what OCR adds, withhold what it merely repeats.

The two observations run concurrently and are fused. What survives the fusion changed once
the fused list started feeding one-shot selectors: a reading that only repeats text the tree
already reports is not evidence to reconcile, and keeping it made `tap --text Submit` raise
"matches 2 elements" on a screen with one Submit button. Pixel-only text always survives -
that is what the OCR pass is for.
"""

from __future__ import annotations

import threading
import time

from android_ui_analyser.providers.base import (
    Availability,
    ChainSpec,
    OcrProvider,
    ScreenImage,
    TextBox,
)
from android_ui_analyser.schema import ScreenSource, Source
from conftest import FakeDevice, make_config, make_engine

XML = (
    '<hierarchy rotation="0">'
    '<node class="android.widget.Button" text="Submit" resource-id="x:id/submit" '
    'clickable="true" enabled="true" bounds="[100,200][300,260]"/>'
    '<node class="android.widget.TextView" text="Welcome" enabled="true" '
    'bounds="[0,40][300,100]"/>'
    "</hierarchy>"
)


class _ParallelDevice(FakeDevice):
    def __init__(self, barrier: threading.Barrier) -> None:
        super().__init__(hierarchy_xml=XML)
        self._barrier = barrier

    def dump_hierarchy(self, compressed: bool = False) -> str:
        self.hierarchy_calls += 1
        self._barrier.wait(timeout=1.0)
        time.sleep(0.08)
        return self._xml


class _ParallelApple(OcrProvider):
    name = "apple_vision"

    def __init__(self, barrier: threading.Barrier) -> None:
        super().__init__()
        self._barrier = barrier
        self.calls = 0

    def is_available(self) -> Availability:
        return Availability(True, "test apple vision")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        self.calls += 1
        self._barrier.wait(timeout=1.0)
        time.sleep(0.08)
        return [
            # Sits on top of the hierarchy's Submit button and says the same thing, so it
            # is withheld by default: a second copy of text already described is not
            # evidence, and it made `tap --text Submit` ambiguous.
            TextBox(text="Submit", bounds=(105, 205, 295, 255), confidence=0.99),
            # Pixel-only text absent from the hierarchy. This is what OCR is for, and it
            # must always survive.
            TextBox(text="Canvas total 42", bounds=(40, 400, 300, 450), confidence=0.91),
        ]


class _Factory:
    def __init__(self, provider: OcrProvider) -> None:
        self.provider = provider

    def is_enabled(self, kind: str) -> bool:
        return kind == "ocr"

    def build_chain(self, kind: str) -> ChainSpec:
        return ChainSpec(kind=kind, providers=[self.provider] if kind == "ocr" else [])


def test_hierarchy_and_apple_ocr_run_in_parallel_and_both_survive() -> None:
    barrier = threading.Barrier(2)
    provider = _ParallelApple(barrier)
    device = _ParallelDevice(barrier)
    cfg = make_config(
        ocr={"enabled": True, "chain": ["apple_vision"], "augment_hierarchy": True},
        perf={"skip_unchanged_analyze": False},
    )
    engine = make_engine(config=cfg, device=device, factory=_Factory(provider))  # type: ignore[arg-type]

    started = time.perf_counter()
    result = engine.analyze(source="hierarchy")
    elapsed = time.perf_counter() - started

    submit = [element for element in result.elements if element.text == "Submit"]
    assert {element.source for element in submit} == {Source.hierarchy}, (
        "the OCR reading of Submit is a second copy of text the tree already reports; "
        "keeping it made `tap --text Submit` raise 'matches 2 elements'"
    )
    assert any(
        element.text == "Canvas total 42" and element.source is Source.ocr
        for element in result.elements
    ), "pixel-only text is the whole point of the OCR pass and must survive"
    assert result.screen.source is ScreenSource.mixed
    assert result.meta.providers_used == ["apple_vision"]
    assert provider.calls == 1
    # Each side sleeps for 80ms; a sequential implementation would take at least 160ms.
    assert elapsed < 0.15


def test_redundant_readings_can_be_kept_for_inspection() -> None:
    """`ocr.drop_redundant: false` returns the unfiltered pass, for debugging OCR itself."""
    barrier = threading.Barrier(2)
    provider = _ParallelApple(barrier)
    device = _ParallelDevice(barrier)
    cfg = make_config(
        ocr={
            "enabled": True,
            "chain": ["apple_vision"],
            "augment_hierarchy": True,
            "drop_redundant": False,
        },
        perf={"skip_unchanged_analyze": False},
    )
    engine = make_engine(config=cfg, device=device, factory=_Factory(provider))  # type: ignore[arg-type]

    result = engine.analyze(source="hierarchy")

    submit = [element for element in result.elements if element.text == "Submit"]
    assert {element.source for element in submit} == {Source.hierarchy, Source.ocr}


def test_explicit_no_ocr_keeps_hierarchy_only() -> None:
    provider = _ParallelApple(threading.Barrier(2))
    cfg = make_config(
        ocr={"enabled": True, "chain": ["apple_vision"], "augment_hierarchy": True},
    )
    engine = make_engine(
        config=cfg, device=FakeDevice(hierarchy_xml=XML), factory=_Factory(provider)
    )  # type: ignore[arg-type]

    result = engine.analyze(source="hierarchy", with_ocr=False)

    assert {element.source for element in result.elements} == {Source.hierarchy}
    assert result.screen.source is ScreenSource.hierarchy
    assert provider.calls == 0
