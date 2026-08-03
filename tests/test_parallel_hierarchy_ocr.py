"""Parallel Apple Vision + hierarchy observations keep both raw data sources."""

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
            # Deliberately duplicates hierarchy text: both observations must survive.
            TextBox(text="Submit", bounds=(105, 205, 295, 255), confidence=0.99),
            # Pixel-only text absent from the hierarchy.
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
    assert {element.source for element in submit} == {Source.hierarchy, Source.ocr}
    assert any(
        element.text == "Canvas total 42" and element.source is Source.ocr
        for element in result.elements
    )
    assert result.screen.source is ScreenSource.mixed
    assert result.meta.providers_used == ["apple_vision"]
    assert provider.calls == 1
    # Each side sleeps for 80ms; a sequential implementation would take at least 160ms.
    assert elapsed < 0.15


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
