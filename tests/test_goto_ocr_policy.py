"""`goto` keeps known routes hierarchy-fast and pays for OCR only on a miss."""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMemoryStore, RouteStep
from android_ui_analyser.providers.base import (
    Availability,
    ChainSpec,
    OcrProvider,
    ScreenImage,
    TextBox,
)
from android_ui_analyser.schema import Source
from conftest import FakeDevice, make_config
from test_memory import APPS, HOME, P, _elements


class _AdvancingDevice(FakeDevice):
    def __init__(self, screens: list[str], **kwargs: object) -> None:
        super().__init__(hierarchy_xml=screens[0], **kwargs)  # type: ignore[arg-type]
        self._screens = screens
        self._index = 0

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._index = min(self._index + 1, len(self._screens) - 1)
        self._xml = self._screens[self._index]


class _Apple(OcrProvider):
    name = "apple_vision"

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
        self.calls = 0

    def is_available(self) -> Availability:
        return Availability(True, "test apple vision")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        self.calls += 1
        return [TextBox(text=self.text, bounds=(100, 200, 300, 260), confidence=0.99)]


class _Factory:
    def __init__(self, provider: OcrProvider) -> None:
        self.provider = provider

    def is_enabled(self, kind: str) -> bool:
        return kind == "ocr"

    def build_chain(self, kind: str) -> ChainSpec:
        return ChainSpec(kind=kind, providers=[self.provider] if kind == "ocr" else [])


def _engine(tmp_path, provider: _Apple, *, route_label: str) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        daemon={"enabled": False},
        ocr={"enabled": True, "chain": ["apple_vision"], "augment_hierarchy": True},
    )
    store = AppMemoryStore(cfg.memory)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(P, "home", "apps", steps=[RouteStep(kind="tap", label=route_label)])
    device = _AdvancingDevice([HOME, APPS], package=P, serial=f"goto-{route_label}")
    return Engine(cfg, device=device, factory=_Factory(provider))  # type: ignore[arg-type]


def test_goto_does_not_run_ocr_when_hierarchy_matches_route(tmp_path) -> None:
    provider = _Apple("Apps")
    engine = _engine(tmp_path, provider, route_label="Apps")

    result = engine.goto("apps")

    assert result["ok"] is True
    assert provider.calls == 0
    assert {element["source"] for element in result["elements"] if "source" in element} == set()


def test_goto_retries_ocr_when_route_label_is_missing_from_hierarchy(tmp_path) -> None:
    provider = _Apple("Continue")
    engine = _engine(tmp_path, provider, route_label="Continue")

    result = engine.goto("apps")

    assert result["ok"] is True
    assert provider.calls == 1
    assert all(element.get("source") != Source.ocr.value for element in result["elements"])
