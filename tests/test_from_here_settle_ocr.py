"""Mid-edge goto --from-here, smarter settle, and map-based OCR skip."""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import (
    AppMemoryStore,
    RouteStep,
    ScreenRecord,
    screen_skips_ocr,
)
from android_ui_analyser.providers.base import (
    Availability,
    ChainSpec,
    OcrProvider,
    ScreenImage,
    TextBox,
)
from conftest import FakeDevice, make_config
from test_memory import APPS, HOME, P, _elements
from test_navigation import IMAGES, ScriptedDevice


class _Apple(OcrProvider):
    name = "apple_vision"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def is_available(self) -> Availability:
        return Availability(True, "test apple vision")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        self.calls += 1
        return [TextBox(text="noise", bounds=(10, 10, 40, 30), confidence=0.9)]


class _Factory:
    def __init__(self, provider: OcrProvider) -> None:
        self.provider = provider

    def is_enabled(self, kind: str) -> bool:
        return kind == "ocr"

    def build_chain(self, kind: str) -> ChainSpec:
        return ChainSpec(kind=kind, providers=[self.provider] if kind == "ocr" else [])


def test_screen_skips_ocr_requires_evidence() -> None:
    base = ScreenRecord(
        name="home",
        signature="x",
        first_seen="t",
        last_seen="t",
        last_verified="t",
        surface="native",
        hierarchy_only_ok=2,
        ocr_helped=0,
    )
    assert screen_skips_ocr(base) is False
    base.hierarchy_only_ok = 3
    assert screen_skips_ocr(base) is True
    base.ocr_helped = 1
    assert screen_skips_ocr(base) is False
    base.ocr_helped = 0
    base.surface = "canvas"
    assert screen_skips_ocr(base) is False


def test_goto_from_here_skips_matched_prefix(tmp_path) -> None:
    """Multi-step edge; already on Apps → --from-here resumes at tap Images."""
    mem_dir = tmp_path / "home"
    cfg = make_config(memory={"dir": str(mem_dir)}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_screen(package=P, elements=_elements(IMAGES), name_hint="images")
    store.record_route(
        P,
        "home",
        "images",
        steps=[
            RouteStep(kind="tap", label="Apps", resource_id="nav_apps"),
            RouteStep(kind="tap", label="Images", resource_id="tool_images"),
        ],
    )
    # Mid-edge: session still says "home", but the device already shows Apps.
    device = ScriptedDevice([APPS, IMAGES], package=P, serial="from-here")
    store.save_session(
        "from-here",
        store.load_session("from-here").model_copy(
            update={"package": P, "current_screen": "home"}
        ),
    )
    engine = Engine(cfg, device=device)

    clicks_before = len([c for c in device.calls if c[0] == "click"])
    result = engine.goto("images", from_here=True)
    clicks_after = len([c for c in device.calls if c[0] == "click"])

    assert result["ok"] is True
    assert result.get("arrived") is True
    assert clicks_after - clicks_before == 1  # only Images — Apps already done


def test_map_skips_ocr_after_enough_hierarchy_only_visits(tmp_path) -> None:
    mem_dir = tmp_path / "home"
    cfg = make_config(
        memory={"dir": str(mem_dir)},
        daemon={"enabled": False},
        ocr={"enabled": True, "chain": ["apple_vision"], "augment_hierarchy": True},
    )
    provider = _Apple()
    store = AppMemoryStore(cfg.memory)
    name = store.record_screen(
        package=P, elements=_elements(HOME), name_hint="home", ocr_helped=False
    ).name
    app = store.load(P)
    assert app is not None
    rec = app.screens[name]
    rec.hierarchy_only_ok = 5
    rec.ocr_helped = 0
    rec.surface = "native"
    store.save(app)

    device = FakeDevice(hierarchy_xml=HOME, package=P, serial="ocr-skip")
    engine = Engine(cfg, device=device, factory=_Factory(provider))  # type: ignore[arg-type]

    engine.analyze(source="hierarchy")
    assert provider.calls == 0


def test_settle_for_next_step_uses_has_not_wait_stable(tmp_path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    # FakeDevice.find_text uses text_index / resource_index (not the XML dump).
    device = FakeDevice(
        hierarchy_xml=HOME,
        package=P,
        text_index={"Apps": (40, 300, 1040, 400)},
        resource_index={"nav_apps": (40, 300, 1040, 400)},
    )
    engine = Engine(cfg, device=device)

    stable_calls = {"n": 0}
    orig = engine.wait_stable

    def _counting(**kwargs):
        stable_calls["n"] += 1
        return orig(**kwargs)

    engine.wait_stable = _counting  # type: ignore[method-assign]

    nxt = RouteStep(kind="tap", label="Apps")
    assert engine._settle_for_next_step(nxt) is True
    assert stable_calls["n"] == 0
    by_id = RouteStep(kind="tap", resource_id="nav_apps")
    assert engine._settle_for_next_step(by_id) is True
    assert engine._settle_for_next_step(None) is False
