"""A double-sampled tree settle earns its confirmation exemption only by growing.

`via=hierarchy` — two identical tree dumps ~60ms apart — was exempt from the extended
stability confirmation on the grounds that double-sampling is stronger evidence than the
single-sample exits. The recorded failure shows the hole: during an Activity transition
uiautomator kept serving the *old* window's tree minus one label, twice, while the new
Activity waited on the network. Two agreeing samples of a frozen transitional frame are not
arrival evidence.

The discriminator is growth. A tree that gained parts relative to the pre-action tree has
rendered something; a tree that only *lost* parts has proven departure, nothing more. So the
settle loop now reports `tree_added`, and a shrink-only `via=hierarchy` joins the exits that
get the longer quiet window before analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser import imaging
from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

PKG = "com.example.fiction"


def _screen(*rows: str) -> str:
    return '<hierarchy rotation="0">' + "".join(rows) + "</hierarchy>"


def _node(text: str, *, rid: str = "", y: int, clickable: bool = False) -> str:
    return (
        f'<node class="android.view.View" package="{PKG}" text="{text}"'
        f' resource-id="{rid}" clickable="{str(clickable).lower()}" enabled="true"'
        f' bounds="[20,{y}][500,{y + 80}]"/>'
    )


HOME = _screen(
    _node("Welcome back", rid=f"{PKG}:id/entryTitle", y=20),
    _node("Continue with MegaID", rid=f"{PKG}:id/loginBtn", y=120, clickable=True),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
)
# One label lost, nothing gained: the recorded transitional shape.
SHRUNK = _screen(
    _node("Welcome back", rid=f"{PKG}:id/entryTitle", y=20),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
)


def test_the_gate_confirms_a_shrink_only_hierarchy_settle() -> None:
    ready = {"changed": True, "timeout": False, "via": "hierarchy", "tree_added": 0}
    assert Engine._tap_settle_needs_confirmation("tap", ready) is True


def test_a_grown_hierarchy_settle_keeps_its_exemption() -> None:
    ready = {"changed": True, "timeout": False, "via": "hierarchy", "tree_added": 3}
    assert Engine._tap_settle_needs_confirmation("tap", ready) is False


def test_a_legacy_ready_without_the_count_keeps_the_old_behaviour() -> None:
    """Monkeypatched settles in older tests carry no `tree_added`; they must not regress."""
    ready = {"changed": True, "timeout": False, "via": "hierarchy"}
    assert Engine._tap_settle_needs_confirmation("tap", ready) is False


def test_the_settle_loop_reports_tree_growth(tmp_path: Path, monkeypatch: Any) -> None:
    """`_await_post_action_ready` itself must stamp the count on a hierarchy exit."""

    class ShrinkOnTap(FakeDevice):
        def click(self, x: int, y: int) -> None:
            super().click(x, y)
            self._xml = SHRUNK

    dev = ShrinkOnTap(hierarchy_xml=HOME, package=PKG)
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perf={"stable_delay_ms": {"default": 0}},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    eng.analyze(source="hierarchy")  # seeds the pre-action tree fingerprint

    # Force the pixel path to stay busy so the loop must decide on the tree: never visually
    # idle, and every frame differs from the pre-action signature.
    monkeypatch.setattr(imaging.GridSettle, "feed", lambda self, img: False)
    monkeypatch.setattr(imaging, "frames_differ", lambda a, b, **kw: True)

    confirmations: list[dict[str, Any]] = []

    def confirm(**kwargs: Any) -> ActionResult:
        confirmations.append(kwargs)
        return ActionResult(ok=True, action="wait-stable")

    monkeypatch.setattr(eng, "wait_stable", confirm)

    login = f"{PKG}:id/loginBtn"
    out = eng.tap(selector={"rid": login}, observe=True)

    assert out.settle is not None, f"expected a settle report: {out.model_dump(exclude_none=True)}"
    assert out.settle.get("via") == "hierarchy", f"fixture drifted: {out.settle}"
    assert confirmations, "a shrink-only double-sampled settle must be confirmed before analyze"
