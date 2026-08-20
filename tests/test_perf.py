"""Unit tests for latency helpers (prefetch, diffs, settle profiles, gate cache)."""

from __future__ import annotations

import time

from android_ui_analyser.perf import (
    GateCache,
    HierarchyPrefetch,
    SettleProfiles,
    element_diff,
    elements_fingerprint,
)
from android_ui_analyser.schema import Element, Source


def _el(eid: int, text: str, *, rid: str | None = None) -> Element:
    return Element(
        id=eid,
        type="TextView",
        text=text,
        resource_id=rid,
        content_desc=None,
        bounds=(0, eid * 10, 100, eid * 10 + 8),
        center=(50, eid * 10 + 4),
        clickable=True,
        enabled=True,
        focused=False,
        source=Source.hierarchy,
    )


def test_hierarchy_prefetch_take_and_invalidate() -> None:
    pref = HierarchyPrefetch(max_age_ms=500)
    calls = {"n": 0}

    def dump() -> str:
        calls["n"] += 1
        return "<hierarchy/>"

    def parse(xml: str) -> tuple[list[Element], str | None]:
        return [_el(0, "Hi")], "com.x"

    pref.kick(dump, parse)
    deadline = time.monotonic() + 2.0
    slot = None
    while time.monotonic() < deadline:
        slot = pref.take()
        if slot is not None:
            break
        time.sleep(0.01)
    assert slot is not None
    assert slot.package == "com.x"
    assert calls["n"] == 1
    assert pref.take() is None  # consumed

    pref.kick(dump, parse)
    time.sleep(0.05)
    pref.invalidate()
    assert pref.take() is None


def test_settle_profiles_learn() -> None:
    sp = SettleProfiles()
    settle, total = sp.budget("tap")
    assert settle == 45
    assert total == 1100
    sp.observe("tap", 200)
    sp.observe("tap", 200)
    settle2, total2 = sp.budget("tap")
    # Settle window stays short (same-screen taps must not inherit transition idles).
    assert settle2 == 45
    assert 400 <= total2 <= 1600


def test_await_same_tree_exits_hierarchy_same() -> None:
    """In-screen taps: tree unchanged → via=hierarchy-same, not a long pixel settle."""
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice, make_config, make_png

    xml = """<?xml version='1.0'?>
    <hierarchy>
      <node class="android.widget.TextView" text="Hello" clickable="true"
            bounds="[0,0][100,50]" package="com.x"/>
      <node class="android.widget.TextView" text="World" clickable="true"
            bounds="[0,60][100,110]" package="com.x"/>
    </hierarchy>"""
    # Alternating frames so GridSettle never goes idle (ripple / spinner noise).
    a = make_png(100, 120, color=(240, 240, 240), boxes=[((0, 0, 20, 20), (0, 0, 0))])
    b = make_png(100, 120, color=(240, 240, 240), boxes=[((80, 0, 100, 20), (0, 0, 0))])
    cfg = make_config(daemon={"enabled": False})
    dev = FakeDevice(hierarchy_xml=xml, screenshots=[a, b] * 80, width=100, height=120)
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    eng.analyze(source="hierarchy", record=False)
    # Pre-action tree matches what dump_hierarchy still returns (same-screen tap).
    eng._pre_action_tree_fp = eng._tree_fingerprint()
    eng._pre_action_sig = None  # skip pixel identical path; tree should short-circuit
    t0 = time.perf_counter()
    ready = eng._await_post_action_ready(settle_ms=200, total_timeout_ms=2000, poll_ms=5)
    elapsed = (time.perf_counter() - t0) * 1000
    assert ready["via"] == "hierarchy-same", ready
    assert ready["changed"] is False
    assert elapsed < 600, f"same-tree settle took {elapsed:.0f}ms"


def test_gate_cache_roundtrip() -> None:
    cache = GateCache()
    els = [_el(0, "A"), _el(1, "B")]
    key = GateCache.key(els, package="com.x", activity=".Main")
    assert cache.get(key) is None
    cache.put(key, ("vision", "too few"))
    assert cache.get(key) == ("vision", "too few")


def test_element_diff_and_fingerprint() -> None:
    a = [_el(0, "Hello", rid="app:id/t"), _el(1, "Bye")]
    b = [_el(0, "Hello!", rid="app:id/t"), _el(2, "New")]
    fp_a = elements_fingerprint(a)
    fp_b = elements_fingerprint(b)
    assert fp_a != fp_b
    diff = element_diff(a, b)
    assert diff["curr_count"] == 2
    assert diff["prev_count"] == 2
    assert any(c.get("text") for c in diff["changed"]) or diff["added"] or diff["removed"]
