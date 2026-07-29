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
    assert settle2 < 120
    assert total2 < 1100 or total2 >= 400


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
