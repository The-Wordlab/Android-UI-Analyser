"""Durable cold-start Activity pin for multi-launcher builds.

Dev flavours often declare a product MAIN/LAUNCHER next to a Dev Tools one. Without a
pin, bare ``app launch`` coin-flips; with one, every later launch (and flag/proxy restart
fallback) is deterministic.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMemoryStore, _playbook_lines, launch_payload
from conftest import FakeDevice, make_config

PKG = "com.example.app.dev"
# Keep the test self-contained with FQNs under PKG so normalization is visible.
PRODUCT = f"{PKG}.ui.activity.launch.LaunchActivity"
DEVTOOLS = f"{PKG}.devtools.DevToolsActivity"
PRODUCT_SHORT = ".ui.activity.launch.LaunchActivity"


def _store(tmp_path: Path) -> AppMemoryStore:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    return AppMemoryStore(cfg.memory)


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(daemon={"enabled": False}, memory={"dir": str(tmp_path / "memory")})
    return Engine(cfg, device=device)


def test_remember_launch_entry_normalizes_and_pins(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = store.remember_launch_entry(PKG, PRODUCT_SHORT, source="user")
    assert entry is not None
    assert entry.activity == PRODUCT
    assert entry.source == "user"
    assert store.launch_activity(PKG) == PRODUCT


def test_user_pin_is_not_overwritten_by_resolved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.remember_launch_entry(PKG, PRODUCT, source="user")
    store.remember_launch_entry(PKG, DEVTOOLS, source="resolved")
    assert store.launch_activity(PKG) == PRODUCT


def test_record_launcher_activities_auto_pins_singleton(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = store.record_launcher_activities(PKG, [PRODUCT])
    assert entry is not None
    assert entry.source == "resolved"
    assert entry.activity == PRODUCT


def test_record_launcher_activities_surfaces_ambiguity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.record_launcher_activities(PKG, [PRODUCT, DEVTOOLS]) is None
    app = store.load(PKG)
    assert app is not None
    assert app.launch is None
    assert app.launcher_activities == [PRODUCT, DEVTOOLS]
    playbook = "\n".join(_playbook_lines(app))
    assert "launch AMBIGUOUS" in playbook
    assert DEVTOOLS in playbook


def test_launch_reuses_learned_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.remember_launch_entry(PKG, PRODUCT, source="user")
    dev = FakeDevice(package="com.android.launcher")
    eng = _engine(tmp_path, dev)
    res = eng.app("launch", package=PKG, observe=False)
    assert res.ok
    assert ("launch_app", (PKG, PRODUCT)) in dev.calls
    assert res.detail == f"{PKG}/{PRODUCT}"


def test_explicit_activity_teaches_the_map(tmp_path: Path) -> None:
    dev = FakeDevice(package="com.android.launcher")
    dev._launcher_activities = [PRODUCT, DEVTOOLS]
    eng = _engine(tmp_path, dev)
    res = eng.app("launch", package=PKG, activity=PRODUCT, observe=False)
    assert res.ok
    assert AppMemoryStore(eng.config.memory).launch_activity(PKG) == PRODUCT


def test_ambiguous_unpinned_launch_returns_a_note(tmp_path: Path) -> None:
    dev = FakeDevice(package="com.android.launcher")
    # Without a pin, FakeDevice.launch_app(activity=None) still fronts the package —
    # matching u2's unpinned app_start. The teach step then discovers the ambiguity.
    dev._launcher_activities = [PRODUCT, DEVTOOLS]
    eng = _engine(tmp_path, dev)
    res = eng.app("launch", package=PKG, observe=False)
    assert res.ok
    assert res.note is not None
    assert "multi-launcher" in res.note
    assert "remember --launch-activity" in res.note
    app = AppMemoryStore(eng.config.memory).load(PKG)
    assert app is not None
    assert app.launch is None
    assert set(app.launcher_activities) == {PRODUCT, DEVTOOLS}


def test_restart_falls_back_to_learned_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.remember_launch_entry(PKG, PRODUCT, source="user")
    dev = FakeDevice(package=PKG)
    eng = _engine(tmp_path, dev)
    # No mid-flow Activity → the restart must use the learned pin instead of an
    # unpinned platform resolve (the coin flip that opens Dev Tools).
    result = eng._restart_app(PKG, activity=None)
    assert result.ok
    assert ("launch_app", (PKG, PRODUCT)) in dev.calls
    assert ("launch_app", (PKG,)) not in dev.calls


def test_launch_payload_reports_the_pin(tmp_path: Path) -> None:
    # `about --format json` consumers branch on the field, not on prose in the playbook text.
    store = _store(tmp_path)
    store.remember_launch_entry(PKG, PRODUCT, source="user", alternatives=[DEVTOOLS])
    app = store.load(PKG)
    assert app is not None
    assert launch_payload(app) == {
        "launch": {"activity": PRODUCT, "source": "user", "alternatives": [DEVTOOLS]}
    }


def test_launch_payload_reports_ambiguity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_launcher_activities(PKG, [PRODUCT, DEVTOOLS])
    app = store.load(PKG)
    assert app is not None
    assert launch_payload(app) == {"launch_ambiguous": [PRODUCT, DEVTOOLS]}


def test_launch_payload_is_empty_when_there_is_nothing_to_say(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_launcher_activities(PKG, [])
    app = store.load(PKG)
    assert app is None or launch_payload(app) == {}


def test_orient_reports_a_pin_as_the_only_known_fact(tmp_path: Path) -> None:
    # A launch pin alone must make the app "known": otherwise `daemon start` reports known=False
    # and the agent re-discovers an ambiguity that was already resolved.
    store = _store(tmp_path)
    store.remember_launch_entry(PKG, PRODUCT, source="user", alternatives=[DEVTOOLS])
    eng = _engine(tmp_path, FakeDevice(package=PKG))
    out = eng.orient()
    assert out["known"] is True
    assert out["launch"] == {
        "activity": PRODUCT,
        "source": "user",
        "alternatives": [DEVTOOLS],
    }


def test_orient_surfaces_an_unresolved_ambiguity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_launcher_activities(PKG, [PRODUCT, DEVTOOLS])
    eng = _engine(tmp_path, FakeDevice(package=PKG))
    out = eng.orient()
    assert out["known"] is True
    assert out["launch_ambiguous"] == [PRODUCT, DEVTOOLS]
    assert "launch" not in out  # the two fields are mutually exclusive
