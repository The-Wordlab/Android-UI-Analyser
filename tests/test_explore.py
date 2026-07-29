"""Codebase deeplink mining (PRD §6b) — `aua explore mine`.

The miner walks an app source tree and records the deeplinks it declares (shortcuts the
agent can `aua open`). Tests use a tiny synthetic repo: a manifest declares the custom
scheme, main source holds real `myapp://` literals, a test source holds a throwaway URI
(must be skipped), and there are https/mailto URLs (must be ignored).
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.explore import mine_deeplinks
from android_ui_analyser.memory import AppMemoryStore
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

P = "com.example.app"

MANIFEST = """<?xml version="1.0"?>
<manifest package="com.example.app">
  <application>
    <activity>
      <intent-filter><action android:name="android.intent.action.VIEW"/>
        <data android:scheme="myapp"/>
      </intent-filter>
      <intent-filter><action android:name="android.intent.action.VIEW"/>
        <data android:scheme="myapp-test" android:host="set-flags"/>
      </intent-filter>
      <intent-filter><action android:name="android.intent.action.VIEW"/>
        <data android:scheme="https" android:host="example.com"/>
      </intent-filter>
    </activity>
  </application>
</manifest>
"""

NAV_KT = """
val routes = listOf(
    navDeepLink { uriPattern = "myapp://landing/tools" },
    navDeepLink { uriPattern = "myapp://landing/home" },
    navDeepLink { uriPattern = "myapp://dynamic_tools/{toolId}" },
    "https://example.com/help",           // web URL — not a deeplink
    "mailto:support@example.com",         // ignored scheme
)
"""

TEST_KT = 'val throwaway = "myapp://feed/123"  // example URI in a test — must be skipped'


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    (root / "feature/src/main").mkdir(parents=True)
    (root / "feature/src/test").mkdir(parents=True)
    (root / "app/src/main").mkdir(parents=True)
    (root / "app/src/main/AndroidManifest.xml").write_text(MANIFEST, encoding="utf-8")
    (root / "feature/src/main/Nav.kt").write_text(NAV_KT, encoding="utf-8")
    (root / "feature/src/test/NavTest.kt").write_text(TEST_KT, encoding="utf-8")
    # a build dir that must be skipped
    (root / "app/build").mkdir(parents=True)
    (root / "app/build/Generated.kt").write_text('"myapp://generated/junk"', encoding="utf-8")
    return root


def test_mine_finds_custom_deeplinks_only(tmp_path: Path) -> None:
    result = mine_deeplinks(_make_repo(tmp_path))
    uris = {d.uri for d in result.deeplinks}
    assert "myapp" in result.schemes and "myapp-test" in result.schemes
    assert "https" not in result.schemes
    assert "myapp://landing/tools" in uris
    assert "myapp://landing/home" in uris
    assert "myapp-test://set-flags" in uris  # reconstructed from manifest scheme+host
    # ignored / skipped
    assert not any(u.startswith("https") or u.startswith("mailto") for u in uris)
    assert "myapp://feed/123" not in uris  # from a test source → skipped
    assert "myapp://generated/junk" not in uris  # from build/ → skipped


def test_mine_flags_templated(tmp_path: Path) -> None:
    result = mine_deeplinks(_make_repo(tmp_path))
    tmpl = {d.uri: d.templated for d in result.deeplinks}
    assert tmpl["myapp://dynamic_tools/{toolId}"] is True
    assert tmpl["myapp://landing/tools"] is False


def test_mine_missing_dir_is_empty(tmp_path: Path) -> None:
    result = mine_deeplinks(tmp_path / "nope")
    assert result.deeplinks == [] and result.schemes == []


def test_engine_explore_mine_saves_to_playbook(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=FakeDevice(package=P), factory=ProviderFactory(cfg))
    out = eng.explore_mine(str(repo), package=P)
    assert out["ok"] and out["found"] >= 3 and out["saved"] == out["found"]

    app_map = AppMemoryStore(cfg.memory).load(P)
    saved = {d.uri for d in app_map.deeplinks}
    assert "myapp://landing/tools" in saved and "myapp-test://set-flags" in saved
    # the note records provenance
    tools = next(d for d in app_map.deeplinks if d.uri == "myapp://landing/tools")
    assert "mined" in (tools.note or "")


def test_engine_explore_mine_no_save(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=FakeDevice(package=P), factory=ProviderFactory(cfg))
    out = eng.explore_mine(str(repo), package=P, save=False)
    assert out["found"] >= 3 and out["saved"] == 0
    assert AppMemoryStore(cfg.memory).load(P) is None  # nothing written


# --------------------------------------------------------------- explore plan (agent worklist)


def _eng(tmp_path: Path):
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=FakeDevice(package=P), factory=ProviderFactory(cfg)), cfg


def test_explore_plan_bootstrap_when_empty(tmp_path: Path) -> None:
    eng, _ = _eng(tmp_path)
    out = eng.explore_plan(package=P)
    assert out["ok"] and out["tasks"] == []
    assert "mine deeplinks" in out["bootstrap"]


def test_explore_plan_lists_unprobed_deeplinks(tmp_path: Path) -> None:
    eng, cfg = _eng(tmp_path)
    store = AppMemoryStore(cfg.memory)
    store.remember_deeplink(P, "myapp://pet", note="mined")  # unprobed, concrete
    store.remember_deeplink(P, "myapp://tools/{toolId}", note="mined")  # templated
    store.remember_deeplink(P, "myapp://home", note="mined", probed=True)  # already probed
    out = eng.explore_plan(package=P)
    kinds = [t["kind"] for t in out["tasks"]]
    dos = " ".join(t["do"] for t in out["tasks"])
    assert "probe_deeplink" in kinds and "myapp://pet" in dos
    assert "probe_template" in kinds  # templated one flagged separately
    assert "myapp://home" not in dos  # probed → not re-suggested


def test_explore_plan_flags_dead_end_screens(tmp_path: Path) -> None:
    from test_memory import HOME, _elements

    eng, cfg = _eng(tmp_path)
    store = AppMemoryStore(cfg.memory)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")  # no routes out
    out = eng.explore_plan(package=P)
    assert any(t["kind"] == "expand_screen" and "home" in t["do"] for t in out["tasks"])


def test_open_link_marks_deeplink_probed(tmp_path: Path) -> None:
    from test_memory import HOME
    from test_navigation import ScriptedDevice

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    dev = ScriptedDevice([HOME], package=P, serial="emu-probe")
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    eng.analyze(source="hierarchy")  # seed cached package
    eng.open_link("myapp://pet")
    dl = next(d for d in AppMemoryStore(cfg.memory).load(P).deeplinks if d.uri == "myapp://pet")
    assert dl.probed is True
