"""Codebase deeplink mining (PRD §6b) — `aua explore mine`.

The miner walks an app source tree and records the deeplinks it declares (shortcuts the
agent can `aua open`). Tests use a tiny synthetic repo: a manifest declares the custom
scheme, main source holds real `luzia://` literals, a test source holds a throwaway URI
(must be skipped), and there are https/mailto URLs (must be ignored).
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.explore import mine_deeplinks
from android_ui_analyser.memory import AppMemoryStore
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

P = "co.thewordlab.luzia"

MANIFEST = """<?xml version="1.0"?>
<manifest package="co.thewordlab.luzia">
  <application>
    <activity>
      <intent-filter><action android:name="android.intent.action.VIEW"/>
        <data android:scheme="luzia"/>
      </intent-filter>
      <intent-filter><action android:name="android.intent.action.VIEW"/>
        <data android:scheme="luzia-test" android:host="set-flags"/>
      </intent-filter>
      <intent-filter><action android:name="android.intent.action.VIEW"/>
        <data android:scheme="https" android:host="luzia.co"/>
      </intent-filter>
    </activity>
  </application>
</manifest>
"""

NAV_KT = """
val routes = listOf(
    navDeepLink { uriPattern = "luzia://landing/tools" },
    navDeepLink { uriPattern = "luzia://landing/home" },
    navDeepLink { uriPattern = "luzia://dynamic_tools/{toolId}" },
    "https://luzia.co/help",           // web URL — not a deeplink
    "mailto:support@luzia.co",         // ignored scheme
)
"""

TEST_KT = 'val throwaway = "luzia://feed/123"  // example URI in a test — must be skipped'


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
    (root / "app/build/Generated.kt").write_text('"luzia://generated/junk"', encoding="utf-8")
    return root


def test_mine_finds_custom_deeplinks_only(tmp_path: Path) -> None:
    result = mine_deeplinks(_make_repo(tmp_path))
    uris = {d.uri for d in result.deeplinks}
    assert "luzia" in result.schemes and "luzia-test" in result.schemes
    assert "https" not in result.schemes
    assert "luzia://landing/tools" in uris
    assert "luzia://landing/home" in uris
    assert "luzia-test://set-flags" in uris  # reconstructed from manifest scheme+host
    # ignored / skipped
    assert not any(u.startswith("https") or u.startswith("mailto") for u in uris)
    assert "luzia://feed/123" not in uris  # from a test source → skipped
    assert "luzia://generated/junk" not in uris  # from build/ → skipped


def test_mine_flags_templated(tmp_path: Path) -> None:
    result = mine_deeplinks(_make_repo(tmp_path))
    tmpl = {d.uri: d.templated for d in result.deeplinks}
    assert tmpl["luzia://dynamic_tools/{toolId}"] is True
    assert tmpl["luzia://landing/tools"] is False


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
    assert "luzia://landing/tools" in saved and "luzia-test://set-flags" in saved
    # the note records provenance
    tools = next(d for d in app_map.deeplinks if d.uri == "luzia://landing/tools")
    assert "mined" in (tools.note or "")


def test_engine_explore_mine_no_save(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=FakeDevice(package=P), factory=ProviderFactory(cfg))
    out = eng.explore_mine(str(repo), package=P, save=False)
    assert out["found"] >= 3 and out["saved"] == 0
    assert AppMemoryStore(cfg.memory).load(P) is None  # nothing written
