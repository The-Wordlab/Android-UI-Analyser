"""SQLite memory backend round-trip + JSON → SQLite migration."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.memory import AppMap, AppMemoryStore, RouteStep, ScreenRecord, SessionState
from conftest import make_config

P = "com.example.app"


def _screen(name: str = "home") -> ScreenRecord:
    now = "2026-01-01T00:00:00+00:00"
    return ScreenRecord(
        name=name,
        signature="abc123def456",
        anchors=["id:header", "tx:home"],
        first_seen=now,
        last_seen=now,
        last_verified=now,
    )


def test_sqlite_app_and_session_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    cfg = make_config(
        memory={
            "dir": str(tmp_path / "home"),
            "backend": "sqlite",
            "sqlite_path": str(db),
        }
    )
    store = AppMemoryStore(cfg.memory)

    app = AppMap(package=P, label="Example App", screens={"home": _screen()})
    store.save(app)
    loaded = store.load(P)
    assert loaded is not None
    assert loaded.package == P
    assert loaded.label == "Example App"
    assert "home" in loaded.screens
    assert store.list_apps() == [P]

    sess = SessionState(
        package=P,
        current_screen="home",
        pending=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")],
        recent=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")],
    )
    store.save_session("emu-1", sess)
    got = store.load_session("emu-1")
    assert got.package == P
    assert got.current_screen == "home"
    assert len(got.pending) == 1
    assert got.pending[0].resource_id == "nav_apps"

    # Fresh store against the same DB recovers the same blobs.
    store2 = AppMemoryStore(cfg.memory)
    assert store2.load(P) is not None
    assert store2.load(P).screens["home"].name == "home"
    assert store2.load_session("emu-1").current_screen == "home"
    assert db.is_file()


def test_sqlite_migrates_legacy_json_index(tmp_path: Path) -> None:
    home = tmp_path / "home"
    pkg_dir = home / "memory" / P
    pkg_dir.mkdir(parents=True)
    app = AppMap(package=P, label="Legacy", screens={"home": _screen()})
    (pkg_dir / "index.json").write_text(app.model_dump_json(indent=2), encoding="utf-8")

    db = tmp_path / "migrated.db"
    cfg = make_config(
        memory={
            "dir": str(home),
            "backend": "sqlite",
            "sqlite_path": str(db),
        }
    )
    store = AppMemoryStore(cfg.memory)
    loaded = store.load(P)
    assert loaded is not None
    assert loaded.label == "Legacy"
    assert store.list_apps() == [P]
