"""Old maps retire themselves when AUA's map schema moves past what they were learned under.

Every install that already has a map must start building the new one on its own, without a
human clearing anything: what was *learned* (screens, routes, contexts, research) is set aside
in an archive file and rebuilt from scratch; what was *taught* (knowledge, deeplinks, recipes,
notes, launch, vocabulary) is kept, because nobody wants to re-teach an app.
"""

from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser.memory import (
    MEMORY_LEARNING_FLOOR,
    MEMORY_SCHEMA_VERSION,
    AppMap,
    AppMemoryStore,
)
from conftest import make_config
from test_memory import HOME, P, _elements


def _store(tmp_path: Path, **memov: object) -> AppMemoryStore:
    cfg = make_config(memory={"dir": str(tmp_path / "home"), **memov})
    return AppMemoryStore(cfg.memory)


def _map_written_by_an_older_aua(store: AppMemoryStore) -> AppMap:
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    app = store.load(P)
    assert app is not None
    app.notes.append("the gear opens settings")
    app.description = "a shop"
    store.save(app)
    raw = json.loads(store.index_path(P).read_text(encoding="utf-8"))
    raw["schema_version"] = MEMORY_LEARNING_FLOOR - 1
    store.index_path(P).write_text(json.dumps(raw), encoding="utf-8")
    return AppMap.model_validate(raw)


def test_a_map_learned_under_an_older_schema_is_rebuilt_but_keeps_what_was_taught(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    old = _map_written_by_an_older_aua(store)
    assert old.screens and old.schema_version < MEMORY_LEARNING_FLOOR

    app = store.load(P)
    assert app is not None
    assert app.schema_version == MEMORY_SCHEMA_VERSION
    assert app.screens == {} and app.routes == [] and app.research_tasks == []
    assert app.notes == ["the gear opens settings"] and app.description == "a shop"

    # Nothing is lost, just set aside, and the file on disk is already the new version.
    archive = store.app_dir(P) / f"index.v{old.schema_version}.json"
    assert json.loads(archive.read_text(encoding="utf-8"))["screens"]
    on_disk = json.loads(store.index_path(P).read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == MEMORY_SCHEMA_VERSION and on_disk["screens"] == {}

    # Once. Later loads leave the archive alone and keep what has been learned since.
    stamp = archive.stat().st_mtime_ns
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    again = store.load(P)
    assert again is not None and again.screens
    assert archive.stat().st_mtime_ns == stamp


def test_a_current_map_is_left_alone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    app = store.load(P)
    assert app is not None and app.screens
    assert not list(store.app_dir(P).glob("index.v*.json"))


def test_the_sqlite_backend_retires_the_same_way(tmp_path: Path) -> None:
    store = _store(tmp_path, backend="sqlite", sqlite_path=str(tmp_path / "memory.db"))
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    app = store.load(P)
    assert app is not None and app.screens
    app.notes.append("taught")
    app.schema_version = MEMORY_LEARNING_FLOOR - 1
    assert store._sqlite is not None
    store._sqlite.save_app(app)  # write it the way the previous AUA would have

    fresh = store.load(P)
    assert fresh is not None
    assert fresh.screens == {} and fresh.notes == ["taught"]
    assert fresh.schema_version == MEMORY_SCHEMA_VERSION
    assert (store.app_dir(P) / f"index.v{MEMORY_LEARNING_FLOOR - 1}.json").is_file()
