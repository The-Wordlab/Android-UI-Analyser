"""A log preference set once must still be there in the next session, and the next project.

An agent learns which of an app's tags are noise the expensive way: by paying for them in
every observation until it notices. Paying that tuition once per session is the whole failure
this guards against, so the preference is *persisted per app*, beside that app's map, under
``memory.dir`` — never in this repository, because an app's own tag names are private product
knowledge.

Stored preferences are deliberately independent of ``memory.enabled``: turning learning off
says "do not record what you discover", not "discard what I explicitly told you".
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.memory import AppLogPrefs, AppMemoryStore
from conftest import make_config

APP = "com.example.notes"


def _store(tmp_path: Path, **memov: object) -> AppMemoryStore:
    cfg = make_config(memory={"dir": str(tmp_path / "home"), **memov})
    return AppMemoryStore(cfg.memory)


def test_an_app_nobody_configured_reads_back_as_no_preference_not_an_error(tmp_path: Path) -> None:
    assert _store(tmp_path).load_log_prefs(APP) is None


def test_a_preference_written_by_one_process_is_read_by_the_next(tmp_path: Path) -> None:
    # Two store objects stand in for two `aua` invocations: every CLI call is a fresh process,
    # so anything held only in memory is the same as never having been set.
    _store(tmp_path).save_log_prefs(
        AppLogPrefs(package=APP, ignore_tags=["ChattyThing"], limit=40)
    )

    again = _store(tmp_path).load_log_prefs(APP)

    assert again is not None
    assert again.ignore_tags == ["ChattyThing"]
    assert again.limit == 40


def test_the_preference_file_sits_beside_that_app_s_map_under_the_memory_dir(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save_log_prefs(AppLogPrefs(package=APP, only_tags=["Checkout"]))

    path = store.log_prefs_path(APP)

    assert path.is_file()
    assert path.parent == store.app_dir(APP)
    assert store.base in path.parents, "nothing may escape memory.dir"


def test_a_corrupt_preference_file_degrades_to_no_preference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_log_prefs(AppLogPrefs(package=APP, limit=40))
    store.log_prefs_path(APP).write_text("{ not json", encoding="utf-8")

    assert store.load_log_prefs(APP) is None, "a half-written file must not break every action"


def test_a_preference_round_trips_when_sqlite_holds_the_map(tmp_path: Path) -> None:
    # The map moves into sqlite; the preference is a small hand-written document, so it stays a
    # file in both backends — the same call has to answer for both.
    memov = {"backend": "sqlite", "sqlite_path": str(tmp_path / "memory.db")}
    _store(tmp_path, **memov).save_log_prefs(AppLogPrefs(package=APP, only_tags=["Checkout"]))

    again = _store(tmp_path, **memov).load_log_prefs(APP)

    assert again is not None and again.only_tags == ["Checkout"]


def test_forgetting_an_app_forgets_what_it_wanted_from_its_logs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_log_prefs(AppLogPrefs(package=APP, limit=40))

    store.forget(APP)

    assert store.load_log_prefs(APP) is None
