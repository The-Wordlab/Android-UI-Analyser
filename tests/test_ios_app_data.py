"""Simulator app data is platform-owned, scoped to its container, and recoverable."""

from __future__ import annotations

import json
import plistlib
import sqlite3

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml
from android_ui_analyser.platforms.ios_files import IOSAppFiles
from test_ios_platform import APP_ID, UDID
from test_ios_platform import adapter as adapter
from test_ios_platform import host as host


def test_preference_flow_preserves_the_explicit_ios_plist_name():
    flow = parse_flow_yaml(
        "name: fixture\nsteps:\n  - prefs_write: {file: com.example.fixture.plist, values: {fixture_flag: true}}"
    )
    assert flow.steps[0].arg == "com.example.fixture.plist"


@pytest.fixture
def container(tmp_path, host):
    root = tmp_path / "Containers" / "Data" / "Application" / "fixture"
    root.mkdir(parents=True)
    host.container = root
    (root / "Documents").mkdir()
    connection = sqlite3.connect(root / "Documents" / "fixture.sqlite")
    connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT NOT NULL)")
    connection.executemany("INSERT INTO items VALUES (?,?)", [(1, "original"), (2, "second")])
    connection.commit()
    connection.close()
    return root


@pytest.fixture
def defaults_import(monkeypatch, host, container):
    original = host._simctl

    def simctl(argv, args, input_bytes):
        if args[:4] == ("spawn", UDID, "defaults", "export"):
            from pathlib import Path

            path = Path(args[4] + ".plist")
            assert path.is_relative_to(container)
            return (
                host._ok(argv, path.read_bytes())
                if path.exists()
                else host._fail(argv, "Domain not found")
            )
        if args[:4] == ("spawn", UDID, "defaults", "import"):
            from pathlib import Path

            assert args[-1] == "-"
            path = Path(args[4] + ".plist")
            assert path.is_relative_to(container)
            plistlib.loads(input_bytes)
            path.write_bytes(input_bytes)
            return host._ok(argv)
        return original(argv, args, input_bytes)

    monkeypatch.setattr(host, "_simctl", simctl)


def test_database_reads_include_live_wal_and_do_not_stop_app(adapter, host, container):
    service = adapter.capability("app_database")
    runtime = adapter.connect(UDID)
    writer = sqlite3.connect(container / "Documents" / "fixture.sqlite")
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO items VALUES (3, 'wal')")
        writer.commit()
        listed = service.list_databases(runtime, APP_ID)
        assert [row["name"] for row in listed["databases"]] == ["Documents/fixture.sqlite"]
        result = service.query_database(
            runtime,
            APP_ID,
            "Documents/fixture.sqlite",
            "SELECT label FROM items WHERE id=?",
            parameters=[3],
        )
        assert result["rows"] == [["wal"]]
        assert result["app_stopped"] is False
        assert not host.argv_of("xcrun", "simctl", "terminate")
        limited = service.query_database(
            runtime, APP_ID, "Documents/fixture.sqlite", "SELECT * FROM items", limit=1
        )
        assert limited["truncated"] and limited["row_count"] == 1
    finally:
        writer.close()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM items",
        "ATTACH ':memory:' AS extra",
        "PRAGMA writable_schema=ON",
        "SELECT load_extension('nothing')",
        "SELECT 1; DELETE FROM items",
    ],
)
def test_queries_cannot_mutate_or_escape_the_snapshot(adapter, container, sql):
    runtime = adapter.connect(UDID)
    with pytest.raises(UsageError):
        adapter.capability("app_database").query_database(
            runtime, APP_ID, "Documents/fixture.sqlite", sql
        )
    with sqlite3.connect(container / "Documents" / "fixture.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM items").fetchone() == (2,)


@pytest.mark.parametrize("relative", ["../secret", "/tmp/secret", ".", "Documents/../../secret"])
def test_container_paths_reject_traversal(adapter, container, relative):
    with pytest.raises(UsageError):
        IOSAppFiles(adapter.tools, UDID).path(APP_ID, relative)


def test_container_paths_reject_escaping_symlinks(adapter, container, tmp_path):
    (container / "escape").symlink_to(tmp_path)
    with pytest.raises(UsageError):
        IOSAppFiles(adapter.tools, UDID).path(APP_ID, "escape/secret")


def test_database_mutations_require_confirmation_and_restore_their_backup(
    adapter, container, tmp_path
):
    service = adapter.capability("app_database")
    runtime = adapter.connect(UDID)
    args = (runtime, tmp_path, APP_ID, "Documents/fixture.sqlite")
    with pytest.raises(UsageError, match="--yes"):
        service.execute_database(*args, "UPDATE items SET label='changed'")
    result = service.execute_database(
        *args, "UPDATE items SET label=? WHERE id=1", parameters=["changed"], confirmed=True
    )
    assert result["rows_affected"] == 1

    def query():
        return service.query_database(
            runtime, APP_ID, "Documents/fixture.sqlite", "SELECT label FROM items ORDER BY id"
        )["rows"]

    assert query() == [["changed"], ["second"]]
    service.restore_database(*args, result["backup_id"], confirmed=True)
    assert query() == [["original"], ["second"]]
    assert len(service.list_backups(*args)["backups"]) == 2


def test_failed_database_mutation_rolls_back(adapter, container, tmp_path):
    runtime = adapter.connect(UDID)
    service = adapter.capability("app_database")
    with pytest.raises(UsageError, match="rolled back"):
        service.execute_database(
            runtime,
            tmp_path,
            APP_ID,
            "Documents/fixture.sqlite",
            "UPDATE items SET label=NULL",
            confirmed=True,
        )
    assert service.query_database(
        runtime, APP_ID, "Documents/fixture.sqlite", "SELECT label FROM items WHERE id=1"
    )["rows"] == [["original"]]


def test_database_schema_matches_the_shared_dashboard_contract(adapter, container):
    service = adapter.capability("app_database")
    runtime = adapter.connect(UDID)
    result = service.database_schema(runtime, APP_ID, "Documents/fixture.sqlite", table="items")
    assert result["count"] == 1
    assert result["objects"][0]["name"] == "items"
    assert [column["name"] for column in result["objects"][0]["columns"]] == ["id", "label"]
    assert result["objects"][0]["columns"][1]["not_null"] is True
    with pytest.raises(UsageError, match="does not exist"):
        service.database_schema(runtime, APP_ID, "Documents/fixture.sqlite", table="absent")


def test_preferences_are_typed_filtered_and_restored(adapter, container, defaults_import, tmp_path):
    runtime = adapter.connect(UDID)
    service = adapter.capability("feature_flags")
    path = container / "Library" / "Preferences" / f"{APP_ID}.plist"
    path.parent.mkdir(parents=True)
    path.write_bytes(plistlib.dumps({"fixture_flag": False, "untouched": [1, 2]}))
    snapshot = service.snapshot_prefs(runtime, APP_ID, APP_ID)
    backup = service.save_prefs_backup(tmp_path, UDID, snapshot)
    result = service.write_prefs(
        runtime, snapshot, {"fixture_flag": True, "count": 2, "ratio": 1.5}
    )
    assert result["ok"] and result["verified"]
    assert plistlib.loads(path.read_bytes())["untouched"] == [1, 2]
    assert service.read_prefs(runtime, APP_ID, {"fixture_flag": "true"}).applied == {
        "fixture_flag": "true"
    }
    assert service.read_context_flags(runtime, APP_ID, keys=["fixture_flag"]).flags == {
        "fixture_flag": "true"
    }
    assert service.read_context_flags(runtime, APP_ID).flags == {}
    service.restore_prefs(runtime, backup)
    assert plistlib.loads(path.read_bytes()) == {"fixture_flag": False, "untouched": [1, 2]}


def test_preference_readback_uses_committed_values_before_the_plist_is_flushed(
    adapter,
    host,
    container,
    defaults_import,
    monkeypatch,
):
    runtime = adapter.connect(UDID)
    original = host._simctl

    def simctl(argv, args, input_bytes):
        if args[:4] == ("spawn", UDID, "defaults", "export"):
            return host._ok(argv, plistlib.dumps({"fixture_flag": "committed"}))
        return original(argv, args, input_bytes)

    monkeypatch.setattr(host, "_simctl", simctl)
    service = adapter.capability("feature_flags")
    result = service.read_prefs(runtime, APP_ID, {"fixture_flag": "committed"})
    assert result.applied == {"fixture_flag": "committed"}
    assert not result.ignored and not result.mismatched
    snapshot = service.snapshot_prefs(runtime, APP_ID, APP_ID)
    assert snapshot.existed and plistlib.loads(snapshot.data) == {"fixture_flag": "committed"}


def test_absent_preferences_are_removed_on_restore(adapter, container, defaults_import, tmp_path):
    runtime = adapter.connect(UDID)
    service = adapter.capability("feature_flags")
    snapshot = service.snapshot_prefs(runtime, APP_ID, APP_ID)
    backup = service.save_prefs_backup(tmp_path, UDID, snapshot)
    assert service.write_prefs(runtime, snapshot, {"fixture_flag": "enabled"})["ok"]
    service.restore_prefs(runtime, backup)
    assert not (container / "Library" / "Preferences" / f"{APP_ID}.plist").exists()


def test_engine_records_preference_undo_before_import_and_keeps_first_backup(
    adapter,
    container,
    defaults_import,
    monkeypatch,
):
    engine = Engine(adapter.config, device=adapter.connect(UDID), platform=adapter)
    engine.config.teardown.enabled = True
    monkeypatch.setattr(engine, "_ledger_identity", lambda: {"owner": "ios-fixture"})
    monkeypatch.setattr(engine, "_ensure_teardown_watchdog", lambda target: None)
    service = adapter.capability("feature_flags")
    original = service.write_prefs
    backups = []

    def write(device, snapshot, values, **kwargs):
        entries = engine._pending_device_changes(serial=device.target_id)
        entry = next(item for item in entries if item.op == "restore_app_prefs")
        backups.append(entry.args["backup_path"])
        return original(device, snapshot, values, **kwargs)

    monkeypatch.setattr(service, "write_prefs", write)
    assert engine.prefs_write(APP_ID, APP_ID, {"fixture_flag": "first"})["ok"]
    assert engine.prefs_write(APP_ID, APP_ID, {"fixture_flag": "second"})["ok"]
    assert backups[0] == backups[1]
    from pathlib import Path

    assert json.loads(Path(backups[0]).read_text())["existed"] is False
    service.restore_prefs(engine.device, backups[0])
    engine.forget_device_change(f"app_prefs:{APP_ID}:{APP_ID}.plist")
    assert not engine._pending_device_changes(serial=UDID)
