"""Private app database inspection, guarded mutation, backup, and restore."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser import app_database
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError, UsageError
from android_ui_analyser.mcp_server import build_server
from conftest import FakeDevice, make_config

PKG = "com.example.app"
DB = "app.db"


def _database_snapshot(tmp_path: Path) -> dict[str, bytes]:
    path = tmp_path / DB
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE parents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE children (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER NOT NULL REFERENCES parents(id),
                value TEXT NOT NULL,
                payload BLOB
            );
            CREATE INDEX children_value ON children(value);
            INSERT INTO parents(id, name) VALUES (1, 'one');
            INSERT INTO children(id, parent_id, value, payload)
            VALUES (10, 1, 'before', X'000102');
            """
        )
        connection.commit()
        files = {f"databases/{DB}": path.read_bytes()}
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(f"{path.name}{suffix}")
            if sidecar.is_file():
                files[f"databases/{DB}{suffix}"] = sidecar.read_bytes()
        return files
    finally:
        connection.close()


def _device(tmp_path: Path, **kwargs: object) -> FakeDevice:
    return FakeDevice(package=PKG, app_files=_database_snapshot(tmp_path), **kwargs)


def _read_remote(device: FakeDevice, tmp_path: Path, sql: str) -> list[tuple[object, ...]]:
    path = tmp_path / "remote.db"
    path.write_bytes(device.app_files[f"databases/{DB}"])
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(f"{path.name}{suffix}")
        sidecar.unlink(missing_ok=True)
        remote = device.app_files.get(f"databases/{DB}{suffix}")
        if remote is not None:
            sidecar.write_bytes(remote)
    connection = sqlite3.connect(path)
    try:
        return list(connection.execute(sql))
    finally:
        connection.close()


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    return Engine(
        make_config(
            cache={"dir": str(tmp_path / "cache")},
            daemon={"enabled": False},
            lease={"enabled": False},
        ),
        device=device,
    )


def _first_text(result) -> str:  # type: ignore[no-untyped-def]
    return next(block.text for block in result.content if getattr(block, "type", None) == "text")


def test_list_reports_primary_databases_and_sidecars(tmp_path: Path) -> None:
    device = _device(tmp_path)
    device.app_files["databases/other"] = b"SQLite format 3\x00rest"
    device.app_files["databases/other-journal"] = b"journal"

    result = app_database.list_databases(device, PKG)

    assert [item["name"] for item in result["databases"]] == [DB, "other"]
    app_entry = result["databases"][0]
    assert app_entry["wal_size_bytes"]
    assert app_entry["shm_size_bytes"]
    assert not [call for call in device.calls if call[0] == "stop_app"]


def test_query_is_read_only_parameterized_bounded_and_restarts(tmp_path: Path) -> None:
    device = _device(tmp_path)

    result = app_database.query_database(
        device,
        PKG,
        DB,
        "SELECT id, value, payload FROM children WHERE parent_id = ? ORDER BY id",
        parameters=[1],
        limit=1,
    )

    assert result["columns"] == ["id", "value", "payload"]
    assert result["rows"][0][:2] == [10, "before"]
    assert result["rows"][0][2] == {"base64": "AAEC", "bytes": 3}
    assert result["truncated"] is False
    assert [call[0] for call in device.calls].count("stop_app") == 1
    assert [call[0] for call in device.calls].count("launch_app") == 1

    original = dict(device.app_files)
    with pytest.raises(UsageError, match="query failed"):
        app_database.query_database(device, PKG, DB, "UPDATE children SET value = 'bad'")
    assert device.app_files == original


def test_query_truncates_without_rewriting_the_sql(tmp_path: Path) -> None:
    device = _device(tmp_path)
    result = app_database.query_database(
        device,
        PKG,
        DB,
        "SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3",
        limit=2,
    )
    assert result["rows"] == [[1], [2]]
    assert result["truncated"] is True


def test_schema_returns_columns_indexes_and_foreign_keys(tmp_path: Path) -> None:
    result = app_database.database_schema(_device(tmp_path), PKG, DB, table="children")

    assert result["count"] == 1
    children = result["objects"][0]
    assert [column["name"] for column in children["columns"]] == [
        "id",
        "parent_id",
        "value",
        "payload",
    ]
    assert children["indexes"][0]["name"] == "children_value"
    assert children["foreign_keys"][0]["table"] == "parents"


def test_execute_requires_confirmation_before_touching_the_app(tmp_path: Path) -> None:
    device = _device(tmp_path)
    with pytest.raises(UsageError, match="requires explicit confirmation"):
        app_database.execute_database(
            device,
            tmp_path / "cache",
            PKG,
            DB,
            "UPDATE children SET value = 'after'",
        )
    assert device.calls == []


def test_execute_mutates_in_one_transaction_backs_up_and_removes_sidecars(tmp_path: Path) -> None:
    device = _device(tmp_path)

    result = app_database.execute_database(
        device,
        tmp_path / "cache",
        PKG,
        DB,
        """
        UPDATE children SET value = 'after' WHERE id = 10;
        INSERT INTO children(id, parent_id, value) VALUES (11, 1, 'second');
        """,
        confirmed=True,
    )

    assert result["changes"] == 2
    assert result["statement_count"] == 2
    assert result["backup"]["reason"] == "before-execute"
    assert Path(result["backup"]["path"], "metadata.json").is_file()
    assert _read_remote(device, tmp_path, "SELECT id, value FROM children ORDER BY id") == [
        (10, "after"),
        (11, "second"),
    ]
    assert f"databases/{DB}-wal" not in device.app_files
    assert f"databases/{DB}-shm" not in device.app_files


def test_execute_rejects_schema_and_foreign_key_damage_without_replacing(tmp_path: Path) -> None:
    device = _device(tmp_path)
    original = dict(device.app_files)

    with pytest.raises(UsageError, match="refuses CREATE"):
        app_database.execute_database(
            device,
            tmp_path / "cache",
            PKG,
            DB,
            "CREATE TABLE forbidden(id INTEGER)",
            confirmed=True,
        )
    assert device.calls == []

    with pytest.raises(UsageError, match="execute failed"):
        app_database.execute_database(
            device,
            tmp_path / "cache",
            PKG,
            DB,
            "DELETE FROM parents WHERE id = 1",
            confirmed=True,
        )
    assert device.app_files == original


def test_restore_preserves_current_state_as_a_safety_backup(tmp_path: Path) -> None:
    device = _device(tmp_path)
    cache = tmp_path / "cache"
    changed = app_database.execute_database(
        device,
        cache,
        PKG,
        DB,
        "UPDATE children SET value = 'after' WHERE id = 10",
        confirmed=True,
    )
    original_id = changed["backup"]["id"]

    restored = app_database.restore_database(
        device,
        cache,
        PKG,
        DB,
        original_id,
        confirmed=True,
    )

    assert restored["restored_backup"]["id"] == original_id
    assert restored["safety_backup"]["reason"] == f"before-restore-{original_id}"
    assert _read_remote(device, tmp_path, "SELECT value FROM children WHERE id = 10") == [
        ("before",)
    ]
    backups = app_database.list_backups(device, cache, PKG, DB)
    assert backups["count"] == 2


def test_manual_backup_keeps_wal_and_shm(tmp_path: Path) -> None:
    device = _device(tmp_path)
    result = app_database.backup_database(device, tmp_path / "cache", PKG, DB)
    backup_path = Path(result["backup"]["path"])
    assert (backup_path / DB).is_file()
    assert (backup_path / f"{DB}-wal").is_file()
    assert (backup_path / f"{DB}-shm").is_file()


def test_run_as_refusal_is_a_structured_device_error(tmp_path: Path) -> None:
    device = _device(tmp_path, run_as_error="run-as: package not debuggable")
    with pytest.raises(DeviceError, match="not debuggable") as raised:
        app_database.list_databases(device, PKG)
    assert raised.value.code == "database_access"


def test_engine_daemon_cli_and_mcp_expose_the_same_database_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _device(tmp_path)
    engine = _engine(tmp_path, device)
    daemon_result = dispatch(
        engine,
        {
            "cmd": "database_query",
            "args": {"package": PKG, "database": DB, "sql": "SELECT value FROM children"},
        },
    )
    assert daemon_result["ok"] is True
    assert daemon_result["result"]["rows"] == [["before"]]

    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setattr(
        "android_ui_analyser.cli.load_config",
        lambda *args, **kwargs: make_config(
            cache={"dir": str(tmp_path / "cli-cache")},
            daemon={"enabled": False},
            lease={"enabled": False},
        ),
    )
    cli_result = CliRunner().invoke(
        app,
        ["db", "query", PKG, DB, "SELECT value FROM children"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert json.loads(cli_result.stdout)["rows"] == [["before"]]

    server = build_server(engine)

    async def call_mcp() -> tuple[set[str], dict[str, object]]:
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            response = await client.call_tool(
                "database_query",
                {"package": PKG, "database": DB, "sql": "SELECT value FROM children"},
            )
            return {tool.name for tool in listed.tools}, json.loads(_first_text(response))

    tools, mcp_result = anyio.run(call_mcp)
    assert {
        "database_list",
        "database_schema",
        "database_query",
        "database_execute",
        "database_backup",
        "database_backups",
        "database_restore",
    } <= tools
    assert mcp_result["rows"] == [["before"]]


def test_cli_and_mcp_execute_still_require_explicit_confirmation(tmp_path: Path) -> None:
    device = _device(tmp_path)
    engine = _engine(tmp_path, device)
    server = build_server(engine)

    async def call_mcp() -> dict[str, object]:
        async with create_connected_server_and_client_session(server) as client:
            response = await client.call_tool(
                "database_execute",
                {
                    "package": PKG,
                    "database": DB,
                    "sql": "UPDATE children SET value = 'after'",
                    "confirmed": False,
                },
            )
            return json.loads(_first_text(response))

    result = anyio.run(call_mcp)
    assert result["error"]["code"] == "database_confirmation_required"
    assert _read_remote(device, tmp_path, "SELECT value FROM children") == [("before",)]


def test_cli_execute_requires_yes_then_returns_the_restore_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _device(tmp_path)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    monkeypatch.setattr(
        "android_ui_analyser.cli.load_config",
        lambda *args, **kwargs: make_config(
            cache={"dir": str(tmp_path / "cli-cache")},
            daemon={"enabled": False},
            lease={"enabled": False},
        ),
    )
    runner = CliRunner()
    arguments = [
        "db",
        "execute",
        PKG,
        DB,
        "UPDATE children SET value = :value WHERE id = :id",
        "--params",
        '{"value":"after","id":10}',
    ]

    refused = runner.invoke(app, arguments)
    assert refused.exit_code == 2
    assert "database_confirmation_required" in refused.stderr
    assert _read_remote(device, tmp_path, "SELECT value FROM children") == [("before",)]

    changed = runner.invoke(app, [*arguments, "--yes"])
    assert changed.exit_code == 0, changed.output
    payload = json.loads(changed.stdout)
    assert payload["changes"] == 1
    assert payload["backup"]["id"]
    assert _read_remote(device, tmp_path, "SELECT value FROM children") == [("after",)]
