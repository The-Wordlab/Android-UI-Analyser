"""SQLite access in simulator app containers, with consistent snapshots and restore points."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text
from ..errors import UsageError
from ..sqlite_schema import _schema_entry
from .identity import TargetRef
from .ios_files import IOSAppFiles
from .ios_tools import IOSTools
from .runtime import TargetRuntime

Parameters = Mapping[str, Any] | Sequence[Any] | None


def _copy(
    source: sqlite3.Connection, destination: sqlite3.Connection, timeout_ms: int = 5000
) -> None:
    if timeout_ms <= 0:
        raise UsageError("database timeout must be positive")
    deadline = time.monotonic() + timeout_ms / 1000

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() >= deadline:
            raise UsageError("database snapshot timed out")

    source.backup(destination, pages=128, progress=progress)


def _open(path: Path, *, readonly: bool = True) -> sqlite3.Connection:
    return sqlite3.connect(
        f"{path.as_uri()}?mode={'ro' if readonly else 'rw'}", uri=True, timeout=5
    )


@contextlib.contextmanager
def _stopped(device: TargetRuntime, app: str, restart: bool) -> Iterator[None]:
    device.stop_app(app)
    try:
        yield
    finally:
        if restart:
            device.launch_app(app)


def _guard(connection: sqlite3.Connection, timeout_ms: int, *, readonly: bool) -> None:
    if timeout_ms <= 0:
        raise UsageError("database timeout must be positive")
    deadline = time.monotonic() + timeout_ms / 1000
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    if not readonly:
        allowed |= {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}

    def authorize(
        action: int, arg1: str | None, arg2: str | None, _db: str | None, _source: str | None
    ) -> int:
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    connection.set_authorizer(authorize)


class IOSDatabase:
    def __init__(self, tools: IOSTools) -> None:
        self.tools = tools

    def _path(self, device: TargetRuntime, package: str, database: str) -> Path:
        path = IOSAppFiles(self.tools, device.target_id).path(package, database)
        if not path.is_file():
            raise UsageError("database not found; choose a relative path from `aua db list`")
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise UsageError("database is not unencrypted SQLite")
        return path

    def list_databases(self, device: TargetRuntime, package: str) -> dict[str, Any]:
        files = IOSAppFiles(self.tools, device.target_id)
        root = files.root(package)
        entries = []
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.name.endswith(("-wal", "-shm", "-journal"))
            ):
                continue
            if not path.resolve().is_relative_to(root):
                continue
            with path.open("rb") as stream:
                if stream.read(16) != b"SQLite format 3\x00":
                    continue
            entries.append(
                {"name": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size}
            )
        return {
            "ok": True,
            "action": "database-list",
            "package": package,
            "databases": entries,
            "count": len(entries),
        }

    def query_database(
        self,
        device: TargetRuntime,
        package: str,
        database: str,
        sql: str,
        *,
        parameters: Parameters = None,
        limit: int = 100,
        timeout_ms: int = 5000,
        restart: bool = True,
        live: bool = True,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 1000:
            raise UsageError("row limit must be between 1 and 1000")
        started = time.monotonic()
        path = self._path(device, package, database)
        with (
            contextlib.nullcontext() if live else _stopped(device, package, restart),
            contextlib.closing(_open(path)) as source,
            contextlib.closing(sqlite3.connect(":memory:")) as snapshot,
        ):
            _copy(source, snapshot, timeout_ms)
            snapshot.execute("PRAGMA query_only=ON")
            _guard(snapshot, timeout_ms, readonly=True)
            try:
                cursor = snapshot.execute(sql, parameters or ())
                rows = cursor.fetchmany(limit + 1)
                columns = [item[0] for item in cursor.description or ()]
            except sqlite3.Error as exc:
                raise UsageError(
                    f"read-only database query failed: {exc}", code="database_sql"
                ) from exc
        return {
            "ok": True,
            "action": "database-query",
            "package": package,
            "database": database,
            "columns": columns,
            "rows": [
                [value.hex() if isinstance(value, bytes) else value for value in row]
                for row in rows[:limit]
            ],
            "row_count": min(len(rows), limit),
            "limit": limit,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "truncated": len(rows) > limit,
            "coherent": not live,
            "app_stopped": not live,
            "app_restarted": restart and not live,
        }

    def database_schema(
        self,
        device: TargetRuntime,
        package: str,
        database: str,
        *,
        table: str | None = None,
        restart: bool = True,
    ) -> dict[str, Any]:
        path = self._path(device, package, database)
        with (
            contextlib.closing(_open(path)) as source,
            contextlib.closing(sqlite3.connect(":memory:")) as snapshot,
        ):
            _copy(source, snapshot)
            snapshot.row_factory = sqlite3.Row
            rows = snapshot.execute(
                "SELECT type,name,sql FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' AND (? IS NULL OR name=?) ORDER BY name",
                [table, table],
            ).fetchall()
            objects = [_schema_entry(snapshot, row) for row in rows]
        if table is not None and not objects:
            raise UsageError(
                f"table or view {table!r} does not exist", code="database_table_not_found"
            )
        return {
            "ok": True,
            "action": "database-schema",
            "package": package,
            "database": database,
            "objects": objects,
            "count": len(objects),
            "app_stopped": False,
            "app_restarted": False,
        }

    def _backups(
        self, device: TargetRuntime, cache_dir: str | Path, package: str, database: str
    ) -> Path:
        key = hashlib.sha256(database.encode()).hexdigest()[:16]
        return (
            Path(cache_dir).expanduser()
            / "ios-db-backups"
            / TargetRef("ios", device.target_id).storage_key
            / package
            / key
        )

    def _backup(
        self, device: TargetRuntime, cache_dir: str | Path, package: str, database: str, path: Path
    ) -> dict[str, Any]:
        identifier = uuid.uuid4().hex
        directory = self._backups(device, cache_dir, package, database)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{identifier}.sqlite"
        destination.touch(mode=0o600, exist_ok=False)
        with (
            contextlib.closing(_open(path)) as source,
            contextlib.closing(sqlite3.connect(destination)) as saved,
        ):
            _copy(source, saved)
        metadata = {
            "id": identifier,
            "backup_id": identifier,
            "package": package,
            "database": database,
            "created_at": time.time(),
        }
        atomic_write_text(
            directory / f"{identifier}.json",
            json.dumps(metadata),
        )
        return metadata

    def backup_database(
        self,
        device: TargetRuntime,
        cache_dir: str | Path,
        package: str,
        database: str,
        *,
        restart: bool = True,
    ) -> dict[str, Any]:
        path = self._path(device, package, database)
        backup = self._backup(device, cache_dir, package, database, path)
        return {
            "ok": True,
            "action": "database-backup",
            "package": package,
            "database": database,
            "backup": backup,
            "backup_id": backup["id"],
        }

    def list_backups(
        self, device: TargetRuntime, cache_dir: str | Path, package: str, database: str
    ) -> dict[str, Any]:
        self._path(device, package, database)
        backups = [
            json.loads(path.read_text())
            for path in self._backups(device, cache_dir, package, database).glob("*.json")
        ]
        return {
            "ok": True,
            "action": "database-backups",
            "backups": sorted(backups, key=lambda row: row["created_at"]),
        }

    def execute_database(
        self,
        device: TargetRuntime,
        cache_dir: str | Path,
        package: str,
        database: str,
        sql: str,
        *,
        parameters: Parameters = None,
        timeout_ms: int = 5000,
        restart: bool = True,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        if not confirmed:
            raise UsageError("database mutation requires --yes")
        started = time.monotonic()
        if not sql.strip() or sql.lstrip().split()[0].upper() not in {
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
        }:
            raise UsageError("execute accepts one INSERT, UPDATE, DELETE or REPLACE statement")
        path = self._path(device, package, database)
        with _stopped(device, package, restart):
            backup = self._backup(device, cache_dir, package, database, path)
            with contextlib.closing(_open(path, readonly=False)) as connection:
                try:
                    connection.execute("PRAGMA foreign_keys=ON")
                    connection.execute("BEGIN IMMEDIATE")
                    _guard(connection, timeout_ms, readonly=False)
                    before = connection.total_changes
                    cursor = connection.execute(sql, parameters or ())
                    columns = [item[0] for item in cursor.description or ()]
                    rows = cursor.fetchmany(101) if columns else []
                    lastrowid = cursor.lastrowid
                    count = cursor.rowcount
                    cursor.close()
                    changes = connection.total_changes - before
                    connection.set_authorizer(None)
                    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                        raise UsageError("SQLite integrity check failed")
                    connection.commit()
                except (sqlite3.Error, UsageError) as exc:
                    connection.set_authorizer(None)
                    connection.rollback()
                    raise UsageError(
                        f"database mutation rolled back: {exc}", code="database_sql"
                    ) from exc
        return {
            "ok": True,
            "action": "database-execute",
            "package": package,
            "database": database,
            "rows_affected": count,
            "changes": changes,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "statements": [
                {
                    "statement": 1,
                    "kind": sql.lstrip().split()[0].upper(),
                    "changes": changes,
                    "rowcount": count,
                    "lastrowid": lastrowid,
                    "columns": columns,
                    "rows": [
                        [value.hex() if isinstance(value, bytes) else value for value in row]
                        for row in rows[:100]
                    ],
                    "truncated": len(rows) > 100,
                }
            ],
            "backup": backup,
            "backup_id": backup["id"],
            "app_restarted": restart,
        }

    def restore_database(
        self,
        device: TargetRuntime,
        cache_dir: str | Path,
        package: str,
        database: str,
        backup_id: str,
        *,
        restart: bool = True,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        if (
            not confirmed
            or len(backup_id) != 32
            or any(c not in "0123456789abcdef" for c in backup_id)
        ):
            raise UsageError("restore requires --yes and a backup id from `aua db backups`")
        path = self._path(device, package, database)
        directory = self._backups(device, cache_dir, package, database)
        try:
            metadata = json.loads((directory / f"{backup_id}.json").read_text())
        except (OSError, ValueError) as exc:
            raise UsageError(
                "backup not found or unreadable; choose an id from `aua db backups`"
            ) from exc
        if metadata["package"] != package or metadata["database"] != database:
            raise UsageError("backup does not belong to the selected app/database")
        with _stopped(device, package, restart):
            before = self._backup(device, cache_dir, package, database, path)
            with (
                contextlib.closing(_open(directory / f"{backup_id}.sqlite")) as source,
                contextlib.closing(_open(path, readonly=False)) as target,
            ):
                if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise UsageError("backup failed SQLite integrity check")
                _copy(source, target)
        return {
            "ok": True,
            "action": "database-restore",
            "backup_id": backup_id,
            "package": package,
            "database": database,
            "restored_backup": metadata,
            "safety_backup": before,
            "pre_restore_backup_id": before["id"],
            "app_restarted": restart,
        }
