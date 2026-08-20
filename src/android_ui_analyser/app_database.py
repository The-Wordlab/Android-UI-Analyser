"""Safe host-side inspection and mutation of a debuggable Android app database.

Android system images commonly omit the ``sqlite3`` executable.  AUA therefore snapshots
the selected database through ``run-as``, opens the snapshot with Python's SQLite module,
and writes a validated, consolidated database back only for explicitly confirmed mutations.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import re
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from shlex import quote
from typing import Any

from .device import Device
from .errors import DeviceError, UsageError

_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")
_DATABASE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_BACKUP_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RUN_AS_ERRORS = ("run-as:", "not debuggable", "is unknown", "permission denied")
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_SQLITE_HEADER = b"SQLite format 3\x00"
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000
_DEFAULT_TIMEOUT_MS = 5000
_BACKUP_FORMAT = 1

SqlParameters = Mapping[str, Any] | Sequence[Any] | None


def _validate_package(package: str) -> str:
    value = package.strip()
    if not _PACKAGE_RE.fullmatch(value):
        raise UsageError(
            f"invalid Android package name: {package!r}",
            code="database_package_invalid",
        )
    return value


def _validate_database(database: str) -> str:
    value = database.strip()
    if not value or not _DATABASE_RE.fullmatch(value) or value.endswith(_SIDECAR_SUFFIXES):
        raise UsageError(
            f"invalid database name: {database!r}",
            hint="Pass a database basename from `aua db list <package>`, not a path or sidecar.",
            code="database_name_invalid",
        )
    return value


def _validate_backup_id(backup_id: str) -> str:
    value = backup_id.strip()
    if not value or not _BACKUP_RE.fullmatch(value):
        raise UsageError(f"invalid backup id: {backup_id!r}", code="database_backup_invalid")
    return value


def _remote_path(database: str, suffix: str = "") -> str:
    return f"databases/{database}{suffix}"


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def _run_as(device: Device, package: str, argv: list[str]) -> str:
    command = f"run-as {quote(package)} " + " ".join(quote(arg) for arg in argv)
    try:
        output = device.shell(command)
    except Exception as exc:
        raise DeviceError(
            f"cannot access {package} app data: {exc}",
            hint="Database access requires an installed debuggable build and a connected device.",
            code="database_access",
        ) from exc
    for line in output.splitlines():
        lowered = line.strip().lower()
        if lowered and any(marker in lowered for marker in _RUN_AS_ERRORS):
            raise DeviceError(
                f"cannot access {package} app data: {line.strip()}",
                hint="Use a debuggable app build; Android run-as refuses production builds.",
                code="database_access",
            )
    return output


def _parse_database_listing(raw: str) -> dict[str, int]:
    files: dict[str, int] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        parts = stripped.split(maxsplit=7)
        if len(parts) < 8:
            continue
        try:
            size = int(parts[4])
        except ValueError:
            continue
        name = parts[7]
        if "/" not in name and name not in (".", ".."):
            files[name] = size
    return files


def _database_entries(files: Mapping[str, int]) -> list[dict[str, Any]]:
    primaries = sorted(
        name for name in files if not name.startswith(".") and not name.endswith(_SIDECAR_SUFFIXES)
    )
    entries: list[dict[str, Any]] = []
    for name in primaries:
        entries.append(
            {
                "name": name,
                "size_bytes": files[name],
                "wal_size_bytes": files.get(f"{name}-wal"),
                "shm_size_bytes": files.get(f"{name}-shm"),
                "journal_size_bytes": files.get(f"{name}-journal"),
            }
        )
    return entries


def list_databases(device: Device, package: str) -> dict[str, Any]:
    package = _validate_package(package)
    raw = _run_as(device, package, ["ls", "-la", "databases"])
    if "no such file or directory" in raw.lower():
        files: dict[str, int] = {}
    else:
        files = _parse_database_listing(raw)
    databases = _database_entries(files)
    return {
        "ok": True,
        "action": "database-list",
        "package": package,
        "databases": databases,
        "count": len(databases),
    }


def _available_files(device: Device, package: str) -> dict[str, int]:
    raw = _run_as(device, package, ["ls", "-la", "databases"])
    return _parse_database_listing(raw)


def _capture_files(
    device: Device,
    package: str,
    database: str,
    *,
    include_shm: bool = True,
) -> dict[str, bytes]:
    available = _available_files(device, package)
    if database not in available:
        choices = [entry["name"] for entry in _database_entries(available)]
        suffix = f" Available: {', '.join(choices)}." if choices else ""
        raise UsageError(
            f"database {database!r} does not exist for {package}.{suffix}",
            hint=f"Run `aua db list {package}` and pass one of its names.",
            code="database_not_found",
        )
    suffixes = (
        _SIDECAR_SUFFIXES
        if include_shm
        else tuple(suffix for suffix in _SIDECAR_SUFFIXES if suffix != "-shm")
    )
    names = [database]
    names.extend(f"{database}{suffix}" for suffix in suffixes if f"{database}{suffix}" in available)
    captured: dict[str, bytes] = {}
    for name in names:
        try:
            captured[name] = device.read_app_file(package, f"databases/{name}")
        except Exception as exc:
            raise DeviceError(
                f"failed to snapshot {package}/{name}: {exc}",
                hint="The app must be debuggable and remain installed while AUA reads it.",
                code="database_snapshot_failed",
            ) from exc
    if not captured[database].startswith(_SQLITE_HEADER):
        raise UsageError(
            f"{database!r} is not a plain SQLite database",
            hint="Encrypted or vendor-specific databases cannot be queried through this feature.",
            code="database_not_sqlite",
        )
    return captured


@contextmanager
def _stopped_app(device: Device, package: str, *, restart: bool) -> Iterator[None]:
    try:
        device.stop_app(package)
    except Exception as exc:
        raise DeviceError(
            f"could not stop {package} before accessing its database: {exc}",
            code="database_stop_failed",
        ) from exc
    try:
        yield
    finally:
        if restart:
            try:
                device.launch_app(package)
            except Exception as exc:
                raise DeviceError(
                    f"database operation finished but {package} could not be relaunched: {exc}",
                    hint=f"Relaunch it explicitly with `aua app launch {package}`.",
                    code="database_restart_failed",
                ) from exc


def _state_loss_warning(package: str, *, restarted: bool) -> str:
    """What an agent must know after a call that force-stopped the app to snapshot its data.

    ``app_restarted`` alone reads as bookkeeping, not "you are not where you were" -- an agent
    skimming the result can miss it and keep acting on stale navigation state. Say the
    consequence in words, the same way ``mock_map`` warns an agent about another session's
    live stubs.
    """
    if restarted:
        return (
            f"{package} was force-stopped to get a coherent database snapshot and has been "
            "relaunched. Any in-app navigation or UI state from before this call is gone -- "
            "the app is back at its cold-start screen, not wherever you left it. Re-navigate "
            "before continuing; do not assume you are still where you were."
        )
    return (
        f"{package} was force-stopped to get a coherent database snapshot and was left "
        "stopped (--no-restart). Any in-app navigation or UI state from before this call is "
        "gone, and the app is not running at all. Launch it and re-navigate before continuing."
    )


_LIVE_SNAPSHOT_WARNING = (
    "this read did not stop the app, so the database copy may be torn: the app could have "
    "been mid-write when the files were captured. SQLite only accepts WAL frames whose own "
    "checksum validates, so a torn copy fails safe (a slightly older but internally "
    "consistent state) rather than returning corrupt data -- but this is a point-in-time-ish "
    "read, not a transactionally coherent one. In-app navigation and UI state were NOT "
    "touched. For a guaranteed-coherent snapshot, omit --live."
)


def _materialize(directory: Path, files: Mapping[str, bytes]) -> Path:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    main: Path | None = None
    primary_names = {name for name in files if not name.endswith(_SIDECAR_SUFFIXES)}
    if len(primary_names) != 1:
        raise DeviceError("database snapshot has no unambiguous primary file")
    primary = next(iter(primary_names))
    for name, data in files.items():
        path = directory / name
        path.write_bytes(data)
        path.chmod(0o600)
        if name == primary:
            main = path
    if main is None:
        raise DeviceError("database snapshot is missing its primary file")
    return main


def _backup_root(cache_dir: str | Path, serial: str, package: str, database: str) -> Path:
    return (
        Path(cache_dir).expanduser()
        / "database-backups"
        / _safe_component(serial)
        / _safe_component(package)
        / _safe_component(database)
    )


def _new_backup_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _store_backup(
    *,
    cache_dir: str | Path,
    serial: str,
    package: str,
    database: str,
    files: Mapping[str, bytes],
    reason: str,
) -> dict[str, Any]:
    backup_id = _new_backup_id()
    path = _backup_root(cache_dir, serial, package, database) / backup_id
    _materialize(path, files)
    created_at = datetime.now(UTC).isoformat()
    metadata = {
        "format": _BACKUP_FORMAT,
        "id": backup_id,
        "serial": serial,
        "package": package,
        "database": database,
        "created_at": created_at,
        "reason": reason,
        "files": {name: len(data) for name, data in sorted(files.items())},
    }
    metadata_path = path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path.chmod(0o600)
    return {**metadata, "path": str(path)}


def _load_backup(
    cache_dir: str | Path,
    serial: str,
    package: str,
    database: str,
    backup_id: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    path = _backup_root(cache_dir, serial, package, database) / _validate_backup_id(backup_id)
    metadata_path = path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(
            f"database backup {backup_id!r} was not found or is invalid",
            hint=f"List available restore points with `aua db backups {package} {database}`.",
            code="database_backup_not_found",
        ) from exc
    expected = (serial, package, database)
    actual = (metadata.get("serial"), metadata.get("package"), metadata.get("database"))
    if metadata.get("format") != _BACKUP_FORMAT or actual != expected:
        raise UsageError(
            f"database backup {backup_id!r} does not belong to this device/package/database",
            code="database_backup_mismatch",
        )
    files: dict[str, bytes] = {}
    for name in metadata.get("files", {}):
        if not isinstance(name, str) or Path(name).name != name:
            raise UsageError(f"database backup {backup_id!r} has unsafe file metadata")
        try:
            files[name] = (path / name).read_bytes()
        except OSError as exc:
            raise UsageError(f"database backup {backup_id!r} is incomplete") from exc
    if database not in files or not files[database].startswith(_SQLITE_HEADER):
        raise UsageError(f"database backup {backup_id!r} has no valid primary database")
    return {**metadata, "path": str(path)}, files


def backup_database(
    device: Device,
    cache_dir: str | Path,
    package: str,
    database: str,
    *,
    restart: bool = True,
) -> dict[str, Any]:
    package = _validate_package(package)
    database = _validate_database(database)
    with _stopped_app(device, package, restart=restart):
        files = _capture_files(device, package, database)
        backup = _store_backup(
            cache_dir=cache_dir,
            serial=device.serial,
            package=package,
            database=database,
            files=files,
            reason="manual",
        )
    return {
        "ok": True,
        "action": "database-backup",
        "package": package,
        "database": database,
        "backup": backup,
        "app_restarted": restart,
        "warning": _state_loss_warning(package, restarted=restart),
    }


def list_backups(
    device: Device,
    cache_dir: str | Path,
    package: str,
    database: str,
) -> dict[str, Any]:
    package = _validate_package(package)
    database = _validate_database(database)
    root = _backup_root(cache_dir, device.serial, package, database)
    backups: list[dict[str, Any]] = []
    if root.is_dir():
        for metadata_path in sorted(root.glob("*/metadata.json"), reverse=True):
            with contextlib.suppress(OSError, json.JSONDecodeError):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("format") == _BACKUP_FORMAT
                    and metadata.get("serial") == device.serial
                    and metadata.get("package") == package
                    and metadata.get("database") == database
                ):
                    backups.append({**metadata, "path": str(metadata_path.parent)})
    return {
        "ok": True,
        "action": "database-backups",
        "package": package,
        "database": database,
        "backups": backups,
        "count": len(backups),
    }


def _snapshot_to_temp(
    device: Device,
    package: str,
    database: str,
    *,
    restart: bool,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="aua-database-")
    try:
        with _stopped_app(device, package, restart=restart):
            files = _capture_files(device, package, database)
            main = _materialize(Path(temporary.name), files)
    except Exception:
        temporary.cleanup()
        raise
    return temporary, main


def _snapshot_live_to_temp(
    device: Device,
    package: str,
    database: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Copy the database and its ``-wal`` sidecar while the app keeps running.

    Deliberately excludes ``-shm``. SQLite has been able to read a WAL-mode database with no
    writable ``-shm`` since 3.22.0: it re-derives the wal-index by scanning ``-wal`` frames
    under a read lock and validating each frame's own checksum, instead of trusting a
    memory-mapped index (see https://www.sqlite.org/wal.html, "Read-Only WAL"). That is
    exactly what makes this mode safe to run against a live app: a ``-shm`` captured a moment
    apart from its ``-wal`` could describe frames that do not match what was actually copied,
    and SQLite would trust it -- turning an honestly torn read into a silently wrong one.
    Leaving ``-shm`` out forces the codepath that is built to tolerate a torn WAL instead of
    the one that assumes a coherent snapshot.
    """
    temporary = tempfile.TemporaryDirectory(prefix="aua-database-live-")
    try:
        files = _capture_files(device, package, database, include_shm=False)
        main = _materialize(Path(temporary.name), files)
    except Exception:
        temporary.cleanup()
        raise
    return temporary, main


def _connect_snapshot(main: Path, *, live: bool) -> sqlite3.Connection:
    """Open the host-side snapshot for reading only -- never to write back to it.

    The coherent (stop-first) path keeps its historical plain-path connection; the live path
    additionally opens through the ``mode=ro`` URI parameter, which has SQLite itself open the
    file ``O_RDONLY`` so it cannot checkpoint or otherwise touch the copy even by accident, and
    is what engages the read-only WAL recovery described in `_snapshot_live_to_temp`.
    ``PRAGMA query_only=ON`` is set either way as a second, session-level guard against writes.
    """
    connection = (
        sqlite3.connect(main.as_uri() + "?mode=ro", uri=True)
        if live
        else sqlite3.connect(str(main))
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _configure_timeout(connection: sqlite3.Connection, timeout_ms: int) -> None:
    if timeout_ms <= 0:
        raise UsageError("database SQL timeout must be greater than zero")
    deadline = time.monotonic() + timeout_ms / 1000.0
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)


def _normalize_parameters(parameters: SqlParameters) -> Mapping[str, Any] | Sequence[Any]:
    if parameters is None:
        return ()
    if isinstance(parameters, Mapping):
        return dict(parameters)
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes, bytearray)):
        return list(parameters)
    raise UsageError("SQL parameters must be a JSON object or array", code="database_params")


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "base64": base64.b64encode(value).decode("ascii"),
            "bytes": len(value),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _cursor_rows(cursor: sqlite3.Cursor, limit: int) -> tuple[list[str], list[list[Any]], bool]:
    columns = [item[0] for item in (cursor.description or [])]
    if not columns:
        return [], [], False
    rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    return columns, [[_json_value(value) for value in row] for row in rows[:limit]], truncated


def _bounded_limit(limit: int) -> int:
    if limit <= 0 or limit > _MAX_LIMIT:
        raise UsageError(
            f"database row limit must be between 1 and {_MAX_LIMIT}",
            code="database_limit",
        )
    return limit


def query_database(
    device: Device,
    package: str,
    database: str,
    sql: str,
    *,
    parameters: SqlParameters = None,
    limit: int = _DEFAULT_LIMIT,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    restart: bool = True,
    live: bool = False,
) -> dict[str, Any]:
    """Run one read-only statement against a host-side database snapshot.

    By default this stops the app to get a transactionally coherent snapshot (see
    ``_stopped_app``); pass ``live=True`` to read a copy taken while the app keeps running
    instead -- no navigation/UI state is lost, but the copy may be torn (see
    ``_snapshot_live_to_temp`` / ``_LIVE_SNAPSHOT_WARNING``). ``restart`` is meaningless when
    ``live`` is set, since the app is never stopped.
    """
    package = _validate_package(package)
    database = _validate_database(database)
    if not sql.strip():
        raise UsageError("database query needs a SQL statement")
    limit = _bounded_limit(limit)
    started = time.monotonic()
    if live:
        temporary, main = _snapshot_live_to_temp(device, package, database)
    else:
        temporary, main = _snapshot_to_temp(device, package, database, restart=restart)
    try:
        connection = _connect_snapshot(main, live=live)
        try:
            _configure_timeout(connection, timeout_ms)
            cursor = connection.execute(sql, _normalize_parameters(parameters))
            columns, rows, truncated = _cursor_rows(cursor, limit)
            if not columns:
                raise UsageError(
                    "database query did not return rows",
                    hint="Use `aua db execute ... --yes` for INSERT, UPDATE, or DELETE.",
                    code="database_query_not_readonly",
                )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise UsageError(
            f"database query failed: {exc}",
            hint="Only one read-only SQLite statement is accepted; use parameters for values.",
            code="database_sql",
        ) from exc
    finally:
        temporary.cleanup()
    return {
        "ok": True,
        "action": "database-query",
        "package": package,
        "database": database,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "limit": limit,
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "coherent": not live,
        "app_stopped": not live,
        "app_restarted": False if live else restart,
        "warning": (
            _LIVE_SNAPSHOT_WARNING if live else _state_loss_warning(package, restarted=restart)
        ),
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _schema_entry(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    name = str(row[1])
    identifier = _quote_identifier(name)
    columns = [
        {
            "cid": item[0],
            "name": item[1],
            "type": item[2],
            "not_null": bool(item[3]),
            "default": item[4],
            "primary_key": item[5],
            "hidden": item[6] if len(item) > 6 else 0,
        }
        for item in connection.execute(f"PRAGMA table_xinfo({identifier})")
    ]
    indexes = [
        {
            "name": item[1],
            "unique": bool(item[2]),
            "origin": item[3],
            "partial": bool(item[4]),
        }
        for item in connection.execute(f"PRAGMA index_list({identifier})")
    ]
    foreign_keys = [
        {
            "id": item[0],
            "sequence": item[1],
            "table": item[2],
            "from": item[3],
            "to": item[4],
            "on_update": item[5],
            "on_delete": item[6],
            "match": item[7],
        }
        for item in connection.execute(f"PRAGMA foreign_key_list({identifier})")
    ]
    return {
        "type": row[0],
        "name": name,
        "sql": row[2],
        "columns": columns,
        "indexes": indexes,
        "foreign_keys": foreign_keys,
    }


def database_schema(
    device: Device,
    package: str,
    database: str,
    *,
    table: str | None = None,
    restart: bool = True,
) -> dict[str, Any]:
    package = _validate_package(package)
    database = _validate_database(database)
    temporary, main = _snapshot_to_temp(device, package, database, restart=restart)
    try:
        connection = sqlite3.connect(str(main))
        connection.row_factory = sqlite3.Row
        try:
            sql = (
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
            parameters: tuple[str, ...] = ()
            if table is not None:
                sql += " AND name = ?"
                parameters = (table,)
            sql += " ORDER BY type, name"
            objects = [
                _schema_entry(connection, row) for row in connection.execute(sql, parameters)
            ]
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise UsageError(f"database schema inspection failed: {exc}", code="database_sql") from exc
    finally:
        temporary.cleanup()
    if table is not None and not objects:
        raise UsageError(
            f"table or view {table!r} does not exist in {database!r}",
            code="database_table_not_found",
        )
    return {
        "ok": True,
        "action": "database-schema",
        "package": package,
        "database": database,
        "objects": objects,
        "count": len(objects),
        "app_restarted": restart,
        "warning": _state_loss_warning(package, restarted=restart),
    }


def _split_sql_script(sql: str) -> list[str]:
    statements: list[str] = []
    pending = ""
    for char in sql:
        pending += char
        if char == ";" and sqlite3.complete_statement(pending):
            statement = pending.strip()
            if statement:
                statements.append(statement)
            pending = ""
    if pending.strip():
        statements.append(pending.strip())
    return statements


def _leading_keyword(statement: str) -> str:
    cleaned = re.sub(r"\A(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", statement, flags=re.DOTALL)
    match = re.match(r"([A-Za-z]+)", cleaned)
    return match.group(1).upper() if match else ""


def _validate_mutations(statements: Sequence[str], parameters: SqlParameters) -> None:
    if not statements:
        raise UsageError("database execute needs at least one SQL statement")
    if len(statements) > 1 and parameters is not None:
        raise UsageError(
            "SQL parameters are only supported for a single execute statement",
            code="database_params",
        )
    allowed = {"INSERT", "UPDATE", "DELETE", "REPLACE", "WITH"}
    for statement in statements:
        keyword = _leading_keyword(statement)
        if keyword not in allowed:
            raise UsageError(
                f"database execute refuses {keyword or 'unknown'} statements",
                hint="Only data mutations are accepted: INSERT, UPDATE, DELETE, REPLACE, or WITH. "
                "AUA deliberately refuses schema changes, PRAGMA, ATTACH, and transaction control.",
                code="database_mutation_refused",
            )


def _schema_digest(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_master ORDER BY 1, 2, 3, 4"
    )
    payload = json.dumps(list(rows), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _foreign_key_violations(connection: sqlite3.Connection) -> set[tuple[Any, ...]]:
    return {tuple(row) for row in connection.execute("PRAGMA foreign_key_check")}


def _consolidate(connection: sqlite3.Connection, path: Path) -> bytes:
    consolidated = path.with_name(f"{path.name}.consolidated")
    with contextlib.suppress(OSError):
        consolidated.unlink()
    target = sqlite3.connect(str(consolidated))
    try:
        connection.backup(target)
    finally:
        target.close()
    verify = sqlite3.connect(str(consolidated))
    try:
        integrity = verify.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise DeviceError(
                f"mutated database failed integrity_check: {integrity}",
                code="database_integrity",
            )
    finally:
        verify.close()
    return consolidated.read_bytes()


def _replace_database(device: Device, package: str, database: str, data: bytes) -> None:
    device.write_app_file(package, _remote_path(database), data)
    device.remove_app_files(
        package,
        [_remote_path(database, suffix) for suffix in _SIDECAR_SUFFIXES],
    )


def _restore_files(
    device: Device,
    package: str,
    database: str,
    files: Mapping[str, bytes],
) -> None:
    device.write_app_file(package, _remote_path(database), files[database])
    sidecars = [_remote_path(database, suffix) for suffix in _SIDECAR_SUFFIXES]
    device.remove_app_files(package, sidecars)
    for suffix in _SIDECAR_SUFFIXES:
        name = f"{database}{suffix}"
        if name in files:
            device.write_app_file(package, _remote_path(database, suffix), files[name])


def execute_database(
    device: Device,
    cache_dir: str | Path,
    package: str,
    database: str,
    sql: str,
    *,
    parameters: SqlParameters = None,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    restart: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    package = _validate_package(package)
    database = _validate_database(database)
    if not confirmed:
        raise UsageError(
            "database execute mutates app data and requires explicit confirmation",
            hint="Review the SQL, then pass `--yes` (MCP: `confirmed: true`).",
            code="database_confirmation_required",
        )
    statements = _split_sql_script(sql)
    _validate_mutations(statements, parameters)
    normalized = _normalize_parameters(parameters)
    started = time.monotonic()
    with _stopped_app(device, package, restart=restart):
        original = _capture_files(device, package, database)
        backup = _store_backup(
            cache_dir=cache_dir,
            serial=device.serial,
            package=package,
            database=database,
            files=original,
            reason="before-execute",
        )
        with tempfile.TemporaryDirectory(prefix="aua-database-execute-") as temp_name:
            main = _materialize(Path(temp_name), original)
            connection = sqlite3.connect(str(main))
            results: list[dict[str, Any]] = []
            try:
                connection.execute("PRAGMA foreign_keys=ON")
                _configure_timeout(connection, timeout_ms)
                schema_before = _schema_digest(connection)
                foreign_keys_before = _foreign_key_violations(connection)
                connection.execute("BEGIN IMMEDIATE")
                for index, statement in enumerate(statements):
                    before = connection.total_changes
                    cursor = connection.execute(statement, normalized if index == 0 else ())
                    columns, rows, truncated = _cursor_rows(cursor, _DEFAULT_LIMIT)
                    results.append(
                        {
                            "statement": index + 1,
                            "kind": _leading_keyword(statement),
                            "changes": connection.total_changes - before,
                            "rowcount": cursor.rowcount,
                            "lastrowid": cursor.lastrowid,
                            "columns": columns,
                            "rows": rows,
                            "truncated": truncated,
                        }
                    )
                if _schema_digest(connection) != schema_before:
                    raise UsageError(
                        "database execute changed the schema",
                        hint="AUA only supports data mutations; use an app migration for schema changes.",
                        code="database_schema_changed",
                    )
                new_foreign_keys = _foreign_key_violations(connection) - foreign_keys_before
                if new_foreign_keys:
                    raise UsageError(
                        f"database execute introduced foreign-key violations: {list(new_foreign_keys)[:3]}",
                        code="database_foreign_key",
                    )
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    raise DeviceError(
                        f"database execute failed integrity_check: {integrity}",
                        code="database_integrity",
                    )
                connection.commit()
                connection.set_progress_handler(None, 0)
                consolidated = _consolidate(connection, main)
            except (sqlite3.Error, UsageError, DeviceError) as exc:
                with contextlib.suppress(sqlite3.Error):
                    connection.rollback()
                if isinstance(exc, sqlite3.Error):
                    raise UsageError(
                        f"database execute failed: {exc}",
                        hint="No device data was replaced; the original restore point is intact.",
                        code="database_sql",
                    ) from exc
                raise
            finally:
                connection.close()
        try:
            _replace_database(device, package, database, consolidated)
        except Exception as exc:
            with contextlib.suppress(Exception):
                _restore_files(device, package, database, original)
            raise DeviceError(
                f"failed to install the mutated database: {exc}",
                hint=f"Restore point {backup['id']} remains available with `aua db restore`.",
                code="database_replace_failed",
            ) from exc
    return {
        "ok": True,
        "action": "database-execute",
        "package": package,
        "database": database,
        "statements": results,
        "statement_count": len(results),
        "changes": sum(int(item["changes"]) for item in results),
        "backup": backup,
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "app_restarted": restart,
        "warning": _state_loss_warning(package, restarted=restart),
    }


def restore_database(
    device: Device,
    cache_dir: str | Path,
    package: str,
    database: str,
    backup_id: str,
    *,
    restart: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    package = _validate_package(package)
    database = _validate_database(database)
    if not confirmed:
        raise UsageError(
            "database restore replaces app data and requires explicit confirmation",
            hint="Review the backup id, then pass `--yes` (MCP: `confirmed: true`).",
            code="database_confirmation_required",
        )
    restored, target_files = _load_backup(
        cache_dir,
        device.serial,
        package,
        database,
        backup_id,
    )
    with _stopped_app(device, package, restart=restart):
        current = _capture_files(device, package, database)
        safety_backup = _store_backup(
            cache_dir=cache_dir,
            serial=device.serial,
            package=package,
            database=database,
            files=current,
            reason=f"before-restore-{backup_id}",
        )
        try:
            _restore_files(device, package, database, target_files)
        except Exception as exc:
            with contextlib.suppress(Exception):
                _restore_files(device, package, database, current)
            raise DeviceError(
                f"failed to restore database backup {backup_id}: {exc}",
                hint=f"The pre-restore safety backup is {safety_backup['id']}.",
                code="database_restore_failed",
            ) from exc
    return {
        "ok": True,
        "action": "database-restore",
        "package": package,
        "database": database,
        "restored_backup": restored,
        "safety_backup": safety_backup,
        "app_restarted": restart,
        "warning": _state_loss_warning(package, restarted=restart),
    }


__all__ = [
    "backup_database",
    "database_schema",
    "execute_database",
    "list_backups",
    "list_databases",
    "query_database",
    "restore_database",
]
