"""SQLite backend for :class:`~android_ui_analyser.memory.AppMemoryStore`.

Stores AppMap / SessionState JSON blobs in two tables. Optional one-shot migration
from the legacy ``memory/<pkg>/index.json`` layout when the DB is first opened empty.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .memory import AppMap, SessionState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS apps (
    package TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    serial TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
"""


class SqliteMemoryBackend:
    """Thin SQLite store for AppMap and SessionState JSON blobs."""

    def __init__(self, path: Path, *, migrate_from: Path | None = None) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        if migrate_from is not None:
            self._maybe_migrate(migrate_from)

    def close(self) -> None:
        self._conn.close()

    # -- apps -----------------------------------------------------------------

    def load_app(self, package: str) -> AppMap | None:
        row = self._conn.execute(
            "SELECT data FROM apps WHERE package = ?", (package,)
        ).fetchone()
        if row is None:
            return None
        try:
            return AppMap.model_validate_json(row[0])
        except Exception:  # pragma: no cover - corrupt row → treat as absent
            return None

    def save_app(self, app: AppMap) -> None:
        self._conn.execute(
            "INSERT INTO apps(package, data, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(package) DO UPDATE SET data = excluded.data, "
            "updated_at = excluded.updated_at",
            (app.package, app.model_dump_json(), time.time()),
        )
        self._conn.commit()

    def list_apps(self) -> list[str]:
        rows = self._conn.execute("SELECT package FROM apps ORDER BY package").fetchall()
        return [r[0] for r in rows]

    def delete_app(self, package: str) -> bool:
        cur = self._conn.execute("DELETE FROM apps WHERE package = ?", (package,))
        self._conn.commit()
        return cur.rowcount > 0

    # -- sessions -------------------------------------------------------------

    def load_session(self, serial: str) -> SessionState:
        row = self._conn.execute(
            "SELECT data FROM sessions WHERE serial = ?", (serial,)
        ).fetchone()
        if row is not None:
            try:
                return SessionState.model_validate_json(row[0])
            except Exception:  # pragma: no cover
                pass
        return SessionState()

    def save_session(self, serial: str, sess: SessionState) -> None:
        self._conn.execute(
            "INSERT INTO sessions(serial, data) VALUES (?, ?) "
            "ON CONFLICT(serial) DO UPDATE SET data = excluded.data",
            (serial, sess.model_dump_json()),
        )
        self._conn.commit()

    def latest_session(self, package: str) -> SessionState | None:
        rows = self._conn.execute("SELECT data FROM sessions ORDER BY rowid DESC").fetchall()
        for (raw,) in rows:
            try:
                session = SessionState.model_validate_json(raw)
            except Exception:  # pragma: no cover - skip corrupt cursor
                continue
            if session.package == package:
                return session
        return None

    # -- migration ------------------------------------------------------------

    def _maybe_migrate(self, migrate_from: Path) -> None:
        """Import ``memory/*/index.json`` once when the apps table is empty."""
        count = self._conn.execute("SELECT COUNT(*) FROM apps").fetchone()[0]
        if count or not migrate_from.is_dir():
            return
        now = time.time()
        for child in sorted(migrate_from.iterdir()):
            index = child / "index.json"
            if not index.is_file():
                continue
            try:
                raw = index.read_text(encoding="utf-8")
                app = AppMap.model_validate_json(raw)
            except Exception:  # pragma: no cover - skip corrupt legacy files
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO apps(package, data, updated_at) VALUES (?, ?, ?)",
                (app.package, app.model_dump_json(), now),
            )
        self._conn.commit()
