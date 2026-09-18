"""Platform-neutral SQLite schema introspection for app database services."""

from __future__ import annotations

import sqlite3
from typing import Any


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
