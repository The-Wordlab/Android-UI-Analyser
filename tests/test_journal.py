"""Tests for the append-only command journal."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser import journal


def test_database_sql_and_parameters_are_never_journaled() -> None:
    redacted = journal.redact_args(
        {
            "sql": "UPDATE users SET token = 'private-value'",
            "parameters": {"token": "another-private-value"},
        }
    )
    assert redacted == {
        "sql": "<redacted SQL: 40 chars>",
        "parameters": "<redacted>",
    }


def test_record_and_read_since(tmp_path: Path) -> None:
    import time

    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="test",
        cmd="has",
        args={"text": "hello", "password": "secret"},
        ok=True,
        duration_ms=12.5,
        result={"found": True},
    )
    time.sleep(0.02)
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="test",
        cmd="tap",
        args={"id": 4},
        ok=False,
        duration_ms=2000,
        error={"message": "element gone"},
    )

    events = journal.read_since(tmp_path, "emulator-5554", limit=10)
    assert len(events) == 2
    assert events[0]["cmd"] == "has"
    assert events[0]["ok"] is True
    assert events[0]["args"]["password"] == "<redacted>"
    assert events[1]["cmd"] == "tap"
    assert events[1]["ok"] is False

    since = int(events[0]["ts_ms"]) + 1
    newer = journal.read_since(tmp_path, "emulator-5554", since_ms=since, limit=10)
    assert len(newer) == 1
    assert newer[0]["cmd"] == "tap"

    stats = journal.failure_stats(events)
    assert stats["fail_count"] == 1
    assert stats["by_cmd"]["has"]["ok"] == 1
    assert stats["by_cmd"]["tap"]["fail"] == 1
    assert len(stats["slow"]) == 1
    assert stats["slow"][0]["cmd"] == "tap"


def test_read_since_hides_dashboard_by_default(tmp_path: Path) -> None:
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="cli",
        cmd="tap",
        args={"id": 1},
        ok=True,
    )
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="dashboard",
        cmd="dashboard_start",
        args={},
        ok=True,
    )
    visible = journal.read_since(tmp_path, "emulator-5554")
    assert len(visible) == 1
    assert visible[0]["source"] == "cli"
    all_ev = journal.read_since(tmp_path, "emulator-5554", include_dashboard=True)
    assert len(all_ev) == 2
