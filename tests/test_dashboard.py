"""Tests for the sneak-peek dashboard helpers (no real device required)."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError, UsageError


def test_latest_frame_picks_newest(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    root = tmp_path / "captures" / "emulator-5554"
    old = root / "sess-old" / "frames"
    new = root / "sess-new" / "frames"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"old")
    newer = new / "b.jpg"
    newer.write_bytes(b"new")
    os.utime(old / "a.jpg", (time.time() - 10, time.time() - 10))
    os.utime(newer, None)
    got = dash.latest_frame(tmp_path, "emulator-5554")
    assert got == newer


def test_recent_marks_reads_index(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    sess = tmp_path / "captures" / "emulator-5554" / "s1"
    (sess / "frames").mkdir(parents=True)
    idx = sess / "index.jsonl"
    idx.write_text(
        json.dumps({"t_ms": 1, "path": "frames/1.jpg", "hash": "a"})
        + "\n"
        + json.dumps({"t_ms": 2, "path": "frames/2.jpg", "hash": "b", "action": "tap:4"})
        + "\n",
        encoding="utf-8",
    )
    marks = dash.recent_marks(tmp_path, "emulator-5554")
    assert len(marks) == 1
    assert marks[0]["action"] == "tap:4"


def test_resolve_serial_requires_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import dashboard as dash

    class D:
        def __init__(self, serial: str) -> None:
            self.serial = serial
            self.state = "device"

    monkeypatch.setattr(
        "android_ui_analyser.device.list_devices",
        lambda: [D("emulator-5554"), D("emulator-5556")],
    )
    with pytest.raises(DeviceError, match="multiple"):
        dash.resolve_serial(None)
    assert dash.resolve_serial("emulator-5554") == "emulator-5554"


def test_resolve_dashboard_targets_grid_when_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import dashboard as dash

    monkeypatch.setattr(
        dash,
        "list_online_serials",
        lambda: ["emulator-5554", "emulator-5556"],
    )
    out = dash.resolve_dashboard_targets(None)
    assert out["mode"] == "grid"
    assert out["serials"] == ["emulator-5554", "emulator-5556"]
    assert out["focus"] is None
    forced = dash.resolve_dashboard_targets("emulator-5554")
    assert forced["mode"] == "detail"
    assert forced["focus"] == "emulator-5554"
    grid = dash.resolve_dashboard_targets(None, grid=True)
    assert grid["mode"] == "grid"


def test_owner_for_serial(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    rec = tmp_path / "emulator"
    rec.mkdir()
    (rec / "a.p5554.json").write_text(
        json.dumps(
            {
                "avd": "a",
                "serial": "emulator-5554",
                "owner": "agent-a",
                "started_by_aua": True,
            }
        ),
        encoding="utf-8",
    )
    assert dash.owner_for_serial(tmp_path, "emulator-5554") == "agent-a"
    assert dash.owner_for_serial(tmp_path, "emulator-5556") is None


def test_ensure_capture_falls_back_to_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    cfg.daemon.socket = str(tmp_path / "no-daemon.sock")

    import android_ui_analyser.capture_sidecar as cs

    monkeypatch.setattr(
        cs,
        "start",
        lambda **k: {
            "ok": True,
            "action": "capture-sidecar-start",
            "status": "started",
            "socket": "x",
        },
    )
    out = dash.ensure_capture(serial="emulator-5554", config=cfg)
    assert out["via"] == "sidecar"
    assert out["ok"] is True


def _dashboard_state(tmp_path: Path):
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    return dash._DashboardState(
        serials=["emulator-5554"],
        focus="emulator-5554",
        mode="detail",
        cache_dir=tmp_path,
        ensures={},
        poll_ms=500,
        config=cfg,
    )


def test_dashboard_database_view_is_in_detail_html() -> None:
    from android_ui_analyser import dashboard as dash

    assert "App database workspace" in dash._DASHBOARD_HTML
    assert "/api/database/" in dash._DASHBOARD_HTML
    assert "MUTATE " in dash._DASHBOARD_HTML
    assert "RESTORE " in dash._DASHBOARD_HTML
    assert "__DATABASE_TOKEN__" in dash._DASHBOARD_HTML


def test_dashboard_database_operations_delegate_and_require_typed_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import app_database

    state = _dashboard_state(tmp_path)
    device = object()
    monkeypatch.setattr("android_ui_analyser.device.connect", lambda serial: device)

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_list(actual_device: object, package: str) -> dict[str, object]:
        assert actual_device is device
        calls.append(("list", {"package": package}))
        return {"ok": True, "databases": [], "count": 0}

    def fake_execute(
        actual_device: object,
        cache_dir: Path,
        package: str,
        database: str,
        sql: str,
        **kwargs: object,
    ) -> dict[str, object]:
        assert actual_device is device
        assert cache_dir == tmp_path
        calls.append(
            (
                "execute",
                {
                    "package": package,
                    "database": database,
                    "sql": sql,
                    **kwargs,
                },
            )
        )
        return {"ok": True, "changes": 1}

    def fake_restore(
        actual_device: object,
        cache_dir: Path,
        package: str,
        database: str,
        backup_id: str,
        **kwargs: object,
    ) -> dict[str, object]:
        assert actual_device is device
        assert cache_dir == tmp_path
        calls.append(
            (
                "restore",
                {
                    "package": package,
                    "database": database,
                    "backup_id": backup_id,
                    **kwargs,
                },
            )
        )
        return {"ok": True, "backup_id": backup_id}

    monkeypatch.setattr(app_database, "list_databases", fake_list)
    monkeypatch.setattr(app_database, "execute_database", fake_execute)
    monkeypatch.setattr(app_database, "restore_database", fake_restore)

    listed = state.database_operation("list", {"package": "com.example.debug"})
    assert listed["ok"] is True
    assert calls == [("list", {"package": "com.example.debug"})]
    with pytest.raises(UsageError, match="not part of this dashboard session"):
        state.database_operation(
            "list",
            {"serial": "emulator-9999", "package": "com.example.debug"},
        )
    assert len(calls) == 1

    mutation = {
        "package": "com.example.debug",
        "database": "app.db",
        "sql": "UPDATE items SET done = 1",
    }
    with pytest.raises(UsageError, match="MUTATE app.db"):
        state.database_operation("execute", mutation)
    assert len(calls) == 1

    result = state.database_operation(
        "execute", {**mutation, "confirmation": "MUTATE app.db"}
    )
    assert result == {"ok": True, "changes": 1}
    assert calls[-1][0] == "execute"
    assert calls[-1][1]["confirmed"] is True

    restore = {
        "package": "com.example.debug",
        "database": "app.db",
        "backup_id": "backup-1",
    }
    with pytest.raises(UsageError, match="RESTORE backup-1"):
        state.database_operation("restore", restore)
    restored = state.database_operation(
        "restore", {**restore, "confirmation": "RESTORE backup-1"}
    )
    assert restored == {"ok": True, "backup_id": "backup-1"}
    assert calls[-1][0] == "restore"
    assert calls[-1][1]["confirmed"] is True


def test_dashboard_database_http_requires_session_token(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    state.database_token = "dashboard-test-token"
    state.database_operation = lambda action, payload: {
        "ok": True,
        "action": action,
        "package": payload.get("package"),
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}/"
    url = root_url + "api/database/list"
    body = json.dumps({"package": "com.example.debug"}).encode()
    try:
        with urllib.request.urlopen(root_url, timeout=2) as response:
            html = response.read().decode()
            assert (
                "script-src 'nonce-dashboard-test-token'"
                in response.headers["Content-Security-Policy"]
            )
        assert 'nonce="dashboard-test-token"' in html
        assert "__DATABASE_TOKEN__" not in html

        request = urllib.request.Request(url, data=body, method="POST")
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(request, timeout=2)
        assert unauthorized.value.code == 403

        authorized = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-AUA-Dashboard-Token": state.database_token,
            },
        )
        with urllib.request.urlopen(authorized, timeout=2) as response:
            assert (
                "script-src 'nonce-dashboard-test-token'"
                in response.headers["Content-Security-Policy"]
            )
            payload = json.loads(response.read())
        assert payload == {
            "ok": True,
            "action": "list",
            "package": "com.example.debug",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
