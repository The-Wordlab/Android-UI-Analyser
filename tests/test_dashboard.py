"""Tests for the sneak-peek dashboard helpers (no real device required)."""

from __future__ import annotations

import builtins
import json
import os
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.errors import DeviceError, UsageError


def test_dashboard_service_uses_one_exact_dedicated_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    assert dash.DEFAULT_DASHBOARD_PORT == 48765
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        with pytest.raises(UsageError, match="could not bind"):
            dash.run(
                port=port,
                open_browser=False,
                block=False,
                grid=True,
                exact_port=True,
                config=cfg,
            )


def test_dashboard_service_state_is_private(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    path = dash._write_service_state(tmp_path, {"access_token": "private-token"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["access_token"] == "private-token"


def test_dashboard_access_qr_is_private_and_uses_the_authenticated_lan_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg.cache.dir = "~/.aua-test-cache"
    access_url = "http://192.0.2.10:48765/?token=private-token"
    monkeypatch.setattr(
        dash,
        "service_status",
        lambda *a, **k: {
            "ok": True,
            "running": True,
            "lan_access_urls": [access_url],
        },
    )

    result = dash.create_access_qr(cfg)

    path = Path(result["path"])
    assert result["url"] == access_url
    assert path.parent == tmp_path / ".aua-test-cache"
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_dashboard_access_qr_requires_a_running_lan_dashboard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: {"running": False})
    with pytest.raises(UsageError, match="dashboard is not running"):
        dash.create_access_qr(cfg)

    monkeypatch.setattr(
        dash,
        "service_status",
        lambda *a, **k: {"running": True, "lan_access_urls": []},
    )
    with pytest.raises(UsageError, match="local-only"):
        dash.create_access_qr(cfg)


def test_dashboard_access_qr_reports_a_stale_tool_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import dashboard as dash

    real_import = builtins.__import__

    def import_without_qrcode(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "qrcode" or name.startswith("qrcode."):
            raise ModuleNotFoundError("No module named 'qrcode'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_qrcode)
    with pytest.raises(UsageError, match="QR support is not installed") as missing:
        dash._qr_png("http://192.0.2.10:48765/?token=test")
    assert "uv tool install --force --editable" in str(missing.value.hint)


def test_dashboard_service_start_is_idempotent_but_network_scope_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    running = {
        "ok": True,
        "running": True,
        "status": "running",
        "lan": False,
        "access_url": "http://127.0.0.1:48765/",
    }
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: dict(running))
    result = dash.start_service(cfg)
    assert result["status"] == "already_running"
    with pytest.raises(UsageError, match="different network scope"):
        dash.start_service(cfg, lan=True)


def test_dashboard_status_does_not_adopt_an_unrelated_port_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    monkeypatch.setattr(dash, "_dashboard_health", lambda _port: None)
    monkeypatch.setattr(dash, "_port_is_open", lambda _port: True)
    result = dash.service_status(cfg, port=48765)
    assert result["running"] is False
    assert result["ok"] is False
    assert result["status"] == "port_occupied"


def test_dashboard_stop_requires_matching_private_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    dash._write_service_state(tmp_path, {"pid": 111, "port": 48765})
    monkeypatch.setattr(
        dash,
        "_dashboard_health",
        lambda _port: {"service": "aua-dashboard-v1", "pid": 222},
    )
    with pytest.raises(UsageError, match="matching ownership"):
        dash.stop_service(cfg)


def test_dashboard_cli_exposes_background_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.cli import app

    monkeypatch.setattr(
        dash,
        "service_status",
        lambda *a, **k: {
            "ok": True,
            "running": True,
            "status": "running",
            "port": 48765,
        },
    )
    runner = CliRunner()
    status = runner.invoke(app, ["dashboard", "status"])
    assert status.exit_code == 0
    assert json.loads(status.output)["port"] == 48765

    help_result = runner.invoke(app, ["dashboard", "--help"])
    assert help_result.exit_code == 0
    for command in ("start", "status", "open", "qr", "stop", "run"):
        assert command in help_result.output

    qr_path = Path("/tmp/aua-dashboard-test-qr.png")
    monkeypatch.setattr(
        dash,
        "create_access_qr",
        lambda *a, **k: {
            "ok": True,
            "action": "dashboard-qr",
            "port": 48765,
            "url": "http://192.0.2.10:48765/?token=secret",
            "path": str(qr_path),
        },
    )
    qr = runner.invoke(app, ["dashboard", "qr", "--no-open"])
    assert qr.exit_code == 0
    assert json.loads(qr.output)["action"] == "dashboard-qr"


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


def test_resolve_dashboard_targets_grid_by_default(
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
    detail = dash.resolve_dashboard_targets(None, grid=False)
    assert detail["mode"] == "detail"
    assert detail["focus"] == "emulator-5554"

    monkeypatch.setattr(dash, "list_online_serials", lambda: ["emulator-5554"])
    one_device = dash.resolve_dashboard_targets(None)
    assert one_device["mode"] == "grid"
    assert one_device["focus"] is None


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


def test_device_runtime_status_reports_lease_and_idle_watchdog(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser import leases

    serial = "emulator-5554"
    now = time.time()
    rec = tmp_path / "emulator"
    rec.mkdir()
    meta = rec / "pixel.json"
    meta.write_text(
        json.dumps(
            {
                "avd": "Pixel",
                "instance": "Pixel",
                "serial": serial,
                "owner": "starter-a",
                "started_by_aua": True,
                "started_at": now - 600,
                "last_activity": now - 300,
                "idle_timeout_s": 1200,
                "watchdog_pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    assert leases.acquire(tmp_path, serial, owner="agent-a", ttl_s=900)

    runtime = dash.device_runtime_status(tmp_path, serial, now=now)

    assert runtime["lease"]["held"] is True
    assert runtime["lease"]["owner"] == "agent-a"
    assert runtime["watchdog"] == {
        "managed": True,
        "enabled": True,
        "running": True,
        "idle_s": 300.0,
        "timeout_s": 1200.0,
        "remaining_s": 900.0,
        "instance": "Pixel",
        "explicit": False,
    }

    meta.write_text(
        json.dumps(
            {
                "avd": "Pixel",
                "serial": serial,
                "started_by_aua": True,
                "started_at": now - 600,
                "last_activity": now - 300,
                "idle_timeout_s": 0,
                "idle_stop_explicit": True,
            }
        ),
        encoding="utf-8",
    )
    disabled = dash.device_runtime_status(tmp_path, serial, now=now)
    assert disabled["watchdog"]["enabled"] is False
    assert disabled["watchdog"]["running"] is False
    assert disabled["watchdog"]["remaining_s"] is None


def test_ensure_capture_falls_back_to_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    cfg.memory.dir = str(tmp_path)
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
    cfg.memory.dir = str(tmp_path)
    return dash._DashboardState(
        serials=["emulator-5554"],
        focus="emulator-5554",
        mode="detail",
        cache_dir=tmp_path,
        ensures={},
        poll_ms=500,
        config=cfg,
    )


def test_model_controls_start_the_host_when_no_agent_has_used_it_yet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import daemon as daemon_mod

    state = _dashboard_state(tmp_path)
    calls: list[tuple[str, str, dict[str, Any]]] = []
    replies = iter([None, {"ok": True, "loaded": True}])

    def fake_call(serial: str, cmd: str, timeout: float = 1.5, **args: Any) -> Any:
        calls.append((serial, cmd, args))
        return next(replies)

    starts: list[str | None] = []
    monkeypatch.setattr(state, "_daemon_call", fake_call)
    monkeypatch.setattr(
        daemon_mod,
        "start",
        lambda _config, *, serial=None: (
            starts.append(serial) or {"running": True, "status": "started"}
        ),
    )

    result = state._model_daemon_call(
        "emulator-5554", "model_action", action="load", provider="functiongemma"
    )

    assert result == {"ok": True, "loaded": True}
    assert starts == ["emulator-5554"]
    assert calls == [
        ("emulator-5554", "model_action", {"action": "load", "provider": "functiongemma"}),
        ("emulator-5554", "model_action", {"action": "load", "provider": "functiongemma"}),
    ]


def test_existing_model_playground_routes_agent_samples_to_the_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _dashboard_state(tmp_path)
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_call(serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append((serial, cmd, args))
        return {"ok": True, "status": "selected", "selected_id": 1}

    monkeypatch.setattr(state, "_model_daemon_call", fake_call)
    request = {
        "goal": "Open Settings",
        "candidates": [
            {"id": 0, "label": "Search"},
            {"id": 1, "label": "Settings"},
        ],
    }

    result = state.model_operation(
        "agent-test",
        {"serial": "emulator-5554", "provider": "agent_chain", "request": request},
    )

    assert result["selected_id"] == 1
    assert calls == [
        (
            "emulator-5554",
            "model_agent_test",
            {"timeout": 300.0, "provider": "agent_chain", "request": request},
        )
    ]


def test_dashboard_device_health_ping_never_adopts_the_action_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import daemon as daemon_mod

    state = _dashboard_state(tmp_path)
    socket_path = tmp_path / "daemon.sock"
    socket_path.touch()
    state.config.daemon.socket = str(socket_path)
    owners: list[str | None] = []

    class FakeClient:
        def __init__(self, _path: str, *, timeout: float, owner: str | None = None) -> None:
            owners.append(owner)

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def ping(self) -> bool:
            return True

        def call(self, _cmd: str, **_args: Any) -> dict[str, Any]:
            return {"ok": True, "result": {"screen": {}, "elements": []}}

    monkeypatch.setattr(daemon_mod, "DaemonClient", FakeClient)

    result = state._daemon_call("emulator-5554", "analyze", owner="dashboard-owner", journal=True)

    assert result == {"screen": {}, "elements": []}
    assert owners == [None, "dashboard-owner"]


@pytest.mark.parametrize(
    ("cmd", "args"),
    [
        ("tap", {"element_id": 4}),
        ("goto", {"goal": "Settings"}),
        ("flow_run", {"name": "sign-in"}),
    ],
)
def test_dashboard_never_retries_a_device_action_with_an_unknown_daemon_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cmd: str, args: dict[str, Any]
) -> None:
    from android_ui_analyser import daemon as daemon_mod
    from android_ui_analyser.errors import DaemonOutcomeUnknownError

    state = _dashboard_state(tmp_path)

    def uncertain(*_args: Any, **_kwargs: Any) -> Any:
        raise DaemonOutcomeUnknownError("the action may have arrived")

    monkeypatch.setattr(state, "_daemon_call", uncertain)
    monkeypatch.setattr(
        daemon_mod,
        "start",
        lambda *_args, **_kwargs: pytest.fail("an uncertain action must never start and retry"),
    )

    with pytest.raises(DaemonOutcomeUnknownError, match="may have arrived"):
        state._inspection_daemon_call("emulator-5554", cmd, **args)


def test_dashboard_analyze_returns_one_exact_overlay_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: ["emulator-5554"])
    calls: list[tuple[str, str, dict[str, Any]]] = []
    analyzed = {
        "schema_version": 1,
        "screen": {"width": 1080, "height": 2400, "package": "com.example"},
        "elements": [
            {
                "id": 7,
                "text": "Open app",
                "bounds": [100, 1800, 980, 1950],
                "clickable": True,
            }
        ],
        "meta": {"duration_ms": 12, "tier_used": "hierarchy"},
    }

    def fake_call(serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append((serial, cmd, args))
        Path(args["with_image"]).write_bytes(b"exact-analysis-frame")
        return analyzed

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    result = state.inspection_operation("analyze", {"serial": "emulator-5554"})

    # The panel is a client of the warm daemon like the CLI and MCP, so what it serves is the
    # default observation view, not the daemon's whole answer — see
    # test_the_dashboard_trims_what_it_serves.py. The frame's identity and geometry survive
    # that trim; `duration_ms`/`tier_used` are provenance and do not.
    assert result["view"] is result["result"]
    assert result["result"]["screen"] == analyzed["screen"]
    assert result["result"]["elements"] == [
        {"id": 7, "text": "Open app", "clickable": True, "bounds": [100, 1800, 980, 1950]}
    ]
    assert result["result"]["meta"] == {}
    assert state._inspections["emulator-5554"]["result"] == analyzed, (
        "the stored frame stays the daemon's answer; only the served copy is a view"
    )
    assert result["inspection_id"]
    assert calls[0][0:2] == ("emulator-5554", "analyze")
    # The overlay publishes numbered elements to a human, so those numbers must be recorded;
    # see test_dashboard_acts_on_identity_not_a_number.py for what withholding them cost.
    assert "no_cache" not in calls[0][2]
    assert state.inspection_frame("emulator-5554", result["inspection_id"]) == (
        b"exact-analysis-frame",
        "image/png",
    )


def test_dashboard_overlay_id_is_consumed_by_tap_and_analyze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: ["emulator-5554"])
    analyzed = {
        "screen": {"width": 100, "height": 200},
        "elements": [
            {
                "id": 4,
                "bounds": [10, 20, 80, 60],
                "clickable": True,
                "stable_key": "rid:openButton",
            }
        ],
        "meta": {},
    }
    source_frame = tmp_path / "source.png"
    source_frame.write_bytes(b"source")
    source = state._store_inspection("emulator-5554", "source-id", source_frame, analyzed, analyzed)
    assert source["inspection_id"] == "source-id"
    calls: list[dict[str, Any]] = []

    def fake_call(_serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append({"cmd": cmd, **args})
        Path(args["with_image"]).write_bytes(b"post-action")
        return {"ok": True, "action": "tap", "id": 4, "observation": analyzed}

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    result = state.inspection_operation(
        "tap",
        {"serial": "emulator-5554", "inspection_id": "source-id", "element_id": 4},
    )

    assert calls == [
        {
            "cmd": "tap",
            # The clicked element's own identity, not the frame-local ordinal: the ordinal
            # would be resolved through the per-device id cache this process does not own.
            "selector": {"key": "rid:openButton", "bounds": [10, 20, 80, 60]},
            "observe": True,
            "with_image": str(
                tmp_path
                / "dashboard-inspection"
                / "emulator-5554"
                / f"{result['inspection_id']}.png"
            ),
        }
    ]
    assert result["view"] == analyzed
    assert result["result"]["action"] == "tap"
    with pytest.raises(UsageError, match="no longer current"):
        state.inspection_operation(
            "tap",
            {"serial": "emulator-5554", "inspection_id": "source-id", "element_id": 4},
        )


def test_dashboard_forwards_app_scoped_logcat_to_the_selected_platform(tmp_path: Path) -> None:
    state = _dashboard_state(tmp_path)
    calls: list[tuple[str, int, str | None]] = []

    class FakePlatform:
        def recent_logs(
            self, target_id: str, *, limit: int = 80, app_id: str | None = None
        ) -> list[str]:
            calls.append((target_id, limit, app_id))
            return ["one app only"]

    state.platform = FakePlatform()

    assert state.log_lines("emulator-5554", 40, app_id="com.example.notes") == ["one app only"]
    assert calls == [("emulator-5554", 40, "com.example.notes")]


def test_map_payload_contains_expandable_route_steps_and_all_app_flows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from android_ui_analyser import flows as flows_mod
    from android_ui_analyser import memory as memory_mod

    class FakeStep:
        def __init__(self, **payload: Any) -> None:
            self.payload = payload

        def model_dump(self, *, exclude_none: bool = True) -> dict[str, Any]:
            return dict(self.payload)

    class FakeFlowStore:
        def __init__(self, _config: Any) -> None:
            pass

        def list(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "sign in",
                    "storage_name": "sign-in",
                    "ref": "com.example.notes:sign-in",
                    "app": "com.example.notes",
                    "path": str(tmp_path / "sign-in.yaml"),
                    "steps": 1,
                },
                {
                    "name": "dismiss system dialog",
                    "storage_name": "dismiss-system-dialog",
                    "ref": "dismiss-system-dialog",
                    "app": None,
                    "path": str(tmp_path / "dismiss-system-dialog.yaml"),
                    "steps": 1,
                },
            ]

        def load_file(self, path: Path) -> Any:
            if path.name == "sign-in.yaml":
                return SimpleNamespace(
                    steps=[FakeStep(kind="type", text="private input", data={"token": "x"})]
                )
            return SimpleNamespace(steps=[FakeStep(kind="press", key="back")])

    screen = SimpleNamespace(
        name="Home",
        id="screen-home",
        canonical_name="Home",
        logical_name="Home",
        aliases=["Main"],
        activity="MainActivity",
        visit_count=4,
        stale=False,
        context_id=None,
        surface="native",
        tier="mapped",
        anchors=[],
        notes="Landing screen",
        last_verified="2026-08-21T09:00:00Z",
    )
    route = SimpleNamespace(
        from_screen="Login",
        to_screen="Home",
        action="submit credentials",
        count=3,
        status="verified",
        id="route-login-home",
        context_id=None,
        guards=[],
        verification_count=2,
        last_seen="2026-08-21T09:00:00Z",
        steps=[FakeStep(kind="tap", target="Continue")],
    )
    app = SimpleNamespace(
        label="Notes",
        description="Example app",
        screens={"screen-home": screen},
        routes=[route],
    )

    class FakeAppMemoryStore:
        def __init__(self, _config: Any) -> None:
            pass

        def load(self, package: str) -> Any:
            assert package == "com.example.notes"
            return app

        def list_apps(self) -> list[str]:
            return ["com.example.notes"]

    monkeypatch.setattr(flows_mod, "FlowStore", FakeFlowStore)
    monkeypatch.setattr(memory_mod, "AppMemoryStore", FakeAppMemoryStore)
    state = _dashboard_state(tmp_path)
    monkeypatch.setattr(state, "foreground_package", lambda _serial: "com.example.notes")

    payload = state.map_payload("emulator-5554")

    assert payload["screens"][0]["name"] == "Home"
    assert payload["routes"][0]["steps"] == [{"kind": "tap", "target": "Continue"}]
    assert [flow["app"] for flow in payload["flows"]] == ["com.example.notes", None]
    assert payload["flows"][0]["steps_detail"][0]["text"] == "<redacted>"
    assert payload["flows"][0]["steps_detail"][0]["data"]["token"] == "<redacted>"
    assert all("path" not in flow for flow in payload["flows"])


def test_dashboard_navigation_actions_use_the_shared_daemon_engine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _dashboard_state(tmp_path)
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_call(serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append((serial, cmd, args))
        if cmd == "goto":
            return {"ok": True, "arrived": True, "target": args["goal"]}
        if cmd == "flow_run":
            return {"ok": True, "steps_run": 2}
        return {"ok": True, "deleted": True}

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)

    goto = state.navigation_operation(
        "goto", {"serial": "emulator-5554", "target": "Settings"}
    )
    assert goto["result"]["arrived"] is True

    with pytest.raises(UsageError, match="confirm this navigation-library action"):
        state.navigation_operation(
            "flow-run", {"serial": "emulator-5554", "ref": "com.example:sign-in"}
        )
    flow = state.navigation_operation(
        "flow-run",
        {
            "serial": "emulator-5554",
            "ref": "com.example:sign-in",
            "confirmation": "RUN FLOW com.example:sign-in",
        },
    )
    deleted = state.navigation_operation(
        "flow-delete",
        {
            "serial": "emulator-5554",
            "ref": "com.example:sign-in",
            "confirmation": "DELETE FLOW com.example:sign-in",
        },
    )

    assert flow["result"]["steps_run"] == 2
    assert deleted["result"]["deleted"] is True
    assert calls == [
        ("emulator-5554", "goto", {"timeout": 300.0, "goal": "Settings"}),
        (
            "emulator-5554",
            "flow_run",
            {"timeout": 300.0, "name": "com.example:sign-in"},
        ),
        (
            "emulator-5554",
            "flow_delete",
            {"timeout": 30.0, "name": "com.example:sign-in"},
        ),
    ]


def test_dashboard_can_clear_one_route_and_the_whole_navigation_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser.flows import FlowStore
    from android_ui_analyser.memory import AppMap, AppMemoryStore, RouteEdge

    state = _dashboard_state(tmp_path)
    memory = AppMemoryStore(state.config.memory)
    memory.save(
        AppMap(
            package="com.example.notes",
            routes=[
                RouteEdge(
                    id="route-home-settings",
                    from_screen="Home",
                    to_screen="Settings",
                    action="tap Settings",
                    last_seen="2026-08-21T14:00:00Z",
                )
            ],
        )
    )
    monkeypatch.setattr(
        state,
        "map_payload",
        lambda _serial: {"package": "com.example.notes", "known": True},
    )

    route = state.navigation_operation(
        "route-delete",
        {
            "serial": "emulator-5554",
            "package": "com.example.notes",
            "route_id": "route-home-settings",
            "confirmation": "DELETE ROUTE route-home-settings",
        },
    )
    assert route["deleted"] is True
    assert memory.load("com.example.notes").routes == []  # type: ignore[union-attr]

    memory.save(AppMap(package="com.example.second"))
    flow_root = FlowStore(state.config.memory).flows_dir()
    (flow_root / "com.example.notes").mkdir(parents=True)
    (flow_root / "global.yaml").write_text("name: global\nsteps: []\n", encoding="utf-8")
    (flow_root / "com.example.notes" / "open.yaml").write_text(
        "name: open\napp: com.example.notes\nsteps: []\n", encoding="utf-8"
    )

    with pytest.raises(UsageError, match="CLEAR ALL NAVIGATION"):
        state.navigation_operation("clear-all", {"serial": "emulator-5554"})
    cleared = state.navigation_operation(
        "clear-all",
        {"serial": "emulator-5554", "confirmation": "CLEAR ALL NAVIGATION"},
    )

    assert cleared["maps_deleted"] == 2
    assert cleared["flows_deleted"] == 2
    assert memory.list_apps() == []
    assert FlowStore(state.config.memory).files() == []


def test_dashboard_clear_journal_removes_compact_details_and_rotations(tmp_path: Path) -> None:
    from android_ui_analyser import journal

    state = _dashboard_state(tmp_path)
    roots = [
        journal.journal_path(tmp_path, "emulator-5554"),
        journal.journal_detail_path(tmp_path, "emulator-5554"),
        journal.journal_path(tmp_path, None),
        journal.journal_detail_path(tmp_path, None),
    ]
    for root in roots:
        root.write_text("{}\n", encoding="utf-8")
        root.with_suffix(root.suffix + ".1").write_text("{}\n", encoding="utf-8")

    with pytest.raises(UsageError, match="CLEAR JOURNAL emulator-5554"):
        state.journal_operation("clear", {"serial": "emulator-5554"})
    result = state.journal_operation(
        "clear",
        {
            "serial": "emulator-5554",
            "confirmation": "CLEAR JOURNAL emulator-5554",
        },
    )

    assert result["deleted"] == 8
    assert not any(
        path.exists()
        for root in roots
        for path in (root, root.with_suffix(root.suffix + ".1"))
    )


def test_dashboard_database_view_is_in_detail_html() -> None:
    from android_ui_analyser import dashboard as dash

    assert "App database workspace" in dash._DASHBOARD_HTML
    assert "/api/database/" in dash._DASHBOARD_HTML
    assert "MUTATE " in dash._DASHBOARD_HTML
    assert "RESTORE " in dash._DASHBOARD_HTML
    assert "__DATABASE_TOKEN__" in dash._DASHBOARD_HTML


def test_dashboard_journal_rows_expand_request_and_response_as_text() -> None:
    from android_ui_analyser import dashboard as dash

    assert "document.createElement('details')" in dash._DASHBOARD_HTML
    assert "Agent request" in dash._DASHBOARD_HTML
    assert "AUA response" in dash._DASHBOARD_HTML
    assert "requestPayload.textContent" in dash._DASHBOARD_HTML
    assert "responsePayload.textContent" in dash._DASHBOARD_HTML
    assert "'/api/event'" in dash._DASHBOARD_HTML
    assert "X-AUA-Dashboard-Token" in dash._DASHBOARD_HTML
    assert "refreshOpenEventExchanges" in dash._DASHBOARD_HTML
    assert "details.dataset.refreshPending = 'true'" in dash._DASHBOARD_HTML
    assert "d.detail_revision" in dash._DASHBOARD_HTML
    assert 'id="lease"' in dash._DASHBOARD_HTML
    assert 'id="watchdog"' in dash._DASHBOARD_HTML
    assert "leaseText(lease)" in dash._DASHBOARD_HTML
    assert "watchdogText(watchdog, lease)" in dash._DASHBOARD_HTML
    assert "</script>" not in dash._script_json("</script><script>alert(1)</script>")
    assert "\\u003c/script\\u003e" in dash._script_json("</script>")


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

    def fake_query(
        actual_device: object,
        package: str,
        database: str,
        sql: str,
        **kwargs: object,
    ) -> dict[str, object]:
        assert actual_device is device
        calls.append(
            (
                "query",
                {
                    "package": package,
                    "database": database,
                    "sql": sql,
                    **kwargs,
                },
            )
        )
        return {"ok": True, "rows": []}

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
    monkeypatch.setattr(app_database, "query_database", fake_query)
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

    query = {
        "package": "com.example.debug",
        "database": "app.db",
        "sql": "SELECT 1",
    }
    queried = state.database_operation("query", query)
    assert queried == {"ok": True, "rows": []}
    assert calls[-1][1]["live"] is True

    state.database_operation("query", {**query, "live": False})
    assert calls[-1][1]["live"] is False

    mutation = {
        "package": "com.example.debug",
        "database": "app.db",
        "sql": "UPDATE items SET done = 1",
    }
    call_count = len(calls)
    with pytest.raises(UsageError, match="MUTATE app.db"):
        state.database_operation("execute", mutation)
    assert len(calls) == call_count

    result = state.database_operation("execute", {**mutation, "confirmation": "MUTATE app.db"})
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
    restored = state.database_operation("restore", {**restore, "confirmation": "RESTORE backup-1"})
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


def test_dashboard_serves_the_authenticated_phone_qr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    state.bind_host = "0.0.0.0"
    state.require_auth = True
    state.access_token = "private-access-token"
    monkeypatch.setattr(dash, "_lan_addresses", lambda: ["192.0.2.10"])
    encoded: list[str] = []

    def fake_svg(value: str) -> bytes:
        encoded.append(value)
        return b"<svg>phone qr</svg>"

    monkeypatch.setattr(dash, "_qr_svg", fake_svg)
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}/"
    phone_url = f"http://192.0.2.10:{server.server_port}/?token=private-access-token"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    try:
        with opener.open(root_url + "?token=private-access-token", timeout=2) as response:
            html = response.read().decode()
        assert phone_url in html

        with opener.open(root_url + "api/dashboard-access-qr.svg", timeout=2) as response:
            assert response.headers["Content-Type"] == "image/svg+xml; charset=utf-8"
            assert response.read() == b"<svg>phone qr</svg>"
        assert encoded == [phone_url]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_journal_detail_is_token_protected_and_serial_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser import journal

    state = _dashboard_state(tmp_path)
    state.database_token = "dashboard-test-token"
    full_only = "full response payload " * 40
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="mcp",
        cmd="analyze",
        args={"source": "auto"},
        result={
            "ok": True,
            "full_only": full_only,
            "elements": [{"id": 1, "text": "Ready"}],
        },
    )
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-9999",
        source="mcp",
        cmd="analyze",
        args={"source": "auto"},
        result={"ok": True, "other_device_private": "must stay scoped"},
    )
    event = state.journal_bundle(limit=1)["events"][0]
    detail_id = event["detail_id"]
    other_detail_id = journal.read_since(tmp_path, "emulator-9999", limit=1)[0]["detail_id"]
    assert "full_only" not in event["result"]
    monkeypatch.setattr(
        dash,
        "list_online_serials",
        lambda _config=None: ["emulator-5554", "emulator-9999"],
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}/"
    detail_url = root_url + f"api/event?detail_id={detail_id}&serial=emulator-5554"
    try:
        with urllib.request.urlopen(
            root_url + "api/events?serial=emulator-5554&limit=1", timeout=2
        ) as response:
            compact = json.loads(response.read())
        assert compact["events"][0]["detail_id"] == detail_id
        initial_detail_revision = compact["detail_revision"]
        assert initial_detail_revision
        assert full_only not in json.dumps(compact)

        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(detail_url, timeout=2)
        assert unauthorized.value.code == 403

        authorized = urllib.request.Request(
            detail_url,
            headers={"X-AUA-Dashboard-Token": state.database_token},
        )
        with urllib.request.urlopen(authorized, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["detail"]["request"] == {
            "cmd": "analyze",
            "args": {"source": "auto"},
        }
        assert payload["detail"]["response"]["result"]["full_only"] == full_only
        assert payload["detail"]["response"]["result"]["elements"] == [{"id": 1, "text": "Ready"}]

        assert journal.record_emitted_response(
            cache_dir=tmp_path,
            serial="emulator-5554",
            invocation_id="dashboard-live-revision",
            detail_id=detail_id,
            cmd="analyze",
            args={"source": "auto"},
            result={"ok": True, "full_only": "final agent response"},
        )
        with urllib.request.urlopen(
            root_url + "api/events?serial=emulator-5554&limit=1", timeout=2
        ) as response:
            revised_compact = json.loads(response.read())
        assert revised_compact["detail_revision"] != initial_detail_revision
        with urllib.request.urlopen(authorized, timeout=2) as response:
            revised_payload = json.loads(response.read())
        assert revised_payload["detail"]["response"]["result"]["full_only"] == (
            "final agent response"
        )

        with urllib.request.urlopen(root_url + "api/devices", timeout=2) as response:
            devices = json.loads(response.read())
        assert devices["mode"] == "detail"
        assert [device["serial"] for device in devices["devices"]] == ["emulator-5554"]
        assert state.serials == ["emulator-5554"]

        wrong_serial = urllib.request.Request(
            root_url + f"api/event?detail_id={other_detail_id}&serial=emulator-9999",
            headers={"X-AUA-Dashboard-Token": state.database_token},
        )
        with pytest.raises(urllib.error.HTTPError) as not_found:
            urllib.request.urlopen(wrong_serial, timeout=2)
        assert not_found.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as events_out_of_scope:
            urllib.request.urlopen(root_url + "api/events?serial=emulator-9999&limit=1", timeout=2)
        assert events_out_of_scope.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as logs_out_of_scope:
            urllib.request.urlopen(root_url + "api/logcat?serial=emulator-9999", timeout=2)
        assert logs_out_of_scope.value.code == 400

        injected_serial = root_url + "?serial=%27%3BglobalThis.SERIAL_XSS%3Dtrue%3B%2F%2F"
        with pytest.raises(urllib.error.HTTPError) as injected:
            urllib.request.urlopen(injected_serial, timeout=2)
        assert injected.value.code == 404
        assert state.database_token not in injected.value.read().decode()

        malformed = urllib.request.Request(
            root_url + "api/event?detail_id=..%2F..%2Fprivate",
            headers={"X-AUA-Dashboard-Token": state.database_token},
        )
        with pytest.raises(urllib.error.HTTPError) as invalid:
            urllib.request.urlopen(malformed, timeout=2)
        assert invalid.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_opens_an_empty_grid_when_no_device_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aua dashboard` is a watcher, not a device command: it must open regardless."""
    from android_ui_analyser import dashboard as dash

    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    grid = dash.resolve_dashboard_targets(None)
    assert grid["mode"] == "grid"
    assert grid["serials"] == []
    assert grid["focus"] is None
    assert grid["discovery_error"] is None
    # --detail with nothing to focus still opens; the grid discovers devices later.
    detail = dash.resolve_dashboard_targets(None, grid=False)
    assert detail["mode"] == "grid"
    assert detail["serials"] == []


def test_dashboard_opens_when_device_discovery_itself_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import dashboard as dash

    def boom(*_a: object, **_k: object) -> list[str]:
        raise DeviceError("adb server would not start")

    monkeypatch.setattr(dash, "list_online_serials", boom)
    out = dash.resolve_dashboard_targets(None)
    assert out["mode"] == "grid"
    assert out["serials"] == []
    assert "adb server would not start" in str(out["discovery_error"])
    # A pinned serial is still watchable when discovery is broken.
    pinned = dash.resolve_dashboard_targets("emulator-5554")
    assert pinned["mode"] == "detail"
    assert pinned["focus"] == "emulator-5554"


def test_devices_payload_reports_an_empty_grid_and_its_discovery_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    state.mode = "grid"
    state.serials = []
    state.focus = None
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    payload = state.devices_payload()
    assert payload["ok"] is True
    assert payload["devices"] == []
    assert payload["discovery_error"] is None

    def boom(*_a: object, **_k: object) -> list[str]:
        raise DeviceError("adb went away")

    monkeypatch.setattr(dash, "list_online_serials", boom)
    broken = state.devices_payload()
    assert broken["devices"] == []
    assert "adb went away" in str(broken["discovery_error"])


def test_grid_removes_detached_devices_and_does_not_resurrect_them_on_discovery_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    state.mode = "grid"
    state.serials = ["emulator-5554", "emulator-5556"]
    state._online_serials = {"emulator-5554", "emulator-5556"}
    state.ensures = {
        "emulator-5554": {"ok": True, "via": "daemon"},
        "emulator-5556": {"ok": True, "via": "daemon"},
    }
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: ["emulator-5554"])
    monkeypatch.setattr(
        state,
        "device_tile",
        lambda serial: {"serial": serial},
    )

    payload = state.devices_payload()
    assert payload["devices"] == [{"serial": "emulator-5554"}]
    assert payload["detached_serials"] == ["emulator-5556"]
    assert "emulator-5556" not in state.ensures

    def boom(*_a: object, **_k: object) -> list[str]:
        raise DeviceError("temporary discovery failure")

    monkeypatch.setattr(dash, "list_online_serials", boom)
    retry = state.devices_payload()
    assert retry["devices"] == [{"serial": "emulator-5554"}]
    assert "temporary discovery failure" in str(retry["discovery_error"])


def test_detail_status_reports_detached_before_touching_stale_device_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    monkeypatch.setattr(
        state,
        "foreground_package",
        lambda *_a, **_k: pytest.fail("detached status must not connect to the device"),
    )
    result = state.status("emulator-5554")
    assert result == {
        "ok": False,
        "detached": True,
        "serial": "emulator-5554",
        "online_serials": [],
        "error": "device 'emulator-5554' is no longer attached",
    }


def test_analyze_refuses_a_detached_device_before_calling_the_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    monkeypatch.setattr(
        state,
        "_inspection_daemon_call",
        lambda *_a, **_k: pytest.fail("detached Analyze must not reach the daemon"),
    )
    with pytest.raises(UsageError) as raised:
        state.inspection_operation("analyze", {"serial": "emulator-5554"})
    assert raised.value.to_dict()["error"]["code"] == "dashboard_device_detached"


def test_frame_token_tracks_the_frame_actually_served(tmp_path: Path) -> None:
    """The tile cache key must change exactly when the served bytes change."""
    from android_ui_analyser import dashboard as dash  # noqa: F401

    state = _dashboard_state(tmp_path)
    frames = tmp_path / "captures" / "emulator-5554" / "s1" / "frames"
    frames.mkdir(parents=True)
    shot = frames / "1.jpg"
    shot.write_bytes(b"first")
    state.note_capture_live("emulator-5554", True)

    first = state.frame_token("emulator-5554")
    assert state.frame_token("emulator-5554") == first  # deduped screen → no refetch
    time.sleep(0.02)
    shot.write_bytes(b"second")
    os.utime(shot, None)
    assert state.frame_token("emulator-5554") != first


def test_frame_token_advances_while_capture_is_not_running(tmp_path: Path) -> None:
    """No live capture means the bytes come from a screencap, so the tile must refetch."""
    state = _dashboard_state(tmp_path)
    state.note_capture_live("emulator-5554", False)
    first = state.frame_token("emulator-5554")
    time.sleep(0.02)
    assert state.frame_token("emulator-5554") != first


def test_grid_tiles_refetch_the_frame_instead_of_pinning_a_constant_url() -> None:
    from android_ui_analyser import dashboard as dash

    html = dash._DASHBOARD_HTML
    # The old cache key stripped its own timestamp, so it never changed and the tile
    # image was fetched exactly once for the lifetime of the page.
    assert "src.indexOf('&t=')" not in html
    assert "d.frame_token" in html
    assert "s.frame_token" in html


def test_frame_bytes_screencaps_past_a_dead_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _dashboard_state(tmp_path)
    frames = tmp_path / "captures" / "emulator-5554" / "s1" / "frames"
    frames.mkdir(parents=True)
    shot = frames / "1.jpg"
    shot.write_bytes(b"stale-jpeg")
    long_ago = time.time() - 300
    os.utime(shot, (long_ago, long_ago))

    live_png = b"\x89PNG\r\n\x1a\nlive-bytes"

    class _Img:
        png_bytes = live_png

    class _Dev:
        def screenshot(self) -> _Img:
            return _Img()

    monkeypatch.setattr(state.platform, "connect", lambda _ser: _Dev())

    state.note_capture_live("emulator-5554", False)
    data, mime = state.frame_bytes("emulator-5554")
    assert data == live_png
    assert mime == "image/png"

    # A healthy capture dedupes unchanged screens, so an old file is still correct.
    state._fallback.clear()
    state.note_capture_live("emulator-5554", True)
    assert state.frame_bytes("emulator-5554")[0] == b"stale-jpeg"


def test_dashboard_serves_with_no_devices_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    info = dash.run(
        port=0,
        cache_dir=tmp_path,
        config=cfg,
        open_browser=False,
        block=False,
        grid=True,
    )
    assert info["ok"] is True
    assert info["mode"] == "grid"
    assert info["serials"] == []
    try:
        with urllib.request.urlopen(info["url"], timeout=2) as response:
            assert response.status == 200
        with urllib.request.urlopen(info["url"] + "api/devices", timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["devices"] == []
    finally:
        dash.shutdown(info)


def test_lan_dashboard_serves_token_entry_then_uses_an_http_only_cookie(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    monkeypatch.setattr(dash, "_lan_addresses", lambda: ["192.0.2.10"])
    state = _dashboard_state(tmp_path)
    state.bind_host = "0.0.0.0"
    state.require_auth = True
    state.access_token = "phone-access-token"
    monkeypatch.setattr(state, "devices_payload", lambda: {"ok": True})
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        with urllib.request.urlopen(root + "api/health", timeout=2) as response:
            health = json.loads(response.read())
        assert health["service"] == "aua-dashboard-v1"
        assert health["authenticated"] is True

        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(root, timeout=2)
        assert unauthorized.value.code == 401

        jar = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        with opener.open(root + "?token=phone-access-token", timeout=2) as response:
            assert response.status == 200
            assert "token=phone-access-token" in response.geturl()
            html = response.read().decode()
        assert "cleanAccessUrl.searchParams.delete('token')" in html
        assert "window.history.replaceState" in html
        cookie = next(iter(jar))
        assert cookie.name == "AUA_DASHBOARD_ACCESS"
        assert cookie.has_nonstandard_attr("HttpOnly")
        with opener.open(root + "api/devices", timeout=2) as response:
            assert json.loads(response.read())["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_detached_detail_url_and_status_return_to_the_live_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _dashboard_state(tmp_path)
    state.mode = "grid"
    state.serials = []
    state._online_serials = set()
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: [])
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        with urllib.request.urlopen(root + "?serial=emulator-5556", timeout=2) as response:
            assert "detached=emulator-5556" in response.geturl()
            assert response.status == 200
        with urllib.request.urlopen(
            root + "api/status?serial=emulator-5556", timeout=2
        ) as response:
            payload = json.loads(response.read())
        assert payload["detached"] is True
        assert payload["serial"] == "emulator-5556"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# --------------------------------------------------------------------------
# proxy panel
# --------------------------------------------------------------------------


class _FakeProxyService:
    """Just the members of the PROXY capability contract the panel actually reads."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.health: dict[str, Any] = {
            "ok": True,
            "state": "healthy",
            "port": 8080,
            "intercepting": True,
            "checks": {},
        }
        self.doc: dict[str, Any] = {"mode": "off", "owner": "agent-a", "rules": []}
        self.flows: list[dict[str, Any]] = []
        self.bodies: list[dict[str, Any]] = []
        self.flow_reads: list[str | None] = []
        self.body_reads: list[str | None] = []

    def proxy_health(self, serial: str, cache_dir: Any, *, self_heal: bool = False) -> dict:
        assert self_heal is False, "the dashboard must never heal the device"
        return dict(self.health, serial=serial)

    def rules_path(self, cache_dir: Any, serial: str | None = None) -> Path:
        return self.root / f"mock_rules.{serial or 'shared'}.json"

    def load_doc(self, path: Path) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.doc.items()}

    def backfill_rule_ids(self, rules: list[dict]) -> tuple[list[dict], bool]:
        return list(rules), False

    def read_flows_since(
        self, cache_dir: Any, since_ts: float, serial: str | None = None
    ) -> list[dict]:
        self.flow_reads.append(serial)
        return [f for f in self.flows if float(f.get("ts") or 0) > since_ts]

    def read_flow_bodies(self, cache_dir: Any, serial: str | None = None) -> list[dict]:
        self.body_reads.append(serial)
        return list(self.bodies)


def _proxy_state(tmp_path: Path, service: Any):
    state = _dashboard_state(tmp_path)
    state._proxy_service = lambda: service  # type: ignore[method-assign]
    return state


def test_proxy_payload_reports_health_rules_and_live_traffic(tmp_path: Path) -> None:
    from typing import Any  # noqa: F401

    svc = _FakeProxyService(tmp_path)
    svc.doc["rules"] = [
        {"id": "r1", "action": "stub", "request": {"method": "GET", "path": "/v1/me"}},
        {"id": "r2", "action": "rewrite", "match": {"method": "GET", "path": "/v1/feed"}},
    ]
    svc.health["state"] = "healthy"
    state = _proxy_state(tmp_path, svc)
    before = state._proxy_opened_at - 100
    svc.flows = [
        {"n": 1, "ts": before, "method": "GET", "path": "/earlier", "status": 200},
        {
            "n": 2,
            "ts": time.time() + 1,
            "method": "GET",
            "path": "/v1/me",
            "status": 200,
            "action": "stub",
            "rule": "r1",
        },
        {"n": 3, "ts": time.time() + 1, "method": "POST", "path": "/v1/track", "status": 204},
    ]

    out = state.proxy_payload("emulator-5554")
    assert out["ok"] is True and out["supported"] is True
    assert out["on"] is True and out["intercepting"] is True and out["port"] == 8080
    # Traffic from before the page opened is the whole point — it is what the agent did.
    assert [f["n"] for f in out["flows"]] == [1, 2, 3]
    assert [f["live"] for f in out["flows"]] == [False, True, True]
    assert out["flow_count"] == 3
    assert out["manipulated"] == 1
    by_id = {r["id"]: r for r in out["rules"]}
    assert by_id["r1"]["fired"] == 1
    assert by_id["r2"]["fired"] == 0


def test_proxy_payload_does_not_call_a_clean_unproxied_device_proxied(tmp_path: Path) -> None:
    """`proxy_health` reports ok for an unproxied device too — its network path is fine.
    Reading that as "proxy on" made the panel claim interception on a clean device."""
    svc = _FakeProxyService(tmp_path)
    svc.health = {"ok": True, "state": "unproxied", "intercepting": False, "port": None}
    state = _proxy_state(tmp_path, svc)
    out = state.proxy_payload("emulator-5554")
    assert out["on"] is False
    assert out["intercepting"] is False
    assert out["state"] == "unproxied"


def test_proxy_payload_survives_a_platform_without_a_proxy(tmp_path: Path) -> None:
    from android_ui_analyser.errors import UnsupportedPlatformCapabilityError

    state = _dashboard_state(tmp_path)

    def refuse() -> Any:
        raise UnsupportedPlatformCapabilityError("fake", "proxy")

    state._proxy_service = refuse  # type: ignore[method-assign]
    out = state.proxy_payload("emulator-5554")
    assert out["ok"] is False
    assert out["supported"] is False
    assert "error" in out


def test_proxy_payload_survives_a_broken_health_probe(tmp_path: Path) -> None:
    svc = _FakeProxyService(tmp_path)

    def boom(*_a: object, **_k: object) -> dict:
        raise OSError("adb reverse blew up")

    svc.proxy_health = boom  # type: ignore[assignment]
    state = _proxy_state(tmp_path, svc)
    out = state.proxy_payload("emulator-5554")
    assert out["ok"] is True
    assert out["health"] is None
    assert out["on"] is False


def test_proxy_flow_detail_returns_the_captured_exchange(tmp_path: Path) -> None:
    svc = _FakeProxyService(tmp_path)
    svc.bodies = [
        {"n": 7, "method": "POST", "path": "/v1/chat", "request_body": "{}"},
        {"n": 8, "method": "GET", "path": "/v1/feed", "response_body": "[]"},
    ]
    state = _proxy_state(tmp_path, svc)
    found = state.proxy_flow_detail(8)
    assert found["ok"] is True
    assert found["flow"]["path"] == "/v1/feed"
    missing = state.proxy_flow_detail(99)
    assert missing["ok"] is False
    assert missing["error"]["code"] == "proxy_flow_body_missing"


def test_proxy_operation_routes_through_the_engine(tmp_path: Path) -> None:
    state = _dashboard_state(tmp_path)
    calls: list[tuple[str, dict[str, Any]]] = []

    class _FakeEngine:
        def mock_list(self) -> dict:
            calls.append(("list", {}))
            return {"ok": True, "action": "mock-list"}

        def mock_clear(self) -> dict:
            calls.append(("clear", {}))
            return {"ok": True, "action": "mock-clear"}

        def mock_rm(self, rule_id: str) -> dict:
            calls.append(("rm", {"id": rule_id}))
            return {"ok": True, "action": "mock-rm"}

        def mock_map(self, method: str, path: str, **kw: Any) -> dict:
            calls.append(("stub", {"method": method, "path": path, **kw}))
            return {"ok": True, "action": "mock-map"}

        def mock_rewrite(self, method: str, path: str, **kw: Any) -> dict:
            calls.append(("rewrite", {"method": method, "path": path, **kw}))
            return {"ok": True, "action": "mock-rewrite"}

    state._engine = _FakeEngine()

    assert state.proxy_operation("list", {})["action"] == "mock-list"
    state.proxy_operation("rm", {"id": "r9"})
    state.proxy_operation(
        "stub", {"method": "get", "path": "/v1/me", "status": 402, "body": '{"x":1}'}
    )
    state.proxy_operation(
        "rewrite",
        {
            "method": "GET",
            "path": "/v1/feed",
            "host": "api.example.com",
            "status": 429,
            "set_json": {"items[0].title": "patched"},
            "delete_json": ["meta.cursor"],
            "times": 2,
        },
    )
    kinds = [c[0] for c in calls]
    assert kinds == ["list", "rm", "stub", "rewrite"]
    stub = dict(calls[2][1])
    assert stub["status"] == 402
    assert stub["serial"] == "emulator-5554"
    rewrite = dict(calls[3][1])
    assert rewrite["host"] == "api.example.com"
    assert rewrite["set_json"] == {"items[0].title": "patched"}
    assert rewrite["delete_json"] == ["meta.cursor"]
    assert rewrite["times"] == 2

    with pytest.raises(UsageError):
        state.proxy_operation("drop-tables", {})


def test_proxy_operation_omits_status_when_the_browser_left_it_blank(tmp_path: Path) -> None:
    """A rewrite with no status must not silently become a 200."""
    state = _dashboard_state(tmp_path)
    seen: dict[str, Any] = {}

    class _FakeEngine:
        def mock_rewrite(self, method: str, path: str, **kw: Any) -> dict:
            seen.update(kw)
            return {"ok": True}

    state._engine = _FakeEngine()
    state.proxy_operation(
        "rewrite", {"method": "GET", "path": "/v1/feed", "status": "", "set_json": {"a": 1}}
    )
    assert seen["status"] is None


def test_proxy_operation_rejects_a_device_outside_this_session(tmp_path: Path) -> None:
    state = _dashboard_state(tmp_path)
    state._engine = object()
    with pytest.raises(UsageError) as err:
        state.proxy_operation("list", {"serial": "emulator-9999"})
    assert err.value.to_dict()["error"]["code"] == "dashboard_device_scope"


def test_proxy_http_views_are_serial_scoped_and_writes_need_the_token(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash

    svc = _FakeProxyService(tmp_path)
    state = _proxy_state(tmp_path, svc)
    state.database_token = "dashboard-test-token"

    class _FakeEngine:
        def mock_list(self) -> dict:
            return {"ok": True, "action": "mock-list", "rules": []}

    state._engine = _FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}/"
    try:
        with urllib.request.urlopen(root + "api/proxy?serial=emulator-5554", timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["ok"] is True and payload["supported"] is True

        with pytest.raises(urllib.error.HTTPError) as scoped:
            urllib.request.urlopen(root + "api/proxy?serial=emulator-9999", timeout=2)
        assert scoped.value.code in (400, 404)

        body = json.dumps({}).encode()
        unauthorized = urllib.request.Request(root + "api/proxy/list", data=body, method="POST")
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(unauthorized, timeout=2)
        assert denied.value.code == 403

        authorized = urllib.request.Request(
            root + "api/proxy/list",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-AUA-Dashboard-Token": state.database_token,
            },
        )
        with urllib.request.urlopen(authorized, timeout=2) as r:
            assert json.loads(r.read())["action"] == "mock-list"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_proxy_panel_is_in_the_detail_html() -> None:
    from android_ui_analyser import dashboard as dash

    html = dash._DASHBOARD_HTML
    assert 'id="px-flows"' in html
    assert 'id="px-rulelist"' in html
    assert "/api/proxy/" in html
    # The panel is detail-only: a grid tile has no serial to scope these calls to.
    assert "if (isGrid) return;" in html


def test_dashboard_stub_rules_keep_the_host_and_times_the_form_collected(
    tmp_path: Path,
) -> None:
    """The panel pre-fills the host from the clicked request and offers a Times field.

    `mock_map` could express neither, so both were silently dropped and a stub meant for
    one endpoint on one host was armed against every host, forever.
    """
    state = _dashboard_state(tmp_path)
    seen: dict[str, Any] = {}

    class _FakeEngine:
        def mock_map(self, method: str, path: str, **kw: Any) -> dict:
            seen.update({"method": method, "path": path, **kw})
            return {"ok": True}

    state._engine = _FakeEngine()
    state.proxy_operation(
        "stub",
        {"method": "GET", "path": "/v1/feed", "host": "api.example.test", "times": 1},
    )
    assert seen["host"] == "api.example.test"
    assert seen["times"] == 1


def test_proxy_flow_detail_disambiguates_a_reused_sequence_number(tmp_path: Path) -> None:
    """The addon's `n` restarts at 1 per mitmdump process, the log is append-only."""
    svc = _FakeProxyService(tmp_path)
    svc.bodies = [
        {"n": 1, "ts": 1000.0, "path": "/first-session", "response_body": "a"},
        {"n": 1, "ts": 9000.0, "path": "/second-session", "response_body": "b"},
    ]
    state = _proxy_state(tmp_path, svc)
    assert state.proxy_flow_detail(1, 1000.0)["flow"]["path"] == "/first-session"
    assert state.proxy_flow_detail(1, 9000.0)["flow"]["path"] == "/second-session"
    # With no timestamp the newest is still the sane default.
    assert state.proxy_flow_detail(1)["flow"]["path"] == "/second-session"


def test_a_tile_never_takes_the_uiautomation_slot_from_the_agent_using_the_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The frame fallback goes through platform.connect(), which attaches uiautomator2.

    Tiles now request a frame every poll, so without this the dashboard would grab the
    UiAutomation slot roughly once a second out from under whichever agent is driving.
    """
    from android_ui_analyser import leases

    state = _dashboard_state(tmp_path)

    def refuse(_ser: str) -> Any:
        raise AssertionError("the dashboard must not connect to a device another agent holds")

    monkeypatch.setattr(state.platform, "connect", refuse)
    monkeypatch.setattr(leases, "read_lease", lambda _cache, _ser: {"owner": "some-other-agent"})

    frames = tmp_path / "captures" / "emulator-5554" / "s1" / "frames"
    frames.mkdir(parents=True)
    (frames / "1.jpg").write_bytes(b"last-known")
    long_ago = time.time() - 300
    os.utime(frames / "1.jpg", (long_ago, long_ago))
    state.note_capture_live("emulator-5554", False)

    data, _mime = state.frame_bytes("emulator-5554")
    assert data == b"last-known"

    # Nobody holding it → the screencap fallback is allowed again.
    monkeypatch.setattr(leases, "read_lease", lambda _cache, _ser: None)
    live_png = b"\x89PNG\r\n\x1a\nfresh"

    class _Img:
        png_bytes = live_png

    class _Dev:
        def screenshot(self) -> _Img:
            return _Img()

    monkeypatch.setattr(state.platform, "connect", lambda _ser: _Dev())
    data, mime = state.frame_bytes("emulator-5554")
    assert data == live_png
    # The cached bytes must keep their own mime; a PNG served as JPEG is a broken tile.
    assert state.frame_bytes("emulator-5554") == (live_png, "image/png")
    assert mime == "image/png"


def test_the_flow_detail_view_never_hands_out_a_credential(tmp_path: Path) -> None:
    """The proxy captures whole exchanges, and this endpoint serves them over plain HTTP
    on localhost — into a page people screenshot into bug reports. A real run had it
    returning the app's bearer token, x-api-key, x-device-key and a streamToken."""

    svc = _FakeProxyService(tmp_path)
    svc.bodies = [
        {
            "n": 1,
            "ts": 1.0,
            "method": "POST",
            "path": "/v1/session",
            "query": "access_token=QUERYSECRET&page=2",
            "request_headers": {
                "Authorization": "Bearer SECRET-BEARER",
                "X-Api-Key": "SECRET-KEY",
                "x-device-key": "SECRET-DEVICE",
                "Content-Type": "application/json",
            },
            "response_headers": {"Set-Cookie": "sid=SECRET-COOKIE", "Server": "uvicorn"},
            "response_body": json.dumps(
                {
                    "streamToken": "SECRET-STREAM",
                    "user": {"name": "Ada", "refresh_token": "SECRET-REFRESH"},
                    "items": [{"password": "SECRET-PW"}],
                }
            ),
        }
    ]
    state = _proxy_state(tmp_path, svc)
    flow = state.proxy_flow_detail(1)["flow"]
    rendered = json.dumps(flow)

    for secret in (
        "SECRET-BEARER",
        "SECRET-KEY",
        "SECRET-DEVICE",
        "SECRET-COOKIE",
        "SECRET-STREAM",
        "SECRET-REFRESH",
        "SECRET-PW",
        "QUERYSECRET",
    ):
        assert secret not in rendered, f"{secret} leaked from the proxy panel"

    # Names and non-secret values survive — knowing an endpoint sends a bearer is the point.
    assert "Authorization" in flow["request_headers"]
    assert flow["request_headers"]["Content-Type"] == "application/json"
    assert flow["response_headers"]["Server"] == "uvicorn"
    assert json.loads(flow["response_body"])["user"]["name"] == "Ada"
    assert "page=2" in flow["query"]


def test_the_proxy_panel_reads_only_its_own_device(tmp_path: Path) -> None:
    svc = _FakeProxyService(tmp_path)
    state = _proxy_state(tmp_path, svc)
    state.proxy_payload("emulator-5554")
    state.proxy_flow_detail(1, None, "emulator-5554")
    assert svc.flow_reads == ["emulator-5554"]
    assert svc.body_reads == ["emulator-5554"]
