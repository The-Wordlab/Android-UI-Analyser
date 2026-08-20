"""Daemon tests — unix-socket dispatch, DaemonClient, serve lifecycle.

Uses a background thread (no real device, no subprocess) with FakeDevice so the
tests run fast and without a phone.
"""

from __future__ import annotations

# FakeDevice and helpers are defined in conftest.py, which pytest automatically
# injects into the test module's namespace as fixtures.  For plain-function use
# (non-fixture), we import them directly from the file to avoid hitting a stale
# installed copy in the venv.
import importlib.util as _ilu
import sys as _sys
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from android_ui_analyser.daemon import DaemonClient, dispatch, serve, socket_path

_conftest_path = str(Path(__file__).parent / "conftest.py")
if "conftest" not in _sys.modules:
    _spec = _ilu.spec_from_file_location("conftest", _conftest_path)
    _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    _sys.modules["conftest"] = _mod
else:
    _mod = _sys.modules["conftest"]

FakeDevice = _mod.FakeDevice
make_config = _mod.make_config
make_engine = _mod.make_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sock_path() -> Generator[str, None, None]:
    # AF_UNIX paths on macOS are limited to ~104 chars; use a system tempdir with
    # a short name to stay well under the limit.
    with tempfile.TemporaryDirectory(prefix="aua_") as td:
        yield str(Path(td) / "d.sock")


@pytest.fixture()
def settings_engine(sock_path: str) -> Generator[tuple[object, str], None, None]:
    """Start a serve() thread, yield (engine, sock_path), then shut it down."""
    device = FakeDevice(text_index={"Settings": (10, 20, 100, 60)})
    engine = make_engine(device=device)

    ready = threading.Event()
    stop = threading.Event()

    thread = threading.Thread(
        target=serve,
        args=(engine, sock_path),
        kwargs={"ready_event": ready, "_stop_event": stop},
        daemon=True,
    )
    thread.start()

    assert ready.wait(timeout=3.0), "daemon did not become ready in time"

    yield engine, sock_path

    # Tear down: signal the accept loop and wait for thread exit.
    stop.set()
    thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# dispatch() unit tests (no socket needed)
# ---------------------------------------------------------------------------


def test_dispatch_ping() -> None:
    device = FakeDevice()
    engine = make_engine(device=device)
    resp = dispatch(engine, {"cmd": "ping", "args": {}})
    from android_ui_analyser import __version__, daemon

    assert resp["ok"] is True
    # The reported identity is version + loaded-source fingerprint, so an edited file makes
    # a warm daemon detectably stale (a bare version never differs during development).
    assert resp["result"] == {
        "pong": True,
        "version": daemon._aua_version(),
        "policy_fingerprint": daemon.policy_config_fingerprint(engine.config),
    }
    assert str(resp["result"]["version"]).startswith(f"{__version__}+src")


def test_dispatch_unknown_command() -> None:
    device = FakeDevice()
    engine = make_engine(device=device)
    resp = dispatch(engine, {"cmd": "unknown_xyz", "args": {}})
    assert resp["ok"] is False
    assert "unknown_command" in resp["error"]["code"]


def test_dispatch_analyze_returns_schema_keys() -> None:
    device = FakeDevice()
    engine = make_engine(device=device)
    resp = dispatch(engine, {"cmd": "analyze", "args": {}})
    assert resp["ok"] is True
    result = resp["result"]
    assert "schema_version" in result
    assert "screen" in result
    assert "elements" in result
    assert "meta" in result


def test_dispatch_holds_device_fence_until_the_command_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    monkeypatch.setattr(leases, "_proc_started", lambda pid: "source" if pid == 111 else "")
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    source = leases.LeaseOwner("orchestrator", pid=111, started="source")
    serial = "emulator-5554"
    config = make_config(cache={"dir": str(tmp_path)})
    engine = make_engine(config=config, device=FakeDevice(serial=serial))
    assert leases.acquire(tmp_path, serial, owner=source)
    engine._lease_owner = source
    engine._lease_owner_resolved = source
    engine._lease_serial = serial
    engine._leased_serial_resolved = (True, serial)
    command_started = threading.Event()
    transfer_started = threading.Event()
    let_command_finish = threading.Event()

    class Result:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"ok": True}

    def analyze(**_kwargs: object) -> Result:
        command_started.set()
        assert let_command_finish.wait(timeout=2)
        return Result()

    monkeypatch.setattr(engine, "analyze", analyze)
    response: list[dict[str, object]] = []
    offered: list[dict[str, object]] = []

    command = threading.Thread(
        target=lambda: response.append(dispatch(engine, {"cmd": "analyze", "args": {}}))
    )

    def transfer() -> None:
        transfer_started.set()
        offered.append(leases.create_handoff(tmp_path, serial, owner=source))

    handoff = threading.Thread(target=transfer)
    command.start()
    assert command_started.wait(timeout=2)
    handoff.start()
    assert transfer_started.wait(timeout=2)
    assert not offered
    let_command_finish.set()
    command.join(timeout=2)
    handoff.join(timeout=2)

    assert response == [{"ok": True, "result": {"ok": True}}]
    assert offered and offered[0]["serial"] == serial


def test_daemon_journaling_does_not_reacquire_the_finished_command_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.daemon import _journal_dispatch

    monkeypatch.setattr(leases, "_proc_started", lambda pid: "source" if pid == 111 else "")
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    source = leases.LeaseOwner("orchestrator", pid=111, started="source")
    serial = "emulator-5554"
    engine = make_engine(
        config=make_config(cache={"dir": str(tmp_path)}),
        device=FakeDevice(serial=serial),
    )
    assert leases.acquire(tmp_path, serial, owner=source)
    engine._lease_owner = source
    engine._lease_owner_resolved = source
    engine._lease_serial = serial
    engine._leased_serial_resolved = (True, serial)

    class Result:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"ok": True}

    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: Result())
    request = {"cmd": "analyze", "args": {}, "owner": str(source)}
    response = dispatch(engine, request)
    _journal_dispatch(engine, request, response, duration_ms=1.0)

    offered: list[dict[str, object]] = []
    transfer = threading.Thread(
        target=lambda: offered.append(leases.create_handoff(tmp_path, serial, owner=source))
    )
    transfer.start()
    transfer.join(timeout=2)

    assert offered and offered[0]["serial"] == serial


def test_push_watcher_uses_background_fence_without_blocking_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.push import PushHub

    monkeypatch.setattr(leases, "_proc_started", lambda pid: "source" if pid == 111 else "")
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    source = leases.LeaseOwner("orchestrator", pid=111, started="source")
    serial = "emulator-5554"
    engine = make_engine(
        config=make_config(cache={"dir": str(tmp_path)}),
        device=FakeDevice(serial=serial),
    )
    assert leases.acquire(tmp_path, serial, owner=source)
    engine._lease_owner = source
    engine._lease_owner_resolved = source
    engine._lease_serial = serial
    engine._leased_serial_resolved = (True, serial)
    sampled = threading.Event()

    def dump_tree(_device: object, *, compact: bool) -> str:
        assert isinstance(compact, bool)
        sampled.set()
        return "<hierarchy/>"

    monkeypatch.setattr(engine.platform, "dump_tree", dump_tree)
    hub = PushHub()
    hub.start_watcher(
        lambda: engine.hierarchy_fingerprint(background=True),
        interval_ms=10,
        serial=serial,
    )
    try:
        assert sampled.wait(timeout=2)
        offered: list[dict[str, object]] = []
        transfer = threading.Thread(
            target=lambda: offered.append(
                leases.create_handoff(tmp_path, serial, owner=source)
            )
        )
        transfer.start()
        transfer.join(timeout=2)
        assert offered and offered[0]["serial"] == serial
    finally:
        hub.stop()
        if hub._watch_thread is not None:
            hub._watch_thread.join(timeout=2)


def test_dispatch_host_only_command_adopts_identity_without_claiming_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.engine import Engine

    owner = leases.LeaseOwner("agent-a", pid=111, started="source")
    monkeypatch.setattr(leases, "bind_owner_caller", lambda _owner, _caller: owner)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}))

    def unexpected_device_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host-only daemon command touched a device")

    monkeypatch.setattr(engine, "_lease_device", unexpected_device_access)
    monkeypatch.setattr(engine, "_connect_target", unexpected_device_access)
    monkeypatch.setattr(
        engine,
        "capture_status",
        lambda: {"ok": True, "action": "capture-status", "running": False},
    )

    response = dispatch(
        engine,
        {"cmd": "capture_status", "args": {}, "owner": "agent-a", "caller": {}},
    )

    assert response["ok"] is True
    assert engine._lease_owner_resolved == owner
    assert engine._device is None
    assert not list((tmp_path / "leases").glob("*.json"))


def test_owner_adoption_waits_for_old_async_memory_writer_before_clearing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.daemon import _adopt_client_owner
    from android_ui_analyser.engine import Engine

    source = leases.LeaseOwner("source", pid=111, started="source")
    child = leases.LeaseOwner("child", pid=222, started="child")
    monkeypatch.setattr(leases, "bind_owner_caller", lambda _owner, _caller: child)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}))
    engine._lease_owner = source
    engine._lease_owner_resolved = source
    writer_started = threading.Event()
    release_writer = threading.Event()
    join_started = threading.Event()

    def old_writer() -> None:
        writer_started.set()
        assert release_writer.wait(timeout=2)
        engine._last_known_screen = "old-owner-screen"
        engine._last_mem_fp = "old-owner-fingerprint"

    writer = threading.Thread(target=old_writer)
    with engine._mem_threads_lock:
        engine._mem_thread = writer
        engine._mem_threads.append(writer)
        writer.start()
    assert writer_started.wait(timeout=2)
    original_join = engine._join_memory_writers

    def observed_join(*, timeout_s: float) -> bool:
        join_started.set()
        return original_join(timeout_s=timeout_s)

    monkeypatch.setattr(engine, "_join_memory_writers", observed_join)
    adoption_done = threading.Event()

    def adopt() -> None:
        _adopt_client_owner(engine, "child", {}, claim_device=False)
        adoption_done.set()

    adoption = threading.Thread(target=adopt)
    adoption.start()
    assert join_started.wait(timeout=2)
    assert not adoption_done.is_set()
    release_writer.set()
    adoption.join(timeout=2)

    assert adoption_done.is_set()
    assert engine._lease_owner_resolved == child
    assert engine._last_known_screen is None
    assert engine._last_mem_fp is None


def test_returned_lease_generation_clears_stale_state_for_the_same_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    starts = {111: "source", 222: "child"}
    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    source = leases.LeaseOwner("orchestrator", pid=111, started="source")
    child = leases.LeaseOwner("child", pid=222, started="child")
    serial = "emulator-5554"
    device = FakeDevice(serial=serial)
    engine = make_engine(
        config=make_config(cache={"dir": str(tmp_path)}),
        device=device,
    )
    assert leases.acquire(tmp_path, serial, owner=source)
    engine._lease_owner = source
    engine._lease_owner_resolved = source
    engine._lease_serial = serial
    engine._leased_serial_resolved = (True, serial)
    engine.begin_device_use()
    original_generation = engine._lease_generation_resolved
    engine.release_device_use()
    assert original_generation is not None
    engine._last_known_screen = "stale-screen"
    engine._last_mem_fp = "stale-fingerprint"

    to_child = leases.create_handoff(tmp_path, serial, owner=source)
    leases.accept_handoff(tmp_path, to_child["token"], owner=child)
    to_source = leases.create_handoff(tmp_path, serial, owner=child)
    leases.accept_handoff(tmp_path, to_source["token"], owner=source)

    engine.begin_device_use()
    try:
        assert engine._lease_generation_resolved != original_generation
        assert engine._last_known_screen is None
        assert engine._last_mem_fp is None
        assert engine._device is device
    finally:
        engine.release_device_use()


def test_dispatch_ask_screen(monkeypatch) -> None:
    engine = make_engine(device=FakeDevice())
    monkeypatch.setattr(
        engine,
        "ask_screen",
        lambda question: {"question": question, "analysis": {"answer": "top-right"}},
    )
    resp = dispatch(engine, {"cmd": "ask_screen", "args": {"question": "Where?"}})
    assert resp == {
        "ok": True,
        "result": {"question": "Where?", "analysis": {"answer": "top-right"}},
    }


def test_dispatch_session_autopilot_uses_the_warm_engine(monkeypatch) -> None:
    engine = make_engine(device=FakeDevice())
    calls: list[dict[str, object]] = []

    def autopilot(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"ok": True, "autopilot": {"steps_executed": 2}}

    monkeypatch.setattr(engine, "session_autopilot", autopilot, raising=False)

    resp = dispatch(
        engine,
        {
            "cmd": "session_autopilot",
            "args": {"session_id": "session-1", "max_steps": 4, "max_duration_ms": 12000},
        },
    )

    assert resp == {"ok": True, "result": {"ok": True, "autopilot": {"steps_executed": 2}}}
    assert calls == [{"session_id": "session-1", "max_steps": 4, "max_duration_ms": 12000}]


def test_dispatch_has_found() -> None:
    device = FakeDevice(text_index={"Settings": (10, 20, 100, 60)})
    engine = make_engine(device=device)
    resp = dispatch(engine, {"cmd": "has", "args": {"text": "Settings"}})
    assert resp["ok"] is True
    assert resp["result"]["found"] is True


def test_dispatch_has_not_found() -> None:
    device = FakeDevice(text_index={})
    engine = make_engine(device=device)
    resp = dispatch(engine, {"cmd": "has", "args": {"text": "NonExistent"}})
    assert resp["ok"] is True
    assert resp["result"]["found"] is False


def test_dispatch_list_devices_returns_dict() -> None:
    """list_devices goes to the real adb; we just assert a dict with 'ok' comes back."""
    device = FakeDevice()
    engine = make_engine(device=device)
    resp = dispatch(engine, {"cmd": "list_devices", "args": {}})
    assert "ok" in resp


# ---------------------------------------------------------------------------
# Over-socket integration tests (DaemonClient <-> serve thread)
# ---------------------------------------------------------------------------


def test_ping_true(settings_engine: tuple) -> None:
    _, sock = settings_engine
    with DaemonClient(sock) as client:
        assert client.ping() is True


def test_analyze_over_socket(settings_engine: tuple) -> None:
    _, sock = settings_engine
    with DaemonClient(sock) as client:
        resp = client.call("analyze")
    assert resp["ok"] is True
    result = resp["result"]
    assert "schema_version" in result
    assert "screen" in result
    assert "elements" in result
    assert "meta" in result


def test_has_settings_found_over_socket(settings_engine: tuple) -> None:
    _, sock = settings_engine
    with DaemonClient(sock) as client:
        resp = client.call("has", text="Settings")
    assert resp["ok"] is True
    assert resp["result"]["found"] is True


def test_list_devices_over_socket(settings_engine: tuple) -> None:
    """list_devices may succeed or fail (no real adb needed), but must return a dict."""
    _, sock = settings_engine
    with DaemonClient(sock) as client:
        resp = client.call("list_devices")
    assert isinstance(resp, dict)
    assert "ok" in resp


def test_unknown_cmd_over_socket(settings_engine: tuple) -> None:
    _, sock = settings_engine
    with DaemonClient(sock) as client:
        resp = client.call("does_not_exist")
    assert resp["ok"] is False
    assert resp["error"]["code"] == "unknown_command"


def test_multiple_sequential_connections(settings_engine: tuple) -> None:
    """The server handles multiple independent connections sequentially."""
    _, sock = settings_engine
    for _ in range(3):
        with DaemonClient(sock) as client:
            assert client.ping() is True


def test_socket_file_removed_after_shutdown(sock_path: str) -> None:
    """After serve() exits, the socket file is cleaned up."""
    device = FakeDevice()
    engine = make_engine(device=device)

    ready = threading.Event()
    stop = threading.Event()

    thread = threading.Thread(
        target=serve,
        args=(engine, sock_path),
        kwargs={"ready_event": ready, "_stop_event": stop},
        daemon=True,
    )
    thread.start()
    assert ready.wait(timeout=3.0), "daemon did not become ready"

    # Confirm it's up.
    with DaemonClient(sock_path) as client:
        assert client.ping() is True

    # Signal shutdown.
    stop.set()
    thread.join(timeout=3.0)

    assert not Path(sock_path).exists(), "socket file must be removed after shutdown"


# ---------------------------------------------------------------------------
# socket_path helper
# ---------------------------------------------------------------------------


def test_socket_path_expands_tilde() -> None:
    cfg = make_config()
    result = socket_path(cfg)
    assert "~" not in result
    assert result.startswith("/")


def test_pong_version_reports_and_back_compat() -> None:
    """The client extracts the daemon version; an old 'pong' string → None (trusted)."""
    from unittest.mock import patch

    from android_ui_analyser import __version__
    from android_ui_analyser.daemon import DaemonClient

    client = DaemonClient("/tmp/x.sock")
    with patch.object(
        client, "call", return_value={"ok": True, "result": {"pong": True, "version": __version__}}
    ):
        assert client.pong_version() == __version__
        assert client.ping() is True
    with patch.object(client, "call", return_value={"ok": True, "result": "pong"}):
        assert client.pong_version() is None  # pre-version daemon
        assert client.ping() is True
    with patch.object(client, "call", side_effect=OSError):
        assert client.pong_version() is False
        assert client.ping() is False
