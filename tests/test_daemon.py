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
    assert resp == {"ok": True, "result": "pong"}


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
