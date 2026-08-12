"""Idle lifecycle: capture idle-pause, daemon idle-TTL, and orphan reaping."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser import daemon as daemon_mod
from android_ui_analyser.capture import CaptureBuffer, CaptureCfgView


def _engine_stub(*, idle_pause_s: int, idle_ttl_s: int) -> SimpleNamespace:
    paused: list[bool] = []

    def capture_idle_pause() -> bool:
        paused.append(True)
        return True

    return SimpleNamespace(
        config=SimpleNamespace(
            capture=SimpleNamespace(idle_pause_s=idle_pause_s),
            daemon=SimpleNamespace(idle_ttl_s=idle_ttl_s),
        ),
        capture_idle_pause=capture_idle_pause,
        paused_calls=paused,
    )


class _Clock:
    """An _Activity whose idle time we control."""

    def __init__(self, idle: float) -> None:
        self._idle = idle

    def idle_s(self) -> float:
        return self._idle


def test_idle_tick_pauses_capture_but_keeps_daemon_alive() -> None:
    engine = _engine_stub(idle_pause_s=10, idle_ttl_s=100)
    assert daemon_mod._idle_tick(engine, _Clock(20.0)) is False
    assert engine.paused_calls == [True]


def test_idle_tick_requests_shutdown_past_ttl() -> None:
    engine = _engine_stub(idle_pause_s=10, idle_ttl_s=60)
    assert daemon_mod._idle_tick(engine, _Clock(75.0)) is True


def test_idle_tick_is_inert_while_active() -> None:
    engine = _engine_stub(idle_pause_s=10, idle_ttl_s=60)
    assert daemon_mod._idle_tick(engine, _Clock(3.0)) is False
    assert engine.paused_calls == []


def test_zero_disables_both_policies() -> None:
    engine = _engine_stub(idle_pause_s=0, idle_ttl_s=0)
    assert daemon_mod._idle_tick(engine, _Clock(10_000.0)) is False
    assert engine.paused_calls == []


def test_manual_pause_survives_idle_resume() -> None:
    buf = CaptureBuffer(
        root=Path("/tmp"), serial="x", cfg=CaptureCfgView(), screenshot=lambda: None
    )
    buf.pause()  # user ran `aua capture off`
    buf.resume(only_if_idle=True)  # agent activity must not resurrect it
    assert buf.paused is True
    buf.resume()  # explicit `aua capture on`
    assert buf.paused is False


def test_idle_pause_is_resumed_by_activity() -> None:
    buf = CaptureBuffer(
        root=Path("/tmp"), serial="x", cfg=CaptureCfgView(), screenshot=lambda: None
    )
    buf.pause("idle")
    buf.resume(only_if_idle=True)
    assert buf.paused is False


def _spawn_with_own_copy_of(binary: str, tmp_path: Path) -> subprocess.Popen[bytes]:
    copy = tmp_path / "ghost-exe"
    shutil.copy(binary, copy)
    proc = subprocess.Popen([str(copy), "300"], start_new_session=True)
    time.sleep(0.3)
    return proc


@pytest.mark.skipif(not Path("/bin/sleep").exists(), reason="needs /bin/sleep")
def test_reap_terminates_a_live_daemon_whose_interpreter_was_deleted(tmp_path: Path) -> None:
    """The scratchpad-venv leak: process still alive, its interpreter long gone."""
    cache = tmp_path / "cache"
    cache.mkdir()
    ghost_exe = tmp_path / "ghost-exe"
    proc = _spawn_with_own_copy_of("/bin/sleep", tmp_path)
    try:
        assert daemon_mod._pid_alive(proc.pid) is True
        (cache / "daemon.sock.ghost.pid").write_text(
            json.dumps({"pid": proc.pid, "exe": str(ghost_exe)})
        )
        (cache / "daemon.sock.ghost").touch()
        config = SimpleNamespace(cache=SimpleNamespace(dir=str(cache)))

        assert daemon_mod.reap(config)["count"] == 0  # interpreter still on disk

        ghost_exe.unlink()
        out = daemon_mod.reap(config)

        assert out["count"] == 1
        assert out["reaped"][0]["reason"] == "orphaned_venv"
        assert not (cache / "daemon.sock.ghost.pid").exists()
        assert not (cache / "daemon.sock.ghost").exists()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(not Path("/bin/sleep").exists(), reason="needs /bin/sleep")
def test_reap_spares_a_healthy_daemon(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    proc = subprocess.Popen(["/bin/sleep", "300"], start_new_session=True)
    time.sleep(0.3)
    try:
        (cache / "daemon.sock.live.pid").write_text(
            json.dumps({"pid": proc.pid, "exe": "/bin/sleep"})
        )
        (cache / "daemon.sock.live").touch()
        config = SimpleNamespace(cache=SimpleNamespace(dir=str(cache)))

        out = daemon_mod.reap(config)

        assert out["count"] == 0
        assert proc.poll() is None
        assert (cache / "daemon.sock.live.pid").exists()
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_reap_clears_a_stale_pidfile(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    proc = subprocess.Popen(["/bin/sleep", "0.05"])
    proc.wait(timeout=5)
    dead_pid = proc.pid
    (cache / "daemon.sock.gone.pid").write_text(str(dead_pid))
    (cache / "daemon.sock.gone").touch()

    out = daemon_mod.reap(SimpleNamespace(cache=SimpleNamespace(dir=str(cache))))

    assert out["reaped"][0]["reason"] == "dead"
    assert not (cache / "daemon.sock.gone").exists()


def test_legacy_pidfile_never_reaps_a_live_process(tmp_path: Path) -> None:
    """A bare-int pidfile records no interpreter, so we cannot prove it is an orphan."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "daemon.sock.legacy.pid").write_text(str(os.getpid()))
    (cache / "daemon.sock.legacy").touch()

    out = daemon_mod.reap(SimpleNamespace(cache=SimpleNamespace(dir=str(cache))))

    assert out["count"] == 0
    assert (cache / "daemon.sock.legacy").exists()


def test_pidfile_roundtrip_and_legacy_parse(tmp_path: Path) -> None:
    path = tmp_path / "d.sock.pid"
    daemon_mod.write_pidfile(path)
    pid, exe = daemon_mod.read_pidfile(path)
    assert pid == os.getpid()
    assert exe == sys.executable

    path.write_text("4242")
    assert daemon_mod.read_pidfile(path) == (4242, None)

    path.write_text("not-a-pid")
    assert daemon_mod.read_pidfile(path) == (None, None)


def test_busy_live_pid_counts_as_running_without_spawning_a_competitor(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg = SimpleNamespace(
        cache=SimpleNamespace(dir=str(cache)),
        daemon=SimpleNamespace(socket=str(cache / "daemon.sock")),
        device=SimpleNamespace(serial="fictional-5554"),
    )
    sock = daemon_mod.socket_path(cfg)
    Path(sock).touch()
    Path(sock + ".pid").write_text(
        json.dumps({"pid": os.getpid(), "exe": sys.executable}), encoding="utf-8"
    )
    monkeypatch.setattr(
        daemon_mod.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    monkeypatch.setattr(daemon_mod, "_socket_alive", lambda _sock: False)

    assert daemon_mod.is_running(cfg) is True
    started = daemon_mod.start(cfg)

    assert started["status"] == "already_running"
    assert started["pid"] == os.getpid()


def test_superseded_daemon_cleanup_preserves_successor_socket_and_pidfile(tmp_path: Path) -> None:
    sock = str(tmp_path / "daemon.sock")
    Path(sock).touch()
    Path(sock + ".pid").write_text(
        json.dumps({"pid": 222, "exe": "/fictional/python"}), encoding="utf-8"
    )

    removed = daemon_mod._remove_owned_daemon_files(sock, owner_pid=111)

    assert removed is False
    assert Path(sock).exists()
    assert daemon_mod.read_pidfile(sock + ".pid")[0] == 222


def test_current_daemon_cleanup_removes_its_own_socket_and_pidfile(tmp_path: Path) -> None:
    sock = str(tmp_path / "daemon.sock")
    Path(sock).touch()
    Path(sock + ".pid").write_text(
        json.dumps({"pid": 111, "exe": "/fictional/python"}), encoding="utf-8"
    )

    removed = daemon_mod._remove_owned_daemon_files(sock, owner_pid=111)

    assert removed is True
    assert not Path(sock).exists()
    assert not Path(sock + ".pid").exists()
