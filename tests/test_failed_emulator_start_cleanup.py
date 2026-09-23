"""Failed boots must not leave an untracked emulator behind or signal a reused PID."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

import pytest

from android_ui_analyser import emulator, leases


@pytest.fixture
def failed_start(tmp_path, monkeypatch):
    record = tmp_path / "example.p5998.json"
    meta = {
        "avd": "example",
        "instance": "example.p5998",
        "instance_token": "example-boot",
        "started_by_aua": True,
        "pid": 4242,
        "process_started": "original-start",
        "port": 5998,
    }
    record.write_text(json.dumps(meta))
    reservations = tmp_path / "portlocks"
    reservations.mkdir()
    reservation = reservations / "5998.port"
    reservation.write_text("test-owner\n")
    state = {"alive": True, "started": "original-start"}
    signals = []
    watchdogs = []

    def probe(pid, sig):
        assert pid == 4242 and sig == 0
        if not state["alive"]:
            raise ProcessLookupError()

    monkeypatch.setattr(emulator.os, "kill", probe)
    monkeypatch.setattr(emulator.os, "waitpid", lambda *_args: (0, 0))
    monkeypatch.setattr(emulator, "_signal_emulator", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(leases, "_proc_started", lambda pid: state["started"])
    monkeypatch.setattr(emulator, "_OWNED_STOP_TIMEOUT_S", 0)
    monkeypatch.setattr(emulator, "_reservation_dir", lambda: reservations)
    monkeypatch.setattr(emulator, "_kill_watchdog", watchdogs.append)
    monkeypatch.setattr(
        emulator, "running_emulators", lambda: pytest.fail("cleanup must not consult shared ADB")
    )
    monkeypatch.setattr(
        emulator, "_adb_emu_kill", lambda _serial: pytest.fail("cleanup must not stop by serial")
    )
    return meta, record, reservation, state, signals, watchdogs


def rollback(meta, record):
    return emulator._rollback_failed_start(
        pid=4242, meta=meta, meta_path=record, console_port=5998
    )


@pytest.mark.parametrize("exit_signal", [signal.SIGTERM, signal.SIGKILL])
def test_failed_start_waits_for_exit_and_escalates_only_when_needed(
    failed_start, monkeypatch, exit_signal
):
    meta, record, reservation, state, signals, watchdogs = failed_start

    def signal_group(pid, sig):
        signals.append((pid, sig))
        # The ownership/port claim must still exist at the actual moment of exit.
        assert record.is_file() and reservation.is_file()
        if sig == exit_signal:
            state["alive"] = False

    monkeypatch.setattr(emulator, "_signal_emulator", signal_group)

    assert rollback(meta, record) is True

    assert signals == [(4242, signal.SIGTERM)] + (
        [(4242, signal.SIGKILL)] if exit_signal == signal.SIGKILL else []
    )
    assert not record.exists() and not reservation.exists()
    assert watchdogs == [meta]


@pytest.mark.parametrize("denied", [False, True])
def test_failed_start_keeps_bookkeeping_when_exit_cannot_be_confirmed(
    failed_start, monkeypatch, caplog, denied
):
    meta, record, reservation, _state, signals, watchdogs = failed_start

    if denied:
        def deny_signal(_pid, _sig):
            raise PermissionError("synthetic signal denial")

        monkeypatch.setattr(emulator, "_signal_emulator", deny_signal)

    assert rollback(meta, record) is False

    assert json.loads(record.read_text()) == meta
    assert reservation.exists()
    assert watchdogs == []
    assert signals == ([] if denied else [(4242, signal.SIGTERM), (4242, signal.SIGKILL)])
    assert "could not confirm emulator exit" in caplog.text


@pytest.mark.parametrize("reused_after_term", [False, True])
def test_failed_start_never_signals_a_reused_pid(failed_start, monkeypatch, reused_after_term):
    meta, record, reservation, state, signals, _watchdogs = failed_start
    if not reused_after_term:
        state["started"] = "replacement-start"
    else:
        def signal_group(pid, sig):
            signals.append((pid, sig))
            state["started"] = "replacement-start"

        monkeypatch.setattr(emulator, "_signal_emulator", signal_group)

    assert rollback(meta, record) is True

    assert signals == ([(4242, signal.SIGTERM)] if reused_after_term else [])
    assert not record.exists() and not reservation.exists()


@pytest.mark.parametrize("missing", ["recorded", "current"])
def test_failed_start_preserves_unknown_live_process_identity(failed_start, missing):
    meta, record, reservation, state, signals, _watchdogs = failed_start
    if missing == "recorded":
        meta["process_started"] = ""
    else:
        state["started"] = ""

    assert rollback(meta, record) is False

    assert signals == []
    assert record.exists() and reservation.exists()


def test_failed_start_cleans_an_already_exited_child(failed_start, monkeypatch):
    meta, record, reservation, state, signals, _watchdogs = failed_start
    state["started"] = ""
    # A zombie still responds to kill(pid, 0); waitpid is proof it actually exited.
    monkeypatch.setattr(emulator.os, "waitpid", lambda pid, _flags: (pid, 0))

    assert rollback(meta, record) is True

    assert signals == []
    assert not record.exists() and not reservation.exists()


def test_failed_start_rechecks_identity_before_force_killing(failed_start, monkeypatch):
    meta, record, reservation, _state, signals, _watchdogs = failed_start
    identities = iter(["original-start", "replacement-start"])
    monkeypatch.setattr(leases, "_proc_started", lambda _pid: next(identities))
    # Model replacement immediately after the TERM wait timed out.
    monkeypatch.setattr(emulator, "_wait_owned_process_exit", lambda *_args: False)

    assert rollback(meta, record) is True

    assert signals == [(4242, signal.SIGTERM)]
    assert not record.exists() and not reservation.exists()


def test_failed_start_reaps_a_real_child_that_ignores_term(tmp_path, monkeypatch):
    ready = tmp_path / "ready"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)",
            str(ready),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "synthetic child did not install its TERM handler"
        record = tmp_path / "example.json"
        meta = {
            "avd": "example",
            "instance": "example",
            "started_by_aua": True,
            "pid": proc.pid,
            "process_started": leases._proc_started(proc.pid),
        }
        assert meta["process_started"]
        record.write_text(json.dumps(meta))
        released = []
        monkeypatch.setattr(emulator, "release_console_port", released.append)
        monkeypatch.setattr(emulator, "_OWNED_STOP_TIMEOUT_S", 0.1)

        assert emulator._rollback_failed_start(
            pid=proc.pid, meta=meta, meta_path=record, console_port=5998
        ) is True

        with pytest.raises(ProcessLookupError):
            emulator.os.kill(proc.pid, 0)
        assert not record.exists()
        assert released == [5998]
    finally:
        # Popen remembers/reaps the exact child and never signals a reused PID.
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
