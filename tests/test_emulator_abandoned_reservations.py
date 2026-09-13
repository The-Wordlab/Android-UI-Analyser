"""A dead starter must not block a fresh lane cache (#12)."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from android_ui_analyser import emulator as em
from android_ui_analyser.errors import DeviceError


def test_dead_starter_does_not_block_a_fresh_cache(tmp_path, monkeypatch):
    reservations = tmp_path / "shared-portlocks"
    reservations.mkdir()
    monkeypatch.setattr(em, "_reservation_dir", lambda: reservations)
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    code = (
        "import pathlib, sys, time; from android_ui_analyser import emulator as em; "
        "em._reservation_dir = lambda: pathlib.Path(sys.argv[1]); "
        "assert em._claim_console_port(5556); print('reserved', flush=True); time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(reservations)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "reserved"
        # A live starter must remain protected, even though no socket is bound yet.
        with pytest.raises(DeviceError, match="already in use"):
            em.allocate_console_port(5556, cache_dir=tmp_path / "other-lane")
        child.kill()
        child.wait(timeout=5)
        assert em.allocate_console_port(5556, cache_dir=tmp_path / "fresh-lane") == 5556
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)



def test_live_reservation_stays_reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    assert em.allocate_console_port(5556, cache_dir=tmp_path) == 5556
    assert int((em._reservation_dir() / "5556.port").read_text()) == os.getpid()
    with pytest.raises(DeviceError, match="already in use"):
        em.allocate_console_port(5556, cache_dir=tmp_path)


def test_parallel_reclaim_does_not_delete_a_new_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    for port in range(5554, 5570, 2):
        (em._reservation_dir() / f"{port}.port").write_text("2147483647\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        ports = list(pool.map(lambda i: em.allocate_console_port(cache_dir=tmp_path / str(i)), range(8)))
    assert len(set(ports)) == 8
    assert set(ports) == set(range(5554, 5570, 2))


def test_spawned_child_holds_reservation_until_failed_start_cleanup(tmp_path, monkeypatch):
    launcher = tmp_path / "fake-emulator"
    launcher.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    launcher.chmod(0o755)
    monkeypatch.setattr(em, "emulator_bin", lambda: str(launcher))
    monkeypatch.setattr(em, "list_avds", lambda: {"avds": ["example"]})
    monkeypatch.setattr(em, "running_emulators", lambda: [])

    def during_boot(*_args, **_kwargs):
        pid = int((em._reservation_dir() / "5556.port").read_text())
        assert pid > 1 and pid != os.getpid()
        with pytest.raises(DeviceError, match="already in use"):
            em.allocate_console_port(5556, cache_dir=tmp_path / "fresh-lane")
        raise DeviceError("controlled boot failure")

    monkeypatch.setattr(em, "_wait_for_serial", during_boot)
    with pytest.raises(DeviceError, match="controlled boot failure"):
        em.start("example", port=5556, cache_dir=tmp_path, idle_timeout_s=0, animations=True)
    assert not (em._reservation_dir() / "5556.port").exists()


@pytest.mark.parametrize("contents", ["", "unreadable-owner", "0", "-1"])
def test_unknown_reservation_owner_keeps_bounded_protection(monkeypatch, contents):
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    (em._reservation_dir() / "5556.port").write_text(contents)
    with pytest.raises(DeviceError, match="already in use"):
        em.allocate_console_port(5556)
