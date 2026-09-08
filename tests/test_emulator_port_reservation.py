"""Concurrent port allocation must never hand the same console port to two callers.

The bug this guards: `allocate_console_port` read the set of used ports and returned one
without holding it. adb cannot see an emulator that is still booting, and parallel agents
are told to keep separate caches, so two simultaneous callers read an identical "used" set
and both picked 5554. The loser bound no console port at all - it stayed alive but invisible
to adb, while its serial actually belonged to the winner's AVD. Two agents then drove one
device, each believing it had its own.
"""

from __future__ import annotations

import concurrent.futures as cf

import pytest

from android_ui_analyser import emulator as em
from android_ui_analyser.errors import DeviceError


@pytest.fixture(autouse=True)
def _clean_reservations(monkeypatch, tmp_path):
    """Point reservations at a temp dir and report no emulators as running."""
    resdir = tmp_path / "portlocks"
    resdir.mkdir()
    monkeypatch.setattr(em, "_reservation_dir", lambda: resdir)
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    monkeypatch.setattr(em, "_aua_started_records", lambda _cache: [])
    return resdir


def test_concurrent_callers_get_distinct_ports(tmp_path):
    """Eight simultaneous callers, each with its own cache, must not collide."""
    n = 8

    def pick(i: int) -> int:
        return em.allocate_console_port(None, cache_dir=str(tmp_path / f"cache-{i}"))

    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        ports = list(pool.map(pick, range(n)))

    assert len(set(ports)) == n, f"duplicate ports handed out: {sorted(ports)}"
    assert all(p % 2 == 0 for p in ports), "console ports must be even"


def test_reservation_is_released_and_port_reusable(_clean_reservations):
    port = em.allocate_console_port(None)
    assert (_clean_reservations / f"{port}.port").exists()

    # While reserved, the next caller must move on rather than reuse it.
    assert em.allocate_console_port(None) != port

    em.release_console_port(port)
    assert not (_clean_reservations / f"{port}.port").exists()
    assert em.allocate_console_port(None) == port  # freed, so offered again


def test_stale_reservation_does_not_block_the_range(_clean_reservations, monkeypatch):
    """A reservation only covers the boot gap; an abandoned one must not leak a port."""
    port = em.allocate_console_port(None)
    stale = _clean_reservations / f"{port}.port"
    assert stale.exists()

    monkeypatch.setattr(em, "_RESERVATION_TTL_S", -1)  # everything is now stale
    assert em.allocate_console_port(None) == port
    assert not any(f.stem == str(port) and f.suffix == ".stale" for f in _clean_reservations.iterdir())


def test_explicit_port_is_also_reserved(_clean_reservations):
    """An explicitly requested port must be claimed too, or --port races auto-allocation."""
    em.allocate_console_port(5566)
    assert (_clean_reservations / "5566.port").exists()
    assert em.allocate_console_port(None) != 5566


@pytest.mark.parametrize("preferred", [None, 5554])
def test_a_boot_that_releases_its_reservation_after_the_snapshot_is_not_reallocated(
    monkeypatch, _clean_reservations, preferred,
):
    """A live boot can replace its reservation while another caller holds an old snapshot."""
    running = []
    monkeypatch.setattr(em, "running_emulators", lambda: list(running))
    claim = em._claim_console_port

    def finish_competing_boot_then_claim(port):
        if port == 5554:
            assert claim(port)
            running.append({"serial": "emulator-5554"})
            em.release_console_port(port)
        return claim(port)

    monkeypatch.setattr(em, "_claim_console_port", finish_competing_boot_then_claim)

    if preferred is None:
        assert em.allocate_console_port() == 5556
        assert (_clean_reservations / "5556.port").is_file()
    else:
        with pytest.raises(DeviceError, match="already in use"):
            em.allocate_console_port(preferred)
    assert not (_clean_reservations / "5554.port").exists()


def test_an_explicit_port_losing_its_claim_preserves_the_winners_reservation(
    monkeypatch, _clean_reservations,
):
    claim = em._claim_console_port

    def competing_claim(port):
        assert claim(port)
        return claim(port)

    monkeypatch.setattr(em, "_claim_console_port", competing_claim)

    with pytest.raises(DeviceError, match="already in use"):
        em.allocate_console_port(5554)
    assert (_clean_reservations / "5554.port").is_file()


def test_odd_and_out_of_range_ports_rejected():
    from android_ui_analyser.errors import UsageError

    with pytest.raises(UsageError):
        em.allocate_console_port(5555)
    with pytest.raises(UsageError):
        em.allocate_console_port(9999)
