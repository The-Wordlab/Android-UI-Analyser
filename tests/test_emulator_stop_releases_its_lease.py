"""A stopped emulator's lease goes with it (#12).

``aua emulator stop --mine --owner <label>`` returned ``stopped``: the emulator process was
gone and adb listed nothing, yet the host-wide lease file for its serial was still there.
Leases are bound to the calling agent's process, which is deliberately long-lived - an IDE,
an agent harness, a reused CI worker - so that file never expired, and the next
``emulator start --port`` on that console port was refused as "already in use" without
anyone having done anything wrong.

Only the serial-scoped rollback released its lease. Every path that stops an emulator must
drop the lease of the device it stopped - and only once that device is actually gone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from android_ui_analyser import emulator as em
from android_ui_analyser import leases

# Outside the real console-port range, so a stubbing mistake here can never reach a device.
SERIAL = "emulator-9998"
SYNTHETIC_PID = 4242


def _live_owner(label: str) -> leases.LeaseOwner:
    """A process-bound owner whose process is alive: this test process, like an IDE would be."""
    import os

    pid = os.getpid()
    return leases.LeaseOwner(label, pid=pid, started=leases._proc_started(pid))


def _record(cache: Path, *, owner: str = "agent-a", avd: str = "fake") -> Path:
    path = cache / "emulator" / f"{avd}.p9998.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "avd": avd,
                "instance": f"{avd}.p9998",
                "serial": SERIAL,
                "pid": SYNTHETIC_PID,
                "owner": owner,
                "started_by_aua": True,
                "started_at": time.time() - 30,
            }
        )
    )
    return path


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """No adb visibility, no real kills; the synthetic emulator exits when signalled."""
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    monkeypatch.setattr(em, "_kill_watchdog", lambda _meta: None)
    killed: list[str] = []
    monkeypatch.setattr(em, "_adb_emu_kill", killed.append)
    signalled: list[int] = []
    monkeypatch.setattr(em.os, "killpg", lambda pid, _sig: signalled.append(pid))
    real_kill = em.os.kill

    def probe(pid: int, sig: int) -> None:
        if pid == SYNTHETIC_PID and sig == 0:
            raise ProcessLookupError
        real_kill(pid, sig)

    monkeypatch.setattr(em.os, "kill", probe)
    return {"killed": killed, "signalled": signalled}


def _lease_file(registry: Path) -> Path:
    return registry / "leases" / f"{SERIAL}.json"


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param({"owner": "agent-a"}, id="owner"),
        pytest.param({"mine": True}, id="mine"),
        pytest.param({"avd": "fake"}, id="avd"),
        pytest.param({"all_devices": True}, id="all-residue"),
    ],
)
def test_a_stop_drops_the_lease_of_the_emulator_it_stopped(
    scope: dict, host: dict[str, list], tmp_path: Path
) -> None:
    registry, cache = tmp_path / "registry", tmp_path / "cache"
    record = _record(cache)
    holder = _live_owner("agent-a")
    assert leases.acquire(registry, SERIAL, owner=holder)

    out = em.stop(**scope, cache_dir=cache, lease_registry_dir=registry, lease_owner=holder)

    assert host["signalled"] == [SYNTHETIC_PID]
    if "all_devices" in scope:
        assert out["signalled_pids"] == [SYNTHETIC_PID]
    else:
        assert out["stopped"] == [SERIAL]
    assert not record.exists()
    assert leases.read_lease(registry, SERIAL) is None
    assert not _lease_file(registry).exists(), (
        "the lease must be removed, not left to expire: its owner process is alive and "
        "would keep it for the rest of the session"
    )


def test_stopping_a_running_serial_without_a_record_drops_its_lease(
    host: dict[str, list], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An emulator adb can see but this cache never recorded is stopped over the console."""
    monkeypatch.setattr(em, "running_emulators", lambda: [{"serial": SERIAL}])
    registry, cache = tmp_path / "registry", tmp_path / "cache"
    holder = _live_owner("agent-a")
    assert leases.acquire(registry, SERIAL, owner=holder)

    out = em.stop(serial=SERIAL, cache_dir=cache, lease_registry_dir=registry, lease_owner=holder)

    assert host["killed"] == [SERIAL]
    assert out["stopped"] == [SERIAL]
    assert not _lease_file(registry).exists()


def test_an_emulator_that_survives_the_signal_keeps_its_lease_and_is_not_reported_stopped(
    host: dict[str, list], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A delivered signal is not a stopped device. Nothing is released until it is gone."""
    monkeypatch.setattr(em.os, "kill", lambda _pid, _sig: None)  # still alive
    monkeypatch.setattr(em, "_OWNED_STOP_TIMEOUT_S", 0.05)
    registry, cache = tmp_path / "registry", tmp_path / "cache"
    record = _record(cache)
    holder = _live_owner("agent-a")
    assert leases.acquire(registry, SERIAL, owner=holder)

    out = em.stop(owner="agent-a", cache_dir=cache, lease_registry_dir=registry, lease_owner=holder)

    assert host["signalled"] == [SYNTHETIC_PID]
    assert out["stopped"] == []
    assert out["still_running"] == [SERIAL]
    assert record.exists(), "the record stays so the next stop can find the instance"
    assert leases.read_lease(registry, SERIAL) is not None
