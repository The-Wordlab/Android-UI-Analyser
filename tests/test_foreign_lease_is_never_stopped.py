"""A device with a live foreign lease is never provisioned over, stopped, or cleaned.

The live failure this file guards: a session start collided with another worker's leased
emulator. adb briefly omitted `emulator-5554` from its snapshot, so its console port looked
free; the new boot's expected serial was answered by the foreign device; the lease claim then
failed with `device_leased` — and the claim rollback issued a *serial-scoped* stop that killed
the foreign worker's emulator mid-run (`emulator stop via=serial stopped=[emulator-5554]`).

Three independent fences each break that chain, and each is tested here on its own:

1. port allocation treats live-leased serials as occupied, whatever adb reports;
2. a boot detects the collision (pre-spawn, and post-spawn when its own process died)
   before any settings/install/device mutation;
3. every stop path skips a serial whose live lease belongs to someone else, and rollbacks
   stop by owned instance/pid, never by ambiguous serial.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import emulator as em
from android_ui_analyser import leases
from android_ui_analyser.errors import DeviceError

# Serials outside the real emulator port range wherever a stop could ever be reached, so a
# stubbing bug in this file cannot touch a device someone is using.
SER_GUARD = "emulator-9998"


def _live_owner(label: str = "foreign-worker") -> leases.LeaseOwner:
    """A process-bound owner whose process is genuinely alive (this test process)."""
    pid = os.getpid()
    return leases.LeaseOwner(label, pid=pid, started=leases._proc_started(pid))


def _dead_owner(label: str = "dead-worker") -> leases.LeaseOwner:
    """A process-bound owner that no longer exists (its lease must read as expired)."""
    return leases.LeaseOwner(label, pid=2**22 + 1234, started="long-gone")


@pytest.fixture(autouse=True)
def _no_real_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No adb visibility and no real kills, for every test in this module."""
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    killed: list[str] = []
    monkeypatch.setattr(em, "_adb_emu_kill", killed.append)
    signalled: list[int] = []
    monkeypatch.setattr(em.os, "killpg", lambda pid, _sig: signalled.append(pid))
    return {"killed": killed, "signalled": signalled}


# ------------------------------------------------------------------- port allocation


def test_live_leased_console_port_stays_unavailable_when_adb_omits_it(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry"
    assert leases.acquire(registry, "emulator-5554", owner=_live_owner())

    # adb reports nothing and this cache has no boot records — the registry alone must
    # keep the foreign worker's port out of the allocatable set.
    port = em.allocate_console_port(
        None, cache_dir=tmp_path / "cache", lease_registry_dir=registry
    )

    assert port == 5556


def test_a_dead_owners_leased_port_is_reclaimed_immediately(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    assert leases.acquire(registry, "emulator-5554", owner=_live_owner("worker"))
    entry = json.loads((registry / "leases" / "emulator-5554.json").read_text())
    entry["owner_pid"] = 2**22 + 4321  # rewrite the holder as a process that is gone
    entry["owner_started"] = "long-gone"
    (registry / "leases" / "emulator-5554.json").write_text(json.dumps(entry))

    port = em.allocate_console_port(
        None, cache_dir=tmp_path / "cache", lease_registry_dir=registry
    )

    assert port == 5554


def test_unreadable_lease_metadata_blocks_its_port(tmp_path: Path) -> None:
    if os.name == "nt" or os.geteuid() == 0:  # pragma: no cover - permission model differs
        pytest.skip("chmod-based unreadability needs a non-root POSIX host")
    registry = tmp_path / "registry"
    assert leases.acquire(registry, "emulator-5554", owner=_live_owner())
    path = registry / "leases" / "emulator-5554.json"
    path.chmod(0o000)
    try:
        port = em.allocate_console_port(
            None, cache_dir=tmp_path / "cache", lease_registry_dir=registry
        )
    finally:
        path.chmod(0o644)

    assert port == 5556, "an unreadable lease must fail closed, not read as a free port"


# ------------------------------------------------------------------ boot collision


def _stub_avds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(em, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(
        em, "list_avds", lambda: {"ok": True, "avds": ["fakeavd"], "count": 1, "emulator": "x"}
    )


def test_start_fails_before_spawning_when_the_expected_serial_already_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    _stub_avds(monkeypatch)
    # The port snapshot sees nothing (adb blink) but the `before` snapshot sees the foreign
    # device: the boot must fail before Popen, having touched nothing.
    snapshots = iter([[], [{"serial": "emulator-5554", "state": "device"}]])
    monkeypatch.setattr(
        em, "running_emulators", lambda: next(snapshots, [{"serial": "emulator-5554"}])
    )

    def no_spawn(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a colliding boot must never spawn a process")

    monkeypatch.setattr(em.subprocess, "Popen", no_spawn)

    with pytest.raises(DeviceError, match="already belongs to a running emulator"):
        em.start("fakeavd", headless=True, cache_dir=tmp_path, parallel=True, wait_s=5)

    assert _no_real_devices["killed"] == []
    assert not (tmp_path / "emulator" / "fakeavd.p5554.json").exists()


def test_start_fails_when_its_spawned_process_dies_and_the_serial_is_foreign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    _stub_avds(monkeypatch)

    class DeadProc:
        pid = 4242

        @staticmethod
        def poll() -> int:
            return 1  # our emulator lost the port race and exited

    monkeypatch.setattr(em.subprocess, "Popen", lambda *a, **k: DeadProc())
    monkeypatch.setattr(em, "_wait_for_serial", lambda serial, **_k: serial)
    monkeypatch.setattr(em.time, "sleep", lambda *_: None)

    def no_mutation(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a foreign serial must not have its settings touched")

    monkeypatch.setattr(em, "_clear_inherited_blackholed_proxy", no_mutation)

    with pytest.raises(DeviceError, match="belongs to another instance"):
        em.start("fakeavd", headless=True, cache_dir=tmp_path, parallel=True, wait_s=5)

    assert _no_real_devices["killed"] == [], "the surviving serial is not ours to stop"
    assert not (tmp_path / "emulator" / "fakeavd.p5554.json").exists()


# ----------------------------------------------------------------------- stop fences


def test_serial_scoped_stop_skips_a_live_foreign_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    registry = tmp_path / "registry"
    assert leases.acquire(registry, SER_GUARD, owner=_live_owner())
    monkeypatch.setattr(
        em, "running_emulators", lambda: [{"serial": SER_GUARD, "state": "device"}]
    )

    out = em.stop(
        serial=SER_GUARD,
        cache_dir=tmp_path / "cache",
        lease_registry_dir=registry,
        lease_owner="somebody-else",
    )

    assert out["stopped"] == []
    assert _no_real_devices["killed"] == []
    assert out["skipped_leased"] == [{"serial": SER_GUARD, "holder": "foreign-worker"}]
    assert "lease release" in (out.get("hint") or "")


def test_the_leaseholder_may_still_stop_its_own_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    registry = tmp_path / "registry"
    me = _live_owner("its-me")
    assert leases.acquire(registry, SER_GUARD, owner=me)
    monkeypatch.setattr(
        em, "running_emulators", lambda: [{"serial": SER_GUARD, "state": "device"}]
    )

    out = em.stop(
        serial=SER_GUARD,
        cache_dir=tmp_path / "cache",
        lease_registry_dir=registry,
        lease_owner=me,
    )

    assert out["stopped"] == [SER_GUARD]
    assert out["skipped_leased"] == []


def test_a_dead_owners_lease_does_not_block_a_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    registry = tmp_path / "registry"
    (registry / "leases").mkdir(parents=True)
    (registry / "leases" / f"{SER_GUARD}.json").write_text(
        json.dumps(
            {
                "serial": SER_GUARD,
                "owner": "dead-worker",
                "owner_pid": 2**22 + 99,
                "owner_started": "long-gone",
                "last_activity": time.time(),
                "ttl_s": 900,
            }
        )
    )
    monkeypatch.setattr(
        em, "running_emulators", lambda: [{"serial": SER_GUARD, "state": "device"}]
    )

    out = em.stop(
        serial=SER_GUARD,
        cache_dir=tmp_path / "cache",
        lease_registry_dir=registry,
        lease_owner="somebody-else",
    )

    assert out["stopped"] == [SER_GUARD]


def test_owner_scoped_stop_preserves_a_record_whose_device_another_agent_leases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    registry = tmp_path / "registry"
    cache = tmp_path / "cache"
    record = cache / "emulator" / "fake.p9998.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "avd": "fake",
                "instance": "fake.p9998",
                "serial": SER_GUARD,
                "pid": 4242,
                "owner": "agent-a",
                "started_by_aua": True,
            }
        )
    )
    assert leases.acquire(registry, SER_GUARD, owner=_live_owner("the-new-holder"))

    out = em.stop(
        owner="agent-a",
        cache_dir=cache,
        lease_registry_dir=registry,
        lease_owner="agent-a",
    )

    assert out["stopped"] == []
    assert _no_real_devices["killed"] == []
    assert _no_real_devices["signalled"] == [], "the handed-off process is the holder's device"
    assert record.is_file(), "the record still describes a live, owned instance"


# ----------------------------------------------------------- rollback by owned boot


def _boot_record(cache: Path, *, serial: str, pid: int, started_at: float) -> Path:
    path = cache / "emulator" / "fake.p5554.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "avd": "fake",
                "instance": "fake.p5554",
                "serial": serial,
                "pid": pid,
                "started_by_aua": True,
                "started_at": started_at,
            }
        )
    )
    return path


def test_rollback_of_an_unleased_boot_stops_it_gracefully(
    tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    cache = tmp_path / "cache"
    record = _boot_record(cache, serial=SER_GUARD, pid=4242, started_at=time.time())

    out = em.stop_spawned_instance(
        instance="fake.p5554",
        pid=4242,
        cache_dir=cache,
        lease_registry_dir=tmp_path / "registry",
        owner="me",
        requested_by="session-start-claim-rollback",
    )

    assert out["stopped"] == [SER_GUARD]
    assert _no_real_devices["signalled"] == [4242]
    assert not record.exists()


def test_rollback_never_stops_the_serial_when_a_pre_existing_foreign_lease_holds_it(
    tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    """THE live-failure regression: claim failed because the serial was somebody else's."""
    registry = tmp_path / "registry"
    cache = tmp_path / "cache"
    assert leases.acquire(registry, SER_GUARD, owner=_live_owner())
    # The foreign lease predates our boot — the serial always belonged to the other worker,
    # so our recorded process is an unbound loser we must reap.
    record = _boot_record(cache, serial=SER_GUARD, pid=4242, started_at=time.time() + 60)

    out = em.stop_spawned_instance(
        instance="fake.p5554",
        pid=4242,
        cache_dir=cache,
        lease_registry_dir=registry,
        owner="me",
        requested_by="session-start-claim-rollback",
    )

    assert out["stopped"] == []
    assert _no_real_devices["killed"] == [], "the foreign device must never receive emu kill"
    assert _no_real_devices["signalled"] == [4242], "our own spawned loser is reaped"
    assert not record.exists()
    assert leases.holder(registry, SER_GUARD) == "foreign-worker"


def test_rollback_touches_nothing_when_the_lease_arrived_after_our_boot(
    tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    registry = tmp_path / "registry"
    cache = tmp_path / "cache"
    record = _boot_record(cache, serial=SER_GUARD, pid=4242, started_at=time.time() - 60)
    # Another agent legitimately claimed the fresh device between our boot and our failed
    # claim: the process behind the serial is their device now.
    assert leases.acquire(registry, SER_GUARD, owner=_live_owner("quick-claimer"))

    out = em.stop_spawned_instance(
        instance="fake.p5554",
        pid=4242,
        cache_dir=cache,
        lease_registry_dir=registry,
        owner="me",
    )

    assert out["stopped"] == []
    assert _no_real_devices["killed"] == []
    assert _no_real_devices["signalled"] == []
    assert record.exists()


def test_rollback_with_no_record_signals_only_its_own_pid(
    tmp_path: Path, _no_real_devices: dict[str, list]
) -> None:
    out = em.stop_spawned_instance(
        instance="never-written.p5554",
        pid=777,
        cache_dir=tmp_path / "cache",
        lease_registry_dir=tmp_path / "registry",
        owner="me",
    )

    assert out["stopped"] == []
    assert _no_real_devices["killed"] == []
    assert _no_real_devices["signalled"] == [777]


# ----------------------------------------------------------------- lease atomicity


def test_concurrent_acquire_has_exactly_one_winner(tmp_path: Path) -> None:
    winners: list[str] = []
    barrier = threading.Barrier(8)

    def contend(index: int) -> None:
        owner = f"contender-{index}"  # distinct TTL owners: no process binding, pure race
        barrier.wait()
        if leases.acquire(tmp_path, "emulator-5554", owner=owner):
            winners.append(owner)

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert leases.holder(tmp_path, "emulator-5554") == winners[0]


def test_a_dead_process_bound_owner_cannot_acquire_at_all(tmp_path: Path) -> None:
    """No TTL substitute: a claimed-but-dead caller identity must fail, not soften."""
    assert leases.acquire(tmp_path, "emulator-5554", owner=_dead_owner()) is False
    assert leases.read_lease(tmp_path, "emulator-5554") is None
    # A plain label that never claimed a process identity still gets its TTL lease.
    assert leases.acquire(tmp_path, "emulator-5554", owner="nightly-agent") is True


def test_unreadable_lease_metadata_reads_as_held_not_free(tmp_path: Path) -> None:
    if os.name == "nt" or os.geteuid() == 0:  # pragma: no cover - permission model differs
        pytest.skip("chmod-based unreadability needs a non-root POSIX host")
    assert leases.acquire(tmp_path, "emulator-5554", owner=_live_owner("holder"))
    path = tmp_path / "leases" / "emulator-5554.json"
    path.chmod(0o000)
    try:
        entry = leases.read_lease(tmp_path, "emulator-5554")
        assert entry is not None and entry.get("inaccessible") is True
        assert leases.acquire(tmp_path, "emulator-5554", owner="opportunist") is False
    finally:
        path.chmod(0o644)
    assert leases.holder(tmp_path, "emulator-5554") == "holder"


def test_a_live_process_bound_owner_never_expires_by_ttl(tmp_path: Path) -> None:
    """Process death is the only expiry for a bound owner; no TTL clock runs beside it."""
    assert leases.acquire(tmp_path, "emulator-5554", owner=_live_owner("busy"), ttl_s=1)
    path = tmp_path / "leases" / "emulator-5554.json"
    entry = json.loads(path.read_text())
    entry["last_activity"] = time.time() - 3600  # idle far past any TTL
    path.write_text(json.dumps(entry))

    current = leases.read_lease(tmp_path, "emulator-5554")

    assert current is not None and current["owner"] == "busy", (
        "an idle-but-alive owner is mid-wait, not gone; TTL must not seize its device"
    )
