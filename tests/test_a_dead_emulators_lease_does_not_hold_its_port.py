"""A lease for an emulator that no longer answers does not hold its console port (#12).

Leases are bound to the calling agent's process on purpose: an agent that thinks for ten
minutes must not lose its device to a TTL. But that process is an IDE, an agent harness or a
reused CI worker, and it outlives every emulator it ever started. So a lease for
``emulator-5556`` whose emulator had been dead for 27 hours still read as live, and
``emulator start --port 5556`` was refused on a port nothing listened on.

The registry is in the allocator to keep a *live* emulator's port safe when adb blinks. A
live emulator always answers on its console port; a dead one never does. That is the test.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest

from android_ui_analyser import emulator as em
from android_ui_analyser import leases
from android_ui_analyser.errors import DeviceError


def _live_owner(label: str = "worker") -> leases.LeaseOwner:
    pid = os.getpid()
    return leases.LeaseOwner(label, pid=pid, started=leases._proc_started(pid))


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, set[int]]:
    """Empty adb, no boot records, private reservations; the probe answers only where told."""
    monkeypatch.setattr(em, "running_emulators", lambda: [])
    monkeypatch.setattr(em, "_aua_started_records", lambda _cache: [])
    reservations = tmp_path / "portlocks"
    reservations.mkdir()
    monkeypatch.setattr(em, "_reservation_dir", lambda: reservations)
    answering: set[int] = set()
    monkeypatch.setattr(em, "_console_answers", lambda port: port in answering)
    return {"answering": answering}


def _lease_as_left_by_a_finished_run(registry: Path, serial: str) -> None:
    """Dead ``pid``, live ``owner_pid``, activity a day old: the reported file, exactly."""
    assert leases.acquire(registry, serial, owner=_live_owner("verify-lane0"), ttl_s=120)
    path = registry / "leases" / f"{serial}.json"
    entry = json.loads(path.read_text())
    entry["pid"] = 2**22 + 99
    entry["last_activity"] = time.time() - 27 * 3600
    path.write_text(json.dumps(entry))
    assert leases.read_lease(registry, serial) is not None, "its owner process is alive"


def test_a_leased_port_whose_console_still_answers_stays_unavailable(
    host: dict[str, set[int]], tmp_path: Path
) -> None:
    """The adb-blink guarantee: a live emulator's port is held even when adb omits it."""
    registry = tmp_path / "registry"
    assert leases.acquire(registry, "emulator-5554", owner=_live_owner())
    host["answering"].add(5554)

    port = em.allocate_console_port(None, cache_dir=tmp_path / "cache", lease_registry_dir=registry)

    assert port == 5556


def test_a_leased_port_whose_console_is_silent_is_free_while_its_owner_still_lives(
    host: dict[str, set[int]], tmp_path: Path
) -> None:
    registry = tmp_path / "registry"
    _lease_as_left_by_a_finished_run(registry, "emulator-5556")

    explicit = em.allocate_console_port(
        5556, cache_dir=tmp_path / "cache", lease_registry_dir=registry
    )
    em.release_console_port(explicit)
    automatic = em.allocate_console_port(
        None, cache_dir=tmp_path / "cache", lease_registry_dir=registry
    )

    assert explicit == 5556
    assert automatic == 5554, "and auto-allocation is untouched by the stale record"


def test_a_lease_that_cannot_be_read_holds_its_port_even_when_nothing_answers(
    host: dict[str, set[int]], tmp_path: Path
) -> None:
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

    assert port == 5556, "unreadable is not free: fail closed"


def test_the_refusal_says_the_holder_and_that_its_emulator_is_running(
    host: dict[str, set[int]], tmp_path: Path
) -> None:
    registry = tmp_path / "registry"
    assert leases.acquire(registry, "emulator-5554", owner=_live_owner("worker-b"))
    host["answering"].add(5554)

    with pytest.raises(DeviceError, match="already in use") as excinfo:
        em.allocate_console_port(5554, cache_dir=tmp_path / "cache", lease_registry_dir=registry)

    hint = str(excinfo.value.hint)
    assert "'worker-b'" in hint
    assert "console" in hint and "answers" in hint


def test_the_console_probe_sees_a_listener_and_its_absence() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert em._console_answers(port) is True
    assert em._console_answers(port) is False
