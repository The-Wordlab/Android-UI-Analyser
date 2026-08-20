"""Idleness alone is a weak reason to kill an emulator; idleness plus no lease is a strong one.

A wait can legitimately block for 90-120s, and an orchestrator can pause between steps — so a
short wall-clock idle timeout on its own would stop emulators out from under working agents. The
lease is the signal that makes a short timeout safe, and it is fast in exactly the direction that
matters: a lease whose owner process is gone reads as expired immediately, before its TTL is even
consulted. So "idle AND unleased" means the agent is really gone.

The other half is that the watchdog is a *process* spawned once at boot. Nothing re-spawns it, so
a host reboot or a stray kill leaves that emulator immortal — which is what the adoption sweep
covers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import emulator_watchdog, leases
from android_ui_analyser.config import Config


def _meta(cache: Path, instance: str, *, serial: str, idle_s: float, age_s: float) -> Path:
    path = cache / "emulator" / f"{instance}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "started_by_aua": True,
                "serial": serial,
                "avd": instance,
                "idle_timeout_s": idle_s,
                "last_activity": time.time() - age_s,
                "pid": None,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make the watchdog's one loop pass observable without booting anything."""
    stops: list[dict[str, Any]] = []
    monkeypatch.setattr(emulator_watchdog, "_still_running", lambda serial, pid: True)
    monkeypatch.setattr(emulator_watchdog, "_POLL_S", 0.01)
    monkeypatch.setattr(
        "android_ui_analyser.emulator.stop",
        lambda **kw: stops.append(kw) or {"ok": True},
    )
    monkeypatch.setattr(emulator_watchdog, "_reset_device_changes", lambda cache, serial: None)
    return {"stops": stops}


def test_an_idle_but_leased_emulator_is_left_running(
    tmp_path: Path, _stub_runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    _meta(cache, "pixel", serial="emulator-5554", idle_s=120, age_s=600)
    assert leases.acquire(cache, "emulator-5554", owner="claude-still-working")

    # The loop would poll forever while the lease holds, so stop it after one pass.
    passes = {"n": 0}

    def one_pass(_seconds: float) -> None:
        passes["n"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(emulator_watchdog.time, "sleep", one_pass)
    with pytest.raises(KeyboardInterrupt):
        emulator_watchdog.run_watchdog(cache_dir=str(cache), instance="pixel")

    assert passes["n"] == 1
    assert _stub_runtime["stops"] == [], (
        "an emulator was stopped while an agent still held its lease"
    )


def test_an_idle_and_unleased_emulator_is_stopped(
    tmp_path: Path, _stub_runtime: dict[str, Any]
) -> None:
    cache = tmp_path / "cache"
    _meta(cache, "pixel", serial="emulator-5554", idle_s=120, age_s=600)

    assert emulator_watchdog.run_watchdog(cache_dir=str(cache), instance="pixel") == 0

    assert [s.get("serial") for s in _stub_runtime["stops"]] == ["emulator-5554"]
    assert _stub_runtime["stops"][0]["requested_by"] == "idle-watchdog"


def test_a_dead_owners_lease_frees_the_emulator_immediately(
    tmp_path: Path, _stub_runtime: dict[str, Any]
) -> None:
    """No TTL wait: the pid check runs first, which is what makes a 2-minute timeout usable."""
    import os

    pid = os.fork()
    if pid == 0:  # pragma: no cover — child exits at once
        os._exit(0)
    os.waitpid(pid, 0)

    cache = tmp_path / "cache"
    _meta(cache, "pixel", serial="emulator-5554", idle_s=120, age_s=600)
    owner = leases.LeaseOwner("ghost-agent", pid=pid, started="Jan1")
    assert leases.acquire(cache, "emulator-5554", owner=owner, ttl_s=99_999)

    assert emulator_watchdog.run_watchdog(cache_dir=str(cache), instance="pixel") == 0
    assert [s.get("serial") for s in _stub_runtime["stops"]] == ["emulator-5554"]


def test_an_unreadable_lease_directory_is_treated_as_held(tmp_path: Path) -> None:
    """Never guess a device is free: the cost of a wrong guess is a killed working emulator."""

    def explode(*_a: Any, **_k: Any) -> None:
        raise OSError("lease store unreadable")

    import android_ui_analyser.leases as lease_mod

    original = lease_mod.read_lease
    try:
        lease_mod.read_lease = explode  # type: ignore[assignment]
        assert emulator_watchdog._leased(tmp_path, "emulator-5554") == "unknown"
    finally:
        lease_mod.read_lease = original  # type: ignore[assignment]


def test_the_idle_timeout_is_twenty_minutes_and_configurable() -> None:
    """Long enough that a human driving a windowed AVD by hand never crosses it.

    `last_activity` measures AUA activity, not human activity — manual taps are invisible to us.
    That is the whole reason this is twenty minutes and not two: at two minutes, gated only on
    idleness, we would stop an emulator someone was in the middle of using.
    """
    assert Config().teardown.emulator_idle_stop_s == 1200.0


# --------------------------------------------------------------- orphan adoption


def _record(
    cache: Path,
    instance: str,
    *,
    serial: str,
    idle_s: float,
    watchdog_pid: int | None,
    explicit: bool = False,
) -> Path:
    path = cache / "emulator" / f"{instance}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "started_by_aua": True,
                "avd": instance.split(".")[0],
                "serial": serial,
                "idle_timeout_s": idle_s,
                "idle_stop_explicit": explicit,
                "watchdog_pid": watchdog_pid,
                "started_at": time.time() - 3600,
                "cmd": ["emulator"],
                "pid": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def _adoption(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from android_ui_analyser import emulator as emu

    spawned: list[str] = []
    monkeypatch.setattr(
        emu, "_spawn_idle_watchdog", lambda **kw: spawned.append(kw["instance"]) or 9001
    )
    monkeypatch.setattr(
        emu, "running_emulators", lambda: [{"serial": "emulator-5554"}, {"serial": "emulator-5560"}]
    )
    return {"spawned": spawned}


def test_an_emulator_whose_watchdog_died_gets_a_new_one(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    """A host reboot or a stray kill must not make an emulator immortal.

    Seen on a dev host: a record with idle_timeout_s 900 and watchdog_pid None. Nothing in the
    codebase looked for that, so the only thing that would ever have stopped it was gone.
    """
    from android_ui_analyser import emulator as emu

    _record(tmp_path, "pixel.p5554", serial="emulator-5554", idle_s=900, watchdog_pid=None)

    adopted = emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200)

    assert [a["serial"] for a in adopted] == ["emulator-5554"]
    assert _adoption["spawned"] == ["pixel.p5554"]
    meta = json.loads((tmp_path / "emulator" / "pixel.p5554.json").read_text(encoding="utf-8"))
    assert meta["watchdog_pid"] == 9001
    assert meta["idle_timeout_s"] == 1200


def test_a_still_guarded_emulator_is_left_alone(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    """Two watchdogs on one instance would race to stop it and to rewrite its meta."""
    import os

    from android_ui_analyser import emulator as emu

    _record(
        tmp_path, "pixel.p5554", serial="emulator-5554", idle_s=1200, watchdog_pid=os.getpid()
    )

    assert emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200) == []
    assert _adoption["spawned"] == []


def test_an_explicit_never_stop_is_honoured(tmp_path: Path, _adoption: dict[str, Any]) -> None:
    """`--idle-stop 0` is an instruction, not an oversight."""
    from android_ui_analyser import emulator as emu

    _record(
        tmp_path,
        "pixel.p5554",
        serial="emulator-5554",
        idle_s=0,
        watchdog_pid=None,
        explicit=True,
    )

    assert emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200) == []
    assert _adoption["spawned"] == []


def test_a_record_with_no_stated_timeout_is_adopted(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    """Instances booted before this existed stored 0 for windowed; they still need a net."""
    from android_ui_analyser import emulator as emu

    _record(tmp_path, "pixel.p5554", serial="emulator-5554", idle_s=0, watchdog_pid=None)

    adopted = emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200)

    assert [a["idle_timeout_s"] for a in adopted] == [1200]


def test_a_longer_deliberate_timeout_is_never_shortened(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    """Adopting must not silently retune someone's boot-time choice downwards."""
    from android_ui_analyser import emulator as emu

    _record(
        tmp_path, "pixel.p5554", serial="emulator-5554", idle_s=7200, watchdog_pid=None, explicit=True
    )

    adopted = emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200)

    assert [a["idle_timeout_s"] for a in adopted] == [7200]


def test_an_emulator_that_is_no_longer_running_is_not_adopted(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    """A stale record must not spawn a watchdog for a device that is already gone."""
    from android_ui_analyser import emulator as emu

    _record(tmp_path, "ghost.p5900", serial="emulator-5900", idle_s=1200, watchdog_pid=None)

    assert emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200) == []
    assert _adoption["spawned"] == []


def test_an_emulator_aua_did_not_start_is_never_adopted(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    """Someone else's Android Studio session is not ours to supervise, or to stop."""
    from android_ui_analyser import emulator as emu

    path = tmp_path / "emulator" / "theirs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"serial": "emulator-5554", "avd": "theirs", "idle_timeout_s": 0}),
        encoding="utf-8",
    )

    assert emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=1200) == []
    assert _adoption["spawned"] == []


def test_adoption_is_off_when_the_timeout_is_disabled(
    tmp_path: Path, _adoption: dict[str, Any]
) -> None:
    from android_ui_analyser import emulator as emu

    _record(tmp_path, "pixel.p5554", serial="emulator-5554", idle_s=0, watchdog_pid=None)

    assert emu.adopt_idle_watchdogs(cache_dir=tmp_path, idle_timeout_s=0) == []
    assert _adoption["spawned"] == []
