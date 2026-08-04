"""`flow save --last N` must not return steps from a scenario that used the same serial.

Observed in a real sweep, reported independently by three lanes: `flow save` produced an
`open_link` to a screen it never opened, a consent tap, a flag cycle it never ran and a tap
labelled from *another* test's content — none of which the saving lane had performed. One
lane nearly filed a hand-written flow containing another test's journey.

The action journal is keyed by device serial, and serials come from a small pool that is
handed out and reclaimed as workers come and go: `emulator-5554` may be one AVD, then a
second, then that second one again for a *different* run, all inside an hour. Giving
each worker its own `AUA_CACHE__DIR` cannot fix it, because the memory directory is
deliberately shared so learned routes accumulate — and it should stay shared. Only the
session is instance-specific.

So the cursor records which boot of the device it belongs to, and a mismatch discards it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMemoryStore, RouteStep, SessionState
from conftest import FakeDevice, make_config

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""

_SERIAL = "emulator-5554"


def _store(tmp_path: Path) -> AppMemoryStore:
    cfg = make_config(memory={"dir": str(tmp_path / "mem")})
    return AppMemoryStore(cfg.memory)


def _journey(*labels: str) -> list[RouteStep]:
    return [RouteStep(kind="tap", label=label) for label in labels]


def test_a_new_instance_discards_the_previous_workers_journal(tmp_path: Path) -> None:
    """The reported bug, in miniature: two scenarios, one recycled serial."""
    store = _store(tmp_path)
    store.save_session(
        _SERIAL,
        SessionState(
            package="com.example.app",
            instance="boot-of-the-previous-worker",
            recent=_journey("a title from another test's content"),
        ),
    )

    discarded = store.claim_session(_SERIAL, "boot-of-this-worker")

    assert discarded is True
    fresh = store.load_session(_SERIAL)
    assert fresh.recent == []
    assert fresh.package is None
    assert fresh.instance == "boot-of-this-worker"


def test_the_same_instance_keeps_its_own_journey(tmp_path: Path) -> None:
    """The fix must not amputate a live journey — that would break every `flow save`."""
    store = _store(tmp_path)
    store.save_session(
        _SERIAL,
        SessionState(package="com.example.app", instance="one-boot", recent=_journey("Home")),
    )

    assert store.claim_session(_SERIAL, "one-boot") is False
    kept = store.load_session(_SERIAL)
    assert [s.label for s in kept.recent] == ["Home"]
    assert kept.package == "com.example.app"


def test_an_unreadable_instance_never_destroys_a_session(tmp_path: Path) -> None:
    """No instance token means no observation, and an unobserved thing is not evidence.

    The same rule as the recording fix: state is only discarded on the strength of
    something actually read from the device.
    """
    store = _store(tmp_path)
    store.save_session(
        _SERIAL, SessionState(package="com.example.app", instance="one-boot", recent=_journey("Home"))
    )

    assert store.claim_session(_SERIAL, None) is False
    assert [s.label for s in store.load_session(_SERIAL).recent] == ["Home"]


def test_an_empty_cursor_is_stamped_without_being_reported_as_foreign(tmp_path: Path) -> None:
    """A first run has nothing to discard; it should just take ownership quietly."""
    store = _store(tmp_path)

    assert store.claim_session(_SERIAL, "first-boot") is False
    assert store.load_session(_SERIAL).instance == "first-boot"
    # And the stamp survives, so the next command is a no-op rather than a re-claim.
    assert store.claim_session(_SERIAL, "first-boot") is False


def test_an_unstamped_legacy_session_with_history_is_not_trusted(tmp_path: Path) -> None:
    """Pre-upgrade cursors carry no instance, so they cannot be vouched for.

    Discarding costs one journey's history once, at upgrade; trusting it risks handing a
    lane another lane's steps, which is the bug.
    """
    store = _store(tmp_path)
    store.save_session(_SERIAL, SessionState(package="com.example.app", recent=_journey("Home")))

    assert store.claim_session(_SERIAL, "this-boot") is True
    assert store.load_session(_SERIAL).recent == []


def test_the_engine_claims_the_session_on_a_real_journey(tmp_path: Path) -> None:
    """End to end: a device that reports an instance gets a cursor scoped to it.

    Without the wiring the store fix is dead code, so this pins the seam and not just the
    helper.
    """

    class _StampedDevice(FakeDevice):
        def instance_token(self) -> str | None:
            return "boot-of-this-worker"

    cfg = make_config(memory={"dir": str(tmp_path / "mem")})
    store = AppMemoryStore(cfg.memory)
    store.save_session(
        _SERIAL,
        SessionState(
            package="com.example.app",
            instance="boot-of-the-previous-worker",
            recent=_journey("someone else's tap"),
        ),
    )

    dev = _StampedDevice(hierarchy_xml=_XML, serial=_SERIAL)
    eng = Engine(cfg, device=dev)
    eng.analyze()

    sess = AppMemoryStore(cfg.memory).load_session(_SERIAL)
    assert sess.instance == "boot-of-this-worker"
    assert "someone else's tap" not in [s.label for s in sess.recent]


def test_reading_memory_without_a_device_does_not_connect_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Offline commands (`aua map --app …`) must not pay a connect timeout for this check.

    The claim needs a device, but asking for one here would make every offline memory read
    wait out a uiautomator2 connection attempt to learn nothing.
    """
    from android_ui_analyser import engine as engine_mod

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline memory access must not connect to a device")

    monkeypatch.setattr(engine_mod, "connect", _refuse)
    eng = Engine(make_config(memory={"dir": str(tmp_path / "mem")}))

    assert eng._memory is not None  # the store is usable; no device was touched
