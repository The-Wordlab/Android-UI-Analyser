"""AUA can pre-set a debuggable app's Jetpack DataStore state instead of driving the UI for it.

The gap this closes. ``prefs_write`` covers ``shared_prefs/<file>.xml``, and that is where
feature flags live -- but an app's *own* persisted state (the theme, whether onboarding is
done, a launch counter) sits in Jetpack DataStore instead: a protobuf map under
``files/datastore`` that no ``adb`` or ``settings`` command can read, let alone write. The
existing prefs-write test says so in its own docstring: "some state (an environment held in a
DataStore protobuf) has no flow step at all, so setup had to walk a developer screen". Walking
the screen is right when the *setting change* is the thing under test, and pure cost when the
setting is only a precondition.

What this file pins is the wiring, not the codec -- ``test_app_datastore.py`` owns the wire
format and the per-call safety rules. Here:

- **The undo is journalled before the write.** A theme or an onboarding flag outlives the agent
  that set it, and the next agent inherits an app configured for someone else's precondition
  with nothing on screen to say why. ``set_datastore`` does take its own restore point, but it
  takes it *inside* the write, which is too late for the ledger -- so the engine takes the
  pre-write backup itself and records against that.
- **The registered undo actually replays.** A recorded undo nobody can run is theatre.
- **Repeated writes keep the first restore point**, so teardown lands on the state before AUA
  touched the store at all, not on the last intermediate value. (The ledger is idempotent on
  key, so the naive version overwrites the original backup with an intermediate one.)
- **An unconfirmed call journals nothing.** It never reaches the device, so an undo entry for it
  would be a lie teardown then acts on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import app_datastore, device_ledger
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import CAPABILITY_METHODS
from android_ui_analyser.platforms.services import APP_DATASTORE
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config
from test_app_datastore import DIRECTORY, FILE, NAME, PKG, _device, _written


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perf={"async_memory": False},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _context(device: FakeDevice) -> device_ledger.UndoContext:
    return device_ledger.UndoContext(
        serial=device.serial,
        device=device,
        capability=lambda name: app_datastore if name == APP_DATASTORE else None,
    )


# --------------------------------------------------------------------------- capability


def test_the_capability_is_declared_with_the_methods_the_engine_calls() -> None:
    """A capability whose method set drifts from its module fails open at runtime, not here."""
    declared = CAPABILITY_METHODS[APP_DATASTORE]

    assert declared <= set(app_datastore.__all__)
    for method in declared:
        assert callable(getattr(app_datastore, method))


# --------------------------------------------------------------------------- write-ahead undo


def test_the_previous_store_is_journalled_before_it_is_replaced(tmp_path: Path) -> None:
    device = _device()
    engine = _engine(tmp_path, device)
    written: list[str] = []
    real_write = device.write_app_file

    def spy(package: str, path: str, data: bytes) -> None:
        written.append(path)
        real_write(package, path, data)

    device.write_app_file = spy  # type: ignore[method-assign]
    recorded_when_written: list[list[str]] = []
    real_record = engine.record_device_change

    def record_spy(**kwargs: Any) -> None:
        real_record(**kwargs)
        recorded_when_written.append(list(written))

    engine.record_device_change = record_spy  # type: ignore[method-assign]

    engine.datastore_set(
        PKG, NAME, {"user_theme_mode": {"type": "int", "value": 1}}, restart=False, confirmed=True
    )

    assert recorded_when_written == [[]], "the undo was journalled after the write"
    entries = device_ledger.read_ledger(device.serial)
    assert [entry.kind for entry in entries] == ["app_datastore"]
    assert entries[0].op == "restore_app_datastore"
    assert entries[0].args["datastore"] == NAME
    assert Path(entries[0].args["cache_dir"]).is_dir()


def test_the_registered_undo_puts_the_previous_datastore_back(tmp_path: Path) -> None:
    device = _device()
    engine = _engine(tmp_path, device)
    before = device.app_files[FILE]

    engine.datastore_set(
        PKG, NAME, {"user_theme_mode": {"type": "int", "value": 1}}, restart=False, confirmed=True
    )
    assert _written(device)["user_theme_mode"] == ("int", 1)

    outcome = device_ledger.replay(device.serial, context=_context(device))

    assert not outcome["failed"], outcome
    assert device.app_files[FILE] == before
    assert device_ledger.read_ledger(device.serial) == []


def test_repeated_writes_keep_the_original_restore_point(tmp_path: Path) -> None:
    """Teardown must land on the state before AUA touched the store, not on step two of three."""
    device = _device()
    engine = _engine(tmp_path, device)
    before = device.app_files[FILE]

    engine.datastore_set(
        PKG, NAME, {"user_theme_mode": {"type": "int", "value": 1}}, restart=False, confirmed=True
    )
    engine.datastore_set(
        PKG, NAME, {"user_theme_mode": {"type": "int", "value": 0}}, restart=False, confirmed=True
    )
    assert _written(device)["user_theme_mode"] == ("int", 0)
    assert len(device_ledger.read_ledger(device.serial)) == 1

    outcome = device_ledger.replay(device.serial, context=_context(device))

    assert not outcome["failed"], outcome
    assert device.app_files[FILE] == before


def test_an_unconfirmed_write_journals_nothing_and_touches_nothing(tmp_path: Path) -> None:
    device = _device()
    engine = _engine(tmp_path, device)
    before = device.app_files[FILE]

    with pytest.raises(DeviceError) as raised:
        engine.datastore_set(PKG, NAME, {"user_theme_mode": {"type": "int", "value": 1}})

    assert raised.value.code == "datastore_confirmation_required"
    assert device_ledger.read_ledger(device.serial) == []
    assert device.app_files[FILE] == before


# --------------------------------------------------------------------------- read passthroughs


def test_reads_and_backups_go_through_the_engine_without_journalling(tmp_path: Path) -> None:
    """Nothing here changes the device, so an undo entry would be residue teardown acts on."""
    device = _device()
    engine = _engine(tmp_path, device)

    assert engine.datastore_list(PKG)["datastores"] == [
        {"name": NAME, "path": FILE, "bytes": len(device.app_files[FILE])}
    ]
    assert engine.datastore_get(PKG, NAME, keys=["user_theme_mode"])["values"] == {
        "user_theme_mode": {"type": "int", "value": 2}
    }

    backup = engine.datastore_backup(PKG, NAME)
    listed = engine.datastore_backups(PKG, NAME)
    assert [item["id"] for item in listed["backups"]] == [backup["backup_id"]]

    assert device_ledger.read_ledger(device.serial) == []
    assert DIRECTORY  # the layout this whole feature is pinned to


def test_restore_through_the_engine_round_trips(tmp_path: Path) -> None:
    device = _device()
    engine = _engine(tmp_path, device)
    before = device.app_files[FILE]

    backup = engine.datastore_backup(PKG, NAME)
    engine.datastore_set(
        PKG, NAME, {"user_theme_mode": {"type": "int", "value": 1}}, restart=False, confirmed=True
    )
    assert device.app_files[FILE] != before

    engine.datastore_restore(PKG, NAME, backup["backup_id"], restart=False, confirmed=True)

    assert device.app_files[FILE] == before
