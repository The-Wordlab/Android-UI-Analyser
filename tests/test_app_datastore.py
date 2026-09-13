"""Jetpack DataStore preference inspection, guarded mutation, backup, and restore."""

from __future__ import annotations

import shlex
import struct
from pathlib import Path

import pytest

from android_ui_analyser import app_datastore
from android_ui_analyser.errors import DeviceError, UsageError
from conftest import FakeDevice

PKG = "com.example.app"
NAME = "settings"
DIRECTORY = "files/datastore"
FILE = f"{DIRECTORY}/{NAME}.preferences_pb"


# --------------------------------------------------------------------------- #
# a synthetic .preferences_pb, written the way the app's protobuf runtime writes one
#
# Deliberately assembled here with its own tiny writer rather than with the module's
# `encode`: a fixture produced by the encoder under test would prove only that the codec
# agrees with itself. A real device file is not an option -- these stores hold session
# tokens, so nothing copied off a device belongs in the repository.
# --------------------------------------------------------------------------- #


def _varint(number: int) -> bytes:
    out = bytearray()
    while True:
        byte = number & 0x7F
        number >>= 7
        if number:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _blob(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _preference(key: str, value: bytes) -> bytes:
    return _blob(1, _blob(1, key.encode("utf-8")) + _blob(2, value))


FIXTURE_ENTRIES = (
    ("user_theme_mode", _tag(3, 0) + _varint(2)),
    ("onboarding_complete", _tag(1, 0) + _varint(1)),
    ("launch_count", _tag(4, 0) + _varint(2**40)),
    ("mic_gain", _tag(2, 5) + struct.pack("<f", 0.5)),
    ("sampling_ratio", _tag(7, 1) + struct.pack("<d", 0.1)),
    ("display_name", _blob(5, b"qa-persona")),
    ("seen_tips", _blob(6, _blob(1, b"beat") + _blob(1, b"paint"))),
    ("device_salt", _blob(8, b"\x00\x01\x02")),
)

DECODED = {
    "user_theme_mode": ("int", 2),
    "onboarding_complete": ("bool", True),
    "launch_count": ("long", 2**40),
    "mic_gain": ("float", 0.5),
    "sampling_ratio": ("double", 0.1),
    "display_name": ("string", "qa-persona"),
    "seen_tips": ("string_set", ["beat", "paint"]),
    "device_salt": ("bytes", b"\x00\x01\x02"),
}


def _fixture() -> bytes:
    return b"".join(_preference(key, value) for key, value in FIXTURE_ENTRIES)


class DatastoreDevice(FakeDevice):
    """A fake whose ``run-as ls`` also serves ``files/datastore``, as a real app data dir does."""

    def _run_as(self, command: str) -> str:
        if self.run_as_error:
            return self.run_as_error
        argv = shlex.split(command)[2:]
        if argv and argv[0] == "ls" and argv[-1].rstrip("/").endswith("datastore"):
            directory = argv[-1].rstrip("/")
            rows = ["total 8"]
            for path, data in sorted(self.app_files.items()):
                parent, _, entry = path.rpartition("/")
                if parent == directory:
                    rows.append(
                        f"-rw------- 1 u0_a1 u0_a1 {len(data)} 2026-01-01 00:00 {entry}"
                    )
            return "\n".join(rows)
        return super()._run_as(command)


def _device(*, payload: bytes | None = None, **kwargs: object) -> DatastoreDevice:
    data = _fixture() if payload is None else payload
    return DatastoreDevice(package=PKG, app_files={FILE: data}, **kwargs)


def _written(device: DatastoreDevice) -> dict[str, tuple[str, object]]:
    return app_datastore.decode(device.app_files[FILE])


def _calls(device: DatastoreDevice) -> list[str]:
    return [name for name, _ in device.calls]


def test_codec_round_trips_an_app_written_file_byte_for_byte() -> None:
    raw = _fixture()

    preferences = app_datastore.decode(raw)

    assert preferences == DECODED
    assert list(preferences) == [key for key, _ in FIXTURE_ENTRIES]
    assert app_datastore.encode(preferences) == raw


def test_list_and_get_read_every_type_and_leave_the_app_alone(tmp_path: Path) -> None:
    device = _device()

    listed = app_datastore.list_datastores(device, PKG)
    assert listed["datastores"] == [
        {"name": NAME, "path": FILE, "bytes": len(_fixture())}
    ]

    result = app_datastore.get_datastore(device, PKG, NAME)
    assert result["values"]["user_theme_mode"] == {"type": "int", "value": 2}
    assert result["values"]["onboarding_complete"] == {"type": "bool", "value": True}
    assert result["values"]["launch_count"] == {"type": "long", "value": 2**40}
    assert result["values"]["mic_gain"] == {"type": "float", "value": 0.5}
    assert result["values"]["sampling_ratio"] == {"type": "double", "value": 0.1}
    assert result["values"]["display_name"] == {"type": "string", "value": "qa-persona"}
    assert result["values"]["seen_tips"] == {"type": "string_set", "value": ["beat", "paint"]}
    assert result["values"]["device_salt"] == {"type": "bytes", "value": "AAEC"}

    # The stored suffix is accepted as a name, and a key the app never wrote is simply absent.
    narrowed = app_datastore.get_datastore(
        device,
        PKG,
        f"{NAME}.preferences_pb",
        keys=["user_theme_mode", "never_written"],
    )
    assert narrowed["values"] == {"user_theme_mode": {"type": "int", "value": 2}}

    # A read is safe against a live app, so it must not cost the caller its navigation state.
    assert not [name for name in _calls(device) if name in ("stop_app", "launch_app")]


def test_set_requires_confirmation_before_touching_the_app(tmp_path: Path) -> None:
    device = _device()

    with pytest.raises(DeviceError, match="requires explicit confirmation") as raised:
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"user_theme_mode": {"type": "int", "value": 1}},
            cache_dir=tmp_path / "cache",
        )

    assert raised.value.code == "datastore_confirmation_required"
    assert device.calls == []
    assert device.app_files[FILE] == _fixture()


def test_set_stops_the_app_backs_up_merges_and_relaunches(tmp_path: Path) -> None:
    device = _device()

    result = app_datastore.set_datastore(
        device,
        PKG,
        NAME,
        {
            "user_theme_mode": {"type": "int", "value": 1},
            "display_name": {"type": "string", "value": "night-owl"},
        },
        confirmed=True,
        cache_dir=tmp_path / "cache",
    )

    # Stopped first because a live SingleProcessCoordinator would never re-read the file.
    order = _calls(device)
    assert order.index("stop_app") < order.index("write_app_file") < order.index("launch_app")

    written = _written(device)
    assert written["user_theme_mode"] == ("int", 1)
    assert written["display_name"] == ("string", "night-owl")
    # Keys nobody asked about survive untouched, in their original file order.
    assert written["launch_count"] == DECODED["launch_count"]
    assert written["device_salt"] == DECODED["device_salt"]
    assert list(written) == [key for key, _ in FIXTURE_ENTRIES]

    assert result["applied"] == {
        "user_theme_mode": {"type": "int", "value": 1},
        "display_name": {"type": "string", "value": "night-owl"},
    }
    assert result["app_restarted"] is True
    assert "force-stopped" in result["warning"]

    # The restore point holds the bytes that were there before the write.
    assert result["backup"]["reason"] == "before-set"
    backup_file = Path(result["backup"]["path"], f"{NAME}.preferences_pb")
    assert backup_file.read_bytes() == _fixture()
    assert result["backup_id"] == result["backup"]["id"]


def test_set_accepts_every_type_including_datastores_own_spellings(tmp_path: Path) -> None:
    device = _device()

    result = app_datastore.set_datastore(
        device,
        PKG,
        NAME,
        {
            "onboarding_complete": {"type": "boolean", "value": False},
            "user_theme_mode": {"type": "integer", "value": 0},
            "launch_count": {"type": "long", "value": -5},
            "mic_gain": {"type": "float", "value": 0.25},
            "sampling_ratio": {"type": "double", "value": 2.5},
            "display_name": {"type": "string", "value": "qa"},
            "seen_tips": {"type": "string_set", "value": ["a", "b"]},
            "device_salt": {"type": "bytes", "value": "//8A"},
        },
        confirmed=True,
        cache_dir=tmp_path / "cache",
    )

    assert _written(device) == {
        "onboarding_complete": ("bool", False),
        "user_theme_mode": ("int", 0),
        "launch_count": ("long", -5),
        "mic_gain": ("float", 0.25),
        "sampling_ratio": ("double", 2.5),
        "display_name": ("string", "qa"),
        "seen_tips": ("string_set", ["a", "b"]),
        "device_salt": ("bytes", b"\xff\xff\x00"),
    }
    # The Kotlin spellings are accepted on the way in and normalised on the way out.
    assert result["applied"]["onboarding_complete"] == {"type": "bool", "value": False}
    assert result["applied"]["user_theme_mode"] == {"type": "int", "value": 0}
    assert result["applied"]["device_salt"] == {"type": "bytes", "value": "//8A"}


def test_a_bare_value_is_typed_from_the_entry_it_replaces(tmp_path: Path) -> None:
    """The store already knows each key's type, so making the caller restate it only adds risk.

    ``int`` and ``long`` are different oneof fields: a caller who writes a ``launch_count`` of 3
    as an ``int`` because 3 is small has not changed a setting, it has planted a
    ``ClassCastException`` in the app's own reader. Reading the type off the entry being
    replaced is both the easier call and the only one that cannot get this wrong.
    """
    device = _device()

    result = app_datastore.set_datastore(
        device,
        PKG,
        NAME,
        {
            "user_theme_mode": 1,
            "onboarding_complete": False,
            "launch_count": 7,
            "display_name": "night-owl",
            "seen_tips": ["paint"],
        },
        confirmed=True,
        cache_dir=tmp_path / "cache",
    )

    written = _written(device)
    assert written["user_theme_mode"] == ("int", 1)
    assert written["onboarding_complete"] == ("bool", False)
    # 7 fits in an int, and is still written as the long the app declared it to be.
    assert written["launch_count"] == ("long", 7)
    assert written["display_name"] == ("string", "night-owl")
    assert written["seen_tips"] == ("string_set", ["paint"])
    assert result["applied"]["launch_count"] == {"type": "long", "value": 7}


def test_a_bare_value_for_an_unknown_key_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    device = _device()

    with pytest.raises(DeviceError, match="cannot be inferred") as raised:
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"never_written": 1},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )

    assert raised.value.code == "datastore_type_invalid"
    assert device.app_files[FILE] == _fixture(), "the store was written despite the refusal"


def test_a_bare_value_of_the_wrong_shape_is_still_checked(tmp_path: Path) -> None:
    """Inferring the type is not the same as accepting anything: the value still has to fit it."""
    device = _device()

    with pytest.raises(DeviceError, match="is an int"):
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"user_theme_mode": "dark"},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )

    assert device.app_files[FILE] == _fixture()


def test_a_mapping_missing_type_or_value_names_both_forms(tmp_path: Path) -> None:
    """A half-filled {"value": ...} is a typo, not a bare value, and must not be written as one."""
    device = _device()

    with pytest.raises(DeviceError, match="without both 'type' and 'value'"):
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"user_theme_mode": {"value": 1}},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )

    assert device.app_files[FILE] == _fixture()


def test_a_store_that_does_not_parse_is_never_overwritten(tmp_path: Path) -> None:
    # A top-level varint where the map expects a length-delimited entry: whatever this file
    # is, it is not a preferences map, and guessing at it is how every key gets deleted.
    device = _device(payload=b"\x08\x01not-a-preference-map")
    original = dict(device.app_files)

    with pytest.raises(DeviceError, match="not a readable DataStore") as read_error:
        app_datastore.get_datastore(device, PKG, NAME)
    assert read_error.value.code == "datastore_corrupt"

    with pytest.raises(DeviceError) as write_error:
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"user_theme_mode": {"type": "int", "value": 1}},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )
    assert write_error.value.code == "datastore_corrupt"
    assert device.app_files == original
    assert "write_app_file" not in _calls(device)


def test_unknown_types_and_mismatched_values_are_refused_before_the_device(
    tmp_path: Path,
) -> None:
    device = _device()

    with pytest.raises(DeviceError, match="unknown datastore type") as unknown:
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"user_theme_mode": {"type": "int32", "value": 1}},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )
    assert unknown.value.code == "datastore_type_invalid"

    with pytest.raises(DeviceError, match="is a bool") as mismatch:
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"onboarding_complete": {"type": "bool", "value": "true"}},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )
    assert mismatch.value.code == "datastore_type_invalid"

    with pytest.raises(DeviceError, match="out of range") as overflow:
        app_datastore.set_datastore(
            device,
            PKG,
            NAME,
            {"user_theme_mode": {"type": "int", "value": 2**31}},
            confirmed=True,
            cache_dir=tmp_path / "cache",
        )
    assert overflow.value.code == "datastore_type_invalid"

    # Nothing was stopped, read, or written to find any of that out.
    assert device.calls == []
    assert device.app_files[FILE] == _fixture()


def test_run_as_refusal_is_a_structured_device_error() -> None:
    device = _device(run_as_error="run-as: package not debuggable")

    with pytest.raises(DeviceError, match="not debuggable") as listing:
        app_datastore.list_datastores(device, PKG)
    assert listing.value.code == "datastore_access"

    with pytest.raises(DeviceError, match="not debuggable") as read:
        app_datastore.get_datastore(device, PKG, NAME)
    assert read.value.code == "datastore_access"


def test_a_name_is_a_basename_not_a_path() -> None:
    device = _device()

    for bad in ("../databases/app.db", "settings/../../app.db", "..", ".hidden", "", "   "):
        with pytest.raises(UsageError) as raised:
            app_datastore.get_datastore(device, PKG, bad)
        assert raised.value.code == "datastore_name_invalid", bad

    # Traversal hidden behind the stored suffix is refused on the same grounds.
    with pytest.raises(UsageError) as suffixed:
        app_datastore.get_datastore(device, PKG, "../other/app.preferences_pb")
    assert suffixed.value.code == "datastore_name_invalid"

    assert device.calls == []


def test_an_unknown_datastore_names_the_ones_that_exist() -> None:
    device = _device()

    with pytest.raises(DeviceError, match="does not exist") as raised:
        app_datastore.get_datastore(device, PKG, "missing")

    assert raised.value.code == "datastore_not_found"
    assert "Available: settings." in str(raised.value)


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    device = _device()
    cache = tmp_path / "cache"

    backup = app_datastore.backup_datastore(device, PKG, NAME, cache_dir=cache)
    # A backup is a plain read, so it costs no navigation state.
    assert not [name for name in _calls(device) if name in ("stop_app", "launch_app")]

    app_datastore.set_datastore(
        device,
        PKG,
        NAME,
        {"user_theme_mode": {"type": "int", "value": 1}},
        confirmed=True,
        cache_dir=cache,
    )
    assert _written(device)["user_theme_mode"] == ("int", 1)

    with pytest.raises(DeviceError) as unconfirmed:
        app_datastore.restore_datastore(device, PKG, NAME, backup["backup_id"], cache_dir=cache)
    assert unconfirmed.value.code == "datastore_confirmation_required"

    restored = app_datastore.restore_datastore(
        device,
        PKG,
        NAME,
        backup["backup_id"],
        confirmed=True,
        cache_dir=cache,
    )

    assert device.app_files[FILE] == _fixture()
    assert restored["restored_backup"]["id"] == backup["backup_id"]
    assert restored["safety_backup"]["reason"] == f"before-restore-{backup['backup_id']}"
    assert restored["app_restarted"] is True
    order = _calls(device)
    assert order.index("stop_app") < order.index("write_app_file") < order.index("launch_app")

    listed = app_datastore.list_backups(device, PKG, NAME, cache_dir=cache)
    assert [item["reason"] for item in listed["backups"]] == [
        f"before-restore-{backup['backup_id']}",
        "before-set",
        "manual",
    ]
