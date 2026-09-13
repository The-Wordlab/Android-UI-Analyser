"""Safe host-side inspection and mutation of a debuggable app's DataStore Preferences.

Some scenarios need a setting as a *precondition* rather than as the thing under test.  Feature
flags live in ``shared_prefs`` and are written by :mod:`android_ui_analyser.flags`; everything an
app keeps in Jetpack DataStore -- theme, onboarding state, counters -- lives instead in
``files/datastore/<name>.preferences_pb``, a protobuf map that no Android system binary can edit.
AUA therefore reads that file through ``run-as``, decodes it with the dependency-free codec below,
and writes back only bytes that same codec produced.

Three rules are enforced rather than documented, because each one is a real, silent failure:

1. **A mutation force-stops the app first.**  DataStore 1.2.1 mediates a file through
   ``SingleProcessCoordinator``, which holds a *process-local* version counter and an empty
   ``updateNotifications`` flow.  A running app therefore never re-reads the file after its first
   load: the write appears to succeed and changes nothing, and if the app writes its own copy back
   before the process dies, the edit is lost for the life of that process.  Reads do not stop the
   app -- DataStore replaces the file by rename, so a live read returns one whole version or
   another, and a stale read is a stale read, not a lost write.
2. **Never write bytes the codec did not produce, and always take a restore point first.**  A
   parse failure raises ``CorruptionException``, and apps install
   ``ReplaceFileCorruptionHandler { emptyPreferences() }`` -- which deletes *every* key, the
   session included, without crashing or warning.  :func:`set_datastore` re-parses what it is
   about to write and refuses to install anything that does not round-trip.
3. **``run-as`` only; there is deliberately no ``adb root`` path.**  ``run-as`` carries the app's
   uid and SELinux categories.  A file created by root keeps root's owner and label through the
   rename, and the app then reads it either as absent (``emptyPreferences()``, silently) or as an
   unhandled ``IOException``.  Only debuggable builds have a ``run-as`` route at all, which is what
   keeps this a QA tool rather than something that can be pointed at a shipped build.

Wire schema (``androidx.datastore.preferences.protobuf``)::

    PreferenceMap { map<string, Value> preferences = 1; }
    map entry:  key = 1 (string), value = 2 (Value)
    Value oneof: boolean=1(varint) float=2(fixed32) integer=3(varint) long=4(varint)
                 string=5(len) string_set=6(len StringSet{strings=1 repeated})
                 double=7(fixed64) bytes=8(len)

:func:`decode` returns an insertion-ordered mapping, so ``encode(decode(x)) == x`` for any file a
canonical protobuf implementation wrote -- which is what makes an edit a surgical one-key change
rather than a rewrite of the whole map.
"""

from __future__ import annotations

import base64
import contextlib
import json
import math
import re
import struct
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from shlex import quote
from typing import Any

from .device import Device
from .errors import DeviceError, UsageError

_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")
_DATASTORE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_BACKUP_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RUN_AS_ERRORS = ("run-as:", "not debuggable", "is unknown", "permission denied")
_DATASTORE_DIR = "files/datastore"
_SUFFIX = ".preferences_pb"
_BACKUP_FORMAT = 1
# Callers that have a loaded config should pass ``cache_dir=config.cache.dir`` so concurrent
# workers keep separate restore points; this mirrors ``config.CacheCfg.dir`` for the ones that
# do not (the CLI's own default).
_DEFAULT_CACHE_DIR = "~/.cache/android-ui-analyser"

WIRE_VARINT, WIRE_I64, WIRE_LEN, WIRE_I32 = 0, 1, 2, 5

# Value oneof field number -> the type name this module speaks.
VALUE_TYPES = {
    1: "bool",
    2: "float",
    3: "int",
    4: "long",
    5: "string",
    6: "string_set",
    7: "double",
    8: "bytes",
}
TYPE_FIELDS = {name: field for field, name in VALUE_TYPES.items()}
# DataStore's own Kotlin spellings, accepted so a caller can copy a type straight out of
# `booleanPreferencesKey` / `intPreferencesKey` without translating it first.
_TYPE_ALIASES = {"boolean": "bool", "integer": "int"}

# int32 vs int64 range, used to normalise varints back to signed Java values.
_I64 = 1 << 64
_I63 = 1 << 63
_I32 = 1 << 32
_I31 = 1 << 31

PreferenceMap = dict[str, tuple[str, Any]]


# --------------------------------------------------------------------------- #
# codec primitives
# --------------------------------------------------------------------------- #


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError(f"truncated varint at offset {pos}")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError(f"varint longer than 10 bytes at offset {pos}")


def _write_varint(value: int) -> bytes:
    """Canonical (minimal-length) base-128 varint.

    Negatives are two's complement over 64 bits, exactly as protobuf encodes int32/int64 -- a
    shorter encoding would parse on the host and be rejected by the app's parser.
    """
    if value < 0:
        value += _I64
        if value < 0:
            raise ValueError("integer too negative for a 64-bit varint")
    elif value >= _I64:
        raise ValueError("integer too large for a 64-bit varint")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _write_varint((field << 3) | wire)


def _lendelim(field: int, payload: bytes) -> bytes:
    return _tag(field, WIRE_LEN) + _write_varint(len(payload)) + payload


def _skip(buf: bytes, pos: int, wire: int) -> int:
    """Advance past one field body of the given wire type."""
    if wire == WIRE_VARINT:
        _, pos = _read_varint(buf, pos)
    elif wire == WIRE_I64:
        pos += 8
    elif wire == WIRE_LEN:
        length, pos = _read_varint(buf, pos)
        pos += length
    elif wire == WIRE_I32:
        pos += 4
    else:
        raise ValueError(f"unsupported wire type {wire} at offset {pos}")
    if pos > len(buf):
        raise ValueError("field body runs past the end of the buffer")
    return pos


def _text(raw: bytes) -> str:
    # surrogateescape keeps decode/encode lossless even if a writer emitted a string field that
    # is not valid UTF-8; re-encoding restores the original bytes rather than U+FFFD.
    return raw.decode("utf-8", "surrogateescape")


def _bin(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


def _decode_string_set(buf: bytes) -> list[str]:
    strings: list[str] = []
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            strings.append(_text(buf[pos : pos + length]))
            pos += length
        else:
            pos = _skip(buf, pos, wire)
    return strings


def _decode_value(buf: bytes) -> tuple[str, Any]:
    """Return ``(type, value)`` for one ``Value`` message.

    The last oneof field wins, which is what protobuf itself does when a oneof appears twice.
    """
    found: tuple[str, Any] = ("unset", None)
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        name = VALUE_TYPES.get(field)
        if name is None:
            pos = _skip(buf, pos, wire)
            continue
        if name == "bool" and wire == WIRE_VARINT:
            raw, pos = _read_varint(buf, pos)
            found = ("bool", raw != 0)
        elif name == "float" and wire == WIRE_I32:
            found = ("float", struct.unpack("<f", buf[pos : pos + 4])[0])
            pos += 4
        elif name in ("int", "long") and wire == WIRE_VARINT:
            raw, pos = _read_varint(buf, pos)
            if raw >= _I63:
                raw -= _I64
            if name == "int":
                raw &= _I32 - 1
                if raw >= _I31:
                    raw -= _I32
            found = (name, raw)
        elif name == "string" and wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            found = ("string", _text(buf[pos : pos + length]))
            pos += length
        elif name == "string_set" and wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            found = ("string_set", _decode_string_set(buf[pos : pos + length]))
            pos += length
        elif name == "double" and wire == WIRE_I64:
            found = ("double", struct.unpack("<d", buf[pos : pos + 8])[0])
            pos += 8
        elif name == "bytes" and wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            found = ("bytes", bytes(buf[pos : pos + length]))
            pos += length
        else:
            raise ValueError(f"preference field {field} ({name}) has unexpected wire type {wire}")
    return found


def _decode_entry(buf: bytes) -> tuple[str, tuple[str, Any]]:
    name = ""
    value: tuple[str, Any] = ("unset", None)
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            name = _text(buf[pos : pos + length])
            pos += length
        elif field == 2 and wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            value = _decode_value(buf[pos : pos + length])
            pos += length
        else:
            pos = _skip(buf, pos, wire)
    return name, value


def decode(data: bytes) -> PreferenceMap:
    """Parse a ``.preferences_pb`` payload into ``name -> (type, value)``, in file order."""
    out: PreferenceMap = {}
    pos, end = 0, len(data)
    while pos < end:
        key, pos = _read_varint(data, pos)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == WIRE_LEN:
            length, pos = _read_varint(data, pos)
            if pos + length > end:
                raise ValueError("map entry runs past the end of the file")
            name, value = _decode_entry(data[pos : pos + length])
            out[name] = value
            pos += length
        else:
            raise ValueError(
                f"unexpected top-level field {field} (wire {wire}) at offset {pos}; "
                "refusing to drop data silently"
            )
    return out


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #


def _encode_value(kind: str, value: Any) -> bytes:
    if kind == "unset":
        return b""
    kind = _TYPE_ALIASES.get(kind, kind)
    try:
        field = TYPE_FIELDS[kind]
    except KeyError:
        raise ValueError(f"unknown preference type {kind!r}") from None

    if kind == "bool":
        return _tag(field, WIRE_VARINT) + _write_varint(1 if value else 0)
    if kind == "float":
        return _tag(field, WIRE_I32) + struct.pack("<f", float(value))
    if kind == "double":
        return _tag(field, WIRE_I64) + struct.pack("<d", float(value))
    if kind in ("int", "long"):
        number = int(value)
        if kind == "int" and not (-_I31 <= number < _I31):
            raise ValueError(f"integer {number} is out of int32 range")
        return _tag(field, WIRE_VARINT) + _write_varint(number)
    if kind == "string":
        return _lendelim(field, _bin(value) if isinstance(value, str) else bytes(value))
    if kind == "bytes":
        return _lendelim(field, bytes(value))
    if kind == "string_set":
        items = value if isinstance(value, (list, tuple)) else sorted(value)
        return _lendelim(field, b"".join(_lendelim(1, _bin(item)) for item in items))
    raise ValueError(f"unhandled preference type {kind!r}")


def encode(preferences: Mapping[str, tuple[str, Any]]) -> bytes:
    """Serialise ``name -> (type, value)`` back to a ``.preferences_pb`` payload, in map order."""
    out = bytearray()
    for name, (kind, value) in preferences.items():
        entry = _lendelim(1, _bin(name)) + _lendelim(2, _encode_value(kind, value))
        out += _lendelim(1, bytes(entry))
    return bytes(out)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _validate_package(package: str) -> str:
    value = package.strip()
    if not _PACKAGE_RE.fullmatch(value):
        raise UsageError(
            f"invalid Android package name: {package!r}",
            code="datastore_package_invalid",
        )
    return value


def _validate_datastore(name: str) -> str:
    """Normalise a DataStore reference to the stem Android stores, or refuse it.

    ``preferencesDataStore(name = "settings")`` writes ``settings.preferences_pb``, and callers
    name the file either way, so the suffix is stripped rather than demanded.  A path is refused
    instead of resolved: the write replaces whatever it addresses, and ``../databases/app.db``
    addressed from the datastore directory is a different file in a different format.
    """
    value = str(name).strip()
    if value.endswith(_SUFFIX):
        value = value[: -len(_SUFFIX)]
    if not value or value.startswith(".") or not _DATASTORE_RE.fullmatch(value):
        raise UsageError(
            f"invalid datastore name: {name!r}",
            hint="Pass a name from `aua datastore list <package>`, e.g. `settings` or "
            "`settings.preferences_pb` -- not a path.",
            code="datastore_name_invalid",
        )
    return value


def _validate_backup_id(backup_id: str) -> str:
    value = backup_id.strip()
    if not value or not _BACKUP_RE.fullmatch(value):
        raise UsageError(f"invalid backup id: {backup_id!r}", code="datastore_backup_invalid")
    return value


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def _remote_path(name: str) -> str:
    return f"{_DATASTORE_DIR}/{name}{_SUFFIX}"


# --------------------------------------------------------------------------- #
# device access
# --------------------------------------------------------------------------- #


def _run_as(device: Device, package: str, argv: list[str]) -> str:
    command = f"run-as {quote(package)} " + " ".join(quote(arg) for arg in argv)
    try:
        output = device.shell(command)
    except Exception as exc:
        raise DeviceError(
            f"cannot access {package} app data: {exc}",
            hint="DataStore access requires an installed debuggable build and a connected device.",
            code="datastore_access",
        ) from exc
    for line in output.splitlines():
        lowered = line.strip().lower()
        if lowered and any(marker in lowered for marker in _RUN_AS_ERRORS):
            raise DeviceError(
                f"cannot access {package} app data: {line.strip()}",
                hint="Use a debuggable app build; Android run-as refuses production builds.",
                code="datastore_access",
            )
    return output


def _listing(device: Device, package: str) -> dict[str, int]:
    raw = _run_as(device, package, ["ls", "-la", _DATASTORE_DIR])
    if "no such file or directory" in raw.lower():
        return {}
    files: dict[str, int] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        parts = stripped.split(maxsplit=7)
        if len(parts) < 8:
            continue
        try:
            size = int(parts[4])
        except ValueError:
            continue
        name = parts[7]
        if "/" not in name and name not in (".", ".."):
            files[name] = size
    return files


def _require(device: Device, package: str, name: str) -> None:
    files = _listing(device, package)
    if f"{name}{_SUFFIX}" not in files:
        choices = sorted(item[: -len(_SUFFIX)] for item in files if item.endswith(_SUFFIX))
        available = f" Available: {', '.join(choices)}." if choices else ""
        raise DeviceError(
            f"datastore {name!r} does not exist for {package}.{available}",
            hint=f"Run `aua datastore list {package}` and pass one of its names.",
            code="datastore_not_found",
        )


def _read_payload(device: Device, package: str, name: str) -> bytes:
    _require(device, package, name)
    try:
        return device.read_app_file(package, _remote_path(name))
    except Exception as exc:
        raise DeviceError(
            f"could not read {package}/{_remote_path(name)}: {exc}",
            hint="The app must be debuggable and remain installed while AUA reads it.",
            code="datastore_access",
        ) from exc


def _parse_payload(payload: bytes, package: str, name: str) -> PreferenceMap:
    try:
        return decode(payload)
    except ValueError as exc:
        raise DeviceError(
            f"{package}/{_remote_path(name)} is not a readable DataStore preferences file: {exc}",
            hint="AUA will not overwrite a file it cannot parse; inspect it by hand, or let the "
            "app rewrite it. Note that the app's corruption handler empties such a file on its "
            "next read.",
            code="datastore_corrupt",
        ) from exc


@contextmanager
def _stopped_app(device: Device, package: str, *, restart: bool) -> Iterator[None]:
    """Hold the app stopped for the duration of a mutation.

    Not a precaution: ``SingleProcessCoordinator`` keeps its version counter in the app's own
    process, so a running reader never notices the new file, and a running writer overwrites it.
    """
    try:
        device.stop_app(package)
    except Exception as exc:
        raise DeviceError(
            f"could not stop {package} before writing its datastore: {exc}",
            code="datastore_stop_failed",
        ) from exc
    try:
        yield
    finally:
        if restart:
            try:
                device.launch_app(package)
            except Exception as exc:
                raise DeviceError(
                    f"datastore write finished but {package} could not be relaunched: {exc}",
                    hint=f"Relaunch it explicitly with `aua app launch {package}`.",
                    code="datastore_restart_failed",
                ) from exc


def _state_loss_warning(package: str, *, restarted: bool) -> str:
    """What an agent must know after a call that force-stopped the app to write its datastore.

    ``app_restarted`` alone reads as bookkeeping, not "you are not where you were" -- an agent
    skimming the result can miss it and keep acting on stale navigation state.  Say the
    consequence in words, the way ``app_database`` does after a coherent snapshot.
    """
    if restarted:
        return (
            f"{package} was force-stopped so the new DataStore file would actually be read, and "
            "has been relaunched. Any in-app navigation or UI state from before this call is "
            "gone -- the app is back at its cold-start screen, not wherever you left it. "
            "Re-navigate before continuing; do not assume you are still where you were."
        )
    return (
        f"{package} was force-stopped so the new DataStore file would actually be read, and was "
        "left stopped (--no-restart). Any in-app navigation or UI state from before this call is "
        "gone, and the app is not running at all. Launch it and re-navigate before continuing."
    )


# --------------------------------------------------------------------------- #
# restore points
# --------------------------------------------------------------------------- #


def _backup_root(cache_dir: str | Path | None, serial: str, package: str, name: str) -> Path:
    return (
        Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser()
        / "datastore-backups"
        / _safe_component(serial)
        / _safe_component(package)
        / _safe_component(name)
    )


def _new_backup_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _store_backup(
    *,
    cache_dir: str | Path | None,
    serial: str,
    package: str,
    name: str,
    payload: bytes,
    reason: str,
) -> dict[str, Any]:
    """Persist *payload* host-side as the restore point a later call can replay.

    Host-side and outside the app's own storage, because the file this is protecting is the one
    a bad write empties -- a copy kept beside it would be no copy at all.
    """
    backup_id = _new_backup_id()
    path = _backup_root(cache_dir, serial, package, name) / backup_id
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    file_name = f"{name}{_SUFFIX}"
    payload_path = path / file_name
    payload_path.write_bytes(payload)
    payload_path.chmod(0o600)
    metadata = {
        "format": _BACKUP_FORMAT,
        "id": backup_id,
        "serial": serial,
        "package": package,
        "datastore": name,
        "created_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "file": file_name,
        "bytes": len(payload),
    }
    metadata_path = path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path.chmod(0o600)
    return {**metadata, "path": str(path)}


def _load_backup(
    cache_dir: str | Path | None,
    serial: str,
    package: str,
    name: str,
    backup_id: str,
) -> tuple[dict[str, Any], bytes]:
    path = _backup_root(cache_dir, serial, package, name) / _validate_backup_id(backup_id)
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(
            f"datastore backup {backup_id!r} was not found or is invalid",
            hint=f"List available restore points with `aua datastore backups {package} {name}`.",
            code="datastore_backup_not_found",
        ) from exc
    expected = (serial, package, name)
    actual = (metadata.get("serial"), metadata.get("package"), metadata.get("datastore"))
    if metadata.get("format") != _BACKUP_FORMAT or actual != expected:
        raise UsageError(
            f"datastore backup {backup_id!r} does not belong to this device/package/datastore",
            code="datastore_backup_mismatch",
        )
    file_name = metadata.get("file")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise UsageError(f"datastore backup {backup_id!r} has unsafe file metadata")
    try:
        payload = (path / file_name).read_bytes()
    except OSError as exc:
        raise UsageError(f"datastore backup {backup_id!r} is incomplete") from exc
    return {**metadata, "path": str(path)}, payload


# --------------------------------------------------------------------------- #
# value marshalling
# --------------------------------------------------------------------------- #


def _json_value(kind: str, value: Any) -> Any:
    """The JSON-transportable shape of one decoded value.

    ``bytes`` becomes base64 and is accepted back in that form, so a caller can read a value and
    write it somewhere else without a second encoding convention to get wrong.
    """
    if kind == "bytes":
        return base64.b64encode(value).decode("ascii")
    if kind == "string_set":
        return list(value)
    return value


def _type_error(message: str, *, hint: str | None = None) -> DeviceError:
    return DeviceError(message, hint=hint, code="datastore_type_invalid")


def _coerce(key: str, kind: str, value: Any) -> tuple[str, Any]:
    """Validate one requested value against its declared DataStore type.

    The type is the caller's, not a guess from the Python value: the app reads a key with
    ``booleanPreferencesKey`` or ``intPreferencesKey``, and a value stored under the wrong oneof
    field is not a wrong setting but a ``ClassCastException`` in the reader.
    """
    canonical = _TYPE_ALIASES.get(kind, kind)
    if canonical not in TYPE_FIELDS:
        raise _type_error(
            f"unknown datastore type {kind!r} for key {key!r}",
            hint=f"One of: {', '.join(TYPE_FIELDS)}.",
        )
    if canonical == "bool":
        if not isinstance(value, bool):
            raise _type_error(f"datastore key {key!r} is a bool; got {type(value).__name__}")
        return canonical, value
    if canonical in ("int", "long"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(f"datastore key {key!r} is an {canonical}; got {type(value).__name__}")
        limit = _I31 if canonical == "int" else _I63
        if not -limit <= value < limit:
            raise _type_error(f"datastore {canonical} {value} for {key!r} is out of range")
        return canonical, value
    if canonical in ("float", "double"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(f"datastore key {key!r} is a {canonical}; got {type(value).__name__}")
        if not math.isfinite(value):
            raise _type_error(f"datastore {canonical} for {key!r} must be finite")
        return canonical, float(value)
    if canonical == "string":
        if not isinstance(value, str):
            raise _type_error(f"datastore key {key!r} is a string; got {type(value).__name__}")
        return canonical, value
    if canonical == "string_set":
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)):
            raise _type_error(f"datastore key {key!r} is a string_set; pass an array of strings")
        items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        if any(not isinstance(item, str) for item in items):
            raise _type_error(f"datastore string_set {key!r} may only contain strings")
        return canonical, items
    if isinstance(value, (bytes, bytearray)):
        return canonical, bytes(value)
    if isinstance(value, str):
        try:
            # binascii.Error, which b64decode raises on bad padding/alphabet, is a ValueError.
            return canonical, base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise _type_error(f"datastore bytes for {key!r} must be base64: {exc}") from exc
    raise _type_error(f"datastore key {key!r} is bytes; pass raw bytes or a base64 string")


def _coerce_values(values: Mapping[str, Any]) -> dict[str, tuple[str | None, Any]]:
    """Normalise the requested writes, deferring any value whose type the caller left out.

    A bare ``{"user_theme_mode": 1}`` is the natural way to change a setting the app already
    stores, and the alternative -- making the caller name the type -- is the more dangerous
    one: ``int`` and ``long`` are different oneof fields, so a caller who guesses wrong turns a
    setting change into a ``ClassCastException`` in the app's own reader.  The store already
    knows the answer, so a bare value is carried through as ``(None, value)`` and typed in
    :func:`_apply_requested` from the entry it is replacing.  A bare value for a key the app has
    never written has nothing to learn from, and is refused there rather than guessed.
    """
    if not isinstance(values, Mapping) or not values:
        raise UsageError(
            "datastore set needs at least one key to write",
            hint='Pass a mapping of key -> value, or key -> {"type": ..., "value": ...}.',
            code="datastore_values_invalid",
        )
    requested: dict[str, tuple[str | None, Any]] = {}
    for key, spec in values.items():
        if not isinstance(key, str) or not key.strip():
            raise UsageError(
                "a datastore key must be a non-empty string",
                code="datastore_key_invalid",
            )
        if isinstance(spec, Mapping):
            if "type" not in spec or "value" not in spec:
                raise _type_error(
                    f"datastore key {key!r} was given a mapping without both 'type' and "
                    "'value'; pass a bare value, or both fields",
                )
            requested[key] = _coerce(key, str(spec["type"]), spec["value"])
        else:
            requested[key] = (None, spec)
    return requested


def _apply_requested(
    preferences: PreferenceMap,
    requested: Mapping[str, tuple[str | None, Any]],
) -> PreferenceMap:
    """Merge *requested* into *preferences*, typing bare values from the entry they replace."""
    applied: PreferenceMap = {}
    for key, (kind, value) in requested.items():
        if kind is None:
            existing = preferences.get(key)
            if existing is None:
                raise _type_error(
                    f"datastore key {key!r} does not exist yet, so its type cannot be inferred",
                    hint='Pass {"type": "bool|int|long|float|double|string|string_set|bytes", '
                    '"value": ...} for a new key.',
                )
            kind, value = _coerce(key, existing[0], value)
        applied[key] = (kind, value)
    preferences.update(applied)
    return applied


def _encode_checked(preferences: PreferenceMap, package: str, name: str) -> bytes:
    """Serialise *preferences* and prove the result parses back to the same map.

    This is the guard behind rule 2 in the module docstring: an app whose corruption handler is
    ``emptyPreferences()`` answers a malformed file by deleting every key without crashing, so
    "the encoder looked right" is not evidence.  Round-tripping through :func:`decode` costs
    microseconds and is the only check that would actually have caught it.
    """
    try:
        payload = encode(preferences)
    except (ValueError, TypeError) as exc:
        raise _type_error(f"cannot encode the requested datastore values: {exc}") from exc
    reparsed = decode(payload)
    same_keys = list(reparsed) == list(preferences)
    same_types = [kind for kind, _ in reparsed.values()] == [
        kind for kind, _ in preferences.values()
    ]
    if not (same_keys and same_types and encode(reparsed) == payload):
        raise DeviceError(
            f"refusing to write {package}/{_remote_path(name)}: the encoded file does not read "
            "back as the values that were requested",
            hint="This is a codec bug, not a device problem; nothing was written.",
            code="datastore_corrupt",
        )
    return payload


# --------------------------------------------------------------------------- #
# public surface
# --------------------------------------------------------------------------- #


def list_datastores(device: Device, package: str) -> dict[str, Any]:
    """Every ``*.preferences_pb`` file the app owns, by the name its Kotlin code uses."""
    package = _validate_package(package)
    files = _listing(device, package)
    datastores = [
        {
            "name": file_name[: -len(_SUFFIX)],
            "path": f"{_DATASTORE_DIR}/{file_name}",
            "bytes": size,
        }
        for file_name, size in sorted(files.items())
        if file_name.endswith(_SUFFIX)
    ]
    return {
        "ok": True,
        "action": "datastore-list",
        "package": package,
        "datastores": datastores,
        "count": len(datastores),
    }


def get_datastore(
    device: Device,
    package: str,
    name: str,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read one DataStore file, optionally narrowed to *keys*; unknown keys are simply absent.

    This deliberately does not stop the app.  DataStore installs its new file by rename, so a
    concurrent read returns one whole version or another, never a half-written one -- and the
    worst case is a value the app has since changed, which costs a re-read rather than a lost
    write.  Stopping the app to read would throw away the caller's navigation state for nothing.
    """
    package = _validate_package(package)
    name = _validate_datastore(name)
    preferences = _parse_payload(_read_payload(device, package, name), package, name)
    wanted = None if keys is None else [str(key) for key in keys]
    values = {
        key: {"type": kind, "value": _json_value(kind, value)}
        for key, (kind, value) in preferences.items()
        if wanted is None or key in wanted
    }
    return {
        "ok": True,
        "action": "datastore-get",
        "package": package,
        "datastore": name,
        "values": values,
        "count": len(values),
    }


def set_datastore(
    device: Device,
    package: str,
    name: str,
    values: Mapping[str, Any],
    *,
    confirmed: bool = False,
    restart: bool = True,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Merge *values* into one DataStore file: stop the app, back up, write, relaunch.

    Every step is load-bearing (see the module docstring): the app is force-stopped because a
    live ``SingleProcessCoordinator`` would ignore or overwrite the new file; a restore point is
    taken first because an unparseable file makes the app's corruption handler delete every key,
    session included; and the new payload is re-parsed before it is installed so the only bytes
    that ever reach the device are bytes this codec can read back.  Keys absent from *values* are
    preserved, in their original order.  The file must already exist -- AUA edits a store the app
    wrote, it does not invent one.

    A value may be given bare (``{"user_theme_mode": 1}``) for a key the store already holds, in
    which case its type is read from the entry being replaced; a new key must name its type.
    """
    package = _validate_package(package)
    name = _validate_datastore(name)
    if not confirmed:
        raise DeviceError(
            "writing a datastore mutates app data and requires explicit confirmation",
            hint="Review the values, then pass `--yes` (MCP: `confirmed: true`).",
            code="datastore_confirmation_required",
        )
    requested = _coerce_values(values)
    with _stopped_app(device, package, restart=restart):
        original = _read_payload(device, package, name)
        preferences = _parse_payload(original, package, name)
        applied = _apply_requested(preferences, requested)
        payload = _encode_checked(preferences, package, name)
        backup = _store_backup(
            cache_dir=cache_dir,
            serial=device.serial,
            package=package,
            name=name,
            payload=original,
            reason="before-set",
        )
        try:
            device.write_app_file(package, _remote_path(name), payload)
        except Exception as exc:
            with contextlib.suppress(Exception):
                device.write_app_file(package, _remote_path(name), original)
            raise DeviceError(
                f"failed to install the new datastore file for {package}: {exc}",
                hint=f"Restore point {backup['id']} remains available with `aua datastore restore`.",
                code="datastore_access",
            ) from exc
    return {
        "ok": True,
        "action": "datastore-set",
        "package": package,
        "datastore": name,
        "applied": {
            key: {"type": kind, "value": _json_value(kind, value)}
            for key, (kind, value) in applied.items()
        },
        "backup_id": backup["id"],
        "backup": backup,
        "app_restarted": restart,
        "warning": _state_loss_warning(package, restarted=restart),
    }


def backup_datastore(
    device: Device,
    package: str,
    name: str,
    *,
    reason: str = "manual",
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Copy one DataStore file to a host-side restore point, leaving the app running.

    Unlike a database snapshot this needs no force-stop: the file is replaced by rename, so the
    copy is some complete version of the store, and a backup that costs the caller its navigation
    state is a backup nobody takes.
    """
    package = _validate_package(package)
    name = _validate_datastore(name)
    payload = _read_payload(device, package, name)
    backup = _store_backup(
        cache_dir=cache_dir,
        serial=device.serial,
        package=package,
        name=name,
        payload=payload,
        reason=reason,
    )
    return {
        "ok": True,
        "action": "datastore-backup",
        "package": package,
        "datastore": name,
        "backup_id": backup["id"],
        "backup": backup,
    }


def list_backups(
    device: Device,
    package: str,
    name: str,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Restore points recorded for this device/package/datastore, newest first."""
    package = _validate_package(package)
    name = _validate_datastore(name)
    root = _backup_root(cache_dir, device.serial, package, name)
    backups: list[dict[str, Any]] = []
    if root.is_dir():
        for metadata_path in sorted(root.glob("*/metadata.json"), reverse=True):
            with contextlib.suppress(OSError, json.JSONDecodeError):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("format") == _BACKUP_FORMAT
                    and metadata.get("serial") == device.serial
                    and metadata.get("package") == package
                    and metadata.get("datastore") == name
                ):
                    backups.append({**metadata, "path": str(metadata_path.parent)})
    return {
        "ok": True,
        "action": "datastore-backups",
        "package": package,
        "datastore": name,
        "backups": backups,
        "count": len(backups),
    }


def restore_datastore(
    device: Device,
    package: str,
    name: str,
    backup_id: str,
    *,
    confirmed: bool = False,
    restart: bool = True,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Put a recorded restore point back, keeping the state it replaces as a safety backup.

    Force-stops for the same reason :func:`set_datastore` does, and re-parses the stored payload
    *before* stopping anything: a restore point that no longer decodes would empty the store on
    the app's next read, and refusing early costs the caller nothing.
    """
    package = _validate_package(package)
    name = _validate_datastore(name)
    if not confirmed:
        raise DeviceError(
            "restoring a datastore replaces app data and requires explicit confirmation",
            hint="Review the backup id, then pass `--yes` (MCP: `confirmed: true`).",
            code="datastore_confirmation_required",
        )
    restored, payload = _load_backup(cache_dir, device.serial, package, name, backup_id)
    _parse_payload(payload, package, name)
    with _stopped_app(device, package, restart=restart):
        current = _read_payload(device, package, name)
        safety_backup = _store_backup(
            cache_dir=cache_dir,
            serial=device.serial,
            package=package,
            name=name,
            payload=current,
            reason=f"before-restore-{restored['id']}",
        )
        try:
            device.write_app_file(package, _remote_path(name), payload)
        except Exception as exc:
            with contextlib.suppress(Exception):
                device.write_app_file(package, _remote_path(name), current)
            raise DeviceError(
                f"failed to restore datastore backup {restored['id']}: {exc}",
                hint=f"The pre-restore safety backup is {safety_backup['id']}.",
                code="datastore_access",
            ) from exc
    return {
        "ok": True,
        "action": "datastore-restore",
        "package": package,
        "datastore": name,
        "restored_backup": restored,
        "safety_backup": safety_backup,
        "backup_id": safety_backup["id"],
        "app_restarted": restart,
        "warning": _state_loss_warning(package, restarted=restart),
    }


__all__ = [
    "backup_datastore",
    "decode",
    "encode",
    "get_datastore",
    "list_backups",
    "list_datastores",
    "restore_datastore",
    "set_datastore",
]
