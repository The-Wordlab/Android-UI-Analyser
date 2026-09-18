"""UserDefaults/plist preferences and configured feature-flag deeplinks on simulators."""

from __future__ import annotations

import base64
import json
import math
import plistlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from ..atomic import atomic_write_text
from ..errors import DeviceError, UsageError
from ..flag_values import (
    ContextPrefsRead,
    PrefsRead,
    build_uri,
    dump_result,
    load_flags_file,
    parse_assignments,
)
from .identity import TargetRef
from .ios_files import IOSAppFiles
from .ios_tools import IOSTools
from .runtime import TargetRuntime


class PreferencesSnapshot(NamedTuple):
    package: str
    file: str
    existed: bool
    data: bytes | None


def _name(name: str) -> str:
    value = name if name.endswith(".plist") else f"{name}.plist"
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.plist", value):
        raise UsageError("preferences need a plist basename, usually the app bundle id")
    return value


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class IOSPreferences:
    build_uri = staticmethod(build_uri)
    dump_result = staticmethod(dump_result)
    load_flags_file = staticmethod(load_flags_file)
    parse_assignments = staticmethod(parse_assignments)

    def __init__(self, tools: IOSTools) -> None:
        self.tools = tools

    def _read(self, device: TargetRuntime, package: str, name: str) -> dict[str, Any]:
        path = IOSAppFiles(self.tools, device.target_id).path(
            package, f"Library/Preferences/{_name(name)}"
        )
        # UserDefaults can be committed to cfprefsd before its backing file is flushed.
        # Read the same preference service used for imports, not a stale on-disk snapshot.
        result = self.tools.simctl(
            "spawn",
            device.target_id,
            "defaults",
            "export",
            str(path.with_suffix("")),
            "-",
            check=False,
        )
        if not result.ok:
            if not path.exists():
                return {}
            raise self.tools.error(result, "defaults export")
        try:
            value = plistlib.loads(result.stdout)
        except plistlib.InvalidFileException as exc:
            raise DeviceError(
                "preferences are not a readable plist", code="prefs_unreadable"
            ) from exc
        if not isinstance(value, dict):
            raise DeviceError("preference plist is not a dictionary", code="prefs_unreadable")
        return value

    def read_prefs(
        self,
        device: TargetRuntime,
        package: str,
        pairs: dict[str, str],
        *,
        prefs_file: str | None = None,
    ) -> PrefsRead:
        name = _name(prefs_file or package)
        values = self._read(device, package, name)
        applied = {
            key: want for key, want in pairs.items() if key in values and _text(values[key]) == want
        }
        ignored = [key for key in pairs if key not in values]
        mismatched = {
            key: _text(values[key]) for key in pairs if key in values and key not in applied
        }
        return PrefsRead(applied, ignored, mismatched, [name], None)

    def read_context_flags(
        self,
        device: TargetRuntime,
        package: str,
        *,
        prefs_file: str | None = None,
        keys: list[str] | None = None,
        key_patterns: list[str] | None = None,
    ) -> ContextPrefsRead:
        name = _name(prefs_file or package)
        if not keys and not key_patterns:
            return ContextPrefsRead({}, [], None)
        values = self._read(device, package, name)
        patterns = [re.compile(pattern) for pattern in key_patterns or ()]
        selected = {
            key: _text(value)
            for key, value in values.items()
            if (key in (keys or ()) or any(pattern.search(key) for pattern in patterns))
            and isinstance(value, (bool, int, float, str))
        }
        return ContextPrefsRead(selected, [name], None)

    def snapshot_prefs(self, device: TargetRuntime, package: str, file: str) -> PreferencesSnapshot:
        name = _name(file)
        path = IOSAppFiles(self.tools, device.target_id).path(
            package, f"Library/Preferences/{name}"
        )
        device.stop_app(package)
        values = self._read(device, package, name)
        existed = path.exists() or bool(values)
        return PreferencesSnapshot(
            package, name, existed, plistlib.dumps(values) if existed else None
        )

    def save_prefs_backup(
        self, cache_dir: str | Path, serial: str, snapshot: PreferencesSnapshot
    ) -> Path:
        path = (
            Path(cache_dir).expanduser()
            / "prefs"
            / TargetRef("ios", serial).storage_key
            / f"{snapshot.package}-{snapshot.file}.json"
        )
        atomic_write_text(
            path,
            json.dumps(
                {
                    "package": snapshot.package,
                    "file": snapshot.file,
                    "existed": snapshot.existed,
                    "data": base64.b64encode(snapshot.data).decode() if snapshot.data else None,
                }
            ),
        )
        path.chmod(0o600)
        return path

    def _import(self, device: TargetRuntime, package: str, file: str, data: bytes) -> None:
        files = IOSAppFiles(self.tools, device.target_id)
        path = files.path(package, f"Library/Preferences/{_name(file)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # defaults talks to cfprefsd; replacing the file alone leaves its cached values live.
        self.tools.simctl(
            "spawn",
            device.target_id,
            "defaults",
            "import",
            str(path.with_suffix("")),
            "-",
            input_bytes=data,
        )

    def write_prefs(
        self,
        device: TargetRuntime,
        snapshot: PreferencesSnapshot,
        values: Mapping[str, Any],
        *,
        relaunch: bool = True,
    ) -> dict[str, Any]:
        if not values:
            raise UsageError("a prefs write needs at least one value")
        for key, value in values.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(value, (str, bool, int, float))
            ):
                raise UsageError(
                    "preference keys are strings; values are strings, booleans or numbers"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise UsageError("preference numbers must be finite")
        original = plistlib.loads(snapshot.data) if snapshot.data else {}
        if not isinstance(original, dict):
            raise UsageError("existing preference plist is not a dictionary")
        merged = {**original, **values}
        self._import(device, snapshot.package, snapshot.file, plistlib.dumps(merged))
        readback = self._read(device, snapshot.package, snapshot.file)
        verified = all(
            type(readback.get(k)) is type(v) and readback.get(k) == v for k, v in values.items()
        )
        if relaunch:
            device.launch_app(snapshot.package)
        return {
            "ok": verified,
            "action": "prefs-write",
            "package": snapshot.package,
            "file": snapshot.file,
            "written": dict(values),
            "verified": verified,
            "relaunched": relaunch,
        }

    def restore_prefs(self, device: TargetRuntime, backup_path: str | Path) -> str:
        raw = json.loads(Path(backup_path).read_text())
        package, name = str(raw["package"]), _name(str(raw["file"]))
        device.stop_app(package)
        if raw["existed"]:
            original = base64.b64decode(raw["data"], validate=True)
            self._import(device, package, name, original)
            if self._read(device, package, name) != plistlib.loads(original):
                raise DeviceError(
                    "preference restore readback did not match", code="prefs_restore_failed"
                )
        else:
            self._import(device, package, name, plistlib.dumps({}))
            IOSAppFiles(self.tools, device.target_id).remove(
                package, [f"Library/Preferences/{name}"]
            )
        return f"{package}/{name} restored"
