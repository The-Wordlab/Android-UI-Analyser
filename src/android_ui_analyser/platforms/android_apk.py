"""Android-only APK bundle inspection and installation.

Reachable **only** through :class:`~android_ui_analyser.platforms.android.AndroidPlatform`
so the generic engine never learns that "install an app bundle" means ``adb install``.

Two host-side jobs live here:

* :func:`inspect_bundle` reads ``package`` / ``versionName`` / ``versionCode`` straight out of
  an APK's binary ``AndroidManifest.xml``.  Doing it in-process rather than shelling out to
  ``aapt2 dump badging`` matters: build-tools are absent from plenty of otherwise working SDK
  installs, and an install that cannot name its own package cannot answer "is this already on
  the device?" — which is the whole point of the idempotent path.
* :func:`install_bundle` / :func:`installed_app` / :func:`uninstall` wrap the adb package
  manager, translating its famously unstructured stdout into typed failures.

``adb install`` reports failure in three different ways depending on platform-tools version:
a non-zero exit, or exit 0 with ``Failure [INSTALL_FAILED_…]`` on stdout, or exit 0 with the
same text on stderr.  Trusting the exit code alone silently reports success for an install
that never landed, so every path here parses the output too.
"""

from __future__ import annotations

import re
import struct
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import DeviceError, UsageError

# -- binary AndroidManifest.xml (AXML) --------------------------------------------------

_RES_STRING_POOL_TYPE = 0x0001
_RES_XML_START_ELEMENT_TYPE = 0x0102
_UTF8_FLAG = 1 << 8
_TYPE_STRING = 0x03
_TYPE_INT_DEC = 0x10
_TYPE_INT_HEX = 0x11
_NO_ENTRY = 0xFFFFFFFF


@dataclass(frozen=True)
class BundleInfo:
    """Identity read out of an app bundle on the host, before any device is involved."""

    package: str
    version_name: str | None = None
    version_code: str | None = None


def _pool_strings(blob: bytes, offset: int) -> list[str]:
    """Decode a ``ResStringPool`` chunk into its ordered string list."""

    chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", blob, offset)
    if chunk_type != _RES_STRING_POOL_TYPE:  # pragma: no cover - defensive
        raise ValueError("expected a string pool chunk")
    count, _styles, flags, strings_start = struct.unpack_from("<IIII", blob, offset + 8)
    utf8 = bool(flags & _UTF8_FLAG)
    offsets = struct.unpack_from(f"<{count}I", blob, offset + header_size)
    base = offset + strings_start
    limit = offset + chunk_size
    out: list[str] = []
    for entry in offsets:
        at = base + entry
        if not 0 <= at < limit:  # pragma: no cover - defensive
            out.append("")
            continue
        if utf8:
            # Two length prefixes (UTF-16 length, then byte length); either may use the
            # two-byte high-bit encoding, so both have to be stepped over independently.
            at += 2 if blob[at] & 0x80 else 1
            if blob[at] & 0x80:
                length = ((blob[at] & 0x7F) << 8) | blob[at + 1]
                at += 2
            else:
                length = blob[at]
                at += 1
            out.append(blob[at : at + length].decode("utf-8", errors="replace"))
        else:
            length = struct.unpack_from("<H", blob, at)[0]
            at += 2
            if length & 0x8000:
                length = ((length & 0x7FFF) << 16) | struct.unpack_from("<H", blob, at)[0]
                at += 2
            out.append(blob[at : at + length * 2].decode("utf-16-le", errors="replace"))
    return out


def _manifest_attributes(blob: bytes) -> dict[str, str]:
    """Return the ``<manifest>`` element's attributes as plain ``name -> value`` strings.

    Only the first ``START_ELEMENT`` named ``manifest`` is read; that is where ``package`` and
    the two version attributes live, and stopping there keeps this parser far smaller than a
    general AXML reader.
    """

    if len(blob) < 8:
        raise ValueError("AndroidManifest.xml is too short to be a resource chunk")
    strings: list[str] = []
    offset = 8  # skip the RES_XML_TYPE file header
    total = len(blob)
    while offset + 8 <= total:
        chunk_type, _header_size, chunk_size = struct.unpack_from("<HHI", blob, offset)
        if chunk_size <= 0 or offset + chunk_size > total:
            break
        if chunk_type == _RES_STRING_POOL_TYPE:
            strings = _pool_strings(blob, offset)
        elif chunk_type == _RES_XML_START_ELEMENT_TYPE and strings:
            name_index = struct.unpack_from("<I", blob, offset + 20)[0]
            name = strings[name_index] if name_index < len(strings) else ""
            if name == "manifest":
                return _element_attributes(blob, offset, strings)
        offset += chunk_size
    raise ValueError("no <manifest> element found in AndroidManifest.xml")


def _element_attributes(blob: bytes, offset: int, strings: list[str]) -> dict[str, str]:
    attribute_start, attribute_size, attribute_count = struct.unpack_from("<HHH", blob, offset + 24)
    attributes: dict[str, str] = {}
    # ``attributeStart`` counts from the start of the attrExt struct, which itself follows the
    # 16-byte node header (chunk header + lineNumber + comment) — not from the chunk start.
    for index in range(attribute_count):
        at = offset + 16 + attribute_start + index * attribute_size
        _ns, name_index, raw_index, _size, _res0, data_type, data = struct.unpack_from(
            "<IIIHBBI", blob, at
        )
        name = strings[name_index] if name_index < len(strings) else ""
        if not name:
            continue
        if raw_index != _NO_ENTRY and raw_index < len(strings):
            attributes[name] = strings[raw_index]
        elif data_type == _TYPE_STRING and data < len(strings):
            attributes[name] = strings[data]
        elif data_type in (_TYPE_INT_DEC, _TYPE_INT_HEX):
            attributes[name] = str(data)
    return attributes


def inspect_bundle(bundle: Path) -> BundleInfo:
    """Read package identity out of an APK without touching a device."""

    path = Path(bundle).expanduser()
    if not path.exists():
        raise UsageError(
            f"no app bundle at {path}",
            hint="Build it first (e.g. `./gradlew :app:assembleDebug`) and pass the .apk path.",
        )
    if path.is_dir():
        raise UsageError(
            f"{path} is a directory, not an app bundle",
            hint="Pass the .apk file itself, e.g. app/build/outputs/apk/debug/app-debug.apk.",
        )
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("AndroidManifest.xml")
    except KeyError as exc:
        raise UsageError(
            f"{path} has no AndroidManifest.xml — it is not an APK",
            hint="An .aab app bundle cannot be installed directly; build a debug .apk instead.",
        ) from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise UsageError(f"could not read {path} as an APK: {exc}") from exc
    try:
        attributes = _manifest_attributes(raw)
    except (ValueError, struct.error, IndexError) as exc:
        raise UsageError(
            f"could not read the package id out of {path}: {exc}",
            hint="Pass --package to name it explicitly.",
        ) from exc
    package = attributes.get("package", "").strip()
    if not package:
        raise UsageError(
            f"{path} declares no package id",
            hint="Pass --package to name it explicitly.",
        )
    return BundleInfo(
        package=package,
        version_name=attributes.get("versionName") or None,
        version_code=attributes.get("versionCode") or None,
    )


# -- device-side package manager -------------------------------------------------------


@dataclass(frozen=True)
class InstalledApp:
    """What the device's package manager reports about one package."""

    package: str
    installed: bool
    version_name: str | None = None
    version_code: str | None = None


_VERSION_NAME_RE = re.compile(r"versionName=(\S+)")
_VERSION_CODE_RE = re.compile(r"versionCode=(\d+)")
_FAILURE_RE = re.compile(r"(?:Failure|Error)\s*\[?([A-Z_]*INSTALL_[A-Z_]+|[A-Z_]{4,})\]?")


def _adb(serial: str, *args: str, timeout_s: float) -> subprocess.CompletedProcess[str]:
    # `adb_bin()` rather than the bare literal: it falls back to $ANDROID_HOME/platform-tools,
    # which is exactly the "adb is installed but not on PATH" case the surrounding tooling
    # already normalises. The bare-"adb" calls elsewhere in this codebase are migration debt.
    from ..emulator import adb_bin

    try:
        return subprocess.run(  # noqa: S603
            [adb_bin(), "-s", serial, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise DeviceError(
            "adb is not on PATH",
            code="adb_missing",
            hint="Install Android SDK platform-tools, then re-run `aua doctor`.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DeviceError(
            f"adb {' '.join(args[:1])} timed out after {timeout_s:.0f}s on {serial}",
            code="adb_timeout",
            hint="Check the device is responsive (`aua doctor`); a cold emulator may need longer "
            "— raise --install-timeout.",
        ) from exc


def installed_app(serial: str, package: str, *, timeout_s: float = 30.0) -> InstalledApp:
    """Ask the device whether *package* is installed, and at which version.

    Presence comes from ``pm path``, not from the exit code and not from ``dumpsys``: ``pm path``
    answers with an APK path or nothing at all, which is the only unambiguous signal the package
    manager offers. ``dumpsys package`` is then read only to *describe* a package already known
    to be there.
    """

    listed = _adb(serial, "shell", "pm", "path", package, timeout_s=timeout_s)
    if not any(line.strip().startswith("package:") for line in (listed.stdout or "").splitlines()):
        return InstalledApp(package=package, installed=False)
    dumped = _adb(serial, "shell", "dumpsys", "package", package, timeout_s=timeout_s)
    text = dumped.stdout or ""
    name = _VERSION_NAME_RE.search(text)
    code = _VERSION_CODE_RE.search(text)
    return InstalledApp(
        package=package,
        installed=True,
        version_name=name.group(1) if name else None,
        version_code=code.group(1) if code else None,
    )


def uninstall(serial: str, package: str, *, timeout_s: float = 120.0) -> None:
    """Remove *package* and its data. Absent packages are not an error."""

    done = _adb(serial, "uninstall", package, timeout_s=timeout_s)
    blob = f"{done.stdout}\n{done.stderr}"
    if done.returncode == 0 and "Failure" not in blob:
        return
    if "DELETE_FAILED_INTERNAL_ERROR" in blob or "Unknown package" in blob:
        return  # not installed — the requested end state already holds
    raise DeviceError(
        f"could not uninstall {package}: {blob.strip() or 'adb uninstall failed'}",
        code="uninstall_failed",
        hint="A device-admin or system package cannot be removed; install without --fresh.",
    )


def install_bundle(
    serial: str,
    bundle: Path,
    *,
    reinstall: bool = True,
    grant_permissions: bool = False,
    allow_test: bool = True,
    allow_downgrade: bool = True,
    timeout_s: float = 300.0,
) -> None:
    """Push an APK onto the device, raising a typed error on any failure."""

    flags: list[str] = []
    if reinstall:
        flags.append("-r")
    if allow_test:
        # Debug builds routinely carry android:testOnly, and pm refuses those without -t. The
        # bundles driven through here are debug builds by definition, so this is on by default.
        flags.append("-t")
    if allow_downgrade:
        # The caller named a bundle; installing it is the request. Refusing because its version
        # is lower than what happens to be on the device would block the ordinary "check the
        # previous build" case, and the answer would be to run adb by hand anyway.
        flags.append("-d")
    if grant_permissions:
        flags.append("-g")
    done = _adb(serial, "install", *flags, str(bundle), timeout_s=timeout_s)
    blob = f"{done.stdout}\n{done.stderr}"
    # Only an explicit `Failure` or a non-zero exit is a failure here. Treating any output
    # containing "Error" as one misreads a successful install that happened to print a warning,
    # and it is not needed: the engine re-reads the package manager afterwards, so an install that
    # silently did not land is caught there rather than by guessing at adb's prose.
    if done.returncode == 0 and "Failure" not in blob:
        return
    match = _FAILURE_RE.search(blob)
    reason = match.group(1) if match else ""
    hint = (
        "Reinstall over a different signing key is impossible; pass --fresh to uninstall "
        "first (this wipes app data)."
        if reason == "INSTALL_FAILED_UPDATE_INCOMPATIBLE"
        else "Check the APK's ABI/minSdk against the target, then retry."
        if reason
        else "Run the same install by hand to see the package manager's own message."
    )
    raise DeviceError(
        f"installing {Path(bundle).name} on {serial} failed"
        + (f": {reason}" if reason else f": {blob.strip() or 'adb install failed'}"),
        code="install_failed",
        hint=hint,
    )


__all__ = [
    "BundleInfo",
    "InstalledApp",
    "install_bundle",
    "inspect_bundle",
    "installed_app",
    "uninstall",
]
