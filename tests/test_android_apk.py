"""Android-only APK inspection and package-manager wrapping.

The binary-manifest reader is tested against manifests this module builds itself, in both string
pool encodings, because that parser is what lets ``install --if-needed`` answer "is this build
already there?" without pushing the bundle.
"""

from __future__ import annotations

import struct
import subprocess
import zipfile
from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError, UsageError
from android_ui_analyser.platforms import android_apk

# --------------------------------------------------------------- binary manifest fixtures

_NO_ENTRY = 0xFFFFFFFF


def _string_pool(strings: list[str], *, utf8: bool) -> bytes:
    offsets: list[int] = []
    data = b""
    for value in strings:
        offsets.append(len(data))
        if utf8:
            encoded = value.encode("utf-8")
            data += bytes([len(value), len(encoded)]) + encoded + b"\x00"
        else:
            data += struct.pack("<H", len(value)) + value.encode("utf-16-le") + b"\x00\x00"
    data += b"\x00" * (-len(data) % 4)
    header_size = 28
    strings_start = header_size + 4 * len(strings)
    return (
        struct.pack(
            "<HHIIIIII",
            0x0001,
            header_size,
            strings_start + len(data),
            len(strings),
            0,
            0x100 if utf8 else 0,
            strings_start,
            0,
        )
        + struct.pack(f"<{len(strings)}I", *offsets)
        + data
    )


def _start_element(name_index: int, attributes: list[tuple[int, int, int, int]]) -> bytes:
    """One START_ELEMENT chunk; each attribute is (name_index, raw_index, data_type, data)."""

    body = b""
    for name, raw, data_type, data in attributes:
        body += struct.pack("<IIIHBBI", _NO_ENTRY, name, raw, 8, 0, data_type, data)
    return (
        struct.pack("<HHI", 0x0102, 16, 36 + len(body))
        + struct.pack("<II", 1, _NO_ENTRY)
        + struct.pack("<II", _NO_ENTRY, name_index)
        + struct.pack("<HHHHHH", 20, 20, len(attributes), 0, 0, 0)
        + body
    )


def build_manifest(
    *, package: str = "com.example.app", version_name: str = "2.1.0", version_code: int = 7, utf8: bool
) -> bytes:
    strings = ["package", "versionCode", "versionName", "manifest", package, version_name]
    pool = _string_pool(strings, utf8=utf8)
    element = _start_element(
        3,
        [
            (0, 4, 0x03, 4),  # package="com.example.app" (raw string)
            (1, _NO_ENTRY, 0x10, version_code),  # versionCode=7 (typed int, no raw value)
            (2, 5, 0x03, 5),  # versionName="2.1.0"
        ],
    )
    body = pool + element
    return struct.pack("<HHI", 0x0003, 8, 8 + len(body)) + body


def write_apk(path: Path, manifest: bytes | None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        if manifest is not None:
            archive.writestr("AndroidManifest.xml", manifest)
        archive.writestr("classes.dex", b"not a real dex")
    return path


# --------------------------------------------------------------- bundle inspection


@pytest.mark.parametrize("utf8", [True, False], ids=["utf8-pool", "utf16-pool"])
def test_inspect_bundle_reads_package_and_both_versions(tmp_path, utf8: bool) -> None:
    apk = write_apk(tmp_path / "example-debug.apk", build_manifest(utf8=utf8))

    info = android_apk.inspect_bundle(apk)

    assert info.package == "com.example.app"
    assert info.version_name == "2.1.0"
    # versionCode is a typed integer with no raw string, so it exercises the other decode path.
    assert info.version_code == "7"


def test_inspect_bundle_names_the_build_step_when_the_file_is_missing(tmp_path) -> None:
    with pytest.raises(UsageError) as raised:
        android_apk.inspect_bundle(tmp_path / "never-built.apk")

    assert "assembleDebug" in (raised.value.hint or "")


def test_inspect_bundle_rejects_a_directory(tmp_path) -> None:
    with pytest.raises(UsageError, match="directory"):
        android_apk.inspect_bundle(tmp_path)


def test_inspect_bundle_explains_that_an_app_bundle_is_not_installable(tmp_path) -> None:
    apk = write_apk(tmp_path / "example-release.apk", None)

    with pytest.raises(UsageError) as raised:
        android_apk.inspect_bundle(apk)

    assert ".aab" in (raised.value.hint or "")


def test_inspect_bundle_rejects_a_file_that_is_not_an_archive(tmp_path) -> None:
    plain = tmp_path / "example-debug.apk"
    plain.write_bytes(b"this is not a zip")

    with pytest.raises(UsageError, match="as an APK"):
        android_apk.inspect_bundle(plain)


def test_inspect_bundle_offers_package_override_when_the_manifest_is_unreadable(tmp_path) -> None:
    apk = write_apk(tmp_path / "example-debug.apk", b"\x03\x00\x08\x00garbage")

    with pytest.raises(UsageError) as raised:
        android_apk.inspect_bundle(apk)

    assert "--package" in (raised.value.hint or "")


# --------------------------------------------------------------- package manager


class _Adb:
    """Records every adb argv and replays canned results in order."""

    def __init__(self, results: list[tuple[int, str, str]]) -> None:
        self.results = list(results)
        self.commands: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
        self.commands.append(list(argv))
        code, out, err = self.results.pop(0) if self.results else (0, "Success\n", "")
        return subprocess.CompletedProcess(argv, code, out, err)


@pytest.fixture
def adb(monkeypatch):
    def install(results: list[tuple[int, str, str]]) -> _Adb:
        fake = _Adb(results)
        monkeypatch.setattr(android_apk.subprocess, "run", fake)
        monkeypatch.setattr("android_ui_analyser.emulator.adb_bin", lambda: "adb")
        return fake

    return install


def test_install_bundle_builds_the_expected_argv(adb, tmp_path) -> None:
    fake = adb([(0, "Performing Streamed Install\nSuccess\n", "")])
    bundle = tmp_path / "example-debug.apk"

    android_apk.install_bundle("emulator-5554", bundle)

    assert fake.commands == [
        ["adb", "-s", "emulator-5554", "install", "-r", "-t", "-d", str(bundle)]
    ]


def test_install_bundle_drops_replace_and_adds_grant_when_asked(adb, tmp_path) -> None:
    fake = adb([(0, "Success\n", "")])
    bundle = tmp_path / "example-debug.apk"

    android_apk.install_bundle(
        "emulator-5554", bundle, reinstall=False, grant_permissions=True
    )

    assert "-r" not in fake.commands[0]
    assert "-g" in fake.commands[0]


def test_install_bundle_fails_on_failure_printed_with_a_zero_exit(adb, tmp_path) -> None:
    # The whole reason this wrapper exists: some platform-tools report a failed install by
    # printing `Failure [...]` and exiting 0. Trusting the exit code reports a phantom success.
    adb([(0, "Failure [INSTALL_FAILED_INSUFFICIENT_STORAGE]\n", "")])

    with pytest.raises(DeviceError) as raised:
        android_apk.install_bundle("emulator-5554", tmp_path / "example-debug.apk")

    assert raised.value.code == "install_failed"
    assert "INSTALL_FAILED_INSUFFICIENT_STORAGE" in raised.value.message


def test_install_bundle_reads_a_failure_reported_only_on_stderr(adb, tmp_path) -> None:
    adb([(1, "", "adb: failed to install: Failure [INSTALL_FAILED_OLDER_SDK]\n")])

    with pytest.raises(DeviceError, match="INSTALL_FAILED_OLDER_SDK"):
        android_apk.install_bundle("emulator-5554", tmp_path / "example-debug.apk")


def test_signature_mismatch_names_the_only_recovery(adb, tmp_path) -> None:
    adb([(0, "Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE]\n", "")])

    with pytest.raises(DeviceError) as raised:
        android_apk.install_bundle("emulator-5554", tmp_path / "example-debug.apk")

    assert "--fresh" in (raised.value.hint or "")


def test_a_successful_install_that_prints_the_word_error_is_still_a_success(adb, tmp_path) -> None:
    # Only an explicit `Failure` or a non-zero exit is a failure; the engine re-reads the package
    # manager afterwards, so guessing at adb's prose would only invent false negatives.
    adb([(0, "Warning: ErrorProne check skipped\nSuccess\n", "")])

    android_apk.install_bundle("emulator-5554", tmp_path / "example-debug.apk")


def test_installed_app_reports_absent_when_pm_path_prints_nothing(adb) -> None:
    fake = adb([(0, "", "")])

    state = android_apk.installed_app("emulator-5554", "com.example.app")

    assert state == android_apk.InstalledApp(package="com.example.app", installed=False)
    # No second call: there is nothing to describe, so `dumpsys package` is never paid for.
    assert len(fake.commands) == 1


def test_installed_app_parses_the_first_version_out_of_a_realistic_dump(adb) -> None:
    dump = "\n".join(
        [
            "Packages:",
            "  Package [com.example.app] (a1b2c3):",
            "    versionCode=7 minSdk=24 targetSdk=34",
            "    versionName=2.1.0",
            "    Hidden system packages:",
            "      versionCode=1",
            "      versionName=0.9.0",
        ]
    )
    adb([(0, "package:/data/app/~~abc==/com.example.app-1/base.apk\n", ""), (0, dump, "")])

    state = android_apk.installed_app("emulator-5554", "com.example.app")

    assert state.installed is True
    assert state.version_name == "2.1.0"
    assert state.version_code == "7"


def test_uninstall_treats_an_absent_package_as_the_requested_end_state(adb) -> None:
    adb([(1, "Failure [DELETE_FAILED_INTERNAL_ERROR]\n", "")])

    android_apk.uninstall("emulator-5554", "com.example.app")


def test_uninstall_still_raises_on_a_package_it_cannot_remove(adb) -> None:
    adb([(1, "Failure [DELETE_FAILED_DEVICE_POLICY_MANAGER]\n", "")])

    with pytest.raises(DeviceError) as raised:
        android_apk.uninstall("emulator-5554", "com.example.app")

    assert raised.value.code == "uninstall_failed"


def test_a_missing_adb_is_a_typed_error_not_a_traceback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("android_ui_analyser.emulator.adb_bin", lambda: "adb")

    def missing(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("adb")

    monkeypatch.setattr(android_apk.subprocess, "run", missing)

    with pytest.raises(DeviceError) as raised:
        android_apk.install_bundle("emulator-5554", tmp_path / "example-debug.apk")

    assert raised.value.code == "adb_missing"


def test_a_timed_out_install_says_so_rather_than_implying_nothing_happened(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("android_ui_analyser.emulator.adb_bin", lambda: "adb")

    def slow(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired("adb", 1.0)

    monkeypatch.setattr(android_apk.subprocess, "run", slow)

    with pytest.raises(DeviceError) as raised:
        android_apk.install_bundle("emulator-5554", tmp_path / "example-debug.apk")

    assert raised.value.code == "adb_timeout"
