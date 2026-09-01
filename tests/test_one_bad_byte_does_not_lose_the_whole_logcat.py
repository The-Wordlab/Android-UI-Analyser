"""A single non-UTF-8 byte in the buffer must not take the whole logcat dump with it.

Reported 2026-09-01 by a QA runner: `aua logcat` was clean at `--since 60s` and died at 300s,
900s and 1800s with

    'utf-8' codec can't decode byte 0xc0 in position 211106

at positions 211106 / 1457485 / 6793208 — one bad byte further into the buffer each time.
`--tag` failed identically, because filtering happens after the whole buffer is decoded, so
narrowing the request could not rescue it. That took the suite's own redacting reader
(`tools/safe-logcat.sh`) with it, and every tier, quota and crash check for the sweep had to be
re-done against raw shell output.

Logcat carries whatever bytes an app logged. It is not a UTF-8 channel and cannot be treated as
one: a strict decode makes one misbehaving line on the device delete every other line for the
reader. Replacement characters in one line are a far better answer than no lines at all.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from android_ui_analyser.device import Uiautomator2Device


@pytest.fixture()
def fake_adb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An `adb` on PATH whose logcat emits a line, one 0xc0 byte, and another line."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    adb = bin_dir / "adb"
    adb.write_text(
        "#!/bin/sh\n"
        "printf 'first readable line\\n'\n"
        "printf '\\300'\n"
        "printf '\\nsecond readable line\\n'\n",
        encoding="utf-8",
    )
    adb.chmod(adb.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return adb


def _device() -> Uiautomator2Device:
    return Uiautomator2Device.__new__(Uiautomator2Device)


def test_the_dump_survives_a_byte_that_is_not_utf_8(fake_adb: Path) -> None:
    """The regression: this used to raise UnicodeDecodeError and return nothing at all."""

    device = _device()
    device.serial = "emulator-5554"  # type: ignore[attr-defined]

    out = device._logcat_dump([])

    assert "first readable line" in out
    assert "second readable line" in out


def test_the_bad_byte_is_replaced_rather_than_dropping_its_line(fake_adb: Path) -> None:
    """The undecodable byte becomes U+FFFD. Its neighbours are not silently discarded."""

    device = _device()
    device.serial = "emulator-5554"  # type: ignore[attr-defined]

    out = device._logcat_dump([])

    assert "\ufffd" in out, "expected the replacement character where 0xc0 was"
    assert len(out.splitlines()) >= 2
