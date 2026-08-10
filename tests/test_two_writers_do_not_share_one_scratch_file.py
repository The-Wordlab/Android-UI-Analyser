"""`os.replace` made the write atomic; the shared scratch name made it crash anyway.

Every writer used `path.with_suffix(".tmp")`, so the daemon, the CLI and each parallel agent
all wrote the same scratch file. Writer A renames it into place, writer B's rename finds
nothing there. Measured on a live agent run, 2026-08-10, on the agent's first `goto`:

    {"error": {"code": "internal_error", "message": "[Errno 2] No such file or directory:
      '…/state/session_emulator-5554.json.tmp' -> '…/state/session_emulator-5554.json'"}}

The cost was not the crash. The session cursor never got written, so the next `goto` planned
from a screen the device had already left and pressed `back` twice on the Android home screen.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from android_ui_analyser.atomic import atomic_write_text


def test_concurrent_writers_all_survive(tmp_path: Path) -> None:
    """The live failure was two processes writing one device's cursor. Threads reproduce it."""
    target = tmp_path / "session.json"
    errors: list[BaseException] = []

    def write(n: int) -> None:
        try:
            for _ in range(40):
                atomic_write_text(target, f"writer-{n}")
        except BaseException as exc:  # noqa: BLE001 - the bug was an uncaught FileNotFoundError
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert target.read_text().startswith("writer-"), "last writer wins, intact"
    assert not list(tmp_path.glob("*.tmp")), "no scratch files left behind"


def test_each_write_uses_its_own_scratch_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "session.json"
    seen: list[str] = []
    real_replace = os.replace

    def record(src: object, dst: object) -> None:
        seen.append(str(src))
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", record)
    atomic_write_text(target, "a")
    atomic_write_text(target, "b")

    assert seen[0] != seen[1], "two writes must not contend for one scratch file"
    assert target.read_text() == "b"


def test_the_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "state" / "nested" / "session.json"

    atomic_write_text(target, "hello")

    assert target.read_text() == "hello"


def test_a_failed_write_leaves_no_litter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "session.json"

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "a")

    assert list(tmp_path.iterdir()) == [], "a dead scratch file would break the next reader"


def test_the_reader_never_sees_a_half_written_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the original code was written for, kept intact."""
    target = tmp_path / "session.json"
    atomic_write_text(target, '{"complete": true}')
    during: list[str] = []
    real_replace = os.replace

    def peek(src: object, dst: object) -> None:
        during.append(target.read_text())
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", peek)
    atomic_write_text(target, '{"complete": false}')

    assert during == ['{"complete": true}'], "the old contents stand until the swap"
