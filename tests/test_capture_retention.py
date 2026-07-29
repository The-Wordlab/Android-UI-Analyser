"""Capture must stay bounded ACROSS sessions, not just within one.

The per-session TTL/size caps are enforced by the thread that owns the session, so when a
daemon stops, frames still inside the TTL window are orphaned — no process owns them, and
nothing ever prunes them. Every restart mints another session directory, so the aggregate
grew without bound: 9 sessions and 11 MB in one afternoon on a 180s TTL, the oldest 95
minutes past it, while `capture status` reported 114 kB.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from android_ui_analyser.capture import CaptureBuffer, CaptureCfgView


def _session(root: Path, serial: str, name: str, *, frames: int, kb: int, age_s: float) -> Path:
    """Fabricate a dead session directory with *frames* files of *kb* each, aged *age_s*."""
    d = root / serial / name / "frames"
    d.mkdir(parents=True, exist_ok=True)
    when = time.time() - age_s
    for i in range(frames):
        p = d / f"{i:06d}.jpg"
        p.write_bytes(b"\xff\xd8" + b"0" * (kb * 1024))
        os.utime(p, (when, when))
    return d.parent


def _buffer(root: Path, **cfg: object) -> CaptureBuffer:
    view = CaptureCfgView(ttl_s=60, max_mb=1, **cfg)  # type: ignore[arg-type]
    return CaptureBuffer(root=root, serial="emu-test", cfg=view, screenshot=lambda: None)


def test_dead_session_past_ttl_is_swept(tmp_path: Path) -> None:
    old = _session(tmp_path, "emu-test", "20200101-000000-aaaaaa", frames=3, kb=10, age_s=999)
    assert old.is_dir()
    _buffer(tmp_path).sweep_sessions()
    assert not old.is_dir(), "a dead session older than the TTL must be removed"


def test_dead_session_inside_ttl_survives(tmp_path: Path) -> None:
    """Only what is genuinely stale goes — a recent dead session may still be wanted."""
    recent = _session(tmp_path, "emu-test", "20990101-000000-bbbbbb", frames=1, kb=1, age_s=1)
    _buffer(tmp_path).sweep_sessions()
    assert recent.is_dir()


def test_empty_session_directory_is_removed(tmp_path: Path) -> None:
    empty = tmp_path / "emu-test" / "20990101-000000-cccccc" / "frames"
    empty.mkdir(parents=True)
    _buffer(tmp_path).sweep_sessions()
    assert not empty.parent.is_dir(), "a session that captured nothing must not linger"


def test_aggregate_is_capped_even_when_every_session_is_recent(tmp_path: Path) -> None:
    """The real failure mode: many recent sessions, each individually within its caps.

    max_mb is a budget for the tool, not per run — 10 sessions of 400 kB must not add up to
    4 MB under a 1 MB cap just because none of them is old enough to expire.
    """
    for i in range(10):
        _session(tmp_path, "emu-test", f"20990101-0000{i:02d}-dddddd", frames=4, kb=100, age_s=2)
    buf = _buffer(tmp_path)
    before = buf.total_disk_bytes()
    assert before > 1024 * 1024, "fixture should start over the 1 MB budget"
    buf.sweep_sessions()
    assert buf.total_disk_bytes() <= 1024 * 1024, (
        f"aggregate still {buf.total_disk_bytes()} bytes after sweep — max_mb is not a budget"
    )


def test_sweep_never_touches_the_live_session(tmp_path: Path) -> None:
    """`_prune` owns the live session; the sweep must not delete frames under it."""
    buf = _buffer(tmp_path)
    live = buf.dir / "frames"
    live.mkdir(parents=True)
    for i in range(4):
        (live / f"{i}.jpg").write_bytes(b"\xff\xd8" + b"0" * (200 * 1024))
    old = time.time() - 9999
    for p in live.glob("*.jpg"):
        os.utime(p, (old, old))  # stale AND over budget, yet still ours
    buf.sweep_sessions()
    assert live.is_dir() and list(live.glob("*.jpg")), "the live session must survive a sweep"


def test_status_reports_the_aggregate_not_just_this_session(tmp_path: Path) -> None:
    _session(tmp_path, "emu-test", "20990101-000000-eeeeee", frames=2, kb=50, age_s=1)
    buf = _buffer(tmp_path)
    (buf.dir / "frames").mkdir(parents=True)
    (buf.dir / "frames" / "0.jpg").write_bytes(b"\xff\xd8" + b"0" * 1024)
    st = buf.status()
    assert st["total_disk_bytes"] > st["disk_bytes"], (
        "status under-reports: it must show what the tool consumes, not one session"
    )
