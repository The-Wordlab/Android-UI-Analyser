"""A daemon that has been outdated should step aside on its own time, not on the caller's.

The correctness check is not negotiable: a daemon holds modules in memory, so an edited file is
invisible to it, and because `__version__` still matches, a plain version check never fires. It
keeps answering out of bytes the caller never wrote -- including rejecting kwargs added since it
started. That is why identity is a digest of the loaded source.

The cost is that in an editable install with several agents editing the tree, skew is the
resting state rather than an exception, and every path out of it used to be paid by whoever
called next: either the in-process fallback (a full device attach, ~6x slower, warned about only
on stderr) or an inline stop+start on the critical path -- up to 10 seconds, and if it fails,
paid again by the following call.

An idle daemon can notice this about itself for free. It already wakes every 0.5s on the accept
timeout, already refuses to act while a job is in flight, and already has a clean self-shutdown
path; `perf.auto_daemon` then starts a current one on the next call. That moves the whole cost
off the critical path and out of mid-flight work.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import android_ui_analyser.daemon as daemon_mod
from conftest import make_config


class _Activity:
    def __init__(self, idle: float) -> None:
        self._idle = idle

    def touch(self) -> None:  # pragma: no cover - not exercised here
        self._idle = 0.0

    def idle_s(self) -> float:
        return self._idle


def _engine(tmp_path: Path, *, active_job: Any = None) -> Any:
    cfg = make_config(cache={"dir": str(tmp_path)})
    cfg.perf.auto_daemon = True
    cfg.daemon.idle_ttl_s = 0  # isolate source-skew retirement from the idle-TTL shutdown
    cfg.capture.idle_pause_s = 0
    return SimpleNamespace(
        config=cfg,
        _aua_job_manager=SimpleNamespace(active=lambda: active_job),
        capture_idle_pause=lambda: False,
    )


def test_an_idle_daemon_whose_source_moved_retires_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_mod, "_source_fingerprint", lambda: "a-tree-that-moved-on")
    monkeypatch.setattr(daemon_mod, "_last_source_check_at", 0.0)

    assert daemon_mod._idle_tick(_engine(tmp_path), _Activity(3.0)) is True


def test_a_daemon_whose_source_still_matches_stays_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the guard: retiring on every tick would make the daemon useless."""
    monkeypatch.setattr(daemon_mod, "_source_fingerprint", lambda: daemon_mod._LOADED_SOURCE)
    monkeypatch.setattr(daemon_mod, "_last_source_check_at", 0.0)

    assert daemon_mod._idle_tick(_engine(tmp_path), _Activity(3.0)) is False


def test_a_daemon_with_a_job_in_flight_does_not_retire_however_stale_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job worker is a thread inside this process; retiring would abandon another agent."""
    monkeypatch.setattr(daemon_mod, "_source_fingerprint", lambda: "a-tree-that-moved-on")
    monkeypatch.setattr(daemon_mod, "_last_source_check_at", 0.0)
    engine = _engine(tmp_path, active_job=SimpleNamespace(job_id="busy"))

    assert daemon_mod._idle_tick(engine, _Activity(3600.0)) is False


def test_retirement_is_reported_so_a_puzzled_agent_can_find_out_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Another agent's next call pays one cold start. It must be able to learn the reason."""
    monkeypatch.setattr(daemon_mod, "_source_fingerprint", lambda: "a-tree-that-moved-on")
    monkeypatch.setattr(daemon_mod, "_last_source_check_at", 0.0)

    with caplog.at_level("INFO", logger="android_ui_analyser.daemon"):
        assert daemon_mod._idle_tick(_engine(tmp_path), _Activity(3.0)) is True

    assert any("source" in record.getMessage() for record in caplog.records), caplog.text
