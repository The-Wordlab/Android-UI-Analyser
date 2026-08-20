"""A short-lived CLI call must not lose the map update it just made.

Screen writes are handed to a background thread, and that thread is a daemon: when the
interpreter exits it is killed wherever it happens to be. Every `aua analyze` /
`aua tap-and-analyze` is a process that starts a writer and exits milliseconds later, so the
daemon-less path can silently lose a map update even though the call reported the screen
correctly.

The engine already knows how to wait (`_join_memory_writers`); it was simply never asked to
before shutdown.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config


def _engine(tmp_path: Path) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=FakeDevice())


def test_a_queued_map_write_is_flushed_when_the_process_exits(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    landed: list[str] = []

    def slow_write() -> None:
        time.sleep(0.3)
        landed.append("written")

    writer = threading.Thread(target=slow_write, name="aua-mem-record", daemon=True)
    with engine._mem_threads_lock:
        engine._mem_threads.append(writer)
        writer.start()

    engine_mod._flush_memory_writers_at_exit()

    assert landed == ["written"], "the queued map write was dropped at exit"
    assert not writer.is_alive()


def test_every_engine_is_reachable_from_the_exit_flush(tmp_path: Path) -> None:
    """The hook has to find engines it never saw constructed — CLI, MCP, embedded alike."""

    engine = _engine(tmp_path)
    assert engine in set(engine_mod._LIVE_ENGINES)


def test_the_exit_flush_survives_an_engine_that_has_already_gone(tmp_path: Path) -> None:
    """Shutdown must not raise because something was garbage collected first."""

    _engine(tmp_path)
    import gc

    gc.collect()
    engine_mod._flush_memory_writers_at_exit()  # must not raise
