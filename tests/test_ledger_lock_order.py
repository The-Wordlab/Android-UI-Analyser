"""Cross-process regression tests for ledger/device lock ordering."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from android_ui_analyser import leases
from android_ui_analyser.platforms.identity import TargetRef


def _ordinary_worker(root: str, events, control, done) -> None:
    ref = TargetRef("strict-fake", "fictional")
    with leases.device_command(root, ref):
        events.put("ordinary-device-lock")
        control.get(timeout=5)
    events.put("ordinary-done")
    events.close()
    events.join_thread()
    done.set()


def _recovery_worker(root: str, events, control, done) -> None:
    ref = TargetRef("strict-fake", "fictional")
    with leases.host_transaction(root, f"ledger|{ref.storage_key}"):
        events.put("recovery-host-lock")
        control.get(timeout=5)
        with leases.device_command(root, ref):
            pass
    events.put("recovery-done")
    events.close()
    events.join_thread()
    done.set()


@pytest.mark.skipif(
    multiprocessing.get_start_method(allow_none=True) == "spawn", reason="fork test"
)
def test_ordinary_and_recovery_paths_have_one_lock_order(tmp_path: Path):
    ctx = multiprocessing.get_context("fork")
    events = ctx.Queue()
    ordinary_control = ctx.Queue()
    recovery_control = ctx.Queue()
    ordinary_done = ctx.Event()
    recovery_done = ctx.Event()
    ordinary = ctx.Process(
        target=_ordinary_worker, args=(str(tmp_path), events, ordinary_control, ordinary_done)
    )
    recovery = ctx.Process(
        target=_recovery_worker, args=(str(tmp_path), events, recovery_control, recovery_done)
    )
    recovery.start()
    try:
        assert events.get(timeout=5) == "recovery-host-lock"
        ordinary.start()
        recovery_control.put("release")
        # With the old order both workers now wait forever: ordinary wants host while recovery
        # wants device. The new order lets recovery finish, then ordinary acquires both locks.
        # Queue ordering is guaranteed only per producer, so either worker's final message may
        # arrive first after recovery releases the locks.
        assert recovery_done.wait(timeout=5)
        assert {events.get(timeout=5), events.get(timeout=5)} == {
            "recovery-done",
            "ordinary-device-lock",
        }
        ordinary_control.put("release")
        assert ordinary_done.wait(timeout=5)
        ordinary.join(timeout=5)
        recovery.join(timeout=5)
    finally:
        for process in (ordinary, recovery):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert ordinary.exitcode == 0
    assert recovery.exitcode == 0
