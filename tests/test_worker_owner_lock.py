"""Sibling agent workers must not block each other's target selection."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from android_ui_analyser import leases


def _hold_owner_lock(root, scope, entered, release):
    os.environ["AUA_WORKER_SCOPE"] = scope
    with leases.owner_transaction(root, "fictional-shared-agent"):
        entered.set()
        release.wait(timeout=10)


@pytest.mark.parametrize("sibling_scope", ["worker-a", "worker-b"])
def test_owner_transactions_follow_worker_scope(tmp_path: Path, sibling_scope: str):
    ctx = multiprocessing.get_context("spawn")
    first_entered, next_entered = ctx.Event(), ctx.Event()
    first_release, next_release = ctx.Event(), ctx.Event()
    first = ctx.Process(target=_hold_owner_lock, args=(str(tmp_path), "worker-a", first_entered, first_release))
    second = ctx.Process(target=_hold_owner_lock, args=(str(tmp_path), sibling_scope, next_entered, next_release))
    first.start()
    try:
        assert first_entered.wait(timeout=5)
        second.start()
        if sibling_scope == "worker-a":
            assert not next_entered.wait(timeout=0.3)
        else:
            assert next_entered.wait(timeout=3), "a different worker is blocked by its sibling's owner lock"
        first_release.set()
        assert next_entered.wait(timeout=5)
        next_release.set()
    finally:
        first_release.set()
        next_release.set()
        for process in (first, second):
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
    assert first.exitcode == second.exitcode == 0
