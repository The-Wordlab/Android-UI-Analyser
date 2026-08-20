from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser import leases
from android_ui_analyser.errors import DeviceLeasedError


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.on_sleep = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(self.now)


def test_wait_for_explicit_device_keeps_pin_until_holder_releases(tmp_path: Path) -> None:
    assert leases.acquire(tmp_path, "example-1", owner="first")
    clock = Clock()

    def release(at: float) -> None:
        if at >= 0.5:
            leases.release(tmp_path, "example-1", owner="first")

    clock.on_sleep = release
    selected, why, waited_ms = leases.wait_for_device(
        tmp_path,
        owner="second",
        explicit="example-1",
        candidates=lambda: [("example-1", {}), ("example-2", {})],
        wait_s=2,
        poll_s=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert selected == "example-1"
    assert why == "explicit"
    assert waited_ms == 500
    assert leases.holder(tmp_path, "example-2") is None


def test_wait_timeout_reports_elapsed_and_never_steals(tmp_path: Path) -> None:
    assert leases.acquire(tmp_path, "example-1", owner="first")
    clock = Clock()

    with pytest.raises(DeviceLeasedError) as raised:
        leases.wait_for_device(
            tmp_path,
            owner="second",
            explicit="example-1",
            candidates=lambda: [("example-1", {})],
            wait_s=0.5,
            poll_s=0.25,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert "after waiting 500ms" in raised.value.message
    assert leases.holder(tmp_path, "example-1") == "first"


def test_zero_wait_preserves_fail_fast_behavior(tmp_path: Path) -> None:
    assert leases.acquire(tmp_path, "example-1", owner="first")

    with pytest.raises(DeviceLeasedError) as raised:
        leases.wait_for_device(
            tmp_path,
            owner="second",
            explicit="example-1",
            candidates=lambda: [("example-1", {})],
            wait_s=0,
        )

    assert "after waiting 0ms" in raised.value.message
