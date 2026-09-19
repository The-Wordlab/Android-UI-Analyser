"""Browser reads propagate deadlines and drain cancellation on their owner thread."""

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from android_ui_analyser import read_budget
from android_ui_analyser.errors import DeviceError, JobCancelledError
from android_ui_analyser.platforms.web_bounded_reads import read
from android_ui_analyser.platforms.web_runtime import WebRuntime
from android_ui_analyser.platforms.web_tools import (
    PlaywrightConnection,
    PlaywrightLauncher,
    WebLaunchOptions,
)


@pytest.fixture
def connection():
    with ThreadPoolExecutor(max_workers=1) as executor:
        conn = PlaywrightConnection(
            executor, object(), object(), object(), WebLaunchOptions(), "https://fixture.test"
        )
        conn._page = SimpleNamespace(is_closed=lambda: False)
        conn._context = object()
        yield conn


@pytest.mark.parametrize("method", ["evaluate", "title", "screenshot", "bounding_box"])
def test_slow_read_cancels_before_returning_and_restores_context(connection, method):
    seen = []
    drained = []

    async def slow():
        seen.append((read_budget.current(), threading.get_ident()))
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.001)
            drained.append(True)

    page = SimpleNamespace(_impl_obj=SimpleNamespace(**{method: slow}), _sync=asyncio.run)
    runtime = WebRuntime(connection, "https://fixture.test")
    budget = read_budget.ReadBudget(time.monotonic() + 0.08, time.monotonic)
    started = time.monotonic()
    with pytest.raises(read_budget.ReadDeadlineExceeded), runtime.read_deadline(budget):
        connection._call(lambda: read(page, method))
    assert time.monotonic() - started < 0.5
    assert drained == [True]
    assert seen[0][0] is budget
    assert seen[0][1] != threading.get_ident()
    assert read_budget.current() is None
    assert connection._call(read_budget.current) is None


def test_job_cancellation_is_not_wrapped_as_a_browser_error(connection):
    cancelled = threading.Event()
    drained = []

    async def slow():
        cancelled.set()
        try:
            await asyncio.sleep(10)
        finally:
            drained.append(True)

    page = SimpleNamespace(_impl_obj=SimpleNamespace(evaluate=slow), _sync=asyncio.run)
    budget = read_budget.ReadBudget(time.monotonic() + 2, time.monotonic, cancelled.is_set)
    with pytest.raises(JobCancelledError), read_budget.activate(budget):
        connection._call(lambda: read(page, "evaluate"))
    assert drained == [True]


def test_expired_queued_read_is_cancelled_without_ever_running(connection):
    blocked = threading.Event()
    entered = threading.Event()

    def busy():
        entered.set()
        blocked.wait(2)

    connection._executor.submit(busy)
    assert entered.wait(1)
    budget = read_budget.ReadBudget(time.monotonic() + 0.05, time.monotonic)
    try:
        with pytest.raises(read_budget.ReadDeadlineExceeded), read_budget.activate(budget):
            connection._call(lambda: pytest.fail("expired queued read ran"))
    finally:
        blocked.set()
    assert connection._call(lambda: True)


def test_passive_read_never_reopens_a_closed_page(connection):
    connection._page = None
    connection._context = SimpleNamespace(
        pages=[], new_page=lambda: pytest.fail("passive wait must not create a page")
    )
    budget = read_budget.ReadBudget(time.monotonic() + 1, time.monotonic)
    with pytest.raises(DeviceError) as error, read_budget.activate(budget):
        _ = connection.url
    assert error.value.code == "web_page_closed"


def test_driver_without_protocol_cancellation_fails_before_launch(monkeypatch):
    stopped = []
    driver = SimpleNamespace(
        _impl_obj=SimpleNamespace(_connection=object()),
        chromium=SimpleNamespace(launch=lambda **kwargs: pytest.fail("must not launch")),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        SimpleNamespace(sync_playwright=lambda: SimpleNamespace(start=lambda: driver)),
    )
    with pytest.raises(DeviceError) as error:
        PlaywrightLauncher().launch("https://fixture.test", WebLaunchOptions())
    assert error.value.code == "web_driver_outdated"
    assert stopped == [True]
