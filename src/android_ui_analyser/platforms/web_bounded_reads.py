"""Deadline-aware passive reads on the existing Playwright owner thread.

The public sync API has no timeout on evaluate, title, or frame geometry. Use its
sync-to-async bridge for these reads so cancellation reaches Playwright's protocol
abort and is acknowledged before the caller returns. Call the generated public
wrapper to retain its defaults, argument conversion and returned handle mapping.
"""

from __future__ import annotations

import asyncio
from importlib.metadata import version
from typing import Any

from .. import read_budget
from ..errors import DeviceError


def require_cancellable_driver() -> None:
    """Cancellation arrived in 1.63; its private implementation moved in 1.65.

    In 1.63 Connection._abort handles it. In 1.65 cancelling the protocol future
    sends __abort__ directly, without that method. Check the distribution version,
    not the presence of a private function that AUA never calls.
    """
    installed = version("playwright")
    parts = installed.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]) or tuple(map(int, parts[:2])) < (1, 63):
        raise DeviceError(
            f"web UI deadlines require Playwright >=1.63 (installed: {installed})",
            code="web_driver_outdated",
            hint="Upgrade android-ui-analyser[web], then install its browser binaries.",
        )


def read(owner: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    budget = read_budget.current()
    if budget is None:
        return getattr(owner, method)(*args, **kwargs)
    budget.check()

    async def bounded(awaitable: Any) -> Any:
        connection = getattr(owner._impl_obj, "_connection", None)
        callbacks = getattr(connection, "_callbacks", {})
        previous = set(callbacks)
        task = asyncio.ensure_future(awaitable)
        try:
            while not task.done():
                # Check cancellation as well as the deadline while transport I/O is pending.
                await asyncio.wait({task}, timeout=min(0.05, budget.remaining()))
            budget.check()
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # 1.63 awaits the abort reply inside the task. 1.65 sends the abort
            # from a cancelled future's callback and removes its registry entry
            # only when the reply arrives. Drain only this read's cancelled calls.
            pending = {key for key, callback in callbacks.items()
                       if key not in previous and callback.future.cancelled()}
            while pending.intersection(callbacks):
                await asyncio.sleep(0.001)

    # Every use is on the connection's one owner thread. Temporarily wrap just the
    # bridge; the generated public method supplies version-specific implementation
    # defaults and converts returned handles into the appropriate sync API objects.
    sync = owner._sync
    missing = object()
    original = vars(owner).get("_sync", missing)
    owner._sync = lambda awaitable: sync(bounded(awaitable))
    try:
        return getattr(owner, method)(*args, **kwargs)
    finally:
        if original is missing:
            del owner._sync
        else:
            owner._sync = original
