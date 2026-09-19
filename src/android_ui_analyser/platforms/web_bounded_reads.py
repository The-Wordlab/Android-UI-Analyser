"""Deadline-aware passive reads on the existing Playwright owner thread.

The public sync API has no timeout on evaluate, title, or frame geometry. Use its
sync-to-async bridge for these reads so cancellation reaches Playwright's protocol
abort and is acknowledged before the caller returns. Keep this private-API seam
isolated here; Playwright >=1.63 is required and checked when the driver starts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .. import read_budget


def read(owner: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    budget = read_budget.current()
    if budget is None:
        return getattr(owner, method)(*args, **kwargs)
    budget.check()

    async def bounded() -> Any:
        task = asyncio.create_task(getattr(owner._impl_obj, method)(*args, **kwargs))
        try:
            while not task.done():
                # Check cancellation as well as the deadline while transport I/O is pending.
                await asyncio.wait({task}, timeout=min(0.05, budget.remaining()))
            budget.check()
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            # Playwright's cancellation handler sends protocol abort and drains the reply.
            # Never leave an in-flight read or a detached worker after returning timeout.
            await asyncio.gather(task, return_exceptions=True)

    result = owner._sync(bounded())
    if hasattr(result, "_channel"):
        from playwright._impl._sync_base import mapping

        return mapping.from_maybe_impl(result)
    return result
