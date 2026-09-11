"""A caller's absolute deadline for synchronous, read-only UI work.

The scope is thread-local through ContextVar. It never cancels cleanup or journal writes;
adapters opt in and must bound their transport I/O before advertising the capability.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .errors import JobCancelledError


class ReadDeadlineExceeded(TimeoutError):
    """The caller's UI-read deadline expired, including any final observation."""


@dataclass
class ReadBudget:
    deadline: float
    clock: Callable[[], float]
    cancelled: Callable[[], bool] = lambda: False
    result: Any = None
    terms: list[dict[str, Any]] = field(default_factory=list)
    polling_deadline: float | None = None
    locale_hint: str | None = None

    def remaining(self) -> float:
        if self.cancelled():
            raise JobCancelledError("background wait cancelled")
        seconds = self.deadline - self.clock()
        if seconds <= 0:
            raise ReadDeadlineExceeded("UI-read deadline reached")
        return seconds

    def check(self) -> None:
        self.remaining()


_CURRENT: ContextVar[ReadBudget | None] = ContextVar("aua_read_budget", default=None)


def current() -> ReadBudget | None:
    return _CURRENT.get()


def poll_deadline(deadline: float) -> float:
    budget = current()
    return (
        min(deadline, budget.polling_deadline)
        if budget is not None and budget.polling_deadline is not None
        else deadline
    )


@contextmanager
def activate(budget: ReadBudget) -> Iterator[None]:
    token = _CURRENT.set(budget)
    try:
        # Entering/leaving a scope performs no I/O. In particular, expiry finalization
        # must retain the scope so lazy metadata cannot issue a new unbounded device read.
        yield
    finally:
        _CURRENT.reset(token)
