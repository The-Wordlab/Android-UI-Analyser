"""How often a policy provider's output was actually usable, per provider.

The policy guard already refuses to act on output it cannot parse, which is why an unusable
model never executed anything. What it did not do is *notice*. Observed live: the configured
chain was a local primary with a small local fallback, and roughly four in five of the
fallback's answers never resolved to an offered candidate ID. Each one was rejected correctly
and then silently replaced by another attempt costing real seconds, so the run reported itself
as working while it mostly could not steer.

An 80% invalid rate is not a fallback. This module keeps a small rolling window of selection
attempts per provider so that rate is:

* **measured** — every attempt is counted valid or invalid at the one place the guard decides;
* **reported** — the rate rides along in the decision trace and in ``aua policy status``;
* **acted on** — once the recent window is majority-invalid, the provider is refused instead
  of consulted, and a command that depends on it (``aua session autopilot``) fails once,
  up front, carrying the measured rate instead of handing off on every step.

Deliberately in-memory and process-local: it describes the model runtime this process is
actually talking to, and a fresh process (or ``reset()``) gets to form its own opinion. The
counters are only ever counters — nothing here can authorize a call.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

__all__ = [
    "MIN_ATTEMPTS",
    "PolicyHealthRegistry",
    "WINDOW",
    "record",
    "registry",
    "report",
    "unusable_reason",
]

#: Attempts kept per provider. Small on purpose: this is a "is it working right now" signal.
WINDOW = 20
#: A single bad decision can spend several attempts, so one is never a verdict. Six is about
#: two decisions' worth of evidence — enough that a provider failing four times in five is
#: condemned within roughly a minute of wall clock instead of a whole run.
MIN_ATTEMPTS = 6
#: Majority-invalid is the line. Below it a provider is merely imperfect, which the chain's
#: consensus and fallback layers already handle.
MAX_INVALID_RATE = 0.5


class PolicyHealthRegistry:
    """Rolling per-provider validity counters with a fail-loud verdict."""

    def __init__(
        self,
        *,
        window: int = WINDOW,
        min_attempts: int = MIN_ATTEMPTS,
        max_invalid_rate: float = MAX_INVALID_RATE,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if min_attempts < 1:
            raise ValueError("min_attempts must be positive")
        if not 0.0 < max_invalid_rate <= 1.0:
            raise ValueError("max_invalid_rate must be in (0, 1]")
        self._window = window
        self._min_attempts = min_attempts
        self._max_invalid_rate = max_invalid_rate
        self._lock = threading.Lock()
        self._attempts: dict[str, deque[bool]] = {}

    def record(self, provider: str, *, attempts: int, invalid: int) -> None:
        """Record *attempts* selection attempts of which *invalid* produced no usable ID."""

        name = str(provider or "").strip() or "unknown"
        attempts = max(0, int(attempts))
        invalid = min(max(0, int(invalid)), attempts)
        if attempts == 0:
            return
        with self._lock:
            window = self._attempts.setdefault(name, deque(maxlen=self._window))
            # Order inside one decision does not matter; recency between decisions does.
            for _ in range(attempts - invalid):
                window.append(True)
            for _ in range(invalid):
                window.append(False)

    def report(self, provider: str) -> dict[str, Any]:
        """Return the recent-window validity report for *provider*."""

        name = str(provider or "").strip() or "unknown"
        with self._lock:
            window = list(self._attempts.get(name, ()))
        attempts = len(window)
        invalid = sum(1 for ok in window if not ok)
        value: dict[str, Any] = {
            "provider": name,
            "attempts": attempts,
            "invalid": invalid,
            "invalid_rate": round(invalid / attempts, 3) if attempts else 0.0,
            "window": self._window,
            "usable": True,
        }
        reason = self._unusable_reason(attempts, invalid)
        if reason is not None:
            value["usable"] = False
            value["reason"] = reason
        return value

    def unusable_reason(self, provider: str) -> str | None:
        """Return why *provider* must not be consulted, or ``None`` when it may be."""

        name = str(provider or "").strip() or "unknown"
        with self._lock:
            window = list(self._attempts.get(name, ()))
        return self._unusable_reason(len(window), sum(1 for ok in window if not ok))

    def reset(self, provider: str | None = None) -> None:
        """Forget the window for one provider, or for all of them."""

        with self._lock:
            if provider is None:
                self._attempts.clear()
                return
            self._attempts.pop(str(provider or "").strip() or "unknown", None)

    def known_providers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._attempts))

    def _unusable_reason(self, attempts: int, invalid: int) -> str | None:
        if attempts < self._min_attempts:
            return None
        rate = invalid / attempts
        if rate <= self._max_invalid_rate:
            return None
        return (
            f"the provider returned unusable output in {invalid} of {attempts} recent selection "
            f"attempts ({round(rate * 100)}%), so it is refused instead of consulted; "
            "run `aua policy status` for the per-provider report"
        )


_REGISTRY = PolicyHealthRegistry()


def registry() -> PolicyHealthRegistry:
    """Return the process-wide registry."""

    return _REGISTRY


def record(provider: str, *, attempts: int, invalid: int) -> None:
    _REGISTRY.record(provider, attempts=attempts, invalid=invalid)


def report(provider: str) -> dict[str, Any]:
    return _REGISTRY.report(provider)


def unusable_reason(provider: str) -> str | None:
    return _REGISTRY.unusable_reason(provider)
