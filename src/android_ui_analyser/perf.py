"""Latency helpers: hierarchy prefetch, settle profiles, element diffs, gate cache."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .schema import Element

logger = logging.getLogger(__name__)

DumpFn = Callable[[], str]
ParseFn = Callable[[str], tuple[list[Element], str | None]]


@dataclass
class PrefetchSlot:
    """One warm hierarchy dump ready for the next analyze."""

    xml: str
    elements: list[Element]
    package: str | None
    at: float
    gen: int


@dataclass
class HierarchyPrefetch:
    """Speculative background ``dump_hierarchy`` so the next analyze can skip the RPC."""

    max_age_ms: float = 350.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _slot: PrefetchSlot | None = None
    _gen: int = 0
    _thread: threading.Thread | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._gen += 1
            self._slot = None

    def take(self, *, max_age_ms: float | None = None) -> PrefetchSlot | None:
        age_limit = self.max_age_ms if max_age_ms is None else max_age_ms
        with self._lock:
            slot = self._slot
            if slot is None:
                return None
            if slot.gen != self._gen:
                self._slot = None
                return None
            if (time.monotonic() - slot.at) * 1000.0 > age_limit:
                self._slot = None
                return None
            self._slot = None
            return slot

    def quiesce(self, timeout_s: float = 5.0) -> bool:
        """Stop speculating and wait for any dump already in flight. True if the device is idle.

        A prefetch is a real ``dump_hierarchy`` on a background thread, and it is invisible to
        anything reasoning about who is driving the device. That is harmless while AUA owns
        the UiAutomation slot and actively dangerous when it is about to hand the slot to the
        on-device helper: the dump fails, uiautomator2 answers a failed call by restarting its
        server, and the restart silently suppresses the accessibility service the helper needs
        — mid-run, so steps start failing several at a time with nothing pointing at the cause.

        Invalidating first means the result is discarded even if it lands; the join is about
        the *call*, not the value. Returning False rather than pressing on lets the caller
        decline the handover, which is always the safe answer.
        """

        self.invalidate()
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    def kick(self, dump: DumpFn, parse: ParseFn) -> None:
        """Start a background dump if one isn't already in flight for this generation."""
        with self._lock:
            gen = self._gen
            if self._thread is not None and self._thread.is_alive():
                return

            def _run() -> None:
                try:
                    xml = dump()
                    elements, package = parse(xml)
                except Exception as exc:  # noqa: BLE001 — prefetch is best-effort
                    logger.debug("hierarchy prefetch failed: %s", exc)
                    return
                with self._lock:
                    if gen != self._gen:
                        return
                    self._slot = PrefetchSlot(
                        xml=xml,
                        elements=elements,
                        package=package,
                        at=time.monotonic(),
                        gen=gen,
                    )

            t = threading.Thread(target=_run, name="aua-hier-prefetch", daemon=True)
            self._thread = t
            t.start()


# --------------------------------------------------------------------------- wait policy
#
# Two knobs the agent loop is measured against, kept together because they trade off:
# `clamp_wait_ms` bounds how long any single observation may block, and `stable_delay_for`
# buys back the accuracy that a short ceiling would otherwise cost.
#
# Provisioning is deliberately NOT routed through here. Installing an APK or booting an AVD
# legitimately takes minutes and is not an observation, so those keep their own budgets.

_PROVISIONING_KINDS = frozenset(
    {"install", "emulator-start", "emulator-stop", "network", "job", "flow", "boot"}
)


def clamp_wait_ms(
    requested_ms: int | None, config: Any, ceiling_ms: int | None = None
) -> tuple[int, bool]:
    """Bound one observation wait to ``perf.max_wait_ms``, or to *ceiling_ms* when given.

    Returns ``(effective_ms, was_clamped)``. A caller asking for more is not an error and is
    not silently obeyed either: re-invoking the agent costs one function call, whereas
    honouring a 45s request costs 45s of a blocked session. The caller is expected to surface
    ``was_clamped`` so "not yet" is never mistaken for "not there".

    ``ceiling_ms`` exists so the caller-adaptive ceiling (:func:`wait_ceiling_ms`) is enforced
    *through* this function rather than beside it. Two clamps that disagree is worse than one
    that is occasionally too tight: the tighter one silently wins and the looser one reads as a
    guarantee it is not. It can only ever be at or below ``perf.max_wait_ms`` — the adaptive
    policy shortens, never lengthens — so passing it never widens what this function permits.
    """
    ceiling = int(getattr(getattr(config, "perf", None), "max_wait_ms", 5000) or 5000)
    if ceiling_ms is not None:
        ceiling = min(ceiling, max(0, int(ceiling_ms)))
    if requested_ms is None:
        return ceiling, False
    requested = int(requested_ms)
    if requested <= ceiling:
        return max(0, requested), False
    return ceiling, True


#: Pin the observation-wait ceiling for one run. An adaptive budget is exactly what a
#: measurement run must not have: the same script would get a different budget on a different
#: day, and the numbers would not be comparable. Config can pin it too
#: (``perf.wait_ceiling_mode = "fixed"``); this exists because a sweep is a shell loop, not an
#: edit to the config file. It cannot exceed ``perf.max_wait_ms`` — a convenience for sweeps is
#: not an escape hatch from the ceiling, and raising the maximum is a config decision.
WAIT_CEILING_ENV = "AUA_WAIT_CEILING_MS"


def wait_ceiling_ms(cap_ms: int, config: Any, profile: Any = None) -> tuple[int, str]:
    """The ceiling for one observation wait, and which policy produced it.

    Returns ``(ceiling_ms, mode)`` where mode is ``pinned`` (env), ``fixed`` (config),
    ``cold`` (no caller samples yet) or ``adaptive`` (sized from the measured caller gap).
    The mode rides on the response so a reader can tell a reproducible run from an adaptive one.

    *cap_ms* is ``perf.max_wait_ms``, passed in rather than read here so that the ceiling has
    exactly one reader in the engine — the single gate ``Engine._bounded_wait_ms``. **Every
    branch is capped by it, including the pins.** That is the whole contract: this function can
    hand back a number lower than the cap but never a number above it, so a caller cannot buy a
    longer wait through an env var, a config mode, or a slow measurement.

    Why adapt downward at all, when the cap is the answer for a slow caller anyway: because a
    fast caller is real too. A shell script's re-call costs roughly one aua call (~3.9s
    measured) and no thinking, so holding it for the full 5s spends time to avoid a cheaper
    round trip. Sizing the ceiling from ``recall_cost_ms`` — wait no longer than asking again
    would have taken — collapses that case without touching the slow one.

    Cold start uses the cap, deliberately: with nothing measured there is nothing to be clever
    with, and the existing 5s behaviour is the tested default. This feature adds no new number
    to get wrong at install time.
    """
    perf = getattr(config, "perf", None)
    cap = max(0, int(cap_ms))
    floor = min(cap, max(0, int(getattr(perf, "wait_ceiling_min_ms", 2000) or 0)))

    pinned = os.environ.get(WAIT_CEILING_ENV)
    if pinned is not None:
        try:
            parsed = float(pinned)
            if not math.isfinite(parsed):
                raise ValueError("wait ceiling must be finite")
            return min(cap, max(0, int(parsed))), "pinned"
        except (OverflowError, TypeError, ValueError):
            logger.warning("ignoring non-numeric %s=%r", WAIT_CEILING_ENV, pinned)
    if str(getattr(perf, "wait_ceiling_mode", "adaptive") or "adaptive").lower() != "adaptive":
        return cap, "fixed"
    if profile is None or int(getattr(profile, "samples", 0) or 0) <= 0:
        return cap, "cold"
    try:
        raw_priced = float(getattr(profile, "recall_cost_ms", cap) or cap)
        priced = int(raw_priced) if math.isfinite(raw_priced) else cap
    except (OverflowError, TypeError, ValueError):
        priced = cap
    return max(floor, min(priced, cap)), "adaptive"


STABLE_DELAY_ENV = "AUA_STABLE_DELAY_MS"


def stable_delay_for(kind: str | None, config: Any) -> int:
    """The deliberate post-action pause for ``kind``, in milliseconds.

    Falls back to the ``default`` entry so a new action kind is never accidentally given a
    zero pause, which is the state that produced empty observations in the field.

    ``AUA_STABLE_DELAY_MS`` overrides every kind at once with a flat value. Config merges are
    per-key, which is right for tuning one slow action but wrong for a sweep: setting
    ``default`` alone would leave every named kind at its old value and the sweep would measure
    nothing. One env var moves the whole curve, including to a true zero baseline.
    """
    flat = os.environ.get(STABLE_DELAY_ENV)
    if flat is not None:
        try:
            return max(0, int(float(flat)))
        except (TypeError, ValueError):
            logger.warning("ignoring non-numeric %s=%r", STABLE_DELAY_ENV, flat)
    table = getattr(getattr(config, "perf", None), "stable_delay_ms", None) or {}
    if not isinstance(table, dict):
        return 0
    fallback = int(table.get("default", 0) or 0)
    if not kind:
        return max(0, fallback)
    return max(0, int(table.get(kind, fallback) or 0))


def is_provisioning_wait(kind: str | None) -> bool:
    """True when ``kind`` is a provisioning budget that the observation ceiling must not cap."""
    return bool(kind) and str(kind).lower() in _PROVISIONING_KINDS


@dataclass
class SettleProfiles:
    """EMA of observed settle times per action kind (tap/key/swipe/…)."""

    half_life: float = 4.0  # samples
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ema_ms: dict[str, float] = field(default_factory=dict)

    def observe(self, kind: str, ms: float) -> None:
        if not kind or ms < 0:
            return
        alpha = 2.0 / (self.half_life + 1.0)
        with self._lock:
            prev = self._ema_ms.get(kind)
            self._ema_ms[kind] = ms if prev is None else (alpha * ms + (1 - alpha) * prev)

    def budget(
        self,
        kind: str,
        *,
        default_settle_ms: int = 45,
        default_total_ms: int = 1100,
        total_max_ms: int = 1600,
    ) -> tuple[int, int]:
        """Return ``(settle_ms, total_timeout_ms)`` tuned from history.

        Only stretches the *deadline* from past transitions. Keeping ``settle_ms`` at the
        default avoids same-screen taps paying a transition-sized idle wait after a ripple.
        """
        with self._lock:
            ema = self._ema_ms.get(kind)
        if ema is None:
            return default_settle_ms, default_total_ms
        total = int(max(400, min(total_max_ms, ema * 1.8 + 100)))
        return default_settle_ms, total


class GateCache:
    """Memoize ``gate.decide`` for identical tree fingerprints within a session."""

    def __init__(self, *, max_size: int = 64) -> None:
        self._max = max_size
        self._lock = threading.Lock()
        self._items: dict[str, Any] = {}

    @staticmethod
    def key(
        elements: list[Element],
        *,
        package: str | None,
        activity: str | None,
    ) -> str:
        n = len(elements)
        labeled = sum(1 for e in elements if e.text or e.content_desc)
        clickable = sum(1 for e in elements if e.clickable)
        types = ",".join(sorted({(e.type or "")[:24] for e in elements[:40]}))
        raw = f"{package}|{activity}|{n}|{labeled}|{clickable}|{types}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._items.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key not in self._items and len(self._items) >= self._max:
                # Drop an arbitrary oldest-ish entry (insertion order in 3.7+).
                self._items.pop(next(iter(self._items)), None)
            self._items[key] = value


def elements_fingerprint(elements: list[Element]) -> str:
    """Cheap stable fingerprint for skip-unchanged memory."""
    parts: list[str] = []
    for e in elements:
        if getattr(e, "window", None) == "system":
            continue
        rid = (e.resource_id or "").split("/")[-1]
        label = (e.text or e.content_desc or "")[:40]
        parts.append(f"{e.id}:{rid}:{label}:{e.bounds}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def element_diff(prev: list[Element], curr: list[Element]) -> dict[str, Any]:
    """Token-cheap delta between two element lists, reported by stable identity.

    Ids in the returned lists are the same stable keys the payload publishes, so a reader can
    take one straight from `added` and act on it.
    """

    def _key(e: Element) -> str:
        sk = getattr(e, "stable_key", None)
        if sk:
            return str(sk)
        rid = e.resource_id or ""
        label = e.text or e.content_desc or ""
        return f"{e.type}|{rid}|{label}|{e.bounds}"

    prev_map = {_key(e): e for e in prev}
    curr_map = {_key(e): e for e in curr}
    # Report the identity, not the ordinal. `removed` is the case that forced it: those
    # elements are gone from the current frame, so a reader handed their ordinals could not
    # look any of them up — and the numbers it got back were *reused* by whatever occupies
    # that reading position now, which reads as "row 5 disappeared" about a row that is on
    # screen. The key that identified them in this very function is the honest answer.
    added = [k for k in curr_map if k not in prev_map]
    removed = [k for k in prev_map if k not in curr_map]
    changed: list[dict[str, Any]] = []
    for k, e in curr_map.items():
        old = prev_map.get(k)
        if old is None:
            continue
        delta: dict[str, Any] = {"id": k}
        if (old.text or "") != (e.text or ""):
            delta["text"] = {"from": old.text, "to": e.text}
        if (old.content_desc or "") != (e.content_desc or ""):
            delta["content_desc"] = {"from": old.content_desc, "to": e.content_desc}
        if old.bounds != e.bounds:
            delta["bounds"] = {"from": list(old.bounds), "to": list(e.bounds)}
        if len(delta) > 1:
            changed.append(delta)
    return {
        "added": added[:40],
        "removed": removed[:40],
        "changed": changed[:40],
        "prev_count": len(prev),
        "curr_count": len(curr),
    }
