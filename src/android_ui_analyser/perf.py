"""Latency helpers: hierarchy prefetch, settle profiles, element diffs, gate cache."""

from __future__ import annotations

import hashlib
import logging
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
    """Token-cheap delta between two element lists (by stable_key / id fallback)."""

    def _key(e: Element) -> str:
        sk = getattr(e, "stable_key", None)
        if sk:
            return str(sk)
        rid = e.resource_id or ""
        label = e.text or e.content_desc or ""
        return f"{e.type}|{rid}|{label}|{e.bounds}"

    prev_map = {_key(e): e for e in prev}
    curr_map = {_key(e): e for e in curr}
    added = [e.id for k, e in curr_map.items() if k not in prev_map]
    removed = [e.id for k, e in prev_map.items() if k not in curr_map]
    changed: list[dict[str, Any]] = []
    for k, e in curr_map.items():
        old = prev_map.get(k)
        if old is None:
            continue
        delta: dict[str, Any] = {"id": e.id}
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
