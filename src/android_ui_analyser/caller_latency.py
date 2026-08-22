"""What the caller costs to think, measured, so a wait can be priced against a re-call.

The feature is exercised with a rounded synthetic trace whose gaps range from a few seconds to a
deliberately slow turn. Nothing is pending during those gaps: they model the time a caller takes to
read a response, decide, and write the next call. AUA knows both ends of every gap — the stamp it
wrote when its last call returned and the clock when this one started — so it need not discard that
signal.

What this module does NOT do: decide how long a wait may be. It measures the caller and
prices a notional re-call; `perf.wait_ceiling_ms` turns that into a ceiling, and
`perf.max_wait_ms` (5s) is a standing hard maximum the estimate can only sit *below*. The
numbers here therefore explain why a fast caller earns a shorter ceiling than 5s, not why any
caller should earn a longer one — an agent that needs longer makes another call.

Two reasons the number has to be on disk. Every CLI call is a new process, so an in-memory EMA
would learn one sample and forget it before it could be used. And it must be readable by the
warm daemon, which serves the wait but never sees the caller.

Its own file rather than ``SessionState``: that cursor is rewritten wholesale by every command,
so folding samples into it would drop whichever writer lost the race — and a dropped sample is
a ceiling that stays cold. Here the read-modify-write happens under an exclusive lock on a file
nothing else owns.

Keyed by *caller*, not by device. The gap is a property of whoever is generating the calls: the
same agent driving two emulators has one thinking speed, and splitting its samples per target
would just halve the history. Keying this way is also what makes two agents sharing one device
harmless — a shell script hammering the same emulator as a slow model must not drag that
model's ceiling down to its own. ``leases.resolve_owner`` already answers "which agent is
asking" without touching a device, and is the same discriminator the redundant-analyze lint
uses to avoid accusing you of someone else's command.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Beyond this a gap is not the caller generating: it is a human who walked away, a paused
#: debugger, a session resumed the next morning. The synthetic slow-turn fixture stays well below
#: this boundary, while an overnight pause cannot inflate the ceiling. It is
#: deliberately the same number as `cli._SAME_TURN_MS`, which already draws the boundary of "one
#: agent turn" for the redundant-analyze lint.
IDLE_GAP_MS = 120_000

#: A conservative synthetic cost for one AUA call on top of the caller's own gap. A re-call is
#: never just the gap — it is the gap plus another call — so a wait has to beat ``gap + this``
#: before shortening it is the cheaper option. In
#: practice this is the floor of the useful range: a caller with no think time at all still
#: pays this much to ask again, which is why a scripted caller's ceiling lands here rather than
#: at zero.
RECALL_TOOL_MS = 3_900

#: In samples, matching :class:`perf.SettleProfiles`. Short on purpose: a caller's speed changes
#: with the model, the harness and the size of the transcript, and a long memory would keep
#: pricing waits for a caller that is no longer on the other end.
HALF_LIFE_SAMPLES = 4.0

_SCHEMA = 1
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_key(value: str) -> str:
    """One filename component per caller. Owner labels carry pids and process names."""
    cleaned = _UNSAFE.sub("-", (value or "").strip()).strip("-")
    cleaned = cleaned or "anonymous"
    if len(cleaned) <= 120:
        return cleaned
    # Keeping only a prefix made two long owner identities share one latency history.  The
    # readable prefix remains useful in diagnostics; the digest makes the truncation injective
    # for practical purposes without changing existing short-key filenames.
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{cleaned[:103]}-{digest}"


def _finite_number(value: Any) -> float | None:
    """Return a finite numeric value, rejecting JSON's non-standard NaN/Infinity tokens."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


@dataclass(frozen=True)
class CallerProfile:
    """Smoothed caller think time, plus enough spread to price a wait against a re-call."""

    ema_ms: float | None = None
    spread_ms: float = 0.0
    samples: int = 0
    #: This turn's measured gap, reported even when it was not learned from.
    gap_ms: int | None = None
    #: Why this gap was excluded from the estimate: ``"idle"``, ``"clock"``, or None.
    gap_ignored: str | None = None

    @property
    def recall_cost_ms(self) -> int:
        """End-to-end cost of asking the caller to call again instead of waiting.

        Literally what a re-call takes: the caller's gap plus one more aua call. This is an
        *input* to the ceiling, not the ceiling: it is clamped into
        ``[perf.wait_ceiling_min_ms, perf.max_wait_ms]`` by :func:`perf.wait_ceiling_ms`, so a
        caller whose re-call costs 16s does not get a 16s wait — it gets the 5s cap, and the
        number is still reported so the caller can see why calling again is its cheaper move.

        The spread is
        added rather than trusting the mean, for the reason ``_learned_action_budget`` already
        gives about deadlines — a budget set from the average is by construction too short half
        the time, and here being too short costs a whole round trip *plus* an observation the
        app has already moved past, while being too long costs tool time nobody was using.
        """
        if self.ema_ms is None or not math.isfinite(self.ema_ms + self.spread_ms):
            return 0
        return int(round(self.ema_ms + self.spread_ms + RECALL_TOOL_MS))

    def as_response(self) -> dict[str, Any]:
        """The caller-facing form: integers, and only what a reader can act on.

        Sparse on purpose. This rides on *every* response, so a key whose value is None is pure
        token cost on the one path this project is most sensitive about — and a block of nulls
        also reads as "measured, and the answer is nothing", which is the opposite of what an
        absent key means.
        """
        out: dict[str, Any] = {}
        if self.gap_ms is not None:
            out["gap_ms"] = self.gap_ms
        if self.gap_ignored:
            out["gap_ignored"] = self.gap_ignored
        if self.ema_ms is not None:
            out["ema_ms"] = int(self.ema_ms)
            if self.spread_ms:
                out["spread_ms"] = int(self.spread_ms)
        if self.samples:
            out["samples"] = self.samples
        return out


@dataclass(frozen=True)
class CallerTurnFacts:
    """Everything one call learns about the caller before it does any work."""

    profile: CallerProfile
    #: Hierarchy fingerprint of the screen handed to the caller by its previous call.
    previous_fingerprint: str | None = None
    #: How old that observation is now — i.e. how long the caller spent holding it.
    previous_age_ms: int | None = None


class CallerLatencyStore:
    """Per-caller record of "when did my last call return, and how fast is this caller"."""

    def __init__(
        self,
        root: str | Path,
        key: str,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root).expanduser()
        self.key = _safe_key(key)
        # Wall clock, not monotonic: the two ends of a gap are measured by different processes.
        self._clock = clock

    @property
    def path(self) -> Path:
        return self.root / f"caller_{self.key}.json"

    # -- the two ends of a caller turn ------------------------------------

    def open_turn(self) -> CallerTurnFacts:
        """Measure the gap since this caller's last call returned, and fold it in."""
        now = self._clock()
        with _locked(self.root, self.key):
            record = self._read()
            gap_ms, ignored = _classify(record.get("ended_at"), now)
            if gap_ms is not None and ignored is None:
                _learn(record, gap_ms)
            record["last_gap_ms"] = gap_ms
            self._write(record)
        fp = record.get("fingerprint")
        fp_at = record.get("fingerprint_at")
        finite_fp_at = _finite_number(fp_at)
        age_ms = (
            int((now - finite_fp_at) * 1000.0)
            if finite_fp_at is not None and now >= finite_fp_at
            else None
        )
        return CallerTurnFacts(
            profile=_profile(record, gap_ms=gap_ms, gap_ignored=ignored),
            previous_fingerprint=str(fp) if fp else None,
            previous_age_ms=age_ms,
        )

    def peek_turn(self) -> CallerTurnFacts:
        """The same facts as :meth:`open_turn`, without opening one or learning from it.

        The warm daemon must never open a caller turn — a daemon round trip is aua's own
        transport, and counting it would halve every gap — but it *is* the process that reads
        the screen, so it is the only one that can compare what is on the device now against
        what the caller was last handed. Both ends of the turn are stamped into this record by
        the CLI process, ``open_turn`` included, so this call needs no lock and no write: the
        gap it reports is the one the CLI already measured and classified for this turn.
        """
        now = self._clock()
        record = self._read()
        raw_gap = record.get("last_gap_ms")
        gap_ms = int(raw_gap) if isinstance(raw_gap, int) and not isinstance(raw_gap, bool) else None
        finite_fp_at = _finite_number(record.get("fingerprint_at"))
        fp = record.get("fingerprint")
        return CallerTurnFacts(
            profile=_profile(record, gap_ms=gap_ms, gap_ignored=_ignored_reason(gap_ms)),
            previous_fingerprint=str(fp) if fp else None,
            previous_age_ms=(
                int((now - finite_fp_at) * 1000.0)
                if finite_fp_at is not None and now >= finite_fp_at
                else None
            ),
        )

    def close_turn(self, fingerprint: str | None = None) -> None:
        """Stamp when this call returned, and which screen it handed back.

        The fingerprint is what makes "your screen is gone" answerable on the next call without
        another device read: the next observation already computes one, so the comparison is
        free where a re-dump would not be.
        """
        now = self._clock()
        with _locked(self.root, self.key):
            record = self._read()
            record["ended_at"] = now
            if fingerprint:
                record["fingerprint"] = fingerprint
                record["fingerprint_at"] = now
            self._write(record)

    def profile(self) -> CallerProfile:
        """The stored estimate, without opening a turn (the daemon's read path)."""
        record = self._read()
        return _profile(record, gap_ms=None, gap_ignored=None)

    # -- storage ----------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema": _SCHEMA}
        except (OSError, ValueError) as exc:
            # A ceiling is an optimisation. Never fail a caller's command over its bookkeeping;
            # an unreadable history is indistinguishable from no history, which is a state this
            # already handles by erring long.
            logger.debug("caller latency file unreadable (%s); starting over", exc)
            return {"schema": _SCHEMA}
        return raw if isinstance(raw, dict) else {"schema": _SCHEMA}

    def _write(self, record: dict[str, Any]) -> None:
        record["schema"] = _SCHEMA
        from .atomic import atomic_write_text

        with contextlib.suppress(OSError):
            atomic_write_text(self.path, json.dumps(record, ensure_ascii=False))


def _ignored_reason(gap_ms: int | None) -> str | None:
    """Why a measured gap must not be believed, or None when it can be.

    Its own function because two readers need the same verdict from different inputs:
    :func:`_classify` has the raw stamps, while :meth:`CallerLatencyStore.peek_turn` has only
    the gap this turn already recorded. A second copy of the rule would be a second place for
    "was anyone actually driving" to drift.
    """
    if gap_ms is None:
        return None
    if gap_ms < 0:
        # An NTP correction or a restored snapshot. Believing it would poison the EMA with a
        # negative think time, and there is no way to tell how much of the gap was real.
        return "clock"
    if gap_ms > IDLE_GAP_MS:
        return "idle"
    return None


def _classify(ended_at: Any, now: float) -> tuple[int | None, str | None]:
    """``(gap_ms, why_it_was_ignored)`` for the interval since the last call returned."""
    finite_ended_at = _finite_number(ended_at)
    if finite_ended_at is None or not math.isfinite(now):
        return None, None
    gap_ms = int(round((now - finite_ended_at) * 1000.0))
    return gap_ms, _ignored_reason(gap_ms)


def _learn(record: dict[str, Any], gap_ms: int) -> None:
    """Fold one accepted gap into the EMA and its spread, in place."""
    alpha = 2.0 / (HALF_LIFE_SAMPLES + 1.0)
    previous = _finite_number(record.get("ema_ms"))
    if previous is not None:
        deviation = abs(gap_ms - previous)
        prior_spread = _finite_number(record.get("spread_ms")) or 0.0
        record["ema_ms"] = alpha * gap_ms + (1.0 - alpha) * previous
        record["spread_ms"] = alpha * deviation + (1.0 - alpha) * prior_spread
    else:
        record["ema_ms"] = float(gap_ms)
        record["spread_ms"] = 0.0
    samples = record.get("samples")
    record["samples"] = (int(samples) if isinstance(samples, int) else 0) + 1


def _profile(
    record: dict[str, Any], *, gap_ms: int | None, gap_ignored: str | None
) -> CallerProfile:
    ema = _finite_number(record.get("ema_ms"))
    spread = _finite_number(record.get("spread_ms"))
    samples = record.get("samples")
    return CallerProfile(
        ema_ms=ema,
        spread_ms=spread or 0.0,
        samples=int(samples) if isinstance(samples, int) else 0,
        gap_ms=gap_ms,
        gap_ignored=gap_ignored,
    )


@contextlib.contextmanager
def _locked(root: Path, key: str) -> Any:
    """Exclusive lock around one caller's read-modify-write.

    Two aua processes for one caller are legitimate — a background wait and a foreground
    check — and both fold samples into the same estimate. ``atomic_write_text`` alone keeps the
    file readable but still loses the loser's sample, so the whole read-modify-write is held,
    not just the publish.
    """
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / f"caller_{key}.lock").open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - best effort off Unix
            pass
        yield
    finally:
        with contextlib.suppress(Exception):
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
