"""Logcat marks + filtered dump helpers.

Marks are **device-clock** bookmarks persisted under the cache dir so an agent can dump
only the lines since the last action (or a named mark) without re-reading the whole
buffer.

The device clock is the whole point. Every logcat line is stamped by the device, and on
an emulator the device wall clock drifts seconds away from the host's — measured at
+9.4s on a plain AVD. A host-derived boundary therefore lands in the *future* relative
to the log, so ``mark`` → act → ``--since <mark>`` silently returns nothing and reads
exactly like "the app never logged anything". Windows are computed in device time and
handed to logcat's own ``-T`` filter; the host clock only ever appears as a measured
skew, reported back so drift is visible instead of silent.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# threadtime: "01-15 10:30:45.123  1234  5678 I Tag: msg"
_THREADTIME = re.compile(
    r"^(?P<mon>\d{2})-(?P<day>\d{2})\s+"
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})\b"
)
# epoch / usec: "1700000000.123  ..." or "1700000000123 ..."
_EPOCH = re.compile(r"^(?P<sec>\d{10})(?:\.(?P<frac>\d+))?(?:\s|$)")
_EPOCH_MS = re.compile(r"^(?P<ms>\d{13})\b")

_TAG_RE = re.compile(
    r"^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+\d+\s+\d+\s+[VDIWEF]\s+(?P<tag>[^:]+):"
)


def marks_path(cache_dir: str | Path, serial: str) -> Path:
    safe = str(serial).replace(":", "_")
    return Path(cache_dir).expanduser() / f"logcat_marks_{safe}.json"


def clock_path(cache_dir: str | Path, serial: str) -> Path:
    """Sidecar for the measured host↔device skew — deliberately NOT the marks file.

    A reserved key inside ``marks`` would be resolvable as ``--since <that key>``.
    """
    safe = str(serial).replace(":", "_")
    return Path(cache_dir).expanduser() / f"logcat_clock_{safe}.json"


def load_marks(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_marks(path: Path, marks: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Re-measure the skew at most this often. Any single window stays self-consistent as long
# as mark and dump share one skew, and wall-clock drift over a minute is sub-millisecond.
SKEW_TTL_MS = 60_000


@dataclass(frozen=True)
class DeviceClock:
    """The device wall clock, expressed as a measured offset from the host's.

    ``skew_ms`` is ``host - device``: positive means the host runs ahead. ``measured`` is
    ``False`` when the device clock could not be read, in which case the skew is assumed
    zero and every timestamp derived here is host time — surfaced as ``clock: "host"`` so
    a caller can tell an unverified window from a verified one.
    """

    skew_ms: int = 0
    measured: bool = False

    @property
    def name(self) -> str:
        return "device" if self.measured else "host"

    def now_ms(self) -> int:
        return self.to_device(int(time.time() * 1000))

    def to_device(self, host_ms: int) -> int:
        return int(host_ms) - self.skew_ms


def measure_skew(device: Any) -> int | None:
    """Measured ``host - device`` ms, or ``None`` when the device clock is unreadable.

    The host is sampled either side of the round-trip and the midpoint used, so adb
    latency lands on both halves instead of inflating the skew.
    """
    before = time.time() * 1000
    try:
        device_ms = device.get_clock_ms()
    except Exception:
        return None
    after = time.time() * 1000
    if device_ms is None:
        return None
    return int((before + after) / 2) - int(device_ms)


def load_clock(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_clock(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def resolve_clock(
    device: Any,
    cache_dir: str | Path,
    *,
    force: bool = False,
    ttl_ms: int = SKEW_TTL_MS,
) -> DeviceClock:
    """The device clock for *device*, re-using a cached skew within *ttl_ms*.

    Caching is what keeps this affordable: ``last-action`` is stamped after every single
    action, and an adb round-trip per tap would be a visible latency regression on a tool
    that sells itself on tens of milliseconds. Each ``aua`` run is a fresh process, so the
    cache has to live on disk to help the CLI at all.
    """
    path = clock_path(cache_dir, getattr(device, "serial", "unknown"))
    now = int(time.time() * 1000)
    if not force:
        cached = load_clock(path)
        if cached and isinstance(cached.get("skew_ms"), int):
            age = now - int(cached.get("measured_host_ms") or 0)
            if 0 <= age <= ttl_ms:
                return DeviceClock(skew_ms=int(cached["skew_ms"]), measured=True)
    skew = measure_skew(device)
    if skew is None:
        return DeviceClock()
    save_clock(path, {"skew_ms": skew, "measured_host_ms": now})
    return DeviceClock(skew_ms=skew, measured=True)


def make_mark(*, unix_ms: int | None = None, clock: DeviceClock | None = None) -> dict[str, Any]:
    """A mark entry. ``unix_ms`` is DEVICE time; ``host_unix_ms``/``skew_ms`` expose drift."""
    dc = clock or DeviceClock()
    host_ms = int(time.time() * 1000)
    ms = int(unix_ms if unix_ms is not None else dc.to_device(host_ms))
    iso = datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")
    entry: dict[str, Any] = {"unix_ms": ms, "iso": iso, "clock": dc.name}
    if dc.measured:
        entry["host_unix_ms"] = host_ms
        entry["skew_ms"] = dc.skew_ms
    return entry


def set_mark(
    cache_dir: str | Path,
    serial: str,
    name: str = "default",
    *,
    clock: DeviceClock | None = None,
) -> dict[str, Any]:
    """Persist a named mark; returns ``{name, unix_ms, iso, clock, …}``."""
    path = marks_path(cache_dir, serial)
    marks = load_marks(path)
    entry = make_mark(clock=clock)
    marks[name] = entry
    save_marks(path, marks)
    return {"name": name, **entry}


def mark_device_ms(entry: dict[str, Any], *, skew_ms: int = 0) -> int:
    """A stored mark as device ms, converting entries written before marks moved clocks."""
    ms = int(entry["unix_ms"])
    return ms if entry.get("clock") == "device" else ms - skew_ms


def resolve_since_ms(
    marks: dict[str, dict[str, Any]],
    since: str | None,
    *,
    clock: DeviceClock | None = None,
    default_window_ms: int = 30_000,
) -> tuple[int, str]:
    """Resolve a since token to ``(device_unix_ms, label)``.

    Every form is anchored to the device clock: durations count back from *device* now,
    and a bare epoch is taken as device ms (it is compared against device-stamped lines).
    Default: ``last-action`` mark if present, else now − 30s.
    """
    dc = clock or DeviceClock()
    now_ms = dc.now_ms()
    if since is None or since == "":
        if "last-action" in marks:
            return mark_device_ms(marks["last-action"], skew_ms=dc.skew_ms), "last-action"
        return now_ms - default_window_ms, "30s"

    if since in marks:
        return mark_device_ms(marks[since], skew_ms=dc.skew_ms), since

    # bare milliseconds
    if since.isdigit():
        return int(since), since

    # duration suffixes: 30s / 5m
    m = re.fullmatch(r"(\d+)(s|m|ms)?", since.strip())
    if m:
        n = int(m.group(1))
        unit = m.group(2) or "s"
        mult = {"ms": 1, "s": 1000, "m": 60_000}[unit]
        return now_ms - n * mult, since

    raise KeyError(since)


def line_unix_ms(
    line: str, *, ref_year: int | None = None, tz_offset_minutes: int = 0
) -> int | None:
    """Best-effort parse of a logcat line's timestamp → unix ms (or None).

    ``-v threadtime`` carries neither a year nor a zone and is stamped in the *device's*
    local time, so ``tz_offset_minutes`` (the device's UTC offset) is what makes the result
    a real epoch. Epoch formats are absolute and ignore it.
    """
    s = line.lstrip()
    m = _EPOCH_MS.match(s)
    if m:
        return int(m.group("ms"))
    m = _EPOCH.match(s)
    if m:
        frac = (m.group("frac") or "0")[:3].ljust(3, "0")
        return int(m.group("sec")) * 1000 + int(frac)

    m = _THREADTIME.match(s)
    if not m:
        return None
    year = ref_year if ref_year is not None else datetime.now(tz=UTC).year
    try:
        dt = datetime(
            year,
            int(m.group("mon")),
            int(m.group("day")),
            int(m.group("h")),
            int(m.group("m")),
            int(m.group("s")),
            int(m.group("ms")) * 1000,
            tzinfo=UTC,
        )
    except ValueError:
        return None
    return int(dt.timestamp() * 1000) - tz_offset_minutes * 60_000


def line_tag(line: str) -> str | None:
    m = _TAG_RE.match(line.lstrip())
    if m:
        return m.group("tag").strip()
    # fallback: "I Tag: msg" after pid/tid columns
    parts = line.split(None, 5)
    if len(parts) >= 6 and parts[4] in "VDIWEF" and ":" in parts[5]:
        return parts[5].split(":", 1)[0].strip()
    return None


def filter_logcat(
    raw: str,
    *,
    since_ms: int | None = None,
    grep: str | None = None,
    tag: str | None = None,
    lines: int | None = None,
    ref_year: int | None = None,
    tz_offset_minutes: int = 0,
) -> list[str]:
    """Split *raw* into lines and apply since / grep / tag / lines filters.

    ``since_ms`` is device ms and ``tz_offset_minutes`` the device's UTC offset — both
    sides of the comparison have to sit on the device's clock.
    """
    out: list[str] = []
    gre = re.compile(grep) if grep else None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if since_ms is not None:
            ts = line_unix_ms(line, ref_year=ref_year, tz_offset_minutes=tz_offset_minutes)
            if ts is not None and ts < since_ms:
                continue
            # lines without a parseable timestamp are kept (header / continuation)
        if tag is not None:
            lt = line_tag(line)
            if lt is None or lt != tag:
                continue
        if gre is not None and gre.search(line) is None:
            continue
        out.append(line)
    if lines is not None and lines >= 0:
        out = out[-lines:]
    return out
