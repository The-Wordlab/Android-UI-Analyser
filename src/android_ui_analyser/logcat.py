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
from collections.abc import Sequence
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
_DETAIL_RE = re.compile(
    r"^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+"
    r"(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<priority>[VDIWEF])\s+"
    r"(?P<tag>[^:]+):\s?(?P<message>.*)$"
)
_CRASH_MARKERS = (
    "fatal exception",
    "anr in ",
    "fatal signal",
    "am_crash",
    "has crashed",
)
_STACK_MARKERS = (
    "process:",
    "java.lang.",
    "kotlin.",
    "caused by:",
    "suppressed:",
    "at ",
    "... ",
    "reason:",
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


def extract_crash_evidence(
    raw: str,
    *,
    app_id: str | None = None,
    limit: int = 60,
) -> dict[str, Any]:
    """Extract one bounded fatal/ANR/error block from a diagnostic log window.

    A line-only ``grep FATAL`` drops the exception type and stack that explain the crash. This
    keeps the surrounding same-process error block, prefers a block naming *app_id*, and falls
    back to error-priority lines when a platform uses a different crash marker. The caller is
    expected to pass an already bounded time window (normally ``last-action``).
    """

    source = [line for line in raw.splitlines() if line.strip()]
    records: list[dict[str, str] | None] = []
    for line in source:
        match = _DETAIL_RE.match(line.lstrip())
        records.append(match.groupdict() if match else None)

    anchors = [
        index
        for index, line in enumerate(source)
        if any(marker in line.casefold() for marker in _CRASH_MARKERS)
    ]
    selected: set[int] = set()
    selected_kind = "none"
    app_folded = (app_id or "").casefold()

    if anchors:
        blocks: list[tuple[set[int], str, bool]] = []
        for anchor in anchors:
            anchor_record = records[anchor]
            anchor_pid = anchor_record.get("pid") if anchor_record else None
            anchor_tag = anchor_record.get("tag") if anchor_record else None
            anchor_text = source[anchor].casefold()
            kind = (
                "anr"
                if "anr in " in anchor_text
                else "fatal"
                if "fatal exception" in anchor_text or "fatal signal" in anchor_text
                else "crash"
            )
            block: set[int] = {anchor}
            start = max(0, anchor - 2)
            end = min(len(source), anchor + 81)
            for index in range(start, end):
                line = source[index]
                folded = line.casefold()
                record = records[index]
                priority = record.get("priority") if record else None
                pid = record.get("pid") if record else None
                tag = record.get("tag") if record else None
                message = (record.get("message") or line if record else line).strip().casefold()
                same_process = bool(anchor_pid and pid == anchor_pid)
                same_tag = bool(anchor_tag and tag == anchor_tag)
                if (
                    index == anchor
                    or (app_folded and app_folded in folded)
                    or (
                        priority in {"E", "F"}
                        and (same_process or (anchor_pid is None and same_tag))
                    )
                    or (same_process and index <= anchor + 40)
                    or (
                        index <= anchor + 40
                        and any(message.startswith(marker) for marker in _STACK_MARKERS)
                    )
                ):
                    block.add(index)
            names_app = bool(app_folded and any(app_folded in source[i].casefold() for i in block))
            blocks.append((block, kind, names_app))

        relevant = [block for block in blocks if block[2]] or blocks
        for block, kind, _names_app in relevant:
            selected.update(block)
            if (
                kind == "fatal"
                or selected_kind == "none"
                or (kind == "anr" and selected_kind == "crash")
            ):
                selected_kind = kind
    else:
        error_indices = {
            index
            for index, record in enumerate(records)
            if record is not None and record.get("priority") in {"E", "F"}
        }
        app_pids = {
            record["pid"]
            for index, record in enumerate(records)
            if record is not None and index in error_indices and app_folded in source[index].casefold()
        }
        if app_pids:
            selected = {
                index
                for index in error_indices
                if (record := records[index]) is not None and record.get("pid") in app_pids
            }
        else:
            selected = error_indices
        if selected:
            selected_kind = "error"

    ordered = sorted(selected)
    bounded_limit = max(1, int(limit))
    truncated = len(ordered) > bounded_limit
    if truncated:
        head_count = max(1, (bounded_limit * 2) // 3)
        tail_count = bounded_limit - head_count
        ordered = ordered[:head_count] + (ordered[-tail_count:] if tail_count else [])
    lines = [source[index] for index in ordered]
    return {
        "kind": selected_kind,
        "lines": lines,
        "count": len(lines),
        "total_count": len(selected),
        "truncated": truncated,
        "matched_app": bool(app_folded and any(app_folded in line.casefold() for line in lines)),
    }


# --------------------------------------------------------------------------- action log digest

# Priority set attached to a folded action observation by default. ``V`` and ``I`` are
# deliberately absent, from measurement rather than taste: in one real app's cold-launch window
# all 113 ``I`` lines came from a third-party HTTP client, an attribution SDK, the advertising-id
# client, or the ART runtime, and not one carried app logic. ``D`` is kept because that is where
# an app writes its own breadcrumbs — dropping it would leave the digest technically cheaper and
# practically useless.
DEFAULT_LEVELS = "DWEF"

# Tags that are noise *by construction*: platform framework internals, and third-party SDKs that
# log per request, per frame, or per attribution event. Deliberately generic — an app's own tags
# are never listed, so they always survive the filter, and this file never learns the name of a
# real app's logger. Matched as a case-insensitive prefix so one entry covers the versioned
# (``AppsFlyer_6.17.6``) and namespaced (``TRuntime.CctTransportBackend``) shapes SDKs actually
# emit.
DEFAULT_DENY_TAG_PREFIXES: tuple[str, ...] = (
    "AdvertisingIdClient",
    "AppsFlyer",
    "ApplicationLoaders",
    "Choreographer",
    "Chucker",
    "CompatChangeReporter",
    "DesktopExperienceFlags",
    "FirebaseSessions",
    "HWUI",
    "ImeTracker",
    "InsetsController",
    "LeakCanary",
    "OkHttp",
    "StrictMode",
    "TRuntime",
    "VRI[",
    "ViewRootImpl",
    "WindowOnBackDispatcher",
    "ashmem",
    "com.facebook.",
    "libEGL",
    "nativeloader",
)

# Note on what is deliberately NOT here. `System` would be the single biggest win by line count
# on some apps, and it is refused: `System.out` and `System.err` are how an app prints, so a
# `System` prefix would delete real app output to remove framework chatter. Payment and billing
# SDK tags are likewise kept — a failed purchase is exactly the kind of thing the screen will not
# tell you. Both stay bounded by the per-tag cap instead, and an app's own chatty logger belongs
# in the user's `logs.deny_tags`, never in this file.

# Below this length a tag is too short to be a meaningful suffix of a package name, so the
# runtime-tag rule stops applying. Without the floor an app whose own tag happened to be a
# two-letter suffix of its package ("es" in "…example.notes") would be silently deleted.
_RUNTIME_TAG_MIN = 8

# One logcat line has no length limit worth relying on. Measured on a real app, a single `I`
# line carried a 145 KB HTTP response body — one such line is seven times the entire line
# budget this digest is supposed to enforce, and an app that logs a payload under its *own* tag
# defeats every tag filter. So the cap is per line as well as per window, and a clipped line is
# marked as clipped: a reader must never mistake a truncated message for a complete one.
DEFAULT_MAX_LINE_CHARS = 300


def _is_runtime_tag(tag: str, app_id: str | None) -> bool:
    """Whether *tag* is ART/libcore logging under the app's own (truncated) process name.

    The Android runtime logs GC, JIT, lock-verification and hidden-api messages under the
    process name truncated to fit logcat's tag field, so the tag is a *suffix* of the package.
    Deriving it beats listing it: a hardcoded list would have to name real applications, and
    this repository is public.
    """
    if not app_id or len(tag) < _RUNTIME_TAG_MIN:
        return False
    return app_id == tag or app_id.endswith(tag)


def _clip(line: str, limit: int) -> str:
    """Bound one line, saying so where the cut is rather than trailing off silently."""
    if len(line) <= limit:
        return line
    return f"{line[:limit]}…[+{len(line) - limit} chars]"


def digest_app_logs(
    raw: str,
    *,
    app_id: str | None = None,
    levels: str = DEFAULT_LEVELS,
    deny_tag_prefixes: Sequence[str] = DEFAULT_DENY_TAG_PREFIXES,
    limit: int = 20,
    per_tag: int = 5,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> dict[str, Any]:
    """Reduce one action's log window to something affordable to attach to every observation.

    Three filters, in the order of how much they remove. Priority first (a level set, not a
    floor: ``I`` is noisier than ``D`` on Android, so a floor would keep the wrong half). Then
    known-noisy tags. Then a per-tag cap, so a single chatty logger cannot spend the whole line
    budget and push out the one error that explains the failure — measured on a real launch,
    44 near-identical config lines would otherwise have filled a 20-line cap by themselves.

    ``F`` is always included whatever *levels* asks for: a caller narrowing the filter must not
    be able to hide the line that explains a crash.

    *max_line_chars* bounds each line as well, because a line budget alone does not bound the
    output: a single measured line held a 145 KB response body.
    """

    wanted = {ch.upper() for ch in levels if ch.strip()} | {"F"}
    deny = tuple(prefix.casefold() for prefix in deny_tag_prefixes)

    kept: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _DETAIL_RE.match(line.lstrip())
        if match is None:
            # Buffer separators ("--------- beginning of main") and continuation lines carry no
            # priority. They are structure, not app output, so they are not reported as such.
            continue
        if match.group("priority") not in wanted:
            continue
        tag = (match.group("tag") or "").strip()
        folded = tag.casefold()
        if any(folded.startswith(prefix) for prefix in deny):
            continue
        if _is_runtime_tag(tag, app_id):
            continue
        kept.append(line)

    total = len(kept)
    if per_tag > 0:
        seen: dict[str, int] = {}
        capped: list[str] = []
        for line in kept:
            tag = line_tag(line) or ""
            seen[tag] = seen.get(tag, 0) + 1
            if seen[tag] <= per_tag:
                capped.append(line)
        kept = capped

    bounded = max(1, int(limit))
    truncated = len(kept) > bounded
    if truncated:
        # Head and tail, as `extract_crash_evidence` does: the first thing the app said after
        # the action and the last thing it said before we looked are both load-bearing, and a
        # plain head slice throws the second one away.
        head = max(1, (bounded * 2) // 3)
        tail = bounded - head
        kept = kept[:head] + (kept[-tail:] if tail else [])

    if max_line_chars > 0:
        kept = [_clip(line, max_line_chars) for line in kept]

    return {
        # Android's own priority order, not alphabetical: `DWEF` is how a reader thinks about
        # this filter, and `DEFW` would look like a different, wrong answer.
        "levels": "".join(ch for ch in "VDIWEF" if ch in wanted),
        "lines": kept,
        "count": len(kept),
        "total_count": total,
        "omitted": max(0, total - len(kept)),
        "truncated": truncated,
    }
