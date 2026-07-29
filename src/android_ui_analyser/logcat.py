"""Logcat marks + filtered dump helpers.

Marks are host-time bookmarks persisted under the cache dir so an agent can dump
only the lines since the last action (or a named mark) without re-reading the
whole buffer.
"""

from __future__ import annotations

import json
import re
import time
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


def make_mark(*, unix_ms: int | None = None) -> dict[str, Any]:
    ms = int(unix_ms if unix_ms is not None else time.time() * 1000)
    iso = datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")
    return {"unix_ms": ms, "iso": iso}


def set_mark(
    cache_dir: str | Path,
    serial: str,
    name: str = "default",
) -> dict[str, Any]:
    """Persist a named mark; returns ``{name, unix_ms, iso}``."""
    path = marks_path(cache_dir, serial)
    marks = load_marks(path)
    entry = make_mark()
    marks[name] = entry
    save_marks(path, marks)
    return {"name": name, **entry}


def resolve_since_ms(
    marks: dict[str, dict[str, Any]],
    since: str | None,
    *,
    default_window_ms: int = 30_000,
) -> tuple[int, str]:
    """Resolve a since token to ``(unix_ms, label)``.

    Default: ``last-action`` mark if present, else now − 30s.
    """
    now_ms = int(time.time() * 1000)
    if since is None or since == "":
        if "last-action" in marks:
            entry = marks["last-action"]
            return int(entry["unix_ms"]), "last-action"
        return now_ms - default_window_ms, "30s"

    if since in marks:
        return int(marks[since]["unix_ms"]), since

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


def line_unix_ms(line: str, *, ref_year: int | None = None) -> int | None:
    """Best-effort parse of a logcat line's timestamp → unix ms (or None)."""
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
    return int(dt.timestamp() * 1000)


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
) -> list[str]:
    """Split *raw* into lines and apply since / grep / tag / lines filters."""
    out: list[str] = []
    gre = re.compile(grep) if grep else None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if since_ms is not None:
            ts = line_unix_ms(line, ref_year=ref_year)
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
