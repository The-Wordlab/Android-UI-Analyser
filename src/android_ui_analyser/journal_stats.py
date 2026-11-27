"""How AUA has been behaving on this host: one summary over every journal file."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .journal import journal_dir


def _surface(record: dict[str, Any], fallback: str) -> str:
    platform = str(record.get("platform") or fallback or "android")
    if platform != "web":
        return platform
    target = str(record.get("target_id") or record.get("serial") or "")
    if target == "existing-chrome":
        return "web:extension"
    if target.startswith("cdp:"):
        return "web:cdp"
    return "web:dedicated"


def _platform_of(path: Path) -> str:
    name = path.name
    return name.split("@")[1] if name.startswith("@") and name.count("@") >= 2 else "android"


def _records(cache_dir: str | Path, since: float | None) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for path in sorted(journal_dir(cache_dir).glob("*.jsonl")):
        if path.name.endswith(".details.jsonl"):
            continue
        platform = _platform_of(path)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if since is not None and float(record.get("ts") or 0) < since:
                    continue
                found.append((_surface(record, platform), record))
    return found


def _group(records: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [record for record in records if record.get("ok") is False]
    durations = sorted(
        float(record["duration_ms"])
        for record in records
        if isinstance(record.get("duration_ms"), (int, float))
    )
    by_command: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if isinstance(record.get("duration_ms"), (int, float)):
            by_command[str(record.get("cmd"))].append(float(record["duration_ms"]))
    slowest = sorted(
        ((statistics.median(values), command, len(values))
         for command, values in by_command.items() if len(values) >= 5),
        reverse=True,
    )[:5]
    usage = Counter(
        str((record.get("error") or {}).get("message") or "")
        for record in failed
        if record.get("cmd") == "cli_usage_error"
    )

    def percentile(share: float) -> int | None:
        if not durations:
            return None
        return round(durations[min(len(durations) - 1, int(share * len(durations)))])

    return {
        "calls": len(records),
        "failed": len(failed),
        "failure_rate": round(len(failed) / len(records), 3) if records else 0.0,
        "duration_ms": {"p50": percentile(0.5), "p90": percentile(0.9)},
        "session_starts": sum(1 for record in records if record.get("cmd") == "session_start"),
        "help_calls": sum(1 for record in records if record.get("cmd") == "cli_help"),
        "top_errors": Counter(
            str((record.get("error") or {}).get("code") or "none") for record in failed
        ).most_common(8),
        "top_failing_commands": Counter(str(record.get("cmd")) for record in failed).most_common(8),
        "slowest_commands_ms": [
            {"cmd": command, "median": round(median), "calls": count}
            for median, command, count in slowest
        ],
        "usage_errors": usage.most_common(5),
    }


def summarize(
    cache_dir: str | Path, *, platform: str | None = None, days: float | None = None
) -> dict[str, Any]:
    """Calls, failures, error codes and slow commands per surface (web split by attachment)."""

    since = time.time() - days * 86400 if days else None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stamps = []
    for surface, record in _records(cache_dir, since):
        if platform and surface.split(":")[0] != platform:
            continue
        grouped[surface].append(record)
        if record.get("ts"):
            stamps.append(float(record["ts"]))
    every = [record for records in grouped.values() for record in records]
    return {
        "ok": True,
        "action": "journal-stats",
        "journal": str(journal_dir(cache_dir)),
        "from": time.strftime("%Y-%m-%d", time.localtime(min(stamps))) if stamps else None,
        "to": time.strftime("%Y-%m-%d", time.localtime(max(stamps))) if stamps else None,
        "total": _group(every),
        "surfaces": {surface: _group(records) for surface, records in sorted(grouped.items())},
    }


__all__ = ["summarize"]
