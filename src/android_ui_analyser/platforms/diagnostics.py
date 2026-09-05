"""Platform-neutral diagnostic evidence returned by ``device.logs`` adapters.

Native log formats belong to their adapters.  Shared Engine code receives this module's
normalized events and windows, so waiting on a message, filtering a public dump, and folding a
small app-log digest into an observation do not need to know whether the source was Android
logcat, an iOS unified log, or a test adapter.

``display_text`` deliberately preserves the adapter's established public rendering.  The
normalized fields are for decisions; the display text is for compatibility and human evidence.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .identity import TargetRef

DIAGNOSTIC_LEVEL_CODES = frozenset("VDIWEF")


class UnknownDiagnosticMark(KeyError):
    """A named adapter-clock cursor was not found for this target."""

    def __init__(self, name: str, known: Sequence[str] = ()) -> None:
        super().__init__(name)
        self.name = name
        self.known = tuple(known)


class DiagnosticLevel(StrEnum):
    """Severity independent of a platform's native priority spelling."""

    VERBOSE = "verbose"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    @property
    def compatibility_code(self) -> str:
        """The long-standing one-letter AUA spelling used by app-log preferences."""

        return {
            DiagnosticLevel.VERBOSE: "V",
            DiagnosticLevel.DEBUG: "D",
            DiagnosticLevel.INFO: "I",
            DiagnosticLevel.WARNING: "W",
            DiagnosticLevel.ERROR: "E",
            DiagnosticLevel.FATAL: "F",
        }[self]

    @classmethod
    def from_compatibility_code(cls, code: str) -> DiagnosticLevel | None:
        return {
            "V": cls.VERBOSE,
            "D": cls.DEBUG,
            "I": cls.INFO,
            "W": cls.WARNING,
            "E": cls.ERROR,
            "F": cls.FATAL,
        }.get(str(code).strip().upper())


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    """One diagnostic record after adapter-owned parsing and attribution."""

    message: str
    level: DiagnosticLevel | None = None
    source: str | None = None
    timestamp_ms: int | None = None
    process_id: str | None = None
    thread_id: str | None = None
    app_id: str | None = None
    display_text: str | None = None
    hidden_by_default: bool = False

    @property
    def text(self) -> str:
        """Stable human rendering, preserving the native adapter's existing output."""

        return self.display_text if self.display_text is not None else self.message


@dataclass(frozen=True, slots=True)
class CrashEvidence:
    """A bounded, adapter-attributed fatal/ANR/error evidence block."""

    kind: str = "none"
    events: tuple[DiagnosticEvent, ...] = ()
    total_count: int = 0
    truncated: bool = False
    matched_app: bool = False

    def as_dict(self) -> dict[str, Any]:
        lines = [event.text for event in self.events]
        return {
            "kind": self.kind,
            "lines": lines,
            "count": len(lines),
            "total_count": self.total_count,
            "truncated": self.truncated,
            "matched_app": self.matched_app,
        }


HiddenSourcePredicate = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class DiagnosticSourcePolicy:
    """Adapter-owned defaults for source/tag filtering.

    Prefixes are exposed because the existing ``logcat prefs`` response lists the defaults.
    ``derived_hidden`` covers platform rules that cannot be represented as a fixed list (for
    example a runtime logger derived from the current app id).  Shared code only asks whether a
    source is hidden; it never learns the native rule.
    """

    hidden_prefixes: tuple[str, ...] = ()
    app_is_subject: bool = True
    derived_hidden: HiddenSourcePredicate | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def hides(self, source: str) -> bool:
        folded = source.casefold()
        if any(folded.startswith(prefix.casefold()) for prefix in self.hidden_prefixes if prefix):
            return True
        return bool(self.derived_hidden is not None and self.derived_hidden(source))


@dataclass(frozen=True, slots=True)
class AppExitEvidence:
    """Normalized evidence that the app under test unexpectedly left the foreground."""

    from_app_id: str
    to_app_id: str
    crash_dialog: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_app_id,
            "to": self.to_app_id,
            "crash_dialog": self.crash_dialog,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticWindow:
    """A platform-clock window of normalized diagnostic events."""

    events: tuple[DiagnosticEvent, ...] = ()
    target: TargetRef | None = None
    since: str | None = None
    since_unix_ms: int | None = None
    clock: str = "host"
    skew_ms: int = 0
    crash_evidence: CrashEvidence = field(default_factory=CrashEvidence)

    @property
    def lines(self) -> list[str]:
        return [event.text for event in self.events]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def select_lines(
        self,
        *,
        grep: str | None = None,
        source: str | None = None,
        lines: int | None = None,
    ) -> list[str]:
        """Filter normalized records while retaining their adapter rendering."""

        pattern = re.compile(grep) if grep else None
        selected = [
            event.text
            for event in self.events
            if (source is None or event.source == source)
            and (pattern is None or pattern.search(event.text) is not None)
        ]
        if lines is not None and lines >= 0:
            # Preserve the historical ``[-0:]`` behavior of the public logcat helper.
            selected = selected[-lines:]
        return selected

    def contains(self, text: str, *, ignore_case: bool = False) -> bool:
        needle = text.casefold() if ignore_case else text
        return any(
            needle in (event.text.casefold() if ignore_case else event.text)
            for event in self.events
        )

    def digest(
        self,
        *,
        levels: str,
        drop_source_prefixes: Sequence[str] = (),
        keep_source_prefixes: Sequence[str] = (),
        only_source_prefixes: Sequence[str] = (),
        limit: int = 20,
        per_source: int = 5,
        max_line_chars: int = 300,
    ) -> dict[str, Any]:
        """Reduce a normalized window for per-action evidence.

        ``levels`` retains AUA's public one-letter preference syntax, but comparison happens
        against normalized severities. Fatal events always survive user filters.
        """

        wanted_codes = {character.upper() for character in levels if character.strip()} | {"F"}
        wanted = {
            level
            for code in wanted_codes
            if (level := DiagnosticLevel.from_compatibility_code(code)) is not None
        }
        drop = tuple(prefix.casefold() for prefix in drop_source_prefixes if prefix.strip())
        keep = tuple(prefix.casefold() for prefix in keep_source_prefixes if prefix.strip())
        only = tuple(prefix.casefold() for prefix in only_source_prefixes if prefix.strip())

        kept: list[DiagnosticEvent] = []
        explicitly_selected_sources: set[str] = set()
        for event in self.events:
            if event.level is None or event.source is None or event.level not in wanted:
                continue
            source = event.source.strip()
            folded = source.casefold()
            fatal = event.level is DiagnosticLevel.FATAL
            exempt = any(folded.startswith(prefix) for prefix in keep)
            if not fatal and not exempt and any(folded.startswith(prefix) for prefix in drop):
                continue
            explicitly_selected = any(folded.startswith(prefix) for prefix in only)
            if only and not explicitly_selected and not fatal:
                continue
            if not fatal and not explicitly_selected and not exempt and event.hidden_by_default:
                continue
            if explicitly_selected:
                explicitly_selected_sources.add(source)
            kept.append(event)

        total = len(kept)
        if per_source > 0:
            seen: dict[str, int] = {}
            capped: list[DiagnosticEvent] = []
            for event in kept:
                source = event.source or ""
                if source in explicitly_selected_sources:
                    capped.append(event)
                    continue
                seen[source] = seen.get(source, 0) + 1
                if seen[source] <= per_source:
                    capped.append(event)
            kept = capped

        bounded = max(1, int(limit))
        truncated = len(kept) > bounded
        if truncated:
            head = max(1, (bounded * 2) // 3)
            tail = bounded - head
            kept = kept[:head] + (kept[-tail:] if tail else [])

        rendered = [_clip(event.text, max_line_chars) for event in kept]
        digest: dict[str, Any] = {
            "levels": "".join(code for code in "VDIWEF" if code in wanted_codes),
            "lines": rendered,
            "count": len(rendered),
            "total_count": total,
            "omitted": max(0, total - len(rendered)),
            "truncated": truncated,
        }
        if only:
            digest["only"] = [prefix for prefix in only_source_prefixes if prefix.strip()]
        return digest


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…[+{len(text) - limit} chars]"


__all__ = [
    "AppExitEvidence",
    "CrashEvidence",
    "DIAGNOSTIC_LEVEL_CODES",
    "DiagnosticEvent",
    "DiagnosticLevel",
    "DiagnosticSourcePolicy",
    "DiagnosticWindow",
    "UnknownDiagnosticMark",
]
