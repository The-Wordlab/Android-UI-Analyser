"""Projection, filtering and TSV rendering of an ``analyze`` payload (PRD §8 views).

A screen is 20-40 elements of which most are status-bar chrome, and a caller almost always
wants a handful of columns from a handful of rows. Without this, every call site
re-implements the same ad-hoc JSON filter by hand — which is exactly where projection bugs
live. The CLI exposes the whole thing as flags on ``analyze`` (``--fields``,
``--format tsv``, ``--nonempty``, ``--no-system``, ``--where-text``, ``--where-rid``,
``--clickable``, ``--region``, ``--limit``, ``--meta``/``--no-meta``, ``--all``) and this
module owns their semantics.

Two invariants:

* Everything operates on the **full** dict form of :class:`~.schema.AnalyzeResult`, never
  the ``compact`` trim, so a field is projectable whatever ``--format`` asked for, and the
  same code path serves both the in-process result and the daemon's dict response.
* A :class:`Projection` that no flag activated is :attr:`~Projection.active` ``False`` and
  the caller renders as before — the default ``json``/``compact`` bytes never move.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import UsageError
from .schema import Meta, OutputFormat

# Short agent-facing names → the canonical element key they read. Two spellings are
# deliberate: `rid` yields the *tail* after "/" (what a human recognises), `resource_id`
# the fully-qualified id (what a selector needs).
FIELD_ALIASES: dict[str, str] = {
    "id": "id",
    "type": "type",
    "text": "text",
    "rid": "resource_id",
    "resource_id": "resource_id",
    "desc": "content_desc",
    "content_desc": "content_desc",
    "bounds": "bounds",
    "center": "center",
    "clickable": "clickable",
    "enabled": "enabled",
    "focused": "focused",
    "checkable": "checkable",
    "checked": "checked",
    "selected": "selected",
    "scrollable": "scrollable",
    "long_clickable": "long_clickable",
    "password": "password",
    "source": "source",
    "confidence": "confidence",
}

# Aliases that render the shortened form of their underlying key.
_SHORT_FORM_ALIASES = frozenset({"rid"})

# What `--format tsv` shows when no `--fields` was given: who am I, what does it say,
# what is its selector, can I tap it.
TSV_DEFAULT_FIELDS: tuple[str, ...] = ("id", "text", "rid", "clickable")

# Packages whose resource-ids are never app content. `android:` is deliberately absent —
# `android:id/button1` is a dialog's OK button and `android:id/content` the app's own root.
_SYSTEM_ID_PACKAGES = frozenset(
    {
        "com.android.systemui",
        "com.android.systemui.plugins",
        "com.android.launcher3",
    }
)

# Status-bar / notification-shade chrome, matched against content_desc.
_SYSTEM_DESC_RE = re.compile(
    r"\b(?:"
    r"battery|charging|wi-?fi|mobile\s+(?:signal|data)|phone\s+signal|cellular|"
    r"signal\s+(?:full|strength)|no\s+(?:signal|sim|service)|roaming|"
    r"bluetooth|airplane\s+mode|vpn|do\s+not\s+disturb|ringer|"
    r"google\s+play\s+services|play\s+services"
    r")\b",
    re.IGNORECASE,
)

# A content_desc match alone is not proof of chrome — an app's own settings screen may
# legitimately have a "Bluetooth" row. We additionally require the element to sit inside
# the top strip of the screen, where only the status bar lives. Real status bars are
# 84-130px on a 2400px-tall device; 6% (144px) clears them without reaching app content.
_STATUS_BAND_FRACTION = 0.06

Bounds = tuple[int, int, int, int]


def short_rid(resource_id: str | None) -> str | None:
    """``com.app:id/tab_explore`` → ``tab_explore`` (the part a human recognises)."""
    if not resource_id:
        return resource_id
    return resource_id.rsplit("/", 1)[-1]


def is_system_rid(resource_id: str | None) -> bool:
    """True when a resource-id belongs to system chrome rather than the app under test.

    The package list lives here because ``--no-system`` already owns it; candidate ranking
    reads the same one so a status-bar id can never be offered as a "did you mean".
    """
    if not resource_id or ":" not in resource_id:
        return False
    return resource_id.split(":", 1)[0] in _SYSTEM_ID_PACKAGES


def _valid_field_names() -> str:
    return ", ".join(sorted(FIELD_ALIASES))


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_region(raw: str) -> Bounds:
    parts = _split_csv(raw)
    if len(parts) != 4:
        raise UsageError(
            f"invalid --region '{raw}'",
            hint="Give four integers: --region x1,y1,x2,y2 (e.g. 0,0,1080,300).",
        )
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError as exc:
        raise UsageError(
            f"invalid --region '{raw}'",
            hint="All four values must be integers, e.g. --region 0,0,1080,300.",
        ) from exc
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _intersects(bounds: Any, region: Bounds) -> bool:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return False
    x1, y1, x2, y2 = (int(v) for v in bounds)
    rx1, ry1, rx2, ry2 = region
    return not (x2 <= rx1 or rx2 <= x1 or y2 <= ry1 or ry2 <= y1)


def _cell(value: Any) -> str:
    """One TSV cell: never contains a tab or newline; ``None`` renders empty.

    Empty is reserved for *unknown* so a tri-state flag stays readable as
    ``true``/``false``/`` `` (see :class:`~.schema.Element`).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_cell(v) for v in value)
    return re.sub(r"\s+", " ", str(value)).strip()


@dataclass(frozen=True)
class Projection:
    """A validated element view: which rows survive, which columns come back."""

    fields: tuple[str, ...] = ()
    nonempty: bool = False
    no_system: bool = False
    where_text: tuple[str, ...] = ()
    where_rid: tuple[str, ...] = ()
    clickable_only: bool = False
    regions: tuple[Bounds, ...] = ()
    limit: int | None = None
    meta_keys: tuple[str, ...] | None = None
    no_meta: bool = False
    tsv: bool = False
    _explicit: bool = field(default=False, repr=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def parse(
        cls,
        *,
        fmt: OutputFormat | str = OutputFormat.json,
        fields: str | None = None,
        nonempty: bool = False,
        no_system: bool = False,
        show_all: bool = False,
        where_text: Sequence[str] | None = None,
        where_rid: Sequence[str] | None = None,
        clickable: bool = False,
        region: Sequence[str] | None = None,
        limit: int | None = None,
        meta: str | None = None,
        no_meta: bool = False,
    ) -> Projection:
        """Validate raw CLI values into a :class:`Projection` (raises :class:`UsageError`).

        ``tsv`` is opinionated — it is a new format nothing consumes, so it defaults to the
        "show me the app" view (``--nonempty --no-system``) that a caller wants 90% of the
        time. ``--all`` opts back out. JSON formats stay byte-identical unless a flag asks
        otherwise.
        """
        is_tsv = OutputFormat(fmt) is OutputFormat.tsv
        columns = cls._parse_fields(fields)
        explicit = bool(
            columns
            or nonempty
            or no_system
            or show_all
            or where_text
            or where_rid
            or clickable
            or region
            or limit is not None
            or meta
            or no_meta
        )
        if limit is not None and limit < 0:
            raise UsageError("invalid --limit", hint="--limit takes a non-negative integer.")
        return cls(
            fields=columns,
            nonempty=(nonempty or is_tsv) and not show_all,
            no_system=(no_system or is_tsv) and not show_all,
            where_text=tuple(s.lower() for s in (where_text or ()) if s),
            where_rid=tuple(s.lower() for s in (where_rid or ()) if s),
            clickable_only=clickable,
            regions=tuple(_parse_region(r) for r in (region or ())),
            limit=limit,
            meta_keys=cls._parse_meta(meta),
            no_meta=no_meta,
            tsv=is_tsv,
            _explicit=explicit,
        )

    @staticmethod
    def _parse_fields(raw: str | None) -> tuple[str, ...]:
        if not raw:
            return ()
        names = _split_csv(raw)
        unknown = [n for n in names if n not in FIELD_ALIASES]
        if unknown:
            raise UsageError(
                f"unknown --fields name(s): {', '.join(unknown)}",
                hint=f"Valid names: {_valid_field_names()}.",
            )
        return tuple(names)

    @staticmethod
    def _parse_meta(raw: str | None) -> tuple[str, ...] | None:
        if not raw:
            return None
        names = _split_csv(raw)
        valid = set(Meta.model_fields)
        unknown = [n for n in names if n not in valid]
        if unknown:
            raise UsageError(
                f"unknown --meta key(s): {', '.join(unknown)}",
                hint=f"Valid keys: {', '.join(sorted(valid))}.",
            )
        return tuple(names)

    # -- introspection -----------------------------------------------------

    @property
    def active(self) -> bool:
        """True when the caller asked for a view (so the default bytes are untouched)."""
        return self.tsv or self._explicit

    def columns(self) -> tuple[str, ...]:
        """The column order for a projected view (TSV falls back to its defaults)."""
        if self.fields:
            return self.fields
        return TSV_DEFAULT_FIELDS if self.tsv else ()

    # -- filtering ---------------------------------------------------------

    def select(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """The elements of *payload* that survive every filter, in payload order."""
        height = int((payload.get("screen") or {}).get("height") or 0)
        kept = [e for e in (payload.get("elements") or []) if self._keep(e, height)]
        if self.limit is not None:
            kept = kept[: self.limit]
        return kept

    def _keep(self, element: dict[str, Any], screen_height: int) -> bool:
        """Filters of different kinds AND together; repeats of one kind OR together."""
        checks = (
            not self.nonempty or _has_label(element),
            not self.no_system or not _is_system(element, screen_height),
            not self.clickable_only or bool(element.get("clickable")),
            not self.where_text or _matches(element.get("text"), self.where_text),
            not self.where_rid or _matches(element.get("resource_id"), self.where_rid),
            not self.regions or any(_intersects(element.get("bounds"), r) for r in self.regions),
        )
        return all(checks)

    # -- projection --------------------------------------------------------

    def value_of(self, element: dict[str, Any], alias: str) -> Any:
        key = FIELD_ALIASES[alias]
        value = element.get(key)
        if alias in _SHORT_FORM_ALIASES:
            return short_rid(value)
        return value

    def project(self, element: dict[str, Any], *, drop_null: bool = False) -> dict[str, Any]:
        """*element* reduced to the requested columns (unprojected when none were asked for).

        ``drop_null`` serves ``--format compact``, which promises the smallest footprint:
        absence then carries the same "unknown" meaning the tri-state flags already give it,
        so nothing is lost by omitting a null.
        """
        cols = self.columns()
        if not cols:
            return element
        out = {alias: self.value_of(element, alias) for alias in cols}
        if drop_null:
            return {k: v for k, v in out.items() if v is not None}
        return out

    def apply(self, payload: dict[str, Any], *, fmt: OutputFormat | str = OutputFormat.json) -> dict:
        """The payload as a JSON view: filtered rows, projected columns, trimmed meta."""
        drop_null = OutputFormat(fmt) is OutputFormat.compact
        out = dict(payload)
        out["elements"] = [self.project(e, drop_null=drop_null) for e in self.select(payload)]
        out["meta"] = self._meta(payload.get("meta") or {})
        if self.no_meta:
            out.pop("meta", None)
        elif drop_null:
            out["meta"] = {k: v for k, v in out["meta"].items() if v not in (None, [])}
        return out

    def _meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        if self.meta_keys is None:
            return dict(meta)
        return {k: v for k, v in meta.items() if k in self.meta_keys}

    # -- rendering ---------------------------------------------------------

    def render_tsv(self, payload: dict[str, Any]) -> str:
        """Tab-separated rows with a ``#``-prefixed summary, so the payload stays greppable."""
        elements = payload.get("elements") or []
        kept = self.select(payload)
        lines = self._comment_lines(payload, total=len(elements), shown=len(kept))
        cols = self.columns() or TSV_DEFAULT_FIELDS
        lines.append("\t".join(cols))
        lines += ["\t".join(_cell(self.value_of(e, c)) for c in cols) for e in kept]
        return "\n".join(lines)

    def _comment_lines(self, payload: dict[str, Any], *, total: int, shown: int) -> list[str]:
        if self.no_meta:
            return []
        screen = payload.get("screen") or {}
        meta = payload.get("meta") or {}
        head = [f"screen={meta.get('known_screen') or '-'}"]
        if screen.get("package"):
            head.append(f"package={screen['package']}")
        head.append(f"{screen.get('width', 0)}x{screen.get('height', 0)}")
        if screen.get("activity"):
            head.append(f"activity={screen['activity']}")
        lines = ["# " + " ".join(head)]
        if self.meta_keys is not None:
            lines += [f"# {k}={_comment_value(meta.get(k))}" for k in self.meta_keys]
            return lines
        summary = [f"elements={total}", f"shown={shown}"]
        for key in ("tier_used", "duration_ms"):
            if meta.get(key) is not None:
                summary.append(f"{key}={meta[key]}")
        lines.append("# " + " ".join(summary))
        return lines


def _comment_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(_cell(v) for v in value)
    return _cell(value)


def _has_label(element: dict[str, Any]) -> bool:
    return bool(element.get("text") or element.get("resource_id") or element.get("content_desc"))


def _matches(value: Any, needles: Iterable[str]) -> bool:
    hay = str(value or "").lower()
    return any(n in hay for n in needles)


def _is_system(element: dict[str, Any], screen_height: int) -> bool:
    if is_system_rid(element.get("resource_id")):
        return True
    desc = element.get("content_desc") or ""
    if not desc or not _SYSTEM_DESC_RE.search(desc):
        return False
    return _in_status_band(element.get("bounds"), screen_height)


def _in_status_band(bounds: Any, screen_height: int) -> bool:
    if not screen_height or not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return False
    return int(bounds[3]) <= max(1, round(screen_height * _STATUS_BAND_FRACTION))
