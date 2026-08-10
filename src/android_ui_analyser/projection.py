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

import difflib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
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
    "window": "window",
    "stable_key": "stable_key",
}

# Aliases that render the shortened form of their underlying key.
_SHORT_FORM_ALIASES = frozenset({"rid"})

_SQUASHED_FIELD_ALIASES: dict[str, str] = {
    name.replace("_", ""): name for name in FIELD_ALIASES
}

# The names these columns carry in Android itself. An agent that knows the platform reaches for
# `contentDescription` and `resource-id` before it reaches for this tool's abbreviations, and it
# is right to — they are the same attribute on the same node, so this is an alias rather than a
# guess. Measured 2026-08-10: `--fields id,text,rid,clickable,contentDescription,...` was refused
# whole, costing a call to correct one word out of seven.
_ANDROID_FIELD_SPELLINGS: dict[str, str] = {
    "contentdescription": "content_desc",
    "description": "content_desc",
    "resourceid": "resource_id",
    "resourcename": "resource_id",
    "classname": "type",
    "class": "type",
    "clazz": "type",
    "packagename": "window",
    "longclickable": "long_clickable",
    "stablekey": "stable_key",
    "elementid": "id",
    "index": "id",
}


def resolve_field_name(name: str) -> str:
    """Accept a column under any spelling that unambiguously names the same column.

    Case, `-` and `_` carry no meaning here (`content-desc`, `contentDesc` and `CONTENT_DESC`
    are one column), and neither does an `is` prefix on a boolean, which is how the attribute
    reads in Java (`isClickable`).
    """
    if name in FIELD_ALIASES:
        return name
    squashed = name.lower().replace("-", "").replace("_", "")
    if squashed in _ANDROID_FIELD_SPELLINGS:
        return _ANDROID_FIELD_SPELLINGS[squashed]
    if squashed.startswith("is") and squashed[2:] in _SQUASHED_FIELD_ALIASES:
        return _SQUASHED_FIELD_ALIASES[squashed[2:]]
    return _SQUASHED_FIELD_ALIASES.get(squashed, name)

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
    """``com.app:id/tab_browse`` → ``tab_browse`` (the part a human recognises)."""
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
    # With ``nonempty``, also keep a node that carries no label but *can be acted on*. A
    # design-system tile puts its click handler on an inner container and renders the title as a
    # separate non-clickable node outside it, so that container has no text, no content-desc and
    # no resource-id — and dropping it leaves a view showing the label with ``clickable: false``
    # and no clickable node at all, which reads as "the control is absent or disabled". That
    # exact structure produced a false FAIL_CRITICAL against the maths composer. Off for
    # ``analyze --nonempty``, whose contract is "rows a human can read"; on for the folded
    # post-action observation, whose contract is "rows you can act on next".
    keep_actionable: bool = False
    no_system: bool = False
    no_ime: bool = False
    no_wrappers: bool = False
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
        no_ime: bool = False,
        no_wrappers: bool = False,
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
            or no_ime
            or no_wrappers
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
            no_ime=no_ime and not show_all,
            no_wrappers=no_wrappers and not show_all,
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

    @classmethod
    def for_observation(
        cls, spec: str | None, *, fmt: OutputFormat | str = OutputFormat.json
    ) -> Projection | None:
        """The view applied to a folded post-action ``observation``; ``None`` means "don't touch".

        ``spec`` of ``"all"`` (or empty) returns ``None`` so the full dump is emitted verbatim.
        Anything else is a field list, filtered to the app's own labelled nodes — the point is
        that the *default* observation is cheap enough that no caller needs ``--no-observe``.
        """
        if spec is None:
            return None
        cleaned = spec.strip()
        if not cleaned or cleaned.lower() == "all":
            return None
        parsed = cls.parse(
            fmt=fmt,
            fields=cleaned,
            nonempty=True,
            no_system=True,
            no_ime=True,
        )
        return replace(parsed, keep_actionable=True)

    @staticmethod
    def _parse_fields(raw: str | None) -> tuple[str, ...]:
        if not raw:
            return ()
        names = [resolve_field_name(n) for n in _split_csv(raw)]
        unknown = [n for n in names if n not in FIELD_ALIASES]
        if unknown:
            near = [
                f"{n} -> did you mean {', '.join(guesses)}?"
                for n in unknown
                if (guesses := difflib.get_close_matches(n.lower(), FIELD_ALIASES, n=2, cutoff=0.6))
            ]
            raise UsageError(
                f"unknown --fields name(s): {', '.join(unknown)}",
                hint=(f"{' '.join(near)} " if near else "") + f"Valid names: {_valid_field_names()}.",
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
        elements = payload.get("elements") or []
        height = int((payload.get("screen") or {}).get("height") or 0)
        # Structural, so it is derived from the whole screen: whether a node wraps something
        # must not change because another flag filtered its children out of the view.
        wrappers = _wrapper_ids(elements) if self.no_wrappers else frozenset()
        kept = [
            e for e in elements if e.get("id") not in wrappers and self._keep(e, height)
        ]
        if self.limit is not None:
            kept = kept[: self.limit]
        return kept

    def _keep(self, element: dict[str, Any], screen_height: int) -> bool:
        """Filters of different kinds AND together; repeats of one kind OR together."""
        checks = (
            not self.nonempty
            or _has_label(element)
            or (self.keep_actionable and _is_actionable(element)),
            not self.no_system or not _is_system(element, screen_height),
            not self.no_ime or not _is_ime(element),
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
        lines += _route_comment(meta)
        return lines


def _route_comment(meta: dict[str, Any]) -> list[str]:
    """What this app's map already knows how to reach from here.

    `suggested_gotos` rides on every JSON `analyze`, and TSV dropped it — while the guide and
    the orientation block both teach `--format tsv analyze --fields …` as the way to read a
    screen. So the recommended call was the one that hid the map. Measured 2026-08-10: across
    five fresh-agent runs on an app with 135 remembered screens and 613 routes, not one agent
    used `goto`, and none could have known it had anything to offer.

    One line, only when there is something to replay, and `--no-meta` still silences it.
    """
    gotos = [str(g) for g in (meta.get("suggested_gotos") or [])]
    if not gotos:
        return []
    names = [g.removeprefix("goto ").strip() for g in gotos]
    return [
        "# goto: " + " | ".join(names[:4]) + "  (aua goto \"<name>\" replays the remembered "
        "route — no tapping; `aua map --find \"<goal>\"` for one not listed)"
    ]


def _comment_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(_cell(v) for v in value)
    return _cell(value)


def render_action_tsv(data: dict[str, Any], view: Projection | None = None) -> str:
    """An action envelope as TSV: the envelope as ``#`` lines, then its observation's rows.

    ``--format tsv`` was silently ignored on action responses, so an agent that had settled on
    TSV for ``analyze`` got JSON back the moment it tapped anything and had to hand-parse it.
    The envelope is kept because it carries the verdict — ``change.text_added`` is usually the
    whole reason the action was run.
    """
    lines = _envelope_comments(data)
    payload = data.get("observation")
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return "\n".join(lines)
    projection = view or Projection.parse(fmt=OutputFormat.tsv)
    return "\n".join([*lines, projection.render_tsv(payload)])


def _envelope_comments(data: dict[str, Any]) -> list[str]:
    """Every scalar (and one nested level of scalars) in the envelope, as ``#key=value``."""
    lines: list[str] = []
    for key, value in data.items():
        if key == "observation":
            continue
        lines += [f"# {key}{suffix}={_comment_value(v)}" for suffix, v in _scalars(value)]
    return lines


def _scalars(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, (str, int, float, bool)):
        return [("", value)]
    if _is_scalar_list(value):
        return [("", list(value))]
    if isinstance(value, dict):
        return [
            (f".{k}", v)
            for k, v in value.items()
            if isinstance(v, (str, int, float, bool)) or _is_scalar_list(v)
        ]
    return []


def _is_scalar_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(v is None or isinstance(v, (str, int, float, bool)) for v in value)
    )


def _has_label(element: dict[str, Any]) -> bool:
    return bool(element.get("text") or element.get("resource_id") or element.get("content_desc"))


def _is_actionable(element: dict[str, Any]) -> bool:
    """Can this node be acted on, whatever it is called?

    Deliberately narrow: the interaction flags only. It is not "anything that might matter" — the
    point is to keep a view small while never dropping a node the caller could tap next.
    """
    return bool(
        element.get("clickable")
        or element.get("checkable")
        or element.get("long_clickable")
        or element.get("scrollable")
    )


def _matches(value: Any, needles: Iterable[str]) -> bool:
    hay = str(value or "").lower()
    return any(n in hay for n in needles)


def _has_app_rid(resource_id: Any) -> bool:
    """Mirrors the parse rule that keeps id-bearing nodes (see :mod:`hierarchy`)."""
    rid = str(resource_id or "")
    return ":id/" in rid and not rid.startswith("android:id/")


def _is_inert(element: dict[str, Any]) -> bool:
    return not any(
        element.get(k) for k in ("clickable", "long_clickable", "checkable", "scrollable")
    )


def _box(bounds: Any) -> Bounds | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    x1, y1, x2, y2 = (int(v) for v in bounds)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _area(box: Bounds) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _strictly_contains(outer: Bounds, inner: Bounds) -> bool:
    """*outer* wraps *inner* and is genuinely bigger.

    Strictness matters: Compose siblings routinely share identical bounds, and a plain
    containment test would make each of them "wrap" the other and drop both.
    """
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
        and _area(outer) > _area(inner)
    )


def _wrapper_ids(elements: Sequence[dict[str, Any]]) -> frozenset[int]:
    """Ids of pure layout containers — the rows ``--no-wrappers`` drops.

    A wrapper is a node that survived the parse *only* because it carries an app
    resource-id: nothing to read, nothing to act on, and it encloses something else. Leaves
    are never wrappers — an unlabeled leaf is the concrete drawn thing (an icon, an image, a
    custom view), which is exactly what a caller taps.

    "Encloses something else" is read off the emitted boxes, so a container whose children
    were all absorbed into a clickable ancestor looks like a leaf here and survives. That
    errs toward keeping a row rather than hiding an addressable one, which is the safe bias.
    """
    boxed = [(el, _box(el.get("bounds"))) for el in elements]
    candidates = [
        (el, box)
        for el, box in boxed
        if box is not None
        and _has_app_rid(el.get("resource_id"))
        and not (el.get("text") or el.get("content_desc"))
        and _is_inert(el)
    ]
    return frozenset(
        int(el["id"])
        for el, box in candidates
        if any(other is not el and ob is not None and _strictly_contains(box, ob) for other, ob in boxed)
    )


def _is_ime(element: dict[str, Any]) -> bool:
    """Soft-keyboard chrome — drop with ``--no-ime`` so chat trees stay readable."""
    if element.get("window") == "ime":
        return True
    rid = element.get("resource_id") or ""
    return "inputmethod" in rid.lower()


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


def trim_observation_payload(
    data: dict[str, Any],
    view: Projection | None,
    *,
    fmt: OutputFormat | str = OutputFormat.json,
) -> dict[str, Any]:
    """Trim a nested ``observation`` in an action payload, keeping derived lists consistent.

    Shared by the CLI and the MCP server on purpose. These two surfaces had already drifted: the
    CLI trimmed the folded observation while MCP returned every field of every element on every
    action, so the measured cost win ("37 taps produced 73 separate analyze calls") applied to one
    caller and not the other. One implementation means a future change lands on both.

    ``stable_elements`` and ``next_actions`` are both derived from the same tree, so they are
    filtered to the ids that survived. Leaving them whole re-adds exactly the nodes the view just
    dropped — the status bar alone was a third of `stable_elements` — and worse, lets
    ``next_actions`` name an id that is not in the observation the caller was given.
    """
    if view is None or not isinstance(data, dict):
        return data
    payload = data.get("observation")
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return data
    projected = view.apply(payload, fmt=fmt)
    data["observation"] = projected
    kept = {e.get("id") for e in projected.get("elements", []) if isinstance(e, dict)}
    for key in ("stable_elements", "next_actions"):
        rows = data.get(key)
        if isinstance(rows, list):
            data[key] = [r for r in rows if not isinstance(r, dict) or r.get("id") in kept]
    return data
