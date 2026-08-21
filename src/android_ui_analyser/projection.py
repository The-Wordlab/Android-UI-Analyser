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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .errors import UsageError
from .schema import Meta, OutputFormat, drop_default_flags

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
    "parent": "parent",
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

# Named `meta` budgets for a folded post-action observation. The full `meta` is sized for a
# question a caller *asked* — it carries research tasks, deeplink suggestions, a capture hint,
# the provider list. An action asks a much narrower question, and measured on one real settings
# screen the full block was 299 of the observation's 919 tokens: 16 keys empty, and three
# (`research_tasks`, `suggested_deeplinks`, `capture_hint`) alone accounting for over half of
# what was left. Paying that on every tap is what pushes an agent into calling `analyze`
# separately, which is the cost this whole path exists to avoid.
#
# `changed` therefore answers only the questions an *action* raises: did the screen change,
# where am I now, which device, and is there anything I am not allowed to miss. `ask`,
# `goal_progress` and `observation_contract` are protocol obligations rather than information,
# so they are never budget-trimmed; `lossy_*` warns that the text may be wrong, which a caller
# must see before believing a label it is about to assert on. `raw_image` stays because it is
# the escape hatch from the whole element view — the frame an agent can actually look at.
#
# Telemetry is out, and that is the test a key has to pass: not "is this true" but "can the
# caller do anything differently knowing it". `tier_used`, `via` and `path` describe how the
# read was obtained and `duration_ms` how long it took; none changes the next call. The first
# two were also the same word twice on the overwhelming majority of reads — both said
# `hierarchy` — so between them they spent two keys to say nothing once. A human debugging
# perception wants all four and gets them from `analyze`, which is where that question is
# actually being asked.
# Declaration order is the emitted order (see :meth:`Projection._meta`), and it is chosen, not
# alphabetical: a reader — human or model — takes the first keys most seriously, so the frame it
# can actually look at and the answer to "did anything change" lead, and the provenance of the
# read trails.
OBSERVATION_META_PRESETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "changed": (
            # the frame a reader can actually look at, first
            "raw_image",
            # protocol obligations — never trimmed for budget, never buried
            "ask",
            "goal_progress",
            "observation_contract",
            # did anything change, and can I trust what I am looking at
            "unchanged",
            "element_diff",
            "stale_risk",
            "lossy_text",
            "lossy_hint",
            "known_screen",
            # shortcuts worth knowing the moment they exist; empty (and free) otherwise
            "suggested_gotos",
            "flows",
            "map_hint",
            # `fingerprint` is here because something *reads* it, not as provenance:
            # `coaching.emitted_fingerprint` recovers it from the emitted payload to stamp the
            # caller turn, and under the warm daemon the answering engine is another process,
            # so the payload is the only place it can come from. `device_serial` answers "which
            # target did this happen on" for anyone driving more than one.
            "fingerprint",
            "device_serial",
        ),
    }
)

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
    # Omit every key that carries nothing — a null, an empty list, a flag at its default —
    # whatever `--format` asked for. Set only for the folded observation, whose contract is
    # "the cheapest read of the new screen"; `analyze` keeps its byte-for-byte output so a
    # caller that wants the full shape still has one call that gives it. See
    # :func:`~.schema.drop_default_flags` for the one flag this must never drop.
    drop_defaults: bool = False
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
            # `--where-rid a,b,c` used to be ONE substring that matched nothing, and an empty
            # `elements` list is indistinguishable from "verified absent" — the most dangerous
            # answer this tool can give. Android resource ids cannot contain a comma, so
            # splitting is unambiguous here. `where_text` is deliberately NOT split: real UI
            # copy contains commas, and splitting it would silently widen the caller's filter.
            where_rid=tuple(
                part.strip().lower()
                for raw in (where_rid or ())
                if raw
                for part in str(raw).split(",")
                if part.strip()
            ),
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
        cls,
        spec: str | None,
        *,
        meta: str | None = None,
        fmt: OutputFormat | str = OutputFormat.json,
    ) -> Projection | None:
        """The view applied to a folded post-action ``observation``; ``None`` means "don't touch".

        Two independent dials, because they answer different questions and a caller needs one
        without the other: *which columns* per element (``spec``) and *which* ``meta`` *keys*
        (``meta``, a name from :data:`OBSERVATION_META_PRESETS` or a comma-separated key list).
        Either at ``"all"`` opts that dial out; only both at ``"all"`` returns ``None`` and
        emits the payload verbatim. Wiring them together was the earlier mistake — asking for
        every column silently re-added every `meta` key too, so the cheap dial could not be
        used at the one moment a caller wanted the full element shape.

        Rows are filtered to the app's own nodes: no status bar, no keyboard, no pure wrapper
        layouts, and nothing unlabelled *unless it can be acted on* — an unnamed switch is
        exactly the row a caller needs next, so ``keep_actionable`` overrides ``nonempty``.
        The whole point is that the default observation is cheap enough that no caller needs
        ``--no-observe`` and reaches for a separate ``analyze`` instead.
        """
        if spec is None and meta is None:
            return None
        fields = (spec or "all").strip() or "all"
        meta_spec = (meta or "all").strip() or "all"
        fields_all = fields.lower() == "all"
        meta_all = meta_spec.lower() == "all"
        if fields_all and meta_all:
            return None
        parsed = cls.parse(
            fmt=fmt,
            fields=None if fields_all else fields,
            nonempty=True,
            no_system=True,
            no_ime=True,
            no_wrappers=True,
        )
        return replace(
            parsed,
            keep_actionable=True,
            drop_defaults=True,
            meta_keys=None if meta_all else cls._parse_observation_meta(meta_spec),
        )

    @staticmethod
    def _parse_observation_meta(raw: str) -> tuple[str, ...]:
        """A preset name from :data:`OBSERVATION_META_PRESETS`, or an explicit key list."""
        preset = OBSERVATION_META_PRESETS.get(raw.strip().lower())
        if preset is not None:
            # Declared order, not sorted: the preset encodes what a reader should see first,
            # and it is a literal, so the emitted order is deterministic either way.
            return tuple(preset)
        names = _split_csv(raw)
        valid = set(Meta.model_fields)
        unknown = [n for n in names if n not in valid]
        if unknown:
            raise UsageError(
                f"unknown observation meta key(s): {', '.join(unknown)}",
                hint=(
                    f"Use a preset ({', '.join(sorted(OBSERVATION_META_PRESETS))}), 'all', "
                    f"or any of: {', '.join(sorted(valid))}."
                ),
            )
        return tuple(names)

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
        drop_null = self.drop_defaults or OutputFormat(fmt) is OutputFormat.compact
        out = dict(payload)
        rows = self.select(payload)
        if self.drop_defaults:
            # Default-trim the *whole* element, then let the columns select — never the reverse.
            # `checked: false` survives only on a checkable node, and `checkable` is not in the
            # default column set: trimming a projected row would look at a dict with no
            # `checkable` key, conclude the node is not checkable, and delete the one field that
            # makes an off switch readable. Order is the fix, not a longer column list.
            rows = [drop_default_flags(row) for row in rows]
        out["elements"] = [self.project(row, drop_null=drop_null) for row in rows]
        screen = payload.get("screen")
        if self.drop_defaults and isinstance(screen, dict):
            out["screen"] = {k: v for k, v in screen.items() if v is not None}
        out["meta"] = self._meta(payload.get("meta") or {})
        if self.no_meta:
            out.pop("meta", None)
        elif drop_null:
            strict = self.drop_defaults
            out["meta"] = {
                k: v for k, v in out["meta"].items() if not _is_empty(v, strict=strict)
            }
        return out

    def _meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """The requested `meta` keys, in the order they were requested.

        Order is part of the answer. A reader weights what it sees first, so the caller (or the
        preset) decides what leads; following the payload's own field order instead would bury
        the frame path and the did-anything-change keys behind whatever the model happens to
        declare first.
        """
        if self.meta_keys is None:
            return dict(meta)
        return {k: meta[k] for k in self.meta_keys if k in meta}

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
        meta = payload.get("meta") or {}
        if self.no_meta:
            # `--no-meta` drops diagnostics, not affordances. Measured 2026-08-10: an agent's
            # first call was `--format tsv analyze --no-meta` — the guide recommends it to cut
            # noise — so it never saw the `# goto:` or `# aua asks:` lines and navigated the
            # whole task by tapping. What the call cost is metadata about the call; a route you
            # can replay and a question you can answer are things to DO, and suppressing those
            # to save two lines is what made the map invisible in the first place.
            return _route_comment(meta) + _flows_comment(meta) + _ask_comment(meta)
        screen = payload.get("screen") or {}
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
        lines += _flows_comment(meta)
        lines += _ask_comment(meta)
        return lines


def _flows_comment(meta: dict[str, Any]) -> list[str]:
    """Saved journeys for this app — one call instead of a dozen.

    A parameterised flow had been sitting saved for this project and no agent had ever run one.
    `flow` appeared 19 times in the long guide and zero times in the analyze header or the
    orientation block, which is everything an agent actually reads.
    """
    flows = [str(f) for f in (meta.get("flows") or [])]
    if not flows:
        return []
    return [
        "# flows: " + " | ".join(flows[:3]) + "  (aua flow run <name> --param K=V — replays a "
        "whole saved journey: launch, taps, waits, even cross-app sign-in)"
    ]


def _ask_comment(meta: dict[str, Any]) -> list[str]:
    """The one question this screen raises, and exactly how to answer it in passing."""
    ask = meta.get("ask")
    if not isinstance(ask, dict) or not ask.get("id"):
        return []
    return [f"# aua asks: {ask.get('q', '')}  -> {ask.get('how', '')}"]


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


def _is_empty(value: Any, *, strict: bool = False) -> bool:
    """A ``meta`` value that carries nothing; ``strict`` also counts ``False`` as nothing.

    Only the observation is strict. ``--format compact`` has emitted ``lossy_text: false`` for
    as long as it has existed and a consumer may be branching on its presence, whereas the
    observation's contract is new and states that absence means the default — so the stricter
    rule applies where it was announced and nowhere else.

    ``False`` is tested by identity: ``0 == False`` in Python, and a ``duration_ms`` of 0 is a
    real measurement, not an absence. That exact equality trap already cost this file's sibling
    a bug (see :meth:`~.schema.AnalyzeResult.as_dict`).
    """
    if value is None or value == []:
        return True
    return strict and value is False


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


def _wrapper_ids(elements: Sequence[dict[str, Any]]) -> frozenset[Any]:
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
        el["id"]
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

    ``next_actions`` is derived from the same tree, so it is filtered to the ids that survived.
    Leaving it whole re-adds exactly the nodes the view just dropped, and worse, lets it name an
    id that is not in the observation the caller was given.
    """
    if view is None or not isinstance(data, dict):
        return data
    payload = data.get("observation")
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return data
    projected = view.apply(payload, fmt=fmt)
    data["observation"] = projected
    kept = {e.get("id") for e in projected.get("elements", []) if isinstance(e, dict)}
    for key in ("next_actions",):
        rows = data.get(key)
        if isinstance(rows, list):
            data[key] = [r for r in rows if not isinstance(r, dict) or r.get("id") in kept]
    return data
