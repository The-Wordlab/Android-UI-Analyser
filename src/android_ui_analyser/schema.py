"""Canonical, versioned output schema (PRD §8).

This module is the **single source of truth** for the shape of everything the CLI and
MCP server emit. Pydantic models here are imported by the engine, the CLI, the MCP
wrapper, and the tests. Do not duplicate these shapes elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- enums


class Source(str, Enum):
    """Where a single element came from."""

    hierarchy = "hierarchy"
    detection = "detection"
    ocr = "ocr"
    grounding = "grounding"
    webview = "webview"


class ScreenSource(str, Enum):
    """Aggregate provenance of the whole screen result."""

    hierarchy = "hierarchy"
    vision = "vision"
    mixed = "mixed"


class TargetStatus(str, Enum):
    """Platform-neutral reachability state for an automation target."""

    online = "online"
    offline = "offline"
    booting = "booting"
    unavailable = "unavailable"
    unknown = "unknown"


class PathKind(str, Enum):
    """Which high-level perception path produced the result."""

    hierarchy = "hierarchy"
    vision = "vision"


class Tier(str, Enum):
    """Escalation ladder rungs (PRD §6a), cheapest → most expensive."""

    text = "text"
    selector = "selector"
    hierarchy = "hierarchy"
    vision = "vision"
    grounding = "grounding"


# Canonical ordering for the escalation ladder. Index == cost rank.
TIER_ORDER: tuple[Tier, ...] = (
    Tier.text,
    Tier.selector,
    Tier.hierarchy,
    Tier.vision,
    Tier.grounding,
)


def tier_rank(tier: Tier | str) -> int:
    """Return the cost rank of a tier (lower == cheaper)."""
    t = Tier(tier)
    return TIER_ORDER.index(t)


class OutputFormat(str, Enum):
    json = "json"
    pretty = "pretty"
    compact = "compact"
    tsv = "tsv"  # one element per line, tab-separated (see :mod:`projection`)
    delta = "delta"  # omit elements when meta.unchanged; else compact + element_diff
    msgpack = "msgpack"  # binary dump (host-side; see binary_dump.py / hierarchy.fbs)


class MatchMode(str, Enum):
    exact = "exact"
    contains = "contains"
    regex = "regex"


# --------------------------------------------------------------------------- models

Bounds = tuple[int, int, int, int]
Center = tuple[int, int]

#: How an element is addressed. An ``int`` is a frame-local ordinal (reading order, renumbered
#: every analyze); a ``str`` is the stable id that the payload actually publishes. Both flow
#: through the same code paths because a caller may hold either — a fresh observation gives it
#: the string, an older script still has the number.
ElementId = int | str


def center_of(bounds: Bounds) -> Center:
    """Geometric center of an ``[x1, y1, x2, y2]`` box."""
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# Flag → the value that carries no information. A flag sitting at its default is omitted from
# a trimmed payload, and absence reads back as this value. ``checked`` is deliberately absent
# from the table: whether it says anything depends on ``checkable``, which is the one rule a
# flat table cannot express (see :func:`drop_default_flags`).
FLAG_DEFAULTS: Mapping[str, bool] = MappingProxyType(
    {
        "clickable": False,
        "enabled": True,
        "focused": False,
        "checkable": False,
        "selected": False,
        "scrollable": False,
        "long_clickable": False,
        "password": False,
    }
)


def drop_default_flags(element: Mapping[str, Any]) -> dict[str, Any]:
    """*element* without the keys that say nothing: nulls, and flags at their default.

    The dict-level twin of :meth:`Element.compact`, for the projection layer — which operates
    on payload dicts and has no model instance to ask. Both consult :data:`FLAG_DEFAULTS`, and
    ``tests/test_observation_payload_is_slim.py`` pins them to each other so they cannot drift.

    One exception, and it is the whole reason this is a function rather than a comprehension:
    ``checked: false`` on a **checkable** node is the reading of a switch, not a default.
    Omitting it turns an off toggle into something indistinguishable from a plain button — the
    same trap :meth:`Element._compact_state` documents.

    Membership is tested by identity, not equality: ``0 == False`` in Python, so an equality
    test would silently delete a legitimate ``duration_ms``-style zero if this ever grew past
    booleans.
    """
    checkable = bool(element.get("checkable"))
    out: dict[str, Any] = {}
    for key, value in element.items():
        if value is None:
            continue
        if key == "checked":
            if not checkable and not value:
                continue
        elif value is FLAG_DEFAULTS.get(key):
            continue
        out[key] = value
    return out


class Element(BaseModel):
    """One actionable thing on screen, identified by a stable integer ``id``.

    The interaction-state flags (``checkable``/``checked``/``selected``/``scrollable``/
    ``long_clickable``/``password``) are **tri-state**: ``True``/``False`` when the
    accessibility node reported the attribute, ``None`` when it is genuinely unknown (a
    vision-derived element has no a11y attributes at all). A caller reading a toggle must
    be able to tell "off" from "unknown", so we never coerce a missing attribute to
    ``False``. ``clickable``/``enabled``/``focused`` predate this rule and keep their
    long-standing plain-``bool`` semantics.
    """

    model_config = ConfigDict(extra="forbid")

    # A frame-local ordinal in memory, a stable id once published. Both are accepted so a
    # payload round-trips: `as_dict` rewrites every id to the element's stable key (see
    # `AnalyzeResult._publish_identity`), and re-validating that payload must not be an error —
    # the alternative is a shape the tool emits but cannot read back, which every consumer
    # eventually trips over. Python paths index by whatever they were given.
    id: int | str
    type: str
    text: str | None = None
    resource_id: str | None = None
    content_desc: str | None = None
    bounds: Bounds
    center: Center
    clickable: bool = False
    enabled: bool = True
    focused: bool = False
    checkable: bool | None = None
    checked: bool | None = None
    selected: bool | None = None
    scrollable: bool | None = None
    long_clickable: bool | None = None
    password: bool | None = None
    source: Source = Source.hierarchy
    confidence: float | None = None
    # Cross-frame fingerprint — survives re-analyze ID churn (see ``identity.stable_key``).
    stable_key: str | None = None
    # Window layer: app | ime | system | overlay (hierarchy package heuristics).
    window: str | None = None
    # ``id`` of the nearest *collected* ancestor, or None for a root / vision element. Kept
    # because the acting control is often not a geometric container of the label that names
    # it: a design-system tile puts the click on an inner Box and renders the title as a
    # sibling **outside** those bounds, so containment cannot find one from the other and
    # only the tree can. See :func:`selectors.acting_node`.
    parent: int | str | None = None
    # What this control cost last time it was acted on: ``{"avg_ms": 4800, "max_ms": 4800,
    # "n": 3}``, learned per (screen, control) and scoped to the flag context. Absent unless
    # this exact control has history, so an unmeasured screen pays nothing.
    #
    # It lives on the element because it is the one thing about a control that the tree cannot
    # tell you, and every other home for it was worse. `meta.slow_controls` carries the same
    # numbers but is not in the `changed` meta preset every folded observation is trimmed to;
    # the derived `next_actions` list carried it and cost more than the whole element list to
    # do so. Priced onto the row it belongs to, "tap this next, and it takes ~4.8s" is one
    # read rather than a cross-reference.
    cost: dict[str, Any] | None = None

    def compact(self) -> dict[str, Any]:
        """Token-minimal dict: drop nulls and default-valued verbose fields."""
        out: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "bounds": list(self.bounds),
            "center": list(self.center),
        }
        if self.text is not None:
            out["text"] = self.text
        if self.resource_id is not None:
            out["resource_id"] = self.resource_id
        if self.content_desc is not None:
            out["content_desc"] = self.content_desc
        if self.clickable:
            out["clickable"] = True
        if not self.enabled:
            out["enabled"] = False
        if self.focused:
            out["focused"] = True
        out.update(self._compact_state())
        if self.source is not Source.hierarchy:
            out["source"] = self.source.value
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        if self.stable_key is not None:
            out["stable_key"] = self.stable_key
        if self.window is not None:
            out["window"] = self.window
        if self.parent is not None:
            out["parent"] = self.parent
        if self.cost is not None:
            out["cost"] = self.cost
        return out

    def _compact_state(self) -> dict[str, Any]:
        """Interaction-state flags worth their tokens.

        ``checked`` rides along whenever the node is ``checkable`` even when it is
        ``False``, because on a checkable node the *off* state is the payload — dropping
        it as a "default" would make a switch unreadable, which is the whole point of
        these fields.
        """
        out: dict[str, Any] = {}
        if self.checkable:
            out["checkable"] = True
            out["checked"] = self.checked
        elif self.checked:
            out["checked"] = True
        for name in ("selected", "scrollable", "long_clickable", "password"):
            if getattr(self, name):
                out[name] = True
        return out


class AppContext(BaseModel):
    """Foreground application identity independent of Android package/activity names.

    Runtime implementations should return this model. ``package``/``activity`` and ``get`` are
    compatibility projections for existing engine and plugin code while those public spellings
    remain supported.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: str | None = None
    surface_id: str | None = None

    @property
    def package(self) -> str | None:
        return self.app_id

    @property
    def activity(self) -> str | None:
        return self.surface_id

    def get(self, key: str, default: Any = None) -> Any:
        aliases = {
            "app_id": self.app_id,
            "package": self.app_id,
            "surface_id": self.surface_id,
            "activity": self.surface_id,
        }
        return aliases.get(key, default)

    def compatibility_dict(self) -> dict[str, str]:
        """Legacy Android-shaped mapping, omitting unavailable values."""

        return {
            key: value
            for key, value in {"package": self.app_id, "activity": self.surface_id}.items()
            if value is not None
        }

    @classmethod
    def coerce(cls, value: AppContext | Mapping[str, Any] | None) -> AppContext:
        """Read a neutral context or a legacy ``package``/``activity`` mapping."""

        if isinstance(value, cls):
            return value
        raw = value or {}
        return cls(
            app_id=raw.get("app_id") or raw.get("package"),
            surface_id=raw.get("surface_id") or raw.get("activity"),
        )


class Screen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    package: str | None = None
    activity: str | None = None
    source: ScreenSource

    @property
    def app_id(self) -> str | None:
        """Platform-neutral alias for the public ``package`` compatibility field."""

        return self.package

    @property
    def surface_id(self) -> str | None:
        """Platform-neutral alias for the public ``activity`` compatibility field."""

        return self.activity

    @property
    def app_context(self) -> AppContext:
        return AppContext(app_id=self.app_id, surface_id=self.surface_id)


class ObservationContract(BaseModel):
    """Machine-readable guidance for safely reusing one caller-visible frame."""

    model_config = ConfigDict(extra="forbid")

    fingerprint: str | None = None
    evidence_id: str | None = None
    produced_by: str
    reusable: bool
    analyze_needed: bool
    reason: str


class CallerTurn(BaseModel):
    """What the caller's own latency cost, and whether the screen it was holding is still up.

    aua knows both ends of every gap — the stamp it wrote when its last call returned, and the
    clock when this one started — so the caller's think time is free to compute. Measured over
    one 13-call agent session it was 75% of the elapsed wall time: the dominant term, and until
    now the one number aua discarded.

    It is reported because it is what the wait ceiling is sized from, and a caller that can see
    `ema_ms` can see why its budget is what it is. Note the arithmetic runs one way only: the
    estimate can lower the ceiling below `perf.max_wait_ms` but never raise it, so a caller
    reading a large `ema_ms` next to a 5000ms `wait_ceiling_ms` is being told that calling again
    is its cheaper move, not that aua will wait that long for it.
    """

    model_config = ConfigDict(extra="forbid")

    # Think time between the previous call returning and this one starting.
    gap_ms: int | None = None
    # Present when this gap was measured but deliberately not learned from: `idle` (longer than
    # any caller turn could be — someone walked away) or `clock` (time went backwards).
    gap_ignored: str | None = None
    ema_ms: int | None = None
    spread_ms: int | None = None
    samples: int | None = None
    # The ceiling one observation wait may not exceed, and which policy produced it:
    # `pinned` / `fixed` (reproducible) or `cold` / `adaptive` (sized from the measurement).
    wait_ceiling_ms: int | None = None
    wait_ceiling_mode: str | None = None
    # True when the screen described by the caller's *previous* result is no longer on the
    # device. A stale element id already raises; a wholly replaced screen said nothing at all,
    # so a caller could reason about a screen that had been gone for its entire thinking gap.
    # Deliberately a field and not prose, and deliberately not an error: acting anyway is a
    # legitimate choice and only the caller can make it.
    previous_screen_gone: bool | None = None
    previous_screen_age_ms: int | None = None

    @model_serializer
    def _only_what_was_measured(self) -> dict[str, Any]:
        """Drop unset fields on the way out, at every nesting depth.

        This block rides on every response, so the difference between an absent key and a null
        one is not cosmetic twice over: nulls are pure token cost on the path this project is
        most sensitive about, and `previous_screen_gone: null` actively misreads as "checked,
        the screen is fine" when the truth is "there was nothing to compare". The enclosing
        `Meta`/`ActionResult` dumps filter their own top-level Nones but cannot see inside a
        nested model, which is why this lives here rather than at the call site.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_ms: int
    tier_used: Tier
    path: PathKind
    providers_used: list[str] = Field(default_factory=list)
    known_screen: str | None = None  # recognised app-map screen name (PRD §6b, §8)
    # Navigation affordances pushed inline from app memory (§6b) so an agent gets them on
    # the analyze it already runs, instead of having to remember to call `aua map`.
    known_routes: list[str] = Field(default_factory=list)  # ["tap 'Catalog' → catalog", ...]
    suggested_gotos: list[str] = Field(default_factory=list)  # ["goto product_detail", ...]
    suggested_deeplinks: list[str] = Field(  # ["open myapp://dashboard", ...] — shortcuts
        default_factory=list
    )
    research_tasks: list[str] = Field(default_factory=list)
    # Saved journeys for THIS app, as ``name(PARAM, …)``. A flow replays a whole sequence in one
    # call, so it belongs next to the routes rather than in a manual nobody opens.
    flows: list[str] = Field(default_factory=list)
    # One question about THIS screen, answerable on the caller's next command via `--answers`.
    # Scoped to where the caller is standing, because that is the only question they can answer
    # from what is in front of them.
    ask: dict[str, str] | None = None
    map_hint: str | None = None  # e.g. "12 screens mapped — run `aua map`"
    # Controls on THIS screen that history says are slow, worst first:
    # [{"control": "buttonContinue", "avg_ms": 4800, "max_ms": 6100, "n": 3, ...}].
    # Told to the agent on arrival so a long wait is planned rather than discovered as a timeout.
    slow_controls: list[dict[str, Any]] = Field(default_factory=list)
    capture_hint: str | None = None  # rolling buffer saw post-action change
    # The hierarchy handed us text it could not represent (U+FFFD). Silence here is worse
    # than slowness: an agent reads "Divide both sides by 2 to solve for <?>: <?>", cannot
    # see the formula, and either reports a wrong observation or falls back to eyeballing
    # screenshots on its own. Say so, and name the flag that recovers it.
    lossy_text: bool = False
    lossy_hint: str | None = None
    # How many broken labels OCR successfully repaired in place on this analyze.
    ocr_repaired: int = 0
    annotated_image: str | None = None
    raw_image: str | None = None  # unannotated screenshot saved on request (--with-image)
    device_serial: str | None = None
    device_locale: str | None = None  # BCP-47 UI locale (e.g. "es-ES"); labels render in it
    observation_contract: ObservationContract | None = None
    # Optional token-cheap delta vs the previous analyze (perf.differential).
    element_diff: dict[str, Any] | None = None
    # Host-side incremental a11y: True when hierarchy XML matched the previous analyze.
    unchanged: bool = False
    # The screen the caller was last handed was already gone *before* this call touched the
    # device: an interstitial, a permission dialog, a session-expiry sheet, a push. Settle and
    # arrival logic are blind to this by construction — nothing the caller did caused it — and
    # `caller.previous_screen_gone` answered it only for whoever remembered to read a telemetry
    # block that costs 199 B on every response. This is the same fact, delivered instead of
    # measured: absent unless something the caller could act on moved on its own, which makes
    # its mere presence the warning. A *string* for `stale_risk`'s reasons (compact drops
    # falsey values), and it carries the drift so a reader can judge it.
    screen_moved: str | None = None
    # Why `unchanged` / `element_diff` must NOT be read as evidence about an action's effect:
    # set on a post-action observation whose settle wait never confirmed the screen had moved,
    # so the frame it describes may predate the action. A *string* rather than a False flag
    # because `compact` drops falsey values and `delta` allowlists keys — a boolean would
    # vanish from output in exactly the cases that need it.
    stale_risk: str | None = None
    # Where the arrival state machine landed for the action this observation is folded
    # into: settled / no_change / loading / transitioning / unconfirmed. None on a plain
    # `analyze` and on a settled action — like `screen_moved` and `stale_risk`, the key
    # appearing IS the warning, so a healthy response pays nothing for it.
    arrival_state: str | None = None
    # SHA1 of the raw hierarchy XML (or elements fingerprint for vision paths).
    fingerprint: str | None = None
    # How the result was produced (e.g. hierarchy, hierarchy-unchanged, vision).
    via: str | None = None
    caller: CallerTurn | None = None
    # Active goal-session phase and one exact next call. Attached best-effort after perception;
    # absent outside a goal session.
    goal_progress: dict[str, Any] | None = None


class AnalyzeResult(BaseModel):
    """Top-level ``analyze`` payload (PRD §8)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    screen: Screen
    elements: list[Element]
    meta: Meta

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _publish_identity(data: dict[str, Any]) -> dict[str, Any]:
        """Rewrite every published ``id`` to the element's stable key.

        The integer ``id`` is a frame-local ordinal: reading order, renumbered on every
        analyze, and validated through one cache file per device that all callers of that
        device share. Publishing it made the number the caller's handle on the element, which
        is wrong in two directions — it goes stale the moment the screen moves, and a caller
        holding an observation produced by another process (the dashboard, a second agent, a
        saved report) is validated against whoever wrote that file last.

        So the ordinal stays internal — every Python path still indexes by it — and what goes
        out is the one name that outlives the frame. `parent` is remapped through the same
        table, because a parent pointer that still held an ordinal would name a different
        element than the id it points at. `meta.element_diff` too: an added/removed list of
        ordinals is unreadable next to string ids, and worse, silently comparable to the wrong
        row. Actions accept what is published — see `Engine._resolve_action_key`.
        """
        elements = data.get("elements")
        if not isinstance(elements, list):
            return data
        # Every element gets one, with no exceptions: a payload that mixed string ids with
        # leftover integers would make the caller guess which kind it is holding, and
        # `identity.stable_key` always has an answer — a resource-id, a label hash, or a
        # geometry fingerprint as the last resort. Imported here because `identity` imports
        # this module.
        from .identity import stable_key as _stable_key
        from .identity import uniquify_keys as _uniquify_keys

        # Uniqueness is applied here, not merely inherited. `attach_stable_keys` numbers
        # repeats during analyze, but a payload can reach a boundary without having gone
        # through it, and then two rows built from one reusable layout publish the same id.
        # That was survivable while `rid` rode along in the default view; once the id became
        # the only name on the row it made them indistinguishable.
        rows: list[tuple[Any, str, Any]] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            key = element.get("stable_key") or _stable_key(element)
            if key:
                rows.append((element.get("id"), str(key), element.get("bounds")))
        if not rows:
            return data
        by_ordinal: dict[Any, Any] = _uniquify_keys(rows)
        for element in elements:
            if isinstance(element, dict) and element.get("id") in by_ordinal:
                element["stable_key"] = by_ordinal[element["id"]]
        for element in elements:
            if not isinstance(element, dict):
                continue
            if element.get("id") in by_ordinal:
                element["id"] = by_ordinal[element["id"]]
            if element.get("parent") in by_ordinal:
                element["parent"] = by_ordinal[element["parent"]]
        meta = data.get("meta")
        diff = meta.get("element_diff") if isinstance(meta, dict) else None
        if isinstance(diff, dict):
            # `added`/`removed` are flat id lists; `changed` is a list of
            # `{"id": …, "text": {"from": …, "to": …}}` rows. Treating all three the same way
            # meant calling `by_ordinal.get(row)` with a dict, which raises
            # `TypeError: unhashable type: 'dict'` — and only when something actually changed
            # between two frames, so it surfaced as an intermittent `internal_error` on real
            # taps while every fixture with an empty diff passed.
            for key in ("added", "removed"):
                flat = diff.get(key)
                if isinstance(flat, list):
                    diff[key] = [
                        by_ordinal.get(row, row) if isinstance(row, (int, str)) else row
                        for row in flat
                    ]
            changed = diff.get("changed")
            if isinstance(changed, list):
                for row in changed:
                    if isinstance(row, dict) and row.get("id") in by_ordinal:
                        row["id"] = by_ordinal[row["id"]]
        return data

    def as_dict(self, fmt: OutputFormat | str = OutputFormat.json) -> dict[str, Any]:
        """The serialisable payload for *fmt* (``compact`` trims to the smallest footprint).

        Shared by :meth:`render` and by :class:`ActionResult` so an embedded ``observation``
        renders in the same format as a standalone ``analyze``.
        """
        fmt = OutputFormat(fmt)
        if fmt is OutputFormat.delta:
            base = self.as_dict(OutputFormat.compact)
            if self.meta.unchanged:
                base["elements"] = []
                base["meta"] = {
                    k: v
                    for k, v in base.get("meta", {}).items()
                    if k
                    in {
                        "duration_ms",
                        "tier_used",
                        "path",
                        "unchanged",
                        "fingerprint",
                        "via",
                        "element_diff",
                        "device_serial",
                        "known_screen",
                        # Must survive the delta trim: `delta` fires precisely when
                        # `unchanged` is True, which is the case whose caveat matters.
                        "stale_risk",
                        # Same argument, one step further out: `unchanged` compares this read
                        # against the previous one *this engine* took, which under a warm daemon
                        # is not necessarily the screen this caller was handed. A trim that drops
                        # the warning is a trim that hides the only case it exists for.
                        "screen_moved",
                        # The machine-readable form of the same warning: `delta` fires when
                        # `unchanged` is True, which is exactly the `no_change` state.
                        "arrival_state",
                    }
                    and v not in (None, [], False)
                }
                base["meta"]["unchanged"] = True
            return base
        if fmt is OutputFormat.compact:
            return self._publish_identity({
                "schema_version": self.schema_version,
                "screen": {
                    k: v for k, v in self.screen.model_dump(mode="json").items() if v is not None
                },
                "elements": [e.compact() for e in self.elements],
                "meta": {
                    k: v
                    for k, v in self.meta.model_dump(mode="json").items()
                    # `v not in (None, [], False)` compares with ==, and 0 == False, so a
                    # `duration_ms` of 0 was dropped — the payload then failed its own schema
                    # (duration_ms is required), and only on FAST calls, i.e. the ones the
                    # trimming exists for. Test `False` by identity.
                    if v is not None and v != [] and v is not False
                },
            })
        return self._publish_identity(self.model_dump(mode="json"))

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        """Serialise to one of the output formats (PRD §8)."""
        fmt = OutputFormat(fmt)
        if fmt is OutputFormat.msgpack:
            from .binary_dump import pack_analyze_b64

            return pack_analyze_b64(self)
        data = self.as_dict(fmt)
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)

    def element_by_id(self, element_id: ElementId) -> Element | None:
        for e in self.elements:
            if e.id == element_id:
                return e
        return None



def publish_ids(payload: Any) -> Any:
    """Rewrite ids to stable ids anywhere an observation sits in *payload*, in place.

    The boundary function. :meth:`AnalyzeResult.as_dict` publishes for callers who use it, but
    `model_dump(mode="json")` is the more obvious method and three surfaces reached for it —
    MCP, the CLI's projection path, and the engine sites that embed an observation in a larger
    response — so those kept handing out frame ordinals.

    Deliberately *not* a model serializer, which would be the tidier place: `model_dump` is
    also the internal form. The analyze cache is written and read through it and a numeric
    action resolves against that file, so publishing inside the model turned the cache into
    something its own resolver could not read. The ordinal stays internal; this is where it
    stops being internal.

    Idempotent, so a payload that passed a boundary twice is unchanged by the second.
    """
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("elements"), list):
        AnalyzeResult._publish_identity(payload)
    nested = payload.get("observation")
    if isinstance(nested, dict) and isinstance(nested.get("elements"), list):
        AnalyzeResult._publish_identity(nested)
    return payload


# --------------------------------------------------------- lightweight result models


class HasResult(BaseModel):
    """Result of the ``has`` quick-check (PRD §5 quick checks)."""

    model_config = ConfigDict(extra="forbid")

    found: bool
    source: str | None = None  # "hierarchy" | "ocr"
    bounds: Bounds | None = None
    text: str | None = None
    # On a miss only: the device UI locale plus, for text lookups, a hint that labels
    # render in that locale — match observed labels or go --by id. Language-neutral:
    # the query may be written in any language, so no locale is treated as "safe".
    # On a hit through the mined-strings bridge, `hint` names the key and rendering
    # that matched (and `text` carries the rendering that was actually found).
    device_locale: str | None = None
    hint: str | None = None
    wait_clamped_from_ms: int | None = None
    wait_ceiling_ms: int | None = None
    wait_ceiling_mode: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class AppStatusResult(BaseModel):
    """Read-only package-manager status for one app on the selected target."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    action: str = "app-status"
    package: str
    installed: bool
    serial: str
    version_name: str | None = None
    version_code: str | None = None
    mode: str = "read-only"

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ShellResult(BaseModel):
    """Structured result of one bounded, read-only command on the selected target."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str = "shell"
    serial: str
    argv: list[str]
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_limit_bytes: int = 256 * 1024
    exit_code: int
    duration_ms: int
    mode: str = "read-only"

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = self.model_dump(mode="json")
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ActionResult(BaseModel):
    """Result of an action command (tap/input/swipe/key/...)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str
    # Inline hint when an action already returns usable screen state. Declared this high on
    # purpose: it is the sentence that stops a caller spending a second round trip on an
    # `analyze` it does not need, and it has to be read *before* the observation it describes,
    # not after it. Buried under the optional diagnostics it was read last, once the decision it
    # exists to prevent had already been made.
    note: str | None = None
    id: ElementId | None = None
    target: list[int] | None = None  # coords or bounds acted on
    detail: str | None = None
    # ``--submit`` dispatches an IME action, but many apps keep the text in the composer and
    # expose their own semantic Send/Confirm control. Distinguish "text was typed" from "the
    # app accepted a submission" so a caller never reports a sent message from the IME call
    # alone. None means the returned frame could not prove either outcome.
    submitted: bool | None = None
    # One exact recovery call when the action itself can identify it, for example the unique
    # semantic send control after an IME submit left the composer populated.
    recommended_call: dict[str, Any] | None = None
    # The screen right after the action (when called with observe=True), so an agent gets
    # fresh element ids without a separate `analyze` round-trip (act + observe in one call).
    observation: AnalyzeResult | None = None
    # Stable contract fields for downstream agents. `observation_present` is always returned on
    # an *action* response, so callers can branch without checking for key existence. Both are
    # None by default rather than False/[] because `render()` only strips None: a plain default
    # would emit these on commands that perform no action at all, and "observation_present:
    # false" on a `screenshot` claims the effect of an action was not observed when there was no
    # action to observe. `_observe` sets `observation_present` on both of its branches, which is
    # what makes the field reliably present exactly where the contract promises it.
    observation_present: bool | None = None
    # Route context on the action response itself.
    known_screen: str | None = None
    # Compact diff summary from the folded observation (`meta.element_diff` transformed).
    action_diff_summary: dict[str, Any] | None = None
    # Structured one-call efficiency recommendation (CLI and MCP share the same ids/text).
    advice: list[dict[str, str]] | None = None
    # What an install actually did to the target, as data rather than prose: {"package": ...,
    # "installed": bool, "pushed": bool, "reason": "already-present"|"missing"|…, "version_name":
    # …, "bundle_version_name": …}. `pushed` is the field that matters to a caller deciding
    # whether app state survived — an idempotent skip preserves data, a push may not — and a
    # sentence in `detail` cannot be branched on.
    app_install: dict[str, Any] | None = None
    goal_progress: dict[str, Any] | None = None
    observation_contract: ObservationContract | None = None
    # Why the folded observation may not show the action's effect — e.g. the post-action screen was
    # byte-identical to the pre-action one. Carries the reason, not a bare flag, so a reader can
    # judge it. Absent when the observation is trustworthy, so its presence is the signal.
    stale_risk: str | None = None
    # The arrival verdict, machine-readable: {"state", "evidence": [detector names],
    # "waited_ms"?}. Present only when the state is not a silent settled arrival — absence
    # means "arrived, nothing to say", presence means "read me before acting". Named
    # evidence, never a confidence score. `stale_risk` and `note` carry the same verdict as
    # prose and remain the compatibility surface.
    arrival: dict[str, Any] | None = None
    # What the post-action wait cost: {"ms": int, "via": str, "anim": ...}. Answers "why did that
    # tap take 300 ms" as data rather than as a tag glued onto `detail`.
    settle: dict[str, Any] | None = None
    # Real wall time for the whole call, device round trips included. `settle.ms` and the
    # observation's `meta.duration_ms` each measure one phase, and in the field their sum
    # understated a 5.26s call as 247ms — a caller budgeting from those numbers is budgeting
    # from a fiction. This is the number a stopwatch would show.
    wall_ms: int | None = None
    # The deliberate post-action pause actually spent, so a latency sweep can attribute cost
    # to the knob rather than inferring it (see `perf.stable_delay_ms`).
    stable_delay_ms: int | None = None
    # Set when the caller asked to wait longer than `perf.max_wait_ms` allows. Carries the
    # *requested* value so the response explains itself: the wait was shortened on purpose and
    # "nothing yet" must not be read as "nothing there". `wait_ceiling_ms` is what was enforced.
    wait_clamped_from_ms: int | None = None
    wait_ceiling_ms: int | None = None
    # Whether that ceiling was measured (`cold`/`adaptive`) or pinned (`fixed`/`pinned`).
    wait_ceiling_mode: str | None = None
    # What the caller's own latency cost, and whether its previous screen is still there.
    caller: CallerTurn | None = None
    # True when the folded observation came back with no visible elements. An action that
    # reports success while handing back an empty screen is the defect this names: downstream
    # the agent burns a turn, and the screen it wanted may already have arrived.
    observation_empty: bool | None = None
    # A bounded wait reached its ceiling with the screen still moving. Distinct from an error:
    # the screen is returned, nothing is known to be wrong, and the caller may simply ask again.
    settled_unmet: bool | None = None
    # What can be done from the screen this action landed on, as a filtered projection of
    # `observation.elements`: [{"id": "rid:submit", "label": "Submit", "rid": "submit"}].
    #
    # **Off unless `output.next_actions` is set**, and that default is measured. It existed to
    # remove a *reasoning* step — an agent scanning ~50 observation nodes for the ones it could
    # act on — and that scan no longer exists: the folded observation is trimmed to ~20 rows
    # with `clickable` on each, so `[e for e in observation.elements if e["clickable"]]` is the
    # same answer for free. On one real journalled response the list cost 1384 bytes / 346
    # tokens (25% of the whole response) to restate 12 rows of a 1301-byte `elements` list. The
    # learned per-control cost, the one thing `elements` could not express, moved to
    # `Element.cost`.
    next_actions: list[dict[str, Any]] | None = None
    # Navigation shortcuts out of here that memory already knows — merged `known_routes` and
    # `suggested_gotos`, hoisted so they are visible without opening `observation.meta`.
    routes: list[str] | None = None
    # Rolling capture saw a post-action pixel change — pull `aua capture last`.
    capture_hint: str | None = None
    # Did the action's effect get *observed*, as opposed to merely attempted?
    # True  = read back and confirmed, False = read back and provably did not happen,
    # None  = not checked, or checked and genuinely ambiguous. Absent from output when None,
    # so `ok` keeps its meaning for actions that cannot verify themselves.
    verified: bool | None = None
    # Which node actually received the interaction, and how it relates to the one named. The
    # control a label belongs to is frequently a different node — a design-system tile puts
    # the click on an inner Box and renders the title outside it — so "I acted on what you
    # named" and "I acted on its sibling" must not look the same in output.
    acting: dict[str, Any] | None = None
    # `await`: which terminal condition ended the wait.  In addition to satisfied /
    # screen-changed / timeout, an action-bound wait can return settled-unmet when the action
    # demonstrably reached a different, stable destination but the caller's positive arrival
    # term names something that is not there.  Standalone waits deliberately retain their
    # package/activity-only three-outcome semantics.
    # What actually changed, rather than a claim that the action dispatched: activity before and
    # after, node-count delta, focus movement, text added/removed. Deliberately carries no
    # confidence score — a number invites trusting a figure over evidence, and "a command
    # reporting success is not evidence of effect" is this project's first lesson. `changed` is
    # an explicit boolean (None = no baseline to compare) so "nothing changed" is checkable.
    change: dict[str, Any] | None = None
    # A bounded fatal/ANR/error block read automatically from the action's own diagnostic-log
    # window when the app leaves the foreground. This is structured and top-level so CLI and MCP
    # callers receive the cause without parsing the warning or spending another tool call.
    crash_evidence: dict[str, Any] | None = None
    # What the app itself logged in this action's own window, scoped to its process and reduced
    # to a priority set plus a line budget. The screen is the app's conclusion; this is its
    # reasoning, and between them sits the failure an agent otherwise cannot see: the tap
    # landed, the screen looks plausible, and the app quietly logged the refusal. Absent when
    # the window was empty, so a quiet action costs nothing.
    app_logs: dict[str, Any] | None = None
    await_outcome: str | None = None
    # Per-term results, reported satisfied or not: *which* term is missing is how a reader
    # tells a failed load from a slow one.
    await_terms: list[dict[str, Any]] | None = None
    # Structured correction for ``await_outcome=settled-unmet``.  Keeping this separate from
    # prose lets adapters and agents offer a corrected predicate without parsing ``detail``.
    arrival_mismatch: dict[str, Any] | None = None
    # Unmet positive ``rid:`` terms that no mapped screen of this app has ever carried. An
    # unmet term reads as "not there yet" whether or not it *can* be there; naming the ones
    # that cannot is what stops an agent inventing a second id and waiting on that too.
    unknown_selectors: list[dict[str, Any]] | None = None
    elapsed_ms: int | None = None
    # Bounded multi-step actions expose why they stopped and a compact semantic hop trace.
    stop_reason: str | None = None
    steps_run: list[dict[str, Any]] | None = None
    # Locale-bridge diagnosis: on a translated hit, which string key and rendering matched;
    # on a text miss, what the target renders as in the device locale (see Meta.device_locale).
    hint: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        fmt = OutputFormat(fmt)
        # `observation` is re-rendered in *fmt* rather than taken from `model_dump`, but it keeps
        # its declared position instead of being appended: excluding it and adding it back put
        # the screen — the thing the caller asked for — dead last, behind a dozen optional
        # diagnostics. A reader that stops early should hit the screen, not miss it.
        rendered = self.observation.as_dict(fmt) if self.observation is not None else None
        data = {
            k: (rendered if k == "observation" else v)
            for k, v in self.model_dump(mode="json").items()
            if v is not None
        }
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class NetworkState(BaseModel):
    """Observed Android connectivity controls and the active default network."""

    model_config = ConfigDict(extra="forbid")

    airplane_mode: bool | None = None
    wifi_supported: bool | None = None
    wifi_enabled: bool | None = None
    cellular_supported: bool | None = None
    mobile_data_enabled: bool | None = None
    active_network: bool | None = None
    active_network_id: str | None = None
    active_transports: list[str] = Field(default_factory=list)
    internet_validated: bool | None = None
    offline: bool | None = None


class NetworkShaping(BaseModel):
    """Observed bandwidth, latency, or packet-loss shaping for a network profile."""

    model_config = ConfigDict(extra="forbid")

    mechanism: str
    upload_bps: int | None = None
    download_bps: int | None = None
    min_latency_ms: int | None = None
    max_latency_ms: int | None = None
    interface: str | None = None
    loss_percent: float | None = None
    qdisc: str | None = None
    root_enabled: bool | None = None


class NetworkResult(BaseModel):
    """Result of deterministic network status/offline/restore operations."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str
    state: NetworkState
    saved_state: NetworkState | None = None
    profile: str | None = None
    shaping: NetworkShaping | None = None
    verified: bool | None = None
    detail: str | None = None
    goal_progress: dict[str, Any] | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ResolveResult(BaseModel):
    """Remap a previous-frame id or ``stable_key`` onto the current screen."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    from_id: ElementId | None = None
    to_id: ElementId | None = None
    stable_key: str | None = None
    element: Element | None = None
    detail: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class TargetInfo(BaseModel):
    """Neutral discovery result returned by independently packaged adapters.

    The compatibility properties let the existing engine consume this model while Android keeps
    returning its established :class:`DeviceInfo` wire shape.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str
    platform: str
    status: TargetStatus = TargetStatus.online
    model: str | None = None
    os_name: str | None = None
    os_version: str | None = None
    locale: str | None = None
    native_status: str | None = None

    @property
    def serial(self) -> str:
        return self.target_id

    @property
    def state(self) -> str:
        """Legacy transport state used by pre-platform target selection."""

        if self.native_status:
            return self.native_status
        return "device" if self.status is TargetStatus.online else self.status.value

    @property
    def android_version(self) -> str | None:
        return self.os_version if self.platform == "android" else None


class DeviceInfo(BaseModel):
    """Android discovery compatibility shape.

    New platform plugins return :class:`TargetInfo`; keeping this model unchanged preserves the
    exact JSON emitted by ``aua devices`` for Android callers.
    """

    model_config = ConfigDict(extra="forbid")

    serial: str
    model: str | None = None
    android_version: str | None = None
    locale: str | None = None
    state: str = "device"

    @property
    def target_id(self) -> str:
        return self.serial

    @property
    def platform(self) -> str:
        return "android"

    @property
    def status(self) -> TargetStatus:
        return {
            "device": TargetStatus.online,
            "offline": TargetStatus.offline,
        }.get(self.state, TargetStatus.unavailable)

    @property
    def os_name(self) -> str:
        return "android"

    @property
    def os_version(self) -> str | None:
        return self.android_version

    def as_target_info(self) -> TargetInfo:
        return TargetInfo(
            target_id=self.target_id,
            platform=self.platform,
            status=self.status,
            model=self.model,
            os_name=self.os_name,
            os_version=self.os_version,
            locale=self.locale,
            native_status=self.state,
        )
