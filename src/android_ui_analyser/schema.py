"""Canonical, versioned output schema (PRD §8).

This module is the **single source of truth** for the shape of everything the CLI and
MCP server emit. Pydantic models here are imported by the engine, the CLI, the MCP
wrapper, and the tests. Do not duplicate these shapes elsewhere.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


def center_of(bounds: Bounds) -> Center:
    """Geometric center of an ``[x1, y1, x2, y2]`` box."""
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) // 2, (y1 + y2) // 2)


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

    id: int
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


class Screen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    package: str | None = None
    activity: str | None = None
    source: ScreenSource


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_ms: int
    tier_used: Tier
    path: PathKind
    providers_used: list[str] = Field(default_factory=list)
    known_screen: str | None = None  # recognised app-map screen name (PRD §6b, §8)
    # Navigation affordances pushed inline from app memory (§6b) so an agent gets them on
    # the analyze it already runs, instead of having to remember to call `aua map`.
    known_routes: list[str] = Field(default_factory=list)  # ["tap 'Apps' → apps", ...]
    suggested_gotos: list[str] = Field(default_factory=list)  # ["goto image_creator", ...]
    suggested_deeplinks: list[str] = Field(  # ["open myapp://home", ...] — shortcuts
        default_factory=list
    )
    research_tasks: list[str] = Field(default_factory=list)
    map_hint: str | None = None  # e.g. "12 screens mapped — run `aua map`"
    capture_hint: str | None = None  # rolling buffer saw post-action change
    # The hierarchy handed us text it could not represent (U+FFFD). Silence here is worse
    # than slowness: an agent reads "Divide both sides by 2 to solve for <?>: <?>", cannot
    # see the formula, and either reports a wrong observation or falls back to eyeballing
    # screenshots on its own. Say so, and name the flag that recovers it.
    lossy_text: bool = False
    lossy_hint: str | None = None
    annotated_image: str | None = None
    raw_image: str | None = None  # unannotated screenshot saved on request (--with-image)
    device_serial: str | None = None
    # Optional token-cheap delta vs the previous analyze (perf.differential).
    element_diff: dict[str, Any] | None = None
    # Host-side incremental a11y: True when hierarchy XML matched the previous analyze.
    unchanged: bool = False
    # SHA1 of the raw hierarchy XML (or elements fingerprint for vision paths).
    fingerprint: str | None = None
    # How the result was produced (e.g. hierarchy, hierarchy-unchanged, vision).
    via: str | None = None


class AnalyzeResult(BaseModel):
    """Top-level ``analyze`` payload (PRD §8)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    screen: Screen
    elements: list[Element]
    meta: Meta

    # -- rendering ---------------------------------------------------------

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
                    }
                    and v not in (None, [], False)
                }
                base["meta"]["unchanged"] = True
            return base
        if fmt is OutputFormat.compact:
            return {
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
            }
        return self.model_dump(mode="json")

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

    def element_by_id(self, element_id: int) -> Element | None:
        for e in self.elements:
            if e.id == element_id:
                return e
        return None


# --------------------------------------------------------- lightweight result models


class HasResult(BaseModel):
    """Result of the ``has`` quick-check (PRD §5 quick checks)."""

    model_config = ConfigDict(extra="forbid")

    found: bool
    source: str | None = None  # "hierarchy" | "ocr"
    bounds: Bounds | None = None
    text: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ActionResult(BaseModel):
    """Result of an action command (tap/input/swipe/key/...)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str
    id: int | None = None
    target: list[int] | None = None  # coords or bounds acted on
    detail: str | None = None
    # The screen right after the action (when called with observe=True), so an agent gets
    # fresh element ids without a separate `analyze` round-trip (act + observe in one call).
    observation: AnalyzeResult | None = None
    # Rolling capture saw a post-action pixel change — pull `aua capture last`.
    capture_hint: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        fmt = OutputFormat(fmt)
        data = {
            k: v
            for k, v in self.model_dump(mode="json").items()
            if v is not None and k != "observation"
        }
        if self.observation is not None:
            data["observation"] = self.observation.as_dict(fmt)
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ResolveResult(BaseModel):
    """Remap a previous-frame id or ``stable_key`` onto the current screen."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    from_id: int | None = None
    to_id: int | None = None
    stable_key: str | None = None
    element: Element | None = None
    detail: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class DeviceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serial: str
    model: str | None = None
    android_version: str | None = None
    state: str = "device"
