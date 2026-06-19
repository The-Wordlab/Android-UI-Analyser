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
    """One actionable thing on screen, identified by a stable integer ``id``."""

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
    source: Source = Source.hierarchy
    confidence: float | None = None

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
        if self.source is not Source.hierarchy:
            out["source"] = self.source.value
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
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
    map_hint: str | None = None  # e.g. "12 screens mapped — run `aua map`"
    annotated_image: str | None = None
    device_serial: str | None = None


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
                    if v not in (None, [])
                },
            }
        return self.model_dump(mode="json")

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        """Serialise to one of the three output formats (PRD §8)."""
        fmt = OutputFormat(fmt)
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


class DeviceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serial: str
    model: str | None = None
    android_version: str | None = None
    state: str = "device"
