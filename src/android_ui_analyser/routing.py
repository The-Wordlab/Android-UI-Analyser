"""Cost-aware routing / escalation ladder (PRD §6a) — pure, testable helpers.

The router decides the **entry tier** from the command's intent + a cheap (regex/keyword,
NO LLM) query classifier, and the **ceiling** from ``routing.max_tier`` plus the
per-call ``--cheap`` / ``--deep`` overrides. The engine consumes these and escalates
only on a miss/low-confidence, never silently past the ceiling and never to a paid
provider unless the ceiling explicitly allows it.
"""

from __future__ import annotations

import re
from enum import Enum

from .schema import TIER_ORDER, Tier, tier_rank

RESOURCE_ID_RE = re.compile(r"[\w.]+:id/[\w.]+")

VISUAL_WORDS: frozenset[str] = frozenset(
    {
        "icon",
        "image",
        "picture",
        "logo",
        "avatar",
        "thumbnail",
        "banner",
        "glyph",
        "top",
        "bottom",
        "left",
        "right",
        "corner",
        "center",
        "centre",
        "middle",
        "near",
        "above",
        "below",
        "beside",
        "leftmost",
        "rightmost",
        "upper",
        "lower",
        "color",
        "colour",
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "orange",
        "purple",
        "pink",
        "gray",
        "grey",
        "circle",
        "square",
        "arrow",
        "hamburger",
        "gear",
        "cog",
        "magnifier",
        "magnifying",
        "looks",
    }
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "button",
        "btn",
        "icon",
        "tab",
        "field",
        "input",
        "box",
        "on",
        "screen",
        "please",
        "click",
        "tap",
        "press",
        "select",
        "choose",
        "that",
        "says",
        "saying",
        "labeled",
        "labelled",
        "label",
        "with",
        "text",
        "to",
        "of",
        "for",
        "my",
        "this",
        "in",
        "at",
        "is",
        "and",
        "or",
        "element",
        "item",
        "menu",
    }
)


class QueryKind(str, Enum):
    resource_id = "resource_id"  # looks like pkg:id/name → selector
    literal = "literal"  # a quoted/short exact phrase → selector/hierarchy
    visual = "visual"  # visual/relational language → candidate for vision/grounding
    general = "general"  # default: try hierarchy first


class Intent(str, Enum):
    has = "has"  # boolean presence check
    locate = "locate"  # find a known selector to act on
    analyze = "analyze"  # enumerate the screen
    query = "query"  # analyze --query "<nl>"


def classify_query(query: str) -> QueryKind:
    q = query.strip()
    if RESOURCE_ID_RE.search(q):
        return QueryKind.resource_id
    low = q.lower()
    tokens = re.findall(r"[a-z0-9]+", low)
    if any(tok in VISUAL_WORDS for tok in tokens):
        return QueryKind.visual
    # short, mostly-non-stopword phrase → treat as a literal we can match in the tree
    salient = [t for t in tokens if t not in _STOPWORDS]
    if 0 < len(salient) <= 4:
        return QueryKind.literal
    return QueryKind.general


def salient_tokens(query: str) -> list[str]:
    """Content tokens for hierarchy matching (strip command/UI stopwords)."""
    low = query.lower()
    toks = re.findall(r"[a-z0-9]+", low)
    out = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
    return out or [t for t in toks if len(t) > 1]


def entry_tier(
    intent: Intent, *, query: str | None = None, semantic_hierarchy_first: bool = True
) -> Tier:
    """The cheapest tier that *could* answer this request."""
    if intent is Intent.has:
        return Tier.text
    if intent is Intent.locate:
        return Tier.selector
    if intent is Intent.analyze:
        return Tier.hierarchy
    # query
    kind = classify_query(query or "")
    if kind in (QueryKind.resource_id, QueryKind.literal):
        return Tier.selector
    if kind is QueryKind.visual and not semantic_hierarchy_first:
        return Tier.vision
    return Tier.hierarchy


def resolve_ceiling(max_tier: Tier | str, *, cheap: bool = False, deep: bool = False) -> Tier:
    """Effective ceiling for one call: ``--cheap`` lowers it, ``--deep`` raises it."""
    rank = tier_rank(max_tier)
    if deep:
        rank = tier_rank(Tier.grounding)
    if cheap:
        rank = max(0, rank - 1)
    return TIER_ORDER[rank]


def next_tier(current: Tier, ceiling: Tier) -> Tier | None:
    """The next rung up, or None if already at the ceiling."""
    cur, cap = tier_rank(current), tier_rank(ceiling)
    if cur >= cap:
        return None
    return TIER_ORDER[cur + 1]


def allows(tier: Tier, ceiling: Tier) -> bool:
    return tier_rank(tier) <= tier_rank(ceiling)
