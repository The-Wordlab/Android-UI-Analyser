"""Quality gate: is the parsed hierarchy good enough, or do we need vision? (PRD §6 step 3)

This is the T2→T3 rung of the escalation ladder. Given the elements parsed from the
hierarchy (plus the foreground package/activity), :func:`decide` applies a few cheap,
fully-configurable heuristics and returns whether the engine should fall back to the
local vision path — together with a one-line human ``reason`` (surfaced in logs / so a
caller can see *why* it climbed).

The rules, in order (first match wins, all thresholds from :class:`GateCfg`):

1. **Too few elements** — ``len(elements) < cfg.min_elements`` → vision.
2. **No semantics** — not a single element carries ``text`` or ``content_desc``
   (likely a custom-drawn / canvas / game surface) → vision.
3. **Known-opaque package/class** — the package, activity, or any element's type
   matches one of ``cfg.vision_packages`` (Flutter, Unity, SDL, WebView, …). Patterns
   are matched both as a glob (``*.WebView``) and as a plain substring (``io.flutter``)
   → vision.
4. **Poorly-labeled controls** — of the *clickable* elements, the fraction that carry a
   text/content-desc label is below ``cfg.min_labeled_ratio`` → vision. Skipped (never
   fires) when there are no clickable elements, to avoid divide-by-zero.

If none fire, the hierarchy is trusted (``use_vision=False``).
"""

from __future__ import annotations

import typing
from fnmatch import fnmatch

from .config import GateCfg
from .schema import Element


class GateDecision(typing.NamedTuple):
    """Outcome of :func:`decide`. Unpacks as ``(use_vision, reason)``."""

    use_vision: bool
    reason: str


def _has_label(el: Element) -> bool:
    """An element is "labeled" if it has any text or content-desc the agent can read."""
    return bool(el.text) or bool(el.content_desc)


def _pattern_matches(pattern: str, value: str | None) -> bool:
    """Match ``value`` against a vision_packages ``pattern``.

    Supports shell globs (``*.WebView``) via :func:`fnmatch`, and treats a plain
    pattern (no glob metacharacters) as a case-insensitive substring so ``"io.flutter"``
    matches ``"io.flutter.embedding.android.FlutterActivity"``.
    """
    if not value:
        return False
    value_l = value.lower()
    pattern_l = pattern.lower()
    if any(ch in pattern for ch in "*?["):
        return fnmatch(value_l, pattern_l)
    return pattern_l in value_l


def _pattern_matches_class(pattern: str, short_type: str | None) -> bool:
    """Match a pattern against an element's *short* class name.

    Element ``type`` is the short name (``WebView``), but vision_packages patterns are
    typically written against fully-qualified names (``*.WebView``, ``android.webkit.*``).
    So we test the pattern against the short type directly *and* against a synthetic
    fully-qualified-looking form, and also fnmatch the pattern's trailing segment — so
    ``*.WebView`` still catches a ``WebView`` element.
    """
    if not short_type:
        return False
    if _pattern_matches(pattern, short_type):
        return True
    # treat a dotted glob's trailing literal as a class-name pattern
    trailing = pattern.rsplit(".", 1)[-1]
    return bool(trailing and trailing != pattern and fnmatch(short_type.lower(), trailing.lower()))


def _matched_vision_package(
    patterns: list[str], package: str | None, activity: str | None, elements: list[Element]
) -> str | None:
    """Return the first ``(pattern -> matched-value)`` description, or ``None``.

    Patterns are checked against the package, the activity, and each element's ``type``
    (its short class name) — so ``*.WebView`` catches a ``WebView`` element even when the
    host package looks ordinary.
    """
    for pattern in patterns:
        if _pattern_matches(pattern, package):
            return f"package {package!r} matches vision pattern {pattern!r}"
        if _pattern_matches(pattern, activity):
            return f"activity {activity!r} matches vision pattern {pattern!r}"
        for el in elements:
            if _pattern_matches_class(pattern, el.type):
                return f"element class {el.type!r} matches vision pattern {pattern!r}"
    return None


def decide(
    elements: list[Element],
    *,
    package: str | None,
    activity: str | None,
    cfg: GateCfg,
) -> GateDecision:
    """Decide whether to fall back to vision for this screen (see module docstring)."""
    # 1. too few elements
    if len(elements) < cfg.min_elements:
        return GateDecision(
            True,
            f"only {len(elements)} element(s) < min_elements={cfg.min_elements}",
        )

    # 2. no semantics anywhere
    if not any(_has_label(el) for el in elements):
        return GateDecision(
            True,
            "no element carries text or content-desc (likely custom-drawn)",
        )

    # 3. known-opaque package/activity/class
    matched = _matched_vision_package(cfg.vision_packages, package, activity, elements)
    if matched is not None:
        return GateDecision(True, matched)

    # 4. poorly-labeled clickable controls (guard divide-by-zero)
    clickables = [el for el in elements if el.clickable]
    if clickables:
        labeled = sum(1 for el in clickables if _has_label(el))
        ratio = labeled / len(clickables)
        if ratio < cfg.min_labeled_ratio:
            return GateDecision(
                True,
                f"labeled-clickable ratio {ratio:.2f} < min_labeled_ratio={cfg.min_labeled_ratio} "
                f"({labeled}/{len(clickables)} clickables labeled)",
            )

    return GateDecision(
        False,
        f"hierarchy sufficient: {len(elements)} elements, semantics present",
    )
