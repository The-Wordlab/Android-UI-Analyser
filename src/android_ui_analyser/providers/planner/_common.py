"""Shared helpers for planner (objective + elements -> next action) LLM providers.

Two concerns, mirroring ``grounding/_common.py`` so every planner provider stays tiny:

1. **Request building** — a strict system prompt that constrains the model to pick ONE
   next action, choosing ``id`` ONLY from the numbered element list we provide (so it can
   never target an off-screen element), and to answer JSON-only. Plus a compact
   text rendering of the element list (``render_elements``) that keeps the prompt cheap.
2. **Defensive parsing** — pull the first balanced ``{...}``, ``json.loads`` it, and map
   it to a :class:`PlannerDecision` (validating the action vocabulary). Never raises — a
   bad reply yields ``None`` so the chain advances / the caller hands off.

Network errors propagate (the chain runner logs and moves on).
"""

from __future__ import annotations

import json
from typing import Any

from ..base import PLANNER_ACTIONS, PlannerDecision

# Balanced-brace + fence scanners are shared with grounding (defensive JSON extraction).
from ..grounding._common import _first_json_object, _strip_fences

# --------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = (
    "You are a UI navigation planner for an Android app. You are given a GOAL and a "
    "numbered list of the elements currently on screen. Choose the SINGLE next action "
    "that best makes progress toward the goal — usually tapping a button/list item, or "
    "dismissing a blocking dialog/popup (Allow, Not now, Skip, Close, Continue) that is "
    "in the way. Respond with JSON ONLY — no prose, no markdown, no code fences — using "
    "these fields:\n"
    '  {"action": "tap", "id": <id from the list>, "reason": "<short>"}\n'
    '  {"action": "input", "id": <id>, "text": "<value>", "reason": "..."}\n'
    '  {"action": "key", "arg": "back"|"home"|"enter", "reason": "..."}\n'
    '  {"action": "scroll-to", "arg": "<visible text to reveal>", "reason": "..."}\n'
    '  {"action": "swipe", "arg": "up"|"down"|"left"|"right", "reason": "..."}\n'
    '  {"action": "done", "reason": "the goal is satisfied on this screen"}\n'
    '  {"action": "give-up", "reason": "cannot make progress toward the goal"}\n'
    "Rules: the `id` MUST be one of the ids shown; never invent an id. Prefer `done` the "
    "moment the goal is visibly satisfied. Do NOT choose destructive actions (delete, "
    "sign out, pay, purchase) unless the goal explicitly asks for one."
)


def render_elements(elements: list[dict[str, Any]]) -> str:
    """Token-light one-line-per-element rendering: ``[id] "label" (type, flags)``."""
    lines: list[str] = []
    for el in elements:
        label = el.get("label") or el.get("text") or el.get("content_desc") or ""
        flags = []
        if el.get("clickable"):
            flags.append("clickable")
        if el.get("input"):
            flags.append("input")
        tail = f" ({', '.join(flags)})" if flags else ""
        lines.append(f'[{el.get("id")}] "{label}"{tail}')
    return "\n".join(lines)


def build_user_prompt(objective: str, elements: list[dict[str, Any]]) -> str:
    return (
        f"GOAL: {objective}\n\nON-SCREEN ELEMENTS (id, label):\n"
        f"{render_elements(elements)}\n\n"
        "Return the single next action as JSON only."
    )


# --------------------------------------------------------------------------- parsing


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def parse_planner_json(text: str | None) -> PlannerDecision | None:
    """Best-effort parse of an LLM reply into a :class:`PlannerDecision`. Never raises."""
    if not text:
        return None
    try:
        candidate = _first_json_object(_strip_fences(text))
        if candidate is None:
            return None
        data = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    action = str(data.get("action", "")).strip().lower()
    if action not in PLANNER_ACTIONS:
        return None
    arg = data.get("arg")
    text_val = data.get("text")
    reason = data.get("reason")
    return PlannerDecision(
        action=action,
        target_id=_as_int(data.get("id")),
        text=str(text_val) if text_val is not None else None,
        arg=str(arg) if arg is not None else None,
        reason=str(reason) if reason is not None else None,
    )
