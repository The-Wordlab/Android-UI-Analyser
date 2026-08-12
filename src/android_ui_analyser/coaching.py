"""Owner-scoped, response-local efficiency coaching for CLI and MCP adapters."""

from __future__ import annotations

import contextlib
from typing import Any

_MANUAL_ACTIONS = frozenset(
    {
        "tap",
        "long_press",
        "tap_point",
        "input",
        "input_text",
        "clear",
        "swipe",
        "scroll",
        "scroll_to",
        "key",
    }
)


def _base_command(value: Any) -> str:
    command = str(value or "").removesuffix("_and_analyze")
    return "input" if command == "input_text" else command


def _payload(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        with contextlib.suppress(Exception):
            value = result.model_dump(mode="json")
            return value if isinstance(value, dict) else None
    return None


def _append(result: Any, advice: dict[str, str]) -> Any:
    if isinstance(result, dict):
        rows = result.setdefault("advice", [])
        if isinstance(rows, list) and advice not in rows:
            rows.append(advice)
        return result
    if hasattr(result, "advice"):
        rows = list(getattr(result, "advice", None) or [])
        if advice not in rows:
            rows.append(advice)
        with contextlib.suppress(Exception):
            result.advice = rows
        return result
    note = getattr(result, "note", None)
    message = f"AUA efficiency: {advice['message']} Next: {advice['recommended_call']}"
    if not note or message not in str(note):
        with contextlib.suppress(Exception):
            result.note = f"{note} {message}".strip() if note else message
    return result


def decorate_result(engine: Any, cmd: str, result: Any) -> Any:
    """Attach one actionable hint when the current call reveals an avoidable pattern."""
    serial = None
    with contextlib.suppress(Exception):
        serial = engine.device.serial
    try:
        from . import journal

        events = journal.read_since(engine.config.cache.dir, serial, limit=12)
    except Exception:  # pragma: no cover - coaching is always best effort
        events = []
    owner = getattr(engine, "_lease_owner_resolved", None)
    if owner:
        events = [event for event in events if event.get("owner") in (None, owner)]
    normalized = _base_command(cmd)
    prior = events[:-1] if events and _base_command(events[-1].get("cmd")) == normalized else events

    if normalized == "analyze" and prior:
        previous = prior[-1]
        previous_result = previous.get("result")
        intentionally_different = False
        current_args = events[-1].get("args") if events else {}
        if isinstance(current_args, dict):
            intentionally_different = bool(
                current_args.get("source") == "vision"
                or current_args.get("query")
                or current_args.get("with_ocr") is not None
                or current_args.get("fields")
            )
        if (
            isinstance(previous_result, dict)
            and previous_result.get("observation")
            and not intentionally_different
        ):
            return _append(
                result,
                {
                    "id": "reuse_observation",
                    "message": f"{previous.get('cmd')} already returned the current screen",
                    "recommended_call": "Reuse its observation and take the next action.",
                },
            )

    if normalized == "has" and prior and _base_command(prior[-1].get("cmd")) == "has":
        return _append(
            result,
            {
                "id": "combine_assertions",
                "message": "consecutive has calls can be one semantic assertion",
                "recommended_call": "aua await-and-analyze 'rid:<first>,rid:<second>' --observe",
            },
        )

    if normalized in {"open_link", "open_link_and_analyze", "open_and_analyze"}:
        data = _payload(result) or {}
        route_values = data.get("routes")
        routes = route_values if isinstance(route_values, list) else []
        observation = data.get("observation") or {}
        meta = observation.get("meta") if isinstance(observation, dict) else {}
        flow_values = meta.get("flows") if isinstance(meta, dict) else None
        flows = flow_values if isinstance(flow_values, list) else []
        if routes or flows:
            offered = f"aua goto {routes[0]!r}" if routes else f"aua flow run {flows[0]}"
            return _append(
                result,
                {
                    "id": "prefer_verified_navigation",
                    "message": "this screen offered a verified route or saved flow before a deeplink",
                    "recommended_call": offered,
                },
            )

    if normalized in _MANUAL_ACTIONS:
        streak = 1
        for event in reversed(prior):
            if _base_command(event.get("cmd")) not in _MANUAL_ACTIONS:
                break
            streak += 1
        if streak == 4:
            return _append(
                result,
                {
                    "id": "save_repeated_path",
                    "message": "four manual navigation calls form a reusable journey",
                    "recommended_call": "aua flow save <descriptive-name> --last 4",
                },
            )
    return result
