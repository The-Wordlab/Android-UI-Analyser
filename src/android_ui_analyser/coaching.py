"""Owner-scoped, response-local efficiency coaching for CLI and MCP adapters."""

from __future__ import annotations

import contextlib
from typing import Any

from .selectors import is_back_resource_id

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
_ENGINE_PROGRESS_COMMANDS = frozenset({"session_start", "session_progress", "session_finish"})


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _is_back_event(event: dict[str, Any]) -> bool:
    args = _mapping(event.get("args"))
    selector = _mapping(args.get("selector"))
    command = _base_command(event.get("cmd"))
    semantic_selector = (
        is_back_resource_id(str(selector.get("rid") or ""))
        or str(selector.get("desc") or "").casefold() in {"back", "navigate up", "up"}
        or str(selector.get("text") or "").casefold() == "back"
    )
    return (command == "key" and str(args.get("name")).casefold() == "back") or (
        command == "tap" and semantic_selector
    )


def _reuses_numeric_id_across_frames(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    current_args = _mapping(current.get("args"))
    previous_args = _mapping(previous.get("args"))
    element_id = current_args.get("element_id")
    if not isinstance(element_id, int) or element_id != previous_args.get("element_id"):
        return False

    def screen(event: dict[str, Any]) -> str | None:
        result = _mapping(event.get("result"))
        observation = _mapping(result.get("observation"))
        meta = observation.get("meta")
        if isinstance(meta, str):
            return meta
        if isinstance(meta, dict) and meta.get("known_screen"):
            return str(meta["known_screen"])
        return None

    return bool(screen(current) and screen(previous) and screen(current) != screen(previous))


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


def decorate_result(
    engine: Any,
    cmd: str,
    result: Any,
    *,
    args: dict[str, Any] | None = None,
    current_recorded: bool = True,
) -> Any:
    """Attach one actionable hint when the current call reveals an avoidable pattern."""
    # Host-only flow metadata calls already know their explicit/leased serial. Do not connect
    # uiautomator2 merely to attach coaching or goal progress to an idempotent delete.
    serial = getattr(engine, "_lease_serial", None) or getattr(
        getattr(engine.config, "device", None), "serial", None
    )
    if not serial:
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
    # A global action `--until` journals its internal predicate adoption. It is part of the
    # same agent call, so it must not break a manual-path/back-navigation streak.
    events = [
        event
        for event in events
        if not (
            _base_command(event.get("cmd")) == "await_predicate"
            and isinstance(event.get("args"), dict)
            and event["args"].get("adopt_action") is True
        )
    ]
    normalized = _base_command(cmd)
    # Goal progress is durable across short-lived CLI processes. Environment checkpoints advance
    # only from structured, verified tool events; UI checkpoints advance only when the agent's
    # --phase-done / phase_done evidence passes the active phase's relevance contract.
    try:
        data = _payload(result) or {}
        existing_progress = data.get("goal_progress")
        if normalized in _ENGINE_PROGRESS_COMMANDS and isinstance(existing_progress, dict):
            # These Engine methods already performed the complete session/policy refresh. Reusing
            # their answer avoids a second local-model generation in MCP, in-process CLI, and the
            # warm daemon response decorator while preserving the established nested envelope.
            progress = dict(existing_progress)
            for key in ("policy", "policy_suggestion"):
                if key in data:
                    progress[key] = data[key]
        else:
            from .session import complete_environment_phase, phase_progress

            state = engine._session_state()  # noqa: SLF001 - shared Engine/session contract
            state = complete_environment_phase(
                engine.config.cache.dir,
                state,
                command=normalized,
                result=data,
            )
            observation_payload: Any = data.get("observation")
            if observation_payload is None and "screen" in data and "elements" in data:
                observation_payload = data
            observation = None
            if isinstance(observation_payload, dict):
                from .schema import AnalyzeResult

                with contextlib.suppress(Exception):
                    observation = AnalyzeResult.model_validate(observation_payload)
            stale_deeplink = normalized in {
                "open_link",
                "open_link_and_analyze",
                "open_and_analyze",
            } and bool(data.get("stale_risk"))
            # A background wait's observation is not phase proof until JobState has reached a
            # successful terminal state and its session/owner/serial/predicate correlation has
            # been checked. jobs.py performs that check and refreshes this snapshot afterward.
            progress_observation = None if normalized.startswith("job:") else observation
            refreshed = engine.session_progress(
                state.session_id,
                observation=progress_observation,
                _avoid_deeplinks=stale_deeplink,
            )
            refreshed_progress = refreshed.get("goal_progress")
            if isinstance(refreshed_progress, dict):
                refreshed_state = engine._session_state(state.session_id)  # noqa: SLF001
                progress = phase_progress(refreshed_state, compact=True)
                # Policy is an optional response-local side channel. Preserve it inside the
                # existing goal_progress envelope on ordinary action results; otherwise only
                # explicit session_progress calls would expose the advisory and the hot path
                # would silently discard the model's work.
                for key in ("policy", "policy_suggestion"):
                    if key in refreshed:
                        progress[key] = refreshed[key]
            else:
                progress = phase_progress(state, compact=True)
        if isinstance(result, dict):
            result["goal_progress"] = progress
        elif hasattr(result, "goal_progress"):
            result.goal_progress = progress
        elif hasattr(result, "meta") and hasattr(result.meta, "goal_progress"):
            result.meta.goal_progress = progress
    except Exception:  # pragma: no cover - session coaching must never break a tool call
        pass
    if not current_recorded:
        events.append(
            {
                "cmd": cmd,
                "args": args or {},
                "result": _payload(result) or {},
                "owner": owner,
            }
        )
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
        observation_data = data.get("observation") or {}
        meta_value = observation_data.get("meta") if isinstance(observation_data, dict) else None
        meta: dict[str, Any] = meta_value if isinstance(meta_value, dict) else {}
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
        if (
            normalized == "tap"
            and events
            and prior
            and _reuses_numeric_id_across_frames(events[-1], prior[-1])
        ):
            return _append(
                result,
                {
                    "id": "do_not_reuse_frame_id",
                    "message": "the same numeric id was reused after the screen changed",
                    "recommended_call": (
                        "Use the fresh observation's --rid/stable_key; for nested Back use "
                        "aua back-until-and-analyze '<known_screen>'."
                    ),
                },
            )
        if events and _is_back_event(events[-1]) and prior and _is_back_event(prior[-1]):
            return _append(
                result,
                {
                    "id": "bounded_back_navigation",
                    "message": "repeated Back navigation can stop on semantic destination evidence",
                    "recommended_call": "aua back-until-and-analyze '<known_screen>'",
                },
            )
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
