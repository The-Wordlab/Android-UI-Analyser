"""Owner-scoped, response-local efficiency coaching for CLI and MCP adapters."""

from __future__ import annotations

import contextlib
import uuid
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
_ENGINE_PROGRESS_COMMANDS = frozenset(
    {"session_start", "session_progress", "session_autopilot", "session_finish"}
)


# A relaunch resets app state and cannot change what a screen contains; a third one in one
# window is churn. A repeated call on one target means the first answer was already the true one.
_RELAUNCH_CHURN = 3
_REPEAT_CHURN = 3
_LAUNCH_COMMANDS = frozenset({"app_launch", "app_launch_and_analyze"})
# Fields that name *what* a call acted on, most specific first.
_TARGET_KEYS = ("rid", "resource_id", "stable_key", "label", "text", "desc", "content_desc", "id")


def _target_of(args: Any) -> str | None:
    """What a call acted on, for deciding whether two calls did the same thing."""
    mapping = _mapping(args)
    for key in _TARGET_KEYS:
        value = mapping.get(key)
        if value not in (None, ""):
            return f"{key}={value}"
    return None


def looping_advice(history: Any) -> dict[str, str] | None:
    """One hint when the recent history is churn rather than progress, else ``None``.

    Read from the same durable journal the other rules use, so it survives the short-lived CLI
    processes an agent actually runs. Relaunching outranks repeating: a relaunch destroys app
    state — it cost one live run its login — while a repeated tap only wastes a call.
    """
    rows = [row for row in (history or []) if isinstance(row, dict)]
    if not rows:
        return None
    launches = sum(1 for row in rows if _base_command(row.get("cmd")) in _LAUNCH_COMMANDS)
    if launches >= _RELAUNCH_CHURN:
        return {
            "id": "session_looping",
            "message": (
                f"the app was relaunched {launches} times in the last {len(rows)} calls; a "
                "relaunch resets app state (a live run lost its login this way) and cannot "
                "change what a screen contains"
            ),
            "recommended_call": (
                "Read the current observation and say what is actually missing. If the screen "
                "is empty, the data has to be seeded before it can be verified — "
                "`aua map --find '<goal>'` shows what this app is known to reach."
            ),
        }
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        target = _target_of(row.get("args"))
        if target is None:
            continue
        key = (_base_command(row.get("cmd")), target)
        counts[key] = counts.get(key, 0) + 1
    repeated = max(counts.items(), key=lambda item: item[1], default=None)
    if repeated is not None and repeated[1] >= _REPEAT_CHURN:
        (command, target), count = repeated
        return {
            "id": "session_looping",
            "message": (
                f"{command} on {target} ran {count} times in the last {len(rows)} calls; the "
                "first answer was already the true one"
            ),
            "recommended_call": (
                "Take the fresh observation as the answer and change approach, or run "
                "`aua session review` to see what the run has already spent."
            ),
        }
    return None


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


def _observation_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    nested = data.get("observation")
    if isinstance(nested, dict):
        return nested
    if isinstance(data.get("screen"), dict) and isinstance(data.get("elements"), list):
        return data
    return None


def _attach_observation_contract(result: Any, contract: dict[str, Any]) -> None:
    if isinstance(result, dict):
        if isinstance(result.get("screen"), dict) and isinstance(result.get("meta"), dict):
            result["meta"]["observation_contract"] = contract
        else:
            result["observation_contract"] = contract
        return
    if hasattr(result, "screen") and hasattr(result, "meta"):
        with contextlib.suppress(Exception):
            from .schema import ObservationContract

            result.meta.observation_contract = ObservationContract.model_validate(contract)
        return
    if hasattr(result, "observation_contract"):
        with contextlib.suppress(Exception):
            from .schema import ObservationContract

            result.observation_contract = ObservationContract.model_validate(contract)


def emitted_fingerprint(result: Any) -> str | None:
    """The hierarchy fingerprint in the payload about to be emitted, wherever it sits.

    This is what the caller will actually be holding while it thinks, so it — not whatever this
    process happens to have cached — is the right thing to stamp for the next call to compare
    against. Under the warm daemon the two differ: the answering engine is in another process.
    """
    for probe in (result, getattr(result, "observation", None)):
        if probe is None:
            continue
        if isinstance(probe, dict):
            for key in ("meta", "observation"):
                block = probe.get(key)
                if isinstance(block, dict):
                    meta = block.get("meta") if key == "observation" else block
                    if isinstance(meta, dict) and meta.get("fingerprint"):
                        return str(meta["fingerprint"])
            continue
        fingerprint = getattr(getattr(probe, "meta", None), "fingerprint", None)
        if fingerprint:
            return str(fingerprint)
    return None


def attach_caller_turn(engine: Any, result: Any) -> Any:
    """Put this call's caller-latency facts on *result*, wherever they fit its shape.

    Reported on every decorated response rather than only on waits, because the two things it
    carries are both about the caller's own turn: what its think time cost (which is what the
    wait ceiling is sized from) and whether the screen it has been holding is still there.
    Silent when nothing measured a turn — a daemon round trip is aua's transport, not a caller,
    and reporting one would halve every gap.
    """
    report: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        report = engine.caller_turn_report(emitted_fingerprint(result))
    if not report:
        return result
    data = _payload(result) or {}
    if data.get("wait_ceiling_ms") is not None:
        # A clamped wait already reports these at top level. Repeating the same constants in
        # the caller block spends tokens and lets two independently recomputed values drift.
        report.pop("wait_ceiling_ms", None)
        report.pop("wait_ceiling_mode", None)
    if isinstance(result, dict):
        if isinstance(result.get("screen"), dict) and isinstance(result.get("meta"), dict):
            result["meta"]["caller"] = report
        elif "action" in result or "observation" in result:
            result["caller"] = report
        return result
    with contextlib.suppress(Exception):
        from .schema import CallerTurn

        turn = CallerTurn.model_validate(report)
        if hasattr(result, "meta") and hasattr(result.meta, "caller"):
            result.meta.caller = turn
        elif hasattr(result, "caller"):
            result.caller = turn
    return result


def _record_session_artifact(
    engine: Any,
    cmd: str,
    result: Any,
    *,
    invocation_id: str | None,
    duration_ms: float | None,
    args: dict[str, Any] | None,
) -> None:
    """Attach reuse metadata and append to an active session bundle, best effort."""

    data = _payload(result) or {}
    observation = _observation_payload(data)
    if observation is None and "action" not in data and not data.get("artifacts_dir"):
        return
    state = None
    with contextlib.suppress(Exception):
        state = engine._session_state()  # noqa: SLF001 - shared lifecycle boundary

    contract: dict[str, Any] | None = None
    if observation is not None:
        from .session_artifacts import observation_evidence_id

        raw_meta = observation.get("meta")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        stale = data.get("stale_risk") or meta.get("stale_risk")
        fingerprint = meta.get("fingerprint")
        contract = {
            "fingerprint": str(fingerprint) if fingerprint else None,
            "evidence_id": (
                observation_evidence_id(state.session_id, observation)
                if state is not None
                else None
            ),
            "produced_by": cmd,
            "reusable": stale is None,
            "analyze_needed": stale is not None,
            "reason": str(stale or "fresh settled observation"),
        }
        _attach_observation_contract(result, contract)
    elif "action" in data:
        contract = {
            "fingerprint": None,
            "evidence_id": None,
            "produced_by": cmd,
            "reusable": False,
            "analyze_needed": True,
            "reason": "this result did not contain an observation",
        }
        _attach_observation_contract(result, contract)

    artifact_dir = getattr(state, "artifact_dir", None) if state is not None else None
    if not artifact_dir:
        return
    from pathlib import Path

    from .session_artifacts import SessionArtifactStore

    store = SessionArtifactStore(artifact_dir)

    def screenshot(path: Path) -> str:
        image = engine.platform.capture_screenshot(engine.device)
        image.save(str(path))
        return str(path)

    stored = store.record(
        command=cmd,
        result=result,
        invocation_id=invocation_id or uuid.uuid4().hex,
        duration_ms=duration_ms,
        args=args,
        screenshot=screenshot,
        diagnostics=(
            lambda: (
                engine.platform.diagnostic_logs(engine.device, lines=400)
                if engine.platform.supports("device.logs")
                else None
            )
        ),
    )
    if stored is not None:
        _attach_observation_contract(result, stored)

    if cmd == "session_finish" and data.get("terminated") is True:
        raw_progress = data.get("goal_progress")
        progress: dict[str, Any] = raw_progress if isinstance(raw_progress, dict) else {}
        raw_checkpoints = progress.get("phases")
        checkpoints = (
            [item for item in raw_checkpoints if isinstance(item, dict)]
            if isinstance(raw_checkpoints, list)
            else []
        )
        candidate = data.get("candidate_flow")
        candidate_yaml = candidate.get("yaml") if isinstance(candidate, dict) else None
        finalized = store.finalize(
            data,
            verdict="passed" if data.get("ok") and data.get("finished") else "failed",
            checkpoints=checkpoints,
            candidate_yaml=candidate_yaml if isinstance(candidate_yaml, str) else None,
        )
        if isinstance(result, dict):
            result.update(finalized)


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
    invocation_id: str | None = None,
    duration_ms: float | None = None,
) -> Any:
    """Attach one actionable hint when the current call reveals an avoidable pattern."""
    # Before any of the early returns below: the caller's own latency is not a hint that may or
    # may not apply, it is a measurement of this call.
    attach_caller_turn(engine, result)
    # Host-only flow metadata calls already know their explicit/leased serial. Do not connect
    # uiautomator2 merely to attach coaching or goal progress to an idempotent delete.
    device = getattr(engine, "_device", None)
    serial = (
        getattr(engine, "_lease_serial", None)
        or getattr(getattr(engine.config, "device", None), "serial", None)
        or getattr(device, "serial", None)
    )
    try:
        from . import journal

        # Two windows from one read. The per-command rules below were tuned on the last 12
        # calls and stay on 12; churn needs a longer view — a run can relaunch seven times
        # across twelve minutes and never show three in any twelve-call slice.
        events = journal.read_since(engine.config.cache.dir, serial, limit=36)
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
    history = events
    events = events[-12:]
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
            for key in ("policy", "policy_suggestion", "policy_handoff"):
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
                for key in ("policy", "policy_suggestion", "policy_handoff"):
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
    with contextlib.suppress(Exception):
        _record_session_artifact(
            engine,
            normalized,
            result,
            invocation_id=invocation_id,
            duration_ms=duration_ms,
            args=args,
        )
    if not current_recorded:
        events.append(
            {
                "cmd": cmd,
                "args": args or {},
                "result": _payload(result) or {},
                "owner": owner,
            }
        )
    # Additive, not an early return: "you are looping" and "reuse that observation" are both
    # worth saying, and the churn hint must not suppress the more specific one.
    churn = looping_advice(history)
    if churn is not None:
        result = _append(result, churn)

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
