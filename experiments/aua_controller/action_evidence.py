"""Host-observed action targets for evidence, never action authorization or verdicts."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.aua_controller.session_state import observation_frame

ID_ACTIONS = frozenset({
    "tap", "tap_and_analyze", "long_press", "long_press_and_analyze",
    "input", "input_and_analyze",
})


def resolved_action_target(
    tool: str,
    arguments: Any,
    previous: Any,
    *,
    evidence_ref: str | None = None,
) -> dict[str, Any] | None:
    """Bind an opaque handle only to one element in the immediately prior fresh frame.

    This records what the host offered as the target, not whether the dispatch succeeded.
    Never search older frames, derive labels from the model, or use a post-action screen.
    """
    if tool not in ID_ACTIONS or not isinstance(arguments, Mapping):
        return None
    handle = arguments.get("id")
    if not isinstance(handle, (str, int)) or isinstance(handle, bool):
        return None
    frame = observation_frame(previous)
    if frame is None:
        return None
    matches = [element for element in frame["elements"] if element["id"] == handle]
    if len(matches) != 1:
        return None
    element = matches[0]
    target: dict[str, Any] = {
        "source": "previous_fresh_observation",
        "observation_fingerprint": frame["meta"]["fingerprint"],
    }
    if evidence_ref:
        target["source_evidence_ref"] = evidence_ref
    for key in ("text", "content_desc", "desc", "resource_id", "rid", "type"):
        value = element.get(key)
        if key == "text" and element.get("password") is True:
            continue
        if isinstance(value, str) and value.strip():
            target[key] = value[:200]
    if "bounds" in element:
        target["bounds"] = copy.deepcopy(element["bounds"])
    return target


def reported_app_stop(tool: Any, result: Any) -> bool:
    """AUA's own receipt that it force-stopped the app, which leaves no screen of its own."""
    return (tool == "app_force_stop" and isinstance(result, dict) and result.get("ok") is True
            and result.get("action") == "app-stop")


def definitive_selector_miss(result: Any) -> bool:
    """Recognize only AUA's pre-dispatch addressing refusal with a recovery observation."""
    if (not isinstance(result, dict) or result.get("ok") is True or result.get("mcp_is_error")
            or result.get("action") or result.get("capture_evidence") or result.get("action_sent") is True):
        return False
    error = result.get("error")
    if not isinstance(error, dict):
        return False
    observation = error.get("observation")
    refused = ((error.get("code") == "element_not_found"
                and str(error.get("hint") or "").startswith("No action was sent"))
               # A text or resource-id selector that matched nothing is refused while it is
               # resolved, before any press. Counted as a failed action, one miss on a prompt
               # that was not there turned a judged pass into QA_ERROR (2026-09-28).
               or error.get("code") == "selector_not_found")
    return (refused
            and error.get("action_sent") is not True
            and error.get("observation_present") is True and isinstance(observation, dict)
            and isinstance(observation.get("screen"), dict)
            and isinstance(observation.get("elements"), list)
            and isinstance(observation.get("meta"), dict)
            and bool(observation["meta"].get("fingerprint")))


def judge_action_history(
    records: Sequence[dict[str, Any]],
    initial_observation: Any,
    *,
    initial_evidence_ref: str = "E0000",
) -> list[dict[str, Any]]:
    """Project executed actions in recorded order, reconstructing targets from raw evidence.

    Even an existing resolved_target is reconstructed rather than trusted. A receipt without a
    fresh observation invalidates the previous binding; no historical search fills that gap.
    """
    actions = []
    previous = initial_observation
    previous_ref: str | None = initial_evidence_ref
    for record in records:
        if record.get("executed") is True:
            action = {key: copy.deepcopy(record.get(key)) for key in ("step", "tool", "arguments")}
            target = resolved_action_target(
                str(record.get("tool") or ""), record.get("arguments"), previous,
                evidence_ref=previous_ref,
            )
            if target is not None:
                action["resolved_target"] = target
            if definitive_selector_miss(record.get("result")):
                # Dispatched to AUA, which refused it before sending: nothing was pressed.
                action["not_sent"] = True
            if reported_app_stop(record.get("tool"), record.get("result")):
                action["app_stopped"] = True
            actions.append(action)
        previous = record.get("result")
        previous_ref = record.get("evidence_ref")
    return actions
