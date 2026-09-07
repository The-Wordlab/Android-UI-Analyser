"""One interpretation of caller-visible evidence, readiness, and observation reuse.

Fresh pixels, visible selectors, and a satisfied destination predicate answer different
questions. Keep those facts separate and use the same evaluator after projection and when
reviewing a previous answer. This module only reads payloads; it never observes a device.
"""

from __future__ import annotations

from typing import Any

_ARTIFACT_READS = frozenset(
    {"capture_status", "capture_last", "capture_sheet", "capture_export", "capture_explain"}
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def observation_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Find both full observations and the compact journal's observation summaries."""
    nested = data.get("observation")
    if isinstance(nested, dict):
        return nested
    if isinstance(data.get("screen"), dict) and (
        isinstance(data.get("elements"), list) or "elements_count" in data
    ):
        return data
    return None


def _existing_contract(data: dict[str, Any]) -> dict[str, Any]:
    top = data.get("observation_contract")
    if isinstance(top, dict):
        return top
    observation = observation_payload(data)
    return _mapping(_mapping((observation or {}).get("meta")).get("observation_contract"))


def preserves_previous_observation(command: str) -> bool:
    """A local capture read/export does not change the previously observed UI."""
    return command.replace("-", "_").removesuffix("_and_analyze") in _ARTIFACT_READS


def build_observation_contract(
    data: dict[str, Any],
    *,
    command: str,
    evidence_id: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    """Describe the actual returned view without inferring success from freshness.

    ``reusable`` concerns using this view's elements for the next semantic action. An opaque
    frame can remain fresh visual evidence without exposing any such elements. ``readiness``
    only says ready when there is explicit arrival/predicate evidence; an ordinary analyze
    does not certify an arbitrary destination. Existing contract claims are deliberately not
    inputs, because a folded wait or a projection may have replaced the view they described.
    """
    observation = observation_payload(data)
    contract: dict[str, Any] = {
        "fingerprint": fingerprint,
        "evidence_id": evidence_id,
        "produced_by": command,
        "reusable": False,
        "analyze_needed": True,
        "reason": "this result did not contain an observation",
    }
    if isinstance(data.get("ok"), bool) and "action" in data:
        contract["action_succeeded"] = data["ok"]
    if observation is None:
        if preserves_previous_observation(command):
            contract.update(
                analyze_needed=False,
                previous_observation_validity="unchanged",
                reason=(
                    "this artifact read does not change the previous observation's validity; "
                    "it contains no new UI observation"
                ),
            )
        return contract

    meta = _mapping(observation.get("meta"))
    stale = data.get("stale_risk") or meta.get("stale_risk")
    if meta.get("fingerprint"):
        contract["fingerprint"] = str(meta["fingerprint"])
    elements = observation.get("elements")
    if isinstance(elements, list):
        elements_available = bool(elements)
    else:
        count = observation.get("elements_count")
        elements_available = isinstance(count, int) and count > 0
    image_path = meta.get("raw_image")
    if isinstance(image_path, str) and image_path:
        contract["image_path"] = image_path
    else:
        image_path = None
    arrival = _mapping(data.get("arrival"))
    arrival_state = arrival.get("state") or meta.get("arrival_state")
    outcome = data.get("await_outcome")
    unmet = bool(data.get("settled_unmet")) or outcome in {"timeout", "settled-unmet"}
    unconfirmed = bool(stale) or arrival_state in {
        "loading", "transitioning", "unconfirmed", "no_change"
    }
    empty = not elements_available or bool(data.get("observation_empty"))
    if unmet:
        readiness = "unmet"
        reason = "the requested wait condition was not met; inspect this frame before recovery"
    elif unconfirmed:
        readiness = "unconfirmed"
        reason = str(stale or "the returned frame does not confirm arrival")
    elif empty:
        readiness = "unconfirmed"
        reason = (
            "no elements were returned in this view; inspect the existing image or request "
            "a different view only if semantic controls are needed"
        )
    elif outcome == "satisfied" or arrival_state == "settled":
        readiness = "ready"
        reason = "fresh observation with confirmed arrival evidence"
    else:
        readiness = "not_checked"
        reason = "fresh observation; no destination readiness claim"
    contract.update(
        evidence_fresh=not bool(stale),
        elements_available=elements_available,
        readiness=readiness,
        reusable=not (unmet or unconfirmed or empty),
        # An unmet destination condition does not make a just-read frame disappear. Inspect
        # that evidence first; another analyze is needed only for stale/missing evidence,
        # while a follow-up predicate wait is a separate readiness decision.
        analyze_needed=bool(stale or (empty and not image_path)),
        reason=reason,
    )
    return contract


def result_has_reusable_observation(value: Any) -> bool:
    """Apply the emitted contract and raw caveats to live or journal-shaped results."""
    if not isinstance(value, dict) or observation_payload(value) is None:
        return False
    existing = _existing_contract(value)
    if existing.get("reusable") is False:
        return False
    return bool(build_observation_contract(value, command="review")["reusable"])


def refresh_observation_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Refresh an existing contract after folding a wait or trimming the emitted elements.

    Preserve evidence identity and placement. An explicit no-meta projection remains no-meta;
    this helper does not add a contract where the caller deliberately omitted it.
    """
    existing = _existing_contract(data)
    if not existing:
        return data
    contract = build_observation_contract(
        data,
        command=str(existing.get("produced_by") or data.get("action") or "analyze"),
        evidence_id=existing.get("evidence_id"),
        fingerprint=existing.get("fingerprint"),
    )
    if isinstance(data.get("observation_contract"), dict):
        data["observation_contract"] = contract
    else:
        observation = observation_payload(data)
        if observation is not None and isinstance(observation.get("meta"), dict):
            observation["meta"]["observation_contract"] = contract
    return data
