"""A single caller view of CLI and MCP results, without another device read.

The transport hands this module the payload it actually received. Normalization does not
recover missing output, replay an action, load an image, or carry an older screen forward.
An unsuccessful action may still return useful evidence; those are separate facts here.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .observation_contract import build_observation_contract, observation_payload

_UI_KEYS = frozenset({"schema_version", "screen", "elements", "elements_count", "meta"})


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _existing_contract(data: dict[str, Any]) -> dict[str, Any]:
    existing = data.get("observation_contract")
    if isinstance(existing, dict):
        return existing
    observation = observation_payload(data)
    meta = observation.get("meta") if observation else None
    existing = meta.get("observation_contract") if isinstance(meta, dict) else None
    return existing if isinstance(existing, dict) else {}


def _without_observation(data: dict[str, Any]) -> dict[str, Any]:
    """Move only a recognized UI shape; unrelated result metadata stays in place."""
    out = dict(data)
    observation = observation_payload(data)
    if observation is data:
        for key in _UI_KEYS:
            out.pop(key, None)
    elif observation is not None:
        out.pop("observation", None)
    out.pop("observation_contract", None)
    return out


def normalize_result(
    payload: Any,
    *,
    command: str,
    context: dict[str, Any] | None = None,
    exit_code: int | None = None,
    transport_error: bool = False,
) -> dict[str, Any]:
    """Normalize one already-decoded AUA payload into the versioned agent envelope.

    Arrays are valid non-UI responses (for example ``list_devices``). Scalars, empty
    objects, and malformed success/error indicators are explicit protocol failures.
    The caller-supplied context is descriptive; this function never resolves ownership.
    """
    copied = deepcopy(payload)
    data = copied if isinstance(copied, dict) else {}
    error: dict[str, Any] | None = None
    source = data
    result: Any = copied
    invalid = not isinstance(copied, (dict, list)) or copied == {}
    if "ok" in data and not isinstance(data["ok"], bool):
        invalid = True
    raw_error = data.get("error")
    if raw_error is not None:
        if not isinstance(raw_error, dict) or not raw_error:
            invalid = True
        else:
            error = dict(raw_error)
            attached = error.pop("result", None)
            if isinstance(attached, dict):
                # Error result fields are the command result. Preserve outer fields too;
                # an attached recovery result takes precedence over earlier stdout data.
                source = {
                    **{key: value for key, value in data.items() if key != "error"},
                    **attached,
                }
                result = source
            elif attached is not None:
                result = attached
            if isinstance(error.get("observation"), dict) and (
                not isinstance(attached, dict) or observation_payload(attached) is None
            ):
                source = {**source, "observation": error["observation"]}
            if observation_payload(source) is not None:
                error.pop("observation", None)
    if invalid and error is None:
        error = _error("invalid_response", "AUA did not return a valid structured result.")
    if error is None and transport_error:
        error = _error("transport_error", "The transport reported an unsuccessful tool call.")
    if error is None and exit_code not in (None, 0):
        error = _error(
            "cli_exit_error",
            "AUA exited unsuccessfully without a structured error.",
            exit_code=exit_code,
        )

    observation = observation_payload(source)
    existing = _existing_contract(source)
    contract_source = source
    if observation is not None and not isinstance(observation.get("elements"), list):
        # Journal summaries can report a count without exposing any selectors. They remain
        # useful result data, but an agent cannot address elements it was never given.
        visible = {**observation, "elements": []}
        contract_source = visible if observation is source else {**source, "observation": visible}
    try:
        contract = build_observation_contract(
            contract_source,
            command=command,
            evidence_id=existing.get("evidence_id"),
            fingerprint=existing.get("fingerprint"),
        )
    except (TypeError, ValueError):
        invalid = True
        if error is None:
            error = _error("invalid_response", "AUA returned malformed observation metadata.")
        contract = build_observation_contract({}, command=command)
    # Projection can omit the underlying caveat. A producer's explicit refusal is still
    # authoritative; never turn it back into permission to reuse the returned selectors.
    if existing.get("reusable") is False:
        if contract["reusable"]:
            contract["reason"] = (
                existing.get("reason") or "The producer did not permit selector reuse."
            )
            contract["readiness"] = existing.get("readiness") or "unconfirmed"
        contract["reusable"] = False
    if existing.get("evidence_fresh") is False:
        contract.update(evidence_fresh=False, reusable=False, analyze_needed=True)
    if existing.get("analyze_needed") is True:
        contract["analyze_needed"] = True

    if observation is not None:
        if observation is source:
            observation = {key: value for key, value in observation.items() if key in _UI_KEYS}
        else:
            observation = dict(observation)
        meta = observation.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("observation_contract"), dict):
            observation["meta"] = {
                key: value for key, value in meta.items() if key != "observation_contract"
            }
    if isinstance(result, dict):
        result = _without_observation(result)
        result.pop("error", None)

    ok = not (
        invalid
        or error is not None
        or transport_error
        or exit_code not in (None, 0)
        or data.get("ok") is False
        or source.get("ok") is False
    )
    if not ok and contract.get("action_succeeded") is True:
        # A recovery/partial result may report a successful step inside a failed command.
        # Keep that original fact in result, without claiming overall action success here.
        contract.pop("action_succeeded")
    return {
        "schema_version": 1,
        "ok": ok,
        "error": error,
        "observation": observation,
        "observation_contract": contract,
        "result": result,
        "context": deepcopy(context) if context is not None else {},
    }


def _decode(text: str) -> tuple[Any, bool]:
    try:
        return json.loads(text), True
    except (ValueError, TypeError):
        return None, False


def _stderr_error(stderr: str) -> dict[str, Any] | None:
    # Progress/log lines may precede the structured error, whose emitter writes one JSON
    # line. Inspect complete lines only; never search arbitrary prose for a JSON substring.
    candidates = [stderr, *reversed(stderr.splitlines())]
    for text in candidates:
        value, decoded = _decode(text)
        if decoded and isinstance(value, dict) and value.get("error") is not None:
            return value
    return None


def from_cli(
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    command: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read JSON stdout and structured stderr without masking a command failure.

    Ordinary stderr logs do not invalidate valid stdout. A structured stderr error wins
    even if stdout also contains a result or the subprocess incorrectly exits zero.
    Empty/non-JSON stdout is never fabricated into a successful UI observation.
    """
    payload, decoded = _decode(stdout)
    stderr_payload = _stderr_error(stderr)
    if stderr_payload is not None:
        if decoded and isinstance(payload, dict):
            payload = {**payload, **stderr_payload}
        else:
            payload = stderr_payload
    elif not decoded:
        payload = {
            "error": _error(
                "invalid_response",
                "AUA did not return JSON output.",
                exit_code=exit_code,
            )
        }
        if stderr.strip():
            # Click can reject syntax before AUA's structured error emitter runs. Return
            # its bounded diagnostic to this caller; do not log it or attempt recovery.
            payload["error"]["diagnostic"] = stderr[:2000]
    return normalize_result(payload, command=command, context=context, exit_code=exit_code)
