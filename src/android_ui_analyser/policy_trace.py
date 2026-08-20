"""Opt-in, local-only trace of what the policy was asked and what happened next.

A decision record alone is not training data. The public autopilot output reports *that* a
candidate was chosen — its opaque id, the deciding provider, the compiler's counts — but not the
candidate semantics the model actually read, so the decision cannot be reconstructed, and a
reconstruction is exactly what a training row is. This module closes that gap for local
experiments without widening what AUA ever emits by default.

Three properties make it safe to leave in the tree:

* **Off unless asked.** Tracing happens only when ``AUA_POLICY_TRACE_DIR`` names a writable
  directory. There is no config key, so it cannot be switched on by configuration drift, and
  nothing reaches the journal, the dashboard, telemetry, or the wheel.
* **Screened by construction.** The recorded prompt is the output of
  :func:`~android_ui_analyser.policy.compile_policy_context` — the same privacy-screened
  projection the model receives. Trusted call arguments, session paths, device identity, typed
  input, and hierarchy data are not in that projection and are not added here.
* **Fail closed and quiet.** A decision that cannot be correlated is not written, and any I/O
  problem disables tracing for the process rather than interrupting a device run.

A record is only useful once its outcome is known, so decisions and outcomes are written as
separate lines joined by ``decision_id``. What the local model chose is deliberately *not* a
label: it is the state that a later expert pass relabels.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .policy import PolicyCandidate, PolicyContext, PolicyDecision

ENV_VAR = "AUA_POLICY_TRACE_DIR"
SCHEMA = "aua-policy-training-trace-v1"

_lock = threading.Lock()
_disabled_reason: str | None = None
# The autopilot loop decides and then acts, in order, on one thread. Remembering the most
# recent decision is therefore an exact correlation, and avoids rebuilding a context that
# the caller no longer holds.
_last_decision_id: str | None = None


def trace_directory() -> Path | None:
    """Return the configured trace directory, or ``None`` when tracing is off."""

    if _disabled_reason is not None:
        return None
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def enabled() -> bool:
    return trace_directory() is not None


def _disable(reason: str) -> None:
    global _disabled_reason
    _disabled_reason = reason


def _now() -> str:
    """UTC stamp for one record.

    Without it a trace cannot be ordered, split by day, or lined up with the run that produced it.
    That gap cost a diagnosis: asked which run had failed, the records could say what was decided
    but not when, so the answer had to come from file mtimes and installed-package dates instead.
    """

    return datetime.now(UTC).isoformat(timespec="seconds")


def _append(record: dict[str, Any]) -> None:
    directory = trace_directory()
    if directory is None:
        return
    record = {"recorded_at": _now(), **record}
    try:
        with _lock:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "decisions.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
    except OSError as exc:
        # A device run must never fail because a local experiment could not write a file.
        _disable(f"{type(exc).__name__}: {exc}")


def decision_id(context: PolicyContext, fingerprint: str | None) -> str:
    """Stable identity for one decision point: session, phase, frame, and candidate set."""

    material = json.dumps(
        {
            "session": context.session_id,
            "phase": context.phase,
            "fingerprint": fingerprint or context.observation_fingerprint,
            "goal": context.goal,
            "candidates": sorted(
                json.dumps(candidate.trusted_call(), sort_keys=True)
                for candidate in context.candidates
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def record_decision(
    context: PolicyContext,
    eligible: tuple[PolicyCandidate, ...],
    decision: PolicyDecision,
) -> str | None:
    """Write the exact model-facing prompt plus every provider's verdict. Returns its id."""

    if not enabled():
        return None
    from .policy import compile_policy_context

    try:
        prompt = compile_policy_context(context, eligible)
        identity = decision_id(context, context.observation_fingerprint)
        selected = decision.selected_candidate
        _append(
            {
                "schema": SCHEMA,
                "kind": "decision",
                "decision_id": identity,
                "session_id": context.session_id,
                "phase": context.phase,
                "package": context.package,
                "observation_fingerprint": context.observation_fingerprint,
                "allow_handoff": context.allow_handoff,
                # The privacy-screened projection the model read, verbatim.
                "prompt": prompt,
                "status": decision.status,
                "provider": decision.provider,
                "model_used": decision.model_used,
                "selection_strategy": decision.selection_strategy,
                "selected_candidate_id": (selected.candidate_id if selected is not None else None),
                "selected_tool": selected.tool if selected is not None else None,
                "selection_trace": [dict(entry) for entry in decision.selection_trace],
                # Explicitly not a label. A later expert pass supplies that.
                "label": None,
            }
        )
        global _last_decision_id
        _last_decision_id = identity
        return identity
    except Exception as exc:  # pragma: no cover - never break a run for a trace
        _disable(f"{type(exc).__name__}: {exc}")
        return None


def record_outcome(
    identity: str | None,
    *,
    executed: bool,
    verdict: str,
    action_ok: bool | None = None,
    before_fingerprint: str | None = None,
    after_fingerprint: str | None = None,
    phase_progressed: bool | None = None,
    terminal_reason: str | None = None,
) -> None:
    """Write what actually happened after a decision.

    ``verdict`` is the correlation the handover requires: followed, rejected, stale, failed,
    proved, cleanup_complete, or handoff. A decision with no correlatable outcome is dropped
    rather than guessed at.
    """

    if identity is None or not enabled():
        return
    _append(
        {
            "schema": SCHEMA,
            "kind": "outcome",
            "decision_id": identity,
            "executed": executed,
            "verdict": verdict,
            "action_ok": action_ok,
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": after_fingerprint,
            "frame_changed": (
                None
                if before_fingerprint is None or after_fingerprint is None
                else before_fingerprint != after_fingerprint
            ),
            "phase_progressed": phase_progressed,
            "terminal_reason": terminal_reason,
        }
    )


def last_decision_id() -> str | None:
    """Identity of the decision most recently recorded on this thread of execution."""

    return _last_decision_id


def status() -> dict[str, Any]:
    """Report tracing state for diagnostics without revealing recorded content."""

    directory = trace_directory()
    return {
        "enabled": directory is not None,
        "env_var": ENV_VAR,
        "directory": str(directory) if directory else None,
        "disabled_reason": _disabled_reason,
        "schema": SCHEMA,
    }
