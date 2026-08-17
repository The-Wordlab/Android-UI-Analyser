"""Guarded, optional next-call policy selection.

The policy boundary is deliberately narrower than an AUA planner.  Deterministic
code constructs complete calls and retains them in a trusted map; a policy model
may only select one opaque integer ID.  Model output never authors arguments,
grants authorization, executes a call, or relaxes a safety check.

This module is dependency-free and contains the production guard.  Optional model
providers live under :mod:`android_ui_analyser.providers.policy` and are loaded only
when policy selection is explicitly enabled.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

PolicyMode = Literal["off", "shadow", "advisory"]

ACTIVATION_PREFIX = "You are a model that can do function calling with the following functions."
SELECTOR_POLICY = (
    f"{ACTIVATION_PREFIX} "
    "You are an AUA policy selector for Android UI testing. Select exactly one "
    "supplied candidate. Candidate IDs are opaque and their order is arbitrary. "
    "Prefer direct semantic proof, current observations, bounded waits, and "
    "required cleanup. Never invent or rewrite a call."
)
SELECTION_REQUEST = "Choose the next action that advances the goal with direct proof."

SELECT_CANDIDATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "select_candidate",
        "description": (
            "Select exactly one supplied candidate by its opaque integer ID. "
            "Do not invent, rewrite, or execute an AUA call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "integer",
                    "description": "The candidate_id of the single safest next call.",
                }
            },
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
    },
}

_ESCALATION_ARGUMENTS = frozenset(
    {
        "allow_unsafe",
        "allow_destructive",
        "allow_replay",
        "force",
    }
)
_CURRENT_FRAME_TOOL_PREFIXES = ("tap", "input", "long_press", "swipe", "key", "scroll")
_OBSERVATION_KEYS = frozenset(
    {
        "fresh",
        "known_screen",
        "outcome",
        "screen_changed",
        "goal_checkpoint_reached",
        "element_count",
        "source",
    }
)


@dataclass(frozen=True)
class PolicyCandidate:
    """One trusted, fully-authored call that a policy may select by opaque ID.

    ``call`` remains on the trusted side of the boundary. ``model_arguments`` is
    an independently-authored, privacy-screened projection used only to preserve
    the adapter's training schema; it defaults to empty and can never rewrite the
    trusted call. Session identifiers, paths, typed input, and raw hierarchy data
    are never placed in the prompt. Provenance fields support both pre-inference
    filtering and pre-emission TOCTOU revalidation by the caller.
    """

    candidate_id: int
    call: Mapping[str, Any]
    purpose: str
    proof: str
    model_arguments: Mapping[str, Any] | None = None
    cleanup: str = "none"
    risk: str = "safe"
    safe: bool = True
    authorized: bool = True
    redundant: bool = False
    current: bool = True
    session_id: str | None = None
    phase: str | None = None
    observation_fingerprint: str | None = None
    package: str | None = None

    @property
    def tool(self) -> str | None:
        value = self.call.get("tool") if isinstance(self.call, Mapping) else None
        return value if isinstance(value, str) and value.strip() else None

    @property
    def arguments(self) -> Mapping[str, Any] | None:
        value = self.call.get("arguments") if isinstance(self.call, Mapping) else None
        return value if isinstance(value, Mapping) else None

    def is_current_for(self, context: PolicyContext) -> bool:
        """Return whether this candidate was compiled from *context*'s live state."""

        if not self.current:
            return False
        if self.phase != context.phase:
            return False
        for expected, actual in (
            (context.session_id, self.session_id),
            (context.observation_fingerprint, self.observation_fingerprint),
            (context.package, self.package),
        ):
            if expected is not None and actual != expected:
                return False
        tool = self.tool or ""
        if tool.startswith(_CURRENT_FRAME_TOOL_PREFIXES):
            return bool(
                context.observation_fingerprint
                and context.package
                and self.observation_fingerprint == context.observation_fingerprint
                and self.package == context.package
            )
        return True

    def as_model_value(self) -> dict[str, Any]:
        """Return training-shaped metadata with only the explicit safe projection."""

        model_arguments = self.model_arguments or {}
        return {
            "id": self.candidate_id,
            "call": {
                "tool": self.tool,
                "arguments": copy.deepcopy(dict(model_arguments)),
            },
            "purpose": self.purpose,
            "risk": self.risk,
            "authorized": self.authorized,
            "redundant": self.redundant,
            "proof": self.proof,
            "cleanup": self.cleanup,
        }

    def trusted_call(self) -> dict[str, Any]:
        """Copy the exact call for advisory output after all guards pass."""

        return copy.deepcopy(dict(self.call))


@dataclass(frozen=True)
class PolicyContext:
    """Current goal state and the bounded calls available at this decision point."""

    goal: str
    phase: str
    candidates: tuple[PolicyCandidate, ...]
    observation: Mapping[str, Any] = field(default_factory=dict)
    recent_outcomes: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    session_id: str | None = None
    observation_fingerprint: str | None = None
    package: str | None = None


class PolicySelector(Protocol):
    """Minimal interface implemented by optional local policy providers."""

    name: str

    def is_available(self) -> Any:
        """Return an object with boolean ``ok`` and string ``reason`` fields."""

    def select(self, context: PolicyContext) -> int | None:
        """Return one offered candidate ID, or ``None`` on any failure."""

    def supports_candidate_count(self, count: int) -> bool:
        """Return whether this provider was trained for *count* candidates."""

    def supports_mode(self, mode: str) -> bool:
        """Return whether provenance authorizes this rollout mode."""


@dataclass(frozen=True)
class PolicyDecision:
    """Fail-closed, auditable result of guarded candidate selection."""

    mode: PolicyMode
    status: str
    eligible_candidates: tuple[PolicyCandidate, ...] = ()
    selected_candidate: PolicyCandidate | None = None
    provider: str | None = None
    model_used: bool = False
    error: str | None = None

    @property
    def selected_candidate_id(self) -> int | None:
        return self.selected_candidate.candidate_id if self.selected_candidate is not None else None

    def as_json(self) -> dict[str, Any]:
        """Serialize safe policy metadata.

        Shadow mode intentionally withholds the exact call: it is evaluation-only
        and must not become an accidental execution recommendation.  Advisory mode
        discloses the already-authored trusted call after the guard accepts it.
        """

        value: dict[str, Any] = {
            "mode": self.mode,
            "status": self.status,
            "provider": self.provider,
            "model_used": self.model_used,
            "candidate_count": len(self.eligible_candidates),
            "eligible_candidate_ids": [
                candidate.candidate_id for candidate in self.eligible_candidates
            ],
        }
        if self.selected_candidate is not None:
            value["selected_candidate_id"] = self.selected_candidate.candidate_id
            if self.mode == "advisory" and self.model_used:
                value["recommended_call"] = self.selected_candidate.trusted_call()
        if self.error:
            value["error"] = self.error
        return value


def _has_valid_call(candidate: PolicyCandidate) -> bool:
    tool = candidate.tool
    arguments = candidate.arguments
    if tool is None or arguments is None:
        return False
    # A selector is never an authorization boundary.  Calls containing an
    # escalation switch are withheld even if their planner metadata says safe.
    for key, value in arguments.items():
        if str(key).strip().lower() in _ESCALATION_ARGUMENTS and bool(value):
            return False
    model_arguments = candidate.model_arguments or {}
    return isinstance(model_arguments, Mapping) and set(model_arguments).issubset(arguments)


def guard_candidates(
    context: PolicyContext,
    *,
    max_candidates: int = 4,
) -> tuple[PolicyCandidate, ...]:
    """Return only uniquely-addressed, safe, authorized, fresh exact calls.

    The model never sees rejected candidates.  Duplicate or boolean IDs are
    withheld rather than resolved by order, keeping the ID-to-call map unambiguous.
    """

    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or not 1 <= max_candidates <= 4
    ):
        raise ValueError("max_candidates must be an integer from 1 to 4")
    ids = Counter(
        candidate.candidate_id
        for candidate in context.candidates
        if isinstance(candidate.candidate_id, int) and not isinstance(candidate.candidate_id, bool)
    )
    eligible: list[PolicyCandidate] = []
    for candidate in context.candidates:
        candidate_id = candidate.candidate_id
        if (
            not isinstance(candidate_id, int)
            or isinstance(candidate_id, bool)
            or ids[candidate_id] != 1
            or not candidate.safe
            or str(candidate.risk).strip().lower() != "safe"
            or not candidate.authorized
            or candidate.redundant
            or not candidate.is_current_for(context)
            or not _has_valid_call(candidate)
        ):
            continue
        eligible.append(candidate)
        if len(eligible) == max_candidates:
            break
    return tuple(eligible)


def compile_policy_context(
    context: PolicyContext,
    candidates: Sequence[PolicyCandidate] | None = None,
) -> dict[str, Any]:
    """Compile the model-facing, training-shaped JSON value.

    Only explicit ``model_arguments`` cross this boundary; trusted call values,
    session/device identity, and hierarchy data do not. The caller should pass
    the result of :func:`guard_candidates`; :func:`evaluate_policy` does this
    automatically.
    """

    visible = tuple(candidates) if candidates is not None else context.candidates
    observation = {
        str(key): value
        for key, value in context.observation.items()
        if str(key) in _OBSERVATION_KEYS
        and (value is None or isinstance(value, (str, int, float, bool)))
    }
    return {
        "fixture_ref": "aua-live-policy-v1",
        "request": SELECTION_REQUEST,
        "goal": context.goal,
        "phase": context.phase,
        "observation": observation,
        "recent_outcomes": list(context.recent_outcomes),
        "constraints": [
            *context.constraints,
            "Select exactly one supplied candidate",
            "Require direct proof",
        ],
        "candidates": [candidate.as_model_value() for candidate in visible],
    }


def policy_messages(
    context: PolicyContext,
    candidates: Sequence[PolicyCandidate] | None = None,
) -> list[dict[str, Any]]:
    """Build the developer/user turns used by FunctionGemma providers."""

    import json

    return [
        {"role": "developer", "content": SELECTOR_POLICY},
        {
            "role": "user",
            "content": json.dumps(
                compile_policy_context(context, candidates),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    ]


def policy_tools() -> list[dict[str, Any]]:
    """Return an isolated copy of the policy's sole function definition."""

    return [copy.deepcopy(SELECT_CANDIDATE_TOOL)]


def evaluate_policy(
    context: PolicyContext,
    selector: PolicySelector | None,
    *,
    mode: PolicyMode,
    max_candidates: int = 4,
) -> PolicyDecision:
    """Guard candidates, optionally invoke a selector, and validate its opaque ID.

    Zero eligible candidates fail closed.  One eligible candidate is selected
    deterministically without checking dependencies or loading a model.  Any
    availability, load, generation, parse, or off-list failure produces metadata
    only and no selected call.
    """

    if mode not in {"off", "shadow", "advisory"}:
        raise ValueError(f"unsupported policy mode: {mode!r}")
    if mode == "off":
        return PolicyDecision(mode=mode, status="off")

    eligible = guard_candidates(context, max_candidates=max_candidates)
    if not eligible:
        return PolicyDecision(mode=mode, status="no_candidate")
    if len(eligible) == 1:
        return PolicyDecision(
            mode=mode,
            status="deterministic",
            eligible_candidates=eligible,
            selected_candidate=eligible[0],
            provider="deterministic",
            model_used=False,
        )
    if selector is None:
        return PolicyDecision(
            mode=mode,
            status="no_provider",
            eligible_candidates=eligible,
            error="no policy provider is configured",
        )

    provider_name = str(getattr(selector, "name", type(selector).__name__))
    supports_mode = getattr(selector, "supports_mode", None)
    if callable(supports_mode):
        try:
            mode_supported = bool(supports_mode(mode))
        except Exception:
            mode_supported = False
        if not mode_supported:
            error = f"policy provider is not authorized for {mode} mode"
            capability_method = getattr(selector, "rollout_capability", None)
            if callable(capability_method):
                try:
                    capability = capability_method()
                    if isinstance(capability, Mapping) and capability.get("reason"):
                        error = str(capability["reason"])
                except Exception:
                    pass
            return PolicyDecision(
                mode=mode,
                status="unsupported_mode",
                eligible_candidates=eligible,
                provider=provider_name,
                model_used=False,
                error=error,
            )
    supports_count = getattr(selector, "supports_candidate_count", None)
    if callable(supports_count):
        try:
            cardinality_supported = bool(supports_count(len(eligible)))
        except Exception:
            cardinality_supported = False
        if not cardinality_supported:
            return PolicyDecision(
                mode=mode,
                status="unsupported_cardinality",
                eligible_candidates=eligible,
                provider=provider_name,
                model_used=False,
                error=f"policy provider does not support {len(eligible)} candidates",
            )
    try:
        availability = selector.is_available()
        available = bool(getattr(availability, "ok", False))
        reason = str(getattr(availability, "reason", "unavailable"))
    except Exception:
        return PolicyDecision(
            mode=mode,
            status="unavailable",
            eligible_candidates=eligible,
            provider=provider_name,
            error="policy provider availability check failed",
        )
    if not available:
        return PolicyDecision(
            mode=mode,
            status="unavailable",
            eligible_candidates=eligible,
            provider=provider_name,
            error=reason,
        )

    guarded_context = replace(context, candidates=eligible)
    try:
        selected_id = selector.select(guarded_context)
    except Exception:
        return PolicyDecision(
            mode=mode,
            status="provider_error",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error="policy provider failed closed",
        )
    if not isinstance(selected_id, int) or isinstance(selected_id, bool):
        return PolicyDecision(
            mode=mode,
            status="invalid_selection",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error="provider did not return one integer candidate ID",
        )
    selected = next(
        (candidate for candidate in eligible if candidate.candidate_id == selected_id), None
    )
    if selected is None:
        return PolicyDecision(
            mode=mode,
            status="invalid_selection",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error="provider selected an ID outside the guarded candidate set",
        )
    return PolicyDecision(
        mode=mode,
        status="selected",
        eligible_candidates=eligible,
        selected_candidate=selected,
        provider=provider_name,
        model_used=True,
    )


def policy_status(config: Any, *, factory: Any | None = None) -> dict[str, Any]:
    """Return host-only policy readiness without touching a device or loading a model.

    Provider modules may inspect platform/dependency presence and local artifact
    provenance, but their optional inference dependencies remain lazily imported.
    This small surface is shared by CLI and MCP status commands.
    """

    from .providers.registry import ProviderFactory

    section = getattr(config, "policy", None)
    enabled = bool(getattr(section, "enabled", False))
    mode = str(getattr(section, "mode", "off"))
    chain = list(getattr(section, "chain", []) or [])
    max_candidates = int(getattr(section, "max_candidates", 4))
    provider_factory = factory or ProviderFactory(config)
    providers: list[dict[str, Any]] = []
    for name in chain:
        try:
            provider = provider_factory.create("policy", str(name))
            status_method = getattr(provider, "status", None)
            if callable(status_method):
                provider_value = dict(status_method())
            else:
                availability = provider.is_available()
                provider_value = {
                    "provider": provider.name,
                    "available": bool(availability.ok),
                    "reason": str(availability.reason),
                    "loaded": False,
                }
            supports_mode = getattr(provider, "supports_mode", None)
            provider_value["configured_mode_supported"] = (
                bool(supports_mode(mode)) if callable(supports_mode) else True
            )
        except Exception as exc:
            provider_value = {
                "provider": str(name),
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "loaded": False,
            }
        providers.append(provider_value)
    available = any(
        bool(provider.get("available")) and bool(provider.get("configured_mode_supported", True))
        for provider in providers
    )
    return {
        "enabled": enabled,
        "mode": mode,
        "chain": chain,
        "max_candidates": max_candidates,
        "ready": enabled and mode in {"shadow", "advisory"} and available,
        "providers": providers,
    }
