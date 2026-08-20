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
import hashlib
import json
import pathlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from . import policy_health

PolicyMode = Literal["off", "shadow", "advisory"]
POLICY_HANDOFF_ID = -1

ACTIVATION_PREFIX = "You are a model that can do function calling with the following functions."
SELECTOR_POLICY = (
    f"{ACTIVATION_PREFIX} "
    "You are an AUA policy selector for Android UI testing. Select exactly one "
    "supplied candidate. Candidate IDs are opaque and their order is arbitrary. "
    "Prefer direct semantic proof, current observations, bounded waits, and "
    "required cleanup. Never invent or rewrite a call."
)
SELECTOR_POLICY_WITH_HANDOFF = (
    f"{ACTIVATION_PREFIX} "
    "You are an AUA policy selector for Android UI testing. Select exactly one "
    "supplied candidate, or select candidate ID -1 to hand control back when none "
    "of the supplied actions directly advances the requested goal. Candidate IDs "
    "are opaque and their order is arbitrary. Prefer direct semantic proof, current "
    "observations, bounded waits, and required cleanup. Never invent or rewrite a call."
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


def _selection_tool(*, allow_handoff: bool) -> dict[str, Any]:
    tool = copy.deepcopy(SELECT_CANDIDATE_TOOL)
    if allow_handoff:
        tool["function"]["description"] = (
            "Select one supplied candidate by its opaque integer ID, or select -1 to hand "
            "control back when no supplied action directly advances the goal. Do not invent, "
            "rewrite, or execute an AUA call."
        )
        tool["function"]["parameters"]["properties"]["candidate_id"]["description"] = (
            "A supplied candidate_id, or -1 only when none of the supplied actions is correct."
        )
    return tool


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
    allow_handoff: bool = False


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

    def supports_handoff(self) -> bool:
        """Return whether the authenticated prompt schema learned the handoff sentinel."""


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
    selection_strategy: str | None = None
    selection_trace: tuple[Mapping[str, Any], ...] = ()

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
        if self.status == "handoff":
            value["handoff_reason"] = "no_supplied_candidate_advances_goal"
        if self.error:
            value["error"] = self.error
        if self.selection_strategy:
            value["selection_strategy"] = self.selection_strategy
        if self.selection_trace:
            value["selection_trace"] = [dict(item) for item in self.selection_trace]
        return value


_SEMANTIC_STOPWORDS = frozenset(
    {
        "action",
        "advance",
        "and",
        "android",
        "app",
        "call",
        "choose",
        "control",
        "current",
        "direct",
        "exact",
        "frame",
        "goal",
        "next",
        "observe",
        "open",
        "proof",
        "result",
        "screen",
        "select",
        "setting",
        "tap",
        "the",
        "then",
        "this",
        "visible",
    }
)


def _split_identifier(token: str) -> list[str]:
    """Split a camelCase / snake_case identifier into its words.

    Android controls are frequently identified only by a resource id, so ``buttonSettings``
    is the *only* place the word "settings" appears for that candidate. Tokenising on
    non-alphanumerics alone yields the single opaque term ``buttonsettings``, which can never
    match a goal term — every rid-only candidate then looks equally unrelated to every goal,
    and the overlap comparison silently carries no information at all.
    """

    return [part for part in re.split(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+", token) if part]


def _semantic_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9]+", value):
        # Keep the whole token as well as its parts: a label that is genuinely one word is
        # unchanged, while an identifier contributes both forms.
        for candidate in (raw, *_split_identifier(raw)):
            token = candidate.casefold()
            if len(token) > 1 and token not in _SEMANTIC_STOPWORDS:
                terms.add(token)
    return terms


def _candidate_semantic_terms(candidate: PolicyCandidate) -> set[str]:
    projected = json.dumps(
        candidate.model_arguments or {},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _semantic_terms(f"{candidate.purpose} {projected}")


def selection_requires_review(
    context: PolicyContext,
    selected: PolicyCandidate,
    candidates: Sequence[PolicyCandidate],
) -> tuple[bool, str]:
    """Conservatively detect a semantically *off-goal* model choice.

    This is a routing signal, never an authorization or correctness claim. It exists to
    catch a confident but unrelated choice — one that reaches for a control the goal never
    named, or a weaker match while a strictly stronger one was on offer.

    It deliberately does **not** fire when several supplied candidates tie for the
    strongest overlap. A real screen routinely exposes more than one control that reaches
    the same destination (a bottom tab and a card carrying the same label), and breaking
    that tie is exactly the judgement the model is here to supply. Vetoing an agreed
    choice there stalls navigation without preventing a single wrong action.
    """

    goal_terms = _semantic_terms(context.goal)
    if not goal_terms:
        return True, "goal_has_no_discriminating_terms"
    scores = [len(goal_terms & _candidate_semantic_terms(candidate)) for candidate in candidates]
    selected_index = next(
        (index for index, candidate in enumerate(candidates) if candidate is selected),
        None,
    )
    if selected_index is None:
        return True, "selected_candidate_is_not_guarded"
    selected_score = scores[selected_index]
    best = max(scores, default=0)
    if selected_score == 0:
        return True, "selected_candidate_has_no_goal_overlap"
    if selected_score < best:
        return True, "another_candidate_has_stronger_goal_overlap"
    if scores.count(best) != 1:
        return False, "tied_best_goal_overlap"
    return False, "unique_direct_semantic_match"


# Verdicts that must never execute, even from the last configured reviewer. Everything else
# the chain may decide for itself.
_TERMINAL_REFUSAL_REASONS = frozenset(
    {
        "selected_candidate_has_no_goal_overlap",
        "selected_candidate_is_not_guarded",
    }
)


def _candidate_identity(candidate: PolicyCandidate) -> str:
    material = json.dumps(
        {
            "call": candidate.trusted_call(),
            "purpose": candidate.purpose,
            "proof": candidate.proof,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _counterfactual_context(
    context: PolicyContext,
    candidates: Sequence[PolicyCandidate],
    *,
    provider: str,
    review: int,
) -> tuple[PolicyContext, dict[int, PolicyCandidate]]:
    rows = [(_candidate_identity(candidate), candidate) for candidate in candidates]
    id_rows = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"policy-review-id\0{provider}\0{review}\0{row[0]}".encode()
        ).hexdigest(),
    )
    assigned = {
        identity: candidate_id for candidate_id, (identity, _candidate) in enumerate(id_rows)
    }
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"policy-review-order\0{provider}\0{review}\0{row[0]}".encode()
        ).hexdigest(),
    )
    remapped = tuple(
        replace(candidate, candidate_id=assigned[identity]) for identity, candidate in ordered
    )
    by_id = {assigned[identity]: candidate for identity, candidate in rows}
    return replace(context, candidates=remapped), by_id


def _selector_preflight(
    selector: PolicySelector,
    *,
    mode: PolicyMode,
    candidate_count: int,
    allow_handoff: bool,
) -> str | None:
    supports_mode = getattr(selector, "supports_mode", None)
    if callable(supports_mode):
        try:
            if not bool(supports_mode(mode)):
                return f"provider is not authorized for {mode} mode"
        except Exception:
            return "provider mode capability check failed"
    supports_count = getattr(selector, "supports_candidate_count", None)
    if callable(supports_count):
        try:
            if not bool(supports_count(candidate_count)):
                return f"provider does not support {candidate_count} candidates"
        except Exception:
            return "provider candidate capability check failed"
    supports_handoff = getattr(selector, "supports_handoff", None)
    if allow_handoff and callable(supports_handoff):
        try:
            if not bool(supports_handoff()):
                return "provider does not support policy handoff"
        except Exception:
            return "provider handoff capability check failed"
    try:
        availability = selector.is_available()
        if not bool(getattr(availability, "ok", False)):
            return str(getattr(availability, "reason", "provider unavailable"))
    except Exception:
        return "provider availability check failed"
    return None


def _selector_consensus(
    selector: PolicySelector,
    context: PolicyContext,
    candidates: Sequence[PolicyCandidate],
    *,
    reviews: int,
    record_health: bool,
) -> tuple[PolicyCandidate | int | None, dict[str, Any]]:
    provider = _provider_name(selector)
    votes: list[PolicyCandidate | int] = []
    vote_ids: list[int | None] = []
    invalid_attempts = 0
    attempts = 0
    # A malformed/truncated verdict is not a semantic disagreement. Permit two bounded
    # replacement attempts while still requiring the requested number of valid unanimous
    # counterfactual votes before a call can be exposed.
    for review in range(reviews + 2):
        attempts += 1
        remapped_context, originals = _counterfactual_context(
            context,
            candidates,
            provider=provider,
            review=review,
        )
        try:
            selected_id = selector.select(remapped_context)
        except Exception:
            selected_id = None
        vote_ids.append(selected_id if isinstance(selected_id, int) else None)
        if selected_id == POLICY_HANDOFF_ID and context.allow_handoff:
            votes.append(POLICY_HANDOFF_ID)
        elif isinstance(selected_id, int) and not isinstance(selected_id, bool):
            original = originals.get(selected_id)
            if original is None:
                invalid_attempts += 1
                continue
            votes.append(original)
        else:
            invalid_attempts += 1
            continue
        valid_identities = [
            "handoff" if isinstance(vote, int) else _candidate_identity(vote) for vote in votes
        ]
        if len(set(valid_identities)) > 1 or len(votes) == reviews:
            break
    identities = [
        "handoff" if isinstance(vote, int) else _candidate_identity(vote) for vote in votes
    ]
    unanimous = len(votes) == reviews and len(set(identities)) == 1
    selected: PolicyCandidate | int | None = votes[0] if unanimous else None
    # A malformed verdict is a measurement, not just a retry. Recording it here is what lets a
    # provider whose output is mostly unusable be refused instead of asked again forever.
    if record_health:
        policy_health.record(provider, attempts=attempts, invalid=invalid_attempts)
    health = policy_health.report(provider)
    return selected, {
        "provider": provider,
        "reviews": reviews,
        "attempts": attempts,
        "invalid_attempts": invalid_attempts,
        "recent_invalid_rate": health["invalid_rate"],
        "unanimous": unanimous,
        "outcomes": identities,
        "raw_candidate_ids": vote_ids,
    }


def _evaluate_selective_policy(
    context: PolicyContext,
    selectors: Sequence[PolicySelector],
    *,
    mode: PolicyMode,
    max_candidates: int = 4,
    primary_reviews: int = 2,
    reviewer_reviews: int = 3,
) -> PolicyDecision:
    """Use a fast primary selector and invoke stronger reviewers only when needed.

    The primary must be counterfactually unanimous and the unique strongest semantic
    match to avoid review. Any ambiguity, unavailable provider, handoff, or invalid
    output advances to the next provider. A reviewer must be unanimous; otherwise the
    result is a fail-closed handoff with no suggested call.
    """

    if mode not in {"off", "shadow", "advisory"}:
        raise ValueError(f"unsupported policy mode: {mode!r}")
    if mode == "off":
        return PolicyDecision(mode=mode, status="off")
    if primary_reviews < 1 or reviewer_reviews < 1:
        raise ValueError("policy review counts must be positive")
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
            selection_strategy="selective_hybrid",
        )
    if not selectors:
        return PolicyDecision(
            mode=mode,
            status="no_provider",
            eligible_candidates=eligible,
            error="no policy provider is configured",
            selection_strategy="selective_hybrid",
        )

    guarded_context = replace(context, candidates=eligible)
    trace: list[Mapping[str, Any]] = []
    model_used = False
    for index, selector in enumerate(selectors):
        provider = _provider_name(selector)
        # Shadow mode exists to *measure* a provider, so it always calls it; refusing a
        # condemned provider is about not acting on one, and not paying for it per step.
        unusable = policy_health.unusable_reason(provider) if mode == "advisory" else None
        if unusable:
            # Not a transient failure: the chain has measured this one and stops paying for it.
            trace.append({"provider": provider, "status": "provider_unusable", "reason": unusable})
            continue
        preflight_error = _selector_preflight(
            selector,
            mode=mode,
            candidate_count=len(eligible),
            allow_handoff=context.allow_handoff,
        )
        if preflight_error:
            trace.append({"provider": provider, "status": "unavailable", "reason": preflight_error})
            continue
        reviews = primary_reviews if index == 0 else reviewer_reviews
        selected, attempt = _selector_consensus(
            selector,
            guarded_context,
            eligible,
            reviews=reviews,
            record_health=mode == "advisory",
        )
        model_used = True
        attempt = dict(attempt)
        if selected == POLICY_HANDOFF_ID:
            attempt["status"] = "handoff"
            trace.append(attempt)
            continue
        if not isinstance(selected, PolicyCandidate):
            attempt["status"] = "no_consensus"
            trace.append(attempt)
            continue
        requires_review, reason = selection_requires_review(
            guarded_context,
            selected,
            eligible,
        )
        attempt["semantic_review_required"] = requires_review
        attempt["semantic_reason"] = reason
        if requires_review:
            if index < len(selectors) - 1:
                # An off-goal-looking choice is never executed on the fast primary's word
                # alone. This is a routing signal only: spend the stronger reviewer here.
                attempt["status"] = "review_required"
                trace.append(attempt)
                continue
            if reason in _TERMINAL_REFUSAL_REASONS:
                # The last reviewer is the final *local* authority, but authority over a
                # judgement call is not authority to act on a control that shares nothing
                # with the goal. Observed live: on a goal naming a destination the app does
                # not contain, the terminal reviewer confidently selected an unrelated
                # navigation tab and the turn executed it. Zero overlap is the one verdict
                # that must return control instead of tapping.
                attempt["status"] = "rejected_semantic"
                trace.append(attempt)
                continue
        # Any remaining uncertainty is a judgement the local chain is allowed to make: a tie
        # between equivalent entry points, or a weaker-but-deliberate match.
        attempt["status"] = "selected"
        trace.append(attempt)
        return PolicyDecision(
            mode=mode,
            status="selected",
            eligible_candidates=eligible,
            selected_candidate=selected,
            provider=provider,
            model_used=model_used,
            selection_strategy="selective_hybrid",
            selection_trace=tuple(trace),
        )

    return PolicyDecision(
        mode=mode,
        status="handoff",
        eligible_candidates=eligible,
        provider=_provider_name(selectors[-1]),
        model_used=model_used,
        error="no configured policy provider produced a trusted consensus",
        selection_strategy="selective_hybrid",
        selection_trace=tuple(trace),
    )


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


def _provider_name(selector: Any) -> str:
    """A diagnostic name must never be able to abort a fail-closed provider chain."""
    try:
        value = getattr(selector, "name", None)
        return str(value or type(selector).__name__)
    except Exception:
        return type(selector).__name__


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
            *(
                ["Select candidate ID -1 only when no supplied action directly advances the goal"]
                if context.allow_handoff
                else []
            ),
        ],
        "candidates": [candidate.as_model_value() for candidate in visible],
        **(
            {
                "handoff": {
                    "allowed": True,
                    "candidate_id": POLICY_HANDOFF_ID,
                    "reason": "no_supplied_candidate_advances_goal",
                }
            }
            if context.allow_handoff
            else {}
        ),
    }


def policy_messages(
    context: PolicyContext,
    candidates: Sequence[PolicyCandidate] | None = None,
) -> list[dict[str, Any]]:
    """Build the developer/user turns used by FunctionGemma providers."""

    import json

    return [
        {
            "role": "developer",
            "content": SELECTOR_POLICY_WITH_HANDOFF if context.allow_handoff else SELECTOR_POLICY,
        },
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


def policy_tools(*, allow_handoff: bool = False) -> list[dict[str, Any]]:
    """Return an isolated copy of the policy's sole function definition."""

    return [_selection_tool(allow_handoff=allow_handoff)]


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

    provider_name = _provider_name(selector)
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
    unusable = policy_health.unusable_reason(provider_name) if mode == "advisory" else None
    if unusable:
        return PolicyDecision(
            mode=mode,
            status="provider_unusable",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=False,
            error=unusable,
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
        if mode == "advisory":
            policy_health.record(provider_name, attempts=1, invalid=1)
        return PolicyDecision(
            mode=mode,
            status="provider_error",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error="policy provider failed closed",
        )
    if not isinstance(selected_id, int) or isinstance(selected_id, bool):
        if mode == "advisory":
            policy_health.record(provider_name, attempts=1, invalid=1)
        return PolicyDecision(
            mode=mode,
            status="invalid_selection",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error="provider did not return one integer candidate ID",
        )
    if selected_id == POLICY_HANDOFF_ID:
        if not context.allow_handoff:
            if mode == "advisory":
                policy_health.record(provider_name, attempts=1, invalid=1)
            return PolicyDecision(
                mode=mode,
                status="invalid_selection",
                eligible_candidates=eligible,
                provider=provider_name,
                model_used=True,
                error="provider requested handoff without an authenticated handoff protocol",
            )
        # Declining every offered candidate is a usable answer, not a malformed one.
        if mode == "advisory":
            policy_health.record(provider_name, attempts=1, invalid=0)
        return PolicyDecision(
            mode=mode,
            status="handoff",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
        )
    selected = next(
        (candidate for candidate in eligible if candidate.candidate_id == selected_id), None
    )
    if selected is None:
        if mode == "advisory":
            policy_health.record(provider_name, attempts=1, invalid=1)
        return PolicyDecision(
            mode=mode,
            status="invalid_selection",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error="provider selected an ID outside the guarded candidate set",
        )
    requires_review, semantic_reason = selection_requires_review(
        guarded_context, selected, eligible
    )
    if requires_review:
        if mode == "advisory":
            policy_health.record(provider_name, attempts=1, invalid=0)
        return PolicyDecision(
            mode=mode,
            status="rejected_semantic",
            eligible_candidates=eligible,
            provider=provider_name,
            model_used=True,
            error=semantic_reason,
        )
    if mode == "advisory":
        policy_health.record(provider_name, attempts=1, invalid=0)
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
        # What the provider claims about itself, next to what its recent output actually was.
        # A chain member that mostly emits unusable selections is reported here rather than
        # discovered by watching a run go nowhere.
        health = policy_health.report(str(provider_value.get("provider") or name))
        if health["attempts"]:
            provider_value["selection_health"] = health
            if not health["usable"]:
                provider_value["available"] = False
                provider_value["reason"] = str(health["reason"])
        providers.append(provider_value)
    available = any(
        bool(provider.get("available")) and bool(provider.get("configured_mode_supported", True))
        for provider in providers
    )
    from . import policy_trace

    trace = policy_trace.status()
    if trace.get("enabled"):
        # Recorded decisions are only useful if someone can confirm they are being recorded.
        # The trace is intentionally env-var-only with no config key, which also means nothing
        # reports it — so a run that silently recorded nothing (the variable set in a different
        # shell from the one that started the daemon, the usual mistake) looked identical to a
        # working one until the directory turned up empty. Counting the file here is host-only
        # and reads no record content.
        directory = trace.get("directory")
        path = pathlib.Path(str(directory)) / "decisions.jsonl" if directory else None
        try:
            trace["records"] = (
                sum(1 for line in path.open(encoding="utf-8") if line.strip())
                if path is not None and path.is_file()
                else 0
            )
        except OSError as exc:
            trace["records"] = None
            trace["records_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "enabled": enabled,
        "mode": mode,
        "chain": chain,
        "max_candidates": max_candidates,
        "ready": enabled and mode in {"shadow", "advisory"} and available,
        "providers": providers,
        "training_trace": trace,
    }


def evaluate_selective_policy(
    context: PolicyContext,
    selectors: Sequence[PolicySelector],
    *,
    mode: PolicyMode,
    max_candidates: int = 4,
    primary_reviews: int = 2,
    reviewer_reviews: int = 3,
) -> PolicyDecision:
    """Evaluate the chain, recording the decision when local tracing is switched on.

    Tracing wraps the whole evaluation rather than hooking each return, so no future branch can
    escape it and the traced object is exactly the decision the caller receives. When tracing is
    off this costs one boolean check.
    """

    decision = _evaluate_selective_policy(
        context,
        selectors,
        mode=mode,
        max_candidates=max_candidates,
        primary_reviews=primary_reviews,
        reviewer_reviews=reviewer_reviews,
    )
    from . import policy_trace

    if policy_trace.enabled():
        policy_trace.record_decision(context, decision.eligible_candidates, decision)
    return decision
