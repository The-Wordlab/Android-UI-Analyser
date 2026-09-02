"""The optional local policy model: model_control status/action/chat/agent-test, policy tap-candidate and selection helpers, the session policy side channel, and session_autopilot which lets the policy drive a bounded stretch.

Engine methods for policy. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_policy.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any, cast

from .engine_support import _GENERIC_MANUAL_MATCH_TERMS, logger
from .errors import AuaError, UsageError
from .memory import (
    RouteStep,
    arrival_destination_terms,
    is_destructive_step,
    recorded_selector,
    redact_label,
    title_of,
)
from .providers.base import PolicyProvider
from .schema import ActionResult, AnalyzeResult, Element

if TYPE_CHECKING:
    from .engine import Engine
    from .policy import PolicyContext, PolicyDecision, PolicyMode, PolicySelector


def _configured_policy_mode(self: Engine) -> PolicyMode:
    """The mode the operator configured, ignoring the `enabled` resource switch."""
    section = getattr(self.config, "policy", None)
    mode = str(getattr(section, "mode", "off") or "off").strip().casefold()
    return cast("PolicyMode", mode) if mode in {"off", "shadow", "advisory"} else "off"


def _session_policy_mode(self: Engine) -> PolicyMode:
    """Resolve the opt-in selector mode without making policy a base dependency."""
    # `session autopilot` sets this for its own duration. Running the chain costs real time —
    # around twenty seconds per analyze with the reviewer in play — so `policy.enabled` is the
    # switch that keeps ordinary navigation from paying it. That left no way to use autopilot
    # without also taxing every unrelated analyze, and the observed outcome was the policy
    # being switched off entirely, which then made autopilot refuse with "set enabled=true".
    # Typing the command is the opt-in; the flag governs the passive advice, not this.
    operator_override = self.model_control.intercept_override()
    if operator_override is False:
        return "off"
    override = self._policy_mode_override
    if override is not None:
        return override
    section = getattr(self.config, "policy", None)
    if section is None or (
        operator_override is not True and not bool(getattr(section, "enabled", False))
    ):
        return "off"
    mode = str(getattr(section, "mode", "off") or "off").strip().casefold()
    return cast("PolicyMode", mode) if mode in {"off", "shadow", "advisory"} else "off"


def model_control_status(self: Engine, *, limit: int = 100) -> dict[str, Any]:
    """Return the shared operator switches, resident state, and recent model exchanges."""

    from .model_control import MODEL_NAMES, model_context_window

    control = self.model_control
    state = control.read_state()
    providers: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        try:
            provider = self.factory.create("policy", name)
            status_method = getattr(provider, "status", None)
            availability = provider.is_available()
            value = (
                dict(status_method())
                if callable(status_method)
                else {
                    "provider": name,
                    "available": availability.ok,
                    "reason": availability.reason,
                }
            )
            value["context_window"] = value.get("context_window") or model_context_window(
                provider.settings
            )
            value["enabled"] = control.provider_enabled(name)
        except Exception as exc:
            value = {
                "provider": name,
                "available": False,
                "loaded": False,
                "enabled": control.provider_enabled(name),
                "reason": f"{type(exc).__name__}: {exc}",
                "context_window": None,
            }
        providers.append(value)
    return {
        "ok": True,
        "control": state,
        "configured_mode": self._configured_policy_mode(),
        "configured_chain": list(getattr(self.config.policy, "chain", []) or []),
        "strategy": str(getattr(self.config.policy, "strategy", "single")),
        "providers": providers,
        "events": control.events(limit=limit),
    }


def model_control_action(
    self: Engine,
    action: str,
    *,
    provider: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Apply one host-local model operation without touching the Android device."""

    from .model_control import MODEL_NAMES

    control = self.model_control
    if action == "set_intercept":
        if not isinstance(enabled, bool):
            raise UsageError("model intercept action requires enabled=true or false")
        control.update(intercept_enabled=enabled)
        control.record(
            {
                "source": "dashboard",
                "phase": "complete",
                "operation": "intercept_on" if enabled else "intercept_off",
            }
        )
        return self.model_control_status()
    if action == "clear_events":
        control.clear_events()
        return self.model_control_status()
    if provider not in MODEL_NAMES:
        raise UsageError(f"unknown local model {provider!r}")
    if action == "set_provider":
        if not isinstance(enabled, bool):
            raise UsageError("model provider action requires enabled=true or false")
        control.update(provider=provider, provider_enabled=enabled)
        control.record(
            {
                "provider": provider,
                "source": "dashboard",
                "phase": "complete",
                "operation": "enable" if enabled else "disable",
            }
        )
        return self.model_control_status()
    instance = cast("PolicyProvider", self.factory.create("policy", provider))
    if action == "load":
        result = instance.load_model()
    elif action == "unload":
        result = instance.unload_model()
    else:
        raise UsageError(f"unknown model control action {action!r}")
    return {**result, "status": self.model_control_status()}


def model_control_chat(
    self: Engine,
    provider: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run a bounded direct local-model exchange on the same resident daemon instance."""

    from .model_control import MODEL_NAMES

    if provider not in MODEL_NAMES:
        raise UsageError(f"unknown local model {provider!r}")
    if not self.model_control.provider_enabled(provider):
        raise UsageError(f"local model {provider!r} is disabled in the dashboard")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 30:
        raise UsageError("model playground needs between 1 and 30 messages")
    clean: list[dict[str, str]] = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            raise UsageError("each model message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise UsageError("model messages need a system, user, or assistant role and text")
        total_chars += len(content)
        if total_chars > 100_000:
            raise UsageError("model playground context exceeds 100000 characters")
        clean.append({"role": role, "content": content})
    instance = cast("PolicyProvider", self.factory.create("policy", provider))
    return instance.interact(clean, max_tokens=max_tokens)


def _evaluate_policy_context(
    self: Engine,
    context: PolicyContext,
    *,
    mode: PolicyMode,
) -> tuple[PolicyContext, PolicyDecision, tuple[PolicySelector, ...]]:
    """Run the configured agent policy evaluator for an already-compiled context."""

    from .policy import (
        evaluate_policy,
        evaluate_selective_policy,
        guard_candidates,
    )

    max_candidates = max(1, int(getattr(self.config.policy, "max_candidates", 4)))
    eligible = guard_candidates(context, max_candidates=max_candidates)
    selector: PolicySelector | None = None
    selectors: tuple[PolicySelector, ...] = ()
    if len(eligible) > 1:
        chain = self.factory.build_chain("policy")
        selectors = tuple(cast("PolicySelector", provider) for provider in chain.providers)
        selector = selectors[0] if selectors else None
        supports_handoff = getattr(selector, "supports_handoff", None)
        if callable(supports_handoff):
            # This is an authenticated provider capability, not caller-controlled input.
            with contextlib.suppress(Exception):
                context = dataclass_replace(
                    context,
                    allow_handoff=bool(supports_handoff()),
                )
    if getattr(self.config.policy, "strategy", "single") == "selective_hybrid":
        decision = evaluate_selective_policy(
            context,
            selectors,
            mode=mode,
            max_candidates=max_candidates,
            primary_reviews=int(getattr(self.config.policy, "primary_reviews", 3)),
            reviewer_reviews=int(getattr(self.config.policy, "reviewer_reviews", 3)),
        )
    else:
        decision = evaluate_policy(
            context,
            selector,
            mode=mode,
            max_candidates=max_candidates,
        )
    return context, decision, selectors


def model_control_agent_test(
    self: Engine,
    provider: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a synthetic request through the same policy path as an agent turn."""

    from .policy import PolicyCandidate, PolicyContext, compile_policy_context, policy_tools

    if provider != "agent_chain":
        raise UsageError("agent model tests must use the configured agent policy chain")
    if not isinstance(request, dict):
        raise UsageError("agent model test request must be a JSON object")

    def bounded_text(name: str, value: Any, *, maximum: int, required: bool = True) -> str:
        if not isinstance(value, str) or (required and not value.strip()):
            raise UsageError(f"agent model test field {name!r} must be text")
        clean = value.strip()
        if len(clean) > maximum:
            raise UsageError(f"agent model test field {name!r} exceeds {maximum} characters")
        return clean

    goal = bounded_text("goal", request.get("goal"), maximum=500)
    phase = bounded_text("phase", request.get("phase", "Choose the next action"), maximum=200)
    raw_candidates = request.get("candidates")
    if not isinstance(raw_candidates, list) or not 2 <= len(raw_candidates) <= 4:
        raise UsageError("agent model test needs between 2 and 4 candidates")
    fingerprint = "dashboard-agent-model-test-v1"
    package = "dashboard.sample"
    candidates: list[PolicyCandidate] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise UsageError("each agent model test candidate must be an object")
        candidate_id = raw.get("id", index)
        if (
            isinstance(candidate_id, bool)
            or not isinstance(candidate_id, int)
            or candidate_id != index
        ):
            raise UsageError("agent model test candidate ids must be dense integers from 0")
        label = bounded_text("candidate.label", raw.get("label"), maximum=200)
        purpose = bounded_text(
            "candidate.purpose", raw.get("purpose", f"Tap {label}"), maximum=300
        )
        proof = bounded_text(
            "candidate.proof",
            raw.get("proof", f"Visible current-frame control labelled {label}"),
            maximum=300,
        )
        arguments = {"element_id": candidate_id}
        candidates.append(
            PolicyCandidate(
                candidate_id=candidate_id,
                call={"tool": "tap_and_analyze", "arguments": arguments},
                model_arguments=arguments,
                purpose=purpose,
                proof=proof,
                phase=phase,
                observation_fingerprint=fingerprint,
                package=package,
            )
        )
    constraints_raw = request.get("constraints", [])
    if not isinstance(constraints_raw, list) or len(constraints_raw) > 8:
        raise UsageError("agent model test constraints must be a list of at most 8 strings")
    constraints = tuple(
        bounded_text("constraint", value, maximum=300) for value in constraints_raw
    )
    observation_raw = request.get("observation", {})
    if not isinstance(observation_raw, dict):
        raise UsageError("agent model test observation must be an object")
    observation = {
        "fresh": bool(observation_raw.get("fresh", True)),
        "known_screen": bounded_text(
            "observation.known_screen",
            observation_raw.get("known_screen", "sample_screen"),
            maximum=200,
        ),
        "element_count": len(candidates),
        "source": bounded_text(
            "observation.source",
            observation_raw.get("source", "hierarchy"),
            maximum=50,
        ),
    }
    allow_handoff = request.get("allow_handoff", True)
    if not isinstance(allow_handoff, bool):
        raise UsageError("agent model test allow_handoff must be true or false")
    context = PolicyContext(
        goal=goal,
        phase=phase,
        candidates=tuple(candidates),
        observation=observation,
        constraints=constraints,
        observation_fingerprint=fingerprint,
        package=package,
        allow_handoff=allow_handoff,
    )
    started_ms = int(time.time() * 1000)
    evaluated_context, decision, selectors = self._evaluate_policy_context(
        context,
        mode="advisory",
    )
    provider_names = [
        str(getattr(selector, "name", type(selector).__name__)) for selector in selectors
    ]
    exchanges = [
        event
        for event in self.model_control.events(limit=200)
        if event.get("provider") in provider_names
        and event.get("source") == "agent"
        and int(event.get("timestamp_ms") or 0) >= started_ms - 5
        and event.get("phase") in {"complete", "error"}
    ]
    exchange = exchanges[-1] if exchanges else None
    selected = decision.selected_candidate
    selected_id = selected.candidate_id if selected is not None else None
    selected_candidate = selected.as_model_value() if selected is not None else None
    decision_json = decision.as_json()
    return {
        "ok": True,
        "provider": decision.provider,
        "providers": provider_names,
        "status": decision.status,
        "selected_id": selected_id,
        "selected_candidate": selected_candidate,
        "decision": decision_json,
        "compiled_context": compile_policy_context(evaluated_context),
        "tool_schema": policy_tools(allow_handoff=evaluated_context.allow_handoff)
        if "functiongemma" in provider_names
        else [],
        "exchange": exchange,
        "exchanges": exchanges,
        "provider_error": decision.error,
    }


def _policy_selector_arguments(
    element: Element,
    elements: Sequence[Element],
) -> tuple[dict[str, Any], str] | None:
    """Return one privacy-filtered selector and its safe display label.

        Durable selectors remain preferred.  A frame-bound element ID is allowed only when
        the copy is durable by itself and ambiguity comes solely from a passive duplicate
        (normally the page title beside one clickable row).  The candidate is still bound to
        the current observation fingerprint, so this fallback is never reusable across frames.
        """
    selector = recorded_selector(element, elements=elements)
    by = selector.get("by")
    redacted = redact_label(element)
    if redacted == "<redacted>":
        return None
    # A stable id can make the *action* reusable even when its adjacent copy is volatile.
    # For policy selection we additionally need safe semantic evidence, so withhold the
    # whole candidate when the durable-selector filter refused non-empty copy/description.
    if by == "id" and (
        (bool((element.text or "").strip()) and not selector.get("label"))
        or (bool((element.content_desc or "").strip()) and not selector.get("content_desc"))
    ):
        return None
    if by == "id" and selector.get("resource_id"):
        value = str(selector["resource_id"])
        args: dict[str, Any] = {"rid": value}
    elif by == "desc" and selector.get("content_desc"):
        value = str(selector["content_desc"])
        args = {"desc": value}
    elif by == "text" and selector.get("label"):
        value = str(selector["label"])
        args = {"text": value}
    else:
        # Re-run without neighbouring elements to distinguish safe copy that is ambiguous
        # only because a passive title duplicates it from copy that is itself volatile,
        # secret, or otherwise unsuitable. Two clickable duplicates remain ambiguous.
        standalone = recorded_selector(element, elements=())
        standalone_by = standalone.get("by")
        if element.resource_id or standalone_by not in {"text", "desc"}:
            return None
        if standalone_by == "desc":
            value = str(standalone.get("content_desc") or "")
            matches = [
                other for other in elements if (other.content_desc or "").strip()[:60] == value
            ]
        else:
            value = str(standalone.get("label") or "")
            matches = [other for other in elements if (other.text or "").strip()[:60] == value]
        clickable_matches = [
            other
            for other in matches
            if other.clickable and other.enabled is not False and other.window in {None, "app"}
        ]
        if (
            not value
            or len(matches) <= 1
            or len(clickable_matches) != 1
            or clickable_matches[0].id != element.id
        ):
            return None
        args = {"id": element.id}

    # `recorded_selector` already refuses secrets, PII, typed values, dynamic values, and
    # ambiguous selectors. `redact_label` is an independent final check, but its result is
    # used only when the stricter durable-selector filter preserved the same copy. Dynamic
    # copy beside a safe resource id must not sneak back into the prompt through prose.
    persisted_label = selector.get("label") or selector.get("content_desc")
    safe_label = (
        str(persisted_label)
        if persisted_label and redacted not in {None, "<redacted>"}
        else value
    )
    return args, safe_label


def _policy_target_terms(objective: str) -> list[str]:
    """Extract action-object terms without proof-contract scaffolding.

        Authored checkpoints often read ``Prove the real X destination, not the search
        result``. Words such as ``prove``, ``destination``, and ``search result`` describe
        the evidence contract, not controls the policy should offer. Navigation verbs are
        handled by :func:`arrival_destination_terms`; this extra lane handles proof-led
        checkpoints conservatively and app-agnostically.
        """
    from .session import _goal_terms

    if re.search(
        r"\b(?:open|tap|press|click|reach|enter|visit|view|inspect|verify|select|choose|"
        r"navigate(?:\s+once)?\s+to|go\s+to|return\s+to)\b",
        objective,
        flags=re.IGNORECASE,
    ):
        return _goal_terms(" ".join(arrival_destination_terms(objective)) or objective)
    proof = re.search(
        r"\b(?:prove|confirm|assert|check)\s+(?:the\s+)?"
        r"(?:(?:real|actual|requested)\s+)?(?P<destination>.+?)"
        r"(?=\s+(?:destination|page|screen|view|panel)\b|"
        r"\s+reached\b|,\s*(?:not|rather)\b|$)",
        objective,
        flags=re.IGNORECASE,
    )
    if proof is not None:
        terms = _goal_terms(proof.group("destination"))
        if terms:
            return terms
    return _goal_terms(" ".join(arrival_destination_terms(objective)) or objective)


def _policy_tap_candidates(
    self: Engine,
    state: Any,
    phase: Any,
    observation: AnalyzeResult,
    *,
    objective: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> list[Any]:
    """Compile guard-owned exact calls from one fresh frame.

        The optional model never sees the hierarchy and never authors arguments. It receives
        only enabled app controls with a unique durable selector, locally-proved semantic
        relevance, and a clean destructive-risk check. Toggle/input/system controls stay out of
        this first integration because a generic tap can mutate state without proving progress.
        """
    from .policy import PolicyCandidate
    from .session import _goal_terms, _match_score

    fingerprint = observation.meta.fingerprint
    package = observation.screen.package
    # A compound goal may enumerate the visible alternatives after naming the requested
    # destination (``Open History from these choices: Grammar, History, Physics``).  Those
    # alternatives are context, not evidence that every row is goal-relevant.  Reuse the
    # same destination-object extraction as arrival proof so candidate recall and model
    # selection are both conditioned on the requested target only.
    policy_objective = objective or phase.objective
    target_terms = self._policy_target_terms(policy_objective)
    policy_goal = " ".join(target_terms) or policy_objective
    goal_terms = set(target_terms)
    ranked: list[tuple[int, str, Any]] = []
    max_candidates = max(1, int(getattr(self.config.policy, "max_candidates", 4)))
    stage_counts = {
        "elements": len(observation.elements),
        "enabled_clickable": 0,
        "safe_control": 0,
        "stable_selector": 0,
        "frame_selector": 0,
        "non_destructive": 0,
        "target_matched": 0,
        "offered": 0,
    }

    for element in observation.elements:
        if not element.clickable or element.enabled is False:
            continue
        stage_counts["enabled_clickable"] += 1
        if element.checkable or element.selected is True or element.window not in {None, "app"}:
            continue
        element_type = element.type.casefold()
        if "edittext" in element_type or "textfield" in element_type or "input" in element_type:
            continue
        stage_counts["safe_control"] += 1

        selector_value = self._policy_selector_arguments(element, observation.elements)
        if selector_value is None:
            continue
        arguments, safe_label = selector_value
        if "id" in arguments:
            stage_counts["frame_selector"] += 1
        else:
            stage_counts["stable_selector"] += 1

        rid_label = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])",
            " ",
            (element.resource_id or "").rsplit("/", 1)[-1],
        ).replace("_", " ")
        # Raw copy is used only by deterministic in-process classification. It is never
        # placed in the PolicyContext, response, journal, or model prompt.
        risk_label = " ".join(
            value for value in (element.text, element.content_desc, rid_label) if value
        )
        if is_destructive_step(
            RouteStep(kind="tap", label=risk_label),
            self.config.memory.destructive_labels,
        ):
            continue
        stage_counts["non_destructive"] += 1

        semantic_label = " ".join(value for value in (safe_label, rid_label) if value)
        matched_terms = goal_terms & set(_goal_terms(semantic_label))
        target_matched = bool(
            matched_terms and not matched_terms <= _GENERIC_MANUAL_MATCH_TERMS
        )
        if target_matched:
            stage_counts["target_matched"] += 1
        if (
            not target_matched
            and getattr(self.config.policy, "candidate_scope", "goal_matched") != "safe_visible"
        ):
            continue
        score = _match_score(policy_goal, semantic_label, exactness=safe_label)
        call = {"tool": "tap_and_analyze", "arguments": arguments}
        material = json.dumps(
            {
                "session_id": state.session_id,
                "phase_id": phase.id,
                "fingerprint": fingerprint,
                "package": package,
                "call": call,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        candidate = PolicyCandidate(
            # Dense opaque IDs are assigned after the bounded candidate set is known.
            # FunctionGemma v3 was trained on exactly this 0..N-1 ID vocabulary.
            candidate_id=0,
            call=call,
            model_arguments=arguments,
            purpose=f"Tap the current-frame {safe_label!r} control and observe the result.",
            proof="The exact call returns a folded post-action observation.",
            safe=True,
            authorized=True,
            redundant=False,
            session_id=state.session_id,
            phase=phase.id,
            observation_fingerprint=fingerprint,
            package=package,
        )
        ranked.append((score, material, candidate))

    # Keep the most goal-relevant guarded calls. Assign a dense hidden permutation of
    # 0..N-1 because those are the only opaque IDs in the frozen adapter's vocabulary;
    # use a separate stable permutation for display order so neither position nor source
    # hierarchy order leaks a preference.
    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected = ranked[:max_candidates]
    id_rows = sorted(
        selected,
        key=lambda row: hashlib.sha256(f"policy-id\0{row[1]}".encode()).hexdigest(),
    )
    ids = {row[1]: candidate_id for candidate_id, row in enumerate(id_rows)}
    ordered = sorted(
        selected,
        key=lambda row: hashlib.sha256(f"policy-order\0{row[1]}".encode()).hexdigest(),
    )
    candidates = [dataclass_replace(row[2], candidate_id=ids[row[1]]) for row in ordered]
    stage_counts["offered"] = len(candidates)
    if diagnostics is not None:
        diagnostics.update(
            {
                "schema_version": 1,
                "target_term_count": len(goal_terms),
                "stages": stage_counts,
            }
        )
    return candidates


def _policy_navigation_waypoints(objective: str) -> list[str]:
    """Extract ordered, explicitly-authored tap destinations from a compound phase.

        Goal compilation intentionally keeps ordinary ``and`` inside one proof checkpoint.
        A bounded local navigator still needs to distinguish ``open Catalog, then open
        Archive`` from the later input/proof clauses.  This helper does not invent a route:
        it preserves only objects that immediately follow an authored navigation verb and
        stops each object at the next authored action or assertion.
        """

    verb = (
        r"(?:open|tap|press|click|select|choose|visit|view|"
        r"navigate(?:\s+(?:once\s+)?to)?|go\s+to)"
    )
    boundary = (
        r"(?=\s*(?:,|;|\.|\bthen\b|\band\b)?\s*(?:"
        + verb
        + r"|enter|type|input|write|generate|submit|send|wait|"
        r"verify|prove|confirm|assert|check|ensure)\b|\s*$)"
    )
    pattern = re.compile(
        rf"\b{verb}\s+(?:the\s+)?(?P<object>.+?){boundary}",
        flags=re.IGNORECASE,
    )
    waypoints: list[str] = []
    for match in pattern.finditer(objective):
        value = " ".join(match.group("object").strip(" ,;.").split())
        value = re.sub(r"^(?:to\s+)", "", value, flags=re.IGNORECASE)
        if value and value.casefold() not in {item.casefold() for item in waypoints}:
            waypoints.append(value[:160])
    return waypoints


def _policy_waypoint_arrived(waypoint: str, observation: AnalyzeResult) -> bool:
    """Return whether a passive current-screen title proves *waypoint* arrival."""

    from .session import _goal_terms

    terms = set(_goal_terms(waypoint))
    if not terms:
        return False
    visible = [element for element in observation.elements if element.clickable is not True]
    title = title_of(visible, observation.screen.height)
    if not title:
        return False
    title_terms = set(_goal_terms(title))
    # A one-word child must not be declared reached by a broader parent title such as
    # ``Network & internet``. Multi-word destinations retain the established title-evidence
    # lane because every discriminating term must be present.
    return terms == title_terms


def _restore_term_case(objective: str, terms: Sequence[str]) -> list[str]:
    """Return *terms* spelled the way the objective spells them.

        Term extraction case-folds so that matching is case-insensitive, which is
        correct for matching and wrong for the prompt: the model compares the goal
        against candidate labels that keep their original capitalisation. A rare
        label like ``Stylist`` stops binding once it arrives as ``stylist``, and the
        model then settles on a commoner neighbour. Only the spelling is restored —
        which terms survive filtering is decided upstream and unchanged here.
        """

    restored: list[str] = []
    for term in terms:
        match = re.search(rf"\b{re.escape(term)}\b", objective, flags=re.IGNORECASE)
        restored.append(match.group(0) if match is not None else term)
    return restored


def _policy_selection_goal(objective: str, candidates: Sequence[Any]) -> str:
    """Keep safe disambiguating evidence without reintroducing alternative-list bias.

        Candidate filtering intentionally uses only the requested destination.  The selector can
        still need a qualifier when several safe rows share that destination (for example four
        ``History`` rows with different summaries).  Preserve only objective terms that also
        occur in privacy-screened candidate prose, after removing explicit alternative lists.
        User text, typed values, and unrelated private vocabulary therefore cannot enter the
        local-model prompt through this seam.
        """

    target_terms = _policy_target_terms(objective)
    target_goal = " ".join(_restore_term_case(objective, target_terms)) or objective
    cleaned = re.sub(
        r"\s+(?:from|among)\s+(?:the\s+)?(?:these\s+)?(?:visible\s+)?"
        r"(?:[A-Za-z]+\s+)?(?:destinations|choices)\s*:\s*.*$",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:rather\s+than|choices\s+are|the\s+alternatives\s+are|"
        r"available\s+destinations\s+are)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    from .session import _goal_terms

    candidate_terms: set[str] = set()
    for candidate in candidates:
        candidate_terms.update(_goal_terms(str(getattr(candidate, "purpose", ""))))
    generic = {
        "action",
        "call",
        "control",
        "current",
        "exact",
        "folded",
        "frame",
        "observation",
        "observe",
        "post",
        "result",
        "returns",
        "tap",
    }
    target_set = set(target_terms)
    qualifiers: list[str] = []
    for term in _goal_terms(cleaned):
        if (
            term in candidate_terms
            and term not in target_set
            and term not in generic
            and term not in qualifiers
        ):
            qualifiers.append(term)
    if not qualifiers:
        return target_goal
    evidence = " ".join(_restore_term_case(cleaned, qualifiers))
    return f"Requested destination: {target_goal}. Matching evidence: {evidence}."


def _policy_suggestion(candidate: Any) -> dict[str, Any]:
    """Render a selected guard-owned call for advisory mode only."""
    call: dict[str, Any] = {
        "tool": str(candidate.call["tool"]),
        "arguments": dict(candidate.call["arguments"]),
    }
    arguments = call["arguments"]
    if "id" in arguments:
        cli = f"aua tap-and-analyze {int(arguments['id'])}"
    elif "rid" in arguments:
        cli = f"aua tap-and-analyze --rid {shlex.quote(str(arguments['rid']))}"
    elif "desc" in arguments:
        cli = f"aua tap-and-analyze --desc {shlex.quote(str(arguments['desc']))}"
    else:
        cli = f"aua tap-and-analyze --text {shlex.quote(str(arguments['text']))}"
    return {
        "kind": "policy_advisory",
        "candidate_id": candidate.candidate_id,
        "cli": cli,
        "mcp": call,
        "reason": (
            "The optional local policy selected this guard-approved current-frame call. "
            "AUA has not executed it and has not replaced the deterministic recommendation."
        ),
        "executes": True,
    }


def _policy_handoff(*, model_used: bool, reason_code: str) -> dict[str, Any]:
    """Render a structured, non-executing return to the parent agent."""
    if reason_code == "no_guard_approved_candidate":
        reason = (
            "The optional local policy found no supplied guard-approved action that "
            "directly advances the active goal. It has executed nothing; return control "
            "to the parent agent for a fresh observation, broader recovery, or a clear "
            "target-absent result."
        )
    else:
        reason = (
            "The optional local policy judged that none of the supplied guard-approved "
            "actions directly advances the active goal. It has executed nothing; return "
            "control to the parent agent for broader recovery or a clear target-absent result."
        )
    return {
        "kind": "policy_handoff",
        "reason_code": reason_code,
        "reason": reason,
        "model_used": model_used,
        "executes": False,
    }


def _policy_context_is_current(
    self: Engine,
    state: Any,
    phase: Any,
    observation: AnalyzeResult,
    candidate: Any,
) -> tuple[bool, str | None]:
    """Revalidate session, phase, and frame provenance after model latency."""
    fingerprint = observation.meta.fingerprint
    package = observation.screen.package
    try:
        current_state = self._session_state(state.session_id)
    except Exception as exc:  # pragma: no cover - defensive, surfaced as policy metadata
        return False, f"session revalidation failed: {type(exc).__name__}"
    current_phase = next(
        (item for item in current_state.phases if item.status != "completed"),
        None,
    )
    if current_state.session_id != state.session_id or current_state.finished_ms is not None:
        return False, "the goal session changed or finished during policy evaluation"
    if current_phase is None or current_phase.id != phase.id:
        return False, "the active goal phase changed during policy evaluation"
    if candidate.observation_fingerprint != fingerprint or candidate.package != package:
        return False, "the selected candidate is not bound to the supplied observation"
    if observation.meta.device_serial not in {None, current_state.serial}:
        return False, "the supplied observation belongs to another device"

    # A warm Engine can observe a newer frame while a slow policy call is in flight. Never
    # expose a selector from the older frame in that case. A short-lived Engine may have no
    # cache here; session/phase/candidate provenance still provides the binding.
    latest = self._last_analyze_result
    if latest is not None:
        latest_fingerprint = latest.meta.fingerprint
        if latest_fingerprint and latest_fingerprint != fingerprint:
            return False, "a newer observation replaced the policy input frame"
        if latest.screen.package != package:
            return False, "the foreground package changed during policy evaluation"
    return True, None


def _session_policy_output(
    self: Engine,
    state: Any,
    phase: Any,
    observation: AnalyzeResult | None,
    *,
    recommended_call: Any,
    policy_objective: str | None = None,
    recent_outcomes: Sequence[str] = (),
    _return_selected: bool = False,
) -> dict[str, Any]:
    """Evaluate the optional policy as a non-fatal, non-executing side channel."""
    mode = self._session_policy_mode()
    if mode == "off":
        return {}

    deterministic_kind = (
        recommended_call.get("kind") if isinstance(recommended_call, dict) else None
    )
    if phase.kind != "verify" or deterministic_kind not in {
        "manual_action",
        "manual_observation",
    }:
        audit: dict[str, Any] = {
            "mode": mode,
            "status": "skipped_deterministic",
            "provider": None,
            "model_used": False,
            "candidate_count": 0,
            "eligible_candidate_ids": [],
            "error": None,
        }
        return {"policy": audit}
    if (
        observation is None
        or not observation.meta.fingerprint
        or bool(observation.meta.stale_risk)
        or not observation.screen.package
    ):
        audit = {
            "mode": mode,
            "status": "skipped_unbound_observation",
            "provider": None,
            "model_used": False,
            "candidate_count": 0,
            "eligible_candidate_ids": [],
            "error": "policy requires a fresh fingerprinted observation",
        }
        return {"policy": audit}

    try:
        from .policy import PolicyContext

        compiler_audit: dict[str, Any] = {}
        candidates = self._policy_tap_candidates(
            state,
            phase,
            observation,
            objective=policy_objective,
            diagnostics=compiler_audit,
        )
        deterministic_mcp = (
            recommended_call.get("mcp") if isinstance(recommended_call, dict) else None
        )
        compiler_audit["recommended_call_offered"] = (
            any(candidate.trusted_call() == deterministic_mcp for candidate in candidates)
            if isinstance(deterministic_mcp, dict)
            else None
        )
        objective = policy_objective or phase.objective
        policy_goal = self._policy_selection_goal(objective, candidates)
        context = PolicyContext(
            goal=policy_goal,
            phase=phase.id,
            session_id=state.session_id,
            candidates=tuple(candidates),
            observation={
                "fresh": True,
                **(
                    {"known_screen": observation.meta.known_screen}
                    if observation.meta.known_screen
                    else {}
                ),
            },
            constraints=(
                "Select only a supplied guard-approved candidate.",
                "Do not invent or execute a call.",
            ),
            recent_outcomes=(
                "session_active=true",
                "outcome=known",
                "goal_checkpoint_reached=false",
                *tuple(recent_outcomes),
            ),
            observation_fingerprint=observation.meta.fingerprint,
            package=observation.screen.package,
        )
        context, decision, _selectors = self._evaluate_policy_context(context, mode=mode)
        audit = decision.as_json()
        audit["compiler"] = compiler_audit
        # Exact calls belong only in the separate advisory field, never in shadow/audit.
        audit.pop("recommended_call", None)
        selected = decision.selected_candidate
        # The dashboard switch is out-of-band specifically so an operator can kill a bad
        # model while the serialized daemon is still inside MLX. Generation itself may finish,
        # but a result completed after OFF must never become advice or an autopilot action.
        if self.model_control.intercept_override() is False:
            selected = None
            audit["status"] = "discarded_operator_disabled"
            audit["model_used"] = bool(decision.model_used)
            audit.pop("selected_candidate_id", None)
        suggestion = None
        if selected is not None:
            current, stale_reason = self._policy_context_is_current(
                state,
                phase,
                observation,
                selected,
            )
            if not current:
                audit["status"] = "rejected_stale_context"
                audit["error"] = stale_reason
                audit.pop("selected_candidate_id", None)
            elif mode == "advisory" and decision.model_used:
                suggestion = self._policy_suggestion(selected)
        out: dict[str, Any] = {"policy": audit}
        if suggestion is not None:
            out["policy_suggestion"] = suggestion
        if (
            _return_selected
            and mode == "advisory"
            and selected is not None
            and audit.get("status") not in {"rejected_stale_context", "handoff"}
        ):
            # Private return lane for ``session_autopilot``. The exact trusted call never
            # enters shadow metadata and is removed before the public result is serialized.
            out["_selected_policy_call"] = selected.trusted_call()
            out["_selected_policy_candidate_id"] = selected.candidate_id
        if mode == "advisory" and decision.status == "no_candidate":
            out["policy_handoff"] = self._policy_handoff(
                model_used=False,
                reason_code="no_guard_approved_candidate",
            )
        elif mode == "advisory" and decision.status == "handoff":
            out["policy_handoff"] = self._policy_handoff(
                model_used=True,
                reason_code="no_supplied_candidate_advances_goal",
            )
        return out
    except Exception as exc:  # policy is optional and must never break a UI result
        logger.warning("optional policy evaluation failed: %s", exc)
        audit = {
            "mode": mode,
            "status": "error",
            "provider": None,
            "model_used": False,
            "candidate_count": 0,
            "eligible_candidate_ids": [],
            "error": f"policy evaluation failed: {type(exc).__name__}",
        }
        return {"policy": audit}


def _autopilot_public_policy_output(value: dict[str, Any]) -> dict[str, Any]:
    """Strip the private trusted-call lane before recording a policy decision."""

    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "_selected_policy_call",
            "_selected_policy_candidate_id",
            # The ordinary advisory text says the call was not executed. Autopilot records
            # the exact trusted call and its executed boolean itself, so retaining that
            # sentence inside the trace would contradict the actual result.
            "policy_suggestion",
        }
    }


def _autopilot_provider_failure(audit: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return (terminal_reason, detail) when the *model*, not the screen, ended the step.

        Observed live: a chain whose fallback returned unparsable output roughly four times in
        five reported every stop as "no visible guard-approved tap advances the navigation".
        The guard was right to reject the output, but the run named the wrong cause, and the
        measured rate — the one number that identifies a broken provider — appeared nowhere.
        """

    from . import policy_health

    status = str(audit.get("status") or "")
    reasons: list[str] = []
    providers: list[str] = []
    for item in audit.get("selection_trace") or ():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status")) == "provider_unusable":
            providers.append(str(item.get("provider") or "?"))
            reasons.append(f"{item.get('provider')}: {item.get('reason')}")
        if (
            str(item.get("status")) == "no_consensus"
            and int(item.get("attempts") or 0) > 0
            and int(item.get("invalid_attempts") or 0) >= int(item.get("attempts") or 0)
        ):
            providers.append(str(item.get("provider") or "?"))
            reasons.append(
                f"{item.get('provider')}: every bounded selection attempt was invalid"
            )
    if status in {"provider_unusable", "invalid_selection", "provider_error", "unavailable"}:
        providers.append(str(audit.get("provider") or "?"))
        reasons.append(f"{audit.get('provider')}: {audit.get('error') or status}")
    if not reasons:
        return None
    rates = []
    for provider in dict.fromkeys(providers):
        health = policy_health.report(provider)
        if health["attempts"]:
            rates.append(
                f"{provider} was invalid in {health['invalid']} of {health['attempts']} "
                f"recent attempts"
            )
    reason = "unavailable" if status == "unavailable" else "unusable"
    detail = (
        f"The local policy produced {reason} output, so nothing was executed: "
        + "; ".join(reasons)
    )
    if rates:
        detail += ". Measured: " + "; ".join(rates)
    return (
        "provider_unavailable" if status == "unavailable" else "provider_output_unusable",
        detail,
    )


def _execute_guarded_policy_call(self: Engine, call: dict[str, Any]) -> ActionResult:
    """Execute one already-guarded local-policy call through the normal Engine action."""

    if call.get("tool") != "tap_and_analyze":
        raise UsageError(
            "local autopilot received an unsupported guarded call",
            hint="No action was executed; return control to the parent agent.",
            code="policy_call_unsupported",
        )
    raw_arguments = call.get("arguments")
    arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}
    allowed = {"id", "rid", "text", "desc"}
    if set(arguments) - allowed or len(arguments) != 1:
        raise UsageError(
            "local autopilot received malformed guarded tap arguments",
            hint="No action was executed; return control to the parent agent.",
            code="policy_call_invalid",
        )
    if "id" in arguments:
        element_id = arguments["id"]
        if not isinstance(element_id, int) or isinstance(element_id, bool):
            raise UsageError(
                "local autopilot received a non-integer frame id",
                code="policy_call_invalid",
            )
        return self.tap(element_id, observe=True)
    selector = {key: arguments[key] for key in ("rid", "text", "desc") if key in arguments}
    if not selector or not all(isinstance(value, str) and value for value in selector.values()):
        raise UsageError(
            "local autopilot received an empty semantic selector",
            code="policy_call_invalid",
        )
    return self.tap(selector=selector, observe=True)


def session_autopilot(
    self: Engine,
    session_id: str | None = None,
    *,
    max_steps: int = 6,
    max_duration_ms: int = 30_000,
    observation: AnalyzeResult | None = None,
) -> dict[str, Any]:
    """Let the warm local policy execute a bounded safe navigation stretch.

        The model still selects only an opaque ID. AUA owns the exact call, revalidates the
        frame/session/phase after inference, executes through the ordinary action method, and
        consumes the folded observation. Any ambiguity, stale result, repeated call, lack of
        screen progress, unsupported action, or exhausted budget returns control to the parent
        agent without replaying a mutation.
        """

    if self._configured_policy_mode() != "advisory":
        raise UsageError(
            "local session autopilot requires policy advisory mode",
            hint=(
                "Set policy.mode=advisory, then restart the daemon. `policy.enabled` is not "
                "required for this command — it governs the passive advice on ordinary "
                "analyze calls, which is where the per-call inference cost is paid."
            ),
            code="policy_autopilot_disabled",
        )
    if not 1 <= max_steps <= 20:
        raise UsageError("max_steps must be between 1 and 20")
    if not 1_000 <= max_duration_ms <= 300_000:
        raise UsageError("max_duration_ms must be between 1000 and 300000")

    # Unlike ordinary advisory metadata, this command can mutate the UI. Require at least
    # one configured provider to authenticate advisory rollout before even the one-candidate
    # deterministic fast path is allowed to act, so a shadow-capped adapter cannot turn
    # configuration drift into execution. The bundled adapter does authenticate advisory, which
    # is why enabling the policy at all stays an explicit operator action rather than a default.
    try:
        policy_chain = self.factory.build_chain("policy")
        rollout_authorized = False
        for provider in policy_chain.providers:
            supports_mode = getattr(provider, "supports_mode", None)
            if callable(supports_mode) and bool(supports_mode("advisory")):
                rollout_authorized = True
                break
    except Exception:
        rollout_authorized = False
    if not rollout_authorized:
        raise UsageError(
            "no configured local policy is authenticated for autopilot execution",
            hint=(
                "Use shadow/advisory output for evaluation, or configure a pinned adapter "
                "whose manifest explicitly authorizes advisory rollout."
            ),
            code="policy_autopilot_unauthorized",
        )

    # Authorised is not the same as able. `supports_mode` reads the adapter's manifest, which
    # says nothing about whether the runtime can load it — so with the optional MLX extras
    # absent this check passed, autopilot started, and every single step found the provider
    # unavailable and handed off. Observed live: 32 of 41 handoffs in one session were nothing
    # but "optional dependency missing", which from the outside is indistinguishable from a
    # slow, useless model. Refusing once with the reason is worth more than a bounded run that
    # cannot possibly act.
    # Able to load is still not the same as able to steer. A provider whose recent output was
    # mostly unparsable cannot drive a bounded run either — measured live at roughly four
    # invalid responses in five for one chain member — and every one of those costs seconds.
    # That verdict belongs here, once, and not as a per-step handoff.
    from . import policy_health

    blocked: list[str] = []
    condemned: list[str] = []
    for provider in policy_chain.providers:
        try:
            provider_name = str(provider.name)
        except Exception:
            provider_name = type(provider).__name__
        unusable = policy_health.unusable_reason(provider_name)
        if unusable:
            condemned.append(f"{provider_name}: {unusable}")
            blocked.append(f"{provider_name}: {unusable}")
            continue
        try:
            availability = provider.is_available()
        except Exception as exc:  # a broken provider must not mask the others
            blocked.append(f"{provider_name}: {type(exc).__name__}: {exc}")
            continue
        if availability.ok:
            break
        blocked.append(f"{provider_name}: {availability.reason}")
    else:
        if condemned and len(condemned) == len(blocked):
            raise UsageError(
                "every configured local policy provider is producing unusable output: "
                + "; ".join(condemned),
                hint=(
                    "This is a broken provider, not a slow one: its recent selections did not "
                    "parse into an offered candidate ID. Fix or replace it in `policy.chain` "
                    "(`aua policy status` shows the per-provider rate), or drive the steps "
                    "yourself. A restarted daemon re-measures from scratch."
                ),
                code="policy_autopilot_unusable",
            )
        raise UsageError(
            "the local policy is configured for autopilot but no provider can run: "
            + "; ".join(blocked or ["no policy providers are configured"]),
            hint=(
                "Run `aua policy status` for the full readiness report. A missing optional "
                "dependency means the model was never installed in the environment running "
                "`aua` — install the extras there (`functiongemma` for the small selector, "
                "`hybrid-policy` for the reviewer). If `aua` is a `uv tool` install, add them "
                "to the tool's own requirements, or the next `uv tool upgrade` will drop them "
                "again."
            ),
            code="policy_autopilot_unavailable",
        )

    from functools import partial

    from .autopilot import plan_waypoints

    self._policy_mode_override = "advisory"
    try:
        started = time.monotonic()
        state = self._session_state(session_id)
        current_observation = observation or self.analyze(no_cache=True)
        self._last_analyze_result = current_observation
        trace: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        completed_waypoints: list[str] = []
        # Waypoints nothing on screen matched. Kept apart from the completed list, which was
        # absorbing them and so reporting navigation the run never performed.
        skipped_waypoints: list[str] = []
        terminal_reason = "handoff"
        detail = "Local navigation could not safely continue."

        for step_number in range(1, max_steps + 1):
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if elapsed_ms >= max_duration_ms:
                terminal_reason = "time_limit"
                detail = "The bounded local-policy time budget expired before another action."
                break

            self.session_progress(
                state.session_id,
                observation=current_observation,
                _include_policy=False,
            )
            state = self._session_state(state.session_id)
            active_phase = next(
                (phase for phase in state.phases if phase.status != "completed"),
                None,
            )
            if active_phase is None:
                terminal_reason = "goal_complete"
                detail = "Every goal phase has fresh deterministic proof."
                break

            # Goal phases are proof checkpoints, not necessarily one navigation action each,
            # but only the phase the run is actually on may supply a destination. Folding
            # every remaining phase into one flat list let autopilot steer toward a phase-3
            # waypoint while the session said phase 1, and report nothing about the jump.
            # `plan_waypoints` owns that decision and names every crossing it allows.
            plan = plan_waypoints(
                state.phases,
                active_phase_id=active_phase.id,
                completed=completed_waypoints,
                skipped=skipped_waypoints,
                waypoints_of=self._policy_navigation_waypoints,
                # Bound to *this* step's frame, never a later one.
                arrived=partial(
                    self._policy_waypoint_arrived,
                    observation=current_observation,
                ),
            )
            # Passive title evidence can advance navigation bookkeeping, but never the
            # session proof checkpoint itself.
            completed_waypoints.extend(plan.arrived_waypoints)
            if not plan.can_steer:
                terminal_reason = plan.blocked_reason or "navigation_complete"
                detail = plan.blocked_detail
                trace.append(
                    {
                        "step": step_number,
                        "active_phase": active_phase.id,
                        "phase": plan.phase_id or active_phase.id,
                        "crossed_phases": list(plan.crossed_phases),
                        "arrived_waypoints": list(plan.arrived_waypoints),
                        "executed": False,
                        "stop_reason": terminal_reason,
                    }
                )
                break
            # Authored waypoints are ordered. If the first is not visible, trying a later
            # one crosses a navigation prerequisite without evidence.
            objectives = list(plan.objectives[:1])
            # The provenance anchor stays the *active* phase — that is where the run is, and
            # `_policy_context_is_current` revalidates it after inference. The plan's phase is
            # reported alongside it so a look-ahead is visible instead of implied.
            waypoint_phase = plan.phase_id or active_phase.id

            policy_result: dict[str, Any] | None = None
            chosen_objective: str | None = None
            step_skipped: list[str] = []
            for objective in objectives:
                candidate_result = self._session_policy_output(
                    state,
                    active_phase,
                    current_observation,
                    # Ordinary response advice is gated by the deterministic phase call. This
                    # explicit execution loop follows a later authored waypoint even while the
                    # prior arrival still awaits proof acknowledgement, so it deliberately uses
                    # the manual-action lane. Candidate compilation and post-inference provenance
                    # checks remain unchanged.
                    recommended_call={"kind": "manual_action"},
                    policy_objective=objective,
                    recent_outcomes=tuple(
                        [f"completed_navigation={item}" for item in completed_waypoints]
                    ),
                    _return_selected=True,
                )
                status = (candidate_result.get("policy") or {}).get("status")
                if candidate_result.get("_selected_policy_call") is not None:
                    policy_result = candidate_result
                    chosen_objective = objective
                    break
                if status == "no_candidate":
                    policy_result = candidate_result
                    chosen_objective = objective
                    break
                policy_result = candidate_result
                chosen_objective = objective
                break

            if policy_result is None or policy_result.get("_selected_policy_call") is None:
                public_policy = self._autopilot_public_policy_output(policy_result or {})
                audit = public_policy.get("policy") or {}
                policy_status = audit.get("status")
                terminal_reason = (
                    "no_guard_approved_candidate"
                    if policy_status in {None, "no_candidate"}
                    else "policy_handoff"
                )
                detail = (
                    "No visible guard-approved tap advances the remaining authored navigation; "
                    "the parent agent must recover, scroll, provide input, or report absence."
                )
                # …unless the screen was never the problem. Output the guard could not
                # resolve to an offered ID is the provider's failure, and reporting it as
                # "nothing on screen advances the goal" sends the reader to inspect the app.
                provider_failure = self._autopilot_provider_failure(audit)
                if provider_failure is not None:
                    terminal_reason, detail = provider_failure
                trace.append(
                    {
                        "step": step_number,
                        "active_phase": active_phase.id,
                        "phase": waypoint_phase,
                        "crossed_phases": list(plan.crossed_phases),
                        "waypoint": chosen_objective,
                        "skipped_waypoints": step_skipped,
                        "executed": False,
                        **public_policy,
                    }
                )
                break

            raw_call = policy_result.pop("_selected_policy_call")
            candidate_id = policy_result.pop("_selected_policy_candidate_id", None)
            call = dict(raw_call) if isinstance(raw_call, dict) else {}
            call_key = json.dumps(
                call, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            )
            if call_key in seen_calls:
                terminal_reason = "repeated_action"
                detail = "The local policy repeated a prior action, so AUA handed off without replay."
                trace.append(
                    {
                        "step": step_number,
                        "active_phase": active_phase.id,
                        "phase": waypoint_phase,
                        "crossed_phases": list(plan.crossed_phases),
                        "skipped_waypoints": step_skipped,
                        "waypoint": chosen_objective,
                        "candidate_id": candidate_id,
                        "call": call,
                        "executed": False,
                        **self._autopilot_public_policy_output(policy_result),
                    }
                )
                break
            seen_calls.add(call_key)

            before_fingerprint = current_observation.meta.fingerprint
            action_started = time.monotonic()
            try:
                action_result = self._execute_guarded_policy_call(call)
            except AuaError as exc:
                terminal_reason = "action_rejected"
                detail = str(exc)
                trace.append(
                    {
                        "step": step_number,
                        "active_phase": active_phase.id,
                        "phase": waypoint_phase,
                        "crossed_phases": list(plan.crossed_phases),
                        "skipped_waypoints": step_skipped,
                        "waypoint": chosen_objective,
                        "candidate_id": candidate_id,
                        "call": call,
                        "executed": False,
                        "error": exc.code,
                        **self._autopilot_public_policy_output(policy_result),
                    }
                )
                break
            action_ms = round((time.monotonic() - action_started) * 1000.0, 3)
            observed = action_result.observation
            after_fingerprint = observed.meta.fingerprint if observed is not None else None
            # Local training trace: a decision only becomes useful once its outcome is known.
            from . import policy_trace

            if policy_trace.enabled():
                policy_trace.record_outcome(
                    policy_trace.last_decision_id(),
                    executed=True,
                    verdict="followed" if action_result.ok else "failed",
                    action_ok=action_result.ok,
                    before_fingerprint=before_fingerprint,
                    after_fingerprint=after_fingerprint,
                )
            trace.append(
                {
                    "step": step_number,
                    "active_phase": active_phase.id,
                    "phase": waypoint_phase,
                    "crossed_phases": list(plan.crossed_phases),
                    "waypoint": chosen_objective,
                    "skipped_waypoints": step_skipped,
                    "candidate_id": candidate_id,
                    "call": call,
                    "executed": True,
                    "action_ok": action_result.ok,
                    "action_duration_ms": action_ms,
                    "before_fingerprint": before_fingerprint,
                    "after_fingerprint": after_fingerprint,
                    **self._autopilot_public_policy_output(policy_result),
                }
            )
            if (
                not action_result.ok
                or observed is None
                or not after_fingerprint
                or bool(observed.meta.stale_risk)
            ):
                terminal_reason = "outcome_unknown"
                detail = (
                    "The selected action lacks a fresh trustworthy folded observation; "
                    "AUA will not repeat it."
                )
                if observed is not None:
                    current_observation = observed
                break
            current_observation = observed
            if after_fingerprint == before_fingerprint:
                terminal_reason = "no_progress"
                detail = (
                    "The selected action did not change the observed frame, so AUA stopped."
                )
                break
            if chosen_objective:
                arrived = self._policy_waypoint_arrived(chosen_objective, observed)
                trace[-1]["waypoint_arrived"] = arrived
                if not arrived:
                    terminal_reason = "waypoint_unverified"
                    detail = (
                        "The frame changed, but its passive title does not exactly prove "
                        f"arrival at {chosen_objective!r}; AUA handed off without marking it complete."
                    )
                    break
                completed_waypoints.append(chosen_objective)
        else:
            terminal_reason = "step_limit"
            detail = "The bounded local-policy step limit was reached."

        final_progress = self.session_progress(
            state.session_id,
            observation=current_observation,
            _include_policy=False,
        ).get("goal_progress")
        return {
            # A broken model is a failed command, not a clean handoff: the caller asked
            # autopilot to drive and it could not, so this must not read as success.
            "ok": terminal_reason
            not in {
                "action_rejected",
                "outcome_unknown",
                "provider_output_unusable",
                "provider_unavailable",
            },
            "autopilot": {
                "executed_by": "aua_daemon_local_policy",
                "terminal_reason": terminal_reason,
                "detail": detail,
                "steps_executed": sum(1 for item in trace if item.get("executed") is True),
                "max_steps": max_steps,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "completed_waypoints": completed_waypoints,
                "skipped_waypoints": skipped_waypoints,
                "trace": trace,
                "handoff_required": terminal_reason != "goal_complete",
            },
            "goal_progress": final_progress,
            "observation": current_observation.model_dump(mode="json"),
        }

    finally:
        self._policy_mode_override = None
