"""Model-backed chooser for the host-only FunctionGemma AUA experiment.

The model is intentionally limited to one function: it selects an opaque ID
from calls that deterministic AUA code has already constructed and annotated.
It never authors or executes an Android action.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.functiongemma.closed_loop import DecisionContext
from experiments.functiongemma.curriculum import SELECT_CANDIDATE_TOOL
from experiments.functiongemma.evaluate import (
    STRICT_CALL,
    _validate_adapter,
    _validate_tokenizer,
)

ACTIVATION_PREFIX = "You are a model that can do function calling with the following functions."
SELECTOR_POLICY = (
    f"{ACTIVATION_PREFIX} "
    "You are an AUA policy selector for Android UI testing. Select exactly one "
    "supplied candidate. Candidate IDs are opaque and their order is arbitrary. "
    "Prefer direct semantic proof, current observations, bounded waits, and "
    "required cleanup. Never invent or rewrite a call."
)
SELECTION_REQUEST = "Choose the next action that advances the goal with direct proof."
INVALID_CANDIDATE_ID = -1


class SelectionProtocolError(ValueError):
    """Raised when model output is not exactly one valid selector call."""


@dataclass(frozen=True)
class ModelDecision:
    """Auditable result of one model invocation."""

    phase: str
    candidate_ids: tuple[int, ...]
    selected_candidate_id: int | None
    raw_output: str
    valid_protocol: bool
    candidate_exists: bool
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate_ids"] = list(self.candidate_ids)
        return value


def serialize_context(context: DecisionContext) -> dict[str, Any]:
    """Convert live simulator state to the curriculum's prompt vocabulary.

    The closed-loop simulator deliberately exposes no oracle label.  This
    adapter preserves every exposed state value, while expressing observations,
    recent outcomes, and constraints with the same top-level keys used by the
    synthetic training corpus.
    """

    state = dict(context.state)
    observed_screen = state.get("observed_screen")
    outcome = state.get("outcome", "known")
    observation: dict[str, Any] = {"fresh": outcome != "unknown"}
    if observed_screen is not None:
        observation["known_screen"] = observed_screen

    recent_outcomes = [
        f"session_active={str(bool(state.get('session_active'))).lower()}",
        f"network={state.get('network', 'unknown')}",
        f"outcome={outcome}",
        (
            "goal_checkpoint_reached=true"
            if state.get("goal_checkpoint_reached")
            else "goal_checkpoint_reached=false"
        ),
    ]
    constraints = ["Select exactly one supplied candidate", "Require direct proof"]
    if state.get("cleanup_required"):
        constraints.append("Complete required cleanup before finishing")
    if outcome == "unknown":
        constraints.append("Observe before replaying a mutation with an unknown outcome")

    return {
        "fixture_ref": "fixture-closed-loop-0001",
        "request": SELECTION_REQUEST,
        "goal": context.goal,
        "phase": context.phase,
        "observation": observation,
        "recent_outcomes": recent_outcomes,
        "constraints": constraints,
        "candidates": [candidate.as_prompt_value() for candidate in context.candidates],
    }


def policy_messages(context: DecisionContext) -> list[dict[str, Any]]:
    """Build the exact two-turn selector prompt used before target generation."""

    return [
        {"role": "developer", "content": SELECTOR_POLICY},
        {
            "role": "user",
            "content": json.dumps(
                serialize_context(context),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    ]


def policy_tools() -> list[dict[str, Any]]:
    """Return an isolated copy of the experiment's sole function definition."""

    return [copy.deepcopy(SELECT_CANDIDATE_TOOL)]


def parse_candidate_id(output: str, tokenizer: Any, tools: list[dict[str, Any]]) -> int:
    """Parse one canonical FunctionGemma selector call and reject all extra output."""

    if not isinstance(output, str):
        raise SelectionProtocolError(f"model output is not text: {type(output).__name__}")
    strict_match = STRICT_CALL.fullmatch(output)
    if strict_match is None:
        raise SelectionProtocolError("output is not exactly one canonical FunctionGemma call")
    try:
        parsed = tokenizer.tool_parser(output, tools)
    except Exception as exc:
        raise SelectionProtocolError(f"FunctionGemma tool parser rejected output: {exc}") from exc
    if not isinstance(parsed, Mapping) or parsed.get("name") != "select_candidate":
        name = parsed.get("name") if isinstance(parsed, Mapping) else None
        raise SelectionProtocolError(f"unexpected function {name!r}")
    arguments = parsed.get("arguments")
    candidate_id = arguments.get("candidate_id") if isinstance(arguments, Mapping) else None
    if type(candidate_id) is not int:
        raise SelectionProtocolError(f"candidate_id is not an integer: {candidate_id!r}")
    if candidate_id != int(strict_match.group(1)):
        raise SelectionProtocolError("protocol parser and strict call disagree")
    return candidate_id


class FunctionGemmaChooser:
    """Lazy, greedy MLX implementation of the closed-loop ``Chooser`` protocol."""

    def __init__(
        self,
        model: str,
        *,
        adapter: Path | None = None,
        max_tokens: int = 48,
        model_loader: Callable[..., tuple[Any, Any]] | None = None,
        generator: Callable[..., str] | None = None,
        sampler_factory: Callable[..., Any] | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model_name = model
        self.adapter = adapter
        self.max_tokens = max_tokens
        self._model_loader = model_loader
        self._generator = generator
        self._sampler_factory = sampler_factory
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._sampler: Any | None = None
        self.adapter_provenance: dict[str, Any] | None = None
        self.decisions: list[ModelDecision] = []

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        if self._model_loader is None or self._generator is None or self._sampler_factory is None:
            # MLX is an experiment dependency, not an import-time dependency of
            # the repository or its ordinary unit-test environment.
            from mlx_lm import load
            from mlx_lm.generate import generate
            from mlx_lm.sample_utils import make_sampler

            self._model_loader = self._model_loader or load
            self._generator = self._generator or generate
            self._sampler_factory = self._sampler_factory or make_sampler

        self.adapter_provenance = _validate_adapter(self.model_name, self.adapter)
        self._model, self._tokenizer = self._model_loader(
            self.model_name,
            adapter_path=str(self.adapter) if self.adapter else None,
        )
        if self._tokenizer.tool_call_end:
            self._tokenizer.add_eos_token(self._tokenizer.tool_call_end)
        _validate_tokenizer(self._tokenizer)
        self._sampler = self._sampler_factory(temp=0.0)
        return self._model, self._tokenizer

    def select(self, context: DecisionContext) -> ModelDecision:
        """Generate and validate one decision without executing its underlying call."""

        model, tokenizer = self._ensure_loaded()
        tools = policy_tools()
        prompt = tokenizer.apply_chat_template(
            policy_messages(context),
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        assert self._generator is not None
        output = self._generator(
            model,
            tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        candidate_ids = tuple(candidate.id for candidate in context.candidates)
        try:
            selected_id = parse_candidate_id(output, tokenizer, tools)
            exists = selected_id in candidate_ids
            decision = ModelDecision(
                phase=context.phase,
                candidate_ids=candidate_ids,
                selected_candidate_id=selected_id,
                raw_output=output,
                valid_protocol=True,
                candidate_exists=exists,
                error=None if exists else f"candidate {selected_id} was not supplied",
            )
        except SelectionProtocolError as exc:
            decision = ModelDecision(
                phase=context.phase,
                candidate_ids=candidate_ids,
                selected_candidate_id=None,
                raw_output=output,
                valid_protocol=False,
                candidate_exists=False,
                error=str(exc),
            )
        self.decisions.append(decision)
        return decision

    def __call__(self, context: DecisionContext) -> int:
        decision = self.select(context)
        return (
            decision.selected_candidate_id
            if decision.selected_candidate_id is not None
            else INVALID_CANDIDATE_ID
        )
