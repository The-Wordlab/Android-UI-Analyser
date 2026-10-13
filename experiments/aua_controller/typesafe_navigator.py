"""TypeSafe/Jev transport for the shared bounded navigator (explicit legacy option)."""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from experiments.aua_controller.bounded_navigator import (  # noqa: F401
    ACTION_SPACES,
    BACK_TOOL,
    FIELD_SUFFIX,
    INPUT_TOOL,
    MAX_JOURNEY_CHARS,
    MIN_CONFIDENCE,
    OPERATIONS,
    SCROLL_KINDS,
    SCROLL_TOOL,
    TAP_TOOL,
    WAIT_TOOL,
    BoundedNavigator,
    build_questions,
    candidates,
    numbered,
    plain,
    quoted_texts,
    screen_for_model,
    sketch,
    tool_names,
    where,
)
from experiments.aua_controller.typesafe_cost import USD_PER_INPUT_TOKEN, client_options

MODEL = "jev-latest"


class TypeSafeNavigator(BoundedNavigator):
    engine = "typesafe_system_one"
    provider = "typesafe"
    def __init__(self, goal: str, *, client: Any = None, model: str = MODEL, **kwargs: Any):
        super().__init__(goal, model=model, **kwargs)
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient
            client = AsyncTypeSafeClient(**client_options())
        self.client = client

    async def _ask(self, compact: Mapping[str, Any], questions: Mapping[str, Any]):
        """One request; ``None`` when it failed, else the transcript turn and every answer."""
        from typesafe_sdk import Choice

        questions = {name: Choice(instructions=q.instructions, criteria=dict(q.criteria))
                     for name, q in questions.items()}
        state = self._state(compact)
        started = time.perf_counter()
        try:
            response = await self.client.system_one(
                state=state, questions=questions, model=self.model, timeout=self.timeout_s,
            )
        except Exception as exc:
            # The chat model is the fallback for every step, so a navigator failure costs a
            # round trip and never a run.
            self._decline(f"request_failed:{type(exc).__name__}")
            return None
        self.requests += 1
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.request_ms.append(elapsed_ms)
        tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        self.input_tokens += tokens
        self.usd += tokens * USD_PER_INPUT_TOKEN
        turn = {
            "call": self.requests,
            "request_ms": round(elapsed_ms, 1),
            "input_tokens": tokens,
            "usd": round(tokens * USD_PER_INPUT_TOKEN, 9),
            # The request body verbatim, in the shape the SDK puts on the wire. Anything trimmed
            # or prettified here is a step a reader cannot check.
            "request": {"model": self.model, "state": state,
                        "questions": {n: plain(q) for n, q in questions.items()}},
            "response": plain(response),
            "menu": numbered(self._options)[0],
        }
        return turn, response.answers

    def report(self) -> dict[str, Any]:
        report = super().report()
        report.update(engine="typesafe_system_one", provider="typesafe")
        return report
