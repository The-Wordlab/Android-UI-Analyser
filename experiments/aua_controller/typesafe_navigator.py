"""A System One navigator that answers the easy taps and declines everything else.

The controller asks a chat model for the next tool call, which is the dominant cost and
latency of a run: tens of requests against the judge's two. Most of those requests decide
something narrow — which of the controls on this screen moves toward the goal — and that is a
Choice over ids AUA already enumerates. This plugs into ``run_agent``'s ``host_next`` seam, so
returning ``None`` simply hands the step back to the chat model and nothing else changes.

It declines far more than it answers, on purpose. Measured over 120 saved real steps, the
unfiltered pick matched the chat controller's 42% of the time — useless alone — but the model's
own confidence separates those cases sharply (mean 0.80 when right against 0.46 when wrong), so
at a 0.80 gate it answers about 30% of steps at about 90% fidelity and passes the rest on.

Two refusals are structural rather than tuned:

* **Only taps.** The same measurement showed the worst confusions were about *stopping*:
  15 steps where the controller finished and this model wanted to tap, and 10 where the
  controller went back and this model wanted to finish. Ending a run early, or missing that it
  ended, corrupts the verdict rather than costing a step. Stopping, going back, scrolling and
  typing all stay with the chat model.
* **No text.** A System One model generates nothing, so a step that needs a typed string is not
  one it can answer even in principle.

A proposal is an opinion with no authority beyond the tools it was offered. ``shadow`` records
what it would have done and returns ``None`` every time, which is how a run proves the gate on
real screens before anything depends on it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

MODEL = "jev-latest"
MIN_CONFIDENCE = 0.80
MAX_OPTIONS = 60  # a Choice takes up to 255 options; a screen offering more is not a decision
TAP_TOOL = "tap_and_analyze"

# Offered so the model can say "none of these", which is what a low-confidence tap looks like
# before it is thrown away. Only `tap` is ever acted on.
ACTION_KINDS: dict[str, str] = {
    "tap": "Press a control that is visible on this screen now",
    "type": "Type text into a field on this screen",
    "scroll": "What is needed is not on screen; scroll to reveal more",
    "back": "This screen is wrong or finished; go back",
    "done": "The goal is already satisfied; stop",
}


def candidates(observation: Mapping[str, Any] | None, *, limit: int = MAX_OPTIONS) -> dict[str, str]:
    """Interactive ids on this screen, labelled the way a person would read them."""
    if not isinstance(observation, Mapping):
        return {}
    options: dict[str, str] = {}
    for element in observation.get("elements") or []:
        if not isinstance(element, Mapping):
            continue
        handle = element.get("id")
        if not isinstance(handle, str) or not handle:
            continue
        if not (element.get("clickable") is True or element.get("editable") is True
                or "checked" in element):
            continue
        label = next((element[key] for key in ("text", "desc", "content_desc", "resource_id", "rid")
                      if isinstance(element.get(key), str) and element[key].strip()), handle)
        if "checked" in element:
            label = f"{label} [switch is {'ON' if element['checked'] else 'OFF'}]"
        options[handle] = label[:90]
        if len(options) >= limit:
            break
    return options


def tool_names(tools: Sequence[Any]) -> set[str]:
    """Accept the offered tools however the caller holds them.

    The controller carries OpenAI-shaped entries, where the name sits under ``function``.
    A first cut read the top level, found nothing, and silently declined every step of a live
    run -- which is exactly the failure shadow mode exists to catch.
    """
    names: set[str] = set()
    for tool in tools or ():
        if isinstance(tool, str) and tool:
            names.add(tool)
        elif isinstance(tool, Mapping):
            function = tool.get("function")
            name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def build_questions(options: Mapping[str, str]) -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul

    return {
        "action": Choice(instructions="What is the single best next action to reach the goal?",
                         criteria=dict(ACTION_KINDS)),
        "target": Choice(instructions="Which control should that action operate on?",
                         criteria=dict(options)),
        "settled": Noul(instructions="Is the goal already fully satisfied on this screen, "
                                     "with nothing further to do?"),
    }


class TypeSafeNavigator:
    """Propose a confident tap, or decline and let the chat model take the step."""

    def __init__(
        self,
        goal: str,
        *,
        client: Any = None,
        tools: Sequence[str] = (),
        model: str = MODEL,
        min_confidence: float = MIN_CONFIDENCE,
        shadow: bool = False,
        timeout_s: float = 10.0,
    ) -> None:
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must sit in (0, 1]")
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            client = AsyncTypeSafeClient()
        self.client = client
        self.goal = goal
        self.model = model
        self.min_confidence = min_confidence
        self.shadow = shadow
        self.timeout_s = timeout_s
        # A tap it was never offered is not a proposal this navigator may make.
        offered = tool_names(tools)
        self.can_tap = not offered or TAP_TOOL in offered
        self.history: list[str] = []
        self.proposals: list[dict[str, Any]] = []
        self.declined: dict[str, int] = {}
        self.requests = 0
        self.input_tokens = 0
        self.request_ms: list[float] = []

    def _decline(self, why: str) -> None:
        self.declined[why] = self.declined.get(why, 0) + 1

    def observed(self, tool: str) -> None:
        """Record what the run actually did, whoever chose it."""
        self.history.append(tool)

    async def __call__(self, result: Any) -> dict[str, Any] | None:
        from experiments.aua_controller.compaction import compact_frame

        if not self.can_tap:
            self._decline("tap_not_offered")
            return None
        compact = compact_frame(result, keep_ids=True)
        observation = compact.get("observation") if isinstance(compact, Mapping) else None
        options = candidates(observation)
        if len(options) < 2:
            # One control is not a choice, and none is not a screen this can help with.
            self._decline("too_few_controls")
            return None

        state = {"goal": self.goal, "actions_so_far": self.history[-20:], "screen": observation}
        started = time.perf_counter()
        try:
            response = await self.client.system_one(
                state=state, questions=build_questions(options),
                model=self.model, timeout=self.timeout_s,
            )
        except Exception as exc:
            # The chat model is the fallback for every step, so a navigator failure costs a
            # round trip and never a run.
            self._decline(f"request_failed:{type(exc).__name__}")
            return None
        self.requests += 1
        self.request_ms.append((time.perf_counter() - started) * 1000)
        self.input_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)

        action, target = response.answers["action"], response.answers["target"]
        record = {
            "kind": action.choice, "kind_confidence": round(action.confidence, 4),
            "target": target.choice, "target_confidence": round(target.confidence, 4),
            "settled": round(response.answers["settled"].noul, 4),
            "options": len(options),
        }
        if action.choice != "tap":
            # Everything that ends, rewinds, scrolls or types a run stays with the chat model.
            self._decline(f"kind:{action.choice}")
            record["accepted"] = False
            self.proposals.append(record)
            return None
        gate = min(action.confidence, target.confidence)
        if gate < self.min_confidence or target.choice not in options:
            self._decline("below_confidence" if gate < self.min_confidence else "unknown_target")
            record["accepted"] = False
            self.proposals.append(record)
            return None

        record["accepted"] = not self.shadow
        self.proposals.append(record)
        if self.shadow:
            self._decline("shadow")
            return None
        return {
            "tool": TAP_TOOL,
            "arguments": {"id": target.choice},
            "reason": (f"System One tap at confidence {gate:.2f}: {options[target.choice]}"),
        }

    def report(self) -> dict[str, Any]:
        accepted = sum(1 for item in self.proposals if item.get("accepted"))
        return {
            "engine": "typesafe_system_one", "model": self.model, "shadow": self.shadow,
            "min_confidence": self.min_confidence, "requests": self.requests,
            "input_tokens": self.input_tokens, "proposals": len(self.proposals),
            "accepted": accepted, "declined": dict(self.declined),
            "mean_request_ms": (round(sum(self.request_ms) / len(self.request_ms), 1)
                                if self.request_ms else None),
            # A shadow run is only worth taking if you can read back what it would have done,
            # step by step, against what the controller actually chose.
            "proposals_detail": self.proposals[:64],
        }


__all__ = ["TypeSafeNavigator", "ACTION_KINDS", "MODEL", "MIN_CONFIDENCE", "TAP_TOOL",
           "build_questions", "candidates", "tool_names"]
