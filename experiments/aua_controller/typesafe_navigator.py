"""A System One navigator that answers the easy taps and declines everything else.

The controller asks a chat model for the next tool call, which is the dominant cost and
latency of a run: tens of requests against the judge's two. Most of those requests decide
something narrow — which of the controls on this screen moves toward the goal — and that is a
Choice over ids AUA already enumerates. This plugs into ``run_agent``'s ``host_next`` seam, so
returning ``None`` simply hands the step back to the chat model and nothing else changes.

It declines far more than it answers, on purpose. Measured over 120 saved real steps, the
unfiltered pick matched the chat controller's 42% of the time — useless alone — but the model's
own confidence separates those cases sharply (mean 0.80 when right against 0.46 when wrong).
Counting every step it would actually take, including the ones whose right answer was not a tap
at all:

===========  ========  =========
gate         coverage  correct
===========  ========  =========
0.80         18%       77%
**0.85**     **12%**   **93%**
0.90         8%        100%
===========  ========  =========

Hence the 0.85 default. The earlier reading of this that quoted 30% coverage at 90% fidelity
was scoring only steps that were already taps, so it never counted wanting to tap when the run
should have scrolled, typed or stopped; those are the misses that matter.

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

import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

MODEL = "jev-latest"
MIN_CONFIDENCE = 0.85  # measured: 0.80 gives 77% correct, 0.85 gives 93%
MAX_JOURNEY_CHARS = 60_000  # state and question share a 32k-token budget
MAX_OPTIONS = 60  # a Choice takes up to 255 options; a screen offering more is not a decision
#: $42 per billion input tokens, output free (typesafe.ai pricing, Sep 2026).
USD_PER_INPUT_TOKEN = 42 / 1e9
TAP_TOOL = "tap_and_analyze"
SCROLL_TOOL = "scroll_and_analyze"
BACK_TOOL = "back_gesture_and_analyze"
FINISH_TOOL = "session_finish"
WAIT_TOOL = "wait_and_analyze"

# Widened action space, after reading how public Jev browser agents are built
# (browser-use/jev-ultrafast): one call returns the operation and every operand it might need, and
# the harness acts on whichever operand the chosen operation names. The extra questions ride the
# same state for free, because a System One request prices the state once and the answers in
# parallel. `type` is still refused here for the same reason their harness hands it to a small
# LLM: a non-generative model cannot write the string.
#: `in_progress` is not one of the harness's finish outcomes and is never passed to one. It is
#: here because the other four describe a *finished* run, and on a step in the middle of one none
#: of them is true -- so the model had to answer something anyway. Measured over 10 saved screens
#: from a real run, it answered `blocked` on 6 of them, with blocked probability between 0.41 and
#: 0.81 while nothing whatsoever was blocking the run. Worse, on the step that finally chose
#: `done`, `blocked` was outscoring `achieved` 0.41 to 0.35 -- one coin flip from recording a
#: working run as blocked. Adding the true option drained it: `in_progress` on 9 of 10 at 0.78 to
#: 1.00, and blocked fell to 0.00-0.05 everywhere.
FINISH_OUTCOMES: dict[str, str] = {
    "in_progress": "The run is still working toward the goal; it is neither finished nor stopped",
    "achieved": "The goal was carried out during this run",
    "already_satisfied": "The goal was already true when the run started; nothing was needed",
    "blocked": "Something outside the goal stops it being carried out",
    "not_achievable": "This app cannot do what the goal asks",
}
ACTION_SPACES = ("taps", "full")

# Offered so the model can say "none of these", which is what a low-confidence tap looks like
# before it is thrown away. `wait` and `blocked` are never acted on and exist only to be chosen:
# jev-1.13 is documented as literal and answers the question as written, so a screen that is
# still loading, or a run that is stuck, had nowhere to go but an outright wrong tap. The public
# browser harnesses carry WAIT and BLOCKED in their action space for the same reason. Giving a
# boundary case its own option is the doc's own advice, and it costs nothing: both decline.
ACTION_KINDS: dict[str, str] = {
    "tap": "Press a control that is visible on this screen now",
    "type": "Type text into a field on this screen",
    "scroll_down": "What is needed is below; scroll down to reveal it",
    "scroll_up": "What is needed is above; scroll up to reveal it",
    "back": "This screen is wrong or finished; go back",
    "done": "The goal is already satisfied; stop",
    "wait": "This screen is still loading or mid-animation; nothing should be pressed yet",
    "blocked": "Something outside the goal stops this run going further",
}
#: Scroll is two actions rather than one action plus a direction question. The public browser
#: harnesses carry SCROLL_UP and SCROLL_DOWN as operations for the same reason it is right here:
#: "which way should this screen be scrolled" is a hop of indirection jev-1.13's own notes warn
#: about, and gating on min(action, direction) mixed the confidences of two separate questions,
#: which those notes also warn about. Replayed over 11 saved screens the merged form chose the
#: same action 11 times out of 11 and asked 2% fewer tokens, so the extra question was buying
#: nothing. What it picks when a scroll is genuinely needed is untested either way.
SCROLL_KINDS = {"scroll_down": "down", "scroll_up": "up"}

#: `blocked` is chosen to be declined: acting on it means ending the run, and ending a run early
#: is this model's worst measured skill. `wait` is not in here because waiting is a real tool the
#: harness already offers -- asking "is this screen still loading?" and then paying a chat model
#: to answer the same question was the option costing a round trip to say nothing.
NON_ACTIONS = ("blocked",)
#: The outcome that means "do not finish". Never a `session_finish` argument.
UNFINISHED = "in_progress"


def where(element: Mapping[str, Any], screen: Mapping[str, Any] | None) -> str:
    """Roughly where a control sits, for the ones the app never named.

    Around 11% of the options handed over carried no text, desc or resource id at all, so the
    label fell back to the element's own 32-character digest -- exactly the opaque value the
    numbering was introduced to keep out of the request. Dropping them is not safe: this class of
    app leaves many genuinely pressable controls unnamed, including the one the run needs. A
    position is something a reader of the screen can actually use.
    """
    bounds = element.get("bounds")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
        return "unlabelled control"
    width = (screen or {}).get("width") or 0
    height = (screen or {}).get("height") or 0
    if not (width and height):
        return "unlabelled control"
    x = (float(bounds[0]) + float(bounds[2])) / 2 / width
    y = (float(bounds[1]) + float(bounds[3])) / 2 / height
    down = "top" if y < 0.33 else ("bottom" if y > 0.66 else "middle")
    across = "left" if x < 0.33 else ("right" if x > 0.66 else "centre")
    return f"unlabelled control, {down} {across} of the screen"


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
                      if isinstance(element.get(key), str) and element[key].strip()),
                     None)
        if label is None:
            label = where(element, observation.get("screen") if isinstance(observation, Mapping) else None)
        if "checked" in element:
            label = f"{label} [switch is {'ON' if element['checked'] else 'OFF'}]"
        options[handle] = label[:90]
        if len(options) >= limit:
            break
    return options


def plain(value: Any) -> Any:
    """Whatever the SDK handed back, as something json.dumps will take."""
    for attribute in ("model_dump", "dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001 - a transcript must never fail a run
                pass
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, Mapping) and fields:
        return {str(k): plain(v) for k, v in fields.items() if not str(k).startswith("_")}
    named = {key: plain(getattr(value, key)) for key in
             ("choice", "confidence", "probabilities", "noul", "expectation", "legend")
             if hasattr(value, key)}
    return named or str(value)


def what_happened(result: Any, moved: bool) -> str:
    """Say what the last action did, in the terms that separate progress from a redraw.

    "screen changed" was a boolean off the fingerprint, so a button losing its label while a
    login was in flight read exactly like arriving somewhere new. Observed live: the tap on the
    sign-in button left the activity unchanged and swapped 2 of 32 controls -- AUA settled it and
    reported a change -- and the navigator, told only that the screen had changed, pressed the
    same button again. The frame already carries what actually happened; it was being thrown away
    by compaction before anyone read it.
    """
    if not moved:
        return "SCREEN DID NOT CHANGE"
    change = result.get("change") if isinstance(result, Mapping) else None
    diff = result.get("action_diff_summary") if isinstance(result, Mapping) else None
    if isinstance(change, Mapping) and change.get("activity_changed") is True:
        return "a different screen opened"
    if isinstance(diff, Mapping):
        # A control swapped for another counts once, not twice: it is one slot that differs.
        moved_count = max(int(diff.get("added") or 0), int(diff.get("removed") or 0))
        total = int(diff.get("curr_count") or 0)
        if total and moved_count * 4 <= total:
            # Same screen, a handful of controls redrawn: a spinner, a label, a disabled button.
            return (f"the SAME screen redrew -- only {moved_count} of {total} controls changed, "
                    "which usually means it is still working on the last action")
    return "screen changed"


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


def numbered(options: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Present the controls as a small numbered menu, not as their internal ids.

    AUA ids are 32-character hex digests. jev-1.13 is documented to do worse on numeric and
    opaque representations than on semantic ones, and the public browser harnesses hand it an
    indexed element table (`[1] button Settings`) rather than a DOM handle. So the Choice is
    over "1", "2", "3" described by what a person would read, and code maps the answer back.

    Returns ``(criteria, by_index)``: what the model is asked, and how to undo the numbering.
    """
    criteria: dict[str, str] = {}
    by_index: dict[str, str] = {}
    for position, (handle, label) in enumerate(options.items(), start=1):
        criteria[str(position)] = label
        by_index[str(position)] = handle
    return criteria, by_index


def build_questions(options: Mapping[str, str], *, action_space: str = "taps") -> dict[str, Any]:
    """The action, plus one operand question per action that takes one.

    Every operand is asked speculatively, on the same state, in the same request -- the operands
    belonging to actions that lose cost nothing and are simply discarded. That is the whole
    economy of a System One call and it is how the public browser harnesses use it.
    """
    from typesafe_sdk import Choice, Noul

    questions = {
        "action": Choice(instructions="What is the single best next action to reach the goal?",
                         criteria=dict(ACTION_KINDS)),
        "target": Choice(instructions="Which control should that action operate on?",
                         criteria=numbered(options)[0]),
        "settled": Noul(instructions="Is the goal already fully satisfied on this screen, "
                                     "with nothing further to do?"),
    }
    if action_space == "full":
        # Asked directly, not as "if the run stopped here...": a hypothetical is a hop of
        # indirection, and jev-1.13's documented jaggedness names indirection as a cost.
        questions["outcome"] = Choice(
            instructions="What has become of the goal on this screen?",
            criteria=dict(FINISH_OUTCOMES))
    return questions


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
        action_space: str = "taps",
        shadow: bool = False,
        timeout_s: float = 10.0,
        transcript_path: Any = None,
    ) -> None:
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must sit in (0, 1]")
        if action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}")
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            client = AsyncTypeSafeClient()
        self.client = client
        self.goal = goal
        self.model = model
        self.min_confidence = min_confidence
        self.action_space = action_space
        self.shadow = shadow
        self.timeout_s = timeout_s
        # Everything sent and everything returned, one JSON object per call. The report keeps
        # only the decision; this keeps the evidence, which is what an audit or a write-up needs.
        self.transcript_path = pathlib.Path(transcript_path) if transcript_path else None
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.usd = 0.0
        # A tap it was never offered is not a proposal this navigator may make.
        offered = tool_names(tools)
        self.offered = offered
        self.can_tap = not offered or TAP_TOOL in offered
        self.history: list[str] = []
        # (screen fingerprint, target) pairs already proposed. A System One model answers each
        # screen from scratch with no memory of the last one, so on a screen that did not change
        # it confidently repeats the tap that failed to change it -- observed live as a 20-step
        # loop that burned a whole run's budget. The chat model carries the transcript and can
        # see that, so a repeat is its problem, not this one's.
        self._seen: set[tuple[str, str]] = set()
        # The journey so far, in the state rather than in the model. A System One model keeps
        # nothing between calls, but it does not need to: the whole run fits in one request.
        # Sending only tool names measured 33% target accuracy against 41% for the journey, and
        # the 0.90 gate went from 83% correct to 100%. It is also what lets it see that a screen
        # did not change, which is the loop it otherwise walks straight into.
        self._journey: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._last_fingerprint: str | None = None
        self.proposals: list[dict[str, Any]] = []
        self.declined: dict[str, int] = {}
        self.requests = 0
        self._why: str | None = None
        self.input_tokens = 0
        self.request_ms: list[float] = []

    def _decline(self, why: str) -> None:
        self.declined[why] = self.declined.get(why, 0) + 1
        # The last reason given, so the proposal and the transcript can both name it. A run that
        # hands a step to the chat model without saying why is unreadable afterwards.
        self._why = why

    def observed(self, tool: str) -> None:
        """Record what the run actually did, whoever chose it."""
        self.history.append(tool)
        if self._pending is not None:
            self._pending["you_chose"] = f"{tool} on '{self._pending.pop('_label', '?')}'"

    async def __call__(self, result: Any) -> dict[str, Any] | None:
        from experiments.aua_controller.compaction import compact_frame

        if not self.can_tap:
            self._decline("tap_not_offered")
            return None
        self._why = None
        compact = compact_frame(result, keep_ids=True)
        observation = compact.get("observation") if isinstance(compact, Mapping) else None
        meta = observation.get("meta") if isinstance(observation, Mapping) else None
        fingerprint = meta.get("fingerprint") if isinstance(meta, Mapping) else None
        options = candidates(observation)
        if len(options) < 2:
            # One control is not a choice, and none is not a screen this can help with.
            self._decline("too_few_controls")
            return None

        # Close the previous turn now that its result is on screen: whether the chosen action
        # moved anything is the single most useful fact about it.
        if self._pending is not None:
            self._pending["what_happened"] = what_happened(
                result, moved=fingerprint != self._last_fingerprint)
            self._journey.append(self._pending)
            self._pending = None
        self._last_fingerprint = fingerprint if isinstance(fingerprint, str) else None

        journey = list(self._journey)
        # State plus the longest question share a 32k-token budget, so the oldest turns go
        # first; a loop is made of the recent ones.
        while len(json.dumps(journey, default=str)) > MAX_JOURNEY_CHARS and journey:
            journey.pop(0)
        state = {"goal": self.goal, "journey_so_far": journey,
                 "this_is_the_new_screen": observation}
        started = time.perf_counter()
        try:
            response = await self.client.system_one(
                state=state, questions=build_questions(options, action_space=self.action_space),
                model=self.model, timeout=self.timeout_s,
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

        # Held, not written: the useful half of a turn is what the harness decided to do with the
        # answer, and that has not happened yet. It is written once, below, verdict included.
        turn = {
            "call": self.requests,
            "request_ms": round(elapsed_ms, 1),
            "input_tokens": tokens,
            "usd": round(tokens * USD_PER_INPUT_TOKEN, 9),
            # The request body verbatim, in the shape the SDK puts on the wire: state, model and
            # the questions as they serialise. Anything trimmed or prettified here is a step a
            # reader cannot check, which defeats the point of keeping it.
            "request": {
                "model": self.model,
                "state": state,
                "questions": {name: plain(question)
                              for name, question in build_questions(
                                  options, action_space=self.action_space).items()},
            },
            "response": plain(response),
            "menu": numbered(options)[0],
        }
        action, target = response.answers["action"], response.answers["target"]
        record = {
            "kind": action.choice, "kind_confidence": round(action.confidence, 4),
            "target": target.choice, "target_confidence": round(target.confidence, 4),
            "target_id": numbered(options)[1].get(target.choice),
            "settled": round(response.answers["settled"].noul, 4),
            "options": len(options),
        }
        def settle(accepted: bool) -> None:
            record["accepted"] = accepted
            if not accepted:
                record["declined_because"] = self._why
            self.proposals.append(record)
            turn["verdict"] = dict(record)
            self._record(turn)

        plan = self._plan(action, target, response.answers, options, numbered(options)[1])
        if plan is None:
            settle(False)
            return None
        tool, arguments, operand, operand_confidence, label = plan
        record["tool"] = tool
        # Name the operand the gate actually read. `target_confidence` belongs to the tap
        # question and says nothing about a scroll, a back or a finish -- printing it beside a
        # gate computed from a different answer made the log contradict itself.
        record["operand"] = operand
        record["operand_confidence"] = round(operand_confidence, 4)

        gate = min(action.confidence, operand_confidence)
        record["gate"] = round(gate, 4)
        record["gate_needed"] = self.min_confidence
        if gate < self.min_confidence:
            self._decline("below_confidence")
            settle(False)
            return None

        # Repeating an action on a screen that action already failed to move is the one loop a
        # model reading each screen from scratch walks straight into, whatever the action is.
        pair = (str(fingerprint), f"{action.choice}:{operand}")
        if fingerprint is not None and pair in self._seen:
            self._decline("repeat_on_unchanged_screen")
            settle(False)
            return None
        if self.shadow:
            self._decline("shadow")
        settle(not self.shadow)
        if fingerprint is not None:
            self._seen.add(pair)
        self._pending = {"n": len(self._journey) + 1,
                         "screen_you_saw": list(options.values())[:40],
                         "_label": label}
        if self.shadow:
            self._decline("shadow")
            return None
        return {"tool": tool, "arguments": arguments,
                "reason": f"System One {action.choice} at confidence {gate:.2f}: {label}"}

    def _plan(self, action, target, answers, options, by_index):
        """Bind the chosen action to an offered tool, or decline and say why.

        Returns ``(tool, arguments, operand, operand_confidence, label)``. The operand is what the
        repeat guard keys on and what the confidence gate reads, so an action with no operand --
        going back -- is gated on the action choice alone.
        """
        kind = action.choice
        if kind in NON_ACTIONS:
            # Chosen on purpose, declined on purpose: these exist so a loading screen or a stuck
            # run has somewhere to go other than a confidently wrong tap.
            self._decline(f"kind:{kind}")
            return None
        if kind == "tap":
            handle = by_index.get(target.choice)
            if handle is None:
                self._decline("unknown_target")
                return None
            return (TAP_TOOL, {"id": handle}, handle, target.confidence, options[handle])
        # Everything below exists only in the widened space. Keeping the narrow default is what
        # lets the two be compared on the same code.
        if self.action_space != "full":
            self._decline(f"kind:{kind}")
            return None
        if kind == "type":
            # A System One model returns a choice, never a string; the public harnesses call a
            # small generative model here and so does this one, by handing the step back.
            self._decline("kind:type")
            return None
        if kind in SCROLL_KINDS:
            if SCROLL_TOOL not in self.offered:
                self._decline("scroll_not_offered")
                return None
            direction = SCROLL_KINDS[kind]
            # The direction is the action, so there is no second answer to gate on.
            return (SCROLL_TOOL, {"direction": direction}, direction,
                    action.confidence, f"scroll {direction}")
        if kind == "wait":
            if WAIT_TOOL not in self.offered:
                self._decline("wait_not_offered")
                return None
            # The repeat guard bounds this without a counter: a wait that leaves the screen
            # identical is the same (fingerprint, action) pair, so the second one is refused and
            # the step goes to the chat model. One wait per screen, then escalate.
            return (WAIT_TOOL, {"idle": True}, "idle", action.confidence, "wait for the screen")
        if kind == "back":
            if BACK_TOOL not in self.offered:
                self._decline("back_not_offered")
                return None
            return (BACK_TOOL, {}, "back", action.confidence, "go back")
        if kind == "done":
            if FINISH_TOOL not in self.offered:
                self._decline("finish_not_offered")
                return None
            outcome = answers.get("outcome")
            if outcome is None:
                self._decline("no_outcome")
                return None
            if outcome.choice == UNFINISHED:
                # It asked to stop and said the goal is not finished. Those cannot both be acted
                # on, and the contradiction is exactly the kind of step the chat model should own.
                self._decline("done_but_unfinished")
                return None
            # No note: it is free text, and a fabricated one would reach the judge as evidence.
            return (FINISH_TOOL, {"outcome": outcome.choice}, outcome.choice,
                    outcome.confidence, f"finish as {outcome.choice}")
        self._decline(f"kind:{kind}")
        return None

    def _record(self, entry: dict[str, Any]) -> None:
        if self.transcript_path is None:
            return
        try:
            with self.transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            # A transcript is evidence, never a dependency: losing it costs a write-up, not a run.
            pass

    def report(self) -> dict[str, Any]:
        accepted = sum(1 for item in self.proposals if item.get("accepted"))
        return {
            "engine": "typesafe_system_one", "model": self.model, "shadow": self.shadow,
            "action_space": self.action_space,
            "min_confidence": self.min_confidence, "requests": self.requests,
            "input_tokens": self.input_tokens, "usd": round(self.usd, 8),
            "transcript": str(self.transcript_path) if self.transcript_path else None,
            "proposals": len(self.proposals),
            "accepted": accepted, "declined": dict(self.declined),
            "mean_request_ms": (round(sum(self.request_ms) / len(self.request_ms), 1)
                                if self.request_ms else None),
            # A shadow run is only worth taking if you can read back what it would have done,
            # step by step, against what the controller actually chose.
            "proposals_detail": self.proposals[:64],
        }


__all__ = ["TypeSafeNavigator", "ACTION_KINDS", "ACTION_SPACES", "NON_ACTIONS", "numbered",
           "MODEL", "MIN_CONFIDENCE",
           "TAP_TOOL", "SCROLL_TOOL", "BACK_TOOL", "FINISH_TOOL", "WAIT_TOOL", "SCROLL_KINDS",
           "FINISH_OUTCOMES", "UNFINISHED", "build_questions", "what_happened", "candidates", "tool_names"]
