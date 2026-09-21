"""A System One navigator: it answers the steps it is sure of and declines the rest.

The controller asks a chat model for the next tool call, which is the dominant call count of a
run: tens of requests against the judge's two. Most of them decide something narrow — which of
the controls on this screen moves toward the goal — and that is a Choice over controls the
harness already enumerates. This plugs into ``run_agent``'s ``host_next`` seam, so returning
``None`` simply hands the step back to the chat model and nothing else changes.

It declines far more than it answers, on purpose, and two of those refusals are structural
rather than tuned:

* **No text.** A System One model generates nothing, so a step needing a typed string is not one
  it can answer even in principle. The public browser harnesses call a small generative model at
  exactly this point.
* **It never ends a run.** ``blocked`` is offered so a stuck run has somewhere to put the truth,
  but acting on it is refused: the worst measured confusions were about stopping, and ending a
  run early corrupts the verdict rather than costing a step.

**The gate default is not currently backed by a measurement of this code.** 0.85 came from 120
saved steps scored against a request that no longer exists: five action options rather than
eight, element digests as the Choice keys, a history of bare tool names, no ``what_happened``,
and a journey that omitted every step the chat model took. Adding options redistributes
probability mass and lowers the maximum, so the old coverage/fidelity table is not transferable
and has been removed rather than left to be quoted. Re-measure before relying on a number.

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
MIN_CONFIDENCE = 0.85  # not re-measured against the current questions; see the module docstring
#: A ceiling on the journey, not a documented API limit -- the SDK publishes none. It exists
#: because this model is documented to lose accuracy as the state fills with material that is not
#: about the decision, and a run's journey grows every step.
MAX_JOURNEY_CHARS = 30_000
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
    "already_satisfied": "No step was ever needed; it was true before the run began",
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
    "back": "This is not the screen the goal needs; the previous screen was closer",
    "done": "The goal has been reached; nothing further is needed",
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
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, Mapping):
        return {str(k): plain(v) for k, v in fields.items() if not str(k).startswith("_")}
    return str(value)


def what_happened(result: Any, moved: bool) -> str:
    """State what the last action did. State it, do not interpret it.

    This began as a boolean off the fingerprint, so a button losing its label mid-login read like
    arriving somewhere new. The first repair guessed the other way -- "which usually means it is
    still working on the last action" -- which is wrong on a toggled switch, where one control
    changing IS the completed action, and wrong again on a relabel, which AUA reports as `changed`
    with nothing added or removed. Replacing one false inference with another is not a fix. The
    counts are facts; what they mean is the model's job.
    """
    if not moved:
        return "the screen did not change at all"
    change = result.get("change") if isinstance(result, Mapping) else None
    diff = result.get("action_diff_summary") if isinstance(result, Mapping) else None
    if isinstance(change, Mapping) and change.get("activity_changed") is True:
        return "a different screen opened"
    if isinstance(diff, Mapping):
        added = int(diff.get("added") or 0)
        removed = int(diff.get("removed") or 0)
        changed = int(diff.get("changed") or 0)
        total = int(diff.get("curr_count") or 0)
        if total:
            return (f"same screen: {added} controls appeared, {removed} went away, "
                    f"{changed} were relabelled, out of {total}")
    return "the screen changed"


#: Sent to the model as-is. Anything not on this list is either an internal handle or a number
#: the model cannot use, and this model is documented to lose accuracy to irrelevant state.
READABLE = ("text", "desc", "content_desc", "resource_id", "rid", "checked")


def screen_for_model(compact: Mapping[str, Any] | None) -> dict[str, Any]:
    """The screen with everything the model cannot read taken out.

    The numbered menu was introduced to keep 32-character element digests out of the request, and
    then the state body carried the same digests on every element anyway, plus the fingerprint and
    raw pixel bounds. Those are the harness's vocabulary, not the screen's.
    """
    observation = (compact or {}).get("observation") if isinstance(compact, Mapping) else None
    if not isinstance(observation, Mapping):
        return {}
    screen = observation.get("screen") if isinstance(observation.get("screen"), Mapping) else {}
    elements = []
    for element in observation.get("elements") or []:
        if not isinstance(element, Mapping):
            continue
        kept = {key: element[key] for key in READABLE if element.get(key) not in (None, "")}
        if kept:
            elements.append(kept)
    return {"app": screen.get("package"), "elements": elements}


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
    from typesafe_sdk import Choice

    questions = {
        # Neither question may presuppose the other's answer. "the single best next action to
        # reach the goal" is false the moment the goal is reached, and "which control should THAT
        # ACTION operate on" is a question about another answer -- indirection this model is
        # documented to pay for, and the reason a "no control" option was never taken.
        "action": Choice(instructions="What should happen next on this screen?",
                         criteria=dict(ACTION_KINDS)),
        "target": Choice(instructions="Which control on this screen moves toward the goal?",
                         criteria=numbered(options)[0]),
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
        # One entry per step the RUN took, not per step this navigator won. The chat model takes
        # most of them in the default space, and a journey that silently omits them told the model
        # it was on step 6 of a run that was on step 12. The 33%-vs-41% measurement behind this
        # field was taken on a journey built from every step, which is not what shipped.
        self._journey: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._options: dict[str, str] = {}
        # (screen key, action:operand) pairs already proposed. This model reads each screen from
        # scratch, so on a screen its own action failed to move it repeats that action -- observed
        # live as a 20-step loop. A screen mid-load re-fingerprints on every frame, so a wait is
        # keyed on the activity instead, or waiting would never be bounded at all.
        self._seen: set[tuple[str, str]] = set()
        self._last_fingerprint: str | None = None
        self._last_activity: str | None = None
        self.proposals: list[dict[str, Any]] = []
        self.declined: dict[str, int] = {}
        self.requests = 0
        self.input_tokens = 0
        self.request_ms: list[float] = []

    def _decline(self, why: str) -> None:
        self.declined[why] = self.declined.get(why, 0) + 1

    def observed(self, tool: str, arguments: Mapping[str, Any] | None = None) -> None:
        """Record what the run actually did on this step, whoever chose it."""
        if self._pending is None:
            return
        handle = (arguments or {}).get("id")
        label = self._options.get(handle) if isinstance(handle, str) else None
        self._pending["you_chose"] = f"{tool} on '{label}'" if label else tool

    async def __call__(self, result: Any) -> dict[str, Any] | None:
        from experiments.aua_controller.compaction import compact_frame

        compact = compact_frame(result, keep_ids=True)
        observation = compact.get("observation") if isinstance(compact, Mapping) else None
        meta = observation.get("meta") if isinstance(observation, Mapping) else None
        fingerprint = meta.get("fingerprint") if isinstance(meta, Mapping) else None
        change = result.get("change") if isinstance(result, Mapping) else None
        activity = change.get("activity_after") if isinstance(change, Mapping) else None

        # Close the open turn and move the screen markers on BEFORE anything can return early.
        # Leaving a turn open across a decline closed it later against a screen it never saw,
        # which is how a tap on Notifications came back as "scroll_and_analyze on '?'".
        if self._pending is not None:
            self._pending["what_happened"] = what_happened(
                result, moved=fingerprint != self._last_fingerprint)
            self._journey.append(self._pending)
        self._last_fingerprint = fingerprint if isinstance(fingerprint, str) else None
        self._last_activity = activity if isinstance(activity, str) else self._last_activity
        # Opened for every step. Whoever acts, `observed` fills it in.
        self._pending = {"n": len(self._journey) + 1, "you_chose": "(nothing yet)"}

        if not self.can_tap:
            self._decline("tap_not_offered")
            return None
        self._options = candidates(observation)
        if len(self._options) < 2:
            # One control is not a choice, and none is not a screen this can help with.
            self._decline("too_few_controls")
            return None

        journey = list(self._journey)
        # The state is what the model reads; every token in it that is not about this decision is
        # documented to cost accuracy. Oldest turns go first -- a loop is made of the recent ones.
        while len(json.dumps(journey, default=str)) > MAX_JOURNEY_CHARS and journey:
            journey.pop(0)
        state = {"goal": self.goal, "journey_so_far": journey,
                 "this_is_the_new_screen": screen_for_model(compact)}
        questions = build_questions(self._options, action_space=self.action_space)
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
        action, target = response.answers["action"], response.answers["target"]
        record = {
            "kind": action.choice, "kind_confidence": round(action.confidence, 4),
            "target": target.choice, "target_confidence": round(target.confidence, 4),
            "target_id": numbered(self._options)[1].get(target.choice),
            "options": len(self._options),
        }

        def settle(accepted: bool, why: str | None = None) -> None:
            record["accepted"] = accepted
            if not accepted:
                record["declined_because"] = why
            self.proposals.append(record)
            turn["verdict"] = dict(record)
            self._record(turn)

        plan, why = self._plan(action, target, response.answers, numbered(self._options)[1])
        if plan is None:
            settle(False, why)
            return None
        tool, arguments, operand, operand_confidence, label = plan
        record["tool"] = tool
        # Name the operand the gate actually read: `target_confidence` belongs to the tap question
        # and says nothing about a scroll, a back or a finish.
        record["operand"] = operand
        record["operand_confidence"] = round(operand_confidence, 4)
        gate = min(action.confidence, operand_confidence)
        record["gate"] = round(gate, 4)
        record["gate_needed"] = self.min_confidence
        if gate < self.min_confidence:
            self._decline("below_confidence")
            settle(False, "below_confidence")
            return None

        # A waiting screen re-fingerprints on every frame it redraws, so keying a wait on the
        # fingerprint would never repeat and never escalate. The activity is what holds still.
        screen_key = self._last_activity if action.choice == "wait" else str(fingerprint)
        pair = (str(screen_key), f"{action.choice}:{operand}")
        if screen_key is not None and pair in self._seen:
            self._decline("repeat_on_unchanged_screen")
            settle(False, "repeat_on_unchanged_screen")
            return None
        if self.shadow:
            self._decline("shadow")
            settle(False, "shadow")
            return None
        settle(True)
        if screen_key is not None:
            self._seen.add(pair)
        self._pending["you_chose"] = f"{tool} on '{label}'"
        return {"tool": tool, "arguments": arguments,
                "reason": f"System One {action.choice} at confidence {gate:.2f}: {label}"}

    def _plan(self, action, target, answers, by_index):
        """Bind the chosen action to an offered tool, or say why it cannot be.

        Returns ``(plan, why)``. A plan is ``(tool, arguments, operand, confidence, label)``; the
        operand is what the gate reads and what the repeat guard keys on, so an action that takes
        none -- going back, waiting -- is gated on the action choice alone.
        """
        kind = action.choice
        if kind in NON_ACTIONS:
            # Chosen on purpose, declined on purpose: acting on `blocked` ends the run, and
            # ending a run early is this model's worst measured skill.
            self._decline(f"kind:{kind}")
            return None, f"kind:{kind}"
        if kind == "tap":
            handle = by_index.get(target.choice)
            if handle is None:
                self._decline("unknown_target")
                return None, "unknown_target"
            return (TAP_TOOL, {"id": handle}, handle, target.confidence,
                    self._options[handle]), None
        if kind == "type":
            # A System One model returns a choice, never a string; the public harnesses call a
            # small generative model here and so does this one, by handing the step back.
            self._decline("kind:type")
            return None, "kind:type"
        if kind == "wait":
            # Honoured in both spaces. Offering it and then paying a chat model to answer the
            # same question about the same screen was the defect, not the option.
            if WAIT_TOOL not in self.offered:
                self._decline("wait_not_offered")
                return None, "wait_not_offered"
            return (WAIT_TOOL, {"idle": True}, "idle", action.confidence,
                    "wait for the screen"), None
        if self.action_space != "full":
            # Keeping the narrow default is what lets the two be compared on the same code.
            self._decline(f"kind:{kind}")
            return None, f"kind:{kind}"
        if kind in SCROLL_KINDS:
            if SCROLL_TOOL not in self.offered:
                self._decline("scroll_not_offered")
                return None, "scroll_not_offered"
            direction = SCROLL_KINDS[kind]
            return (SCROLL_TOOL, {"direction": direction}, direction, action.confidence,
                    f"scroll {direction}"), None
        if kind == "back":
            if BACK_TOOL not in self.offered:
                self._decline("back_not_offered")
                return None, "back_not_offered"
            return (BACK_TOOL, {}, "back", action.confidence, "go back"), None
        if kind == "done":
            if FINISH_TOOL not in self.offered:
                self._decline("finish_not_offered")
                return None, "finish_not_offered"
            outcome = answers.get("outcome")
            if outcome is None:
                self._decline("no_outcome")
                return None, "no_outcome"
            if outcome.choice == UNFINISHED:
                # It asked to stop and said the goal is not finished. Both cannot be acted on.
                self._decline("done_but_unfinished")
                return None, "done_but_unfinished"
            # No note: it is free text, and a fabricated one would reach the judge as evidence.
            return (FINISH_TOOL, {"outcome": outcome.choice}, outcome.choice,
                    outcome.confidence, f"finish as {outcome.choice}"), None
        self._decline(f"kind:{kind}")
        return None, f"kind:{kind}"

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
