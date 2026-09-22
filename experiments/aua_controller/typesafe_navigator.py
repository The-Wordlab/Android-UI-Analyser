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

**One question, and the gate measured against it.** A press is not an action plus a separate
operand -- each pressable control *is* an action, listed beside the actions that operate on
nothing. The earlier shape asked "what kind of move" and "which control" as two independent
questions, because the API has no question conditional on another answer, so the model named a
control even when it chose to wait and the gate was the minimum of two confidences about
different things. Measured over 60 real screens three times, the merged form is steadier (median
confidence 0.54-0.55 against 0.47-0.48) and acts on the same taps at the same accuracy. It is not
faster. It is one question with one answer and nothing discarded.

200 steps replayed out of verified-pass runs of a real app, scored against what the run did next
-- a floor on correctness, not correctness: a different tap is not a wrong tap, and the chat model
itself takes recoverable detours. Only a tap is ever acted on in the default space. Three samples,
the spread shown where they differ:

====  ==============  ===========================
gate  steps acted on  same control the run tapped
====  ==============  ===========================
0.00       106 (53%)                    ~49%
0.70        33 (16%)                    ~76%
0.80        28 (14%)                     79%
0.85        26 (13%)                    ~83%
0.90        19  (9%)                    ~93%
====  ==============  ===========================

Every row of that table moved when the journey started quoting screens instead of counting
controls (see ``what_happened``): at 0.85 it was 19 steps at 74%, and is now 26 at 83% -- more
coverage *and* more accuracy, which is not a trade. That was the single largest measured change
to this navigator, larger than the question shape and far larger than the threshold.

0.85 stays the default. 0.90 is now a real alternative for the first time -- it was within noise
of 0.85 under the old journey and is worth about ten points of fidelity under this one, for a
third fewer steps. Naming a control is still where the accuracy goes: 49% of presses match the
run when nothing is gated at all.

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
MIN_CONFIDENCE = 0.85  # measured; see the module docstring -- the threshold is not the lever
#: A near miss under the gate buys one wait and a fresh read, not a retry. The same request
#: replayed eight times scored 0.66-0.78 and never crossed 0.80; the screen read again after a
#: wait scored 0.85 eight times out of eight -- the first frame was the login page still
#: finishing. Below this floor the chat model takes the step at once.
SECOND_LOOK_FLOOR = 0.60
#: A pick whose own probability clears the gate is taken when its confidence is at least this,
#: although that confidence sits under the gate. Jev reports both numbers: over 399 saved answers
#: the confidence ran a median 0.03 under the top probability and never more than 0.07, so a
#: probability over the gate is the same judgement in the model's other voice. Measured on 112
#: aligned steps, the picks this admits were three for three right (0.79/0.82, 0.79/0.81,
#: 0.77/0.80); each had cost a wait, a re-ask and a chat-model call for the very press it named.
PROBABILITY_GATE_FLOOR = 0.60
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
#: The harness's finish outcomes, word for word what `session_finish` accepts. They used to be a
#: second question beside the move, and on a step in the middle of a run none of them is true,
#: so that question had to offer `in_progress` -- which then vetoed a confident `done`. Over a
#: 27-row run it vetoed 17 times: the 10 premature ones were already under the gate, and the 2
#: right ones above it were lost (an observe-only row, 13s of chat-model time to say nothing).
#: Finishing is now a move like any other, so a mid-run step simply picks a press or a scroll and
#: nothing is asked about a run that has not stopped.
FINISH_OUTCOMES: dict[str, str] = {
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
    "type": "Type text into a text field on this screen",
    "scroll_down": "What is needed is below; scroll down to reveal it",
    "scroll_up": "What is needed is above; scroll up to reveal it",
    "back": "This is not the screen the goal needs; the previous screen was closer",
    "achieved": "The goal was carried out during this run; nothing further is needed",
    "already_satisfied": "Nothing was ever needed; the goal was already true before the run began",
    "wait": "This screen is still loading or mid-animation; nothing should be pressed yet",
    "blocked": "Something outside the goal stops this run going further",
    "not_achievable": "This app cannot do what the goal asks",
}
#: Scroll is two actions rather than one action plus a direction question. The public browser
#: harnesses carry SCROLL_UP and SCROLL_DOWN as operations for the same reason it is right here:
#: "which way should this screen be scrolled" is a hop of indirection jev-1.13's own notes warn
#: about, and gating on min(action, direction) mixed the confidences of two separate questions,
#: which those notes also warn about. Replayed over 11 saved screens the merged form chose the
#: same action 11 times out of 11 and asked 2% fewer tokens, so the extra question was buying
#: nothing. What it picks when a scroll is genuinely needed is untested either way.
SCROLL_KINDS = {"scroll_down": "down", "scroll_up": "up"}
#: AUA's tool names said back in the vocabulary the model answers in, so a journey the chat model
#: half-wrote still reads as one story rather than two.
TOOL_WORDS = {
    TAP_TOOL: "press a control",
    SCROLL_TOOL: "scroll",
    "swipe_and_analyze": "scroll",
    BACK_TOOL: "back",
    "key_and_analyze": "back",
    WAIT_TOOL: "wait",
    FINISH_TOOL: "done",
    "input_and_analyze": "type",
}

#: `blocked` is chosen to be declined: acting on it means ending the run, and ending a run early
#: is this model's worst measured skill. `wait` is not in here because waiting is a real tool the
#: harness already offers -- asking "is this screen still loading?" and then paying a chat model
#: to answer the same question was the option costing a round trip to say nothing.
NON_ACTIONS = ("blocked", "not_achievable")
#: The two finish moves a System One answer may act on; each is its own `session_finish` outcome.
FINISH_KINDS = ("achieved", "already_satisfied")


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


def is_switch(element: Mapping[str, Any]) -> bool:
    """A control whose checked state means something.

    Compact observations carry ``checked`` only on switches. A raw hierarchy dump carries
    ``checked: false`` on every node -- the status-bar clock, the battery icon, static text --
    with ``checkable: false`` beside it; that first frame turned 22 status-bar nodes into
    "switches" and thirteen junk options.
    """
    return "checked" in element and element.get("checkable") is not False


FIELD_SUFFIX = " (text field)"


def move_phrase(label: str) -> str:
    """The menu line for one control: what a person would do to it, not just its name.

    A button is pressed. A field is tapped so text can be typed into it -- the same words the
    goal uses when it asks for typing, which is how a literal reader tells the field apart from
    the button beside it whose id happens to contain a word from the goal.
    """
    if label.endswith(FIELD_SUFFIX):
        return f"Tap the text field '{label[: -len(FIELD_SUFFIX)]}' so text can be typed into it"
    return f"Press '{label}'"


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
                or is_switch(element)):
            continue
        label = next((element[key] for key in ("text", "desc", "content_desc", "resource_id", "rid")
                      if isinstance(element.get(key), str) and element[key].strip()),
                     None)
        if label is None:
            label = where(element, observation.get("screen") if isinstance(observation, Mapping) else None)
        if element.get("editable") is True:
            # A field is labelled by its hint, so the menu read "Press 'Ask me anything'" beside
            # "Press 'buttonOpenComposerAttachments'" -- and a goal that said "tap the composer"
            # matched the word, not the field, twice at 0.96 and 0.93. The role is the fact a
            # reader uses to tell a field from the button next to it.
            label = f"{label}{FIELD_SUFFIX}"
        if is_switch(element):
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


#: Element fields the model can actually read. Everything else on an element is the harness's
#: vocabulary: digests, pixel bounds, internal flags.
READABLE = ("text", "desc", "content_desc", "resource_id", "rid", "checked", "editable")
MAX_JOURNEY_LABELS = 10  # a turn is a reminder of a screen, not a second copy of one
MAX_LABEL_CHARS = 34


def sketch(result: Any) -> str:
    """The screen in one line of its own words: ``Welcome back · [Sign in] · [Browse as a guest]``.

    Pressable controls are bracketed, because "what could I have pressed there" is the question a
    journey turn is read for. Cut short on purpose: the current screen is already in the state in
    full, and a turn that reproduces one is a second copy of it in a model documented to lose
    accuracy as the state fills.
    """
    observation = result.get("observation") if isinstance(result, Mapping) else None
    elements = (observation or result or {}).get("elements") if isinstance(result, Mapping) else None
    labels: list[str] = []
    for element in elements or []:
        if not isinstance(element, Mapping):
            continue
        text = str(element.get("text") or element.get("desc") or element.get("content_desc") or "")
        text = " ".join(text.split())[:MAX_LABEL_CHARS]
        if not text:
            continue
        labels.append(f"[{text}]" if element.get("clickable") else text)
        if len(labels) >= MAX_JOURNEY_LABELS:
            labels.append("…")
            break
    return " · ".join(labels)


def what_happened(result: Any, moved: bool) -> str:
    """Whether the last action moved the screen, and what the screen then said.

    This began as a boolean off the fingerprint, so a button losing its label mid-login read like
    arriving somewhere new. The repair after that reported counts -- "7 controls appeared, 2 went
    away, out of 32" -- which are facts, but facts about a screen the model never sees: they
    cannot tell a login page from a settings list, and telling those apart is exactly how a model
    knows it is going in circles. The labels can. The one-bit answer stays in front of them,
    because "your tap did nothing" is not recoverable from a screen that looks plausible.
    """
    seen = sketch(result)
    if not moved:
        return f"the screen did not change: {seen}" if seen else "the screen did not change at all"
    return f"now showing: {seen}" if seen else "the screen changed"

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
    meta = observation.get("meta") if isinstance(observation.get("meta"), Mapping) else {}
    elements = []
    for element in observation.get("elements") or []:
        if not isinstance(element, Mapping):
            continue
        kept = {key: element[key] for key in READABLE if element.get(key) not in (None, "")}
        if kept:
            elements.append(kept)
    out: dict[str, Any] = {"app": screen.get("package"), "elements": elements}
    # Only when there is something to say. A screen mid-load and an idle screen are the same
    # hierarchy, and told nothing about the network this model re-pressed a button it had already
    # pressed; told the login POST had not answered, it waited instead. The key is absent on a
    # quiet screen because every line of state that is not about the decision costs accuracy.
    network = meta.get("network_calls")
    if isinstance(network, list) and network:
        out["network"] = [str(item) for item in network]
    return out


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


#: What finishing means while the script has steps to go: the step is done, not the run. Live,
#: the run-level line ("nothing further is needed") sat beside a list of steps still to do, and a
#: true "this step is done" came back at 0.73 and 0.49 -- under the gate, so the pointer never
#: moved and every later ask was about a step the run had long finished. Reworded, the same
#: screen came back at 0.88.
STEP_DONE_KINDS: dict[str, str] = {
    "achieved": "This step is done; the run should move on to the next step",
    "already_satisfied": "This step was already true before anything was done; move on to the next step",
}


#: Asked beside the move while steps remain. Live, a step the chat model had already carried out
#: stayed current for twelve asks because the model never *picked* "done" among fifteen moves
#: (0.03-0.49); asked this directly on the same saved screens it said done at up to 0.86. It is
#: a second decision, not a second opinion on the move, so it is judged on its own gate and never
#: mixed into the move's -- the docstring's warning is about min() over one decision.
STEP_QUESTION = {
    "instructions": "Look at the screen and the journey. Has the current step (`goal`) already been carried out?",
    "criteria": {"done": "Yes, the step is already done; nothing on this screen is left to do for it",
                 "not_yet": "No, something still has to happen on this screen for this step"},
}


def build_questions(options: Mapping[str, str], *, action_space: str = "taps",
                    steps_remain: bool = False) -> dict[str, Any]:
    """One question naming every move this screen allows, finishing included.

    A press is not an action plus a separate operand; each pressable control *is* an action, and
    so is each way of finishing: the outcome `session_finish` records is the move itself. With
    ``steps_remain`` the two finish lines speak of the current step, not the run, and a second
    question asks outright whether that step is already done.
    """
    from typesafe_sdk import Choice

    # Every pressable control is its own action, beside the actions that operate on nothing.
    # The earlier shape asked "what kind of move" and "which control" as two independent
    # questions -- the API has no question conditional on another answer -- so the model named a
    # control even when it chose to wait, and the gate was then the minimum of two confidences
    # about different things. Measured over 60 real screens, three times: the merged form is
    # steadier (median confidence 0.54-0.55 against 0.47-0.48) and acts on the same taps at the
    # same accuracy. It is not faster; it is one question with one answer and nothing discarded.
    # One text field made two menu lines, "tap the field so text can be typed" and "type", and
    # the vote split between them (0.54 / 0.46 on a real chat screen) although both meant the
    # same thing. The harness built both from the same element, so it also knows they are one:
    # with a single field on the screen, `type` names that field and the tap line is not offered.
    # Typing hands the step to the chat model, which focuses the field and types in one call.
    fields = [label for label in options.values() if label.endswith(FIELD_SUFFIX)]
    lone = fields[0][: -len(FIELD_SUFFIX)] if len(fields) == 1 else None
    criteria: dict[str, str] = {index: move_phrase(label)
                                for index, label in numbered(options)[0].items()
                                if lone is None or not label.endswith(FIELD_SUFFIX)}
    actions = dict(ACTION_KINDS)
    if steps_remain:
        actions.update(STEP_DONE_KINDS)
    if lone is not None:
        actions["type"] = f"Type text into the text field '{lone}'"
    criteria.update({kind: text for kind, text in actions.items() if kind != "tap"})
    questions: dict[str, Any] = {"move": Choice(instructions="What should happen next on this screen?",
                                                criteria=criteria)}
    if steps_remain:
        questions["step"] = Choice(**STEP_QUESTION)
    return questions


def goal_steps(goal: str) -> list[str]:
    """The goal's own ordered steps, cut only where the prose says so; a one-step goal is [].

    A human-written brief is a short script: "open the menu and look, then close it. Send a
    message and wait for the reply, then open the menu again." Sent whole on every turn, the
    model has to work out from the journey how far the script has run -- and a System One model
    is bad at counting. Measured on one row: the two picks that opened the menu were 0.96 and
    0.85; the picks that had to know *which* phase the run was in were 0.21 to 0.51, every one
    of them declined and paid for twice. AUA's own `goal_phases` already cuts a goal at its
    sequence words (then, next, after that, a full stop) and never invents a step, so the
    author keeps writing prose and the navigator hands the model one step at a time.
    """
    from android_ui_analyser.session import goal_phases

    steps = [phase.objective for phase in goal_phases(goal) if phase.kind == "verify"]
    return steps if len(steps) > 1 else []


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
        second_look_floor: float | None = None,
        action_space: str = "taps",
        shadow: bool = False,
        timeout_s: float = 10.0,
        transcript_path: Any = None,
    ) -> None:
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must sit in (0, 1]")
        if action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}")
        if second_look_floor is None:
            # The default floor follows a gate set under it; an explicit one above the gate is a mistake.
            second_look_floor = min(SECOND_LOOK_FLOOR, min_confidence)
        if not 0 <= second_look_floor <= min_confidence:
            raise ValueError("second_look_floor must sit in [0, min_confidence]")
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            client = AsyncTypeSafeClient()
        self.client = client
        self.goal = goal
        # The script's steps and where the run is in it. A finish answered while steps remain is
        # "this step is done", moves the pointer and is asked again on the same screen.
        self.steps = goal_steps(goal)
        self.step_index = 0
        self.model = model
        self.min_confidence = min_confidence
        self.second_look_floor = second_look_floor
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
        self._second_looks: set[str] = set()
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
        """Record what the run did on this step, in the words the model itself answers in.

        The tool is AUA's function name. The model never says `tap_and_analyze`; it answers
        "press 'Privacy'", and a history written in the harness's vocabulary is one the model has
        to translate before it can read what it did. Steps the chat model took are named the same
        way, because the journey is one story.
        """
        if self._pending is None:
            return
        handle = (arguments or {}).get("id")
        label = self._options.get(handle) if isinstance(handle, str) else None
        if label:
            phrase = move_phrase(label)
            self._pending["you_chose"] = phrase[0].lower() + phrase[1:]
        else:
            self._pending["you_chose"] = TOOL_WORDS.get(tool, tool)

    def forget(self) -> None:
        """Drop the open turn: its action was never sent.

        AUA refused it as stale -- the screen moved on between the read and the press -- so
        it is not part of the story the model reads on the next step. Leaving it in taught
        the model that pressing the same control twice is what a run does.
        """
        self._pending = None

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

        by_index = numbered(self._options)[1]
        # One ask per step the model may declare done on this screen, plus the move itself. A
        # step declared done costs a second question, never a device step, and the pointer only
        # ever moves forward, so this is bounded by the script's length.
        for _ in range(len(self.steps) + 1):
            questions = build_questions(self._options, action_space=self.action_space,
                                        steps_remain=bool(self.steps) and self.step_index < len(self.steps) - 1)
            asked = await self._ask(compact, questions)
            if asked is None:
                return None
            turn, answers = asked
            move = answers["move"]
            record = {
                "kind": "tap" if move.choice in by_index else move.choice,
                "choice": move.choice,
                "confidence": round(move.confidence, 4),
                "target_id": by_index.get(move.choice),
                "options": len(self._options),
            }

            def settle(accepted: bool, why: str | None = None, *, record=record, turn=turn) -> None:
                record["accepted"] = accepted
                if not accepted:
                    record["declined_because"] = why
                self.proposals.append(record)
                turn["verdict"] = dict(record)
                self._record(turn)

            steps_remain = bool(self.steps) and self.step_index < len(self.steps) - 1
            step = answers.get("step") if isinstance(answers, Mapping) else None
            if steps_remain and step is not None and step.choice == "done":
                # A confident "already done" moves the pointer; the move it came with was about a
                # step that is over, so it is not taken. An unsure one changes nothing.
                verdict = {"kind": "phase_done", "via": "step_question", "choice": "done",
                           "confidence": round(step.confidence, 4), "step": self.step_index + 1,
                           "options": len(self._options)}
                if self._gate(step, verdict):
                    verdict["accepted"] = True
                    self.proposals.append(verdict)
                    turn["verdict"] = dict(verdict)
                    self._record(turn)
                    self._advance()
                    continue
            if steps_remain and move.choice in FINISH_KINDS:
                # "Done" with steps still to go is a claim about the current step, not the run.
                record["kind"] = "phase_done"
                record["via"] = "move"
                record["step"] = self.step_index + 1
                if not self._gate(move, record):
                    self._decline("below_confidence")
                    settle(False, "below_confidence")
                    return None
                settle(True)
                self._advance()
                continue
            break

        plan, why = self._plan(move, by_index)
        if plan is None:
            settle(False, why)
            return None
        tool, arguments, operand, label = plan
        record["tool"] = tool
        record["operand"] = operand
        passes = self._gate(move, record)
        gate, probability = record["gate"], record["probability"]
        if not passes:
            look_key = self._last_activity or str(fingerprint)
            if (gate >= self.second_look_floor and WAIT_TOOL in self.offered
                    and look_key not in self._second_looks):
                # One more look, not one more ask: the same screen re-asked gives the same
                # number, a screen read again after a wait may not be the same screen.
                self._second_looks.add(look_key)
                record["second_look"] = True
                self._decline("second_look")
                settle(False, "below_confidence")
                self._pending["you_chose"] = TOOL_WORDS[WAIT_TOOL]
                return {"tool": WAIT_TOOL, "arguments": {"idle": True},
                        "reason": (f'System One second look: {record["kind"]} at {gate:.2f} is under '
                                   f"{self.min_confidence:.2f}; waiting for the screen once before asking again")}
            self._decline("below_confidence")
            settle(False, "below_confidence")
            return None

        # A waiting screen re-fingerprints on every frame it redraws, so keying a wait on the
        # fingerprint would never repeat and never escalate. The activity is what holds still.
        screen_key = self._last_activity if record["kind"] == "wait" else str(fingerprint)
        pair = (str(screen_key), f'{record["kind"]}:{operand}')
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
        voice = f" (probability {probability:.2f})" if record.get("accepted_by") == "probability" else ""
        return {"tool": tool, "arguments": arguments,
                "reason": f'System One {record["kind"]} at confidence {gate:.2f}{voice}: {label}'}

    def _state(self, compact: Mapping[str, Any]) -> dict[str, Any]:
        """What the model reads: the current step, the script around it, the journey, the screen."""
        journey = list(self._journey)
        # Every token in the state that is not about this decision is documented to cost
        # accuracy. Oldest turns go first -- a loop is made of the recent ones.
        while len(json.dumps(journey, default=str)) > MAX_JOURNEY_CHARS and journey:
            journey.pop(0)
        state: dict[str, Any] = {"goal": self.goal}
        if self.steps:
            state["goal"] = self.steps[self.step_index]
            state["done_before_this"] = self.steps[: self.step_index]
            state["still_to_do_after_this"] = self.steps[self.step_index + 1:]
        state["journey_so_far"] = journey
        state["this_is_the_new_screen"] = screen_for_model(compact)
        return state

    def _advance(self) -> None:
        """The current step is done: move the pointer and say so in the journey."""
        done, self.step_index = self.steps[self.step_index], self.step_index + 1
        self._journey.append({"n": len(self._journey) + 1,
                              "you_chose": f"said the step '{done}' was done",
                              "what_happened": f"the next step is '{self.steps[self.step_index]}'"})
        if self._pending is not None:
            self._pending["n"] = len(self._journey) + 1

    async def _ask(self, compact: Mapping[str, Any], questions: Mapping[str, Any]):
        """One request; ``None`` when it failed, else the transcript turn and every answer."""
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

    def _gate(self, move, record: dict[str, Any]) -> bool:
        """Does this one answer clear the gate? Writes the numbers it judged into the record.

        One question, one answer, one number -- `min()` of two confidences about different
        things was never a statement about this decision. A pick under the gate still passes
        when its own probability clears it (see PROBABILITY_GATE_FLOOR).
        """
        gate = move.confidence
        probability = float((getattr(move, "probabilities", None) or {}).get(str(move.choice), 0.0) or 0.0)
        record["gate"] = round(gate, 4)
        record["gate_needed"] = self.min_confidence
        record["probability"] = round(probability, 4)
        if gate >= self.min_confidence:
            return True
        if gate >= PROBABILITY_GATE_FLOOR and probability >= self.min_confidence:
            record["accepted_by"] = "probability"
            return True
        return False

    def _plan(self, move, by_index):
        """Bind the one chosen move to an offered tool, or say why it cannot be.

        Returns ``(plan, why)``. A plan is ``(tool, arguments, operand, label)``. The operand is
        what the repeat guard keys on.
        """
        # A numbered answer IS a press: the menu of controls and the list of actions are one
        # list, so naming a control names the whole move.
        handle = by_index.get(move.choice)
        if handle is not None:
            return (TAP_TOOL, {"id": handle}, handle, self._options[handle]), None
        kind = move.choice
        if kind not in ACTION_KINDS:
            # Not a control on this screen and not an action either -- a stale index, or a raw
            # element id the menu exists to keep out. The menu is rebuilt per screen, so there is
            # nothing safe to press.
            self._decline("unknown_target")
            return None, "unknown_target"
        if kind in NON_ACTIONS:
            # Chosen on purpose, declined on purpose: acting on `blocked` ends the run, and
            # ending a run early is this model's worst measured skill.
            self._decline(f"kind:{kind}")
            return None, f"kind:{kind}"
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
            return (WAIT_TOOL, {"idle": True}, "idle", "wait for the screen"), None
        if self.action_space != "full":
            # Keeping the narrow default is what lets the two be compared on the same code.
            self._decline(f"kind:{kind}")
            return None, f"kind:{kind}"
        if kind in SCROLL_KINDS:
            if SCROLL_TOOL not in self.offered:
                self._decline("scroll_not_offered")
                return None, "scroll_not_offered"
            direction = SCROLL_KINDS[kind]
            return (SCROLL_TOOL, {"direction": direction}, direction,
                    f"scroll {direction}"), None
        if kind == "back":
            if BACK_TOOL not in self.offered:
                self._decline("back_not_offered")
                return None, "back_not_offered"
            return (BACK_TOOL, {}, "back", "go back"), None
        if kind in FINISH_KINDS:
            if FINISH_TOOL not in self.offered:
                self._decline("finish_not_offered")
                return None, "finish_not_offered"
            # No note: it is free text, and a fabricated one would reach the judge as evidence.
            # The enum the model answered is the whole claim, and the gate above applies to it
            # exactly as to a press -- ending a run early is this model's worst measured skill.
            return (FINISH_TOOL, {"outcome": kind}, kind, f"finish as {kind}"), None
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
            "min_confidence": self.min_confidence, "second_look_floor": self.second_look_floor,
            "requests": self.requests,
            "input_tokens": self.input_tokens, "usd": round(self.usd, 8),
            "transcript": str(self.transcript_path) if self.transcript_path else None,
            "proposals": len(self.proposals),
            "phase": ({"current": self.step_index + 1, "of": len(self.steps)} if self.steps else None),
            "accepted": accepted, "declined": dict(self.declined),
            "mean_request_ms": (round(sum(self.request_ms) / len(self.request_ms), 1)
                                if self.request_ms else None),
            # A shadow run is only worth taking if you can read back what it would have done,
            # step by step, against what the controller actually chose.
            "proposals_detail": self.proposals[:64],
        }


__all__ = ["TypeSafeNavigator", "ACTION_KINDS", "ACTION_SPACES", "NON_ACTIONS", "numbered", "goal_steps", "STEP_DONE_KINDS", "STEP_QUESTION",
           "MODEL", "MIN_CONFIDENCE",
           "TAP_TOOL", "SCROLL_TOOL", "BACK_TOOL", "FINISH_TOOL", "WAIT_TOOL", "SCROLL_KINDS",
           "FINISH_OUTCOMES", "FINISH_KINDS", "build_questions", "what_happened", "candidates", "tool_names"]
