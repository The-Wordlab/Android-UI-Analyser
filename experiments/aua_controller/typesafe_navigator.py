"""A System One navigator: it answers the steps it is sure of and declines the rest.

The controller asks a chat model for the next tool call, which is the dominant call count of a
run. Most of those calls decide something narrow -- which control on this screen moves the goal
on -- and that is a Choice over controls the harness already enumerates. This plugs into
``run_agent``'s ``host_next`` seam, so returning ``None`` hands the step back to the chat model
and nothing else changes.

**Three narrow questions about the whole brief, in one request.** ``operation`` asks what kind of
move comes next, ``target`` which control a press would be on, and ``text`` which of the brief's
own quoted texts a typing move would type. This is the shape TypeSafe's docs and the public Jev
browser harness use: one narrow judgment per question, only the operations this screen allows,
and the operand questions asked speculatively beside the operation because a System One request
prices the state once.

The earlier shape asked one merged question -- every control plus ten boundary moves -- about one
step of the brief, tracked by a pointer. Replayed over 378 real screens from a QA group, both
halves of that were the cause of the hand-overs, not the gate:

* On screens the chat model took, the merged question's top pick was "this step is done" 41% of
  the time and "back" 25%: boundary moves absorbed the probability whenever the screen did not
  match the step's words.
* The pointer was often on the wrong step. Of 90 chat-model presses on a listed control, the step
  Jev had been shown named that control 17 times; 40 named it nowhere and 33 in another step.

Asked only which control, with the whole brief, the same 90 presses were matched 83% of the time
instead of 53%. On scenarios held out from every choice made here, the three-question form acts
on 58% of screens where the merged form acted on 21%. It agrees with what the run did on 73% of
them, and most of the rest are the same move made another way: a back gesture where the run
pressed the on-screen back arrow.

It still declines on purpose, and two refusals are structural rather than tuned:

* **No free text.** A System One model generates nothing. It types only a text the brief itself
  quotes after "type exactly", chosen among those quotes; anything else goes to the chat model.
* **It never ends a run.** ``finished`` and ``other`` are offered so the truth has somewhere to
  go, and both hand the step back: the worst measured confusions were about stopping.

A proposal is an opinion with no authority beyond the tools it was offered. ``shadow`` records
what it would have done and returns ``None`` every time, which is how a run proves the gate on
real screens before anything depends on it.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.aua_controller.typesafe_cost import USD_PER_INPUT_TOKEN, client_options

MODEL = "jev-latest"
MIN_CONFIDENCE = 0.85  # applied to the operation and to the operand it acts on
#: A ceiling on the journey, not a documented API limit -- the SDK publishes none. It exists
#: because this model is documented to lose accuracy as the state fills with material that is not
#: about the decision, and a run's journey grows every step.
MAX_JOURNEY_CHARS = 30_000
MAX_OPTIONS = 60  # a Choice takes up to 255 options; a screen offering more is not a decision
TAP_TOOL = "tap_and_analyze"
SCROLL_TOOL = "scroll_and_analyze"
BACK_TOOL = "back_gesture_and_analyze"
WAIT_TOOL = "wait_and_analyze"
INPUT_TOOL = "input_and_analyze"
# "type exactly `Hi!`": text the author wrote into the brief, so it is no secret and needs no model.
EXACT_TEXT = re.compile(r"\btype\s+exactly\s+`([^`\n]{1,200})`", re.IGNORECASE)

ACTION_SPACES = ("taps", "full")

#: Every operation, one meaning each. Only the ones this screen and action space allow are
#: offered: a scroll on a screen with nothing to scroll is an option that can only be wrong.
#: `finished` and `other` exist to be chosen and declined -- jev-1.13 answers the question as
#: written, so a screen whose true move is a long-press had nowhere to go but a confident wrong
#: press, and a finished run nowhere but another tap.
OPERATIONS: dict[str, str] = {
    "press": "Press one of the controls on this screen",
    "type": "Type text into a text field on this screen",
    "scroll_down": "Scroll down: what is needed is further down this screen",
    "scroll_up": "Scroll up: what is needed is further up this screen",
    "back": "Go back to the previous screen",
    "wait": "Wait: the screen is still loading or the app is still answering",
    "finished": "Stop: every step of `goal` has been carried out",
    "other": ("Something else: a long-press, a system key, relaunching or force-closing the app, "
              "or a tool `goal` names"),
}
FULL_ONLY = ("scroll_down", "scroll_up", "back")
HANDED_OVER = ("finished", "other")
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
    "session_finish": "done",
    INPUT_TOOL: "type",
}


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

    A button is pressed. A field is tapped so text can be typed into it -- the words a goal uses
    when it asks for typing, which is how a literal reader tells the field apart from the button
    beside it whose id happens to contain a word from the goal.
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
            # "Press 'buttonAttach'" and a goal that named the field matched the button's id.
            # The role is the fact a reader uses to tell a field from the button next to it.
            label = f"{label}{FIELD_SUFFIX}"
        if is_switch(element):
            label = f"{label} [switch is {'ON' if element['checked'] else 'OFF'}]"
        options[handle] = label[:90]
        if len(options) >= limit:
            break
    return options


def scrollable(observation: Mapping[str, Any] | None) -> bool:
    return isinstance(observation, Mapping) and any(
        isinstance(element, Mapping) and element.get("scrollable") is True
        for element in observation.get("elements") or [])


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


def quoted_texts(goal: str) -> list[str]:
    """The texts the brief says to type, once each, in the order it names them."""
    return list(dict.fromkeys(EXACT_TEXT.findall(goal)))


def build_questions(options: Mapping[str, str], *, action_space: str = "taps",
                    can_scroll: bool = True, texts: Sequence[str] = ()) -> dict[str, Any]:
    """The operation, the control a press would be on, and the text a typing move would type."""
    from typesafe_sdk import Choice

    fields = any(label.endswith(FIELD_SUFFIX) for label in options.values())
    offered = [kind for kind in OPERATIONS
               if (kind != "type" or fields)
               and (kind not in FULL_ONLY or action_space == "full")
               and (kind not in SCROLL_KINDS or can_scroll)]
    questions: dict[str, Any] = {
        "operation": Choice(instructions="What should happen next on this screen to carry on with `goal`?",
                            criteria={kind: OPERATIONS[kind] for kind in offered}),
        "target": Choice(instructions="If a control on this screen is pressed next to carry on with `goal`, which one?",
                         criteria={index: move_phrase(label) for index, label in numbered(options)[0].items()}),
    }
    if fields and texts:
        questions["text"] = Choice(instructions="If text is typed next, which of these texts from `goal` is it?",
                                   criteria={str(n): text for n, text in enumerate(texts, start=1)})
    return questions


class TypeSafeNavigator:
    """Propose a confident move, or decline and let the chat model take the step."""

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

            client = AsyncTypeSafeClient(**client_options())
        self.client = client
        self.goal = goal
        self.texts = quoted_texts(goal)
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
        # Steps the brief names by a tool Jev cannot issue ("force-close with app_force_stop") are
        # the chat model's. Read whole, the brief let Jev walk past one: a cold-start row did its
        # post-restart checks before the restart (2026-09-29). Until the run has called such a
        # tool, Jev reads the brief only up to that step, so it cannot get ahead of it.
        own = {TAP_TOOL, INPUT_TOOL, WAIT_TOOL, "session_finish"}
        if action_space == "full":
            own |= {SCROLL_TOOL, BACK_TOOL}
        self._barriers = {name for name in offered - own
                          if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", goal)}
        self._called: set[str] = set()
        # One entry per step the RUN took, not per step this navigator won: a journey that omits
        # the chat model's steps tells the model it is on step 6 of a run that is on step 12.
        self._journey: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._options: dict[str, str] = {}
        # The brief's quoted texts typed so far, in order. They are the author's words, never a
        # secret, and they are how a typing move knows which quote comes next.
        self._typed: list[str] = []
        # (screen key, action:operand) pairs already proposed. This model reads each screen from
        # scratch, so on a screen its own action failed to move it repeats that action -- observed
        # live as a 20-step loop. A screen mid-load re-fingerprints on every frame, so a wait is
        # keyed on the activity instead, or waiting would never be bounded at all.
        self._seen: set[tuple[str, str]] = set()
        self._previous_screen = ""
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
        self._called.add(tool)
        arguments = arguments or {}
        handle = arguments.get("id")
        label = self._options.get(handle) if isinstance(handle, str) else None
        typed = str(arguments.get("text") or "")
        if tool == INPUT_TOOL and typed in self.texts:
            self._typed.append(typed)
        if self._pending is None:
            return
        if tool == INPUT_TOOL and label and label.endswith(FIELD_SUFFIX):
            # Named after the field's tap line, typing read "tap the text field so text can be
            # typed into it", and the model asked to type again on every later screen. Only a text
            # the brief quotes is ever repeated here; anything else may be a secret.
            field = label[: -len(FIELD_SUFFIX)]
            what = f"`{typed}`" if typed in self.texts else "text"
            sent = " and send it" if arguments.get("submit") else ""
            self._pending["you_chose"] = f"type {what} into the text field '{field}'{sent}"
        elif label:
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
            # The screen a turn was chosen on is kept only for the last turn. "Your tap did
            # nothing" is the one fact a screen that looks plausible cannot tell, so an unmoved
            # screen still says so.
            self._previous_screen = self._pending.pop("_screen", "")
            if fingerprint == self._last_fingerprint:
                self._pending["screen_did_not_change"] = True
            self._journey.append(self._pending)
        self._last_fingerprint = fingerprint if isinstance(fingerprint, str) else None
        self._last_activity = activity if isinstance(activity, str) else self._last_activity
        # Opened for every step. Whoever acts, `observed` fills it in.
        self._pending = {"n": len(self._journey) + 1, "you_chose": "(nothing yet)", "_screen": sketch(compact)}

        if not self.can_tap:
            self._decline("tap_not_offered")
            return None
        self._options = candidates(observation)
        if len(self._options) < 2:
            # One control is not a choice, and none is not a screen this can help with.
            self._decline("too_few_controls")
            return None

        by_index = numbered(self._options)[1]
        questions = build_questions(self._options, action_space=self.action_space,
                                    can_scroll=scrollable(observation), texts=self.texts)
        asked = await self._ask(compact, questions)
        if asked is None:
            return None
        turn, answers = asked
        operation = answers["operation"]
        record: dict[str, Any] = {"kind": operation.choice, "gate": round(operation.confidence, 4),
                                  "gate_needed": self.min_confidence, "options": len(self._options)}

        def settle(accepted: bool, why: str | None = None) -> None:
            record["accepted"] = accepted
            if not accepted:
                record["declined_because"] = why
                self._decline(str(why))
            self.proposals.append(record)
            turn["verdict"] = dict(record)
            self._record(turn)

        if operation.confidence < self.min_confidence:
            settle(False, "below_confidence")
            return None
        plan, why = self._plan(operation.choice, answers, by_index, record)
        if plan is None:
            settle(False, why)
            return None
        tool, arguments, operand, label = plan
        record["tool"] = tool
        record["operand"] = operand

        # A waiting screen re-fingerprints on every frame it redraws, so keying a wait on the
        # fingerprint would never repeat and never escalate. The activity is what holds still.
        screen_key = self._last_activity if operation.choice == "wait" else fingerprint
        pair = (str(screen_key), f"{operation.choice}:{operand}")
        if screen_key is not None and pair in self._seen:
            settle(False, "repeat_on_unchanged_screen")
            return None
        if self.shadow:
            settle(False, "shadow")
            return None
        settle(True)
        if screen_key is not None:
            self._seen.add(pair)
        return {"tool": tool, "arguments": arguments,
                "reason": f"System One {operation.choice} at confidence {record['gate']:.2f}: {label}"}

    def _plan(self, kind: str, answers: Mapping[str, Any], by_index: Mapping[str, str],
              record: dict[str, Any]):
        """Bind a confident operation to an offered tool, or say why it cannot be.

        Returns ``(plan, why)``. A plan is ``(tool, arguments, operand, label)``; the operand is
        what the repeat guard keys on. An operation that needs an operand is gated on the
        operand's own confidence as well: a sure "press" with an unsure control is not a press.
        """
        if kind == "press":
            target = answers["target"]
            handle = by_index.get(target.choice)
            record["target_id"] = handle
            record["operand_confidence"] = round(target.confidence, 4)
            if handle is None:
                # Not on this screen's menu -- a stale index, or a raw id the menu exists to keep
                # out. The menu is rebuilt per screen, so there is nothing safe to press.
                return None, "unknown_target"
            if target.confidence < self.min_confidence:
                return None, "below_confidence"
            return (TAP_TOOL, {"id": handle}, handle, self._options[handle]), None
        if kind == "type":
            return self._typing(answers, by_index, record)
        if kind == "wait":
            if WAIT_TOOL not in self.offered:
                return None, "wait_not_offered"
            return (WAIT_TOOL, {"idle": True}, "idle", "wait for the screen"), None
        if kind in SCROLL_KINDS and self.action_space == "full":
            if SCROLL_TOOL not in self.offered:
                return None, "scroll_not_offered"
            direction = SCROLL_KINDS[kind]
            return (SCROLL_TOOL, {"direction": direction}, direction, f"scroll {direction}"), None
        if kind == "back" and self.action_space == "full":
            if BACK_TOOL not in self.offered:
                return None, "back_not_offered"
            return (BACK_TOOL, {}, "back", "go back"), None
        # `finished`, `other`, and anything this space does not act on: the chat model's call.
        return None, f"kind:{kind}"

    def _typing(self, answers: Mapping[str, Any], by_index: Mapping[str, str], record: dict[str, Any]):
        """Type a text the brief quotes into a field, or hand the step to the chat model.

        The text is one of the author's own quotes, chosen by Jev among them, so nothing is
        generated. The field is the only one on the screen, or the one the target answer names.
        """
        text = answers.get("text") if isinstance(answers, Mapping) else None
        if text is None or INPUT_TOOL not in self.offered:
            return None, "kind:type"
        record["operand_confidence"] = round(text.confidence, 4)
        choice = str(text.choice)
        if not (choice.isdigit() and 1 <= int(choice) <= len(self.texts)):
            return None, "unknown_text"
        if text.confidence < self.min_confidence:
            return None, "below_confidence"
        fields = [handle for handle, label in self._options.items() if label.endswith(FIELD_SUFFIX)]
        target = by_index.get(str(answers["target"].choice))
        if len(fields) != 1:
            fields = [target] if target in fields else []
        if not fields:
            return None, "kind:type"
        typed = self.texts[int(choice) - 1]
        field = self._options[fields[0]][: -len(FIELD_SUFFIX)]
        return (INPUT_TOOL, {"id": fields[0], "text": typed, "submit": False},
                f"type:{typed}", f"type '{typed}' into '{field}'"), None

    def _state(self, compact: Mapping[str, Any]) -> dict[str, Any]:
        """What the model reads: the brief, the journey, the texts typed, the screen."""
        journey = list(self._journey)
        # Every token in the state that is not about this decision is documented to cost
        # accuracy. Oldest turns go first -- a loop is made of the recent ones.
        while len(json.dumps(journey, default=str)) > MAX_JOURNEY_CHARS and journey:
            journey.pop(0)
        state: dict[str, Any] = {
            "goal": self.visible_goal(),
            "journey_so_far": [{k: v for k, v in turn.items() if not k.startswith("_")} for turn in journey],
        }
        if self._typed:
            state["typed_so_far"] = list(self._typed)
        if self._previous_screen:
            state["previous_screen"] = self._previous_screen
        state["this_is_the_new_screen"] = screen_for_model(compact)
        return state

    def visible_goal(self) -> str:
        """The brief up to and including the first step whose named tool the run has not called."""
        cut = len(self.goal)
        for name in self._barriers - self._called:
            match = re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", self.goal)
            if match:
                cut = min(cut, match.end())
        if cut == len(self.goal):
            return self.goal
        end = self.goal.find(". ", cut)
        return self.goal[: end + 1 if end != -1 else len(self.goal)] + " The brief goes on after this step."

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
            "min_confidence": self.min_confidence,
            "requests": self.requests,
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


__all__ = ["TypeSafeNavigator", "OPERATIONS", "ACTION_SPACES", "numbered", "quoted_texts",
           "MODEL", "MIN_CONFIDENCE", "TAP_TOOL", "SCROLL_TOOL", "BACK_TOOL", "WAIT_TOOL",
           "INPUT_TOOL", "SCROLL_KINDS", "build_questions", "sketch",
           "candidates", "tool_names"]
