"""The V11 decision contract: one on-device step, in AUA's own step language.

V10 trained a *chooser*. AUA compiled 2-4 pre-approved calls and the model returned an opaque
``candidate_id``. That model cannot drive anything: it never authors a selector, so it can only
answer a question the host has already almost answered. It also cannot live on the device, because
the candidate list is built host-side from a host-side observation.

V11 trains a *driver*. It reads one screen and emits one step, and the step it emits is a
:class:`android_ui_analyser.memory.RouteStep` — the shared step model AUA already uses for routes,
flows, and the on-device helper's ``flow.run``. Nothing new is invented: the helper's
``FlowFeature.runStep`` already executes exactly these kinds, and ``flows.py`` already validates
them. That is the point. A model that emits RouteSteps is immediately executable by the helper APK
with no translation layer, which is what makes a fully on-device loop possible.

Three things this module pins down, because getting any of them wrong produces training data the
helper silently rejects:

**Exactly one selector.** This is a restriction *this corpus imposes*, not one the host enforces —
an earlier version of this docstring claimed ``flows.py`` errors on two selectors and that was
simply wrong. What ``flows.py`` actually enforces for element steps is *at least* one
(``if not (resource_id or content_desc or label): raise``), and auto-recorded ``RouteStep``s
deliberately carry a resource id *and* a label. The exactly-one rule is kept anyway because the
helper's ``match()`` is an else-if chain — resource id, then label, then description — so a second
selector is silently ignored rather than combined, and a model that emits two has expressed an
intent the device will not honour. Emitting none matches nothing at all.

**Action steps match on equality; predicate steps match on containment.** ``tap``/``long-press``/
``input``/``clear`` match a named selector field exactly. ``wait-for``/``assert-visible``/
``assert-not-visible``/``scroll-to`` instead carry the query in ``arg`` with a ``by`` field and
match on *contains*. The helper's own comment calls this out as a distinction that must stay:
"Matching those on ``label`` with equality made the same step mean two different things depending
on where it ran."

**``by`` has a closed vocabulary of three.** ``text`` (which searches the content description too,
matching the host's ``_BY_FIELDS["text"] == ["text", "description"]``), ``desc``, and ``rid``/``id``.
The helper raises on anything else rather than degrading to a text search.

The device lane is not the whole AUA surface. An accessibility service cannot start a host
mitmproxy, copy a database through ``run-as``, install an APK, or boot an emulator. Those stay
host-side, and a goal that needs one is a handoff, not a step. ``DEVICE_KINDS`` is therefore
deliberately smaller than ``flows._KINDS``.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- step vocabulary

#: Step kinds the on-device helper can execute itself (``FlowFeature.runStep``).
#: Verified against helper/app/src/main/java/dev/aua/helper/FlowFeature.java.
DEVICE_ACTION_KINDS = (
    "tap",
    "long-press",
    "input",
    "clear",
    "key",
    "tap-point",
    "swipe",
    "scroll",
    "hide-keyboard",
    "paste",
)

#: Kinds whose target is a *predicate* in ``arg`` + ``by``, matched on containment.
DEVICE_PREDICATE_KINDS = (
    "wait-for",
    "assert-visible",
    "assert-not-visible",
    "scroll-to",
)

#: Kinds that take no target at all.
DEVICE_BARE_KINDS = ("wait-stable", "hide-keyboard", "paste")

DEVICE_KINDS = tuple(sorted(set(DEVICE_ACTION_KINDS + DEVICE_PREDICATE_KINDS + ("wait-stable",))))

#: Kinds that select a single element by an exact-match selector field.
ELEMENT_KINDS = ("tap", "long-press", "input", "clear")

#: The three selector fields, in the helper's own priority order.
SELECTOR_FIELDS = ("resource_id", "label", "content_desc")

#: ``by`` values the helper accepts. Anything else raises there, so never emit anything else.
BY_VALUES = ("text", "desc", "rid", "id")

#: Directions accepted by ``swipe``, ``scroll`` and ``scroll-to``.
DIRECTIONS = ("up", "down", "left", "right")

#: Global actions reachable through ``key`` from an accessibility service. An arbitrary keycode
#: needs input injection, which the helper explicitly cannot do.
KEY_NAMES = ("back", "home", "recents")

#: Why a driver stops without acting. ``target_absent`` and ``no_progress`` are relevance
#: refusals — the majority case V10's corpus got wrong by confounding refusal with destructiveness.
HANDOFF_REASONS = (
    "target_absent",
    "no_progress",
    "needs_host_lane",
    "needs_authorization",
    "ambiguous_target",
)

#: Host-only capabilities. A goal that needs one of these is `needs_host_lane`, never a step.
HOST_ONLY_CAPABILITIES = (
    "proxy",
    "database",
    "install",
    "emulator",
    "flags",
    "logcat",
    "clipboard_host",
    "network_profile",
)


# --------------------------------------------------------------------------- tool definitions


def tools() -> list[dict[str, Any]]:
    """The three functions a V11 driver may call, as compact JSON-schema definitions.

    Deliberately terse. LFM2.5's chat template inlines the whole tool list into the system turn of
    *every* prompt, so prose here is paid for on every single decision — in training tokens, and
    then again in on-device latency for the rest of the model's life. A first draft with full
    descriptions cost 569 tokens per row against 133 tokens of actual screen: the schema was 57% of
    the input and the question was 13% of it.

    Prose is also the part a fine-tuned model needs least. Forty thousand labelled examples teach
    the vocabulary far more precisely than a sentence can, so what stays is the part that is real
    information — the closed enums — and what goes is the part that merely restates them. The
    contract in ``DRIVER_POLICY`` carries the few rules that are not inferable from the enums.

    Three named functions rather than one with a mode argument, so a terminal decision cannot be
    expressed as a malformed step.
    """

    return [
        {
            "type": "function",
            "function": {
                "name": "next_step",
                "description": "Run one step on the device.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(DEVICE_KINDS)},
                        "resource_id": {"type": "string"},
                        "label": {"type": "string"},
                        "content_desc": {"type": "string"},
                        "arg": {"type": "string"},
                        "by": {"type": "string", "enum": list(BY_VALUES)},
                        "direction": {"type": "string", "enum": list(DIRECTIONS)},
                        "text": {"type": "string"},
                    },
                    "required": ["kind"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "The screen proves the goal is reached.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "handoff",
                "description": "Return control to the host.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "enum": list(HANDOFF_REASONS)},
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


# --------------------------------------------------------------------------- validation


class ContractError(ValueError):
    """A step that the helper or the host flow parser would reject."""


def validate_step(step: dict[str, Any]) -> None:
    """Raise :class:`ContractError` unless *step* is executable by the on-device helper.

    This is the gate that keeps the corpus honest. Every generated label passes through it, so a
    generator bug becomes a build failure instead of a model that learns to emit rejected steps.
    """

    kind = step.get("kind")
    if kind not in DEVICE_KINDS:
        raise ContractError(f"kind {kind!r} is not executable on the device")

    selectors = [field for field in SELECTOR_FIELDS if step.get(field)]
    arg = step.get("arg")
    by = step.get("by")

    direction = step.get("direction")
    if direction is not None:
        # The helper reads `direction` only in `case "scroll-to"`; anywhere else it is dead weight
        # that would mislead a reader of the flow into thinking the step is directional.
        if kind != "scroll-to":
            raise ContractError(f"{kind} takes no direction")
        if direction not in DIRECTIONS:
            raise ContractError(f"direction={direction!r} is not one of {DIRECTIONS}")

    if kind in DEVICE_PREDICATE_KINDS:
        # Predicate kinds carry their target in `arg`; the helper's match() checks `arg` first and
        # only falls through to the selector fields when it is empty.
        if not arg:
            raise ContractError(f"{kind} needs its query in arg")
        if by is not None and by not in BY_VALUES:
            raise ContractError(f"by={by!r} is not one of {BY_VALUES}")
        if selectors:
            raise ContractError(f"{kind} takes arg/by, not a {selectors[0]} selector")
        return

    if step.get("submit"):
        # The helper's `case "input"` performs only ACTION_SET_TEXT: no IME action, no focus step,
        # and the performAction result is not even checked. A `submit` would be silently dropped and
        # the following wait-for would time out, so the field is refused rather than ignored.
        raise ContractError(
            "submit is not executable on the device; the helper never fires the IME"
        )
    if kind in ELEMENT_KINDS:
        if len(selectors) != 1:
            raise ContractError(
                f"{kind} needs exactly one of {SELECTOR_FIELDS}, got {len(selectors)}"
            )
        if by is not None:
            raise ContractError(f"{kind} matches its selector exactly; by is for predicates")
        if kind == "input" and not step.get("text"):
            raise ContractError("input needs text")
        if kind != "input" and step.get("text"):
            raise ContractError(f"{kind} must not carry a typed value")
        return

    # Remaining kinds are argument-only or bare.
    if selectors:
        raise ContractError(f"{kind} does not take a selector")
    if kind == "key":
        if arg not in KEY_NAMES:
            raise ContractError(f"key arg must be one of {KEY_NAMES}, got {arg!r}")
        return
    if kind in ("swipe", "scroll"):
        if arg not in DIRECTIONS:
            raise ContractError(f"{kind} arg must be one of {DIRECTIONS}, got {arg!r}")
        return
    if kind == "tap-point":
        if not arg or "," not in str(arg):
            raise ContractError("tap-point needs an 'x,y' arg")
        return
    if kind in DEVICE_BARE_KINDS:
        if arg:
            raise ContractError(f"{kind} takes no arg")
        return
    raise ContractError(f"unhandled kind {kind!r}")


def render_call(name: str, arguments: dict[str, Any]) -> str:
    """Render one decision as the Pythonic tool call LFM2.5 was trained to emit.

    LFM2.5's chat template ignores ``tool_calls`` entirely and raises on ``content: None``, so the
    label has to be a content string. Argument order follows the declaration order in
    :func:`tools` so the surface form is deterministic for a given decision.
    """

    order = (
        "kind",
        "resource_id",
        "label",
        "content_desc",
        "arg",
        "by",
        "direction",
        "text",
        "reason",
    )
    parts: list[str] = []
    for key in order:
        if key not in arguments:
            continue
        value = arguments[key]
        if value is None:
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={'True' if value else 'False'}")
        else:
            parts.append(f'{key}="{value}"')
    unknown = set(arguments) - set(order)
    if unknown:
        raise ContractError(f"unknown argument(s) for {name}: {sorted(unknown)}")
    return f"<|tool_call_start|>[{name}({', '.join(parts)})]<|tool_call_end|>"
