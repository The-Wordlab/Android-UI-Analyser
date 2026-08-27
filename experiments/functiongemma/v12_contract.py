"""The pointing contract: four calls, no free text anywhere.

V11 asked the model to *spell* its target — ``next_step(kind="tap", label="Display & touch Dark
theme, font size, touch")``. Across 16 checkpoints and 496 answers that produced a selector matching
something on screen **0 times**, because a model that writes strings writes the strings it was
trained on. It emitted corpus destinations (``Digest``, ``Chancercy``) at screens that had never
contained them.

So this contract removes the ability. Every argument is either an index out of the list the model was
just shown, or a member of a closed set. There is no field a name can be invented into. That is not
a quality improvement, it is a structural one: the same property is why the scoring rule in
:mod:`drive_rule` grounds at 100% while every trained checkpoint grounded at 0%.

Four calls, and the reason each exists:

``tap(n)``
    Act on a node. ``n`` must be one of the ``n1..n14`` the projection listed and must be tappable.

``scroll(dir)``
    Reveal more screen. The honest move when nothing matches *and* there is reason to believe more
    exists — the projection says so via ``more`` or a scrollable node.

``done()``
    The screen already proves the goal. No arguments: judging arrival needs the screen, which the
    caller has, and a free-text "proof" field is another place to hallucinate.

``handoff(reason)``
    Stop, with one of four reasons. This is half the point of training a model at all: the scoring
    rule cannot tell "absent" from "present under another name", and got 4 of 7 refusals wrong.

What is deliberately absent, and why, because each was in V11 and each cost something:

* **No ``resource_id``/``label``/``content_desc``.** See above. Also: the V11 policy told the model
  to *prefer* ``resource_id``, while only 7.6% of live Settings nodes carry one — the contract asked
  for a field the runtime mostly does not have.
* **No ``assert-visible``/``wait-for``/``assert-not-visible``.** The V11 corpus let history end in an
  assert, and the shortcut gate found ``history_tail == assert-visible -> done`` at precision 1.000.
  Asserting is the caller's job; it holds the goal's success criteria.
* **No ``submit`` flag, no ``input``.** The helper's ``case "input"`` performs ``ACTION_SET_TEXT``
  only, so a model trained to send IME actions would be trained on something the runtime cannot do.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Where a scroll can go. Two directions, because the projection's ``more`` flag only ever means
#: "there is content below", and back-tracking upward is the only other useful move.
DIRECTIONS = ("up", "down")

#: Why the run stops. Closed on purpose — a free-text reason is a free-text hallucination site, and
#: these four are the ones the live run actually produced.
REASONS = (
    # Not on this screen, and scrolling has already been exhausted or is impossible.
    "target_absent",
    # On screen or reachable, but pressing it needs something the device agent does not have:
    # a host capability, a permission dialog, a sign-in.
    "needs_host",
    # Actions are landing and the screen is not changing. The one the model must learn from
    # *outcomes*, never from step count — V11 learned the step count and this is the fix.
    "no_progress",
    # Forward requires authorization a driver must not give itself.
    "needs_auth",
)

CALLS = ("tap", "scroll", "done", "handoff")


def tools() -> list[dict[str, Any]]:
    """The four calls as a tool schema, kept minimal because it is re-sent every single turn.

    At roughly 200 tokens this is a fifth of a typical prompt. Every word here is paid for on every
    decision, so the descriptions say what a call *is* and nothing about when to prefer it — that
    belongs in the policy, which is sent once.
    """

    return [
        {
            "type": "function",
            "function": {
                "name": "tap",
                "description": "Press one of the listed nodes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "string", "description": "A listed node id, e.g. n3."}
                    },
                    "required": ["n"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scroll",
                "description": "Reveal more of the screen.",
                "parameters": {
                    "type": "object",
                    "properties": {"dir": {"type": "string", "enum": list(DIRECTIONS)}},
                    "required": ["dir"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "The screen already satisfies the goal.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "handoff",
                "description": "Stop and give the run back.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string", "enum": list(REASONS)}},
                    "required": ["reason"],
                },
            },
        },
    ]


def render(call: Mapping[str, Any]) -> str:
    """One decision as the transport line the model is trained to emit.

    Argument order is fixed and spacing is exact: a decoder that has seen one spelling learns that
    spelling, and any drift between training and serving shows up as a parse failure rather than as a
    wrong answer, which is much harder to notice.
    """

    name = call["call"]
    if name == "tap":
        return f'[tap(n="{call["n"]}")]'
    if name == "scroll":
        return f'[scroll(dir="{call.get("dir", "up")}")]'
    if name == "done":
        return "[done()]"
    if name == "handoff":
        return f'[handoff(reason="{call["reason"]}")]'
    raise ValueError(f"not a call: {name!r}")


def validate(call: Mapping[str, Any], projection: Mapping[str, Any]) -> None:
    """Reject anything the runtime could not carry out, against the screen it was decided on.

    Validating against the projection rather than against the schema alone is the point: ``tap(n7)``
    is well-formed and still wrong if the screen listed six nodes, or if n7 is a heading. Every such
    row would teach the model that pointing off the end of the list is acceptable, and the corpus is
    the only place to catch it.
    """

    name = call.get("call")
    if name not in CALLS:
        raise ValueError(f"unknown call: {name!r}")

    nodes = list(projection.get("nodes") or [])

    if name == "tap":
        listed = {node.get("n"): node for node in nodes}
        target = listed.get(call.get("n"))
        if target is None:
            raise ValueError(f"tap({call.get('n')!r}) — not one of {sorted(listed)}")
        if not target.get("tap"):
            raise ValueError(f"tap({call['n']!r}) — listed but not tappable")
        return

    if name == "scroll":
        if call.get("dir") not in DIRECTIONS:
            raise ValueError(f"scroll dir {call.get('dir')!r} not in {DIRECTIONS}")
        # Scrolling a screen with nothing to scroll is a wasted step the model would learn to spend.
        if not projection.get("more") and not any(node.get("scroll") for node in nodes):
            raise ValueError("scroll — nothing on this screen scrolls and `more` is false")
        return

    if name == "handoff":
        if call.get("reason") not in REASONS:
            raise ValueError(f"handoff reason {call.get('reason')!r} not in {REASONS}")
        return

    if len(call) > 1:
        raise ValueError(f"done() takes no arguments, got {sorted(set(call) - {'call'})}")
