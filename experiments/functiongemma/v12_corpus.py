"""Builds the V12 corpus: real screens, a pointing contract, and progress written onto the nodes.

Answers four measured V11 failures and one of its own.

**It never named a real element.** 0 of 496 answers across 16 checkpoints produced a selector that
existed on screen. Answered by :mod:`v12_contract`: the model emits an index out of the list it was
shown, so there is nothing to invent.

**It never looked at the screen.** V11's training screens carried 4-10 clean synthetic nodes; real
ones have a median of 41 raw and 10 projected, with labels like "Face detection Use face detection
instead of focus areas". Answered by building every row on one of 637 real screens harvested from
live emulators, in English, Arabic and German.

**It counted steps instead of remembering.** Given "Display" plainly visible, checkpoint 576 refused
seven times out of seven and picked its reason purely by history length.

**And the first fix for that was worse than the problem.** Sending a separate history list to be
joined against the screen produced a corpus where "a history entry names a node above n8" predicted
``no_progress`` at precision 1.000 over 1,004 rows — because history ids were invented from
``randrange(1, 9)`` while real target ids reach ``n14``. The shortcut gate could not see it: no
feature read the number in an id. Worse, the two families meant to share an evidence shape did not,
so the probe testing them was asking the model to contradict its own training data.

:mod:`v12_progress` removes the join. Progress lives on the node — ``"tried": 2, "last": "unchanged"``
— which the helper can compute for free because it already holds every ``AccessibilityNodeInfo``.
``no_progress`` becomes one field on the node the goal names, and the distinction that matters is
carried by ``tap_despite_stalls``: other nodes stalled, the target untried, so **tap it**. Those rows
are what stop "something on this screen stalled" from being enough to refuse.

Deliberately absent: semantic bridging ("make the text bigger" -> "Display"). It needs verified
pairs, the harvest is too shallow on Settings to supply them, and generating them from head-word
matches produced 998 wrong ones on the first attempt. See :mod:`v12_goals`.
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from experiments.functiongemma import v12_progress as prog
from experiments.functiongemma.v12_contract import render, validate
from experiments.functiongemma.v12_goals import (
    ARRIVAL_FRAMES,
    FRAMES,
    absent_goal,
    goal_for,
    label_of,
)

from android_ui_analyser.drive_projection import project

#: Sent once per conversation. It names no field the runtime might not have — the V11 policy told the
#: model to prefer ``resource_id``, which 7.6% of live nodes carry — and it describes ``tried``/``last``
#: as belonging to a node, because that is the reading the whole corpus turns on.
POLICY = (
    "Drive the screen to the goal. One call per turn.\n"
    "You are given numbered nodes. tap only a node in that list.\n"
    "A node may carry tried (how often it was already acted on) and last "
    "(changed, unchanged, blocked). A node with no tried has not been touched.\n"
    "scroll when nothing listed fits and there is more to see; scrolls says how many are spent.\n"
    "done when the listed screen already satisfies the goal.\n"
    "handoff: target_absent when it is not here and there is no more screen; "
    "no_progress when the node the goal needs was already tried and did not move; "
    "needs_host when the goal needs the host, not the screen; "
    "needs_auth when going on would mean granting access."
)

#: Goals no on-device driver can carry out, whatever is on screen. Real host-side AUA capabilities,
#: so the refusal being taught is the one the runtime actually needs.
HOST_GOALS = (
    "take a screenshot of this screen",
    "capture the screen and save it",
    "read the logcat output",
    "check the app logs for errors",
    "query the app's database",
    "dump the sqlite tables",
    "install the new build on this device",
    "push an apk onto the device",
    "set the device clock to midnight",
    "change the system time",
    "turn the network off at the host",
    "put the device offline",
    "start the recording proxy",
    "record the network traffic",
    "record a video of the screen",
    "copy a photo into the gallery",
    "rotate the device to landscape",
    "list the connected devices",
    "restore the database backup",
    "export the last capture",
)

#: How often each family appears. Refusal is nearly half, because refusal is the job: the scoring rule
#: already reaches 17 of 19 destinations and gets 4 of 7 refusals wrong. ``tap`` keeps the largest
#: single share regardless — a policy that cannot act is not useful for refusing either.
#:
#: ``tap_despite_stalls`` is weighted heavily for its size. It is the only family that makes "a node
#: on this screen stalled" insufficient grounds to refuse, and without it the model can answer
#: ``no_progress`` by pattern-matching the presence of a stalled node anywhere.
FAMILY_WEIGHTS = (
    ("tap", 22),
    ("tap_despite_stalls", 14),
    ("scroll", 11),
    ("done", 6),
    ("absent", 13),
    ("host", 11),
    ("no_progress", 13),
    ("needs_auth", 5),
    ("decline", 5),
)

#: The refusal control on a permission dialog. Tapping one grants nothing, which is why a driver may
#: press it and why the ``decline`` family exists — it is also what stops "a dialog is up" from
#: predicting ``needs_auth`` on its own.
_DECLINE_LABELS = frozenset({"don't allow", "dont allow", "deny", "no thanks", "not now", "cancel"})


def script_of(text: str) -> str:
    """Which writing system a string is mostly in.

    The harvest deliberately includes the same screens in English, Arabic and German, and goals for
    absent targets were once drawn from every label regardless of script. That put German goals on
    Arabic screens and made script mismatch worth a 2x shift — ``target_absent`` plus ``scroll`` went
    from 22.4% of matched rows to 44.4% of mismatched ones, while ``tap`` fell from 43% to 24%.

    Nothing real causes that. It is an artefact of the sampling pool, and no shortcut-gate feature
    could ever have seen it — the standing reminder that a clean gate is necessary, not sufficient.
    """

    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        key = "arabic" if "ARABIC" in name else "cjk" if "CJK" in name else "latin"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda pair: pair[1])[0]


def load_screens(root: Any) -> list[dict[str, Any]]:
    """Every harvested screen, projected once.

    Projection is the expensive part and deterministic, so it happens here rather than per row.
    ``locale.jsonl`` names its fields ``resource_id``/``content_desc`` while the others use
    ``rid``/``desc``; normalising that was worth a correction once already.
    """

    out: list[dict[str, Any]] = []
    for name in ("aosp", "apps", "locale"):
        path = root / f"{name}.jsonl"
        if not path.is_file():
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            elements = [
                {
                    **element,
                    "rid": element.get("rid", element.get("resource_id")),
                    "desc": element.get("desc", element.get("content_desc")),
                }
                for element in (row.get("elements") or [])
            ]
            projection = project(elements)
            if not projection["nodes"]:
                continue
            projection["source"] = name
            projection["package"] = row.get("package") or "app"
            projection["script"] = script_of(
                " ".join(
                    str(node.get("text") or node.get("desc") or "")
                    for node in projection["nodes"]
                )
            )
            out.append(projection)
    return out


def _can_scroll(projection: Mapping[str, Any]) -> bool:
    return bool(projection.get("more")) or any(
        node.get("scroll") for node in projection.get("nodes") or []
    )


def _is_auth(projection: Mapping[str, Any]) -> bool:
    for node in projection.get("nodes") or []:
        if node.get("tap") and label_of(node).strip().lower() in _DECLINE_LABELS - {"cancel"}:
            return True
    labels = " ".join(label_of(node) for node in projection.get("nodes") or []).lower()
    return "don't allow" in labels or "dont allow" in labels


def _payload(projection: Mapping[str, Any]) -> list[MutableMapping[str, Any]]:
    """A fresh, mutable copy of the screen's nodes, ready to be annotated.

    Copied per row because progress is written onto them: annotating the shared projection would leak
    one row's ``tried`` counts into every later row built on the same screen.
    """

    nodes: list[MutableMapping[str, Any]] = []
    for node in projection.get("nodes") or []:
        entry: MutableMapping[str, Any] = {"n": node.get("n")}
        for key in ("text", "desc", "rid"):
            if node.get(key):
                entry[key] = node[key]
        if node.get("tap"):
            entry["tap"] = True
        if node.get("scroll"):
            entry["scroll"] = True
        nodes.append(entry)
    return nodes


def _row(
    goal: str,
    projection: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    scrolls: int,
    call: Mapping[str, Any],
    family: str,
    style: str,
) -> dict[str, Any]:
    screen: dict[str, Any] = {
        "package": projection.get("package") or "app",
        "nodes": list(nodes),
    }
    if projection.get("more"):
        screen["more"] = True
    if scrolls:
        screen["scrolls"] = scrolls
    validate(call, {"nodes": nodes, "more": projection.get("more")})
    context = {"goal": goal, "step": _step(nodes, scrolls), "screen": screen}
    return {
        "messages": [
            {"role": "system", "content": POLICY},
            {
                "role": "user",
                "content": json.dumps(
                    context, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
            },
            {"role": "assistant", "content": f"<|tool_call_start|>{render(call)}<|tool_call_end|>"},
        ],
        "meta": {"family": family, "style": style, "source": projection.get("source")},
    }


def _step(nodes: Sequence[Mapping[str, Any]], scrolls: int) -> int:
    """Steps taken so far, derived from the node counts rather than chosen.

    Derived on purpose: V11 sent ``step`` as an independent field and the model read it instead of the
    screen — checkpoint 576 chose its refusal reason from it alone. Here it is the sum of what the
    nodes already say, so it carries nothing they do not.
    """

    return sum(int(node.get("tried") or 0) for node in nodes) + scrolls


def build_row(
    family: str,
    screens: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[str]],
    rng: random.Random,
) -> dict[str, Any] | None:
    """One training row of *family*, or ``None`` when the sampled screen cannot support it.

    Returning ``None`` rather than forcing a row is why the ground truth holds: a screen with nothing
    to scroll cannot teach scrolling, and a screen whose only match is ambiguous cannot teach tapping.
    """

    projection = screens[rng.randrange(len(screens))]
    original = list(projection.get("nodes") or [])
    nodes = _payload(projection)
    tappable = [node for node in nodes if node.get("tap")]
    if not tappable:
        return None
    pool = labels.get(projection.get("script") or "none") or labels["all"]

    def pick_goal() -> tuple[Mapping[str, Any], str, str] | None:
        target = tappable[rng.randrange(len(tappable))]
        made = goal_for(target, original, rng)
        return (target, made[0], made[1]) if made else None

    # ---------------------------------------------------------------- act on what is here
    if family == "tap":
        chosen = pick_goal()
        if not chosen:
            return None
        target, goal, style = chosen
        # Other nodes may already be tried, with any outcome. The target is untouched.
        prog.scatter(nodes, rng, skip=str(target.get("n")))
        scrolls = rng.randrange(0, prog.MAX_SCROLLS + 1) if _can_scroll(projection) else 0
        return _row(
            goal, projection, nodes, scrolls, {"call": "tap", "n": target.get("n")}, family, style
        )

    if family == "tap_despite_stalls":
        chosen = pick_goal()
        if not chosen:
            return None
        target, goal, style = chosen
        others = [node for node in tappable if node.get("n") != target.get("n")]
        if not others:
            return None
        # One other node, repeatedly tried, stalled. This is the exact evidence shape that must NOT
        # mean "refuse": the goal's own node has never been touched, so tapping it is the answer.
        # Without these rows, "some node stalled" is sufficient to refuse and the model never has to
        # work out which node the goal is about.
        stalled_node = others[rng.randrange(len(others))]
        prog.annotate(
            stalled_node,
            tried=max(2, prog.sample_tried(rng)),
            last=prog.STALLED[rng.randrange(len(prog.STALLED))],
        )
        prog.scatter(nodes, rng, skip=str(target.get("n")), count=rng.randrange(0, 2))
        scrolls = rng.randrange(0, prog.MAX_SCROLLS + 1) if _can_scroll(projection) else 0
        return _row(
            goal, projection, nodes, scrolls, {"call": "tap", "n": target.get("n")}, family, style
        )

    # ---------------------------------------------------------------- look further
    if family == "scroll":
        if not _can_scroll(projection):
            return None
        goal = absent_goal(original, pool, rng)
        if not goal:
            return None
        prog.scatter(nodes, rng)
        return _row(
            goal,
            projection,
            nodes,
            rng.randrange(0, prog.MAX_SCROLLS),
            {"call": "scroll", "dir": "up"},
            family,
            "absent",
        )

    # ---------------------------------------------------------------- already there
    if family == "done":
        target = tappable[rng.randrange(len(tappable))]
        made = goal_for(target, original, rng, frames=ARRIVAL_FRAMES)
        if not made:
            return None
        goal, style = made
        prog.scatter(nodes, rng)
        scrolls = rng.randrange(0, prog.MAX_SCROLLS + 1) if _can_scroll(projection) else 0
        return _row(goal, projection, nodes, scrolls, {"call": "done"}, family, style)

    # ---------------------------------------------------------------- refuse: not here
    if family == "absent":
        # Half, not a third. `done` is 6% of the corpus and this family 13%, so an even split is what
        # makes an arrival phrasing carry no information: at 30% the gate found
        # `goal_first_word == "check"` predicting `done` at precision 0.987.
        arrival = rng.random() < 0.5
        goal = absent_goal(
            original, pool, rng, frames=ARRIVAL_FRAMES if arrival else FRAMES
        )
        if not goal:
            return None
        prog.scatter(nodes, rng)
        # Refusing while there is more screen left is only correct once scrolling is spent, so these
        # rows say so. On a screen that cannot scroll, no count is needed.
        # Scrolling must be spent before refusing is correct, so this is the maximum. Every other
        # family also reaches the maximum (see `MAX_SCROLLS + 1` above) — when this family alone did,
        # `scrolls == 3` predicted target_absent at precision 1.000 over 45% of the class.
        scrolls = prog.MAX_SCROLLS if _can_scroll(projection) else 0
        return _row(
            goal,
            projection,
            nodes,
            scrolls,
            {"call": "handoff", "reason": "target_absent"},
            family,
            "absent-arrival" if arrival else "absent",
        )

    # ---------------------------------------------------------------- refuse: not the screen's job
    if family == "host":
        # English-only goals, so restricting them to Latin screens stops script mismatch from
        # predicting this class. 599 of 637 harvested screens are Latin, so it costs almost nothing.
        if (projection.get("script") or "latin") != "latin":
            return None
        goal = HOST_GOALS[rng.randrange(len(HOST_GOALS))]
        prog.scatter(nodes, rng)
        scrolls = rng.randrange(0, prog.MAX_SCROLLS + 1) if _can_scroll(projection) else 0
        return _row(
            goal, projection, nodes, scrolls, {"call": "handoff", "reason": "needs_host"}, family, "host"
        )

    # ---------------------------------------------------------------- refuse: tried, nothing moved
    if family == "no_progress":
        chosen = pick_goal()
        if not chosen:
            return None
        target, goal, style = chosen
        # The stall is on the goal's own node. That single field is the whole distinction from
        # tap_despite_stalls, and both families reach every `tried` count and every `scrolls` value.
        for node in nodes:
            if node.get("n") == target.get("n"):
                prog.annotate(
                    node,
                    tried=prog.sample_tried(rng),
                    last=prog.STALLED[rng.randrange(len(prog.STALLED))],
                )
        prog.scatter(nodes, rng, skip=str(target.get("n")))
        scrolls = rng.randrange(0, prog.MAX_SCROLLS + 1) if _can_scroll(projection) else 0
        return _row(
            goal,
            projection,
            nodes,
            scrolls,
            {"call": "handoff", "reason": "no_progress"},
            family,
            style,
        )

    # ---------------------------------------------------------------- refuse: needs permission
    if family == "needs_auth":
        if not _is_auth(projection):
            return None
        # An ordinary navigation goal, phrased like every other one. An earlier version gave this
        # family its own vocabulary — "proceed", "continue past this" — and the gate found that the
        # goal's first word alone predicted the class at precision 1.000.
        goal = absent_goal(original, pool, rng)
        if not goal:
            return None
        prog.scatter(nodes, rng)
        return _row(
            goal, projection, nodes, 0, {"call": "handoff", "reason": "needs_auth"}, family, "auth"
        )

    # ---------------------------------------------------------------- decline, which is allowed
    if family == "decline":
        if not _is_auth(projection):
            return None
        target = None
        for node in tappable:
            if label_of(node).strip().lower() in _DECLINE_LABELS:
                target = node
                break
        if target is None:
            return None
        # A dialog does not mean "always stop". Refusing it is a legitimate action, and these rows are
        # what stop the screen alone from predicting needs_auth.
        goal = rng.choice(
            (
                "decline this request",
                "turn this permission down",
                "say no to this",
                "refuse it and move on",
                "dismiss this without granting anything",
            )
        )
        prog.scatter(nodes, rng, skip=str(target.get("n")))
        return _row(
            goal, projection, nodes, 0, {"call": "tap", "n": target.get("n")}, family, "decline"
        )

    raise ValueError(f"unknown family: {family!r}")


def answer_of(row: Mapping[str, Any]) -> str:
    """The full decision class, reason included.

    Reason included because leaving it out is how V11's actual failure survived a rebuild: balancing
    and auditing on the call alone left ``needs_host`` at 51% of refusals when nothing had been tried
    and ``no_progress`` at 40% once several had — checkpoint 576's learned mapping, reproduced, under
    a reported spread of 0.0001.
    """

    text = row["messages"][-1]["content"]
    inner = text.split("[", 1)[-1].rsplit("]", 1)[0]
    name = inner.split("(", 1)[0]
    if name == "handoff":
        return f"handoff:{inner.split('reason=', 1)[-1].strip(chr(34) + ')')}"
    return name


def _touched_band(row: Mapping[str, Any]) -> int:
    """How much has already happened, as the model can see it. The balancing axis."""

    context = json.loads(row["messages"][1]["content"])
    return min(int(context.get("step") or 0), 5)


def balance(rows: Sequence[Mapping[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Drop rows until every progress level carries the same mix of answers, reasons included.

    V11's model chose its refusal *reason* by how much had happened. The first V12 build balanced the
    call and not the reason, so that mapping was still in the corpus while the manifest reported a
    spread of 0.0001. This balances on the full class.

    **Level zero is excluded, and that is a real distinction rather than a convenience.**
    ``no_progress`` means the node the goal names was already tried; at zero progress nothing has
    been, so the class does not exist there. Demanding a flat mix across zero would delete the class
    corpus-wide — the starvation guard below caught exactly that. What must be flat is the mix *given
    that something has happened*, because that is where V11's counting lived: it read "more has
    happened" as "give up". Learning that a refusal for no progress is impossible before any attempt
    is correct, not a shortcut, and level zero stays at 13% of the corpus so the regime is well
    represented.

    Trimming rather than generating extra is the cheaper direction: rows cost nothing to make, and
    over-generating one (level, answer) cell means sampling screens non-uniformly, which trades a
    progress bias for a screen bias.
    """

    untouched: list[Mapping[str, Any]] = []
    buckets: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        level = _touched_band(row)
        if level == 0:
            untouched.append(row)
            continue
        buckets.setdefault(level, {}).setdefault(answer_of(row), []).append(row)
    if not buckets:
        return list(rows)

    classes = sorted({name for by_class in buckets.values() for name in by_class})
    share: dict[str, float] = {}
    for name in classes:
        share[name] = min(
            len(by_class.get(name, ())) / sum(len(v) for v in by_class.values())
            for by_class in buckets.values()
        )
    total = sum(share.values())
    if total <= 0:
        return list(rows)

    # A class missing from any single level gets share 0 and would be deleted corpus-wide. That
    # happened to `done` once and every check still reported clean. Silent truncation is the one
    # failure a balancer must not have.
    starved = [name for name in classes if share[name] <= 0]
    if starved:
        raise ValueError(
            f"balance would delete {starved} entirely: absent from at least one progress level "
            f"above zero. Generate those rows at every level, or exclude them from balancing."
        )

    kept: list[dict[str, Any]] = []
    for by_class in (buckets[level] for level in sorted(buckets)):
        size = min(
            int(len(by_class.get(name, ())) / (share[name] / total)) for name in classes
        )
        for name in classes:
            candidates = list(by_class.get(name, ()))
            rng.shuffle(candidates)
            kept.extend(candidates[: int(size * share[name] / total)])  # type: ignore[arg-type]
    kept.extend(untouched)  # type: ignore[arg-type]
    rng.shuffle(kept)
    return kept


def build(
    screens: Sequence[Mapping[str, Any]],
    count: int,
    seed: int = 0,
    *,
    do_balance: bool = True,
) -> list[dict[str, Any]]:
    """*count* rows, families in proportion to :data:`FAMILY_WEIGHTS`.

    Families are drawn from a running deficit rather than sampled independently, because a family
    that only fits rare screens — ``needs_auth`` needs a permission dialog — would otherwise be
    squeezed out by its own rejection rate. A V11 run shipped with zero refusal rows for exactly this.
    """

    rng = random.Random(seed)
    every = sorted(
        {
            label_of(node)
            for projection in screens
            for node in projection.get("nodes") or []
            if node.get("tap") and label_of(node)
        }
    )
    labels: dict[str, list[str]] = {"all": every}
    for text in every:
        labels.setdefault(script_of(text), []).append(text)

    total = sum(weight for _, weight in FAMILY_WEIGHTS)
    want = {name: round(count * weight / total) for name, weight in FAMILY_WEIGHTS}
    made: dict[str, int] = dict.fromkeys(want, 0)

    rows: list[dict[str, Any]] = []
    attempts = 0
    limit = count * 80
    while len(rows) < count and attempts < limit:
        attempts += 1
        short = [name for name in want if made[name] < want[name]]
        if not short:
            break
        family = short[rng.randrange(len(short))]
        row = build_row(family, screens, labels, rng)
        if row is None:
            continue
        rows.append(row)
        made[family] += 1
    rng.shuffle(rows)
    return balance(rows, rng) if do_balance else rows
