"""The probes that decide whether V12 worked. Rewritten, because the first version measured nothing.

V11 was evaluated on held-out accuracy, looked healthy, and then reached 0 of 19 destinations on a
real device. Accuracy on rows from the same generator cannot separate a model that reads screens from
one that has memorised the generator, so these measure behaviour instead.

The first version of this file got that wrong in four ways, all found in review, and each is worth
recording because they are the standard ways a probe flatters a model:

**It asked the model to contradict its training.** The "should tap" arm built two identical stalls on
one real node — an input the corpus labelled ``no_progress`` 377 times against ``tap`` 3 times. The
model answering ``no_progress`` was *correct*; the 68.8% reported as "ignored the history" was the
model being right, and the 9.4% flip rate was noise above a 0.8% floor. A probe has to be drawn from
the same distribution as training, or it measures disagreement with the data rather than skill.

**Its screen key was the node count.** ``json.dumps([node["n"] for node in nodes])`` is
``["n1","n2",...]`` — a function of length alone, so 637 screens fell into 13 buckets and
"distinct answers per screen" pooled seventeen unrelated screens. The reassuring 2.91 and V11's
damning 1.0 were both measured this way.

**It fed off-distribution histories.** Passing an empty label list made every history entry
``scroll up -> moved``, a shape training never contains. The 0.951 grounding was measured on inputs
the model had never seen.

**It always used the first two nodes.** ``tappable[0], tappable[1]`` is usually navigation chrome.

What is measured now:

``grounded``
    Does it point at a node that exists and can be tapped? V11: 0 of 496. Under the pointing contract
    this should be near 1.0, and if it is not, the model is emitting malformed calls.

``stall_probe``
    One screen, one goal, one difference: whether the stall sits on the node the goal names or on a
    different one. The corpus answers ``no_progress`` 87% of the time for the first and ``tap`` 56% for
    the second, so both arms are in-distribution and the flip is a fair thing to ask for.

``goal_sensitivity``
    Distinct answers for distinct goals on one *actual* screen, keyed by node identity. V11 gave one
    answer for 26 goals.

``refusal``
    Host-capability goals, which no on-device driver can satisfy. The scoring rule gets 4 of 7 wrong
    and this is the main reason to train a model at all.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from experiments.functiongemma import v12_progress as prog
from experiments.functiongemma.v12_contract import tools
from experiments.functiongemma.v12_corpus import (
    HOST_GOALS,
    POLICY,
    _payload,
    load_screens,
)
from experiments.functiongemma.v12_goals import goal_for

MODEL = "LiquidAI/LFM2.5-350M-MLX-bf16"


def _context(
    goal: str, projection: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]], scrolls: int
) -> dict[str, Any]:
    screen: dict[str, Any] = {"package": projection.get("package") or "app", "nodes": list(nodes)}
    if projection.get("more"):
        screen["more"] = True
    if scrolls:
        screen["scrolls"] = scrolls
    step = sum(int(node.get("tried") or 0) for node in nodes) + scrolls
    return {"goal": goal, "step": step, "screen": screen}


def ask(model: Any, tokenizer: Any, context: Mapping[str, Any], schema: Any) -> str:
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": POLICY},
            {
                "role": "user",
                "content": json.dumps(
                    context, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
            },
        ],
        tools=schema,
        tokenize=False,
        add_generation_prompt=True,
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=24, verbose=False)
    return out.replace("<|tool_call_start|>", "").replace("<|tool_call_end|>", "").strip()


def parse(answer: str) -> tuple[str, str]:
    """``(call, argument)`` from an emitted line, or ``("<malformed>", answer)``."""

    if "[" not in answer or "(" not in answer:
        return "<malformed>", answer
    inner = answer.split("[", 1)[1]
    inner = inner.rsplit("]", 1)[0] if "]" in inner else inner
    name = inner.split("(", 1)[0].strip()
    arg = inner.split('="', 1)[1].split('"', 1)[0] if '="' in inner else ""
    return name, arg


def _screen_key(projection: Mapping[str, Any]) -> str:
    """Screen identity by its labels, not its node count. The V11-era key was length alone."""

    return "|".join(
        " ".join(str(node.get(key) or "") for key in ("text", "desc", "rid")).strip()
        for node in projection.get("nodes") or []
    )


def run(
    adapter: Path | None,
    screens: Sequence[Mapping[str, Any]],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    from mlx_lm import load

    model, tokenizer = load(MODEL, adapter_path=str(adapter) if adapter else None)[:2]
    schema = tools()
    rng = random.Random(seed)

    grounded = decisions = malformed = 0
    calls: dict[str, int] = {}
    invented: list[str] = []
    answers_by_screen: dict[str, set[str]] = {}

    def pick(projection: Mapping[str, Any]) -> tuple[list[MutableMapping[str, Any]], Any, str] | None:
        original = list(projection.get("nodes") or [])
        nodes = _payload(projection)
        tappable = [node for node in nodes if node.get("tap")]
        if not tappable:
            return None
        target = tappable[rng.randrange(len(tappable))]
        made = goal_for(target, original, rng)
        return (nodes, target, made[0]) if made else None

    # ---- grounding and goal sensitivity, on in-distribution inputs
    for _ in range(trials):
        projection = screens[rng.randrange(len(screens))]
        chosen = pick(projection)
        if not chosen:
            continue
        nodes, target, goal = chosen
        prog.scatter(nodes, rng, skip=str(target.get("n")))
        scrolls = rng.randrange(0, prog.MAX_SCROLLS + 1) if projection.get("more") else 0
        answer = ask(model, tokenizer, _context(goal, projection, nodes, scrolls), schema)
        call, arg = parse(answer)
        decisions += 1
        calls[call] = calls.get(call, 0) + 1
        if call == "<malformed>":
            malformed += 1
            continue
        if call != "tap":
            grounded += 1  # a non-tap call names no element, so it cannot be ungrounded
        elif arg in {node.get("n") for node in nodes if node.get("tap")}:
            grounded += 1
        else:
            invented.append(arg)
        answers_by_screen.setdefault(_screen_key(projection), set()).add(answer)

    # ---- the stall probe: same screen, same goal, the stall moved
    pairs = flipped = same = 0
    for _ in range(trials):
        projection = screens[rng.randrange(len(screens))]
        chosen = pick(projection)
        if not chosen:
            continue
        nodes, target, goal = chosen
        others = [n for n in nodes if n.get("tap") and n.get("n") != target.get("n")]
        if not others:
            continue
        other = others[rng.randrange(len(others))]
        tried = max(2, prog.sample_tried(rng))

        on_other = [dict(node) for node in nodes]
        for node in on_other:
            if node["n"] == other["n"]:
                prog.annotate(node, tried=tried, last=prog.UNCHANGED)
        on_target = [dict(node) for node in nodes]
        for node in on_target:
            if node["n"] == target["n"]:
                prog.annotate(node, tried=tried, last=prog.UNCHANGED)

        a = parse(ask(model, tokenizer, _context(goal, projection, on_other, 0), schema))
        b = parse(ask(model, tokenizer, _context(goal, projection, on_target, 0), schema))
        pairs += 1
        if a == b:
            same += 1
        if a[0] == "tap" and b[0] == "handoff" and b[1] == "no_progress":
            flipped += 1

    # ---- refusal: goals no on-device driver can satisfy
    host_right = host_total = 0
    for index in range(min(trials, len(HOST_GOALS) * 2)):
        projection = screens[rng.randrange(len(screens))]
        nodes = _payload(projection)
        if not any(node.get("tap") for node in nodes):
            continue
        prog.scatter(nodes, rng)
        goal = HOST_GOALS[index % len(HOST_GOALS)]
        call, arg = parse(ask(model, tokenizer, _context(goal, projection, nodes, 0), schema))
        host_total += 1
        if call == "handoff" and arg == "needs_host":
            host_right += 1

    spread = [len(v) for v in answers_by_screen.values() if len(v) >= 1]
    return {
        "adapter": str(adapter) if adapter else "base",
        "decisions": decisions,
        "grounded": round(grounded / decisions, 4) if decisions else 0.0,
        "malformed": round(malformed / decisions, 4) if decisions else 0.0,
        "calls": dict(sorted(calls.items())),
        "invented_examples": invented[:5],
        "goal_sensitivity": {
            "screens": len(answers_by_screen),
            "distinct_answers_mean": round(sum(spread) / len(spread), 2) if spread else 0.0,
        },
        "stall_probe": {
            "pairs": pairs,
            "identical_answer": same,
            "correctly_flipped": flipped,
            "flip_rate": round(flipped / pairs, 4) if pairs else 0.0,
        },
        "refusal_needs_host": {
            "asked": host_total,
            "right": host_right,
            "rate": round(host_right / host_total, 4) if host_total else 0.0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", nargs="*", type=Path)
    parser.add_argument("--screens", type=Path, default=Path("runs/functiongemma/screens"))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    screens = load_screens(args.screens)
    for adapter in args.adapters or [None]:
        print(json.dumps(run(adapter, screens, args.trials, args.seed), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
