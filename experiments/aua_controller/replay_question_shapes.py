"""Replay saved QA screens through the Decisions navigator in both question shapes.

Each sampled screen is one the run actually reached, with the screenshot AUA captured and the
journey the navigator would have had there, rebuilt by replaying the run's own actions in order.
The label is the move the run made on that screen. Only rows whose verdict was PASS or
PASS_WITH_WARNING are used, so the labels come from runs that got where they were going.

    aua config exec --env-file .env --require OPENAI_API_KEY -- \\
        python -m experiments.aua_controller.replay_question_shapes --artifacts ../qa-repo/artifacts --out /tmp/ab

It spends real Decisions requests: one per sampled screen per shape.

Measured 2026-10-14 with GPT-6 Luna (seed 7, 800 screens sampled, 643 replayable, full action space,
gate 0.85): the split questions acted correctly on 198 screens and wrongly on 10; the flat list on
134 and 9. Split typed a quoted text right 44 times to flat's 12. The navigator keeps split.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.aua_controller.bounded_navigator import (
    BACK_TOOL,
    FIELD_SUFFIX,
    INPUT_TOOL,
    OPERATIONS,
    READABLE,
    SCROLL_TOOL,
    TAP_TOOL,
    WAIT_TOOL,
    Choice,
    move_phrase,
    numbered,
)
from experiments.aua_controller.decisions_navigator import DecisionsNavigator
from experiments.aua_controller.judgement import frame_screenshot, screenshot_index
from experiments.aua_controller.run_realapp import CONTROLLER_TOOLS

GOOD = {"PASS", "PASS_WITH_WARNING"}
READS = {"analyze_screen", "session_progress", "expect_and_analyze", "app_status", "has"}


class Replay(DecisionsNavigator):
    """Asks only on the sampled screens; every other screen is bookkeeping, as in a live run."""
    ask_now = False

    async def _ask(self, compact, questions):
        return await super()._ask(compact, questions) if self.ask_now else None


class FlatReplay(Replay):
    """The single list the navigator asked before 2026-09-29, with the whole brief.

    Every move is one option -- each control, each quoted text into each field, and each other
    operation the screen allows -- and its one confidence gates both the move and its operand.
    The answer is unfolded into the split answers, so gating and planning are the navigator's own.
    """

    async def _ask(self, compact, questions):
        menu = numbered(self._options)[0]
        criteria = {index: move_phrase(label) for index, label in menu.items()}
        for index, label in menu.items():
            if label.endswith(FIELD_SUFFIX):
                for number, text in enumerate(self.texts, start=1):
                    criteria[f"type:{index}:{number}"] = f"Type `{text}` into the text field '{label[: -len(FIELD_SUFFIX)]}'"
        # The operations the split question offers on this screen, so both shapes allow the same moves.
        for kind in questions["operation"].criteria:
            if kind not in ("press", "type"):
                criteria[kind] = OPERATIONS[kind]
        flat = {"move": Choice(instructions="What should happen next on this screen to carry on with `goal`?",
                               criteria=criteria)}
        asked = await super()._ask(compact, flat)
        if asked is None:
            return None
        turn, answers = asked
        return turn, unfold(answers["move"])


class BoundsReplay(Replay):
    """The split questions, with each element's pixel bounds and the screen size, as the chat model gets them."""

    def _state(self, compact):
        state = super()._state(compact)
        observation = (compact or {}).get("observation") or {}
        screen = observation.get("screen") or {}
        elements = []
        for element in observation.get("elements") or []:
            kept = {key: element[key] for key in READABLE if element.get(key) not in (None, "")}
            if kept and element.get("bounds"):
                kept["bounds"] = element["bounds"]
            if kept:
                elements.append(kept)
        state["this_is_the_new_screen"] = {**state["this_is_the_new_screen"], "elements": elements,
                                           "screen_size": [screen.get("width"), screen.get("height")]}
        return state


VARIANTS = {"split": Replay, "flat": FlatReplay, "bounds": BoundsReplay}


def unfold(move) -> dict:
    """A flat answer as the operation, control and text answers the split questions give."""
    def answer(choice):
        return SimpleNamespace(choice=choice, confidence=move.confidence)

    choice = str(move.choice)
    if choice.isdigit():
        return {"operation": answer("press"), "target": answer(choice)}
    if choice.startswith("type:"):
        _, field, number = choice.split(":")
        return {"operation": answer("type"), "target": answer(field), "text": answer(number)}
    return {"operation": answer(choice), "target": answer("")}


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def center(bounds: Any) -> tuple[float, float] | None:
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        return (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    return None


def inside(point: Any, bounds: Any) -> bool:
    return (point is not None and isinstance(bounds, (list, tuple)) and len(bounds) == 4
            and bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3])


def label_of(call: dict, observation: dict) -> dict | None:
    """The move the run made on this screen, in the navigator's vocabulary; None for a read."""
    tool, arguments = call.get("tool"), call.get("arguments") or {}
    elements = {e.get("id"): e for e in observation.get("elements") or [] if isinstance(e, dict)}
    if tool in READS:
        return None
    if tool in (TAP_TOOL, INPUT_TOOL):
        element = elements.get(arguments.get("id"))
        bounds = (element or {}).get("bounds") or (call.get("resolved_target") or {}).get("bounds")
        if not bounds:
            return None
        if tool == INPUT_TOOL:
            return {"kind": "type", "bounds": bounds, "text": arguments.get("text")}
        return {"kind": "press", "bounds": bounds}
    if tool in (WAIT_TOOL, "await_and_analyze", "wait_stable_and_analyze", "wait_changed_and_analyze"):
        return {"kind": "wait"}
    if tool == BACK_TOOL or (tool == "key_and_analyze" and "BACK" in str(arguments.get("key", "")).upper()):
        return {"kind": "back"}
    if tool in (SCROLL_TOOL, "swipe_and_analyze"):
        direction = str(arguments.get("direction", "")).lower()
        return {"kind": "scroll_" + direction if direction in ("up", "down") else "scroll"}
    if tool == "session_finish":
        return {"kind": "finished"}
    return {"kind": "other"}


def matches(proposal: dict, label: dict, observation: dict) -> bool:
    kind = proposal.get("kind")
    if kind != label["kind"]:
        return False
    if kind == "press":
        chosen = next((e for e in observation.get("elements") or [] if e.get("id") == proposal.get("target_id")), {})
        return inside(center(label["bounds"]), chosen.get("bounds")) or inside(center(chosen.get("bounds")), label["bounds"])
    if kind == "type":
        return str(proposal.get("operand", "")).removeprefix("type:") == label.get("text")
    return True


def rows(artifacts: Path):
    """Every scenario run with a good verdict and a controller transcript."""
    for calls in sorted(artifacts.glob("**/harness/controller/tool-calls.jsonl")):
        controller = calls.parent
        harness = controller.parent
        try:
            verdict = json.loads((harness.parent / "result.json").read_text(encoding="utf-8")).get("verdict")
            goal = json.loads((harness / "result.json").read_text(encoding="utf-8")).get("goal")
        except (OSError, ValueError, AttributeError):
            continue
        if verdict in GOOD and isinstance(goal, str) and goal:
            yield controller, goal


def plan_row(controller: Path) -> list[dict]:
    """The run's steps in order: the screen each started on, its actions, and its label."""
    index = screenshot_index(controller.parent / "aua/manifest.json")
    calls = [c for c in read_jsonl(controller / "tool-calls.jsonl") if c.get("executed") is True]
    steps: dict[int, list[dict]] = {}
    for call in calls:
        steps.setdefault(call.get("step"), []).append(call)
    latest, out = "E0000", []
    for step in sorted(s for s in steps if isinstance(s, int)):
        try:
            saved = json.loads((controller / "evidence" / f"{latest}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        # Evidence is a whole tool result; the first frame of a run is a bare observation.
        frame = saved if isinstance(saved, dict) and isinstance(saved.get("observation"), dict) else \
            {"observation": saved} if isinstance(saved, dict) and "elements" in saved else None
        observation = frame["observation"] if frame else None
        label = label_of(steps[step][0], observation) if observation else None
        shot = frame_screenshot(frame, index) if frame else None
        out.append({"step": step, "frame": frame, "observation": observation, "calls": steps[step],
                    "label": label if shot else None, "evidence": latest})
        for call in steps[step]:
            if isinstance(call.get("evidence_ref"), str):
                latest = call["evidence_ref"]
    return out


def offered_tools(controller: Path) -> list[str]:
    first = next(iter(read_jsonl(controller / "model-turns.jsonl")), {})
    tools = (first.get("request") or {}).get("tools") or []
    names = [((t.get("function") or {}).get("name")) for t in tools if isinstance(t, dict)]
    return [n for n in names if n] or list(CONTROLLER_TOOLS)


async def replay_row(controller: Path, goal: str, plan: list[dict], sampled: set[int], out: Path,
                     key: str, gate: asyncio.Semaphore, variants: list[str]) -> list[dict]:
    tools = offered_tools(controller)
    name = "__".join(controller.parts[-5:-2])
    navigators = {shape: VARIANTS[shape](goal, api_key=key, tools=tools, action_space="full",
                                         manifest_path=controller.parent / "aua/manifest.json",
                                         transcript_path=out / shape / f"{name}.jsonl")
                  for shape in variants}
    results = []
    for entry in plan:
        if entry["observation"] is None:
            break
        frame = entry["frame"]
        ask = entry["step"] in sampled
        for navigator in navigators.values():
            navigator.ask_now = ask
        async with gate if ask else _nothing():
            answers = await asyncio.gather(*(n(frame) for n in navigators.values()))
        if ask:
            row = {"row": name, "step": entry["step"], "label": entry["label"]["kind"]}
            for (shape, navigator), action in zip(navigators.items(), answers, strict=True):
                proposal = (navigator.last_decision or {}).get("verdict") or {}
                row[shape] = {"kind": proposal.get("kind"), "gate": proposal.get("gate"),
                              "operand_confidence": proposal.get("operand_confidence"),
                              "acted": action is not None, "declined": navigator.last_declined,
                              "right_kind_and_operand": bool(proposal) and matches(
                                  {**proposal, "operand": proposal.get("operand")}, entry["label"], entry["observation"]),
                              "usd": (navigator.last_decision or {}).get("usd"),
                              "ms": (navigator.last_decision or {}).get("request_ms")}
            results.append(row)
        for call in entry["calls"]:
            for navigator in navigators.values():
                navigator.observed(call.get("tool"), call.get("arguments") or {})
    return results


class _nothing:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def summarise(results: list[dict], variants: list[str]) -> dict:
    summary = {}
    for shape in variants:
        acted = [r for r in results if r[shape]["acted"]]
        right = [r for r in acted if r[shape]["right_kind_and_operand"]]
        finished = [r for r in results if r[shape]["kind"] == "finished" and (r[shape]["gate"] or 0) >= 0.85]
        summary[shape] = {
            "screens": len(results),
            "acted": len(acted), "acted_right": len(right), "acted_wrong": len(acted) - len(right),
            "net_right_minus_2x_wrong": len(right) - 2 * (len(acted) - len(right)),
            "confident_finished": len(finished),
            "confident_finished_right": sum(r["label"] == "finished" for r in finished),
            "errors": sum(1 for r in results if str(r[shape]["declined"] or "").startswith("request_failed")),
            "usd": round(sum(r[shape]["usd"] or 0 for r in results), 4),
            "median_ms": sorted(r[shape]["ms"] or 0 for r in results)[len(results) // 2] if results else None,
            "by_label": {label: {"screens": n,
                                 "acted_right": sum(1 for r in results if r["label"] == label and r[shape]["acted"]
                                                    and r[shape]["right_kind_and_operand"]),
                                 "acted_wrong": sum(1 for r in results if r["label"] == label and r[shape]["acted"]
                                                    and not r[shape]["right_kind_and_operand"])}
                         for label, n in Counter(r["label"] for r in results).most_common()},
        }
    return summary


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--variants", default="split,flat",
                        help=f"comma-separated, from {', '.join(VARIANTS)}")
    parser.add_argument("--dry-run", action="store_true", help="sample and count; send nothing")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if not variants or any(v not in VARIANTS for v in variants):
        parser.error(f"--variants takes names from {', '.join(VARIANTS)}")
    import os
    key = os.environ.get("OPENAI_API_KEY", "")
    plans = [(controller, goal, plan_row(controller)) for controller, goal in rows(args.artifacts)]
    pool = [(i, e["step"]) for i, (_, _, plan) in enumerate(plans) for e in plan if e["label"]]
    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.screens]
    print(json.dumps({"rows": len(plans), "labelled_screens": len(pool), "sampled": len(chosen),
                      "labels": Counter(next(e["label"]["kind"] for e in plans[i][2] if e["step"] == s)
                                        for i, s in chosen)}))
    if args.dry_run:
        return 0
    if not key:
        raise SystemExit("OPENAI_API_KEY is required")
    by_row: dict[int, set[int]] = {}
    for i, step in chosen:
        by_row.setdefault(i, set()).add(step)
    args.out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)
    batches = await asyncio.gather(*(replay_row(plans[i][0], plans[i][1], plans[i][2], steps, args.out, key, gate, variants)
                                     for i, steps in by_row.items()))
    results = [r for batch in batches for r in batch]
    (args.out / "screens.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summary = summarise(results, variants)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
