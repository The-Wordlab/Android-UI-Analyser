"""The V12 corpus contract: the properties whose absence made V11 unlearnable.

Every test names a measured failure. Several name a failure in a *previous version of this file*,
because the first V12 suite passed while the corpus was broken, and how it managed that is the most
useful thing in here:

* it asserted ``seen[("tap_despite_stalls", "handoff:no_progress")] == 0`` while ``build_row``
  returned that label unconditionally for the family — restating the generator's control flow and
  calling it a property. The test could not fail. The real distribution was 377 ``no_progress``
  against 3 ``tap`` for the evidence shape it claimed was balanced.
* it stripped the refusal *reason* before checking that progress does not predict the answer, so
  V11's exact learned mapping — ``needs_host`` at 51% of refusals with nothing tried, ``no_progress``
  at 40% with several — sat in the corpus under a reported spread of 0.0001.

So the tests below check distributions rather than control flow, and they check the full class label.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.v11_baseline import content_words, score  # noqa: E402
from experiments.functiongemma.v12_contract import CALLS, REASONS, render, validate  # noqa: E402
from experiments.functiongemma.v12_corpus import (  # noqa: E402
    answer_of,
    balance,
    build,
    load_screens,
    script_of,
)
from experiments.functiongemma.v12_progress import STALLED, annotate, sample_tried  # noqa: E402
from experiments.functiongemma.v12_shortcut_gate import check, features  # noqa: E402

SCREENS = Path(__file__).resolve().parents[1] / "runs/functiongemma/screens"


@pytest.fixture(scope="module")
def screens() -> list[dict]:
    if not SCREENS.is_dir():
        pytest.skip("no harvested screens in this checkout")
    loaded = load_screens(SCREENS)
    if not loaded:
        pytest.skip("harvest present but produced no projectable screens")
    return loaded


@pytest.fixture(scope="module")
def rows(screens: list[dict]) -> list[dict]:
    return build(screens, 9000, seed=17)


def _ctx(row: dict) -> dict:
    return json.loads(row["messages"][1]["content"])


def _stalled(nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if str(n.get("last") or "") in STALLED]


def _goal_node(ctx: dict) -> str | None:
    """The node a word-overlap reader would pick — the same operation the shipped rule performs."""

    terms = content_words(ctx["goal"])
    ranked = sorted(
        ((score(terms, n), n.get("n")) for n in ctx["screen"]["nodes"] if n.get("tap")),
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] > 0 else None


# ----------------------------------------------------------- it cannot name what is not there


def test_the_contract_has_no_field_a_name_can_be_invented_into() -> None:
    """V11's largest failure: 0 of 496 answers named an element that existed.

    ``label="Digest"`` is well-formed for any screen, so the model emitted training-corpus
    destinations at screens that had never contained them. Here every argument is an index or a
    closed enum, so the failure has no expression.
    """

    assert set(CALLS) == {"tap", "scroll", "done", "handoff"}
    assert render({"call": "tap", "n": "n3"}) == '[tap(n="n3")]'
    with pytest.raises(ValueError):
        validate({"call": "tap", "n": "n9"}, {"nodes": [{"n": "n1", "tap": True}]})
    with pytest.raises(ValueError):
        validate({"call": "handoff", "reason": "because"}, {"nodes": []})


def test_every_tap_points_at_a_node_that_exists_and_is_tappable(rows: list[dict]) -> None:
    taps = 0
    for row in rows:
        answer = row["messages"][-1]["content"]
        if 'tap(n="' not in answer:
            continue
        taps += 1
        want = answer.split('tap(n="', 1)[1].split('"', 1)[0]
        listed = {n["n"]: n for n in _ctx(row)["screen"]["nodes"]}
        assert want in listed, f"{want} not among {sorted(listed)}"
        assert listed[want].get("tap"), f"{want} is listed but not tappable"
    assert taps > 1000


def test_every_handoff_reason_is_one_of_the_four(rows: list[dict]) -> None:
    for row in rows:
        answer = row["messages"][-1]["content"]
        if 'reason="' in answer:
            assert answer.split('reason="', 1)[1].split('"', 1)[0] in REASONS


# ----------------------------------------------------------- the screens are the real ones


def test_screens_are_harvested_not_synthesised(rows: list[dict], screens: list[dict]) -> None:
    """V11 trained on 4-10 clean nodes and met a median of 41 raw / 10 projected on device."""

    assert len(screens) > 400
    counts = sorted(len(_ctx(r)["screen"]["nodes"]) for r in rows)
    assert counts[len(counts) // 2] >= 6
    labels = {
        " ".join(str(n.get(k) or "") for k in ("text", "desc"))
        for r in rows
        for n in _ctx(r)["screen"]["nodes"]
    }
    assert any(len(text) > 90 for text in labels), "no long folded real label present"


def test_the_goal_rarely_contains_the_answer_verbatim(rows: list[dict]) -> None:
    """73.6% of V11 rows had the target sitting in the goal, so it learned to copy the goal."""

    taps = verbatim = 0
    for row in rows:
        if row["meta"]["family"] == "decline":
            continue  # fixed phrases; structurally never verbatim, and counting them flatters this
        answer = row["messages"][-1]["content"]
        if 'tap(n="' not in answer:
            continue
        want = answer.split('tap(n="', 1)[1].split('"', 1)[0]
        ctx = _ctx(row)
        taps += 1
        for node in ctx["screen"]["nodes"]:
            if node.get("n") != want:
                continue
            text = " ".join(str(node.get(k) or "") for k in ("text", "desc")).strip().casefold()
            if text and text in str(ctx["goal"]).casefold():
                verbatim += 1
    assert taps
    assert verbatim / taps < 0.45, f"verbatim share {verbatim / taps:.1%} — V11 was 73.6%"


# ----------------------------------------------------------- progress must not predict the answer


def test_progress_does_not_predict_the_answer_reason_included(rows: list[dict]) -> None:
    """The V11 failure, checked on the full class label.

    Checkpoint 576, shown "Display" plainly on screen, refused seven times out of seven and chose the
    *reason* by how much had happened: nothing tried -> target_absent, a little -> needs_host, a lot
    -> no_progress. Checking only the call name misses this entirely, which is how it survived the
    first V12 build.

    Level zero is excluded because ``no_progress`` cannot exist there: it means the node the goal
    names was already tried, and at zero nothing has been. That is a definition, not a shortcut.
    """

    by_level: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        level = features(row)["step_band"]
        if level > 0:
            by_level[level][answer_of(row)] += 1

    levels = {k: v for k, v in by_level.items() if sum(v.values()) >= 150}
    assert len(levels) >= 3, "not enough progress levels represented"
    classes = {name for counter in levels.values() for name in counter}
    for name in classes:
        rates = [counter[name] / sum(counter.values()) for counter in levels.values()]
        assert max(rates) - min(rates) < 0.05, (
            f"{name} varies with progress: "
            + ", ".join(f"L{k}={v[name] / sum(v.values()):.3f}" for k, v in sorted(levels.items()))
        )


def test_a_stalled_node_anywhere_is_not_grounds_to_refuse(rows: list[dict]) -> None:
    """The property the old tautology test claimed to check, stated so it can fail.

    ``no_progress`` is correct when *the node the goal names* was tried and did not move. A model can
    approximate that with "some node on this screen has stalled", which needs no goal at all. In the
    corpus this predecessor design shipped, that approximation was right 377 times out of 380. Here
    the same evidence must be dominated by ``tap``.
    """

    with_stall: Counter[str] = Counter()
    for row in rows:
        if _stalled(_ctx(row)["screen"]["nodes"]):
            with_stall[answer_of(row)] += 1
    total = sum(with_stall.values())
    assert total > 500, "too few rows with a stalled node to judge"
    assert with_stall["tap"] / total > 0.35, (
        f"a stalled node mostly means refuse: {dict(with_stall)}"
    )
    assert with_stall["handoff:no_progress"] / total < 0.30


def test_but_a_stall_on_the_goals_own_node_does_mean_refuse(rows: list[dict]) -> None:
    """And the distinction has to be learnable by the operation the shipped rule already performs.

    Word overlap picks the goal's node; the answer then follows from one field on it. If this failed,
    the corpus would be asking for a judgement the prompt does not support.
    """

    on_target: Counter[str] = Counter()
    off_target: Counter[str] = Counter()
    for row in rows:
        ctx = _ctx(row)
        stalled = {n.get("n") for n in _stalled(ctx["screen"]["nodes"])}
        if not stalled:
            continue
        chosen = _goal_node(ctx)
        (on_target if chosen and chosen in stalled else off_target)[answer_of(row)] += 1

    assert sum(on_target.values()) > 100
    assert on_target["handoff:no_progress"] / sum(on_target.values()) > 0.6
    assert off_target["handoff:no_progress"] / sum(off_target.values()) < 0.10


def test_no_history_list_exists_to_be_joined_against(rows: list[dict]) -> None:
    """Progress lives on the node. The joinable list is where every earlier bug came from."""

    for row in rows:
        ctx = _ctx(row)
        assert "history" not in ctx, "a joinable history list is back"
        for node in ctx["screen"]["nodes"]:
            if "tried" in node:
                assert int(node["tried"]) > 0, "tried:0 teaches a field the runtime would omit"
                assert node.get("last") in ("changed", "unchanged", "blocked")
            else:
                assert "last" not in node, "last without tried"


def test_the_node_id_numeric_leak_is_gone(rows: list[dict]) -> None:
    """A precision-1.000 rule the gate could not see, because no feature read the number in an id.

    History entries invented ids from ``randrange(1, 9)`` while a real target id reaches ``n14``, so
    "a history entry names a node above n8" identified ``no_progress`` in 1,004 of 1,004 rows. It
    passed every one of the search's firing conditions and the manifest still said clean.
    """

    high: Counter[str] = Counter()
    for row in rows:
        for node in _ctx(row)["screen"]["nodes"]:
            index = str(node.get("n") or "")[1:]
            if int(node.get("tried") or 0) > 0 and index.isdigit() and int(index) > 8:
                high[answer_of(row)] += 1
                break
    total = sum(high.values())
    if total < 40:
        pytest.skip("too few high-index touched nodes in this sample")
    assert max(high.values()) / total < 0.6, f"high node indices still tell: {dict(high)}"


# ----------------------------------------------------------- the gate, and what it cannot see


def test_no_cheap_rule_predicts_any_answer(rows: list[dict]) -> None:
    report = check(rows)
    assert report["clean"], "\n".join(
        f"  P={f['precision']:.3f} R={f['recall']:.3f} {f['predicts']} <- {f['rule']}"
        for f in report["blocking"]
    )


def test_writing_system_does_not_predict_the_answer(rows: list[dict]) -> None:
    """The confound no gate feature could see, and why a clean gate is not sufficient.

    Absent-target goals were once drawn from every harvested label regardless of script, which put
    German goals on Arabic screens. Mismatch then moved ``target_absent`` + ``scroll`` from 22.4% of
    rows to 44.4%, and ``tap`` from 43% down to 24%.
    """

    by_match: dict[bool, Counter[str]] = defaultdict(Counter)
    for row in rows:
        ctx = _ctx(row)
        screen_text = " ".join(
            str(n.get("text") or n.get("desc") or "") for n in ctx["screen"]["nodes"]
        )
        goal_script, screen_script = script_of(str(ctx["goal"])), script_of(screen_text)
        if "none" in (goal_script, screen_script):
            continue
        by_match[goal_script == screen_script][answer_of(row).split(":")[0]] += 1

    if sum(by_match[False].values()) < 60:
        pytest.skip("too few cross-script rows to measure")
    for answers in by_match.values():
        total = sum(answers.values())
        assert answers["tap"] / total > 0.3, f"tap collapses on one script group: {dict(answers)}"


# ----------------------------------------------------------- the machinery itself


def test_the_balancer_refuses_to_delete_a_class_silently() -> None:
    """It once took ``done`` to 0% of the corpus while every check reported clean."""

    rng = random.Random(0)
    only_at_one_level = [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": json.dumps({"goal": "g", "step": step, "screen": {}})},
                {"role": "assistant", "content": "<|tool_call_start|>[done()]<|tool_call_end|>"},
            ]
        }
        for step in (1, 1, 2, 2)
    ] + [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": json.dumps({"goal": "g", "step": 1, "screen": {}})},
                {"role": "assistant", "content": '<|tool_call_start|>[tap(n="n1")]<|tool_call_end|>'},
            ]
        }
    ] * 4
    with pytest.raises(ValueError, match="would delete"):
        balance(only_at_one_level, rng)


def test_progress_annotation_omits_itself_when_nothing_was_tried() -> None:
    """A node carrying ``tried: 0`` would teach a field the runtime omits — a train/serve gap."""

    node: dict[str, object] = {"n": "n1", "tap": True}
    annotate(node, tried=0)
    assert "tried" not in node and "last" not in node
    annotate(node, tried=2, last="unchanged")
    assert node["tried"] == 2 and node["last"] == "unchanged"
    annotate(node, tried=0)
    assert "tried" not in node


def test_the_build_is_reproducible(screens: list[dict]) -> None:
    """``hash(name)`` is salted by PYTHONHASHSEED, so the split seeds were random every run."""

    # Unbalanced: `balance` refuses on a sample too small to carry every class at every level, and
    # what is under test here is the generator, whose seed was the thing being salted.
    first = build(screens, 1200, seed=5, do_balance=False)
    second = build(screens, 1200, seed=5, do_balance=False)
    assert [r["messages"] for r in first] == [r["messages"] for r in second]
    assert first != build(screens, 1200, seed=6, do_balance=False)


def test_tried_counts_come_from_one_shared_distribution() -> None:
    rng = random.Random(4)
    counts = Counter(sample_tried(rng) for _ in range(20000))
    assert set(counts) == {1, 2, 3, 4}
    assert min(counts.values()) / max(counts.values()) > 0.15
