"""The training corpus must ask the model the question the runtime actually asks.

The curriculum imports ``compile_policy_context`` from the live runtime, so corpus and runtime can
never disagree about the *structure* of a policy context — and they do not. They disagree about its
*content*, in four fields at once, and a model trained on one vocabulary is being queried in
another. Every live context opens with two constraint strings and three ``recent_outcomes`` strings
that appear nowhere in the corpus, and five of every six training candidates name a tool the live
path cannot offer.

This is the regression test for that skew. It compares the vocabulary of a rendered training
context against the vocabulary of a context shaped exactly like the one the live autopilot builds,
and fails on any field where the corpus teaches something the runtime never says.

**Status: these record a defect in a superseded generator, and are marked xfail for that reason.**
They were written red on purpose, against V10, and they did their job — the measurement they encode
is what redirected the work. V11 replaced the V10 chooser entirely: its output is a
``memory.RouteStep`` executed by the on-device helper rather than an opaque candidate id chosen from
a host-built list, so the host ``PolicyContext`` these tests compare against is no longer the
contract V11 is trained for.

The equivalent guarantees for V11 live in ``tests/test_functiongemma_v11_curriculum.py``, which
checks the contract against the helper's own Java, and in
``experiments/functiongemma/v11_shortcut_gate.py``, which searches every cheap feature for rules
that answer without reading the goal.

Do not delete these: if the V10 corpus is ever regenerated for a comparison run, the skew they
measure is still real, and ``strict=False`` means they will simply start passing if it is fixed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser.policy import (  # noqa: E402
    PolicyCandidate,
    PolicyContext,
    compile_policy_context,
)

# Sampling breadth. Small enough to stay a unit test, wide enough that a field appearing in
# "100% of rows" is a real statement rather than an accident of one family.
GROUPS_PER_SPLIT = 120
VARIANTS = 2

#: The single tool the live policy path can offer. Both ``PolicyCandidate(`` construction sites in
#: ``engine.py`` hardcode it; every other ``"tool":`` there is a recommended_call handed to the
#: parent agent, never a candidate this model chooses among.
LIVE_TOOL = "tap_and_analyze"


def _live_context() -> dict[str, object]:
    """Compile a context shaped exactly like the live autopilot's (``engine.py:8529``)."""

    arguments = {"rid": "navTabPlaceholder"}
    candidates = tuple(
        PolicyCandidate(
            candidate_id=index,
            call={"tool": LIVE_TOOL, "arguments": arguments},
            model_arguments=arguments,
            purpose="Tap the current-frame 'Placeholder' control and observe the result.",
            proof="The exact call returns a folded post-action observation.",
            session_id="session",
            phase="navigate",
            observation_fingerprint="fingerprint",
            package="fictional.package",
        )
        for index in range(2)
    )
    context = PolicyContext(
        goal="Open the Placeholder",
        phase="navigate",
        session_id="session",
        candidates=candidates,
        # engine.py sends `fresh`, plus `known_screen` only when the screen is known.
        observation={"fresh": True},
        constraints=(
            "Select only a supplied guard-approved candidate.",
            "Do not invent or execute a call.",
        ),
        recent_outcomes=(
            "session_active=true",
            "outcome=known",
            "goal_checkpoint_reached=false",
        ),
        observation_fingerprint="fingerprint",
        package="fictional.package",
        allow_handoff=True,
    )
    return compile_policy_context(context)


def _corpus_contexts() -> list[dict[str, object]]:
    """Render training contexts through the same serializer the runtime uses."""

    pytest.importorskip("android_ui_analyser")
    from experiments.functiongemma.v9_curriculum import render_case
    from experiments.functiongemma.v10_learning_material import generate

    contexts: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        for case in generate(split, GROUPS_PER_SPLIT):
            for variant in range(VARIANTS):
                row = render_case(case, variant)
                contexts.append(json.loads(row["messages"][1]["content"]))
    assert contexts, "the generator produced no rows"
    return contexts


def test_structure_still_matches() -> None:
    """The shared serializer keeps the key sets identical. This half must never regress."""

    live = _live_context()
    corpus = _corpus_contexts()[0]
    assert set(corpus) == set(live)
    assert set(corpus["candidates"][0]) == set(live["candidates"][0])  # type: ignore[index]


@pytest.mark.xfail(reason="records the V10 skew; V11 supersedes this contract", strict=False)
def test_every_live_constraint_string_appears_in_the_corpus() -> None:
    """A constraint the runtime always sends must be one the model has read before."""

    live_head = [str(value) for value in _live_context()["constraints"]][:2]  # type: ignore[index]
    corpus_vocabulary = {
        str(value) for context in _corpus_contexts() for value in context["constraints"]
    }
    missing = [value for value in live_head if value not in corpus_vocabulary]
    assert not missing, (
        f"{len(missing)} of {len(live_head)} caller-supplied constraint strings never appear in "
        f"the corpus: {missing}"
    )


@pytest.mark.xfail(reason="records the V10 skew; V11 supersedes this contract", strict=False)
def test_every_live_recent_outcome_appears_in_the_corpus() -> None:
    """``recent_outcomes`` is a fixed triple at runtime; the corpus must speak it."""

    live_outcomes = [str(value) for value in _live_context()["recent_outcomes"]]  # type: ignore[index]
    corpus_vocabulary = {
        str(value) for context in _corpus_contexts() for value in context["recent_outcomes"]
    }
    missing = [value for value in live_outcomes if value not in corpus_vocabulary]
    assert not missing, (
        f"{len(missing)} of {len(live_outcomes)} runtime recent_outcomes strings never appear in "
        f"the corpus: {missing}"
    )


@pytest.mark.xfail(reason="records the V10 skew; V11 supersedes this contract", strict=False)
def test_the_corpus_does_not_rely_on_observation_keys_the_runtime_withholds() -> None:
    """A key present in nearly every training row and no live context is a phantom feature."""

    live_keys = set(_live_context()["observation"])  # type: ignore[arg-type]
    contexts = _corpus_contexts()
    counts: Counter[str] = Counter()
    for context in contexts:
        # Count keys, not values: ``Counter.update`` on a mapping would add the values as counts.
        counts.update(dict(context["observation"]).keys())  # type: ignore[arg-type]
    phantom = {
        key: round(count / len(contexts), 3)
        for key, count in counts.items()
        if key not in live_keys and count / len(contexts) > 0.10
    }
    assert not phantom, (
        "observation keys the corpus teaches but the runtime never supplies "
        f"(key -> share of rows): {phantom}"
    )


@pytest.mark.xfail(reason="records the V10 skew; V11 supersedes this contract", strict=False)
def test_most_training_candidates_name_a_tool_the_runtime_can_offer() -> None:
    """The runtime asks one question. The corpus should mostly answer that question."""

    tools: Counter[str] = Counter()
    for context in _corpus_contexts():
        for candidate in context["candidates"]:  # type: ignore[union-attr]
            tools[str(candidate["call"]["tool"])] += 1
    total = sum(tools.values())
    live_share = tools[LIVE_TOOL] / total
    assert live_share >= 0.90, (
        f"only {live_share:.1%} of {total} training candidates use {LIVE_TOOL!r}, the sole tool "
        f"the live policy path can offer; {len(tools)} distinct tools appear in the corpus"
    )
