"""Gates for the V11 on-device driver corpus.

The V11 corpus is only useful if every label it teaches is a step the on-device helper can actually
run, and if none of its labels can be predicted from something cheaper than reading the goal. Those
are the two ways every earlier cycle in this experiment went wrong, so both are tested here rather
than discovered after a training run.

The contract assertions are deliberately written against the helper's own Java source
(``helper/app/src/main/java/dev/aua/helper/FlowFeature.java``) rather than against the generator, so
a drift on either side fails here instead of silently producing steps the device rejects.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma import v11_contract as contract  # noqa: E402
from experiments.functiongemma import v11_curriculum as curriculum  # noqa: E402
from experiments.functiongemma import v11_learning_material as material  # noqa: E402
from experiments.functiongemma import v11_shortcut_gate as shortcut_gate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FLOW_FEATURE = REPO / "helper/app/src/main/java/dev/aua/helper/FlowFeature.java"

SMALL = {"train": 300, "valid": 120, "test": 120}
#: Rate assertions need real samples. `audit` refuses to estimate a bucket under 200 rows, and two
#: buckets near a 0.15 rate can differ by ~0.10 on noise alone at that size, so the statistical
#: tests use their own wider fixture rather than the shape fixture.
WIDE = {"train": 1400, "valid": 260, "test": 260}


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict[str, object]]]:
    return {
        split: curriculum.build_split(split, count, variants=2) for split, count in SMALL.items()
    }


@pytest.fixture(scope="module")
def wide_rows() -> list[dict[str, object]]:
    """One flat, wider sample for the tests that estimate a rate rather than check a shape."""

    return [
        row
        for split, count in WIDE.items()
        for row in curriculum.build_split(split, count, variants=2)
    ]


# --------------------------------------------------------------------------- contract


def _java() -> str:
    return FLOW_FEATURE.read_text(encoding="utf-8")


def _switch_labels(body: str) -> set[str]:
    """Every ``case "x":`` label in a Java switch body."""

    return set(re.findall(r'case\s+"([a-z-]+)"\s*:', body))


def _function_body(source: str, signature: str) -> str:
    """The text of one Java method, from its signature to its closing brace at method indent."""

    start = source.index(signature)
    end = source.index("\n    }", start)
    return source[start:end]


def test_every_device_kind_is_one_the_helper_implements() -> None:
    """A kind the generator teaches but ``runStep`` has no case for is a step that always fails."""

    body = _function_body(_java(), "private boolean runStep(")
    missing = sorted(set(contract.DEVICE_KINDS) - _switch_labels(body))
    assert not missing, f"kinds the helper cannot execute: {missing}"


def test_predicate_kinds_match_the_helper_s_predicate_set() -> None:
    """``PREDICATE_KINDS`` decides arg/by versus selector matching; both sides must agree."""

    source = _java()
    block = source[source.index("PREDICATE_KINDS") :].split(";", 1)[0]
    declared = set(re.findall(r'"([a-z-]+)"', block))
    assert set(contract.DEVICE_PREDICATE_KINDS) == declared


def test_by_vocabulary_matches_the_helper() -> None:
    """The helper raises on an unknown ``by`` rather than degrading, so the sets must be equal."""

    body = _function_body(_java(), "private static boolean matchesPredicate(")
    assert set(contract.BY_VALUES) == _switch_labels(body)


@pytest.mark.parametrize(
    "step",
    [
        {"kind": "tap"},  # no selector
        {"kind": "tap", "resource_id": "a", "label": "b"},  # two selectors
        {"kind": "tap", "resource_id": "a", "by": "text"},  # by is for predicates
        {"kind": "input", "resource_id": "a"},  # nothing to type
        {"kind": "tap", "resource_id": "a", "text": "typed"},  # tap carries no value
        {"kind": "assert-visible"},  # no query
        {"kind": "assert-visible", "arg": "x", "by": "resource_id"},  # not a by value
        {"kind": "assert-visible", "label": "x", "arg": "x"},  # predicate with a selector
        {"kind": "key", "arg": "enter"},  # not a global action
        {"kind": "swipe", "arg": "sideways"},
        {"kind": "wait-stable", "arg": "x"},
        {"kind": "goto", "arg": "x"},  # host-lane kind
        # `submit` is silently dropped by the helper, so the contract refuses it outright.
        {"kind": "input", "resource_id": "f", "text": "x", "submit": True},
        {"kind": "tap", "label": "X", "direction": "down"},  # direction is scroll-to only
        {"kind": "scroll-to", "arg": "X", "by": "text", "direction": "sideways"},
    ],
)
def test_validate_step_rejects_unexecutable_steps(step: dict[str, object]) -> None:
    with pytest.raises(contract.ContractError):
        contract.validate_step(step)


@pytest.mark.parametrize(
    "step",
    [
        {"kind": "tap", "resource_id": "navTabAtlas"},
        {"kind": "tap", "label": "Atlas"},
        {"kind": "tap", "content_desc": "Atlas"},
        {"kind": "input", "resource_id": "searchField", "text": "Atlas"},
        {"kind": "assert-visible", "arg": "Atlas", "by": "text"},
        {"kind": "scroll-to", "arg": "Atlas", "by": "text"},
        {"kind": "key", "arg": "back"},
        {"kind": "swipe", "arg": "up"},
        {"kind": "wait-stable"},
        {"kind": "hide-keyboard"},
    ],
)
def test_validate_step_accepts_executable_steps(step: dict[str, object]) -> None:
    contract.validate_step(step)


def test_render_call_rejects_an_unknown_argument() -> None:
    with pytest.raises(contract.ContractError):
        contract.render_call("next_step", {"kind": "tap", "nonsense": 1})


# --------------------------------------------------------------------------- corpus shape


def test_every_label_is_an_executable_step(splits) -> None:
    """Nothing may reach the corpus that the helper would refuse."""

    for rows in splits.values():
        for row in rows:
            if row["metadata"]["call"] != "next_step":
                continue
            body = row["messages"][2]["content"]
            assert body.startswith("<|tool_call_start|>[next_step(")
            assert body.endswith(")]<|tool_call_end|>")


def test_the_model_never_sees_a_node_index(splits) -> None:
    """Authoring a selector is the whole task; an index would let the model dodge it."""

    for rows in splits.values():
        for row in rows:
            context = json.loads(row["messages"][1]["content"])
            for node in context["screen"]["nodes"]:
                assert set(node) <= {"rid", "text", "desc", "clickable", "scrollable"}, node


def test_every_emitted_selector_exists_on_the_screen(splits) -> None:
    """A label that names a control the screen does not show teaches hallucination."""

    for rows in splits.values():
        for row in rows:
            if row["metadata"]["call"] != "next_step":
                continue
            context = json.loads(row["messages"][1]["content"])
            nodes = context["screen"]["nodes"]
            body = row["messages"][2]["content"]
            for field, key in (("resource_id", "rid"), ("label", "text"), ("content_desc", "desc")):
                marker = f'{field}="'
                if marker not in body:
                    continue
                value = body.split(marker, 1)[1].split('"', 1)[0]
                assert any(node.get(key) == value for node in nodes), (
                    f"{row['id']} selects {field}={value!r}, absent from the screen"
                )


def test_transport_is_lfm2_5_shaped(splits) -> None:
    """LFM2.5 ignores tool_calls and raises on content: None, and only messages[0] is system."""

    for rows in splits.values():
        for row in rows:
            roles = [message["role"] for message in row["messages"]]
            assert roles == ["system", "user", "assistant"]
            for message in row["messages"]:
                assert isinstance(message["content"], str) and message["content"]
                assert "tool_calls" not in message


def test_groups_and_vocabulary_do_not_cross_splits(splits) -> None:
    curriculum.check_group_isolation(splits)
    curriculum.check_vocabulary_isolation(splits)


def test_held_out_destinations_are_disjoint_from_training(splits) -> None:
    """A held-out score has to be about the model, not about a name it memorised."""

    def destinations(split: str) -> set[str]:
        found: set[str] = set()
        for row in splits[split]:
            context = json.loads(row["messages"][1]["content"])
            for node in context["screen"]["nodes"]:
                for key in ("text", "desc", "rid"):
                    if node.get(key):
                        found.add(str(node[key]))
        return found

    train = destinations("train")
    for split in ("valid", "test"):
        # Inert filler rows are shared vocabulary on purpose; section names must not be.
        sections = {name for name in destinations(split) if name in material._SECTION_POOLS[split]}
        assert sections, f"{split} produced no section names to check"
        assert not (sections & train), f"{split} reuses training section names"


# --------------------------------------------------------------------------- confounds


def test_refusal_is_not_predictable_from_a_visible_destructive_control(wide_rows) -> None:
    """The V10 defect, as a test. Its corpus made 'something dangerous is on screen' sufficient."""

    report = curriculum.audit(wide_rows)
    gap = report["refusal_rate_by_destructive"]["gap"]
    assert gap <= curriculum.MAX_DESTRUCTIVE_REFUSAL_GAP, report["refusal_rate_by_destructive"]
    # And the flag has to actually be present often enough for the comparison to mean anything.
    assert 0.2 <= report["destructive_visible_share"] <= 0.45


def test_refusal_is_not_predictable_from_node_count(wide_rows) -> None:
    """The V9 defect, as a test."""

    report = curriculum.audit(wide_rows)
    rates = report["refusal_rate_by_node_count"]
    if len(rates) < 2:
        pytest.skip("sample too small to estimate per-width refusal rates")
    spread = max(rates.values()) - min(rates.values())
    assert spread <= curriculum.MAX_NODE_COUNT_REFUSAL_SPREAD, rates


def test_relevance_refusal_dominates_authorization_refusal(wide_rows) -> None:
    """V10 had this at 0.118 and the model learned the cheap rule. It must be inverted."""

    report = curriculum.audit(wide_rows)
    assert report["relevance_share_of_handoffs"] >= curriculum.MIN_RELEVANCE_SHARE


def test_acting_still_dominates(wide_rows) -> None:
    """V9 regressed on taps when they fell to 40% of a self-generated corpus."""

    report = curriculum.audit(wide_rows)
    acting = report["calls"]["next_step"] / report["rows"]
    assert acting >= curriculum.MIN_ACTING_SHARE


def test_all_three_selector_fields_are_exercised(wide_rows) -> None:
    """A corpus that only ever answers with a resource id cannot name an icon."""

    report = curriculum.audit(wide_rows)
    fields = report["selector_fields"]
    for field in contract.SELECTOR_FIELDS:
        assert fields.get(field, 0) > 0, f"{field} never appears as an answer"


def test_every_family_and_every_handoff_reason_is_represented(wide_rows) -> None:
    report = curriculum.audit(wide_rows)
    for family in material.FAMILY_NAMES:
        assert family in report["families"], f"{family} produced no rows"
    for reason in ("target_absent", "no_progress", "needs_host_lane", "needs_authorization"):
        assert report["handoff_reasons"].get(reason, 0) > 0, f"{reason} never appears"


def test_multi_hop_goals_name_their_route(splits) -> None:
    """An intermediate hop is unlearnable unless the goal names it.

    Nothing on a home screen reveals which section contains a destination, so a goal naming only
    the destination would be asking the model to guess an app's information architecture. This
    caught a real generator bug: the first hop tapped a section the goal never mentioned.
    """

    for rows in splits.values():
        for row in rows:
            if row["metadata"]["family"] != "multi_hop":
                continue
            context = json.loads(row["messages"][1]["content"])
            body = row["messages"][2]["content"]
            for field in ("label", "content_desc"):
                marker = f'{field}="'
                if marker not in body:
                    continue
                value = body.split(marker, 1)[1].split('"', 1)[0]
                assert value in context["goal"], (
                    f"{row['id']} taps {value!r}, which its goal never names: {context['goal']!r}"
                )


@pytest.mark.xfail(
    reason=(
        "Deliberately red. Two changes made the gate stricter than the current corpus: the "
        "`needs_host_lane` residual was withdrawn after the trained model was caught using it as a "
        "reverse router on device goals, and RECALL_FLOOR dropped 0.25 -> 0.15 because that very "
        "rule measured 0.257 on a sample and under 0.25 on the full corpus — too weak to trip the "
        "gate, strong enough to shape gradients. Ten rules now surface, including one this session "
        "introduced (`history_tail_kind == 'assert-not-visible' -> done`, from the double-proof fix "
        "for the assert-visible leak: the leak moved one step down the chain rather than leaving). "
        "strict=False, so this flips green on its own once the corpus is rebuilt on the real-screen "
        "projection, the 2,000-name pools and the non-verbatim goal templates."
    ),
    strict=False,
)
def test_the_generic_shortcut_gate_passes(wide_rows) -> None:
    """No cheap feature may be sufficient for a minority answer.

    This is the gate the earlier ones should have been. `check_confounds` knew only about node count
    and visible destructiveness — the two confounds V9 and V10 had already paid for — and passed a
    corpus in which seven other rules predicted the answer at precision 1.000. Training then scored
    exactly 0.000 on all ten families that require naming a control, and 1.000 on the
    shortcut-decidable ones.

    Any residual tolerated here has to be declared in `ACCEPTED_RESIDUALS` with a written reason and
    a recall ceiling, so a new leak cannot hide behind an old exemption.
    """

    report = shortcut_gate.check(wide_rows)
    assert report["shortcuts"] == []
    for residual in report["accepted_residuals"]:
        assert residual["accepted_because"]


def test_no_family_teaches_a_step_the_helper_ignores(wide_rows) -> None:
    """`submit` is the case in point: the helper's input branch never fires the IME."""

    for row in wide_rows:
        body = row["messages"][2]["content"]
        assert "submit=" not in body
