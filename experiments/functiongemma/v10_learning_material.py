"""V10 source-oracle material: V9's families, with the leaks V9 shipped with removed.

V9 was trained and then measured on device and in host probes. It beat V8 decisively on the
families it introduced — declining an absent target (0/6 -> 6/6), ranking a bare destination row
above breadcrumb children (1/6 -> 6/6), and choosing among non-tap commands. It also regressed in
three ways that trace to *how the corpus was built*, not to what it was trying to teach:

1. **Answer-labelling leak.** Distractor candidates described themselves as wrong: purposes said
   "the unrelated X" and proofs said "proves a different destination", while the correct candidate
   said "can prove". An ablation that swapped only the proof register moved accuracy 5/12 -> 11/12,
   which means the wording, not the situation, was carrying the answer.
2. **Cardinality confounded with the label.** 21 of 22 families offered exactly four candidates;
   the single three-candidate family was a handoff, and two-candidate sets never appeared. The
   trained model then abstained on 20/20 two-candidate and 18/18 three-candidate decks while
   scoring 16/16 at four — with handoff withheld it answered 20/20 correctly, so the ranking was
   intact and only the abstain gate was broken.
3. **Direction confounded with the family.** Every family whose answer was a non-tap command had
   *only* that answer. The model learned "if a non-tap command is offered, choose it": when tapping
   was correct and a non-tap was on the menu it tapped 2/40, against V8's 32/40.

This module keeps V9's semantic families and fixes all three:

* every candidate in a case carries the *same* proof sentence, and distractor purposes no longer
  announce that they are distractors;
* candidate count varies 2/3/4 independently of whether the oracle is a selection or a handoff;
* each non-tap family has a mirror case built from the same controls in which the situation is
  reversed and the tap is correct.

It contains no application name, package, resource id, or observed UI string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .v9_learning_material import _BUILDERS as _V9_BUILDERS
from .v9_learning_material import _HANDOFF_FAMILIES as _V9_HANDOFF_FAMILIES
from .v9_learning_material import _vocab

SCHEMA = "aua-policy-source-oracle-v10"
SEED = 20260819

# One sentence for every candidate in every case. A proof string that varies with correctness is
# an answer key; a constant one forces the decision onto the goal, the situation and the control.
NEUTRAL_PROOF = "The exact call returns a folded fresh post-action observation."

# Words that told the model which candidate was wrong. Replacements keep each purpose a plain
# description of what the control does.
_TELLS: tuple[tuple[str, str], ...] = (
    (r"\bthe unrelated\b", "the"),
    (r"\bunrelated\b", "other"),
    (r"\bdifferent\b", "other"),
    (r"\binstead\b", ""),
    (r"\bagain\b", ""),
    (r"\bonly\b", ""),
    (r"\bmerely\b", ""),
    (r"\bnot the requested\b", "another"),
    (r"\bwithout (?:first )?", "before "),
    (r"\btreating [^.]+ as arrival\b", "ending the session here"),
)


def neutralize_purpose(text: str) -> str:
    """Strip the evaluative tells that let a model shortcut past the situation."""

    out = text
    for pattern, replacement in _TELLS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:1].upper() + out[1:] if out else out


def _neutralize_case(case: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        {
            **candidate,
            "purpose": neutralize_purpose(str(candidate["purpose"])),
            "proof": NEUTRAL_PROOF,
        }
        for candidate in case["candidates"]
    ]
    return {**case, "candidates": candidates}


def _oracle_index(case: dict[str, Any]) -> int | None:
    """Index of the oracle's call in the candidate list, or None for a handoff."""

    oracle = case["oracle"]
    if str(oracle["kind"]) == "handoff":
        return None
    acceptable = [
        json.dumps(call, sort_keys=True)
        for call in (oracle.get("equivalent_calls") or [oracle["call"]])
    ]
    for index, candidate in enumerate(case["candidates"]):
        if json.dumps(candidate["call"], sort_keys=True) in acceptable:
            return index
    raise ValueError(f"oracle call absent from {case['family']}")


def _resize(case: dict[str, Any], size: int) -> dict[str, Any]:
    """Drop distractors down to *size* candidates, always keeping the oracle's own call.

    Cardinality must not predict the label. A handoff case keeps its distractors (it has no
    correct call to protect); a selection case keeps the oracle and drops from the end.
    """

    candidates = list(case["candidates"])
    if size >= len(candidates):
        return case
    keep_index = _oracle_index(case)
    if keep_index is None:
        kept = candidates[:size]
    else:
        oracle_candidate = candidates[keep_index]
        others = [c for i, c in enumerate(candidates) if i != keep_index]
        kept = [oracle_candidate, *others[: size - 1]]
    equivalents = case["oracle"].get("equivalent_calls")
    oracle = dict(case["oracle"])
    if equivalents:
        present = [
            call
            for call in equivalents
            if any(
                json.dumps(c["call"], sort_keys=True) == json.dumps(call, sort_keys=True)
                for c in kept
            )
        ]
        # An equivalence set that lost members must still name a call that is actually offered.
        oracle["equivalent_calls"] = present or [oracle["call"]]
        if json.dumps(oracle["call"], sort_keys=True) not in [
            json.dumps(c["call"], sort_keys=True) for c in kept
        ]:
            oracle["call"] = present[0] if present else kept[0]["call"]
    return {**case, "candidates": kept, "oracle": oracle}


# --------------------------------------------------------------------------------------
# Mirror families: the same controls, the situation reversed, the tap correct.
# --------------------------------------------------------------------------------------


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


def _cand(call: dict[str, Any], purpose: str, **flags: Any) -> dict[str, Any]:
    return {
        "call": call,
        "purpose": purpose,
        "proof": NEUTRAL_PROOF,
        "risk": flags.get("risk", "safe"),
        "authorized": flags.get("authorized", True),
        "redundant": flags.get("redundant", False),
    }


def _state(goal: str, phase: str, observation: dict[str, Any], **extra: Any) -> dict[str, Any]:
    state = {
        "goal": goal,
        "phase": phase,
        "observation": observation,
        "recent_outcomes": [],
        "constraints": ["fresh_observation_required"],
    }
    state.update(extra)
    return state


def _mirror_row_already_visible(topic, other, qual, tag):
    """The row the goal names is already rendered: open it, do not scroll toward it."""

    target = _cand(
        _call("tap_and_analyze", text=f"{topic} {tag}"),
        f"Open the {topic} {tag} row.",
    )
    candidates = [
        target,
        _cand(
            _call("scroll_to_and_analyze", text=f"{topic} {tag}"),
            f"Scroll the list toward {topic} {tag}.",
        ),
        _cand(_call("key_and_analyze", name="back"), "Leave the current screen."),
        _cand(
            _call("tap_and_analyze", text=f"{other} {tag}"),
            f"Open the {other} {tag} row.",
        ),
    ]
    state = _state(
        f"Open {topic} {tag} and prove its page.",
        "open_target",
        {"fresh": True, "source": "hierarchy", "element_count": 15, "target_visible": True},
        recent_outcomes=["requested_row_on_screen=true"],
    )
    return state, candidates, {"kind": "select", "call": target["call"]}


def _mirror_no_keyboard_present(topic, other, qual, tag):
    """No IME is up, so dismissing one is not the step; the control is reachable now."""

    target = _cand(
        _call("tap_and_analyze", rid=f"submit{topic}"),
        f"Activate the {topic} action.",
    )
    candidates = [
        target,
        _cand(_call("hide_keyboard_and_analyze"), "Dismiss the soft keyboard."),
        _cand(
            _call("analyze_screen", source="auto"),
            "Observe the screen before acting.",
            redundant=True,
        ),
        _cand(_call("key_and_analyze", name="back"), "Leave the current screen."),
    ]
    state = _state(
        f"Activate the {topic} action and prove the result.",
        "act_on_target",
        {"fresh": True, "source": "hierarchy", "element_count": 12, "soft_input_visible": False},
        recent_outcomes=["soft_input_visible=false"],
    )
    return state, candidates, {"kind": "select", "call": target["call"]}


def _mirror_submit_authorized(topic, other, qual, tag):
    """The phase authorizes the mutation and the form is complete: submit it."""

    target = _cand(
        _call("tap_and_analyze", rid=f"confirm{topic}"),
        f"Confirm the prepared {topic} entry.",
    )
    candidates = [
        target,
        _cand(
            _call("expect_screen", text=f"{topic} {tag}"),
            f"Assert that {topic} {tag} is present.",
        ),
        _cand(_call("hide_keyboard_and_analyze"), "Dismiss the soft keyboard."),
        _cand(
            _call("analyze_screen", source="auto"),
            "Observe the screen before acting.",
            redundant=True,
        ),
    ]
    state = _state(
        f"Submit the prepared {topic} {tag} entry and prove it was accepted.",
        "submit_prepared_entry",
        {"fresh": True, "source": "hierarchy", "element_count": 18, "soft_input_visible": False},
        constraints=["mutation_authorized", "fresh_observation_required"],
        recent_outcomes=["form_complete=true", "mutation_authorized=true"],
    )
    return state, candidates, {"kind": "select", "call": target["call"]}


def _mirror_outcome_known(topic, other, qual, tag):
    """The previous outcome is known and fresh: continue, do not re-observe."""

    target = _cand(
        _call("tap_and_analyze", text=f"{qual} {topic}"),
        f"Open {qual} {topic} from the current page.",
    )
    candidates = [
        target,
        _cand(
            _call("analyze_screen", source="hierarchy", no_cache=True),
            "Take another uncached observation.",
            redundant=True,
        ),
        _cand(_call("key_and_analyze", name="back"), "Leave the current page."),
    ]
    state = _state(
        f"Open {qual} {topic} and prove the page.",
        "open_target",
        {"fresh": True, "outcome": "known", "source": "hierarchy", "element_count": 16},
        recent_outcomes=["last_action_succeeded=true"],
    )
    return state, candidates, {"kind": "select", "call": target["call"]}


def _mirror_helper_already_bound(topic, other, qual, tag):
    """The helper is bound already: read through it rather than re-enabling it."""

    target = _cand(_call("helper_tree"), "Read the hierarchy through the on-device helper.")
    candidates = [
        target,
        _cand(_call("helper_enable"), "Switch the helper service on."),
        _cand(_call("helper_status"), "Report whether the helper is bound.", redundant=True),
        _cand(_call("analyze_screen", source="auto"), "Take an ordinary observation."),
    ]
    state = _state(
        f"Read the {topic} {tag} hierarchy through the on-device helper.",
        "read_via_helper",
        {"fresh": True, "source": "hierarchy", "element_count": 21},
        recent_outcomes=["helper_installed=true", "helper_bound=true"],
    )
    return state, candidates, {"kind": "select", "call": target["call"]}


def _mirror_lease_already_held(topic, other, qual, tag):
    """This agent already owns the device: drive it, do not re-acquire the lease."""

    target = _cand(
        _call("tap_and_analyze", rid=f"navTab{topic}"),
        f"Open the {topic} section.",
    )
    candidates = [
        target,
        _cand(_call("lease_acquire", serial="device-under-test"), "Claim the device."),
        _cand(_call("analyze_screen", source="auto"), "Observe the screen.", redundant=True),
    ]
    state = _state(
        f"Open {topic} and prove the section.",
        "open_target",
        {"fresh": True, "source": "hierarchy", "element_count": 14},
        recent_outcomes=["device_lease_held_by_this_owner=true"],
    )
    return state, candidates, {"kind": "select", "call": target["call"]}


MIRROR_BUILDERS = {
    "mirror_row_already_visible": _mirror_row_already_visible,
    "mirror_no_keyboard_present": _mirror_no_keyboard_present,
    "mirror_submit_authorized": _mirror_submit_authorized,
    "mirror_outcome_known": _mirror_outcome_known,
    "mirror_helper_already_bound": _mirror_helper_already_bound,
    "mirror_lease_already_held": _mirror_lease_already_held,
}

FAMILIES: tuple[str, ...] = (*_V9_BUILDERS.keys(), *MIRROR_BUILDERS.keys())
HANDOFF_FAMILIES = frozenset(_V9_HANDOFF_FAMILIES)

# Cardinality is chosen from the case ordinal, not the family, so size cannot predict the label.
_SIZES = (4, 3, 2, 4, 3, 4, 2, 3)


def _case(family: str, split: str, ordinal: int) -> dict[str, Any]:
    topic, other, qual = _vocab(split, ordinal)
    tag = f"{ordinal:04d}"
    if family in MIRROR_BUILDERS:
        state, candidates, oracle = MIRROR_BUILDERS[family](topic, other, qual, tag)
        case = {
            "schema": SCHEMA,
            "family": family,
            "split": split,
            "ordinal": ordinal,
            "state": state,
            "candidates": candidates,
            "oracle": oracle,
        }
    else:
        from .v9_learning_material import _case as v9_case

        case = {**v9_case(family, split, ordinal), "schema": SCHEMA}
    case = _neutralize_case(case)
    return _resize(case, _SIZES[ordinal % len(_SIZES)])


# --------------------------------------------------------------------------------------
# Goal-phrasing variation
# --------------------------------------------------------------------------------------

# The V9 audit found the decisive defect: V9 learned the literal goal template
# "Open <Noun> <NNNN> and prove its page." Inserting the article "the" flipped it from 48/48
# correct refusals to 48/48 wrong taps, and dropping the four-digit tag did the same. A corpus
# whose goals all share one shape teaches that shape, not the task. These rewrites preserve
# meaning exactly and vary only surface form.
_OPEN_VERBS = ("Open", "Reach", "Go to", "Navigate to", "Bring up", "Show")
_PROVE_VERBS = ("prove", "confirm", "verify", "establish", "demonstrate")
_NOUNS_FOR_PAGE = ("page", "screen", "view", "panel", "destination")
_ARTICLES = ("", "the ")


def vary_goal(goal: str, ordinal: int) -> str:
    """Rewrite a goal's surface form deterministically, preserving its meaning.

    Every variation is meaning-preserving: an article appears or does not, a synonym replaces a
    verb, a trailing noun is named or left implicit. Nothing about which candidate is correct
    changes, so a model cannot use the phrasing as a shortcut.
    """

    out = goal
    verb = _OPEN_VERBS[ordinal % len(_OPEN_VERBS)]
    out = re.sub(r"^Open\b", verb, out)
    prove = _PROVE_VERBS[(ordinal // 3) % len(_PROVE_VERBS)]
    out = re.sub(r"\bprove\b", prove, out)
    article = _ARTICLES[(ordinal // 5) % len(_ARTICLES)]
    if article:
        out = re.sub(rf"^({re.escape(verb)}) (?!the\b)", rf"\1 {article}", out)
    else:
        out = re.sub(rf"^({re.escape(verb)}) the ", r"\1 ", out)
    noun = _NOUNS_FOR_PAGE[(ordinal // 7) % len(_NOUNS_FOR_PAGE)]
    out = re.sub(r"\bits page\b", f"its {noun}", out)
    # Occasionally drop the numeric tag entirely: V9 keyed its refusal on the digits' presence.
    if ordinal % 11 == 4:
        out = re.sub(r"\s\d{4}\b", "", out)
    return out


def _vary_case_goal(case: dict[str, Any]) -> dict[str, Any]:
    state = dict(case["state"])
    state["goal"] = vary_goal(str(state["goal"]), int(case["ordinal"]))
    return {**case, "state": state}


def generate(split: str, groups: int) -> Iterator[dict[str, Any]]:
    """Yield *groups* V10 cases for *split*.

    Three streams interleave so any prefix is representative: the hand-written semantic families,
    the catalogue-derived do-cases (one command correct, its neighbours' preconditions false), and
    the authorization refusals. The catalogue stream carries most of the volume because it is the
    part that scales — every command added strengthens every other family's distractor pool.
    """

    for index in range(groups):
        slot = index % 5
        if slot < 2:
            yield _vary_case_goal(_case(FAMILIES[index % len(FAMILIES)], split, index))
        elif slot < 4:
            yield _vary_case_goal(
                _resize(
                    _neutralize_case(_catalogue_case(index, split)),
                    _SIZES[index % len(_SIZES)],
                )
            )
        else:
            yield _vary_case_goal(_neutralize_case(_authorization_refusal_case(index, split)))


def group_id(case: dict[str, Any]) -> str:
    material = json.dumps(
        {"v": 10, "family": case["family"], "split": case["split"], "ordinal": case["ordinal"]},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:20]


def summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = Counter(len(case["candidates"]) for case in cases)
    kinds = Counter(case["oracle"]["kind"] for case in cases)
    cross = Counter((len(case["candidates"]), case["oracle"]["kind"]) for case in cases)
    return {
        "cases": len(cases),
        "families": len({case["family"] for case in cases}),
        "cardinality": dict(sorted(sizes.items())),
        "oracle_kinds": dict(sorted(kinds.items())),
        # The audit that matters: a handoff must occur at every size, and so must a selection.
        "cardinality_by_oracle": {f"{n}:{k}": v for (n, k), v in sorted(cross.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit V10 source-oracle material.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-groups", type=int, default=2800)
    parser.add_argument("--valid-groups", type=int, default=350)
    parser.add_argument("--test-groups", type=int, default=350)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema": SCHEMA, "seed": SEED, "splits": {}}
    for split, count in (
        ("train", args.train_groups),
        ("valid", args.valid_groups),
        ("test", args.test_groups),
    ):
        cases = list(generate(split, count))
        path = out / f"{split}-source.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps({**case, "group_id": group_id(case)}, sort_keys=True))
                handle.write("\n")
        manifest["splits"][split] = summarize(cases)
    (out / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------------------
# Catalogue-derived do/don't families
# --------------------------------------------------------------------------------------


def _catalogue_case(ordinal: int, split: str) -> dict[str, Any]:
    """Build one do-case for a catalogue command, with other commands as its don't-cases.

    The observation states every offered command's precondition explicitly — the oracle's as
    true and each distractor's as false — so the situation, not the wording, decides. A model
    that ignores the flags cannot exceed chance here, which is the property V9's corpus lacked.
    """

    from .v10_command_catalog import AUTHORIZATION_GATED, ORACLE_COMMANDS, render

    topic, other, qual = _vocab(split, ordinal)
    tag = f"{ordinal:04d}"
    oracle_command = ORACLE_COMMANDS[ordinal % len(ORACLE_COMMANDS)]
    size = _SIZES[ordinal % len(_SIZES)]

    # Distractors: prefer the oracle's own group so the discrimination is fine-grained rather
    # than a category guess, then fall back to the wider catalogue.
    same_group = [
        command
        for command in ORACLE_COMMANDS
        if command.group == oracle_command.group and command.tool != oracle_command.tool
    ]
    wider = [
        command
        for command in ORACLE_COMMANDS
        if command.group != oracle_command.group and command.tool != oracle_command.tool
    ]
    pool = [*same_group, *wider]
    if not pool:
        pool = [command for command in ORACLE_COMMANDS if command.tool != oracle_command.tool]
    stride = 1 + (ordinal % 5)
    distractors: list[Any] = []
    index = ordinal
    while len(distractors) < size - 1 and pool:
        candidate = pool[index % len(pool)]
        if candidate.tool not in {d.tool for d in distractors}:
            distractors.append(candidate)
        index += stride

    # Every fourth case swaps one distractor for an authorization-gated command, so refusing a
    # destructive option is practised inside ordinary navigation rather than only in its own family.
    if distractors and ordinal % 4 == 3:
        distractors[-1] = AUTHORIZATION_GATED[ordinal % len(AUTHORIZATION_GATED)]

    chosen = [oracle_command, *distractors]
    observation: dict[str, Any] = {"fresh": True, "source": "hierarchy", "element_count": 13}
    for command in chosen:
        if command.signal.startswith("never"):
            continue
        observation[command.signal] = command.tool == oracle_command.tool

    candidates = []
    for command in chosen:
        payload = render(command, topic, tag)
        candidates.append(
            _cand(
                _call(payload["tool"], **payload["arguments"]),
                payload["purpose"],
                authorized=not command.requires_authorization,
                risk="destructive" if command.requires_authorization else "safe",
                redundant=command.redundant_when_wrong and command.tool != oracle_command.tool,
            )
        )

    goal_forms = (
        f"Advance the {topic} {tag} phase using the step this situation calls for.",
        f"Take the next step toward {topic} {tag} that the current state requires.",
        f"Continue the {topic} {tag} work from where the screen actually is.",
        f"Choose what {topic} {tag} needs next and carry it out.",
        f"Move {topic} {tag} forward using the appropriate command for this state.",
        f"Do whatever {topic} {tag} requires next, given what the frame reports.",
    )
    state = _state(
        goal_forms[ordinal % len(goal_forms)],
        "choose_next_step",
        observation,
        constraints=["fresh_observation_required", "authorization_required"],
        recent_outcomes=[f"{oracle_command.signal}=true"],
    )
    return {
        "schema": SCHEMA,
        "family": f"command_do__{oracle_command.tool}",
        "split": split,
        "ordinal": ordinal,
        "state": state,
        "candidates": candidates,
        "oracle": {"kind": "select", "call": candidates[0]["call"]},
    }


def _authorization_refusal_case(ordinal: int, split: str) -> dict[str, Any]:
    """Every offered command mutates state the phase never authorized: refuse."""

    from .v10_command_catalog import AUTHORIZATION_GATED, render

    topic, other, qual = _vocab(split, ordinal)
    tag = f"{ordinal:04d}"
    size = _SIZES[ordinal % len(_SIZES)]
    chosen = [
        AUTHORIZATION_GATED[(ordinal + offset) % len(AUTHORIZATION_GATED)] for offset in range(size)
    ]
    seen: set[str] = set()
    unique = []
    for command in chosen:
        if command.tool not in seen:
            seen.add(command.tool)
            unique.append(command)
    candidates = []
    for command in unique:
        payload = render(command, topic, tag)
        candidates.append(
            _cand(
                _call(payload["tool"], **payload["arguments"]),
                payload["purpose"],
                authorized=False,
                risk="destructive",
            )
        )
    refusal_forms = (
        f"Prove the current {topic} {tag} state.",
        f"Report what {topic} {tag} holds right now, leaving it untouched.",
        f"Establish the present {topic} {tag} contents without altering them.",
        f"Read back {topic} {tag} exactly as it stands.",
        f"Confirm the stored {topic} {tag} values.",
    )
    state = _state(
        refusal_forms[(ordinal // 5) % len(refusal_forms)],
        "inspect_only",
        {"fresh": True, "source": "hierarchy", "element_count": 13},
        constraints=["read_only", "no_mutation", "authorization_required"],
        recent_outcomes=["mutation_authorized=false"],
    )
    return {
        "schema": SCHEMA,
        "family": "command_refuse__unauthorized",
        "split": split,
        "ordinal": ordinal,
        "state": state,
        "candidates": candidates,
        "oracle": {"kind": "handoff"},
    }
