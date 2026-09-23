"""A contract judge backed by a TypeSafe System One model (Jev).

A System One model does not generate text and calls no tools. It answers typed questions
against one shared ``state`` in a single parallel pass, returning a level, a probability
distribution and a calibrated confidence. That is a good fit for the *decision* half of
:mod:`judgement` and no fit at all for its *narrative* half, so this module splits them:

* each authored contract bullet becomes one ``Score`` question on :data:`EVIDENCE_LEVELS`;
* two ``Noul`` questions ask whether something outside the feature stopped the run and
  whether the run ever arrived at the screen the goal is about;
* :func:`compose` turns those numbers into the same verdict vocabulary the incumbent judge
  uses, in ordinary Python, with thresholds you can read and change.

What this judge cannot do, and does not fake: quoted evidence. ``judgement.OUTCOME_SCHEMA``
asks a model to write the sentence that justifies a verdict. Jev writes nothing, so every
string here is derived from the authored criterion and the level it landed on, and says so.
A run that needs a written justification still needs the incumbent judge.

Like every judgement in this harness, a result is a model opinion: it carries
``oracle="typesafe_system_one_v1"`` and ``verified=False``, and never outranks AUA's own
contract oracle. These are paid calls, billed on input tokens only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from experiments.aua_controller.typesafe_cost import USD_PER_INPUT_TOKEN

ORACLE = "typesafe_system_one_v1"
DEFAULT_MODEL = "jev-latest"

# One level, one situation. An earlier draft folded "contradicted" and "indirect" into a
# single level; the model placed a contradicted criterion there correctly and the composed
# verdict still came out wrong, because that one level meant two opposite things. Levels are
# judged independently and their numbering is their order, so never describe a level by
# reference to its neighbours.
EVIDENCE_LEVELS: tuple[str, ...] = (
    "Contradicted: an observed screen plainly shows the OPPOSITE of this",
    "Absent: nothing on the observed screens speaks to this either way",
    "Indirect: implied by what was observed, but not shown outright",
    "Directly evidenced: an observed screen shows this plainly and in full",
)

# Normalised onto 0..1 by dividing by the top level number, so these read as fractions of
# full proof regardless of how many levels are authored.
FAILED_BELOW = 0.25  # at or under level 0: the screens show the opposite
IMPLIED_AT = 0.50  # at or over: implied rather than absent, so a warning beats silence
VERIFIED_AT = 0.80  # at or over: near enough to level 3 to call it shown
BLOCKED_OVER = 0.60
REACHED_UNDER = 0.50

# Jev's request ceiling is 32k tokens for state plus the longest question, and its accuracy
# falls as the state fills with material unrelated to the decision. Both argue for the same
# cap the incumbent judge already applies to text evidence.
MAX_FRAMES = 8
MAX_ACTIONS = 30

_ACTION_FIELDS = ("step", "tool")


def _level(answer: Any) -> float:
    return float(answer.score)


def _probability(answer: Any) -> float:
    return float(answer.noul)


def _noul_confidence(probability: float) -> float:
    """A Noul carries no confidence of its own; distance from 0.5 is the honest stand-in."""
    return abs(probability - 0.5) * 2


def build_questions(criteria: Sequence[str]) -> dict[str, Any]:
    """One Score per authored bullet plus the two run-level Nouls, for a single call.

    Questions are evaluated in parallel against one state, so asking all of them together
    costs one copy of the state rather than one request each.
    """
    if not criteria:
        raise ValueError("a contract judge needs at least one authored criterion")
    from typesafe_sdk import Noul, Score

    questions: dict[str, Any] = {
        f"c{index}": Score(
            instructions={
                "question": "How well do the observed screens evidence this criterion?",
                "criterion": criterion,
                "focus": "Judge only what the observed screens show, not what the run intended.",
            },
            criteria=list(EVIDENCE_LEVELS),
        )
        for index, criterion in enumerate(criteria)
    }
    questions["blocked"] = Noul(
        instructions="Did something OUTSIDE the feature under test stop this run?",
        criteria={
            "true": "A login wall, crash, network error, permission prompt or other "
                    "interruption unrelated to the feature prevented progress",
            "false": "Nothing outside the feature interfered; the run was free to proceed",
        },
    )
    questions["reached"] = Noul(
        instructions="Did the run ever reach the screen the goal is about?",
        criteria={
            "true": "An observed screen is the one the goal describes",
            "false": "The run stopped somewhere else and never arrived",
        },
    )
    return questions


def build_state(
    *,
    goal: str,
    criteria: Sequence[str],
    final_frame: Any,
    frames: Sequence[Any] = (),
    actions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The same evidence the incumbent judge reads, shaped as one JSON state.

    Frames go through :func:`judgement.evidence_frame`, so a comparison between this judge
    and the incumbent is a comparison of models rather than of two different screens.
    """
    from experiments.aua_controller.judgement import evidence_frame

    return {
        "goal": goal,
        "contract": list(criteria),
        "actions_taken": [
            {key: action[key] for key in _ACTION_FIELDS if key in action}
            for action in list(actions)[-MAX_ACTIONS:]
        ],
        "intermediate_screens": [evidence_frame(frame) for frame in list(frames)[:MAX_FRAMES]],
        "final_screen": evidence_frame(final_frame),
    }


def _criterion_result(normalised: float, failed_below: float, verified_at: float) -> str:
    if normalised < failed_below:
        return "failed"
    if normalised < verified_at:
        return "not_verified"
    return "verified"


def _nearest_level(normalised: float) -> str:
    index = min(
        range(len(EVIDENCE_LEVELS)),
        key=lambda level: abs(normalised - level / (len(EVIDENCE_LEVELS) - 1)),
    )
    return EVIDENCE_LEVELS[index].split(":")[0]


def compose(
    answers: Mapping[str, Any],
    criteria: Sequence[str],
    *,
    failed_below: float = FAILED_BELOW,
    implied_at: float = IMPLIED_AT,
    verified_at: float = VERIFIED_AT,
    blocked_over: float = BLOCKED_OVER,
    reached_under: float = REACHED_UNDER,
) -> dict[str, Any]:
    """Turn one System One answer set into the harness's five-way verdict.

    Every threshold is an argument because the right cut depends on what a wrong verdict
    costs you, not on the model. Raising ``verified_at`` buys fewer false passes and more
    ``unverified``; there is no setting that buys both.
    """
    if not criteria:
        raise ValueError("a contract judge needs at least one authored criterion")
    top = len(EVIDENCE_LEVELS) - 1
    # A missing key is a programming error, not a degraded answer: composing a verdict from
    # a criterion nobody judged would report silence as a result.
    levels = [_level(answers[f"c{index}"]) / top for index in range(len(criteria))]
    blocked = _probability(answers["blocked"])
    reached = _probability(answers["reached"])

    entries: list[dict[str, Any]] = []
    satisfied: list[str] = []
    unsatisfied: list[str] = []
    for index, (criterion, normalised) in enumerate(zip(criteria, levels, strict=True)):
        outcome = _criterion_result(normalised, failed_below, verified_at)
        entries.append({
            "criterion_index": index,
            "result": outcome,
            # Derived, never observed. A System One model returns no text, so naming the
            # level it chose is the most this judge can honestly say it saw.
            "evidence": f"{_nearest_level(normalised)} ({normalised:.2f} of full proof): {criterion}",
        })
        (satisfied if outcome == "verified" else unsatisfied).append(criterion)

    weakest = min(levels)
    statuses = [entry["result"] for entry in entries]
    blocker: str | None = None
    if blocked > blocked_over:
        verdict = "blocked"
        blocker = "A screen outside the feature under test interrupted the run."
        confidence = _noul_confidence(blocked)
    elif reached < reached_under:
        verdict = "unverified"
        confidence = _noul_confidence(reached)
    else:
        if "failed" in statuses:
            verdict = "fail"
        elif "not_verified" not in statuses:
            verdict = "pass"
        elif weakest >= implied_at:
            verdict = "pass_with_warning"
        else:
            verdict = "unverified"
        # The chain is only as good as its weakest link, so the verdict inherits the
        # least confident criterion rather than an average that hides it.
        confidence = min(float(answers[f"c{index}"].confidence) for index in range(len(criteria)))

    reasons = [
        f"{len(satisfied)} of {len(criteria)} authored criteria are directly evidenced.",
        f"Weakest criterion sits at {weakest:.2f} of full proof ({_nearest_level(weakest)}).",
    ]
    if blocker:
        reasons.append(blocker)
    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "reasons": reasons,
        "satisfied": satisfied,
        "unsatisfied": unsatisfied,
        "blocker": blocker,
        "criteria": entries,
        "oracle": ORACLE,
        "verified": False,
    }


class TypeSafeJudge:
    """Judge a finished run against its authored contract in one System One request."""

    def __init__(self, client: Any = None, *, model: str = DEFAULT_MODEL) -> None:
        if client is None:
            from experiments.aua_controller.typesafe_cost import client_options
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(**client_options())
        self.client = client
        self.model = model
        self.requests = 0
        self.input_tokens = 0

    def judge(
        self,
        *,
        goal: str,
        criteria: Sequence[str],
        final_frame: Any,
        frames: Sequence[Any] = (),
        actions: Sequence[Mapping[str, Any]] = (),
        **thresholds: float,
    ) -> dict[str, Any]:
        state = build_state(
            goal=goal, criteria=criteria, final_frame=final_frame, frames=frames, actions=actions
        )
        response = self.client.system_one(
            state=state, questions=build_questions(criteria), model=self.model
        )
        self.requests += 1
        self.input_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)
        return compose(response.answers, criteria, **thresholds)

    def report(self) -> dict[str, Any]:
        """Output tokens are not billed, so input tokens are the whole cost story."""
        return {
            "oracle": ORACLE,
            "model": self.model,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "usd": round(self.input_tokens * USD_PER_INPUT_TOKEN, 8),
        }


__all__ = [
    "ORACLE", "DEFAULT_MODEL", "EVIDENCE_LEVELS", "TypeSafeJudge",
    "build_questions", "build_state", "compose",
]
