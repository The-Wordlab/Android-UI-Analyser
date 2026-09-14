"""What AUA must be told before it can prove a claim about an app it has not seen.

"The hub badge shows once on first open" is not yet a test.  It needs a build, a way past
sign-in, a definition of *first open* that a machine can establish, a decision about whether the
backend is in scope, and - the part that is easiest to skip - a statement of what a person would
actually see, written so it can be asserted rather than believed.

The agent that just implemented the feature has all of that.  AUA does not.  So this module is the
question list AUA hands over, plus the bookkeeping that stops the same question being asked twice.

Three deliberate constraints:

* **No device, no network, no model.**  This is a catalogue, a matcher against what the app map
  already knows, and a validator.  An interview that needed a model would put a provider outage in
  front of the cheapest part of a run, and would invent answers when it should be asking for them.
* **The oracle is never guessed.**  ``success``/``repeat`` answers are AUA predicate terms, so
  turning them into contract assertions is a mechanical translation.  A module that read English
  and emitted assertions would sometimes produce a contract that looks right and asserts the wrong
  thing, which is worse than no contract at all.
* **Additive.**  ``session start`` is untouched.  Preparation produces a contract and a command;
  running without it remains the normal path.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from .errors import UsageError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .memory import AppMap

PREPARE_SCHEMA_VERSION = 1

AnswerKind = Literal["text", "path", "choice", "predicate"]


# --------------------------------------------------------------------------- seeding strategies


@dataclass(frozen=True)
class SeedingStrategy:
    """One way to put the device into the state the claim is about, and its honest price.

    ``fakes`` is the load-bearing field.  Every strategy but the last two substitutes something for
    the real thing, and which substitution is acceptable depends entirely on what the test is for:
    mocking the backend is the obvious choice for a UI-only check and disqualifying for an
    end-to-end one.  Ranking without that distinction recommends the fastest option every time.
    """

    key: str
    summary: str
    calls: tuple[str, ...]
    seconds: int
    reversible: bool
    # What this substitutes: "local" app state, the code "behaviour", or the "backend".
    fakes: tuple[str, ...]
    # Does it destroy an existing signed-in session?
    wipes_signin: bool
    proves_nothing_about: str


SEEDING_STRATEGIES: tuple[SeedingStrategy, ...] = (
    SeedingStrategy(
        key="datastore",
        summary="Write the app's own Preferences/DataStore key to the pre-condition value.",
        calls=("aua datastore backup", "aua datastore set", "aua datastore restore"),
        seconds=1,
        reversible=True,
        fakes=("local",),
        wipes_signin=False,
        proves_nothing_about="how that key gets its very first value on a genuinely new install",
    ),
    SeedingStrategy(
        key="database",
        summary="Update the row the feature reads, in the app's own SQLite database.",
        calls=("aua database backup", "aua database execute", "aua database restore"),
        seconds=3,
        reversible=True,
        fakes=("local",),
        wipes_signin=False,
        proves_nothing_about="the migration or writer that normally produces that row",
    ),
    SeedingStrategy(
        key="flags",
        summary="Apply a feature flag that forces the branch under test.",
        calls=("aua flags apply",),
        seconds=5,
        reversible=True,
        fakes=("behaviour",),
        wipes_signin=False,
        proves_nothing_about="the default branch real users get when the flag is off",
    ),
    SeedingStrategy(
        key="mock",
        summary="Serve a recorded or rewritten backend response through the AUA proxy.",
        calls=("aua proxy start", "aua mock map", "aua mock replay", "aua proxy stop"),
        seconds=10,
        reversible=True,
        fakes=("backend",),
        wipes_signin=False,
        proves_nothing_about="whether the real backend sends that shape at all",
    ),
    SeedingStrategy(
        key="reinstall",
        summary="Uninstall and install the build, so the app really is on its first run.",
        calls=("aua session start --apk <path> --fresh --yes",),
        seconds=45,
        reversible=False,
        fakes=(),
        wipes_signin=True,
        proves_nothing_about="nothing - this is the real thing, at the price of the whole app state",
    ),
    SeedingStrategy(
        key="ui",
        summary="Drive the app by hand into the state, using no back door at all.",
        calls=("aua goto", "aua tap"),
        seconds=60,
        reversible=True,
        fakes=(),
        wipes_signin=False,
        proves_nothing_about="nothing - but a once-only state usually cannot be re-entered this way",
    ),
)

_STRATEGY_BY_KEY = {strategy.key: strategy for strategy in SEEDING_STRATEGIES}
SEEDING_KEYS: tuple[str, ...] = tuple(strategy.key for strategy in SEEDING_STRATEGIES)

SCOPE_CHOICES: tuple[str, ...] = ("ui", "e2e")


def _speed_points(seconds: int) -> int:
    if seconds <= 2:
        return 3
    if seconds <= 10:
        return 2
    return 0


def rank_seeding(
    *,
    scope: str = "ui",
    available: Sequence[str] | None = None,
    signin_required: bool = True,
) -> list[dict[str, Any]]:
    """Rank the ways to reach the pre-condition, best first, each with its reason.

    The ranking is a small integer score rather than a judgement call so that it is inspectable:
    the caller sees the same arithmetic AUA used and can disagree with a specific line of it.
    """

    scope_value = scope if scope in SCOPE_CHOICES else "ui"
    allowed = set(available) if available is not None else set(SEEDING_KEYS)
    unknown = sorted(allowed - set(SEEDING_KEYS))
    if unknown:
        raise UsageError("unknown seeding strategy: " + ", ".join(unknown))

    ranked: list[dict[str, Any]] = []
    for strategy in SEEDING_STRATEGIES:
        if strategy.key not in allowed:
            continue
        reasons: list[str] = []
        score = 0
        if strategy.reversible:
            score += 3
            reasons.append("AUA can put it back")
        else:
            reasons.append("not reversible; the app's data is gone")
        speed = _speed_points(strategy.seconds)
        score += speed
        reasons.append(f"about {strategy.seconds}s to apply")
        if scope_value == "e2e" and "backend" in strategy.fakes:
            score -= 4
            reasons.append("faking the backend removes the thing an end-to-end run is for")
        if "behaviour" in strategy.fakes:
            score -= 1
            reasons.append("a forced branch is not the branch real users get")
        if strategy.wipes_signin and signin_required:
            score -= 2
            reasons.append("sign-in has to be done again afterwards")
        ranked.append(
            {
                "strategy": strategy.key,
                "score": score,
                "summary": strategy.summary,
                "calls": list(strategy.calls),
                "reasons": reasons,
                "proves_nothing_about": strategy.proves_nothing_about,
            }
        )
    # Ties keep catalogue order, which runs cheapest-and-most-local first; that is the order a
    # reader expects, and a stable order keeps the recommendation reproducible across runs.
    ranked.sort(key=lambda entry: -entry["score"])
    return ranked


def seeding_recommendation(**kwargs: Any) -> dict[str, Any] | None:
    """The single strategy AUA suggests, or ``None`` when every option was excluded."""

    ranked = rank_seeding(**kwargs)
    return ranked[0] if ranked else None


# --------------------------------------------------------------------------- the question list


@dataclass(frozen=True)
class Question:
    """One thing AUA cannot find out for itself."""

    key: str
    ask: str
    why: str
    kind: AnswerKind = "text"
    choices: tuple[str, ...] = ()
    # Goal phrasings that make an existing knowledge item an answer to this question.
    aliases: tuple[str, ...] = ()
    required: bool = True
    # Worth keeping in the app map afterwards?  A build path and a sign-in recipe are facts about
    # the app; the scope of one particular test is not, and remembering it would quietly answer a
    # question the next agent should be asked.
    durable: bool = False
    knowledge_kind: str = "note"
    example: str | None = None
    # Asked only when this returns True for (goal, answers so far).
    applies: Callable[[str, Mapping[str, str]], bool] | None = None


_ONCE_ONLY = re.compile(
    r"\bonce\b"
    r"|\bfirst[- ]?(?:run|open|launch|time)\b"
    r"|\bone[- ]?shot\b"
    # "is not shown again", "never appears again": the negation and `again` can be a few words
    # apart, and it is the pair that means bounded, not either word alone.
    r"|\b(?:not|never|no longer|n't)\b(?:\W+\w+){0,4}\W+\bagain\b",
    re.IGNORECASE,
)


def goal_is_once_only(goal: str) -> bool:
    """Does the claim say the thing happens a bounded number of times?

    This is the difference between a test that can pass by accident and one that cannot: showing a
    badge is trivial, showing it exactly once is the actual claim, and only the second half needs a
    second checkpoint.
    """

    return bool(_ONCE_ONLY.search(goal or ""))


QUESTIONS: tuple[Question, ...] = (
    Question(
        key="build",
        ask="Which build should AUA install, and what package does it launch?",
        why="AUA installs and launches it; without the path it can only drive whatever is already on the device.",
        kind="path",
        aliases=("apk path", "where is the build", "install the app", "debug build"),
        durable=True,
        knowledge_kind="note",
        example="/Users/me/app/build/outputs/apk/debug/app-debug.apk",
    ),
    Question(
        key="signin",
        ask="How does a test user get past sign-in? Say `none` if the app opens straight in.",
        why="Every later screen is behind it, so a wrong guess here blocks the whole run.",
        aliases=("log in", "sign in", "authentication", "guest user", "account"),
        durable=True,
        knowledge_kind="recipe",
        example="tap `Continue as guest` on the welcome screen",
    ),
    Question(
        key="precondition",
        ask="What has to be true before the check, in terms the app actually stores?",
        why="AUA has to put the device into that state; a description of the user's experience is not enough to do it.",
        aliases=("first open", "first run", "first launch", "fresh install", "precondition"),
        durable=True,
        knowledge_kind="note",
        example="the DataStore key `hub_badge_seen` is absent or false",
    ),
    Question(
        key="seeding",
        ask="How should AUA reach that state?",
        why="Each option fakes something different, and which substitution is acceptable is your call, not AUA's.",
        kind="choice",
        choices=SEEDING_KEYS,
        aliases=("seed state", "reset state", "mock", "fake state"),
        durable=True,
        knowledge_kind="recipe",
    ),
    Question(
        key="scope",
        ask="Is this a UI-only check, or must the real backend be in the loop?",
        why="It decides whether mocking is a shortcut or a way of not testing the thing.",
        kind="choice",
        choices=SCOPE_CHOICES,
        durable=False,
    ),
    Question(
        key="success",
        ask="What must be on screen when it works? Use AUA predicate terms, comma separated.",
        why="This becomes the contract. AUA will not translate a sentence into an assertion - a guessed oracle is worse than none.",
        kind="predicate",
        durable=False,
        example="rid:hubBadge,text:New",
    ),
    Question(
        key="repeat",
        ask="And what must be on screen the *second* time, when it should not happen again?",
        why="The claim is `once`. Without this the run proves only that it happened at all.",
        kind="predicate",
        required=True,
        durable=False,
        example="!rid:hubBadge",
        applies=lambda goal, answers: goal_is_once_only(goal),
    ),
    Question(
        key="restore",
        ask="Anything AUA should put back afterwards beyond what it changed itself? `none` is a fine answer.",
        why="AUA already undoes its own device changes; this covers state only you know about.",
        required=False,
        durable=False,
        example="none",
    ),
)

QUESTION_BY_KEY = {question.key: question for question in QUESTIONS}


# --------------------------------------------------------------------------- answer validation


_PREDICATE_TERM = re.compile(r"\A(!?)(?:(rid|id|text|desc):)?(.+)\Z", re.DOTALL)


def parse_predicate_terms(value: str, *, where: str) -> list[dict[str, Any]]:
    """Turn `rid:hubBadge,!text:New` into assert mappings, with no interpretation.

    A bare term is treated as visible text, which is what `until:` already does, so an agent that
    has learned one predicate grammar has learned this one.  ``!`` asserts absence.
    """

    terms = [term.strip() for term in str(value or "").split(",")]
    terms = [term for term in terms if term]
    if not terms:
        raise UsageError(f"{where} needs at least one predicate term, e.g. `rid:hubBadge`")
    out: list[dict[str, Any]] = []
    for term in terms:
        match = _PREDICATE_TERM.fullmatch(term)
        if match is None:  # pragma: no cover - the pattern accepts any non-empty term
            raise UsageError(f"{where} could not read the term {term!r}")
        negated, selector, body = match.group(1), match.group(2) or "text", match.group(3).strip()
        if not body:
            raise UsageError(f"{where} term {term!r} has no selector value")
        key = "rid" if selector in {"rid", "id"} else selector
        assertion: dict[str, Any] = {key: body}
        assertion["absent" if negated else "exists"] = True
        out.append({"assert": assertion})
    return out


def validate_answer(question: Question, value: str) -> str:
    """Normalize one answer, or say precisely what is wrong with it."""

    text = str(value if value is not None else "").strip()
    if not text:
        raise UsageError(f"answer for `{question.key}` is empty")
    if question.kind == "choice" and text not in question.choices:
        raise UsageError(
            f"`{question.key}` must be one of " + ", ".join(question.choices) + f", got {text!r}"
        )
    if question.kind == "predicate":
        parse_predicate_terms(text, where=f"`{question.key}`")
    return text


# --------------------------------------------------------------------------- the prepare session


@dataclass
class PrepareSession:
    """The interview so far: what was already known, what was answered, what is still missing."""

    id: str
    package: str
    goal: str
    created_at: str
    answers: dict[str, str] = field(default_factory=dict)
    # question key -> the knowledge items that already answer it
    known: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    schema_version: int = PREPARE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "package": self.package,
            "goal": self.goal,
            "created_at": self.created_at,
            "answers": dict(self.answers),
            "known": {key: list(items) for key, items in self.known.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PrepareSession:
        if not isinstance(value, Mapping):
            raise UsageError("a prepare session must be a mapping")
        version = value.get("schema_version", PREPARE_SCHEMA_VERSION)
        if version != PREPARE_SCHEMA_VERSION:
            raise UsageError(
                f"unsupported prepare session version {version}; expected {PREPARE_SCHEMA_VERSION}"
            )
        for required in ("id", "package", "goal", "created_at"):
            if not isinstance(value.get(required), str) or not value[required].strip():
                raise UsageError(f"prepare session is missing `{required}`")
        answers = value.get("answers") or {}
        known = value.get("known") or {}
        if not isinstance(answers, Mapping) or not isinstance(known, Mapping):
            raise UsageError("prepare session `answers`/`known` must be mappings")
        return cls(
            id=str(value["id"]),
            package=str(value["package"]),
            goal=str(value["goal"]),
            created_at=str(value["created_at"]),
            answers={str(k): str(v) for k, v in answers.items()},
            known={str(k): list(v) for k, v in known.items() if isinstance(v, list)},
        )


def applicable_questions(goal: str, answers: Mapping[str, str]) -> list[Question]:
    """The questions this goal actually raises, in catalogue order."""

    return [
        question
        for question in QUESTIONS
        if question.applies is None or question.applies(goal, answers)
    ]


def open_preparation(
    *,
    package: str,
    goal: str,
    app_map: AppMap | None = None,
    now: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
) -> PrepareSession:
    """Start an interview, pre-filling every question the app map can already answer."""

    if not str(package or "").strip():
        raise UsageError("prepare needs a package")
    if not str(goal or "").strip():
        raise UsageError("prepare needs a goal; it decides which questions apply")
    session = PrepareSession(
        id=session_id or f"prep-{uuid.uuid4().hex[:12]}",
        package=str(package).strip(),
        goal=str(goal).strip(),
        created_at=now or datetime.now(UTC).isoformat(),
    )
    if app_map is not None:
        session.known = recall(app_map, goal=session.goal, context_id=context_id)
    return session


def recall(
    app_map: AppMap, *, goal: str, context_id: str | None = None, per_question: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Knowledge already on file that answers a question, keyed by question.

    Matching runs per question rather than once against the goal, because a goal phrase and a
    question are looking for different things: "badge on first open" should surface the *first
    open* note whatever else it matches, and should surface the sign-in recipe even though the word
    never appears in it.
    """

    from .session import relevant_knowledge  # local: session.py is large and rarely needed here

    found: dict[str, list[dict[str, Any]]] = {}
    for question in QUESTIONS:
        if not question.aliases:
            continue
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for phrase in (*question.aliases, goal):
            for item in relevant_knowledge(
                app_map, phrase, context_id=context_id, limit=per_question
            ):
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                hits.append(item)
        if hits:
            found[question.key] = hits[:per_question]
    return found


def outstanding(session: PrepareSession, *, include_known: bool = False) -> list[Question]:
    """Questions still unanswered.  A question the app map already answers is not asked again."""

    return [
        question
        for question in applicable_questions(session.goal, session.answers)
        if question.key not in session.answers
        and (include_known or question.key not in session.known)
    ]


def record_answers(session: PrepareSession, answers: Mapping[str, str]) -> list[str]:
    """Validate and store answers; returns the keys accepted, in the order given."""

    accepted: list[str] = []
    for key, value in answers.items():
        question = QUESTION_BY_KEY.get(key)
        if question is None:
            raise UsageError(
                f"unknown prepare answer `{key}`; expected one of "
                + ", ".join(sorted(QUESTION_BY_KEY))
            )
        session.answers[key] = validate_answer(question, value)
        accepted.append(key)
    return accepted


def parse_answer_pairs(pairs: Iterable[str]) -> dict[str, str]:
    """Read repeated `key=value` CLI arguments, keeping `=` inside the value."""

    out: dict[str, str] = {}
    for pair in pairs or ():
        text = str(pair)
        if "=" not in text:
            raise UsageError(f"expected `key=value`, got {text!r}")
        key, _, value = text.partition("=")
        key = key.strip()
        if not key:
            raise UsageError(f"expected `key=value`, got {text!r}")
        out[key] = value.strip()
    return out


def missing_required(session: PrepareSession) -> list[str]:
    """Required questions with neither an answer nor prior knowledge."""

    return [
        question.key
        for question in outstanding(session)
        if question.required and question.key not in session.known
    ]


def is_ready(session: PrepareSession) -> bool:
    """Can a contract be built from what is on file?"""

    # A contract needs a stated oracle, and knowledge can never supply one: the app map records
    # what an app *is*, not what this particular claim should look like when it holds.
    return not missing_required(session) and "success" in session.answers


def interview(session: PrepareSession) -> dict[str, Any]:
    """The payload handed back to the calling agent: what AUA knows, and what it still needs."""

    pending = outstanding(session)
    scope = session.answers.get("scope", "ui")
    signin = session.answers.get("signin", "").strip().lower()
    ranked = rank_seeding(scope=scope, signin_required=signin not in {"none", "not needed", ""})
    questions: list[dict[str, Any]] = []
    for question in pending:
        entry: dict[str, Any] = {
            "key": question.key,
            "ask": question.ask,
            "why": question.why,
            "kind": question.kind,
            "required": question.required,
        }
        if question.choices:
            entry["choices"] = list(question.choices)
        if question.example:
            entry["example"] = question.example
        if question.key == "seeding":
            entry["aua_suggests"] = ranked[0]["strategy"] if ranked else None
            entry["options"] = ranked
        questions.append(entry)
    return {
        "ok": True,
        "prepare_id": session.id,
        "package": session.package,
        "goal": session.goal,
        "already_known": {
            key: [
                {"id": item["id"], "name": item.get("name"), "text": item["text"]} for item in items
            ]
            for key, items in session.known.items()
        },
        "answers": dict(session.answers),
        "questions": questions,
        "missing_required": missing_required(session),
        "ready": is_ready(session),
        "answer_with": (
            "aua prepare answer "
            + session.id
            + " "
            + " ".join(f"--answer {question.key}=<value>" for question in pending)
        ).strip()
        if pending
        else None,
    }
