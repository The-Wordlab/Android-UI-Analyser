"""Preparation is an interview, and the value of an interview is in what it does not ask.

These cover the pure half: which questions a goal raises, which ones the app map has already
answered, what a bad answer is told, and why AUA suggests the seeding strategy it does.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.memory import AppMap, AppMemoryStore
from android_ui_analyser.prepare import (
    QUESTION_BY_KEY,
    SEEDING_KEYS,
    PrepareSession,
    goal_is_once_only,
    interview,
    is_ready,
    missing_required,
    open_preparation,
    outstanding,
    parse_answer_pairs,
    parse_predicate_terms,
    rank_seeding,
    record_answers,
    seeding_recommendation,
    validate_answer,
)
from conftest import make_config

GOAL = "the hub badge shows once on first open"


def _session(**kwargs: object) -> PrepareSession:
    return open_preparation(
        package="com.example.app",
        goal=str(kwargs.pop("goal", GOAL)),
        now="2026-09-14T00:00:00+00:00",
        session_id="prep-test",
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_once_only_claim_raises_the_second_look_and_a_plain_one_does_not() -> None:
    once = {question.key for question in outstanding(_session())}
    plain = {question.key for question in outstanding(_session(goal="the hub lists my apps"))}
    assert "repeat" in once
    assert "repeat" not in plain
    # Everything else is asked either way; the conditional question is the only difference.
    assert once - plain == {"repeat"}


@pytest.mark.parametrize(
    "goal",
    [
        "badge shows once",
        "the tooltip appears on first run",
        "shown on first-launch only",
        "the banner is not shown again after dismissal",
        "a one-shot coach mark",
    ],
)
def test_the_phrasings_that_mean_bounded_are_recognised(goal: str) -> None:
    assert goal_is_once_only(goal)


@pytest.mark.parametrize("goal", ["the hub lists my apps", "search returns results", ""])
def test_an_unbounded_claim_is_not_treated_as_bounded(goal: str) -> None:
    assert not goal_is_once_only(goal)


def test_a_fact_already_on_file_is_not_asked_for_again(tmp_path) -> None:
    store = AppMemoryStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.remember_knowledge(
        "com.example.app",
        kind="recipe",
        text="Tap `Continue as guest` on the welcome screen.",
        name="guest_signin",
        aliases=["log in", "sign in"],
        source="agent",
    )
    app_map = store.load("com.example.app") or AppMap(package="com.example.app")

    session = _session(app_map=app_map)

    assert "signin" in session.known
    assert "signin" not in {question.key for question in outstanding(session)}
    # It is still visible to the caller, so an agent can correct a stale recipe rather than
    # discover at run time that AUA used one.
    assert interview(session)["already_known"]["signin"][0]["name"] == "guest_signin"


def test_an_empty_app_map_asks_everything_it_should() -> None:
    keys = {question.key for question in outstanding(_session())}
    assert keys == {
        "build",
        "signin",
        "precondition",
        "seeding",
        "scope",
        "success",
        "repeat",
        "restore",
    }


def test_an_answer_outside_the_choices_is_refused_by_name() -> None:
    with pytest.raises(UsageError) as excinfo:
        validate_answer(QUESTION_BY_KEY["scope"], "integration")
    assert "ui" in str(excinfo.value) and "e2e" in str(excinfo.value)


def test_an_unknown_answer_key_is_refused_rather_than_silently_kept() -> None:
    session = _session()
    with pytest.raises(UsageError, match="unknown prepare answer"):
        record_answers(session, {"scoop": "ui"})
    assert session.answers == {}


def test_an_empty_answer_is_not_an_answer() -> None:
    with pytest.raises(UsageError, match="empty"):
        validate_answer(QUESTION_BY_KEY["signin"], "   ")


def test_an_answer_pair_keeps_equals_signs_inside_its_value() -> None:
    assert parse_answer_pairs(["precondition=key hub_badge_seen=false"]) == {
        "precondition": "key hub_badge_seen=false"
    }
    with pytest.raises(UsageError, match="key=value"):
        parse_answer_pairs(["justakey"])


@pytest.mark.parametrize(
    ("terms", "expected"),
    [
        ("rid:hubBadge", [{"assert": {"rid": "hubBadge", "exists": True}}]),
        ("id:hubBadge", [{"assert": {"rid": "hubBadge", "exists": True}}]),
        ("text:New", [{"assert": {"text": "New", "exists": True}}]),
        ("desc:Badge", [{"assert": {"desc": "Badge", "exists": True}}]),
        ("New", [{"assert": {"text": "New", "exists": True}}]),
        ("!rid:hubBadge", [{"assert": {"rid": "hubBadge", "absent": True}}]),
        (
            "rid:hubBadge, !text:Loading",
            [
                {"assert": {"rid": "hubBadge", "exists": True}},
                {"assert": {"text": "Loading", "absent": True}},
            ],
        ),
    ],
)
def test_a_predicate_becomes_an_assertion_without_interpretation(terms, expected) -> None:
    assert parse_predicate_terms(terms, where="`success`") == expected


def test_a_predicate_with_nothing_in_it_is_refused() -> None:
    with pytest.raises(UsageError, match="at least one predicate term"):
        parse_predicate_terms("  ,  ", where="`success`")


def test_aua_suggests_the_reversible_local_seed_for_a_ui_check() -> None:
    best = seeding_recommendation(scope="ui")
    assert best is not None and best["strategy"] == "datastore"
    assert best["proves_nothing_about"]
    assert any("put it back" in reason for reason in best["reasons"])


def test_faking_the_backend_is_pushed_down_when_the_backend_is_the_point() -> None:
    ui_rank = [entry["strategy"] for entry in rank_seeding(scope="ui")]
    e2e_rank = [entry["strategy"] for entry in rank_seeding(scope="e2e")]
    assert ui_rank.index("mock") < e2e_rank.index("mock")
    e2e_mock = next(entry for entry in rank_seeding(scope="e2e") if entry["strategy"] == "mock")
    assert any("end-to-end" in reason for reason in e2e_mock["reasons"])


def test_a_wipe_costs_less_when_there_was_no_sign_in_to_lose() -> None:
    with_signin = next(
        entry for entry in rank_seeding(signin_required=True) if entry["strategy"] == "reinstall"
    )
    without = next(
        entry for entry in rank_seeding(signin_required=False) if entry["strategy"] == "reinstall"
    )
    assert without["score"] > with_signin["score"]


def test_only_the_offered_strategies_are_ranked() -> None:
    ranked = rank_seeding(available=["datastore", "reinstall"])
    assert [entry["strategy"] for entry in ranked] == ["datastore", "reinstall"]
    with pytest.raises(UsageError, match="unknown seeding strategy"):
        rank_seeding(available=["telepathy"])
    assert set(SEEDING_KEYS) >= {"datastore", "database", "flags", "mock", "reinstall", "ui"}


def test_a_session_is_not_ready_until_the_oracle_is_stated(tmp_path) -> None:
    session = _session()
    assert not is_ready(session)
    record_answers(
        session,
        {
            "build": "/tmp/app-debug.apk",
            "signin": "none",
            "precondition": "datastore key hub_badge_seen absent",
            "seeding": "datastore",
            "scope": "ui",
            "repeat": "!rid:hubBadge",
            "restore": "none",
        },
    )
    # Everything but the oracle is answered, and it is still not ready.
    assert missing_required(session) == ["success"]
    assert not is_ready(session)

    record_answers(session, {"success": "rid:hubBadge,text:New"})
    assert is_ready(session) and missing_required(session) == []


def test_the_interview_tells_the_agent_the_exact_command_to_answer_with() -> None:
    payload = interview(_session())
    assert payload["prepare_id"] == "prep-test"
    assert payload["answer_with"].startswith("aua prepare answer prep-test ")
    assert "--answer success=<value>" in payload["answer_with"]
    seeding = next(entry for entry in payload["questions"] if entry["key"] == "seeding")
    assert seeding["aua_suggests"] == "datastore"
    assert {option["strategy"] for option in seeding["options"]} == set(SEEDING_KEYS)


def test_a_prepare_session_survives_a_round_trip_and_rejects_a_foreign_version() -> None:
    session = _session()
    record_answers(session, {"scope": "e2e"})
    restored = PrepareSession.from_dict(session.to_dict())
    assert restored.to_dict() == session.to_dict()
    with pytest.raises(UsageError, match="unsupported prepare session version"):
        PrepareSession.from_dict({**session.to_dict(), "schema_version": 99})


@pytest.mark.parametrize("terms", ["rid:", "!text:  ", "desc:,text:New"])
def test_a_selector_with_no_value_is_refused_rather_than_read_as_text(terms: str) -> None:
    # `rid:` once fell through to "text matching the literal string rid:", which is an assertion
    # that passes on the wrong screen and never explains itself.
    with pytest.raises(UsageError, match="selector with no value"):
        parse_predicate_terms(terms, where="`success`")


def test_a_label_with_a_comma_in_it_stays_one_assertion() -> None:
    # Found the first time this ran against a real app: the badged tab's accessibility
    # description was `Create, New`, and a bare comma split turned one assertion into two - the
    # second of them asserting the word "New" somewhere, anywhere, on the screen.
    assert parse_predicate_terms('desc:"Create, New"', where="`success`") == [
        {"assert": {"desc": "Create, New", "exists": True}}
    ]
    assert parse_predicate_terms("rid:bottomBarAppsHub, !desc:'Create, New'", where="`repeat`") == [
        {"assert": {"rid": "bottomBarAppsHub", "exists": True}},
        {"assert": {"desc": "Create, New", "absent": True}},
    ]


def test_a_predicate_that_never_closes_its_quote_is_refused() -> None:
    with pytest.raises(UsageError, match="unbalanced"):
        parse_predicate_terms('desc:"Create, New', where="`success`")


def test_a_loosely_related_note_does_not_stand_in_for_an_answer(tmp_path) -> None:
    # The first real app this ran against had a note about forced dark theme. A permissive match
    # let it answer "where is the build", "how do you sign in" and "what is the pre-condition" at
    # once - which does not merely add noise, it REMOVES those questions, so the run would start
    # with no APK and nobody would ever have been asked for one.
    store = AppMemoryStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.remember_knowledge(
        "com.example.app",
        kind="note",
        text=(
            "A fresh install is forced to dark theme on every launch where the session is null. "
            "Set the theme only after a session exists."
        ),
        source="agent",
    )
    app_map = store.load("com.example.app") or AppMap(package="com.example.app")

    session = open_preparation(
        package="com.example.app",
        goal="the badge shows once on first open after a fresh install",
        app_map=app_map,
        session_id="prep-loose",
    )

    assert session.known == {}
    assert {"build", "signin", "precondition", "seeding"} <= {
        question.key for question in outstanding(session)
    }
