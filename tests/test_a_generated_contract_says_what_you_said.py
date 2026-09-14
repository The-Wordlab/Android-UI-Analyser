"""A generated oracle is only safe if it is a translation, and if you can see the translation.

The risk in generating a contract from an interview is a checkpoint that looks right and asserts
something else.  These pin the two properties that make that impossible to do quietly: every
checkpoint traces to exactly one answer, and the emitted YAML is re-parsed by the authored
contract schema before anyone sees it.
"""

from __future__ import annotations

import pytest
import yaml

from android_ui_analyser.errors import UsageError
from android_ui_analyser.prepare import (
    PrepareSession,
    build_contract_document,
    knowledge_writes,
    open_preparation,
    prepared,
    record_answers,
    render_contract_yaml,
    scenario_name,
    setup_plan,
)
from android_ui_analyser.session_contracts import parse_session_contract_yaml

GOAL = "the hub badge shows once on first open"

ANSWERS = {
    "build": "/tmp/app-debug.apk",
    "signin": "tap `Continue as guest` on the welcome screen",
    "precondition": "the DataStore key hubBadgeSeen is absent",
    "seeding": "datastore",
    "scope": "ui",
    "success": "rid:hubBadge,text:New",
    "repeat": "!rid:hubBadge",
    "restore": "none",
}


def _answered(**overrides: str) -> PrepareSession:
    session = open_preparation(
        package="com.example.app", goal=GOAL, now="2026-09-14T00:00:00+00:00", session_id="prep-1"
    )
    record_answers(session, {**ANSWERS, **overrides})
    return session


def test_each_checkpoint_names_the_answer_it_came_from() -> None:
    built = build_contract_document(_answered())
    assert [trace["checkpoint"] for trace in built["provenance"]] == ["observed", "not_repeated"]
    assert built["provenance"][0]["from_answer"] == "success"
    assert built["provenance"][0]["you_said"] == "rid:hubBadge,text:New"
    assert built["provenance"][1]["asserts"] == [{"assert": {"rid": "hubBadge", "absent": True}}]


def test_the_negative_half_of_a_once_only_claim_becomes_its_own_checkpoint() -> None:
    document = build_contract_document(_answered())["document"]
    ids = [checkpoint["id"] for checkpoint in document["checkpoints"]]
    assert ids == ["observed", "not_repeated"]

    # A claim with no second half gets one checkpoint, not an invented one.
    session = open_preparation(
        package="com.example.app", goal="the hub lists my apps", session_id="prep-2"
    )
    record_answers(session, {**{k: v for k, v in ANSWERS.items() if k != "repeat"}})
    single = build_contract_document(session)["document"]
    assert [checkpoint["id"] for checkpoint in single["checkpoints"]] == ["observed"]


def test_the_emitted_yaml_is_what_the_authored_schema_accepts() -> None:
    text = render_contract_yaml(_answered())
    contract = parse_session_contract_yaml(text)
    assert [checkpoint.id for checkpoint in contract.checkpoints] == ["observed", "not_repeated"]
    assert contract.checkpoints[0].description == GOAL
    # Internal completion policy is never authored into the file.
    assert "proof_mode" not in text and "manual_completion_allowed" not in text
    assert yaml.safe_load(text)["version"] == 1


def test_a_contract_is_refused_while_the_oracle_is_unstated() -> None:
    session = open_preparation(package="com.example.app", goal=GOAL, session_id="prep-3")
    record_answers(session, {"scope": "ui", "seeding": "datastore"})
    with pytest.raises(UsageError, match="`success` is unanswered"):
        build_contract_document(session)


def test_an_assertion_the_contract_schema_would_reject_fails_here_not_on_a_leased_device() -> None:
    session = _answered()
    # Bypass answer validation the way a future caller might, and prove the round trip still bites.
    session.answers["success"] = "rid:"
    with pytest.raises(UsageError):
        render_contract_yaml(session)


def test_the_setup_plan_is_ordered_and_traceable() -> None:
    steps = setup_plan(_answered())
    assert [step["step"] for step in steps] == ["install", "seed", "sign in"]
    assert steps[1]["calls"][0].startswith("aua datastore")
    assert steps[1]["proves_nothing_about"]
    assert all(step["from_answer"] for step in steps)


def test_a_sign_in_that_is_not_needed_is_not_a_step() -> None:
    steps = setup_plan(_answered(signin="none"))
    assert [step["step"] for step in steps] == ["install", "seed"]


def test_only_facts_about_the_app_are_remembered() -> None:
    remembered = {item["text"] for item in knowledge_writes(_answered())}
    assert ANSWERS["signin"] in remembered
    assert ANSWERS["precondition"] in remembered
    # The scope of one test and its expected screen are decisions, not facts about the app.
    assert ANSWERS["success"] not in remembered
    assert "ui" not in remembered


def test_the_handback_carries_the_run_command_and_the_way_to_repeat_it() -> None:
    payload = prepared(_answered(), contract_path="/tmp/badge.yaml", artifacts_dir="/tmp/art")
    assert payload["scenario"] == "the-hub-badge-shows-once-on-first-open"
    assert "--contract /tmp/badge.yaml" in payload["run"]
    assert "--apk /tmp/app-debug.apk" in payload["run"]
    assert "--artifacts-dir /tmp/art --evidence all" in payload["run"]
    assert "--fresh" not in payload["run"]  # seeded via datastore, so nothing is wiped
    assert payload["run_again"].startswith("aua prepare run ")
    assert payload["provenance"] and payload["confirm"]


def test_choosing_the_wipe_puts_the_wipe_in_the_command() -> None:
    payload = prepared(_answered(seeding="reinstall"), contract_path="/tmp/c.yaml")
    assert "--fresh --yes" in payload["run"]


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("Badge shows once!", "badge-shows-once"),
        ("   ", "scenario"),
        ("a" * 80, "a" * 48),
    ],
)
def test_a_scenario_name_is_stable_and_file_safe(goal: str, expected: str) -> None:
    assert scenario_name(goal) == expected
