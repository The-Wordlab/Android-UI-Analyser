"""Preparation spans processes, so the interview has to survive between them.

AUA asks, the calling agent goes and reads its own source, and comes back a minute later in a
different process.  These drive the real CLI end to end to pin what has to persist: the questions,
the answers, the durable facts, and the scenario the next run looks up instead of asking again.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.session_contracts import parse_session_contract_yaml

runner = CliRunner()

PACKAGE = "com.example.app"
GOAL = "the hub badge shows once on first open"


def _aua(tmp_path: Path, *args: str) -> dict:
    # `_aua_isolate_state` already points memory and cache at this test's tmp_path, which is
    # what makes each of these a genuinely cold app map.
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return json.loads(result.stdout)


def _start(tmp_path: Path, goal: str = GOAL) -> dict:
    return _aua(tmp_path, "prepare", "start", "--goal", goal, "--app", PACKAGE)


def test_the_first_call_returns_questions_and_the_command_that_answers_them(tmp_path) -> None:
    payload = _start(tmp_path)

    assert payload["ok"] and payload["prepare_id"].startswith("prep-")
    keys = {question["key"] for question in payload["questions"]}
    assert {"build", "signin", "precondition", "seeding", "scope", "success", "repeat"} <= keys
    assert payload["answer_with"].startswith(f"aua prepare answer {payload['prepare_id']}")
    assert not payload["ready"]

    seeding = next(q for q in payload["questions"] if q["key"] == "seeding")
    assert seeding["aua_suggests"] == "datastore"
    assert all(option["proves_nothing_about"] for option in seeding["options"])


def test_an_interview_is_resumable_from_a_separate_process(tmp_path) -> None:
    opened = _start(tmp_path)
    prepare_id = opened["prepare_id"]

    part = _aua(
        tmp_path, "prepare", "answer", prepare_id, "--app", PACKAGE, "--answer", "scope=e2e"
    )
    assert part["answers"]["scope"] == "e2e"
    assert not part["ready"]

    # A third invocation still sees it, and the seeding advice has moved with the scope.
    shown = _aua(tmp_path, "prepare", "show", prepare_id, "--app", PACKAGE)
    assert shown["answers"]["scope"] == "e2e"
    seeding = next(q for q in shown["questions"] if q["key"] == "seeding")
    assert [option["strategy"] for option in seeding["options"]].index("mock") > 0

    listed = _aua(tmp_path, "prepare", "list", "--app", PACKAGE)
    assert [entry["prepare_id"] for entry in listed["in_progress"]] == [prepare_id]


def _answer_everything(tmp_path: Path, prepare_id: str, **extra: str) -> dict:
    answers = {
        "build": "/tmp/app-debug.apk",
        "signin": "tap `Continue as guest` on the welcome screen",
        "precondition": "the DataStore key hubBadgeSeen is absent",
        "seeding": "datastore",
        "scope": "ui",
        "success": "rid:hubBadge,text:New",
        "repeat": "!rid:hubBadge",
        "restore": "none",
        **extra,
    }
    flags: list[str] = []
    for key, value in answers.items():
        flags += ["--answer", f"{key}={value}"]
    return _aua(tmp_path, "prepare", "answer", prepare_id, "--app", PACKAGE, *flags)


def test_the_last_answer_writes_a_contract_that_says_what_was_agreed(tmp_path) -> None:
    done = _answer_everything(tmp_path, _start(tmp_path)["prepare_id"])

    assert done["saved"] and done["scenario"] == "the-hub-badge-shows-once-on-first-open"
    contract = parse_session_contract_yaml(Path(done["contract_path"]).read_text())
    assert [checkpoint.id for checkpoint in contract.checkpoints] == ["observed", "not_repeated"]
    # The caller is shown the translation, never asked to trust it.
    assert {trace["from_answer"] for trace in done["provenance"]} == {"success", "repeat"}
    assert "--contract" in done["run"] and "--apk /tmp/app-debug.apk" in done["run"]


def test_the_second_time_the_same_app_is_prepared_the_known_facts_are_not_re_asked(
    tmp_path,
) -> None:
    _answer_everything(tmp_path, _start(tmp_path)["prepare_id"])

    again = _start(tmp_path, goal="the streak badge shows once on first open")
    still_asked = {question["key"] for question in again["questions"]}
    assert "signin" not in still_asked
    assert "build" not in still_asked
    # The oracle is never inherited: what *this* claim looks like is not a fact about the app.
    assert {"success", "repeat", "scope"} <= still_asked
    assert again["reused_from_memory"]["signin"]["answer"].startswith("tap `Continue as guest`")


def test_a_saved_scenario_is_listed_with_the_command_that_repeats_it(tmp_path) -> None:
    _answer_everything(tmp_path, _start(tmp_path)["prepare_id"])

    listed = _aua(tmp_path, "prepare", "list", "--app", PACKAGE)
    assert listed["in_progress"] == []  # the finished interview is not left lying around
    scenario = listed["scenarios"][0]
    assert scenario["scenario"] == "the-hub-badge-shows-once-on-first-open"
    assert scenario["run"].startswith("aua prepare run ")
    assert yaml.safe_load(Path(scenario["contract"]).read_text())["version"] == 1


def test_declining_to_remember_leaves_the_app_map_alone(tmp_path) -> None:
    prepare_id = _start(tmp_path)["prepare_id"]
    flags: list[str] = []
    for key, value in {
        "build": "/tmp/app-debug.apk",
        "signin": "guest",
        "precondition": "hubBadgeSeen absent",
        "seeding": "datastore",
        "scope": "ui",
        "success": "rid:hubBadge",
        "repeat": "!rid:hubBadge",
    }.items():
        flags += ["--answer", f"{key}={value}"]
    done = _aua(
        tmp_path, "prepare", "answer", prepare_id, "--app", PACKAGE, "--no-remember", *flags
    )
    assert done["remembered_ids"] == []
    again = _start(tmp_path, goal="another badge on first open")
    assert "signin" in {question["key"] for question in again["questions"]}


def test_a_bad_answer_is_refused_by_name_and_nothing_is_written(tmp_path) -> None:
    prepare_id = _start(tmp_path)["prepare_id"]
    result = runner.invoke(
        app, ["prepare", "answer", prepare_id, "--app", PACKAGE, "--answer", "scope=integration"]
    )

    assert result.exit_code != 0
    assert "e2e" in result.output
    assert _aua(tmp_path, "prepare", "show", prepare_id, "--app", PACKAGE)["answers"] == {}


def test_an_unknown_prepare_id_says_how_to_get_a_real_one(tmp_path) -> None:
    result = runner.invoke(app, ["prepare", "show", "prep-nope", "--app", PACKAGE])

    assert result.exit_code != 0
    assert "aua prepare start" in result.output


def test_an_abandoned_interview_can_be_dropped_without_touching_the_scenarios(tmp_path) -> None:
    kept = _start(tmp_path)["prepare_id"]
    _answer_everything(tmp_path, kept)
    abandoned = _start(tmp_path, goal="something else entirely")["prepare_id"]

    assert _aua(tmp_path, "prepare", "discard", abandoned, "--app", PACKAGE)["discarded"]

    listed = _aua(tmp_path, "prepare", "list", "--app", PACKAGE)
    assert listed["in_progress"] == []
    assert len(listed["scenarios"]) == 1

    # Dropping one that is not there says so rather than pretending.
    assert not _aua(tmp_path, "prepare", "discard", abandoned, "--app", PACKAGE)["discarded"]
