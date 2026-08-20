"""One unusable answer in a batch must cost one row, never the whole batch.

`answer` raises on a task it cannot settle, which is right for an inline answer typed by hand: it
is loud and the caller is standing right there. A batch is different. Measured 2026-08-18, the
496 open naming questions on one map include one that names a screen no longer present and twenty
about screens already named explicitly — so a single hard raise would discard hundreds of good
answers because of a handful of stale rows.

Every input therefore gets an outcome with a code and a sentence the caller can act on, and only
the rows that could not be applied are left `open`.
"""

from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser.reconcile import ReconciliationStore
from test_hundreds_of_answers_are_one_transaction import P, _naming_tasks, _seed


def _add_task(store, *, task_id: str, issue_type: str, affected: list[str]) -> None:
    app = store.load(P)
    app.research_tasks.append(
        {
            "id": task_id,
            "package": P,
            "app_version": None,
            "context_id": None,
            "flags": {},
            "issue_type": issue_type,
            "affected_ids": affected,
            "observations": ["synthetic row for this test"],
            "questions": ["What is this?"],
            "created_at": "2026-08-18T00:00:00Z",
            "status": "open",
        }
    )
    store.save(app)


def test_a_colliding_name_is_skipped_and_the_rest_apply(tmp_path) -> None:
    store, names = _seed(tmp_path, 4)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    taken = names[0]  # an existing screen name nobody vacates

    result = rec.answer_many(
        P,
        [
            (tasks[names[1]], taken),
            (tasks[names[2]], "good one"),
            (tasks[names[3]], "good two"),
        ],
    )

    assert result["renamed"] == 2
    assert result["skipped"] == 1
    codes = {row["task_id"]: row["code"] for row in result["skipped_answers"]}
    assert codes[tasks[names[1]]] == "name_collision"
    app = store.load(P)
    assert {"good_one", "good_two"} <= set(app.screens)


def test_two_answers_that_slug_alike_do_not_both_win(tmp_path) -> None:
    store, names = _seed(tmp_path, 3)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)

    result = rec.answer_many(P, [(tasks[names[0]], "Dev Tools"), (tasks[names[1]], "dev-tools")])

    assert result["renamed"] == 1, "both slug to dev_tools; the first claims it"
    assert result["skipped"] == 1
    assert result["skipped_answers"][0]["code"] == "name_taken_in_batch"
    assert "dev_tools" in store.load(P).screens


def test_a_name_freed_by_an_earlier_rename_is_not_reused(tmp_path) -> None:
    """The vacated name is still the first screen's alias, so handing it over shadows it."""
    store, names = _seed(tmp_path, 3)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    vacated = names[0]

    result = rec.answer_many(P, [(tasks[vacated], "moved away"), (tasks[names[1]], vacated)])

    assert result["renamed"] == 1
    assert result["skipped_answers"][0]["code"] == "name_freed_in_batch"
    app = store.load(P)
    assert "moved_away" in app.screens
    assert vacated in app.screens["moved_away"].aliases


def test_a_skipped_task_stays_open_and_its_screen_is_untouched(tmp_path) -> None:
    """A skipped row must not become a zombie.

    Only a row that was actually applied may be stamped `applied`: the rename sets
    `name_source = "explicit"`, which permanently stops `ask_about_current_screen` offering that
    screen, so a task closed without a rename could never be answered by anyone again.
    """
    store, names = _seed(tmp_path, 3)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    blocked = tasks[names[1]]

    rec.answer_many(P, [(blocked, names[0]), (tasks[names[2]], "fine")])

    app = store.load(P)
    row = next(t for t in app.research_tasks if t["id"] == blocked)
    assert row["status"] == "open", "an unapplied answer leaves its question answerable"
    assert names[1] in app.screens, "and leaves its screen exactly as it was"
    assert app.screens[names[1]].aliases == [], "no half-applied rename"


def test_a_route_question_in_the_batch_is_skipped_not_fatal(tmp_path) -> None:
    store, names = _seed(tmp_path, 3)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    _add_task(
        store, task_id="research_routeconflict", issue_type="route_conflict", affected=["route_x"]
    )

    result = rec.answer_many(
        P, [("research_routeconflict", "nope"), (tasks[names[0]], "still lands")]
    )

    assert result["renamed"] == 1
    assert result["skipped_answers"][0]["code"] == "not_a_naming_question"
    assert "still_lands" in store.load(P).screens


def test_a_task_naming_a_deleted_screen_is_skipped(tmp_path) -> None:
    store, names = _seed(tmp_path, 2)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    _add_task(store, task_id="research_ghost", issue_type="poor_name", affected=["screen_gone"])

    result = rec.answer_many(P, [("research_ghost", "ghost"), (tasks[names[0]], "real")])

    assert result["renamed"] == 1
    assert result["skipped_answers"][0]["code"] == "unknown_screen"
    assert "real" in store.load(P).screens


def test_the_event_records_one_outcome_per_answer_with_a_reason(tmp_path) -> None:
    store, names = _seed(tmp_path, 4)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    rows = [
        (tasks[names[0]], "one"),
        (tasks[names[1]], "one"),  # slug collision inside the batch
        (tasks[names[2]], "two"),
        ("research_unknown", "three"),
    ]

    result = rec.answer_many(P, rows)

    event = json.loads(Path(str(result["event"])).read_text(encoding="utf-8"))
    assert len(event["outcomes"]) == len(rows), "every input is accounted for"
    for outcome in event["outcomes"]:
        assert outcome["reason"], f"no reason on {outcome}"
        assert outcome["code"]
    assert event["task_ids"] == [
        o["task_id"] for o in event["outcomes"] if o["status"] == "applied"
    ]
    assert event["report"] is None, "a bulk event has no single authoring report"
    assert event["task_id"] == ""
