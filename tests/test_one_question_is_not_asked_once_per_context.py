"""A question about a screen is one question, however many flag contexts asked it.

`plan(context_id=X)` mints `_stable_id("research", package, context_id or "all", issue.id)` and
preserves the rows belonging to every *other* context. So auditing under a new flag context adds a
fresh copy of every finding that context can see, and nothing ever removes the old copy.

The synthetic regression creates the same screen question under several contexts and proves the
merge keeps one context-independent task. The backlog should describe how much is wrong, not how
often the map was audited.
"""

from __future__ import annotations

from android_ui_analyser.reconcile import (
    ReconciliationStore,
    _dedupe_research_tasks,
    _question_identity,
)
from test_memory import HOME, P, _elements, _store

_WHEN = "2026-08-18T00:00:00Z"


def _task(
    task_id: str, *, context_id, screen="screen_abc", status="open", created=_WHEN, kind="poor_name"
):
    return {
        "id": task_id,
        "package": P,
        "app_version": None,
        "context_id": context_id,
        "flags": {},
        "issue_type": kind,
        "affected_ids": [screen],
        "observations": ["Screen has a generated name."],
        "questions": ["Which app destination does this screen represent?"],
        "created_at": created,
        "status": status,
    }


def test_the_same_question_from_ten_contexts_becomes_one_row() -> None:
    """The worst real case: ten rows, one screen, one question."""
    rows = [_task("research_all", context_id=None), _task("research_default", context_id="default")]
    rows += [_task(f"research_flag{n}", context_id=f"flags-experiment_{n}") for n in range(8)]

    merged = _dedupe_research_tasks(rows)

    assert len(merged) == 1
    assert len({_question_identity(r) for r in rows}) == 1, "they were always one question"


def test_the_surviving_row_keeps_the_age_the_question_really_has() -> None:
    rows = [
        _task("research_new", context_id="flags-b", created="2026-08-18T00:00:00Z"),
        _task("research_old", context_id="flags-a", created="2026-08-03T00:00:00Z"),
    ]

    merged = _dedupe_research_tasks(rows)

    assert merged[0]["created_at"] == "2026-08-03T00:00:00Z"


def test_a_question_answered_in_one_context_is_not_reopened_in_another() -> None:
    """This is the zombie: the rename stamps `explicit`, so a reopened row can never be answered."""
    rows = [
        _task("research_fresh", context_id="flags-b", status="open"),
        _task("research_done", context_id="flags-a", status="applied"),
    ]

    merged = _dedupe_research_tasks(rows)

    assert len(merged) == 1
    assert merged[0]["status"] == "applied", "a settled question stays settled"


def test_the_survivor_is_answerable_wherever_the_caller_is_standing() -> None:
    """`ask_about_current_screen` only offers a task whose context matches, or is None."""
    rows = [
        _task("research_a", context_id="flags-a"),
        _task("research_b", context_id="flags-b"),
    ]

    merged = _dedupe_research_tasks(rows)

    assert merged[0]["context_id"] is None, "collapsed across contexts, so not per-context"


def test_a_question_only_one_context_asks_keeps_that_context() -> None:
    rows = [_task("research_only", context_id="flags-a")]

    assert _dedupe_research_tasks(rows)[0]["context_id"] == "flags-a"


def test_different_questions_about_one_screen_are_both_kept() -> None:
    """Merging is by question, not by screen — a screen can be wrong in more than one way."""
    rows = [
        _task("research_name", context_id=None, kind="poor_name"),
        _task("research_stale", context_id=None, kind="stale_screen"),
    ]

    merged = _dedupe_research_tasks(rows)

    assert {r["issue_type"] for r in merged} == {"poor_name", "stale_screen"}


def test_auditing_the_same_map_again_does_not_grow_the_backlog(tmp_path) -> None:
    """The invariant that keeps it from coming back: one stored row per distinct question."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    reconciliation = ReconciliationStore(store)

    for context_id in (None, "default", "flags-a", "flags-b", None, "flags-a"):
        reconciliation.plan(P, context_id=context_id)

    stored = store.load(P).research_tasks
    identities = [_question_identity(task) for task in stored]
    assert len(identities) == len(set(identities)), (
        f"{len(identities)} rows for {len(set(identities))} distinct questions: {stored}"
    )


def test_a_row_without_an_id_is_not_silently_dropped() -> None:
    rows = [
        {"issue_type": "poor_name", "affected_ids": ["s"]},
        _task("research_x", context_id=None),
    ]

    merged = _dedupe_research_tasks(rows)

    assert len(merged) == 2, "malformed rows pass through rather than vanish"
