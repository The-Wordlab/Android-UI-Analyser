"""Many questions about one screen are one rename, and one disagreement is not settled by order.

The synthetic regression creates ten open naming questions for one screen — one per flag context
audited, because `plan(context_id=X)` only replaces the tasks for context `X`. Those rows are
siblings of a question another row already settles.

A sibling that agrees is `applied`, because the rename stamps `name_source = "explicit"` and its
question can never be asked again — leaving it open would mint a permanent zombie. A sibling that
disagrees is left open and reported, because ten different names for one screen is a real
disagreement and list order is not an answer to it.
"""

from __future__ import annotations

from android_ui_analyser.reconcile import ReconciliationStore
from test_hundreds_of_answers_are_one_transaction import P, _naming_tasks, _seed


def _siblings(store, screen_id: str, count: int) -> list[str]:
    """`count` extra open naming questions about one screen, as flag contexts accumulate them."""
    app = store.load(P)
    ids = []
    for n in range(count):
        task_id = f"research_sib{n:04d}"
        ids.append(task_id)
        app.research_tasks.append(
            {
                "id": task_id,
                "package": P,
                "app_version": None,
                "context_id": f"flags-experiment_{n}",
                "flags": {},
                "issue_type": "poor_name",
                "affected_ids": [screen_id],
                "observations": ["Screen has a generated name."],
                "questions": ["Which app destination does this screen represent?"],
                "created_at": "2026-08-18T00:00:00Z",
                "status": "open",
            }
        )
    store.save(app)
    return ids


def test_sibling_tasks_for_one_screen_are_settled_by_one_rename(tmp_path) -> None:
    store, names = _seed(tmp_path, 2)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    target = names[0]
    screen_id = store.load(P).screens[target].id
    sibs = _siblings(store, screen_id, 9)

    result = rec.answer_many(
        P,
        [(tasks[target], "catalog overview")] + [(s, "catalog overview") for s in sibs],
    )

    assert result["renamed"] == 1, "ten questions, one rename"
    assert result["settled"] == 9
    assert result["skipped"] == 0
    app = store.load(P)
    assert "catalog_overview" in app.screens
    closed = {t["id"] for t in app.research_tasks if t["status"] == "applied"}
    assert set(sibs) <= closed, "no sibling is left as a permanently unanswerable zombie"
    assert tasks[target] in closed


def test_disagreeing_answers_for_one_screen_surface_instead_of_being_picked(tmp_path) -> None:
    store, names = _seed(tmp_path, 2)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    target = names[0]
    screen_id = store.load(P).screens[target].id
    sibs = _siblings(store, screen_id, 3)

    result = rec.answer_many(
        P,
        [(tasks[target], "catalog overview")]
        + [
            (s, name)
            for s, name in zip(
                sibs,
                ("catalog overview", "catalog tab", "browse"),
                strict=True,
            )
        ],
    )

    assert result["renamed"] == 1
    assert result["settled"] == 1, "the one that agrees"
    assert result["skipped"] == 2, "the two that disagree"
    codes = {row["code"] for row in result["skipped_answers"]}
    assert codes == {"conflicting_answer"}
    app = store.load(P)
    assert "catalog_overview" in app.screens
    still_open = {t["id"] for t in app.research_tasks if t["status"] == "open"}
    assert set(sibs[1:]) <= still_open, "a disagreement stays answerable"


def test_confirming_the_name_a_screen_already_has_does_not_self_alias(tmp_path) -> None:
    """The old rename branch popped and re-inserted the record, aliasing the screen to itself."""
    store, names = _seed(tmp_path, 2)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    target = names[0]

    result = rec.answer_many(P, [(tasks[target], target)])

    assert result["confirmed"] == 1
    assert result["renamed"] == 0
    app = store.load(P)
    assert target in app.screens
    assert app.screens[target].aliases == [], "a confirmation is not a rename to itself"
    assert app.screens[target].name_source == "explicit"
    closed = {t["id"] for t in app.research_tasks if t["status"] == "applied"}
    assert tasks[target] in closed, "a confirmation still settles the question"
