"""A rollback must never refuse to return the map to a state it demonstrably was in.

`apply` was taught on 2026-08-10 to judge a correction only by what it *introduces*, because
validating the candidate absolutely let one stale row veto every unrelated repair. `rollback` was
never given the same lesson: it validates the restored snapshot absolutely and raises
`snapshot is invalid: ...` on anything it finds, including damage the snapshot already carried.

The synthetic fixture below gives a map two pre-existing violations — duplicate route ids and a
route whose source screen was deleted — then applies and rolls back an unrelated correction. The
rollback must remain available precisely because the restored snapshot did not introduce that
damage.

A snapshot is by definition a state the map was in. Refusing to go back to it is strictly worse
than going back to visible inherited damage, so inherited errors are carried onto the event record
the same way `apply` carries them.
"""

from __future__ import annotations

from android_ui_analyser.memory import RouteEdge
from android_ui_analyser.reconcile import (
    CorrectionOperation,
    ReconciliationStore,
    ResearchReport,
    validate_map,
)
from test_memory import HOME, P, _elements, _store

_WHEN = "2026-08-18T00:00:00Z"


def _damage(store, package: str) -> None:
    """Add the exact shape measured in the wild: a route whose source no longer exists."""
    app = store.load(package)
    app.routes.append(
        RouteEdge(
            id="route_orphan",
            from_screen="deleted_screen",
            to_screen=next(iter(app.screens)),
            action="tap 'Back'",
            last_seen=_WHEN,
        )
    )
    store.save(app)


def _rename(reconciliation, package: str, screen: str, to: str) -> dict:
    task = next(t for t in reconciliation.plan(package) if t.issue_type == "poor_name")
    return reconciliation.submit(
        package,
        ResearchReport(
            task_id=task.id,
            agent="codex",
            verdict="apply",
            rationale="The title identifies the destination.",
            operations=[CorrectionOperation(op="rename", screen_id=screen, value=to)],
        ),
    )


def test_a_rollback_is_not_blocked_by_damage_the_snapshot_already_had(tmp_path) -> None:
    store = _store(tmp_path)
    outcome = store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    _damage(store, P)
    reconciliation = ReconciliationStore(store)
    assert validate_map(store.load(P)), "the map starts damaged, as the real one does"

    result = _rename(reconciliation, P, outcome.name, "settings")
    assert "settings" in store.load(P).screens

    reconciliation.rollback(P, result["event"]["rollback_id"])

    restored = store.load(P)
    assert outcome.name in restored.screens, "the undo must work on a damaged map"
    assert "settings" not in restored.screens


def test_the_inherited_damage_is_recorded_on_the_event_not_swallowed(tmp_path) -> None:
    store = _store(tmp_path)
    outcome = store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    _damage(store, P)
    reconciliation = ReconciliationStore(store)

    result = _rename(reconciliation, P, outcome.name, "settings")
    event = reconciliation.rollback(P, result["event"]["rollback_id"])

    carried = [line for line in event.validation if "deleted_screen" in line]
    assert carried, f"inherited damage must stay visible, got {event.validation}"
    assert any("restored" in line for line in carried), "and be attributed to the restore"
    assert event.rolled_back_at, "the event still records that it was rolled back"


def test_a_rollback_of_a_clean_map_is_unchanged(tmp_path) -> None:
    """The existing clean-map path must behave exactly as before."""
    store = _store(tmp_path)
    outcome = store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    reconciliation = ReconciliationStore(store)

    result = _rename(reconciliation, P, outcome.name, "settings")
    event = reconciliation.rollback(P, result["event"]["rollback_id"])

    restored = store.load(P)
    assert outcome.name in restored.screens and "settings" not in restored.screens
    assert validate_map(restored) == []
    assert not [line for line in event.validation if "restored with pre-existing" in line]


def test_an_unknown_rollback_id_still_fails_loudly(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    reconciliation = ReconciliationStore(store)

    try:
        reconciliation.rollback(P, "correction_does_not_exist")
    except ValueError as err:
        assert "unknown rollback id" in str(err)
    else:  # pragma: no cover - the guard must not be removed by this change
        raise AssertionError("an unknown rollback id must still raise")
