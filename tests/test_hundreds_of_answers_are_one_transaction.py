"""Answering many naming questions must cost one transaction, not one per answer.

`apply` writes a full before/after copy of the map per correction. On a large map, settling a
naming backlog one row at a time multiplies both snapshots and validation passes until a bulk
operation is impractical.

`answer_many` makes the whole batch one deep copy, one validation pass, one snapshot pair, and one
rollback id. Per-item skips, batch-level atomicity: one bad row costs one row, a failed save costs
the whole batch and puts the map back.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.reconcile import ReconciliationStore
from test_memory import _elements, _hier, _node, _store

P = "com.example.app"


# Distinct vocabularies per screen: the map merges screens whose signatures agree within
# `drift_threshold`, so near-identical fixtures collapse into one record.
_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
]


def _xml(n: int) -> str:
    word = _WORDS[n]
    nodes = [
        _node(
            "android.widget.TextView",
            text=word.title(),
            rid=f"x:id/{word}_header",
            b="[40,120][1040,210]",
        )
    ]
    for k in range(3):
        nodes.append(
            _node(
                "android.widget.Button",
                text=f"{word} {k}",
                rid=f"x:id/{word}_act{k}",
                clk=True,
                b=f"[40,{300 + k * 140}][1040,{400 + k * 140}]",
            )
        )
    return _hier(*nodes)


def _seed(tmp_path, count: int):
    """`count` distinct screens, each named something the auditor calls poor."""
    store = _store(tmp_path)
    names = [
        store.record_screen(
            package=P, elements=_elements(_xml(i)), name_hint=f"screen_{i + 2}"
        ).name
        for i in range(count)
    ]
    assert len(set(names)) == count, f"expected {count} distinct screens, got {set(names)}"
    # `name_hint` stamps `explicit`, and a deliberate name is no longer a research question. Real
    # poor names come from a weak source — 400 of the 496 measured on one real map were `legacy`.
    app = store.load(P)
    for name in names:
        app.screens[name].name_source = "legacy"
    store.save(app)
    return store, names


def _naming_tasks(reconciliation, screen_names, store):
    """One open poor_name task per screen, keyed by the screen's current name."""
    app = store.load(P)
    id_of = {name: app.screens[name].id for name in screen_names}
    tasks = [t for t in reconciliation.plan(P) if t.issue_type == "poor_name"]
    out = {}
    for name in screen_names:
        match = next(t for t in tasks if id_of[name] in t.affected_ids)
        out[name] = match.id
    return out


def _corrections(store):
    directory = ReconciliationStore(store).corrections_dir(P)
    if not directory.is_dir():
        return {"before": [], "after": [], "event": []}
    return {kind: sorted(directory.glob(f"*.{kind}.json")) for kind in ("before", "after", "event")}


def test_one_batch_writes_one_snapshot_trio(tmp_path) -> None:
    store, names = _seed(tmp_path, 12)
    reconciliation = ReconciliationStore(store)
    tasks = _naming_tasks(reconciliation, names, store)

    result = reconciliation.answer_many(
        P, [(tasks[n], f"destination {i}") for i, n in enumerate(names)], agent="needle"
    )

    files = _corrections(store)
    assert len(files["before"]) == 1, "twelve answers, one before-snapshot"
    assert len(files["after"]) == 1
    assert len(files["event"]) == 1
    assert result["renamed"] == 12
    assert result["skipped"] == 0
    assert result["rollback_id"]


def test_every_answered_task_is_applied(tmp_path) -> None:
    store, names = _seed(tmp_path, 6)
    reconciliation = ReconciliationStore(store)
    tasks = _naming_tasks(reconciliation, names, store)

    reconciliation.answer_many(P, [(tasks[n], f"place {i}") for i, n in enumerate(names)])

    app = store.load(P)
    assert {f"place_{i}" for i in range(6)} <= set(app.screens)
    answered = {t["id"] for t in app.research_tasks if t["status"] == "applied"}
    assert set(tasks.values()) <= answered
    for name in (f"place_{i}" for i in range(6)):
        assert app.screens[name].name_source == "explicit"


def test_one_rollback_id_undoes_the_whole_batch(tmp_path) -> None:
    store, names = _seed(tmp_path, 8)
    reconciliation = ReconciliationStore(store)
    tasks = _naming_tasks(reconciliation, names, store)
    before = set(store.load(P).screens)

    result = reconciliation.answer_many(P, [(tasks[n], f"spot {i}") for i, n in enumerate(names)])
    reconciliation.rollback(P, str(result["rollback_id"]))

    app = store.load(P)
    assert set(app.screens) == before, "one undo returns every renamed screen"
    assert not [t for t in app.research_tasks if t["status"] == "applied"]


def _timeless(value):
    drop = {"first_seen", "last_seen", "last_verified", "created_at", "applied_at", "generated_at"}
    if isinstance(value, dict):
        return {k: _timeless(v) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


def test_a_batch_of_one_is_the_same_as_a_single_answer(tmp_path) -> None:
    """The bulk path must not be a second, subtly different way to answer one question."""
    single_store, single_names = _seed(tmp_path / "single", 1)
    bulk_store, bulk_names = _seed(tmp_path / "bulk", 1)
    single = ReconciliationStore(single_store)
    bulk = ReconciliationStore(bulk_store)
    single_task = _naming_tasks(single, single_names, single_store)[single_names[0]]
    bulk_task = _naming_tasks(bulk, bulk_names, bulk_store)[bulk_names[0]]

    single.answer(P, single_task, "settings")
    bulk.answer_many(P, [(bulk_task, "settings")])

    assert _timeless(single_store.load(P).model_dump(mode="json")) == _timeless(
        bulk_store.load(P).model_dump(mode="json")
    )


def test_nothing_is_written_when_no_answer_survives(tmp_path) -> None:
    store, names = _seed(tmp_path, 3)
    reconciliation = ReconciliationStore(store)
    _naming_tasks(reconciliation, names, store)
    before = store.load(P).model_dump(mode="json")

    result = reconciliation.answer_many(P, [("research_nope", "x"), ("research_also_nope", "y")])

    assert result["status"] == "rejected"
    assert result["rollback_id"] is None
    assert result["skipped"] == 2
    assert _corrections(store)["event"] == [], "no snapshot pair for zero changes"
    assert store.load(P).model_dump(mode="json") == before


def test_the_map_is_restored_when_the_save_fails(tmp_path) -> None:
    store, names = _seed(tmp_path, 4)
    reconciliation = ReconciliationStore(store)
    tasks = _naming_tasks(reconciliation, names, store)
    before = set(store.load(P).screens)
    real_save = store.save
    calls = {"n": 0}

    def flaky(app):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk went away")
        return real_save(app)

    store.save = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        reconciliation.answer_many(P, [(tasks[n], f"zone {i}") for i, n in enumerate(names)])
    store.save = real_save  # type: ignore[method-assign]

    app = store.load(P)
    assert set(app.screens) == before, "a failed save leaves the map exactly as it was"
    assert not [t for t in app.research_tasks if t["status"] == "applied"]
    assert _corrections(store)["event"] == [], "and no event points at the orphaned snapshots"
