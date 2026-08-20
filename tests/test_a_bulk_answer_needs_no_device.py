"""Draining a naming backlog must not require an emulator, and must be one transaction.

`--answers` is deliberately device-bound: it answers the one question about the screen the caller
is standing on, and `engine.current_package()` is the only thing that knows which app that is. That
is right for one answer and wrong for a backlog, where every row is already tied to an explicit
screen and package.

So `reconcile answers` takes an explicit `--app`, reads a researched document, and commits every
row in a single transaction with a single rollback id. Several `--answers` pairs on one command now
take the same path, because N answers used to mean N full copies of the map on disk.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from android_ui_analyser import cli as cli_mod
from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMemoryStore
from android_ui_analyser.reconcile import ReconciliationStore
from conftest import make_config
from test_hundreds_of_answers_are_one_transaction import P, _naming_tasks, _xml
from test_memory import _elements

runner = CliRunner()


def _seed(count: int):
    """Seed the *default* isolated store, which is the one the CLI will open."""
    store = AppMemoryStore(make_config().memory)
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


def _no_device(monkeypatch) -> None:
    """Any attempt to reach a device is a failure, not a slow path."""

    def boom(*_args, **_kwargs):
        raise AssertionError("a bulk answer must not touch a device")

    monkeypatch.setattr(Engine, "current_package", boom)
    monkeypatch.setattr(cli_mod, "connect", boom, raising=False)


def test_a_researched_backlog_is_drained_without_a_device(tmp_path, monkeypatch) -> None:
    store, names = _seed(5)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    doc = tmp_path / "answers.json"
    doc.write_text(
        json.dumps({"answers": {tasks[n]: f"place {i}" for i, n in enumerate(names)}}),
        encoding="utf-8",
    )
    _no_device(monkeypatch)

    result = runner.invoke(cli_app, ["reconcile", "answers", str(doc), "--app", P])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["renamed"] == 5
    assert payload["rollback_id"]
    app = store.load(P)
    assert {f"place_{i}" for i in range(5)} <= set(app.screens)


def test_the_whole_drain_is_one_snapshot_pair(tmp_path, monkeypatch) -> None:
    store, names = _seed(6)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    doc = tmp_path / "answers.json"
    doc.write_text(
        json.dumps({"answers": {tasks[n]: f"zone {i}" for i, n in enumerate(names)}}),
        encoding="utf-8",
    )
    _no_device(monkeypatch)

    runner.invoke(cli_app, ["reconcile", "answers", str(doc), "--app", P])

    directory = rec.corrections_dir(P)
    assert len(list(directory.glob("*.before.json"))) == 1
    assert len(list(directory.glob("*.event.json"))) == 1


def test_a_list_form_lets_the_caller_set_priority(tmp_path, monkeypatch) -> None:
    """Two answers that slug alike: the first in the list wins, and the second is reported."""
    store, names = _seed(3)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    doc = tmp_path / "answers.json"
    doc.write_text(
        json.dumps(
            {
                "answers": [
                    {"task_id": tasks[names[0]], "value": "Dev Tools"},
                    {"task_id": tasks[names[1]], "value": "dev-tools"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _no_device(monkeypatch)

    result = runner.invoke(cli_app, ["reconcile", "answers", str(doc), "--app", P])

    payload = json.loads(result.output)
    assert payload["renamed"] == 1
    assert payload["skipped"] == 1
    assert payload["skipped_answers"][0]["code"] == "name_taken_in_batch"
    assert "dev_tools" in store.load(P).screens


def test_a_malformed_document_fails_before_touching_the_map(tmp_path, monkeypatch) -> None:
    store, names = _seed(2)
    ReconciliationStore(store)
    before = store.load(P).model_dump(mode="json")
    doc = tmp_path / "answers.json"
    doc.write_text(json.dumps({"answers": [{"task_id": "research_x"}]}), encoding="utf-8")
    _no_device(monkeypatch)

    result = runner.invoke(cli_app, ["reconcile", "answers", str(doc), "--app", P])

    assert result.exit_code != 0
    assert store.load(P).model_dump(mode="json") == before


def test_a_rollback_id_from_the_drain_undoes_all_of_it(tmp_path, monkeypatch) -> None:
    store, names = _seed(4)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    before = set(store.load(P).screens)
    doc = tmp_path / "answers.json"
    doc.write_text(
        json.dumps({"answers": {tasks[n]: f"spot {i}" for i, n in enumerate(names)}}),
        encoding="utf-8",
    )
    _no_device(monkeypatch)

    payload = json.loads(
        runner.invoke(cli_app, ["reconcile", "answers", str(doc), "--app", P]).output
    )
    rec.rollback(P, str(payload["rollback_id"]))

    assert set(store.load(P).screens) == before


class _StubEngine:
    """Only what `_apply_answers` reads: which app, and where memory lives."""

    def __init__(self, config) -> None:
        self.config = config

    def current_package(self) -> str:
        return P


def test_several_inline_answers_on_one_command_are_one_transaction() -> None:
    store, names = _seed(4)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    engine = _StubEngine(make_config())

    cli_mod._apply_answers(
        engine,  # type: ignore[arg-type]
        tuple(f'{tasks[n]}="area {i}"' for i, n in enumerate(names)),
    )

    directory = rec.corrections_dir(P)
    assert len(list(directory.glob("*.event.json"))) == 1, "four answers, one correction"
    assert {f"area_{i}" for i in range(4)} <= set(store.load(P).screens)


def test_one_unusable_inline_answer_still_fails_loudly() -> None:
    """`--answers` must keep stopping the command rather than quietly renaming the wrong screen."""
    store, names = _seed(3)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    engine = _StubEngine(make_config())
    before = set(store.load(P).screens)

    try:
        cli_mod._apply_answers(
            engine,  # type: ignore[arg-type]
            (f'{tasks[names[0]]}="fine"', 'research_nope="bad"'),
        )
    except Exception as err:
        assert "could not use" in str(err), err
    else:  # pragma: no cover - the loud contract must not be softened
        raise AssertionError("an unusable answer must stop the command")

    assert set(store.load(P).screens) == before, "and nothing is committed"


def test_a_dry_run_reports_the_whole_outcome_and_writes_nothing(tmp_path, monkeypatch) -> None:
    """A bulk rename of a real map deserves a look before it happens."""
    store, names = _seed(4)
    rec = ReconciliationStore(store)
    tasks = _naming_tasks(rec, names, store)
    before = store.load(P).model_dump(mode="json")
    doc = tmp_path / "answers.json"
    doc.write_text(
        json.dumps(
            {
                "answers": [
                    {"task_id": tasks[names[0]], "value": "Dev Tools"},
                    {"task_id": tasks[names[1]], "value": "dev-tools"},
                    {"task_id": tasks[names[2]], "value": "Ideas"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _no_device(monkeypatch)

    payload = json.loads(
        runner.invoke(cli_app, ["reconcile", "answers", str(doc), "--app", P, "--dry-run"]).output
    )

    assert payload["status"] == "dry_run"
    assert payload["would_apply"] == 2
    assert payload["skipped_answers"][0]["code"] == "name_taken_in_batch"
    assert payload["rollback_id"] is None
    assert store.load(P).model_dump(mode="json") == before, "a dry run writes nothing"
    assert not list(rec.corrections_dir(P).glob("*.json"))
