"""Atomic publication guarantees for new and replacement flow files."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import android_ui_analyser.atomic as atomic_mod
from android_ui_analyser.atomic import atomic_create_text
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import Flow, FlowStore, parse_flow_yaml
from android_ui_analyser.memory import RouteStep
from conftest import make_config


def _flow(label: str) -> Flow:
    return Flow(name="race", steps=[RouteStep(kind="tap", label=label)])


def test_atomic_create_never_exposes_a_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "flow.yaml"
    real_link = atomic_mod.os.link
    observed: list[tuple[bool, str]] = []

    def publish(source: Path, destination: Path) -> None:
        observed.append((destination.exists(), source.read_text(encoding="utf-8")))
        real_link(source, destination)

    monkeypatch.setattr(atomic_mod.os, "link", publish)
    atomic_create_text(target, "complete: true\n")

    assert observed == [(False, "complete: true\n")]
    assert target.read_text(encoding="utf-8") == "complete: true\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupted_first_save_leaves_no_destination_or_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    real_write_text = Path.write_text

    def interrupt(
        path: Path,
        text: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        real_write_text(path, text[:8], encoding=encoding, errors=errors, newline=newline)
        raise KeyboardInterrupt("fictional interruption")

    monkeypatch.setattr(Path, "write_text", interrupt)
    with pytest.raises(KeyboardInterrupt, match="fictional interruption"):
        store.save(_flow("Complete"))

    assert not store.path("race").exists()
    assert not list(store.flows_dir().glob("*.tmp"))


def test_concurrent_first_saves_publish_one_complete_flow(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def save(label: str) -> None:
        barrier.wait()
        try:
            store.save(_flow(label))
            outcomes.append("saved")
        except UsageError as exc:
            assert "already exists" in str(exc)
            outcomes.append("exists")

    threads = [threading.Thread(target=save, args=(f"Writer {n}",)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("saved") == 1
    assert outcomes.count("exists") == 7
    assert parse_flow_yaml(store.path("race").read_text(encoding="utf-8")).steps[0].label in {
        f"Writer {n}" for n in range(8)
    }
    assert not list(store.flows_dir().glob("*.tmp"))


def test_non_force_save_never_overwrites_an_existing_flow(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.save(_flow("Original"))
    original = store.path("race").read_bytes()

    with pytest.raises(UsageError, match="already exists"):
        store.save(_flow("Replacement"))

    assert store.path("race").read_bytes() == original
    assert not list(store.flows_dir().glob("*.tmp"))


@pytest.mark.parametrize("arrival", ["!text:Loading", "unknown:state"])
def test_invalid_arrival_never_creates_a_flow_file(tmp_path: Path, arrival: str) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    flow = Flow(
        name="invalid-arrival",
        arrival=arrival,
        steps=[RouteStep(kind="key", arg="back")],
    )

    with pytest.raises(UsageError):
        store.save(flow)

    assert not store.path(flow.name).exists()
    assert not list(store.flows_dir().glob("*.tmp"))


def test_invalid_default_never_creates_a_flow_file(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    flow = Flow(
        name="invalid-default",
        params={"DIRECTION": "diagonal"},
        steps=[RouteStep(kind="swipe", arg="${DIRECTION}")],
    )

    with pytest.raises(UsageError):
        store.save(flow)

    assert not store.path(flow.name).exists()
    assert not list(store.flows_dir().glob("*.tmp"))


def test_required_empty_finite_parameter_is_saved_as_metadata(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    flow = Flow(
        name="required-direction",
        params={"DIRECTION": ""},
        steps=[RouteStep(kind="swipe", arg="${DIRECTION}")],
    )

    path = store.save(flow)
    loaded = parse_flow_yaml(path.read_text(encoding="utf-8"))

    assert loaded.params == {"DIRECTION": ""}
    assert loaded.steps[0].arg == "${DIRECTION}"


def test_invalid_arrival_force_save_preserves_existing_flow(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.save(_flow("Original"))
    original = store.path("race").read_bytes()
    invalid_replacement = Flow(
        name="race",
        arrival="!text:Loading",
        steps=[RouteStep(kind="tap", label="Replacement")],
    )

    with pytest.raises(UsageError, match="positive arrival"):
        store.save(invalid_replacement, force=True)

    assert store.path("race").read_bytes() == original
    assert not list(store.flows_dir().glob("*.tmp"))
