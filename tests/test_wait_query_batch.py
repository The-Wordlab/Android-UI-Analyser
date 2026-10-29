"""Compound waits use an optional adapter batch without adding native calls to core."""

from __future__ import annotations

import time
from contextlib import closing, contextmanager
from types import SimpleNamespace

import pytest

from android_ui_analyser import engine_waits, read_budget
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms.android import AndroidPlatform
from conftest import FakeDevice, make_config
from test_platform_runtime import _NeutralAdapter, _NeutralRuntime


class BatchRuntime(_NeutralRuntime):
    def __init__(self):
        self.batches = []
        self.failure = None

    def read_deadline(self, budget):
        return read_budget.activate(budget)

    def find_text(self, *args, **kwargs):
        pytest.fail("compound query fell back to individual native reads")

    def find_texts(self, queries, **kwargs):
        self.batches.append((list(queries), kwargs))
        if self.failure:
            raise self.failure
        return [(1, 2, 3, 4) if text == "Ready" else None for text, _ in queries]


class BatchAdapter(_NeutralAdapter):
    capabilities = frozenset({"ui.tree", "ui.input", "ui.read_deadline", "ui.query_batch"})


def make_batch_engine(runtime):
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False}, ocr={"enabled": False})
    return Engine(cfg, device=runtime, platform=BatchAdapter(cfg))


def test_compound_wait_uses_the_selected_adapter_without_android(monkeypatch):
    monkeypatch.setattr(AndroidPlatform, "connect", lambda *a: pytest.fail("Android fallback"))
    runtime = BatchRuntime()
    with closing(make_batch_engine(runtime)) as engine:
        result = engine.await_predicate("text:Ready,!rid:spinner", observe=False)
    assert result.await_outcome == "satisfied"
    assert runtime.batches[0][0] == [("Ready", "text"), ("spinner", "rid")]
    assert len(runtime.batches) == 1


def test_failed_batch_cannot_satisfy_an_absence_predicate_or_retry_singly():
    runtime = BatchRuntime()
    runtime.failure = DeviceError("temporary read failure")
    with closing(make_batch_engine(runtime)) as engine:
        result = engine.await_predicate("!rid:spinner,!rid:dialog", timeout_ms=30, observe=False)
    assert result.await_outcome == "timeout"
    assert all(term["present"] is None and not term["satisfied"] for term in result.await_terms)


@pytest.fixture
def wait_clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        engine_waits, "time",
        SimpleNamespace(monotonic=lambda: now[0], time=time.time),
    )
    return now


@pytest.mark.parametrize("expiry_boundary", ["scope_entry", "foreground_read"])
def test_deadline_before_first_query_reports_unknown_terms(
    monkeypatch, wait_clock, expiry_boundary,
):
    runtime = BatchRuntime()

    @contextmanager
    def deadline_scope(budget):
        with read_budget.activate(budget):
            if expiry_boundary == "scope_entry":
                wait_clock[0] += 0.031
            yield

    def current_app():
        wait_clock[0] += 0.031
        return {"package": "com.example.fixture", "activity": "Fixture"}

    def bounded_query(*args, **kwargs):
        read_budget.current().check()
        pytest.fail("the expired read budget must prevent the first query")

    monkeypatch.setattr(runtime, "read_deadline", deadline_scope)
    monkeypatch.setattr(runtime, "current_app", current_app)
    monkeypatch.setattr(runtime, "find_texts", bounded_query)
    with closing(make_batch_engine(runtime)) as engine:
        result = engine.await_predicate("text:Ready,!rid:spinner", timeout_ms=30, observe=False)

    assert result.await_outcome == "timeout"
    assert len(result.await_terms) == 2
    assert all(row["present"] is None and not row["satisfied"] for row in result.await_terms)
    assert all(row["evidence"] == "unconfirmed" for row in result.await_terms)
    assert not runtime.batches


@pytest.mark.parametrize("prior_poll", [False, True])
def test_failed_batch_publishes_unknowns_before_its_deadline_check(
    monkeypatch, wait_clock, prior_poll,
):
    runtime = BatchRuntime()

    def query(queries, **kwargs):
        runtime.batches.append((list(queries), kwargs))
        if prior_poll and len(runtime.batches) == 1:
            return [None, (1, 2, 3, 4)]
        wait_clock[0] += 0.031
        raise DeviceError("read failed after consuming the remaining deadline")

    def next_poll(*args):
        wait_clock[0] += 0.005

    monkeypatch.setattr(runtime, "find_texts", query)
    with closing(make_batch_engine(runtime)) as engine:
        monkeypatch.setattr(engine, "_sleep_between_polls", next_poll)
        result = engine.await_predicate("!rid:spinner,!rid:dialog", timeout_ms=30, observe=False)

    assert result.await_outcome == "timeout"
    assert len(result.await_terms) == 2
    assert all(row["present"] is None and not row["satisfied"] for row in result.await_terms)
    assert all(row["reason"] == "ui_read_failed" for row in result.await_terms)
    assert len(runtime.batches) == (2 if prior_poll else 1)


def test_permanent_batch_error_is_preserved_without_individual_retry():
    runtime = BatchRuntime()
    runtime.failure = DeviceError("tool unavailable", code="ios_tool_missing")
    with closing(make_batch_engine(runtime)) as engine, pytest.raises(DeviceError) as caught:
        engine.await_predicate("text:Ready,!rid:spinner", observe=False)
    assert caught.value.code == "ios_tool_missing"
    assert len(runtime.batches) == 1


def test_android_batch_preserves_its_existing_selector_results(tmp_path):
    cfg = make_config(cache={"dir": str(tmp_path)}, lease={"enabled": False})
    runtime = FakeDevice(text_index={"Ready": (1, 2, 3, 4)}, resource_index={"spinner": (5, 6, 7, 8)})
    batch = AndroidPlatform(cfg).runtime_capability("ui.query_batch", runtime)
    queries = [("Ready", "text"), ("spinner", "rid"), ("Missing", "text")]
    assert batch.find_texts(queries) == [runtime.find_text(text, by=by) for text, by in queries]


def test_adapter_without_batch_keeps_its_existing_query_path():
    class LegacyRuntime(BatchRuntime):
        singles = 0

        def find_text(self, text, **kwargs):
            self.singles += 1
            return (1, 2, 3, 4) if text == "Ready" else None

    runtime = LegacyRuntime()
    with closing(make_batch_engine(runtime)) as engine:
        engine.platform.capabilities = BatchAdapter.capabilities - {"ui.query_batch"}
        result = engine.await_predicate("text:Ready,!rid:spinner", observe=False)
    assert result.await_outcome == "satisfied"
    assert runtime.singles == 2 and not runtime.batches


def test_malformed_batch_is_an_explicit_error_not_false_absence(monkeypatch):
    runtime = BatchRuntime()
    monkeypatch.setattr(runtime, "find_texts", lambda *a, **kw: [])
    with closing(make_batch_engine(runtime)) as engine, pytest.raises(DeviceError) as caught:
        engine.await_predicate("text:Ready,!rid:spinner", observe=False)
    assert caught.value.code == "invalid_platform_result"
