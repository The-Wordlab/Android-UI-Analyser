"""Whole-call deadlines use the selected runtime, including a slow final observation."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from android_ui_analyser import engine_waits, read_budget
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    JobCancelledError,
    UnsupportedPlatformCapabilityError,
    UsageError,
)
from android_ui_analyser.platforms.base import NormalizedTree
from android_ui_analyser.providers.base import TextBox
from android_ui_analyser.schema import Element
from conftest import FakeDevice, StubOcr, make_config
from test_platform_runtime import _NeutralAdapter, _NeutralRuntime


class Runtime(_NeutralRuntime):
    def __init__(self, *, probe_delay=0.0, foreground_delay=0.0, tree_delay=0.0):
        self.probe_delay = probe_delay
        self.foreground_delay = foreground_delay
        self.tree_delay = tree_delay
        self.reads = []

    def read_deadline(self, budget):
        return read_budget.activate(budget)

    def _read(self, name, duration):
        self.reads.append(name)
        budget = read_budget.current()
        assert budget is not None
        threading.Event().wait(min(duration, budget.remaining()))
        budget.check()

    def find_text(self, text, **kwargs):
        self._read("probe", self.probe_delay)
        return (10, 10, 40, 40) if text == "Ready" else None

    def current_app(self):
        self._read("foreground", self.foreground_delay)
        return super().current_app()

    def dump_hierarchy(self, compressed=False):
        self._read("tree", self.tree_delay)
        return "native-tree"


class Adapter(_NeutralAdapter):
    capabilities = frozenset({"ui.tree", "ui.input", "ui.read_deadline"})

    def connect(self, target_id=None):
        raise AssertionError("bounded wait must not lazily connect")

    def normalize_tree(self, raw_tree, screen_size, **kwargs):
        return NormalizedTree(
            [Element(id=0, text="Ready", type="Text", bounds=(10, 10, 40, 40), center=(25, 25))],
            app_id="example.app",
        )


def engine(runtime=None, *, adapter=Adapter):
    config = make_config(memory={"enabled": False}, lease={"enabled": False})
    return Engine(config, device=runtime, platform=adapter(config))


@pytest.mark.parametrize("method", ["wait", "await_predicate"])
def test_slow_probe_is_bounded_without_background_device_work(method):
    runtime = Runtime(probe_delay=2)
    eng = engine(runtime)
    started = time.monotonic()
    result = (
        eng.wait(for_="Ready", timeout_ms=100, observe=True)
        if method == "wait"
        else eng.await_predicate("text:Ready", timeout_ms=100, observe=True)
    )
    elapsed = time.monotonic() - started
    assert 0.08 <= elapsed < 0.5, elapsed
    assert not result.ok
    assert result.observation is None
    assert result.elapsed_ms is not None and result.elapsed_ms < 500
    assert runtime.reads.count("probe") == 1
    assert "tree" not in runtime.reads
    assert read_budget.current() is None


def test_foreground_snapshot_is_inside_the_same_budget():
    runtime = Runtime(foreground_delay=2)
    started = time.monotonic()
    result = engine(runtime).await_predicate("text:Ready", timeout_ms=100, observe=True)
    assert time.monotonic() - started < 0.5
    assert not result.ok and result.await_outcome == "timeout"
    assert runtime.reads == ["foreground"]


@pytest.mark.parametrize("absent", [False, True])
def test_a_late_predicate_cannot_pass_even_if_adapter_returns_a_match(monkeypatch, absent):
    now = [10.0]
    monkeypatch.setattr(engine_waits, "time", SimpleNamespace(monotonic=lambda: now[0]))
    runtime = Runtime()

    def late(*args, **kwargs):
        now[0] += 0.3
        return None if absent else (10, 10, 40, 40)

    monkeypatch.setattr(runtime, "find_text", late)
    result = engine(runtime).wait(for_="Ready", absent=absent, timeout_ms=100, by="rid")
    assert not result.ok
    assert result.observation is None


def test_timely_predicate_is_distinct_from_final_observation_expiry():
    runtime = Runtime(tree_delay=2)
    started = time.monotonic()
    result = engine(runtime).wait(for_="Ready", timeout_ms=100, observe=True)
    assert time.monotonic() - started < 0.5
    assert result.ok  # the predicate held before readback started
    assert result.observation is None and result.observation_present is False
    assert "final observation" in result.note.lower()
    assert runtime.reads.count("tree") == 1


def test_timely_final_observation_keeps_ids_on_the_shared_analyze_path():
    runtime = Runtime()
    result = engine(runtime).wait(for_="Ready", timeout_ms=500, observe=True, with_image=False)
    assert result.ok and result.observation is not None
    assert result.observation.elements[0].text == "Ready"
    assert result.observation.elements[0].published_id.startswith("el:")
    assert runtime.reads.count("tree") == 1


def test_cold_runtime_refuses_without_connecting():
    with pytest.raises(UsageError) as error:
        engine().wait(for_="Ready", timeout_ms=100)
    assert error.value.code == "wait_runtime_not_ready"


def test_zero_performs_one_probe_with_an_explicit_bounded_read_budget():
    runtime = Runtime()
    eng = engine(runtime)
    eng.config.perf.max_wait_ms = 200
    result = eng.wait(for_="Missing", timeout_ms=0, by="rid")
    assert not result.ok
    assert runtime.reads == ["probe"]
    assert result.wait_budget_ms == 200
    assert "one probe without polling" in result.note


def test_missing_predicate_still_returns_one_fresh_readback_inside_the_budget():
    runtime = Runtime()
    started = time.monotonic()
    result = engine(runtime).wait(for_="Missing", timeout_ms=200, observe=True, with_image=False)
    assert time.monotonic() - started < 0.5
    assert not result.ok and result.observation is not None
    assert runtime.reads.count("tree") == 1
    assert result.observation.elements[0].text == "Ready"


def test_nested_wait_reports_the_parent_limited_effective_budget():
    runtime = Runtime()
    parent = read_budget.ReadBudget(time.monotonic() + 0.15, time.monotonic)
    with runtime.read_deadline(parent):
        result = engine(runtime).wait(for_="Missing", timeout_ms=1000, by="rid")
        assert read_budget.current() is parent
    assert not result.ok
    assert 0 < result.wait_budget_ms <= 150
    assert read_budget.current() is None


def test_expiry_journal_initialization_cannot_start_unbounded_device_metadata(monkeypatch):
    runtime = Runtime(probe_delay=2)
    config = make_config(lease={"enabled": False})
    eng = Engine(config, device=runtime, platform=Adapter(config))
    seen = []

    def token():
        budget = read_budget.current()
        seen.append(budget)
        assert budget is not None, "expiry finalization escaped the read deadline"
        budget.check()
        raise AssertionError("expired journal metadata must not perform device I/O")

    monkeypatch.setattr(runtime, "instance_token", token)
    assert eng._mem is None
    started = time.monotonic()
    result = eng.wait(for_="Missing", timeout_ms=100, by="rid")
    assert time.monotonic() - started < 0.5
    assert not result.ok and seen and all(item is not None for item in seen)
    assert eng._mem is not None
    assert eng._mem.load_session(runtime.target_id).calls[-1].outcome == "timeout"


def test_adapter_without_deadline_contract_refuses_before_any_reads():
    class Unsupported(Adapter):
        capabilities = Adapter.capabilities - {"ui.read_deadline"}

    runtime = Runtime()
    with pytest.raises(UnsupportedPlatformCapabilityError):
        engine(runtime, adapter=Unsupported).wait(for_="Ready", timeout_ms=100)
    assert runtime.reads == []


def test_job_cancellation_remains_cancellation_and_restores_scope():
    eng = engine(Runtime())
    event = threading.Event()
    event.set()
    eng._job_cancel_event = event
    with pytest.raises(JobCancelledError):
        eng.wait(for_="Ready", timeout_ms=100)
    assert read_budget.current() is None


def test_cancellation_during_final_observation_is_not_reported_as_success(monkeypatch):
    runtime = Runtime()
    eng = engine(runtime)
    event = threading.Event()
    eng._job_cancel_event = event

    def cancelled_tree(*args, **kwargs):
        event.set()
        read_budget.current().check()

    monkeypatch.setattr(runtime, "dump_hierarchy", cancelled_tree)
    with pytest.raises(JobCancelledError):
        eng.wait(for_="Ready", timeout_ms=100, observe=True, with_image=False)
    assert read_budget.current() is None


def test_warm_ocr_only_positive_is_checked_before_deadline():
    config = make_config(ocr={"enabled": True, "augment_hierarchy": True, "chain": ["stub_ocr"]})
    eng = Engine(config, device=FakeDevice())
    provider = StubOcr(
        result=[TextBox(text="Canvas ready", bounds=(10, 10, 200, 50), confidence=0.99)]
    )
    eng.factory._instances[("ocr", "stub_ocr")] = provider
    result = eng.await_predicate("text:Canvas ready", timeout_ms=500, observe=False)
    assert result.ok and result.await_outcome == "satisfied"
    assert provider.calls >= 1
    assert result.elapsed_ms < 500


def test_cold_ocr_cannot_prove_a_canvas_label_absent():
    config = make_config(ocr={"enabled": True, "augment_hierarchy": True, "chain": ["stub_ocr"]})
    eng = Engine(config, device=FakeDevice())
    result = eng.await_predicate("!text:Canvas loading", timeout_ms=50, observe=False)
    assert not result.ok and result.await_outcome == "timeout"
    assert result.await_terms[0]["satisfied"] is False
    assert result.await_terms[0]["evidence"] == "unconfirmed"


@pytest.mark.parametrize("warm", [False, True])
def test_action_destination_does_not_override_unconfirmed_or_visible_canvas_text(monkeypatch, warm):
    from test_action_arrival_mismatch import _observation, _seed_action_baseline

    config = make_config(ocr={"enabled": True, "augment_hierarchy": True, "chain": ["stub_ocr"]})
    runtime = FakeDevice(resource_index={"catalogItemCard": (20, 220, 900, 360)})
    eng = Engine(config, device=runtime)
    _seed_action_baseline(eng)
    if warm:
        eng.factory._instances[("ocr", "stub_ocr")] = StubOcr(
            result=[TextBox(text="Canvas loading", bounds=(10, 10, 200, 50), confidence=0.99)]
        )
    monkeypatch.setattr(eng, "_sample_action_destination", _observation)
    result = eng.await_predicate(
        "rid:catalogItemCard,!text:Canvas loading", timeout_ms=100, poll_ms=10, adopt_action=True
    )
    assert not result.ok
    assert result.await_outcome == "timeout"
    assert not all(row["satisfied"] for row in result.await_terms)
