"""Whole-call deadlines use the selected runtime, including a slow final observation."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from android_ui_analyser import engine_waits, read_budget
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    DeviceError,
    JobCancelledError,
    UnsupportedPlatformCapabilityError,
    UsageError,
)
from android_ui_analyser.platforms.base import NormalizedTree
from android_ui_analyser.providers.base import TextBox
from android_ui_analyser.schema import Element
from conftest import FakeDevice, StubOcr, make_config
from test_platform_runtime import _NeutralAdapter, _NeutralRuntime

REAL_WAIT_MS = 500
MAX_ELAPSED_S = 1.5


class Runtime(_NeutralRuntime):
    def __init__(self, *, probe_delay=0.0, foreground_delay=0.0, tree_delay=0.0):
        self.probe_delay = probe_delay
        self.foreground_delay = foreground_delay
        self.tree_delay = tree_delay
        self.reads = []
        self.read_threads = []
        self.read_waits = []
        self.active_reads = 0

    def read_deadline(self, budget):
        return read_budget.activate(budget)

    def _wait(self, seconds):
        threading.Event().wait(seconds)

    def _read(self, name, duration):
        self.reads.append(name)
        self.read_threads.append(threading.get_ident())
        budget = read_budget.current()
        assert budget is not None
        self.active_reads += 1
        try:
            wait_s = min(duration, budget.remaining())
            self.read_waits.append((name, wait_s))
            self._wait(wait_s)
            budget.check()
        finally:
            self.active_reads -= 1

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
        eng.wait(for_="Ready", timeout_ms=REAL_WAIT_MS, observe=True)
        if method == "wait"
        else eng.await_predicate("text:Ready", timeout_ms=REAL_WAIT_MS, observe=True)
    )
    elapsed = time.monotonic() - started
    assert elapsed < MAX_ELAPSED_S, elapsed
    assert not result.ok
    assert result.observation is None
    assert result.elapsed_ms is not None and result.elapsed_ms < MAX_ELAPSED_S * 1000
    assert runtime.reads.count("probe") == 1
    assert 0 < dict(runtime.read_waits)["probe"] <= REAL_WAIT_MS / 1000
    assert "tree" not in runtime.reads
    assert set(runtime.read_threads) == {threading.get_ident()}
    assert runtime.active_reads == 0
    assert read_budget.current() is None


def test_foreground_snapshot_is_inside_the_same_budget():
    runtime = Runtime(foreground_delay=2)
    eng = engine(runtime)
    started = time.monotonic()
    result = eng.await_predicate("text:Ready", timeout_ms=REAL_WAIT_MS, observe=True)
    assert time.monotonic() - started < MAX_ELAPSED_S
    assert not result.ok and result.await_outcome == "timeout"
    assert runtime.reads == ["foreground"]
    assert 0 < dict(runtime.read_waits)["foreground"] <= REAL_WAIT_MS / 1000
    assert runtime.read_threads == [threading.get_ident()]
    assert runtime.active_reads == 0


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
    eng = engine(runtime)
    started = time.monotonic()
    result = eng.wait(for_="Ready", timeout_ms=REAL_WAIT_MS, observe=True)
    assert time.monotonic() - started < MAX_ELAPSED_S
    assert result.ok  # the predicate held before readback started
    assert result.observation is None and result.observation_present is False
    assert "final observation" in result.note.lower()
    assert runtime.reads.count("tree") == 1
    assert 0 < dict(runtime.read_waits)["tree"] <= REAL_WAIT_MS / 1000
    assert set(runtime.read_threads) == {threading.get_ident()}
    assert runtime.active_reads == 0


def test_timely_final_observation_keeps_ids_on_the_shared_analyze_path():
    runtime = Runtime()
    result = engine(runtime).wait(for_="Ready", timeout_ms=500, observe=True, with_image=False)
    assert result.ok and result.observation is not None
    assert result.observation.elements[0].text == "Ready"
    assert result.observation.elements[0].published_id.startswith("el:")
    assert runtime.reads.count("tree") == 1


def test_folded_action_until_uses_neutral_runtime_and_never_repeats_action(monkeypatch):
    from android_ui_analyser.mcp_server import _dump, _fold_action_until
    from android_ui_analyser.platforms.android import AndroidPlatform

    monkeypatch.setattr(
        AndroidPlatform, "connect", lambda *args: pytest.fail("must use the selected adapter")
    )

    class ActionRuntime(Runtime):
        clicks = 0

        def _read(self, name, duration):
            if read_budget.current() is not None:
                super()._read(name, duration)

        def click(self, x, y):
            self.clicks += 1

        def find_text(self, text, **kwargs):
            assert self.clicks == 1
            return super().find_text(text, **kwargs)

    runtime = ActionRuntime()
    config = make_config(
        memory={"enabled": False}, lease={"enabled": False}, output={"with_image": False}
    )
    eng = Engine(config, device=runtime, platform=Adapter(config))
    try:
        observation = eng.analyze(source="hierarchy", with_image=False)
        action = eng.tap(observation.elements[0].published_id, observe=False)
        payload = _fold_action_until(
            eng, "tap_and_analyze", {"until": "text:Ready", "until_timeout": 500}, _dump(action)
        )
        assert payload["ok"] and payload["await_outcome"] == "satisfied"
        assert payload["observation_present"]
        assert runtime.clicks == 1
        assert "probe" in runtime.reads
        assert runtime.active_reads == 0
    finally:
        eng.close()


def test_cold_runtime_refuses_without_connecting():
    with pytest.raises(UsageError) as error:
        engine().wait(for_="Ready", timeout_ms=100)
    assert error.value.code == "wait_runtime_not_ready"


def test_zero_performs_one_probe_with_an_explicit_bounded_read_budget():
    runtime = Runtime()
    eng = engine(runtime)
    eng.config.perf.max_wait_ms = REAL_WAIT_MS
    result = eng.wait(for_="Missing", timeout_ms=0, by="rid")
    assert not result.ok
    assert runtime.reads == ["probe"]
    assert result.wait_budget_ms == REAL_WAIT_MS
    assert "one probe without polling" in result.note


def test_missing_predicate_still_returns_one_fresh_readback_inside_the_budget(monkeypatch):
    # Model timely adapter reads and the polling reserve explicitly. The real-wall tests
    # above separately prove that slow I/O expires without detached background work.
    now = [100.0]

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(
        engine_waits,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=advance, time=time.time),
    )
    runtime = Runtime(probe_delay=0.01, foreground_delay=0.01, tree_delay=0.02)
    monkeypatch.setattr(runtime, "_wait", advance)
    eng = engine(runtime)
    result = eng.wait(for_="Missing", timeout_ms=REAL_WAIT_MS, observe=True, with_image=False)
    assert not result.ok and result.observation is not None
    assert runtime.reads.count("probe") > 1
    assert runtime.reads.count("tree") == 1
    assert dict(runtime.read_waits)["tree"] == 0.02
    assert result.observation.elements[0].text == "Ready"
    assert 0 < result.elapsed_ms < result.wait_budget_ms == REAL_WAIT_MS
    assert set(runtime.read_threads) == {threading.get_ident()}
    assert runtime.active_reads == 0
    assert read_budget.current() is None


def test_nested_wait_reports_the_parent_limited_effective_budget():
    runtime = Runtime()
    eng = engine(runtime)
    parent = read_budget.ReadBudget(time.monotonic() + 0.15, time.monotonic)
    with runtime.read_deadline(parent):
        result = eng.wait(for_="Missing", timeout_ms=1000, by="rid")
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
    result = eng.wait(for_="Missing", timeout_ms=REAL_WAIT_MS, by="rid")
    assert time.monotonic() - started < MAX_ELAPSED_S
    assert runtime.reads.count("probe") == 1
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


def test_await_retries_a_failed_passive_read_within_the_same_deadline(monkeypatch):
    runtime = Runtime()
    eng = engine(runtime)
    probes = []

    def probe(*args, **kwargs):
        probes.append(read_budget.current().deadline)
        if len(probes) == 1:
            raise DeviceError("passive UI read failed")
        return (10, 10, 40, 40)

    monkeypatch.setattr(runtime, "find_text", probe)
    result = eng.await_predicate("text:Ready", timeout_ms=300, poll_ms=10,
                                 rich_ui=False, observe=False)
    assert result.ok and result.await_outcome == "satisfied"
    assert len(probes) == 2 and probes[0] == probes[1]
    assert read_budget.current() is None


@pytest.mark.parametrize("predicate", ["text:Ready", "!text:Loading"])
@pytest.mark.parametrize("scheduler_pause", [0.0, 0.15], ids=["repeated-polls", "deadline-after-first"])
def test_repeated_passive_read_errors_timeout_and_never_prove_absence(
    monkeypatch, predicate, scheduler_pause
):
    # Poll count is a clock property, not a guarantee that a loaded CI worker will
    # be scheduled twice within 150 ms. Exercise both schedules deterministically;
    # the slow-runtime tests above independently enforce the real wall deadline.
    now = [100.0]

    def sleep(seconds):
        now[0] += max(seconds, scheduler_pause)

    monkeypatch.setattr(
        engine_waits, "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=sleep, time=time.time),
    )
    runtime = Runtime()
    monkeypatch.setattr(runtime, "_wait", lambda seconds: None)
    eng = engine(runtime)
    probes = []

    def probe(*args, **kwargs):
        probes.append(read_budget.current().deadline)
        raise DeviceError("passive UI read failed")

    monkeypatch.setattr(runtime, "find_text", probe)
    result = eng.await_predicate(predicate, timeout_ms=150, poll_ms=10,
                                 rich_ui=False, observe=False)
    assert now[0] == pytest.approx(100.15)
    assert result.wait_budget_ms == 150
    assert not result.ok and result.await_outcome == "timeout"
    if scheduler_pause:
        assert len(probes) == 1
    else:
        assert len(probes) >= 2
    assert set(probes) == {100.15}
    assert result.await_terms[0]["satisfied"] is False
    assert result.await_terms[0]["reason"] == "ui_read_failed"
    assert read_budget.current() is None


def test_passive_read_retry_does_not_delay_cancellation(monkeypatch):
    runtime = Runtime()
    eng = engine(runtime)
    event = threading.Event()
    eng._job_cancel_event = event
    probes = []

    def probe(*args, **kwargs):
        probes.append(True)
        event.set()
        raise DeviceError("passive UI read failed")

    monkeypatch.setattr(runtime, "find_text", probe)
    started = time.monotonic()
    with pytest.raises(JobCancelledError):
        eng.await_predicate("text:Ready", timeout_ms=5000, poll_ms=1000, rich_ui=False)
    assert time.monotonic() - started < 0.5 and len(probes) == 1
    assert read_budget.current() is None


def test_passive_read_retry_does_not_hide_unsupported_capability(monkeypatch):
    runtime = Runtime()

    def probe(*args, **kwargs):
        raise DeviceError("unsupported read", code="unsupported_capability")

    monkeypatch.setattr(runtime, "find_text", probe)
    with pytest.raises(DeviceError, match="unsupported read"):
        engine(runtime).await_predicate("text:Ready", timeout_ms=500, rich_ui=False)


def test_detached_await_recovers_after_one_passive_read_failure(monkeypatch):
    from android_ui_analyser.jobs import manager_for

    runtime = Runtime()
    probes = []

    def probe(*args, **kwargs):
        probes.append(True)
        if len(probes) == 1:
            raise DeviceError("passive UI read failed")
        return (10, 10, 40, 40)

    monkeypatch.setattr(runtime, "find_text", probe)
    manager = manager_for(engine(runtime))
    started = manager.start("await", {"predicate": "rid:ready_control", "timeout_ms": 500,
                                      "poll_ms": 10, "observe": False})
    terminal = manager.wait(started["job_id"], timeout_ms=1500)
    assert terminal["status"] == "succeeded" and terminal["run_ok"] is True
    assert terminal["result"]["await_outcome"] == "satisfied"
    assert len(probes) >= 2


def test_rich_text_recheck_cannot_turn_failed_negative_id_probe_into_absence(monkeypatch):
    runtime = Runtime()

    def probe(*args, **kwargs):
        if kwargs.get("by") == "rid":
            raise DeviceError("passive UI read failed")
        return None  # Rich observation independently finds the Ready text.

    monkeypatch.setattr(runtime, "find_text", probe)
    result = engine(runtime).await_predicate(
        "!rid:loading_control,text:Ready", timeout_ms=150, poll_ms=10, observe=False
    )
    assert not result.ok and result.await_outcome == "timeout"
    assert result.await_terms[0]["satisfied"] is False
    assert result.await_terms[0]["reason"] == "ui_read_failed"


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
        eng.wait(for_="Ready", timeout_ms=REAL_WAIT_MS, observe=True, with_image=False)
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
    runtime = FakeDevice()
    eng = Engine(config, device=runtime)
    result = eng.await_predicate("!text:Canvas loading", timeout_ms=0, observe=False)
    assert not result.ok and result.await_outcome == "timeout"
    assert result.await_terms[0]["satisfied"] is False
    assert result.await_terms[0]["evidence"] == "unconfirmed"
    assert runtime.hierarchy_calls > 0


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
    destinations = []
    monkeypatch.setattr(
        eng, "_sample_action_destination", lambda: destinations.append(True) or _observation()
    )
    result = eng.await_predicate(
        "rid:catalogItemCard,!text:Canvas loading", timeout_ms=0, adopt_action=True
    )
    assert not result.ok
    assert result.await_outcome == "timeout"
    assert not all(row["satisfied"] for row in result.await_terms)
    assert destinations == [True]
    assert runtime.hierarchy_calls > 0
    if warm:
        assert eng.factory._instances[("ocr", "stub_ocr")].calls > 0
