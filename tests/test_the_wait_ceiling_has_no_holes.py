"""No single observation wait may block longer than ``perf.max_wait_ms``.

The ceiling itself already exists (`perf.clamp_wait_ms`, `Engine._bounded_wait_ms`,
`Engine._say_the_wait_was_shortened`) and is asserted by
``tests/test_a_wait_is_capped_and_deliberate.py``. This file is about the paths that were
still walking around it, because a ceiling with a hole in it is worse than no ceiling: it
reads as a guarantee and is not one.

Four holes, each found by hand rather than by this suite:

* ``wait_stable`` and ``wait_changed`` sized their deadline straight from the caller's
  ``timeout_ms``. They were deliberately left alone when the ceiling landed, to avoid
  colliding with a parallel edit, and nobody came back to them.
* the poll interval. A clamped deadline buys nothing while one sleep between polls can
  outlast the whole budget: ``--interval 30000`` spends 30s inside a 5s ceiling, because the
  loop only notices it is late *after* waking up.
* ``wait --for`` / ``wait --idle``, which handed ``timeout_ms`` to the device untouched.
* ``expect --timeout``, the assertion poll, which is an observation wait wearing a different
  name.

Everything here routes through the one existing gate. The deliberate exemption is also
pinned: a background job is the single wait nobody is blocked on, so it keeps the budget it
was given — which is the whole reason the `job` vocabulary exists.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.errors import StabilityTimeout
from android_ui_analyser.perf import clamp_wait_ms, is_provisioning_wait
from conftest import FakeDevice, make_config, make_engine, make_png

FRAME = make_png(width=120, height=200, color=(240, 240, 240))

_XML = '<hierarchy rotation="0"><node text="Initial" bounds="[0,0][100,60]"/></hierarchy>'


def _engine(device: FakeDevice, *, ceiling_ms: int):
    return make_engine(
        device=device,
        perf={"max_wait_ms": ceiling_ms},
        daemon={"enabled": False},
    )


class ChangingHierarchy(FakeDevice):
    """A tree that changes once, on the second dump."""

    def dump_hierarchy(self, compressed: bool = False) -> str:
        self.hierarchy_calls += 1
        if self.hierarchy_calls <= 1:
            return _XML
        return '<hierarchy rotation="0"><node text="Arrived" bounds="[0,0][100,60]"/></hierarchy>'


# --------------------------------------------------------------- the ceiling binds every wait


def test_a_sixty_second_wait_stable_returns_well_inside_the_ceiling() -> None:
    # settle_ms far past the budget: this screen can never satisfy the wait, so the only thing
    # that can end it is the deadline — which must be the ceiling, not the request.
    dev = FakeDevice(screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    with pytest.raises(StabilityTimeout) as ei:
        eng.wait_stable(interval_ms=1, settle_ms=30_000, timeout_ms=60_000)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 3_000, f"a clamped 60s wait blocked {elapsed_ms:.0f}ms"
    assert "150" in str(ei.value)
    assert "60000" in (ei.value.hint or "")


def test_a_sixty_second_wait_changed_returns_well_inside_the_ceiling() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    with pytest.raises(StabilityTimeout) as ei:
        eng.wait_changed(timeout_ms=60_000, interval_ms=10)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 3_000, f"a clamped 60s wait blocked {elapsed_ms:.0f}ms"
    assert "150" in str(ei.value)
    assert "60000" in (ei.value.hint or "")


def test_wait_stable_still_raises_stability_timeout_when_clamped() -> None:
    """Clamping changes how long, not what happens: the raise contract is untouched."""
    dev = FakeDevice(screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=100)

    with pytest.raises(StabilityTimeout):
        eng.wait_stable(interval_ms=1, settle_ms=30_000, timeout_ms=60_000)


def test_a_sixty_second_wait_for_text_returns_well_inside_the_ceiling() -> None:
    """`wait --for` handed the caller's budget to the device untouched."""
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    result = eng.wait(for_="NothingOnThisScreenEver", timeout_ms=60_000)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert result.ok is False
    assert elapsed_ms < 3_000, f"a clamped 60s wait blocked {elapsed_ms:.0f}ms"


def test_has_shares_one_ceiling_between_hierarchy_and_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev, ceiling_ms=150)
    hierarchy_budgets: list[int] = []
    ocr_budgets: list[int | None] = []

    def miss_hierarchy(*args: object, timeout_ms: int, **kwargs: object) -> None:
        hierarchy_budgets.append(timeout_ms)

    def miss_ocr(*args: object, timeout_ms: int | None = None, **kwargs: object) -> None:
        ocr_budgets.append(timeout_ms)

    monkeypatch.setattr(dev, "wait_for", miss_hierarchy)
    monkeypatch.setattr(eng, "_ocr_contains", miss_ocr)

    result = eng.has("NeverThere", timeout_ms=60_000)

    assert hierarchy_budgets == [150]
    assert ocr_budgets and 0 < int(ocr_budgets[0] or 0) <= 150
    assert result.wait_clamped_from_ms == 60_000
    assert result.wait_ceiling_ms == 150


def test_a_sixty_second_absent_wait_returns_well_inside_the_ceiling() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, text_index={"Initial": (0, 0, 100, 60)})
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    result = eng.wait(for_="Initial", timeout_ms=60_000, absent=True)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert result.ok is False, "premise: the text never disappears"
    assert elapsed_ms < 3_000, f"a clamped 60s absent-wait blocked {elapsed_ms:.0f}ms"


def test_a_sixty_second_expect_poll_returns_well_inside_the_ceiling() -> None:
    """`expect --timeout` is an observation wait under another name."""
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    result = eng.expect(text="NothingOnThisScreenEver", exists=True, timeout_ms=60_000)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert result.ok is False
    assert elapsed_ms < 3_000, f"a clamped 60s expect blocked {elapsed_ms:.0f}ms"


# ------------------------------------------------------- a coarse poll cannot outlast the budget


def test_a_giant_poll_interval_cannot_outlast_the_ceiling_in_wait_stable() -> None:
    """A clamped deadline is worthless if one sleep between polls is longer than the budget."""
    dev = FakeDevice(screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    with pytest.raises(StabilityTimeout):
        eng.wait_stable(interval_ms=30_000, settle_ms=30_000, timeout_ms=60_000)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 3_000, f"one poll interval blocked {elapsed_ms:.0f}ms"


def test_a_giant_poll_interval_cannot_outlast_the_ceiling_in_wait_changed() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=150)

    started = time.monotonic()
    with pytest.raises(StabilityTimeout):
        eng.wait_changed(timeout_ms=60_000, interval_ms=30_000)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 3_000, f"one poll interval blocked {elapsed_ms:.0f}ms"


def test_a_giant_poll_interval_cannot_outlast_the_ceiling_in_await() -> None:
    """`await` clamps its deadline already; its poll sleep did not respect it."""
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev, ceiling_ms=150)
    eng.analyze()

    started = time.monotonic()
    eng.await_predicate(
        "text:NothingOnThisScreenEver", timeout_ms=60_000, poll_ms=30_000, observe=False
    )
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 3_000, f"one poll interval blocked {elapsed_ms:.0f}ms"


def test_a_giant_poll_interval_cannot_outlast_the_ceiling_in_wait_after_change() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME] * 200)
    eng = _engine(dev, ceiling_ms=300)

    started = time.monotonic()
    eng.wait_after_change(
        timeout_ms=60_000, interval_ms=30_000, settle_ms=1, confirmation_ms=30_000, observe=False
    )
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms < 3_000, f"one poll interval blocked {elapsed_ms:.0f}ms"


# ------------------------------------------------------------------- a shortened wait says so


def test_a_shortened_wait_stable_explains_itself_on_the_result() -> None:
    dev = FakeDevice(screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=2_000)

    res = eng.wait_stable(interval_ms=1, settle_ms=2, timeout_ms=60_000)

    assert res.ok
    note = res.note or ""
    assert "2000" in note and "60000" in note, note
    assert "max_wait_ms" in note
    assert res.wait_clamped_from_ms == 60_000
    assert res.wait_ceiling_ms == 2_000


def test_a_shortened_wait_changed_explains_itself_on_the_result() -> None:
    dev = ChangingHierarchy(hierarchy_xml=_XML, screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=2_000)

    res = eng.wait_changed(timeout_ms=60_000, interval_ms=10)

    assert res.ok
    note = res.note or ""
    assert "2000" in note and "60000" in note, note


def test_a_wait_inside_the_ceiling_says_nothing_extra() -> None:
    dev = FakeDevice(screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=5_000)

    res = eng.wait_stable(interval_ms=1, settle_ms=2, timeout_ms=1_000)

    assert res.ok
    assert "max_wait_ms" not in (res.note or "")
    assert res.wait_clamped_from_ms is None


# ------------------------------------------------------------------------- configurable ceiling


def test_the_shipped_ceiling_is_five_seconds() -> None:
    assert Config().perf.max_wait_ms == 5_000


def test_the_ceiling_is_configurable() -> None:
    eng = _engine(FakeDevice(), ceiling_ms=12_000)

    assert eng._bounded_wait_ms(60_000) == (12_000, 60_000, 12_000)
    assert eng._bounded_wait_ms(9_000) == (9_000, None, 12_000)


def test_there_is_exactly_one_clamp_function() -> None:
    """A second ceiling is a second thing to forget. Every path uses this one."""
    import inspect

    from android_ui_analyser import engine as engine_mod
    from android_ui_analyser import perf as perf_mod

    # The engine is engine.py plus its engine_*.py domain modules; a clamp hiding in any of them
    # is the hole this guards against.
    engine_dir = Path(engine_mod.__file__).parent
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(engine_dir.glob("engine*.py")))
    assert src.count("def _bounded_wait_ms") == 1
    assert inspect.getsource(perf_mod).count("def clamp_wait_ms") == 1
    # The ceiling is read from config in exactly one place — `_bounded_wait_ms`. A second read
    # is a wait sizing its own budget, which is how the first set of holes got in.
    assert src.count("self.config.perf.max_wait_ms") == 1, "a wait is reading the ceiling itself"


# -------------------------------------------------------------------- provisioning is exempt


def test_provisioning_budgets_are_not_observation_waits() -> None:
    assert is_provisioning_wait("install")
    assert is_provisioning_wait("emulator-start")
    assert is_provisioning_wait("network")
    assert is_provisioning_wait("boot")
    assert is_provisioning_wait("job")
    assert not is_provisioning_wait("observation")
    assert not is_provisioning_wait("wait-stable")
    assert not is_provisioning_wait("wait-after-change")
    assert not is_provisioning_wait(None)
    assert not is_provisioning_wait("")


def test_a_background_job_keeps_its_full_budget() -> None:
    """The ceiling protects a blocked session; a job's caller polls `job status` instead."""
    eng = _engine(FakeDevice(), ceiling_ms=50)
    eng._job_cancel_event = threading.Event()

    assert eng._bounded_wait_ms(600_000) == (600_000, None, 50)


def test_a_background_job_wait_changed_spends_the_budget_it_was_given() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME] * 40)
    eng = _engine(dev, ceiling_ms=50)
    eng._job_cancel_event = threading.Event()

    started = time.monotonic()
    with pytest.raises(StabilityTimeout) as ei:
        eng.wait_changed(timeout_ms=400, interval_ms=10)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert elapsed_ms >= 300, f"the job's 400ms budget was cut to {elapsed_ms:.0f}ms"
    assert "400" in str(ei.value)


def test_a_foreground_caller_does_not_get_the_job_exemption() -> None:
    """The exemption is the job's, not the flag's: a normal engine is still clamped."""
    eng = _engine(FakeDevice(), ceiling_ms=50)

    assert eng._bounded_wait_ms(600_000) == (50, 600_000, 50)


def test_clamping_still_applies_when_the_ceiling_is_the_default() -> None:
    cfg = make_config()

    assert clamp_wait_ms(600_000, cfg) == (cfg.perf.max_wait_ms, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
