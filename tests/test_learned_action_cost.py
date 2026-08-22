"""AUA remembers what a control cost last time, tells the agent, and waits longer itself.

The pre-existing learning was one EMA per *action kind* (tap/swipe/key), app-wide, held in a dict
built fresh in ``Engine.__init__`` — so it reset every run, and it averaged a 40ms same-screen tap
together with a login button that waits on a network round-trip. ``perf.py`` then capped it at
1.6s, while real screens in this app take 18-60s. A per-kind number cannot answer the question an
agent actually has, which is "what will *this* button on *this* screen cost me".

Timing is stored per (screen, control), scoped to the flag context, and used three ways: surfaced
in ``meta.slow_controls`` when the agent arrives, priced onto the element itself as ``cost`` in
the folded observation an action hands back (``meta.slow_controls`` is not in the ``changed`` meta
preset those are trimmed to), and spent as a deadline when it acts.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.identity import stable_key
from android_ui_analyser.memory import AppMemoryStore
from conftest import make_config


def _store(tmp_path: Path) -> AppMemoryStore:
    cfg = make_config(memory={"enabled": True, "dir": str(tmp_path)})
    return AppMemoryStore(cfg.memory)


P = "com.example.app"


def _seeded(tmp_path: Path) -> AppMemoryStore:
    from test_memory import HOME, _elements

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="welcome")
    return store


def test_a_timing_is_remembered_per_screen_and_control(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    store.record_action_timing(
        P, screen="welcome", control="buttonContinue", ms=5000.0, outcome="changed"
    )

    t = store.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.n == 1
    assert t.ema_ms == 5000.0 and t.max_ms == 5000.0

    # A different control on the same screen is a different cost, not the same one.
    assert store.action_timing(P, screen="welcome", control="buttonCancel") is None


def test_it_survives_a_new_store_instance(tmp_path: Path) -> None:
    """The old profile died with the process; this is the half that makes it *memory*."""
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=5000.0)

    fresh = _store(tmp_path)
    t = fresh.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.ema_ms == 5000.0


def test_repeated_samples_keep_the_worst_case_not_just_the_average(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    for ms in (5000.0, 1000.0, 1000.0):
        store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=ms)

    t = store.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.n == 3
    assert t.max_ms == 5000.0, "a deadline built from the mean is too short half the time"
    assert t.ema_ms < 5000.0, "and the average still tracks the recent trend"


def test_slow_controls_are_offered_worst_first_and_the_fast_ones_are_not(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=6000.0)
    store.record_action_timing(P, screen="welcome", control="buttonSlowish", ms=2000.0)
    store.record_action_timing(P, screen="welcome", control="linkFast", ms=40.0)

    rows = store.slow_controls(P, screen="welcome")
    assert [r["control"] for r in rows] == ["buttonContinue", "buttonSlowish"], (
        "worst first, and an ordinary sub-second tap is not a warning"
    )
    assert rows[0]["n"] == 1, "sample count rides along so one observation is not read as ten"


def test_a_timeout_is_recorded_rather_than_discarded(tmp_path: Path) -> None:
    """The coarse EMA refuses timeouts because they poison an app-wide average. Per control,
    "this one timed out" is exactly what the next run needs to be told."""
    store = _seeded(tmp_path)
    store.record_action_timing(
        P, screen="welcome", control="buttonContinue", ms=8000.0, outcome="unchanged"
    )

    t = store.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.last_outcome == "unchanged"


def test_an_unknown_screen_or_control_records_nothing(tmp_path: Path) -> None:
    # No screen record → no home for the timing. Silently filing it under a name that means
    # something else next run would be worse than not learning it, because it gets spent.
    store = _store(tmp_path)
    store.record_action_timing(P, screen="never-seen", control="buttonContinue", ms=5000.0)
    assert store.action_timing(P, screen="never-seen", control="buttonContinue") is None


def test_timings_do_not_leak_across_flag_contexts(tmp_path: Path) -> None:
    """An experiment arm that adds a network call changes the cost; averaging arms describes
    neither. CLAUDE.md requires learned data be scoped to the exact flag context."""
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=5000.0)

    assert (
        store.action_timing(
            P, screen="welcome", control="buttonContinue", context_id="flags:experiment-on"
        )
        is None
    )


# ------------------------------------------------------------------- wiring: store → agent
#
# The store above was always sound; what was missing were tests that a timing survives the trip
# through `Engine`. Every one of the four uses read the package from a helper that did not exist,
# and each call sat inside `contextlib.suppress(Exception)` — so the whole feature failed silently
# and the store-level tests above stayed green. These four cover each use through the Engine.


def _engine_on_apps(tmp_path: Path, serial: str):
    """An Engine on a known screen — the precondition for any per-(screen, control) timing."""
    from conftest import FakeDevice
    from test_memory import APPS, _elements, _engine

    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial=serial)
    eng = _engine(tmp_path, dev)
    assert eng._memory is not None
    eng._memory.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    observation = eng.analyze(source="hierarchy")
    screen = observation.meta.known_screen
    assert screen, "a timing needs a recognised screen to live on"
    element = next(e for e in observation.elements if (e.text or "") == "Reports")
    control = element.stable_key or (element.resource_id or "").rsplit("/", 1)[-1]
    assert control
    return eng, observation, screen, element, control


def test_an_arriving_agent_is_warned_which_control_is_slow(tmp_path: Path) -> None:
    """The arrival half of the contract: `meta.slow_controls` on the analyze that lands here.

    Seeded before the first observation because that is the case worth having — cost learned in
    an earlier run, spent by the agent that arrives next.
    """
    from conftest import FakeDevice
    from test_memory import APPS, _elements, _engine

    elements = _elements(APPS)
    control = next(e.stable_key for e in elements if (e.text or "") == "Reports")
    eng = _engine(tmp_path, FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-cost-arrival"))
    assert eng._memory is not None
    eng._memory.record_screen(package=P, elements=elements, name_hint="apps")
    eng._memory.record_action_timing(
        P, screen="apps", control=control, ms=6000.0, outcome="changed"
    )

    observation = eng.analyze(source="hierarchy")  # the agent arrives

    assert observation.meta.known_screen == "apps"
    assert [row["control"] for row in observation.meta.slow_controls] == [control], (
        "a learned 6s control must reach the agent that is about to tap it"
    )


def test_a_cost_learned_mid_session_reaches_a_screen_that_has_not_changed(tmp_path: Path) -> None:
    """A still screen is exactly when the agent is deciding what to tap next.

    `analyze` reuses the previous payload when the tree hashes the same, and that reuse refreshes
    `known_screen` precisely because the map keeps learning between calls. A learned cost is the
    same kind of fact — memory, not a property of the tree — so it has to be re-read too, or the
    warning is stale for as long as the screen sits still.
    """
    eng, _, screen, _, control = _engine_on_apps(tmp_path, "emu-cost-unchanged")
    eng._memory.record_action_timing(  # type: ignore[union-attr]
        P, screen=screen, control=control, ms=6000.0, outcome="changed"
    )

    again = eng.analyze(source="hierarchy")

    assert again.meta.unchanged is True, "an identical tree must take the reuse path"
    assert [row["control"] for row in again.meta.slow_controls] == [control], (
        "a cost learned since the last analyze must still be told on an unchanged screen"
    )


def test_a_learned_cost_rides_on_the_element_it_belongs_to(tmp_path: Path) -> None:
    """ "tap this next, and it takes ~4.8s" is one read — now on the element, not beside it.

    This used to be asserted through the derived `next_actions` list, which was the only route
    from memory to an acting agent: `meta.slow_controls` carries the same numbers and is not in
    the `changed` meta preset every folded observation is trimmed to. That list cost more than
    the whole `elements` list it was a filtered subset of, so it went behind an opt-in and the
    cost moved onto the row it describes. Same promise, one fewer list.
    """
    eng, observation, screen, element, control = _engine_on_apps(tmp_path, "emu-cost-priced")
    eng._memory.record_action_timing(  # type: ignore[union-attr]
        P, screen=screen, control=control, ms=4800.0, outcome="changed"
    )

    eng._price_elements(observation)
    priced = {stable_key(e): e.cost for e in observation.elements if e.cost is not None}

    assert priced[stable_key(element)] == {
        "avg_ms": 4800,
        "max_ms": 4800,
        # One observation must not read as ten.
        "n": 1,
    }, "the cost must be priced onto its own control"
    assert list(priced) == [stable_key(element)], "a control with no history stays unpriced"


def test_an_observed_action_records_what_it_cost(tmp_path: Path) -> None:
    """Nothing is ever learned unless the action files its own measurement.

    The id cache is deleted the moment the device is touched, so this bookkeeping runs when
    `_cached_package()` has nothing left to read — the package has to be the one captured with
    the action site, before the gesture.
    """
    eng, _, screen, element, control = _engine_on_apps(tmp_path, "emu-cost-record")

    eng.tap(element.id, observe=True)

    timing = eng._memory.action_timing(P, screen=screen, control=control)  # type: ignore[union-attr]
    assert timing is not None, "an observed action must record its own cost"
    assert timing.n == 1 and timing.max_ms > 0


def test_a_measured_control_earns_a_longer_deadline_than_the_coarse_profile(
    tmp_path: Path,
) -> None:
    """The spending half: a control measured at 20s must not be waited on for 1.1s."""
    eng, _, screen, element, control = _engine_on_apps(tmp_path, "emu-cost-budget")
    eng._memory.record_action_timing(  # type: ignore[union-attr]
        P, screen=screen, control=control, ms=20_000.0, outcome="changed"
    )
    eng._step("tap", element)  # what a real action does just before touching the device

    assert eng._learned_action_budget(1100) == 30_000, (
        "history must extend the deadline (max_ms x 1.5), never shorten it"
    )
