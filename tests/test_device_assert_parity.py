"""A step must mean the same thing on the device as it does on the host.

The offload is a shortcut, and a shortcut is only allowed to change the *cost*. That
guarantee has an asymmetry worth stating plainly, because it decides what is worth
guarding: a device-side check that wrongly FAILS is nearly free, since the host re-runs the
step and produces the authoritative answer. A device-side check that wrongly PASSES is
unrecoverable — the run continues, the flow reports success, and nobody ever learns the
assertion was never true.

So every divergence pinned here is one that can manufacture a false pass. They were all live
in shipped code: the device accepted selector vocabularies the host refuses, waited five
seconds where the host waits none, and matched nodes the host's projection had already
filtered off the screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import RouteStep
from conftest import FakeDevice, make_config


class _Agent:
    """Records the payload sent to the device without needing one."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def is_bound(self, serial: str) -> bool:
        return True

    def is_enabled(self, serial: str) -> bool:
        return True

    def is_installed(self, serial: str) -> bool:
        return True

    def rootable(self, serial: str) -> bool:
        return True

    def uiautomation_held(self, serial: str) -> bool:
        return False

    def release_uiautomation(self, serial: str) -> None:
        return None

    def enable(self, serial: str) -> dict[str, Any]:
        return {"enabled": True, "bound": True}

    def open_channel(self, serial: str, *, timeout: float = 5.0) -> Any:
        outer = self

        class _Channel:
            def request(self, method: str, params: Any = None, *, timeout: float = 5.0) -> Any:
                outer.requests.append({"method": method, "params": params})
                steps = (params or {}).get("steps") or []
                return {
                    "completed": len(steps),
                    "total": len(steps),
                    "stopped_at": -1,
                    "stopped_reason": None,
                    "steps": [
                        {"index": i, "kind": s.get("kind"), "ok": True, "ms": 1}
                        for i, s in enumerate(steps)
                    ],
                }

            def close(self) -> None:
                return None

        return _Channel()


def _engine(tmp_path: Path, agent: _Agent, **helper: Any) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        helper={"enabled": True, "min_flow_steps": 1, "min_midflow_steps": 1, **helper},
    )
    engine = Engine(cfg, device=FakeDevice())
    engine.platform.capability = lambda name: agent  # type: ignore[method-assign]
    return engine


def test_the_device_accepts_no_selector_field_the_host_would_refuse() -> None:
    """Anything the device is allowed to match on, the host must be able to re-run.

    ``Uiautomator2Device._fields_for`` refuses a ``by`` it does not know, on purpose — a
    silent fall-through to a text search once reported a screen wrong for a full 15s timeout.
    The offload keeps its own list of the fields it will hand over, and the two drifted: it
    contained ``content_desc`` and ``resource_id``, which the host refuses outright, so a
    step could be run on the device and then be a hard usage error the moment the host
    touched it.
    """

    unknown = set(Engine._DEVICE_BY_FIELDS) - set(Uiautomator2Device._BY_FIELDS)

    assert not unknown, (
        f"the offload would hand the device selector fields the host refuses: {sorted(unknown)}"
    )


def test_a_resource_id_predicate_is_offered_to_the_device(tmp_path: Path) -> None:
    """``by: id`` is what the flow parser actually emits, and it was never handed over.

    ``flows.py`` normalizes ``wait_for: {id: foo}`` to ``by=\"id\"``. The offload's field list
    knew ``rid`` but not ``id``, so every resource-id predicate in every saved flow was
    silently disqualified — the run stopped there and the rest went back to the host one
    round trip at a time. Nothing was wrong with the step; the two spellings had simply never
    been reconciled.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent)
    steps = [
        RouteStep(kind="tap", resource_id="rowOne"),
        RouteStep(kind="assert-visible", arg="panelCheckout", by="id"),
        RouteStep(kind="tap", resource_id="rowTwo"),
    ]

    ran = engine._offload_steps_to_device(steps, executed=None, allow_destructive=True)

    assert ran == 3, "a `by: id` predicate stopped the run from being handed over"


def test_an_unknown_selector_field_still_stays_on_the_host(tmp_path: Path) -> None:
    """Widening the vocabulary must not turn into accepting anything at all."""

    agent = _Agent()
    engine = _engine(tmp_path, agent)
    steps = [
        RouteStep(kind="tap", resource_id="rowOne"),
        RouteStep(kind="assert-visible", arg="something", by="regex"),
        RouteStep(kind="tap", resource_id="rowTwo"),
    ]

    ran = engine._offload_steps_to_device(steps, executed=None, allow_destructive=True)

    assert ran == 1, "a selector field neither side understands was handed to the device"


def test_the_device_is_told_how_long_each_check_may_wait(tmp_path: Path) -> None:
    """A check that waits longer on the device than on the host can invent a pass.

    The host gives ``assert-visible`` no wait at all (``timeout_ms=s.timeout_ms or 0``): it
    asserts about the screen as it is. The device defaulted to five seconds, so an element
    that appeared 400ms after the assertion ran made the device pass a check the host would
    have failed — and because a device pass is final, the run carried on with an assertion
    that was never true when it was made.

    The step's own timeout therefore has to travel with it, explicitly, rather than each side
    applying a default the other has never heard of.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent)
    steps = [
        RouteStep(kind="tap", resource_id="rowOne"),
        RouteStep(kind="assert-visible", arg="Total", by="text"),
        RouteStep(kind="wait-for", arg="Receipt", by="text"),
    ]

    engine._offload_steps_to_device(steps, executed=None, allow_destructive=True)

    sent = agent.requests[0]["params"]["steps"]
    by_kind = {s["kind"]: s for s in sent}

    assert by_kind["assert-visible"]["timeout_ms"] == 0, (
        "assert-visible went to the device with no timeout, so it used the device default "
        "and waited where the host would not have"
    )
    assert by_kind["wait-for"]["timeout_ms"] == 10000, (
        "wait-for went to the device without the host's own 10s budget"
    )


def test_an_authored_timeout_is_passed_through_untouched(tmp_path: Path) -> None:
    """Filling in the default must not overwrite a timeout the flow actually asked for."""

    agent = _Agent()
    engine = _engine(tmp_path, agent)
    steps = [
        RouteStep(kind="tap", resource_id="rowOne"),
        RouteStep(kind="assert-visible", arg="Total", by="text", timeout_ms=2500),
    ]

    engine._offload_steps_to_device(steps, executed=None, allow_destructive=True)

    sent = agent.requests[0]["params"]["steps"]
    assert sent[1]["timeout_ms"] == 2500
