"""One call, not a guess-loop: an unready destination is waited on, bounded, then reported.

The owner's framing, which this contract implements: *"remove blind waits and extra function
calls from the agent, and make the agent wait for the response, instead of multiple function
calls and waits."* AUA already does this when the caller can name the destination (`--until`).
This is for when it cannot — you tap a login control you have never tapped before, and the new
Activity waits on a network round trip before rendering. Recorded shape (fictional names):
settle `via=hierarchy` at 199ms, activity changed, node count identical, nothing added.

Phase 0 made that frame honest (`stale_risk`). Phase 1 makes it *correct* when possible: the
action holds on, re-reads cheaply, and returns the rendered destination in the same call —
`arrival` says what happened either way. The extension spends time only where the answer was
already wrong; a settled tap must not pay a millisecond for it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PKG = "com.example.fiction"

# The recorded settle shape. `tree_added` deliberately absent: these tests isolate the
# content-wait; the shrinking-tree confirmation has its own file.
JOURNALLED_SETTLE = {"changed": True, "masked": 0, "ms": 199, "timeout": False, "via": "hierarchy"}


def _screen(*rows: str) -> str:
    return '<hierarchy rotation="0">' + "".join(rows) + "</hierarchy>"


def _node(
    text: str = "",
    *,
    rid: str = "",
    y: int,
    clickable: bool = False,
    cls: str = "android.view.View",
) -> str:
    return (
        f'<node class="{cls}" package="{PKG}" text="{text}"'
        f' resource-id="{rid}" clickable="{str(clickable).lower()}" enabled="true"'
        f' bounds="[20,{y}][500,{y + 80}]"/>'
    )


ENTRY = _screen(
    _node("Welcome back", rid=f"{PKG}:id/entryTitle", y=20),
    _node("Continue with MegaID", rid=f"{PKG}:id/loginBtn", y=120, clickable=True),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)

# The new Activity is foreground but has drawn nothing: old content minus the tapped control.
AUTH_BLANK = _screen(
    _node("Welcome back", rid=f"{PKG}:id/entryTitle", y=20),
    _node(y=120),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)

AUTH_READY = _screen(
    _node("Choose an account", rid=f"{PKG}:id/accountTitle", y=20),
    _node("Add account", rid=f"{PKG}:id/addAccount", y=120, clickable=True),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)

AUTH_SPINNER = _screen(
    _node("Signing in", rid=f"{PKG}:id/authStatus", y=20),
    _node(y=120, cls="android.widget.ProgressBar"),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)


class SlowRenderAuth(FakeDevice):
    """Tapping login fronts a new Activity that renders *ready_xml* only after a delay."""

    def __init__(self, ready_xml: str, *, render_after_s: float, blank_xml: str = AUTH_BLANK):
        super().__init__(hierarchy_xml=ENTRY, package=PKG, activity=".launch.EntryActivity")
        self._ready_xml = ready_xml
        self._blank_xml = blank_xml
        self._render_after_s = render_after_s
        self._tapped_at: float | None = None

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._tapped_at = time.monotonic()
        self._act = ".auth.AuthActivity"

    def dump_hierarchy(self, compressed: bool = False) -> str:
        self.hierarchy_calls += 1
        if self._tapped_at is None:
            return ENTRY
        if time.monotonic() - self._tapped_at < self._render_after_s:
            return self._blank_xml
        return self._ready_xml


def _engine(tmp_path: Path, device: FakeDevice, **perf: Any) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perception={"observe_escalates_to_vision": False},
        perf={"stable_delay_ms": {"default": 0}, **perf},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _tap_login(eng: Engine, monkeypatch: Any) -> Any:
    login = next(
        e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Continue with MegaID"
    )
    monkeypatch.setattr(eng, "_await_post_action_ready", lambda **_: dict(JOURNALLED_SETTLE))
    return eng.tap(login, observe=True)


def test_the_call_returns_the_rendered_destination_not_the_blank_frame(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The recorded case, with the network completing inside the budget: one call, right answer."""
    dev = SlowRenderAuth(AUTH_READY, render_after_s=0.45)
    eng = _engine(tmp_path, dev)

    out = _tap_login(eng, monkeypatch)

    assert out.ok and out.observation is not None
    texts = {e.text for e in out.observation.elements if e.text}
    assert "Choose an account" in texts, f"the caller must be handed the rendered screen: {texts}"
    assert out.stale_risk is None, f"content arrived — nothing left to warn about: {out.stale_risk}"
    assert out.arrival is not None and out.arrival["state"] == "settled"
    assert "content_arrived" in out.arrival["evidence"]
    assert out.arrival.get("waited_ms", 0) > 0
    assert out.settle is not None and out.settle.get("content_ms", 0) > 0
    # The rendered screen's ids must be immediately actionable (published id space).
    assert any(e.text == "Add account" for e in out.observation.elements)


def test_a_destination_that_never_renders_is_reported_not_guessed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dev = SlowRenderAuth(AUTH_READY, render_after_s=30.0)
    eng = _engine(tmp_path, dev, arrival_extension_ms=400)

    started = time.monotonic()
    out = _tap_login(eng, monkeypatch)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert out.ok
    assert out.stale_risk and "rendered nothing" in out.stale_risk
    assert "waited" in out.stale_risk, "the caveat must say the wait already happened"
    assert out.arrival is not None and out.arrival["state"] == "unconfirmed"
    assert "departure_without_arrival" in out.arrival["evidence"]
    assert "content_wait_expired" in out.arrival["evidence"]
    assert out.observation is not None
    assert out.observation.meta.arrival_state == "unconfirmed"
    assert elapsed_ms < 3000, f"the extension is bounded; this took {elapsed_ms:.0f}ms"


def test_a_persistent_loading_indicator_is_reported_as_loading(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dev = SlowRenderAuth(AUTH_SPINNER, render_after_s=0.0, blank_xml=AUTH_SPINNER)
    eng = _engine(tmp_path, dev, arrival_extension_ms=300)

    out = _tap_login(eng, monkeypatch)

    assert out.stale_risk and "loading" in out.stale_risk.lower()
    assert out.arrival is not None and out.arrival["state"] == "loading"
    assert "loading_indicator" in out.arrival["evidence"]
    assert out.observation is not None and out.observation.meta.arrival_state == "loading"


def test_the_env_knob_disables_the_wait_but_not_the_honesty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """`AUA_ARRIVAL_EXTENSION_MS=0` is the sweep baseline: Phase 0 behaviour, verdict intact."""
    monkeypatch.setenv("AUA_ARRIVAL_EXTENSION_MS", "0")
    dev = SlowRenderAuth(AUTH_READY, render_after_s=30.0)
    eng = _engine(tmp_path, dev)

    out = _tap_login(eng, monkeypatch)

    assert out.stale_risk and "rendered nothing" in out.stale_risk
    assert out.arrival is not None and out.arrival["state"] == "unconfirmed"
    assert "content_wait_expired" not in out.arrival["evidence"], "no wait ran"
    assert not (out.settle or {}).get("content_ms")


def test_a_settled_tap_carries_no_arrival_field_and_pays_nothing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Absence is the signal: a clean arrival adds zero bytes and zero waiting."""
    dev = SlowRenderAuth(AUTH_READY, render_after_s=0.0)
    eng = _engine(tmp_path, dev)

    started = time.monotonic()
    out = _tap_login(eng, monkeypatch)
    elapsed_ms = (time.monotonic() - started) * 1000

    assert out.stale_risk is None
    assert out.arrival is None, f"settled without waiting must stay silent: {out.arrival}"
    assert out.observation is not None and out.observation.meta.arrival_state is None
    assert elapsed_ms < 1500, f"a settled tap must not pay for the extension: {elapsed_ms:.0f}ms"


def test_a_no_op_tap_names_its_state_and_keeps_the_repeat_mutation_caveat(
    tmp_path: Path,
) -> None:
    dev = FakeDevice(hierarchy_xml=ENTRY, package=PKG)
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perf={"stable_delay_ms": {"default": 0}},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    login = next(
        e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Continue with MegaID"
    )

    out = eng.tap(login, observe=True)

    assert out.arrival is not None and out.arrival["state"] == "no_change"
    assert out.stale_risk and "not evidence" in out.stale_risk.lower(), (
        "the repeat-mutation caveat is the best-written invariant in the file; it survives"
    )
