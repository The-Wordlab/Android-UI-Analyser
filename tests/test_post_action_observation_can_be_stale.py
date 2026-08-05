"""A post-action observation can be a snapshot of the *previous* screen, and must say so.

Observed in the field: a `tap` succeeded and the device advanced, yet the returned observation
reported `unchanged=true` with an empty `element_diff` — measured against a snapshot taken before
the screen changed. A screenshot plus a fresh `analyze` proved the app had in fact moved on.

Mechanism: `_await_post_action_ready` gives up early on purpose. It returns `via=unchanged` after
~80ms of pixel-identical frames and `via=hierarchy-same` on two matching trees, because waiting
longer made same-screen taps ~2x slower. The folded `analyze` then dumps a tree that still matches
the previous one, `skip_unchanged_analyze` reuses the *previous payload*, and the result is stamped
`unchanged=true`. Every individual step is correct; the claim is simply older than it looks.

This is the dangerous direction. The other four ways a tap can look inert all risk an agent giving
up too early. This one risks an agent **repeating an action that already happened** — a second
submit, a second message, a second purchase attempt. A runner cannot safely retry a mutating action
on the strength of `unchanged`, which removes most of the value of reporting it at all.

So the engine must not present it as evidence. It genuinely cannot distinguish "no effect" from
"not yet", and the honest answer is to say that — which is why a real in-screen no-op carries the
caveat too, and why the caveat is a string rather than a `False` flag (`compact` drops falsey
values and `delta` allowlists keys, so a boolean would vanish from the output in exactly the cases
that need it).
"""

from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import OutputFormat
from conftest import FakeDevice, make_config

PKG = "com.test.app"

SCREEN = (
    '<hierarchy rotation="0">'
    '<node class="android.widget.TextView" package="com.test.app" text="Compose"'
    ' resource-id="x:id/header" clickable="false" enabled="true" bounds="[40,120][1040,210]"/>'
    '<node class="android.widget.Button" package="com.test.app" text="Send"'
    ' resource-id="x:id/send" clickable="true" enabled="true" bounds="[40,640][400,740]"/>'
    "</hierarchy>"
)

CONFIRMED_CHANGE = {"changed": True, "masked": 0, "ms": 90, "timeout": False, "via": "hierarchy"}


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def test_an_unchanged_observation_is_not_offered_as_evidence(tmp_path: Path) -> None:
    """The reported shape, reproduced: a device that never moves yields `unchanged` + a caveat.

    A `FakeDevice` renders one fixed screen, so the settle wait takes the same early-exit branch
    the field hit (`via=unchanged`) and the folded analyze reuses the previous payload.
    """
    dev = FakeDevice(hierarchy_xml=SCREEN, package=PKG, serial="emu-stale")
    eng = _engine(tmp_path, dev)
    send = next(e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Send")

    out = eng.tap(send, observe=True)

    assert out.ok and out.observation is not None
    meta = out.observation.meta
    assert meta.unchanged is True, "precondition: this is the reported shape"
    assert meta.stale_risk, (
        "`unchanged` with no caveat is the trap: it invites a second submit. "
        f"meta={meta.model_dump(exclude_none=True)}"
    )
    assert "not evidence" in meta.stale_risk.lower()
    assert "retry" in meta.stale_risk.lower(), "the caveat must name the expensive mistake"


def test_a_confirmed_transition_carries_no_caveat(tmp_path: Path, monkeypatch) -> None:
    """The caveat must be earned. A wait that saw the screen change and stop is trustworthy."""
    dev = FakeDevice(hierarchy_xml=SCREEN, package=PKG, serial="emu-fresh")
    eng = _engine(tmp_path, dev)
    send = next(e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Send")
    monkeypatch.setattr(
        Engine, "_await_post_action_ready", lambda self, **kw: dict(CONFIRMED_CHANGE)
    )

    out = eng.tap(send, observe=True)

    assert out.observation is not None
    assert out.observation.meta.stale_risk is None, "a confirmed change is not a stale risk"


def test_a_settle_timeout_is_also_flagged(tmp_path: Path, monkeypatch) -> None:
    """A timed-out wait may hand back a mid-transition tree — the other stale direction."""
    dev = FakeDevice(hierarchy_xml=SCREEN, package=PKG, serial="emu-to")
    eng = _engine(tmp_path, dev)
    send = next(e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Send")
    monkeypatch.setattr(
        Engine,
        "_await_post_action_ready",
        lambda self, **kw: {
            "changed": True,
            "masked": 0,
            "ms": 1100,
            "timeout": True,
            "via": "timeout",
        },
    )

    out = eng.tap(send, observe=True)

    assert out.observation is not None
    risk = out.observation.meta.stale_risk
    assert risk and "timed out" in risk


def test_the_risk_is_visible_at_the_top_level_without_corrupting_detail(tmp_path: Path) -> None:
    """A runner reading only the terse form must find the caveat — in its own field.

    This first appended the marker to ``detail``, which was wrong: ``detail`` carries a semantic
    value for several actions (``app launch`` puts the launched package/activity there), so
    appending to it corrupts the thing a caller parses. Caught when ``app launch`` started folding
    in an observation and its detail came back as ``"com.example.app stale_risk"``.
    """
    dev = FakeDevice(hierarchy_xml=SCREEN, package=PKG, serial="emu-detail")
    eng = _engine(tmp_path, dev)
    send = next(e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Send")

    out = eng.tap(send, observe=True)

    assert out.stale_risk, "the caveat must be reachable without opening `observation.meta`"
    assert "not evidence" in out.stale_risk.lower(), "and must carry the reason, not a bare flag"
    assert "stale_risk" not in (out.detail or ""), "detail must not be polluted with the marker"


def test_the_caveat_survives_the_compact_and_delta_trims(tmp_path: Path) -> None:
    """`delta` fires exactly when `unchanged` is True, so the caveat has to be allowlisted.

    Both trims exist to save tokens: `compact` drops falsey values, `delta` keeps a fixed key set.
    A caveat that only appears in full `json` output would be missing from the two formats an
    agent actually runs under, in the one case it is about.
    """
    dev = FakeDevice(hierarchy_xml=SCREEN, package=PKG, serial="emu-fmt")
    eng = _engine(tmp_path, dev)
    send = next(e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Send")
    obs = eng.tap(send, observe=True).observation
    assert obs is not None and obs.meta.stale_risk

    for fmt in (OutputFormat.compact, OutputFormat.delta):
        payload = obs.as_dict(fmt)
        assert payload["meta"].get("stale_risk"), f"{fmt} dropped the caveat: {payload['meta']!r}"

    rendered = json.loads(obs.render(OutputFormat.delta))
    assert rendered["meta"]["unchanged"] is True
    assert rendered["meta"]["stale_risk"]


def test_a_raw_read_is_not_flagged() -> None:
    """`observe` without a settle is a deliberate raw look; there is nothing to be stale against."""
    assert Engine._stale_observation_risk(False, None) is None
    assert Engine._stale_observation_risk(False, dict(CONFIRMED_CHANGE)) is None


def test_a_missing_wait_result_is_treated_as_unverified() -> None:
    """Absence of evidence is not evidence — an unavailable wait must not read as confirmation."""
    risk = Engine._stale_observation_risk(True, None)
    assert risk and "did not run" in risk
