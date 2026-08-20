"""AC14 — ``wait --for-stable`` settles on a (stubbed) screenshot stream and times out
with a clear, structured error, WITHOUT running OCR or a hierarchy parse.

The settle check is a cheap perceptual-hash over screenshots only — we assert that by
spying on the device: ``dump_hierarchy`` is never called, and the OCR provider chain is
never built/invoked.
"""

from __future__ import annotations

import time

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ExitCode, StabilityTimeout
from android_ui_analyser.imaging import dhash, hamming, is_stable
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config, make_png

runner = CliRunner()

# Two structurally-distinct frames (a moving solid box → different dHash gradients).
FRAME_A = make_png(
    width=200, height=400, color=(240, 240, 240), boxes=[((10, 40, 90, 360), (0, 0, 0))]
)
FRAME_B = make_png(
    width=200, height=400, color=(240, 240, 240), boxes=[((110, 40, 190, 360), (0, 0, 0))]
)


def _engine(device: FakeDevice) -> Engine:
    cfg = make_config(daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


# --------------------------------------------------------------------------- hash unit


def test_dhash_identical_is_stable_distinct_is_not() -> None:
    a = ScreenImage(FRAME_A, width=200, height=400)
    b = ScreenImage(FRAME_B, width=200, height=400)
    assert is_stable(dhash(a), dhash(a))  # a frame equals itself
    assert hamming(dhash(a), dhash(b)) > 8  # the two frames are clearly different


# --------------------------------------------------------------------------- settle


def test_wait_stable_settles_without_ocr_or_hierarchy() -> None:
    # changes for two frames, then holds steady → settles on the steady run.
    stream = [FRAME_A, FRAME_B] + [FRAME_A] * 20
    dev = FakeDevice(screenshots=stream)
    eng = _engine(dev)
    res = eng.wait_stable(interval_ms=1, settle_ms=3, timeout_ms=3000)
    assert res.ok and res.action == "wait-stable"
    assert dev.hierarchy_calls == 0  # NO hierarchy parse
    assert dev.screenshot_calls > 0  # it polled screenshots


def test_wait_stable_does_not_build_the_ocr_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(screenshots=[FRAME_A] * 10)
    eng = _engine(dev)

    built: list[str] = []
    real_build = eng.factory.build_chain
    monkeypatch.setattr(
        eng.factory, "build_chain", lambda kind: built.append(kind) or real_build(kind)
    )
    eng.wait_stable(interval_ms=1, settle_ms=2, timeout_ms=2000)
    assert built == []  # no provider chain (OCR/detection) is ever constructed


# --------------------------------------------------------------------------- timeout


def test_wait_stable_times_out_with_structured_error() -> None:
    # Alternating frames never hold steady → never settles.
    dev = FakeDevice(screenshots=[FRAME_A, FRAME_B] * 200)
    eng = _engine(dev)
    with pytest.raises(StabilityTimeout) as ei:
        eng.wait_stable(interval_ms=1, settle_ms=50, timeout_ms=25)
    assert ei.value.exit_code == ExitCode.DEVICE == 3
    assert ei.value.hint  # actionable hint
    assert dev.hierarchy_calls == 0


# --------------------------------------------------------------------------- CLI wiring


def test_cli_wait_for_stable_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(screenshots=[FRAME_A] * 12)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    res = runner.invoke(
        app,
        ["wait-and-analyze", "--for-stable", "--interval", "1", "--settle", "2", "--timeout", "3000"],
    )
    assert res.exit_code == 0, res.stderr
    assert "wait-stable" in res.stdout


def test_cli_wait_for_stable_timeout_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(screenshots=[FRAME_A, FRAME_B] * 200)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    res = runner.invoke(
        app,
        # Settle deliberately outlasts the timeout, so an alternating screen cannot be called
        # stable and the wait must give up. Both were an order of magnitude smaller and this
        # failed roughly one full-suite run in eight — 25ms is not a window a shared machine can
        # be relied on to land inside. The relationship is what the test is about, not the size.
        [
            "wait-and-analyze",
            "--for-stable",
            "--interval",
            "5",
            "--settle",
            "500",
            "--timeout",
            "150",
        ],
    )
    assert res.exit_code == 3
    import json

    err = json.loads(res.stderr)
    assert err["error"]["code"] == "wait_timeout"


# --------------------------------------------------------------- wait --observe (fresh ids)

_XML = (
    '<hierarchy rotation="0">'
    '<node class="android.widget.Button" text="Continue" resource-id="x:id/go"'
    ' clickable="true" enabled="true" bounds="[0,0][100,60]"/>'
    "</hierarchy>"
)


def test_wait_for_observe_returns_fresh_ids() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, text_index={"Continue": (0, 0, 100, 60)})
    eng = _engine(dev)
    res = eng.wait(for_="Continue", timeout_ms=1000, observe=True)
    assert res.ok and res.observation is not None
    # the id is available immediately — no separate analyze needed
    assert any(e.text == "Continue" for e in res.observation.elements)


def test_wait_for_no_observe_has_no_observation() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, text_index={"Continue": (0, 0, 100, 60)})
    eng = _engine(dev)
    res = eng.wait(for_="Continue", timeout_ms=1000)
    assert res.ok and res.observation is None


def test_wait_stable_observe_returns_screen() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME_A] * 10)
    eng = _engine(dev)
    res = eng.wait_stable(interval_ms=1, settle_ms=2, timeout_ms=2000, observe=True)
    assert res.ok and res.observation is not None
    assert any(e.text == "Continue" for e in res.observation.elements)


def test_cli_wait_accepts_timeout_ms_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, text_index={"Continue": (0, 0, 100, 60)})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    # --timeout-ms is accepted as an alias for --timeout (both are ms).
    r = runner.invoke(app, ["wait-and-analyze", "--for", "Continue", "--timeout-ms", "1500"])
    assert r.exit_code == 0, r.stderr


def test_wait_after_change_restarts_when_result_arrives_after_quiet_loading_shell() -> None:
    """A first stable frame is provisional until the bounded confirmation stays quiet."""

    class ScriptedHierarchy(FakeDevice):
        """Working, then Result a fixed time *after the screen went quiet*.

        Scripting by call index instead made which phase the Result landed in depend on how
        many polls the host fitted into a 2ms settle window: on a 3-core CI runner the fifth
        dump arrived while the wait was still settling, so it was folded into the original
        change and there was no late change left to restabilize. Anchoring to elapsed time
        since the change keeps the ordering the test is about — quiet first, result second —
        on any machine.
        """

        QUIET_BEFORE_RESULT_S = 0.10

        def __init__(self) -> None:
            super().__init__(screenshots=[FRAME_A] * 80)
            self._changed_at: float | None = None

        def dump_hierarchy(self, compressed: bool = False) -> str:
            self.hierarchy_calls += 1
            if self.hierarchy_calls == 1:
                return '<hierarchy><node text="Initial"/></hierarchy>'
            if self._changed_at is None:
                self._changed_at = time.monotonic()
            if time.monotonic() - self._changed_at >= self.QUIET_BEFORE_RESULT_S:
                return '<hierarchy><node text="Result"/></hierarchy>'
            return '<hierarchy><node text="Working"/></hierarchy>'

    dev = ScriptedHierarchy()
    eng = _engine(dev)

    out = eng.wait_after_change(
        interval_ms=1,
        settle_ms=2,
        # The confirmation window has to outlast the 100ms quiet stretch above, or the wait
        # returns before the late change arrives and there is nothing to restabilize.
        confirmation_ms=500,
        timeout_ms=5000,
    )

    assert out.ok and out.action == "wait-after-change"
    assert "1 late change(s) restabilized" in (out.detail or "")


def test_cli_after_change_uses_the_confirmed_engine_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME_A] * 80)
    eng = _engine(dev)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_route(_engine, method: str, **kwargs):
        calls.append((method, kwargs))
        return engine_mod.ActionResult(ok=True, action="wait-after-change")

    from android_ui_analyser import cli

    monkeypatch.setattr(cli, "_route", fake_route)
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: eng)
    res = runner.invoke(
        app,
        ["wait-and-analyze", "--after-change", "--timeout", "9000", "--observe"],
    )

    assert res.exit_code == 0, res.stderr
    assert [method for method, _ in calls] == ["wait_after_change"]
    assert calls[0][1]["timeout_ms"] == 9000


def test_wait_and_await_accept_analyze_no_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, screenshots=[FRAME_A] * 10)
    eng = _engine(dev)

    from android_ui_analyser import cli

    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: eng)
    wait = runner.invoke(
        app,
        ["wait-and-analyze", "--for", "Ready", "--observe", "--no-meta", "--timeout", "1"],
    )
    assert wait.exit_code != 2, wait.output
    assert cli._OBSERVATION_VIEW is not None and cli._OBSERVATION_VIEW.no_meta

    awaited = runner.invoke(
        app,
        ["await-and-analyze", "text:Ready", "--observe", "--no-meta", "--timeout-ms", "1"],
    )
    assert awaited.exit_code != 2, awaited.output
    assert cli._OBSERVATION_VIEW is not None and cli._OBSERVATION_VIEW.no_meta


# --------------------------------------------------------------- grid-based settle (animation masking)


def test_grid_settle_masks_animation_and_settles() -> None:
    """A spinner in one corner doesn't block the rest of the screen from settling."""
    from android_ui_analyser.imaging import GridSettle

    w, h = 200, 200
    stable_color = (200, 200, 200)

    def frame(i: int) -> bytes:
        spinner = ((i * 60) % 255, 30, 30)
        return make_png(w, h, color=stable_color, boxes=[((0, 0, 50, 50), spinner)])

    frames = [ScreenImage(frame(i), width=w, height=h) for i in range(12)]
    gs = GridSettle(streak=3)
    settled_at: int | None = None
    for i, img in enumerate(frames):
        if gs.feed(img):
            settled_at = i
            break
    assert settled_at is not None, "GridSettle should have reported stable"
    assert len(gs.masked_cells) >= 1, "spinner cell should be masked"


def test_grid_settle_does_not_settle_when_everything_changes() -> None:
    from android_ui_analyser.imaging import GridSettle

    w, h = 100, 100
    frames = [
        ScreenImage(make_png(w, h, color=(i * 30, i * 30, i * 30)), width=w, height=h)
        for i in range(8)
    ]
    gs = GridSettle(streak=3)
    for img in frames:
        # After streak, all cells may mask → then "stable" over an empty set. That is
        # intentional: pure full-screen animation is treated as settled-except-anim. But at
        # least one cell must still be unmasked and changing while cells are being classified.
        if gs.feed(img) and gs.samples < 4:
            raise AssertionError("should not settle while cells are still being classified")


def test_wait_stable_ignore_animation_settles() -> None:
    """Integration: wait_stable with ignore_animation=True handles a looping spinner."""
    w, h = 200, 200
    stable_body = (200, 200, 200)

    def frame(i: int) -> bytes:
        spinner = ((i * 50) % 255, 20, 20)
        return make_png(w, h, color=stable_body, boxes=[((0, 0, 50, 50), spinner)])

    stream = [frame(i) for i in range(40)]
    # pad with identical non-spinner end so settle_ms can elapse on FakeDevice stream end
    stream += [frame(39)] * 20
    dev = FakeDevice(screenshots=stream, width=w, height=h)
    eng = _engine(dev)
    res = eng.wait_stable(
        interval_ms=1, settle_ms=5, timeout_ms=3000, ignore_animation=True
    )
    assert res.ok
    assert "settled" in (res.detail or "").lower()


def test_wait_stable_legacy_mode_times_out_on_animation() -> None:
    """Legacy (whole-frame) mode DOES time out when a spinner keeps flipping."""
    w, h = 100, 100

    def frame(i: int) -> bytes:
        # Moving bar so dHash differs every frame (uniform colour alone won't).
        x = 10 + (i * 17) % 60
        return make_png(w, h, color=(240, 240, 240), boxes=[((x, 20, x + 20, 80), (0, 0, 0))])

    stream = [frame(i) for i in range(80)]
    dev = FakeDevice(screenshots=stream, width=w, height=h)
    eng = _engine(dev)
    with pytest.raises(StabilityTimeout):
        eng.wait_stable(
            interval_ms=1, settle_ms=30, timeout_ms=40, ignore_animation=False
        )


def test_frames_differ_detects_solid_colour_flip() -> None:
    from android_ui_analyser.imaging import frame_signature, frames_differ

    a = ScreenImage(make_png(80, 80, color=(10, 10, 10)), width=80, height=80)
    b = ScreenImage(make_png(80, 80, color=(200, 20, 20)), width=80, height=80)
    assert frames_differ(frame_signature(a), frame_signature(b))
    assert not frames_differ(frame_signature(a), frame_signature(a))


# --------------------------------------------------------------- batch-3 fixes


def test_wait_absent_returns_when_gone() -> None:
    # find_text returns None (nothing in text_index) → the target is already absent.
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev)
    res = eng.wait(for_="Loading", timeout_ms=500, absent=True)
    assert res.ok is True and res.detail == "absent:Loading"


def test_wait_absent_times_out_while_present() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, text_index={"Loading": (0, 0, 50, 50)})
    eng = _engine(dev)
    res = eng.wait(for_="Loading", timeout_ms=300, absent=True)
    assert res.ok is False  # still present → not gone within the timeout


def test_wait_observe_attaches_screen_even_on_miss() -> None:
    # A failed wait still returns the current screen so the agent can diagnose in one call.
    dev = FakeDevice(hierarchy_xml=_XML)  # "Nope" is not in the tree
    eng = _engine(dev)
    res = eng.wait(for_="Nope", timeout_ms=200, observe=True)
    assert res.ok is False and res.observation is not None


def test_app_launch_activity_threads_to_device() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    eng = _engine(dev)
    eng.app("launch", package="com.x", activity=".LaunchActivity")
    assert ("launch_app", ("com.x", ".LaunchActivity")) in dev.calls
    # An explicit --activity is remembered, so the next bare launch reuses that entry rather
    # than falling back to an unpinned resolve (see tests/test_launch_entry.py).
    eng.app("launch", package="com.x")
    assert ("launch_app", ("com.x", "com.x.LaunchActivity")) in dev.calls


def test_bare_launch_stays_unpinned_when_nothing_is_known() -> None:
    # No pin and no declared MAIN/LAUNCHER: the platform still resolves the entry itself, which
    # is the 1-tuple call shape the device layer has always received.
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    eng = _engine(dev)
    eng.app("launch", package="com.x")
    assert ("launch_app", ("com.x",)) in dev.calls


def test_cli_wait_absent_and_app_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    r = runner.invoke(app, ["wait-and-analyze", "--for", "Loading", "--absent", "--timeout", "200"])
    assert r.exit_code == 0, r.stderr  # already absent → ok
    r2 = runner.invoke(app, ["app", "launch", "com.x", "--activity", ".LaunchActivity"])
    assert r2.exit_code == 0, r2.stderr
    assert ("launch_app", ("com.x", ".LaunchActivity")) in dev.calls


# ------------------------------------------------ wait --for honours a leading `!` predicate
#
# `--until`/`await-and-analyze` already understand `!field:value` as "must be absent"
# (`_parse_await_terms`). `wait-and-analyze --for` predates that grammar and instead takes a
# plain string plus separate `--by`/`--absent` flags, so an agent reaching for the syntax it
# already uses elsewhere — `--for '!text:Loading'` — got no error: the bang and the `text:`
# prefix were both swallowed into the literal search needle, so the wait looked for the
# *presence* of a string that could never appear and burned the full timeout even though the
# absence it actually asked for was already true.


def test_wait_for_bang_prefix_is_honoured_when_already_absent() -> None:
    # Nothing in text_index — "Gone for good" is genuinely absent right now, so a correctly
    # parsed `!text:` predicate must return ok=True well inside the timeout.
    dev = FakeDevice(hierarchy_xml=_XML)
    eng = _engine(dev)
    res = eng.wait(for_="!text:Gone for good", timeout_ms=300)
    assert res.ok is True
    assert res.detail == "absent:Gone for good"


def test_wait_for_bang_prefix_times_out_while_still_present() -> None:
    dev = FakeDevice(hierarchy_xml=_XML, text_index={"Gone for good": (0, 0, 50, 50)})
    eng = _engine(dev)
    res = eng.wait(for_="!text:Gone for good", timeout_ms=300)
    assert res.ok is False  # still present → the absence predicate is not satisfied


def test_wait_for_bang_prefix_honours_embedded_rid_field() -> None:
    # The `field:` prefix must also switch `by`, not just strip the `!` — a resource-id term
    # searched as text would coincidentally "pass" for the wrong reason.
    dev = FakeDevice(hierarchy_xml=_XML, resource_index={"x:id/spinner": (0, 0, 10, 10)})
    eng = _engine(dev)
    res = eng.wait(for_="!rid:missingSpinner", timeout_ms=300)
    assert res.ok is True
    assert ("find_text", ("missingSpinner", "MatchMode.contains", False, "rid")) in dev.calls


def test_cli_wait_for_bang_prefix_exits_zero_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev = FakeDevice(hierarchy_xml=_XML, package="com.x")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    r = runner.invoke(
        app, ["wait-and-analyze", "--for", "!text:Gone for good", "--timeout", "200"]
    )
    assert r.exit_code == 0, r.stderr
