"""Cross-package transit (PRD §6b): an external-auth leg becomes ONE replayable edge.

The synthetic fixture covers tapping "Continue with Example ID" to open a Chrome custom tab
(different package). Route edges must still be recorded across the auth leg and `goto
onboarding` was `route_unknown` forever. Now the journey cursor stays on the origin app
through `memory.transit_packages`; the return records one edge whose steps carry their
packages, `goto` replays it end-to-end (auto-source analyzes mid-transit), a redacted
account row hands off cleanly, and a re-run resumes mid-transit from the first step that
matches the current screen.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import _package_from_xml
from android_ui_analyser.memory import RouteStep
from test_memory import P, _elements, _engine, _hier, _node, _store
from test_navigation import ScriptedDevice

CHROME = "com.android.chrome"

LOGIN = _hier(
    _node("android.widget.TextView", text="Welcome", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Continue with Example ID",
        rid="x:id/exampleProvider",
        clk=True,
        b="[40,300][1040,400]",
    ),
    _node(
        "android.widget.Button",
        text="Use demo account",
        rid="x:id/demoAccount",
        clk=True,
        b="[40,440][1040,540]",
    ),
)
CHROME_PICKER = _hier(
    _node(
        "android.widget.TextView",
        text="Choose an account",
        b="[40,120][1040,210]",
        pkg=CHROME,
    ),
    _node("android.view.View", text="Demo Account", clk=True, b="[40,300][1040,400]", pkg=CHROME),
    _node(
        "android.view.View",
        text="Use another account",
        clk=True,
        b="[40,440][1040,540]",
        pkg=CHROME,
    ),
)
CHROME_CONSENT = _hier(
    _node("android.widget.TextView", text="Signing back in", b="[40,120][1040,210]", pkg=CHROME),
    _node("android.view.View", text="Cancel", clk=True, b="[40,300][500,400]", pkg=CHROME),
    _node("android.view.View", text="Continue", clk=True, b="[540,300][1040,400]", pkg=CHROME),
)
ONBOARDING = _hier(
    _node("android.widget.TextView", text="Onboarding", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Get started",
        rid="x:id/start",
        clk=True,
        b="[40,300][1040,400]",
    ),
)
OTHER_APP = _hier(
    _node(
        "android.widget.TextView",
        text="Calculator",
        b="[40,120][1040,210]",
        pkg="com.other.calc",
    ),
    _node(
        "android.widget.Button",
        text="Equals",
        clk=True,
        b="[40,300][400,400]",
        pkg="com.other.calc",
    ),
)


class TransitScriptedDevice(ScriptedDevice):
    """ScriptedDevice whose foreground package tracks the current screen's XML."""

    def current_app(self) -> dict[str, str]:
        return {"package": _package_from_xml(self._xml) or self._pkg, "activity": self._act}


def _walk_auth_leg(tmp_path: Path, serial: str):
    """Drive login → chrome picker → chrome consent → onboarding with plain calls."""
    dev = TransitScriptedDevice(
        [LOGIN, CHROME_PICKER, CHROME_CONSENT, ONBOARDING], package=P, serial=serial
    )
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")  # login (origin cursor seeds here)
    provider_id = next(e.id for e in res.elements if e.text == "Continue with Example ID")
    eng.tap(provider_id, observe=False)
    res = eng.analyze(source="hierarchy")  # chrome picker (transit)
    row = next(e.id for e in res.elements if e.text == "Demo Account")
    eng.tap(row, observe=False)
    res = eng.analyze(source="hierarchy")  # chrome consent (transit)
    cont = next(e.id for e in res.elements if e.text == "Continue")
    eng.tap(cont, observe=False)
    eng.analyze(source="hierarchy")  # back in the origin app: onboarding
    return dev, eng


# --------------------------------------------------------------- passive learning


def test_auth_leg_records_one_replayable_edge(tmp_path: Path) -> None:
    """The user's exact scenario: the whole excursion becomes ONE origin-app edge."""
    _, eng = _walk_auth_leg(tmp_path, "emu-transit")
    store = _store(tmp_path)

    origin = store.load(P)
    assert origin is not None
    assert len(origin.routes) == 1
    edge = origin.routes[0]
    assert edge.from_screen == "welcome" and edge.to_screen == "onboarding"
    assert [s.kind for s in edge.steps] == ["tap", "tap", "tap"]
    assert edge.steps[0].package is None  # origin step normalized
    assert edge.steps[1].package == CHROME and edge.steps[2].package == CHROME
    assert "⇢ (via com.android.chrome)" in edge.action

    chrome = store.load(CHROME)
    assert chrome is not None and chrome.screens  # transit screens keep their own map
    assert chrome.routes == []  # but no journey edges of their own

    sess = store.load_session("emu-transit")
    assert sess.package == P and sess.current_screen == "onboarding"
    assert sess.pending == []


def test_transit_to_foreign_app_resets_cleanly(tmp_path: Path) -> None:
    dev = TransitScriptedDevice([LOGIN, OTHER_APP], package=P, serial="emu-foreign")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)  # lands in a NON-transit app
    eng.analyze(source="hierarchy")
    store = _store(tmp_path)
    assert store.load(P).routes == []  # no cross-app edge, pending dropped
    sess = store.load_session("emu-foreign")
    assert sess.package == "com.other.calc" and sess.pending == []


# --------------------------------------------------------------- goto transit replay


def test_goto_replays_the_auth_leg_end_to_end(tmp_path: Path) -> None:
    _walk_auth_leg(tmp_path, "emu-learn")  # teach the map once

    dev = TransitScriptedDevice(
        [LOGIN, CHROME_PICKER, CHROME_CONSENT, ONBOARDING], package=P, serial="emu-replay"
    )
    eng = _engine(tmp_path, dev)
    refused = eng.goto("onboarding")
    assert refused["code"] == "unsafe_route"
    assert refused["required_opt_in"] == ["--allow-unsafe"]
    assert not any(call[0] == "click" for call in dev.calls)

    out = eng.goto("onboarding", allow_unsafe=True)
    assert out["ok"] is True and out["arrived"] is True, out
    assert sum(1 for c in dev.calls if c[0] == "click") == 3  # the whole boring leg
    assert out["final_screen"] == "onboarding"


def test_goto_hands_off_on_redacted_transit_step(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(LOGIN), name_hint="welcome")
    store.record_screen(package=P, elements=_elements(ONBOARDING), name_hint="onboarding")
    store.record_route(
        P,
        "welcome",
        "onboarding",
        steps=[
            RouteStep(kind="tap", label="Continue with Example ID", resource_id="exampleProvider"),
            RouteStep(kind="tap", label="<redacted>", package=CHROME),
            RouteStep(kind="tap", label="Continue", package=CHROME),
        ],
    )
    dev = TransitScriptedDevice(
        [LOGIN, CHROME_PICKER, CHROME_CONSENT, ONBOARDING], package=P, serial="emu-redact"
    )
    eng = _engine(tmp_path, dev)
    out = eng.goto("onboarding", allow_unsafe=True)
    assert out["ok"] is False and out["code"] == "element_not_found"
    assert out["step"]["label"] == "<redacted>"  # the identity-bearing tap is manual
    assert out["expected_package"] == CHROME
    assert len(out["remaining_steps"]) == 2
    assert sum(1 for c in dev.calls if c[0] == "click") == 1  # only the first hop ran


def test_goto_resumes_mid_transit_after_manual_step(tmp_path: Path) -> None:
    """After the handoff, one manual tap + a re-run finishes the journey."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(LOGIN), name_hint="welcome")
    store.record_screen(package=P, elements=_elements(ONBOARDING), name_hint="onboarding")
    store.record_route(
        P,
        "welcome",
        "onboarding",
        steps=[
            RouteStep(kind="tap", label="Continue with Example ID", resource_id="exampleProvider"),
            RouteStep(kind="tap", label="<redacted>", package=CHROME),
            RouteStep(kind="tap", label="Continue", package=CHROME),
        ],
    )
    dev = TransitScriptedDevice(
        [LOGIN, CHROME_PICKER, CHROME_CONSENT, ONBOARDING], package=P, serial="emu-resume"
    )
    eng = _engine(tmp_path, dev)
    first = eng.goto("onboarding", allow_unsafe=True)
    assert first["ok"] is False  # stops at the redacted account row (on CHROME_PICKER)

    res = eng.analyze(source="hierarchy")  # the agent's one manual step
    row = next(e.id for e in res.elements if e.text == "Demo Account")
    eng.tap(row, observe=False)  # → CHROME_CONSENT

    # Foreground=chrome, journey=origin app → resume.
    resumed = eng.goto("onboarding", allow_unsafe=True)
    assert resumed["ok"] is True and resumed["arrived"] is True, resumed
    # resume matched 'Continue' on the consent screen and only tapped that
    clicks_total = sum(1 for c in dev.calls if c[0] == "click")
    assert clicks_total == 3  # goto#1 (google) + manual (row) + resume (continue)


def test_goto_resume_hands_off_when_nothing_matches(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(LOGIN), name_hint="welcome")
    store.record_screen(package=P, elements=_elements(ONBOARDING), name_hint="onboarding")
    store.record_route(
        P,
        "welcome",
        "onboarding",
        steps=[RouteStep(kind="tap", label="No Such Thing", package=CHROME)],
    )
    # Start ALREADY mid-transit: session says the journey is the origin app's, screen is chrome.
    sess = store.load_session("emu-stuck")
    sess.package = P
    sess.current_screen = "welcome"
    store.save_session("emu-stuck", sess)
    dev = TransitScriptedDevice([CHROME_PICKER], package=CHROME, serial="emu-stuck")
    eng = _engine(tmp_path, dev)
    out = eng.goto("onboarding", allow_unsafe=True)
    assert out["ok"] is False and out["code"] == "element_not_found"
    assert "manually" in out["hint"] and out["remaining_steps"]
    assert not any(c[0] == "click" for c in dev.calls)


def test_transitional_same_screen_frame_does_not_eat_the_edge(tmp_path: Path) -> None:
    """Auth returns often flash a frame recognised as the screen we LEFT — pending must
    survive it so the eventual different-screen analyze still records the transit edge."""
    dev = TransitScriptedDevice(
        [LOGIN, CHROME_PICKER, LOGIN, ONBOARDING], package=P, serial="emu-flash"
    )
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")  # login
    provider_id = next(e.id for e in res.elements if e.text == "Continue with Example ID")
    eng.tap(provider_id, observe=False)
    eng.analyze(source="hierarchy")  # chrome picker (transit, pending kept)
    res = eng.analyze(source="hierarchy")  # ScriptedDevice held → still picker; harmless
    row = next(e.id for e in res.elements if e.text == "Demo Account")
    eng.tap(row, observe=False)  # → transitional LOGIN frame
    eng.analyze(source="hierarchy")  # recognised as "welcome" (same as cursor) → KEEP pending
    eng.key("enter", observe=False)  # any action; ScriptedDevice advances → ONBOARDING
    eng.analyze(source="hierarchy")
    store = _store(tmp_path)
    origin = store.load(P)
    edge = next(e for e in origin.routes if e.from_screen == "welcome")
    assert edge.to_screen != "welcome"
    labels = [s.label for s in edge.steps]
    assert "Continue with Example ID" in labels  # survived the transitional same-screen frame
    assert "Demo Account" in labels
    assert edge.steps[-1].kind == "key"
