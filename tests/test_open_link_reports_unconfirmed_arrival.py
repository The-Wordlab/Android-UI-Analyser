"""`open-and-analyze` reported a clean, unqualified success for a deeplink that did nothing.

Measured 2026-08-19, twice, with two distinct fresh identifiers, device offline:
``aua open-and-analyze "<scheme>://<path>/<never-seen-id>"`` returned ``ok: true`` with
``action_diff_summary: {added: 0, removed: 0, changed: 0}``. Nothing had actually happened — the
screen still showed the PREVIOUS app's content, which the caller then read as being about the id
it had just requested. The existing skill guidance already told agents to "trust it only if the
returned screen changed" — an admission the command could not tell them so itself.

The engine already computes, and can say with certainty, whether the post-deeplink screen is
BYTE-IDENTICAL to the pre-deeplink one (`_change_summary` — same activity, same node count, no
text added/removed, focus unmoved). What it never did was attach that certainty to the one field
built for exactly this: `ActionResult.verified` — "True = confirmed effect, False = confirmed no
effect, None = genuinely could not tell" (see its docstring, and `keyboard_visible`'s: "'cannot
tell' must not read as 'hidden'"). `ok` stays `True` in every case here: `am start` really did
deliver the intent, and an app that legitimately has nothing to do in response (already on the
target screen, or a fire-and-forget action) is not an aua failure — only `verified` should carry
the news that no destination change was observed.

Deliberately does NOT try to guess "silently failed" vs "legitimately already there" — the two
are mechanically indistinguishable from a before/after screen diff alone, and asserting one over
the other would be exactly the false confidence this project keeps removing. Both get the same
honest `verified: false`; only a hard failure (`ok: false`) would be crying wolf on the
legitimate case, so `ok` is left alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PKG = "com.example.app"
HOME = (
    f'<hierarchy rotation="0"><node class="android.widget.TextView" package="{PKG}" '
    'text="Home" clickable="false" enabled="true" bounds="[0,0][1080,100]"/></hierarchy>'
)
WIDGET = (
    f'<hierarchy rotation="0"><node class="android.widget.TextView" package="{PKG}" '
    'text="Widget detail" clickable="false" enabled="true" bounds="[0,0][1080,100]"/>'
    f'<node class="android.widget.TextView" package="{PKG}" text="left-handed smoke shifter" '
    'clickable="false" enabled="true" bounds="[0,100][1080,200]"/></hierarchy>'
)


class ScriptedDeeplinkDevice(FakeDevice):
    """Serves whatever `_screen` currently is; a deeplink can change it or leave it alone."""

    def __init__(self, *, screen: str = HOME, activity: str = ".MainActivity") -> None:
        super().__init__(hierarchy_xml=screen, package=PKG)
        self._screen = screen
        self._activity = activity
        self.opened: list[str] = []

    def current_app(self) -> dict[str, str]:
        return {"package": PKG, "activity": self._activity}

    def dump_hierarchy(self, *a: Any, **k: Any) -> str:
        return self._screen

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        self.opened.append(uri)
        # Base behaviour: nothing happens. Subclasses/tests override to model real navigation.


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


# --------------------------------------------------------------------------- the reported bug


def test_a_confirmed_no_op_deeplink_no_longer_reads_as_a_clean_unqualified_success(
    tmp_path: Path,
) -> None:
    """The exact repro: fresh/unresolvable id, offline app, device does not move at all."""
    device = ScriptedDeeplinkDevice()
    eng = _engine(tmp_path, device)
    eng.analyze(source="hierarchy")  # a real flow always has a screen in view first

    out = eng.open_link("example-app://widgets/never-seen-id-999", observe=True)

    assert out.ok is True, "the intent WAS delivered — `am start` succeeded, that part is true"
    assert out.change is not None and out.change["changed"] is False
    assert out.action_diff_summary == {
        "added": 0,
        "removed": 0,
        "changed": 0,
        "prev_count": 1,
        "curr_count": 1,
    }, "reproduces the exact field the report says looked like success"
    # This is the fix: a caller checking only `ok` and `action_diff_summary` — exactly what the
    # report says was misleading — now has a THIRD, impossible-to-miss field confirming no
    # destination change was observed, not just a prose hint it has to know to go looking for.
    assert out.verified is False, (
        "a confirmed no-op deeplink must say so through the structured `verified` field, not "
        "only through `stale_risk` prose a caller has to already know to check"
    )
    assert out.stale_risk, "the prose explanation must still be there too"


def test_no_op_deeplink_stale_risk_names_the_ambiguity_rather_than_asserting_failure(
    tmp_path: Path,
) -> None:
    """The message must not claim the deeplink failed — it may have legitimately no-opped."""
    device = ScriptedDeeplinkDevice()
    eng = _engine(tmp_path, device)
    eng.analyze(source="hierarchy")

    out = eng.open_link("example-app://widgets/never-seen-id-999", observe=True)

    assert out.stale_risk is not None
    lowered = out.stale_risk.lower()
    assert "delivered" in lowered  # honest about what IS known
    assert "fail" not in lowered, "must not assert failure — that is not something aua can prove"


# --------------------------------------------------------------------------- must not cry wolf


def test_a_deeplink_that_really_navigates_is_reported_as_verified_true(tmp_path: Path) -> None:
    """A genuine landing must not be lumped in with the ambiguous/no-op case."""

    class Navigating(ScriptedDeeplinkDevice):
        def open_link(self, uri: str, *, package: str | None = None) -> None:
            super().open_link(uri, package=package)
            self._screen = WIDGET
            self._activity = ".WidgetDetailActivity"

    device = Navigating()
    eng = _engine(tmp_path, device)
    eng.analyze(source="hierarchy")

    out = eng.open_link("example-app://widgets/42", observe=True)

    assert out.ok is True
    assert out.change is not None and out.change["changed"] is True
    assert out.verified is True, "a confirmed real destination change must not be left ambiguous"
    # A confirmed landing must never carry the deeplink-specific "did not move" caveat — that
    # is a distinct question from the generic settle-timing caveat other actions can also
    # carry, which this test does not exercise.
    assert not out.stale_risk or "did not move" not in out.stale_risk


def test_repeat_firing_a_deeplink_already_satisfied_is_not_reported_as_a_failure(
    tmp_path: Path,
) -> None:
    """Firing the same deeplink again while already on its target is legitimate, not broken.

    This is mechanically identical, at the screen-diff level, to the silent-failure case
    above — aua cannot and does not try to tell them apart. What it must not do is call this
    an error: `ok` stays true and no exception is raised, which is the whole "do not cry wolf"
    contract. `verified: false` still applies (nothing observably changed, which is also true
    here), but that is honest information, not an assertion that anything went wrong.
    """

    class Navigating(ScriptedDeeplinkDevice):
        def open_link(self, uri: str, *, package: str | None = None) -> None:
            super().open_link(uri, package=package)
            self._screen = WIDGET
            self._activity = ".WidgetDetailActivity"

    device = Navigating()
    eng = _engine(tmp_path, device)
    eng.analyze(source="hierarchy")
    first = eng.open_link("example-app://widgets/42", observe=True)
    assert first.verified is True

    # Fire it again — already there, nothing left to do.
    second = eng.open_link("example-app://widgets/42", observe=True)

    assert second.ok is True, "landing where you already are must never be reported as a failure"
    assert second.verified is False, "honest: no NEW destination change was observed this time"


def test_a_genuinely_unknown_baseline_stays_ambiguous_rather_than_forced_false(
    tmp_path: Path,
) -> None:
    """No prior analyze means aua cannot know what the screen looked like before — say so."""
    device = ScriptedDeeplinkDevice()
    eng = _engine(tmp_path, device)
    # Deliberately no `eng.analyze(...)` first — cold engine, nothing cached to diff against.

    out = eng.open_link("example-app://widgets/never-seen-id-999", observe=True)

    assert out.change is not None and out.change["changed"] is not False
    assert out.verified is not False, (
        "'I could not compare' must not be reported the same way as 'confirmed no effect' — "
        "collapsing them is the same false-certainty bug this project keeps removing"
    )
