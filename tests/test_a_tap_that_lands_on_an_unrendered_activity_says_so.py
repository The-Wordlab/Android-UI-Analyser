"""A tap that starts an Activity which has not rendered yet must say so, not claim arrival.

Journalled 2026-08-22, reproduced here with fictional names: a tap on a login button started a
new Activity that then waited on the network before drawing anything. The settle wait truthfully
reported quiet pixels and a stable tree (``via=hierarchy``, 199ms), the change summary said the
Activity moved while the node count stayed identical (43 -> 43) and nothing was added — and the
result still claimed full confidence (``stale_risk: None``). The agent picked its next target
from that transitional frame, tapped a control that no longer existed, and had to spend a
recovery ``analyze`` — six calls to do the work of three.

The screen was *physically settled and semantically loading*. No settle-loop tuning can fix
that; classification can. Two facts the engine already computes are the verdict:

* departure without arrival — the Activity changed while nothing was added. Removal-only
  change proves the old screen was left, not that the new one has rendered.
* an explicit loading indicator (progress widget / loading text / mapped loading screen) is
  loading whatever the settle wait said about pixels.

A truthful "this screen has not rendered yet" is a success of the observation contract, not a
failure of the action: ``ok`` stays true, and the honesty lives in ``stale_risk`` + ``note``.
``next_actions`` withholding is best-effort — the field is being independently slimmed/gated,
so these tests only assert it is not *offered*, never that it exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PKG = "com.example.fiction"

# The journalled settle shape, verbatim: a double-sampled hierarchy exit that skips the
# extended confirmation window and used to clear every caveat.
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
    _node(y=320),  # unlabelled wrapper, present on both screens
)

# The new Activity is foreground but has drawn nothing: the tree still shows the old content
# minus the tapped control (its node lost the label), so the node count is unchanged, one text
# is removed, and nothing is added — the exact journalled change shape.
AUTH_BLANK = _screen(
    _node("Welcome back", rid=f"{PKG}:id/entryTitle", y=20),
    _node(y=120),  # the login control's label is gone; nothing replaced it
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)

# The same navigation, but the destination genuinely rendered: new labels, new control.
AUTH_READY = _screen(
    _node("Choose an account", rid=f"{PKG}:id/accountTitle", y=20),
    _node("Add account", rid=f"{PKG}:id/addAccount", y=120, clickable=True),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)

# A destination that landed with content *and* an explicit progress widget: the indicator
# outranks the additive change — content may still be replaced when the request completes.
AUTH_SPINNER = _screen(
    _node("Signing in", rid=f"{PKG}:id/authStatus", y=20),
    _node(y=120, cls="android.widget.ProgressBar"),
    _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
    _node(y=320),
)


class NetworkBoundAuth(FakeDevice):
    """Tapping login fronts a new Activity whose first frame is *after_xml*."""

    def __init__(self, after_xml: str) -> None:
        super().__init__(hierarchy_xml=ENTRY, package=PKG, activity=".launch.EntryActivity")
        self._after = after_xml

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._xml = self._after
        self._act = ".auth.AuthActivity"


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "memory")},
        daemon={"enabled": False},
        perception={"observe_escalates_to_vision": False},
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _tap_login(tmp_path: Path, monkeypatch: Any, after_xml: str) -> Any:
    dev = NetworkBoundAuth(after_xml)
    eng = _engine(tmp_path, dev)
    login = next(
        e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Continue with MegaID"
    )
    monkeypatch.setattr(eng, "_await_post_action_ready", lambda **_: dict(JOURNALLED_SETTLE))
    return eng.tap(login, observe=True)


def test_departure_without_arrival_is_not_reported_as_arrival(
    tmp_path: Path, monkeypatch: Any
) -> None:
    out = _tap_login(tmp_path, monkeypatch, AUTH_BLANK)

    assert out.ok and out.observation is not None
    change = out.change or {}
    # Preconditions: the fixture reproduces the journalled change shape, or this test is
    # measuring something else.
    assert change.get("activity_changed") is True, f"fixture drifted: {change}"
    assert change.get("text_added") == [], f"fixture drifted: {change}"
    assert change.get("node_count_delta") == 0, f"fixture drifted: {change}"
    assert "Continue with MegaID" in (change.get("text_removed") or [])
    assert out.settle is not None and out.settle.get("via") == "hierarchy"

    # The verdict: this frame must not be offered as a trustworthy destination.
    assert out.stale_risk, (
        "activity changed with nothing added is departure evidence, not arrival evidence — "
        f"the result must say so. change={change}"
    )
    assert out.observation.meta.stale_risk == out.stale_risk
    assert "rendered nothing" in out.stale_risk
    assert not out.next_actions, "ids from an unrendered destination must not be advertised"
    assert out.note is not None and "wait-and-analyze" in out.note


def test_an_explicit_loading_indicator_outranks_an_additive_change(
    tmp_path: Path, monkeypatch: Any
) -> None:
    out = _tap_login(tmp_path, monkeypatch, AUTH_SPINNER)

    assert out.ok and out.observation is not None
    change = out.change or {}
    assert change.get("text_added"), f"fixture drifted: the spinner screen adds a label {change}"
    assert out.stale_risk and "loading" in out.stale_risk.lower()
    assert not out.next_actions


def test_a_destination_that_rendered_content_carries_no_caveat(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The hot path must stay clean: an additive arrival is settled, not suspected."""
    out = _tap_login(tmp_path, monkeypatch, AUTH_READY)

    assert out.ok and out.observation is not None
    change = out.change or {}
    assert change.get("activity_changed") is True
    assert change.get("text_added"), f"fixture drifted: {change}"
    assert out.stale_risk is None, f"a rendered destination is not a stale risk: {out.stale_risk}"
    assert out.next_actions, "a settled destination keeps advertising what to do next"


def test_recognition_of_a_known_destination_suppresses_the_departure_verdict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Map recognition beats the inference — but only with a *known origin* to differ from.

    Found while building this fix: ``destination_confirmed`` is "recognized name differs from
    the before name", and on a cold session (async memory, first action after the first
    analyze) the before name is still unstamped — so recognizing *anything* counts as a
    confirmed destination. Under this detector's own preconditions the after tree barely
    differs from the before tree, so that recognition is at least as likely to be the origin's
    own map entry. Recognition without a known origin must not suppress the verdict.
    """
    dev = NetworkBoundAuth(AUTH_BLANK)
    eng = _engine(tmp_path, dev)
    obs = eng.analyze(source="hierarchy")
    departure_change = {
        "activity_changed": True,
        "text_added": [],
        "text_removed": ["Continue with MegaID"],
        "node_count_delta": 0,
    }

    known_origin = {"known_screen": "entry", "rids": []}
    assert (
        eng._unready_destination_risk(
            departure_change, obs, before_state=known_origin, destination_confirmed=True
        )
        is None
    ), "a recognized navigation away from a known screen is arrival evidence"
    assert (
        eng._unready_destination_risk(
            departure_change, obs, before_state=None, destination_confirmed=True
        )
        is not None
    ), "recognition with no known origin has no differential power"
    assert (
        eng._unready_destination_risk(
            departure_change, obs, before_state=known_origin, destination_confirmed=False
        )
        is not None
    )


def test_a_new_unlabelled_actionable_control_counts_as_arrival(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Labels alone miss icon buttons: a clickable with a *new* resource id proves content."""
    icon_button = _screen(
        _node("Welcome back", rid=f"{PKG}:id/entryTitle", y=20),
        _node(rid=f"{PKG}:id/nextArrow", y=120, clickable=True),  # unlabelled, new rid
        _node("Terms apply", rid=f"{PKG}:id/terms", y=220),
        _node(y=320),
    )
    out = _tap_login(tmp_path, monkeypatch, icon_button)

    change = out.change or {}
    assert change.get("activity_changed") is True and change.get("text_added") == []
    assert out.stale_risk is None, f"a new actionable control is arrival evidence: {out.stale_risk}"
