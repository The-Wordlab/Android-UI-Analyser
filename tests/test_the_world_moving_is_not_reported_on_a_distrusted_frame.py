"""`screen_moved` must stay silent when the tool itself said the held frame would change.

A non-settled arrival verdict tells the caller "content may replace these ids when it lands".
When that replacement then happens before the caller's next command, it is the predicted
consequence of the caller's *own* action — not the world moving on its own. Warning
"nothing you sent caused that" would be a false attribution stacked on a screen the tool had
already told the caller not to hold. This is the sixth silence of `_screen_moved_verdict`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

PKG = "com.example.fiction"


def _screen(*rows: str) -> str:
    return '<hierarchy rotation="0">' + "".join(rows) + "</hierarchy>"


def _node(text: str, *, rid: str, y: int, clickable: bool = False) -> str:
    return (
        f'<node class="android.view.View" package="{PKG}" text="{text}"'
        f' resource-id="{rid}" clickable="{str(clickable).lower()}" enabled="true"'
        f' bounds="[20,{y}][500,{y + 80}]"/>'
    )


BEFORE = _screen(
    _node("Continue with MegaID", rid=f"{PKG}:id/loginBtn", y=120, clickable=True),
)
AFTER = _screen(
    _node("Add account", rid=f"{PKG}:id/addAccount", y=120, clickable=True),
)


def test_a_distrusted_frame_being_replaced_is_not_the_world_moving(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dev = FakeDevice(hierarchy_xml=BEFORE, package=PKG)
    cfg = make_config(memory={"dir": str(tmp_path / "m")}, daemon={"enabled": False})
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    shown = eng.analyze(source="hierarchy")
    dev._xml = AFTER
    fresh = eng.analyze(source="hierarchy", no_cache=True)
    monkeypatch.setattr(
        eng,
        "_caller_turn_facts",
        lambda: SimpleNamespace(
            previous_fingerprint=shown.meta.fingerprint, previous_age_ms=1200, profile=None
        ),
    )

    # Control: an ordinary replacement with the caller holding a trusted frame does warn.
    assert eng._screen_moved_verdict(shown, fresh) is not None

    # The same replacement on a frame the tool already flagged is the *predicted* outcome of
    # the caller's own action; attributing it to the world would be false.
    shown.meta.stale_risk = "the destination has rendered nothing yet"
    assert eng._screen_moved_verdict(shown, fresh) is None
