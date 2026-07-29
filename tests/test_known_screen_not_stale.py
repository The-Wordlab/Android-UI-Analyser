"""`known_screen` must name the screen in front of you, not the last one.

With `perf.async_memory` on (the default), `_record_screen_safe` handed the map write to a
background thread and returned `self._last_known_screen` — a value the thread had not updated
yet, i.e. the PREVIOUS screen's name. Observed on device: the device launcher and a system ANR
dialog were both reported under names belonging to the app under test's own map, and a caller
that had just navigated back was told it was still on the screen it had left — the one answer
that makes "where am I?" unanswerable.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config
from test_memory import P as PKG
from test_memory import _hier, _node


def _screen(rows: list[tuple[str, str]]) -> str:
    return _hier(
        *(
            _node(
                "android.widget.Button",
                text=text,
                rid=f"x:id/{rid}",
                clk=True,
                b=f"[40,{300 + i * 140}][1040,{400 + i * 140}]",
            )
            for i, (text, rid) in enumerate(rows)
        )
    )


HOME = _screen(
    [
        ("Inbox", "inboxRow"),
        ("Starred", "starredRow"),
        ("Drafts", "draftsRow"),
        ("Archive", "archiveRow"),
        ("Compose", "composeFab"),
    ]
)

DETAIL = _screen(
    [
        ("Playback speed", "speedSetting"),
        ("Download quality", "qualitySetting"),
        ("Subtitles", "subtitleSetting"),
        ("Autoplay", "autoplaySetting"),
        ("Clear watch history", "clearHistoryButton"),
        ("Sign out", "signOutButton"),
    ]
)


class SwitchableDevice(FakeDevice):
    """A device whose screen the test changes directly, with no interaction in between."""

    def show(self, xml: str) -> None:
        self._xml = xml


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(
        daemon={"enabled": False},
        memory={"enabled": True, "auto_record": True, "dir": str(tmp_path)},
        perf={"async_memory": True, "skip_unchanged_memory": False},
    )
    return Engine(cfg, device=device)


def _name_now(eng: Engine) -> str | None:
    """The reported screen name, after the async map write has landed."""
    res = eng.analyze()
    if eng._mem_thread is not None:
        eng._mem_thread.join(timeout=5)
    return res.meta.known_screen


def test_switching_screens_does_not_report_the_previous_name(tmp_path: Path) -> None:
    dev = SwitchableDevice(hierarchy_xml=HOME, package=PKG, serial="emu-known")
    eng = _engine(tmp_path, dev)
    _name_now(eng)  # first pass records HOME
    home = _name_now(eng)
    assert home, "HOME should be mapped and recognised by the second analyze"

    dev.show(DETAIL)
    assert _name_now(eng) != home, "reported the screen we just left"


def test_returning_to_a_mapped_screen_reports_that_screen(tmp_path: Path) -> None:
    """The other direction: coming back must not keep naming the screen we came from."""
    dev = SwitchableDevice(hierarchy_xml=HOME, package=PKG, serial="emu-return")
    eng = _engine(tmp_path, dev)
    _name_now(eng)
    home = _name_now(eng)

    dev.show(DETAIL)
    _name_now(eng)
    detail = _name_now(eng)
    assert detail and detail != home, "DETAIL should get its own name"

    dev.show(HOME)
    assert _name_now(eng) == home, "back on HOME, so HOME's name — not DETAIL's"


def test_a_reported_name_exists_in_the_map(tmp_path: Path) -> None:
    """A name that is not in the package's map cannot describe the current screen."""
    dev = SwitchableDevice(hierarchy_xml=HOME, package=PKG, serial="emu-inmap")
    eng = _engine(tmp_path, dev)
    _name_now(eng)
    dev.show(DETAIL)
    name = _name_now(eng)
    if name is not None:
        app = eng._memory.load(PKG)
        assert app is not None and name in app.screens, f"{name!r} is not a screen of {PKG}"
