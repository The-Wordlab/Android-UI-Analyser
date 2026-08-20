"""A tap must not be aimed under the system navigation bar.

11 controls across 6 apps in one sweep published accessibility bounds extending below the
nav bar window, which starts at y=1184 on that pool. A touch at the element's centre then
did one of two things, neither of them the caller's intent:

- delivered to **Home**, backgrounding the app under test and losing unsaved input
  (bounds `[36,1164,684,1252]`, centre y=1208);
- swallowed while the app stayed foregrounded, confirmed via `dumpsys mCurrentFocus`
  (bounds `[32,1144,688,1248]`, centre y=1196).

Both reported `ok:true`, so "the app is still in the foreground" was not evidence the touch
landed, and a run could lose its journey and then report that the *product* ignored it. The
measured workaround was ~30px above the reported centre; the fix aims at the middle of
whatever part of the element is above the bar, which is that, derived rather than guessed.

The geometry below is the real geometry from those reports. Half of these tests exist to pin
the cases that must **not** move, because `tap` is the hottest path in the tool and a clamp
that fires when it should not is worse than the bug it fixes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ElementNotFoundError
from conftest import FakeDevice, make_config

# The pool's geometry: 720x1280, three-button nav bar occupying y 1184-1280.
_W, _H = 720, 1280
_BAR_TOP = 1184


def _screen(*nodes: str, nav_bar: bool = True) -> str:
    bar = (
        f"""
  <node index="90" package="com.android.systemui" class="android.widget.ImageView"
        content-desc="Back" resource-id="com.android.systemui:id/back"
        clickable="true" bounds="[120,{_BAR_TOP}][240,{_H}]"/>
  <node index="91" package="com.android.systemui" class="android.widget.ImageView"
        content-desc="Home" resource-id="com.android.systemui:id/home"
        clickable="true" bounds="[300,{_BAR_TOP}][420,{_H}]"/>
"""
        if nav_bar
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<hierarchy rotation="0">\n'
        + "".join(nodes)
        + bar
        + "</hierarchy>"
    )


def _button(text: str, bounds: str, index: int = 0) -> str:
    return (
        f'  <node index="{index}" package="com.example.app" class="android.widget.Button"\n'
        f'        text="{text}" resource-id="com.example.app:id/b{index}"\n'
        f'        clickable="true" enabled="true" bounds="{bounds}"/>\n'
    )


def _engine(xml: str, tmp_path: Path) -> tuple[Engine, FakeDevice]:
    dev = FakeDevice(hierarchy_xml=xml, width=_W, height=_H, serial="emu-navbar")
    eng = Engine(make_config(memory={"dir": str(tmp_path / "mem")}), device=dev)
    eng.analyze()
    return eng, dev


def _clicked(dev: FakeDevice) -> tuple[int, int]:
    clicks = [args for name, args in dev.calls if name == "click"]
    assert clicks, "no click was delivered"
    return clicks[-1]


def _find(eng: Engine, text: str) -> int:
    el = next(e for e in eng.analyze().elements if e.text == text)
    return el.id


# --------------------------------------------------- the reported cases, real geometry


def test_a_control_overlapping_the_bar_is_tapped_above_it(tmp_path: Path) -> None:
    """The Home-capture case: centre y=1208 is inside the bar, so aim into the visible strip."""
    eng, dev = _engine(_screen(_button("Save period", "[36,1164][684,1252]")), tmp_path)
    out = eng.tap(_find(eng, "Save period"), observe=False)

    x, y = _clicked(dev)
    assert out.ok is True
    assert y < _BAR_TOP, f"aimed at y={y}, which is inside the navigation bar"
    assert y >= 1164, "aimed above the element itself"
    assert x == 360, "the horizontal aim must not move"
    assert out.target == [x, y], "the reported target must be where the touch went"


def test_the_silently_swallowed_case_is_also_lifted(tmp_path: Path) -> None:
    """The second reported shape: centre y=1196, app stayed foregrounded, nothing happened."""
    eng, dev = _engine(_screen(_button("New story", "[32,1144][688,1248]")), tmp_path)
    eng.tap(_find(eng, "New story"), observe=False)

    _x, y = _clicked(dev)
    assert y < _BAR_TOP
    assert y >= 1144


def test_an_element_wholly_under_the_bar_refuses_instead_of_hitting_home(
    tmp_path: Path,
) -> None:
    """There is no honest aim point, and a tap here backgrounds the app under test.

    An error the caller can act on beats a success that corrupted the run's state.
    """
    eng, dev = _engine(_screen(_button("Buried", "[40,1200][680,1270]")), tmp_path)
    target = _find(eng, "Buried")
    before = len([1 for name, _ in dev.calls if name == "click"])

    with pytest.raises(ElementNotFoundError, match="under the system navigation bar"):
        eng.tap(target, observe=False)

    assert len([1 for name, _ in dev.calls if name == "click"]) == before, "it touched anyway"


def test_long_press_and_input_are_lifted_too(tmp_path: Path) -> None:
    """Same occlusion, same consequence — a click to focus a field lands on Home as well."""
    eng, dev = _engine(_screen(_button("Save period", "[36,1164][684,1252]")), tmp_path)
    eng.long_press(_find(eng, "Save period"), observe=False)
    presses = [args for name, args in dev.calls if name == "long_click"]
    assert presses and presses[-1][1] < _BAR_TOP

    eng2, dev2 = _engine(_screen(_button("Save period", "[36,1164][684,1252]")), tmp_path)
    eng2.input_text(_find(eng2, "Save period"), "hello", observe=False)
    assert _clicked(dev2)[1] < _BAR_TOP


# ------------------------------------------------------- what must NOT move (the risk half)


def test_an_ordinary_control_is_untouched(tmp_path: Path) -> None:
    """The hot path: nothing near the bar must be adjusted at all."""
    eng, dev = _engine(_screen(_button("Continue", "[40,600][680,700]")), tmp_path)
    eng.tap(_find(eng, "Continue"), observe=False)
    assert _clicked(dev) == (360, 650)


def test_a_control_that_merely_sits_low_is_untouched(tmp_path: Path) -> None:
    """Above the bar but in the bottom band — its centre is legitimate, so leave it alone."""
    eng, dev = _engine(_screen(_button("Low", "[40,1080][680,1170]")), tmp_path)
    eng.tap(_find(eng, "Low"), observe=False)
    assert _clicked(dev) == (360, 1125)


def test_the_system_back_button_is_tapped_where_it_actually_is(tmp_path: Path) -> None:
    """A caller who resolved a nav bar button *means* the bar. Clamping would break it."""
    eng, dev = _engine(_screen(_button("Continue", "[40,600][680,700]")), tmp_path)
    back = next(e for e in eng.analyze().elements if e.content_desc == "Back")
    eng.tap(back.id, observe=False)

    _x, y = _clicked(dev)
    assert y >= _BAR_TOP, "the system Back button must be tapped inside the bar"


def test_without_a_nav_bar_nothing_is_adjusted(tmp_path: Path) -> None:
    """Gesture navigation publishes no buttons, so the bar cannot be located — leave the aim.

    Degrading to the old behaviour is the correct failure mode: a missing observation must
    never produce a differently-wrong aim point.
    """
    eng, dev = _engine(
        _screen(_button("Save period", "[36,1164][684,1252]"), nav_bar=False), tmp_path
    )
    eng.tap(_find(eng, "Save period"), observe=False)
    assert _clicked(dev) == (360, 1208)


def test_a_full_screen_system_overlay_is_not_mistaken_for_the_bar(tmp_path: Path) -> None:
    """The expanded notification shade reaches the bottom edge but is not a bar.

    If it were read as one, every tap on the screen would be "clamped" upward.
    """
    shade = (
        '  <node index="80" package="com.android.systemui" class="android.widget.FrameLayout"\n'
        '        content-desc="Notification shade" resource-id="com.android.systemui:id/shade"\n'
        f'        clickable="true" bounds="[0,0][{_W},{_H}]"/>\n'
    )
    eng, dev = _engine(
        _screen(_button("Continue", "[40,600][680,700]"), shade, nav_bar=False), tmp_path
    )
    eng.tap(_find(eng, "Continue"), observe=False)
    assert _clicked(dev) == (360, 650)
