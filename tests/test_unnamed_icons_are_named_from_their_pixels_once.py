"""A clickable control the app never named gets a name from its pixels, once.

Measured on one real app: 26% of the controls a navigator was offered carried no text, no
content description and no resource id, so they were offered as "unlabelled control, top left
of the screen". A hosted vision model names such a crop correctly ("hamburger menu button,
opens the side drawer") for about $0.00005, but takes 0.6-2.6 s, so the name is asked for once
per distinct icon and kept in the cache directory; every later sight of the same pixels is
free. The feature is off by default: it is a paid call and needs a key.

Only controls that could carry an icon are sent: clickable, in the app's own window, with no
name of their own, of a plausible size, and with something drawn in them. An invisible touch
target is a blank crop and is skipped; the vision model would only say "unanswerable".
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from android_ui_analyser.icon_names import KEY_TOLERANCE_BITS, icon_key, key_distance
from android_ui_analyser.providers.base import Availability, IconNamerProvider, ScreenImage
from android_ui_analyser.providers.registry import register_icon_names
from conftest import FakeDevice, make_config, make_engine

W, H = 720, 1280
ICON = (16, 56, 112, 152)  # unnamed, drawn: three white bars on a dark disc
BLANK = (500, 56, 596, 152)  # unnamed, nothing drawn: an invisible touch target
XML = (
    '<hierarchy rotation="0">'
    '<node class="android.widget.ImageButton" clickable="true" enabled="true" '
    f'bounds="[{ICON[0]},{ICON[1]}][{ICON[2]},{ICON[3]}]"/>'
    '<node class="android.widget.Button" text="Save" clickable="true" enabled="true" '
    'bounds="[200,56][400,152]"/>'
    '<node class="android.widget.ImageView" enabled="true" bounds="[16,300][112,396]"/>'
    '<node class="android.widget.ImageButton" clickable="true" enabled="true" '
    f'bounds="[{BLANK[0]},{BLANK[1]}][{BLANK[2]},{BLANK[3]}]"/>'
    "</hierarchy>"
)


def hamburger_png(jitter: int = 0) -> bytes:
    img = Image.new("RGB", (W, H), (12, 14, 18))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = ICON
    draw.ellipse((x0 + 4, y0 + 4, x1 - 4, y1 - 4), fill=(40, 42, 48))
    for k in range(3):
        y = y0 + 34 + k * 14 + jitter
        draw.rectangle((x0 + 30, y, x1 - 30, y + 4), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@register_icon_names("fake_namer")
class FakeNamer(IconNamerProvider):
    crops: list[tuple[int, int]] = []

    def is_available(self) -> Availability:
        return Availability(True, "fake namer")

    def name_icon(self, image: ScreenImage) -> str | None:
        FakeNamer.crops.append((image.width, image.height))
        return "hamburger menu button, opens the side drawer."


def engine_with(png: bytes, **overrides):
    device = FakeDevice(hierarchy_xml=XML, width=W, height=H, screenshot_bytes=png)
    return make_engine(device=device, **overrides)


def by_bounds(result, bounds):
    return next(el for el in result.elements if tuple(el.bounds) == bounds)


def test_an_unnamed_drawn_icon_is_named_and_everything_else_is_left_alone(tmp_path: Path) -> None:
    FakeNamer.crops.clear()
    engine = engine_with(hamburger_png(), icon_names={"enabled": True, "chain": ["fake_namer"]})

    result = engine.analyze(source="hierarchy", with_ocr=False)

    icon = by_bounds(result, ICON)
    assert icon.content_desc == "hamburger menu button, opens the side drawer"
    assert icon.named_by == "fake_namer"
    assert by_bounds(result, (200, 56, 400, 152)).content_desc is None, "a named button is not sent"
    assert by_bounds(result, (16, 300, 112, 396)).content_desc is None, "a picture is not a control"
    assert by_bounds(result, BLANK).content_desc is None, "nothing drawn, nothing to ask about"
    assert len(FakeNamer.crops) == 1, "one paid call for the one icon worth naming"
    cache_dir = Path(engine.config.cache.dir).expanduser() / "icon-names"
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["name"] == "hamburger menu button, opens the side drawer"
    assert saved["provider"] == "fake_namer"


def test_the_second_sight_of_the_same_pixels_costs_nothing() -> None:
    FakeNamer.crops.clear()
    overrides = {"icon_names": {"enabled": True, "chain": ["fake_namer"]}}
    engine_with(hamburger_png(), **overrides).analyze(source="hierarchy", with_ocr=False)
    assert len(FakeNamer.crops) == 1

    # A new engine, a fresh process as far as the cache is concerned, and a screenshot whose
    # icon sits one pixel lower: the same icon as a person sees it.
    result = engine_with(hamburger_png(jitter=1), **overrides).analyze(source="hierarchy", with_ocr=False)

    assert by_bounds(result, ICON).content_desc == "hamburger menu button, opens the side drawer"
    assert len(FakeNamer.crops) == 1, "the cache answered; no second call"


def test_naming_is_off_unless_switched_on() -> None:
    FakeNamer.crops.clear()
    result = engine_with(hamburger_png()).analyze(source="hierarchy", with_ocr=False)
    assert by_bounds(result, ICON).content_desc is None
    assert FakeNamer.crops == []
    assert make_config().icon_names.enabled is False


def test_the_key_forgives_a_pixel_of_antialiasing_but_not_a_different_icon() -> None:
    bars = Image.open(io.BytesIO(hamburger_png())).crop(ICON)
    bars_shifted = Image.open(io.BytesIO(hamburger_png(jitter=1))).crop(ICON)
    other = Image.new("RGB", bars.size, (40, 42, 48))
    ImageDraw.Draw(other).ellipse((20, 20, 76, 76), outline=(255, 255, 255), width=6)
    assert key_distance(icon_key(bars), icon_key(bars_shifted)) <= KEY_TOLERANCE_BITS
    assert key_distance(icon_key(bars), icon_key(other)) > 4 * KEY_TOLERANCE_BITS


def test_the_payload_says_the_name_came_from_pixels() -> None:
    FakeNamer.crops.clear()
    engine = engine_with(hamburger_png(), icon_names={"enabled": True, "chain": ["fake_namer"]})
    result = engine.analyze(source="hierarchy", with_ocr=False)
    for fmt in ("json", "compact"):
        element = next(e for e in result.as_dict(fmt)["elements"] if tuple(e["bounds"]) == ICON)
        assert element["named_by"] == "fake_namer", fmt
        assert element["content_desc"] == "hamburger menu button, opens the side drawer", fmt
