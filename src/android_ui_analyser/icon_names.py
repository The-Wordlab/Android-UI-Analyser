"""Names for clickable controls the app never named, read once from their pixels.

A Compose or custom-drawn control often reaches the accessibility tree with no text, no
content description and no resource id. Every reader of the screen then sees "unlabelled
control, top left of the screen": measured on one real app, 26% of the controls a navigator
was offered. A hosted vision model names such a crop correctly ("hamburger menu button, opens
the side drawer") for about $0.00005, but takes 0.6-2.6 s per call, so the name is asked for
once per distinct icon and kept under the cache directory. Every later sight of the same
pixels, on any screen, in any run, costs nothing.

Off by default (`icon_names.enabled`): it is a paid call and needs a key. The provider is
the only thing that talks to a network; this module crops, hashes, caches and writes the
name onto the element as its content description, marked `named_by` so a reader can tell a
name the app gave from a name read off pixels.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from PIL import Image

from .providers.base import IconNamerProvider, Provider, ScreenImage
from .schema import Element

logger = logging.getLogger("android_ui_analyser.icon_names")

#: Pixels of context around the control's bounds; an icon's disc often bleeds past its node.
PAD_PX = 4
#: The key is a difference hash: the icon as a 16x16 grey thumbnail, one bit per neighbouring
#: pair saying which is brighter, across and down (480 bits). Brightness and a little noise
#: cancel out; only the drawn shape is left. Measured on 17 real crops: the same icon on three
#: different screens was 0 bits apart, a one-pixel shift 7, the two most alike different icons
#: 31. Two keys within KEY_TOLERANCE_BITS of each other are the same icon.
KEY_SIDE = 16
KEY_TOLERANCE_BITS = 12
#: A crop whose brightest and darkest pixels are this close has nothing drawn in it: an
#: invisible touch target. The vision model would only answer "unanswerable".
BLANK_RANGE = 24
MAX_NAME_CHARS = 90


def icon_key(crop: Image.Image) -> str:
    """One key per icon as a person sees it, not per exact byte (hex of the difference hash)."""
    grey = crop.convert("L").resize((KEY_SIDE + 1, KEY_SIDE + 1), Image.Resampling.BOX)
    side = KEY_SIDE + 1
    values = list(grey.getdata())  # one int per pixel, row-major
    bits = 0
    for y in range(KEY_SIDE):
        for x in range(KEY_SIDE):
            here = values[y * side + x]
            bits = (bits << 1) | int(values[y * side + x + 1] > here)
            bits = (bits << 1) | int(values[(y + 1) * side + x] > here)
    return f"{bits:0{KEY_SIDE * KEY_SIDE * 2 // 4}x}"


def key_distance(a: str, b: str) -> int:
    """Bits in which two keys differ; the same icon is within KEY_TOLERANCE_BITS."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def is_blank(crop: Image.Image) -> bool:
    values = list(crop.convert("L").getdata())  # one int per pixel
    return max(values) - min(values) < BLANK_RANGE


def clean_name(text: str | None) -> str | None:
    if not text:
        return None
    name = " ".join(text.split()).strip().rstrip(".").strip().strip('"')
    return name[:MAX_NAME_CHARS] or None


class IconNameCache:
    """One JSON file per icon key under ``<cache dir>/icon-names/``, found by nearest key."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir).expanduser() / "icon-names"
        self._keys: list[str] | None = None

    def _known_keys(self) -> list[str]:
        if self._keys is None:
            try:
                self._keys = [p.stem for p in self.dir.glob("*.json")]
            except OSError:
                self._keys = []
        return self._keys

    def get(self, key: str) -> dict[str, Any] | None:
        """The record of the nearest known icon within tolerance, or ``None``."""
        nearest = min(
            (k for k in self._known_keys() if len(k) == len(key)),
            key=lambda k: key_distance(k, key),
            default=None,
        )
        if nearest is None or key_distance(nearest, key) > KEY_TOLERANCE_BITS:
            return None
        try:
            data = json.loads((self.dir / f"{nearest}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("name") else None

    def put(self, key: str, name: str, *, provider: str, model: str | None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"name": name, "provider": provider, "model": model,
                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (self.dir / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        if self._keys is not None:
            self._keys.append(key)


def unnamed_controls(elements: Sequence[Element], *, min_side: int, max_side: int) -> list[Element]:
    """Clickable controls in the app's own window that carry no name and could hold an icon."""
    out: list[Element] = []
    for element in elements:
        if element.clickable is not True or element.window not in (None, "app"):
            continue
        if any((getattr(element, field) or "").strip() for field in ("text", "content_desc", "resource_id")):
            continue
        x0, y0, x1, y1 = element.bounds
        width, height = x1 - x0, y1 - y0
        if not (min_side <= width <= max_side and min_side <= height <= max_side):
            continue
        out.append(element)
    return out


def _crop(image: Image.Image, bounds: Sequence[int]) -> Image.Image:
    x0, y0, x1, y1 = bounds
    return image.crop((max(0, x0 - PAD_PX), max(0, y0 - PAD_PX),
                       min(image.width, x1 + PAD_PX), min(image.height, y1 + PAD_PX)))


def _apply(element: Element, name: str, provider: str) -> None:
    element.content_desc = name
    element.named_by = provider


def name_unlabelled(
    elements: Sequence[Element],
    image: ScreenImage,
    providers: Sequence[Provider],
    cache: IconNameCache,
    *,
    max_per_screen: int,
    min_side: int,
    max_side: int,
    timeout_s: float,
) -> int:
    """Name the unnamed controls on this screen; cache first, then at most a few paid calls.

    Returns how many elements were named. Never raises: a provider failure costs the name,
    not the observation.
    """
    candidates = unnamed_controls(elements, min_side=min_side, max_side=max_side)
    if not candidates:
        return 0
    screen = image.pil()
    named = 0
    misses: list[tuple[Element, str, Image.Image]] = []
    for element in candidates:
        crop = _crop(screen, element.bounds)
        if is_blank(crop):
            continue
        key = icon_key(crop)
        hit = cache.get(key)
        if hit is not None:
            _apply(element, str(hit["name"]), str(hit.get("provider") or "cache"))
            named += 1
        else:
            misses.append((element, key, crop))
    if not misses:
        return named
    provider = next((p for p in providers if isinstance(p, IconNamerProvider) and p.is_available().ok), None)
    if provider is None:
        logger.info("icon naming: %d unnamed control(s) and no available provider", len(misses))
        return named
    misses = misses[:max_per_screen]
    model = provider.settings.get("model") if isinstance(provider.settings, dict) else None
    pool = ThreadPoolExecutor(max_workers=min(4, len(misses)), thread_name_prefix="aua-icon-names")
    futures = {pool.submit(provider.name_icon, ScreenImage.from_pil(crop)): (element, key) for element, key, crop in misses}
    deadline = time.monotonic() + timeout_s
    try:
        for future, (element, key) in futures.items():
            try:
                name = clean_name(future.result(timeout=max(0.0, deadline - time.monotonic())))
            except FuturesTimeout:
                logger.info("icon naming: %s did not answer within %.1fs", provider.name, timeout_s)
                continue
            except Exception as exc:
                logger.info("icon naming: %s failed: %s", provider.name, exc)
                continue
            if name is None:
                continue
            cache.put(key, name, provider=provider.name, model=str(model) if model else None)
            _apply(element, name, provider.name)
            named += 1
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return named
