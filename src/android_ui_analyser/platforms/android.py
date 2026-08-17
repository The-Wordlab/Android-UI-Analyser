"""Built-in Android platform strategy."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .. import hierarchy
from ..device import Device, connect, list_devices
from ..memory import matches_any
from ..schema import DeviceInfo
from .base import NormalizedTree, PlatformAdapter
from .registry import register_platform

_PACKAGE_RE = re.compile(r'package="([^"]+)"')


def package_from_tree(
    raw_tree: str, ignore: Sequence[str] = ("com.android.systemui",)
) -> str | None:
    """Guess Android's foreground package from its hierarchy XML."""

    packages = _PACKAGE_RE.findall(raw_tree)
    if not packages:
        return None
    counts = Counter(package for package in packages if package and not matches_any(package, ignore))
    if not counts:
        counts = Counter(packages)
    return counts.most_common(1)[0][0]


@register_platform("android")
class AndroidPlatform(PlatformAdapter):
    """Android implementation backed by adb and uiautomator2."""

    capabilities = frozenset(
        {
            "app.files",
            "app.lifecycle",
            "device.logs",
            "device.network",
            "device.proxy",
            "device.shell",
            "emulator",
            "ui.input",
            "ui.screenshot",
            "ui.tree",
        }
    )

    def prepare_host(self) -> None:
        from ..emulator import ensure_adb_on_path

        ensure_adb_on_path()

    def connect(self, target_id: str | None = None) -> Device:
        self.prepare_host()
        return connect(target_id)

    def list_targets(self) -> list[DeviceInfo]:
        self.prepare_host()
        return list_devices()

    def target_preference(self, target: DeviceInfo) -> int:
        # Prefer a disposable emulator over a physical USB phone when the user did not pin one.
        return 0 if target.serial.startswith("emulator-") else 1

    def probe_target_capabilities(self, target_id: str) -> dict[str, bool]:
        from .. import leases

        return leases.probe_capabilities(self.config.cache.dir, target_id)

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree(
            elements=hierarchy.parse_hierarchy(raw_tree, screen_size),
            app_id=package_from_tree(raw_tree, ignored_app_ids),
        )
