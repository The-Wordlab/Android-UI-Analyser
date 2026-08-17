"""Built-in Android platform strategy."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .. import hierarchy
from ..device import Device, connect, list_devices
from ..memory import matches_any
from ..providers.base import ScreenImage
from ..schema import DeviceInfo, Element
from ..scroll_geom import _iter_nodes, _node_box
from . import android_apk
from .base import AppBundle, InstalledApp, NormalizedTree, PlatformAdapter
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
            "app.install",
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

    def diagnostic_logs(self, runtime: Device, *, lines: int = 400) -> str:
        raw = runtime.logcat(dump=True)
        return "\n".join(raw.splitlines()[-max(1, lines) :])

    def capture_screenshot(self, runtime: Device) -> ScreenImage:
        return runtime.screenshot()

    def inspect_app_bundle(self, bundle: Path) -> AppBundle:
        info = android_apk.inspect_bundle(bundle)
        return AppBundle(
            app_id=info.package,
            version_name=info.version_name,
            version_code=info.version_code,
        )

    def installed_app(self, runtime: Device, app_id: str) -> InstalledApp:
        state = android_apk.installed_app(runtime.serial, app_id)
        return InstalledApp(
            app_id=state.package,
            installed=state.installed,
            version_name=state.version_name,
            version_code=state.version_code,
        )

    def install_app_bundle(
        self,
        runtime: Device,
        bundle: Path,
        *,
        replace: bool = True,
        grant_permissions: bool = False,
        timeout_s: float = 300.0,
    ) -> None:
        self.prepare_host()
        android_apk.install_bundle(
            runtime.serial,
            bundle,
            reinstall=replace,
            grant_permissions=grant_permissions,
            timeout_s=timeout_s,
        )

    def uninstall_app(self, runtime: Device, app_id: str) -> None:
        self.prepare_host()
        android_apk.uninstall(runtime.serial, app_id)

    def install_persistence_warning(self, runtime: Device) -> str | None:
        # A `-read-only` emulator (what `--parallel` implies) puts disk writes in an overlay it
        # throws away on stop. The install genuinely works for the life of this instance, so
        # refusing would break the ordinary parallel-agent run — but saying nothing is worse: a
        # caller who boots read-only, installs, and expects the build to still be there next
        # session gets `Success` now and "not installed" later, with nothing explaining the gap.
        from ..emulator import discards_writes

        if not discards_writes(runtime.serial, cache_dir=self.config.cache.dir):
            return None
        return (
            f"{runtime.serial} was booted -read-only (--parallel implies it), so this install "
            "lives only until the emulator stops. Boot with `--parallel --no-read-only` if the "
            "build must survive a restart."
        )

    def element_state(self, raw_tree: str, element: Element) -> dict[str, object]:
        state = super().element_state(raw_tree, element)
        node = next(
            (node for node in _iter_nodes(raw_tree) if _node_box(node) == tuple(element.bounds)),
            None,
        )
        if node is not None:
            holder = node
            if node.get("checkable") != "true":
                holder = next(
                    (child for child in node.iter("node") if child.get("checkable") == "true"),
                    node,
                )
            state.update(
                checkable=holder.get("checkable") == "true",
                checked=holder.get("checked") == "true",
                enabled=node.get("enabled") == "true",
                selected=node.get("selected") == "true",
                focused=node.get("focused") == "true",
            )
        if element.checkable:
            state["checkable"] = True
            state["checked"] = bool(element.checked)
        if element.selected is not None:
            state["selected"] = element.selected
        return state

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
