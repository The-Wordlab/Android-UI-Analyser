"""Common platform boundary for target discovery, connection, and UI normalization.

The engine still speaks the existing :class:`~android_ui_analyser.device.Device`
runtime protocol.  A platform adapter is the replaceable strategy that creates that
runtime and translates its native UI tree into AUA's canonical ``Element`` schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..device import Device
from ..errors import DeviceError
from ..schema import DeviceInfo, Element

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config
    from ..providers.base import ScreenImage


@dataclass(frozen=True)
class NormalizedTree:
    """A platform-native UI tree translated into AUA's stable core schema."""

    elements: list[Element]
    app_id: str | None = None


@dataclass(frozen=True)
class AppBundle:
    """Identity read out of an installable app bundle on the host, before any device call.

    ``app_id`` is the platform's package/bundle identifier.  The engine needs it *before*
    connecting so an "install only if needed" request can be answered without pushing bytes.
    """

    app_id: str
    version_name: str | None = None
    version_code: str | None = None


@dataclass(frozen=True)
class InstalledApp:
    """What a target reports about one already-installed app."""

    app_id: str
    installed: bool
    version_name: str | None = None
    version_code: str | None = None


class PlatformAdapter(ABC):
    """Strategy selected by :class:`PlatformFactory` for one AUA engine.

    New platforms can override the small native boundary here while reusing AUA's
    analysis, history, navigation, memory, and action orchestration.  ``capabilities``
    is deliberately declarative: later platform-specific features can be discovered
    or gated without adding platform conditionals throughout the engine.
    """

    name: ClassVar[str]
    capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare_host(self) -> None:
        """Make host-side tooling discoverable before connecting, if necessary."""

        return None

    @abstractmethod
    def connect(self, target_id: str | None = None) -> Device:
        """Connect to a target, or choose the sole available target."""

    @abstractmethod
    def list_targets(self) -> list[DeviceInfo]:
        """List targets using AUA's current cross-process target schema."""

    def target_preference(self, target: DeviceInfo) -> int:
        """Lower values win when an unpinned lease chooses among free targets."""

        return 0

    def probe_target_capabilities(self, target_id: str) -> dict[str, bool]:
        """Return runtime capabilities used by lease requirements such as ``--needs``."""

        return {}

    def dump_tree(self, runtime: Device, *, compact: bool = False) -> str:
        """Capture the platform-native UI tree from an already connected runtime."""

        return runtime.dump_hierarchy(compressed=compact)

    @abstractmethod
    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        """Translate a native UI tree into canonical elements and foreground app id."""

    def element_state(self, raw_tree: str, element: Element) -> dict[str, object]:
        """Return assertion state for one normalized element.

        Generic platforms can rely on canonical element attributes. A platform whose native
        tree represents compound controls differently may override this without leaking its
        tree grammar into the engine.
        """

        return {
            "checkable": bool(element.checkable),
            "checked": bool(element.checked),
            "enabled": element.enabled,
            "selected": bool(element.selected),
            "focused": element.focused,
            "text": element.text,
            "content_desc": element.content_desc,
        }

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def diagnostic_logs(self, runtime: Device, *, lines: int = 400) -> str:
        """Return recent platform diagnostics for a failed test artifact.

        This optional operation is deliberately capability-gated.  A platform that advertises
        ``device.logs`` must implement it; every other platform gets an explicit unsupported
        result instead of an Android fallback in the engine.
        """

        raise DeviceError(
            f"platform '{self.name}' does not support diagnostic logs",
            code="unsupported_capability",
        )

    # -- app bundle delivery (capability ``app.install``) -----------------
    #
    # "Get the build under test onto the target" is the first step of every run, and it was the
    # one step AUA could not do — callers dropped to a raw platform tool for it. These three
    # methods are the platform-neutral contract for it: inspect a bundle on the host, ask the
    # target what it already has, and push the bundle. Splitting them is what makes the
    # idempotent path possible; a single fused `install` would have to push bytes to find out
    # whether it needed to.

    def inspect_app_bundle(self, bundle: Path) -> AppBundle:
        """Read an installable bundle's app id and version on the host, without a target."""

        raise DeviceError(
            f"platform '{self.name}' does not support app bundle installs",
            code="unsupported_capability",
        )

    def installed_app(self, runtime: Device, app_id: str) -> InstalledApp:
        """Report whether *app_id* is installed on the target, and at which version."""

        raise DeviceError(
            f"platform '{self.name}' does not support app bundle installs",
            code="unsupported_capability",
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
        """Install *bundle* onto the target, raising a typed error if it does not land."""

        raise DeviceError(
            f"platform '{self.name}' does not support app bundle installs",
            code="unsupported_capability",
        )

    def uninstall_app(self, runtime: Device, app_id: str) -> None:
        """Remove *app_id* and its data. An app that is already absent is not an error."""

        raise DeviceError(
            f"platform '{self.name}' does not support app bundle installs",
            code="unsupported_capability",
        )

    def install_persistence_warning(self, runtime: Device) -> str | None:
        """Why an install on this target may not outlive the session, or ``None`` if it will.

        A disposable target can accept an install, confirm it, and lose it on restart. That is a
        legitimate way to run — so this reports rather than refuses — but it has to be *reported*,
        because the caller cannot tell it apart from a durable install until much later.
        """

        return None

    def capture_screenshot(self, runtime: Device) -> ScreenImage:
        """Capture the current UI frame through the selected platform runtime.

        Evidence writers use this capability instead of reaching through the generic engine to
        an Android-backed ``Device``.  Adapters that do not advertise ``ui.screenshot`` fail
        explicitly so unsupported evidence never falls back to Android tooling.
        """

        raise DeviceError(
            f"platform '{self.name}' does not support screenshots",
            code="unsupported_capability",
        )
