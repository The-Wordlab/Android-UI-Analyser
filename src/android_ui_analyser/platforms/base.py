"""Common platform boundary for target discovery, connection, and UI normalization.

The engine still speaks the existing :class:`~android_ui_analyser.device.Device`
runtime protocol.  A platform adapter is the replaceable strategy that creates that
runtime and translates its native UI tree into AUA's canonical ``Element`` schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ..device import Device
from ..errors import DeviceError
from ..schema import DeviceInfo, Element

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config


@dataclass(frozen=True)
class NormalizedTree:
    """A platform-native UI tree translated into AUA's stable core schema."""

    elements: list[Element]
    app_id: str | None = None


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
