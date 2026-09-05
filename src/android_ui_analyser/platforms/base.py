"""Common platform boundary for target discovery, connection, and UI normalization.

A platform adapter creates a platform-neutral :class:`TargetRuntime` and translates its
native UI tree into AUA's canonical ``Element`` schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

from ..errors import (
    ConfigError,
    DeviceError,
    InvalidPlatformCapabilityError,
    UnsupportedPlatformCapabilityError,
    UsageError,
)
from ..schema import AppContext, DeviceInfo, Element, TargetInfo
from .api import PLATFORM_API_VERSION
from .contracts import (
    ADAPTER_CAPABILITIES,
    DIRECT_CAPABILITIES,
    RUNTIME_CAPABILITIES,
    missing_structural_members,
    normalize_capability,
)
from .geometry import DisplayGeometry
from .runtime import TargetRuntime
from .services import (
    CAPABILITY_METHODS,
    TargetSupervisionService,
    VirtualTargetsService,
    missing_members,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config
    from ..providers.base import ScreenImage
    from .diagnostics import AppExitEvidence, DiagnosticSourcePolicy, DiagnosticWindow


DiscoveredTarget = TargetInfo | DeviceInfo


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
    platform_api_version: ClassVar[int] = PLATFORM_API_VERSION
    api_version: ClassVar[int] = PLATFORM_API_VERSION  # compatibility for early plugin drafts
    capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.options: Mapping[str, Any] = {}
        self._capability_cache: dict[str, Any] = {}

    def prepare_host(self) -> None:
        """Make host-side tooling discoverable before connecting, if necessary."""

        return None

    def validate_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate adapter-owned configuration and return its normalized representation.

        The default is deliberately closed: silently accepting unknown options makes a misspelled
        plugin setting look configured while the adapter runs with another behavior. Platforms
        with options override this method and may return a copied/normalized mapping.
        """

        if options:
            unknown = ", ".join(sorted(str(key) for key in options))
            raise ConfigError(
                f"platform {self.name!r} does not accept configuration options: {unknown}",
                hint="Remove these options or install an adapter version that declares them.",
            )
        return {}

    def forget_app_process(self, app_id: str | None = None) -> None:
        """Forget anything cached about an app's *process*, after a lifecycle event.

        Launching, restarting, clearing or reinstalling an app replaces its process. A platform
        that caches process identity to keep per-action log scoping to a single round trip must
        drop it here, or the next action reads the dead process's window — which comes back
        empty, and an empty window is indistinguishable from "the app said nothing".

        Platform-neutral by design: the core calls this after every lifecycle action rather than
        knowing which platforms cache what. Default is a no-op.
        """

        return None

    @abstractmethod
    def connect(self, target_id: str | None = None) -> TargetRuntime:
        """Connect to a target, or choose the sole available target."""

    @abstractmethod
    def list_targets(self) -> list[DiscoveredTarget]:
        """List targets using AUA's current cross-process target schema."""

    def target_preference(self, target: DiscoveredTarget) -> int:
        """Lower values win when an unpinned lease chooses among free targets."""

        return 0

    def normalize_key(self, name: str) -> str:
        """Validate one platform-neutral key name before it reaches target input.

        The base contract accepts any non-empty semantic name so a plugin may define keys its
        native framework supports. Built-in adapters may retain validation for historical native
        aliases without leaking those aliases into Engine or portable flow parsing.
        """

        candidate = str(name).strip()
        if not candidate:
            raise UsageError("key name must not be empty")
        return candidate

    def probe_target_capabilities(self, target_id: str) -> dict[str, Any]:
        """Return runtime capabilities used by lease requirements such as ``--needs``."""

        return {}

    def dump_tree(self, runtime: TargetRuntime, *, compact: bool = False) -> str:
        """Capture the platform-native UI tree from an already connected runtime."""

        return runtime.dump_hierarchy(compressed=compact)

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        """Return recent target logs, optionally scoped to one app's current process."""

        raise UnsupportedPlatformCapabilityError(self.name, "device.logs")

    @abstractmethod
    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        """Translate a native UI tree into canonical elements and foreground app id.

        ``geometry`` maps the platform automation API's coordinate space into the physical
        pixels of the captured frame. Adapters whose trees use logical points or rotated native
        coordinates must transform every returned bound through it. It is optional only so
        direct compatibility calls made without a connected runtime keep working.
        """

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
        key = normalize_capability(capability)
        return key in {normalize_capability(item) for item in self.capabilities}

    def validate_declared_capabilities(self) -> None:
        """Reject unknown and incomplete adapter-scoped capability promises.

        Services stay lazy and runtime operations are validated once a target is connected.
        Adapter operations can be verified immediately without importing any optional service.
        """

        declared = {normalize_capability(item) for item in self.capabilities}
        known = set(DIRECT_CAPABILITIES) | set(CAPABILITY_METHODS)
        unknown = sorted(declared - known)
        if unknown:
            raise InvalidPlatformCapabilityError(
                self.name,
                unknown[0],
                ["registered capability specification"],
            )
        for name, spec in ADAPTER_CAPABILITIES.items():
            if name not in declared:
                continue
            missing = missing_structural_members(
                self,
                spec.members,
                default_owner=PlatformAdapter,
                inherited_defaults=spec.inherited_defaults,
            )
            if missing:
                raise InvalidPlatformCapabilityError(self.name, name, missing)

    def validate_runtime(self, runtime: TargetRuntime) -> TargetRuntime:
        """Validate every runtime capability claimed by this adapter after connection."""

        target_id = getattr(runtime, "target_id", None)
        if not isinstance(target_id, str) or not target_id.strip():
            raise InvalidPlatformCapabilityError(
                self.name,
                "target_runtime",
                ["non-empty string target_id"],
            )
        for name, spec in RUNTIME_CAPABILITIES.items():
            if not self.supports(name):
                continue
            missing = missing_structural_members(
                runtime,
                spec.members,
                default_owner=TargetRuntime,
                inherited_defaults=spec.inherited_defaults,
            )
            if missing:
                raise InvalidPlatformCapabilityError(self.name, name, missing)
        return runtime

    def runtime_capability(
        self,
        capability: str,
        runtime: TargetRuntime,
    ) -> TargetRuntime:
        """Resolve one per-target capability and validate its complete runtime surface."""

        key = normalize_capability(capability)
        spec = RUNTIME_CAPABILITIES.get(key)
        if spec is None or not self.supports(key):
            raise UnsupportedPlatformCapabilityError(self.name, key)
        missing = missing_structural_members(
            runtime,
            spec.members,
            default_owner=TargetRuntime,
            inherited_defaults=spec.inherited_defaults,
        )
        if missing:
            raise InvalidPlatformCapabilityError(self.name, key, missing)
        return runtime

    def adapter_capability(self, capability: str) -> PlatformAdapter:
        """Resolve one adapter-owned capability without loading a service module."""

        key = normalize_capability(capability)
        spec = ADAPTER_CAPABILITIES.get(key)
        if spec is None or not self.supports(key):
            raise UnsupportedPlatformCapabilityError(self.name, key)
        missing = missing_structural_members(
            self,
            spec.members,
            default_owner=PlatformAdapter,
            inherited_defaults=spec.inherited_defaults,
        )
        if missing:
            raise InvalidPlatformCapabilityError(self.name, key, missing)
        return self

    def diagnostic_logs(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        """Return recent platform diagnostics for a failed test artifact.

        This optional operation is deliberately capability-gated.  A platform that advertises
        ``device.logs`` must implement it; every other platform gets an explicit unsupported
        result instead of an Android fallback in the engine. ``since_ms`` is the target-clock
        start of the relevant action window when the platform supports time-bounded logs.

        *app_id* asks for the window scoped to that app's process. A platform that cannot scope
        may ignore it and return the unscoped window — the caller filters what it can host-side
        — but it must never return another app's logs *as* this app's, so a platform whose logs
        cannot be attributed at all should advertise no ``device.logs`` capability.
        """

        raise DeviceError(
            f"platform '{self.name}' does not support diagnostic logs",
            code="unsupported_capability",
        )

    def diagnostic_window(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        """Return normalized diagnostics owned and interpreted by this adapter."""

        raise UnsupportedPlatformCapabilityError(self.name, "device.logs")

    def mark_diagnostics(
        self,
        runtime: TargetRuntime,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, Any]:
        """Persist an adapter-clock cursor for a later diagnostic window."""

        raise UnsupportedPlatformCapabilityError(self.name, "device.logs")

    def clear_diagnostics(self, runtime: TargetRuntime) -> None:
        """Clear this target's platform diagnostic buffer when supported."""

        raise UnsupportedPlatformCapabilityError(self.name, "device.logs")

    def diagnostic_source_policy(self, app_id: str | None = None) -> DiagnosticSourcePolicy:
        """Adapter-owned default source filtering, available without a connected target."""

        from .diagnostics import DiagnosticSourcePolicy

        return DiagnosticSourcePolicy()

    def app_exit_evidence(
        self,
        before: AppContext | str | None,
        after: AppContext | str | None,
        elements: Sequence[Element],
    ) -> AppExitEvidence | None:
        """Interpret platform-owned evidence that the tested app exited unexpectedly.

        App-to-app hand-offs are ordinary navigation, while a launcher fallback or native crash
        surface may prove that the app died.  Those distinctions depend on platform-owned app
        identities and system UI, so the shared engine must not infer them from Android package
        names or resource ids.  Adapters return normalized evidence when they can prove an exit.
        """

        return None

    def link_chooser_visible(self, runtime: TargetRuntime) -> bool:
        """Whether a platform-owned app-selection surface intercepted an opened link.

        The generic engine understands the consequence (the requested destination has not
        landed), but package names, native resolver activities and localized system labels are
        platform grammar. Adapters that expose such a surface normalize it here. Platforms that
        dispatch links deterministically can keep the conservative default.
        """

        return False

    def link_chooser_candidates(
        self,
        elements: Sequence[Element],
        *,
        preferred_app_id: str | None = None,
    ) -> list[Element]:
        """Return selectable app rows from a normalized link chooser, best candidate first."""

        return []

    def link_chooser_confirmation(self, elements: Sequence[Element]) -> Element | None:
        """Return a follow-up confirmation control after choosing a link handler, if any."""

        return None

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

    def installed_app(self, runtime: TargetRuntime, app_id: str) -> InstalledApp:
        """Report whether *app_id* is installed on the target, and at which version."""

        raise DeviceError(
            f"platform '{self.name}' does not support app bundle installs",
            code="unsupported_capability",
        )

    def install_app_bundle(
        self,
        runtime: TargetRuntime,
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

    def uninstall_app(self, runtime: TargetRuntime, app_id: str) -> None:
        """Remove *app_id* and its data. An app that is already absent is not an error."""

        raise DeviceError(
            f"platform '{self.name}' does not support app bundle installs",
            code="unsupported_capability",
        )

    def install_persistence_warning(self, runtime: TargetRuntime) -> str | None:
        """Why an install on this target may not outlive the session, or ``None`` if it will.

        A disposable target can accept an install, confirm it, and lose it on restart. That is a
        legitimate way to run — so this reports rather than refuses — but it has to be *reported*,
        because the caller cannot tell it apart from a durable install until much later.
        """

        return None

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        """Capture the current UI frame through the selected platform runtime.

        Evidence writers use this capability instead of reaching through the generic engine to
        a native runtime. Adapters that do not advertise ``ui.screenshot`` fail
        explicitly so unsupported evidence never falls back to Android tooling.
        """

        raise DeviceError(
            f"platform '{self.name}' does not support screenshots",
            code="unsupported_capability",
        )

    def load_capability(self, capability: str) -> Any | None:
        """Return the implementation of one optional semantic capability.

        Target-level operations live on :class:`TargetRuntime`; host/platform-wide operations use
        this second gate. Implementations are structural services: for example,
        ``virtual_targets`` exposes neutral discover/start/stop operations and ``app_database``
        exposes ``list_databases/query_database/execute_database``. The stable capability names and
        their public method surfaces are the plugin contract; core code must never import a
        concrete platform module as a fallback.

        Returning ``None`` means unsupported.  Subclasses should load lazily so an iOS or web
        adapter does not need Android dependencies installed merely to start AUA.
        """

        return None

    @overload
    def capability(
        self,
        capability: Literal["virtual_targets", "virtual_devices", "emulator"],
    ) -> VirtualTargetsService: ...

    @overload
    def capability(
        self,
        capability: Literal["target_supervision"],
    ) -> TargetSupervisionService: ...

    @overload
    def capability(self, capability: str) -> Any: ...

    def capability(self, capability: str) -> Any:
        """Resolve and memoize a semantic service, or raise a typed refusal."""

        key = normalize_capability(capability)
        if key in ADAPTER_CAPABILITIES:
            return self.adapter_capability(key)
        if key in RUNTIME_CAPABILITIES:
            raise UnsupportedPlatformCapabilityError(self.name, key)
        if not self.supports(key):
            raise UnsupportedPlatformCapabilityError(self.name, key)
        if key not in self._capability_cache:
            service = self.load_capability(key)
            if service is None:
                raise UnsupportedPlatformCapabilityError(self.name, key)
            missing = missing_members(key, service)
            if missing:
                raise InvalidPlatformCapabilityError(self.name, key, missing)
            self._capability_cache[key] = service
        return self._capability_cache[key]

    def doctor_checks(self) -> dict[str, Any]:
        """Platform-owned environment checks for ``aua doctor``."""

        return {
            "platform": {
                "ok": True,
                "detail": self.name,
                "capabilities": sorted(self.capabilities),
            }
        }
