"""Executable contract checks for independently packaged platform adapters.

This module is intentionally usable without pytest.  A platform plugin can construct a
deterministic disposable target in its own test suite and call
:func:`run_attached_target_conformance` to exercise the same public boundary AUA uses: entry
discovery is handled by :class:`~android_ui_analyser.platforms.registry.PlatformFactory`, while
this profile checks discovery, connection, hierarchy normalization, screenshot geometry, semantic
tap/wait behavior, optional text/key/verified-scroll behavior, and a typed refusal for one omitted
capability.

The profile sends input (at minimum a tap, and any optional actions requested by the case).  It is
therefore for fake, disposable, or deliberately prepared conformance targets, never an arbitrary
user's foreground application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import IncompatiblePlatformPluginError, UnsupportedPlatformCapabilityError
from ..providers.base import Bounds
from .api import PLATFORM_API_VERSION
from .geometry import AffineTransform

if TYPE_CHECKING:  # pragma: no cover - imports used only by type checkers
    from .base import PlatformAdapter


ATTACHED_TARGET_CAPABILITIES = frozenset({"ui.tree", "ui.input", "ui.screenshot"})
"""Capabilities required by the API-v1 attached-target conformance profile."""

_ANDROID_RUNTIME_MEMBERS = frozenset(
    {
        "adb",
        "adb_forward",
        "adb_reverse",
        "dumpsys",
        "logcat",
        "run_as",
        "shell",
        "uiautomator",
    }
)


class PlatformConformanceError(RuntimeError):
    """One behavioral promise in a published conformance profile was not met."""

    def __init__(self, check: str, detail: str) -> None:
        self.check = check
        self.detail = detail
        super().__init__(f"platform conformance failed at {check}: {detail}")


@dataclass(frozen=True, slots=True)
class AttachedTargetCase:
    """Expected state of the disposable screen used by the attached-target profile.

    ``expected_bounds`` and the tap result use AUA's canonical screenshot-pixel coordinates,
    even when the runtime's native accessibility/input coordinate system is scaled or rotated.
    The named element must be unique on the fixture screen.
    """

    target_id: str
    element_text: str
    expected_bounds: Bounds
    expected_app_id: str | None = None
    require_non_identity_geometry: bool = False
    unsupported_capability: str | None = "virtual_targets"
    input_element_text: str | None = None
    input_value: str = "aua conformance"
    key_name: str | None = None
    expected_scrollable_bounds: Bounds | None = None


@dataclass(frozen=True, slots=True)
class AttachedTargetReport:
    """Evidence returned after every attached-target profile check succeeds."""

    platform: str
    requested_target_id: str
    runtime_target_id: str
    discovered_target_ids: tuple[str, ...]
    geometry: AffineTransform
    screenshot_size: tuple[int, int]
    element_bounds: Bounds
    tap_point: tuple[int, int]
    checks: tuple[str, ...]


def _fail(check: str, detail: str) -> None:
    raise PlatformConformanceError(check, detail)


def _conformance_config(adapter: PlatformAdapter):
    """Copy plugin configuration with background/persistent test behavior disabled."""

    from ..config import Config

    data = adapter.config.model_dump(mode="python")
    data["memory"]["enabled"] = False
    data["lease"]["enabled"] = False
    data["teardown"]["enabled"] = False
    data["ocr"]["enabled"] = False
    data["perf"]["prefetch"] = False
    data["perf"]["predictive_prefetch"] = False
    data["perf"]["auto_daemon"] = False
    return Config.model_validate(data)


def run_attached_target_conformance(
    adapter: PlatformAdapter,
    case: AttachedTargetCase,
) -> AttachedTargetReport:
    """Run the API-v1 external attached-target profile against ``adapter``.

    The function raises :class:`PlatformConformanceError` for behavioral failures and preserves
    AUA's typed plugin/capability errors for version or declaration failures.  It intentionally
    goes through :class:`~android_ui_analyser.engine.Engine` for hierarchy analysis and input, so
    a plugin cannot pass merely by making its adapter methods work in isolation.

    The connected runtime is closed before return.  Plugin tests that need transport-level proof
    (for example, checking the native point obtained after inverse rotation) should retain that
    evidence on their adapter or transport fixture.
    """

    from ..engine import Engine

    actual_version = getattr(type(adapter), "platform_api_version", None)
    if actual_version != PLATFORM_API_VERSION:
        raise IncompatiblePlatformPluginError(
            adapter.name,
            expected=PLATFORM_API_VERSION,
            actual=actual_version,
        )

    missing = sorted(
        capability
        for capability in ATTACHED_TARGET_CAPABILITIES
        if not adapter.supports(capability)
    )
    if missing:
        _fail("capabilities", f"missing attached-target capabilities: {', '.join(missing)}")
    adapter.validate_declared_capabilities()

    discovered = adapter.list_targets()
    discovered_ids = tuple(item.target_id for item in discovered)
    if case.target_id not in discovered_ids:
        _fail(
            "discovery",
            f"target {case.target_id!r} was not present in {list(discovered_ids)!r}",
        )
    for item in discovered:
        if item.target_id == case.target_id and item.platform != adapter.name:
            _fail(
                "discovery",
                f"target reports platform {item.platform!r}, expected {adapter.name!r}",
            )

    runtime = adapter.connect(case.target_id)
    try:
        runtime = adapter.validate_runtime(runtime)
        android_members = sorted(name for name in _ANDROID_RUNTIME_MEMBERS if hasattr(runtime, name))
        if android_members:
            _fail(
                "runtime-surface",
                "external runtime exposes Android transport members: "
                + ", ".join(android_members),
            )

        width, height = runtime.window_size()
        geometry = runtime.display_geometry()
        if geometry.canonical_size != (width, height):
            _fail(
                "geometry",
                f"geometry size {geometry.canonical_size!r} != runtime size {(width, height)!r}",
            )
        if case.require_non_identity_geometry and geometry.native_to_canonical == (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ):
            _fail("geometry", "fixture must provide a non-identity coordinate transform")

        screenshot = adapter.adapter_capability("ui.screenshot").capture_screenshot(runtime)
        screenshot_size = (screenshot.width, screenshot.height)
        if screenshot_size != geometry.canonical_size:
            _fail(
                "screenshot",
                f"frame size {screenshot_size!r} != canonical size {geometry.canonical_size!r}",
            )

        engine = Engine(_conformance_config(adapter), device=runtime, platform=adapter)
        # Keep the profile hermetic: caller-latency attribution normally consults host process
        # ancestry, which is unrelated to a platform contract and forbidden by strict fixtures.
        engine._caller_latency_key = "platform-conformance"
        result = engine.analyze(
            source="hierarchy",
            with_ocr=False,
            no_cache=True,
            record=False,
            record_ids=False,
        )
        matches = [element for element in result.elements if element.text == case.element_text]
        if len(matches) != 1:
            _fail(
                "hierarchy",
                f"expected one element with text {case.element_text!r}, found {len(matches)}",
            )
        element = matches[0]
        if element.bounds != case.expected_bounds:
            _fail(
                "geometry",
                f"normalized bounds {element.bounds!r} != expected {case.expected_bounds!r}",
            )
        if case.expected_app_id is not None and result.screen.app_id != case.expected_app_id:
            _fail(
                "app-context",
                f"foreground app {result.screen.app_id!r} != expected {case.expected_app_id!r}",
            )

        checks = [
            "api-version",
            "capabilities",
            "discovery",
            "connection",
            "runtime-surface",
            "geometry",
            "screenshot",
            "hierarchy",
        ]

        tap_point = element.center
        action = engine.tap_point(*tap_point, observe=False)
        if action.target != [*tap_point]:
            _fail("input", f"Engine reported tap target {action.target!r}, expected {tap_point!r}")
        checks.append("engine-tap")

        waited = engine.wait(
            for_=case.element_text,
            timeout_ms=100,
            observe=False,
        )
        if not waited.ok:
            _fail("wait", f"Engine could not find prepared text {case.element_text!r}")
        checks.append("engine-wait")

        if case.input_element_text is not None:
            typed = engine.input_text(
                selector={"text": case.input_element_text},
                text=case.input_value,
                observe=False,
            )
            if not typed.ok:
                _fail("text-input", "Engine-routed text input reported failure")
            checks.append("engine-text-input")

        if case.key_name is not None:
            keyed = engine.key(case.key_name, observe=False)
            if not keyed.ok:
                _fail("key-input", "Engine-routed key input reported failure")
            checks.append("engine-key")

        if case.expected_scrollable_bounds is not None:
            scrollable_box, real_container = engine._scroll_box()
            if not real_container or scrollable_box != case.expected_scrollable_bounds:
                _fail(
                    "scroll-container",
                    f"normalized scroll box {scrollable_box!r} != expected "
                    f"{case.expected_scrollable_bounds!r}",
                )
            swiped = engine.swipe("up", observe=False, verify=True)
            if "moved" not in str(swiped.detail or ""):
                _fail("swipe", f"verified swipe did not report movement: {swiped.detail!r}")
            checks.extend(("scroll-container", "engine-verified-swipe"))

        if case.unsupported_capability is not None:
            if adapter.supports(case.unsupported_capability):
                _fail(
                    "unsupported-capability",
                    f"fixture unexpectedly declares {case.unsupported_capability!r}",
                )
            try:
                adapter.capability(case.unsupported_capability)
            except UnsupportedPlatformCapabilityError as exc:
                if exc.code != "platform_capability_unsupported":
                    _fail(
                        "unsupported-capability",
                        f"typed refusal used error code {exc.code!r}",
                    )
            else:
                _fail(
                    "unsupported-capability",
                    f"missing {case.unsupported_capability!r} did not raise a typed refusal",
                )
            checks.append("unsupported-capability")

        return AttachedTargetReport(
            platform=adapter.name,
            requested_target_id=case.target_id,
            runtime_target_id=runtime.target_id,
            discovered_target_ids=discovered_ids,
            geometry=geometry.native_to_canonical,
            screenshot_size=screenshot_size,
            element_bounds=element.bounds,
            tap_point=tap_point,
            checks=tuple(checks),
        )
    finally:
        runtime.close()


__all__ = [
    "ATTACHED_TARGET_CAPABILITIES",
    "AttachedTargetCase",
    "AttachedTargetReport",
    "PlatformConformanceError",
    "run_attached_target_conformance",
]
