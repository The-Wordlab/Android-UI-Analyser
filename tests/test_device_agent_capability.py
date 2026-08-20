"""The optional on-device helper is a platform capability, not an Android special case.

Two things are guarded here. First, a non-Android adapter that does not claim
``device_agent`` must get a typed refusal rather than an Android import — that is what keeps
an iOS or web plugin from needing adb installed. Second, the Android adapter must really
resolve the service and expose its whole declared surface, so a half-implemented helper fails
at the gate instead of at the first missing method on a live device.

The helper itself is deliberately inert by default: an absent or unbindable helper must never
change a result, only cost the slower polling path.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.errors import (
    InvalidPlatformCapabilityError,
    UnsupportedPlatformCapabilityError,
)
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import PlatformAdapter
from android_ui_analyser.platforms.services import DEVICE_AGENT, missing_members


class _FakePlatform(PlatformAdapter):
    """A plugin that supports nothing optional — the iOS/web shape."""

    name = "fake"
    capabilities = frozenset()

    def connect(self, target_id: str | None = None):  # pragma: no cover - never called
        raise AssertionError("the capability gate must refuse before connecting")

    def list_targets(self):  # pragma: no cover - never called
        return []

    def normalize_tree(self, raw_tree, screen_size, *, ignored_app_ids=()):
        raise AssertionError("not used by this test")


class _ClaimsButIncomplete(_FakePlatform):
    """Claims the capability and then under-delivers."""

    name = "incomplete"
    capabilities = frozenset({DEVICE_AGENT})

    def load_capability(self, capability: str):
        return object()


def test_core_gets_a_typed_refusal_when_a_platform_has_no_device_agent() -> None:
    platform = _FakePlatform(Config())
    with pytest.raises(UnsupportedPlatformCapabilityError):
        platform.capability(DEVICE_AGENT)


def test_a_partial_device_agent_is_rejected_at_the_gate() -> None:
    platform = _ClaimsButIncomplete(Config())
    with pytest.raises(InvalidPlatformCapabilityError):
        platform.capability(DEVICE_AGENT)


def test_android_resolves_the_device_agent_with_its_full_surface() -> None:
    platform = AndroidPlatform(Config())
    assert DEVICE_AGENT in platform.capabilities
    service = platform.capability(DEVICE_AGENT)
    assert missing_members(DEVICE_AGENT, service) == []
    assert service.__name__ == "android_ui_analyser.device_agent"


def test_the_helper_ships_disabled() -> None:
    helper = Config().helper
    assert helper.enabled is False
    assert helper.auto_setup is True


def test_the_bundled_apk_is_present_and_is_an_apk() -> None:
    """A committed binary drifts silently; assert it exists and is really a zip/APK."""

    from android_ui_analyser import device_agent

    apk = device_agent.apk_path()
    assert apk.exists(), f"bundled helper APK missing at {apk}"
    assert apk.read_bytes()[:2] == b"PK", "bundled helper APK is not a zip archive"


def test_enable_refuses_without_root_and_changes_nothing(monkeypatch) -> None:
    """A retail phone cannot bind a sideloaded service; say so, and leave it untouched.

    "Untouched" means nothing was *changed* — reading two properties to decide is fine and is
    how the decision is made cheaply. What must never happen is a settings write, an appop
    grant, or an APK landing on a device that can never run it.
    """

    from android_ui_analyser import device_agent

    ran: list[str] = []
    installed: list[str] = []
    monkeypatch.setattr(device_agent, "is_installed", lambda serial: False)
    monkeypatch.setattr(
        device_agent, "install", lambda serial, **kw: installed.append(serial) or {}
    )
    monkeypatch.setattr(
        device_agent,
        "_shell",
        lambda serial, command, **kw: (ran.append(command) or "")
        if "getprop" in command or command == "id -u"
        else (ran.append(command) or ""),
    )

    with pytest.raises(device_agent.HelperUnavailableError) as excinfo:
        device_agent.enable("serial-without-root")

    assert excinfo.value.code == "helper_needs_root"
    assert installed == [], "nothing may be installed on a device that cannot run it"
    mutations = [c for c in ran if "settings put" in c or "appops" in c or "pm install" in c]
    assert mutations == [], f"refusing must not change the device: {mutations}"
