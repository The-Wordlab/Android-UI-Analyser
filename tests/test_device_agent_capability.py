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


# --------------------------------------------------------------------------- disable really disables


def _disable_with(monkeypatch, listed: str) -> tuple[dict, list[str]]:
    """Run `disable` against a device whose accessibility list is *listed*, capturing the shell."""

    from android_ui_analyser import device_agent

    ran: list[str] = []

    def shell(serial: str, command: str, **_kw: object) -> str:
        ran.append(command)
        if command.startswith("settings get secure enabled_accessibility_services"):
            return listed
        if command.startswith("settings get secure accessibility_enabled"):
            return "1"
        return ""

    monkeypatch.setattr(device_agent, "_shell", shell)
    return device_agent.disable("serial"), ran


def test_disable_actually_clears_the_list_when_we_were_the_only_service(monkeypatch) -> None:
    """`disable` reported success and left the helper listed, which is its whole job undone.

    ``settings put <key> <value>`` with an empty value is rejected by the platform — literally
    ``Bad arguments`` — and ``_shell`` does not check the exit status, so the write failed silently.
    With no other service enabled, ``':'.join([])`` produced exactly that command. The master switch
    was then set to 0, so nothing was running and the device *looked* clean; the helper came back the
    moment any other accessibility service was enabled, which is a service the user never re-consented
    to. Clearing needs ``settings delete``.
    """

    from android_ui_analyser import device_agent

    result, ran = _disable_with(monkeypatch, device_agent.SERVICE)
    writes = [c for c in ran if c.startswith("settings put secure enabled_accessibility_services")]
    assert not any(
        c.rstrip().endswith("enabled_accessibility_services") for c in writes
    ), f"wrote an empty value, which the platform rejects: {writes}"
    assert any(
        c.startswith("settings delete secure enabled_accessibility_services") for c in ran
    ), f"the list was never cleared: {ran}"
    assert result == {"enabled": False, "remaining": []}


def test_disable_leaves_another_service_alone(monkeypatch) -> None:
    """The reason this is surgical rather than a blanket clear: TalkBack may be someone's only
    way to use the device, and a QA helper must not take it away."""

    from android_ui_analyser import device_agent

    other = "com.example.reader/com.example.reader.ReaderService"
    result, ran = _disable_with(monkeypatch, f"{other}:{device_agent.SERVICE}")

    assert result["remaining"] == [other]
    assert f"settings put secure enabled_accessibility_services {other}" in ran
    assert not any(c.startswith("settings delete") for c in ran), "cleared a list still in use"
    # The master switch stays on, or the surviving service would be disabled by proxy.
    assert "settings put secure accessibility_enabled 0" not in ran


def test_disable_is_idempotent_when_the_helper_was_never_listed(monkeypatch) -> None:
    """Called twice, or on a device that never had it, this must be a no-op and not an error."""

    from android_ui_analyser import device_agent

    result, ran = _disable_with(monkeypatch, "")
    assert result["enabled"] is False
    assert not any(
        c.startswith("settings put secure enabled_accessibility_services") for c in ran
    ), f"wrote to a list that had nothing of ours in it: {ran}"
    assert (
        f"cmd appops set {device_agent.PACKAGE} ACCESS_RESTRICTED_SETTINGS default" in ran
    ), "disabling must retract the restricted-settings grant even after the service vanished"


def test_restore_state_reinstates_master_appop_and_non_root_adbd(monkeypatch) -> None:
    from android_ui_analyser import device_agent

    other = "com.example.reader/com.example.reader.ReaderService"
    ran: list[str] = []
    unrooted: list[str] = []
    services = f"{other}:{device_agent.SERVICE}"
    master = "1"
    appop = "allow"

    def shell(serial: str, command: str, **_kw: object) -> str:
        nonlocal appop, master, services
        ran.append(command)
        if command == "settings get secure enabled_accessibility_services":
            return services
        if command == "settings get secure accessibility_enabled":
            return master
        if command.startswith("settings put secure enabled_accessibility_services "):
            services = command.rsplit(" ", 1)[-1]
        elif command == "settings delete secure enabled_accessibility_services":
            services = ""
        elif command.startswith("settings put secure accessibility_enabled "):
            master = command.rsplit(" ", 1)[-1]
        elif command == "settings delete secure accessibility_enabled":
            master = ""
        elif command.startswith(
            f"cmd appops set {device_agent.PACKAGE} ACCESS_RESTRICTED_SETTINGS "
        ):
            appop = command.rsplit(" ", 1)[-1]
        elif command == (
            f"cmd appops get {device_agent.PACKAGE} ACCESS_RESTRICTED_SETTINGS"
        ):
            return f"ACCESS_RESTRICTED_SETTINGS: {appop}"
        return ""

    monkeypatch.setattr(device_agent, "_shell", shell)
    monkeypatch.setattr(
        device_agent,
        "unroot",
        lambda serial: unrooted.append(serial) or {"root": False},
    )

    result = device_agent.restore_state(
        "serial",
        {
            "enabled_services": [other],
            "accessibility_enabled": "0",
            "restricted_settings_appop": "ignore",
            "adbd_root": False,
        },
    )

    assert result == {"enabled": False, "remaining": [other]}
    assert f"settings put secure enabled_accessibility_services {other}" in ran
    assert "settings put secure accessibility_enabled 0" in ran
    assert (
        f"cmd appops set {device_agent.PACKAGE} ACCESS_RESTRICTED_SETTINGS ignore" in ran
    )
    assert unrooted == ["serial"]


def test_strict_touch_cleanup_refuses_to_claim_a_live_capture_was_removed(
    monkeypatch,
) -> None:
    from android_ui_analyser import device_agent

    commands: list[str] = []

    def shell(serial: str, command: str, **_kw: object) -> str:
        commands.append(command)
        return "sh -c getevent -lt" if command == "ps -A -o ARGS" else ""

    monkeypatch.setattr(device_agent, "_shell", shell)

    with pytest.raises(device_agent.HelperUnavailableError) as raised:
        device_agent.discard_touch_capture("serial")

    assert raised.value.code == "helper_touch_capture_cleanup_failed"
    assert not any(command.startswith("rm -f") for command in commands)
