"""The order `http_proxy` and its exclusion list are written in.

Writing a *new* `http_proxy` value makes Android rebuild the network's ProxyInfo, and the
exclusion list goes with it. Exclusions written before the proxy are therefore erased the
moment the proxy is armed — silently, because `settings put` reports nothing.

That is not a cosmetic ordering nit. The exclusion list is what keeps Android's own
connectivity probes off the proxy; without it the probes fail whenever the proxy cannot
serve them, the OS marks the network unvalidated, and every app on the device renders its
offline state. It is exactly the "Wi-Fi has a !, everything is offline" report this whole
area exists to prevent.

It is also nearly impossible to catch by hand: re-writing the *same* proxy value is a no-op
that leaves the list alone, so the obvious manual check passes while the real code path
fails. Hence a fake that models the destructive write.
"""

from __future__ import annotations

import logging

import pytest

from android_ui_analyser.device import Uiautomator2Device


class _AndroidSettings:
    """`settings get/put/delete global …` with Android's ProxyInfo rebuild semantics."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.log: list[str] = []

    def shell(self, cmd: str) -> str:
        self.log.append(cmd)
        parts = cmd.split()
        if parts[:3] == ["settings", "get", "global"]:
            return self.values.get(parts[3], "null")
        if parts[:3] == ["settings", "put", "global"]:
            key, value = parts[3], " ".join(parts[4:])
            if key == "http_proxy" and self.values.get(key) != value:
                # The whole point: a changed proxy drops the bypass list with it.
                self.values.pop("global_http_proxy_exclusion_list", None)
            self.values[key] = value
            return ""
        if parts[:3] == ["settings", "delete", "global"]:
            self.values.pop(parts[3], None)
            return ""
        return ""


def _device(settings: _AndroidSettings) -> Uiautomator2Device:
    dev = object.__new__(Uiautomator2Device)  # no adb, no uiautomator2
    dev.serial = "emulator-5554"
    dev.shell = settings.shell  # type: ignore[method-assign]
    return dev


def test_exclusion_list_survives_arming_the_proxy() -> None:
    settings = _AndroidSettings()
    device = _device(settings)

    device.set_http_proxy("127.0.0.1:8080", exclusion_list=["a.example.com", "b.example.com"])

    assert device.get_http_proxy() == "127.0.0.1:8080"
    assert device.get_proxy_exclusion_list() == ["a.example.com", "b.example.com"]


def test_the_proxy_is_written_before_its_exclusion_list() -> None:
    """Pinned as an ordering, not just an end state.

    The read-back-and-retry in `set_http_proxy` repairs the wrong order too, so asserting
    only on the final settings passes either way. The order is the actual fix; the retry is
    the safety net for devices that drop the list for some other reason.
    """
    settings = _AndroidSettings()
    device = _device(settings)

    device.set_http_proxy("127.0.0.1:8080", exclusion_list=["a.example.com"])

    puts = [c for c in settings.log if c.startswith("settings put global")]
    proxy_at = next(i for i, c in enumerate(puts) if "http_proxy" in c)
    bypass_at = next(i for i, c in enumerate(puts) if "exclusion_list" in c)
    assert proxy_at < bypass_at, f"exclusions must be written after the proxy: {puts}"


def test_changing_ports_between_runs_keeps_the_exclusion_list() -> None:
    """`proxy start` picks a fresh random port each time, so the value always changes."""
    settings = _AndroidSettings()
    device = _device(settings)

    device.set_http_proxy("127.0.0.1:41111", exclusion_list=["probe.example.com"])
    device.set_http_proxy("127.0.0.1:52222", exclusion_list=["probe.example.com"])

    assert device.get_http_proxy() == "127.0.0.1:52222"
    assert device.get_proxy_exclusion_list() == ["probe.example.com"]


def test_clearing_the_proxy_clears_its_exclusion_list() -> None:
    settings = _AndroidSettings()
    device = _device(settings)
    device.set_http_proxy("127.0.0.1:8080", exclusion_list=["a.example.com"])

    device.set_http_proxy(None)

    assert device.get_http_proxy() is None
    assert device.get_proxy_exclusion_list() == []


def test_a_device_that_refuses_the_exclusion_list_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence here is what let the bypass be broken on every device for a whole session."""
    settings = _AndroidSettings()
    device = _device(settings)
    real_put = settings.shell

    def refuse(cmd: str) -> str:
        if "put global global_http_proxy_exclusion_list" in cmd:
            settings.log.append(cmd)
            return ""  # accepted, then quietly dropped
        return real_put(cmd)

    device.shell = refuse  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        device.set_http_proxy("127.0.0.1:8080", exclusion_list=["a.example.com"])

    assert device.get_http_proxy() == "127.0.0.1:8080"  # the proxy itself still works
    assert "exclusion list" in caplog.text
