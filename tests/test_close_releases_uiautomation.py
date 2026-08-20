"""``Device.close()`` must actually release the UiAutomation slot, as its docstring promises.

uiautomator2's ``stop_uiautomator`` kills the subprocess handle the *calling client* created.
A client that merely attached to an already-running server has none — and that is the normal
case, because the server is an ``app_process`` that outlives the command which started it.

Measured before the fix: two clients, both closed, server still running. The slot stayed held,
so ``adb uiautomator dump``, Maestro and AUA's own on-device helper all stayed locked out by a
command that had already finished.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser.device import Uiautomator2Device


class _FakeU2:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop_uiautomator(self) -> None:
        self.stop_calls += 1


def test_close_kills_the_server_by_name_not_just_the_client_handle(monkeypatch) -> None:
    runs: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any):
        runs.append(cmd)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr("android_ui_analyser.device.subprocess.run", fake_run)

    device = Uiautomator2Device.__new__(Uiautomator2Device)
    device.serial = "emulator-9999"
    fake = _FakeU2()
    device._d = fake

    device.close()

    assert fake.stop_calls == 1, "the client handle should still be stopped"
    killed = [c for c in runs if "pkill" in c]
    assert killed, "close() must also stop the server an attached client cannot own"
    cmd = killed[0]
    assert cmd[:3] == ["adb", "-s", "emulator-9999"], (
        "the kill must be scoped to this serial, or it would disturb a parallel agent's device"
    )
    assert "com.wetest.uia2.Main" in cmd


def test_close_is_safe_when_no_client_was_ever_connected(monkeypatch) -> None:
    monkeypatch.setattr(
        "android_ui_analyser.device.subprocess.run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    device = Uiautomator2Device.__new__(Uiautomator2Device)
    device.serial = "emulator-9999"
    device._d = None
    device.close()  # must not raise
