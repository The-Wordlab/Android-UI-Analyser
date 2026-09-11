"""Fake-only bounds for inventory and emulator readiness under a stuck transport."""

from __future__ import annotations

import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from android_ui_analyser import emulator
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_device


def test_inventory_bounds_handshake_and_properties_and_continues_after_timeout(monkeypatch) -> None:
    timeouts = []
    clients = []
    commands = []
    transports = [
        SimpleNamespace(serial="slow", state="device", tags={"model": "Slow"}),
        SimpleNamespace(serial="ready", state="device", tags={"model": "Ready"}),
    ]

    def open_transport(serial, *, timeout):  # type: ignore[no-untyped-def]
        assert 0 < timeout <= 1
        timeouts.append(timeout)
        if serial == "slow":
            raise TimeoutError("transport handshake timed out")
        return nullcontext(
            SimpleNamespace(
                send_command=commands.append,
                check_okay=lambda: None,
                read_until_close=lambda: (
                    "[ro.product.model]: [Example]\n[ro.product.locale]: [en-GB]"
                ),
            )
        )

    def client(*, socket_timeout):  # type: ignore[no-untyped-def]
        clients.append(socket_timeout)
        return SimpleNamespace(
            list=lambda *, extended: transports,
            device=lambda *, serial: SimpleNamespace(
                open_transport=lambda **kwargs: open_transport(serial, **kwargs)
            ),
        )

    monkeypatch.setitem(sys.modules, "adbutils", SimpleNamespace(AdbClient=client))
    result = android_device.list_devices()
    assert clients == [1.0]
    assert len(timeouts) == 2 and commands == ["shell:getprop"]
    assert [(item.serial, item.state) for item in result] == [
        ("slow", "offline"),
        ("ready", "device"),
    ]
    assert result[1].model == "Example" and result[1].locale == "en-GB"


def test_inventory_enrichment_budget_preserves_unenriched_transport_rows(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(android_device.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(android_device, "_INVENTORY_ENRICHMENT_BUDGET_S", 2.0)
    inspected = []

    def open_transport(serial, *, timeout):  # type: ignore[no-untyped-def]
        inspected.append(serial)
        now[0] += timeout
        raise TimeoutError("bounded failure")

    client = SimpleNamespace(
        list=lambda **kwargs: [
            SimpleNamespace(serial=f"target-{index}", state="device", tags={}) for index in range(4)
        ],
        device=lambda *, serial: SimpleNamespace(
            open_transport=lambda **kwargs: open_transport(serial, **kwargs)
        ),
    )
    monkeypatch.setitem(sys.modules, "adbutils", SimpleNamespace(AdbClient=lambda **kwargs: client))
    result = android_device.list_devices()
    assert inspected == ["target-0", "target-1"]
    assert len(result) == 4
    assert result[2].state == "device" and result[2].android_version is None


@pytest.mark.parametrize("wait_kind", ["exact", "new"])
def test_serial_wait_does_not_issue_another_inventory_or_accept_late_read(
    wait_kind, monkeypatch
) -> None:
    now = [0.0]
    calls = []
    monkeypatch.setattr(emulator.time, "monotonic", lambda: now[0])

    def inventory():  # type: ignore[no-untyped-def]
        calls.append(True)
        now[0] += 2.0
        return [{"serial": "emulator-5998", "state": "device"}]

    monkeypatch.setattr(emulator, "running_emulators", inventory)
    result = (
        emulator._wait_for_serial("emulator-5998", timeout_s=1)
        if wait_kind == "exact"
        else emulator._wait_for_new_emulator(set(), timeout_s=1)
    )
    assert result is None and len(calls) == 1


def test_boot_shell_receives_remaining_deadline_and_does_not_probe_after_it(monkeypatch) -> None:
    now = [9.5]
    monkeypatch.setattr(emulator.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(emulator, "adb_bin", lambda: "/fake/adb")
    calls = []

    def run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((command, kwargs["timeout"]))
        now[0] = 10.1
        return SimpleNamespace(stdout="1")

    monkeypatch.setattr(emulator.subprocess, "run", run)
    shell = emulator._serial_shell("emulator-5998", deadline=10.0)
    assert emulator._wait_for_boot(shell, timeout_s=0.5) is False
    assert len(calls) == 1 and calls[0][1] == 0.5
    with pytest.raises(DeviceError) as raised:
        shell("pm path android")
    assert raised.value.code == "emulator_boot_timeout"
    assert len(calls) == 1


@pytest.mark.parametrize("stalled_command", ["getprop sys.boot_completed", "pm path android"])
def test_boot_read_timeout_retries_readiness_inside_the_original_budget(
    monkeypatch, stalled_command
) -> None:
    now = [0.0]
    monkeypatch.setattr(
        emulator,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ),
    )
    monkeypatch.setattr(emulator, "adb_bin", lambda: "/fake/adb")
    calls = []
    stalled = False

    def run(command, *, timeout, **kwargs):
        nonlocal stalled
        assert command[:4] == ["/fake/adb", "-s", "emulator-5998", "shell"]
        assert timeout == min(30.0, 45.0 - now[0])
        calls.append(command[-1])
        if command[-1] == stalled_command and not stalled:
            stalled = True
            now[0] += timeout
            raise subprocess.TimeoutExpired(command, timeout)
        now[0] += 0.1
        return SimpleNamespace(
            stdout="1"
            if command[-1].startswith("getprop")
            else "package:/system/framework/framework-res.apk"
        )

    monkeypatch.setattr(emulator.subprocess, "run", run)
    shell = emulator._serial_shell("emulator-5998", deadline=45.0)
    assert emulator._wait_for_boot(shell, timeout_s=45.0) is True
    assert calls == (
        ["getprop sys.boot_completed", "getprop sys.boot_completed", "pm path android"]
        if stalled_command.startswith("getprop")
        else ["getprop sys.boot_completed", "pm path android", "pm path android"]
    )
    assert 30.0 < now[0] < 45.0


@pytest.mark.parametrize("remaining_s", [5.0, 35.0])
def test_exhausted_readiness_retries_clean_only_the_owned_start(monkeypatch, tmp_path, remaining_s):
    now = [0.0]
    monkeypatch.setattr(
        emulator,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            time=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ),
    )
    monkeypatch.setattr(emulator, "list_avds", lambda: {"avds": ["example"], "count": 1})
    monkeypatch.setattr(emulator, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(emulator, "adb_bin", lambda: "/fake/adb")
    monkeypatch.setattr(emulator, "running_emulators", lambda: [{"serial": "emulator-5996"}])
    monkeypatch.setattr(emulator, "allocate_console_port", lambda *args, **kwargs: 5998)
    spawned = []

    def spawn(command, **kwargs):
        spawned.append(command)
        return SimpleNamespace(pid=4242, poll=lambda: None)

    def serial(*args, **kwargs):
        now[0] = 8.0
        return "emulator-5998"

    monkeypatch.setattr(emulator.subprocess, "Popen", spawn)
    monkeypatch.setattr(emulator, "_wait_for_serial", serial)
    monkeypatch.setattr(emulator, "_clear_inherited_blackholed_proxy", lambda *a, **k: {"ok": True})
    calls = []

    def run(command, *, timeout, **kwargs):
        assert command == [
            "/fake/adb",
            "-s",
            "emulator-5998",
            "shell",
            "getprop sys.boot_completed",
        ]
        assert 0 < timeout <= remaining_s + 8.0 - now[0]
        calls.append(timeout)
        now[0] += timeout
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(emulator.subprocess, "run", run)
    killed, released = [], []
    monkeypatch.setattr(emulator.os, "killpg", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(emulator, "release_console_port", released.append)
    monkeypatch.setattr(
        emulator, "_adb_emu_kill", lambda *args: pytest.fail("shared ADB must remain untouched")
    )
    with pytest.raises(DeviceError) as raised:
        emulator.start(
            "example",
            cache_dir=tmp_path,
            port=5998,
            wait_s=8.0 + remaining_s,
            animations=True,
            idle_timeout_s=0,
        )
    assert raised.value.code == "emulator_boot_timeout"
    assert calls == ([5.0] if remaining_s == 5.0 else [30.0, 4.0])
    assert now[0] == 8.0 + remaining_s
    assert len(spawned) == 1 and killed == [4242]
    assert released and set(released) == {5998}
    assert not (tmp_path / "emulator" / "example.p5998.json").exists()


def test_start_shares_readiness_budget_and_rolls_back_an_unready_boot(
    monkeypatch, tmp_path
) -> None:
    now = [0.0]
    monkeypatch.setattr(emulator.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(emulator, "list_avds", lambda: {"avds": ["example"], "count": 1})
    monkeypatch.setattr(emulator, "emulator_bin", lambda: "/fake/emulator")
    monkeypatch.setattr(emulator, "running_emulators", lambda: [])
    monkeypatch.setattr(emulator, "allocate_console_port", lambda *args, **kwargs: 5998)
    monkeypatch.setattr(
        emulator.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=4242, poll=lambda: None)
    )
    monkeypatch.setattr(emulator, "_clear_inherited_blackholed_proxy", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(emulator, "_serial_shell", lambda *a, **k: lambda command: "")
    boot_budgets = []

    def serial(*args, **kwargs):  # type: ignore[no-untyped-def]
        now[0] += 8.0
        return "emulator-5998"

    monkeypatch.setattr(emulator, "_wait_for_serial", serial)
    monkeypatch.setattr(
        emulator,
        "_wait_for_boot",
        lambda shell, *, timeout_s: boot_budgets.append(timeout_s) or False,
    )
    killed = []
    released = []
    monkeypatch.setattr(emulator.os, "killpg", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(emulator, "release_console_port", released.append)
    monkeypatch.setattr(
        emulator, "_adb_emu_kill", lambda *args: pytest.fail("must not touch shared ADB")
    )
    with pytest.raises(DeviceError) as raised:
        emulator.start(
            "example", cache_dir=tmp_path, port=5998, wait_s=10, animations=True, idle_timeout_s=0
        )
    assert raised.value.code == "emulator_boot_timeout"
    assert boot_budgets == [2.0]
    assert killed == [4242]
    assert released and set(released) == {5998}
    assert not (tmp_path / "emulator" / "example.p5998.json").exists()
