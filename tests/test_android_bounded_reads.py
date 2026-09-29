"""Real loopback I/O verifies deadlines beyond a mocked timeout argument."""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.android_bounded_reads import rpc
from android_ui_analyser.platforms.android_device import Uiautomator2Device
from android_ui_analyser.read_budget import ReadBudget, ReadDeadlineExceeded
from conftest import make_config


@contextlib.contextmanager
def server(stage):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(1)
    stopped = threading.Event()
    commands = []

    def serve():
        with listener:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(1)
                incoming = connection.makefile("rb")
                try:
                    for index in range(2):
                        size = int(incoming.read(4), 16)
                        commands.append(incoming.read(size).decode())
                        if stage == "handshake" and index == 0:
                            stopped.wait(1)
                            return
                        connection.sendall(b"OKAY")
                    assert incoming.readline().startswith(b"POST /jsonrpc/0")
                    length = 0
                    while line := incoming.readline():
                        if line == b"\r\n":
                            break
                        if line.lower().startswith(b"content-length:"):
                            length = int(line.split(b":")[1])
                    commands.append(json.loads(incoming.read(length)))
                    if stage == "headers":
                        connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                    elif stage == "body":
                        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n")
                    else:
                        payload = (b'{"error": {"code": -32001, "message": "UiAutomation busy"}}'
                                   if stage == "error" else
                                   b'{"result": {"displayWidth": 320, "displayHeight": 640}}')
                        connection.sendall(
                            f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
                            + payload
                        )
                        return
                    while not stopped.wait(0.01):
                        connection.sendall(b"x")
                except (OSError, ValueError):
                    pass  # deadline shutdown closes the fictional peer
                finally:
                    incoming.close()

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        yield listener.getsockname()[1], commands
    finally:
        stopped.set()
        worker.join(2)
        assert not worker.is_alive()


@pytest.mark.parametrize("stage", ["handshake", "headers", "body"])
def test_absolute_deadline_bounds_handshake_and_trickling_response(stage):
    with server(stage) as (port, commands):
        started = time.monotonic()
        budget = ReadBudget(started + 0.12, time.monotonic)
        with pytest.raises(ReadDeadlineExceeded):
            rpc("127.0.0.1", port, "fictional-target", 9008, "deviceInfo", [], budget)
        elapsed = time.monotonic() - started
        assert 0.10 <= elapsed < 0.5, elapsed
        assert commands[0] == "host:transport:fictional-target"
        # A timer has no device runtime and is joined before rpc returns.
        assert not any(t.name.startswith("aua-read-deadline") for t in threading.enumerate())


def test_success_joins_timer_and_reads_the_existing_server_without_reconnect():
    with server("success") as (port, commands):
        budget = ReadBudget(time.monotonic() + 1, time.monotonic)
        result = rpc("127.0.0.1", port, "fictional-target", 9008, "deviceInfo", [], budget)
        assert result == {"displayWidth": 320, "displayHeight": 640}
        assert commands[:2] == ["host:transport:fictional-target", "tcp:9008"]
        assert commands[2]["method"] == "deviceInfo"
        assert len(commands) == 3
        assert not any(t.name.startswith("aua-read-deadline") for t in threading.enumerate())


def device():
    runtime = Uiautomator2Device.__new__(Uiautomator2Device)
    runtime.serial = "fictional-target"
    runtime._winsize = (320, 640)
    runtime._device_locale_read = True
    runtime._device_locale_memo = None
    return runtime


@pytest.mark.parametrize("operation", ["current_app", "_logcat_dump", "get_clock_ms"])
def test_subordinate_android_process_receives_only_remaining_budget(monkeypatch, operation):
    runtime = device()
    timeouts = []

    def stalled(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        # Model subprocess.run's kill-and-reap timeout boundary without a device process.
        time.sleep(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", stalled)
    budget = ReadBudget(time.monotonic() + 0.10, time.monotonic)
    started = time.monotonic()
    with runtime.read_deadline(budget):
        if operation == "get_clock_ms":
            assert runtime.get_clock_ms() is None
        else:
            with pytest.raises(ReadDeadlineExceeded):
                getattr(runtime, operation)(*([[]] if operation == "_logcat_dump" else []))
    assert time.monotonic() - started < 0.5
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 0.10


def test_wait_observation_rechecks_android_boot_identity_before_publishing_ids(monkeypatch):
    runtime = device()
    boot = ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]
    reads = []
    xml = '<hierarchy><node package="example.app" class="android.widget.Button" text="Ready" clickable="true" bounds="[10,10][80,60]"/></hierarchy>'

    def command(args, **kwargs):
        reads.append(args)
        if args[-1] == "cat /proc/sys/kernel/random/boot_id":
            output = boot[0]
        elif args[-1] == "dumpsys window":
            output = "mCurrentFocus=Window{123 u0 example.app/.MainActivity}"
        else:
            raise AssertionError(args)
        return SimpleNamespace(returncode=0, stdout=output)

    def bounded_rpc(method, params):
        if method == "exist":
            return True
        if method == "objInfo":
            return {"bounds": {"left": 10, "top": 10, "right": 80, "bottom": 60}}
        if method == "dumpWindowHierarchy":
            return xml
        raise AssertionError(method)

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(runtime, "_bounded_rpc", bounded_rpc)
    eng = Engine(make_config(memory={"enabled": False}, lease={"enabled": False}), device=runtime)
    first = eng.wait(for_="Ready", timeout_ms=500, observe=True, with_image=False)
    boot[0] = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    second = eng.wait(for_="Ready", timeout_ms=500, observe=True, with_image=False)
    assert first.observation is not None and second.observation is not None
    assert first.observation.elements[0].published_id != second.observation.elements[0].published_id
    assert sum(args[-1] == "cat /proc/sys/kernel/random/boot_id" for args in reads) >= 2


def _rpc_device():
    runtime = device()
    client = SimpleNamespace(host="127.0.0.1", port=5037)
    runtime._d = SimpleNamespace(_dev=SimpleNamespace(_client=client), _device_server_port=9008)
    return runtime


def test_one_failed_reply_is_read_again_within_the_budget(monkeypatch):
    from android_ui_analyser import read_budget
    from android_ui_analyser.errors import DeviceError
    from android_ui_analyser.platforms import android_bounded_reads

    replies = [DeviceError("bounded Android UI read failed (UiAutomation busy); no reconnect attempted"),
               {"displayWidth": 320}]

    def fake_rpc(*_args):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(android_bounded_reads, "rpc", fake_rpc)
    with read_budget.activate(ReadBudget(time.monotonic() + 5, time.monotonic)):
        assert _rpc_device()._bounded_rpc("deviceInfo", []) == {"displayWidth": 320}
    assert replies == []


def test_a_second_failure_and_an_exhausted_budget_are_not_retried(monkeypatch):
    from android_ui_analyser import read_budget
    from android_ui_analyser.errors import DeviceError
    from android_ui_analyser.platforms import android_bounded_reads

    calls = []

    def always_fails(*_args):
        calls.append(1)
        raise DeviceError("bounded Android read transport unavailable")

    monkeypatch.setattr(android_bounded_reads, "rpc", always_fails)
    with read_budget.activate(ReadBudget(time.monotonic() + 5, time.monotonic)), pytest.raises(DeviceError):
        _rpc_device()._bounded_rpc("deviceInfo", [])
    assert len(calls) == 2, "one retry, then the failure is the answer"
    calls.clear()
    with read_budget.activate(ReadBudget(time.monotonic() + 0.2, time.monotonic)), pytest.raises(DeviceError):
        _rpc_device()._bounded_rpc("deviceInfo", [])
    assert len(calls) == 1, "no retry once the budget cannot hold one"


def test_a_tree_that_changed_under_the_dump_is_read_again_while_the_budget_lasts(monkeypatch):
    from android_ui_analyser import read_budget
    from android_ui_analyser.errors import DeviceError
    from android_ui_analyser.platforms import android_bounded_reads

    calls = []
    stale = DeviceError("bounded Android UI read failed (androidx.test.uiautomator.StaleObjectException); "
                        "no reconnect attempted")

    def settles_on_the_fourth_read(*_args):
        calls.append(1)
        if len(calls) < 4:
            raise stale
        return "<hierarchy/>"

    monkeypatch.setattr(android_bounded_reads, "rpc", settles_on_the_fourth_read)
    with read_budget.activate(ReadBudget(time.monotonic() + 5, time.monotonic)):
        assert _rpc_device()._bounded_rpc("dumpWindowHierarchy", [False, 50]) == "<hierarchy/>"
    assert len(calls) == 4

    def never_settles(*_args):
        calls.append(1)
        raise stale

    calls.clear()
    monkeypatch.setattr(android_bounded_reads, "rpc", never_settles)
    started = time.monotonic()
    with read_budget.activate(ReadBudget(time.monotonic() + 1.5, time.monotonic)), pytest.raises(DeviceError):
        _rpc_device()._bounded_rpc("dumpWindowHierarchy", [False, 50])
    assert time.monotonic() - started < 1.5, "the budget still bounds it"
    assert len(calls) > 2


def test_a_server_error_reply_keeps_its_reason():
    from android_ui_analyser.errors import DeviceError

    with server("error") as (port, _commands):
        budget = ReadBudget(time.monotonic() + 1, time.monotonic)
        with pytest.raises(DeviceError, match=r"\(UiAutomation busy\); no reconnect attempted"):
            rpc("127.0.0.1", port, "fictional-target", 9008, "dumpWindowHierarchy", [], budget)
