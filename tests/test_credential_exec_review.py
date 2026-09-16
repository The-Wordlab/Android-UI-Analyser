"""Independent boundary checks for private credential continuation."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser import credential_exec as runner
from android_ui_analyser import credentials


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal forwarding has process sessions")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_wrapper_signal_reaches_consumer_cleanup_before_wrapper_exits(
    tmp_path, signum
):
    """A wrapper interrupt must not TERM the harness while its finally block is running."""

    name = "AUA_REVIEW_SIGNAL_TOKEN"
    ready = tmp_path / "consumer-ready"
    cleaned = tmp_path / "consumer-cleaned"
    consumer_pid = tmp_path / "consumer-pid"
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import pathlib, signal, sys, time\n"
        "ready, cleaned, pid = map(pathlib.Path, sys.argv[1:])\n"
        "received = None\n"
        "def interrupt(signum, _frame):\n"
        "    global received\n"
        "    received = signum\n"
        "    raise InterruptedError\n"
        "signal.signal(signal.SIGINT, interrupt)\n"
        "signal.signal(signal.SIGTERM, interrupt)\n"
        "pid.write_text(str(__import__('os').getpid()))\n"
        "try:\n"
        # Readiness must mean cleanup is armed, not just handlers installed. A
        # signal after writing ready but before entering try used to skip finally.
        "    ready.write_text('ready')\n"
        "    while True: time.sleep(1)\n"
        "except InterruptedError:\n"
        "    pass\n"
        "finally:\n"
        "    time.sleep(0.05)\n"  # The wrapper must wait for real consumer cleanup.
        "    cleaned.write_text(signal.Signals(received).name)\n"
    )
    outer_script = """import json, sys
sys.path.insert(0, sys.argv[1])
from android_ui_analyser.credential_exec import run_with_credentials
result, code = run_with_credentials(
    [sys.executable, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]],
    required=[sys.argv[6]], env_file=sys.argv[7], prompt=False,
)
print(json.dumps(result), flush=True)
raise SystemExit(code)
"""
    env = {**os.environ, name: "review-only-signal-token"}
    outer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            outer_script,
            str(Path(runner.__file__).parents[1]),
            str(consumer),
            str(ready),
            str(cleaned),
            str(consumer_pid),
            name,
            str(tmp_path / ".env"),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and outer.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "consumer did not reach its signal handler"
        pid = int(consumer_pid.read_text())
        os.kill(outer.pid, signum)
        stdout, stderr = outer.communicate(timeout=10)
        assert outer.returncode == 128 + signum
        assert cleaned.read_text() == signal.Signals(signum).name
        assert json.loads(stdout)["status"] == "cancelled"
        assert stderr == b""
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(outer.pid, signal.SIGKILL)
        outer.communicate(timeout=5)


def test_keyboard_interrupt_during_setup_is_cancelled_without_a_consumer(tmp_path, monkeypatch):
    name = "AUA_REVIEW_INTERRUPT_TOKEN"
    monkeypatch.delenv(name, raising=False)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(credentials, "request_secret", interrupt)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("Consumer must not start")
    )
    try:
        result, code = runner.run_with_credentials(
            ["consumer"], required=[name], env_file=tmp_path / ".env"
        )
    except KeyboardInterrupt:
        pytest.fail("Setup interruption escaped the structured cancellation boundary")
    assert code == 130
    assert result == {"ok": False, "status": "cancelled", "started": False}
    assert not (tmp_path / ".env").exists()


@pytest.mark.skipif(os.name != "posix", reason="Isolated process-group cleanup requires POSIX")
def test_exited_consumer_with_inherited_descendant_pipes_does_not_hang(tmp_path, monkeypatch):
    name = "AUA_REVIEW_DESCENDANT_TOKEN"
    monkeypatch.setenv(name, "review-only-long-credential-value")
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import os, subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print('direct-consumer-finished' + '.' * 128, flush=True)\n"
        f"os.write(1, os.environ[{name!r}].encode()[:12])\n"
    )
    outer_script = """import json, sys, threading
sys.path.insert(0, sys.argv[1])
from android_ui_analyser.credential_exec import run_with_credentials
before = threading.active_count()
result, code = run_with_credentials(
    [sys.executable, sys.argv[2]], required=[sys.argv[3]], env_file=sys.argv[4], prompt=False,
)
result['thread_delta'] = threading.active_count() - before
print(json.dumps(result), flush=True)
raise SystemExit(code)
"""
    outer = subprocess.Popen(
        [sys.executable, "-c", outer_script, str(Path(runner.__file__).parents[1]),
         str(consumer), name, str(tmp_path / ".env")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = outer.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            pytest.fail("Exited consumer left the wrapper waiting indefinitely on descendant pipes")
        assert b"direct-consumer-finished" in stdout
        assert b'"started": true' in stdout
        assert b'"thread_delta": 0' in stdout
        assert b"credential_output_incomplete" in stdout
        assert outer.returncode == 1
        assert b"review-only-long-credential-value" not in stdout + stderr
        assert b"review-only-" not in stdout + stderr
    finally:
        # Only this test's process group: remove the deliberately sleeping descendant.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(outer.pid, signal.SIGKILL)
        outer.communicate(timeout=5)


def test_large_both_stream_output_drains_completely(tmp_path, monkeypatch, capsys):
    name = "AUA_REVIEW_LARGE_OUTPUT_TOKEN"
    monkeypatch.setenv(name, "review-only-long-token-for-large-output")
    command = [sys.executable, "-c", (
        "import os; "
        "os.write(1, b'a' * 1048576); os.write(2, b'b' * 1048576); "
        f"value=os.environ[{name!r}].encode(); "
        "os.write(1, value + b'\\n'); os.write(2, value + b'\\n')"
    )]
    result, code = runner.run_with_credentials(
        command, required=[name], env_file=tmp_path / ".env", prompt=False,
    )
    captured = capsys.readouterr()
    assert code == 0 and result["status"] == "completed"
    assert captured.out == "a" * 1048576 + "[REDACTED]\n"
    assert captured.err == "b" * 1048576 + "[REDACTED]\n"


def test_windows_peek_contract_distinguishes_wait_bytes_eof_and_error(monkeypatch):
    class FakePeek:
        count = 0
        error = 0

        def __call__(self, handle, buffer, size, read, count, left):
            assert handle.value == 123
            assert buffer is None and size == 0 and read is None and left is None
            count._obj.value = self.count
            return not self.error

    peek = FakePeek()
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda _fd: 123))
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda *_a, **_kw: SimpleNamespace(PeekNamedPipe=peek), raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: peek.error, raising=False)
    available = runner._windows_pipe_available(45)
    assert available() == 0
    peek.count = 37
    assert available() == 37
    peek.error = 109
    assert available() is None
    peek.error = 5
    with pytest.raises(OSError, match="could not be inspected"):
        available()


@pytest.mark.parametrize("available,expected", [(0, None), (None, b""), (9, b"data"), (10000, b"data")])
def test_windows_reader_never_reads_more_than_available(monkeypatch, available, expected):
    calls = []

    def read(fd, count):
        calls.append((fd, count))
        return b"data"

    with monkeypatch.context() as windows:
        windows.setattr(runner.os, "name", "nt")
        windows.setattr(runner, "_windows_pipe_available", lambda _fd: lambda: available)
        windows.setattr(runner.os, "read", read)
        reader = runner._PipeReader(SimpleNamespace(fileno=lambda: 45))
        assert reader.read() == expected
    assert calls == ([(45, min(8192, available))] if available else [])
