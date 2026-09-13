"""Resolve explicitly requested credentials and launch a consumer exactly once.

Only selected dotenv values are loaded, into a child environment, never os.environ.
Public results contain status only. Child text output masks literal selected values;
consumers remain responsible for not emitting transformed secrets or saving them in logs.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

from . import credentials

_EXIT_DRAIN_SECONDS = 1.0
_PIPE_POLL_SECONDS = 0.01


def _failure(code: str, message: str, *, started: bool = False) -> dict[str, Any]:
    return {
        "ok": False, "status": "error", "started": started,
        "error": {"code": code, "message": message},
    }


def _file_values(path: Path, names: list[str]) -> dict[str, str]:
    # parse_stream preserves literals without dotenv interpolation or warning logs.
    _, records = credentials._bindings(credentials._snapshot(path))
    return {
        record.key: record.value or ""
        for record in records if record.key in names
    }


def _prepare(
    required: list[str], env_file: str | Path, *, timeout_s: int, prompt: bool,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Internal boundary: the environment must never be serialized or logged."""
    try:
        if not required or any(
            not isinstance(name, str) or not credentials._NAME.fullmatch(name)
            for name in required
        ):
            return _failure(
                "credential_invalid_name", "Require at least one valid environment variable name."
            ), None
        if type(prompt) is not bool or type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
            return _failure(
                "credential_invalid_options", "Use a boolean prompt and a timeout from 1 to 3600 seconds."
            ), None
        names = list(dict.fromkeys(required))
        child_env = os.environ.copy()
        missing = [name for name in names if not child_env.get(name, "").strip()]
        if missing:
            target = Path(env_file).expanduser().absolute()
            path = target.parent.resolve(strict=True) / target.name
            values = _file_values(path, missing)
            for name in missing:
                if values.get(name, "").strip():
                    continue
                if not prompt:
                    return _failure(
                        "credential_missing", f"Required credential {name} is not configured; no command was started."
                    ), None
                result = credentials.request_secret(name, path, timeout_s=timeout_s)
                if not result.get("ok"):
                    return {**result, "started": False}, None
                values = _file_values(path, missing)
            # Reload after all dialogs: an intervening edit must not cause an old or
            # now-missing file value to be passed to the consumer accidentally.
            values = _file_values(path, missing)
            for name in missing:
                value = values.get(name, "")
                if not value.strip():
                    return _failure(
                        "credential_missing", f"Required credential {name} is absent after setup; no command was started."
                    ), None
                child_env[name] = value
        for name in names:
            value = child_env[name]
            if any(char in value for char in "\r\n\0") or len(value.encode("utf-8")) > 65_536:
                return _failure(
                    "credential_invalid_value", "Required credentials must be single-line values of at most 64 KiB."
                ), None
        return {"ok": True, "status": "ready", "started": False}, child_env
    except credentials._CredentialError as error:
        return _failure(error.code, error.message), None
    except (OSError, ValueError, TypeError):
        return _failure(
            "credential_file_error", "Could not load the credential file; check its path and permissions. No command was started."
        ), None


class _Redactor:
    """Bounded literal matching across pipe chunks, before UTF-8 decoding."""

    def __init__(self, values: list[str]):
        secrets = sorted({value.encode("utf-8") for value in values if value}, key=len, reverse=True)
        self._pattern = re.compile(b"|".join(re.escape(value) for value in secrets))
        self._keep = max(map(len, secrets), default=1) - 1
        self._pending = b""

    def feed(self, data: bytes, *, final: bool = False) -> bytes:
        self._pending += data
        limit = len(self._pending) if final else max(0, len(self._pending) - self._keep)
        parts = []
        cursor = 0
        while cursor < limit:
            match = self._pattern.search(self._pending, cursor)
            if match is None or match.start() >= limit:
                parts.append(self._pending[cursor:limit])
                cursor = limit
            else:
                parts.extend((self._pending[cursor:match.start()], b"[REDACTED]"))
                cursor = match.end()
        self._pending = self._pending[cursor:]
        return b"".join(parts)


def _windows_pipe_available(fd: int) -> Callable[[], int | None]:
    """Inspect an anonymous pipe without a blocking read or another IO thread.

    PeekNamedPipe supports anonymous read handles and reports buffered bytes:
    https://learn.microsoft.com/windows/win32/api/namedpipeapi/nf-namedpipeapi-peeknamedpipe
    This pump is the only reader and never leaves another operation pending on the
    synchronous handle. Read at most the reported bytes; a closed writer is EOF.
    """
    import ctypes
    import msvcrt
    from ctypes import wintypes

    windows: Any = ctypes
    runtime: Any = msvcrt
    kernel = windows.WinDLL("kernel32", use_last_error=True)
    peek = kernel.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                     ctypes.POINTER(wintypes.DWORD)]
    peek.restype = wintypes.BOOL
    handle = wintypes.HANDLE(runtime.get_osfhandle(fd))

    def available() -> int | None:
        count = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(count), None):
            # ERROR_BROKEN_PIPE: the final inherited write handle has closed.
            if windows.get_last_error() == 109:
                return None
            raise OSError("The consumer output pipe could not be inspected.")
        return int(count.value)

    return available


class _PipeReader:
    def __init__(self, source: BinaryIO):
        self.fd = source.fileno()
        self.available: Callable[[], int | None] | None = None
        if os.name == "nt":
            self.available = _windows_pipe_available(self.fd)
        else:
            os.set_blocking(self.fd, False)

    def read(self) -> bytes | None:
        size = 8192
        if self.available is not None:
            count = self.available()
            if count is None:
                return b""
            if count == 0:
                return None
            size = min(size, count)
        try:
            return os.read(self.fd, size)
        except BlockingIOError:
            return None


class _Output:
    def __init__(self, destination: TextIO, values: list[str]):
        self.destination = destination
        self.redactor = _Redactor(values)
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.failed = False

    def feed(self, chunk: bytes, *, final: bool = False) -> None:
        text = self.decoder.decode(self.redactor.feed(chunk, final=final), final=final)
        if not text or self.failed:
            return
        try:
            self.destination.write(text)
            self.destination.flush()
        except Exception:
            # A custom stream's error can include the rejected text. Keep draining
            # without exposing it or invoking a thread exception hook.
            self.failed = True


def _terminate_consumer(child: subprocess.Popen[bytes]) -> None:
    """Stop only our direct consumer; do not terminate its intentional descendants."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        child.kill()
        child.wait()


def _stream_consumer(
    child: subprocess.Popen[bytes], values: list[str],
) -> tuple[int, bool, bool, bool]:
    """Drain both pipes without worker threads; bound inherited-pipe waits after exit."""
    assert child.stdout is not None and child.stderr is not None
    sources = [cast(BinaryIO, child.stdout), cast(BinaryIO, child.stderr)]
    outputs = [_Output(sys.stdout, values), _Output(sys.stderr, values)]
    cancelled = False
    failed = False
    try:
        readers = [_PipeReader(source) for source in sources]
        active = [0, 1]
        deadline: float | None = None
        while True:
            try:
                exit_code = child.poll()
                if exit_code is not None and deadline is None:
                    deadline = time.monotonic() + _EXIT_DRAIN_SECONDS
                progress = False
                for index in list(active):
                    try:
                        chunk = readers[index].read()
                    except OSError:
                        failed = True
                        active.remove(index)
                        continue
                    if chunk is None:
                        continue
                    outputs[index].feed(chunk, final=not chunk)
                    if not chunk:
                        active.remove(index)
                    progress = True
                if exit_code is not None and not active:
                    return exit_code, cancelled, failed or any(o.failed for o in outputs), False
                if deadline is not None and time.monotonic() >= deadline:
                    # A descendant may still complete a secret prefix. Do not mark
                    # EOF or flush pending redactor/decoder bytes on forced cutoff.
                    return exit_code or 0, cancelled, failed or any(o.failed for o in outputs), True
                if not progress:
                    time.sleep(_PIPE_POLL_SECONDS)
            except KeyboardInterrupt:
                cancelled = True
                _terminate_consumer(child)
    except KeyboardInterrupt:
        _terminate_consumer(child)
        return child.returncode or 0, True, False, False
    except (OSError, ValueError):
        _terminate_consumer(child)
        return child.returncode or 0, cancelled, True, False
    finally:
        # Reads are nonblocking/bounded and owned by this thread: close cannot wait
        # for a background reader and no output thread can run after return.
        for source in sources:
            with contextlib.suppress(OSError):
                source.close()


def run_with_credentials(
    command: list[str], *, required: list[str], env_file: str | Path,
    timeout_s: int = 300, prompt: bool = True,
) -> tuple[dict[str, Any], int]:
    """Wait for required credentials, then start one command without a shell or retry.

    Existing nonempty process variables win. Only requested names are loaded from
    env_file. Cancellation or setup failure starts nothing; a failing consumer is
    never replayed. stdout/stderr stream with literal required values masked.
    """
    if not command or any(not isinstance(arg, str) or "\0" in arg for arg in command) or not command[0]:
        return _failure("credential_command_invalid", "Provide a command after --."), 1
    try:
        result, child_env = _prepare(required, env_file, timeout_s=timeout_s, prompt=prompt)
    except KeyboardInterrupt:
        return {"ok": False, "status": "cancelled", "started": False}, 130
    if child_env is None:
        return result, 130 if result.get("status") == "cancelled" else 1
    values = [child_env[name] for name in dict.fromkeys(required)]
    # A caller must never place a selected credential in process arguments. Pass it
    # via the environment only, including when the value was already configured.
    if any(value in arg for value in values for arg in command):
        return _failure(
            "credential_value_in_command", "A required credential occurs in command arguments; pass it through the environment only."
        ), 1
    try:
        child = subprocess.Popen(
            command, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, ValueError):
        return _failure(
            "credential_command_failed", "The consumer could not start; check the executable and arguments."
        ), 1
    exit_code, cancelled, failed, incomplete = _stream_consumer(child, values)
    if incomplete:
        return _failure(
            "credential_output_incomplete", "The consumer exited but inherited output pipes remained open. Output forwarding stopped; the command was not replayed.",
            started=True,
        ), 1
    if failed:
        return _failure(
            "credential_output_failed", "The consumer's output could not be forwarded completely; the command was not replayed.",
            started=True,
        ), 1
    if cancelled:
        exit_code = 130
    if exit_code < 0:
        exit_code = 128 - exit_code
    return {
        "ok": exit_code == 0, "status": "cancelled" if cancelled else "completed",
        "started": True, "exit_code": exit_code,
    }, exit_code
