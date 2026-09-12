"""Local, masked credential entry. Values never cross the CLI/MCP result boundary.

The GUI child returns its input through a private pipe to this process. Only this
module parses/writes it; callers receive status, variable name and destination.
No device, daemon, network, environment export, shell, or clipboard is involved.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv.parser import parse_stream

_MAX_FILE_BYTES = 1_048_576
_MAX_VALUE_BYTES = 65_536
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,255}\Z")

# Only public metadata is supplied as argv. Do not interpolate a value into code,
# use shell=True, or surface a child error: it can contain the entered value.
_MAC_DIALOG = '''on run argv
    set envName to item 1 of argv
    set destination to item 2 of argv
    set waitSeconds to (item 3 of argv) as integer
    try
        activate
        set answer to display dialog ("Enter " & envName & return & return & "Save to:" & return & destination) with title "AUA — Save credential" default answer "" buttons {"Cancel", "Save"} default button "Save" cancel button "Cancel" with hidden answer giving up after waitSeconds
        if gave up of answer then return "timeout:"
        return "saved:" & text returned of answer
    on error number -128
        return "cancelled:"
    end try
end run
'''

_TK_DIALOG = '''import sys
try:
    import tkinter as tk
    root = tk.Tk()
except Exception:
    print("unavailable:")
    raise SystemExit(0)
root.title("AUA — Save credential")
root.resizable(False, False)
name, destination, timeout = sys.argv[1:]
frame = tk.Frame(root, padx=20, pady=18)
frame.pack()
tk.Label(frame, text="Enter " + name, anchor="w").pack(fill="x")
tk.Label(frame, text="Save to:\\n" + destination, anchor="w", justify="left", wraplength=480).pack(fill="x", pady=(8, 12))
entry = tk.Entry(frame, show="*", width=56)
entry.pack(fill="x")
message = tk.StringVar()
tk.Label(frame, textvariable=message, fg="red").pack(fill="x")
def finish(status):
    print(status, flush=True)
    root.destroy()
def save(event=None):
    value = entry.get()
    if not value.strip() or any(c in value for c in "\\r\\n\\0") or len(value.encode("utf-8")) > 65536:
        message.set("Enter a nonempty, single-line value.")
        return
    finish("saved:" + value)
buttons = tk.Frame(frame)
buttons.pack(anchor="e", pady=(8, 0))
tk.Button(buttons, text="Cancel", command=lambda: finish("cancelled:")).pack(side="left", padx=6)
tk.Button(buttons, text="Save", command=save, default="active").pack(side="left")
root.protocol("WM_DELETE_WINDOW", lambda: finish("cancelled:"))
root.bind("<Escape>", lambda event: finish("cancelled:"))
root.bind("<Return>", save)
root.after(int(timeout) * 1000, lambda: finish("timeout:"))
root.lift()
entry.focus_force()
root.mainloop()
'''


class _CredentialError(Exception):
    """Only fixed, public messages belong in this exception."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class _Snapshot:
    # repr must never expose existing credentials if a caller logs an exception.
    data: bytes = field(repr=False)
    identity: tuple[int, int, int, int, int] | None


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _snapshot(path: Path) -> _Snapshot:
    try:
        before = path.lstat()
    except FileNotFoundError:
        return _Snapshot(b"", None)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise _CredentialError("credential_unsafe_file", "The destination must be a regular file, not a link.")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source:
        if _identity(os.fstat(source.fileno())) != _identity(before):
            raise _CredentialError("credential_file_changed", "The destination changed; retry the request.")
        data = source.read(_MAX_FILE_BYTES + 1)
        if len(data) > _MAX_FILE_BYTES:
            raise _CredentialError("credential_file_too_large", "The dotenv file exceeds the 1 MiB limit.")
        if _identity(os.fstat(source.fileno())) != _identity(before):
            raise _CredentialError("credential_file_changed", "The destination changed; retry the request.")
    if _identity(path.lstat()) != _identity(before):
        raise _CredentialError("credential_file_changed", "The destination changed; retry the request.")
    return _Snapshot(data, _identity(before))


def _bindings(snapshot: _Snapshot):
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeError:
        raise _CredentialError("credential_invalid_env", "The dotenv file must use UTF-8.") from None
    # Retain the BOM outside parser input, so the first assignment still matches.
    bom = "\ufeff" if text.startswith("\ufeff") else ""
    records = list(parse_stream(io.StringIO(text.removeprefix(bom) if bom else text)))
    if any(record.error for record in records):
        raise _CredentialError("credential_invalid_env", "The dotenv file contains invalid syntax; correct it before saving.")
    return bom, records


def _updated(snapshot: _Snapshot, name: str, value: str) -> bytes:
    bom, records = _bindings(snapshot)
    newline = "\r\n" if b"\r\n" in snapshot.data else "\n"
    # python-dotenv single quotes decode \\ and \'. Interpolation is a loader
    # option, independent of quoting: arbitrary credentials need interpolate=False.
    quoted = "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    assignment = f"{name}={quoted}{newline}"
    chunks = [bom]
    replaced = False
    for record in records:
        if record.key != name:
            chunks.append(record.original.string)
            continue
        leading = re.match(r"[\r\n]*", record.original.string).group(0)  # type: ignore[union-attr]
        chunks.append(leading)
        if not replaced:
            prefix = "export " if re.match(r"\s*export\s+", record.original.string) else ""
            chunks.append(prefix + assignment)
            replaced = True
        # Remove other bindings for this key so a later duplicate cannot shadow it.
    if not replaced:
        if chunks[-1] and not chunks[-1].endswith(("\n", "\r", "\ufeff")):
            chunks.append(newline)
        chunks.append(assignment)
    result = "".join(chunks).encode("utf-8")
    if len(result) > _MAX_FILE_BYTES:
        raise _CredentialError("credential_file_too_large", "The saved dotenv file would exceed the 1 MiB limit.")
    return result


def _write(path: Path, before: _Snapshot, data: bytes) -> None:
    # Serialize cooperating AUA saves only during commit, never while the person
    # types. The snapshot also detects ordinary editor changes during the dialog.
    lock = path.with_name(path.name + ".aua-lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise _CredentialError("credential_save_busy", "A credential save lock exists beside this file. Wait for active saves; if a save was interrupted, see docs/credentials.md for recovery.") from None
    temporary: str | None = None
    try:
        os.close(lock_fd)
        if _snapshot(path) != before:
            raise _CredentialError("credential_file_changed", "The dotenv file changed while the dialog was open; retry to preserve those edits.")
        fd, temporary = tempfile.mkstemp(prefix=".aua-credential-", dir=path.parent)
        with os.fdopen(fd, "wb") as target:
            if hasattr(os, "fchmod"):
                os.fchmod(target.fileno(), 0o600)
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        if _snapshot(path) != before:
            raise _CredentialError("credential_file_changed", "The dotenv file changed during saving; retry to preserve those edits.")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
        with contextlib.suppress(OSError):
            lock.unlink()


def _dialog(name: str, path: Path, timeout_s: int) -> tuple[str, str]:
    if sys.platform == "darwin":
        command = ["/usr/bin/osascript", "-", name, str(path), str(timeout_s)]
        script: str | None = _MAC_DIALOG
    else:
        command = [sys.executable, "-c", _TK_DIALOG, name, str(path), str(timeout_s)]
        script = None
    try:
        child = subprocess.run(command, input=script, capture_output=True, text=True,
                               encoding="utf-8", timeout=timeout_s + 10, check=False)
    except subprocess.TimeoutExpired:
        raise _CredentialError("credential_dialog_timeout", "The credential dialog timed out; nothing was saved.") from None
    except (OSError, UnicodeError):
        raise _CredentialError("credential_dialog_unavailable", "A local desktop dialog is unavailable. Run this command on your desktop; macOS uses its native dialog, other systems need Python Tk.") from None
    if child.returncode != 0:
        raise _CredentialError("credential_dialog_unavailable", "The local dialog could not open. Run it in an interactive desktop session; nothing was saved.")
    output = child.stdout.removesuffix("\n")
    status, separator, value = output.partition(":")
    if separator and status in {"saved", "cancelled"}:
        return status, value if status == "saved" else ""
    if status == "timeout":
        raise _CredentialError("credential_dialog_timeout", "The credential dialog timed out; nothing was saved.")
    raise _CredentialError("credential_dialog_unavailable", "A local desktop dialog is unavailable. Run on a desktop with native macOS dialogs or Python Tk.")


def request_secret(name: str, env_file: str | Path = ".env", *,
                   replace: bool = False, timeout_s: int = 300) -> dict[str, Any]:
    """Ask a person privately and save dotenv data; return no credential value.

    An existing nonempty binding skips the GUI unless replace=True. The process
    environment is neither inspected nor changed. Cancel and timeout write nothing.
    The destination parent must exist; its resolved path is shown in the dialog.
    """
    result: dict[str, Any] = {"ok": False, "status": "error"}
    try:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise _CredentialError("credential_invalid_name", "Use an environment variable name containing letters, digits and underscores, starting with a letter or underscore.")
        if type(replace) is not bool or type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
            raise _CredentialError("credential_invalid_options", "replace must be boolean and timeout_s must be between 1 and 3600 seconds.")
        target = Path(env_file).expanduser().absolute()
        path = target.parent.resolve(strict=True) / target.name
        result.update(name=name, env_file=str(path))
        if not path.parent.is_dir():
            raise _CredentialError("credential_invalid_path", "The destination parent must be an existing directory.")
        before = _snapshot(path)
        _, records = _bindings(before)
        values = [record.value for record in records if record.key == name]
        if values and values[-1] and values[-1].strip() and not replace:
            return {**result, "ok": True, "status": "already_set"}
        status, value = _dialog(name, path, timeout_s)
        if status == "cancelled":
            return {**result, "status": "cancelled"}
        if not value.strip() or any(char in value for char in "\r\n\0") or len(value.encode("utf-8")) > _MAX_VALUE_BYTES:
            raise _CredentialError("credential_invalid_value", "Enter a nonempty, single-line value of at most 64 KiB; nothing was saved.")
        _write(path, before, _updated(before, name, value))
        return {**result, "ok": True, "status": "saved"}
    except _CredentialError as exc:
        return {**result, "error": {"code": exc.code, "message": exc.message}}
    except (OSError, ValueError, TypeError):
        # Exception text can contain existing file bytes or GUI output. Do not
        # log it, re-raise it, or attach it to a public error result.
        return {**result, "error": {"code": "credential_file_error", "message": "The credential could not be saved. Check the destination path and local file permissions; no credential value is returned."}}
