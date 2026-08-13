"""Crash-free atomic file publication for state two processes both write.

Every writer here used ``path.with_suffix(".tmp")`` — one temp name shared by the daemon, the
CLI, and every parallel agent. `os.replace` is atomic, so nobody ever read a torn file; but the
*second* writer's rename found its scratch file already consumed by the first:

    goto → internal_error
    [Errno 2] No such file or directory:
      '…/state/session_emulator-5554.json.tmp' -> '…/state/session_emulator-5554.json'

Measured 2026-08-10 on a live agent run: a crash on the agent's very first `goto`, which then
left the session cursor pointing at a screen the device had since left.

Giving each writer its own scratch file keeps last-writer-wins and removes the collision.
For create-only files, publishing that completed scratch file with a hard link also gives us
``O_EXCL``-style no-clobber semantics without ever exposing a partially written destination.
"""

from __future__ import annotations

import contextlib
import os
from itertools import count
from pathlib import Path

_SEQ = count()


def _scratch_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{next(_SEQ)}.tmp")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace *path* with *text*, never leaving a partial file and never racing a peer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _scratch_path(path)
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_create_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Create *path* atomically, raising :class:`FileExistsError` rather than replacing it.

    The scratch file lives beside the destination, so linking it into place is an atomic,
    same-filesystem publish.  Unlike opening the destination with ``"x"``, readers cannot see
    the name until all of *text* has been written successfully.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _scratch_path(path)
    try:
        tmp.write_text(text, encoding=encoding)
        os.link(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
