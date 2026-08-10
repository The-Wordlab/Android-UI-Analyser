"""Crash-free atomic file replacement for state two processes both write.

Every writer here used ``path.with_suffix(".tmp")`` — one temp name shared by the daemon, the
CLI, and every parallel agent. `os.replace` is atomic, so nobody ever read a torn file; but the
*second* writer's rename found its scratch file already consumed by the first:

    goto → internal_error
    [Errno 2] No such file or directory:
      '…/state/session_emulator-5554.json.tmp' -> '…/state/session_emulator-5554.json'

Measured 2026-08-10 on a live agent run: a crash on the agent's very first `goto`, which then
left the session cursor pointing at a screen the device had since left.

Giving each writer its own scratch file keeps last-writer-wins and removes the collision.
"""

from __future__ import annotations

import contextlib
import os
from itertools import count
from pathlib import Path

_SEQ = count()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace *path* with *text*, never leaving a partial file and never racing a peer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{next(_SEQ)}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
