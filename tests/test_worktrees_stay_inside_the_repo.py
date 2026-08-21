"""Guard: every registered git worktree lives inside the repository.

A worktree created beside the repo (``../android-ui-analyser-wt-<topic>``) is outside this
repo's ``.gitignore``, so it never shows up in ``git status`` and nobody notices it outliving
the branch it was made for. Three of them once sat in the parent directory holding ~1 GB of
``.venv`` between them, every commit already merged into ``main``.

``.worktree/`` and ``.claude/worktrees/`` are gitignored, which makes them the only acceptable
places to put one. See CLAUDE.md → "If you're DEVELOPING the tool" and AGENTS.md.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def stray_worktrees(main: Path, worktrees: list[Path], tolerated: list[Path]) -> list[Path]:
    """Worktrees registered outside ``main`` and outside every ``tolerated`` root.

    Ephemeral agent worktrees created under the system temp directory are tolerated: they are
    torn down with their session and never accumulate in a human's project directory.
    """
    return [
        wt
        for wt in worktrees
        if not _is_inside(wt, main)
        and not any(_is_inside(wt, root) for root in tolerated)
    ]


def _registered_worktrees() -> list[Path]:
    out = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        Path(line.split(" ", 1)[1]).resolve()
        for line in out.splitlines()
        if line.startswith("worktree ")
    ]


def test_stray_detector_flags_a_sibling_worktree() -> None:
    main = Path("/repos/ai/android-ui-analyser")
    tmp = Path("/var/folders/xx/T")
    found = stray_worktrees(
        main,
        [
            main,
            main / ".worktree" / "some-fix",
            main / ".claude" / "worktrees" / "agent-1",
            tmp / "claude-503" / "agent-worktree",
            Path("/repos/ai/android-ui-analyser-wt-daemon-startup"),
        ],
        [tmp],
    )
    assert found == [Path("/repos/ai/android-ui-analyser-wt-daemon-startup")]


def test_every_worktree_of_this_repo_lives_inside_it() -> None:
    worktrees = _registered_worktrees()
    if not worktrees:
        pytest.skip("not a git checkout")

    # The first porcelain entry is always the main worktree.
    main, *linked = worktrees
    stray = stray_worktrees(main, linked, [Path(tempfile.gettempdir()).resolve()])

    assert not stray, (
        "These worktrees are registered outside the repository, where .gitignore cannot see "
        "them:\n  "
        + "\n  ".join(str(p) for p in stray)
        + "\n\nMove them under .worktree/ instead:\n"
        "  git worktree add .worktree/<slug> <branch>\n"
        "and retire a landed one with:\n"
        "  git worktree remove <path> && git branch -d <branch>\n"
        "A branch whose `git log --oneline main..<branch>` is empty is fully merged — "
        "removing it loses nothing."
    )
