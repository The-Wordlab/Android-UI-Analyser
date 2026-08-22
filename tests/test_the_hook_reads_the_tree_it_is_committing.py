"""The pre-commit hook must generate from, and gate, the worktree being committed.

This repo works in worktrees under `.worktree/`, and a linked worktree has no `.venv` of its
own. The hook resolved its tooling by looking for `.venv/bin/…` relative to the worktree and
falling back to whatever was on `PATH`, which produced two silent failures at once:

* `aua` on `PATH` is the *installed* CLI, whose `src` is the **main checkout** — so committing
  from a worktree regenerated SKILL.md from a different tree's `guide.py`. When the main
  checkout happened to sit several commits behind, the hook quietly stamped stale agent
  guidance over correct guidance, inside a commit about something else. That is how
  `SKILL.md` ended up telling agents an observation's ids "belong to that frame only" after
  ids had become stable identities.
* `ruff` and `mypy` are the only gate this repo has, and both are guarded by
  `[ -x ".venv/bin/… ]`. In a worktree that path does not exist, so the gate skipped
  entirely and the commit reported nothing — the failure mode is a commit that looks checked.

Both come from the same mistake: locating tools by a path relative to the *worktree* while
needing an interpreter that only the main checkout has. The fix keeps the two apart — the
interpreter comes from the main checkout, the source and the files to check come from the
worktree — so this file asserts on the hook text, because a hook that resolves the wrong tree
produces a passing commit and no error to catch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / ".githooks/pre-commit"


@pytest.fixture(scope="module")
def hook() -> str:
    return HOOK.read_text(encoding="utf-8")


def test_the_hook_exists(hook: str) -> None:
    assert hook.startswith("#!"), "the hook must be an executable script"


def test_generation_pins_the_source_to_this_worktree(hook: str) -> None:
    """`PYTHONPATH` naming the worktree's own `src` is what stops the wrong tree being read."""
    assert re.search(r'PYTHONPATH="?\$\{?root\}?/src', hook), (
        "generation must run against $root/src — the worktree being committed — not against "
        "whatever tree the `aua` on PATH was installed from"
    )


def test_generation_never_falls_back_to_the_installed_cli(hook: str) -> None:
    """The installed `aua` reads the main checkout, which is a different commit."""
    assert "command -v aua" not in hook, (
        "`aua` on PATH belongs to the main checkout; using it to regenerate a worktree's "
        "SKILL.md is what silently reintroduced stale agent guidance"
    )


def test_the_interpreter_may_come_from_the_main_checkout(hook: str) -> None:
    """A linked worktree has no `.venv`, so the interpreter has to be found elsewhere."""
    assert "git rev-parse --git-common-dir" in hook, (
        "without the common git dir there is no way to find the main checkout's venv, which "
        "is the only one with the dependencies installed"
    )


def test_the_lint_and_type_gate_is_reachable_from_a_worktree(hook: str) -> None:
    """`[ -x ".venv/bin/ruff" ]` is false in every worktree, so the gate skipped silently."""
    for tool in ("ruff", "mypy"):
        assert f'-x ".venv/bin/{tool}"' not in hook, (
            f"guarding {tool} on a worktree-relative .venv path skips the gate in every "
            "worktree, and a skipped gate reports success"
        )


def test_the_gate_still_blocks(hook: str) -> None:
    """Skipping when a tool is genuinely absent is fine; passing when it fails is not."""
    assert hook.count("exit 1") >= 2, "ruff and mypy must each still fail the commit"


def test_generated_paths_are_still_staged(hook: str) -> None:
    """Regenerating without staging would leave the drift in the working tree instead."""
    assert "git add --" in hook
