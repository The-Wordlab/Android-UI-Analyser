"""An agent that cannot re-lease its own device is worse off than one with no lease at all.

`derive_identity()` walked to the first non-shell ancestor and stopped there. Under `uv run aua`
that ancestor is `uv` — a process that exists for exactly one command. Measured 2026-08-10, three
consecutive invocations from a single session:

    uv-37288-5:242026
    uv-37298-5:242026
    uv-37307-5:242026

Three names, one agent. Command 1 took the lease; command 2 was a stranger to it and got
`device_leased` naming a holder that no longer existed, for the full 900s TTL. Every runner that
wraps the CLI — uvx, pipx, poetry, npx, nix, and plain `env`/`sudo` — has the same shape, so the
walk skips the whole family and keeps climbing to whoever actually persists.
"""

from __future__ import annotations

import pytest

import android_ui_analyser.leases as leases


def _ancestry(monkeypatch: pytest.MonkeyPatch, chain: list[tuple[int, str]]) -> None:
    """Install a fake process tree: chain[0] is this process, each entry is (pid, comm)."""
    pids = dict(chain)
    parents = {chain[i][0]: chain[i + 1][0] for i in range(len(chain) - 1)}

    monkeypatch.setattr(leases.os, "getpid", lambda: chain[0][0])
    monkeypatch.setattr(leases, "_proc_ppid", lambda pid: parents.get(pid))
    monkeypatch.setattr(leases, "_proc_name", lambda pid: pids.get(pid, ""))
    monkeypatch.setattr(leases, "_proc_started", lambda pid: "start")


def test_the_launcher_is_not_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    _ancestry(monkeypatch, [(100, "aua"), (200, "uv"), (300, "zsh"), (400, "codex"), (1, "init")])

    assert leases.derive_identity() == "codex-400-start"


def test_two_invocations_of_one_agent_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different launcher pids, same agent: the name must not move."""
    _ancestry(monkeypatch, [(100, "aua"), (201, "uv"), (300, "zsh"), (400, "codex"), (1, "init")])
    first = leases.derive_identity()

    _ancestry(monkeypatch, [(101, "aua"), (202, "uv"), (300, "zsh"), (400, "codex"), (1, "init")])

    assert leases.derive_identity() == first


@pytest.mark.parametrize(
    "wrapper", ["uvx", "pipx", "poetry", "npx", "nix", "env", "sudo", "timeout"]
)
def test_every_wrapper_in_the_family_is_skipped(
    monkeypatch: pytest.MonkeyPatch, wrapper: str
) -> None:
    _ancestry(monkeypatch, [(100, "aua"), (200, wrapper), (400, "claude"), (1, "init")])

    assert leases.derive_identity() == "claude-400-start"


def test_a_real_agent_is_still_claimed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The skip list must not swallow the thing we are trying to name."""
    _ancestry(monkeypatch, [(100, "aua"), (400, "node"), (1, "init")])

    assert leases.derive_identity() == "node-400-start"


def test_a_human_at_a_terminal_falls_back_to_themselves(monkeypatch: pytest.MonkeyPatch) -> None:
    _ancestry(monkeypatch, [(100, "aua"), (200, "uv"), (300, "zsh"), (1, "init")])

    assert leases.derive_identity() == "pid-100-start"
