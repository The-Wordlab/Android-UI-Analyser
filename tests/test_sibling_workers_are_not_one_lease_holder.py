"""One process, several agents: a lease held by one must not read as another's.

Measured 2026-09-01, running three QA scenario workers as subagents of one `claude` process.
`_derived_owner` walks up to the first ancestor that outlives a command, which for every in-process
subagent is that same shared process, so all three resolved to the identical owner
`claude-1708-2:242026`. `select` then handed two of them `emulator-5554` through its sticky branch
and they drove one screen for six overlapping minutes (cache writes 18:10:21-18:15:43 against
18:09:18-18:17:56).

**How it surfaced is why this test exists.** Nothing errored. Worker 2 completed setup, proved the
build device-side, reached the screen it needed, then watched `DevToolsActivity` appear on top of its
own task with no command of its own — the sibling's `app launch`. It reported BLOCKED and called the
cause "unexplained external interference", because `aua lease list` said the lease was `mine: true`.
That answer was true and useless: under a shared owner the one lease file is "mine" to every sibling
at once, so the tool could not distinguish "mine alone" from "mine and my sibling's".

**The discriminator goes in the lease record, not in the owner label.** A first attempt folded the
run-cache digest into the owner string itself and broke eleven tests in
`test_an_agent_keeps_one_name_across_commands.py`, because `tests/conftest.py` sets `AUA_CACHE__DIR`
for every test — which is the real lesson: an explicit run cache is far more common than "a parallel
worker", so it must not rewrite an identity that other machinery reads back. Recording it as a field
leaves every owner string byte-identical and every existing lease valid.

Not to be confused with `LeaseCfg.registry_dir`, which stays deliberately independent of
`cache.dir`. That is lease *storage*: if the registry followed the override, two agents would each
read an empty registry and see the same device as free. Matching is the opposite case — two callers
with different run caches are, by the parallel-harness contract, two different callers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser import leases  # noqa: E402

OWNER = "claude-1708-2:242026"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    monkeypatch.delenv("AUA_OWNER", raising=False)


def _entry(scope: str | None) -> dict[str, object]:
    entry: dict[str, object] = {"serial": "emulator-5554", "owner": OWNER}
    if scope is not None:
        entry["scope"] = scope
    return entry


def _scope_for(monkeypatch: pytest.MonkeyPatch, cache_dir: str) -> str:
    monkeypatch.setenv("AUA_CACHE__DIR", cache_dir)
    return leases._worker_scope()


def test_a_siblings_lease_is_not_mine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug, directly: same owner label, different run cache, must not match."""

    theirs = _entry(_scope_for(monkeypatch, "/tmp/run/w3-tools"))
    monkeypatch.setenv("AUA_CACHE__DIR", "/tmp/run/w2-media")

    assert not leases._entry_matches_owner(theirs, OWNER), (
        "a sibling worker's lease reads as this worker's, so `select` returns it through the "
        "sticky branch and both drive one screen"
    )


def test_my_own_lease_is_still_mine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lease must survive the same worker's next command, or it can never be renewed."""

    mine = _entry(_scope_for(monkeypatch, "/tmp/run/w2-media"))
    assert leases._entry_matches_owner(mine, OWNER)


def test_the_owner_label_is_left_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first attempt at this changed the label and broke eleven tests that read it back.

    `test_an_agent_keeps_one_name_across_commands` pins the label precisely, and lease expiry parses
    the pid and start time out of the stored string. Neither may notice this feature exists.
    """

    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    plain = str(leases.derive_identity())
    monkeypatch.setenv("AUA_CACHE__DIR", "/tmp/run/w2-media")
    scoped = str(leases.derive_identity())
    assert plain == scoped, f"the run cache leaked into the owner label: {scoped!r}"


def test_a_scope_is_compared_by_path_and_not_by_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One worker naming its own directory two ways is one worker."""

    assert _scope_for(monkeypatch, "/tmp/run/w1") == _scope_for(monkeypatch, "/tmp/run/w1/")


def test_an_unscoped_caller_keeps_a_legacy_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary case — one agent, no override — must keep leases written before this existed."""

    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    assert leases._entry_matches_owner(_entry(None), OWNER)


def test_a_scoped_caller_treats_a_legacy_lease_as_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lease written before scoping existed cannot be attributed to one sibling over another, so
    a scoped worker must not claim it. It picks another device instead, and the orphan ages out with
    its process. Wasting a device once on upgrade beats two workers sharing one screen."""

    monkeypatch.setenv("AUA_CACHE__DIR", "/tmp/run/w2-media")
    assert not leases._entry_matches_owner(_entry(None), OWNER)


def test_an_explicit_owner_is_still_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """`$AUA_OWNER` replaces the label, which is the caller naming itself — but two workers that
    both forget to vary it are still two workers, so the record's scope still separates them."""

    monkeypatch.setenv("AUA_OWNER", "shared-ci-name")
    theirs = _entry(_scope_for(monkeypatch, "/tmp/run/w3-tools"))
    monkeypatch.setenv("AUA_CACHE__DIR", "/tmp/run/w2-media")
    monkeypatch.setenv("AUA_OWNER", "shared-ci-name")
    assert not leases._entry_matches_owner(theirs, str(leases.resolve_owner(None)))
