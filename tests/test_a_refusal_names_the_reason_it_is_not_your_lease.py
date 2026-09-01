"""When the owner label matches, the refusal has to say the scope is what differs.

Scoping a lease by its worker's run cache (see
`test_sibling_workers_are_not_one_lease_holder`) fixed two agents driving one screen. It also
made a refusal that reads as a contradiction: measured on 2026-09-01, `aua record start
--serial emulator-5556` refused with

    emulator-5556 is leased by rec-probe (active 539s ago)
    … You are `rec-probe`: pass `--owner rec-probe` only if that holder is you under another
    name, or `aua lease release emulator-5556 --force` …

The caller *is* `rec-probe`. The advice it was given — pass the name it already has — cannot
work, so the only route left in the message is `--force`, which is the one move that is wrong
here: the sibling may be alive. What actually differed was `AUA_CACHE__DIR`, and nothing in
the message mentioned it. A hint that names the wrong difference sends the caller at the
wrong remedy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser import leases  # noqa: E402
from android_ui_analyser.errors import DeviceLeasedError  # noqa: E402

SERIAL = "emulator-5556"
OWNER = "rec-probe"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUA_OWNER", raising=False)


def _hold(cache_dir: Path, monkeypatch: pytest.MonkeyPatch, *, owner: str, scope_dir: str) -> None:
    """Write a lease as it would look if a sibling under *scope_dir* had claimed the device."""
    monkeypatch.setenv("AUA_CACHE__DIR", scope_dir)
    leases.acquire(cache_dir, SERIAL, owner=owner)


def _refuse(cache_dir: Path, monkeypatch: pytest.MonkeyPatch, *, owner: str) -> DeviceLeasedError:
    with pytest.raises(DeviceLeasedError) as refusal:
        leases._choose_device_unlocked(
            cache_dir,
            owner=owner,
            explicit=SERIAL,
            candidates=[(SERIAL, {})],
        )
    return refusal.value


def test_a_same_name_different_scope_refusal_names_the_run_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured message: it must not tell the caller to pass the name it already has."""
    cache = tmp_path / "registry"
    _hold(cache, monkeypatch, owner=OWNER, scope_dir=str(tmp_path / "w1"))
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "w2"))

    hint = str(_refuse(cache, monkeypatch, owner=OWNER).hint or "")

    assert "AUA_CACHE__DIR" in hint, hint
    assert f"--owner {OWNER}" not in hint, (
        "the hint tells the caller to pass the owner name it is already using"
    )


def test_a_genuinely_different_owner_still_gets_the_owner_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing here weakens the ordinary case: a different name still says so."""
    cache = tmp_path / "registry"
    _hold(cache, monkeypatch, owner="some-other-agent", scope_dir=str(tmp_path / "w1"))
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "w1"))

    hint = str(_refuse(cache, monkeypatch, owner=OWNER).hint or "")

    assert "--owner some-other-agent" in hint, hint
    assert "AUA_CACHE__DIR" not in hint, hint


def test_the_message_itself_still_names_the_holder_and_its_idle_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-line message is what agents log; it keeps naming who holds the device."""
    cache = tmp_path / "registry"
    _hold(cache, monkeypatch, owner=OWNER, scope_dir=str(tmp_path / "w1"))
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "w2"))

    message = str(_refuse(cache, monkeypatch, owner=OWNER))

    assert SERIAL in message and OWNER in message


def test_the_refusal_is_unchanged_when_no_scope_is_in_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No run-cache override anywhere: the legacy shape of the hint is untouched."""
    cache = tmp_path / "registry"
    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    leases.acquire(cache, SERIAL, owner="some-other-agent")

    hint = str(_refuse(cache, monkeypatch, owner=OWNER).hint or "")

    assert "--owner some-other-agent" in hint
    assert "AUA_CACHE__DIR" not in hint
