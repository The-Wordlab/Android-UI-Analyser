"""Two refusals that named no way forward, both hit by the tool's own author.

1. `wait-and-analyze --until 'text:X'` answered "wait needs --for <text> or --idle". `--until`
   is the global every action takes, it parsed fine, and it names exactly what the caller is
   waiting for — in a richer predicate language than `--for` (`!text:`, `rid:`, comma-separated
   terms). Refusing it reads as "`--until` is not a thing here", which is false.

2. The no-free-device refusal advised "wait, start another emulator, or widen --needs". "Wait"
   is not actionable without saying how long, and `lease release` — the command that actually
   frees a held device, and which needs the *holder's* `--owner` to do it — went unmentioned,
   so it stayed invisible. Its sibling branch was fixed earlier the same day; this one kept the
   old text and stranded the same caller three more times in one session.
"""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.errors import DeviceLeasedError
from android_ui_analyser.leases import DEFAULT_TTL_S, acquire, choose_device


def _no_free_hint(tmp_path: Any, *, needs: list[str] | None = None) -> str:
    acquire(tmp_path, "emulator-5554", owner="someone-else:1", ttl_s=DEFAULT_TTL_S)

    with pytest.raises(DeviceLeasedError) as caught:
        choose_device(
            tmp_path,
            owner="me:1",
            explicit=None,  # no --serial: this is the auto-pick branch
            candidates=[("emulator-5554", {})],
            needs=needs,
        )
    return caught.value.hint or ""


def test_the_refusal_names_the_command_that_frees_a_device(tmp_path: Any) -> None:
    hint = _no_free_hint(tmp_path)

    assert "lease release" in hint, "the one move that always works was never mentioned"
    assert "--owner someone-else:1" in hint, "release needs the holder's name, not yours"


def test_the_refusal_says_how_long_waiting_takes(tmp_path: Any) -> None:
    hint = _no_free_hint(tmp_path)

    assert "idle_s" in hint, "`aua lease list` is how you watch it count down"
    assert str(DEFAULT_TTL_S) in hint, '"wait" is not actionable without a number'


def test_the_refusal_still_offers_your_own_device(tmp_path: Any) -> None:
    assert "aua emulator start" in _no_free_hint(tmp_path)


def test_widening_needs_is_only_suggested_when_needs_were_given(tmp_path: Any) -> None:
    assert "--needs" not in _no_free_hint(tmp_path)


def test_widening_needs_is_suggested_when_they_were(tmp_path: Any) -> None:
    assert "--needs" in _no_free_hint(tmp_path, needs=["root"])
