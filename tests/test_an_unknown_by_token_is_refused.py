"""An unrecognised `by` token silently became a text search, and then lied about the screen.

The half of the matcher divergence left open when `--by rid` was made a synonym. `_fields_for`
resolved `by` with `.get(token, ["text", "description"])`, so anything it did not recognise fell
through to a label search. That is how `aua wait --for containerLogin --by rid` spent a full 15s
timeout hunting for the literal string "containerLogin" in the *text* of a screen where the
container was plainly present, and then reported it absent.

Why refusing beats degrading, specifically: a refusal says "you held it wrong", costs one turn, and
names the fix. A silent text search says "that element is not on this screen" — a claim about the
**product**, arriving next to a screenshot that contradicts it. One lane believed it, spent time on
the wrong hypothesis, and wrote a misleading handoff note off the back of it.

The two surfaces also disagreed about the same token: `tap --by <bogus>` was refused loudly by
`cli._selector`, while `has`/`wait`/`scroll-to` passed the token straight through and degraded. This
closes that, and pins the vocabularies together so they cannot drift apart again.

Deliberately checked here rather than assumed: this changes behaviour for every `has`/`wait`/
`scroll-to` caller, so the whole legitimate vocabulary is enumerated below and must keep working.
Every `by` value anywhere in the repository is one of `id`, `rid`, `text`, `desc` (recording never
writes the field at all, and flow YAML has no `by:` key — the parser sets it only for an `id:`
selector), so nothing that works today can start failing.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.cli import _BY_KINDS
from android_ui_analyser.device import Device, Uiautomator2Device
from android_ui_analyser.errors import ExitCode, UsageError
from conftest import FakeDevice


def _fields(by: str) -> list[str]:
    return Uiautomator2Device._fields_for(object.__new__(Uiautomator2Device), by)


@pytest.mark.parametrize("by", ["id", "rid", "text", "desc", "RID", "Text", "DESC", "", None])
def test_every_legitimate_token_still_resolves(by: str | None) -> None:
    """The no-regression check this change had to earn before it could land."""
    assert _fields(by) , f"by={by!r} must keep working"  # noqa: E203


@pytest.mark.parametrize("by", ["label", "resource-id", "resourceId", "content_desc", "ids", "x"])
def test_an_unrecognised_token_is_refused_not_degraded(by: str) -> None:
    with pytest.raises(UsageError) as err:
        _fields(by)
    assert err.value.exit_code == ExitCode.USAGE
    # The message has to name the token AND the way out, or it just moves the confusion.
    assert by in str(err.value)
    for token in ("id", "rid", "text", "desc"):
        assert token in (err.value.hint or "")


def test_the_refusal_happens_before_the_timeout_is_spent() -> None:
    """The original cost was 15s of waiting followed by a wrong answer about the product.

    A wrong `by` cannot become known-good by waiting, so `wait_for` must fail on its first
    attempt rather than polling out the whole budget and then reporting "not found".
    """
    dev = FakeDevice()
    with pytest.raises(UsageError):
        dev.wait_for("anything", timeout_ms=60_000, by="label")
    assert len(dev.calls) == 1, f"it must not have polled: {dev.calls!r}"


def test_a_legitimate_wait_still_returns_none_rather_than_raising() -> None:
    """Absent is not the same as misspelled, and must stay a `None`, not an exception."""
    dev = FakeDevice()
    assert dev.wait_for("nothing here", timeout_ms=1, by="text") is None


def test_the_two_surfaces_accept_exactly_the_same_vocabulary() -> None:
    """The divergence itself, pinned: one vocabulary behind one spelling.

    `tap --by <bogus>` refused while `wait --by <bogus>` text-searched is what made a runner
    conclude the screen was wrong. Equal sets is the property that keeps them honest; a token
    added to one surface alone fails here rather than in a sweep.
    """
    assert set(_BY_KINDS) == set(Device._BY_FIELDS)


def test_the_test_double_shares_the_vocabulary() -> None:
    """A double that maps `by` itself drifts, and then a real fix looks green without being one.

    `FakeDevice.find_text` used to compare `by == "id"` directly, so it kept answering a *text*
    search for `by="rid"` after the real device had learned the synonym.
    """
    dev = FakeDevice(resource_index={"com.test.app:id/containerDetail": (0, 0, 10, 10)})
    assert dev.find_text("containerDetail", by="rid") == (0, 0, 10, 10)
    assert dev.find_text("containerDetail", by="id") == (0, 0, 10, 10)
    # And it must refuse what the real device refuses.
    with pytest.raises(UsageError):
        dev.find_text("containerDetail", by="resourceId")
