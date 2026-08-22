"""A failed lookup already read the screen, so it must hand that screen back.

`_resolve_action_key` analyzes the current screen to discover the key is absent — that read is
how it knows. It then discarded the observation and told the caller:

    "No action was sent. … re-analyze and use an id from that fresh observation …"

One guaranteed wasted round trip per miss, on the exact path where the caller is already
confused about what is on screen. Worse in the case that motivated this: the screen had changed
underneath the caller (a promotional interstitial appeared on its own and covered the control),
so the answer the caller needed *was* the observation being thrown away.

`SelectorNotFoundError` already carries one and reports `observation_present: true`. This gives
the stable-key and stale-id misses the same treatment, using the read that is already paid for.
"""

from __future__ import annotations

from android_ui_analyser.errors import ElementNotFoundError, SelectorNotFoundError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen

PACKAGE = "com.example.fiction"


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[
            Element(
                id=1,
                type="Button",
                text="Start my predictions",
                resource_id=f"{PACKAGE}:id/onboardingPrimary",
                bounds=[0, 400, 1080, 520],
                center=[540, 460],
                clickable=True,
                enabled=True,
                window="app",
            )
        ],
        meta=Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy"),
    )


def test_a_key_miss_carries_the_screen_it_read() -> None:
    err = ElementNotFoundError(
        "no element with stable_key 'rid:feedTab' on the current screen for tap",
        hint="the screen changed",
        observation=_observation(),
    )

    error = err.to_dict()["error"]
    assert isinstance(error, dict)
    assert error["observation_present"] is True
    assert [e["id"] for e in error["observation"]["elements"]] == ["rid:onboardingPrimary"]


def test_the_returned_screen_is_the_trimmed_form() -> None:
    """No reason for a failure to cost more bytes than a success."""
    error = ElementNotFoundError("gone", observation=_observation()).to_dict()["error"]
    assert isinstance(error, dict)
    row = error["observation"]["elements"][0]
    assert "enabled" not in row, "default flags are dropped, as everywhere else"


def test_a_miss_with_nothing_read_stays_exactly_as_it_was() -> None:
    """Not every raise site has an observation; those must not grow an empty key."""
    error = ElementNotFoundError("gone", hint="run analyze").to_dict()["error"]
    assert isinstance(error, dict)
    assert "observation" not in error and "observation_present" not in error


def test_it_matches_the_selector_error_it_copies() -> None:
    """One shape for "your target is not here", whichever way the target was named."""
    obs = _observation()
    key_miss = ElementNotFoundError("gone", observation=obs).to_dict()["error"]
    sel_miss = SelectorNotFoundError("gone", observation=obs).to_dict()["error"]
    assert isinstance(key_miss, dict) and isinstance(sel_miss, dict)
    assert key_miss["observation"] == sel_miss["observation"]
    assert key_miss["observation_present"] == sel_miss["observation_present"]
