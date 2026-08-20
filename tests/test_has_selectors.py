"""`has` must accept the same one-shot selectors as the action commands.

`has` predates them and took only a positional, so `aua has --rid buttonSettings` exited with a
usage error while `aua tap --rid buttonSettings` worked. A guard loop written the obvious way —
check with `has`, then act with `tap` — failed on every iteration with an empty stdout, which
reads as "not on screen" rather than "wrong flag". Observed: four consecutive cycles reported
"not on home" while the app was, in fact, on home.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.cli import _has_target
from android_ui_analyser.errors import ExitCode, UsageError


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"rid": "buttonSettings"}, ("buttonSettings", "id")),
        ({"text_sel": "SEE MORE"}, ("SEE MORE", "text")),
        ({"desc": "Close"}, ("Close", "desc")),
    ],
)
def test_each_one_shot_selector_resolves(kwargs: dict, expected: tuple[str, str]) -> None:
    args = {"by": "text", "rid": None, "text_sel": None, "desc": None, **kwargs}
    assert _has_target(None, **args) == expected


def test_the_positional_spelling_still_works() -> None:
    """The old form is what every existing flow and doc uses."""
    assert _has_target("Hello", by="text", rid=None, text_sel=None, desc=None) == ("Hello", "text")


def test_positional_with_by_id_still_works() -> None:
    assert _has_target("someId", by="id", rid=None, text_sel=None, desc=None) == ("someId", "id")


def test_no_target_at_all_is_a_usage_error() -> None:
    with pytest.raises(UsageError) as err:
        _has_target(None, by="text", rid=None, text_sel=None, desc=None)
    assert err.value.exit_code == ExitCode.USAGE


def test_mixing_both_spellings_is_refused_with_the_right_flag_named() -> None:
    """The hint has to name --rid, not the internal kind 'id'."""
    with pytest.raises(UsageError) as err:
        _has_target("foo", by="text", rid="bar", text_sel=None, desc=None)
    assert "--rid bar" in (err.value.hint or "")


def test_two_selectors_at_once_are_refused() -> None:
    with pytest.raises(UsageError):
        _has_target(None, by="text", rid="a", text_sel="b", desc=None)
