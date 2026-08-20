"""Three refusals that cost a call each, from two runs on 2026-08-10.

`aua tree` is not here: it already answered "`aua tree` is not a command. Use `aua analyze`" and
the agent that hit it recovered from that text alone and said so. The other two did not.

1. `aua app` answered `Missing argument 'ACTION'` over a usage line that names the argument again
   and nothing else. The accepted values sit in the parameter's own help and never reached the
   caller, so the only way on was a second call to `--help`. Two separate agents ran `aua app`,
   and both spent that extra command.

2. `--fields id,text,rid,clickable,contentDescription,selected,checkable` was refused whole for
   one word out of seven. `contentDescription` is what the column is called in Android — an agent
   that knows the platform reaches for it before this tool's abbreviation, and it is naming the
   same attribute on the same node. That is an alias, by the same test used for `--fields` →
   `--observe-fields`: two spellings, one concept, one set of data.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.projection import FIELD_ALIASES, Projection, resolve_field_name


@pytest.mark.parametrize(
    ("written", "means"),
    [
        ("contentDescription", "content_desc"),
        ("content-desc", "content_desc"),
        ("CONTENT_DESC", "content_desc"),
        ("resourceId", "resource_id"),
        ("resource-id", "resource_id"),
        ("longClickable", "long_clickable"),
        ("className", "type"),
        ("stableKey", "stable_key"),
    ],
)
def test_the_android_spelling_names_the_same_column(written: str, means: str) -> None:
    assert resolve_field_name(written) == means
    assert resolve_field_name(written) in FIELD_ALIASES


@pytest.mark.parametrize("written", ["isClickable", "isChecked", "isSelected", "isScrollable"])
def test_a_java_style_boolean_prefix_is_dropped(written: str) -> None:
    """`isClickable` is how the attribute reads in Java; it is not a different column."""
    assert resolve_field_name(written) in FIELD_ALIASES


def test_the_canonical_spellings_are_untouched() -> None:
    for name in FIELD_ALIASES:
        assert resolve_field_name(name) == name


def test_sparks_exact_command_is_accepted() -> None:
    parsed = Projection._parse_fields("id,text,rid,clickable,contentDescription,selected,checkable")

    assert "content_desc" in parsed
    assert len(parsed) == 7, "the other six were always valid and must survive intact"


def test_a_real_typo_is_still_refused() -> None:
    with pytest.raises(UsageError):
        Projection._parse_fields("id,clikable")


def test_a_real_typo_gets_a_guess_and_not_just_the_whole_list() -> None:
    with pytest.raises(UsageError) as caught:
        Projection._parse_fields("id,clikable")

    assert "did you mean" in (caught.value.hint or "")
    assert "clickable" in (caught.value.hint or "")


def test_nonsense_still_gets_the_full_list_to_choose_from() -> None:
    with pytest.raises(UsageError) as caught:
        Projection._parse_fields("id,zzzqqqxyz")

    assert "Valid names:" in (caught.value.hint or "")
