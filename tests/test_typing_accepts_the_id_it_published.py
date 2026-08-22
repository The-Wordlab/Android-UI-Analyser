"""`input-and-analyze <published-id> "text"` must work, like every other action.

`input` is the one action with two positionals — the field and the text — and it decides which
is which by asking whether a *selector* was built. That test used to be equivalent to "a
`--rid`/`--desc` flag was passed", so a bare positional meant an integer id and the text was
the second argument.

Publishing stable ids broke it. A bare non-numeric positional now legitimately builds a key
selector, so `aua input-and-analyze rid:searchField "Brazil"` took `rid:searchField` as the
**text to type** and then refused `"Brazil"` as an unexpected extra argument:

    with --rid/--desc, pass only the text to type

Found by using the tool: the failing form is the exact one the command's own docstring
promises ("with a plain id the first positional addresses the field and the second is the
text") and the exact shape every payload now hands the caller.

The distinction the code needs is not "is there a selector" but **where the selector came
from**: a flag (`--rid`/`--desc`/`--key`) leaves one positional, which is the text; a bare
positional consumes the first, so the text is the second.
"""

from __future__ import annotations

from android_ui_analyser import cli


def _typed(first: str | None, second: str | None, **kw: object) -> str | None:
    """What `input-and-analyze` would type, given these arguments."""
    selector = cli._selector(ident=first, **kw)  # type: ignore[arg-type]
    # `--by` consumes the first positional, so it is not one of the text-leaving flags.
    from_flag = any(kw.get(name) is not None for name in ("rid", "desc", "key"))
    return cli._input_text_argument(first, second, selector=selector, from_flag=from_flag)


def test_a_published_id_addresses_the_field_and_the_second_argument_is_typed() -> None:
    assert _typed("rid:searchField", "Brazil") == "Brazil"


def test_a_numeric_id_still_addresses_the_field() -> None:
    assert _typed("9", "hello") == "hello"


def test_a_rid_flag_still_leaves_the_lone_positional_as_the_text() -> None:
    assert _typed("hello", None, rid="promptField") == "hello"


def test_a_desc_flag_still_leaves_the_lone_positional_as_the_text() -> None:
    assert _typed("hello", None, desc="Prompt") == "hello"


def test_by_id_consumes_the_first_positional() -> None:
    assert _typed("promptField", "hello", by="id") == "hello"


def test_a_stable_id_is_not_mistaken_for_the_text() -> None:
    """The regression in one line: the id must never end up in the text field."""
    assert _typed("rid:searchField", "Brazil") != "rid:searchField"
