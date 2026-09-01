"""`--by id` must accept the element id AUA itself printed.

Measured 2026-09-01 (`mini-apps-nutrition`, run `w1-miniapps-r2`). The Nexo mini-app composer
has no resource-id, so observations publish it under a pixel-signature id. The bare positional
form works:

    aua input-and-analyze "px:EditText:0005252425040400" "text"      # → ok

and the flag whose name matches the thing being passed does not:

    aua input-and-analyze --by id "px:EditText:0005252425040400" "text"
    → {"error": {"code": "selector_not_found",
                 "message": "no element matches rid:px:EditText:0005252425040400 …"}}

`--by id` was read as "resource-id, always", so a published id became a resource-id lookup for
a resource-id that cannot exist. It fails loudly, so it costs a round trip rather than a wrong
verdict — but the same string being accepted one way and refused the other is the part that
makes it a trap rather than a limitation.

A published id is recognisable by its minted prefix (`px:`, `geo:`, `cd:`, `tx:`); anything
else under `--by id` is still a resource-id, bare tail included, which is what that flag is
for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser import cli  # noqa: E402


@pytest.mark.parametrize(
    "published",
    [
        "px:EditText:0005252425040400",
        "geo:EditText:q51:458af95f22",
        "cd:9f2c1ab077",
        "tx:0b71de55aa",
        "px:EditText:0005252425040400#2",
    ],
)
def test_a_published_id_under_by_id_is_the_key_it_looks_like(published: str) -> None:
    """The measured failure: it became `rid:px:…`, a resource-id that cannot exist."""
    selector = cli._selector(ident=published, by="id")
    assert selector == {"key": published}, selector


@pytest.mark.parametrize("published", ["px:EditText:0005252425040400", "geo:X:q1:abc0123456"])
def test_the_bare_and_flagged_spellings_agree(published: str) -> None:
    """The trap was that one worked and the other did not; they must now be the same."""
    assert cli._selector(ident=published, by="id") == cli._selector(ident=published)


@pytest.mark.parametrize(
    ("ident", "expected_rid"),
    [
        ("homeTabBROWSE", "homeTabBROWSE"),
        ("rid:continue_btn", "continue_btn"),
        ("com.example:id/continue_btn", "com.example:id/continue_btn"),
        ("id:continue_btn", "continue_btn"),
    ],
)
def test_an_ordinary_resource_id_under_by_id_is_untouched(ident: str, expected_rid: str) -> None:
    """`--by id` keeps meaning resource-id — bare tail, prefixed, or fully qualified."""
    selector = cli._selector(ident=ident, by="id")
    assert selector is not None
    assert selector.get("rid") == expected_rid, selector
    assert "key" not in selector


def test_by_text_is_never_reinterpreted() -> None:
    """A label that happens to look like a key is still a label under `--by text`."""
    selector = cli._selector(ident="px:EditText:0005252425040400", by="text")
    assert selector is not None
    assert selector.get("text") == "px:EditText:0005252425040400"
    assert "key" not in selector
