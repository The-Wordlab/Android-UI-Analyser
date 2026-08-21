"""A ``stable_key`` names exactly one element on the screen it was published with.

``stable_key`` deliberately carries no exact position — that is what lets it survive a
re-analyze. The cost of that was ambiguity: a reusable row layout hands every row the same
resource-id, so a list of ten chat rows published *one* key ten times, and a caller holding
that key could not say which row it meant. Disambiguating after the fact needs the bounds the
caller happened to be given, which turns "here is the name of a thing" into "here is a name
plus a coordinate, good luck".

So the key is made unique where it is assigned: colliding keys get an ordinal suffix, ordered
by position on screen, and ``rid:row#2`` is the third row from the top whatever the tree
order was.

Two rules this file exists to hold:

* Uniqueness must be **additive**. A key that does not collide is byte-identical to what it
  always was, so nothing that stored one is invalidated by this.
* A bare key must still find the whole group. A caller that saved ``rid:row`` before the
  suffix existed — or that legitimately means "the row, disambiguate it for me" — gets every
  row back and the existing bounds logic decides, rather than a silent miss. A *suffixed*
  key is exact and returns one.
"""

from __future__ import annotations

from android_ui_analyser.identity import (
    KEY_ORDINAL_SEP,
    attach_stable_keys,
    base_stable_key,
    find_by_stable_key,
    stable_key,
)
from android_ui_analyser.schema import Element

PACKAGE = "com.example.fiction"


def _row(element_id: int, y: int, *, rid: str | None = None, text: str | None = None) -> Element:
    return Element(
        id=element_id,
        type="Button",
        text=text,
        resource_id=f"{PACKAGE}:id/{rid}" if rid else None,
        bounds=[40, y, 1040, y + 100],
        center=[540, y + 50],
        clickable=True,
        enabled=True,
        window="app",
    )


# ------------------------------------------------------------------ the collision itself


def test_a_repeated_resource_id_yields_distinct_keys() -> None:
    rows = [_row(1, 200, rid="row"), _row(2, 400, rid="row"), _row(3, 600, rid="row")]

    keys = [el.stable_key for el in attach_stable_keys(rows)]

    assert len(set(keys)) == 3, f"still ambiguous: {keys}"


def test_the_suffix_follows_position_on_screen_not_tree_order() -> None:
    """A reader that sees `#2` must be able to count to it down the screen."""
    # Deliberately handed to the function bottom-first.
    rows = [_row(3, 600, rid="row"), _row(1, 200, rid="row"), _row(2, 400, rid="row")]

    by_id = {el.id: el.stable_key for el in attach_stable_keys(rows)}

    assert by_id[1].endswith(f"{KEY_ORDINAL_SEP}1")
    assert by_id[2].endswith(f"{KEY_ORDINAL_SEP}2")
    assert by_id[3].endswith(f"{KEY_ORDINAL_SEP}3")


def test_the_order_survives_a_reanalyze() -> None:
    """Same screen read twice must produce the same key for the same row."""
    first = {el.id: el.stable_key for el in attach_stable_keys(
        [_row(1, 200, rid="row"), _row(2, 400, rid="row")]
    )}
    # A re-analyze renumbers the integer ids; the keys must not move with them.
    second = {el.id: el.stable_key for el in attach_stable_keys(
        [_row(7, 200, rid="row"), _row(8, 400, rid="row")]
    )}

    assert first[1] == second[7]
    assert first[2] == second[8]


def test_repeated_labels_are_disambiguated_too() -> None:
    rows = [_row(1, 200, text="Retry"), _row(2, 400, text="Retry")]

    keys = [el.stable_key for el in attach_stable_keys(rows)]

    assert len(set(keys)) == 2, f"still ambiguous: {keys}"


# ------------------------------------------------------------------------ additive-ness


def test_a_key_that_does_not_collide_is_byte_identical() -> None:
    """No suffix on a unique element: anything that stored a key keeps working."""
    rows = [_row(1, 200, rid="continue_btn"), _row(2, 400, rid="cancel_btn")]

    attached = attach_stable_keys(rows)

    assert [el.stable_key for el in attached] == [stable_key(el) for el in rows]
    assert all(KEY_ORDINAL_SEP not in (el.stable_key or "") for el in attached)


def test_the_separator_cannot_occur_in_a_resource_id() -> None:
    """`_1` was rejected: an app may legitimately name a view `row_1`.

    The suffix has to be impossible to confuse with the name it is attached to, or
    disambiguating one screen quietly mis-addresses another.
    """
    assert KEY_ORDINAL_SEP not in "abcdefghijklmnopqrstuvwxyz0123456789_"
    natural = attach_stable_keys([_row(1, 200, rid="row_1"), _row(2, 400, rid="row_2")])
    assert all(KEY_ORDINAL_SEP not in (el.stable_key or "") for el in natural)


def test_base_stable_key_strips_the_ordinal() -> None:
    assert base_stable_key(f"rid:row{KEY_ORDINAL_SEP}2") == "rid:row"
    assert base_stable_key("rid:row") == "rid:row"
    assert base_stable_key(None) is None


# ---------------------------------------------------------------------------- lookup


def test_a_suffixed_key_addresses_exactly_one_element() -> None:
    rows = attach_stable_keys([_row(1, 200, rid="row"), _row(2, 400, rid="row")])
    wanted = next(el for el in rows if el.id == 2)

    hits = find_by_stable_key(rows, wanted.stable_key or "")

    assert [el.id for el in hits] == [2]


def test_a_bare_key_still_returns_the_whole_group() -> None:
    """Backward compatible on purpose: a saved pre-suffix key must not become a silent miss."""
    rows = attach_stable_keys([_row(1, 200, rid="row"), _row(2, 400, rid="row")])

    hits = find_by_stable_key(rows, "rid:row")

    assert [el.id for el in hits] == [1, 2], "a bare key must stay ambiguous, not miss"


def test_a_bare_key_still_finds_a_unique_element() -> None:
    rows = attach_stable_keys([_row(1, 200, rid="continue_btn"), _row(2, 400, rid="row")])

    assert [el.id for el in find_by_stable_key(rows, "rid:continue_btn")] == [1]


def test_an_absent_key_is_still_a_miss() -> None:
    rows = attach_stable_keys([_row(1, 200, rid="row")])

    assert find_by_stable_key(rows, "rid:nothing_here") == []
