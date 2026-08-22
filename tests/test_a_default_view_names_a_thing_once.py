"""A default view must not spend bytes saying the same thing twice.

Once `id` became the element's stable identity, `rid` in the default columns became mostly a
restatement of it: measured across 40 rows of one real screen, 23 had `id == "rid:" + rid`
exactly, 12 differed only by the screen-uniqueness ordinal (`rid:host#1` beside `host`), and 5
had no `rid` at all. So on that screen the column was pure duplication on well over half the
rows and near-duplication on the rest — the same "two names for one thing" the id change
existed to remove, left behind in the view every caller reads by default.

It is dropped from the *defaults*, not removed. `rid` is still the right thing to ask for and
the right thing to match on, and the two jobs are genuinely different:

* `id` is a **handle** — one element, send it back to act on that element;
* `rid` is a **class** — `--where-rid host` finds every row built from that layout, and
  `--rid continue_btn` names a control without analyzing first.

A query input and an addressing handle. Keeping the flags while dropping the column is what
lets both stay true without paying for the overlap on every row of every response.
"""

from __future__ import annotations

import json

from android_ui_analyser.config import Config
from android_ui_analyser.projection import TSV_DEFAULT_FIELDS, Projection
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, OutputFormat, Screen

PACKAGE = "com.example.fiction"


def _screen() -> AnalyzeResult:
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[
            Element(
                id=1,
                type="Button",
                text="Continue",
                resource_id=f"{PACKAGE}:id/continue_btn",
                bounds=[0, 400, 1080, 520],
                center=[540, 460],
                clickable=True,
                enabled=True,
                window="app",
            ),
            # Two rows from one reusable layout: the ordinal is what tells them apart.
            Element(
                id=2,
                type="Switch",
                resource_id=f"{PACKAGE}:id/row",
                bounds=[0, 600, 1080, 700],
                center=[540, 650],
                clickable=True,
                checkable=True,
                checked=True,
                window="app",
            ),
            Element(
                id=3,
                type="Switch",
                resource_id=f"{PACKAGE}:id/row",
                bounds=[0, 700, 1080, 800],
                center=[540, 750],
                clickable=True,
                checkable=True,
                checked=False,
                window="app",
            ),
        ],
        meta=Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy"),
    )


def _payload() -> dict:
    return _screen().as_dict(OutputFormat.json)


# ------------------------------------------------------------------- rid is not a default


def test_the_tsv_default_columns_do_not_repeat_the_id() -> None:
    assert "rid" not in TSV_DEFAULT_FIELDS
    assert "id" in TSV_DEFAULT_FIELDS, "the handle is the one column that must always be there"


def test_the_observation_default_columns_do_not_repeat_the_id() -> None:
    assert "rid" not in Config().output.observation_fields.split(",")


def test_the_default_observation_row_carries_no_rid() -> None:
    view = Projection.for_observation(Config().output.observation_fields, meta="changed")
    assert view is not None

    rows = view.apply(_payload())["elements"]

    assert rows, "the fixture must produce rows"
    assert not any("rid" in row for row in rows), f"rid is still in the default view: {rows}"


# ------------------------------------------------------------ but nothing lost a capability


def test_asking_for_rid_still_returns_it() -> None:
    view = Projection.parse(fmt=OutputFormat.json, fields="id,rid")

    rows = view.apply(_payload())["elements"]

    assert [r["rid"] for r in rows] == ["continue_btn", "row", "row"]


def test_matching_on_rid_still_works_without_the_column() -> None:
    """`--where-rid` reads the full payload, so it never depended on the projected columns."""
    view = Projection.parse(fmt=OutputFormat.json, fields="id", where_rid=["row"])

    rows = view.apply(_payload())["elements"]

    assert len(rows) == 2, "the class-level query is what rid is for, and it must still match"
    assert all("rid" not in row for row in rows)


def test_the_id_still_tells_the_two_rows_apart() -> None:
    """Dropping the column is only safe because the ordinal survives in the id."""
    view = Projection.for_observation(Config().output.observation_fields, meta="changed")
    assert view is not None

    ids = [r["id"] for r in view.apply(_payload())["elements"]]

    assert len(set(ids)) == len(ids), f"rows became indistinguishable: {ids}"
    assert any("#" in str(i) for i in ids), "the reusable layout must still be numbered"


def test_the_default_view_is_smaller_for_it() -> None:
    with_rid = Projection.for_observation(
        "id,text,desc,rid,clickable,enabled,checked,selected", meta="changed"
    )
    without = Projection.for_observation(Config().output.observation_fields, meta="changed")
    assert with_rid is not None and without is not None

    before = len(json.dumps(with_rid.apply(_payload()), separators=(",", ":")))
    after = len(json.dumps(without.apply(_payload()), separators=(",", ":")))

    assert after < before, f"no saving: {after}B vs {before}B"
