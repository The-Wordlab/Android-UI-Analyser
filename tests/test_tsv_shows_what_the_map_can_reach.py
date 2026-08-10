"""The recommended way to read a screen was the one that hid the map.

`suggested_gotos` rides on every JSON `analyze`. TSV dropped it — while the guide calls
`aua --format tsv analyze` "the default way to look at a screen" and the orientation block leads
with `--format tsv analyze --fields id,text,rid,clickable`. So the call the tool teaches was the
one call that never mentioned `goto`.

Measured 2026-08-10: five fresh-agent runs against an app with 135 remembered screens and 613
recorded routes. Every one of them navigated by tapping. Not one used `goto`, and none of them
could have known it had anything to offer — nothing they ran said so.

Numeric element ids are not the reason. A recorded route step stores `resource_id` and `label`,
never an id, so `goto` was always immune to the per-call renumbering.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser.projection import OutputFormat, Projection, _route_comment


def _screen(**meta: Any) -> dict[str, Any]:
    return {
        "screen": {"width": 1080, "height": 2400, "package": "com.example.app"},
        "elements": [
            {"id": 0, "text": "Continue", "resource_id": "buttonContinue", "clickable": True}
        ],
        "meta": meta,
    }


def test_a_screen_with_routes_says_where_it_can_go() -> None:
    rendered = Projection.parse(fmt=OutputFormat.tsv).render_tsv(
        _screen(suggested_gotos=["goto settings", "goto home"])
    )

    assert "# goto: settings | home" in rendered


def test_the_line_says_what_goto_does_for_you() -> None:
    """A name alone is not an offer; the point is that the route is already recorded."""
    rendered = Projection.parse(fmt=OutputFormat.tsv).render_tsv(
        _screen(suggested_gotos=["goto settings"])
    )

    assert "aua goto" in rendered
    assert "map --find" in rendered, "the listed four are rarely the goal"


def test_a_screen_with_no_routes_stays_quiet() -> None:
    rendered = Projection.parse(fmt=OutputFormat.tsv).render_tsv(_screen())

    assert "goto" not in rendered, "an empty map must not cost a line on every call"


def test_no_meta_does_not_silence_it() -> None:
    """The first version of this fix put the route behind `--no-meta`, which hid it again.

    Measured 2026-08-10, after that first fix shipped: a fresh agent's opening call was
    `aua --format tsv analyze --no-meta` — the guide recommends `--no-meta` to cut noise — so
    it never saw the line, and navigated the whole task by tapping. `--no-meta` drops the
    diagnostics; a route you can replay is not a diagnostic.
    """
    view = Projection.parse(fmt=OutputFormat.tsv, no_meta=True)
    rendered = view.render_tsv(_screen(suggested_gotos=["goto settings"]))

    assert "# goto: settings" in rendered
    assert "# elements=" not in rendered, "the diagnostics are still cut"


def test_the_element_rows_are_untouched() -> None:
    rendered = Projection.parse(fmt=OutputFormat.tsv).render_tsv(
        _screen(suggested_gotos=["goto settings"])
    )

    assert "buttonContinue" in rendered
    assert rendered.count("\n# ") + rendered.startswith("# ") <= 4, "comments stay a header"


def test_only_a_handful_are_listed() -> None:
    """A map with hundreds of screens must not print hundreds of names."""
    comment = _route_comment({"suggested_gotos": [f"goto s{i}" for i in range(20)]})

    assert len(comment) == 1
    assert comment[0].count("|") <= 3


def test_the_goto_prefix_is_stripped_so_the_name_is_usable() -> None:
    comment = _route_comment({"suggested_gotos": ["goto empty_state"]})

    assert "goto: empty_state" in comment[0], "`goto goto empty_state` is not a command"
