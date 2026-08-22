"""A saved journey nobody is told about is a journey nobody runs.

`aua flow run <name>` replays a whole sequence in one call — launch, taps, waits, `${PARAM}`
interpolation, even a cross-app sign-in leg. A parameterised flow had been sitting saved on this
machine and no agent had ever run one.

Measured 2026-08-10: the word `flow` appeared 19 times in the long guide, and zero times in the
orientation block or the analyze header — everything a fresh agent actually reads. That is the
identical omission that left `goto` unused across five agent runs on a map of 135 screens.

So flows ride out on the same header line as the routes, and — like the routes — they survive
`--no-meta`, because the first version of the goto fix put them behind exactly that flag and the
next agent hid them again.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser.projection import OutputFormat, Projection, _flows_comment


def _screen(**meta: Any) -> dict[str, Any]:
    return {
        "screen": {"width": 1080, "height": 2400, "package": "com.example.app"},
        "elements": [{"id": 0, "text": "Continue", "resource_id": "buttonContinue"}],
        "meta": meta,
    }


def test_a_saved_flow_is_offered() -> None:
    rendered = Projection.parse(fmt=OutputFormat.tsv).render_tsv(
        _screen(flows=["reset_account(ACCOUNT)"])
    )

    assert "# flows: reset_account(ACCOUNT)" in rendered


def test_it_says_what_a_flow_buys_you() -> None:
    """A bare name is not an offer; "one call instead of a dozen" is the reason to use it."""
    comment = _flows_comment({"flows": ["reset_account(ACCOUNT)"]})

    assert "aua flow run" in comment[0]
    assert "--param" in comment[0], "every saved flow here is parameterised"


def test_an_app_with_no_flows_stays_quiet() -> None:
    assert _flows_comment({}) == []
    assert "flows" not in Projection.parse(fmt=OutputFormat.tsv).render_tsv(_screen())


def test_flows_survive_no_meta() -> None:
    """The goto fix was shipped behind `--no-meta` once already; do not repeat it."""
    view = Projection.parse(fmt=OutputFormat.tsv, no_meta=True)
    rendered = view.render_tsv(_screen(flows=["reset_account(ACCOUNT)"]))

    assert "# flows: reset_account(ACCOUNT)" in rendered
    assert "# elements=" not in rendered, "the diagnostics are still cut"


def test_only_a_handful_are_listed() -> None:
    comment = _flows_comment({"flows": [f"flow_{i}()" for i in range(20)]})

    assert len(comment) == 1
    assert comment[0].count("|") <= 2


def test_routes_and_flows_and_questions_coexist() -> None:
    rendered = Projection.parse(fmt=OutputFormat.tsv).render_tsv(
        _screen(
            suggested_gotos=["goto settings"],
            flows=["reset_account(ACCOUNT)"],
            ask={"id": "research_1", "q": "what is this?", "how": "--answers research_1=..."},
        )
    )

    assert "# goto:" in rendered
    assert "# flows:" in rendered
    assert "# aua asks:" in rendered
    # The element row, whatever columns the default view carries — the claim here is about
    # placement (rows last, after the comment block), not about which columns exist.
    assert rendered.splitlines()[-1].split("\t")[1] == "Continue", (
        "the element rows still come last and intact"
    )
