"""The compact projection must survive real Android trees, not just tidy synthetic ones.

Each case here is a shape taken from screens harvested off live emulators, because every earlier
version of this projection was tuned against invented screens and every one of them was wrong about
something structural. The measured facts it has to hold up against: a median of 50 nodes per screen
of which about 6 are actionable, visible text on only 29% of clickable nodes, a description on 58%,
*neither* on 15.5%, and a quarter of clickable texts duplicated elsewhere on the same screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.functiongemma.v11_projection import (  # noqa: E402
    MAX_CONTEXT_NODES,
    MAX_NODES,
    is_structural,
    normalise,
    project,
    unnameable_rate,
)


def test_normalise_flattens_what_real_screens_contain() -> None:
    """ "Wi‑Fi" on a real screen uses U+2011, not a hyphen; a small model should not pay for that."""

    assert normalise("Network & internet Mobile, Wi‑Fi, hotspot") == (
        "Network & internet Mobile, Wi-Fi, hotspot"
    )
    assert normalise("  spaced   out \n") == "spaced out"
    assert normalise("") is None
    assert normalise(None) is None
    assert normalise("​") is None


def test_scaffolding_is_dropped_whatever_its_resource_id_says() -> None:
    """The first version enumerated known container ids and leaked the ones nobody listed."""

    assert is_structural({"rid": "status_bar_launch_animation_container"})
    assert is_structural({"rid": "some_vendor_wrapper_nobody_listed"})
    assert is_structural({"rid": None})
    # Anything actionable or readable stays, however anonymous.
    assert not is_structural({"clickable": True})
    assert not is_structural({"scrollable": True})
    assert not is_structural({"text": "App info"})
    assert not is_structural({"desc": "Profile picture"})


def test_a_real_settings_tree_reduces_to_its_decisions() -> None:
    """38 raw nodes from a live `com.android.settings` app-info screen."""

    elements = [
        {"id": "rid:status_bar", "rid": "status_bar"},
        {"id": "rid:status_bar_contents", "rid": "status_bar_contents"},
        {"id": "rid:clock", "rid": "clock", "text": "2:41", "desc": "2:41 PM"},
        {"id": "rid:battery", "rid": "battery", "desc": "Battery charging, 100 percent."},
        {"id": "cd:9a8241f23f", "desc": "Android System notification:"},
        {"id": "rid:text_frame#1", "rid": "text_frame"},
        {"id": "rid:icon_frame#1", "rid": "icon_frame"},
        {"id": "rid:collapsing_toolbar", "rid": "collapsing_toolbar", "text": "App info"},
        {"id": "tx:aaa", "text": "Chrome"},
        {"id": "rid:scroll", "rid": "main_content_scrollable_container", "scrollable": True},
        {"id": "rid:up", "rid": "up", "text": "Navigate up", "clickable": True},
        {"id": "tx:open", "text": "Open", "clickable": True},
        {"id": "tx:disable", "text": "Disable", "clickable": True},
        {
            "id": "tx:notif",
            "text": "Notifications About 0 notifications per week",
            "clickable": True,
        },
    ]
    out = project(elements)
    kinds = out["nodes"]

    # Every actionable node survives.
    tappable = [n["text"] for n in kinds if n.get("tap")]
    assert tappable == [
        "Navigate up",
        "Open",
        "Disable",
        "Notifications About 0 notifications per week",
    ]
    assert any(n.get("scroll") for n in kinds)

    # Scaffolding, the clock, the battery and the shade summary are all gone. Note the ban list
    # names the systemui *content*, not the substring "notification": the legitimate actionable row
    # "Notifications About 0 notifications per week" contains it and must survive.
    shown = " ".join(str(n) for n in kinds)
    for banned in (
        "status_bar",
        "text_frame",
        "icon_frame",
        "Battery charging",
        "Android System notification:",
        "2:41",
    ):
        assert banned not in shown, banned
    assert any("Notifications About 0" in str(n.get("text")) for n in kinds)

    # Context is the heading and the app name, and no more than the cap.
    context = [n for n in kinds if not n.get("tap") and not n.get("scroll")]
    assert len(context) <= MAX_CONTEXT_NODES
    assert {n.get("text") for n in context} == {"App info", "Chrome"}

    # Ordinals, never AUA's composite keys — ten hex characters are copy poison at 4-bit.
    assert [n["n"] for n in kinds] == [f"n{i}" for i in range(1, len(kinds) + 1)]
    assert "tx:" not in shown and "cd:" not in shown


def test_the_caller_keeps_the_map_back_to_stable_keys() -> None:
    """The model points at `n3`; something host-side still has to act on the real element."""

    elements = [
        {"id": "rid:one", "text": "One", "clickable": True},
        {"id": "tx:9be6e44dfc", "text": "Two", "clickable": True},
    ]
    out = project(elements)
    assert out["keys"] == ["rid:one", "tx:9be6e44dfc"]
    assert len(out["keys"]) == len(out["nodes"])


def test_a_duplicated_heading_is_collapsed_not_taught() -> None:
    """25% of real clickable texts are duplicated. The projection resolves it; the model should
    never have been asked to."""

    elements = [
        {"id": "rid:bar", "rid": "search_action_bar", "text": "Search settings", "clickable": True},
        {"id": "rid:title", "rid": "search_action_bar_title", "text": "Search settings"},
    ]
    out = project(elements)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["tap"] is True


def test_an_unnameable_control_still_gets_something_to_point_at() -> None:
    """One clickable control in six has no text and no description. It must remain addressable."""

    elements = [
        {"id": "rid:fab", "rid": "fab_main", "clickable": True},
        {"id": "geo:ImageView:q02:75957dc13d", "clickable": True},
    ]
    out = project(elements)
    assert len(out["nodes"]) == 2
    # The resource id is shown *only* where nothing else names the node.
    assert out["nodes"][0]["rid"] == "fab_main"
    assert "rid" not in out["nodes"][1]
    # And the ordinal is what makes the anonymous one selectable at all.
    assert out["nodes"][1]["n"] == "n2"
    assert unnameable_rate(elements) == 1.0


def test_a_resource_id_is_hidden_when_the_node_already_has_a_name() -> None:
    """Showing it everywhere is how the `navTab<Section>` leak happened: the id was the label in
    another font, and the model learned to invent ids instead of reading the screen."""

    out = project(
        [{"id": "rid:navTabAtlas", "rid": "navTabAtlas", "text": "Atlas", "clickable": True}]
    )
    assert out["nodes"][0]["text"] == "Atlas"
    assert "rid" not in out["nodes"][0]


def test_truncation_is_reported_because_it_changes_the_right_answer() -> None:
    """With content off-screen, "the target is absent" is not provable and scrolling is the only
    sound step. A corpus that truncates silently teaches "not in the list, therefore refuse"."""

    many = [{"id": f"tx:{i}", "text": f"Row {i}", "clickable": True} for i in range(MAX_NODES + 6)]
    out = project(many)
    assert out["more"] is True
    assert len(out["nodes"]) == MAX_NODES

    few = many[:3]
    assert project(few)["more"] is False


def test_context_never_displaces_an_actionable_node() -> None:
    """A status bar must not be able to crowd a control out of the budget."""

    elements = [{"id": f"tx:{i}", "text": f"Row {i}", "clickable": True} for i in range(MAX_NODES)]
    elements += [{"id": f"cd:{i}", "desc": f"Chatter {i}"} for i in range(10)]
    out = project(elements)
    assert len(out["nodes"]) == MAX_NODES
    assert all(node.get("tap") for node in out["nodes"])


def test_a_fixed_bottom_bar_survives_a_long_list_above_it() -> None:
    """The first live drive on a real app failed exactly here, and the shape is general.

    Asked to reach an app's Apps section, the on-device driver scrolled three times and reported
    ``target_absent`` while an "Apps" control sat plainly on screen. A fixed bottom navigation bar is
    last in tree order and first in importance: the chat list ahead of it filled all 14 slots, so the
    bar was truncated away — and because a fixed bar is not inside the scrollable list, scrolling
    could never have revealed it. The refusal was internally consistent and factually wrong.

    This reproduces the screen shape rather than the app: eighteen actionable rows, a repetitive list
    first and the navigation last. It fails at ``MAX_NODES = 14`` and passes at 28.
    """

    elements = [
        {"rid": "bannerGreeting", "text": "Greeting How can I help you today?", "clickable": True},
        *[
            {"text": f"A conversation {index}", "clickable": True}
            for index in range(14)
        ],
        {"rid": "navBarPrimary", "text": "Sprockets", "clickable": True},
        {"rid": "navBarSecondary", "text": "Grommets", "clickable": True},
        {"rid": "navBarTertiary", "text": "Widgets", "clickable": True},
    ]
    projection = project(elements)
    labels = {str(node.get("text") or node.get("desc") or "") for node in projection["nodes"]}
    assert "Widgets" in labels, (
        "the fixed navigation bar was truncated away by the node cap; "
        f"shown {len(projection['nodes'])} of {len(elements)}"
    )
    # And the whole bar survives, not just its last entry — a driver that can reach one tab and not
    # its neighbours is a worse failure than one that reaches none, because it looks like it works.
    assert {"Sprockets", "Grommets", "Widgets"} <= labels


def test_the_cap_is_high_enough_for_nine_screens_in_ten() -> None:
    """Chosen from the harvest, not picked: p90 is 17 actionable nodes and p95 is 22.

    A cap of 14 truncated 13.2% of the 638 harvested screens. Truncation is not merely lossy — it
    changes what the right answer *is*, because with content hidden "the target is absent" stops being
    provable and scrolling becomes the only sound step. For a fixed bar, scrolling is not sound
    either, which is how the live run got stuck.
    """

    assert MAX_NODES >= 22, "below p95 of real screens"
