"""AXe accessibility JSON becomes AUA elements in screenshot pixels, iOS grammar kept inside."""

from __future__ import annotations

import json

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.platforms import DisplayGeometry, ios_tree
from android_ui_analyser.platforms.ios_tools import parse_launchctl_list

APP_ID = "com.example.fixture"
SCALE = 3.0
GEOMETRY = DisplayGeometry.scaled(native_size=(100.0, 200.0), canonical_size=(300, 600))
SCREEN = (300, 600)


def node(type_name: str, x: float, y: float, w: float, h: float, **extra):
    base = {
        "type": type_name,
        "role": f"AX{type_name}",
        "frame": {"x": x, "y": y, "width": w, "height": h},
        "AXLabel": None,
        "AXValue": None,
        "AXUniqueId": None,
        "enabled": True,
        "traits": [],
        "children": [],
        "pid": 4242,
    }
    base.update(extra)
    return base


def fixture_roots(scroll_offset: float = 0.0) -> list[dict]:
    row_a = node("StaticText", 10, 60 - scroll_offset, 80, 10, AXLabel="Card Alpha")
    row_b = node("StaticText", 10, 75 - scroll_offset, 80, 10, AXLabel="Card Beta")
    scroll = node("ScrollView", 5, 50, 90, 140, children=[row_a, row_b])
    button = node(
        "Button",
        10,
        20,
        20,
        30,
        AXLabel="Continue",
        AXUniqueId="primaryContinueButton",
        children=[
            node("Image", 12, 22, 5, 5, AXUniqueId="chevron.forward"),
            node("StaticText", 18, 25, 10, 5, AXLabel="Continue"),
        ],
    )
    unlabeled_button = node(
        "Button", 50, 20, 40, 10, children=[node("StaticText", 52, 22, 20, 5, AXLabel="Skip intro")]
    )
    field = node(
        "TextField",
        10,
        5,
        80,
        10,
        AXLabel="Email",
        AXValue="me@example.test",
        AXUniqueId="emailField",
    )
    secret = node("SecureTextField", 10, 16, 80, 3, AXLabel="Password", AXValue="••••")
    toggle = node("Switch", 60, 40, 30, 8, AXLabel="Marketing emails", AXValue="1")
    offscreen = node("Button", 500, 500, 10, 10, AXLabel="Ghost")
    zero = node("Button", 5, 5, 0, 0, AXLabel="Nothing")
    keyboard = node("Keyboard", 0, 150, 100, 50, children=[node("Key", 2, 152, 8, 8, AXLabel="q")])
    wrapper = node(
        "Group",
        0,
        0,
        100,
        200,
        children=[
            field,
            secret,
            button,
            unlabeled_button,
            toggle,
            scroll,
            offscreen,
            zero,
            keyboard,
        ],
    )
    return [node("Application", 0, 0, 100, 200, AXLabel="Fixture", children=[wrapper])]


def raw_tree(scroll_offset: float = 0.0, *, apps: dict[int, str] | None = None) -> str:
    return ios_tree.envelope(
        fixture_roots(scroll_offset), apps if apps is not None else {4242: APP_ID}
    )


def by_text(elements, text):
    matches = [element for element in elements if element.text == text]
    assert len(matches) == 1, [element.text for element in elements]
    return matches[0]


def test_frames_in_points_become_screenshot_pixels_and_ids_read_top_to_bottom() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    assert tree.app_id == APP_ID
    button = by_text(tree.elements, "Continue")
    assert button.bounds == (30, 60, 90, 150)
    assert button.center == (60, 105)
    assert button.clickable and button.type == "Button"
    assert button.resource_id == "primaryContinueButton"
    assert [element.id for element in tree.elements] == list(range(len(tree.elements)))
    ys = [element.bounds[1] for element in tree.elements]
    assert ys == sorted(ys)


def test_decorative_children_of_a_button_are_absorbed_into_it() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    ids = {element.resource_id for element in tree.elements}
    assert "chevron.forward" not in ids
    assert sum(element.text == "Continue" for element in tree.elements) == 1


def test_an_unlabelled_button_is_named_from_its_subtree() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    skip = by_text(tree.elements, "Skip intro")
    assert skip.type == "Button" and skip.clickable


def test_text_entries_show_their_value_and_keep_the_caption_as_description() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    field = by_text(tree.elements, "me@example.test")
    assert field.content_desc == "Email"
    assert field.resource_id == "emailField"
    assert field.clickable
    secret = by_text(tree.elements, "••••")
    assert secret.password is True


def test_switches_are_checkable_with_their_state_read_from_the_value() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    toggle = by_text(tree.elements, "Marketing emails")
    assert toggle.checkable is True and toggle.checked is True
    assert toggle.content_desc is None


def test_off_screen_and_zero_area_nodes_are_dropped() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    texts = {element.text for element in tree.elements}
    assert "Ghost" not in texts and "Nothing" not in texts


def test_rows_link_to_their_scroll_container_so_movement_can_be_verified() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    scroll = next(element for element in tree.elements if element.type == "ScrollView")
    assert scroll.scrollable is True
    assert scroll.bounds == (15, 150, 285, 570)
    for label in ("Card Alpha", "Card Beta"):
        assert by_text(tree.elements, label).parent == scroll.id

    moved = ios_tree.normalize(raw_tree(10.0), SCREEN, geometry=GEOMETRY)
    assert (
        by_text(moved.elements, "Card Alpha").bounds[1]
        == by_text(tree.elements, "Card Alpha").bounds[1] - 30
    )


def test_the_keyboard_subtree_is_its_own_window_layer() -> None:
    tree = ios_tree.normalize(raw_tree(), SCREEN, geometry=GEOMETRY)

    key = by_text(tree.elements, "q")
    assert key.window == "ime" and key.clickable
    assert by_text(tree.elements, "Continue").window == "app"


def test_the_home_screen_is_a_system_window_and_can_be_ignored_as_foreground() -> None:
    roots = fixture_roots() + [
        node(
            "Application",
            0,
            0,
            100,
            200,
            pid=7,
            children=[node("Button", 0, 0, 10, 10, AXLabel="Dock", pid=7)],
        )
    ]
    raw = ios_tree.envelope(roots, {4242: APP_ID, 7: "com.apple.springboard"})

    tree = ios_tree.normalize(raw, SCREEN, geometry=GEOMETRY)
    assert by_text(tree.elements, "Dock").window == "system"
    assert tree.app_id == APP_ID

    only_home = ios_tree.envelope(roots[1:], {7: "com.apple.springboard"})
    assert (
        ios_tree.normalize(only_home, SCREEN, geometry=GEOMETRY).app_id == "com.apple.springboard"
    )
    assert (
        ios_tree.normalize(raw, SCREEN, geometry=GEOMETRY, ignored_app_ids=[APP_ID]).app_id
        == "com.apple.springboard"
    )


def test_bare_axe_output_without_an_envelope_still_normalizes() -> None:
    tree = ios_tree.normalize(json.dumps(fixture_roots()), SCREEN, geometry=GEOMETRY)

    assert tree.app_id is None
    assert by_text(tree.elements, "Continue").bounds == (30, 60, 90, 150)
    assert ios_tree.normalize("", SCREEN, geometry=GEOMETRY).elements == []


def test_find_bounds_matches_text_identifier_and_description_in_canonical_pixels() -> None:
    raw = raw_tree()

    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="Contin") == (30, 60, 90, 150)
    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="continue", match="exact") is None
    assert ios_tree.find_bounds(
        raw, geometry=GEOMETRY, query="continue", match="exact", ignore_case=True
    ) == (30, 60, 90, 150)
    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="emailField", by="id") == (
        30,
        15,
        270,
        45,
    )
    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="emailField", by="rid") == (
        30,
        15,
        270,
        45,
    )
    with pytest.raises(UsageError):
        ios_tree.find_bounds(raw, geometry=GEOMETRY, query="emailField", by="resourceId")
    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="Email", by="desc") == (
        30,
        15,
        270,
        45,
    )
    assert ios_tree.find_bounds(
        raw, geometry=GEOMETRY, query=r"Card (Alpha|Beta)", match="regex"
    ) == (30, 180, 270, 210)
    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="Ghost") == (1500, 1500, 1530, 1530)
    assert ios_tree.find_bounds(raw, geometry=GEOMETRY, query="absent") is None


def test_launchctl_rows_map_pids_to_bundles_including_the_home_screen() -> None:
    text = (
        "74458\t0\tcom.apple.SpringBoard\n"
        "75976\t0\tUIKitApplication:com.example.fixture[0e2c][rb-legacy]\n"
        "-\t0\tcom.apple.somedaemon\n"
        "garbage line\n"
    )

    assert parse_launchctl_list(text) == {
        74458: "com.apple.springboard",
        75976: "com.example.fixture",
    }


def test_root_frame_size_is_the_largest_root_in_points() -> None:
    roots = [node("Application", 0, 0, 100, 200), node("Application", 0, 0, 40, 40)]

    assert ios_tree.root_frame_size(roots) == (100.0, 200.0)
    assert ios_tree.root_frame_size([]) is None


def test_empty_layout_containers_and_the_application_root_are_not_elements() -> None:
    roots = [
        node(
            "Application",
            0,
            0,
            100,
            200,
            AXLabel="Fixture",
            children=[
                node("Group", 0, 0, 100, 200),
                node("Other", 0, 0, 100, 10),
                node("Group", 0, 20, 100, 10, AXUniqueId="namedContainer"),
                node("Image", 0, 40, 10, 10),
                node("StaticText", 0, 60, 50, 10, AXLabel="Visible"),
            ],
        )
    ]
    tree = ios_tree.normalize(ios_tree.envelope(roots, {4242: APP_ID}), SCREEN, geometry=GEOMETRY)

    kinds = sorted((element.type, element.text or element.resource_id) for element in tree.elements)
    assert kinds == [("Group", "namedContainer"), ("Image", None), ("StaticText", "Visible")]


def test_a_scrollable_nested_in_a_tappable_row_keeps_its_container_and_parents() -> None:
    items = [
        node(
            "Cell",
            5 + i * 30,
            30,
            28,
            20,
            children=[node("StaticText", 7 + i * 30, 35, 20, 5, AXLabel=f"Item {i}")],
        )
        for i in range(3)
    ]
    carousel = node("CollectionView", 0, 28, 100, 24, children=items)
    row = node(
        "Cell",
        0,
        0,
        100,
        60,
        children=[node("StaticText", 5, 5, 40, 8, AXLabel="Featured"), carousel],
    )
    table = node("Table", 0, 0, 100, 200, children=[row])
    roots = [node("Application", 0, 0, 100, 200, children=[table])]

    tree = ios_tree.normalize(ios_tree.envelope(roots, {4242: APP_ID}), SCREEN, geometry=GEOMETRY)

    by_type = {
        element.type: element
        for element in tree.elements
        if element.type in {"Table", "CollectionView"}
    }
    assert by_type["CollectionView"].scrollable is True
    outer_row = tree.elements[by_type["CollectionView"].parent]
    assert outer_row.type == "Cell" and outer_row.parent == by_type["Table"].id
    item_parents = {
        element.parent for element in tree.elements if (element.text or "").startswith("Item ")
    }
    assert item_parents == {by_type["CollectionView"].id}


def test_value_bearing_controls_keep_their_identity_when_the_value_changes() -> None:
    def roots(value: str) -> list[dict]:
        slider = node("Slider", 10, 50, 80, 10, AXLabel="Volume", AXValue=value)
        return [node("Application", 0, 0, 100, 200, children=[slider])]

    before = ios_tree.normalize(
        ios_tree.envelope(roots("50%"), {4242: APP_ID}), SCREEN, geometry=GEOMETRY
    ).elements[0]
    after = ios_tree.normalize(
        ios_tree.envelope(roots("60%"), {4242: APP_ID}), SCREEN, geometry=GEOMETRY
    ).elements[0]

    assert before.content_desc == after.content_desc == "Volume"
    assert (before.text, after.text) == ("50%", "60%")
    assert before.stable_key == after.stable_key
