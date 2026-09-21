"""A screen's layout tree: what is where, readable in one look.

The app map used to keep only names and anchor strings per screen. These tests pin the
tree that ``layout.build_layout`` derives from the flat ``analyze`` element list — real
nesting from bounds, repeated rows collapsed, system chrome dropped, values redacted —
and the text ``render_layout`` prints for it.
"""

from __future__ import annotations

from android_ui_analyser import hierarchy
from android_ui_analyser.layout import build_layout, render_layout
from android_ui_analyser.memory import redact_label

P = "com.example.shop"
SIZE = (1080, 2400)


def _n(
    cls: str,
    b: str,
    *,
    text: str = "",
    rid: str | None = None,
    desc: str | None = None,
    clk: bool = False,
    scroll: bool = False,
    selected: bool = False,
    pkg: str = P,
) -> str:
    attrs = [f'class="{cls}"', f'package="{pkg}"']
    if text:
        attrs.append(f'text="{text}"')
    if rid:
        attrs.append(f'resource-id="{rid}"')
    if desc:
        attrs.append(f'content-desc="{desc}"')
    attrs += [
        f'clickable="{str(clk).lower()}"',
        f'scrollable="{str(scroll).lower()}"',
        f'selected="{str(selected).lower()}"',
        'enabled="true"',
        f'bounds="{b}"',
    ]
    return "<node " + " ".join(attrs) + "/>"


# An orders list screen, in a11y pre-order: root, status bar, toolbar, search field,
# a scrollable list of three rows, and a three-tab bottom bar with "Home" selected.
ORDERS = (
    '<hierarchy rotation="0">'
    + _n("android.widget.FrameLayout", "[0,0][1080,2400]", rid="android:id/content")
    + _n(
        "android.widget.TextView",
        "[40,20][200,70]",
        text="12:00",
        rid="com.android.systemui:id/clock",
        pkg="com.android.systemui",
    )
    + _n("android.widget.FrameLayout", "[0,100][1080,300]")
    + _n("android.widget.ImageButton", "[20,120][180,280]", desc="Navigate up", clk=True)
    + _n(
        "android.widget.TextView", "[200,140][900,260]", text="Orders", rid=f"{P}:id/toolbar_title"
    )
    + _n(
        "android.widget.EditText",
        "[40,320][1040,420]",
        text="buyer@example.com",
        rid=f"{P}:id/search_field",
    )
    + _n(
        "androidx.recyclerview.widget.RecyclerView",
        "[0,440][1080,2100]",
        rid=f"{P}:id/order_list",
        scroll=True,
    )
    + _n(
        "android.widget.TextView",
        "[0,460][1080,600]",
        text="Order 1",
        rid=f"{P}:id/order_row",
        clk=True,
    )
    + _n(
        "android.widget.TextView",
        "[0,620][1080,760]",
        text="Order 2",
        rid=f"{P}:id/order_row",
        clk=True,
    )
    + _n(
        "android.widget.TextView",
        "[0,780][1080,920]",
        text="Order 3",
        rid=f"{P}:id/order_row",
        clk=True,
    )
    + _n("android.widget.LinearLayout", "[0,2160][1080,2400]", rid=f"{P}:id/bottom_nav")
    + _n("android.widget.Button", "[0,2160][360,2400]", text="Home", clk=True, selected=True)
    + _n("android.widget.Button", "[360,2160][720,2400]", text="Orders", clk=True)
    + _n("android.widget.Button", "[720,2160][1080,2400]", text="Account", clk=True)
    + "</hierarchy>"
)


def _tree(xml: str = ORDERS) -> str:
    elements = hierarchy.parse_hierarchy(xml, SIZE)
    nodes = build_layout(elements, height=SIZE[1], label_of=redact_label)
    return render_layout(nodes, title="orders")


def _line(text: str, needle: str) -> str:
    hits = [line for line in text.splitlines() if needle in line]
    assert hits, f"{needle!r} missing from:\n{text}"
    return hits[0]


def test_tree_nests_rows_under_their_scrolling_list_and_collapses_repeats() -> None:
    text = _tree()
    list_line = _line(text, "order_list")
    assert "↕" in list_line, list_line
    row = _line(text, "Order 1")
    assert "×3 similar" in row, row
    assert "Order 2" not in text and "Order 3" not in text
    # The rows sit one level under the list: their tree prefix is longer.
    assert row.index("◉") > list_line.index("↕")


def test_tree_places_controls_by_screen_band_and_marks_what_they_do() -> None:
    text = _tree()
    assert "[top]" in _line(text, "Navigate up")
    home = _line(text, '"Home"')
    assert "[bottom]" in home and "★" in home and "◉" in home
    assert "[bottom]" in _line(text, '"Account"')
    # Body content carries its vertical position instead of a band.
    assert "@y460" in _line(text, "Order 1")


def test_tree_drops_system_chrome_and_never_prints_a_typed_value() -> None:
    text = _tree()
    assert "12:00" not in text and "clock" not in text
    assert "buyer@example.com" not in text
    search = _line(text, "search field")
    assert "✎" in search


def test_tree_flattens_unlabelled_containers_but_keeps_named_ones() -> None:
    text = _tree()
    # The anonymous toolbar FrameLayout says nothing; its children move up a level.
    assert "FrameLayout" not in text
    assert "bottom_nav" in text
    assert text.startswith("orders")


def test_tree_is_bounded_on_a_huge_screen() -> None:
    rows = "".join(
        _n(
            "android.widget.Button",
            f"[0,{300 + i * 20}][1080,{318 + i * 20}]",
            text=f"Row {i}",
            clk=True,
            rid=f"{P}:id/r{i}",
        )
        for i in range(200)
    )
    xml = '<hierarchy rotation="0">' + rows + "</hierarchy>"
    elements = hierarchy.parse_hierarchy(xml, SIZE)
    nodes = build_layout(elements, height=SIZE[1], label_of=redact_label)

    def count(items: list) -> int:
        return sum(1 + count(node.children) for node in items)

    assert 0 < count(nodes) <= 60
    assert len(render_layout(nodes, title="big").splitlines()) <= 62


def test_empty_screen_renders_just_its_title() -> None:
    assert render_layout([], title="blank") == "blank\n"


# ------------------------------------------------------------------ stored on the app map


def test_observe_screen_remembers_the_layout_tree_and_refreshes_it_on_revisit(tmp_path) -> None:
    from android_ui_analyser.memory import render_map
    from test_memory import _store

    store = _store(tmp_path)
    first = hierarchy.parse_hierarchy(ORDERS, SIZE)
    # A first sighting returns no known name yet; the cursor tells us what it was called.
    store.observe_screen("emulator-1", package=P, elements=first, screen_height=SIZE[1])
    name = store.load_session("emulator-1").current_screen
    assert name
    rec = store.load(P).screens[name]
    assert rec.layout and rec.layout.startswith(name)
    assert "×3 similar" in rec.layout and "buyer@example.com" not in rec.layout

    # The list grew by one row: the same screen is recognised and its tree moves with it.
    fourth = _n(
        "android.widget.TextView",
        "[0,940][1080,1080]",
        text="Order 4",
        rid=f"{P}:id/order_row",
        clk=True,
    )
    grown = ORDERS.replace(
        '<node class="android.widget.LinearLayout"',
        fourth + '<node class="android.widget.LinearLayout"',
        1,
    )
    again = store.observe_screen(
        "emulator-1",
        package=P,
        elements=hierarchy.parse_hierarchy(grown, SIZE),
        screen_height=SIZE[1],
    )
    assert again == name
    assert "×4 similar" in store.load(P).screens[name].layout

    detail = render_map(store.load(P), screen=name)
    assert "## Layout" in detail and "order_list" in detail


def test_maps_saved_before_layout_trees_still_load(tmp_path) -> None:
    from android_ui_analyser.memory import ScreenRecord

    rec = ScreenRecord.model_validate(
        {
            "name": "old",
            "signature": "abc",
            "first_seen": "t",
            "last_seen": "t",
            "last_verified": "t",
        }
    )
    assert rec.layout is None


# ------------------------------------------------------------------------ aua map --screen


def test_cli_map_screen_prints_the_layout_tree(tmp_path, monkeypatch) -> None:
    import json

    from typer.testing import CliRunner

    from android_ui_analyser import engine as engine_mod
    from android_ui_analyser.cli import app as cli_app
    from conftest import FakeDevice

    runner = CliRunner()
    dev = FakeDevice(hierarchy_xml=ORDERS, package=P, serial="emu-tree")
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: dev)
    assert runner.invoke(cli_app, ["analyze", "--source", "hierarchy"]).exit_code == 0
    seen = runner.invoke(cli_app, ["--format", "compact", "analyze", "--source", "hierarchy"])
    name = json.loads(seen.stdout)["meta"]["known_screen"]
    assert name

    shown = runner.invoke(cli_app, ["map", "--app", P, "--screen", name])
    assert shown.exit_code == 0, shown.stderr
    assert "## Layout" in shown.stdout
    assert "↕ order_list" in shown.stdout and "×3 similar" in shown.stdout
    assert "buyer@example.com" not in shown.stdout

    as_json = runner.invoke(cli_app, ["map", "--app", P, "--screen", name, "--json"])
    assert json.loads(as_json.stdout)["layout"].startswith(name)


def test_map_screen_by_logical_name_shows_one_tree_per_flag_context() -> None:
    from android_ui_analyser.memory import AppMap, ContextRecord, ScreenRecord, render_map

    def screen(name: str, context: str, tree: str | None) -> ScreenRecord:
        return ScreenRecord(
            name=name,
            logical_name="orders",
            context_id=context,
            signature="s",
            first_seen="t",
            last_seen="t",
            last_verified="t",
            layout=tree,
        )

    app = AppMap(
        package=P,
        contexts={
            "default": ContextRecord(id="default", first_seen="t", last_seen="t"),
            "flags-list_v2-1": ContextRecord(
                id="flags-list_v2-1", flags={"list_v2": "on"}, first_seen="t", last_seen="t"
            ),
        },
        screens={
            "orders__a": screen(
                "orders__a", "default", "orders__a\n└─ ↕ order_list @y440 1080×1660\n"
            ),
            "orders__b": screen("orders__b", "flags-list_v2-1", None),
        },
    )
    text = render_map(app, screen="orders")
    assert text.startswith("# orders  (com.example.shop, 2 variants)")
    assert "## orders__a  (context: default)" in text
    assert "## orders__b  (context: flags-list_v2-1 · list_v2=on)" in text
    assert "↕ order_list" in text
    assert "no layout recorded yet" in text
