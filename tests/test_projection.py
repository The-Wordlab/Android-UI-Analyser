"""Projection / filtering / TSV semantics (``analyze --fields|--where-*|--region|--format tsv``).

Pure unit tests over :mod:`android_ui_analyser.projection` — no device, no CLI. The CLI
wiring and its exit codes are covered in ``test_analyze_view.py``.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.projection import (
    FIELD_ALIASES,
    TSV_DEFAULT_FIELDS,
    Projection,
    short_rid,
)
from android_ui_analyser.schema import Meta, OutputFormat


def _element(**over: object) -> dict:
    base = {
        "id": 0,
        "type": "Button",
        "text": None,
        "resource_id": None,
        "content_desc": None,
        "bounds": [0, 0, 100, 100],
        "center": [50, 50],
        "clickable": False,
        "enabled": True,
        "focused": False,
        "checkable": None,
        "checked": None,
        "selected": None,
        "scrollable": None,
        "long_clickable": None,
        "password": None,
        "source": "hierarchy",
        "confidence": None,
    }
    base.update(over)
    if "center" not in over:
        x1, y1, x2, y2 = base["bounds"]  # type: ignore[misc]
        base["center"] = [(x1 + x2) // 2, (y1 + y2) // 2]
    return base


def _payload(*elements: dict, **meta: object) -> dict:
    base_meta = {
        "duration_ms": 12,
        "tier_used": "hierarchy",
        "path": "hierarchy",
        "providers_used": [],
        "known_screen": "my_apps",
        "known_routes": ["tap 'Apps' → apps"],
        "suggested_gotos": ["goto apps"],
        "suggested_deeplinks": [],
        "map_hint": None,
        "annotated_image": None,
        "raw_image": None,
        "device_serial": "emulator-5554",
    }
    base_meta.update(meta)
    return {
        "schema_version": 1,
        "screen": {
            "width": 1080,
            "height": 2400,
            "package": "com.example.app.dev",
            "activity": ".MainActivity",
            "source": "hierarchy",
        },
        "elements": list(elements),
        "meta": base_meta,
    }


# A screen shaped like the real thing: status-bar chrome, then app rows.
STATUS_CLOCK = _element(id=0, resource_id="com.android.systemui:id/clock", text="3:37")
STATUS_BATTERY = _element(id=1, content_desc="Battery 100 percent.", bounds=[935, 14, 998, 48])
HEADER_BELL = _element(
    id=2,
    type="ImageButton",
    resource_id="com.example.app.dev:id/notificationsButton",
    content_desc="Notifications",
    bounds=[900, 84, 1010, 200],
    clickable=True,
)
UNLABELLED = _element(id=3, type="View", bounds=[0, 300, 1080, 400])
TAB_BROWSE = _element(
    id=4,
    text="Browse",
    resource_id="com.example.app.dev:id/homeTabBROWSE",
    bounds=[0, 400, 540, 500],
    clickable=True,
    selected=True,
)
SWITCH_OFF = _element(
    id=5,
    type="Switch",
    resource_id="com.example.app.dev:id/settingsSwitch",
    bounds=[859, 600, 996, 700],
    checkable=True,
    checked=False,
)
SCREEN = _payload(STATUS_CLOCK, STATUS_BATTERY, HEADER_BELL, UNLABELLED, TAB_BROWSE, SWITCH_OFF)


# --------------------------------------------------------------------------- aliases


def test_every_alias_maps_to_a_real_element_key() -> None:
    from android_ui_analyser.schema import Element

    assert set(FIELD_ALIASES.values()) <= set(Element.model_fields)


def test_rid_is_short_and_resource_id_is_full() -> None:
    view = Projection.parse(fields="rid,resource_id")
    projected = view.project(TAB_BROWSE)
    assert projected["rid"] == "homeTabBROWSE"
    assert projected["resource_id"] == "com.example.app.dev:id/homeTabBROWSE"


def test_short_rid_passes_through_missing_values() -> None:
    assert short_rid(None) is None
    assert short_rid("") == ""
    assert short_rid("bare_id") == "bare_id"


# --------------------------------------------------------------------------- validation


def test_unknown_field_name_raises_usage_error_listing_valid_names() -> None:
    with pytest.raises(UsageError) as excinfo:
        Projection.parse(fields="id,txt")
    err = excinfo.value
    assert "txt" in str(err)
    assert err.hint is not None and "text" in err.hint


def test_unknown_meta_key_raises_usage_error() -> None:
    with pytest.raises(UsageError) as excinfo:
        Projection.parse(meta="known_screen,nope")
    assert "nope" in str(excinfo.value)


def test_every_meta_key_is_accepted() -> None:
    view = Projection.parse(meta=",".join(Meta.model_fields))
    assert view.meta_keys is not None
    assert set(view.meta_keys) == set(Meta.model_fields)


@pytest.mark.parametrize("raw", ["0,0,10", "a,b,c,d", "", "1,2,3,4,5"])
def test_bad_region_raises_usage_error(raw: str) -> None:
    with pytest.raises(UsageError):
        Projection.parse(region=[raw])


def test_region_is_normalised_to_min_max() -> None:
    view = Projection.parse(region=["1080,300,0,0"])
    assert view.regions == ((0, 0, 1080, 300),)


def test_negative_limit_raises_usage_error() -> None:
    with pytest.raises(UsageError):
        Projection.parse(limit=-1)


# --------------------------------------------------------------------------- inactive default


def test_no_flags_means_inactive_so_default_output_is_untouched() -> None:
    assert Projection.parse().active is False
    assert Projection.parse(fmt=OutputFormat.compact).active is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fields": "id"},
        {"nonempty": True},
        {"no_system": True},
        {"clickable": True},
        {"limit": 5},
        {"meta": "known_screen"},
        {"no_meta": True},
        {"where_text": ["x"]},
        {"where_rid": ["x"]},
        {"region": ["0,0,1,1"]},
        {"show_all": True},
    ],
)
def test_any_flag_activates_the_view(kwargs: dict) -> None:
    assert Projection.parse(**kwargs).active is True


# --------------------------------------------------------------------------- filters


def test_nonempty_drops_elements_with_no_label() -> None:
    kept = Projection.parse(nonempty=True).select(SCREEN)
    assert UNLABELLED["id"] not in [e["id"] for e in kept]
    assert TAB_BROWSE["id"] in [e["id"] for e in kept]


def test_no_system_drops_systemui_ids_and_status_band_chrome() -> None:
    kept = [e["id"] for e in Projection.parse(no_system=True).select(SCREEN)]
    assert STATUS_CLOCK["id"] not in kept  # com.android.systemui resource-id
    assert STATUS_BATTERY["id"] not in kept  # battery content_desc inside the status band
    assert HEADER_BELL["id"] in kept


def test_no_system_keeps_app_content_that_merely_reads_like_chrome() -> None:
    """A "Bluetooth" row in an app's settings list is content, not the status bar."""
    row = _element(
        id=9,
        text="Bluetooth",
        content_desc="Bluetooth",
        bounds=[0, 900, 1080, 1000],
        clickable=True,
    )
    kept = [e["id"] for e in Projection.parse(no_system=True).select(_payload(row))]
    assert kept == [9]


def test_no_system_keeps_dialog_buttons_under_the_android_package() -> None:
    ok = _element(id=1, text="Got it", resource_id="android:id/button1", clickable=True)
    kept = [e["id"] for e in Projection.parse(no_system=True).select(_payload(ok))]
    assert kept == [1]


def test_where_text_is_case_insensitive_substring() -> None:
    kept = Projection.parse(where_text=["BROW"]).select(SCREEN)
    assert [e["id"] for e in kept] == [TAB_BROWSE["id"]]


def test_where_rid_matches_short_and_full_form() -> None:
    for needle in ("homeTab", "example.app.dev:id/homeTabBROWSE", "hometabbrowse"):
        kept = Projection.parse(where_rid=[needle]).select(SCREEN)
        assert [e["id"] for e in kept] == [TAB_BROWSE["id"]], needle


def test_repeated_filters_of_one_kind_or_together() -> None:
    kept = Projection.parse(where_text=["Browse", "3:37"]).select(SCREEN)
    assert [e["id"] for e in kept] == [STATUS_CLOCK["id"], TAB_BROWSE["id"]]


def test_filters_of_different_kinds_and_together() -> None:
    kept = Projection.parse(where_text=["Browse"], clickable=True, nonempty=True).select(SCREEN)
    assert [e["id"] for e in kept] == [TAB_BROWSE["id"]]
    assert Projection.parse(where_text=["Browse"], region=["0,0,1080,10"]).select(SCREEN) == []


def test_clickable_filter() -> None:
    kept = [e["id"] for e in Projection.parse(clickable=True).select(SCREEN)]
    assert kept == [HEADER_BELL["id"], TAB_BROWSE["id"]]


def test_region_keeps_intersecting_elements_only() -> None:
    kept = [e["id"] for e in Projection.parse(region=["0,0,1080,300"]).select(SCREEN)]
    assert kept == [STATUS_CLOCK["id"], STATUS_BATTERY["id"], HEADER_BELL["id"]]


def test_region_touching_edges_does_not_count_as_intersecting() -> None:
    # HEADER_BELL starts at y=84; a region ending exactly at 84 must not match.
    kept = [e["id"] for e in Projection.parse(region=["900,0,1010,84"]).select(SCREEN)]
    assert HEADER_BELL["id"] not in kept


def test_repeated_regions_or_together() -> None:
    kept = [
        e["id"] for e in Projection.parse(region=["0,0,1080,70", "0,590,1080,710"]).select(SCREEN)
    ]
    assert kept == [STATUS_CLOCK["id"], STATUS_BATTERY["id"], SWITCH_OFF["id"]]


def test_limit_applies_after_filtering() -> None:
    kept = Projection.parse(nonempty=True, no_system=True, limit=1).select(SCREEN)
    assert [e["id"] for e in kept] == [HEADER_BELL["id"]]


def test_limit_zero_yields_nothing() -> None:
    assert Projection.parse(limit=0).select(SCREEN) == []


def test_filtering_preserves_the_original_element_ids() -> None:
    """A view is a rendering concern: ids stay addressable by `aua tap <id>`."""
    kept = Projection.parse(clickable=True, nonempty=True).select(SCREEN)
    assert [e["id"] for e in kept] == [2, 4]


# --------------------------------------------------------------------------- projection


def test_apply_projects_columns_and_keeps_the_envelope() -> None:
    out = Projection.parse(fields="id,text,rid").apply(SCREEN)
    assert set(out) == {"schema_version", "screen", "elements", "meta"}
    assert out["elements"][0] == {"id": 0, "text": "3:37", "rid": "clock"}


def test_apply_without_fields_keeps_full_elements() -> None:
    out = Projection.parse(nonempty=True).apply(SCREEN)
    assert set(out["elements"][0]) == set(STATUS_CLOCK)


def test_compact_projection_drops_nulls_so_absence_means_unknown() -> None:
    view = Projection.parse(fields="id,text,checked")
    plain = view.apply(SCREEN, fmt=OutputFormat.json)["elements"]
    compact = view.apply(SCREEN, fmt=OutputFormat.compact)["elements"]
    assert plain[0] == {"id": 0, "text": "3:37", "checked": None}
    assert compact[0] == {"id": 0, "text": "3:37"}
    # A *known* false survives compact — off must stay distinguishable from unknown.
    switch = next(e for e in compact if e["id"] == SWITCH_OFF["id"])
    assert switch == {"id": 5, "checked": False}


def test_meta_selection_and_suppression() -> None:
    only = Projection.parse(meta="known_screen").apply(SCREEN)
    assert only["meta"] == {"known_screen": "my_apps"}
    assert "meta" not in Projection.parse(no_meta=True).apply(SCREEN)


def test_apply_never_mutates_the_input_payload() -> None:
    before = len(SCREEN["elements"])
    Projection.parse(fields="id", nonempty=True, limit=1).apply(SCREEN)
    assert len(SCREEN["elements"]) == before
    assert SCREEN["meta"]["known_screen"] == "my_apps"


# --------------------------------------------------------------------------- tsv


def _tsv(payload: dict, **kwargs: object) -> list[str]:
    view = Projection.parse(fmt=OutputFormat.tsv, **kwargs)
    return view.render_tsv(payload).splitlines()


def test_tsv_defaults_to_the_show_me_the_app_view() -> None:
    view = Projection.parse(fmt=OutputFormat.tsv)
    assert view.nonempty and view.no_system and view.tsv
    assert view.columns() == TSV_DEFAULT_FIELDS


def test_tsv_all_opts_out_of_the_implicit_filters() -> None:
    view = Projection.parse(fmt=OutputFormat.tsv, show_all=True)
    assert not view.nonempty and not view.no_system
    assert len(view.select(SCREEN)) == len(SCREEN["elements"])


def test_tsv_shape_is_comments_then_header_then_rows() -> None:
    lines = _tsv(SCREEN)
    assert lines[0].startswith("# screen=my_apps package=com.example.app.dev 1080x2400")
    assert lines[1].startswith("# elements=6 shown=3")
    assert lines[2] == "id\ttext\trid\tclickable"
    assert lines[3].split("\t") == ["2", "", "notificationsButton", "true"]


def test_tsv_column_order_follows_fields() -> None:
    lines = _tsv(SCREEN, fields="clickable,id,text")
    assert lines[2] == "clickable\tid\ttext"
    assert lines[3].split("\t")[:2] == ["true", "2"]


def test_tsv_tri_state_renders_true_false_and_empty_for_unknown() -> None:
    lines = _tsv(SCREEN, fields="id,checkable,checked", where_rid=["settingsSwitch"])
    assert lines[-1].split("\t") == ["5", "true", "false"]
    unknown = _tsv(SCREEN, fields="id,checked", where_rid=["homeTab"])
    assert unknown[-1].split("\t") == ["4", ""]


def test_tsv_lists_render_comma_joined_without_tabs() -> None:
    lines = _tsv(SCREEN, fields="id,bounds,center", where_rid=["homeTab"])
    assert lines[-1] == "4\t0,400,540,500\t270,450"


def test_tsv_cells_never_contain_tabs_or_newlines() -> None:
    noisy = _element(id=0, text="two\tcols\nand a line", resource_id="app:id/x")
    lines = _tsv(_payload(noisy), fields="id,text")
    assert lines[-1] == "0\ttwo cols and a line"
    assert len(lines[-1].split("\t")) == 2


def test_tsv_no_meta_emits_only_the_payload() -> None:
    lines = _tsv(SCREEN, no_meta=True)
    assert not any(line.startswith("#") for line in lines)
    assert lines[0] == "id\ttext\trid\tclickable"


def test_tsv_meta_keys_become_comment_lines() -> None:
    lines = _tsv(SCREEN, meta="known_screen,suggested_gotos")
    assert lines[1] == "# known_screen=my_apps"
    assert lines[2] == "# suggested_gotos=goto apps"
    assert not any(line.startswith("# elements=") for line in lines)


def test_tsv_summary_reports_how_many_rows_were_hidden() -> None:
    assert "# elements=6 shown=6" in _tsv(SCREEN, show_all=True)[1]
    assert "# elements=6 shown=1" in _tsv(SCREEN, clickable=True, where_text=["browse"])[1]


def test_tsv_survives_an_empty_screen() -> None:
    lines = _tsv(_payload())
    assert lines[1].startswith("# elements=0 shown=0")
    assert lines[-1] == "id\ttext\trid\tclickable"


# ----------------------------------------------------------------- --no-wrappers (opt-in)

# The View-based shape: id'd layout wrappers nested around one real control.
WRAP_OUTER = _element(id=0, resource_id="com.android.settings:id/content_parent",
                      bounds=[0, 0, 1080, 2400])
WRAP_MID = _element(id=1, resource_id="com.android.settings:id/app_bar",
                    bounds=[0, 0, 1080, 400])
WRAP_INNER = _element(id=2, resource_id="com.android.settings:id/collapsing_toolbar",
                      bounds=[0, 0, 1080, 300])
REAL_ROW = _element(id=3, resource_id="com.android.settings:id/title", text="Network",
                    bounds=[40, 100, 1000, 180], clickable=True)
BARE_LEAF = _element(id=4, resource_id="com.android.settings:id/icon",
                     bounds=[40, 500, 120, 580])
FRAMEWORK_ROOT = _element(id=5, resource_id="android:id/content", bounds=[0, 0, 1080, 2400])

_WRAPPED = _payload(WRAP_OUTER, WRAP_MID, WRAP_INNER, REAL_ROW, BARE_LEAF, FRAMEWORK_ROOT)


def _ids(proj: Projection, payload: dict) -> list[int]:
    return [e["id"] for e in proj.select(payload)]


def test_no_wrappers_is_off_by_default() -> None:
    """Container ids stay discoverable unless the caller opts out — the default never moves."""
    assert _ids(Projection.parse(), _WRAPPED) == [0, 1, 2, 3, 4, 5]


def test_no_wrappers_drops_pure_containers_only() -> None:
    kept = _ids(Projection.parse(no_wrappers=True), _WRAPPED)
    assert WRAP_OUTER["id"] not in kept
    assert WRAP_MID["id"] not in kept
    assert WRAP_INNER["id"] not in kept
    # A labeled/actionable row and an unlabeled LEAF are both things a caller acts on.
    assert REAL_ROW["id"] in kept
    assert BARE_LEAF["id"] in kept


def test_no_wrappers_keeps_actionable_and_labeled_containers() -> None:
    tappable = _element(id=6, resource_id="com.app:id/card", bounds=[0, 600, 1080, 900],
                        clickable=True)
    scroller = _element(id=7, resource_id="com.app:id/list", bounds=[0, 900, 1080, 2000],
                        scrollable=True)
    described = _element(id=8, resource_id="com.app:id/hero", bounds=[0, 2000, 1080, 2300],
                         content_desc="Hero banner")
    inner = _element(id=9, resource_id="com.app:id/row", bounds=[10, 950, 500, 1000], text="Row")
    payload = _payload(tappable, scroller, described, inner)
    assert _ids(Projection.parse(no_wrappers=True), payload) == [6, 7, 8, 9]


def test_no_wrappers_ignores_framework_ids_and_identical_bounds() -> None:
    """``android:id/*`` is not an app id, and same-bounds siblings must not cancel out."""
    twin_a = _element(id=0, resource_id="com.app:id/a", bounds=[0, 0, 500, 500])
    twin_b = _element(id=1, resource_id="com.app:id/b", bounds=[0, 0, 500, 500])
    payload = _payload(twin_a, twin_b, FRAMEWORK_ROOT)
    assert _ids(Projection.parse(no_wrappers=True), payload) == [0, 1, 5]


def test_no_wrappers_composes_with_all() -> None:
    assert _ids(Projection.parse(no_wrappers=True, show_all=True), _WRAPPED) == [0, 1, 2, 3, 4, 5]


def test_nonempty_still_keeps_id_only_containers() -> None:
    """DEFECT 3 note: --nonempty means "has identity" and keeps them; --no-wrappers is separate."""
    assert WRAP_OUTER["id"] in _ids(Projection.parse(nonempty=True), _WRAPPED)
