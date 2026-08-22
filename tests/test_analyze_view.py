"""CLI wiring for the analyze views, element state, cropped capture, and quiet daemon start.

Covers the flags an agent reaches for instead of hand-writing a JSON filter
(``--fields``/``--format tsv``/``--nonempty``/``--no-system``/``--where-*``/``--clickable``/
``--region``/``--limit``/``--meta``), the interaction-state fields those flags expose,
``screenshot --region/--scale/--max-width``, and ``daemon start --quiet`` / ``aua orient``.
Device-less: :class:`FakeDevice` is injected by patching ``engine.connect``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.schema import AnalyzeResult
from conftest import FakeDevice

runner = CliRunner()

# A screen with the three shapes that matter: status-bar chrome, app rows, and a switch
# whose on/off state is only readable from the a11y attributes.
SCREEN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="3:37" resource-id="com.android.systemui:id/clock"
        class="android.widget.TextView" package="com.android.systemui" content-desc=""
        checkable="false" checked="false" clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false" password="false"
        selected="false" bounds="[11,2][126,60]"/>
  <node index="1" text="" resource-id="" class="android.widget.ImageView"
        package="com.android.systemui" content-desc="Battery 100 percent."
        checkable="false" checked="false" clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false" password="false"
        selected="false" bounds="[935,14][998,48]"/>
  <node index="2" text="" resource-id="com.test.app:id/notificationsButton"
        class="android.widget.ImageButton" package="com.test.app" content-desc="Notifications"
        checkable="false" checked="false" clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false" password="false"
        selected="false" bounds="[900,84][1010,200]"/>
  <node index="3" text="" resource-id="" class="android.widget.View" package="com.test.app"
        content-desc="" checkable="false" checked="false" clickable="false" enabled="true"
        focusable="false" focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false" bounds="[0,300][1080,400]"/>
  <node index="4" text="Browse" resource-id="com.test.app:id/homeTabBROWSE"
        class="android.widget.TextView" package="com.test.app" content-desc=""
        checkable="false" checked="false" clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="true" password="false"
        selected="true" bounds="[0,400][540,500]"/>
  <node index="5" text="" resource-id="com.test.app:id/settingsSwitch" class="android.widget.Switch"
        package="com.test.app" content-desc="Push notifications" checkable="true" checked="true"
        clickable="true" enabled="true" focusable="true" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false" bounds="[859,600][996,700]"/>
  <node index="6" text="" resource-id="com.test.app:id/feed" class="androidx.recyclerview.widget.RecyclerView"
        package="com.test.app" content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="false" scrollable="true" long-clickable="false"
        password="false" selected="false" bounds="[0,800][1080,2300]"/>
</hierarchy>"""

# A node that reports none of the state attributes at all (a sparse/partial dump).
SPARSE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Bare" bounds="[0,0][100,50]"/>
</hierarchy>"""


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUA_CACHE__DIR", str(cache))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    return cache


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> FakeDevice:
    dev = FakeDevice(hierarchy_xml=SCREEN_XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    return dev


def _run(*args: str, code: int = 0) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == code, f"{args} -> {result.exit_code}\n{result.output}"
    return result.output


def _rows(output: str) -> list[list[str]]:
    body = [ln for ln in output.strip().splitlines() if ln and not ln.startswith("#")]
    return [ln.split("\t") for ln in body]


# ------------------------------------------------------------------ backward compatibility


def test_analyze_without_view_flags_is_unchanged_schema(device: FakeDevice) -> None:
    payload = json.loads(_run("--format", "compact", "analyze"))
    AnalyzeResult.model_validate(payload)
    assert {"schema_version", "screen", "elements", "meta"} == set(payload)


def test_filters_alone_still_validate_as_an_analyze_result(device: FakeDevice) -> None:
    """`--nonempty`/`--no-system` only drop rows, so the payload stays schema-shaped."""
    payload = json.loads(_run("analyze", "--nonempty", "--no-system"))
    AnalyzeResult.model_validate(payload)


# ------------------------------------------------------------------ --fields


def test_fields_projects_and_shortens_rid(device: FakeDevice) -> None:
    payload = json.loads(_run("analyze", "--fields", "id,text,rid", "--where-rid", "homeTab"))
    # A projection is an output path, so its ids are the published stable ids.
    assert payload["elements"] == [
        {"id": "rid:homeTabBROWSE", "text": "Browse", "rid": "homeTabBROWSE"}
    ]


def test_fields_resource_id_keeps_the_full_selector(device: FakeDevice) -> None:
    payload = json.loads(_run("analyze", "--fields", "resource_id", "--where-rid", "settingsSwitch"))
    assert payload["elements"] == [{"resource_id": "com.test.app:id/settingsSwitch"}]


def test_unknown_field_exits_2_and_names_the_offender(device: FakeDevice) -> None:
    out = _run("analyze", "--fields", "id,txt", code=2)
    err = json.loads(out)["error"]
    assert err["code"] == "usage"
    assert "txt" in err["message"]
    assert "text" in err["hint"] and "rid" in err["hint"]


def test_unknown_field_never_touches_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(serial: str | None = None) -> FakeDevice:  # pragma: no cover - must not run
        raise AssertionError("validation must fail before connecting")

    monkeypatch.setattr(engine_mod, "connect", explode)
    _run("analyze", "--fields", "nope", code=2)


def test_unknown_meta_key_exits_2(device: FakeDevice) -> None:
    assert "bogus" in json.loads(_run("analyze", "--meta", "bogus", code=2))["error"]["message"]


@pytest.mark.parametrize("bad", ["0,0,10", "x,y,z,w"])
def test_bad_region_exits_2(device: FakeDevice, bad: str) -> None:
    assert json.loads(_run("analyze", "--region", bad, code=2))["error"]["code"] == "usage"


# ------------------------------------------------------------------ --format tsv


def test_tsv_is_comment_header_then_columns_then_rows(device: FakeDevice) -> None:
    out = _run("--format", "tsv", "analyze")
    lines = out.strip().splitlines()
    assert lines[0].startswith("# screen=")
    assert "package=com.test.app" in lines[0] and "1080x2400" in lines[0]
    assert lines[1].startswith("# elements=")
    assert lines[2] == "id\ttext\trid\tclickable"


def test_tsv_default_view_hides_system_chrome_and_unlabelled_rows(device: FakeDevice) -> None:
    rids = [row[2] for row in _rows(_run("--format", "tsv", "analyze"))[1:]]
    assert "clock" not in rids  # com.android.systemui id
    assert "notificationsButton" in rids and "homeTabBROWSE" in rids
    assert "" not in rids  # the unlabelled View is gone


def test_tsv_all_shows_everything(device: FakeDevice) -> None:
    default = _rows(_run("--format", "tsv", "analyze"))
    everything = _rows(_run("--format", "tsv", "analyze", "--all"))
    assert len(everything) > len(default)


def test_tsv_respects_field_order(device: FakeDevice) -> None:
    rows = _rows(_run("--format", "tsv", "analyze", "--fields", "rid,id"))
    assert rows[0] == ["rid", "id"]


def test_tsv_no_meta_emits_only_columns_and_rows(device: FakeDevice) -> None:
    out = _run("--format", "tsv", "analyze", "--no-meta")
    assert not out.startswith("#")
    assert out.strip().splitlines()[0] == "id\ttext\trid\tclickable"


def test_tsv_meta_selection_becomes_comment_lines(device: FakeDevice) -> None:
    out = _run("--format", "tsv", "analyze", "--meta", "tier_used")
    assert "# tier_used=hierarchy" in out


def test_invalid_global_format_still_exits_2() -> None:
    assert json.loads(_run("--format", "nope", "analyze", code=2))["error"]["code"] == "usage"


# ------------------------------------------------------------------ filters via the CLI


def test_region_and_clickable_compose_to_the_header_only(device: FakeDevice) -> None:
    rows = _rows(_run("--format", "tsv", "analyze", "--region", "0,0,1080,300", "--clickable"))
    assert [r[2] for r in rows[1:]] == ["notificationsButton"]


def test_where_text_is_case_insensitive(device: FakeDevice) -> None:
    rows = _rows(_run("--format", "tsv", "analyze", "--where-text", "brow"))
    assert [r[0] for r in rows[1:]] == ["rid:homeTabBROWSE"]


def test_limit_caps_the_rows(device: FakeDevice) -> None:
    rows = _rows(_run("--format", "tsv", "analyze", "--limit", "2"))
    assert len(rows) == 3  # header + 2


def test_ids_stay_addressable_after_filtering(device: FakeDevice) -> None:
    """A view must not renumber: the id a filtered row shows is the id `tap-and-analyze` takes."""
    rows = _rows(_run("--format", "tsv", "analyze", "--where-rid", "settingsSwitch"))
    element_id = rows[1][0]
    _run("tap-and-analyze", element_id)
    assert ("click", (927, 650)) in device.calls


# ------------------------------------------------------------------ interaction state


def test_state_flags_are_readable_without_a_screenshot(device: FakeDevice) -> None:
    payload = json.loads(_run("analyze", "--where-rid", "settingsSwitch"))
    switch = payload["elements"][0]
    assert switch["checkable"] is True
    assert switch["checked"] is True
    assert switch["long_clickable"] is False
    assert switch["password"] is False


def test_selected_distinguishes_the_active_tab(device: FakeDevice) -> None:
    payload = json.loads(_run("analyze", "--where-rid", "homeTab"))
    assert payload["elements"][0]["selected"] is True


def test_scrollable_lands_on_the_list_container(device: FakeDevice) -> None:
    rows = _rows(_run("--format", "tsv", "analyze", "--fields", "rid,scrollable"))
    assert ["feed", "true"] in rows


def test_missing_attributes_are_null_not_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """"Off" and "unknown" must stay distinguishable."""
    dev = FakeDevice(hierarchy_xml=SPARSE_XML)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    element = json.loads(_run("analyze", "--source", "hierarchy"))["elements"][0]
    for name in ("checkable", "checked", "selected", "scrollable", "long_clickable", "password"):
        assert element[name] is None, name


def test_compact_keeps_a_checkable_nodes_off_state(device: FakeDevice) -> None:
    """`checked: false` is the payload on a checkable node, so compact must not drop it."""
    dumped = json.loads(_run("--format", "compact", "analyze"))
    switch = next(e for e in dumped["elements"] if e.get("resource_id", "").endswith("settingsSwitch"))
    assert switch["checkable"] is True and switch["checked"] is True
    tab = next(e for e in dumped["elements"] if e.get("resource_id", "").endswith("BROWSE"))
    assert "checkable" not in tab  # not checkable → no tokens spent
    assert tab["selected"] is True


# ------------------------------------------------------------------ screenshot


def test_screenshot_region_crops_the_written_png(device: FakeDevice, tmp_path: Path) -> None:
    from PIL import Image

    out = tmp_path / "header.png"
    payload = json.loads(_run("screenshot", "--out", str(out), "--region", "0,0,1080,300"))
    assert payload["detail"] == str(out)
    assert Image.open(out).size == (1080, 300)


def test_screenshot_scale_and_max_width_shrink(device: FakeDevice, tmp_path: Path) -> None:
    from PIL import Image

    half = tmp_path / "half.png"
    _run("screenshot", str(half), "--scale", "0.5")
    assert Image.open(half).size == (540, 1200)

    capped = tmp_path / "capped.png"
    _run("screenshot", "--out", str(capped), "--max-width", "270")
    assert Image.open(capped).size == (270, 600)


def test_screenshot_max_width_never_upscales(device: FakeDevice, tmp_path: Path) -> None:
    from PIL import Image

    out = tmp_path / "big.png"
    _run("screenshot", "--out", str(out), "--max-width", "99999")
    assert Image.open(out).size == (1080, 2400)


def test_screenshot_region_is_clamped_to_the_screen(device: FakeDevice, tmp_path: Path) -> None:
    from PIL import Image

    out = tmp_path / "clamped.png"
    _run("screenshot", "--out", str(out), "--region", "-50,-50,99999,120")
    assert Image.open(out).size == (1080, 120)


def test_screenshot_region_off_screen_exits_2(device: FakeDevice, tmp_path: Path) -> None:
    out = _run("screenshot", "--out", str(tmp_path / "x.png"), "--region", "5000,0,6000,10", code=2)
    assert json.loads(out)["error"]["code"] == "usage"


@pytest.mark.parametrize("flag,value", [("--scale", "0"), ("--max-width", "0")])
def test_screenshot_non_positive_size_exits_2(
    device: FakeDevice, tmp_path: Path, flag: str, value: str
) -> None:
    out = _run("screenshot", "--out", str(tmp_path / "x.png"), flag, value, code=2)
    assert json.loads(out)["error"]["code"] == "usage"


def test_screenshot_annotate_with_region_is_a_usage_error(
    device: FakeDevice, tmp_path: Path
) -> None:
    out = _run(
        "screenshot", "--out", str(tmp_path / "x.png"), "--annotate", "--region", "0,0,10,10", code=2
    )
    assert "annotate" in json.loads(out)["error"]["message"]


def test_plain_screenshot_is_unchanged(device: FakeDevice, tmp_path: Path) -> None:
    from PIL import Image

    out = tmp_path / "plain.png"
    payload = json.loads(_run("screenshot", str(out)))
    assert payload == {"ok": True, "action": "screenshot", "detail": str(out)}
    assert Image.open(out).size == (1080, 2400)


# ------------------------------------------------------------------ orient / daemon --quiet


def test_orient_reports_the_foreground_app(device: FakeDevice) -> None:
    payload = json.loads(_run("orient"))
    assert payload["package"] == "com.test.app"


def test_daemon_start_quiet_omits_the_orientation_blob(
    device: FakeDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    import android_ui_analyser.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "start", lambda cfg: None)
    monkeypatch.setattr(daemon_mod, "is_running", lambda cfg: True)
    monkeypatch.setattr(daemon_mod, "status", lambda cfg: {"running": True})

    quiet = json.loads(_run("daemon", "start", "--quiet"))
    assert "orientation" not in quiet
    assert "aua orient" in quiet["hint"]

    loud = json.loads(_run("daemon", "start"))
    assert "orientation" in loud
