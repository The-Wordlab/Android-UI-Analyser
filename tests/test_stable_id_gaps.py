"""Bootstrap IDs and changing age subtitles must preserve the handle contract."""

import json

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ElementNotFoundError
from android_ui_analyser.mcp_server import _dispatch
from test_element_handles import BootDevice, clicks, engine_for, published, rows


@pytest.mark.parametrize("surface", ["engine", "cli", "mcp"])
def test_bootstrap_ids_are_actionable_after_rows_move(monkeypatch, surface):
    device = BootDevice(hierarchy_xml=rows("Orion", "Vega", nested=True))
    monkeypatch.setattr(
        Engine, "_prepare_session_target", lambda self, **kwargs: {"serial": device.serial}
    )
    engine = engine_for(device)
    if surface == "cli":
        monkeypatch.setattr(Engine, "_connect_target", lambda self, serial=None: device)
        result = CliRunner().invoke(app, ["session", "start", "--goal", "Inspect catalog"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
    elif surface == "mcp":
        payload = _dispatch(engine, "session_start", {"goal": "Inspect catalog"})
    else:
        payload = engine.session_start("Inspect catalog")
    elements = payload["observation"]["elements"]
    assert all(str(el["id"]).startswith("el:") for el in elements)
    target = next(el["id"] for el in elements if el.get("text") == "Orion")
    assert all("handle" not in el for el in elements)
    device._xml = rows("Vega", "Orion", nested=True)
    engine.tap(target, observe=False)
    assert clicks(device)[-1][1] > 400


def list_rows(*labels, nested=False, offset=0):
    content = rows(*labels, nested=nested, offset=offset)
    return content.replace(
        "<hierarchy>",
        '<hierarchy><node class="android.widget.ScrollView" scrollable="true" '
        'resource-id="com.example.fiction:id/list" bounds="[0,0][1080,1920]">',
    ).replace("</hierarchy>", "</node></hierarchy>")


@pytest.mark.parametrize("title,nested", [("Orion", False), ("Orion", True), ("A", False)])
def test_relative_age_changes_keep_row_and_child_handles(title, nested):
    device = BootDevice(hierarchy_xml=list_rows(f"{title} 3 days ago", "Vega 1 hour ago", nested=nested))
    engine = engine_for(device)
    first = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    row = next(el for el in first["elements"] if el.get("text") == f"{title} 3 days ago")
    target = next(
        (el["id"] for el in first["elements"] if el.get("parent") == row["id"]), row["id"]
    )
    device._xml = list_rows("Vega 2 hours ago", f"{title} 4 days ago", nested=nested, offset=100)
    engine.tap(target, observe=False)
    assert clicks(device)[-1][1] > 500
    assert published(engine)[f"{title} 4 days ago"] == row["id"]


def test_same_titles_with_different_ages_are_ambiguous():
    device = BootDevice(hierarchy_xml=list_rows("Orion 3 days ago"))
    engine = engine_for(device)
    target = published(engine)["Orion 3 days ago"]
    device._xml = list_rows("Orion 3 days ago", "Orion 4 days ago")
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)
    device._xml = list_rows("Orion 4 days ago")
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)


@pytest.mark.parametrize("old,new", [
    ("Orion 3 days ago", "Vega 4 days ago"),
    ("3 days ago", "4 days ago"),
    ("Orion 3 days", "Orion 4 days"),
    ("Orion 3 days ago", "Orion"),
])
def test_relative_age_does_not_erase_the_only_identity_or_other_content(old, new):
    device = BootDevice(hierarchy_xml=list_rows(old))
    engine = engine_for(device)
    target = published(engine)[old]
    device._xml = list_rows(new)
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)


def test_relative_age_is_not_removed_from_controls_outside_a_list():
    device = BootDevice(hierarchy_xml=rows("Orion 3 days ago"))
    engine = engine_for(device)
    target = published(engine)["Orion 3 days ago"]
    device._xml = rows("Orion 4 days ago")
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)
