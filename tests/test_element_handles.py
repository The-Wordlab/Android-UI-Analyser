"""A published handle follows the same item, never the slot a new item occupies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.dashboard import _DashboardState
from android_ui_analyser.element_handles import HandleStore
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ElementNotFoundError
from android_ui_analyser.hierarchy import parse_hierarchy
from android_ui_analyser.mcp_server import _selector_from_args
from android_ui_analyser.platforms.identity import TargetRef
from conftest import FakeDevice, make_config


def rows(*labels: str, offset: int = 0, nested: bool = False) -> str:
    nodes = []
    for i, label in enumerate(labels):
        y = 200 + i * 200 + offset
        child = (
            f'<node class="android.widget.Button" text="Delete" clickable="true" '
            f'enabled="true" resource-id="com.example.fiction:id/delete" '
            f'bounds="[800,{y}][1000,{y + 100}]"/>'
            if nested
            else ""
        )
        nodes.append(
            f'<node package="com.example.fiction" class="android.widget.Button" '
            f'text="{label}" resource-id="com.example.fiction:id/row" clickable="true" '
            f'enabled="true" bounds="[40,{y}][1040,{y + 120}]">{child}</node>'
        )
    return f"<hierarchy>{''.join(nodes)}</hierarchy>"


class BootDevice(FakeDevice):
    boot = "fixture-boot-one"

    def instance_token(self) -> str:
        return self.boot


def engine_for(device: FakeDevice, **overrides) -> Engine:
    return Engine(make_config(memory={"enabled": False}, **overrides), device=device)


def published(engine: Engine) -> dict[str, str]:
    payload = engine.analyze(source="hierarchy", with_ocr=False).as_dict()
    return {row["text"]: row["id"] for row in payload["elements"]}


def clicks(device: FakeDevice) -> list[tuple]:
    return [args for name, args in device.calls if name == "click"]


def test_handles_follow_reordered_rows_and_reject_removed_items() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion", "Vega"))
    engine = engine_for(device)
    first = published(engine)
    assert first["Orion"].startswith("el:")
    device._xml = rows("Vega", "Orion")
    assert published(engine) == first
    result = engine.tap(first["Orion"], observe=False)
    assert result.ok and clicks(device)[-1][1] > 400
    device._xml = rows("Vega", "Lyra")
    before = len(clicks(device))
    with pytest.raises(ElementNotFoundError) as exc:
        engine.tap(first["Orion"], observe=False)
    assert exc.value.observation is not None
    assert len(clicks(device)) == before
    assert published(engine)["Lyra"] not in first.values()


def test_duplicate_child_controls_follow_their_owning_row() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion", "Vega", nested=True))
    engine = engine_for(device)
    first = engine.analyze(source="hierarchy").as_dict()
    target = next(el["id"] for el in first["elements"] if el["text"] == "Delete")
    device._xml = rows("Vega", "Orion", nested=True)
    engine.tap(target, observe=False)
    assert clicks(device)[-1][1] > 400
    device._xml = rows("Orion", nested=True)
    remaining = engine.analyze(source="hierarchy").as_dict()
    assert next(el["id"] for el in remaining["elements"] if el["text"] == "Delete") == target


def test_label_without_resource_id_survives_scroll_and_keyboard_reflow() -> None:
    device = BootDevice(
        hierarchy_xml=rows("Orion").replace('resource-id="com.example.fiction:id/row"', "")
    )
    engine = engine_for(device)
    target = published(engine)["Orion"]
    device._xml = rows("Orion", offset=600).replace('resource-id="com.example.fiction:id/row"', "")
    assert published(engine)["Orion"] == target
    engine.tap(target, observe=False)
    assert clicks(device)[-1][1] > 800


def test_editable_value_and_checked_state_are_not_identity() -> None:
    def screen(value: str, checked: str, y: int) -> str:
        return f'''<hierarchy>
          <node class="android.widget.EditText" text="{value}" clickable="true"
            enabled="true" bounds="[40,{y}][1040,{y + 100}]"/>
          <node class="android.widget.Switch" text="Alerts" resource-id="app:id/alerts"
            checkable="true" checked="{checked}" bounds="[40,800][1040,900]"/>
        </hierarchy>'''

    device = BootDevice(hierarchy_xml=screen("", "false", 500))
    engine = engine_for(device)
    first = engine.analyze(source="hierarchy").as_dict()
    device._xml = screen("typed value", "true", 200)
    second = engine.analyze(source="hierarchy").as_dict()
    assert [el["id"] for el in second["elements"]] == [el["id"] for el in first["elements"]]
    state = (Path(engine._lease_registry_dir) / "element-identities").glob("*.json")
    assert all("typed value" not in path.read_text() for path in state)


def test_process_restart_and_different_observation_cache_keep_attested_handles(tmp_path) -> None:
    device = BootDevice(hierarchy_xml=rows("Orion", "Vega"))
    cfg = {"lease": {"registry_dir": str(tmp_path / "coordination")}}
    first = published(engine_for(device, cache={"dir": str(tmp_path / "a")}, **cfg))
    device._xml = rows("Vega", "Orion")
    second = engine_for(device, cache={"dir": str(tmp_path / "b")}, **cfg)
    assert published(second) == first
    second.tap(first["Orion"], observe=False)
    assert clicks(device)[-1][1] > 400


def test_reboot_invalidates_handles_even_on_the_unchanged_hierarchy_fast_path() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion"))
    engine = engine_for(device)
    target = published(engine)["Orion"]
    device.boot = "fixture-boot-two"
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)
    assert published(engine)["Orion"] != target


def test_reboot_publishes_new_handles_instead_of_an_empty_unchanged_delta() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion"))
    engine = engine_for(device)
    old = published(engine)["Orion"]
    device.boot = "fixture-boot-two"
    delta = engine.analyze(source="hierarchy").as_dict("delta")
    assert delta["elements"] and delta["elements"][0]["id"] != old
    assert not delta["meta"].get("unchanged", False)


def test_a_dialog_button_does_not_keep_its_handle_when_its_subject_changes() -> None:
    def dialog(subject: str) -> str:
        return f"""<hierarchy>
          <node class="Dialog" resource-id="app:id/dialog" bounds="[0,0][1080,1000]">
            <node class="TextView" text="Delete {subject}?" bounds="[40,100][900,200]"/>
            <node class="Button" text="Delete" resource-id="app:id/confirm"
              clickable="true" enabled="true" bounds="[40,300][900,400]"/>
          </node>
        </hierarchy>"""

    device = BootDevice(hierarchy_xml=dialog("Orion"))
    engine = engine_for(device)
    target = published(engine)["Delete"]
    device._xml = dialog("Vega")
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)


def test_same_tree_in_another_app_surface_cannot_adopt_a_handle() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion"))
    engine = engine_for(device)
    target = published(engine)["Orion"]
    device._act = ".OtherSurface"
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)


def test_unknown_boot_does_not_adopt_another_runtime_handle() -> None:
    device = FakeDevice(hierarchy_xml=rows("Orion"))
    target = published(engine_for(device))["Orion"]
    with pytest.raises(ElementNotFoundError):
        engine_for(FakeDevice(hierarchy_xml=rows("Orion"))).tap(target, observe=False)
    assert not clicks(device)


def test_indistinguishable_rows_are_refused_instead_of_matched_by_position() -> None:
    device = BootDevice(hierarchy_xml=rows("Duplicate", "Duplicate"))
    engine = engine_for(device)
    payload = engine.analyze(source="hierarchy").as_dict()
    assert len({el["id"] for el in payload["elements"]}) == 2
    with pytest.raises(ElementNotFoundError):
        engine.tap(payload["elements"][0]["id"], observe=False)
    assert not clicks(device)


def test_handle_never_crosses_app_or_resource_namespace() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion"))
    engine = engine_for(device)
    target = published(engine)["Orion"]
    device._xml = rows("Orion").replace("com.example.fiction", "com.example.other")
    with pytest.raises(ElementNotFoundError):
        engine.tap(target, observe=False)
    assert not clicks(device)


def test_corrupt_registry_expires_existing_handles(tmp_path) -> None:
    store = HandleStore(tmp_path, TargetRef("sample-os", "sample-device"), "boot")
    elements = parse_hierarchy(rows("Orion"), (1080, 1920))
    first = store.assign(elements, app="fiction", surface=None)[0].handle
    store.path.write_text("{corrupt")
    second = store.assign(elements, app="fiction", surface=None)[0].handle
    assert first != second


def test_duplicate_handles_in_a_corrupt_registry_are_never_adopted(tmp_path) -> None:
    store = HandleStore(tmp_path, TargetRef("sample-os", "sample-device"), "boot")
    elements = parse_hierarchy(rows("Orion", "Vega"), (1080, 1920))
    first = store.assign(elements, app="fiction", surface=None)
    state = json.loads(store.path.read_text())
    for record in state["records"].values():
        record["handle"] = first[0].handle
    store.path.write_text(json.dumps(state))
    second = store.assign(elements, app="fiction", surface=None)
    assert not {el.handle for el in first} & {el.handle for el in second}


def test_eviction_does_not_recycle_a_handle(tmp_path, monkeypatch) -> None:
    from android_ui_analyser import element_handles

    monkeypatch.setattr(element_handles, "MAX_RECORDS", 2)
    store = HandleStore(tmp_path, TargetRef("sample-os", "sample-device"), "boot")

    def observe(label):
        return store.assign(
            parse_hierarchy(rows(label), (1080, 1920)), app="fiction", surface=None
        )[0]

    first = observe("Orion").handle
    observe("Vega")
    observe("Lyra")
    assert observe("Orion").handle != first
    assert len(json.loads(store.path.read_text())["records"]) == 2


def test_parallel_processes_share_one_registry_without_lost_updates(tmp_path) -> None:
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    script = """
import sys
from pathlib import Path
from android_ui_analyser.element_handles import HandleStore
from android_ui_analyser.platforms.identity import TargetRef
from android_ui_analyser.schema import Element
store = HandleStore(Path(sys.argv[1]), TargetRef("sample-os", "sample-target"), "boot")
element = Element(id=0, type="Button", text="Orion", resource_id="sample/row",
                  bounds=(0,0,100,100), center=(50,50), clickable=True)
print(store.assign([element], app="fiction", surface=None)[0].handle)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))

    def run(_):
        return subprocess.check_output(
            [sys.executable, "-c", script, str(tmp_path)], env=env, text=True, timeout=30
        ).strip()

    with ThreadPoolExecutor(max_workers=3) as pool:
        handles = list(pool.map(run, range(3)))
    assert len(set(handles)) == 1
    assert run(None) == handles[0]


def test_query_and_full_observation_publish_the_same_handle() -> None:
    device = BootDevice(hierarchy_xml=rows("Orion", "Vega"))
    engine = engine_for(device)
    target = published(engine)["Orion"]
    answer = engine.analyze(query="Orion", source="hierarchy").as_dict()
    assert answer["elements"][0]["id"] == target


def test_cli_and_mcp_and_dashboard_resolve_the_published_handle(monkeypatch) -> None:
    device = BootDevice(hierarchy_xml=rows("Orion", "Vega"))
    monkeypatch.setattr(Engine, "_connect_target", lambda self, serial=None: device)
    runner = CliRunner()
    observed = runner.invoke(app, ["analyze", "--source", "hierarchy"])
    assert observed.exit_code == 0, observed.output
    element = next(el for el in json.loads(observed.stdout)["elements"] if el["text"] == "Orion")
    device._xml = rows("Vega", "Orion")
    result = runner.invoke(app, ["tap-and-analyze", "--by", "id", element["id"]])
    assert result.exit_code == 0, result.output
    assert clicks(device)[-1][1] > 400
    engine = engine_for(device)
    engine.tap(
        selector=_selector_from_args(
            {
                "id": element["id"],
                "stable_key": element.get("stable_key"),
            }
        ),
        observe=False,
    )
    assert clicks(device)[-1][1] > 400
    engine.tap(selector=_DashboardState._inspection_selector(element["id"], element), observe=False)
    assert clicks(device)[-1][1] > 400


def test_published_handle_survives_serialization_and_filtering() -> None:
    from android_ui_analyser.schema import AnalyzeResult

    engine = engine_for(BootDevice(hierarchy_xml=rows("Orion", "Vega", nested=True)))
    original = engine.analyze(source="hierarchy").as_dict()
    reloaded = AnalyzeResult.model_validate(original)
    assert reloaded.as_dict() == original
    assert [el.published_id for el in reloaded.elements] == [
        el["id"] for el in original["elements"]
    ]
    filtered = reloaded.model_copy(update={"elements": [reloaded.elements[-1]]}).as_dict()
    assert filtered["elements"][0]["id"] == original["elements"][-1]["id"]


def test_neutral_adapter_tracks_and_acts_without_android_imports(monkeypatch) -> None:
    import builtins

    from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
    from android_ui_analyser.schema import Element
    from test_platform_runtime import _NeutralRuntime

    class Runtime(_NeutralRuntime):
        order = ["Orion", "Vega"]
        tapped = []

        def instance_token(self):
            return "neutral-boot"

        def dump_hierarchy(self, compressed=False):
            return json.dumps(self.order)

        def click(self, x, y):
            self.tapped.append((x, y))

    class Platform(PlatformAdapter):
        name = "neutral-handles"
        capabilities = frozenset({"ui.tree", "ui.input"})

        def connect(self, target_id=None):
            raise AssertionError("the runtime is injected")

        def list_targets(self):
            return []

        def normalize_tree(self, raw_tree, screen_size, *, geometry=None, ignored_app_ids=()):
            return NormalizedTree(
                [
                    Element(
                        id=i,
                        type="Button",
                        text=label,
                        resource_id="sample/row",
                        bounds=(10, 100 + i * 100, 200, 180 + i * 100),
                        center=(105, 140 + i * 100),
                        clickable=True,
                    )
                    for i, label in enumerate(json.loads(raw_tree))
                ],
                app_id="fiction",
            )

    imports = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("adbutils", "uiautomator2", "android_ui_analyser.platforms.android")):
            imports.append(name)
            raise AssertionError(f"native tooling reached: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = make_config(
        device={"platform": "neutral-handles"},
        memory={"enabled": False},
        output={"with_image": False},
        lease={"enabled": False},
    )
    runtime = Runtime()
    engine = Engine(config, platform=Platform(config), device=runtime)
    target = published(engine)["Orion"]
    runtime.order = ["Vega", "Orion"]
    engine.tap(target, observe=False)
    assert runtime.tapped[-1][1] > 200
    assert not imports
