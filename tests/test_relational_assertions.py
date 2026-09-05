"""Shared structural assertions across engine, flows, suites, CLI, and MCP."""

from __future__ import annotations

import json
from collections.abc import Sequence

import anyio
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.config import Config
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import parse_flow_yaml, render_flow_yaml
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.platforms import (
    DisplayGeometry,
    NormalizedTree,
    PlatformAdapter,
    ScreenImage,
)
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import DeviceInfo, Element, Source
from android_ui_analyser.suite import parse_suite, run_suite
from conftest import FakeDevice, make_config


def _elements() -> list[Element]:
    def element(
        id_: int,
        *,
        text: str | None = None,
        rid: str | None = None,
        parent: int | None = None,
        source: Source = Source.hierarchy,
    ) -> Element:
        top = id_ * 20
        return Element(
            id=id_,
            type="Node",
            text=text,
            resource_id=rid,
            bounds=(0, top, 200, top + 10),
            center=(100, top + 5),
            parent=parent,
            source=source,
        )

    return [
        element(0, rid="dev.aua.fixture:id/catalog"),
        element(1, rid="dev.aua.fixture:id/cardA", parent=0),
        element(2, text="Starter", rid="dev.aua.fixture:id/title", parent=1),
        element(3, text="$7.99", rid="dev.aua.fixture:id/price", parent=1),
        element(4, rid="dev.aua.fixture:id/cardB", parent=0),
        element(5, text="Professional", rid="dev.aua.fixture:id/title", parent=4),
        element(6, text="$12.99", rid="dev.aua.fixture:id/price", parent=4),
    ]


class _TreePlatform(PlatformAdapter):
    name = "tree-test"
    capabilities = frozenset({"ui.screenshot", "ui.tree"})

    def __init__(self, config: Config, elements: Sequence[Element]) -> None:
        super().__init__(config)
        self.elements = list(elements)
        self.calls: list[str] = []

    def connect(self, target_id: str | None = None) -> Device:
        raise AssertionError("injected runtime should be used")

    def list_targets(self) -> list[DeviceInfo]:
        return []

    def dump_tree(self, runtime: Device, *, compact: bool = False) -> str:
        self.calls.append("dump_tree")
        return "opaque-native-tree"

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        assert raw_tree == "opaque-native-tree"
        assert geometry is not None
        self.calls.append("normalize_tree")
        return NormalizedTree(list(self.elements), app_id="dev.aua.fixture")

    def capture_screenshot(self, runtime: Device) -> ScreenImage:
        return runtime.screenshot()


def _engine(elements: Sequence[Element] | None = None) -> tuple[Engine, _TreePlatform, FakeDevice]:
    config = make_config(memory={"enabled": False}, lease={"enabled": False})
    platform = _TreePlatform(config, elements or _elements())
    runtime = FakeDevice(package="dev.aua.fixture")
    return Engine(config, device=runtime, platform=platform), platform, runtime


def test_relations_use_only_normalized_parent_links() -> None:
    engine, platform, runtime = _engine()

    within = engine.expect(
        rid="title",
        text_is="Starter",
        within={"rid": "cardA"},
    )
    sibling = engine.expect(
        text="$7.99",
        same_parent_as={"text": "Starter"},
    )
    contains = engine.expect(
        rid="cardA",
        contains_all=[{"text": "Starter"}, {"text": "$7.99"}],
    )

    assert within.ok and sibling.ok and contains.ok
    assert platform.calls == ["dump_tree", "normalize_tree"] * 3
    assert not any(name in {"shell", "logcat", "dump_hierarchy"} for name, _args in runtime.calls)


def test_relations_fail_closed_without_structural_evidence() -> None:
    elements = _elements()
    elements.append(
        Element(
            id=7,
            type="ocr",
            text="Floating",
            bounds=(0, 400, 100, 440),
            center=(50, 420),
            source=Source.ocr,
        )
    )
    engine, _platform, _runtime = _engine(elements)

    result = engine.expect(text="Floating", within={"rid": "catalog"})

    assert not result.ok
    assert "structural_evidence_unavailable" in (result.detail or "")


def test_index_is_strict_and_never_falls_back_to_first_match() -> None:
    engine, _platform, _runtime = _engine()

    result = engine.expect(rid="title", text_is="Starter", index=9)

    assert not result.ok
    assert "predicate=index" in (result.detail or "")
    assert "actual=2 matches" in (result.detail or "")


def test_reading_order_uses_normalized_traversal_not_geometry() -> None:
    elements = _elements()
    # Reverse the geometry while retaining the adapter's structural traversal order.
    elements[2] = elements[2].model_copy(update={"center": (900, 900)})
    elements[5] = elements[5].model_copy(update={"center": (10, 10)})
    engine, _platform, _runtime = _engine(elements)

    result = engine.flow_run(
        yaml="""steps:
  - assert_order:
      axis: reading
      selectors:
        - {text: Starter}
        - {text: Professional}
"""
    )

    assert result["ok"] is True, result
    assert "positions=[2, 5]" in result["steps_run"][0]["assertion"]


def test_flow_and_suite_round_trip_relational_selectors() -> None:
    yaml = """steps:
  - assert:
      id: cardA
      contains_all:
        - {text: Starter}
        - {text: "$7.99"}
  - assert:
      text: "$7.99"
      same_parent_as: {text: Starter}
"""
    flow = parse_flow_yaml(yaml)
    assert parse_flow_yaml(render_flow_yaml(flow)).steps == flow.steps

    suite = parse_suite(
        """name: relational
checks:
  - expect:
      rid: cardA
      contains_all: [{text: Starter}, {text: "$7.99"}]
"""
    )
    engine, _platform, _runtime = _engine()
    assert run_suite(engine, suite).ok


_NESTED_XML = """<hierarchy>
  <node class="android.view.ViewGroup" resource-id="dev.aua.fixture:id/catalog"
        bounds="[0,0][500,500]">
    <node class="android.view.ViewGroup" resource-id="dev.aua.fixture:id/cardA"
          bounds="[0,0][250,200]">
      <node class="android.widget.TextView" text="Starter"
            resource-id="dev.aua.fixture:id/title" bounds="[10,10][200,50]" />
      <node class="android.widget.TextView" text="$7.99"
            resource-id="dev.aua.fixture:id/price" bounds="[10,60][200,100]" />
    </node>
  </node>
</hierarchy>"""


def test_cli_relational_options_reach_shared_engine(monkeypatch) -> None:
    runtime = FakeDevice(hierarchy_xml=_NESTED_XML, package="dev.aua.fixture")
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: runtime)

    result = CliRunner().invoke(
        app,
        [
            "expect-and-analyze",
            "--text",
            "$7.99",
            "--same-parent-as",
            "text:Starter",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_mcp_relational_schema_and_dispatch() -> None:
    runtime = FakeDevice(hierarchy_xml=_NESTED_XML, package="dev.aua.fixture")
    server = build_server(Engine(make_config(), device=runtime))

    async def run() -> tuple[dict, dict]:
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            tool = next(tool for tool in listed.tools if tool.name == "expect_and_analyze")
            called = await client.call_tool(
                "expect_and_analyze",
                {
                    "rid": "cardA",
                    "contains_all": [{"text": "Starter"}, {"text": "$7.99"}],
                },
            )
            text = next(block.text for block in called.content if block.type == "text")
            return tool.inputSchema, json.loads(text)

    schema, result = anyio.run(run)
    assert "within" in schema["properties"]
    assert schema["properties"]["contains_all"]["minItems"] == 1
    assert result["ok"] is True


def test_parent_is_a_supported_projection_field() -> None:
    payload = {
        "screen": {"height": 500},
        "elements": [element.model_dump(mode="json") for element in _elements()],
        "meta": {},
    }
    projected = Projection.parse(fields="id,parent").apply(payload)

    assert projected["elements"][2] == {"id": 2, "parent": 1}
