"""A drag is one platform-neutral runtime operation: press, hold, move, release."""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError, UnsupportedPlatformCapabilityError, UsageError
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.platforms.android_device import Uiautomator2Device
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.drag_path import drag_path
from android_ui_analyser.platforms.runtime import TargetRuntime
from conftest import FakeDevice, make_config

runner = CliRunner()

HIERARCHY = """<hierarchy rotation="0">
  <node class="android.widget.Button" text="Handle" resource-id="example:id/handle"
        clickable="true" enabled="true" bounds="[100,200][200,300]"/>
  <node class="android.widget.FrameLayout" text="Slot" resource-id="example:id/slot"
        enabled="true" bounds="[600,1000][800,1200]"/>
</hierarchy>"""


class _NeutralRuntime(TargetRuntime):
    target_id = "neutral-drag-target"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def window_size(self) -> tuple[int, int]:
        return (1000, 2000)

    def dump_hierarchy(self, compressed: bool = False) -> str:
        return HIERARCHY

    def drag(self, x1, y1, x2, y2, duration_ms=500, hold_ms=0) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(("drag", (x1, y1, x2, y2, duration_ms, hold_ms)))


class _Platform(PlatformAdapter):
    name = "neutral-drag"
    capabilities = frozenset({"device.drag"})

    def connect(self, target_id: str | None = None):  # type: ignore[no-untyped-def]
        raise AssertionError("the engine is handed its runtime")

    def list_targets(self):  # type: ignore[no-untyped-def]
        return []

    def normalize_tree(self, raw_tree, screen_size, *, ignored_app_ids=()):  # type: ignore[no-untyped-def]
        return NormalizedTree(elements=[])


class _NoDragPlatform(_Platform):
    name = "neutral-no-drag"
    capabilities = frozenset()


def test_a_neutral_adapter_drags_without_touching_android_tooling(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in ("click", "swipe", "long_click"):
        monkeypatch.setattr(
            Uiautomator2Device, name, lambda *_a, **_k: pytest.fail("Android tooling was used")
        )
    runtime = _NeutralRuntime()
    eng = Engine(Config(), device=runtime, platform=_Platform(Config()))

    result = eng.drag(coords=(10, 20, 300, 400), duration_ms=250, hold_ms=100, observe=False)

    assert result.ok and result.action == "drag" and result.target == [10, 20, 300, 400]
    assert runtime.calls == [("drag", (10, 20, 300, 400, 250, 100))]


def test_an_adapter_without_the_capability_refuses_explicitly() -> None:
    runtime = _NeutralRuntime()
    eng = Engine(Config(), device=runtime, platform=_NoDragPlatform(Config()))

    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        eng.drag(coords=(1, 2, 3, 4), observe=False)

    assert caught.value.code == "platform_capability_unsupported"
    assert runtime.calls == []


def test_a_runtime_that_does_not_override_drag_reports_it_unsupported() -> None:
    class Bare(TargetRuntime):
        target_id = "bare"

    with pytest.raises(DeviceError) as caught:
        Bare().drag(0, 0, 1, 1)
    assert caught.value.code == "drag_unsupported"


def _engine() -> tuple[Engine, FakeDevice]:
    device = FakeDevice(hierarchy_xml=HIERARCHY)
    return Engine(make_config(), device=device), device


def test_elements_name_both_ends_and_coordinates_mix_with_them() -> None:
    eng, device = _engine()
    eng.analyze(source="hierarchy")

    eng.drag(selector={"rid": "handle"}, to_selector={"rid": "slot"}, observe=False)
    eng.drag(selector={"rid": "handle"}, to_coords=(5, 6), observe=False)
    eng.drag(from_coords=(7, 8), to_selector={"text": "Slot"}, observe=False)

    assert [args[:4] for name, args in device.calls if name == "drag"] == [
        (150, 250, 700, 1100),
        (150, 250, 5, 6),
        (7, 8, 700, 1100),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"to_coords": (1, 2)},  # no start
        {"from_coords": (1, 2)},  # no end
        {"from_coords": (1, 2), "selector": {"rid": "handle"}, "to_coords": (3, 4)},  # two starts
        {"coords": (1, 2, 3, 4), "duration_ms": -1},
    ],
)
def test_an_ambiguous_or_incomplete_drag_sends_nothing(kwargs) -> None:  # type: ignore[no-untyped-def]
    eng, device = _engine()
    with pytest.raises(UsageError):
        eng.drag(observe=False, **kwargs)
    assert not [c for c in device.calls if c[0] == "drag"]


def test_the_path_is_smooth_and_ends_exactly_on_the_target() -> None:
    points, pause = drag_path(0, 0, 100, -50, 400)

    assert len(points) >= 10 and points[-1] == (100, -50)
    assert all(a[0] <= b[0] for a, b in zip(points, points[1:], strict=False))
    assert pause * len(points) == pytest.approx(0.4)
    assert len(drag_path(0, 0, 1, 1, 0)[0]) >= 2  # an instant drag still moves


def test_cli_and_mcp_reach_the_same_engine_drag(monkeypatch, tmp_path, fake_cli_device) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_LEASE__REGISTRY_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    device = fake_cli_device(FakeDevice(hierarchy_xml=HIERARCHY))
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _e, serial=None: device)

    cli = runner.invoke(
        app,
        ["drag-and-analyze", "--rid", "handle", "--to-rid", "slot", "--duration-ms", "300",
         "--hold-ms", "50"],
    )
    assert cli.exit_code == 0, cli.output
    cli_coords = runner.invoke(app, ["drag-and-analyze", "--coords", "1", "2", "3", "4"])
    assert cli_coords.exit_code == 0, cli_coords.output

    server = build_server(Engine(make_config(), device=device))

    async def run() -> None:
        async with create_connected_server_and_client_session(server) as client:
            first = await client.call_tool(
                "drag_and_analyze",
                {"rid": "handle", "to_rid": "slot", "duration_ms": 300, "hold_ms": 50},
            )
            second = await client.call_tool(
                "drag_and_analyze", {"coords": [1, 2, 3, 4]}
            )
            for result in (first, second):
                assert not result.isError, result
                assert json.loads(result.content[0].text)["action"] == "drag"

    anyio.run(run)

    drags = [args for name, args in device.calls if name == "drag"]
    assert drags == [(150, 250, 700, 1100, 300, 50), (1, 2, 3, 4, 500, 0)] * 2
