"""``--with-image`` — raw screenshot return on analyze, actions, and MCP.

The agent-facing contract: any perception or action call can also hand back the actual
pixels (path in ``meta.raw_image``; inline image block over MCP), so vision-capable
callers can *see* the screen in the same round-trip they use to address it.
"""

from __future__ import annotations

import json
from pathlib import Path

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from conftest import FakeDevice, make_config

HIERARCHY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Hello" bounds="[0,0][1080,120]"/>
  <node index="1" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""

PNG_MAGIC = b"\x89PNG"


def _engine() -> Engine:
    return Engine(make_config(), device=FakeDevice(hierarchy_xml=HIERARCHY_XML))


def test_analyze_with_image_saves_raw_screenshot_and_sets_meta() -> None:
    result = _engine().analyze(source="hierarchy", with_image=True)

    assert result.meta.raw_image is not None
    saved = Path(result.meta.raw_image)
    assert saved.exists()
    assert saved.read_bytes()[:4] == PNG_MAGIC
    assert "screen" in saved.name


def test_analyze_with_image_honours_explicit_path(tmp_path: Path) -> None:
    out = tmp_path / "shot.png"

    result = _engine().analyze(source="hierarchy", with_image=str(out))

    assert result.meta.raw_image == str(out)
    assert out.exists()


def test_analyze_without_with_image_leaves_meta_unset() -> None:
    result = _engine().analyze(source="hierarchy")

    assert result.meta.raw_image is None


def test_sequential_captures_never_clobber_each_other() -> None:
    engine = _engine()

    first = engine.analyze(source="hierarchy", with_image=True)
    second = engine.analyze(source="hierarchy", with_image=True)

    assert first.meta.raw_image != second.meta.raw_image
    assert Path(first.meta.raw_image).exists()
    assert Path(second.meta.raw_image).exists()


def test_tap_with_image_lands_in_observation() -> None:
    engine = _engine()
    analyzed = engine.analyze(source="hierarchy")
    button = next(e for e in analyzed.elements if e.text == "Continue")

    result = engine.tap(button.id, with_image=True)

    assert result.observation is not None
    raw = result.observation.meta.raw_image
    assert raw is not None
    assert Path(raw).exists()


def test_mcp_analyze_with_image_returns_inline_image_block() -> None:
    server = build_server(_engine())

    async def run():  # type: ignore[no-untyped-def]
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(
                "analyze_screen", {"source": "hierarchy", "with_image": True}
            )

    result = anyio.run(run)
    assert not result.isError, result
    kinds = [getattr(block, "type", None) for block in result.content]
    assert "text" in kinds
    assert "image" in kinds
    image = next(b for b in result.content if getattr(b, "type", None) == "image")
    assert image.mimeType == "image/png"
    assert len(image.data) > 100
    text = next(b for b in result.content if getattr(b, "type", None) == "text")
    assert json.loads(text.text)["meta"]["raw_image"]


def test_mcp_screenshot_returns_inline_image_block(tmp_path: Path) -> None:
    server = build_server(_engine())
    out = tmp_path / "mcp_shot.png"

    async def run():  # type: ignore[no-untyped-def]
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool("screenshot", {"path": str(out)})

    result = anyio.run(run)
    assert not result.isError, result
    kinds = [getattr(block, "type", None) for block in result.content]
    assert "image" in kinds


def test_session_with_image_default(tmp_path: Path) -> None:
    """Config/session default attaches raw_image without a per-call flag."""
    cfg = make_config(output={"with_image": True}, cache={"dir": str(tmp_path)})
    device = FakeDevice(hierarchy_xml=HIERARCHY_XML)
    engine = Engine(cfg, device=device)
    assert engine._default_with_image is True
    result = engine.analyze(source="hierarchy")
    assert result.meta.raw_image
    assert Path(result.meta.raw_image).is_file()
