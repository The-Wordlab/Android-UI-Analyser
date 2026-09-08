"""An uncertain action keeps its error while publishing usable, projected evidence."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from android_ui_analyser import journal
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.mic import MicDeliveryUncertainError
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

_XML = '''<hierarchy>
  <node class="TextView" package="com.android.systemui" text="System clock"
    resource-id="com.android.systemui:id/clock" bounds="[0,0][400,40]"/>
  <node class="Button" package="com.test.app" text="Send" clickable="true"
    enabled="true" resource-id="com.test.app:id/send" bounds="[40,100][360,200]"/>
</hierarchy>'''


def test_mcp_uncertain_result_publishes_existing_observation_without_replaying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice(hierarchy_xml=_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(make_config(memory={"enabled": False}), device=device)
    observed = engine.analyze(
        source="hierarchy", with_ocr=False, with_image=str(tmp_path / "already-observed.png")
    )
    injected = 0
    closed_fingerprints: list[str | None] = []
    close_turn = engine.close_caller_turn

    def close(fingerprint: str | None = None) -> None:
        closed_fingerprints.append(fingerprint)
        close_turn(fingerprint)

    def uncertain(*_args: object, **_kwargs: object) -> None:
        nonlocal injected
        injected += 1
        raise MicDeliveryUncertainError().with_result(
            ActionResult(ok=False, action="mic-inject", observation=observed).model_dump(mode="json")
        )

    def never_await(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an uncertain action must not run its success-bound predicate")

    monkeypatch.setattr(engine, "mic_inject", uncertain)
    monkeypatch.setattr(engine, "close_caller_turn", close)
    monkeypatch.setattr(engine, "await_predicate", never_await)
    server = build_server(engine)

    async def run() -> tuple[dict, dict]:
        async with create_connected_server_and_client_session(server) as client:
            acquisitions_before = list(device.calls)
            failed = await client.call_tool("mic_inject_and_analyze", {
                "path": str(tmp_path / "fixture.wav"),
                "observe_fields": "id,text,resource_id",
                "until": "text:Delivered",
                "until_timeout": 10,
            })
            assert device.calls == acquisitions_before
            assert any(block.type == "image" for block in failed.content)
            error = json.loads(failed.content[0].text)["error"]
            result = error["result"]
            visible = result["observation"]["elements"]
            assert len(visible) == 1
            control = visible[0]
            assert set(control) == {"id", "text", "resource_id"}
            assert control["id"].startswith("el:")
            assert result["observation_contract"]["action_succeeded"] is False
            assert closed_fingerprints[-1] == observed.meta.fingerprint
            tapped = await client.call_tool("tap_and_analyze", {"id": control["id"]})
            return error, json.loads(tapped.content[0].text)

    try:
        error, tapped = anyio.run(run)
    finally:
        engine.close()
    assert error["code"] == "mic_delivery_uncertain"
    assert "Do not retry blindly" in error["hint"]
    assert error["result"]["ok"] is False
    assert injected == 1
    assert tapped["ok"] is True
    assert len([call for call in device.calls if call[0] == "click"]) == 1
    events = journal.read_since(engine.config.cache.dir, device.serial, limit=10)
    event = next(e for e in events if e["cmd"] == "mic_inject_and_analyze")
    assert event["ok"] is False
    assert event["error"]["code"] == "mic_delivery_uncertain"
    assert event["result"]["observation_contract"]["action_succeeded"] is False


def test_mcp_decoration_failure_preserves_uncertain_error_and_published_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import coaching

    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice(hierarchy_xml=_XML))
    observation = engine.analyze(source="hierarchy", with_ocr=False)
    attempts = 0

    def uncertain(*_args: object, **_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise MicDeliveryUncertainError().with_result(
            ActionResult(ok=False, action="mic-inject", observation=observation).model_dump(mode="json")
        )

    def decoration_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("evidence decoration failed")

    monkeypatch.setattr(engine, "mic_inject", uncertain)
    monkeypatch.setattr(coaching, "decorate_result", decoration_failure)
    server = build_server(engine)

    async def run() -> dict:
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("mic_inject_and_analyze", {"path": "fixture.wav"})
            return json.loads(result.content[0].text)["error"]

    try:
        error = anyio.run(run)
    finally:
        engine.close()
    assert attempts == 1
    assert error["code"] == "mic_delivery_uncertain"
    assert "Do not retry blindly" in error["hint"]
    assert error["result"]["ok"] is False
    assert all(row["id"].startswith("el:") for row in error["result"]["observation"]["elements"])
