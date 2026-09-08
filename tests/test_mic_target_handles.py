"""Published microphone targets survive CLI and MCP parsing without becoming audio-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import cli, mic
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.mcp_server import _dispatch, _optional_mic_target, build_server
from conftest import FakeDevice, make_config
from test_mic import HOLD_XML, _prepared, _write_wav


@pytest.fixture
def controlled_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Engine, list[str]]:
    device = FakeDevice(hierarchy_xml=HOLD_XML, serial="emulator-5554", width=400, height=800)
    engine = Engine(
        make_config(
            cache={"dir": str(tmp_path / "cache")},
            memory={"enabled": False},
            daemon={"enabled": False},
        ),
        device=device,
    )
    prepared = _prepared(tmp_path / "voice.wav")
    _write_wav(prepared.wav.path)
    events: list[str] = []
    monkeypatch.setattr(mic, "prepare_injection", lambda *_a, **_kw: prepared)
    monkeypatch.setattr(mic, "synthesize_speech", lambda *_a, **_kw: prepared.wav.path)
    monkeypatch.setattr(mic, "wait_for_recording", lambda _d: {"ready": True})
    monkeypatch.setattr(mic, "inject_prepared", lambda _p: events.append("audio"))
    monkeypatch.setattr(device, "touch_down", lambda *_a: events.append("down"))
    monkeypatch.setattr(device, "touch_up", lambda *_a: events.append("up"))
    return engine, events


@pytest.mark.parametrize("tool", ["mic_inject", "mic_speak"])
@pytest.mark.parametrize("target_field", ["id", "stable_key"])
def test_mcp_published_handle_reaches_the_control(
    controlled_engine: tuple[Engine, list[str]], tmp_path: Path, tool: str, target_field: str
) -> None:
    engine, events = controlled_engine
    server = build_server(engine)

    async def run() -> None:
        async with create_connected_server_and_client_session(server) as client:
            observed = await client.call_tool("analyze_screen", {"source": "hierarchy"})
            handle = json.loads(observed.content[0].text)["elements"][0]["id"]
            assert handle.startswith("el:")
            args = (
                {"path": str(tmp_path / "voice.wav")}
                if tool == "mic_inject"
                else {"speech": "Example microphone check"}
            )
            args.update(
                {
                    target_field: handle,
                    "bounds": [20, 100, 300, 220],
                    "pre_roll_ms": 0,
                    "post_roll_ms": 0,
                }
            )
            result = await client.call_tool(tool + "_and_analyze", args)
            payload = json.loads(result.content[0].text)
            assert payload["ok"] is True, payload
            assert payload["id"] == handle
            assert payload["target"] == [160, 160]

    try:
        anyio.run(run)
    finally:
        engine.close()
    assert events == ["down", "audio", "up"]


@pytest.mark.parametrize("command", ["inject", "speak"])
def test_cli_published_handle_reaches_the_same_control(
    controlled_engine: tuple[Engine, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    engine, events = controlled_engine
    observed = engine.analyze(source="hierarchy", with_ocr=False)
    handle = observed.elements[0].published_id
    monkeypatch.setattr(cli, "_run", lambda _ctx, go: go(engine, cli.OutputFormat.json))
    value = str(tmp_path / "voice.wav") if command == "inject" else "Example microphone check"
    try:
        result = CliRunner().invoke(
            cli.app,
            [
                "mic",
                command,
                value,
                handle,
                "--pre-roll-ms",
                "0",
                "--post-roll-ms",
                "0",
            ],
        )
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["id"] == handle
        assert payload["target"] == [160, 160]
    finally:
        engine.close()
    assert events == ["down", "audio", "up"]


@pytest.mark.parametrize("command", ["inject", "speak"])
def test_cli_handle_combined_with_selector_refuses_before_audio(
    controlled_engine: tuple[Engine, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    engine, events = controlled_engine
    handle = engine.analyze(source="hierarchy", with_ocr=False).elements[0].published_id
    monkeypatch.setattr(cli, "_run", lambda _ctx, go: go(engine, cli.OutputFormat.json))
    value = str(tmp_path / "voice.wav") if command == "inject" else "Example microphone check"
    try:
        result = CliRunner().invoke(
            cli.app,
            [
                "mic",
                command,
                value,
                handle,
                "--rid",
                "hold_to_talk",
            ],
        )
        assert isinstance(result.exception, UsageError)
        assert "only one" in str(result.exception)
    finally:
        engine.close()
    assert events == []


@pytest.mark.parametrize(
    "target",
    [
        {"id": "el:example", "stable_key": "el:example"},
        {"stable_key": "el:example", "rid": "hold_to_talk"},
        {"stable_key": "el:example", "index": 0},
        {"id": "el:example", "first": True},
        {"bounds": [20, 100, 300, 220]},
        {"rid": "hold_to_talk", "bounds": [20, 100, 300, 220]},
        {"id": 1, "bounds": [20, 100, 300, 220]},
        {"stable_key": "el:example", "bounds": [20, 100, 300]},
        {"stable_key": "el:example", "bounds": [20, 100, 300, "220"]},
        {"id": " "},
        {"stable_key": " "},
    ],
)
@pytest.mark.parametrize("tool", ["mic_inject", "mic_speak"])
def test_invalid_target_cannot_silently_become_audio_only(
    controlled_engine: tuple[Engine, list[str]],
    target: dict[str, Any],
    tool: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, events = controlled_engine

    def forbidden(*_a: Any, **_kw: Any) -> None:
        pytest.fail("invalid target must fail before audio preparation or speech synthesis")

    monkeypatch.setattr(mic, "prepare_injection", forbidden)
    monkeypatch.setattr(mic, "synthesize_speech", forbidden)
    try:
        with pytest.raises(UsageError):
            _dispatch(engine, tool, {"path": "unused.wav", "speech": "Example", **target})
    finally:
        engine.close()
    assert events == []


def test_optional_audio_only_and_selector_bounds_are_explicit() -> None:
    assert _optional_mic_target({}) == (None, None)
    assert _optional_mic_target({"stable_key": "el:example", "bounds": [20, 100, 300, 220]}) == (
        None,
        {"key": "el:example", "bounds": [20, 100, 300, 220]},
    )
