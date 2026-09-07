"""Waits use the same existing-frame image output contract as ordinary actions."""

from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import cli
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.schema import ActionResult, OutputFormat
from conftest import FakeDevice, make_config

XML = '<hierarchy><node class="android.widget.Button" text="Ready" resource-id="ready" clickable="true" enabled="true" bounds="[0,0][100,100]"/></hierarchy>'


def engine():
    return Engine(
        make_config(daemon={"enabled": False}, lease={"enabled": False}),
        device=FakeDevice(hierarchy_xml=XML, text_index={"Ready": (0, 0, 100, 100)}),
    )


@pytest.mark.parametrize(
    "argv", [["await-and-analyze", "text:Ready"], ["wait-and-analyze", "--for", "Ready"]]
)
@pytest.mark.parametrize("explicit_path", [False, True])
def test_wait_cli_optional_image_value_is_parsed_and_saved(
    monkeypatch, tmp_path, argv, explicit_path
):
    runtime = engine()
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: runtime)
    out = tmp_path / "wait.png"
    result = CliRunner().invoke(
        cli.app, [*argv, "--with-image", *([str(out)] if explicit_path else [])]
    )
    assert result.exit_code == 0, result.output
    import json

    payload = json.loads(result.stdout)
    image = payload["observation"]["meta"]["raw_image"]
    assert Path(image).is_file()
    if explicit_path:
        assert image == str(out)
    assert runtime._default_with_image is True


def test_engine_wait_restores_output_default_on_failure(monkeypatch):
    runtime = engine()
    monkeypatch.setattr(
        runtime,
        "_observe",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("observation failed")),
    )
    with pytest.raises(RuntimeError, match="observation failed"):
        runtime.await_predicate("text:Ready", observe=True, with_image=False)
    assert runtime._default_with_image is True


@pytest.mark.parametrize(
    "name,args",
    [("await_and_analyze", {"predicate": "text:Ready"}), ("wait_and_analyze", {"for_": "Ready"})],
)
def test_mcp_wait_accepts_path_and_returns_image(tmp_path, name, args):
    out = tmp_path / "mcp-wait.png"
    server = build_server(engine())

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(name, {**args, "with_image": str(out)})

    result = anyio.run(run)
    assert not result.isError, result
    assert out.is_file()
    assert any(block.type == "image" for block in result.content)


@pytest.mark.parametrize("daemon_dict", [False, True])
@pytest.mark.parametrize("explicit_path", [False, True])
def test_action_until_keeps_explicit_image_destination_for_the_final_frame(
    monkeypatch, tmp_path, capsys, daemon_dict, explicit_path
):
    import json

    runtime = engine()
    initial = runtime.analyze(source="hierarchy", with_ocr=False)
    old_path = tmp_path / "early-auto-frame.png"
    old_path.write_bytes(b"early observation")
    initial.meta.raw_image = str(old_path)
    requested = tmp_path / "requested.png"
    requested.write_bytes(b"early observation")
    action = ActionResult(
        ok=True, action="tap", observation_present=True, observation=initial
    )
    context = cli._CliJournalContext(
        cache_dir=runtime.config.cache.dir,
        serial=runtime.device.serial,
        platform="android",
        invocation_id="image-until-fixture",
        detail_id=None,
        cmd="tap",
        args={"with_image": str(requested)} if explicit_path else {},
        client={},
    )
    calls = []

    def route(_engine, method, **kwargs):
        assert method == "await_predicate"
        kwargs.pop("_journal_privacy_cmd", None)
        calls.append(dict(kwargs))
        result = runtime.await_predicate(**kwargs)
        return result.model_dump(mode="json") if daemon_dict else result

    monkeypatch.setattr(cli, "_route", route)
    monkeypatch.setattr(cli, "_UNTIL", ("text:Ready", 1000, 10))
    monkeypatch.setattr(cli, "_ENGINE", runtime)
    monkeypatch.setattr(cli, "_OBSERVATION_VIEW", None)
    monkeypatch.setattr(cli, "_ANNOTATION_WARNINGS", [])
    cli._emit(
        action.model_dump(mode="json") if daemon_dict else action,
        OutputFormat.json,
        _journal_context=context,
    )
    emitted = json.loads(capsys.readouterr().out)
    final_path = Path(emitted["observation"]["meta"]["raw_image"])
    assert emitted["await_outcome"] == "satisfied"
    assert final_path.read_bytes() == runtime.device._png
    assert old_path.read_bytes() == b"early observation", "never infer intent from an auto filename"
    if explicit_path:
        assert calls[0]["with_image"] == str(requested)
        assert final_path == requested
    else:
        assert "with_image" not in calls[0]
        assert final_path not in {requested, old_path}
