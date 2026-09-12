"""A selector miss returns its recovery screen across the CLI and daemon boundaries."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser.agent_results import from_cli
from android_ui_analyser.cli import _daemon_error, _project_error_observation, app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import SelectorNotFoundError, emit_error
from android_ui_analyser.projection import Projection
from conftest import FakeDevice, make_config, make_engine

_SCREEN = """<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Nebula inbox" bounds="[0,0][400,100]"/>
  <node class="android.widget.Button" text="Continue"
        resource-id="com.example.fiction:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


def _reads(device: FakeDevice) -> tuple[int, int]:
    return device.hierarchy_calls, device.screenshot_calls


def _published_target(observation: dict) -> str:
    target = next(
        element for element in observation["elements"] if element.get("text") == "Continue"
    )
    assert isinstance(target["id"], str) and target["id"].startswith("el:")
    return target["id"]


def _assert_normalized_failure(stdout: str, stderr: str, exit_code: int, observation: dict) -> dict:
    normalized = from_cli(stdout, stderr, exit_code, command="tap-and-analyze")
    assert normalized["ok"] is False
    assert normalized["error"]["code"] == "selector_not_found"
    assert normalized["observation"] == observation
    assert normalized["observation_contract"]["reusable"] is True
    assert normalized["observation_contract"]["analyze_needed"] is False
    assert "observation" not in normalized["error"]
    return normalized


@pytest.mark.parametrize("fields", [None, "id,text", "all"])
def test_cli_selector_miss_returns_a_published_id_usable_by_the_next_call(
    monkeypatch: pytest.MonkeyPatch, fake_cli_device, fields: str | None
) -> None:
    device = fake_cli_device(FakeDevice(hierarchy_xml=_SCREEN))
    monkeypatch.setenv("AUA_OCR__AUGMENT_HIERARCHY", "false")
    monkeypatch.setenv("AUA_PERF__PREDICTIVE_PREFETCH", "false")
    # Arrival timing is separate from the recovery contract being exercised here.
    monkeypatch.setattr(
        Engine,
        "_await_post_action_ready",
        lambda *_args, **_kwargs: {"changed": True, "timeout": False, "via": "hierarchy", "ms": 1},
    )
    runner = CliRunner()

    options = ["--observe-fields", fields] if fields else []
    missed = runner.invoke(app, [*options, "tap-and-analyze", "--rid", "missing_control"])

    assert missed.exit_code == 6, missed.stdout + missed.stderr
    assert missed.stdout == ""
    error = json.loads(missed.stderr)["error"]
    assert error["code"] == "selector_not_found"
    assert error["observation_present"] is True
    observation = error["observation"]
    target_id = _published_target(observation)
    if fields == "id,text":
        assert all(set(element) <= {"id", "text"} for element in observation["elements"])
    elif fields == "all":
        assert observation["elements"][0].get("type")
    assert all(name == "find_text" for name, _args in device.calls), (
        "a missing selector may only issue a read-only native lookup"
    )
    assert device.hierarchy_calls == 1, "the returned screen is the existing resolution read"
    reads_after_miss = _reads(device)

    _assert_normalized_failure(missed.stdout, missed.stderr, missed.exit_code, observation)

    assert _reads(device) == reads_after_miss, "normalization must not acquire another screen"
    recovered = runner.invoke(app, ["tap-and-analyze", target_id])
    assert recovered.exit_code == 0, recovered.stdout + recovered.stderr
    assert json.loads(recovered.stdout)["ok"] is True
    assert [args for name, args in device.calls if name == "click"] == [(540, 260)]


def test_daemon_selector_miss_survives_cli_reconstruction_without_another_read(
    tmp_path: Path,
) -> None:
    device = FakeDevice(hierarchy_xml=_SCREEN)
    engine = make_engine(
        device=device,
        config=make_config(
            cache={"dir": str(tmp_path / "cache")},
            perf={"prefetch": False, "predictive_prefetch": False},
            output={"observation_fields": "id,text", "observation_meta": "changed"},
        ),
    )

    response = dispatch(engine, {"cmd": "tap", "args": {"selector": {"rid": "missing_control"}}})

    assert response["ok"] is False
    assert response["error"]["code"] == "selector_not_found"
    observation = response["error"]["observation"]
    target_id = _published_target(observation)
    assert any("resource_id" in element for element in observation["elements"])
    assert all(name == "find_text" for name, _args in device.calls), (
        "a daemon miss may only issue a read-only native lookup"
    )
    assert device.hierarchy_calls == 1
    reads_after_miss = _reads(device)

    # The daemon sends JSON; the CLI rebuilds the typed exception and writes it to stderr.
    reconstructed = _daemon_error(json.loads(json.dumps(response))["error"])
    assert isinstance(reconstructed, SelectorNotFoundError)
    _project_error_observation(reconstructed, Projection.for_observation("id,text", meta="changed"))
    observation = reconstructed.observation
    assert all(set(element) <= {"id", "text"} for element in observation["elements"])
    stderr = io.StringIO()
    exit_code = emit_error(reconstructed, stream=stderr)
    assert exit_code == 6
    _assert_normalized_failure("", stderr.getvalue(), exit_code, observation)
    assert _reads(device) == reads_after_miss, "serialization and projection reuse the same read"

    recovered = dispatch(
        engine, {"cmd": "tap", "args": {"element_id": target_id, "observe": False}}
    )
    assert recovered["ok"] is True, recovered
    assert recovered["result"]["ok"] is True
    assert [args for name, args in device.calls if name == "click"] == [(540, 260)]


def test_cli_error_projection_preserves_actionable_rows_and_capture_hint() -> None:
    observation = {
        "screen": {"width": 1080, "height": 2400},
        "elements": [
            {
                "id": "el:named",
                "content_desc": "Close",
                "resource_id": "example:id/close_btn",
                "type": "Button",
                "clickable": True,
            },
            {"id": "el:unnamed", "type": "Button", "clickable": True},
        ],
        "meta": {"capture_hint": {"frames": 2}, "fingerprint": "screen-a"},
    }
    error = SelectorNotFoundError("missing", observation=observation)
    view = Projection.for_observation("id,desc,rid", meta="changed")

    _project_error_observation(error, view)

    assert error.observation["elements"][0] == {
        "id": "el:named",
        "desc": "Close",
        "rid": "close_btn",
    }
    assert error.observation["elements"][1]["id"] == "el:unnamed"
    assert error.observation["meta"]["capture_hint"] == {"frames": 2}

    _project_error_observation(error, Projection.parse(fields="id", no_meta=True))

    assert "meta" not in error.observation
    assert [element["id"] for element in error.observation["elements"]] == [
        "el:named",
        "el:unnamed",
    ]
