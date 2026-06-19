"""CLI tests (PRD §13.1 AC1, AC9, AC11 + exit codes + schema validity).

The CLI is a thin Typer adapter over the engine. We drive it with Typer's
``CliRunner`` and inject a device-less :class:`FakeDevice` by monkeypatching
``android_ui_analyser.engine.connect`` (and ``list_devices`` where needed), so no phone
is required. Logs go to stderr; JSON results go to stdout — we assert both streams.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser import __version__
from android_ui_analyser.cli import app
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.providers.base import ChainSpec, TextBox
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import AnalyzeResult
from conftest import FakeDevice, StubOcr

runner = CliRunner()


# A small, well-labeled hierarchy that yields a stable hierarchy-sourced analyze.
HIERARCHY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Welcome" bounds="[0,0][1080,120]"/>
  <node index="1" class="android.widget.Button" text="Continue"
        resource-id="com.test.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
  <node index="2" class="android.widget.EditText" content-desc="Email field"
        resource-id="com.test.app:id/email" clickable="true" enabled="true"
        bounds="[40,400][1040,500]"/>
</hierarchy>"""


@pytest.fixture
def patched_device(monkeypatch: pytest.MonkeyPatch) -> FakeDevice:
    """Patch engine.connect to return a FakeDevice with the labeled hierarchy."""
    device = FakeDevice(
        hierarchy_xml=HIERARCHY_XML,
        text_index={"Continue": (40, 200, 1040, 320), "Welcome": (0, 0, 1080, 120)},
    )
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    return device


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep analyze-cache / annotated-image writes out of the real ~/.cache."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUA_CACHE__DIR", str(cache))
    # Daemon off so commands always run in-process during tests.
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    return cache


# --------------------------------------------------------------------------- AC1


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
    assert "analyze" in result.stdout


def test_version_prints_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


# --------------------------------------------------------------------------- analyze schema


def test_analyze_prints_schema_valid_json(patched_device: FakeDevice) -> None:
    # Force the hierarchy path for a deterministic, provider-independent result.
    result = runner.invoke(app, ["--no-cache", "analyze", "--source", "hierarchy"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert set(data) == {"schema_version", "screen", "elements", "meta"}
    assert data["schema_version"] == 1
    assert data["screen"]["source"] == "hierarchy"
    assert {"width", "height", "source"} <= set(data["screen"])
    assert len(data["elements"]) == 3
    first = data["elements"][0]
    assert {"id", "type", "bounds", "center"} <= set(first)
    # Round-trips through the pydantic model (strict schema).
    AnalyzeResult.model_validate(data)


def test_analyze_compact_is_single_line(patched_device: FakeDevice) -> None:
    result = runner.invoke(
        app, ["--no-cache", "--format", "compact", "analyze", "--source", "hierarchy"]
    )
    assert result.exit_code == 0, result.stderr
    body = result.stdout.strip()
    assert "\n" not in body
    data = json.loads(body)
    AnalyzeResult.model_validate(data)


# --------------------------------------------------------------------------- AC11 has


def test_has_found_via_hierarchy(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["has", "Continue"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["found"] is True
    assert data["source"] == "hierarchy"
    assert data["bounds"] == [40, 200, 1040, 320]


def test_has_not_found_exits_one(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["has", "Nope"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["found"] is False


@pytest.mark.parametrize(
    "args,exit_code",
    [
        (["has", "continue", "--ignore-case"], 0),
        (["has", "continue"], 1),  # case-sensitive miss
        (["has", "Cont", "--match", "contains"], 0),
        (["has", "^Continue$", "--match", "regex"], 0),
        (["has", "Continue", "--match", "exact"], 0),
        (["has", "Contin", "--match", "exact"], 1),  # exact miss
    ],
)
def test_has_match_modes(patched_device: FakeDevice, args: list[str], exit_code: int) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == exit_code, result.stdout + result.stderr


def test_has_ocr_fallback_found_via_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hierarchy misses; the OCR chain finds the text → found via `ocr` (AC11)."""

    class StubFactory(ProviderFactory):
        def is_enabled(self, kind: str) -> bool:
            return kind == "ocr"

        def build_chain(self, kind: str) -> ChainSpec:
            if kind == "ocr":
                provider = StubOcr(result=[TextBox(text="Checkout", bounds=(10, 20, 110, 60))])
                return ChainSpec(kind="ocr", providers=[provider])
            return ChainSpec(kind=kind, providers=[])

    # Empty hierarchy text_index → hierarchy miss; engine falls back to OCR.
    device = FakeDevice(text_index={})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    real_engine = engine_mod.Engine

    def engine_with_stub_factory(cfg, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("factory", StubFactory(cfg))
        return real_engine(cfg, **kwargs)

    monkeypatch.setattr("android_ui_analyser.cli.Engine", engine_with_stub_factory)

    result = runner.invoke(app, ["has", "Checkout"])  # ocr-fallback default on
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["found"] is True
    assert data["source"] == "ocr"


def test_has_no_ocr_fallback_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """With --no-ocr-fallback, a hierarchy miss is final (exit 1), OCR never consulted."""
    device = FakeDevice(text_index={})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)
    result = runner.invoke(app, ["has", "Checkout", "--no-ocr-fallback"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["found"] is False


# --------------------------------------------------------------------------- exit codes


def test_config_error_exit_5(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    result = runner.invoke(app, ["--config", str(missing), "analyze"])
    assert result.exit_code == 5
    err = json.loads(result.stderr)
    assert err["error"]["code"] == "config"


def test_usage_error_bad_format_exit_2() -> None:
    result = runner.invoke(app, ["--format", "banana", "analyze"])
    assert result.exit_code == 2
    err = json.loads(result.stderr)
    assert err["error"]["code"] == "usage"


def test_usage_error_missing_argument_exit_2() -> None:
    # `tap` requires an element id; Typer/Click reports a usage error (exit 2).
    result = runner.invoke(app, ["tap"])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- actions


def test_tap_records_click_and_emits_action(patched_device: FakeDevice) -> None:
    # Seed the analyze cache so the element id resolves.
    seed = runner.invoke(app, ["analyze", "--source", "hierarchy"])
    assert seed.exit_code == 0, seed.stderr
    result = runner.invoke(app, ["tap", "1"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert data["action"] == "tap"
    assert data["id"] == 1
    assert any(call[0] == "click" for call in patched_device.calls)


def test_click_is_alias_of_tap(patched_device: FakeDevice) -> None:
    runner.invoke(app, ["analyze", "--source", "hierarchy"])
    result = runner.invoke(app, ["click", "1"])
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["action"] == "tap"


def test_key_press(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["key", "back"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["action"] == "key"
    assert data["detail"] == "back"
    assert ("press", ("back",)) in patched_device.calls


# --------------------------------------------------------------------------- config commands


def test_config_show_masks_and_is_yaml(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.stderr
    # YAML by default; contains a known config key.
    assert "routing:" in result.stdout
    assert "max_tier" in result.stdout


def test_config_show_json(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["config", "show", "--json"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert "routing" in data
    assert data["routing"]["max_tier"] == "vision"


def test_config_init_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0, result.stderr
    written = Path(result.stdout.strip())
    assert written.is_file()
    assert "android-ui-analyser configuration" in written.read_text()
    # Second run without --force does not overwrite.
    again = runner.invoke(app, ["config", "init"])
    assert again.exit_code == 0
    assert "already exists" in again.stdout


def test_config_path_prefers_explicit(tmp_path: Path) -> None:
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("device: {serial: null}\n")
    result = runner.invoke(app, ["--config", str(cfg), "config", "path"])
    assert result.exit_code == 0
    assert str(cfg) in result.stdout


# --------------------------------------------------------------------------- AC9 doctor


def test_doctor_no_device_no_secret_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """doctor with no device: reports provider availability + reasons, exits 0,
    and never prints the secret value (AC9)."""
    secret = "dummy-secret-value"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    def boom() -> list:
        raise DeviceError("no device found")

    monkeypatch.setattr(engine_mod, "list_devices", boom)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stderr
    combined = result.stdout + result.stderr
    assert secret not in combined
    # Reports each provider kind with availability + reason.
    assert "ocr" in result.stdout
    assert "detection" in result.stdout
    assert "grounding" in result.stdout


def test_doctor_json_no_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "another-secret")
    monkeypatch.setattr(engine_mod, "list_devices", lambda: [])
    result = runner.invoke(app, ["--format", "json", "doctor"])
    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert "checks" in report
    assert "providers" in report
    assert {"ocr", "detection", "grounding"} <= set(report["providers"])
    assert "another-secret" not in result.stdout


# --------------------------------------------------------------------------- devices


def test_devices_lists_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser.schema import DeviceInfo

    infos = [DeviceInfo(serial="emulator-5554", model="Pixel", android_version="14")]
    monkeypatch.setattr(engine_mod, "list_devices", lambda: infos)
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert data[0]["serial"] == "emulator-5554"
