"""CLI tests (PRD §13.1 AC1, AC9, AC11 + exit codes + schema validity).

The CLI is a thin Typer adapter over the engine. We drive it with Typer's
``CliRunner`` and inject a device-less :class:`FakeDevice` by monkeypatching
``android_ui_analyser.engine.connect`` (and ``list_devices`` where needed), so no phone
is required. Logs go to stderr; JSON results go to stdout — we assert both streams.
"""

from __future__ import annotations

import json
import subprocess
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
    monkeypatch.setenv("AUA_LEASE__REGISTRY_DIR", str(cache))
    # Daemon off so commands always run in-process during tests.
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    return cache


# --------------------------------------------------------------------------- AC1


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
    assert "analyze" in result.stdout
    assert "tap-and-analyze" in result.stdout


def test_session_autopilot_help_exposes_bounded_local_loop() -> None:
    result = runner.invoke(app, ["session", "autopilot", "--help"])

    assert result.exit_code == 0
    assert "bounded safe navigation stretch" in result.stdout
    assert "--max-steps" in result.stdout
    assert "--max-duration-ms" in result.stdout


def test_explicit_action_name_cannot_disable_its_analysis(patched_device: FakeDevice) -> None:
    result = runner.invoke(
        app,
        ["--no-cache", "tap-and-analyze", "--rid", "continue_btn", "--no-observe"],
    )

    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["action"] == "tap"
    assert data["observation_present"] is True
    assert data["observation"]["elements"]


def test_version_prints_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


# --------------------------------------------------------------------------- analyze schema


def test_analyze_prints_schema_valid_json(patched_device: FakeDevice) -> None:
    # Force the hierarchy path. On macOS the default parallel Apple OCR augmenter makes
    # aggregate provenance mixed; hosts without Apple Vision remain hierarchy-only.
    result = runner.invoke(app, ["--no-cache", "analyze", "--source", "hierarchy"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert set(data) == {"schema_version", "screen", "elements", "meta"}
    assert data["schema_version"] == 1
    assert data["screen"]["source"] in {"hierarchy", "mixed"}
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


def test_ask_prints_screen_analysis(patched_device: FakeDevice, monkeypatch) -> None:
    monkeypatch.setattr(
        engine_mod.Engine,
        "ask_screen",
        lambda self, question: {
            "question": question,
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "duration_ms": 123,
            "usage": {"total_tokens": 42},
            "analysis": {"answer": "The header is at the top."},
        },
    )
    result = runner.invoke(app, ["ask", "Where is the header?"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["question"] == "Where is the header?"
    assert data["analysis"]["answer"] == "The header is at the top."


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
    result = runner.invoke(app, ["tap-and-analyze"])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- actions


def test_tap_records_click_and_emits_action(patched_device: FakeDevice) -> None:
    # Seed the analyze cache so the element id resolves.
    seed = runner.invoke(app, ["analyze", "--source", "hierarchy"])
    assert seed.exit_code == 0, seed.stderr
    result = runner.invoke(app, ["tap-and-analyze", "1"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["ok"] is True
    assert data["action"] == "tap"
    assert data["id"] == 1
    assert any(call[0] == "click" for call in patched_device.calls)


def test_tap_accepts_explicit_id_flag_for_cli_mcp_symmetry(
    patched_device: FakeDevice,
) -> None:
    seed = runner.invoke(app, ["analyze", "--source", "hierarchy"])
    assert seed.exit_code == 0, seed.stderr

    result = runner.invoke(app, ["tap-and-analyze", "--id", "1"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["id"] == 1
    assert any(call[0] == "click" for call in patched_device.calls)


def test_click_is_alias_of_tap(patched_device: FakeDevice) -> None:
    runner.invoke(app, ["analyze", "--source", "hierarchy"])
    result = runner.invoke(app, ["click-and-analyze", "1"])
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["action"] == "tap"


def test_key_press(patched_device: FakeDevice) -> None:
    result = runner.invoke(app, ["key-and-analyze", "back"])
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["action"] == "key"
    assert data["detail"] == "back" or str(data["detail"]).startswith("back")
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


def test_lease_acquire_positional_serial_overrides_ambient_pin(
    tmp_path: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AUA_SERIAL", "emulator-5554")
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="phone-123", model="Phone", android_version="14"),
            DeviceInfo(serial="emulator-5554", model="Emulator", android_version="14"),
        ],
    )

    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "--owner",
            "agent-a",
            "lease",
            "acquire",
            "phone-123",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["serial"] == "phone-123"
    assert leases.holder(isolated_cache, "phone-123") == "agent-a"
    assert leases.holder(isolated_cache, "emulator-5554") is None


def test_unpinned_lease_acquire_prefers_emulator_when_phone_is_listed_first(
    tmp_path: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="phone-123", model="Phone", android_version="14"),
            DeviceInfo(serial="emulator-5554", model="Emulator", android_version="14"),
        ],
    )

    result = runner.invoke(
        app,
        ["--config", str(config), "--owner", "agent-a", "lease", "acquire"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["serial"] == "emulator-5554"
    assert leases.holder(isolated_cache, "emulator-5554") == "agent-a"
    assert leases.holder(isolated_cache, "phone-123") is None


def test_lease_acquire_without_positional_serial_preserves_aua_serial(
    tmp_path: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AUA_SERIAL", "phone-123")
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="phone-123", model="Phone", android_version="14"),
            DeviceInfo(serial="emulator-5554", model="Emulator", android_version="14"),
        ],
    )

    result = runner.invoke(
        app,
        ["--config", str(config), "--owner", "agent-a", "lease", "acquire"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["serial"] == "phone-123"
    assert leases.holder(isolated_cache, "phone-123") == "agent-a"
    assert leases.holder(isolated_cache, "emulator-5554") is None


def test_lease_switch_requires_warning_then_cleans_and_replaces(
    tmp_path: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="emulator-5554", model="First", android_version="14"),
            DeviceInfo(serial="emulator-5556", model="Second", android_version="14"),
        ],
    )
    reset_calls: list[tuple[str, bool]] = []

    def teardown(_engine: object, *, serial: str, force: bool) -> dict[str, object]:
        reset_calls.append((serial, force))
        return {"ok": True, "reports": []}

    monkeypatch.setattr(engine_mod.Engine, "teardown_run", teardown)
    prefix = ["--config", str(config), "--owner", "agent-a", "lease", "acquire"]
    first = runner.invoke(app, [*prefix, "emulator-5554"])
    assert first.exit_code == 0, first.stdout + first.stderr

    warned = runner.invoke(app, [*prefix, "emulator-5556"])
    assert warned.exit_code == 2
    assert "lease_switch_required" in warned.stderr
    assert "--replace" in warned.stderr
    assert leases.holder(isolated_cache, "emulator-5554") == "agent-a"
    assert leases.holder(isolated_cache, "emulator-5556") is None
    assert reset_calls == []

    replaced = runner.invoke(app, [*prefix, "emulator-5556", "--replace"])
    assert replaced.exit_code == 0, replaced.stdout + replaced.stderr
    payload = json.loads(replaced.stdout)
    assert payload["serial"] == "emulator-5556"
    assert payload["replaced"] == ["emulator-5554"]
    assert reset_calls == [("emulator-5554", True)]
    assert leases.holder(isolated_cache, "emulator-5554") is None
    assert leases.holder(isolated_cache, "emulator-5556") == "agent-a"


def test_lease_replace_keeps_old_lease_when_offline_cleanup_is_deferred(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    owner = leases.resolve_owner("agent-a")
    assert leases.acquire(isolated_cache, "emulator-5554", owner=owner)
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="emulator-5554", model="First", android_version="14"),
            DeviceInfo(serial="emulator-5556", model="Second", android_version="14"),
        ],
    )
    monkeypatch.setattr(
        engine_mod.Engine,
        "teardown_run",
        lambda *_args, **_kwargs: {
            "ok": True,
            "reports": [
                {
                    "serial": "emulator-5554",
                    "skipped": "target unreachable; device-side undos deferred",
                    "undone": [],
                    "failed": [],
                }
            ],
        },
    )

    replaced = runner.invoke(
        app,
        ["--owner", "agent-a", "lease", "acquire", "emulator-5556", "--replace"],
    )

    assert replaced.exit_code == 3
    assert "could not be cleaned" in replaced.stderr
    assert leases.holder(isolated_cache, "emulator-5554") == "agent-a"
    assert leases.holder(isolated_cache, "emulator-5556") is None


def test_lease_replace_recovers_multiple_legacy_primary_leases(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    owner = leases.resolve_owner("agent-a")
    assert leases.acquire(isolated_cache, "emulator-5554", owner=owner)
    assert leases._acquire_unlocked(  # noqa: SLF001 - construct a pre-one-lease registry
        isolated_cache,
        "emulator-5556",
        owner=owner,
        role="primary",
    )
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="emulator-5554", model="First", android_version="14"),
            DeviceInfo(serial="emulator-5556", model="Second", android_version="14"),
        ],
    )
    monkeypatch.setattr(
        engine_mod.Engine,
        "teardown_run",
        lambda *_args, **_kwargs: {"ok": True, "reports": []},
    )

    recovered = runner.invoke(
        app,
        ["--owner", "agent-a", "lease", "acquire", "emulator-5554", "--replace"],
    )

    assert recovered.exit_code == 0, recovered.stdout + recovered.stderr
    assert leases.primary_held_by(isolated_cache, owner) == ["emulator-5554"]
    assert leases.holder(isolated_cache, "emulator-5556") is None


def test_lease_transfer_accept_replay_and_cancel_cli_contract(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    starts = {111: "source", 222: "child"}
    current = [leases.LeaseOwner("source-process", pid=111, started="source")]
    monkeypatch.setattr(leases, "_derived_owner", lambda: current[0])
    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    source = leases.resolve_owner("orchestrator")
    serial = "emulator-5554"
    assert leases.acquire(isolated_cache, serial, owner=source)

    offered = runner.invoke(
        app,
        ["--owner", "orchestrator", "lease", "transfer", serial],
    )
    assert offered.exit_code == 0, offered.stdout + offered.stderr
    offer_payload = json.loads(offered.stdout)
    assert offer_payload["action"] == "lease-transfer"
    token = str(offer_payload["token"])
    assert offer_payload["accept_call"] == f"aua lease accept {token}"

    current[0] = leases.LeaseOwner("child-process", pid=222, started="child")
    accepted = runner.invoke(
        app,
        ["--owner", "child", "lease", "accept", token],
    )
    assert accepted.exit_code == 0, accepted.stdout + accepted.stderr
    accepted_entry = leases.read_lease(isolated_cache, serial)
    assert accepted_entry is not None
    assert accepted_entry["owner"] == "child"
    assert accepted_entry["owner_pid"] == 222

    replayed = runner.invoke(
        app,
        ["--owner", "child", "lease", "accept", token],
    )
    assert replayed.exit_code == 2
    assert "invalid, expired, or already used" in replayed.stderr

    second_offer = runner.invoke(
        app,
        ["--owner", "child", "lease", "transfer", serial],
    )
    assert second_offer.exit_code == 0, second_offer.stdout + second_offer.stderr
    cancelled = runner.invoke(
        app,
        ["--owner", "child", "lease", "cancel-transfer", serial],
    )
    assert cancelled.exit_code == 0, cancelled.stdout + cancelled.stderr
    assert leases.pending_handoff(leases.read_lease(isolated_cache, serial) or {}) is None


def test_lease_list_mine_requires_the_same_process_not_only_the_same_label(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases
    from android_ui_analyser.schema import DeviceInfo

    starts = {111: "first", 222: "second"}
    current = [leases.LeaseOwner("first-process", pid=111, started="first")]
    monkeypatch.setattr(leases, "_derived_owner", lambda: current[0])
    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    assert leases.acquire(
        isolated_cache,
        "emulator-5554",
        owner=leases.resolve_owner("shared-label"),
    )
    monkeypatch.setattr(
        engine_mod,
        "list_devices",
        lambda: [
            DeviceInfo(serial="emulator-5554", model="Emulator", android_version="14")
        ],
    )
    current[0] = leases.LeaseOwner("second-process", pid=222, started="second")

    listed = runner.invoke(app, ["--owner", "shared-label", "lease", "list"])

    assert listed.exit_code == 0, listed.stdout + listed.stderr
    row = json.loads(listed.stdout)["devices"][0]
    assert row["owner"] == "shared-label"
    assert row["mine"] is False


def test_lease_release_keeps_ownership_when_cleanup_fails(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    owner = leases.resolve_owner("agent-a")
    assert leases.acquire(isolated_cache, "emulator-5554", owner=owner)
    monkeypatch.setattr(
        engine_mod.Engine,
        "teardown_run",
        lambda *_args, **_kwargs: {"ok": False, "reports": [{"ok": False}]},
    )

    released = runner.invoke(
        app,
        ["--owner", "agent-a", "lease", "release", "emulator-5554"],
    )

    assert released.exit_code == 3
    assert "could not clean" in released.stderr
    assert leases.holder(isolated_cache, "emulator-5554") == "agent-a"


def test_lease_release_keeps_ownership_when_offline_cleanup_is_deferred(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    owner = leases.resolve_owner("agent-a")
    assert leases.acquire(isolated_cache, "emulator-5554", owner=owner)
    monkeypatch.setattr(
        engine_mod.Engine,
        "teardown_run",
        lambda *_args, **_kwargs: {
            "ok": True,
            "reports": [
                {
                    "serial": "emulator-5554",
                    "skipped": "target unreachable; device-side undos deferred",
                    "undone": [],
                    "failed": [],
                }
            ],
        },
    )

    released = runner.invoke(
        app,
        ["--owner", "agent-a", "lease", "release", "emulator-5554"],
    )

    assert released.exit_code == 3
    assert "could not clean" in released.stderr
    assert leases.holder(isolated_cache, "emulator-5554") == "agent-a"


def test_lease_release_reports_unlink_failure_and_keeps_ownership(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    owner = leases.resolve_owner("agent-a")
    assert leases.acquire(isolated_cache, "emulator-5554", owner=owner)
    monkeypatch.setattr(
        engine_mod.Engine,
        "teardown_run",
        lambda *_args, **_kwargs: {"ok": True, "reports": []},
    )
    lease_path = leases.lease_dir(isolated_cache) / "emulator-5554.json"
    original_unlink = Path.unlink

    def fail_lease_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == lease_path:
            raise OSError("read-only lease store")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_lease_unlink)
    released = runner.invoke(
        app,
        ["--owner", "agent-a", "lease", "release", "emulator-5554"],
    )

    assert released.exit_code == 3
    assert "could not be removed" in released.stderr
    assert leases.holder(isolated_cache, "emulator-5554") == "agent-a"


def test_fanout_uses_one_stable_scoped_owner_per_extra_target(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import leases

    owner = leases.resolve_owner("agent-a")
    assert leases.acquire(isolated_cache, "emulator-5554", owner=owner)
    calls: list[list[str]] = []
    original_run = subprocess.run

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if not command or command[0] != "aua":
            return original_run(command, **_kwargs)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    result = runner.invoke(
        app,
        [
            "--owner",
            "agent-a",
            "fanout",
            "--serials",
            "emulator-5554,emulator-5556",
            "--serial",
            "analyze",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert [row["lease_owner"] for row in payload["devices"]] == [
        "agent-a",
        "agent-a:fanout:emulator-5556",
    ]
    assert calls[0][:5] == ["aua", "--owner", "agent-a", "--serial", "emulator-5554"]
    assert calls[1][:5] == [
        "aua",
        "--owner",
        "agent-a:fanout:emulator-5556",
        "--serial",
        "emulator-5556",
    ]
