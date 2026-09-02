"""Flags: configured templates, read-back verification, and the post-write restart.

Three things a caller cannot see from ``ok: true``, so they are asserted here: the deeplink
template is user config (no built-ins), the keys are read back out of the app's own prefs
(a dropped key exits non-zero), and the app is restarted so a flag that is only read at
cold start is actually re-read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser import flags as flags_mod
from android_ui_analyser.cli import app
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flags import (
    build_uri,
    load_flags_file,
    parse_all_prefs,
    parse_assignments,
    parse_prefs,
    read_context_flags,
)
from android_ui_analyser.memory import AppMemoryStore, context_id_for_flags
from conftest import FakeDevice, make_config

runner = CliRunner()

PKG = "com.example.app"
TEMPLATES = {PKG: "myapp://set-flags?{query}"}
PREFS_FILE = "flag_overrides.xml"


def make_flags_engine(tmp_path: Path, device: Device, **flags_cfg: Any) -> Engine:
    cfg = make_config(cache={"dir": str(tmp_path)}, flags={"templates": TEMPLATES, **flags_cfg})
    return Engine(cfg, device=device)


def device_with(prefs: dict[str, str], **kw: Any) -> FakeDevice:
    return FakeDevice(package=PKG, activity=".MainActivity", prefs={PREFS_FILE: prefs}, **kw)


def call_names(device: FakeDevice) -> list[str]:
    return [c[0] for c in device.calls]


def shell_commands(device: FakeDevice) -> list[str]:
    return [c[1][0] for c in device.calls if c[0] == "shell"]


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake writes prefs and starts activities instantly; no poll windows needed."""
    # These constants are read by flags_set and its helpers from their own module's globals.
    flags_home = sys.modules[Engine.flags_set.__module__]
    monkeypatch.setattr(flags_home, "_FLAGS_VERIFY_DEADLINE_S", 0.0)
    monkeypatch.setattr(flags_home, "_FLAGS_ENTRY_TIMEOUT_S", 0.0)
    monkeypatch.setattr(flags_home, "_FLAGS_FOREGROUND_TIMEOUT_S", 0.0)


class AppLifecycleDevice(FakeDevice):
    """A fake that models the process: force-stop kills it, a working launch brings it back.

    ``exported=False`` reproduces what a real ``am start -n <non-exported activity>`` does:
    prints a SecurityException, exits 0, and starts nothing.
    """

    def __init__(
        self,
        *,
        exported: bool = True,
        launchable: bool = True,
        default_entry: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.exported = exported
        self.launchable = launchable
        self.default_entry = default_entry
        self.running = True

    def stop_app(self, package: str) -> None:
        super().stop_app(package)
        self.running = False

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        super().launch_app(package, activity=activity)
        if activity is not None:
            if self.exported:
                self.running = True
            return
        if self.launchable:
            self.running = True
            self._act = self.default_entry or self._act

    def current_app(self) -> dict[str, str]:
        if self.running:
            return super().current_app()
        return {"package": "com.android.launcher", "activity": ".Launcher"}


def lifecycle_device(**kw: Any) -> AppLifecycleDevice:
    return AppLifecycleDevice(
        package=PKG, activity=".MainActivity", prefs={PREFS_FILE: {"hub": "a"}}, **kw
    )


# ------------------------------------------------------------------ templates come from config


def test_build_uri_uses_the_configured_template() -> None:
    uri = build_uri(PKG, {"some_experiment": "treatment_a", "x": "1"}, TEMPLATES)
    assert uri.startswith("myapp://set-flags?")
    assert "some_experiment=treatment_a" in uri
    assert "x=1" in uri


def test_build_uri_without_a_configured_template_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="no flags deeplink") as exc:
        build_uri(PKG, {"k": "v"})
    assert PKG in str(exc.value)
    assert "flags" in (exc.value.hint or "")


def test_no_templates_ship_with_the_tool() -> None:
    """A set-flags scheme is an app's private contract; a built-in default would be a guess."""
    assert not hasattr(flags_mod, "DEFAULT_TEMPLATES")


def test_parse_and_load(tmp_path: Path) -> None:
    assert parse_assignments(["a=1", "b=two"]) == {"a": "1", "b": "two"}
    path = tmp_path / "flags.yaml"
    path.write_text(f"app: {PKG}\nflags:\n  hub: a\n  other: b\n", encoding="utf-8")
    app_id, flags = load_flags_file(path)
    assert app_id == PKG
    assert flags == {"hub": "a", "other": "b"}


# ------------------------------------------------------------------------------- read-back


def test_parse_prefs_reads_both_entry_shapes() -> None:
    xml = (
        '<map>\n    <string name="hub">a</string>\n'
        '    <boolean name="on" value="true" />\n'
        '    <string name="other">b</string>\n</map>'
    )
    assert parse_prefs(xml, {"hub", "on"}) == {"hub": ["a"], "on": ["true"]}


def test_context_read_filters_private_preferences() -> None:
    device = device_with(
        {
            "catalog_layout_experiment": "a",
            "services_treatment": "b",
            "aua_probe_flag": "true",
            "auth_token": "must-not-leak",
        }
    )

    result = read_context_flags(
        device,
        PKG,
        prefs_file=PREFS_FILE,
        key_patterns=[r"(?i)(experiment|treatment|flag)"],
    )

    assert result.verified
    assert result.flags == {
        "catalog_layout_experiment": "a",
        "aua_probe_flag": "true",
        "services_treatment": "b",
    }
    assert "auth_token" not in result.flags
    assert parse_all_prefs(device.prefs_xml(PREFS_FILE))["auth_token"] == ["must-not-leak"]


def test_analyze_discovers_and_switches_live_flag_context(tmp_path: Path) -> None:
    hierarchy_xml = (
        '<hierarchy rotation="0"><node class="android.widget.TextView" '
        f'package="{PKG}" text="Catalog" resource-id="x:id/catalogGridContent" '
        'clickable="false" enabled="true" bounds="[0,200][400,300]"/></hierarchy>'
    )
    device = device_with(
        {"catalog_layout_experiment": "a", "auth_token": "must-not-leak"},
        hierarchy_xml=hierarchy_xml,
        serial="flag-context",
    )
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"dir": str(tmp_path / "memory")},
        flags={
            "prefs_files": {PKG: PREFS_FILE},
            "context_refresh_s": 0,
        },
        daemon={"enabled": False},
        perf={"async_memory": False},
    )
    engine = Engine(cfg, device=device)

    engine.analyze(source="hierarchy", no_cache=True)
    first = AppMemoryStore(cfg.memory).load_session(device.serial)
    assert first.active_context_id == context_id_for_flags({"catalog_layout_experiment": "a"})
    assert first.context_verified is True

    device.prefs[PREFS_FILE]["catalog_layout_experiment"] = "b"
    engine.analyze(source="hierarchy", no_cache=True)
    store = AppMemoryStore(cfg.memory)
    second = store.load_session(device.serial)
    app_map = store.load(PKG)

    assert second.active_context_id == context_id_for_flags({"catalog_layout_experiment": "b"})
    assert len({record.context_id for record in app_map.screens.values()}) == 2
    assert all("auth_token" not in context.flags for context in app_map.contexts.values())
    assert all(
        context.evidence == [f"shared_prefs:{PREFS_FILE}"]
        for context in app_map.contexts.values()
    )


def test_flags_set_reports_the_keys_the_app_dropped(tmp_path: Path) -> None:
    device = device_with({"hub": "a", "routines": "a"})
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a", "routines=a", "probe=true"], observe=False)

    assert result["verified"] is True
    assert result["applied"] == {"hub": "a", "routines": "a"}
    assert result["ignored"] == ["probe"]
    assert result["prefs"] == [PREFS_FILE]
    assert result["ok"] is False
    assert "probe" in result["detail"]


def test_flags_set_is_ok_when_every_key_lands(tmp_path: Path) -> None:
    device = device_with({"hub": "a", "on": "true"})
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a", "on=true"], observe=False)

    assert result["ok"] is True
    assert result["verified"] is True
    assert result["applied"] == {"hub": "a", "on": "true"}
    assert result["ignored"] == []


def test_flags_set_returns_the_analysis_it_performed(tmp_path: Path) -> None:
    device = device_with(
        {"hub": "a"},
        hierarchy_xml=(
            '<hierarchy rotation="0"><node class="android.widget.TextView" '
            'text="Apps" bounds="[0,0][400,100]"/></hierarchy>'
        ),
    )
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=True)

    assert result["observation_present"] is True
    assert result["observation"]["elements"][0]["text"] == "Apps"


def test_flags_set_treats_a_different_stored_value_as_not_applied(tmp_path: Path) -> None:
    device = device_with({"hub": "control"})
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert result["ok"] is False
    assert result["mismatched"] == {"hub": "control"}
    assert result["applied"] == {}


def test_flags_set_reports_unverifiable_rather_than_claiming_success(tmp_path: Path) -> None:
    """A non-debuggable build cannot be read back — say so, don't fail and don't lie."""
    device = FakeDevice(
        package=PKG,
        activity=".MainActivity",
        run_as_error=f"run-as: package not debuggable: {PKG}",
    )
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert result["ok"] is True
    assert result["verified"] is False
    assert "not debuggable" in result["verify_error"]
    assert "applied" not in result


def test_flags_set_no_verify_reads_nothing_back(tmp_path: Path) -> None:
    device = device_with({"hub": "a"})
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False, verify=False)

    assert result["verified"] is False
    assert not [c for c in shell_commands(device) if c.startswith("run-as ")]


def test_flags_set_prefs_file_scopes_the_read_back(tmp_path: Path) -> None:
    """A key cached in an unrelated prefs file must not pass as the override landing."""
    device = FakeDevice(
        package=PKG,
        activity=".MainActivity",
        prefs={PREFS_FILE: {}, "remote-config-cache.xml": {"hub": "a"}},
    )
    engine = make_flags_engine(tmp_path, device, prefs_files={PKG: PREFS_FILE})

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert result["ok"] is False
    assert result["ignored"] == ["hub"]
    assert result["prefs"] == [PREFS_FILE]


def test_flags_set_reads_back_before_killing_the_app(tmp_path: Path) -> None:
    """Force-stopping first could drop a prefs write the app had not flushed yet."""
    device = device_with({"hub": "a"})
    engine = make_flags_engine(tmp_path, device)

    engine.flags_set(PKG, ["hub=a"], observe=False)

    names = call_names(device)
    last_read = max(i for i, c in enumerate(device.calls) if c[0] == "shell")
    assert last_read < names.index("stop_app")


# --------------------------------------------------------------------------------- restart


def test_flags_set_restarts_the_app_by_default(tmp_path: Path) -> None:
    """Flags read at cold start are invisible to the process that received the deeplink."""
    device = lifecycle_device()
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"])

    names = call_names(device)
    assert names.index("stop_app") < names.index("launch_app")
    assert result["restarted"] is True
    assert result["context_id"].startswith("flags-hub-")
    session = AppMemoryStore(engine.config.memory).load_session(device.serial)
    assert session.active_flags == {"hub": "a"}
    assert session.context_verified is True


def test_flags_set_restart_relaunches_the_activity_that_was_in_front(tmp_path: Path) -> None:
    device = lifecycle_device()
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert ("launch_app", (PKG, ".MainActivity")) in device.calls
    assert result["activity"] == ".MainActivity"


def test_flags_set_restart_takes_an_explicit_activity(tmp_path: Path) -> None:
    device = lifecycle_device()
    engine = make_flags_engine(tmp_path, device)

    engine.flags_set(PKG, ["hub=a"], observe=False, activity=".DevToolsActivity")

    assert ("launch_app", (PKG, ".DevToolsActivity")) in device.calls


def test_flags_set_restart_falls_back_when_the_pinned_entry_starts_nothing(
    tmp_path: Path,
) -> None:
    """A mid-flow Activity is not exported, so `am start -n` is a silent no-op.

    The fallback entry can land somewhere else entirely (a build with two launcher
    activities resolves it ambiguously), so the payload reports where the app IS.
    """
    device = lifecycle_device(exported=False, default_entry=".DevToolsActivity")
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert ("launch_app", (PKG,)) in device.calls
    assert result["restarted"] is True
    assert result["activity"] == ".DevToolsActivity"


def test_flags_set_reports_an_app_that_never_came_back(tmp_path: Path) -> None:
    device = lifecycle_device(exported=False, launchable=False)
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert result["restarted"] is False
    assert result["ok"] is False
    assert "did not come back" in result["restart_error"]
    assert result["applied"] == {"hub": "a"}


def test_flags_set_no_restart_leaves_the_process_alone(tmp_path: Path) -> None:
    device = lifecycle_device()
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False, restart=False)

    names = call_names(device)
    assert "stop_app" not in names
    assert "launch_app" not in names
    assert result["restarted"] is False
    assert any(c[0] == "open_link" for c in device.calls)


def test_flags_apply_verifies_and_restarts_too(tmp_path: Path) -> None:
    device = device_with({"hub": "a"})
    engine = make_flags_engine(tmp_path, device)
    path = tmp_path / "flags.yaml"
    path.write_text(f"app: {PKG}\nflags:\n  hub: a\n  probe: true\n", encoding="utf-8")

    result = engine.flags_apply(str(path), observe=False)

    assert result["ok"] is False
    assert result["ignored"] == ["probe"]
    assert call_names(device).count("launch_app") == 1


# ------------------------------------------------------------------------------------- cli


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "aua.yaml"
    path.write_text(
        f'flags:\n  templates:\n    {PKG}: "myapp://set-flags?{{query}}"\n', encoding="utf-8"
    )
    return path


def test_cli_flags_set_exits_non_zero_when_a_key_does_not_land(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped flag is a broken precondition for whatever runs next, not a success."""
    device = device_with({"hub": "a"})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    res = runner.invoke(
        app,
        ["--config", str(config_file), "flags", "set", PKG, "hub=a", "probe=true", "--no-observe"],
    )

    assert res.exit_code == 8
    payload = json.loads(res.stdout.splitlines()[0])
    assert payload["applied"] == {"hub": "a"}
    assert payload["ignored"] == ["probe"]
    assert "flags_not_applied" in res.stderr


def test_cli_flags_set_exits_zero_when_every_key_lands(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = device_with({"hub": "a"})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    res = runner.invoke(
        app, ["--config", str(config_file), "flags", "set", PKG, "hub=a", "--no-observe"]
    )

    assert res.exit_code == 0, res.stderr
    payload = json.loads(res.stdout.splitlines()[0])
    assert payload["verified"] is True
    assert payload["restarted"] is True


def test_cli_flags_set_exits_non_zero_when_the_app_does_not_come_back(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flags set but no live app is still a broken precondition — and a distinct code."""
    device = lifecycle_device(exported=False, launchable=False)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    res = runner.invoke(
        app, ["--config", str(config_file), "flags", "set", PKG, "hub=a", "--no-observe"]
    )

    assert res.exit_code == 8
    assert "app_not_restarted" in res.stderr
    assert json.loads(res.stdout.splitlines()[0])["applied"] == {"hub": "a"}


def test_cli_flags_set_no_restart(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = device_with({"hub": "a"})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    res = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "flags",
            "set",
            PKG,
            "hub=a",
            "--no-restart",
            "--no-observe",
        ],
    )

    assert res.exit_code == 0, res.stderr
    assert "stop_app" not in call_names(device)


def test_cli_flags_set_without_a_template_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = device_with({})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    res = runner.invoke(app, ["flags", "set", "com.other.app", "hub=a", "--no-observe"])

    assert res.exit_code == 2
    assert "no flags deeplink template" in res.stderr
