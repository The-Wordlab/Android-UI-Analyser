"""Detached workers preserve one exact selected adapter configuration privately."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import capture_sidecar, dashboard
from android_ui_analyser.config import Config
from android_ui_analyser.platforms.identity import TargetRef
from android_ui_analyser.platforms.options_transport import (
    encode_platform_options,
    platform_options_fingerprint,
)


def _plugin_config(tmp_path: Path) -> Config:
    config = Config()
    config.cache.dir = str(tmp_path)
    config.daemon.socket = str(tmp_path / "missing-daemon.sock")
    config.device.platform = "strict-external"
    config.platforms["strict-external"] = {
        "endpoint": "https://grid.invalid",
        "access_token": "literal-private-value",
    }
    return config


def test_dashboard_capture_passes_the_exact_selected_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _plugin_config(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        capture_sidecar,
        "start",
        lambda **kwargs: seen.update(kwargs)
        or {"ok": True, "status": "started", "socket": "fixture.sock"},
    )

    result = dashboard.ensure_capture(serial="shared-target", config=config)

    assert result["via"] == "sidecar"
    assert seen["platform"] == "strict-external"
    assert seen["platform_options"] == config.platforms["strict-external"]


def test_dashboard_service_transports_options_by_inherited_fd_not_argv_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _plugin_config(tmp_path)
    monkeypatch.setattr(dashboard, "service_status", lambda *_args, **_kwargs: {"running": False})
    monkeypatch.setattr(
        dashboard,
        "_dashboard_health",
        lambda _port: {"pid": 4242, "name": "", "name_resolved": False},
    )
    seen: dict[str, Any] = {}

    class Process:
        pid = 4242

        def poll(self) -> None:
            return None

    def popen(command: list[str], **kwargs: Any) -> Process:
        fd = int(command[command.index("--platform-options-fd") + 1])
        os.lseek(fd, 0, os.SEEK_SET)
        seen.update(
            command=command,
            kwargs=kwargs,
            payload=json.loads(os.read(fd, 1_048_576).decode("utf-8")),
        )
        return Process()

    monkeypatch.setattr(dashboard.subprocess, "Popen", popen)

    dashboard.start_service(config, port=48765)

    raw = "literal-private-value"
    assert seen["payload"] == config.platforms["strict-external"]
    assert raw not in " ".join(seen["command"])
    assert raw not in seen["kwargs"]["env"].values()
    assert tuple(seen["kwargs"]["pass_fds"])
    state = dashboard._read_service_state(tmp_path)
    assert state["platform"] == "strict-external"
    assert state["options_fingerprint"]
    assert raw not in json.dumps(state)


def test_dashboard_child_replaces_stale_discovered_platform_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "aua.yaml"
    config_path.write_text(
        "device:\n"
        "  platform: strict-external\n"
        "platforms:\n"
        "  strict-external:\n"
        "    endpoint: https://stale.invalid\n",
        encoding="utf-8",
    )
    expected = {"endpoint": "https://current.invalid", "token_env": "FIXTURE_TOKEN"}
    fingerprint = platform_options_fingerprint(expected, key_dir=tmp_path)
    state_path = dashboard._write_service_state(
        tmp_path,
        {
            "pid": 999999,
            "access_token": "",
            "options_fingerprint": fingerprint,
        },
    )
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, encode_platform_options(expected))
    finally:
        os.close(write_fd)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(dashboard.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        dashboard,
        "run",
        lambda **kwargs: seen.update(kwargs) or {"ok": True},
    )

    result = dashboard._service_main(
        [
            "--serve-service",
            "--state-file",
            str(state_path),
            "--port",
            "48765",
            "--bind",
            "127.0.0.1",
            "--cache-dir",
            str(tmp_path),
            "--config",
            str(config_path),
            "--platform",
            "strict-external",
            "--platform-options-fd",
            str(read_fd),
            "--platform-options-fingerprint",
            fingerprint,
        ]
    )

    assert result == 0
    assert seen["config"].platforms["strict-external"] == expected
    assert seen["options_fingerprint"] == fingerprint


def test_unknown_legacy_capture_sidecar_is_retired_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = TargetRef("android", "emulator-5556")
    scoped = capture_sidecar.socket_path(
        tmp_path, serial=ref.target_id, platform=ref.platform
    )
    legacy = capture_sidecar.socket_path(tmp_path)
    options = {"endpoint": "private-fixture"}
    fingerprint = platform_options_fingerprint(options, key_dir=tmp_path)
    spawned = False
    stopped: list[str] = []

    def ping(path: str) -> dict[str, Any] | None:
        if path == legacy:
            return {"ok": True, "pong": True}  # pre-foundation worker: identity unknown
        if path == scoped and spawned:
            return {
                "ok": True,
                "pong": True,
                "platform": ref.platform,
                "target_id": ref.target_id,
                "options_fingerprint": fingerprint,
            }
        return None

    class Process:
        pid = 31337

    def popen(command: list[str], **kwargs: Any) -> Process:
        nonlocal spawned
        spawned = True
        assert "private-fixture" not in " ".join(command)
        assert "private-fixture" not in kwargs["env"].values()
        return Process()

    monkeypatch.setattr(capture_sidecar, "_ping_response", ping)
    monkeypatch.setattr(
        capture_sidecar,
        "call",
        lambda path, command, **_kwargs: stopped.append(f"{path}:{command}") or {"ok": True},
    )
    monkeypatch.setattr(capture_sidecar.subprocess, "Popen", popen)

    result = capture_sidecar.start(
        serial=ref.target_id,
        cache_dir=tmp_path,
        cfg=SimpleNamespace(idle_fps=1.0, burst_fps=2.0, burst_ms=100),
        platform=ref.platform,
        platform_options=options,
    )

    assert result["status"] == "started"
    assert result["socket"] == scoped
    assert f"{legacy}:stop" in stopped


def test_empty_platform_options_need_no_local_key_even_if_one_is_corrupt(tmp_path: Path) -> None:
    key = tmp_path / ".platform-options-hmac-key"
    first = platform_options_fingerprint({}, key_dir=tmp_path)
    assert not key.exists()
    key.write_bytes(b"broken")
    assert platform_options_fingerprint({}, key_dir=tmp_path) == first
    key.unlink()
    assert platform_options_fingerprint({}, key_dir=tmp_path) == first


def test_option_identity_covers_referenced_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {"endpointEnv": "FIXTURE_ENDPOINT", "nested": {"token_env": "FIXTURE_TOKEN"}}
    monkeypatch.setenv("FIXTURE_ENDPOINT", "https://one.invalid")
    monkeypatch.setenv("FIXTURE_TOKEN", "credential-one")
    first = platform_options_fingerprint(options, key_dir=tmp_path)
    monkeypatch.setenv("FIXTURE_ENDPOINT", "https://two.invalid")
    second = platform_options_fingerprint(options, key_dir=tmp_path)
    monkeypatch.setenv("FIXTURE_TOKEN", "credential-two")
    third = platform_options_fingerprint(options, key_dir=tmp_path)
    assert len({first, second, third}) == 3
    assert "credential" not in first + second + third
