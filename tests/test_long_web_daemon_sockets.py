"""Long targets remain isolated, bindable, and visible to daemon status/stop-all."""

import json
import os
import socket
import tempfile
from pathlib import Path

from android_ui_analyser import daemon
from android_ui_analyser.config import Config


def test_long_urls_have_short_deterministic_distinct_bindable_sockets(monkeypatch):
    monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="aua-sockets-") as directory:
        config = Config()
        config.daemon.socket = f"{directory}/daemon.sock"
        target = "https://example.test/" + "same-prefix/" * 30
        paths = {
            daemon.socket_path(config, target + "a", platform="web"),
            daemon.socket_path(config, target + "b", platform="web"),
            daemon.socket_path(config, target + "a", platform="other"),
        }
        assert len(paths) == 3
        assert daemon.socket_path(config, target + "a", platform="web") in paths
        for path in paths:
            assert len(os.fsencode(path)) <= 103
            assert daemon.serial_for_socket(path) is None
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(path)
        monkeypatch.setattr(daemon, "_socket_process_alive", lambda path: path in paths)
        monkeypatch.setattr(daemon, "_socket_alive", lambda path: False)
        assert set(daemon.live_sockets(config)) == paths


def test_long_base_and_unicode_are_bounded_and_configuration_scoped(monkeypatch):
    monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)
    config = Config()
    config.daemon.socket = "/tmp/" + "長い" * 80 + "/daemon.sock"
    target = "https://example.test/" + "é" * 100
    first = daemon.socket_path(config, target, platform="web")
    config.daemon.socket += "-other"
    second = daemon.socket_path(config, target, platform="web")
    assert first != second
    assert len(os.fsencode(first)) <= 103
    assert len(os.fsencode(second)) <= 103


def test_discovery_includes_relocated_sockets_and_preserves_explicit_override(monkeypatch):
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="aua-sockets-") as directory:
        config = Config()
        config.daemon.socket = f"{directory}/" + "long/" * 40 + "daemon.sock"
        fallback = f"{directory}/short.sock"
        monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)
        monkeypatch.setattr(daemon, "_short_socket_base", lambda base: fallback)
        path = daemon.socket_path(config, "https://example.test/", platform="web")
        Path(path).touch()
        monkeypatch.setattr(daemon, "_socket_process_alive", lambda sock: sock == path)
        monkeypatch.setattr(daemon, "_socket_alive", lambda sock: False)
        assert daemon.live_sockets(config) == [path]
        monkeypatch.setenv("AUA_DAEMON_SOCKET", f"{directory}/explicit.sock")
        assert daemon.socket_path(config, "another-target", platform="other") == (
            f"{directory}/explicit.sock"
        )
        assert daemon.live_sockets(config) == []


def test_reap_removes_only_this_bases_relocated_stale_records(monkeypatch):
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="aua-sockets-") as directory:
        config = Config()
        config.cache.dir = directory
        config.daemon.socket = f"{directory}/" + "long/" * 40 + "daemon.sock"
        fallback = f"{directory}/short.sock"
        monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)
        monkeypatch.setattr(daemon, "_short_socket_base", lambda base: fallback)
        monkeypatch.setattr(daemon, "_pid_alive", lambda pid: False)
        path = daemon.socket_path(config, "https://example.test/", platform="web")
        Path(path).touch()
        pidfile = Path(path + ".pid")
        pidfile.write_text(json.dumps({"pid": 12345}))
        other = Path(directory) / "unrelated.sock.pid"
        other.write_text(json.dumps({"pid": 12345}))
        assert daemon.reap(config)["count"] == 1
        assert not Path(path).exists()
        assert not pidfile.exists()
        assert other.exists()
