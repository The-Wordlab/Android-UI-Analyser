"""``daemon start`` must spawn a daemon bound to the device its socket is named for.

Reproduced against three attached emulators: ``aua --serial emulator-5558 daemon start``
reported success on ``daemon.sock.emulator-5558``, then every routed call failed with
"multiple devices attached (emulator-5554, emulator-5556, emulator-5558)".

``socket_path`` resolved ``serial or config.device.serial or AUA_SERIAL`` while ``start``
gated its ``--serial`` argv on the bare parameter, so ``start(config)`` produced a
*serial-less daemon on a serial-named socket* — it looked healthy and answered every request
through ``connect(None)``. Callers stopped and restarted it, then gave up and ran daemon-less,
losing exactly the warm-state amortization the daemon exists to provide.
"""

from __future__ import annotations

import android_ui_analyser.daemon as daemon_mod
from android_ui_analyser.config import Config


def _config(tmp_path, serial: str | None) -> Config:
    cfg = Config()
    cfg.daemon.socket = str(tmp_path / "daemon.sock")
    cfg.cache.dir = str(tmp_path / "cache")
    if serial is not None:
        cfg.device.serial = serial
    return cfg


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.pid = 4242


def _spawn_argv(monkeypatch, tmp_path, cfg: Config, *, serial: str | None) -> list[str]:
    captured: dict[str, list[str]] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(daemon_mod.subprocess, "Popen", fake_popen)
    # First probe says "nothing listening", the wait-loop probe says "up".
    alive = iter([False, True, True, True])
    monkeypatch.setattr(daemon_mod, "_socket_alive", lambda _s: next(alive, True))
    monkeypatch.setattr(daemon_mod, "reap", lambda _c: None)
    daemon_mod.start(cfg, serial=serial)
    return captured["cmd"]


def test_config_serial_reaches_the_child_argv(monkeypatch, tmp_path) -> None:
    """The regression: an implicit serial named the socket but never reached the process."""
    cfg = _config(tmp_path, "emulator-5558")
    argv = _spawn_argv(monkeypatch, tmp_path, cfg, serial=None)

    assert "--serial" in argv, "daemon spawned without the serial its socket is named for"
    assert argv[argv.index("--serial") + 1] == "emulator-5558"

    sock = argv[argv.index("--socket") + 1]
    assert sock.endswith(".emulator-5558")


def test_explicit_serial_still_wins(monkeypatch, tmp_path) -> None:
    cfg = _config(tmp_path, "emulator-5554")
    argv = _spawn_argv(monkeypatch, tmp_path, cfg, serial="emulator-5556")
    assert argv[argv.index("--serial") + 1] == "emulator-5556"
    assert argv[argv.index("--socket") + 1].endswith(".emulator-5556")


def test_socket_and_argv_cannot_disagree(monkeypatch, tmp_path) -> None:
    """Whatever names the socket must be what the daemon binds to — that is the invariant."""
    monkeypatch.setenv("AUA_SERIAL", "emulator-9999")
    cfg = _config(tmp_path, None)
    argv = _spawn_argv(monkeypatch, tmp_path, cfg, serial=None)

    sock = argv[argv.index("--socket") + 1]
    serial = argv[argv.index("--serial") + 1]
    assert sock.endswith(f".{serial}")
    assert serial == "emulator-9999"


def test_no_serial_anywhere_spawns_an_unpinned_daemon(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    cfg = _config(tmp_path, None)
    argv = _spawn_argv(monkeypatch, tmp_path, cfg, serial=None)

    assert "--serial" not in argv
    assert daemon_mod.effective_serial(cfg, None) is None


def test_selected_platform_reaches_child_and_namespaces_socket(monkeypatch, tmp_path) -> None:
    cfg = _config(tmp_path, "shared-id")
    cfg.device.platform = "ios"

    argv = _spawn_argv(monkeypatch, tmp_path, cfg, serial=None)

    assert argv[argv.index("--platform") + 1] == "ios"
    assert argv[argv.index("--socket") + 1].endswith(".@ios@shared-id")
