"""`daemon status` has to answer about the daemon a routed call would actually reach.

`socket_path` appends the serial only when one is known, and `_route` resolves the process
lease *before* it picks a socket — on purpose, so an unpinned call cannot end up driving
another agent's device. The consequence is that every leased command talks to
`daemon.sock.<serial>` while a bare `aua daemon status` asks about `daemon.sock` and reports
only on that.

Observed 2026-08-19: `daemon status` printed `running: false` for `.../daemon.sock` in the same
session, on the same machine, in which the CLI was reading a live daemon's version off
`.../daemon.sock.emulator-5560` and warning about the skew on every call. Neither reading was
wrong about its own socket; presenting one of them as the verdict on "the" daemon was. `stop`
already names its siblings (`others_still_running`) for exactly this reason.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import android_ui_analyser.daemon as daemon_mod
from android_ui_analyser.cli import app
from android_ui_analyser.config import Config

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_ambient_socket_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """`effective_serial`/`socket_path` read the environment, and the dev host has both set."""
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)


def _pretend_a_daemon_serves(cache: Path, serial: str) -> str:
    """A socket file plus a pidfile naming a live process — what `live_sockets` looks for."""
    cache.mkdir(parents=True, exist_ok=True)
    sock = cache / f"daemon.sock.{serial}"
    sock.write_bytes(b"")
    Path(str(sock) + ".pid").write_text(json.dumps({"pid": os.getpid(), "exe": sys.executable}))
    return str(sock)


def _cfg(tmp_path: Path, serial: str | None = None) -> Config:
    cfg = Config()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg.cache.dir = str(cache)
    cfg.daemon.socket = str(cache / "daemon.sock")
    cfg.device.serial = serial
    return cfg


def _cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUA_CACHE__DIR", str(cache))
    monkeypatch.setenv("AUA_DAEMON__SOCKET", str(cache / "daemon.sock"))
    return cache


def test_an_unpinned_status_names_the_serial_daemon_it_left_out(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    live = _pretend_a_daemon_serves(tmp_path / "cache", "emulator-5560")

    detail = daemon_mod.status(cfg)

    assert detail["socket"].endswith("daemon.sock"), detail
    assert live in detail["others_running"], detail


def test_the_cli_does_not_hide_a_daemon_it_is_not_talking_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrowed on 2026-08-20: same concern, different key.

    This asserted `out["running"] is True` — a bare status must not read "not running" while
    routed commands are using `daemon.sock.emulator-5560`. The concern was right and is kept.
    The mechanism was not: it widened the top-level `running` to mean "any daemon on this host"
    while `detail.running` went on meaning "a daemon at this exact socket", so one payload said
    `running: true` over `running: false`. Both were true of their own socket and the document
    as a whole could not be acted on. Reproduced verbatim against a live daemon on
    emulator-5562 on 2026-08-20.

    Two questions now have two names — `running` for this socket, `any_running` plus a
    `daemons` list for the host — so nothing is concealed and nothing conflicts. See
    `test_daemon_stop_and_status_agree_about_which_daemons_exist.py`.
    """
    cache = _cli_env(tmp_path, monkeypatch)
    live = _pretend_a_daemon_serves(cache, "emulator-5560")

    result = runner.invoke(app, ["--format", "compact", "daemon", "status"])

    assert result.exit_code == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["any_running"] is True, out
    assert live in out["detail"]["others_running"], out
    assert "emulator-5560" in [entry["serial"] for entry in out["daemons"]], out


def test_a_pinned_status_is_not_answered_by_another_devices_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the guard: naming a serial asks about that device, so `true` would be a lie."""
    cache = _cli_env(tmp_path, monkeypatch)
    other = _pretend_a_daemon_serves(cache, "emulator-5560")

    result = runner.invoke(
        app, ["--serial", "emulator-5554", "--format", "compact", "daemon", "status"]
    )

    assert result.exit_code == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["running"] is False, out
    assert out["detail"]["socket"].endswith(".emulator-5554"), out
    assert other in out["detail"]["others_running"], out
