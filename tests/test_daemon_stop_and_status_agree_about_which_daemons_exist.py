"""`daemon status` and `daemon stop` must describe the same reality, in one voice.

Observed 2026-08-20 with a real daemon serving `daemon.sock.emulator-5562`:

    $ aua --format compact daemon status
    {"ok":true,"action":"daemon-status","running":true,
     "detail":{"running":false,"socket":".../daemon.sock",
               "others_running":[".../daemon.sock.emulator-5562"]}}

    $ aua --format compact daemon stop
    {"ok":true,"action":"daemon-stop"}
    $ ls ~/.cache/android-ui-analyser | grep sock
    daemon.sock.emulator-5562            # <- still serving
    daemon.sock.emulator-5562.pid

Two separate defects with one root: `socket_path` appends the serial only when one is known, so
"the" daemon is not a single thing, and both commands pretend it is.

`status` used the key `running` for two different questions in one payload -- host-wide at the
top, this-socket in the detail -- so both readings were individually true and the document as a
whole was incoherent. `stop` was worse: `daemon.stop()` returns `status: "not_running"` *and*
`others_still_running` precisely so a caller can learn it hit the wrong socket, and
`cli.py` threw both away in favour of a hard-coded `{"ok": True}`. That hard-coded success is
what closed the loop behind the `capture last` refusal, whose hint said to run exactly this.

`detail.running` stays scoped to this config's socket: `daemon start` uses it to decide whether
the child it just spawned came up, so a sibling must never answer on its behalf.
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


def _pretend_a_daemon_serves(cache: Path, serial: str | None) -> str:
    """A socket file plus a pidfile naming a live process -- what `live_sockets` looks for."""
    cache.mkdir(parents=True, exist_ok=True)
    sock = cache / ("daemon.sock" if serial is None else f"daemon.sock.{serial}")
    sock.write_bytes(b"")
    Path(str(sock) + ".pid").write_text(json.dumps({"pid": os.getpid(), "exe": sys.executable}))
    return str(sock)


def _cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUA_CACHE__DIR", str(cache))
    monkeypatch.setenv("AUA_DAEMON__SOCKET", str(cache / "daemon.sock"))
    monkeypatch.setenv("AUA_PERF__AUTO_DAEMON", "false")
    return cache


def _cfg(tmp_path: Path, serial: str | None = None) -> Config:
    cfg = Config()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg.cache.dir = str(cache)
    cfg.daemon.socket = str(cache / "daemon.sock")
    cfg.device.serial = serial
    return cfg


# ------------------------------------------------------------------------------ status


def test_status_does_not_contradict_itself_within_one_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`running: true` over `running: false` is not a nuance, it is an unusable answer."""
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")

    result = runner.invoke(app, ["--format", "compact", "daemon", "status"])

    assert result.exit_code == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["running"] == out["detail"]["running"], f"one key, one meaning, per payload: {out}"


def test_an_unpinned_status_still_reveals_the_daemon_it_is_not_talking_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coherence must not cost discoverability.

    This is the concern behind `test_the_cli_does_not_report_not_running_while_a_daemon_runs`,
    which fixed a real bug -- a bare `daemon status` said "not running" while routed commands
    were happily using `daemon.sock.emulator-5560`. That test met the concern by widening the
    top-level `running`, which produced the self-contradiction above. Met here instead by
    answering the two questions under two names, so nothing is hidden and nothing conflicts.
    """
    cache = _cli_env(tmp_path, monkeypatch)
    live = _pretend_a_daemon_serves(cache, "emulator-5562")

    result = runner.invoke(app, ["--format", "compact", "daemon", "status"])

    out = json.loads(result.stdout)
    assert out["any_running"] is True, out
    assert live in out["detail"]["others_running"], out


def test_status_says_which_socket_its_verdict_is_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")

    out = json.loads(runner.invoke(app, ["--format", "compact", "daemon", "status"]).stdout)

    assert out["socket"] == str(cache / "daemon.sock"), out


def test_status_enumerates_every_daemon_and_the_serial_each_one_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both naming schemes exist at once, and a caller cannot act without knowing which."""
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")
    _pretend_a_daemon_serves(cache, "emulator-5560")
    _pretend_a_daemon_serves(cache, None)

    out = json.loads(runner.invoke(app, ["--format", "compact", "daemon", "status"]).stdout)

    listed = {entry["serial"]: entry["socket"] for entry in out["daemons"]}
    assert listed == {
        "emulator-5560": str(cache / "daemon.sock.emulator-5560"),
        "emulator-5562": str(cache / "daemon.sock.emulator-5562"),
        None: str(cache / "daemon.sock"),
    }, out
    for entry in out["daemons"]:
        assert "stop_command" in entry, entry


def test_a_pinned_status_is_still_not_answered_by_another_devices_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cli_env(tmp_path, monkeypatch)
    other = _pretend_a_daemon_serves(cache, "emulator-5560")

    out = json.loads(
        runner.invoke(
            app, ["--serial", "emulator-5562", "--format", "compact", "daemon", "status"]
        ).stdout
    )

    assert out["running"] is False, out
    assert out["detail"]["socket"].endswith(".emulator-5562"), out
    assert other in out["detail"]["others_running"], out


# -------------------------------------------------------------------------------- stop


def test_an_unpinned_stop_does_not_report_success_while_a_daemon_keeps_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-coded `ok: true` that closed the loop behind the `capture last` refusal."""
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")

    result = runner.invoke(app, ["--format", "compact", "daemon", "stop"])

    out = json.loads(result.stdout)
    assert out["ok"] is False, f"nothing was stopped, so this cannot be a success: {out}"


def test_a_stop_that_stopped_nothing_says_exactly_what_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent following the printed advice must not end up where it started."""
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")

    out = json.loads(runner.invoke(app, ["--format", "compact", "daemon", "stop"]).stdout)

    hint = json.dumps(out)
    assert "--serial emulator-5562" in hint, f"the survivor has to be named runnably: {out}"


def test_the_stop_payload_carries_the_real_status_and_the_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`daemon.stop()` already returns both. The CLI discarded them."""
    cache = _cli_env(tmp_path, monkeypatch)
    live = _pretend_a_daemon_serves(cache, "emulator-5562")

    out = json.loads(runner.invoke(app, ["--format", "compact", "daemon", "stop"]).stdout)

    assert out["status"] == "not_running", out
    assert live in out["others_still_running"], out


def test_stop_all_reaches_the_per_serial_daemons_an_unpinned_stop_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There has to be one command that ends every daemon, whatever it is named."""
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")
    _pretend_a_daemon_serves(cache, "emulator-5560")
    killed: list[int] = []
    monkeypatch.setattr(daemon_mod.os, "kill", lambda pid, sig: killed.append(pid))

    result = runner.invoke(app, ["--format", "compact", "daemon", "stop", "--all"])

    assert result.exit_code == 0, result.stderr
    out = json.loads(result.stdout)
    assert len(out["stopped"]) == 2, out


def test_a_pinned_stop_reports_the_success_it_actually_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the guard: the honest failure above must not make real successes look failed."""
    cache = _cli_env(tmp_path, monkeypatch)
    _pretend_a_daemon_serves(cache, "emulator-5562")
    # Alive until the SIGTERM lands, dead after it — what `stop` is written against. A flatly
    # dead socket would take the `not_running` branch and prove nothing about a real stop.
    alive = {"yes": True}

    def _kill(_pid: int, _sig: int) -> None:
        alive["yes"] = False

    monkeypatch.setattr(daemon_mod.os, "kill", _kill)
    monkeypatch.setattr(daemon_mod, "_socket_process_alive", lambda _s: alive["yes"])

    out = json.loads(
        runner.invoke(
            app, ["--serial", "emulator-5562", "--format", "compact", "daemon", "stop"]
        ).stdout
    )

    assert out["ok"] is True, out
    assert out["status"] == "stopped", out
