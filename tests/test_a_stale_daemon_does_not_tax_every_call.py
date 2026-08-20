"""Version skew is the normal state of an editable install, so it must not tax every call.

Observed 2026-08-19: thirteen consecutive CLI calls each logged "daemon runs aua X but this
CLI is Y; using in-process" and each then paid the full device attach the warm daemon exists
to amortize. ``_replace_skewed_daemon`` was written for exactly this, but it refused whenever
the daemon reported a live capture buffer — and ``capture.enabled`` defaults to true, so
``serve`` starts one on every warm daemon. The guard therefore fired on every attempt and the
restart path was unreachable in the default configuration.

Refusing there also protected nothing retrievable: under skew every ``capture_*`` command was
in ``_DAEMON_ONLY_METHODS`` and raised instead of answering, so no caller could read those
frames until the daemon was replaced anyway. The capture reads have since been taken back out
of that set — they have a durable on-disk form, so they can both heal the daemon and, failing
that, answer from the index. The one thing a restart must genuinely never do is kill a live
background job, whose worker is a thread inside the daemon; that is what is pinned here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.cli as cli
import android_ui_analyser.daemon as daemon_mod
import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.config import Config
from conftest import FakeDevice

runner = CliRunner()

STALE = "0.0.0+srcnotthistree"

HIERARCHY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Welcome" bounds="[0,0][1080,120]"/>
  <node index="1" class="android.widget.Button" text="Continue"
        resource-id="com.example.app:id/continue_btn" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


class _Client:
    """A daemon that answers whatever the scenario needs, and records what it was asked."""

    capture_running = True
    asked: list[str] = []
    result: dict[str, Any] = {"ok": True}

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def call(self, cmd: str, **_args: Any) -> dict[str, Any]:
        type(self).asked.append(cmd)
        if cmd == "capture_status":
            return {
                "ok": True,
                "result": {"ok": True, "running": self.capture_running, "frames": 40},
            }
        return {"ok": True, "result": type(self).result}


def _buffer_running() -> type[_Client]:
    """The default warm daemon: `serve` starts the always-on buffer, so this is all of them."""
    return type("BufferRunning", (_Client,), {"capture_running": True, "asked": []})


def _buffer_off() -> type[_Client]:
    return type("BufferOff", (_Client,), {"capture_running": False, "asked": []})


@pytest.fixture(autouse=True)
def _no_ambient_socket_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """`effective_serial`/`socket_path` read the environment, and the dev host has both set."""
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)


def _cfg(tmp_path: Path, serial: str | None = "emulator-5554") -> Config:
    cfg = Config()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg.cache.dir = str(cache)
    cfg.daemon.socket = str(cache / "daemon.sock")
    cfg.device.serial = serial
    return cfg


def _pretend_the_daemon_is_this_process(cfg: Config) -> int:
    """Write the pidfile that ties the socket to a pid, so job ownership can be checked."""
    sock = daemon_mod.socket_path(cfg)
    Path(sock + ".pid").write_text(json.dumps({"pid": os.getpid(), "exe": sys.executable}))
    return os.getpid()


def _pretend_a_job_is_under_way(cfg: Config, *, worker_pid: int) -> str:
    from android_ui_analyser import jobs as jobs_mod

    state = jobs_mod.JobState(
        job_id="jobunderway",
        operation="await",
        args={"predicate": "text:Continue", "timeout_ms": 30_000},
        serial=str(cfg.device.serial or "emulator-5554"),
        owner="agent-b",
        status="running",
        created_ms=1,
        started_ms=2,
        worker_pid=worker_pid,
    )
    jobs_mod._write(cfg.cache.dir, state)
    return state.job_id


def _record_restart(
    monkeypatch: pytest.MonkeyPatch, *, refreshes_version: bool = True
) -> list[tuple[str, str | None]]:
    steps: list[tuple[str, str | None]] = []
    replaced = {"done": False}

    def _stop(cfg: Config, **kwargs: Any) -> dict[str, Any]:
        steps.append(("stop", daemon_mod.socket_path(cfg, kwargs.get("serial"))))
        return {"running": False, "status": "stopped"}

    def _start(cfg: Config, **kwargs: Any) -> dict[str, Any]:
        steps.append(("start", kwargs.get("serial")))
        replaced["done"] = refreshes_version
        return {"running": True, "status": "started"}

    monkeypatch.setattr(daemon_mod, "stop", _stop)
    monkeypatch.setattr(daemon_mod, "start", _start)
    monkeypatch.setattr(
        daemon_mod,
        "running_version",
        lambda _cfg: daemon_mod._aua_version() if replaced["done"] else STALE,
    )
    return steps


def test_the_always_on_capture_buffer_no_longer_blocks_the_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(daemon_mod, "DaemonClient", _buffer_running())
    steps = _record_restart(monkeypatch)

    assert cli._replace_skewed_daemon(daemon_mod, cfg, STALE) is True
    assert [name for name, _ in steps] == ["stop", "start"]
    assert steps[1][1] == "emulator-5554", "the replacement must name the device it is for"


def test_a_live_background_job_is_never_restarted_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job worker is a thread inside the daemon, so SIGTERM would abandon another agent."""
    cfg = _cfg(tmp_path)
    _pretend_a_job_is_under_way(cfg, worker_pid=_pretend_the_daemon_is_this_process(cfg))
    monkeypatch.setattr(daemon_mod, "DaemonClient", _buffer_off())
    steps = _record_restart(monkeypatch)

    assert cli._replace_skewed_daemon(daemon_mod, cfg, STALE) is False
    assert steps == [], "a skew restart must not interrupt work already running on the device"


def test_a_finished_job_does_not_block_the_restart_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the guard: job records outlive the job, so only a live one may hold the restart."""
    from android_ui_analyser import jobs as jobs_mod

    cfg = _cfg(tmp_path)
    pid = _pretend_the_daemon_is_this_process(cfg)
    job_id = _pretend_a_job_is_under_way(cfg, worker_pid=pid)
    done = jobs_mod._read(cfg.cache.dir, job_id)
    assert done is not None
    done.status = "succeeded"
    jobs_mod._write(cfg.cache.dir, done)
    monkeypatch.setattr(daemon_mod, "DaemonClient", _buffer_off())
    steps = _record_restart(monkeypatch)

    assert cli._replace_skewed_daemon(daemon_mod, cfg, STALE) is True
    assert [name for name, _ in steps] == ["stop", "start"]


def test_the_restart_leaves_another_devices_daemon_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agents run concurrently against different emulators, one warm daemon each."""
    cfg = _cfg(tmp_path)
    sibling = Path(cfg.cache.dir) / "daemon.sock.emulator-5560"
    sibling.write_bytes(b"")
    monkeypatch.setattr(daemon_mod, "DaemonClient", _buffer_running())
    monkeypatch.setattr(
        daemon_mod,
        "stop_all",
        lambda *_a, **_k: pytest.fail("skew is per-daemon; it must never stop every device's"),
    )
    steps = _record_restart(monkeypatch)

    assert cli._replace_skewed_daemon(daemon_mod, cfg, STALE) is True
    assert [socket for name, socket in steps if name == "stop"] == [
        str(Path(cfg.cache.dir) / "daemon.sock.emulator-5554")
    ]
    assert sibling.exists(), "another device's daemon socket was removed"


def test_a_capture_read_heals_the_daemon_rather_than_dead_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reversed on 2026-08-20. This asserted the opposite, on a premise that is false.

    It was called `test_a_capture_read_refuses_rather_than_restarting_its_own_answer_away`, and
    its reasoning was "the frames live in the process a restart kills, so an empty answer would
    be a lie". The frames do not live in the process. ``CaptureBuffer`` is a *disk* ring: every
    kept frame is appended to ``<session>/index.jsonl`` — timestamp, path, hash, dimensions and
    the action mark ``--since last-action`` needs — before the call that produced it returns.
    Only ``_entries`` and ``_last_action_ms`` are process-local, and both are reconstructible
    from that file, which is what ``capture.read_session_from_disk`` now does.

    Measured cost of the refusal, against a real daemon on a real emulator: the caller got no
    frames at all. It was told to run `aua daemon stop && aua daemon start`, which stops the
    bare `daemon.sock` while the daemon lives on `daemon.sock.<serial>` — so it stopped
    nothing, and the `daemon start` that followed healed the daemon only as a side effect of
    its own orientation call, a restart that minted a fresh capture session. The retry then
    failed with "no last-action mark in the capture buffer yet". Meanwhile 48 indexed frames
    with 14 action marks sat in a plain JSONL file on the same host the entire time.

    So the refusal protected nothing and cost everything. Restarting first is now the rule for
    capture reads as it already was for every other call, and if the restart cannot be done the
    read degrades to the durable index labelled ``source: "disk-index"`` — see
    ``test_a_stale_daemon_never_leaves_a_capture_read_empty_handed.py``. What a restart must
    still never do is interrupt a live background job; that is pinned above, and `job_*` keeps
    the refusal because a job worker really is a thread with no durable form.
    """
    cfg = _cfg(tmp_path)
    client = _buffer_off()
    client.result = {"ok": True, "action": "capture-last", "frames": [], "count": 0}
    monkeypatch.setattr(daemon_mod, "DaemonClient", client)
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(
        daemon_mod, "running_policy_fingerprint", daemon_mod.policy_config_fingerprint
    )
    steps = _record_restart(monkeypatch)
    engine = SimpleNamespace(
        config=cfg,
        _lease_serial="emulator-5554",
        _lease_owner="agent-a",
        _lease_owner_resolved="agent-a",
        _lease_device=lambda: "emulator-5554",
    )

    cli._route(engine, "capture_last")

    assert [name for name, _ in steps] == ["stop", "start"], (
        "a capture read must be allowed to make the daemon current, like every other call"
    )


def test_the_skew_warning_stays_out_of_machine_readable_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--format compact` stdout must stay parseable, whatever the CLI wants to say.

    A diagnostic printed before the JSON body breaks every `json.load` on the other end. This
    locks the stream down for the skew warning specifically, because that is the one an agent
    sees on every call while a stale daemon is up.
    """
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUA_CACHE__DIR", str(cache))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "true")
    monkeypatch.setenv("AUA_DAEMON__SOCKET", str(cache / "daemon.sock"))
    monkeypatch.setenv("AUA_PERF__AUTO_DAEMON", "false")
    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda serial=None: FakeDevice(hierarchy_xml=HIERARCHY_XML),
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(daemon_mod, "DaemonClient", _buffer_off())
    # Never let this reach the real lifecycle: the pidfile below names the pytest worker, and
    # `stop` signals whatever the pidfile names.
    _record_restart(monkeypatch, refreshes_version=False)
    # A refused restart is the case that keeps warning, so make this one refuse.
    Path(str(cache / "daemon.sock") + ".pid").write_text(
        json.dumps({"pid": os.getpid(), "exe": sys.executable})
    )
    cfg = Config()
    cfg.cache.dir = str(cache)
    cfg.device.serial = None
    _pretend_a_job_is_under_way(cfg, worker_pid=os.getpid())

    result = runner.invoke(
        app,
        ["--no-cache", "--no-lease", "--format", "compact", "analyze", "--source", "hierarchy"],
    )

    assert result.exit_code == 0, result.stderr
    assert "using in-process" in result.stderr, "the skew has to be announced somewhere"
    assert "using in-process" not in result.stdout
    body = json.loads(result.stdout)
    assert body["elements"], body
