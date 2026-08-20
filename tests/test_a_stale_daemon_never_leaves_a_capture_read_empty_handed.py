"""A capture read must never end with the caller holding nothing.

Observed 2026-08-20 against a real daemon on a real emulator. A skewed daemon held a live
buffer and one indexed frame with an action mark. ``aua capture last --since last-action``
refused outright, hinting ``aua daemon stop && aua daemon start``. Followed verbatim, that
hint stopped nothing (it targets the bare ``daemon.sock`` while the daemon lives on
``daemon.sock.<serial>``), and the ``daemon start`` that followed healed the daemon only as a
*side effect* of its own orientation call — a restart which minted a fresh capture session.
The retry then failed with "no last-action mark in the capture buffer yet". Refusal, then
amnesia: two different dead ends, zero screenshots, and no third thing to try.

The refusal's stated premise -- "the buffer lives in the daemon" -- is false. ``CaptureBuffer``
is a *disk* ring: every kept frame is appended to ``index.jsonl`` with its timestamp, path and
action mark before the call that produced it returns. During the failing run above those
records were sitting in a plain JSONL file on the same host, readable by any process, the whole
time. Only ``_entries`` and ``_last_action_ms`` are process-local, and both are reconstructible
from that file.

So the tax and the emptiness are the same bug seen twice, and the fix is one ordering change
plus one reader: try to make the daemon current *first*, and if that cannot be done, answer
from the durable index and label where the answer came from.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import android_ui_analyser.cli as cli
import android_ui_analyser.daemon as daemon_mod
from android_ui_analyser.capture import CaptureBuffer, CaptureCfgView
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import ScreenImage
from conftest import FakeDevice, make_config, make_png

STALE = "0.0.0+srcnotthistree"


@pytest.fixture(autouse=True)
def _no_ambient_socket_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """`effective_serial`/`socket_path` read the environment, and the dev host has both set."""
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    monkeypatch.delenv("AUA_DAEMON_SOCKET", raising=False)


def _a_recorded_session(
    root: Path,
    serial: str,
    *,
    session_id: str,
    frames: int = 3,
    action_on: int | None = 1,
    t0: int = 1_700_000_000_000,
) -> Path:
    """Write the on-disk artefacts a daemon leaves behind: JPEGs plus an append-only index.

    Deliberately written by hand rather than by driving a ``CaptureBuffer``: the point under
    test is that a process which never owned the buffer can still read it.
    """
    session = root / serial / session_id
    (session / "frames").mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(frames):
        jpg = session / "frames" / f"{i:04d}.jpg"
        jpg.write_bytes(make_png(40, 40))
        lines.append(
            {
                "t_ms": t0 + i * 100,
                "path": str(jpg),
                "hash": f"h{i}",
                "bytes": jpg.stat().st_size,
                "w": 40,
                "h": 40,
                "action": "tap:Continue" if i == action_on else None,
            }
        )
    (session / "index.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    return session


# --------------------------------------------------------------------- the durable reader


def test_the_index_is_readable_by_a_process_that_never_owned_the_buffer(tmp_path: Path) -> None:
    """The refusal claimed these frames were unreachable. Reach them."""
    from android_ui_analyser.capture import read_session_from_disk

    _a_recorded_session(tmp_path, "emulator-5562", session_id="20260820-001938-e89b52")

    found = read_session_from_disk(tmp_path, "emulator-5562")

    assert found is not None, "a recorded session on disk must be readable without the daemon"
    assert found.session_id == "20260820-001938-e89b52"
    assert [e.t_ms for e in found.entries] == [
        1_700_000_000_000,
        1_700_000_000_100,
        1_700_000_000_200,
    ]
    assert found.last_action_ms == 1_700_000_000_100, "the action mark is persisted, so use it"
    assert found.indexed == 3
    assert found.available == 3


def test_the_reader_drops_frames_whose_jpeg_was_pruned_but_counts_them(tmp_path: Path) -> None:
    """``_prune`` deletes JPEGs while ``index.jsonl`` only ever grows.

    A session can therefore hold 48 index lines and one surviving image. Silently returning
    entries that point at deleted files would make every downstream reader crash on open;
    silently dropping them without saying so would make a gutted session look complete.
    """
    from android_ui_analyser.capture import read_session_from_disk

    session = _a_recorded_session(tmp_path, "emulator-5562", session_id="s1", frames=4)
    (session / "frames" / "0000.jpg").unlink()
    (session / "frames" / "0001.jpg").unlink()

    found = read_session_from_disk(tmp_path, "emulator-5562")

    assert found is not None
    assert found.indexed == 4
    assert found.available == 2, "the caller has to be told what is no longer on disk"
    assert all(Path(e.path).exists() for e in found.entries)


def test_the_reader_picks_the_newest_session_and_names_it(tmp_path: Path) -> None:
    """Two writers (daemon buffer, capture sidecar) can both own sessions for one serial."""
    from android_ui_analyser.capture import read_session_from_disk

    _a_recorded_session(tmp_path, "emulator-5562", session_id="20260820-001000-aaaaaa", t0=1_000)
    time.sleep(0.01)
    _a_recorded_session(
        tmp_path, "emulator-5562", session_id="20260820-002000-bbbbbb", t0=9_000_000
    )

    found = read_session_from_disk(tmp_path, "emulator-5562")

    assert found is not None
    assert found.session_id == "20260820-002000-bbbbbb", "the newest index wins, deterministically"


def test_a_serial_that_never_recorded_reads_as_nothing_not_as_an_error(tmp_path: Path) -> None:
    from android_ui_analyser.capture import read_session_from_disk

    assert read_session_from_disk(tmp_path, "emulator-9999") is None


# --------------------------------------------------------------------- the engine fallback


def _engine_without_a_buffer(tmp_path: Path, serial: str = "emulator-5562") -> Engine:
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, device={"serial": serial})
    engine = Engine(cfg, device=FakeDevice())
    assert engine._capture is None, "this test is about the process that has no buffer"
    return engine


def test_capture_last_answers_from_disk_instead_of_raising(tmp_path: Path) -> None:
    """The old behaviour raised "capture buffer is not running" and stopped there."""
    engine = _engine_without_a_buffer(tmp_path)
    _a_recorded_session(
        Path(engine.config.cache.dir) / "captures", "emulator-5562", session_id="s1"
    )

    result = engine.capture_last()

    assert result["count"] == 3, result
    assert result["frames"], result
    assert result["source"] == "disk-index", "an answer this indirect has to say what it is"
    assert result["session_id"] == "s1"


def test_the_disk_answer_never_passes_itself_off_as_the_live_buffer(tmp_path: Path) -> None:
    """Requirement: do not quietly hand back one thing labelled as another.

    A disk read is not the live post-action buffer -- it can be older, it can be missing
    pruned frames, and nothing is being appended to it now. Every one of those has to be on
    the face of the payload, in words, not just implied by a missing key.
    """
    engine = _engine_without_a_buffer(tmp_path)
    _a_recorded_session(
        Path(engine.config.cache.dir) / "captures", "emulator-5562", session_id="s1"
    )

    result = engine.capture_last()

    assert result["live"] is False, result
    assert "note" in result, "the payload must explain itself without the caller guessing"
    note = result["note"].lower()
    assert "disk" in note and "not" in note
    assert result["indexed"] == 3 and result["available"] == 3
    assert isinstance(result["newest_frame_age_ms"], int), "staleness is the first question"


def test_capture_status_read_from_disk_does_not_claim_a_running_buffer(tmp_path: Path) -> None:
    """A crashed daemon leaves a session dir that looks live. It is not live."""
    engine = _engine_without_a_buffer(tmp_path)
    _a_recorded_session(
        Path(engine.config.cache.dir) / "captures", "emulator-5562", session_id="s1"
    )

    result = engine.capture_status()

    assert result["running"] is False, "no process is sampling; saying otherwise is a lie"
    assert result["source"] == "disk-index"
    assert result["frames"] == 3
    assert result["last_action_ms"] == 1_700_000_000_100


def test_a_last_action_read_still_fails_loudly_when_nothing_was_ever_marked(
    tmp_path: Path,
) -> None:
    """Guard the guard: the fallback must not invent a window out of an unmarked session."""
    from android_ui_analyser.errors import AuaError

    engine = _engine_without_a_buffer(tmp_path)
    _a_recorded_session(
        Path(engine.config.cache.dir) / "captures",
        "emulator-5562",
        session_id="s1",
        action_on=None,
    )

    with pytest.raises(AuaError, match="last-action"):
        engine.capture_last(since="last-action")


def test_a_capture_read_with_nothing_on_disk_still_says_so_plainly(tmp_path: Path) -> None:
    from android_ui_analyser.errors import AuaError

    engine = _engine_without_a_buffer(tmp_path)

    with pytest.raises(AuaError, match="capture buffer is not running"):
        engine.capture_last()


# --------------------------------------------------------------------- the routing order


class _Client:
    asked: list[str] = []
    result: dict[str, Any] = {"ok": True}

    def __init__(self, *_a: Any, **_k: Any) -> None:
        pass

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_e: Any) -> bool:
        return False

    def call(self, cmd: str, **_a: Any) -> dict[str, Any]:
        type(self).asked.append(cmd)
        if cmd == "capture_status":
            return {"ok": True, "result": {"ok": True, "running": True, "frames": 40}}
        return {"ok": True, "result": type(self).result}


def _cfg(tmp_path: Path, serial: str | None = "emulator-5562") -> Config:
    cfg = Config()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg.cache.dir = str(cache)
    cfg.daemon.socket = str(cache / "daemon.sock")
    cfg.device.serial = serial
    return cfg


def _fake_engine(cfg: Config) -> Any:
    return SimpleNamespace(
        config=cfg,
        _lease_serial=cfg.device.serial,
        _lease_owner="agent-a",
        _lease_owner_resolved="agent-a",
        _lease_device=lambda: cfg.device.serial,
    )


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


def _pretend_the_daemon_is_this_process(cfg: Config) -> int:
    Path(daemon_mod.socket_path(cfg) + ".pid").write_text(
        json.dumps({"pid": os.getpid(), "exe": sys.executable})
    )
    return os.getpid()


def _pretend_a_job_is_under_way(cfg: Config, *, worker_pid: int) -> None:
    from android_ui_analyser import jobs as jobs_mod

    jobs_mod._write(
        cfg.cache.dir,
        jobs_mod.JobState(
            job_id="jobunderway",
            operation="await",
            args={"predicate": "text:Continue", "timeout_ms": 30_000},
            serial=str(cfg.device.serial or "emulator-5562"),
            owner="agent-b",
            status="running",
            created_ms=1,
            started_ms=2,
            worker_pid=worker_pid,
        ),
    )


def test_a_skewed_capture_read_makes_the_daemon_current_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering fix. Restart first; a warm correct answer beats a refusal.

    This inverts ``test_a_capture_read_refuses_rather_than_restarting_its_own_answer_away``,
    whose premise (frames die with the process) the disk-reader tests above disprove.
    """
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(daemon_mod, "DaemonClient", type("C", (_Client,), {"asked": []}))
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(
        daemon_mod, "running_policy_fingerprint", daemon_mod.policy_config_fingerprint
    )
    steps = _record_restart(monkeypatch)

    cli._route(_fake_engine(cfg), "capture_last")

    assert [name for name, _ in steps] == ["stop", "start"], (
        "a capture read must be allowed to heal the daemon it is asking"
    )


def test_a_capture_read_whose_restart_fails_degrades_to_disk_not_to_a_dead_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one case that genuinely cannot restart: another agent's job is in flight.

    Before, this raised and the caller was finished. Now it must fall through to the
    in-process engine, which answers from the durable index.
    """
    cfg = _cfg(tmp_path)
    _pretend_a_job_is_under_way(cfg, worker_pid=_pretend_the_daemon_is_this_process(cfg))
    monkeypatch.setattr(daemon_mod, "DaemonClient", type("C", (_Client,), {"asked": []}))
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(
        daemon_mod, "running_policy_fingerprint", daemon_mod.policy_config_fingerprint
    )
    steps = _record_restart(monkeypatch, refreshes_version=False)
    _a_recorded_session(Path(cfg.cache.dir) / "captures", "emulator-5562", session_id="s1")
    engine = Engine(
        make_config(cache={"dir": cfg.cache.dir}, device={"serial": "emulator-5562"}),
        device=FakeDevice(),
    )
    engine._lease_serial = "emulator-5562"  # type: ignore[attr-defined]
    engine._lease_owner = "agent-a"  # type: ignore[attr-defined]
    engine._lease_owner_resolved = "agent-a"  # type: ignore[attr-defined]
    engine._lease_device = lambda: "emulator-5562"  # type: ignore[attr-defined]
    engine.config.daemon.socket = cfg.daemon.socket
    engine.config.daemon.enabled = True

    result = cli._route(engine, "capture_last")

    assert steps == [], "a live job still must not be restarted away"
    assert result["source"] == "disk-index", result
    assert result["count"] == 3, result


def test_the_jobs_surface_still_refuses_because_it_has_no_durable_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not everything can degrade. A job's worker is a thread; there is no file to read.

    So ``job_*`` keeps the refusal -- but the hint it prints has to be a command that works.
    """
    from android_ui_analyser.errors import AuaError

    cfg = _cfg(tmp_path)
    _pretend_a_job_is_under_way(cfg, worker_pid=_pretend_the_daemon_is_this_process(cfg))
    monkeypatch.setattr(daemon_mod, "DaemonClient", type("C", (_Client,), {"asked": []}))
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(
        daemon_mod, "running_policy_fingerprint", daemon_mod.policy_config_fingerprint
    )
    _record_restart(monkeypatch, refreshes_version=False)

    with pytest.raises(AuaError) as caught:
        cli._route(_fake_engine(cfg), "job_list")

    hint = str(caught.value.hint or "")
    assert "--serial emulator-5562" in hint, (
        f"an unpinned `daemon stop` cannot reach daemon.sock.emulator-5562; hint was: {hint}"
    )


def test_the_skew_message_does_not_claim_to_know_which_side_is_older(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest comparison cannot order two versions; the daemon may be running NEWER code."""
    from android_ui_analyser.errors import AuaError

    cfg = _cfg(tmp_path)
    _pretend_a_job_is_under_way(cfg, worker_pid=_pretend_the_daemon_is_this_process(cfg))
    monkeypatch.setattr(daemon_mod, "DaemonClient", type("C", (_Client,), {"asked": []}))
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(
        daemon_mod, "running_policy_fingerprint", daemon_mod.policy_config_fingerprint
    )
    _record_restart(monkeypatch, refreshes_version=False)

    with pytest.raises(AuaError) as caught:
        cli._route(_fake_engine(cfg), "job_list")

    assert "older" not in str(caught.value).lower(), str(caught.value)


def test_a_restart_that_does_not_take_is_not_retried_on_every_single_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed restart costs stop (up to 5s) + start (up to 5s) and changes nothing.

    In a tree edited every few seconds that is worse than the in-process fallback it was
    meant to avoid, and the next call used to pay it again from scratch.
    """
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(daemon_mod, "DaemonClient", type("C", (_Client,), {"asked": []}))
    steps = _record_restart(monkeypatch, refreshes_version=False)

    assert cli._replace_skewed_daemon(daemon_mod, cfg, STALE) is False
    first = len(steps)
    assert first == 2, steps

    assert cli._replace_skewed_daemon(daemon_mod, cfg, STALE) is False
    assert len(steps) == first, "a restart that just failed must not be attempted again at once"


# ------------------------------------------------- the restart's own session discontinuity


def test_a_live_but_unmarked_buffer_still_finds_the_mark_the_restart_left_behind(
    tmp_path: Path,
) -> None:
    """Found live on emulator-5562 after the routing fix, and not caught by any test above.

    Healing the daemon is not sufficient on its own. A restart mints a *new* capture session,
    so immediately afterwards the buffer is live, correct and current -- and empty. A caller
    that asked `capture last --since last-action` because it just tapped something therefore
    still got nothing: not the old refusal, but "no last-action mark in the capture buffer
    yet", which is the same outcome dressed differently.

        $ aua --serial emulator-5562 capture last --since last-action
        {"error": {"code": "usage",
                   "message": "no last-action mark in the capture buffer yet", ...}}

    The action did happen, and its mark is on disk in the session the restart superseded. So
    the live path has to consult the index too when its own buffer cannot answer -- labelled,
    because frames from a superseded recording are emphatically not the live buffer.
    """
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        device={"serial": "emulator-5562"},
        capture={"enabled": True},
    )
    root = Path(cfg.cache.dir) / "captures"
    # What the restart left behind: a finished session that holds the mark.
    _a_recorded_session(root, "emulator-5562", session_id="20260820-002014-before", action_on=1)
    engine = Engine(cfg, device=FakeDevice())
    # ...and the live-but-empty session the fresh daemon just opened.
    live = CaptureBuffer(
        root=root,
        serial="emulator-5562",
        cfg=CaptureCfgView(enabled=True),
        screenshot=lambda: ScreenImage(make_png(40, 40), width=40, height=40),
        session_id="20260820-003337-after",
    )
    (live.dir / "frames").mkdir(parents=True, exist_ok=True)
    live.index_path.write_text("", encoding="utf-8")
    engine._capture = live
    assert live.last_action_ms() is None, "the fresh session has no mark; that is the premise"

    result = engine.capture_last(since="last-action")

    assert result["count"] == 2, result
    assert result["source"] == "disk-index", result
    assert result["live"] is False, result
    assert result["session_id"] == "20260820-002014-before", result


def test_an_unmarked_buffer_with_no_mark_anywhere_on_disk_still_refuses(
    tmp_path: Path,
) -> None:
    """Guard the guard: the disk consult must not invent a mark that never existed."""
    from android_ui_analyser.errors import AuaError

    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        device={"serial": "emulator-5562"},
        capture={"enabled": True},
    )
    root = Path(cfg.cache.dir) / "captures"
    _a_recorded_session(root, "emulator-5562", session_id="unmarked", action_on=None)
    engine = Engine(cfg, device=FakeDevice())
    live = CaptureBuffer(
        root=root,
        serial="emulator-5562",
        cfg=CaptureCfgView(enabled=True),
        screenshot=lambda: ScreenImage(make_png(40, 40), width=40, height=40),
        session_id="live-empty",
    )
    (live.dir / "frames").mkdir(parents=True, exist_ok=True)
    live.index_path.write_text("", encoding="utf-8")
    engine._capture = live

    with pytest.raises(AuaError, match="last-action"):
        engine.capture_last(since="last-action")


def test_the_marked_session_wins_even_when_a_newer_unmarked_one_exists(
    tmp_path: Path,
) -> None:
    """The realistic shape of the discontinuity, and the one that nearly slipped through.

    A restarted daemon does not stay empty for long -- the sampler is always on, so within a
    second the new session holds frames, just no *action* mark. "Newest session" then resolves
    to a session that cannot answer `--since last-action`, and erroring there would once again
    leave a caller who did tap something holding nothing.

    Choosing the newest session that actually carries a mark is the useful answer, and it is
    honest as long as it says so: `source`, `live: false`, the session id, and the age of its
    newest frame are all on the payload, so a caller can see it is reading a superseded
    recording rather than the buffer it asked about.
    """
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        device={"serial": "emulator-5562"},
        capture={"enabled": True},
    )
    root = Path(cfg.cache.dir) / "captures"
    _a_recorded_session(root, "emulator-5562", session_id="marked-earlier", action_on=1)
    time.sleep(0.01)
    _a_recorded_session(root, "emulator-5562", session_id="unmarked-newer", action_on=None)
    engine = Engine(cfg, device=FakeDevice())

    result = engine.capture_last(since="last-action")

    assert result["session_id"] == "marked-earlier", result
    assert result["count"] == 2, result
    assert result["live"] is False and result["source"] == "disk-index", result
    assert isinstance(result["newest_frame_age_ms"], int), result


def test_a_plain_capture_last_still_reads_the_newest_session(tmp_path: Path) -> None:
    """Guard the guard: only `--since last-action` may prefer an older session."""
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        device={"serial": "emulator-5562"},
        capture={"enabled": True},
    )
    root = Path(cfg.cache.dir) / "captures"
    _a_recorded_session(root, "emulator-5562", session_id="marked-earlier", action_on=1)
    time.sleep(0.01)
    _a_recorded_session(root, "emulator-5562", session_id="unmarked-newer", action_on=None)
    engine = Engine(cfg, device=FakeDevice())

    assert engine.capture_last()["session_id"] == "unmarked-newer"
