"""Retained action/job windows survive thinking time, later actions and daemon turnover."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser.capture import CaptureBuffer, CaptureCfgView
from android_ui_analyser.capture_evidence import EvidenceStore
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.jobs import JobManager
from android_ui_analyser.mcp_server import _dispatch, _tool_definitions
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.schema import ActionResult, OutputFormat
from conftest import FakeDevice, make_config, make_png


class HostEvidencePlatform(PlatformAdapter):
    name = "host-evidence-test"
    capabilities = frozenset()

    def connect(self, target_id: str | None = None):  # type: ignore[no-untyped-def]
        raise AssertionError("an evidence export must not connect to a target")

    def list_targets(self):  # type: ignore[no-untyped-def]
        raise AssertionError("an evidence export must not discover targets")

    def normalize_tree(self, raw_tree, screen_size, *, ignored_app_ids=()):  # type: ignore[no-untyped-def]
        return NormalizedTree(elements=[])


def _buffer(cache: Path, *, platform: str = "android", ttl_s: int = 180) -> CaptureBuffer:
    colors = iter((20, 80, 140, 210, 250))

    def shot() -> ScreenImage:
        value = next(colors)
        return ScreenImage(make_png(24, 24, color=(value, value, value)), width=24, height=24)

    return CaptureBuffer(
        root=cache / "captures",
        serial="example-target",
        platform=platform,
        cfg=CaptureCfgView(ttl_s=ttl_s, max_mb=1),
        screenshot=shot,
    )


def _mark(buf: CaptureBuffer, label: str) -> str:
    buf.mark(label, session_id="example-goal", owner="example-owner")
    value = buf.action_evidence()
    assert value is not None
    return str(value["ref"])


def test_exact_window_survives_next_action_ring_pruning_and_process_restart(tmp_path) -> None:
    buf = _buffer(tmp_path)
    ref = _mark(buf, "tap:First")
    buf._tick()
    buf._tick()
    second = _mark(buf, "tap:Second")  # closes the first window before touching the device
    buf._tick()
    # Prune all rolling frames as if the caller spent longer thinking than the ring TTL.
    buf.cfg.ttl_s = -1
    buf._prune()
    assert not list((buf.dir / "frames").glob("*.jpg"))

    restarted = EvidenceStore(tmp_path / "captures", "android", "example-target")
    metadata, entries = restarted.read(ref)
    other, later = restarted.read(second)
    assert metadata["action"] == "tap:First"
    assert metadata["session_id"] == "example-goal"
    assert metadata["capture_session_id"] == buf.session_id
    assert len(entries) == 2 and len(later) == 1
    assert other["ref"] != metadata["ref"]
    assert all(Path(entry["path"]).is_file() for entry in entries)
    assert {entry["hash"] for entry in entries}.isdisjoint(entry["hash"] for entry in later)


def test_sheet_and_later_gif_reuse_identical_frozen_window_without_device_reads(tmp_path) -> None:
    cfg = make_config(cache={"dir": str(tmp_path)}, device={"serial": "example-target"})
    platform = HostEvidencePlatform(cfg)
    buf = _buffer(tmp_path, platform=platform.name)
    ref = _mark(buf, "tap:Animate")
    buf._tick()
    buf._tick()
    engine = Engine(cfg, platform=platform)  # deliberately has no connected runtime

    sheet = engine.capture_sheet(str(tmp_path / "sheet.png"), evidence_ref=ref)
    buf._tick()  # animation continues; the already exported evidence remains immutable
    exported = engine.capture_export(str(tmp_path / "clip.gif"), evidence_ref=ref)
    assert sheet["source_frames"] == exported["frames"] == 2
    assert sheet["capture_evidence"] == exported["capture_evidence"]
    assert sheet["capture_evidence"]["state"] == "sealed"
    assert Path(sheet["path"]).is_file() and Path(exported["path"]).is_file()
    assert engine._device is None
    assert engine.capture_explain(evidence_ref=ref)["count"] == 2


@pytest.mark.parametrize("command", ["sheet", "export"])
def test_retained_export_refuses_partial_artifact_when_pruned_during_image_loading(
    tmp_path, monkeypatch, command
) -> None:
    from PIL import Image

    from android_ui_analyser import capture

    cfg = make_config(cache={"dir": str(tmp_path)}, device={"serial": "example-target"})
    platform = HostEvidencePlatform(cfg)
    engine = Engine(cfg, platform=platform)
    buf = _buffer(tmp_path, platform=platform.name)
    ref = _mark(buf, "tap:Animate")
    buf._tick()
    buf._tick()
    exporter_name = "export_animation" if command == "export" else "export_contact_sheet"
    exporter = getattr(capture, exporter_name)
    image_open = Image.open
    opened = []

    def prune_after_first_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = image_open(path, *args, **kwargs)
        opened.append(path)
        if len(opened) == 1:
            buf.evidence_store.prune(0)
        return result

    def export_after_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        # The engine has already read/validated the window and summarized its two frames.
        # Lose the files only once the exporter has opened its first (still readable) fd.
        assert kwargs["strict"] is True
        with monkeypatch.context() as race:
            race.setattr(Image, "open", prune_after_first_open)
            return exporter(*args, **kwargs)

    monkeypatch.setattr(capture, exporter_name, export_after_read)
    path = tmp_path / ("clip.gif" if command == "export" else "sheet.png")
    with pytest.raises(UsageError) as raised:
        getattr(engine, f"capture_{command}")(str(path), evidence_ref=ref)
    assert raised.value.code == "capture_evidence_incomplete"
    assert len(opened) == 1
    assert not path.exists()
    assert engine._device is None


def test_failed_new_window_does_not_reuse_previous_actions_reference(tmp_path, monkeypatch) -> None:
    buf = _buffer(tmp_path)
    ref = _mark(buf, "tap:First")
    buf._tick()

    def fail_begin(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("simulated unwritable retention store")

    monkeypatch.setattr(EvidenceStore, "begin", fail_begin)
    with pytest.raises(OSError, match="unwritable"):
        _mark(buf, "tap:Second")
    assert buf.action_evidence() is None
    assert len(buf.evidence_store.read(ref)[1]) == 1


@pytest.mark.parametrize("platform,target", [("other", "example-target"), ("android", "other")])
def test_reference_never_crosses_platform_or_target(tmp_path, platform, target) -> None:
    buf = _buffer(tmp_path)
    ref = _mark(buf, "tap:First")
    buf._tick()
    with pytest.raises(UsageError) as raised:
        EvidenceStore(tmp_path / "captures", platform, target).read(ref)
    assert raised.value.code == "capture_evidence_not_found"


def test_expired_or_incomplete_reference_never_substitutes_newer_frames(tmp_path) -> None:
    buf = _buffer(tmp_path)
    first = _mark(buf, "tap:First")
    buf._tick()
    store = buf.evidence_store
    _, entries = store.read(first)
    _mark(buf, "tap:Second")
    buf._tick()
    Path(entries[0]["path"]).unlink()
    with pytest.raises(UsageError) as incomplete:
        store.read(first)
    assert incomplete.value.code == "capture_evidence_incomplete"

    metadata_path = store._dir(first) / "window.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["expires_ms"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(UsageError) as expired:
        store.read(first)
    assert expired.value.code == "capture_evidence_expired"


def test_empty_read_does_not_freeze_a_still_recording_window(tmp_path) -> None:
    buf = _buffer(tmp_path)
    ref = _mark(buf, "tap:First")
    with pytest.raises(UsageError) as raised:
        buf.evidence_store.read(ref)
    assert raised.value.code == "capture_evidence_empty"
    buf._tick()
    assert len(buf.evidence_store.read(ref)[1]) == 1


def test_evidence_budget_prunes_whole_references_and_counts_retained_bytes(tmp_path) -> None:
    buf = _buffer(tmp_path)
    first = _mark(buf, "tap:First")
    buf._tick()
    second = _mark(buf, "tap:Second")
    buf._tick()
    store = buf.evidence_store
    second_size = sum(entry["bytes"] for entry in store.read(second)[1])
    assert buf.total_disk_bytes() >= store.disk_bytes() > second_size
    store.prune(second_size)
    with pytest.raises(UsageError) as raised:
        store.read(first)
    assert raised.value.code == "capture_evidence_not_found"
    assert len(store.read(second)[1]) == 1
    assert store.disk_bytes() <= second_size


def test_no_frame_windows_are_bounded_too(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("android_ui_analyser.capture_evidence.MAX_REFERENCES", 2)
    buf = _buffer(tmp_path)
    first = _mark(buf, "tap:First")
    _mark(buf, "tap:Second")
    _mark(buf, "tap:Third")
    assert len(list(buf.evidence_store.root.glob("*/window.json"))) == 2
    with pytest.raises(UsageError):
        buf.evidence_store.read(first)


@pytest.mark.parametrize("window", [{"seconds": 2}, {"since": "last-action"}])
def test_reference_and_relative_window_are_mutually_exclusive(tmp_path, window) -> None:
    engine = Engine(make_config(cache={"dir": str(tmp_path)}))
    with pytest.raises(UsageError, match="cannot be combined"):
        engine.capture_last(evidence_ref="cap:invalid", **window)


def test_action_response_exposes_existing_reference_without_sampling(tmp_path) -> None:
    device = FakeDevice()
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)
    buf = _buffer(tmp_path)
    engine._capture = buf
    ref = _mark(buf, "tap:First")
    result = engine._observe(ActionResult(ok=True, action="tap"), False)
    assert result.capture_evidence is not None
    assert result.capture_evidence["ref"] == ref
    assert device.screenshot_calls == 0


@pytest.mark.parametrize(
    "updates",
    [{}, {"finished_ms": 1}, {"serial": "another-target"}, {"owner": "another-owner"}],
)
def test_daemon_action_correlates_only_active_matching_goal_without_local_session_cache(
    tmp_path, monkeypatch, updates
) -> None:
    from android_ui_analyser.session import create_session_state

    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=FakeDevice())
    engine._lease_owner_resolved = "example-owner"
    engine._capture = _buffer(tmp_path)
    state = create_session_state(
        tmp_path,
        goal="Inspect animation",
        serial="example-target",
        owner="example-owner",
        recommended_kind="action",
        recommended_cli="aua tap-and-analyze --text Animate",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    assert getattr(engine, "_session_id", None) is None
    if updates:
        monkeypatch.setattr(
            "android_ui_analyser.session.load_session_state",
            lambda *args, **kwargs: state.model_copy(update=updates),
        )
    engine._capture_mark("tap:Animate")
    assert engine._capture_evidence()["session_id"] == (None if updates else state.session_id)


def test_owner_handoff_seals_old_window_before_another_owner_can_extend_it(tmp_path) -> None:
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=FakeDevice())
    buf = _buffer(tmp_path)
    engine._capture = buf
    ref = _mark(buf, "tap:First")
    buf._tick()
    engine._reset_owner_transient_state()
    buf._tick()
    assert engine._capture_evidence() is None
    assert len(buf.evidence_store.read(ref)[1]) == 1


def test_frame_in_flight_before_action_is_not_relabelled_as_its_response(
    tmp_path, monkeypatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr("android_ui_analyser.capture.time.time", lambda: now[0])
    buf = _buffer(tmp_path)
    reference = []

    def shot() -> ScreenImage:
        now[0] += 1
        reference.append(_mark(buf, "tap:DuringSample"))
        now[0] += 1
        return ScreenImage(make_png(24, 24), width=24, height=24)

    buf.screenshot = shot
    buf._tick()
    assert buf._entries[0].action is None
    with pytest.raises(UsageError) as raised:
        buf.evidence_store.read(reference[0])
    assert raised.value.code == "capture_evidence_empty"


def test_background_job_returns_its_own_persisted_capture_window(tmp_path, monkeypatch) -> None:
    device = FakeDevice(serial="example-target")
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)
    buf = _buffer(tmp_path)
    engine._capture = buf
    original = _mark(buf, "tap:Start")

    def execute(*args):  # type: ignore[no-untyped-def]
        buf._tick()
        return {"ok": True, "action": "await", "found": True}

    monkeypatch.setattr("android_ui_analyser.jobs._execute", execute)
    manager = JobManager(engine)
    started = manager.start("await", {"predicate": "text:Ready", "timeout_ms": 1000})
    terminal = manager.wait(started["job_id"], timeout_ms=1000)
    assert terminal["status"] == "succeeded"
    evidence = terminal["capture_evidence"]
    assert evidence["ref"] != original
    assert evidence["action"] == f"job:{started['job_id']}"
    assert evidence["state"] == "sealed"
    assert terminal["result"]["capture_evidence"] == evidence
    assert JobManager(engine).status(started["job_id"])["capture_evidence"] == evidence
    assert len(buf.evidence_store.read(evidence["ref"])[1]) == 1
    assert device.screenshot_calls == 0


@pytest.mark.parametrize("command", ["last", "sheet", "export", "explain"])
def test_cli_and_mcp_resolve_the_same_retained_evidence_without_connecting(
    tmp_path, monkeypatch, command
) -> None:
    from android_ui_analyser import cli

    cfg = make_config(cache={"dir": str(tmp_path)}, device={"serial": "example-target"})
    platform = HostEvidencePlatform(cfg)
    engine = Engine(cfg, platform=platform)
    buf = _buffer(tmp_path, platform=platform.name)
    ref = _mark(buf, "tap:Animate")
    buf._tick()
    buf._tick()
    # Keep real engine behavior, while replacing only the transport setup. This fake adapter
    # refuses both target discovery and connection; exports must need neither capability.
    monkeypatch.setattr(cli, "_run", lambda ctx, operation: operation(engine, OutputFormat.json))
    routed = []

    def route(current, name, **kwargs):  # type: ignore[no-untyped-def]
        routed.append((name, kwargs))
        return getattr(current, name)(**kwargs)

    monkeypatch.setattr(cli, "_route", route)
    method = f"capture_{command}"
    argv = ["capture", command]
    arguments = {"evidence_ref": ref}
    if command in {"sheet", "export"}:
        extension = "png" if command == "sheet" else "gif"
        path = str(tmp_path / f"cli.{extension}")
        argv.append(path)
        arguments["path"] = str(tmp_path / f"mcp.{extension}")
    result = CliRunner().invoke(cli.app, [*argv, "--evidence", ref])
    assert result.exit_code == 0, result.output
    cli_payload = json.loads(result.stdout)
    assert routed[0][0] == method and routed[0][1]["evidence_ref"] == ref
    definition = next(tool for tool in _tool_definitions() if tool.name == method)
    assert definition.inputSchema["properties"]["evidence_ref"]["type"] == "string"
    mcp_payload = _dispatch(engine, method, arguments)
    assert cli_payload["capture_evidence"] == mcp_payload["capture_evidence"]
    assert cli_payload["capture_evidence"]["frames"] == 2
    assert engine._device is None
