"""Web contexts have separate identities, sticky agent ownership and shared lease fencing."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from android_ui_analyser import daemon, leases
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError, DeviceLeasedError
from android_ui_analyser.platforms.web import WebPlatform
from test_web_platform import URL, FakeConnection, FakeLauncher, _config


def engine_for(tmp_path, owner, *, slots=2, serial=None, connection=None):
    cfg = _config(tmp_path, url=URL, headless=False, context_slots=slots)
    cfg.device.serial = serial
    cfg.lease.enabled = True
    cfg.lease.registry_dir = str(tmp_path / "leases")
    cfg.daemon.socket = str(tmp_path / "daemon.sock")
    adapter = WebPlatform(cfg, launcher=FakeLauncher(connection or FakeConnection()))
    engine = Engine(cfg, platform=adapter)
    engine._lease_owner = owner
    return engine


def test_concurrent_agents_get_different_contexts_and_sticky_daemon_routes(tmp_path):
    first = engine_for(tmp_path, "agent-a")
    second = engine_for(tmp_path, "agent-b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        targets = list(executor.map(lambda engine: engine._lease_device(), [first, second]))
    assert len(set(targets)) == 2
    assert all(target.startswith("browser:") for target in targets)
    assert daemon.socket_path(first.config, targets[0], platform="web") != daemon.socket_path(
        second.config, targets[1], platform="web"
    )
    resumed = engine_for(tmp_path, "agent-a")
    assert resumed._lease_device() == targets[0]
    assert first.platform.probe_target_capabilities(targets[0])["headed"]
    assert leases.holder(first.config.lease.registry_dir, targets[0], platform="web") == "agent-a"


def test_foreign_pinned_context_and_exhausted_pool_refuse_without_native_provisioning(tmp_path):
    first = engine_for(tmp_path, "agent-a", slots=1)
    target = first._lease_device()
    pinned = engine_for(tmp_path, "agent-b", slots=1, serial=target)
    with pytest.raises(DeviceLeasedError):
        pinned._lease_device()
    unpinned = engine_for(tmp_path, "agent-b", slots=1)
    with pytest.raises(DeviceLeasedError) as error:
        unpinned._prepare_session_target(
            provision_target=True, wait_for_lease_s=0, headed=False, audio=False
        )
    assert "context_slots" in error.value.hint
    assert "emulator" not in error.value.hint
    assert leases.holder(first.config.lease.registry_dir, target, platform="web") == "agent-a"


def test_session_records_agent_target_and_releases_only_its_own_context(tmp_path):
    first = engine_for(tmp_path, "agent-a")
    second = engine_for(tmp_path, "agent-b")
    try:
        a = first.session_start("Inspect the first fixture", headed=True)
        b = second.session_start("Inspect the second fixture", headed=True)
        assert first._session_state(a["session_id"]).owner == "agent-a"
        assert second._session_state(b["session_id"]).owner == "agent-b"
        first_target = first.device.serial
        second_target = second.device.serial
        assert first_target != second_target
        result = first.session_finish(a["session_id"], allow_incomplete=True)
        assert result["terminated"]
        assert leases.holder(first.config.lease.registry_dir, first_target, platform="web") is None
        assert (
            leases.holder(second.config.lease.registry_dir, second_target, platform="web")
            == "agent-b"
        )
        assert second.session_finish(b["session_id"], allow_incomplete=True)["terminated"]
    finally:
        first.close()
        second.close()


def test_detached_session_reconnects_in_the_same_warm_engine(tmp_path):
    class DetachingConnection(FakeConnection):
        def session_finish(self, session_id):
            self.closed = True
            return {"ok": True, "detached": True}

    connection = DetachingConnection()
    engine = engine_for(tmp_path, "agent-a", connection=connection)
    try:
        started = engine.session_start("Inspect fixture")
        assert engine.session_finish(started["session_id"], allow_incomplete=True)["terminated"]
        assert connection.closed and engine._device is None
    finally:
        engine.close()


def test_android_keeps_its_existing_lease_recovery_advice(tmp_path):
    from android_ui_analyser.config import Config
    from android_ui_analyser.platforms.android import AndroidPlatform

    assert AndroidPlatform(Config()).lease_conflict_hint() is None
    assert AndroidPlatform(Config()).retain_runtime_on_lease_change()


@pytest.mark.parametrize("next_owner", ["agent-a", "agent-b"])
def test_reclaimed_warm_context_is_fresh_before_the_next_owners_observation(tmp_path, next_owner):
    from test_browser_observation_diagnostics import Connection

    old = Connection()
    new = Connection()
    engine = engine_for(tmp_path, "agent-a", slots=1, connection=old)
    launcher = engine.platform._launcher
    try:
        started = engine.session_start("Inspect first fixture")
        target = engine.device.serial
        old.emit("console", "Previous agent content", level="log")
        engine.release_device_use()
        assert leases.release(
            engine.config.lease.registry_dir, target,
            owner=engine._lease_owner_resolved, platform="web",
        )
        launcher.connection = new
        daemon._adopt_client_owner(engine, next_owner)
        if next_owner == "agent-a":
            # Owner adoption intentionally skips work for an unchanged identity; the new lease
            # generation must still invalidate its previous runtime.
            engine._lease_device()
        observed = engine.analyze(source="hierarchy", with_ocr=False)
        assert old.closed
        assert engine.device._connection is new
        assert observed.meta.browser_diagnostics["events"] == []
        resumed = engine.session_start("Inspect new fixture")
        assert resumed["session_id"] != started["session_id"]
        assert engine._session_state(resumed["session_id"]).owner == next_owner
    finally:
        engine.close()


def test_refused_owner_cannot_close_the_current_agents_browser(tmp_path):
    old = FakeConnection()
    engine = engine_for(tmp_path, "agent-a", slots=1, connection=old)
    try:
        engine.session_start("Inspect fixture")
        target = engine.device.serial
        engine.release_device_use()
        with pytest.raises(DeviceLeasedError):
            daemon._adopt_client_owner(engine, "agent-b")
        assert not old.closed
        assert engine._device._connection is old
        assert leases.holder(engine.config.lease.registry_dir, target, platform="web") == "agent-a"
        daemon._adopt_client_owner(engine, "agent-a")
        engine.analyze(source="hierarchy", with_ocr=False)
        assert not old.closed
    finally:
        engine.close()


@pytest.mark.parametrize("next_owner", ["agent-a", "agent-b"])
def test_daemon_reconnects_reclaimed_context_without_reacquiring_under_its_fence(
    tmp_path, next_owner,
):
    old = FakeConnection()
    new = FakeConnection()
    engine = engine_for(tmp_path, "agent-a", slots=1, connection=old)
    try:
        engine.session_start("Inspect first fixture")
        target = engine.device.serial
        engine.release_device_use()
        assert leases.release(
            engine.config.lease.registry_dir, target,
            owner=engine._lease_owner_resolved, platform="web",
        )
        engine.platform._launcher.connection = new
        if next_owner == "agent-a":
            engine._lease_device()
        result = daemon.dispatch(engine, {
            "cmd": "analyze", "owner": next_owner,
            "args": {"source": "hierarchy", "with_ocr": False},
        })
        assert result["ok"], result
        assert old.closed
        assert engine.device._connection is new
        assert leases.holder(
            engine.config.lease.registry_dir, target, platform="web",
        ) == next_owner
    finally:
        engine.close()


def test_adapter_can_keep_native_runtime_across_a_validated_lease_change(tmp_path, monkeypatch):
    from android_ui_analyser.platforms.base import PlatformAdapter

    connection = FakeConnection()
    engine = engine_for(tmp_path, "agent-a", slots=1, connection=connection)
    monkeypatch.setattr(
        engine.platform, "retain_runtime_on_lease_change",
        lambda: PlatformAdapter.retain_runtime_on_lease_change(engine.platform),
    )
    try:
        engine.session_start("Inspect fixture")
        target = engine.device.serial
        engine.release_device_use()
        assert leases.release(
            engine.config.lease.registry_dir, target,
            owner=engine._lease_owner_resolved, platform="web",
        )
        daemon._adopt_client_owner(engine, "agent-b")
        engine.analyze(source="hierarchy", with_ocr=False)
        assert not connection.closed
        assert engine.device._connection is connection
    finally:
        engine.close()


def test_explicit_url_remains_one_exclusive_target(tmp_path):
    first = engine_for(tmp_path, "agent-a", serial=URL)
    second = engine_for(tmp_path, "agent-b", serial=URL)
    assert first._lease_device() == URL
    with pytest.raises(DeviceLeasedError):
        second._lease_device()


def test_session_bootstrap_releases_the_daemon_read_fence_before_reclaiming(tmp_path):
    engine = engine_for(tmp_path, "agent-a")
    try:
        target = engine._lease_device()
        engine.config.device.serial = target
        engine.begin_device_use()
        started = engine.session_start("Inspect fixture", headed=True)
        assert engine.device.serial == target
        assert engine.session_finish(started["session_id"], allow_incomplete=True)["terminated"]
    finally:
        engine.close()


@pytest.mark.parametrize("new_claim", [True, False])
def test_failed_attachment_releases_only_the_callers_new_claim(tmp_path, new_claim):
    engine = engine_for(tmp_path, "agent-a")
    target = engine._lease_device()
    engine.config.device.serial = target
    engine._lease_was_preexisting = True  # The daemon adopts the caller's claim.

    class FailingLauncher:
        def launch(self, *_):
            raise DeviceError("fixture connection failed")

    engine.platform._launcher = FailingLauncher()
    try:
        engine.begin_device_use()
        with pytest.raises(DeviceError, match="fixture connection failed"):
            engine.session_start("Inspect fixture", _bootstrap_new_lease=new_claim)
        holder = leases.holder(engine.config.lease.registry_dir, target, platform="web")
        assert holder == (None if new_claim else "agent-a")
    finally:
        engine.close()


@pytest.mark.parametrize("preexisting", [True, False])
def test_late_bootstrap_failure_keeps_the_original_claim_provenance(tmp_path, preexisting):
    class FailingBaselineConnection(FakeConnection):
        def session_begin(self, session_id):
            raise DeviceError("fixture baseline failed after observation")

    connection = FailingBaselineConnection()
    engine = engine_for(tmp_path, "agent-a", slots=1, connection=connection)
    target = engine.platform.list_targets()[0].target_id
    if preexisting:
        engine._lease_device()
    try:
        with pytest.raises(DeviceError, match="fixture baseline failed after observation"):
            engine.session_start("Inspect fixture")
        assert connection.calls, "bootstrap reached the first observation"
        holder = leases.holder(engine.config.lease.registry_dir, target, platform="web")
        assert holder == ("agent-a" if preexisting else None)
        assert connection.closed is (not preexisting)
        assert engine._session_bootstrap_lease_preexisting is None
    finally:
        engine.close()


@pytest.mark.parametrize("preexisting", [True, False])
def test_cli_transfers_claim_origin_to_the_session_daemon(tmp_path, monkeypatch, preexisting):
    from android_ui_analyser.cli import _route

    engine = engine_for(tmp_path, "agent-a")
    engine.config.daemon.enabled = True
    if preexisting:
        engine._lease_device()
    # Each CLI invocation has a fresh Engine even when its caller already owns a target.
    engine = engine_for(tmp_path, "agent-a")
    engine.config.daemon.enabled = True
    calls = []

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def call(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"ok": True, "result": {}, "response_decorated": True}

    monkeypatch.setattr(daemon, "is_running", lambda _: True)
    monkeypatch.setattr(daemon, "running_version", lambda _: daemon._aua_version())
    monkeypatch.setattr(daemon, "running_runtime_fingerprint", daemon.runtime_config_fingerprint)
    monkeypatch.setattr(daemon, "running_policy_fingerprint", daemon.policy_config_fingerprint)
    monkeypatch.setattr(daemon, "DaemonClient", Client)
    try:
        _route(engine, "session_start", goal="Inspect fixture")
        assert calls[0][0] == "session_start"
        assert calls[0][1].get("_bootstrap_new_lease", False) is (not preexisting)
    finally:
        engine.close()
