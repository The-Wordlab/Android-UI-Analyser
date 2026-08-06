"""Device network hygiene: orphaned proxies, and putting an inherited emulator right.

`settings put global http_proxy` is persistent; the mitmdump it points at, reached over an
`adb reverse` tunnel, is not. Every case below is a way that pairing goes wrong and leaves
an emulator where every app reports itself offline for a reason nothing on the device
explains — which is then inherited by whichever agent picks up that serial next.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep the shared proxy-ownership dir out of the developer's real cache."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    return tmp_path


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"dir": str(tmp_path / "mem")},
    )
    return Engine(cfg, device=device)


# --------------------------------------------------------------------------- orphan detection


def test_no_record_means_orphaned() -> None:
    """A proxy nothing claims cannot be serviced by anything, so it is dead by definition."""
    assert pm.orphan_reason(None, boot_id="b1") is not None


def test_dead_pid_is_orphaned() -> None:
    state = {"pid": 999_999_999, "port": 41234, "boot_id": "b1"}
    reason = pm.orphan_reason(state, boot_id="b1")
    assert reason and "gone" in reason


def test_reboot_orphans_the_proxy() -> None:
    """`adb reverse` does not survive a reboot; the `http_proxy` setting does."""
    state = {"pid": os.getpid(), "port": 41234, "boot_id": "before"}
    reason = pm.orphan_reason(state, boot_id="after")
    assert reason and "rebooted" in reason


def test_unreachable_port_is_orphaned() -> None:
    state = {"pid": os.getpid(), "port": pm.pick_listen_port(), "boot_id": "b1"}
    reason = pm.orphan_reason(state, boot_id="b1")
    assert reason and "listening" in reason


def test_a_live_proxy_is_left_alone() -> None:
    """Another agent's working proxy must survive: only positive evidence of death counts."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        state = {"pid": os.getpid(), "port": srv.getsockname()[1], "boot_id": "b1"}
        assert pm.orphan_reason(state, boot_id="b1") is None
    finally:
        srv.close()


# --------------------------------------------------------------------------- healing


def test_connect_clears_an_orphaned_proxy(tmp_path: Path) -> None:
    device = FakeDevice(serial="emulator-5554")
    device.set_http_proxy("127.0.0.1:41234")
    device.adb_reverse(41234, 41234)
    pm.write_state("emulator-5554", {"pid": 999_999_999, "port": 41234, "boot_id": None})

    engine = _engine(tmp_path, device)
    engine._heal_device_network(device)

    assert device.get_http_proxy() is None
    assert device.adb_reverse_list() == []
    assert pm.read_state("emulator-5554") is None


def test_heal_leaves_another_agents_live_proxy_alone(tmp_path: Path) -> None:
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        device = FakeDevice(serial="emulator-5554")
        device.set_http_proxy(f"127.0.0.1:{port}")
        pm.write_state(
            "emulator-5554",
            {"pid": os.getpid(), "port": port, "boot_id": None, "owner": "someone-else"},
        )
        engine = _engine(tmp_path, device)
        engine._heal_device_network(device)
        assert device.get_http_proxy() == f"127.0.0.1:{port}"
    finally:
        srv.close()


def test_heal_costs_nothing_when_no_proxy_was_ever_recorded(tmp_path: Path) -> None:
    """The hot path must not pay an adb round-trip on every single command."""
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    engine._lease_fresh = False
    device.calls.clear()
    engine._heal_device_network(device)
    assert device.calls == []


def test_heal_probes_the_device_when_the_emulator_was_just_inherited(tmp_path: Path) -> None:
    """A proxy left by a version that kept no record still has to be found and cleared."""
    device = FakeDevice(serial="emulator-5554")
    device.set_http_proxy("127.0.0.1:41234")
    engine = _engine(tmp_path, device)
    engine._lease_fresh = True
    engine._heal_device_network(device, first_connect=True)
    assert device.get_http_proxy() is None


def test_a_proxy_that_dies_mid_session_is_healed_without_reconnecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon serves a whole session from one engine, and devices break mid-session.

    A sweep that only ran on first connect left exactly the state the user reported: the
    proxy dies, every later command in that session sees an emulator with no internet, and
    the one process able to fix it never looks again.
    """
    import socket

    monkeypatch.setattr("android_ui_analyser.engine._HEAL_INTERVAL_S", 0.0)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    device.set_http_proxy(f"127.0.0.1:{port}")
    device.adb_reverse(port, port)
    pm.write_state("emulator-5554", {"pid": os.getpid(), "port": port, "boot_id": None})

    assert engine.device.get_http_proxy() == f"127.0.0.1:{port}"  # live: left alone

    srv.close()  # the proxy dies, mid-session, with no reconnect and no new engine

    assert engine.device.get_http_proxy() is None
    assert device.adb_reverse_list() == []
    assert pm.read_state("emulator-5554") is None


def test_the_sweep_does_not_undo_our_own_proxy_start(tmp_path: Path) -> None:
    """`proxy start` touches the device before it records ownership.

    In that window the record on disk still describes the previous run, so an un-paused
    sweep would read the proxy we just armed as an orphan and tear it straight back down.
    """
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    pm.write_state("emulator-5554", {"pid": 999_999_999, "port": 41234, "boot_id": None})

    with engine._rearranging_proxy():
        device.set_http_proxy("127.0.0.1:55555")
        assert engine.device.get_http_proxy() == "127.0.0.1:55555"

    assert device.get_http_proxy() == "127.0.0.1:55555"


# --------------------------------------------------------------------------- teardown


def test_proxy_stop_clears_without_the_port_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown must not depend on the cache dir of whoever started the proxy.

    Parallel agents keep separate caches by design, so the worker that has to clean up is
    usually not the one holding the record. The port is read back off the device instead.
    """
    device = FakeDevice(serial="emulator-5554")
    device.set_http_proxy("127.0.0.1:41234")
    device.adb_reverse(41234, 41234)
    engine = _engine(tmp_path, device)
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)

    out = engine.proxy_stop()

    assert out["port"] == 41234
    assert out["reverses_removed"] == [41234]
    assert out["http_proxy"] is None


def test_proxy_start_failure_does_not_strand_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If wiring the device up fails halfway, it must not be left pointing at the proxy."""
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    monkeypatch.setattr(pm, "start_mitm", lambda **_kw: (os.getpid(), 41234))
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)

    def boom(*_a, **_k):
        raise RuntimeError("adb reverse died")

    monkeypatch.setattr(device, "adb_reverse", boom)
    with pytest.raises(RuntimeError):
        engine.proxy_start(install_ca=False)
    assert device.get_http_proxy() is None


def test_proxy_start_drops_a_previous_runs_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache dir outlives the run that filled it; re-arming its stubs looks like a bug."""
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    cache = Path(engine.config.cache.dir).expanduser()
    engine.mock_map("GET", "/v1/stale", status=204)
    assert len(pm.load_rules(pm.rules_path(cache))) == 1

    monkeypatch.setattr(pm, "start_mitm", lambda **_kw: (os.getpid(), 41234))
    out = engine.proxy_start(install_ca=False)

    assert pm.load_rules(pm.rules_path(cache)) == []
    assert "mock_rules.json" in out["cleared_stale"]


def test_proxy_start_keeps_rules_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    engine.mock_map("GET", "/v1/keep", status=204)
    monkeypatch.setattr(pm, "start_mitm", lambda **_kw: (os.getpid(), 41234))
    engine.proxy_start(install_ca=False, keep_rules=True)
    rules = pm.load_rules(pm.rules_path(Path(engine.config.cache.dir)))
    assert len(rules) == 1


def test_proxy_start_bypasses_android_connectivity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keeps Google's probe hosts out of the capture, and app traffic to them off the proxy.

    Not, as previously claimed here, what keeps the network validated: measured on a real
    emulator, NetworkMonitor's own probe goes through the proxy regardless of this list.
    What actually keeps the device online is mitm being able to *serve* that probe, i.e. a
    system CA — see `install_system_ca`.
    """
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    monkeypatch.setattr(pm, "start_mitm", lambda **_kw: (os.getpid(), 41234))
    engine.proxy_start(install_ca=False, exclude=["extra.example.com"])
    excluded = device.get_proxy_exclusion_list()
    assert "connectivitycheck.gstatic.com" in excluded
    assert "extra.example.com" in excluded


def test_proxy_start_leaves_the_app_running_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arming the proxy is a mid-flow action; restarting the app would discard the state.

    You reach for this three screens into a flow, on the one request you got curious about.
    Verified against a real device: an already-running process's next requests go through
    the proxy, and the CA install nsenters into live app PIDs rather than needing a fresh
    Zygote fork.
    """
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    monkeypatch.setattr(pm, "start_mitm", lambda **_kw: (os.getpid(), 41234))

    out = engine.proxy_start(install_ca=False)

    assert "relaunched" not in out
    assert out["kept_running"] == device.current_app()["package"]
    assert not [c for c in device.calls if c[0] in ("stop_app", "launch_app")]


def test_proxy_start_still_restarts_the_app_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    monkeypatch.setattr(pm, "start_mitm", lambda **_kw: (os.getpid(), 41234))

    out = engine.proxy_start(install_ca=False, relaunch=True)

    assert out["relaunched"] == device.current_app()["package"]
    assert [c[0] for c in device.calls if c[0] in ("stop_app", "launch_app")] == [
        "stop_app",
        "launch_app",
    ]


# --------------------------------------------------------------------------- device reset


def test_device_reset_restores_a_wrecked_device(tmp_path: Path) -> None:
    device = FakeDevice(serial="emulator-5554")
    device.set_http_proxy("127.0.0.1:41234", exclusion_list=["a.example.com"])
    device.set_airplane_mode(True)
    device.adb_reverse(41234, 41234)
    engine = _engine(tmp_path, device)

    out = engine.device_reset(reverses=True)

    assert out["ok"]
    assert out["after"]["http_proxy"] is None
    assert out["after"]["airplane"] is False
    assert out["after"]["reverses"] == []
    assert ("set_wifi_enabled", (True,)) in device.calls


def test_device_reset_is_a_no_op_on_a_clean_device(tmp_path: Path) -> None:
    device = FakeDevice(serial="emulator-5554")
    engine = _engine(tmp_path, device)
    out = engine.device_reset()
    assert out["ok"]
    assert out["changed"] == ["nothing to reset — the device's network was already clean"]
    assert "left_alone" not in out


def test_device_reset_does_not_call_a_device_with_leftover_reverses_clean(
    tmp_path: Path,
) -> None:
    """Reporting "clean" while listing debris in the same payload teaches distrust.

    Reverses are left in place on purpose — Metro's `tcp:8081` is indistinguishable from an
    abandoned tunnel — but that is a decision to state, not to hide.
    """
    device = FakeDevice(serial="emulator-5554")
    device.adb_reverse(8081, 8081)
    engine = _engine(tmp_path, device)

    out = engine.device_reset()

    assert out["before"]["reverses"] == [8081]
    assert "already clean" not in " ".join(out["changed"])
    assert out["left_alone"] and "8081" in out["left_alone"][0]
    assert device.adb_reverse_list() == [8081]  # still untouched

    out = engine.device_reset(reverses=True)
    assert device.adb_reverse_list() == []
    assert "left_alone" not in out


# --------------------------------------------------------------------------- rule guards


def test_a_rule_matching_every_host_and_path_is_refused() -> None:
    """Arming one is indistinguishable, from the device, from losing the network."""
    from android_ui_analyser.errors import UsageError

    for path in ("/", "*", ""):
        with pytest.raises(UsageError, match="every request"):
            pm.guard_rule_scope(pm.map_rule("GET", path, status=204))


def test_a_broad_path_is_fine_once_scoped_to_a_host() -> None:
    pm.guard_rule_scope(pm.map_rule("GET", "/", status=204, host="api.example.com"))
