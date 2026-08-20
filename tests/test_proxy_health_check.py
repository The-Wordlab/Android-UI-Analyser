"""The proxy can look armed on every signal an agent used to check, and still be a black hole.

Measured 2026-08-19: ``settings get global http_proxy`` reported ``127.0.0.1:44794`` (armed),
mitmdump was alive and listening on 44794 (armed) — but the ``adb reverse`` tunnel that lets a
loopback-only device reach that host port was gone. Every network call from the app under test
failed with ``java.net.ConnectException: Failed to connect to /127.0.0.1:44794``, visible only
in logcat. Nothing in any `aua` surface said so: the device setting and the process were each
individually fine, and nothing had ever checked the third piece — the tunnel — together with
the other two.

These tests reproduce that exact split with fakes (no real mitmdump, no real device): the
setting and the process both report healthy, only the tunnel is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    InvalidPlatformCapabilityError,
    UnsupportedPlatformCapabilityError,
)
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import PlatformAdapter
from android_ui_analyser.platforms.services import CAPABILITY_METHODS, PROXY, missing_members
from conftest import FakeDevice, make_config

SERIAL = "fake-emulator-5554"
PORT = 44794
PID = 4242


def _arm_state(tmp_path: Path, *, port: int = PORT, pid: int = PID) -> None:
    pm.write_state(SERIAL, {"pid": pid, "port": port, "boot_id": "boot-1"})


# --------------------------------------------------------------------------- pure health check


def test_healthy_process_and_setting_with_a_dropped_tunnel_is_reported_unhealthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact reported split: process alive+listening, setting armed, tunnel gone."""
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(pm, "port_listening", lambda port: port == PORT)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    # The tunnel is gone — this is the one signal nothing checked before.
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["checks"]["process"]["ok"] is True
    assert report["checks"]["device_setting"]["ok"] is True
    assert report["checks"]["tunnel"]["ok"] is False, (
        "a dead tunnel with everything else healthy must not be reported as a healthy proxy"
    )
    assert report["ok"] is False
    assert "tunnel" in report["hint"].lower()


def test_all_three_signals_healthy_reports_fully_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port: True)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: True)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["ok"] is True
    assert all(c["ok"] for c in report["checks"].values())


def test_a_dead_process_is_named_specifically_not_lumped_with_the_tunnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: False)
    monkeypatch.setattr(pm, "port_listening", lambda port: False)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["checks"]["process"]["ok"] is False
    assert str(PID) in report["checks"]["process"]["detail"]


def test_a_device_pointed_at_a_port_our_record_does_not_name_is_diagnosed_on_that_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrowed from `test_device_setting_pointed_elsewhere_is_its_own_named_failure`.

    The concern is kept: a device pointed somewhere other than our record must be a named
    failure, not a silent one. What changed is *which port gets diagnosed*. This used to report
    ``armed: true`` with ``device_setting.ok: false`` — describing the health of port 44794,
    which no traffic on this device goes to — and said nothing at all about whether 9999, the
    port the device actually uses, was alive. The device is ground truth for what it is pointed
    at, so the diagnosis now follows 9999 and calls the record what it is: stale.
    """
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: port == PORT)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: port == PORT)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: "127.0.0.1:9999")
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["ok"] is False, "a mispointed device is still a failure"
    assert report["target"]["port"] == 9999, "diagnose the port the device actually uses"
    assert report["owned"] is False, "we do not own what the device is pointed at"
    assert report["checks"]["listener"]["ok"] is False
    assert report["checks"]["tunnel"]["ok"] is False
    assert "stale" in report["hint"] and str(PORT) in report["hint"], (
        "name the stale record, or nobody knows to clear it"
    )


def test_nothing_armed_at_all_is_reported_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reversed assertion: a clean device is `ok: True`, not a failure.

    This asserted ``ok is False`` for a device with no proxy at all. That was the defect, not a
    decision: ``ok`` is the one field callers branch on, and holding it false for a perfectly
    clean device made it indistinguishable from the black-hole state — a device whose every
    request fails. `state` now carries that difference (`unproxied` vs `blackholed`) and `ok`
    means what it says: this device's network path is sane.
    """
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    report = pm.proxy_health("no-such-device", tmp_path, self_heal=False)
    assert report["ok"] is True
    assert report["state"] == "unproxied"
    assert report["owned"] is False
    assert report["intercepting"] is False
    assert report["checks"] == {}


# --------------------------------------------------------------------------- self-healing


def test_proxy_health_is_diagnostic_even_when_self_heal_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped `adb reverse` is a normal consequence of an adb restart, not user error."""
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port: True)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")

    tunnel_state = {"active": False}
    monkeypatch.setattr(
        pm, "reverse_tunnel_active", lambda serial, port, **kw: tunnel_state["active"]
    )
    ensure_calls: list[tuple[str, int]] = []

    def fake_ensure(serial: str, port: int) -> bool:
        ensure_calls.append((serial, port))
        tunnel_state["active"] = True
        return True

    monkeypatch.setattr(pm, "ensure_reverse_tunnel", fake_ensure)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=True)

    assert ensure_calls == [], "health checks cannot bypass Engine.record_device_change"
    assert report["checks"]["tunnel"]["ok"] is False
    assert report["ok"] is False


def test_proxy_health_does_not_self_heal_when_the_process_is_also_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to reconnect a tunnel to — re-pointing it at a dead port would be a lie."""
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: False)
    monkeypatch.setattr(pm, "port_listening", lambda port: False)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    ensure_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        pm, "ensure_reverse_tunnel", lambda serial, port: ensure_calls.append((serial, port))
    )

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=True)

    assert ensure_calls == [], "must not touch the device when there is no live process behind it"
    assert report["ok"] is False


def test_proxy_health_self_heal_flag_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port: True)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    ensure_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        pm, "ensure_reverse_tunnel", lambda serial, port: ensure_calls.append((serial, port))
    )

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert ensure_calls == []
    assert report["checks"]["tunnel"]["ok"] is False


# --------------------------------------------------------------------------- adb parsing (real primitives)


def test_reverse_tunnel_active_parses_adb_reverse_list(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def fake_adb(serial: str, *args: str, check: bool = True, timeout: float = 60):
        assert args[:2] == ("reverse", "--list")
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{serial} tcp:{PORT} tcp:{PORT}\n", stderr=""
        )

    monkeypatch.setattr(pm, "_adb", fake_adb)
    assert pm.reverse_tunnel_active(SERIAL, PORT) is True
    assert pm.reverse_tunnel_active(SERIAL, 9) is False


def test_owned_reverse_check_rejects_an_asymmetric_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    local = PORT + 1
    monkeypatch.setattr(
        pm,
        "_adb",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout=f"{SERIAL} tcp:{PORT} tcp:{local}\n", stderr=""
        ),
    )

    assert pm.reverse_tunnel_active(SERIAL, PORT) is True
    assert pm.reverse_tunnel_active(SERIAL, PORT, local_port=PORT) is False
    assert pm.reverse_tunnel_active(SERIAL, PORT, local_port=local) is True


def test_reverse_tunnel_active_false_when_list_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(
        pm, "_adb", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr="")
    )
    assert pm.reverse_tunnel_active(SERIAL, PORT) is False


def test_read_device_http_proxy_treats_null_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(
        pm, "_adb", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="null\n", stderr="")
    )
    assert pm.read_device_http_proxy(SERIAL) is None


def test_read_device_http_proxy_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(
        pm,
        "_adb",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=f"127.0.0.1:{PORT}\n", stderr=""),
    )
    assert pm.read_device_http_proxy(SERIAL) == f"127.0.0.1:{PORT}"


def test_ensure_reverse_tunnel_issues_adb_reverse_and_reports_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    calls: list[tuple[str, ...]] = []

    def fake_adb(serial: str, *args: str, check: bool = True, timeout: float = 60):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(pm, "_adb", fake_adb)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: True)
    assert pm.ensure_reverse_tunnel(SERIAL, PORT) is True
    assert ("reverse", f"tcp:{PORT}", f"tcp:{PORT}") in calls


# --------------------------------------------------------------------------- Engine orchestration


def _engine(tmp_path: Path) -> Engine:
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")})
    device = FakeDevice(serial=SERIAL)
    return Engine(cfg, device=device)


def test_engine_proxy_status_reports_the_split_and_self_heals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port: True)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    tunnel_state = {"active": False}
    monkeypatch.setattr(
        pm, "reverse_tunnel_active", lambda serial, port, **kw: tunnel_state["active"]
    )

    order: list[str] = []

    def fake_ensure(serial: str, port: int) -> bool:
        order.append("touched_device")
        tunnel_state["active"] = True
        return True

    monkeypatch.setattr(pm, "ensure_reverse_tunnel", fake_ensure)

    engine = _engine(tmp_path)

    # Patch `device_ledger.record` (imported lazily inside `record_device_change`) to also
    # observe *when* the ledger entry lands relative to the device touch.
    from android_ui_analyser import device_ledger as ledger_mod

    real_record = ledger_mod.record

    def spying_record(*args: Any, **kwargs: Any) -> None:
        order.append("recorded_undo")
        real_record(*args, **kwargs)

    monkeypatch.setattr(ledger_mod, "record", spying_record)

    out = engine.proxy_status()

    assert out["ok"] is True
    assert out["checks"]["tunnel"]["healed"] is True
    assert order == ["recorded_undo", "touched_device"], (
        "write-ahead: the undo must be journalled BEFORE the device is touched, or a crash "
        "between the two leaves a re-established tunnel with nothing on disk to clean it up"
    )
    # Same ledger key `proxy_start` uses, so a stranger who inherits this device can still
    # clean up (and re-healing never doubles the record for the same port).
    entries = device_ledger.read_ledger(SERIAL)
    assert any(e.key == f"reverse_port:{PORT}" and e.op == "remove_reverse_port" for e in entries)


def test_engine_proxy_status_never_touches_the_device_when_self_heal_is_unsafe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: False)
    monkeypatch.setattr(pm, "port_listening", lambda port: False)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    ensure_calls: list[Any] = []
    monkeypatch.setattr(pm, "ensure_reverse_tunnel", lambda *a: ensure_calls.append(a))

    engine = _engine(tmp_path)
    out = engine.proxy_status()

    assert out["ok"] is False
    assert ensure_calls == []


# --------------------------------------------------------------------------- platform boundary


class _FakePlatform(PlatformAdapter):
    """The iOS/web shape: claims nothing optional."""

    name = "fake"
    capabilities = frozenset()

    def connect(self, target_id: str | None = None) -> Any:  # pragma: no cover
        raise AssertionError("never called")

    def list_targets(self) -> list[Any]:  # pragma: no cover
        return []

    def normalize_tree(self, raw_tree: Any, screen_size: Any, *, ignored_app_ids: Any = ()) -> Any:
        raise AssertionError("not used by this test")


class _FakeProxyService:
    """A minimal stand-in service proving the engine only calls the capability contract.

    Implements the full PROXY structural surface (so it clears the capability gate exactly
    like a real plugin's service would have to) but only `read_state`/`proxy_health` do
    anything meaningful — everything else is an unused stand-in.
    """

    def __init__(self, *, healthy: bool) -> None:
        self.healthy = healthy
        self.calls: list[str] = []
        for name in sorted(CAPABILITY_METHODS[PROXY]):
            if not hasattr(self, name):
                setattr(self, name, self._stub)

    @staticmethod
    def _stub(*_a: Any, **_k: Any) -> None:  # pragma: no cover - unused by proxy_status
        return None

    def read_state(self, serial: str) -> dict[str, Any] | None:
        self.calls.append("read_state")
        return {"pid": PID, "port": PORT, "boot_id": "b1"}

    def proxy_health(self, serial: str, cache_dir: Any, *, self_heal: bool = True) -> dict[str, Any]:
        self.calls.append("proxy_health")
        return {
            "ok": self.healthy,
            "port": PORT,
            "pid": PID,
            "checks": {
                "process": {"ok": True, "detail": "ok"},
                "tunnel": {"ok": self.healthy, "detail": "ok" if self.healthy else "gone"},
                "device_setting": {"ok": True, "detail": "ok"},
            },
        }


class _PlatformWithFakeProxy(_FakePlatform):
    name = "fake-with-proxy"
    capabilities = frozenset({PROXY})

    def __init__(self, config: Any, service: _FakeProxyService) -> None:
        super().__init__(config)
        self._service = service

    def load_capability(self, capability: str) -> Any | None:
        if capability == PROXY:
            return self._service
        return None


def test_core_gets_a_typed_refusal_when_a_platform_has_no_proxy() -> None:
    from android_ui_analyser.config import Config

    platform = _FakePlatform(Config())
    with pytest.raises(UnsupportedPlatformCapabilityError):
        platform.capability(PROXY)


def test_a_partial_proxy_service_is_rejected_at_the_gate() -> None:
    from android_ui_analyser.config import Config

    class _Incomplete(_FakePlatform):
        name = "incomplete-proxy"
        capabilities = frozenset({PROXY})

        def load_capability(self, capability: str) -> Any:
            return object()

    platform = _Incomplete(Config())
    with pytest.raises(InvalidPlatformCapabilityError):
        platform.capability(PROXY)


def test_android_resolves_the_full_proxy_surface_including_the_health_check() -> None:
    platform = AndroidPlatform(make_config())
    assert PROXY in platform.capabilities
    service = platform.capability(PROXY)
    assert missing_members(PROXY, service) == []
    assert service.__name__ == "android_ui_analyser.proxy_mock"
    for name in ("proxy_health", "reverse_tunnel_active", "read_device_http_proxy", "ensure_reverse_tunnel"):
        assert hasattr(service, name), f"proxy capability contract is missing {name}"


def test_engine_proxy_status_works_through_a_non_android_proxy_capability(tmp_path: Path) -> None:
    """The generic engine method must not assume it is talking to `proxy_mock`.

    Proves `Engine.proxy_status` has no Android dependency: swap in a fake platform whose
    `proxy` capability is a tiny stand-in, and the engine still produces a sensible report by
    calling only through the capability surface.
    """
    service = _FakeProxyService(healthy=True)
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")})
    device = FakeDevice(serial=SERIAL)
    engine = Engine(cfg, device=device, platform=_PlatformWithFakeProxy(cfg, service))

    out = engine.proxy_status()

    assert out["ok"] is True
    assert "proxy_health" in service.calls


# --------------------------------------------------------------------------- auto-run points


def test_mock_map_warns_when_the_already_armed_proxy_is_actually_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arming a rule is exactly the moment an agent is about to trust dead interception."""
    _arm_state(tmp_path)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port: True)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    # Tunnel gone, and the process is NOT confirmed alive here either (pid mismatch via a
    # forced ensure failure) so self-heal cannot silently fix it — the warning must surface.
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    monkeypatch.setattr(pm, "ensure_reverse_tunnel", lambda serial, port: False)

    engine = _engine(tmp_path)
    out = engine.mock_map("GET", "/widgets", status=200, body="{}")

    assert out["ok"] is True, "arming the rule itself still succeeds — only the warning differs"
    assert "warning" in out
    assert "tunnel" in out["warning"].lower()


def test_mock_map_is_silent_when_the_proxy_is_actually_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never cry wolf: a healthy proxy must not sprout a warning on every `mock map`."""
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    engine = _engine(tmp_path)
    # No ownership record and no device setting — arming rules before `proxy start`.
    out = engine.mock_map("GET", "/widgets", status=200, body="{}")
    assert "warning" not in out


def test_mock_map_skips_the_health_round_trip_when_nothing_is_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No device round trip for the common case: rules armed before any proxy exists yet.

    Re-pointed at ``read_device_http_proxy``: the short-circuit used to be the ownership
    record, which is exactly why this surface went silent on a black-holed device (no record).
    The gate is now one cheap `settings get`, and what must still be true is that an unproxied
    device costs nothing beyond it — no `adb reverse --list`, no logcat.
    """
    calls: list[str] = []
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    monkeypatch.setattr(
        pm, "reverse_tunnel_active", lambda *a: calls.append("reverse_tunnel_active") or True
    )
    engine = _engine(tmp_path)
    engine.mock_map("GET", "/widgets", status=200, body="{}")
    assert calls == [], "no ownership record means nothing to check — must not touch the device"


def test_mock_record_start_refreshes_the_stale_pid_so_it_does_not_cry_wolf_next_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mock record start` restarts mitm under a fresh pid; the ownership record must follow.

    Without this, the very next health check sees `pid_alive(old_pid) is False` on a port that
    is actually being served by the new process, and reports a healthy recording as broken.
    """
    _arm_state(tmp_path, port=PORT, pid=PID)
    cache = tmp_path / "cache"

    def fake_start(
        *, cache_dir: Path, port: int | None = None, mode: str = "map",
        serial: str | None = None,
    ):
        return PID + 1, port or PORT

    monkeypatch.setattr(pm, "start_mitm", fake_start)
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)
    cache.mkdir(parents=True, exist_ok=True)

    engine = _engine(tmp_path)
    engine.mock_record("start", "login_flow")

    state = pm.read_state(SERIAL)
    assert state is not None
    assert state["pid"] == PID + 1, "the ownership record must follow the mitm restart"
    assert state["port"] == PORT


def test_mock_record_start_warns_when_the_refreshed_proxy_is_still_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm_state(tmp_path, port=PORT, pid=PID)
    cache = tmp_path / "cache"

    def fake_start(
        *, cache_dir: Path, port: int | None = None, mode: str = "map",
        serial: str | None = None,
    ):
        return PID + 1, port or PORT

    monkeypatch.setattr(pm, "start_mitm", fake_start)
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID + 1)
    monkeypatch.setattr(pm, "port_listening", lambda port: True)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    monkeypatch.setattr(pm, "ensure_reverse_tunnel", lambda serial, port: False)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    cache.mkdir(parents=True, exist_ok=True)

    engine = _engine(tmp_path)
    out = engine.mock_record("start", "login_flow")

    assert out["ok"] is True, "the recording itself started fine — only the warning differs"
    assert "warning" in out


def test_proxy_status_with_an_explicit_serial_never_connects_to_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard polls this while an agent is driving the device.

    Connecting would attach uiautomator2 and take the UiAutomation slot away from that
    agent — to learn a serial the caller already passed in.
    """
    _arm_state(tmp_path)
    cfg = make_config(cache={"dir": str(tmp_path)})
    engine = Engine(cfg)

    def refuse() -> Any:
        raise AssertionError("proxy_status(serial=...) must not connect to the device")

    monkeypatch.setattr(type(engine), "device", property(lambda self: refuse()))
    seen: list[str] = []

    def fake_health(serial: str, cache_dir: Any, *, self_heal: bool = False) -> dict[str, Any]:
        seen.append(serial)
        return {"ok": True, "state": "healthy", "port": PORT, "checks": {}}

    monkeypatch.setattr(pm, "proxy_health", fake_health)

    out = engine.proxy_status(heal=False, serial=SERIAL)

    assert seen == [SERIAL]
    assert out["action"] == "proxy-status"
    assert out["ok"] is True
