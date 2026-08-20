"""A device pointed at a proxy nobody owns must be diagnosable — never "nothing is armed".

Measured 2026-08-19/20 on a real emulator: the ownership record under
``~/.cache/android-ui-analyser/proxy/<serial>.json`` was gone (a partial teardown, a reaped
watchdog, or the ``contextlib.suppress(Exception)`` around ``proxy_start``'s own
``write_state``), the ``adb reverse`` tunnel with it — but ``settings get global http_proxy``
still read ``127.0.0.1:<port>``. Every app network call on that device then fails with
``java.net.ConnectException: Failed to connect to /127.0.0.1:<port>``, visible only in logcat:
buttons do nothing, screens stay empty, logins never complete.

``proxy_health`` answered that state with ``{"ok": false, "armed": false, "checks": {}}`` and
the words "no aua owns a proxy for this device" — which reads as *nothing is configured*, the
exact opposite of the truth, and is indistinguishable from a genuinely clean device. Everything
needed to tell those two apart is ownership-free: the device's own setting, ``adb reverse
--list``, and a TCP connect to the host port.

These tests pin the five resting states (``unproxied`` / ``healthy`` / ``degraded`` /
``foreign`` / ``blackholed``), the split between "this device's network path is sane" (``ok``)
and "traffic reaches a proxy this aua owns" (``intercepting``), and the one narrow case where an
unowned proxy may be adopted: when this session's own mitm sidecars prove it is ours.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config

SERIAL = "fake-emulator-5554"
PORT = 44794
PID = 4242


def _engine(tmp_path: Path, *, device: FakeDevice | None = None) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")}
    )
    return Engine(cfg, device=device or FakeDevice(serial=SERIAL))


def _unowned_target(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw: str = f"127.0.0.1:{PORT}",
    tunnel: bool = False,
    listening: bool = False,
) -> None:
    """A device pointed at *raw* with NO ownership record — the orphan state, from fakes."""
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: raw)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: tunnel)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: listening)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: False)
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)


# --------------------------------------------------------------- the three different answers


def test_a_blackholed_device_is_named_loudly_instead_of_reported_as_nothing_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported bug, verbatim: proxied, unowned, unreachable — and previously invisible."""
    _unowned_target(monkeypatch)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "blackholed"
    assert report["ok"] is False
    assert report["owned"] is False
    assert report["intercepting"] is False
    assert report["target"] == {
        "raw": f"127.0.0.1:{PORT}",
        "host": "127.0.0.1",
        "port": PORT,
        "kind": "loopback",
    }
    hint = report["hint"]
    assert "BLACK HOLE" in hint
    assert f"127.0.0.1:{PORT}" in hint
    assert "ConnectException" in hint, "name the exception an agent will actually see in logcat"
    assert "proxy stop" in hint, "the hint must name the exact command that fixes it"
    # Both ownership-free checks were actually performed and both are red.
    assert report["checks"]["tunnel"]["ok"] is False
    assert report["checks"]["listener"]["ok"] is False
    # No pid we can vouch for, so no process check may be fabricated.
    assert "process" not in report["checks"]


def test_a_reachable_unowned_proxy_is_foreign_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never cry wolf: another agent's healthy proxy is a working device, plus a caveat."""
    _unowned_target(monkeypatch, raw="127.0.0.1:8081", tunnel=True, listening=True)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "foreign"
    assert report["ok"] is True, "traffic flows, so the device is usable — not a failure"
    assert report["owned"] is False
    assert report["intercepting"] is False, "our mock rules are NOT applied to a foreign proxy"
    warning = report["warning"]
    assert "no aua owns it" in warning or "which no aua owns" in warning
    assert "mock" in warning.lower(), "say that this session's rules are inert"
    assert "do NOT take it over" in warning
    assert "hint" not in report, "nothing is broken, so nothing to fix"


def test_a_device_with_no_proxy_at_all_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean device is the common case and must be silent, and distinguishable."""
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "unproxied"
    assert report["ok"] is True
    assert report["owned"] is False
    assert report["intercepting"] is False
    assert report["target"] is None
    assert report["checks"] == {}
    assert "hint" not in report and "warning" not in report


def test_a_healthy_owned_proxy_reports_owned_and_intercepting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pm.write_state(SERIAL, {"pid": PID, "port": PORT, "boot_id": "boot-1"})
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: port == PORT)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: True)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "healthy"
    assert report["ok"] is True
    assert report["owned"] is True
    assert report["intercepting"] is True
    assert report["checks"]["listener"]["ok"] is True


def test_a_broken_owned_proxy_is_degraded_and_names_the_owner_only_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pm.write_state(SERIAL, {"pid": PID, "port": PORT, "boot_id": "boot-1"})
    monkeypatch.setattr(pm, "pid_alive", lambda pid: False)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: False)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "degraded"
    assert report["ok"] is False
    assert report["owned"] is True
    assert report["intercepting"] is False
    assert "interception is NOT actually working end to end" in report["hint"]
    assert "proxy stop" in report["hint"], "an owner can rebuild it — say how"


# --------------------------------------------------------------- host classification


def test_an_emulator_host_alias_target_never_fabricates_a_tunnel_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``10.0.2.2`` reaches the host without any ``adb reverse`` — asserting one is a lie."""
    _unowned_target(monkeypatch, raw="10.0.2.2:8081", tunnel=False, listening=True)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["target"]["kind"] == "emulator_host"
    assert "tunnel" not in report["checks"], (
        "the emulator host alias needs no reverse tunnel; a red tunnel check here is fabricated"
    )
    assert report["checks"]["listener"]["ok"] is True
    assert report["state"] == "foreign"
    assert report["ok"] is True


def test_an_unreachable_emulator_host_alias_is_still_a_black_hole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _unowned_target(monkeypatch, raw="10.0.2.2:8081", tunnel=False, listening=False)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "blackholed"
    assert report["ok"] is False
    assert "tunnel" not in report["checks"]


def test_an_off_host_proxy_target_is_reported_as_unknowable_not_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither check applies to an address off this host — say so rather than guessing."""
    _unowned_target(monkeypatch, raw="10.1.2.3:3128", tunnel=False, listening=False)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["target"]["kind"] == "external"
    assert report["checks"] == {}
    assert report["state"] == "unknown"
    assert report["ok"] is False
    assert "cannot tell from here" in report["detail"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("127.0.0.1:44794", ("127.0.0.1", 44794, "loopback")),
        ("localhost:8080", ("localhost", 8080, "loopback")),
        ("127.0.0.5:9", ("127.0.0.5", 9, "loopback")),
        ("[::1]:3128", ("::1", 3128, "loopback")),
        ("10.0.2.2:8081", ("10.0.2.2", 8081, "emulator_host")),
        ("proxy.example.com:3128", ("proxy.example.com", 3128, "external")),
        ("http://127.0.0.1:8888", ("127.0.0.1", 8888, "loopback")),
    ],
)
def test_the_target_parser_handles_the_shapes_android_stores(
    raw: str, expected: tuple[str, int, str]
) -> None:
    target = pm.parse_proxy_target(raw)
    assert target is not None
    assert (target["host"], target["port"], target["kind"]) == expected
    assert target["raw"] == raw


@pytest.mark.parametrize("raw", [None, "", "  ", "null", ":0"])
def test_the_target_parser_treats_androids_empty_shapes_as_no_proxy(raw: str | None) -> None:
    assert pm.parse_proxy_target(raw) is None


def test_the_target_parser_preserves_a_malformed_nonempty_setting_as_unknown() -> None:
    target = pm.parse_proxy_target("127.0.0.1:notaport")
    assert target is not None
    assert target["kind"] == "invalid"


def test_the_target_parser_rejects_a_port_outside_the_tcp_range() -> None:
    target = pm.parse_proxy_target("127.0.0.1:65536")
    assert target is not None
    assert target["kind"] == "invalid"


# --------------------------------------------------------------- logcat corroboration


def test_logcat_corroboration_is_appended_when_apps_are_actually_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _unowned_target(monkeypatch)
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 7)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert "Confirmed in logcat: 7" in report["hint"]


def test_absence_of_logcat_evidence_never_softens_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No traffic attempted yet is not health — the setting is still pointed at nothing."""
    _unowned_target(monkeypatch)
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["state"] == "blackholed"
    assert report["ok"] is False
    assert "Confirmed in logcat" not in report["hint"]


def test_the_logcat_scan_counts_connect_failures_for_this_target_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = "\n".join(
        [
            "08-20 01:02:03.100  1234  1234 W System.err: java.net.ConnectException: "
            "Failed to connect to /127.0.0.1:44794",
            "08-20 01:02:03.200  1234  1234 W System.err: java.net.ConnectException: "
            "Failed to connect to /127.0.0.1:44794",
            # A different port — a different problem, must not be counted here.
            "08-20 01:02:04.000  1234  1234 W System.err: java.net.ConnectException: "
            "Failed to connect to /127.0.0.1:9999",
            "08-20 01:02:05.000  1234  1234 I Something: unrelated line",
        ]
    )

    def fake_adb(serial: str, *args: str, check: bool = True, timeout: float = 60):
        assert "logcat" in args
        return subprocess.CompletedProcess(args, 0, stdout=log, stderr="")

    monkeypatch.setattr(pm, "_adb", fake_adb)

    assert pm.connect_failures_in_logcat(SERIAL, "127.0.0.1", 44794) == 2
    assert pm.connect_failures_in_logcat(SERIAL, "127.0.0.1", 12345) == 0


# --------------------------------------------------------------- stale record vs the device


def test_a_record_naming_one_port_while_the_device_points_at_another_follows_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The device is ground truth for what it is pointed at; the record is then stale."""
    pm.write_state(SERIAL, {"pid": PID, "port": PORT, "boot_id": "boot-1"})
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: False)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: "127.0.0.1:55555")
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)

    report = pm.proxy_health(SERIAL, tmp_path, self_heal=False)

    assert report["target"]["port"] == 55555
    assert report["state"] == "blackholed"
    assert report["owned"] is False, "we do not own the thing the device is actually pointed at"
    assert "55555" in report["hint"]
    assert str(PORT) in report["hint"], "name the stale record so someone can clear it"
    assert "stale" in report["hint"]


# --------------------------------------------------------------- adoption on self-proof


def _write_sidecars(cache: Path, *, port: int, pid: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    pm.save_listen_port(cache, port)
    pm.pid_path(cache).write_text(str(pid), encoding="utf-8")


def test_our_own_orphaned_mitm_is_adoptable_via_its_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``proxy_start`` can legitimately leave no record — its ``write_state`` is suppressed."""
    cache = tmp_path / "cache"
    _write_sidecars(cache, port=PORT, pid=PID)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: True)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)

    report = pm.proxy_health(SERIAL, cache, self_heal=False)

    assert report["adoptable"] is True
    assert report["adoptable_pid"] == PID
    assert "own" in report["detail"], "say whose proxy this actually is"


def test_a_stale_sidecar_naming_a_dead_pid_is_not_adoptable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This host really has ``mitmproxy.pid`` files naming long-dead processes."""
    cache = tmp_path / "cache"
    _write_sidecars(cache, port=PORT, pid=PID)
    _unowned_target(monkeypatch, tunnel=True, listening=True)  # pid_alive -> False

    report = pm.proxy_health(SERIAL, cache, self_heal=False)

    assert report.get("adoptable") is not True
    assert report["state"] == "foreign"


def test_a_sidecar_for_a_different_port_is_not_self_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    _write_sidecars(cache, port=PORT, pid=PID)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: "127.0.0.1:8081")
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: True)
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: True)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: True)

    report = pm.proxy_health(SERIAL, cache, self_heal=False)

    assert report.get("adoptable") is not True, (
        "a live mitm of ours on another port says nothing about the port the device uses"
    )


def test_engine_proxy_status_adopts_its_own_orphan_and_journals_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    _write_sidecars(cache, port=PORT, pid=PID)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: True)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)
    tunnel = {"up": False}
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: tunnel["up"])

    order: list[str] = []

    real_write_state = pm.write_state

    def spying_write_state(serial: str, state: dict[str, Any]) -> None:
        order.append("wrote_ownership")
        real_write_state(serial, state)

    def fake_ensure(serial: str, port: int) -> bool:
        order.append("touched_device")
        tunnel["up"] = True
        return True

    monkeypatch.setattr(pm, "write_state", spying_write_state)
    monkeypatch.setattr(pm, "ensure_reverse_tunnel", fake_ensure)

    from android_ui_analyser import device_ledger as ledger_mod

    real_record = ledger_mod.record

    def spying_record(*args: Any, **kwargs: Any) -> None:
        order.append(f"recorded:{kwargs.get('kind')}")
        real_record(*args, **kwargs)

    monkeypatch.setattr(ledger_mod, "record", spying_record)

    engine = _engine(tmp_path)
    out = engine.proxy_status()

    assert out["adopted"] is True
    assert out["state"] == "healthy"
    assert out["owned"] is True
    assert out["intercepting"] is True
    assert "mitmproxy.port" in out["warning"]
    assert order == [
        "recorded:proxy_ownership",
        "wrote_ownership",
        "recorded:reverse_port",
        "touched_device",
    ], f"write-ahead ordering violated: {order}"
    # The record we rebuilt must be readable by any other process.
    state = pm.read_state(SERIAL)
    assert state is not None and state["port"] == PORT and state["pid"] == PID
    kinds = {e.kind for e in device_ledger.read_ledger(SERIAL)}
    assert {"proxy_ownership", "reverse_port"} <= kinds


def test_engine_proxy_status_never_heals_a_proxy_it_cannot_prove_is_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-``adb reverse`` to a stranger's port routes real traffic into an unknown process."""
    _unowned_target(monkeypatch)
    ensure_calls: list[Any] = []
    monkeypatch.setattr(pm, "ensure_reverse_tunnel", lambda *a: ensure_calls.append(a))
    writes: list[Any] = []
    monkeypatch.setattr(pm, "write_state", lambda s, st: writes.append((s, st)))

    engine = _engine(tmp_path)
    out = engine.proxy_status(heal=True)

    assert out["state"] == "blackholed"
    assert out["action"] == "proxy-status"
    assert ensure_calls == [], "never tunnel to a port we cannot prove is ours"
    assert writes == [], "never fabricate ownership from a port that merely answers TCP"
    assert device_ledger.read_ledger(SERIAL) == []


def test_engine_proxy_status_does_not_adopt_when_healing_is_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    _write_sidecars(cache, port=PORT, pid=PID)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: True)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: True)
    writes: list[Any] = []
    monkeypatch.setattr(pm, "write_state", lambda s, st: writes.append((s, st)))

    engine = _engine(tmp_path)
    out = engine.proxy_status(heal=False)

    assert writes == []
    assert out.get("adopted") is not True
    assert out["adoptable"] is True, "still say it is adoptable, just do not do it"


# --------------------------------------------------------------- _proxy_port / proxy stop


def test_proxy_port_does_not_claim_an_unowned_device_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setting alone cannot prove which host-side tunnel belongs to this session."""
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    engine = _engine(tmp_path)

    assert engine._proxy_port() is None


def test_proxy_port_prefers_the_shared_record_over_the_device_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pm.write_state(SERIAL, {"pid": PID, "port": 40001})
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    engine = _engine(tmp_path)

    assert engine._proxy_port() == 40001


def test_proxy_stop_does_not_remove_an_unowned_tunnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "stop_mitm", lambda cache: False)

    removed: list[int] = []
    device = FakeDevice(serial=SERIAL)
    device.remove_reverse_port = removed.append  # type: ignore[method-assign]
    device.set_http_proxy = lambda hp: None  # type: ignore[method-assign]

    engine = _engine(tmp_path, device=device)
    out = engine.proxy_stop()

    assert out["port"] is None
    assert removed == [], "a device setting does not prove ownership of a host tunnel"


# --------------------------------------------------------------- the other warning surfaces


def test_mock_map_warns_on_a_blackholed_device_with_no_ownership_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record-based short-circuit went silent exactly when the device was black-holed."""
    _unowned_target(monkeypatch)
    engine = _engine(tmp_path)

    out = engine.mock_map("GET", "/widgets", status=200, body="{}")

    assert out["ok"] is True
    assert "warning" in out
    assert "BLACK HOLE" in out["warning"]


def test_mock_map_warns_that_a_foreign_proxy_will_not_read_these_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _unowned_target(monkeypatch, raw="127.0.0.1:8081", tunnel=True, listening=True)
    engine = _engine(tmp_path)

    out = engine.mock_map("GET", "/widgets", status=200, body="{}")

    assert "warning" in out
    assert "mock" in out["warning"].lower()


def test_mock_map_stays_silent_on_a_device_with_no_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    calls: list[str] = []
    monkeypatch.setattr(
        pm, "reverse_tunnel_active", lambda *a: calls.append("tunnel") or True
    )
    engine = _engine(tmp_path)

    out = engine.mock_map("GET", "/widgets", status=200, body="{}")

    assert "warning" not in out, "arming rules before `proxy start` is normal — never cry wolf"
    assert calls == [], "an unproxied device needs no tunnel round trip"


# --------------------------------------------------------------- doctor


def test_doctor_names_a_blackholed_serial_while_staying_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent inheriting a device runs `aua doctor`, not `aua proxy status --serial X`."""
    from android_ui_analyser import cli

    _unowned_target(monkeypatch)
    engine = _engine(tmp_path)

    from android_ui_analyser.schema import DeviceInfo

    monkeypatch.setattr(
        Engine, "list_devices", lambda self: [DeviceInfo(serial=SERIAL, state="device")]
    )

    report = cli._build_doctor_report(engine)

    proxy = report["checks"]["proxy"]
    assert proxy["ok"] is False
    assert SERIAL in str(proxy)
    assert "blackholed" in str(proxy)
    rendered = cli._render_doctor_pretty(report)
    assert "proxy" in rendered
    assert SERIAL in rendered


def test_doctor_does_not_fail_on_a_clean_or_foreign_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import cli
    from android_ui_analyser.schema import DeviceInfo

    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    monkeypatch.setattr(
        Engine, "list_devices", lambda self: [DeviceInfo(serial=SERIAL, state="device")]
    )
    engine = _engine(tmp_path)

    report = cli._build_doctor_report(engine)

    assert report["checks"]["proxy"]["ok"] is True


def test_the_doctor_proxy_survey_never_mutates_a_device_it_was_not_pointed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doctor reports; it does not heal. It sweeps serials nobody asked it to touch."""
    from android_ui_analyser.schema import DeviceInfo

    cache = tmp_path / "cache"
    _write_sidecars(cache, port=PORT, pid=PID)
    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: f"127.0.0.1:{PORT}")
    monkeypatch.setattr(pm, "port_listening", lambda port, **kw: True)
    monkeypatch.setattr(pm, "pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(pm, "reverse_tunnel_active", lambda serial, port, **kw: False)
    monkeypatch.setattr(pm, "connect_failures_in_logcat", lambda *a, **k: 0)
    ensure_calls: list[Any] = []
    writes: list[Any] = []
    monkeypatch.setattr(pm, "ensure_reverse_tunnel", lambda *a: ensure_calls.append(a))
    monkeypatch.setattr(pm, "write_state", lambda s, st: writes.append(s))
    monkeypatch.setattr(
        Engine, "list_devices", lambda self: [DeviceInfo(serial="other-emulator-9999")]
    )

    engine = _engine(tmp_path)
    survey = engine.proxy_survey()

    assert [d["serial"] for d in survey["devices"]] == ["other-emulator-9999"]
    assert ensure_calls == [] and writes == [], "a survey must be read-only"


def test_proxy_survey_needs_no_connected_device_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``self.device`` connects and can raise; the survey is serial-based on purpose."""
    from android_ui_analyser.schema import DeviceInfo

    monkeypatch.setattr(pm, "read_device_http_proxy", lambda serial: None)
    monkeypatch.setattr(
        Engine, "list_devices", lambda self: [DeviceInfo(serial="s1"), DeviceInfo(serial="s2")]
    )
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")}
    )
    engine = Engine(cfg)

    def explode(self: Any) -> Any:  # pragma: no cover - must never be reached
        raise AssertionError("proxy_survey must not connect to a device")

    monkeypatch.setattr(Engine, "device", property(explode))

    survey = engine.proxy_survey()

    assert survey["ok"] is True
    assert [d["serial"] for d in survey["devices"]] == ["s1", "s2"]


# --------------------------------------------------------------- CLI / MCP / guide contract


def test_cli_proxy_status_exits_nonzero_when_the_device_is_black_holed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from android_ui_analyser import cli

    monkeypatch.setattr(
        Engine,
        "proxy_status",
        lambda self, heal=True: {
            "ok": False,
            "action": "proxy-status",
            "state": "blackholed",
            "owned": False,
            "intercepting": False,
        },
    )
    result = CliRunner().invoke(cli.app, ["--format", "compact", "proxy", "status"])

    assert result.exit_code == 1, "a black-holed device must be scriptable as a failure"
    assert "blackholed" in result.stdout


def test_mcp_exposes_proxy_status_so_cli_and_mcp_do_not_diverge() -> None:
    from android_ui_analyser.mcp_server import _tool_definitions

    names = {t.name for t in _tool_definitions()}
    assert "proxy_status" in names, "CLAUDE.md requires CLI and MCP to share the engine surface"


def test_the_guide_tells_agents_that_proxy_status_exists() -> None:
    from android_ui_analyser import guide

    text = guide.render_markdown()
    assert "proxy status" in text
    assert "blackholed" in text or "black hole" in text.lower()
