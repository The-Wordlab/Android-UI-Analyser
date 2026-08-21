"""The dashboard's force-unlease button, and the order it has to do things in.

A lease ages out when its owner process dies, so the case that needs a human is the one the
design cannot self-heal: an owner that is still alive but no longer driving. Breaking it from
the browser must behave exactly like ``aua lease release <serial> --force`` — clean the device
first, drop the entry second — or the next agent inherits somebody else's proxy or clock.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.errors import UsageError


def _state_with_wedged_lease(tmp_path: Path) -> tuple[Any, Path]:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser import leases
    from android_ui_analyser.config import Config

    registry = tmp_path / "lease-registry"
    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    cfg.memory.dir = str(tmp_path)
    # Host-wide by design, and never the per-run cache: the dashboard must break the same
    # registry entry an agent claimed.
    cfg.lease.registry_dir = str(registry)
    state = dash._DashboardState(
        serials=["emulator-5554"],
        focus="emulator-5554",
        mode="detail",
        cache_dir=tmp_path,
        ensures={},
        poll_ms=500,
        config=cfg,
    )
    assert leases.acquire(registry, "emulator-5554", owner="wedged-agent")
    assert leases.read_lease(registry, "emulator-5554") is not None
    return state, registry


class _FakeEngine:
    """Stands in for the host engine so no test ever reaches a real device."""

    def __init__(self, reports: list[dict[str, Any]], *, watcher: Any = None) -> None:
        self._reports = reports
        self._watcher = watcher
        self.calls: list[dict[str, Any]] = []

    def teardown_run(
        self, *, serial: str | None = None, force: bool = False, dry_run: bool = False
    ) -> dict[str, Any]:
        self.calls.append({"serial": serial, "force": force, "dry_run": dry_run})
        if self._watcher is not None:
            self._watcher()
        failed = sum(len(r.get("failed") or ()) for r in self._reports)
        return {"ok": failed == 0, "action": "teardown-run", "reports": self._reports}


def test_dashboard_force_unlease_needs_the_typed_confirmation(tmp_path: Path) -> None:
    from android_ui_analyser import leases

    state, registry = _state_with_wedged_lease(tmp_path)
    state._engine = _FakeEngine([{"serial": "emulator-5554", "skipped": "nothing pending"}])

    with pytest.raises(UsageError, match="FORCE UNLEASE emulator-5554"):
        state.lease_operation("release", {"serial": "emulator-5554"})
    with pytest.raises(UsageError, match="FORCE UNLEASE emulator-5554"):
        state.lease_operation(
            "release", {"serial": "emulator-5554", "confirmation": "FORCE UNLEASE wrong-device"}
        )

    assert state._engine.calls == []
    assert leases.read_lease(registry, "emulator-5554") is not None


def test_dashboard_force_unlease_cleans_the_device_then_drops_the_lease(tmp_path: Path) -> None:
    from android_ui_analyser import leases

    state, registry = _state_with_wedged_lease(tmp_path)
    still_leased_during_teardown: list[bool] = []
    state._engine = _FakeEngine(
        [{"serial": "emulator-5554", "undone": [{"kind": "proxy", "result": "cleared"}]}],
        watcher=lambda: still_leased_during_teardown.append(
            leases.read_lease(registry, "emulator-5554") is not None
        ),
    )

    result = state.lease_operation(
        "release",
        {"serial": "emulator-5554", "confirmation": "FORCE UNLEASE emulator-5554"},
    )

    assert result["ok"] is True
    assert result["action"] == "lease-release"
    assert result["forced"] is True
    assert result["was_held"] is True
    assert result["previous_owner"] == "wedged-agent"
    # Cleaning happens while the device is still quarantined by the lease it is about to lose.
    assert still_leased_during_teardown == [True]
    assert state._engine.calls == [{"serial": "emulator-5554", "force": True, "dry_run": False}]
    assert leases.read_lease(registry, "emulator-5554") is None


def test_dashboard_force_unlease_keeps_the_lease_when_cleanup_fails(tmp_path: Path) -> None:
    from android_ui_analyser import leases

    state, registry = _state_with_wedged_lease(tmp_path)
    state._engine = _FakeEngine(
        [{"serial": "emulator-5554", "failed": [{"kind": "proxy", "error": "adb offline"}]}]
    )

    with pytest.raises(UsageError, match="could not clean emulator-5554"):
        state.lease_operation(
            "release",
            {"serial": "emulator-5554", "confirmation": "FORCE UNLEASE emulator-5554"},
        )

    assert leases.read_lease(registry, "emulator-5554") is not None


def test_dashboard_force_unlease_refuses_a_deferred_undo(tmp_path: Path) -> None:
    from android_ui_analyser import leases

    state, registry = _state_with_wedged_lease(tmp_path)
    # A reap that deferred its device-side undos reports ok, so only the skip reason tells the
    # truth: releasing here would advertise a device that is still mutated.
    state._engine = _FakeEngine(
        [{"serial": "emulator-5554", "skipped": "target unreachable; device-side undos deferred"}]
    )

    with pytest.raises(UsageError, match="could not clean emulator-5554"):
        state.lease_operation(
            "release",
            {"serial": "emulator-5554", "confirmation": "FORCE UNLEASE emulator-5554"},
        )

    assert leases.read_lease(registry, "emulator-5554") is not None


def test_dashboard_force_unlease_rejects_a_device_outside_this_session(tmp_path: Path) -> None:
    state, _registry = _state_with_wedged_lease(tmp_path)
    state._engine = _FakeEngine([{"serial": "emulator-5554", "skipped": "nothing pending"}])

    with pytest.raises(UsageError, match="not part of this dashboard session"):
        state.lease_operation(
            "release",
            {"serial": "emulator-5556", "confirmation": "FORCE UNLEASE emulator-5556"},
        )
    with pytest.raises(UsageError, match="unknown dashboard lease action"):
        state.lease_operation("acquire", {"serial": "emulator-5554"})
    assert state._engine.calls == []


def test_dashboard_force_unlease_endpoint_is_token_protected(tmp_path: Path) -> None:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser import leases

    state, registry = _state_with_wedged_lease(tmp_path)
    state.database_token = "dashboard-test-token"
    state._engine = _FakeEngine([{"serial": "emulator-5554", "skipped": "nothing pending"}])
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/lease/release"
    body = b'{"serial": "emulator-5554", "confirmation": "FORCE UNLEASE emulator-5554"}'
    try:
        unauthorized = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as forbidden:
            urllib.request.urlopen(unauthorized, timeout=2)
        assert forbidden.value.code == 403
        assert leases.read_lease(registry, "emulator-5554") is not None

        authorized = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-AUA-Dashboard-Token": "dashboard-test-token",
            },
        )
        with urllib.request.urlopen(authorized, timeout=2) as response:
            assert response.status == 200
        assert leases.read_lease(registry, "emulator-5554") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_the_lease_pill_stops_saying_held_the_moment_the_lease_is_broken(tmp_path: Path) -> None:
    state, _registry = _state_with_wedged_lease(tmp_path)
    state._engine = _FakeEngine([{"serial": "emulator-5554", "skipped": "nothing pending"}])

    before = state.device_runtime("emulator-5554")
    assert before["lease"]["held"] is True
    assert before["lease"]["owner"] == "wedged-agent"

    state.lease_operation(
        "release",
        {"serial": "emulator-5554", "confirmation": "FORCE UNLEASE emulator-5554"},
    )

    # The runtime payload is cached for a second. Without dropping that entry the browser
    # would poll back "held" right after watching its own release succeed.
    assert state.device_runtime("emulator-5554")["lease"]["held"] is False


def test_the_lease_pill_offers_the_escape_hatch_only_while_a_lease_is_held() -> None:
    from android_ui_analyser import dashboard as dash

    page = dash._DASHBOARD_HTML
    assert 'id="lease-release"' in page
    assert "FORCE UNLEASE " in page
    # Hidden by default and shown by the status tick, so a free device offers no button.
    assert 'class="db-button danger detail-status-action hidden"' in page
    assert "leaseReleaseButton.classList.toggle('hidden', !lease.held);" in page
