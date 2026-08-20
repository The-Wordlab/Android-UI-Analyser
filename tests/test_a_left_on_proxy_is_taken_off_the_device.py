"""``proxy start`` must leave a trail any other process can follow, and refuse a foreign one.

The device's ``http_proxy`` is a *device-global* setting pointing at a *non-persistent* host
process. Before this, ``proxy_start`` wrote no ownership record at all and ``_proxy_port`` read
the port from the calling agent's own cache — which parallel agents are told to keep separate.
Two consequences, both silent:

* Agent A crashes; agent B legitimately inherits the emulator. The device still points at A's
  orphan mitmdump. B sees no port, so ``proxy stop`` cannot remove the tunnel it never made,
  and B's own mock rules are read by A's process out of A's cassette dir.
* Agent A runs ``proxy stop`` while B's healthy proxy is live: A clears the device-global
  setting B depends on and B's recording comes out empty while its assertions pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser.errors import UsageError
from conftest import FakeDevice, make_engine


class _FakeProxy:
    """Just enough mitmproxy to exercise the wiring, with a real ownership record."""

    def __init__(self) -> None:
        self.state: dict[str, dict[str, Any]] = {}
        self.stopped = 0
        self.started: list[int] = []
        self.orphan: str | None = None

    # -- ownership -------------------------------------------------------
    def read_state(self, serial: str) -> dict[str, Any] | None:
        return self.state.get(serial)

    def write_state(self, serial: str, state: dict[str, Any]) -> None:
        self.state[serial] = dict(state)

    def clear_state(self, serial: str) -> None:
        self.state.pop(serial, None)

    def orphan_reason(self, state: dict[str, Any], *, boot_id: str | None) -> str | None:
        return self.orphan

    # -- process ---------------------------------------------------------
    def start_mitm(self, *, cache_dir: Path, port: int | None, mode: str) -> tuple[int, int]:
        listen = port or 49097
        self.started.append(listen)
        return 4242, listen

    def stop_mitm(self, cache_dir: Path) -> bool:
        self.stopped += 1
        return True

    def load_listen_port(self, cache_dir: Path) -> int | None:
        return None

    def install_system_ca(self, serial: str) -> dict[str, Any]:
        return {"ok": True}


class _ReversingDevice(FakeDevice):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.reverses: list[tuple[int, int]] = []
        self.removed: list[int] = []

    def instance_token(self) -> str | None:
        return "boot-1"

    def reverse_port(self, device_port: int, host_port: int) -> None:
        self.reverses.append((device_port, host_port))

    def remove_reverse_port(self, device_port: int) -> None:
        self.removed.append(device_port)


def _engine(tmp_path: Path, proxy: _FakeProxy, device: _ReversingDevice):  # noqa: ANN202
    engine = make_engine(device=device, cache={"dir": str(tmp_path / "cache")})
    engine.platform.capabilities = frozenset(  # type: ignore[misc]
        set(engine.platform.capabilities) | {"proxy"}
    )
    real = engine.platform.capability

    def capability(name: str) -> Any:
        return proxy if name == "proxy" else real(name)

    engine.platform.capability = capability  # type: ignore[method-assign]
    return engine


def test_starting_a_proxy_journals_how_to_take_it_off_again(tmp_path: Path) -> None:
    device = _ReversingDevice(serial="emulator-5554")
    proxy = _FakeProxy()
    engine = _engine(tmp_path, proxy, device)

    result = engine.proxy_start(install_ca=False)

    assert result["ok"] and result["port"] == 49097
    kinds = {e.kind for e in device_ledger.read_ledger("emulator-5554")}
    assert kinds == {
        "http_proxy",
        "reverse_port",
        "host_proxy_process",
        "proxy_ownership",
    }, (
        "a crash right now must leave enough on disk for a stranger to un-proxy this device"
    )
    # And the ownership record a parallel agent reads, at the shared path.
    assert proxy.state["emulator-5554"]["port"] == 49097
    assert proxy.state["emulator-5554"]["boot_id"] == "boot-1"


def test_the_undo_is_journalled_before_the_device_is_touched(tmp_path: Path) -> None:
    """The one ordering that matters: a crash between the two must be recoverable."""
    device = _ReversingDevice(serial="emulator-5554")
    proxy = _FakeProxy()
    engine = _engine(tmp_path, proxy, device)

    seen_when_device_changed: list[str] = []

    def exploding_set_http_proxy(host_port: str | None) -> None:
        seen_when_device_changed.extend(
            e.kind for e in device_ledger.read_ledger("emulator-5554")
        )
        raise RuntimeError("agent SIGKILLed here")

    device.set_http_proxy = exploding_set_http_proxy  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        engine.proxy_start(install_ca=False)

    assert "http_proxy" in seen_when_device_changed, (
        "the record must already exist when the device call runs, or a crash there is "
        "unrecoverable — no watchdog, however alive, can undo what nothing wrote down"
    )


def test_stopping_a_proxy_removes_the_tunnel_and_clears_the_record(tmp_path: Path) -> None:
    device = _ReversingDevice(serial="emulator-5554")
    proxy = _FakeProxy()
    engine = _engine(tmp_path, proxy, device)
    engine.proxy_start(install_ca=False)

    result = engine.proxy_stop()

    assert result["port"] == 49097, "the port came from the shared record, not a private cache"
    assert device.removed == [49097]
    assert device_ledger.read_ledger("emulator-5554") == [], (
        "a pending undo after a deliberate stop is a promise to un-point a later proxy"
    )
    assert "emulator-5554" not in proxy.state


def test_a_healthy_foreign_proxy_is_not_overwritten(tmp_path: Path) -> None:
    device = _ReversingDevice(serial="emulator-5554")
    proxy = _FakeProxy()
    proxy.state["emulator-5554"] = {"pid": 999, "port": 40001, "owner": "cursor-77-x"}
    proxy.orphan = None  # positive evidence of health
    engine = _engine(tmp_path, proxy, device)

    with pytest.raises(UsageError) as caught:
        engine.proxy_start(install_ca=False)

    assert "cursor-77-x" in str(caught.value)
    assert "aua teardown run --serial-target emulator-5554 --force" in (caught.value.hint or "")
    assert proxy.started == [], "the second proxy must not launch"
    assert proxy.state["emulator-5554"]["port"] == 40001, "the live owner keeps its port"


def test_a_dead_foreign_proxy_is_reaped_before_a_new_one_starts(tmp_path: Path) -> None:
    device = _ReversingDevice(serial="emulator-5554")
    proxy = _FakeProxy()
    proxy.state["emulator-5554"] = {"pid": 999, "port": 40001, "owner": "cursor-77-x"}
    proxy.orphan = "its mitmdump (pid 999) is gone"
    engine = _engine(tmp_path, proxy, device)

    result = engine.proxy_start(install_ca=False)

    assert result["ok"] and proxy.started == [49097]
    assert proxy.state["emulator-5554"]["port"] == 49097, "ownership moved to this agent"
