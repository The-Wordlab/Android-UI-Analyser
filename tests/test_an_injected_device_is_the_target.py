"""An engine handed a device must not go looking for a different one.

`_leased_serial` exists so the helper offload can learn its target without paying to connect.
It asked the lease for that serial unconditionally, and leasing answers "whichever compatible
device is attached to this host" — so an engine explicitly constructed around one device
reported another one's serial, and the offload would hand a whole flow to the wrong target.

It also made a feature's tests pass only on a machine with a device plugged in: with none
attached the lease answers None, the offload declines, and nine tests that assert it runs failed
in CI while passing on the developer's laptop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config


def _engine(tmp_path: Path, device: FakeDevice | None) -> Engine:
    cfg = make_config(cache={"dir": str(tmp_path / "cache")})
    return Engine(cfg, device=device) if device is not None else Engine(cfg)


def test_the_given_device_is_the_target_whatever_is_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path, FakeDevice(serial="fake-emulator-5554"))
    # Stands in for a host with an unrelated device attached: whatever leasing would pick,
    # the device this engine was built around wins.
    monkeypatch.setattr(Engine, "_lease_device", lambda self: "emulator-9999")

    assert engine._leased_serial() == "fake-emulator-5554"


def test_a_target_is_still_leased_when_none_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path is untouched: with no device supplied, leasing still chooses."""
    engine = _engine(tmp_path, None)
    monkeypatch.setattr(Engine, "_lease_device", lambda self: "emulator-9999")

    assert engine._leased_serial() == "emulator-9999"


def test_the_lease_is_not_consulted_at_all_for_a_given_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just the right answer — no lease call, so no adb round trip and no claim to release."""

    def _refuse(self: Engine) -> str | None:
        raise AssertionError("a supplied device needs no lease")

    engine = _engine(tmp_path, FakeDevice(serial="fake-emulator-5554"))
    monkeypatch.setattr(Engine, "_lease_device", _refuse)

    assert engine._leased_serial() == "fake-emulator-5554"
