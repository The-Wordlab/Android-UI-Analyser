"""``aua app launch --arg`` hands the app its process arguments through the adapter.

A test build reads its launch flags from argv (`--uitesting`, feature-flag overrides), so
without this the agent fell back to the platform tool by hand and then slept a guessed
number of seconds before its first analyze. The engine passes the list through untouched;
what a platform does with it is the platform's contract — an iOS simulator forwards it,
Android refuses it because its apps have no process arguments.
"""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config


def _engine(tmp_path, device: FakeDevice) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def test_launch_arguments_are_passed_to_the_runtime_in_order(tmp_path) -> None:
    device = FakeDevice(package="com.example.app")
    engine = _engine(tmp_path, device)

    engine.app("launch", package="com.example.app", observe=False, arguments=("--a", "--b:1"))

    assert ("launch_arguments", ("--a", "--b:1")) in device.calls


def test_a_launch_without_arguments_records_none(tmp_path) -> None:
    device = FakeDevice(package="com.example.app")
    engine = _engine(tmp_path, device)

    engine.app("launch", package="com.example.app", observe=False)

    assert not [call for call in device.calls if call[0] == "launch_arguments"]
