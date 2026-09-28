"""A session need must reach the boot, however it arrived.

Seen live on 2026-09-28. A harness asked for `--needs audio` without `--audio`: session start
turns the `--audio` flag into a need, but not a need into the flag, so it provisioned a headless
emulator with `-no-audio` that could never satisfy the session, reported "free and matching:
none", and the caller waited ten minutes for a device that was never coming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.virtual_targets import (
    OwnedVirtualTargetStopRequest,
    VirtualTargetInstance,
    VirtualTargetProvisionRequest,
    VirtualTargetStopResult,
)
from android_ui_analyser.schema import DeviceInfo
from conftest import make_config


def engine_that_boots(tmp_path: Path, monkeypatch: Any, needs: list[str]):
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(tmp_path / "coordination")},
    )
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    engine._lease_needs = list(needs)
    online: list[DeviceInfo] = []
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    monkeypatch.setattr(engine.platform, "probe_target_capabilities",
                        lambda _serial: {"audio": True, "headed": True})
    requests: list[VirtualTargetProvisionRequest] = []

    class VirtualDevices:
        def provision_virtual_target(self, request: VirtualTargetProvisionRequest) -> VirtualTargetInstance:
            requests.append(request)
            online.append(DeviceInfo(serial="emulator-5556", model="fresh", android_version="14"))
            return VirtualTargetInstance(target_id="emulator-5556", definition_id="phone",
                                         instance_token="phone.p5556")

        def stop_virtual_target_instance(self, request: OwnedVirtualTargetStopRequest) -> VirtualTargetStopResult:
            return VirtualTargetStopResult(stopped_target_ids=("emulator-5556",))

    monkeypatch.setattr(engine.platform, "capability",
                        lambda name: VirtualDevices() if name == "virtual_targets" else None)
    return engine, requests


@pytest.mark.parametrize(("needs", "audio", "headless"), [
    (["audio"], True, True),
    (["headed"], False, False),
    ([], False, True),
])
def test_a_need_given_only_as_a_need_shapes_the_boot(tmp_path, monkeypatch, needs, audio, headless) -> None:
    engine, requests = engine_that_boots(tmp_path, monkeypatch, needs)

    engine._prepare_session_target(wait_for_lease_s=0, start_emulator=True, headed=False, audio=False)

    (request,) = requests
    assert (request.audio, request.headless) == (audio, headless)
