"""A forbidden package on a device the run booted for itself is removed, not a reason to stop.

Seen live: the preferred emulator was leased elsewhere, so session start booted a spare one --
and that AVD still had the production build from an earlier use. The first-row rule "the
production package must not be installed" then ended the row as QA_ERROR after 32 seconds,
stopping the emulator it had just brought up. The rule protects a *shared* device from being
tested with a stray sibling app present; a device this run provisioned is its own to clean.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_aua_controller_realapp import FakeAua, FakeModel, model_call, run, verdict  # noqa: E402


class DirtyAua(FakeAua):
    """The forbidden app is installed until it is uninstalled; the target may be a fresh boot."""

    def __init__(self, *, provisioned: bool):
        super().__init__()
        self.provisioned = provisioned
        self.installed = True

    async def call_tool(self, name, arguments):
        if name == "session_start":
            start = await super().call_tool(name, arguments)
            start["virtual_target_started"] = self.provisioned
            return start
        if name == "app_status":
            self.calls.append((name, copy.deepcopy(arguments)))
            return {"ok": True, "installed": self.installed}
        if name == "app" and arguments.get("action") == "uninstall":
            self.calls.append((name, copy.deepcopy(arguments)))
            self.installed = False
            return {"ok": True}
        return await super().call_tool(name, arguments)


def model():
    return FakeModel(controller=[model_call("tap_and_analyze", {"id": "el:fp-home-1"})],
                     judgements={"record_verdict": [verdict("pass", "goal state is visible")]})


def test_the_run_removes_the_forbidden_app_from_a_target_it_booted(tmp_path):
    aua = DirtyAua(provisioned=True)

    result = run(tmp_path, aua, model(), max_steps=1, forbidden_packages=["com.example.production"])

    assert result["error"] is None and result["verdict"]["verdict"] in ("pass", "pass_with_warning")
    assert ("app", {"action": "uninstall", "package": "com.example.production", "confirmed": True}) in aua.calls
    facts = " ".join(result["setup_facts"])
    assert "removed" in facts and "com.example.production" in facts


def test_a_shared_target_with_the_forbidden_app_still_stops_the_run(tmp_path):
    aua = DirtyAua(provisioned=False)

    result = run(tmp_path, aua, model(), max_steps=1, forbidden_packages=["com.example.production"])

    assert "forbidden package is installed" in (result["error"] or "")
    assert all(name != "app" for name, _ in aua.calls), "another agent's device is not ours to clean"
