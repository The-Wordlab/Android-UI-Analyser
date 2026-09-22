"""The `aua mcp` child must inherit the caller's AUA cache lane and any pinned serial.

The MCP stdio client scrubs a child's environment to a tiny safe allowlist unless it is given one
explicitly. run_live built its StdioServerParameters without an ``env``, so a caller that set
``AUA_CACHE__DIR`` to isolate its run, or ``AUA_SERIAL`` to pin a device, had the server ignore
both: it wrote to the shared default cache and leased or provisioned whatever it wanted. On a busy
host that meant a full emulator provision and a ~240MB fresh install on every run, and collisions
with other agents' devices. ``mcp_server`` forwards AUA_* and the Android/adb pointers instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.run_live import mcp_server


def test_mcp_server_forwards_the_cache_lane_and_pinned_serial(monkeypatch):
    monkeypatch.setenv("AUA_CACHE__DIR", "/tmp/lane-42")
    monkeypatch.setenv("AUA_SERIAL", "emulator-5554")
    monkeypatch.setenv("AUA_OWNER", "worker-7")
    monkeypatch.setenv("ANDROID_SERIAL", "emulator-5554")
    monkeypatch.setenv("ANDROID_ADB_SERVER_PORT", "5037")

    env = mcp_server("aua").env or {}

    assert env["AUA_CACHE__DIR"] == "/tmp/lane-42"
    assert env["AUA_SERIAL"] == "emulator-5554"
    assert env["AUA_OWNER"] == "worker-7"
    assert env["ANDROID_SERIAL"] == "emulator-5554"
    assert env["ANDROID_ADB_SERVER_PORT"] == "5037"
    # The SDK's safe base is still present, so PATH still resolves adb and the aua binary.
    assert env.get("PATH")


def test_mcp_server_does_not_invent_config_the_caller_did_not_set(monkeypatch):
    for name in ("AUA_CACHE__DIR", "AUA_SERIAL", "AUA_OWNER", "ANDROID_SERIAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SOME_UNRELATED_SECRET", "leak-me")

    server = mcp_server("aua")
    env = server.env or {}

    assert "AUA_SERIAL" not in env and "AUA_CACHE__DIR" not in env
    assert "SOME_UNRELATED_SECRET" not in env, "only AUA_/Android pointers cross the boundary"
    assert server.args == ["mcp"] and server.command == "aua"


def test_mcp_server_forwards_the_provider_keys_the_aua_config_names(monkeypatch):
    """A paid AUA provider inside the child (icon names, grounding) reads its key from the
    variable ``models.<name>.api_key_env`` names; by convention those end in ``_API_KEY``.
    Scrubbed away, the provider reports "key not set" and the feature silently does nothing."""
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "not-a-provider-key")

    env = mcp_server("aua").env or {}

    assert env["OPEN_ROUTER_API_KEY"] == "or-secret"
    assert env["GEMINI_API_KEY"] == "gm-secret"
    assert "DATABASE_PASSWORD" not in env
