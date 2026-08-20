"""Two agents, two devices, two proxies — and none of it shared.

Measured on a real two-emulator run: both mitmdumps read the same `mock_rules.json` and
appended to the same `flow_log.jsonl`. One agent's rewrite rule fired on the other agent's
device and silently corrupted its observation, `mock clear` wiped rules the other agent had
armed, and `/api/proxy?serial=` returned byte-identical traffic for both — so the panel
reported one device's requests as the other's.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser import device_ledger
from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config

A = "emulator-5580"
B = "emulator-5584"


def _engine(tmp_path: Path, serial: str) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")}
    )
    return Engine(cfg, device=FakeDevice(serial=serial))


def test_two_targets_keep_separate_rule_sets(tmp_path: Path) -> None:
    a, b = _engine(tmp_path, A), _engine(tmp_path, B)

    a.mock_map("GET", "/only-on-a", host="api.example.test")
    b.mock_rewrite("GET", "/only-on-b", host="api.example.test", status=503)

    a_paths = [r.get("request", r.get("match", {})).get("path") for r in a.mock_list()["rules"]]
    b_paths = [r.get("request", r.get("match", {})).get("path") for r in b.mock_list()["rules"]]
    assert a_paths == ["/only-on-a"]
    assert b_paths == ["/only-on-b"]

    # And clearing one leaves the other armed — this is what used to wipe another agent.
    assert a.mock_clear()["removed"] == 1
    assert a.mock_list()["count"] == 0
    assert b.mock_list()["count"] == 1


def test_the_addon_for_one_target_is_pointed_at_that_target_s_rules(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    assert pm.rules_path(cache, A) != pm.rules_path(cache, B)
    assert pm.flow_log_path(cache, A) != pm.flow_log_path(cache, B)
    assert pm.flow_bodies_path(cache, A) != pm.flow_bodies_path(cache, B)
    assert pm.record_path(cache, A) != pm.record_path(cache, B)


def test_traffic_is_read_back_per_target(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    import json
    import time

    now = time.time()
    for serial, path in ((A, "/from-a"), (B, "/from-b")):
        log = pm.flow_log_path(cache, serial)
        log.write_text(
            json.dumps({"n": 1, "ts": now, "method": "GET", "path": path, "status": 200}) + "\n",
            encoding="utf-8",
        )
    assert [f["path"] for f in pm.read_flows_since(cache, 0, A)] == ["/from-a"]
    assert [f["path"] for f in pm.read_flows_since(cache, 0, B)] == ["/from-b"]


def test_a_reader_still_sees_a_proxy_started_before_the_split(tmp_path: Path) -> None:
    """An already-running mitmdump is still writing the unscoped file."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    import json
    import time

    pm.flow_log_path(cache).write_text(
        json.dumps({"n": 1, "ts": time.time(), "path": "/legacy", "status": 200}) + "\n",
        encoding="utf-8",
    )
    assert [f["path"] for f in pm.read_flows_since(cache, 0, A)] == ["/legacy"]


def test_the_undo_clears_the_rules_it_actually_armed(tmp_path: Path) -> None:
    """The reaper replays this for a device this process never touches again.

    Scoping the rules without scoping their undo would have left every dashboard-armed
    rule permanently un-retractable while the undo reported success.
    """
    from android_ui_analyser.device_ledger import UNDO_OPS, UndoContext

    a, b = _engine(tmp_path, A), _engine(tmp_path, B)
    a.mock_map("GET", "/only-on-a", host="api.example.test")
    b.mock_map("GET", "/only-on-b", host="api.example.test")

    entry = next(e for e in device_ledger.read_ledger(A) if e.kind == "mock_rules")
    ctx = UndoContext(serial=A, capability=a.platform.capability)
    UNDO_OPS[entry.op].handler(ctx, entry.args)

    assert a.mock_list()["count"] == 0
    assert b.mock_list()["count"] == 1, "the undo reached across to another device"


def test_mock_list_says_which_rules_actually_fired(tmp_path: Path) -> None:
    """A CLI-only caller could not tell an armed rule from one that never matched."""
    import json
    import time

    a = _engine(tmp_path, A)
    rule = a.mock_map("GET", "/hit", host="api.example.test")["rule"]
    a.mock_map("GET", "/never", host="api.example.test")

    cache = Path(a.config.cache.dir)
    pm.flow_log_path(cache, A).write_text(
        json.dumps(
            {"n": 1, "ts": time.time(), "path": "/hit", "status": 200,
             "action": "stub", "rule": rule["id"]}
        )
        + "\n",
        encoding="utf-8",
    )
    by_path = {
        r.get("request", r.get("match", {})).get("path"): r["fired"] for r in a.mock_list()["rules"]
    }
    assert by_path == {"/hit": 1, "/never": 0}
