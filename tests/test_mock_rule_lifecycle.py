"""`aua mock` rule lifecycle: list / clear / rm, ownership warning, and the ledger undo.

Measured on a real cache dir: 14 stale rules from unrelated earlier sessions, `mode: "map"`
armed globally, with no way to see or undo any of it short of hand-editing the JSON. `aua mock
map` happily appended a 15th rule and reported `count: 15`, which reads like success. This file
proves the fix: rules are visible (`mock list`), resettable in one call (`mock clear`),
removable individually even when legacy entries have `id: null` (`mock rm`), inheriting a
foreign session's rules is flagged rather than silent, and arming rules is registered in the
device-change ledger so a crashed agent's mock state does not strand the next one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from conftest import FakeDevice, make_config


def _engine(tmp_path: Path) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")}
    )
    return Engine(cfg, device=FakeDevice())


# --------------------------------------------------------------------------- list / clear / rm


def test_mock_list_shows_mode_and_every_rule_with_a_stable_id(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.mock_map("GET", "/api/v1/widgets", status=200)
    engine.mock_map("POST", "/api/v1/gadgets", status=201)

    out = engine.mock_list()

    assert out["ok"] is True
    assert out["mode"] == "map"
    assert out["count"] == 2
    ids = [r.get("id") for r in out["rules"]]
    assert all(ids) and len(set(ids)) == 2


def test_mock_clear_removes_every_rule_and_disarms_mode(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.mock_map("GET", "/api/v1/widgets")
    cache = Path(engine.config.cache.dir)
    pm.set_mode(pm.rules_path(cache), "record")

    out = engine.mock_clear()

    assert out["ok"] is True
    assert out["removed"] == 1
    doc = pm.load_doc(pm.rules_path(cache))
    assert doc["rules"] == []
    assert doc["mode"] == "map"


def test_mock_rm_removes_one_rule_by_id(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    r1 = engine.mock_map("GET", "/api/v1/widgets")["rule"]
    engine.mock_map("GET", "/api/v1/gadgets")

    out = engine.mock_rm(r1["id"])

    assert out["ok"] is True
    assert out["count"] == 1
    remaining = pm.load_rules(pm.rules_path(Path(engine.config.cache.dir)))
    assert all(r["id"] != r1["id"] for r in remaining)


def test_mock_rm_unknown_id_is_a_clear_usage_error(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.mock_map("GET", "/api/v1/widgets")

    with pytest.raises(UsageError):
        engine.mock_rm("does-not-exist")


def test_mock_rm_copes_with_legacy_rules_that_have_a_null_id(tmp_path: Path) -> None:
    """Several real on-disk rules were found with `id: null` — from before ids were always
    assigned, or from hand-edited cassettes. Removal must be able to target them."""
    engine = _engine(tmp_path)
    cache = Path(engine.config.cache.dir)
    path = pm.rules_path(cache)
    legacy = pm.map_rule("GET", "/api/v1/legacy")
    legacy["id"] = None
    pm.write_rules(path, [legacy])

    listed = engine.mock_list()
    assert listed["count"] == 1
    backfilled_id = listed["rules"][0]["id"]
    assert backfilled_id, "a legacy null id must be backfilled, not left unaddressable"

    out = engine.mock_rm(backfilled_id)

    assert out["count"] == 0


# --------------------------------------------------------------------------- ownership warning


def test_mock_map_warns_when_it_inherits_rules_it_did_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    cache = Path(engine.config.cache.dir)
    # A rule left behind by an unrelated, untagged earlier session — no owner recorded — is
    # exactly the shape of the stale-rules problem this reproduces.
    pm.write_rules(pm.rules_path(cache), [pm.map_rule("GET", "/api/v1/stale")])

    monkeypatch.setenv("AUA_OWNER", "this-session")
    out = engine.mock_map("GET", "/api/v1/widgets")

    assert out.get("warning"), "appending onto a foreign session's rules must not be silent"


def test_mock_map_does_not_warn_within_its_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUA_OWNER", "same-session")
    engine = _engine(tmp_path)

    engine.mock_map("GET", "/api/v1/widgets")
    out = engine.mock_map("GET", "/api/v1/gadgets")

    assert not out.get("warning")


# --------------------------------------------------------------------------- ledger / undo


def test_mock_map_journals_an_undo_before_arming_a_rule(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    engine.mock_map("GET", "/api/v1/widgets")

    kinds = {e.kind for e in device_ledger.read_ledger(engine.device.serial)}
    assert "mock_rules" in kinds, (
        "arming a mock rule is persistent host state that outlives the command and must be "
        "undoable by a stranger — see device_ledger.MUTATION_CATALOGUE"
    )


def test_mock_replay_journals_an_undo(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    cass = pm.cassette_dir(engine.config.memory.dir) / "widgets.yaml"
    pm.save_cassette(cass, "widgets", [pm.map_rule("GET", "/api/v1/widgets")])

    engine.mock_replay("widgets")

    kinds = {e.kind for e in device_ledger.read_ledger(engine.device.serial)}
    assert "mock_rules" in kinds


def test_mock_clear_forgets_the_journalled_undo(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.mock_map("GET", "/api/v1/widgets")
    assert any(
        e.kind == "mock_rules" for e in device_ledger.read_ledger(engine.device.serial)
    )

    engine.mock_clear()

    assert not any(
        e.kind == "mock_rules" for e in device_ledger.read_ledger(engine.device.serial)
    ), "a deliberate clear must retract the pending undo, or a reaper repeats it needlessly"


def test_the_catalogued_mock_rules_undo_actually_clears_the_file(tmp_path: Path) -> None:
    """Exercises the catalogued undo op directly, the way a stranger's reaper would replay it
    on a device this process never touches again."""
    from android_ui_analyser.device_ledger import UNDO_OPS, UndoContext

    engine = _engine(tmp_path)
    engine.mock_map("GET", "/api/v1/widgets")
    cache = Path(engine.config.cache.dir)

    ctx = UndoContext(serial=engine.device.serial, capability=engine.platform.capability)
    entry = next(
        e for e in device_ledger.read_ledger(engine.device.serial) if e.kind == "mock_rules"
    )
    UNDO_OPS[entry.op].handler(ctx, entry.args)

    doc = pm.load_doc(pm.rules_path(cache))
    assert doc["rules"] == []


# --------------------------------------------------------------------- rewrite rules


def test_mock_rewrite_arms_a_response_patch_and_journals_its_undo(tmp_path: Path) -> None:
    """`rewrite_rule` was implemented and addon-tested but had no engine or CLI surface,
    so the whole patch-the-real-response half of the proxy was unreachable to a caller."""
    engine = _engine(tmp_path)

    out = engine.mock_rewrite(
        "GET",
        "/api/v1/widgets",
        host="api.example.test",
        status=429,
        headers={"Retry-After": "30"},
        set_json={"quota.remaining": 0},
        times=1,
    )

    assert out["ok"] is True
    assert out["action"] == "mock-rewrite"
    rule = out["rule"]
    assert rule["action"] == "rewrite"
    assert rule["match"] == {"method": "GET", "path": "/api/v1/widgets", "host": "api.example.test"}
    assert rule["rewrite"]["status"] == 429
    assert rule["rewrite"]["headers"] == {"Retry-After": "30"}
    assert rule["rewrite"]["set_json"] == {"quota.remaining": 0}
    assert rule["times"] == 1

    # Same ledger key as a stub, so one `clear_mock_rules` undo retracts either kind.
    assert any(
        e.kind == "mock_rules" for e in device_ledger.read_ledger(engine.device.serial)
    )


def test_mock_rewrite_and_mock_map_share_one_armed_set(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.mock_map("GET", "/api/v1/widgets", status=200)
    engine.mock_rewrite("POST", "/api/v1/gadgets", status=500)

    listed = engine.mock_list()
    assert listed["count"] == 2
    assert sorted(r["action"] for r in listed["rules"]) == ["rewrite", "stub"]
    assert engine.mock_clear()["removed"] == 2


def test_mock_rewrite_refuses_a_rule_that_matches_every_request(tmp_path: Path) -> None:
    """An unhosted catch-all also intercepts Android's connectivity probes, and the device
    then looks offline rather than mocked."""
    engine = _engine(tmp_path)
    with pytest.raises(UsageError, match="every request"):
        engine.mock_rewrite("*", "*", status=500)
    # Scoping it to a host makes the same rule legitimate.
    assert engine.mock_rewrite("*", "*", host="api.example.test", status=500)["ok"] is True


def test_mock_rewrite_requires_an_actual_change(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(UsageError, match="must change something"):
        engine.mock_rewrite("GET", "/api/v1/widgets")


def test_an_engine_without_a_device_still_journals_the_undo_for_its_target(
    tmp_path: Path,
) -> None:
    """`record_device_change` falls back to this engine's own device and records NOTHING
    when there is none. The dashboard's engine deliberately never connects, so a rule it
    armed used to leave no undo record — unretractable by the reaper that exists for it."""
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")}
    )
    engine = Engine(cfg)  # no device, exactly like the dashboard's

    engine.mock_map("GET", "/api/v1/widgets", host="api.example.test", serial="emulator-5554")
    engine.mock_rewrite("GET", "/api/v1/gadgets", status=429, serial="emulator-5554")

    # One record per key by design — a single `clear_mock_rules` retracts the whole set.
    # What matters is that it exists at all: before, the ledger was empty.
    kinds = [e.kind for e in device_ledger.read_ledger("emulator-5554")]
    assert "mock_rules" in kinds


def test_mock_map_carries_a_host_and_a_times_budget(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    rule = engine.mock_map(
        "GET", "/api/v1/widgets", status=204, host="api.example.test", times=2
    )["rule"]
    assert rule["request"]["host"] == "api.example.test"
    assert rule["times"] == 2


def test_mock_map_refuses_a_stub_that_matches_every_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(UsageError, match="every request"):
        engine.mock_map("*", "*")
    assert engine.mock_map("*", "*", host="api.example.test")["ok"] is True
