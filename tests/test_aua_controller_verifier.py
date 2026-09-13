from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from android_ui_analyser.assertions import evaluate_assertion_step
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen
from android_ui_analyser.session_artifacts import SessionArtifactStore, observation_evidence_id
from android_ui_analyser.session_contracts import (
    load_session_contract,
    render_session_contract_yaml,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.verify_run import CONTRACT_DIR, ORACLE_VERSION, verify_run

PRODUCTS = [
    ("alpine_mug", "Alpine Mug", "$7.99"),
    ("beacon_lamp", "Beacon Lamp", "$12.50"),
    ("cedar_notebook", "Cedar Notebook", "$4.25"),
    ("drift_puzzle", "Drift Puzzle", "$18.00"),
]


def _frame(scenario: str, phase_id: str) -> dict[str, Any]:
    elements: list[Element] = []

    def add(
        rid: str,
        text: str | None = None,
        parent: int | None = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> int:
        identifier = len(elements) + 1
        y = identifier * 30
        bounds = bounds or (0, y, 100, y + 20)
        resource_id = rid if rid.startswith("compose_") else f"dev.aua.fixture:id/{rid}"
        elements.append(
            Element(
                id=identifier,
                type="android.widget.TextView",
                resource_id=resource_id,
                text=text,
                parent=parent,
                bounds=bounds,
                center=((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2),
            )
        )
        return identifier

    if phase_id == "cleanup":
        home = add("fixture_home")
        # Real home buttons must be retained: short absence selectors such as
        # classic_grid fall back to matching open_classic_grid here.
        add("open_classic_grid", "Classic View grid", home)
        add("open_compose_grid", "Compose grid", home)
    elif scenario == "async-recovery":
        add(
            "async_status",
            "Temporary signal error"
            if phase_id == "signal_error"
            else "Expedition ready: Aurora Trail",
        )
        if phase_id == "signal_error":
            add("async_retry", "Retry")
    else:
        mode = "price" if phase_id == "price_sorted" else "name"
        lane = scenario.split("-")[0]
        # AUA's live Compose accessibility result has unique leaf tags, no Card/grid
        # nodes, and parent=None for every product leaf. Do not invent containers
        # that would let an unsupported contains_all assertion pass in tests only.
        if lane == "compose":
            add("compose_sort_name", "Sort by name")
            add("compose_sort_price", "Sort by price")
            grid = None
        else:
            grid = add("classic_grid")
        add(
            "compose_order_status" if lane == "compose" else "order_status",
            f"Current order: {mode}",
        )
        products = [PRODUCTS[i] for i in (2, 0, 1, 3)] if mode == "price" else PRODUCTS
        for index, (slug, name, price) in enumerate(products):
            if lane == "compose":
                row_y = 468 + index * 178
                add(f"compose_name_{slug}", name, bounds=(89, row_y, 420, row_y + 63))
                add(f"compose_price_{slug}", price, bounds=(850, row_y, 991, row_y + 63))
            else:
                card = add("classic_item_card", parent=grid)
                add(f"classic_name_{slug}", name, card)
                add(f"classic_price_{slug}", price, card)
    # Initial and restored name screens deliberately share a fingerprint. Chronology, not
    # uniqueness of an evidence ID, must distinguish their two checkpoint proofs.
    fingerprint = f"{scenario}-{'name' if phase_id in {'default_name_order', 'name_order_restored'} else phase_id}"
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="dev.aua.fixture",
            activity=".MainActivity",
            source="hierarchy",
        ),
        elements=elements,
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            device_serial="fixture-device",
            fingerprint=fingerprint,
        ),
    ).model_dump(mode="json")


def _bundle(
    tmp_path: Path, scenario: str = "classic-sort", *, extra_after_restore: bool = False,
    direct_analyze: bool = False, post_cleanup_phase: str | None = None,
) -> Path:
    contract = load_session_contract(file=CONTRACT_DIR / f"{scenario}.yaml")
    store = SessionArtifactStore.create(
        tmp_path / "bundle",
        session_id="fixture-session",
        goal="Public fixture test",
        evidence="all",
        junit=False,
        contract_yaml=render_session_contract_yaml(contract),
    )
    phases = []
    expected = [(p.id, p.assertions) for p in contract.checkpoints]
    assert contract.cleanup is not None
    expected.append(("cleanup", contract.cleanup.assertions))

    def screenshot(path: Path) -> str:
        Image.new("RGB", (8, 8)).save(path)
        return str(path)

    sequence = 0
    for number, (phase_id, assertions) in enumerate(expected, 1):
        sequence += 1
        observed = _frame(scenario, phase_id)
        store.record(
            command="analyze_screen" if direct_analyze else "tap_and_analyze",
            result=(
                {**observed, "goal_progress": {"completed": number}}
                if direct_analyze else
                {"ok": True, "observation": observed, "goal_progress": {"completed": number}}
            ),
            invocation_id=f"call-{sequence}",
            duration_ms=1,
            screenshot=screenshot,
        )
        phases.append(
            {
                "id": phase_id,
                "status": "completed",
                "proof_mode": "fresh_assertions",
                "manual_completion_allowed": False,
                "proof": {
                    "source": "contract_assertions",
                    "verified": True,
                    "assertions_verified": len(assertions),
                    "capture_order": sequence,
                    "evidence_id": observation_evidence_id("fixture-session", observed),
                    "observation": {
                        "fingerprint": observed["meta"]["fingerprint"],
                        "package": "dev.aua.fixture",
                        "device_serial": "fixture-device",
                    },
                },
            }
        )
        if extra_after_restore and phase_id == "name_order_restored":
            sequence += 1
            store.record(
                command="tap_and_analyze",
                result={
                    "ok": True,
                    "observation": _frame(scenario, "price_sorted"),
                    "goal_progress": {"completed": number},
                },
                invocation_id=f"call-{sequence}",
                duration_ms=1,
                screenshot=screenshot,
            )
    if post_cleanup_phase:
        # Live MCP analyze_screen shape: a direct observation with coaching's compact
        # progress decoration, including a completed historical cleanup checkpoint.
        store.record(
            command="analyze_screen",
            result={
                **_frame(scenario, post_cleanup_phase),
                "goal_progress": {
                    "session_id": "fixture-session", "completed": len(expected),
                    "total": len(expected), "done": True, "terminated": False,
                    "status": "completed", "current": None, "next_call": None,
                    "checkpoint": None, "upcoming": [],
                },
            },
            invocation_id=f"call-{sequence + 1}",
            duration_ms=1,
            screenshot=screenshot,
        )
    result = {
        "session_id": "fixture-session",
        "ok": True,
        "finished": True,
        "terminated": True,
        "verdict": "passed",
        "goal_progress": {"phases": phases},
    }
    store.finalize(result, verdict="passed", checkpoints=phases)
    return store.root


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("scenario", ["classic-sort", "compose-sort", "async-recovery"])
def test_authored_contracts_pass_on_complete_ordered_artifacts(
    tmp_path: Path, scenario: str
) -> None:
    report = verify_run(_bundle(tmp_path, scenario), scenario)
    assert report["passed"] is True, report
    assert report["verified"] is True
    assert report["cleanup_verified"] is True
    assert report["false_pass"] is False
    assert report["evidence_resolved"] == report["evidence_expected"]


@pytest.mark.parametrize("direct_checkpoints", [False, True])
def test_direct_analyze_progress_decoration_is_valid_for_checkpoints_and_cleanup_tail(
    tmp_path: Path, direct_checkpoints: bool,
) -> None:
    root = _bundle(
        tmp_path, "async-recovery", direct_analyze=direct_checkpoints,
        post_cleanup_phase="cleanup",
    )
    report = verify_run(root, "async-recovery")

    assert report["oracle"] == ORACLE_VERSION == "aua_contract_replay_v2"
    assert report["passed"] is True, report
    assert report["verified"] is True and report["cleanup_verified"] is True
    assert report["post_cleanup_observations"] == [{"sequence": 4, "passed": True}]
    raw = _read(Path(_read(root / "manifest.json")["entries"][-1]["observation"]))
    assert raw["goal_progress"]["completed"] == 3  # Normalization never edits evidence.


def test_completed_progress_decoration_cannot_hide_post_cleanup_regression(tmp_path: Path) -> None:
    root = _bundle(tmp_path, "async-recovery", post_cleanup_phase="expedition_ready")
    report = verify_run(root, "async-recovery")

    assert report["verified"] is True
    assert report["passed"] is False and report["cleanup_verified"] is False
    assert report["false_pass"] is True
    assert report["post_cleanup_observations"] == [{"sequence": 4, "passed": False}]


def test_progress_normalization_does_not_ignore_other_unknown_fields(tmp_path: Path) -> None:
    root = _bundle(tmp_path, "async-recovery", direct_analyze=True)
    entry = _read(root / "manifest.json")["entries"][0]
    asset = Path(entry["observation"])
    raw = _read(asset)
    raw["unknown_decoration"] = {"pretend_success": True}
    _write(asset, raw)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    calls[0]["result"] = raw
    (root / "calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls))

    report = verify_run(root, "async-recovery")

    assert report["passed"] is False and report["verified"] is False
    assert "unknown_decoration" in report["checkpoints"][0]["reason"]


def test_filtered_post_cleanup_analyze_without_fingerprint_remains_unverified(tmp_path: Path) -> None:
    root = _bundle(tmp_path, "async-recovery", post_cleanup_phase="cleanup")
    manifest = _read(root / "manifest.json")
    entry = manifest["entries"][-1]
    asset = Path(entry["observation"])
    raw = _read(asset)
    # The real run's final query='Expedition ready' returned a filtered empty view
    # while home was displayed. Accepting its progress decoration must not invent
    # missing whole-screen evidence or silently bypass the remaining freshness check.
    raw["elements"] = []
    raw["meta"]["fingerprint"] = None
    _write(asset, raw)
    entry["evidence_id"] = observation_evidence_id("fixture-session", raw)
    _write(root / "manifest.json", manifest)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    calls[-1]["result"] = raw
    (root / "calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls))

    report = verify_run(root, "async-recovery")

    assert all(row["passed"] for row in report["checkpoints"])
    assert report["verified"] is False and report["passed"] is False
    assert report["false_pass"] is None
    assert report["reasons"] == ["observation lacks a fresh fingerprint"]


@pytest.mark.parametrize("missing", ["observation", "screenshot"])
def test_evidence_id_with_missing_actual_asset_is_unverified(tmp_path: Path, missing: str) -> None:
    root = _bundle(tmp_path)
    manifest = _read(root / "manifest.json")
    Path(manifest["entries"][1][missing]).unlink()
    report = verify_run(root, "classic-sort")
    assert report["passed"] is False
    assert report["verified"] is False
    assert report["false_pass"] is None


def test_corrupt_png_is_not_evidence(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    Path(_read(root / "manifest.json")["entries"][1]["screenshot"]).write_bytes(b"png")
    assert verify_run(root, "classic-sort")["verified"] is False


@pytest.mark.parametrize("scenario", ["classic-sort", "compose-sort"])
def test_wrong_price_pairing_overrules_completed_proof(tmp_path: Path, scenario: str) -> None:
    root = _bundle(tmp_path, scenario)
    entry = _read(root / "manifest.json")["entries"][1]
    asset = Path(entry["observation"])
    observed = _read(asset)
    for element in observed["elements"]:
        if element.get("text") == "$7.99":
            element["text"] = "$700.00"
    _write(asset, observed)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    calls[1]["result"]["observation"] = observed
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))
    report = verify_run(root, scenario)
    assert report["verified"] is True
    assert report["passed"] is False
    assert report["false_pass"] is True


def test_compose_fixture_matches_observable_leaf_structure() -> None:
    observed = _frame("compose-sort", "default_name_order")
    leaves = observed["elements"]
    assert all(element["parent"] is None for element in leaves)
    rids = [element["resource_id"] for element in leaves]
    assert len(rids) == len(set(rids)) == 11
    assert not any(rid.startswith(("compose_item_", "compose_grid")) for rid in rids)
    products = [element for element in leaves if element["resource_id"].startswith(("compose_name_", "compose_price_"))]
    assert [element["text"] for element in products] == [value for _, name, price in PRODUCTS for value in (name, price)]
    for name, price in zip(products[::2], products[1::2], strict=True):
        assert name["bounds"][1] == price["bounds"][1]


def test_compose_interleaved_order_rejects_a_price_from_another_row(tmp_path: Path) -> None:
    root = _bundle(tmp_path, "compose-sort")
    entry = _read(root / "manifest.json")["entries"][1]
    asset = Path(entry["observation"])
    observed = _read(asset)
    prices = [index for index, element in enumerate(observed["elements"])
              if element["resource_id"].startswith("compose_price_")]
    first, second = prices[:2]
    observed["elements"][first], observed["elements"][second] = (
        observed["elements"][second], observed["elements"][first],
    )
    _write(asset, observed)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    calls[1]["result"]["observation"] = observed
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))

    report = verify_run(root, "compose-sort")

    assert report["verified"] is True
    assert report["passed"] is False
    assert report["false_pass"] is True


def test_initial_name_frame_cannot_prove_later_restoration(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    manifest = _read(root / "manifest.json")
    manifest["entries"].pop(2)
    _write(root / "manifest.json", manifest)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    calls.pop(2)
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))
    report = verify_run(root, "classic-sort")
    assert report["passed"] is False
    assert report["false_pass"] is None
    assert "ordered" in report["checkpoints"][2]["reason"]


def test_repeated_name_frames_pass_without_optional_compact_progress(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    # Observed in a real MCP run: both name frames have null progress, while the
    # intervening price frame and later home frame retain their completed counts.
    calls[0]["result"]["goal_progress"] = None
    calls[2]["result"]["goal_progress"] = None
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))

    report = verify_run(root, "classic-sort")

    assert report["passed"] is True, report
    assert [row["sequence"] for row in report["checkpoints"]] == [1, 2, 3, 4]
    assert [row["progress_recorded"] for row in report["checkpoints"]] == [False, True, False, True]


def test_missing_compact_progress_does_not_allow_reusing_initial_name_proof(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    result = _read(root / "result.json")
    phases = result["goal_progress"]["phases"]
    # Both name checkpoints share an evidence ID, but reusing the initial capture
    # cannot establish the later restoration after the intervening price checkpoint.
    phases[2]["proof"] = phases[0]["proof"]
    _write(root / "result.json", result)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    for call in calls:
        call["result"]["goal_progress"] = None
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))

    report = verify_run(root, "classic-sort")

    assert report["passed"] is False
    assert report["verified"] is False
    assert "capture order" in report["checkpoints"][2]["reason"]


def test_recorded_progress_contradiction_is_not_ignored(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    calls[2]["result"]["goal_progress"] = {"completed": 1}
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))

    report = verify_run(root, "classic-sort")

    assert report["passed"] is False
    assert "ordered" in report["checkpoints"][2]["reason"]


def test_missing_progress_does_not_hide_a_later_cleanup_regression(tmp_path: Path) -> None:
    root = _bundle(tmp_path, extra_after_restore=True)
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    for call in calls:
        call["result"]["goal_progress"] = None
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))

    report = verify_run(root, "classic-sort")

    assert report["verified"] is True
    assert report["passed"] is False
    assert report["cleanup_verified"] is False
    assert report["false_pass"] is True


def test_sorting_again_after_restoration_invalidates_cleanup(tmp_path: Path) -> None:
    report = verify_run(_bundle(tmp_path, extra_after_restore=True), "classic-sort")
    assert report["verified"] is True
    assert report["passed"] is False
    assert report["cleanup_verified"] is False
    assert report["false_pass"] is True


def test_manual_claim_cannot_replace_automatic_proof(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    result = _read(root / "result.json")
    result["goal_progress"]["phases"][1]["proof"]["source"] = "manual_evidence"
    _write(root / "result.json", result)
    report = verify_run(root, "classic-sort")
    assert report["passed"] is False
    assert report["false_pass"] is None


def test_different_authored_contract_is_rejected(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    contract = (root / "contract.yaml").read_text().replace("$7.99", "$700.00")
    (root / "contract.yaml").write_text(contract)
    report = verify_run(root, "classic-sort")
    assert report["verified"] is False
    assert report["false_pass"] is None
    assert "contract differs" in report["reasons"][0]


def test_missing_bundle_remains_unknown(tmp_path: Path) -> None:
    report = verify_run(tmp_path, "classic-sort")
    assert report["passed"] is False
    assert report["false_pass"] is None


def test_asset_symlink_cannot_escape_bundle(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    asset = Path(_read(root / "manifest.json")["entries"][1]["screenshot"])
    external = tmp_path / "external.png"
    asset.rename(external)
    asset.symlink_to(external)
    assert verify_run(root, "classic-sort")["verified"] is False


def test_copied_bundle_rebases_absolute_evidence_paths(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    renamed = root.with_name("renamed")
    root.rename(renamed)
    assert verify_run(renamed, "classic-sort")["passed"] is True


def test_missing_later_finish_observation_cannot_be_skipped(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    manifest = _read(root / "manifest.json")
    calls = list(map(json.loads, (root / "calls.jsonl").read_text().splitlines()))
    # A later finish read is part of the evidence boundary even though the home
    # checkpoint was already complete. Omitting its asset must not hide a new state.
    manifest["entries"].append(
        {"sequence": 5, "invocation_id": "finish", "command": "session_finish"}
    )
    calls.append(
        {
            "sequence": 5,
            "invocation_id": "finish",
            "command": "session_finish",
            "result": {"observation": _frame("classic-sort", "price_sorted")},
        }
    )
    _write(root / "manifest.json", manifest)
    (root / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls))
    report = verify_run(root, "classic-sort")
    assert report["passed"] is False
    assert report["verified"] is False
    assert report["false_pass"] is None


@pytest.mark.parametrize("scenario", ["classic-sort", "compose-sort", "async-recovery"])
def test_cleanup_accepts_home_menu_but_rejects_the_test_screen(scenario: str) -> None:
    contract = load_session_contract(file=CONTRACT_DIR / f"{scenario}.yaml")
    assert contract.cleanup is not None
    home = AnalyzeResult.model_validate(_frame(scenario, "cleanup"))
    assert all(
        evaluate_assertion_step(step, home.elements).ok for step in contract.cleanup.assertions
    )
    phase = "expedition_ready" if scenario == "async-recovery" else "name_order_restored"
    test_screen = AnalyzeResult.model_validate(_frame(scenario, phase))
    assert not all(
        evaluate_assertion_step(step, test_screen.elements).ok
        for step in contract.cleanup.assertions
    )
