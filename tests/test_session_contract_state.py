"""Authored contracts persist as strict, ordered session proof phases."""

from pathlib import Path

import pytest

from android_ui_analyser.session import (
    GoalPhase,
    ObservationProvenance,
    PhaseProof,
    contract_phases,
    create_session_state,
    finish_session_state,
    load_session_state,
    mark_phase_complete,
    phase_progress,
    update_session_state,
)
from android_ui_analyser.session_contracts import (
    SessionContract,
    parse_session_contract_yaml,
    render_session_contract_yaml,
)


def _contract(*, cleanup: bool = True) -> SessionContract:
    cleanup_yaml = (
        """
cleanup:
  description: Restore the default ordering
  assertions:
    - assert: {rid: defaultSort, checked: true}
"""
        if cleanup
        else ""
    )
    return parse_session_contract_yaml(
        """\
version: 1
checkpoints:
  - id: catalog_ready
    description: Confirm the catalog and its first card
    assertions:
      - assert: {rid: catalogList, exists: true}
      - assert: {rid: productCard, index: 0, exists: true}
  - id: prices_ordered
    description: Confirm product cards follow reading order
    assertions:
      - assert_order:
          axis: reading
          selectors: [{rid: productCard, index: 0}, {rid: productCard, index: 1}]
"""
        + cleanup_yaml
    )


def _state(tmp_path: Path, contract: SessionContract):
    return create_session_state(
        tmp_path,
        goal="Sort the catalog and restore it",
        serial="example-serial",
        owner="example-agent",
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
        contract=contract,
        artifact_dir="/tmp/example-artifacts",
        evidence="all",
        junit=True,
        capture_package="com.example.catalog",
        capture_context_id="catalog-v2",
        capture_segment=3,
        capture_start_order=12,
    )


def _proof(
    fingerprint: str,
    *,
    assertions: int,
    capture_order: int = 13,
    session_id: str = "example",
) -> PhaseProof:
    return PhaseProof(
        source="contract_assertions",
        command="contract_assertions",
        verified=True,
        observation=ObservationProvenance(
            fingerprint=fingerprint,
            source="hierarchy",
            device_serial="example-serial",
            package="com.example.catalog",
        ),
        evidence_id=f"session-{session_id}:observation:{fingerprint}",
        assertions_verified=assertions,
        capture_order=capture_order,
    )


def test_contract_compiles_to_ordered_non_manual_phases() -> None:
    phases = contract_phases(_contract())

    assert [phase.id for phase in phases] == ["catalog_ready", "prices_ordered", "cleanup"]
    assert [phase.kind for phase in phases] == ["verify", "verify", "cleanup"]
    assert [phase.intent for phase in phases] == [
        "contract_checkpoint",
        "contract_checkpoint",
        "contract_cleanup",
    ]
    assert all(phase.satisfaction == "fresh_assertions" for phase in phases)
    assert all(phase.proof_mode == "fresh_assertions" for phase in phases)
    assert all(phase.manual_completion_allowed is False for phase in phases)
    assert [phase.status for phase in phases] == ["active", "pending", "pending"]
    assert [len(phase.assertions) for phase in phases] == [2, 1, 1]
    assert phases[-1].terminal is True


def test_contract_cleanup_id_is_stable_when_checkpoint_uses_cleanup() -> None:
    contract = parse_session_contract_yaml(
        """\
checkpoints:
  - id: cleanup
    description: Inspect the cleanup control
    assertions: [{assert: {rid: cleanupButton}}]
cleanup:
  description: Restore home
  assertions: [{assert: {rid: homeScreen}}]
"""
    )

    assert [phase.id for phase in contract_phases(contract)] == ["cleanup", "cleanup_2"]


def test_create_persists_contract_artifact_and_capture_metadata(tmp_path: Path) -> None:
    contract = _contract()
    state = _state(tmp_path, contract)
    loaded = load_session_state(tmp_path, session_id=state.session_id)

    assert loaded == state
    assert state.contract == contract
    assert state.contract_yaml == render_session_contract_yaml(contract)
    assert state.artifact_dir == "/tmp/example-artifacts"
    assert state.evidence == "all"
    assert state.junit is True
    assert state.capture_package == "com.example.catalog"
    assert state.capture_context_id == "catalog-v2"
    assert state.capture_segment == 3
    assert state.capture_start_order == 12
    assert state.last_contract_fingerprint is None


def test_natural_goal_sessions_keep_legacy_phase_and_metadata_defaults(tmp_path: Path) -> None:
    state = create_session_state(
        tmp_path,
        goal="Verify the Example catalog title",
        serial="example-serial",
        owner="example-agent",
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )

    assert state.contract is None
    assert state.contract_yaml is None
    assert state.artifact_dir is None
    assert state.evidence == "failures"
    assert state.junit is False
    assert state.capture_package is None
    assert state.capture_context_id is None
    assert state.capture_segment is None
    assert state.capture_start_order is None
    assert state.phases[0].intent == "ui_verification"
    assert state.phases[0].proof_mode == "manual_or_structured"
    assert state.phases[0].manual_completion_allowed is True
    assert state.phases[0].assertions == []


def test_manual_evidence_cannot_complete_an_authored_checkpoint(tmp_path: Path) -> None:
    state = _state(tmp_path, _contract())

    with pytest.raises(ValueError, match="manual evidence cannot complete"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="catalog_ready",
            evidence="The catalog and its first card are visible",
        )

    progress = phase_progress(state)
    assert progress["checkpoint"] is None
    assert progress["current"]["proof_mode"] == "fresh_assertions"


def test_create_accepts_yaml_and_persists_its_canonical_form(tmp_path: Path) -> None:
    contract = _contract(cleanup=False)
    state = create_session_state(
        tmp_path,
        goal="Verify the catalog",
        serial="example-serial",
        owner="example-agent",
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
        contract_yaml="\n" + render_session_contract_yaml(contract),
    )

    assert state.contract == contract
    assert state.contract_yaml == render_session_contract_yaml(contract)


def test_update_session_state_revalidates_metadata_and_preserves_identity(tmp_path: Path) -> None:
    state = _state(tmp_path, _contract())

    updated = update_session_state(tmp_path, state, artifact_dir="/tmp/resolved-run")

    assert updated.artifact_dir == "/tmp/resolved-run"
    assert load_session_state(tmp_path, session_id=state.session_id) == updated
    with pytest.raises(ValueError, match="cannot be changed"):
        update_session_state(tmp_path, updated, serial="another-device")
    with pytest.raises(ValueError):
        update_session_state(tmp_path, updated, evidence="sometimes")


@pytest.mark.parametrize(
    "proof",
    [
        _proof("missing-one", assertions=1),
        PhaseProof(
            source="contract_assertions",
            command="contract_assertions",
            verified=True,
            observation=ObservationProvenance(
                fingerprint="missing-evidence",
                source="hierarchy",
                device_serial="example-serial",
                package="com.example.catalog",
            ),
            assertions_verified=2,
        ),
    ],
)
def test_contract_proof_requires_every_assertion_and_an_evidence_id(
    tmp_path: Path, proof: PhaseProof
) -> None:
    state = _state(tmp_path, _contract())

    with pytest.raises(ValueError, match="structured proof does not satisfy"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="catalog_ready",
            evidence="Authored assertions evaluated",
            _proof=proof,
        )


def test_contract_proof_advances_once_per_fresh_frame_and_records_capture_boundary(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, _contract())
    state = mark_phase_complete(
        tmp_path,
        state,
        phase_id="catalog_ready",
        evidence="Both authored assertions passed on frame-one",
        _proof=_proof("frame-one", assertions=2, session_id=state.session_id),
    )

    assert state.phases[0].status == "completed"
    assert state.phases[0].proof is not None
    assert state.phases[0].proof.evidence_id == (
        f"session-{state.session_id}:observation:frame-one"
    )
    assert state.phases[0].proof.capture_order == 13
    assert state.last_contract_fingerprint == "frame-one"
    assert state.phases[1].status == "active"

    with pytest.raises(ValueError, match="unchanged observation"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="prices_ordered",
            evidence="Ordering assertion passed on the unchanged frame",
            _proof=_proof("frame-one", assertions=1, session_id=state.session_id),
        )

    advanced = mark_phase_complete(
        tmp_path,
        state,
        phase_id="prices_ordered",
        evidence="Ordering assertion passed on frame-two",
        _proof=_proof(
            "frame-two",
            assertions=1,
            capture_order=14,
            session_id=state.session_id,
        ),
    )
    assert advanced.phases[1].status == "completed"
    assert advanced.phases[2].status == "active"
    assert advanced.last_contract_fingerprint == "frame-two"


def test_contract_proof_must_belong_to_the_session_device(tmp_path: Path) -> None:
    state = _state(tmp_path, _contract())
    proof = _proof("foreign-frame", assertions=2, session_id=state.session_id)
    assert proof.observation is not None
    proof.observation.device_serial = "another-device"

    with pytest.raises(ValueError, match="different session device"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="catalog_ready",
            evidence="Assertions passed elsewhere",
            _proof=proof,
        )


def test_contract_proof_evidence_must_belong_to_the_session(tmp_path: Path) -> None:
    state = _state(tmp_path, _contract())

    with pytest.raises(ValueError, match="evidence belongs to a different session"):
        mark_phase_complete(
            tmp_path,
            state,
            phase_id="catalog_ready",
            evidence="Assertions passed in another session",
            _proof=_proof("foreign-frame", assertions=2, session_id="another-session"),
        )


def test_finish_does_not_claim_authored_cleanup_without_ui_proof(tmp_path: Path) -> None:
    contract = parse_session_contract_yaml(
        """\
checkpoints:
  - id: ready
    description: Confirm ready state
    assertions: [{assert: {rid: readyState}}]
cleanup:
  description: Restore home state
  assertions: [{assert: {rid: homeState}}]
"""
    )
    state = _state(tmp_path, contract)
    state = mark_phase_complete(
        tmp_path,
        state,
        phase_id="ready",
        evidence="Ready assertion passed",
        _proof=_proof("ready-frame", assertions=1, session_id=state.session_id),
    )

    finished = finish_session_state(tmp_path, state)
    cleanup = finished.phases[1]

    assert cleanup.status == "active"
    assert cleanup.proof is None
    assert phase_progress(finished)["status"] == "terminated_incomplete"
    assert phase_progress(finished)["blocking_phases"] == [
        {
            "id": "cleanup",
            "objective": "Restore home state",
            "required_evidence": "all authored assertions passing on one fresh observation",
        }
    ]


def test_legacy_goal_phase_payload_gets_backward_compatible_contract_defaults() -> None:
    phase = GoalPhase.model_validate(
        {"id": "phase_1", "objective": "Inspect the catalog", "status": "active"}
    )

    assert phase.assertions == []
    assert phase.proof_mode == "manual_or_structured"
    assert phase.manual_completion_allowed is True
