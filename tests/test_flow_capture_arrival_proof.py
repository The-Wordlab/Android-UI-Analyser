"""A satisfied action-bound predicate may prove an otherwise unmapped flow destination."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.flows import Flow, render_flow_yaml
from android_ui_analyser.memory import (
    AppMemoryStore,
    CaptureArrivalProof,
    CaptureArrivalTerm,
    RouteStep,
    SessionState,
    capture_arrival_for_current,
    capture_arrival_predicate,
)
from conftest import make_config

PACKAGE = "com.example.catalog"
SERIAL = "capture-proof-host-only"


def _store(tmp_path: Path) -> AppMemoryStore:
    config = make_config(memory={"dir": str(tmp_path / "memory")})
    return AppMemoryStore(config.memory)


def _session() -> SessionState:
    return SessionState(
        package=PACKAGE,
        active_context_id="catalog-default",
        capture_segment=7,
        next_capture_order=42,
        recent=[
            RouteStep(
                kind="tap",
                resource_id="openDetails",
                by="id",
                package=PACKAGE,
                origin_package=PACKAGE,
                context_id="catalog-default",
                capture_segment=7,
                capture_order=41,
            )
        ],
    )


def test_satisfied_ui_terms_are_bound_to_the_exact_recorded_action(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, _session())

    proof = store.record_action_arrival(
        SERIAL,
        terms=[
            {"by": "rid", "value": "detailPanel"},
            {"by": "text", "value": "Loading", "negated": True},
        ],
        fingerprint="hierarchy-frame-abc",
        package=PACKAGE,
    )

    assert proof is not None
    assert proof.capture_order == 41
    assert capture_arrival_predicate(proof) == "rid:detailPanel,!text:Loading"
    persisted = store.load_session(SERIAL)
    assert persisted.recent[-1].arrival_proof == proof

    checked = capture_arrival_for_current(
        persisted.recent,
        session=persisted,
        observation_package=PACKAGE,
        observation_fingerprint="hierarchy-frame-abc",
    )
    assert checked.proof == proof
    assert "matches" in checked.reason


def test_capture_proof_is_rejected_after_frame_or_segment_changes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, _session())
    proof = store.record_action_arrival(
        SERIAL,
        terms=[{"by": "text", "value": "Details ready"}],
        fingerprint="destination-frame",
        package=PACKAGE,
    )
    assert proof is not None
    persisted = store.load_session(SERIAL)

    changed_frame = capture_arrival_for_current(
        persisted.recent,
        session=persisted,
        observation_package=PACKAGE,
        observation_fingerprint="different-frame",
    )
    assert changed_frame.proof is None
    assert "fingerprint" in changed_frame.reason

    moved_session = persisted.model_copy(update={"capture_segment": 8})
    changed_segment = capture_arrival_for_current(
        persisted.recent,
        session=moved_session,
        observation_package=PACKAGE,
        observation_fingerprint="destination-frame",
    )
    assert changed_segment.proof is None
    assert "older session segment" in changed_segment.reason


def test_capture_never_promotes_private_volatile_or_offscreen_terms(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for terms in (
        [{"by": "text", "value": "person@example.test"}],
        [{"by": "text", "value": "Result 123456"}],
        [{"by": "net", "value": "GET /private"}],
        [{"by": "text", "value": "Loading", "negated": True}],
    ):
        store.save_session(SERIAL, _session())
        assert (
            store.record_action_arrival(
                SERIAL,
                terms=terms,
                fingerprint="destination-frame",
                package=PACKAGE,
            )
            is None
        )
        assert store.load_session(SERIAL).recent[-1].arrival_proof is None


def test_capture_predicate_escapes_grammar_separators() -> None:
    proof = CaptureArrivalProof(
        terms=[CaptureArrivalTerm(by="text", value=r"Ready, path\name")],
        fingerprint="frame",
        package=PACKAGE,
        origin_package=PACKAGE,
        context_id="catalog-default",
        capture_segment=7,
        capture_order=41,
    )
    assert capture_arrival_predicate(proof) == r"text:Ready\, path\\name"


def test_durable_routes_drop_capture_only_arrival_proof(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, _session())
    proof = store.record_action_arrival(
        SERIAL,
        terms=[{"by": "rid", "value": "detailPanel"}],
        fingerprint="destination-frame",
        package=PACKAGE,
    )
    assert proof is not None

    recorded = store.load_session(SERIAL).recent[-1]
    durable = AppMemoryStore._route_step(recorded, PACKAGE)

    assert durable.arrival_proof is None
    assert durable.capture_order is None
    assert durable.capture_segment is None


def test_capture_provenance_never_renders_as_flow_step_data(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, _session())
    proof = store.record_action_arrival(
        SERIAL,
        terms=[{"by": "rid", "value": "detailPanel"}],
        fingerprint="destination-frame",
        package=PACKAGE,
    )
    assert proof is not None
    recorded = store.load_session(SERIAL).recent[-1]

    rendered = render_flow_yaml(Flow(name="details", app=PACKAGE, steps=[recorded]))

    assert "arrival_proof" not in rendered
    assert "destination-frame" not in rendered
    assert "capture_segment" not in rendered
