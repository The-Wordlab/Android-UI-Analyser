"""Pure authored-session contract schema tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from android_ui_analyser.errors import UsageError
from android_ui_analyser.memory import RouteStep
from android_ui_analyser.session_contracts import (
    ContractCheckpoint,
    ContractCleanup,
    SessionContract,
    load_session_contract,
    parse_session_contract_yaml,
    render_session_contract_yaml,
)

CONTRACT_YAML = """\
version: 1
checkpoints:
  - id: catalog_ready
    description: Confirm the catalog and its first row
    assertions:
      - assert:
          rid: catalogList
          exists: true
      - assert_order:
          axis: vertical
          selectors:
            - {rid: catalogRow, index: 0}
            - {rid: catalogRow, index: 1}
  - id: selection_saved
    description: Confirm the selected row state
    assertions:
      - assert:
          text: Sample item
          selected: true
cleanup:
  description: Restore the default catalog state
  assertions:
    - assert:
        rid: defaultSort
        checked: true
"""


def test_contract_parses_ordered_deterministic_checkpoints_and_cleanup() -> None:
    contract = parse_session_contract_yaml(CONTRACT_YAML)

    assert contract.version == 1
    assert [checkpoint.id for checkpoint in contract.checkpoints] == [
        "catalog_ready",
        "selection_saved",
    ]
    assert [step.kind for step in contract.checkpoints[0].assertions] == [
        "assert",
        "assert-order",
    ]
    assert contract.checkpoints[0].proof_mode == "fresh_assertions"
    assert contract.checkpoints[0].manual_completion_allowed is False
    assert contract.cleanup is not None
    assert contract.cleanup.proof_mode == "fresh_assertions"
    assert contract.cleanup.manual_completion_allowed is False


def test_contract_canonical_yaml_round_trips_without_internal_policy_fields() -> None:
    original = parse_session_contract_yaml(CONTRACT_YAML)

    rendered = render_session_contract_yaml(original)
    reparsed = parse_session_contract_yaml(rendered)

    assert rendered.startswith("version: 1\ncheckpoints:\n")
    assert "proof_mode" not in rendered
    assert "manual_completion_allowed" not in rendered
    assert reparsed == original


def test_contract_inherits_relational_assertions_from_the_flow_schema() -> None:
    contract = parse_session_contract_yaml(
        """\
checkpoints:
  - id: row_content
    description: Prove related content and reading order
    assertions:
      - assert:
          rid: productCard
          index: 0
          within: {rid: catalogList}
          same_parent_as: {text: Sample item}
          contains_all:
            - {text: Sample item}
            - {text: "$7.99"}
      - assert_order:
          axis: reading
          selectors:
            - {rid: productCard, index: 0}
            - {rid: productCard, index: 1}
"""
    )

    assertion = contract.checkpoints[0].assertions[0].assertion
    assert assertion["within"] == {"rid": "catalogList"}
    assert assertion["same_parent_as"] == {"text": "Sample item"}
    assert assertion["contains_all"] == [
        {"text": "Sample item"},
        {"text": "$7.99"},
    ]
    assert parse_session_contract_yaml(render_session_contract_yaml(contract)) == contract


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("[]\n", "must be a mapping"),
        ("version: true\ncheckpoints: []\n", "must be an integer"),
        ("version: 2\ncheckpoints: []\n", "unsupported contract version 2"),
        ("version: 1\ncheckpoints: []\n", "non-empty `checkpoints:`"),
        (
            "version: 1\nunknown: true\ncheckpoints: []\n",
            "unknown top-level contract keys: unknown",
        ),
        (
            "checkpoints:\n"
            "  - id: one\n"
            "    description: First\n"
            "    extra: no\n"
            "    assertions: [{assert: {text: Ready}}]\n",
            "unknown checkpoint[0] keys: extra",
        ),
        (
            "checkpoints:\n"
            "  - id: one\n"
            "    description: First\n"
            "    assertions: [{assert: {text: Ready}}]\n"
            "cleanup:\n"
            "  description: Restore\n"
            "  extra: no\n"
            "  assertions: [{assert: {text: Home}}]\n",
            "unknown cleanup keys: extra",
        ),
    ],
)
def test_contract_rejects_malformed_versions_shapes_and_unknown_keys(
    yaml_text: str, message: str
) -> None:
    with pytest.raises(UsageError) as exc_info:
        parse_session_contract_yaml(yaml_text)
    assert message in str(exc_info.value)


@pytest.mark.parametrize("target", ["checkpoint", "cleanup"])
def test_contract_rejects_empty_assertion_sets(target: str) -> None:
    cleanup = (
        "cleanup:\n  description: Restore home\n  assertions: []\n"
        if target == "cleanup"
        else ""
    )
    assertions = "[]" if target == "checkpoint" else "[{assert: {text: Ready}}]"
    yaml_text = (
        "checkpoints:\n"
        "  - id: ready\n"
        "    description: Confirm readiness\n"
        f"    assertions: {assertions}\n"
        f"{cleanup}"
    )

    with pytest.raises(UsageError, match="non-empty `assertions:`"):
        parse_session_contract_yaml(yaml_text)


def test_contract_rejects_duplicate_checkpoint_ids_after_normalization() -> None:
    yaml_text = """\
checkpoints:
  - id: ready
    description: First proof
    assertions: [{assert: {text: Ready}}]
  - id: " ready "
    description: Duplicate proof
    assertions: [{assert: {text: Ready again}}]
"""

    with pytest.raises(UsageError, match="duplicate checkpoint ids: ready"):
        parse_session_contract_yaml(yaml_text)


@pytest.mark.parametrize(
    ("assertion", "message"),
    [
        ("tap: Continue", "must use `assert` or `assert_order`"),
        ("assert: {text: Ready, mystery: true}", "unknown keys for assert"),
        (
            "assert_order: {axis: diagonal, selectors: [{text: A}, {text: B}]}",
            "`axis:` must be horizontal, vertical, or reading",
        ),
    ],
)
def test_contract_delegates_assertion_validation_to_flow_schema(
    assertion: str, message: str
) -> None:
    yaml_text = (
        "checkpoints:\n"
        "  - id: ready\n"
        "    description: Confirm readiness\n"
        f"    assertions:\n      - {assertion}\n"
    )

    with pytest.raises(UsageError) as exc_info:
        parse_session_contract_yaml(yaml_text)
    assert message in str(exc_info.value)


def test_programmatic_models_cannot_enable_manual_completion_or_action_steps() -> None:
    assertion = RouteStep(kind="assert", label="Ready", assertion={"exists": True})
    checkpoint = ContractCheckpoint(
        id="ready", description="Confirm readiness", assertions=[assertion]
    )
    cleanup = ContractCleanup(description="Restore", assertions=[assertion])
    assert SessionContract(checkpoints=[checkpoint], cleanup=cleanup)

    with pytest.raises(ValidationError):
        ContractCheckpoint(
            id="ready",
            description="Confirm readiness",
            assertions=[assertion],
            manual_completion_allowed=True,
        )
    with pytest.raises(ValidationError):
        ContractCheckpoint(
            id="ready",
            description="Confirm readiness",
            assertions=[RouteStep(kind="tap", label="Continue")],
        )


def test_loader_requires_exactly_one_inline_or_file_source(tmp_path: Path) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text(CONTRACT_YAML, encoding="utf-8")

    assert load_session_contract(file=path) == parse_session_contract_yaml(CONTRACT_YAML)
    assert load_session_contract(yaml=CONTRACT_YAML) == parse_session_contract_yaml(CONTRACT_YAML)

    with pytest.raises(UsageError, match="exactly one"):
        load_session_contract()
    with pytest.raises(UsageError, match="exactly one"):
        load_session_contract(file=path, yaml=CONTRACT_YAML)
    with pytest.raises(UsageError, match="could not read"):
        load_session_contract(file=tmp_path / "missing.yaml")
