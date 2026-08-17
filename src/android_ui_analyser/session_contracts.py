"""Authored, deterministic proof contracts for goal-aware AUA sessions.

The contract schema deliberately reuses the flow assertion grammar.  A contract is only a
declaration of what must be true; it contains no action steps and has no device side effects.
Session lifecycle code can therefore load it before acquiring a device and can only complete a
checkpoint from fresh assertion evidence, never from a free-form manual claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml as yaml_lib
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from .errors import UsageError
from .flows import Flow, parse_flow_yaml, render_flow_yaml
from .memory import RouteStep

SESSION_CONTRACT_VERSION: Literal[1] = 1

ContractProofMode = Literal["fresh_assertions"]


class ContractCheckpoint(BaseModel):
    """One ordered checkpoint whose complete assertion set is the only valid proof."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    assertions: list[RouteStep]
    # These invariants are persisted when the model is embedded in a session, but are not part
    # of the authored YAML surface.  Keeping them typed prevents lifecycle integrations from
    # accidentally routing a contract checkpoint through the legacy `phase_done` path.
    proof_mode: ContractProofMode = "fresh_assertions"
    manual_completion_allowed: Literal[False] = False

    @field_validator("id", "description")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator("assertions")
    @classmethod
    def _assertions_only(cls, assertions: list[RouteStep]) -> list[RouteStep]:
        if not assertions:
            raise ValueError("needs at least one assertion")
        invalid = [step.kind for step in assertions if step.kind not in {"assert", "assert-order"}]
        if invalid:
            raise ValueError(
                "accepts only `assert` and `assert_order`, got " + ", ".join(invalid)
            )
        return assertions


class ContractCleanup(BaseModel):
    """Final UI state that must be freshly proven before environment restoration."""

    model_config = ConfigDict(extra="forbid")

    description: str
    assertions: list[RouteStep]
    proof_mode: ContractProofMode = "fresh_assertions"
    manual_completion_allowed: Literal[False] = False

    @field_validator("description")
    @classmethod
    def _nonempty_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator("assertions")
    @classmethod
    def _assertions_only(cls, assertions: list[RouteStep]) -> list[RouteStep]:
        if not assertions:
            raise ValueError("needs at least one assertion")
        invalid = [step.kind for step in assertions if step.kind not in {"assert", "assert-order"}]
        if invalid:
            raise ValueError(
                "accepts only `assert` and `assert_order`, got " + ", ".join(invalid)
            )
        return assertions


class SessionContract(BaseModel):
    """Versioned authored proof contract for one agent-driven session."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = SESSION_CONTRACT_VERSION
    checkpoints: list[ContractCheckpoint]
    cleanup: ContractCleanup | None = None

    @field_validator("checkpoints")
    @classmethod
    def _checkpoints_are_nonempty(
        cls, checkpoints: list[ContractCheckpoint]
    ) -> list[ContractCheckpoint]:
        if not checkpoints:
            raise ValueError("needs at least one checkpoint")
        return checkpoints

    @model_validator(mode="after")
    def _checkpoint_ids_are_unique(self) -> SessionContract:
        seen: set[str] = set()
        duplicates: list[str] = []
        for checkpoint in self.checkpoints:
            if checkpoint.id in seen and checkpoint.id not in duplicates:
                duplicates.append(checkpoint.id)
            seen.add(checkpoint.id)
        if duplicates:
            raise ValueError("duplicate checkpoint ids: " + ", ".join(duplicates))
        return self


def _mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UsageError(f"{where} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise UsageError(f"every key in {where} must be a string")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = sorted(key for key in mapping if key not in allowed)
    if unknown:
        raise UsageError(f"unknown {where} keys: " + ", ".join(unknown))


def _required_nonempty_string(mapping: dict[str, Any], key: str, *, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UsageError(f"{where} `{key}:` must be a non-empty string")
    return value.strip()


def _parse_assertions(value: Any, *, where: str) -> list[RouteStep]:
    if not isinstance(value, list) or not value:
        raise UsageError(f"{where} needs a non-empty `assertions:` list")
    parsed: list[RouteStep] = []
    for index, assertion in enumerate(value):
        # Flow parsing is the single source of truth for selectors, predicates, relational
        # assertions, timeouts, and canonical aliases.  Wrapping one item also makes an action
        # such as `tap:` parse normally, after which the explicit kind gate rejects it here.
        snippet = yaml_lib.safe_dump(
            {"schema_version": 1, "name": "contract_assertion", "steps": [assertion]},
            sort_keys=False,
            allow_unicode=True,
        )
        try:
            step = parse_flow_yaml(snippet, name="contract_assertion").steps[0]
        except UsageError as exc:
            raise UsageError(f"{where} assertion[{index}] is invalid: {exc.message}") from exc
        if step.kind not in {"assert", "assert-order"}:
            raise UsageError(
                f"{where} assertion[{index}] must use `assert` or `assert_order`, "
                f"not `{step.kind}`"
            )
        parsed.append(step)
    return parsed


def _parse_checkpoint(value: Any, *, index: int) -> ContractCheckpoint:
    where = f"contract checkpoint[{index}]"
    data = _mapping(value, where=where)
    _reject_unknown(data, {"id", "description", "assertions"}, where=f"checkpoint[{index}]")
    try:
        return ContractCheckpoint(
            id=_required_nonempty_string(data, "id", where=where),
            description=_required_nonempty_string(data, "description", where=where),
            assertions=_parse_assertions(data.get("assertions"), where=where),
        )
    except ValidationError as exc:  # pragma: no cover - parser checks give better messages
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise UsageError(f"invalid {where}: {detail}") from exc


def _parse_cleanup(value: Any) -> ContractCleanup | None:
    if value is None:
        return None
    where = "contract cleanup"
    data = _mapping(value, where=where)
    _reject_unknown(data, {"description", "assertions"}, where="cleanup")
    try:
        return ContractCleanup(
            description=_required_nonempty_string(data, "description", where=where),
            assertions=_parse_assertions(data.get("assertions"), where=where),
        )
    except ValidationError as exc:  # pragma: no cover - parser checks give better messages
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise UsageError(f"invalid {where}: {detail}") from exc


def parse_session_contract_yaml(text: str) -> SessionContract:
    """Parse authored contract YAML without touching a device or session store."""

    if not isinstance(text, str):
        raise UsageError("session contract YAML must be a string")
    try:
        loaded = yaml_lib.safe_load(text)
    except yaml_lib.YAMLError as exc:
        raise UsageError(f"session contract YAML does not parse: {exc}") from exc
    data = _mapping(loaded, where="session contract YAML")
    _reject_unknown(data, {"version", "checkpoints", "cleanup"}, where="top-level contract")

    version = data.get("version", SESSION_CONTRACT_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise UsageError("contract `version:` must be an integer")
    if version != SESSION_CONTRACT_VERSION:
        raise UsageError(
            f"unsupported contract version {version}; expected {SESSION_CONTRACT_VERSION}"
        )

    raw_checkpoints = data.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
        raise UsageError("session contract needs a non-empty `checkpoints:` list")
    checkpoints = [
        _parse_checkpoint(checkpoint, index=index)
        for index, checkpoint in enumerate(raw_checkpoints)
    ]
    try:
        return SessionContract(
            version=version,
            checkpoints=checkpoints,
            cleanup=_parse_cleanup(data.get("cleanup")),
        )
    except ValidationError as exc:
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise UsageError(f"invalid session contract: {detail}") from exc


def _render_assertions(assertions: list[RouteStep]) -> list[dict[str, Any] | str]:
    rendered = render_flow_yaml(Flow(name="contract_assertions", steps=assertions))
    document = yaml_lib.safe_load(rendered)
    return document["steps"]


def render_session_contract_yaml(contract: SessionContract) -> str:
    """Render a canonical, authored form without internal completion-policy fields."""

    # Revalidate programmatically created/copy-updated models at the public boundary.
    try:
        checked = SessionContract.model_validate(contract)
    except ValidationError as exc:
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise UsageError(f"invalid session contract: {detail}") from exc

    document: dict[str, Any] = {
        "version": checked.version,
        "checkpoints": [
            {
                "id": checkpoint.id,
                "description": checkpoint.description,
                "assertions": _render_assertions(checkpoint.assertions),
            }
            for checkpoint in checked.checkpoints
        ],
    }
    if checked.cleanup is not None:
        document["cleanup"] = {
            "description": checked.cleanup.description,
            "assertions": _render_assertions(checked.cleanup.assertions),
        }
    return yaml_lib.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)


def load_session_contract(
    *, file: str | Path | None = None, yaml: str | None = None
) -> SessionContract:
    """Load exactly one contract source for the future CLI/MCP integration seam."""

    if (file is None) == (yaml is None):
        raise UsageError("provide exactly one of contract `file` or inline `yaml`")
    if yaml is not None:
        return parse_session_contract_yaml(yaml)

    assert file is not None
    if isinstance(file, str) and not file.strip():
        raise UsageError("contract file path must be non-empty")
    path = Path(file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UsageError(f"could not read session contract {path}: {exc}") from exc
    return parse_session_contract_yaml(text)
