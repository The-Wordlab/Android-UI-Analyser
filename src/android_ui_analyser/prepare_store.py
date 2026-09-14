"""Where an interview and the scenario it produces are kept.

Preparation spans several short-lived CLI/MCP processes: AUA asks, the calling agent goes and
reads its own source, and comes back a minute later.  So the interview has to be durable between
calls, and it is kept beside the app's own memory - per app, because that is the scope at which
"we already know how this app signs in" is true.

A finished interview leaves a *scenario*: the contract, the setup plan, and the goal that produced
them.  That is the artefact the next run looks up instead of interviewing anyone again.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text
from .errors import UsageError
from .memory import AppMemoryStore
from .prepare import PrepareSession, scenario_name

SCENARIO_SCHEMA_VERSION = 1


class PrepareStore:
    """Durable interviews and saved scenarios for one app map."""

    def __init__(self, memory: AppMemoryStore) -> None:
        self.memory = memory

    # ------------------------------------------------------------------ interviews

    def _prepare_dir(self, package: str) -> Path:
        return self.memory.app_dir(package) / "prepare"

    def _prepare_path(self, package: str, prepare_id: str) -> Path:
        safe = "".join(char for char in prepare_id if char.isalnum() or char in "-_")
        if not safe or safe != prepare_id:
            raise UsageError(f"not a prepare id: {prepare_id!r}")
        return self._prepare_dir(package) / f"{safe}.json"

    def save_session(self, session: PrepareSession) -> Path:
        path = self._prepare_path(session.package, session.id)
        atomic_write_text(path, json.dumps(session.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return path

    def load_session(self, package: str, prepare_id: str) -> PrepareSession:
        path = self._prepare_path(package, prepare_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise UsageError(
                f"no prepare session {prepare_id} for {package}",
                hint="Run `aua prepare start --goal ... --app ...` first, or `aua prepare list`.",
            ) from exc
        except ValueError as exc:
            raise UsageError(f"prepare session {prepare_id} is not readable JSON") from exc
        return PrepareSession.from_dict(value)

    def list_sessions(self, package: str) -> list[dict[str, Any]]:
        directory = self._prepare_dir(package)
        out: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "prepare_id": value.get("id"),
                    "goal": value.get("goal"),
                    "created_at": value.get("created_at"),
                    "answered": sorted((value.get("answers") or {}).keys()),
                }
            )
        return out

    def discard_session(self, package: str, prepare_id: str) -> bool:
        path = self._prepare_path(package, prepare_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    # ------------------------------------------------------------------ scenarios

    def _scenario_dir(self, package: str) -> Path:
        return self.memory.app_dir(package) / "scenarios"

    def scenario_paths(self, package: str, name: str) -> tuple[Path, Path]:
        safe = scenario_name(name)
        directory = self._scenario_dir(package)
        return directory / f"{safe}.json", directory / f"{safe}.yaml"

    def save_scenario(
        self,
        *,
        package: str,
        name: str,
        goal: str,
        contract_yaml: str,
        setup: list[dict[str, Any]],
        answers: dict[str, str],
        provenance: list[dict[str, Any]],
        now: str | None = None,
    ) -> dict[str, Any]:
        meta_path, contract_path = self.scenario_paths(package, name)
        atomic_write_text(contract_path, contract_yaml)
        record = {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "name": scenario_name(name),
            "package": package,
            "goal": goal,
            "saved_at": now or datetime.now(UTC).isoformat(),
            "contract": str(contract_path),
            "setup": setup,
            "answers": answers,
            "provenance": provenance,
        }
        atomic_write_text(meta_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return record

    def load_scenario(self, package: str, name: str) -> dict[str, Any]:
        meta_path, _ = self.scenario_paths(package, name)
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise UsageError(
                f"no saved scenario `{scenario_name(name)}` for {package}",
                hint="`aua prepare list --app <package>` shows what has been prepared.",
            ) from exc
        except ValueError as exc:
            raise UsageError(f"scenario `{name}` is not readable JSON") from exc
        if value.get("schema_version") != SCENARIO_SCHEMA_VERSION:
            raise UsageError(
                f"scenario `{name}` was written by a different AUA "
                f"(schema {value.get('schema_version')}); re-prepare it"
            )
        return value

    def list_scenarios(self, package: str) -> list[dict[str, Any]]:
        directory = self._scenario_dir(package)
        out: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "scenario": value.get("name"),
                    "goal": value.get("goal"),
                    "saved_at": value.get("saved_at"),
                    "contract": value.get("contract"),
                    "run": f"aua prepare run {value.get('name')} --app {package}",
                }
            )
        return out


# --------------------------------------------------------------------------- operations
#
# One implementation for both front doors.  The CLI and the MCP server differ only in how the
# arguments arrive, and an interview whose behaviour depended on which one you used would be a
# trap for exactly the agents this feature exists to serve.


def start(
    store: AppMemoryStore,
    *,
    package: str,
    goal: str,
    context: str | None = None,
    now: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Open an interview, pre-filled from what this app's map already knows."""

    from .memory import AppMap
    from .prepare import interview, open_preparation

    app_map = store.load(package) or AppMap(package=package)
    session = open_preparation(
        package=package,
        goal=goal,
        app_map=app_map,
        context_id=context,
        now=now,
        session_id=session_id,
    )
    PrepareStore(store).save_session(session)
    return interview(session)


def answer(
    store: AppMemoryStore,
    *,
    package: str,
    prepare_id: str,
    answers: dict[str, str],
    remember: bool = True,
    artifacts_dir: str | None = None,
    agent: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Record answers.  Returns the remaining questions, or the finished scenario."""

    from .prepare import interview, is_ready, prepared, record_answers

    prepare = PrepareStore(store)
    session = prepare.load_session(package, prepare_id)
    record_answers(session, answers)
    prepare.save_session(session)
    if not is_ready(session):
        return interview(session)

    from .prepare import (
        build_contract_document,
        knowledge_writes,
        render_contract_yaml,
        scenario_name,
        setup_plan,
    )

    built = build_contract_document(session)
    name = scenario_name(session.goal)
    _, contract_path = prepare.scenario_paths(package, name)
    prepare.save_scenario(
        package=package,
        name=name,
        goal=session.goal,
        contract_yaml=render_contract_yaml(session),
        setup=setup_plan(session),
        answers=dict(session.answers),
        provenance=built["provenance"],
        now=now,
    )
    remembered: list[str] = []
    if remember:
        for item in knowledge_writes(session):
            saved = store.remember_knowledge(
                package,
                kind=item["kind"],
                text=item["text"],
                name=item["name"],
                aliases=item["aliases"],
                source="agent",
                agent=agent,
                session=session.id,
            )
            if saved is not None:
                remembered.append(saved.id)
    payload = prepared(session, contract_path=str(contract_path), artifacts_dir=artifacts_dir)
    payload["remembered_ids"] = remembered
    payload["saved"] = True
    # The interview has served its purpose; the scenario is the durable artefact. Keeping the
    # half-finished document around only invites a later answer against a contract that has
    # already been written and possibly edited by hand.
    prepare.discard_session(package, prepare_id)
    return payload


def show(store: AppMemoryStore, *, package: str, prepare_id: str) -> dict[str, Any]:
    """The current state of one interview, without changing it."""

    from .prepare import interview

    return interview(PrepareStore(store).load_session(package, prepare_id))


def catalogue(store: AppMemoryStore, *, package: str) -> dict[str, Any]:
    """Interviews in flight and scenarios already prepared for this app."""

    prepare = PrepareStore(store)
    return {
        "ok": True,
        "package": package,
        "in_progress": prepare.list_sessions(package),
        "scenarios": prepare.list_scenarios(package),
    }
