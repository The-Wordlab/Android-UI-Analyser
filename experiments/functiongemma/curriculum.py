"""Deterministic, privacy-safe curriculum for an AUA candidate-selection policy.

The model is deliberately *not* taught AUA's broad MCP surface.  A deterministic planner
supplies a small set of exact calls, including their risks and required proof, and the model
invokes one narrow function to select the best candidate.  This keeps execution validation and
safety in normal code while still letting a small model learn useful policy decisions.

All examples are synthetic and use only obviously fictional ``com.example`` applications.
No device, emulator, ADB process, transcript, or local AUA memory is read by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20_260_814
VARIANTS_PER_GROUP = 8
# Deliberately unrelated to the final closed-loop evaluator seed.  Fictional episode ordinals
# rotate through these seeds, so no held-out ID/order permutation can be reconstructed from the
# training generator even though every four-row invariant remains exactly counterbalanced.
SEQUENTIAL_CURRICULUM_SEEDS = (73_104_229, 91_337_041, 48_552_887)
SEQUENTIAL_SPLIT_SEED = 64_209_173
DEFAULT_SPLIT_SIZES: dict[str, int] = {
    "train": 12_288,
    "valid": 2_048,
    "test": 2_048,
}

LABELS = (
    "start_session",
    "act_from_fresh_observation",
    "input_from_fresh_observation",
    "await_semantic_evidence",
    "recover_ambiguous_mutation",
    "recover_stale_target",
    "probe_expected_error",
    "enter_verified_offline_state",
    "restore_environment",
    "finish_terminal_session",
    "sequence_start",
    "sequence_prepare_offline",
    "sequence_open_item",
    "sequence_recover_unknown",
    "sequence_restore",
    "sequence_finish",
)

PRESERVED_V1_LABELS = frozenset(
    {
        "start_session",
        "act_from_fresh_observation",
        "input_from_fresh_observation",
        "await_semantic_evidence",
        "recover_stale_target",
        "probe_expected_error",
        "restore_environment",
        "finish_terminal_session",
    }
)

PRESERVED_V2_LABELS = frozenset(
    {
        *PRESERVED_V1_LABELS,
        "recover_ambiguous_mutation",
        "enter_verified_offline_state",
    }
)

SEQUENTIAL_LABELS = frozenset(
    {
        "sequence_start",
        "sequence_prepare_offline",
        "sequence_open_item",
        "sequence_recover_unknown",
        "sequence_restore",
        "sequence_finish",
    }
)

_LABEL_CODES = {
    "start_session": "start",
    "act_from_fresh_observation": "act",
    "input_from_fresh_observation": "input",
    "await_semantic_evidence": "await",
    "recover_ambiguous_mutation": "ambiguous",
    "recover_stale_target": "recover",
    "probe_expected_error": "probe",
    "enter_verified_offline_state": "offline",
    "restore_environment": "restore",
    "finish_terminal_session": "finish",
    "sequence_start": "seq-start",
    "sequence_prepare_offline": "seq-offline",
    "sequence_open_item": "seq-open",
    "sequence_recover_unknown": "seq-recover",
    "sequence_restore": "seq-restore",
    "sequence_finish": "seq-finish",
}

SCENARIO_KIND_BY_LABEL = {
    "start_session": "standard",
    "act_from_fresh_observation": "standard",
    "input_from_fresh_observation": "standard",
    "await_semantic_evidence": "standard",
    "recover_ambiguous_mutation": "recovery",
    "recover_stale_target": "recovery",
    "probe_expected_error": "counterfactual",
    "enter_verified_offline_state": "reversible_mutation",
    "restore_environment": "cleanup",
    "finish_terminal_session": "terminal",
    "sequence_start": "sequential",
    "sequence_prepare_offline": "reversible_mutation",
    "sequence_open_item": "sequential",
    "sequence_recover_unknown": "recovery",
    "sequence_restore": "cleanup",
    "sequence_finish": "terminal",
}

CRITICALITY_BY_LABEL = {
    "start_session": "normal",
    "act_from_fresh_observation": "normal",
    "input_from_fresh_observation": "normal",
    "await_semantic_evidence": "normal",
    "recover_ambiguous_mutation": "critical",
    "recover_stale_target": "critical",
    "probe_expected_error": "critical",
    "enter_verified_offline_state": "critical",
    "restore_environment": "critical",
    "finish_terminal_session": "critical",
    "sequence_start": "normal",
    "sequence_prepare_offline": "critical",
    "sequence_open_item": "normal",
    "sequence_recover_unknown": "critical",
    "sequence_restore": "critical",
    "sequence_finish": "critical",
}

SELECT_CANDIDATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "select_candidate",
        "description": (
            "Select exactly one supplied candidate by its opaque integer ID. "
            "Do not invent, rewrite, or execute an AUA call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "integer",
                    "description": "The candidate_id of the single safest next call.",
                }
            },
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
    },
}

# A deliberately small, copied subset of the public MCP catalogue.  Tests compare these names
# and flat argument keys to ``mcp_server._tool_definitions`` so curriculum drift fails loudly.
# The model itself sees only ``select_candidate``; these schemas validate planner candidates.
PUBLIC_AUA_ARGUMENTS: dict[str, frozenset[str]] = {
    "capabilities": frozenset({"goal", "phase_done", "expect_error"}),
    "session_start": frozenset(
        {
            "goal",
            "contract_yaml",
            "artifacts_dir",
            "evidence",
            "junit",
            "wait_for_lease_s",
            "start_emulator",
            "headed",
            "audio",
            "animations",
            "avd",
            "needs",
            "package",
            "activity",
        }
    ),
    "session_progress": frozenset({"session_id", "phase_done", "expect_error"}),
    "session_review": frozenset({"session_id", "phase_done", "expect_error"}),
    "session_finish": frozenset(
        {"session_id", "allow_incomplete", "summary", "phase_done", "expect_error"}
    ),
    "analyze_screen": frozenset(
        {
            "source",
            "with_ocr",
            "query",
            "with_image",
            "no_cache",
            "phase_done",
            "expect_error",
        }
    ),
    "has": frozenset(
        {"text", "match", "ignore_case", "ocr_fallback", "by", "phase_done", "expect_error"}
    ),
    "tap_and_analyze": frozenset(
        {
            "id",
            "rid",
            "text",
            "desc",
            "stable_key",
            "bounds",
            "index",
            "first",
            "with_image",
            "observe_fields",
            "observe_meta",
            "until",
            "until_timeout",
            "until_poll",
            "phase_done",
            "expect_error",
        }
    ),
    "input_and_analyze": frozenset(
        {
            "id",
            "text",
            "submit",
            "send",
            "with_image",
            "observe_fields",
            "observe_meta",
            "until",
            "until_timeout",
            "until_poll",
            "phase_done",
            "expect_error",
        }
    ),
    "key_and_analyze": frozenset(
        {
            "name",
            "with_image",
            "observe_fields",
            "observe_meta",
            "until",
            "until_timeout",
            "until_poll",
            "phase_done",
            "expect_error",
        }
    ),
    "wait_changed_and_analyze": frozenset(
        {
            "timeout_ms",
            "interval_ms",
            "observe_fields",
            "observe_meta",
            "until",
            "until_timeout",
            "until_poll",
            "phase_done",
            "expect_error",
        }
    ),
    "await_and_analyze": frozenset(
        {
            "predicate",
            "timeout_ms",
            "poll_ms",
            "match",
            "ignore_case",
            "observe_fields",
            "observe_meta",
            "phase_done",
            "expect_error",
        }
    ),
    "reach": frozenset(
        {
            "goal",
            "until",
            "timeout_ms",
            "poll_ms",
            "allow_unsafe",
            "allow_destructive",
            "assist",
            "phase_done",
            "expect_error",
        }
    ),
    "resolve": frozenset({"target", "phase_done", "expect_error"}),
    "network_status": frozenset({"verify", "phase_done", "expect_error"}),
    "network_offline": frozenset({"verify", "timeout_ms", "phase_done", "expect_error"}),
    "network_restore": frozenset({"timeout_ms", "phase_done", "expect_error"}),
}

REQUIRED_AUA_ARGUMENTS: dict[str, frozenset[str]] = {
    "session_start": frozenset({"goal"}),
    "tap_and_analyze": frozenset(),  # one of id/rid/text/desc; checked separately
    "input_and_analyze": frozenset({"id", "text"}),
    "key_and_analyze": frozenset({"name"}),
    "await_and_analyze": frozenset({"predicate"}),
    "reach": frozenset({"goal"}),
    "resolve": frozenset({"target"}),
}

DEFAULT_DENYLIST = (
    "clipboard-",
    "production.example",
    "private-package",
    "real-customer",
    "access_token",
    "api_key",
    "bearer ",
)

_HOST_PATH = re.compile(r"/(?:Users|home|var/folders|data/user)/", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_PACKAGE = re.compile(r"\b(?:[a-z][a-z0-9_]*\.){2,}[a-z][a-z0-9_]*\b")
_RESOURCE_PACKAGE = re.compile(r"\b(?:[a-z][a-z0-9_]*\.)+[a-z][a-z0-9_]*:id/")
_ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9_+/=-]{40,}")

_APPS = (
    ("Example Catalog", "com.example.catalog"),
    ("Sample Notes", "com.example.notes"),
    ("Demo Planner", "com.example.planner"),
    ("Fixture Library", "com.example.library"),
    ("Example Recipes", "com.example.recipes"),
    ("Sample Trails", "com.example.trails"),
    ("Demo Inventory", "com.example.inventory"),
    ("Fixture Calendar", "com.example.calendar"),
)
_ITEMS = (
    "blue notebook",
    "cedar trail",
    "sample recipe",
    "demo task",
    "orange folder",
    "fixture card",
    "green marker",
    "example event",
)
_DETAIL_SCREENS = (
    "catalog_detail",
    "note_detail",
    "plan_detail",
    "book_detail",
    "recipe_detail",
    "trail_detail",
    "inventory_detail",
    "event_detail",
)
_REQUEST_STYLES = (
    "Choose the next action that advances the goal with direct proof.",
    "Select one safe call; avoid adding an observation the last result already returned.",
    "Pick the candidate that preserves state and produces the required evidence.",
    "Continue this test using exactly one of the supplied calls.",
    "Select the lowest-risk next step that still proves the checkpoint.",
    "Choose one candidate and rely on its returned observation when it has one.",
    "Advance the active phase without inventing a new call.",
    "Pick the single call whose proof matches the stated checkpoint.",
)
_HOLDOUT_REQUEST_STYLES = (
    "Decide which offered operation safely moves this multi-stage run forward.",
    "From the supplied choices, select the one justified by current state and obligations.",
)


@dataclass(frozen=True)
class Group:
    """One semantic episode, assigned to a split before paraphrase expansion."""

    group_id: str
    label: str
    ordinal: int


@dataclass(frozen=True)
class ExampleContext:
    app_name: str
    package: str
    item: str
    detail_screen: str
    group_number: int
    session_id: str
    fresh_id: int
    stale_id: int

    @property
    def item_rid(self) -> str:
        return f"{self.package}:id/openDetail"

    @property
    def input_rid(self) -> str:
        return f"{self.package}:id/searchInput"

    @property
    def result_rid(self) -> str:
        return f"{self.package}:id/resultsPanel"

    @property
    def detail_rid(self) -> str:
        return f"{self.package}:id/detailPanel"


def _stable_seed(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _context_for(group: Group) -> ExampleContext:
    label_index = LABELS.index(group.label)
    if group.label in SEQUENTIAL_LABELS:
        # Every phase of a sequential episode uses one fictional fixture identity.  The six
        # labels also share split membership, so this linkage can never cross a split boundary.
        app_name, package = _APPS[group.ordinal % len(_APPS)]
        item = _ITEMS[(group.ordinal * 3) % len(_ITEMS)]
        detail = _DETAIL_SCREENS[(group.ordinal * 5) % len(_DETAIL_SCREENS)]
        number = len(LABELS) * 128 + group.ordinal
    else:
        app_name, package = _APPS[(group.ordinal + label_index) % len(_APPS)]
        item = _ITEMS[(group.ordinal * 3 + label_index) % len(_ITEMS)]
        detail = _DETAIL_SCREENS[(group.ordinal * 5 + label_index) % len(_DETAIL_SCREENS)]
        number = label_index * 128 + group.ordinal
    fresh_id = 10 + (number * 7) % 80
    stale_id = 110 + (number * 11) % 80
    return ExampleContext(
        app_name=app_name,
        package=package,
        item=item,
        detail_screen=detail,
        group_number=number,
        session_id=f"session-example-{number:04d}",
        fresh_id=fresh_id,
        stale_id=stale_id,
    )


def _candidate(
    name: str,
    arguments: Mapping[str, Any],
    *,
    purpose: str,
    risk: str,
    proof: str,
    cleanup: str = "none",
    correct: bool = False,
    risk_level: str | None = None,
    authorized: bool | None = None,
    redundant: bool | None = None,
) -> dict[str, Any]:
    _validate_aua_call(name, arguments)
    folded_risk = risk.casefold()
    unsafe_markers = (
        "wrong element",
        "wrong control",
        "stale frame",
        "predates the transition",
        "double-activate",
        "guess",
        "mutates the environment",
        "too early",
        "unrelated ui mutation",
        "unauthorized",
    )
    redundant_markers = (
        "redundant",
        "wastes a call",
        "repeats evidence",
        "unnecessary",
        "duplicate",
        "adds a call",
        "another copy",
        "again",
    )
    risk_category = risk_level or (
        "unsafe" if any(marker in folded_risk for marker in unsafe_markers) else "safe"
    )
    if risk_category not in {"safe", "unsafe", "destructive"}:
        raise ValueError(f"invalid candidate risk level: {risk_category}")
    inferred_redundant = any(marker in folded_risk for marker in redundant_markers)
    return {
        "call": {"tool": name, "arguments": dict(arguments)},
        "purpose": purpose,
        "risk": risk_category,
        "risk_reason": risk,
        "authorized": (
            authorized if authorized is not None else correct or risk_category == "safe"
        ),
        "redundant": redundant
        if redundant is not None
        else False
        if correct
        else inferred_redundant,
        "proof": proof,
        "cleanup": cleanup,
        "_correct": correct,
    }


def _validate_aua_call(name: str, arguments: Mapping[str, Any]) -> None:
    if name not in PUBLIC_AUA_ARGUMENTS:
        raise ValueError(f"uncurated AUA tool: {name}")
    if "args" in arguments or "arguments" in arguments:
        raise ValueError(f"{name} arguments must be flat")
    unknown = set(arguments) - PUBLIC_AUA_ARGUMENTS[name]
    if unknown:
        raise ValueError(f"unknown {name} argument(s): {sorted(unknown)}")
    missing = REQUIRED_AUA_ARGUMENTS.get(name, frozenset()) - set(arguments)
    if missing:
        raise ValueError(f"missing {name} argument(s): {sorted(missing)}")
    if name == "tap_and_analyze" and not (
        {"id", "rid", "text", "desc", "stable_key"} & set(arguments)
    ):
        raise ValueError("tap_and_analyze requires id, rid, text, desc, or stable_key")


def _common_candidates(context: ExampleContext) -> list[dict[str, Any]]:
    return [
        _candidate(
            "analyze_screen",
            {"source": "auto"},
            purpose="Request another general observation.",
            risk="Redundant when the latest call already returned a fresh analyzed screen.",
            proof="A new screen dump, but no task checkpoint by itself.",
        ),
        _candidate(
            "session_review",
            {"session_id": context.session_id},
            purpose="Inspect efficiency accounting before the task is complete.",
            risk="Does not advance the active UI checkpoint.",
            proof="Call accounting only.",
        ),
    ]


def _sequential_candidate_specs(
    label: str,
    context: ExampleContext,
    goal: str,
) -> list[dict[str, Any]]:
    """Source-oracle choices for a multi-phase run, with no held-out fixture literals."""
    c = context
    finish_early = _candidate(
        "session_finish",
        {"session_id": c.session_id},
        purpose="Close lifecycle ownership before the remaining phase obligations are proved.",
        risk="Unauthorized early termination can strand required work or reversible state.",
        proof="A final review cannot replace missing phase evidence.",
        cleanup=(
            "network_restore remains required"
            if label in {"sequence_open_item", "sequence_recover_unknown", "sequence_restore"}
            else "none"
        ),
        risk_level="unsafe",
        authorized=False,
    )
    observe_redundant = _candidate(
        "analyze_screen",
        {"source": "auto"},
        purpose="Request another general screen description.",
        risk="The latest trustworthy observation already describes this phase's screen.",
        proof="Screen description only; no phase-specific transition.",
        redundant=True,
    )

    if label == "sequence_start":
        return [
            _candidate(
                "session_start",
                {"goal": goal, "package": c.package},
                purpose="Establish goal and cleanup ownership before any reversible change.",
                risk="Low; app data is preserved and an attached target is assumed.",
                proof="Returns active session state and the initial analyzed screen.",
                correct=True,
            ),
            observe_redundant,
            _candidate(
                "network_offline",
                {"verify": True},
                purpose="Enter offline mode before a session owns the saved network state.",
                risk="Unauthorized reversible mutation has no active lifecycle owner.",
                proof="Offline read-back without safe cleanup ownership.",
                cleanup="network_restore becomes mandatory",
                risk_level="unsafe",
                authorized=False,
            ),
            finish_early,
        ]

    if label == "sequence_prepare_offline":
        return [
            _candidate(
                "network_offline",
                {"verify": True},
                purpose="Create the required offline condition under active session ownership.",
                risk="Low; the reversible state is owned and restoration remains explicit.",
                proof="Read-back establishes absence of an active default network.",
                cleanup="network_restore becomes mandatory",
                correct=True,
            ),
            _candidate(
                "network_status",
                {"verify": True},
                purpose="Read connectivity without creating the required offline condition.",
                risk="Safe but does not advance the offline checkpoint.",
                proof="Baseline status only.",
            ),
            observe_redundant,
            finish_early,
        ]

    if label == "sequence_open_item":
        return [
            _candidate(
                "tap_and_analyze",
                {"rid": c.item_rid, "until": f"rid:{c.detail_rid},!text:Loading"},
                purpose=f"Open {c.item} once using a current stable selector.",
                risk="Low; this is the authorized mutation for the active phase.",
                proof="Normally returns settled detail evidence in the same call.",
                correct=True,
            ),
            _candidate(
                "tap_and_analyze",
                {"id": c.stale_id},
                purpose="Act through an integer retained from an expired frame.",
                risk="Stale frame identity makes this action unauthorized and unsafe.",
                proof="No current target identity or semantic arrival predicate.",
                risk_level="unsafe",
                authorized=False,
            ),
            observe_redundant,
            finish_early,
        ]

    if label == "sequence_recover_unknown":
        return [
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Read current UI truth before deciding whether any retry is needed.",
                risk="Low; the read-only recovery cannot duplicate the uncertain mutation.",
                proof="A fresh screen distinguishes landed action from unchanged origin.",
                correct=True,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": c.item_rid},
                purpose="Repeat the action whose first outcome is still unknown.",
                risk="Unauthorized replay can apply the same mutation twice.",
                proof="A later result cannot attribute which attempt caused the state.",
                risk_level="unsafe",
                authorized=False,
                redundant=True,
            ),
            _candidate(
                "session_progress",
                {"session_id": c.session_id},
                purpose="Consult phase bookkeeping instead of observing the uncertain UI.",
                risk="Unauthorized assumption leaves the post-mutation truth unresolved.",
                proof="Plan state is not device arrival evidence.",
                risk_level="unsafe",
                authorized=False,
            ),
            finish_early,
        ]

    if label == "sequence_restore":
        return [
            _candidate(
                "network_restore",
                {"timeout_ms": 15_000},
                purpose="Restore the saved connectivity controls and verify the read-back.",
                risk="Low; this discharges the active reversible-state obligation.",
                proof="The saved online condition is restored before terminal cleanup.",
                cleanup="completes required network cleanup",
                correct=True,
            ),
            _candidate(
                "network_status",
                {"verify": True},
                purpose="Read the leaked offline state without repairing it.",
                risk="Safe inspection is redundant while cleanup remains actionable.",
                proof="Status only; restoration is still unproved.",
                cleanup="network_restore remains required",
                redundant=True,
            ),
            _candidate(
                "network_offline",
                {"verify": True},
                purpose="Apply the already active reversible mutation again.",
                risk="Unauthorized duplicate mutation cannot discharge cleanup ownership.",
                proof="Another offline read-back only.",
                cleanup="network_restore remains required",
                risk_level="unsafe",
                authorized=False,
                redundant=True,
            ),
            finish_early,
        ]

    if label == "sequence_finish":
        return [
            _candidate(
                "session_finish",
                {"session_id": c.session_id},
                purpose="Close lifecycle ownership after goal proof and cleanup both passed.",
                risk="Low; no required phase or reversible state remains open.",
                proof="Terminal accounting confirms the completed run is closed.",
                correct=True,
            ),
            _candidate(
                "session_review",
                {"session_id": c.session_id},
                purpose="Inspect accounting but leave the completed lifecycle active.",
                risk="Safe but redundant after every checkpoint is satisfied.",
                proof="A non-terminal review only.",
                redundant=True,
            ),
            observe_redundant,
            _candidate(
                "network_restore",
                {"timeout_ms": 15_000},
                purpose="Repeat connectivity cleanup that already has read-back proof.",
                risk="Safe but redundant because no cleanup obligation remains.",
                proof="Duplicate restoration evidence.",
                redundant=True,
            ),
        ]

    raise ValueError(f"unknown sequential label: {label}")


def _candidate_specs(label: str, context: ExampleContext, goal: str) -> list[dict[str, Any]]:
    c = context
    if label in SEQUENTIAL_LABELS:
        return _sequential_candidate_specs(label, context, goal)
    if label == "start_session":
        return [
            _candidate(
                "session_start",
                {"goal": goal, "package": c.package},
                purpose="Start goal-aware work, launch the fictional fixture, and observe once.",
                risk="Low; preserves app data and does not start another emulator.",
                proof="Returns a session id, initial observation, phases, and safest next call.",
                correct=True,
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Inspect the current screen without creating a goal session.",
                risk="Loses lifecycle ownership, phase guidance, and automatic cleanup.",
                proof="Only a screen observation.",
            ),
            _candidate(
                "capabilities",
                {"goal": goal},
                purpose="List tool capabilities for the requested goal.",
                risk="Discovery is unnecessary because session_start already performs it.",
                proof="A ranked catalogue, not an active session or app observation.",
            ),
            _candidate(
                "network_offline",
                {"verify": True},
                purpose="Disable network before a session owns the reversible state.",
                risk="Mutates the environment too early and creates avoidable cleanup risk.",
                proof="Offline state only.",
                cleanup="network_restore would become mandatory",
            ),
            _candidate(
                "reach",
                {"goal": c.detail_screen},
                purpose="Navigate directly to a semantic destination.",
                risk="No session is active and the starting app state has not been observed.",
                proof="Destination evidence only if a safe route already exists.",
            ),
            _candidate(
                "session_finish",
                {},
                purpose="Finish an existing session.",
                risk="There is no active session to finish.",
                proof="No starting observation or test progress.",
            ),
        ]

    if label == "act_from_fresh_observation":
        return [
            _candidate(
                "tap_and_analyze",
                {
                    "rid": c.item_rid,
                    "until": f"rid:{c.detail_rid},!text:Loading",
                    "until_timeout": 8_000,
                },
                purpose=f"Open the {c.item} detail using a stable selector and folded arrival wait.",
                risk="Low; the selector is present in the fresh observation.",
                proof=f"The returned observation contains {c.detail_rid} without Loading.",
                correct=True,
            ),
            _candidate(
                "tap_and_analyze",
                {"id": c.stale_id},
                purpose="Tap an integer copied from an older frame.",
                risk="The frame changed, so the numeric id can target the wrong element or be rejected.",
                proof="No reliable destination predicate.",
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Re-observe before acting.",
                risk="Wastes a call because the previous action already returned fresh ids.",
                proof="Another copy of the current screen, not the detail checkpoint.",
            ),
            _candidate(
                "has",
                {"text": c.item, "match": "contains"},
                purpose="Check whether the visible item text exists.",
                risk="Repeats evidence already present and does not open the item.",
                proof="Presence only.",
            ),
            _candidate(
                "reach",
                {"goal": c.detail_screen, "until": f"rid:{c.detail_rid}"},
                purpose="Ask route memory to reach the detail screen.",
                risk="Unnecessary indirection when the target is already actionable on screen.",
                proof="Arrival only if a stored route exists.",
            ),
            _candidate(
                "key_and_analyze",
                {"name": "enter"},
                purpose="Press Enter without focusing the target.",
                risk="May activate an unrelated focused control.",
                proof="No target-specific evidence.",
            ),
        ]

    if label == "input_from_fresh_observation":
        return [
            _candidate(
                "input_and_analyze",
                {
                    "id": c.fresh_id,
                    "text": c.item,
                    "submit": True,
                    "until": f"rid:{c.result_rid},!text:Loading",
                    "until_timeout": 10_000,
                },
                purpose="Type into the fresh search field, submit, and wait for settled results.",
                risk="Low; the id belongs to the current returned observation.",
                proof=f"The returned observation contains {c.result_rid} without Loading.",
                correct=True,
            ),
            _candidate(
                "input_and_analyze",
                {"id": c.stale_id, "text": c.item, "submit": True},
                purpose="Type into a field id retained from an earlier frame.",
                risk="Stale frame-local id; input could be rejected or land in the wrong control.",
                proof="No settled-results predicate.",
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": c.input_rid},
                purpose="Focus the search field without entering the requested text.",
                risk="Adds a call even though input_and_analyze focuses before typing.",
                proof="Only field focus.",
            ),
            _candidate(
                "key_and_analyze",
                {"name": "enter"},
                purpose="Submit whatever text happens to be focused.",
                risk="The required query has not been entered.",
                proof="No query-specific result evidence.",
            ),
            _candidate(
                "analyze_screen",
                {"query": c.item},
                purpose="Search the observation for the desired query text.",
                risk="Does not type into the visible input field.",
                proof="Element match only.",
            ),
            _candidate(
                "await_and_analyze",
                {"predicate": f"rid:{c.result_rid}", "timeout_ms": 2_000},
                purpose="Wait for results without submitting a query.",
                risk="No action has been taken that could make results appear.",
                proof="Likely timeout rather than the requested search.",
            ),
        ]

    if label == "await_semantic_evidence":
        return [
            _candidate(
                "await_and_analyze",
                {
                    "predicate": f"rid:{c.detail_rid},!text:Loading",
                    "timeout_ms": 30_000,
                    "poll_ms": 300,
                },
                purpose="Wait for positive arrival evidence and disappearance of Loading.",
                risk="Low; bounded semantic polling does not repeat the mutating action.",
                proof=f"Per-term evidence for {c.detail_rid} and absence of Loading.",
                correct=True,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": c.item_rid},
                purpose="Repeat the tap that already initiated navigation.",
                risk="May double-activate the control or navigate twice.",
                proof="No explicit settled-state predicate.",
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Poll the screen once immediately.",
                risk="A single observation can capture the same transitional frame.",
                proof="No bounded arrival decision.",
            ),
            _candidate(
                "wait_changed_and_analyze",
                {"timeout_ms": 15_000, "interval_ms": 150},
                purpose="Wait for any hierarchy change.",
                risk="An unrelated animation can satisfy a generic change.",
                proof="Change only, not the named destination.",
            ),
            _candidate(
                "has",
                {"text": "Loading", "match": "exact"},
                purpose="Check the transitional label once.",
                risk="Does not wait for positive destination evidence.",
                proof="One instantaneous text check.",
            ),
            _candidate(
                "session_review",
                {"session_id": c.session_id},
                purpose="Review call efficiency while navigation is unfinished.",
                risk="Does not settle or verify the active phase.",
                proof="Accounting only.",
            ),
        ]

    if label == "recover_ambiguous_mutation":
        return [
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Re-establish current screen truth after an ambiguous mutating result.",
                risk="Low; this read-only observation avoids replaying an action that may have landed.",
                proof="A fresh hierarchy shows whether the destination or original screen is current.",
                correct=True,
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": c.item_rid, "until": f"rid:{c.detail_rid}"},
                purpose="Replay the mutation whose outcome is unknown.",
                risk="Unauthorized replay can double-activate the target if the first tap landed.",
                proof="A later destination cannot distinguish one activation from two.",
            ),
            _candidate(
                "session_progress",
                {"session_id": c.session_id},
                purpose="Read the goal checkpoint without refreshing UI truth.",
                risk="The phase plan cannot resolve whether the device accepted the mutation.",
                proof="Planner state only, not a current-screen observation.",
            ),
            _candidate(
                "session_finish",
                {"session_id": c.session_id},
                purpose="Close the session while the mutation outcome remains unknown.",
                risk="Premature finish abandons the required arrival or recovery proof.",
                proof="Terminal accounting without resolving the ambiguous UI state.",
            ),
        ]

    if label == "recover_stale_target":
        return [
            _candidate(
                "resolve",
                {"target": c.stale_id},
                purpose="Remap the previous-frame target onto the current screen before acting.",
                risk="Low; refuses a remap it cannot justify.",
                proof="Returns a current frame-local target or a structured refusal.",
                correct=True,
            ),
            _candidate(
                "tap_and_analyze",
                {"id": c.stale_id},
                purpose="Reuse the expired numeric id directly.",
                risk="Numeric ids are frame-local and this one predates the transition.",
                proof="No safe identity proof.",
            ),
            _candidate(
                "tap_and_analyze",
                {"id": c.fresh_id},
                purpose="Guess that a current numeric id represents the old target.",
                risk="Matching by coincidental position or number can activate the wrong element.",
                proof="No remap evidence.",
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Get another frame without linking it to the prior target.",
                risk="Leaves the target identity unresolved and invites a later guess.",
                proof="Screen state only.",
            ),
            _candidate(
                "has",
                {"text": c.item, "match": "contains"},
                purpose="Look for similar visible text.",
                risk="Duplicate text can match a different element.",
                proof="Text presence, not target identity.",
            ),
            _candidate(
                "key_and_analyze",
                {"name": "back"},
                purpose="Abandon the changed frame.",
                risk="Moves away from the goal instead of safely recovering the target.",
                proof="A different screen, not a remapped target.",
            ),
        ]

    if label == "probe_expected_error":
        return [
            _candidate(
                "tap_and_analyze",
                {"id": c.stale_id, "expect_error": "stale_element_id"},
                purpose="Run the requested negative probe and annotate its exact expected error.",
                risk="Intentional non-mutation; safe only because this phase requires the rejection.",
                proof="The invocation returns the machine-readable stale_element_id error.",
                correct=True,
            ),
            _candidate(
                "resolve",
                {"target": c.stale_id},
                purpose="Recover the stale target for a successful later action.",
                risk="Defeats the counterfactual requirement to prove stale-id rejection.",
                proof="A remapped target, not the expected error.",
            ),
            _candidate(
                "tap_and_analyze",
                {"id": c.stale_id},
                purpose="Invoke the stale id without declaring the expected error.",
                risk="The intentional failure is counted as unexplained damage.",
                proof="An error without the required expected-error annotation.",
            ),
            _candidate(
                "tap_and_analyze",
                {"rid": c.item_rid},
                purpose="Use a stable selector so the tap succeeds.",
                risk="Tests normal action success instead of stale-id rejection.",
                proof="A navigation result, not the requested counterfactual.",
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto", "expect_error": "stale_element_id"},
                purpose="Attach the expected error to a non-action observation.",
                risk="analyze_screen does not exercise stale element validation.",
                proof="A normal observation, so the error expectation cannot match.",
            ),
            _candidate(
                "session_finish",
                {"session_id": c.session_id},
                purpose="End the session before the negative assertion.",
                risk="Leaves the required counterfactual phase untested.",
                proof="Only a final review.",
            ),
        ]

    if label == "enter_verified_offline_state":
        return [
            _candidate(
                "network_offline",
                {"verify": True, "timeout_ms": 10_000},
                purpose="Save network controls, enter offline state, and verify no default network.",
                risk="Low; the active session owns this authorized reversible mutation.",
                proof="Read-back proves transports are disabled and no default network remains.",
                cleanup="network_restore is recorded as a mandatory session-owned cleanup",
                correct=True,
            ),
            _candidate(
                "session_finish",
                {"session_id": c.session_id},
                purpose="Finish before exercising the authorized offline checkpoint.",
                risk="Premature finish leaves the required offline behavior untested.",
                proof="Terminal accounting only.",
            ),
            _candidate(
                "network_status",
                {"verify": True},
                purpose="Read the connected baseline without entering offline state.",
                risk="Read-only status cannot prove the app's offline behavior.",
                proof="Current network state only.",
            ),
            _candidate(
                "network_offline",
                {"verify": False, "timeout_ms": 10_000},
                purpose="Choose a redundant unverified offline alternative.",
                risk="Redundant mutation drops the direct verification required by the checkpoint.",
                proof="No verified default-network absence.",
                cleanup="network_restore would still be mandatory",
            ),
        ]

    if label == "restore_environment":
        return [
            _candidate(
                "network_restore",
                {"timeout_ms": 15_000},
                purpose="Restore and verify the network controls saved by the offline phase.",
                risk="Low; this is the paired cleanup for network_offline.",
                proof="Read-back confirms the saved network state was restored.",
                cleanup="completes the outstanding network cleanup",
                correct=True,
            ),
            _candidate(
                "network_status",
                {"verify": True},
                purpose="Read the current offline state without restoring it.",
                risk="Confirms leaked state but leaves it active.",
                proof="Status only.",
                cleanup="network_restore still required",
            ),
            _candidate(
                "network_offline",
                {"verify": True},
                purpose="Apply the already-active offline mutation again.",
                risk="Redundant mutation; does not discharge cleanup ownership.",
                proof="Offline state only.",
                cleanup="network_restore still required",
            ),
            _candidate(
                "session_finish",
                {"session_id": c.session_id},
                purpose="Finish and rely on aggregate session cleanup.",
                risk="Closes the session before the explicit restored-state checkpoint is proved.",
                proof="Final review, but not the active cleanup phase's direct read-back.",
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Inspect the app while the device remains offline.",
                risk="Does not restore host-visible network controls.",
                proof="UI state only.",
                cleanup="network_restore still required",
            ),
            _candidate(
                "key_and_analyze",
                {"name": "home"},
                purpose="Leave the app UI.",
                risk="The reversible network mutation survives the navigation key.",
                proof="Home screen only.",
                cleanup="network_restore still required",
            ),
        ]

    if label == "finish_terminal_session":
        return [
            _candidate(
                "session_finish",
                {"session_id": c.session_id},
                purpose="Finish the completed session and return its final efficiency review.",
                risk="Low; all required phases and explicit cleanup checks are complete.",
                proof="Terminal review confirms closure and no outstanding reversible state.",
                cleanup="session-owned residual cleanup is finalized",
                correct=True,
            ),
            _candidate(
                "session_review",
                {"session_id": c.session_id},
                purpose="Review the session without finishing it.",
                risk="Leaves lifecycle ownership open after all work is complete.",
                proof="A pre-finish accounting snapshot.",
                cleanup="session_finish still required",
            ),
            _candidate(
                "session_progress",
                {"session_id": c.session_id},
                purpose="Ask for another active checkpoint.",
                risk="All phases are already complete, so this adds no evidence.",
                proof="Progress state only.",
                cleanup="session_finish still required",
            ),
            _candidate(
                "network_restore",
                {"timeout_ms": 15_000},
                purpose="Restore network controls again.",
                risk="The explicit cleanup phase already proved restoration.",
                proof="Duplicate network read-back.",
            ),
            _candidate(
                "analyze_screen",
                {"source": "auto"},
                purpose="Capture one more app observation.",
                risk="No terminal requirement calls for another screen dump.",
                proof="UI state only.",
                cleanup="session_finish still required",
            ),
            _candidate(
                "key_and_analyze",
                {"name": "home"},
                purpose="Navigate to the launcher before finishing.",
                risk="Adds an unrelated UI mutation after proof is complete.",
                proof="Launcher observation only.",
                cleanup="session_finish still required",
            ),
        ]

    raise ValueError(f"unknown label: {label}")


def _sequential_scenario_state(
    group: Group,
    context: ExampleContext,
    *,
    holdout: bool,
) -> tuple[str, dict[str, Any]]:
    """Build one phase of a linked source-oracle episode in runtime-shaped vocabulary."""
    c = context
    goal = (
        f"Complete a disconnected-mode journey for {c.item}: reach its information view, "
        "reinstate networking, then end cleanly."
        if holdout
        else (
            f"While offline, open {c.item}, prove its detail view, restore connectivity, "
            "and close the test session."
        )
    )
    worlds = {
        "sequence_start": {
            "phase": "not_started",
            "screen": "sample_home",
            "fresh": True,
            "session": False,
            "network": "online",
            "outcome": "known",
            "checkpoint": False,
            "cleanup": False,
        },
        "sequence_prepare_offline": {
            "phase": "prepare_offline",
            "screen": "sample_home",
            "fresh": True,
            "session": True,
            "network": "online",
            "outcome": "known",
            "checkpoint": False,
            "cleanup": False,
        },
        "sequence_open_item": {
            "phase": "open_record",
            "screen": "item_list",
            "fresh": True,
            "session": True,
            "network": "offline",
            "outcome": "known",
            "checkpoint": False,
            "cleanup": True,
        },
        "sequence_recover_unknown": {
            "phase": "recover_unknown",
            "screen": "item_list",
            "fresh": False,
            "session": True,
            "network": "offline",
            "outcome": "unknown",
            "checkpoint": False,
            "cleanup": True,
        },
        "sequence_restore": {
            "phase": "restore_environment",
            "screen": "item_detail",
            "fresh": True,
            "session": True,
            "network": "offline",
            "outcome": "known",
            "checkpoint": True,
            "cleanup": True,
        },
        "sequence_finish": {
            "phase": "finish",
            "screen": "item_detail",
            "fresh": True,
            "session": True,
            "network": "online",
            "outcome": "known",
            "checkpoint": True,
            "cleanup": False,
        },
    }
    world = worlds[group.label]
    if holdout:
        recent_outcomes = [
            "Goal session is active." if world["session"] else "No goal session is active.",
            f"Connectivity is currently {world['network']}.",
            (
                "The last mutation has an unresolved result."
                if world["outcome"] == "unknown"
                else "No action result is awaiting recovery."
            ),
            (
                "The destination checkpoint has direct evidence."
                if world["checkpoint"]
                else "The destination checkpoint is not yet proved."
            ),
        ]
        constraints = [
            "Choose one offered operation",
            "Tie progress to observable evidence",
        ]
        if world["cleanup"]:
            constraints.append("Discharge the saved connectivity obligation before closure")
        if world["outcome"] == "unknown":
            constraints.append("Inspect current state before authorizing another mutation")
    else:
        recent_outcomes = [
            f"session_active={str(world['session']).lower()}",
            f"network={world['network']}",
            f"outcome={world['outcome']}",
            f"goal_checkpoint_reached={str(world['checkpoint']).lower()}",
        ]
        constraints = ["Select one supplied call", "Require phase-specific proof"]
        if world["cleanup"]:
            constraints.append("Restore owned network state before terminal closure")
        if world["outcome"] == "unknown":
            constraints.append("Observe current truth before any replay")

    return goal, {
        "fixture_ref": f"fixture-sequence-{group.ordinal:03d}",
        "phase": world["phase"],
        "observation": {
            "known_screen": world["screen"],
            "fresh": world["fresh"],
        },
        "recent_outcomes": recent_outcomes,
        "constraints": constraints,
    }


def _scenario_state(
    group: Group,
    variant: int,
    *,
    holdout: bool = False,
) -> tuple[str, dict[str, Any]]:
    c = _context_for(group)
    if group.label in SEQUENTIAL_LABELS:
        goal, state = _sequential_scenario_state(group, c, holdout=holdout)
        state["request"] = _HOLDOUT_REQUEST_STYLES[variant] if holdout else _REQUEST_STYLES[variant]
        state["goal"] = goal
        return goal, state
    fixture_ref = f"fixture-{c.group_number:04d}"
    if group.label == "start_session":
        goal = f"Open {c.item} in {c.app_name} and prove the detail screen without clearing data."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "not_started",
            "observation": None,
            "recent_outcomes": ["A device is attached; no AUA goal session is active."],
            "constraints": ["Preserve app data", "Do not start another emulator"],
        }
    elif group.label == "act_from_fresh_observation":
        goal = f"Open the {c.item} detail and prove that loading has settled."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "open_detail",
            "observation": {
                "known_screen": "catalog",
                "fresh": True,
                "elements": [
                    {"id": c.fresh_id, "text": c.item, "rid": c.item_rid, "enabled": True}
                ],
            },
            "recent_outcomes": ["The previous action returned this analyzed observation."],
            "constraints": ["Use fresh evidence", "Prefer a stable selector", "Prove arrival"],
        }
    elif group.label == "input_from_fresh_observation":
        goal = f"Search for {c.item} and prove settled results."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "submit_search",
            "observation": {
                "known_screen": "search",
                "fresh": True,
                "elements": [{"id": c.fresh_id, "text": "", "rid": c.input_rid, "editable": True}],
            },
            "recent_outcomes": ["The current input id came from the latest analyzed result."],
            "constraints": ["Enter the exact query", "Submit once", "Wait for settled results"],
        }
    elif group.label == "await_semantic_evidence":
        goal = f"Prove that navigation to {c.detail_screen} completes after the existing tap."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "prove_arrival",
            "observation": {
                "known_screen": "transition",
                "fresh": True,
                "elements": [{"id": c.fresh_id, "text": "Loading", "enabled": False}],
            },
            "recent_outcomes": ["A single tap succeeded and returned this transitional frame."],
            "constraints": ["Do not repeat the mutation", "Require positive and negative evidence"],
        }
    elif group.label == "recover_ambiguous_mutation":
        goal = f"Determine whether the tap to open {c.item} landed without replaying it."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "recover_ambiguous_action",
            "observation": {
                "known_screen": "unknown",
                "fresh": False,
                "elements": [],
                "reason": "the mutating result did not contain trustworthy post-action state",
            },
            "recent_outcomes": [
                "The tap was dispatched once, but its completion response was ambiguous.",
                "The UI may be on either the original screen or the detail screen.",
            ],
            "constraints": [
                "Do not replay an action that may have landed",
                "Establish current screen truth before choosing another mutation",
            ],
        }
    elif group.label == "recover_stale_target":
        goal = f"Recover the previously observed {c.item} target after the screen changed."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "recover_target",
            "observation": {"known_screen": "catalog_updated", "fresh": True, "elements": []},
            "recent_outcomes": [
                f"Target id {c.stale_id} came from the previous frame.",
                "A transition invalidated numeric action ids before the target was used.",
            ],
            "constraints": ["Do not guess coordinates or ids", "Refuse an unjustified remap"],
        }
    elif group.label == "probe_expected_error":
        goal = "Verify that an expired numeric element id is rejected as stale."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "negative_stale_id_probe",
            "observation": {"known_screen": "catalog_updated", "fresh": True, "elements": []},
            "recent_outcomes": [f"Id {c.stale_id} is intentionally retained from an older frame."],
            "constraints": [
                "Exercise that stale id without remapping",
                "Annotate the exact expected error",
                "Do not treat the expected rejection as damage",
            ],
        }
    elif group.label == "enter_verified_offline_state":
        goal = "Enter a verified offline state for the authorized connectivity checkpoint."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "enter_offline",
            "session": {
                "id": c.session_id,
                "active": True,
                "owns_reversible_state": True,
            },
            "observation": {"known_screen": c.detail_screen, "fresh": True},
            "recent_outcomes": [
                "The goal session is active and owns subsequent reversible mutations.",
                "network_status verified a connected baseline; offline has not been applied.",
            ],
            "constraints": [
                "Mutate only under active session ownership",
                "Verify the transition",
                "Carry network_restore as mandatory cleanup",
            ],
            "cleanup_obligations": [],
        }
    elif group.label == "restore_environment":
        goal = "Restore the saved network controls and prove cleanup before ending the session."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "restore_network",
            "observation": {"known_screen": c.detail_screen, "fresh": True},
            "recent_outcomes": [
                "The offline assertion passed.",
                "network_offline owns a saved reversible network snapshot.",
            ],
            "constraints": ["Restore before finish", "Require state read-back"],
        }
    elif group.label == "finish_terminal_session":
        goal = "Close the completed test session without adding unrelated calls."
        state = {
            "fixture_ref": fixture_ref,
            "phase": "complete",
            "observation": {"known_screen": c.detail_screen, "fresh": True},
            "recent_outcomes": [
                "Every goal phase has direct evidence.",
                "The explicit network cleanup checkpoint passed.",
            ],
            "constraints": ["Finish once", "Return final accounting", "Leave no owned state"],
        }
    else:  # pragma: no cover - guarded by LABELS and _candidate_specs
        raise ValueError(f"unknown label: {group.label}")

    state["request"] = _REQUEST_STYLES[variant]
    state["goal"] = goal
    return goal, state


def _build_record(group: Group, split: str, variant: int, seed: int) -> dict[str, Any]:
    # Four rows share exactly the same semantic state and candidate set.  Their opaque IDs are
    # a hidden permutation of 0..3, so any policy that ignores candidate contents is forced to
    # chance accuracy.  The other quartet gives the group a second wording without weakening
    # that invariant.
    semantic_variant = variant // 4
    permutation_member = variant % 4
    holdout = split == "test" and group.label in SEQUENTIAL_LABELS
    goal, state = _scenario_state(group, semantic_variant, holdout=holdout)
    context = _context_for(group)
    specs = _candidate_specs(group.label, context, goal)
    if group.label in SEQUENTIAL_LABELS:
        # Match the live Candidate.as_prompt_value schema exactly.  Older source-oracle rows
        # retain risk_reason as extra explanatory supervision; runtime-shaped rows do not.
        for spec in specs:
            spec.pop("risk_reason", None)
    correct = [spec for spec in specs if spec.pop("_correct")]
    if len(correct) != 1:
        raise AssertionError(f"{group.label} must have exactly one correct candidate")
    target_spec = correct[0]

    permutation_seed = (
        SEQUENTIAL_CURRICULUM_SEEDS[group.ordinal % len(SEQUENTIAL_CURRICULUM_SEEDS)]
        if group.label in SEQUENTIAL_LABELS
        else seed
    )
    set_rng = random.Random(
        _stable_seed("candidate-set-v1", permutation_seed, group.group_id, semantic_variant)
    )
    distractors = [spec for spec in specs if spec is not target_spec]
    set_rng.shuffle(distractors)
    distractors = distractors[:3]
    candidate_count = len(distractors) + 1

    candidate_set = [target_spec, *distractors]
    order_rng = random.Random(
        _stable_seed("candidate-order-v1", permutation_seed, group.group_id, semantic_variant)
    )
    order_rng.shuffle(candidate_set)
    position_permutation = order_rng.sample(range(candidate_count), candidate_count)
    target_position = position_permutation[permutation_member]
    base_target_position = candidate_set.index(target_spec)
    rotation = (target_position - base_target_position) % candidate_count
    ordered = candidate_set[-rotation:] + candidate_set[:-rotation] if rotation else candidate_set

    # This permutation is intentionally salted and unrelated to fixture numbering, label order,
    # request wording, or display position.  It is not serialized anywhere in the prompt.
    id_rng = random.Random(
        _stable_seed(
            "opaque-candidate-id-v1",
            permutation_seed,
            group.group_id,
            semantic_variant,
        )
    )
    target_id_permutation = id_rng.sample(range(candidate_count), candidate_count)
    target_candidate_id = target_id_permutation[permutation_member]
    other_ids = [value for value in range(candidate_count) if value != target_candidate_id]
    id_rng.shuffle(other_ids)

    emitted_candidates: list[dict[str, Any]] = []
    id_cursor = iter(other_ids)
    for spec in ordered:
        candidate_id = target_candidate_id if spec is target_spec else next(id_cursor)
        emitted = {"id": candidate_id, **copy.deepcopy(spec)}
        emitted_candidates.append(emitted)

    state["candidates"] = emitted_candidates
    target_call = copy.deepcopy(target_spec["call"])
    record = {
        "id": f"fg-{split}-{group.group_id}-v{variant}",
        "messages": [
            {
                "role": "developer",
                "content": (
                    "You are a model that can do function calling with the following functions. "
                    "You are an AUA policy selector for Android UI testing. Select exactly one "
                    "supplied candidate. Candidate IDs are opaque and their order is arbitrary. "
                    "Prefer direct semantic proof, current observations, bounded waits, and "
                    "required cleanup. Never invent or rewrite a call."
                ),
            },
            {"role": "user", "content": _canonical_json(state)},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_select_candidate",
                        "type": "function",
                        "function": {
                            "name": "select_candidate",
                            "arguments": {"candidate_id": target_candidate_id},
                        },
                    }
                ],
            },
        ],
        "tools": [copy.deepcopy(SELECT_CANDIDATE_TOOL)],
        "metadata": {
            "split": split,
            "group_id": group.group_id,
            "episode_id": (
                f"episode-sequence-{group.ordinal:03d}"
                if group.label in SEQUENTIAL_LABELS
                else f"episode-{group.group_id}"
            ),
            "step": (
                tuple(label for label in LABELS if label in SEQUENTIAL_LABELS).index(group.label)
                if group.label in SEQUENTIAL_LABELS
                else LABELS.index(group.label)
            ),
            "variant": variant,
            "intent": goal,
            "family": group.label,
            "label": group.label,
            "scenario_kind": SCENARIO_KIND_BY_LABEL[group.label],
            "criticality": CRITICALITY_BY_LABEL[group.label],
            "template_profile": "heldout_lexical_v3" if holdout else "source_oracle_v3",
            "target_candidate_id": target_candidate_id,
            "target_call": target_call,
            "tool_name": target_call["tool"],
        },
    }
    return record


def build_group_assignments(
    split_sizes: Mapping[str, int] | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[Group]]:
    """Stratify semantic groups into splits before any variants are generated."""
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    if tuple(sizes) != ("train", "valid", "test"):
        raise ValueError("split_sizes must contain train, valid, and test in that order")

    unit = len(LABELS) * VARIANTS_PER_GROUP
    invalid = {name: size for name, size in sizes.items() if size <= 0 or size % unit}
    if invalid:
        raise ValueError(f"every split size must be a positive multiple of {unit}: {invalid}")

    groups_per_label = {
        split: size // (len(LABELS) * VARIANTS_PER_GROUP) for split, size in sizes.items()
    }
    total_per_label = sum(groups_per_label.values())
    assignments = {split: [] for split in sizes}

    for label in LABELS:
        groups = [
            Group(group_id=f"{_LABEL_CODES[label]}-{ordinal:03d}", label=label, ordinal=ordinal)
            for ordinal in range(total_per_label)
        ]
        if label in SEQUENTIAL_LABELS:
            split_rng = random.Random(_stable_seed(SEQUENTIAL_SPLIT_SEED, "episode-split"))
        else:
            split_rng = random.Random(_stable_seed(seed, label, "split"))
        split_rng.shuffle(groups)
        cursor = 0
        for split, count in groups_per_label.items():
            assignments[split].extend(groups[cursor : cursor + count])
            cursor += count

    for split, groups in assignments.items():
        random.Random(_stable_seed(seed, split, "groups")).shuffle(groups)
    return assignments


def build_dataset(
    split_sizes: Mapping[str, int] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, list[dict[str, Any]]]:
    """Build globally deduplicated OpenAI-style rows for FunctionGemma fine-tuning."""
    assignments = build_group_assignments(split_sizes, seed=seed)
    dataset: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()

    for split, groups in assignments.items():
        rows: list[dict[str, Any]] = []
        for group in groups:
            for variant in range(VARIANTS_PER_GROUP):
                record = _build_record(group, split, variant, seed)
                violations = privacy_violations(record, denylist=denylist)
                if violations:
                    raise ValueError(
                        f"privacy audit failed for {record['id']}: " + "; ".join(violations)
                    )
                # Ignore bookkeeping fields: they must not make duplicate learning payloads look
                # unique. Split is already absent from the actual messages/tools.
                dedupe_key = _canonical_json(
                    {"messages": record["messages"], "tools": record["tools"]}
                )
                if dedupe_key in seen:
                    raise ValueError(f"duplicate learning payload: {record['id']}")
                seen.add(dedupe_key)
                rows.append(record)
        random.Random(_stable_seed(seed, split, "rows")).shuffle(rows)
        dataset[split] = rows
    return dataset


def _shannon_entropy(token: str) -> float:
    counts = Counter(token)
    length = len(token)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def privacy_violations(
    value: Any,
    *,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> list[str]:
    """Return deterministic privacy findings for generated or proposed training material."""
    text = value if isinstance(value, str) else _canonical_json(value)
    folded = text.casefold()
    findings: list[str] = []

    for term in denylist:
        if term.casefold() in folded:
            findings.append(f"denylisted term: {term}")
    if _HOST_PATH.search(text):
        findings.append("local host path")
    if _EMAIL.search(text):
        findings.append("email address")
    if _IPV4.search(text):
        findings.append("IPv4 address")
    if _UUID.search(text):
        findings.append("UUID-like identifier")

    for package in _PACKAGE.findall(text):
        if not package.startswith("com.example."):
            findings.append(f"non-fictional package: {package}")
    for match in _RESOURCE_PACKAGE.finditer(text):
        prefix = match.group(0).removesuffix(":id/")
        if not prefix.startswith("com.example."):
            findings.append(f"non-fictional resource package: {prefix}")

    for token in _ENTROPY_TOKEN.findall(text):
        # Natural snake-case tool names are long but low entropy.  Encoded credentials and
        # opaque copied identifiers are both long and high entropy.
        if _shannon_entropy(token) >= 4.25:
            findings.append("high-entropy opaque token")
            break
    return sorted(set(findings))


def dataset_statistics(dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Summarize counts used by tests and the on-disk manifest."""
    stats: dict[str, Any] = {"total_records": 0, "splits": {}}
    all_groups: set[str] = set()
    for split, rows in dataset.items():
        labels = Counter(str(row["metadata"]["label"]) for row in rows)
        kinds = Counter(str(row["metadata"]["scenario_kind"]) for row in rows)
        tools = Counter(str(row["metadata"]["tool_name"]) for row in rows)
        profiles = Counter(str(row["metadata"]["template_profile"]) for row in rows)
        groups = {str(row["metadata"]["group_id"]) for row in rows}
        episodes = {str(row["metadata"]["episode_id"]) for row in rows}
        if all_groups & groups:
            raise ValueError(f"semantic groups overlap split {split}")
        all_groups |= groups
        stats["splits"][split] = {
            "records": len(rows),
            "groups": len(groups),
            "episodes": len(episodes),
            "labels": dict(sorted(labels.items())),
            "scenario_kinds": dict(sorted(kinds.items())),
            "target_tools": dict(sorted(tools.items())),
            "template_profiles": dict(sorted(profiles.items())),
        }
        stats["total_records"] += len(rows)

    kind_totals = Counter()
    for split_stats in stats["splits"].values():
        kind_totals.update(split_stats["scenario_kinds"])
    total = stats["total_records"]
    recovery = kind_totals["recovery"] + kind_totals["counterfactual"]
    cleanup = kind_totals["cleanup"] + kind_totals["terminal"]
    stats["ratios"] = {
        "recovery_or_counterfactual": recovery / total if total else 0.0,
        "cleanup_or_terminal": cleanup / total if total else 0.0,
    }
    return stats


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.functiongemma.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_dataset(
    output_dir: str | Path,
    split_sizes: Mapping[str, int] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    denylist: Sequence[str] = DEFAULT_DENYLIST,
) -> dict[str, Any]:
    """Build and atomically write JSONL splits plus a deterministic hash manifest."""
    output = Path(output_dir)
    dataset = build_dataset(split_sizes, seed=seed, denylist=denylist)
    statistics = dataset_statistics(dataset)
    file_entries: dict[str, dict[str, Any]] = {}

    for split, rows in dataset.items():
        payload = "".join(f"{_canonical_json(row)}\n" for row in rows).encode("utf-8")
        filename = f"{split}.jsonl"
        _atomic_write(output / filename, payload)
        split_stats = statistics["splits"][split]
        file_entries[split] = {
            "path": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            **split_stats,
        }

    combined_hash = hashlib.sha256(
        "".join(file_entries[split]["sha256"] for split in ("train", "valid", "test")).encode()
    ).hexdigest()
    manifest = {
        "format": "functiongemma-aua-candidate-policy-v3",
        "seed": seed,
        "selection_function": "select_candidate(candidate_id: integer)",
        "split_policy": "stratified semantic groups assigned before variant expansion",
        "variants_per_group": VARIANTS_PER_GROUP,
        "total_records": statistics["total_records"],
        "ratios": statistics["ratios"],
        "privacy": {
            "passed": True,
            "checks": [
                "denylist",
                "host paths and direct identifiers",
                "fictional package namespace",
                "high-entropy opaque tokens",
            ],
        },
        "dataset_sha256": combined_hash,
        "splits": file_entries,
    }
    _atomic_write(output / "manifest.json", f"{_canonical_json(manifest)}\n".encode())
    return manifest


def iter_candidate_calls(record: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield candidate calls from one row; useful to independent validators."""
    user_message = record["messages"][1]
    state = json.loads(user_message["content"])
    for candidate in state["candidates"]:
        call = candidate["call"]
        yield str(call["tool"]), call["arguments"]
