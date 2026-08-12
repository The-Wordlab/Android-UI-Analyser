"""Canonical agent-facing AUA capability catalogue.

The CLI, generated skills, and MCP server all teach from this module.  Command adapters stay
typed and explicit in their own modules; this catalogue owns discovery, ordering, examples,
risk, and cleanup so a new agent is not given three subtly different operating protocols.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Capability:
    id: str
    summary: str
    triggers: tuple[str, ...]
    priority: int
    cli: str
    mcp: str | None = None
    risk: str = "read-only"
    cleanup: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, ())}


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "session",
        "Start from a goal: attach, observe once, and receive the safest exact next call.",
        ("test", "verify", "inspect", "navigate", "open", "offline", "android"),
        0,
        'aua session start --goal "<goal>"',
        "session_start",
    ),
    Capability(
        "goto",
        "Replay a verified, context-compatible app-map route and verify every hop.",
        ("reach", "navigate", "open", "screen", "settings", "tool", "thread"),
        10,
        'aua goto "<goal>"',
        "goto",
        risk="previewed; unsafe route effects require explicit opt-in",
    ),
    Capability(
        "flow",
        "Replay a saved multi-step setup or journey in one call.",
        ("repeat", "setup", "login", "journey", "thread", "recent", "offline"),
        20,
        "aua flow run <name>",
        "flow_run",
        risk="expanded steps are preflighted before automatic selection",
    ),
    Capability(
        "back_until",
        "Exit nested screens in one bounded call, re-resolving Back on every fresh frame; "
        "an unlabeled first Back needs its fresh id.",
        ("back", "return", "compare", "nested", "thread", "recent"),
        25,
        "aua back-until-and-analyze 'rid:<destination>'",
        "back_until_and_analyze",
        risk="always stops when the foreground leaves the starting package",
    ),
    Capability(
        "deeplink",
        "Open a known shortcut only after goto/flow; the returned screen must prove arrival.",
        ("deeplink", "deep link", "open", "shortcut"),
        30,
        'aua open-and-analyze "<offered-uri>"',
        "open_link_and_analyze",
        risk="unsafe unless explicitly selected; intent delivery is not arrival",
    ),
    Capability(
        "action_until",
        "Act, wait for semantic evidence, and consume the settled observation in one call.",
        ("tap", "type", "input", "loading", "wait", "result", "interactive", "open"),
        40,
        "aua tap-and-analyze --rid <rid> --until 'rid:<destination>'",
        "tap_and_analyze",
    ),
    Capability(
        "await",
        "Wait for several positive/negative semantic terms in one assertion call.",
        ("wait", "loading", "spinner", "result", "assert", "visible", "absent"),
        45,
        "aua await-and-analyze 'rid:<ready>,!text:Loading' --observe",
        "await_and_analyze",
    ),
    Capability(
        "network_offline",
        "Enter a verified offline state and restore the exact prior connectivity afterward.",
        ("offline", "network", "connectivity", "airplane", "cached"),
        5,
        "aua network offline --verify",
        "network_offline",
        risk="reversible environment mutation",
        cleanup="aua network restore",
    ),
    Capability(
        "network_profile",
        "Apply reversible Wi-Fi, cellular, slow, or lossy connectivity conditions.",
        ("wifi", "cellular", "slow", "lossy", "latency", "network"),
        8,
        "aua network profile apply <profile>",
        "network_profile_apply",
        risk="reversible environment mutation",
        cleanup="aua network profile restore",
    ),
    Capability(
        "analyze",
        "Read the screen once as concise semantic rows with stable selectors.",
        ("inspect", "screen", "visible", "element", "ui"),
        50,
        "aua --format tsv analyze --fields id,text,rid,clickable",
        "analyze_screen",
    ),
    Capability(
        "suite",
        "Evaluate a reusable group of independent acceptance criteria in one call.",
        ("assertions", "criteria", "checklist", "suite", "many checks"),
        55,
        "aua suite run <checks.yaml>",
        "suite_run",
    ),
    Capability(
        "database",
        "Inspect debuggable app SQLite state with coherent WAL-aware snapshots.",
        ("database", "sqlite", "room", "cache", "persisted", "thread"),
        60,
        "aua db list <package>",
        "database_list",
        risk="queries may expose private test data; mutation requires confirmation and backup",
    ),
    Capability(
        "flags",
        "Apply configured feature flags and verify the required app restart.",
        ("flag", "experiment", "variant", "treatment"),
        60,
        "aua flags apply <flags.yaml>",
        "flags_apply_and_analyze",
        risk="reversible app configuration mutation",
    ),
    Capability(
        "capture",
        "Recover transient loading or animation frames from the daemon rolling buffer.",
        ("video", "animation", "flash", "transition", "loading", "frame"),
        65,
        "aua capture last",
        "capture_last",
    ),
    Capability(
        "logcat",
        "Mark and read only the device logs produced by the tested action.",
        ("crash", "log", "error", "exception", "diagnose"),
        65,
        "aua logcat mark",
        "logcat_mark",
    ),
    Capability(
        "map",
        "Inspect or improve context-scoped screens and verified routes.",
        ("map", "route", "unknown screen", "feature flag", "research"),
        70,
        'aua map --find "<goal>"',
        "map_find",
    ),
    Capability(
        "emulator",
        "Discover, lease, boot, and clean up an appropriate Android emulator.",
        ("device", "emulator", "avd", "headed", "headless"),
        75,
        "aua devices",
        "list_devices",
        cleanup="stop only an emulator this session started",
    ),
)


def capability_manifest() -> list[dict[str, Any]]:
    """Return the stable machine-readable discovery surface."""
    return [cap.as_dict() for cap in sorted(CAPABILITIES, key=lambda item: item.priority)]


def capabilities_for_goal(goal: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """Rank capabilities for a natural-language goal, retaining the core session protocol."""
    lowered = goal.casefold()
    core = {"session", "goto", "flow", "action_until"}

    def rank(cap: Capability) -> tuple[int, int, int]:
        matches = sum(1 for trigger in cap.triggers if trigger in lowered)
        return (cap.priority, -matches, 0 if cap.id in core else 1)

    matched = [
        cap
        for cap in CAPABILITIES
        if cap.id in core or any(trigger in lowered for trigger in cap.triggers)
    ]
    return [cap.as_dict() for cap in sorted(matched, key=rank)[: max(1, limit)]]


def render_mcp_instructions() -> str:
    """Short initialization contract for agents that never see shell help or a skill."""
    return (
        "Start Android work with session_start(goal). It observes once and ranks a verified "
        "goto, matching saved flow, proven deeplink, or manual analyzed action in that order. "
        "Use the returned recommended_call. Analyzed actions already return fresh screen ids; "
        "do not call analyze_screen next. Put semantic arrival terms in until, including "
        "comma-separated positive and negative terms. Use back_until_and_analyze for nested "
        "returns; an unlabeled first Back requires its fresh back_id. Use network_offline, never airplane mode, "
        "to prove offline behavior and always call network_restore or session_finish. Use "
        "session_review to find avoidable calls (`ok` is review success; `run_ok` is run health, "
        "and null means an older duplicated invocation has no provable caller-visible outcome). "
        "If a daemon call reports daemon_outcome_unknown, never repeat it: wait, then inspect "
        "one fresh screen. A busy live daemon remains the sole device controller. "
        "Unsafe or destructive effects require explicit "
        "authorization. capabilities(goal) provides progressive discovery for uncommon tasks."
    )
