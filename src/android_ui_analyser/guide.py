"""The agent operating manual — a single canonical source (PRD §5 "Agent guide", §17b).

``aua guide`` prints this so a *future* agent (e.g. a fresh Claude Code session) learns
how to drive the tool: what it is, the recommended session protocol, how perception
self-routes, how memory works, the output schema, exit codes, and the key flags.

This module is the **single source of truth**. The same content renders to:
- ``aua guide``            → markdown manual (``--brief`` for the short form, ``--json`` structured)
- ``aua guide --emit-skill`` → ``.claude/skills/android-ui-analyser/SKILL.md`` (frontmatter + this manual)

Because the skill body *is* the rendered guide, the two can never drift (AC15).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Skill metadata. The description carries the trigger conditions that make Claude Code
# auto-activate the skill on Android-UI tasks — keep it stable across regenerations.
SKILL_NAME = "android-ui-analyser"
SKILL_DESCRIPTION = (
    "Drive and inspect an Android app's UI on a device/emulator with the `aua` "
    "(android-ui-analyser) CLI. Returns the screen as a list of elements with stable integer "
    "IDs + bounding boxes, then acts BY ID — tap/input/swipe/key — so you never guess pixel "
    'coordinates. Use whenever the task involves an Android device/emulator: "test the '
    'Android app", "what\'s on screen", "tap/type/swipe the X", "is <text> visible", "drive '
    'the emulator", automating or debugging an Android UI flow, or checking a screen after a '
    "change. Hierarchy-first (tens of ms); falls back to OCR/detection/grounding vision on "
    "Compose/Flutter/WebView/canvas/game screens the accessibility tree can't see."
)

DEFAULT_SKILL_PATH = Path(".claude/skills/android-ui-analyser/SKILL.md")

# --- structured data (drives both the prose tables and `--json`) ---------------------

SESSION_PROTOCOL: list[tuple[str, str]] = [
    (
        "Start the warm daemon",
        "`aua daemon start` — holds the device connection + loaded models warm so each later "
        "call is ~tens of ms instead of paying Python/connect startup. Optional; every command "
        "still works without it.",
    ),
    (
        "Load the app's known layout",
        "`aua map` (or `aua map --brief` to start) prints what the tool already knows about this "
        "app — its screens, their key elements, and the routes between them — so you navigate "
        'instead of rediscovering. `aua map --find "<goal>"` gives just the route to a target.',
    ),
    (
        "Drive by element ID",
        "`aua --format compact analyze` → a list of elements each with an integer `id` + bounds. "
        'Act on the id: `aua tap <id>`, `aua input <id> "text"`, `aua swipe up`, `aua key back`. '
        'Use `aua has "<text>"` (exit 0/1) to branch cheaply without parsing JSON.',
    ),
    (
        "Re-analyze after every action",
        "IDs are only valid until the screen changes. After any tap/input/swipe/key, run "
        "`analyze` again before acting — old ids are stale.",
    ),
    (
        "Wait on state, never sleep",
        '`aua wait --for "<text>"` waits for text to appear; `aua wait --for-stable` returns once '
        "the screen stops visually changing (cheap perceptual-hash over screenshots — ideal for "
        "image generation / loading / video, works on opaque screens). Prefer these to fixed sleeps.",
    ),
    (
        "Stop the daemon when done",
        "`aua daemon stop` releases the warm connection.",
    ),
]

ESCALATION_LADDER: list[tuple[str, str, str]] = [
    ("T0 text", "hierarchy text match (`has`)", "is this text/element present?"),
    ("T1 selector", "hierarchy selector locate", "give me THIS known element to act on"),
    ("T2 hierarchy", "full hierarchy parse → element list", "what's on screen? (`analyze`)"),
    ("T3 vision", "detection + OCR (local)", "Compose/WebView/canvas/game the tree can't see"),
    ("T4 grounding", "grounding VLM (local or paid)", "fuzzy/visual targets not resolvable above"),
]

EXIT_CODES: list[tuple[str, str]] = [
    ("0", "success (`has`: text present)"),
    ("1", "`has`: text not present"),
    ("2", "usage error"),
    ("3", "no device / device error / `wait --for-stable` timeout"),
    ("4", "provider error (fallback chain exhausted)"),
    ("5", "config error"),
]

KEY_FLAGS: list[tuple[str, str]] = [
    (
        "global, BEFORE the subcommand",
        "`--format json|pretty|compact`, `--serial`, `--config`, "
        "`--profile`, `--timeout`, `--log-level`, `--no-cache`",
    ),
    (
        "analyze",
        '`--source auto|hierarchy|vision`, `--query "<nl>"`, `--deep`, `--cheap`, '
        "`--strategy <tier>`, `--annotate [path]`, `--with-ocr/--no-ocr`",
    ),
    (
        "has",
        "`--match exact|contains|regex`, `--ignore-case`, `--ocr-fallback/--no-ocr-fallback`, "
        "`--timeout <ms>`",
    ),
    ("wait", '`--for "<text>"`, `--idle`, `--for-stable`, `--interval`, `--settle`, `--timeout`'),
    (
        "map",
        '`--app <pkg>`, `--brief`, `--screen <name>`, `--depth N`, `--find "<goal>"`, `--json`',
    ),
]


def _md_table(headers: list[str], rows: Sequence[tuple[str, ...]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- renderers


def render_markdown(*, brief: bool = False) -> str:
    """The manual as markdown. ``brief`` = the short session-protocol form."""
    p: list[str] = []
    p.append("# android-ui-analyser (`aua`) — agent operating manual")
    p.append("")
    p.append(
        "`aua` reports **what's on an Android screen and where**, so you act on **integer "
        "element IDs, not pixels**. It reads the accessibility/view hierarchy first (fast, "
        "exact) and falls back to image vision (detection + OCR, optional grounding VLM) on "
        "screens the hierarchy can't see. It remembers each app's layout so you start each "
        "session already knowing the map."
    )
    p.append("")
    p.append("## Session protocol")
    for i, (title, body) in enumerate(SESSION_PROTOCOL, 1):
        p.append(f"{i}. **{title}.** {body}")

    if brief:
        p.append("")
        p.append("## Escalation (automatic)")
        p.append(
            "Perception climbs only as far as the question needs: "
            + " → ".join(t for t, _, _ in ESCALATION_LADDER)
            + ". Paid grounding (T4) is **never** entered automatically — pass `--deep`."
        )
        p.append("")
        p.append("## Exit codes")
        p.append(", ".join(f"`{c}` {d}" for c, d in EXIT_CODES))
        p.append("")
        p.append("Run `aua guide` (no `--brief`) for the full manual.")
        return "\n".join(p) + "\n"

    p.append("")
    p.append("## Flag placement (this bites people)")
    p.append(
        "**Global** flags go BEFORE the subcommand; **command** flags after.\n"
        "✅ `aua --format compact analyze --source vision`  ·  "
        '❌ `aua analyze --format compact` ("No such option").'
    )
    for scope, flags in KEY_FLAGS:
        p.append(f"- _{scope}_: {flags}")

    p.append("")
    p.append("## The loop")
    p.append("```bash")
    p.append(
        "aua --format compact analyze     # elements[] with id, type, text, bounds, center, clickable"
    )
    p.append("aua tap 4                        # act by id (alias: click)")
    p.append(
        'aua input 2 "hello@example.com"  # focus id 2 and type (--submit fires the IME action)'
    )
    p.append("aua --format compact analyze     # RE-ANALYZE: ids are invalidated after any action")
    p.append("```")
    p.append(
        'Cheap presence check to branch on: `aua has "Sign in"` (exit 0 found / 1 not). '
        '`aua wait --for "Welcome"` polls until present; `aua wait --for-stable` returns once '
        "the screen settles (no OCR/hierarchy — just screenshots)."
    )

    p.append("")
    p.append("## Self-routing — the escalation ladder")
    p.append(
        "No LLM decides the route; the engine starts at the cheapest tier that could answer and "
        "escalates only on a miss, bounded by config (`routing.max_tier`, default `vision`)."
    )
    p.append(_md_table(["Tier", "Method", "Answers"], ESCALATION_LADDER))
    p.append(
        '\n`analyze --query "the gear icon"` resolves from the hierarchy first (free) and only '
        "escalates. The default ceiling is **local vision**; reaching the (paid) grounding VLM "
        "requires `--deep`. `--cheap` forbids escalation; `--strategy <tier>` pins one. "
        "`meta.tier_used` reports which rung actually ran."
    )

    p.append("")
    p.append("## Hard screens (Compose / Flutter / WebView / games)")
    p.append(
        "When the accessibility tree is empty/useless, force vision:\n"
        "```bash\naua --format compact analyze --source vision --annotate\n```\n"
        "`meta.annotated_image` is a PNG with numbered boxes you can open."
    )

    p.append("")
    p.append("## App memory (auto-recorded)")
    p.append(
        "The tool maintains a persistent, **local-only** map per app under `memory.dir` "
        "(default `~/.android-ui-analyser`). Every `analyze` records the current screen and "
        "every state-changing action records a route edge — no extra calls, and the daemon path "
        'records too. Read it back with `aua map` / `aua map --find "<goal>"`. On a revisit, '
        "`meta.known_screen` names the recognised screen; a changed signature or app version "
        "flags it `stale` so you re-verify. Only the **durable skeleton** is stored (screens, "
        "routes, stable elements); dynamic lists are stored as a *shape*, and `EditText` values / "
        "secrets / PII are redacted (`<filled>` / `<redacted>`). Manage with "
        "`aua memory show|path|update|forget`."
    )

    p.append("")
    p.append("## Output schema (read these fields)")
    p.append("```json")
    p.append('{ "schema_version": 1,')
    p.append(
        '  "screen":   { "width", "height", "package", "activity", "source": "hierarchy|vision|mixed" },'
    )
    p.append('  "elements": [ { "id", "type", "text", "resource_id", "content_desc",')
    p.append('                  "bounds": [x1,y1,x2,y2], "center": [x,y],')
    p.append('                  "clickable", "enabled", "focused",')
    p.append('                  "source": "hierarchy|detection|ocr|grounding", "confidence" } ],')
    p.append('  "meta":     { "duration_ms", "tier_used", "path", "providers_used",')
    p.append('                "known_screen", "annotated_image", "device_serial" } }')
    p.append("```")
    p.append("`compact` drops null/default fields for the smallest token footprint.")

    p.append("")
    p.append("## Exit codes")
    p.append(_md_table(["Code", "Meaning"], EXIT_CODES))
    p.append(
        '\nErrors print `{"error":{"code","message","hint"}}` to **stderr**; JSON results go '
        "to **stdout** (pipe-clean)."
    )

    p.append("")
    p.append("## Config & providers (only if asked to change perception)")
    p.append(
        "Config is the nearest `.android-ui-analyser.yaml` (project) → user config; inspect with "
        "`aua config show` / `aua config path`, scaffold with `aua config init`. Swap a model with "
        "one line (e.g. `ocr.chain: [apple_vision, rapidocr]`). **Secrets are env-var names only** "
        "(`api_key_env: OPENAI_API_KEY`); set the env var — never paste keys. Check readiness with "
        "`aua doctor` (it never prints secret values)."
    )
    return "\n".join(p) + "\n"


def render_brief() -> str:
    return render_markdown(brief=True)


def render_json() -> dict[str, object]:
    """Structured form of the manual for programmatic consumers."""
    return {
        "name": SKILL_NAME,
        "summary": (
            "Structured Android UI perception + action for agents: act on element IDs, not pixels."
        ),
        "session_protocol": [{"step": t, "detail": b} for t, b in SESSION_PROTOCOL],
        "escalation_ladder": [
            {"tier": t, "method": m, "answers": a} for t, m, a in ESCALATION_LADDER
        ],
        "memory": (
            "Per-app map auto-recorded locally under memory.dir; read via `aua map`; "
            "meta.known_screen on revisit; durable skeleton only; values/secrets redacted."
        ),
        "schema_fields": {
            "top": ["schema_version", "screen", "elements", "meta"],
            "element": [
                "id",
                "type",
                "text",
                "resource_id",
                "content_desc",
                "bounds",
                "center",
                "clickable",
                "enabled",
                "focused",
                "source",
                "confidence",
            ],
            "meta": [
                "duration_ms",
                "tier_used",
                "path",
                "providers_used",
                "known_screen",
                "annotated_image",
                "device_serial",
            ],
        },
        "exit_codes": [{"code": c, "meaning": d} for c, d in EXIT_CODES],
        "key_flags": [{"scope": s, "flags": f} for s, f in KEY_FLAGS],
    }


# --------------------------------------------------------------------------- skill emit


def render_skill() -> str:
    """The SKILL.md content: YAML frontmatter + the canonical manual body (no drift)."""
    front = [
        "---",
        f"name: {SKILL_NAME}",
        "description: >-",
    ]
    # Fold the description into indented continuation lines for valid YAML block scalar.
    words = SKILL_DESCRIPTION.split(" ")
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 96 and line.strip():
            front.append(line.rstrip())
            line = "  "
        line += w + " "
    if line.strip():
        front.append(line.rstrip())
    front.append("---")
    front.append("")
    front.append(
        "<!-- Generated by `aua guide --emit-skill`. Edit guide.py (the single source), not this file. -->"
    )
    front.append("")
    return "\n".join(front) + render_markdown(brief=False)


def emit_skill(path: str | Path | None = None) -> Path:
    """Write the generated SKILL.md to *path* (default `.claude/skills/.../SKILL.md`)."""
    target = Path(path) if path else DEFAULT_SKILL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_skill(), encoding="utf-8")
    return target
