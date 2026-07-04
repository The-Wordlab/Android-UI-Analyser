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
        "Start from the app playbook",
        "`aua about` prints what the tool already learned about THIS app — a one-line "
        "description, login **recipes** (e.g. how to log in as a test/full user), useful "
        "**deeplinks**, and **notes** (quirks, e.g. a dialog to dismiss after login, or that "
        "the 'Apps' tab is really Tools). Read it first and follow it — it saves you the "
        "discovery the last run already did. As you learn things, teach it back with "
        '`aua remember --about "…" | --note "…" | --recipe NAME --note "…" | --deeplink URI --note "…"` '
        "so the next run starts even more informed.",
    ),
    (
        "Use what memory already knows",
        "`aua map` (or `aua map --brief`) prints the app's known screens + routes — but you "
        "usually don't need to call it: every `analyze` already returns `meta.known_screen` plus "
        "inline `meta.known_routes` / `meta.suggested_gotos` / `meta.map_hint`. Act on those "
        'instead of re-exploring. `aua map --find "<goal>"` gives just the route to a target.',
    ),
    (
        "Take shortcuts with deeplinks",
        "`aua open \"<uri>\"` fires a deeplink — jump straight to a screen or trigger an app "
        "action (e.g. set a feature flag) instead of tapping through the UI. Far faster than "
        "navigating; the deeplink is remembered in the playbook. Some deeplinks need an app "
        "restart to take effect — follow with `aua app stop <pkg>` + `aua app launch <pkg>`.",
    ),
    (
        "Jump to a known screen in one call",
        '`aua goto "<goal>"` replays the remembered steps of each route edge — by resource-id '
        "first, then label — verifying every hop, including cross-app auth legs (Google sign-in "
        "through Chrome/GMS is folded into one edge). Prefer it whenever `suggested_gotos` lists "
        "your target. `--plan` prints the annotated route (steps, replayable, destructive) "
        "without acting. Steps matching `memory.destructive_labels` (delete/sign out/pay/…) are "
        "refused without `--allow-destructive`. On divergence it hands back the failing step, "
        "the remaining steps, and the current elements — finish that one step manually, then "
        "just re-run `aua goto`: it resumes mid-route, even mid-auth.",
    ),
    (
        "Replay whole journeys in one call (flows)",
        'A flow is a Maestro-style YAML journey you can AUTHOR directly (no walking needed) or '
        "record: `aua flow save <name> --last N` materializes your recent actions (typed values "
        "become required `${PARAM_n}` placeholders — fill them in the file). "
        '`aua flow run <name> --param K=V` drives the whole journey — launch, taps, waits, '
        "asserts, cross-app auth, even `goto:` steps — and on divergence returns the failing "
        "step index + remaining steps; fix and resume with `--from-step N`. Flows live under "
        "`<memory.dir>/flows/*.yaml` (`aua flow list|show|delete`); `--dry-run` previews. Use a "
        "flow for any setup you repeat (reset account, log in, reach the screen under test) — "
        "one call instead of a dozen.",
    ),
    (
        "Optional: let a fast model recover or explore (opt-in)",
        "If `planner.enabled` is set (+ an API key like GEMINI_API_KEY), you get two "
        "extras. (1) On a `goto`/`flow` divergence, add `--assist` and a fast planner LLM "
        "tries to recover in the same call (dismiss a popup, find the moved element) "
        "before handing off — the divergence hint tells you when it's worth trying. "
        "(2) `aua navigate \"<goal>\"` drives to a goal with no prior map AND records the "
        "path, so the next `aua goto` is a free deterministic replay. It's OFF by default "
        "and never touches the fast path; destructive taps still need `--allow-destructive`.",
    ),
    (
        "Drive by element ID",
        "`aua --format compact analyze` → a list of elements each with an integer `id` + bounds. "
        'Act on the id: `aua tap <id>`, `aua input <id> "text"`, `aua swipe up`, `aua key back`. '
        'Use `aua has "<text>"` (exit 0/1) to branch cheaply without parsing JSON.',
    ),
    (
        "Verify by resource-id, not just text",
        '`aua has "<id>" --by id` checks a resource-id (a bare tail like "containerChatDetail" '
        "works) — and it finds non-interactive **container** ids that `analyze` prunes from "
        "the element list, so it's the reliable way to assert you reached a screen "
        "(Maestro-style `assertVisible: id:`). `wait --for <id> --by id` and "
        "`scroll-to <id> --by id` take `--by id` too. If a screen is WebView/Compose-backed "
        "and its result text isn't in the tree at all, read it with `analyze --source vision`.",
    ),
    (
        "Act, then read the screen the action gives back",
        "IDs are only valid until the screen changes. By default every state-changing action "
        "(tap/input/swipe/scroll-to/key) returns the next screen inline in `observation` "
        "(elements with fresh ids) — so you rarely need a separate `analyze`: `type → tap "
        "send` is two calls, not three, and `goto` returns the destination's `elements` too. "
        "Pass `--no-observe` to skip it on action-only sequences. `observation` is a no-wait "
        "snapshot taken right after the action — use a plain `analyze` (after `wait --for-stable`) "
        "when the screen is still animating.",
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
    ("T3 vision", "detection + OCR (local)", "Compose/canvas/game (and weak WebView) trees"),
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
        "`--by text|id|desc` (id finds pruned containers), `--match exact|contains|regex`, "
        "`--ignore-case`, `--ocr-fallback/--no-ocr-fallback`, `--timeout <ms>`",
    ),
    (
        "wait",
        '`--for "<text>"` (`--by id`), `--idle`, `--for-stable`, `--interval`, `--settle`, '
        "`--timeout`, `--observe` (fresh ids after the wait)",
    ),
    (
        "map",
        '`--app <pkg>`, `--brief`, `--screen <name>`, `--depth N`, `--find "<goal>"`, `--json`',
    ),
    (
        "goto",
        "`<goal>` (fuzzy), `--plan` (annotated route, no taps), `--max-steps N`, "
        "`--allow-destructive`, `--assist` (opt-in planner recovery)",
    ),
    (
        "flow",
        "`run <name> [--param K=V] [--file PATH] [--dry-run] [--from-step N] "
        "[--no-allow-destructive] [--assist]`, `save <name> [--last N] [--force]`, "
        "`list|show|delete`. Steps incl. `launch_app`/`stop_app`/`open_link`/`goto`/`flow` "
        "(a `flow:` step runs a saved flow inline — reuse a shared `login` recipe).",
    ),
    ("open / about / remember", "`open <uri>` deeplink; `about` app playbook; `remember …` teach it"),
    (
        "navigate (opt-in planner)",
        "`<goal>` (natural language), `--until <text>`, `--max-steps N`, "
        "`--allow-destructive`, `--save-flow <name>` — needs `planner.enabled`",
    ),
    (
        "actions (tap/input/swipe/scroll-to/key/…)",
        "return the post-action screen inline by default (`observation`, fresh ids); "
        "`--no-observe` to skip it",
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
    p.append("## Hard screens (Compose / Flutter / canvas / games)")
    p.append(
        "Compose-without-semantics, Flutter, canvas, and games need vision — the gate escalates "
        "automatically. **WebView pages (Google sign-in, web content) usually expose a rich tree "
        "and stay on the fast hierarchy path**; only weak/hollow WebView trees escalate. If "
        "`analyze` visibly misses content, force it:\n"
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
        "secrets / PII are redacted (`<filled>` / `<redacted>`). The map is pushed to you "
        "inline on every `analyze` (`meta.known_routes` / `meta.suggested_gotos` / "
        "`meta.map_hint`), ranked by your recent navigation so the screens you use most surface "
        'first; `aua goto "<goal>"` drives a remembered route in one call. **Cross-app auth '
        "legs (Google sign-in via Chrome/GMS, permission dialogs) fold into the origin app's "
        "route** and replay step by step — a redacted account row hands off for one manual tap, "
        "then re-running `goto` resumes. Replay refuses destructive steps (delete/sign out/…) "
        "without `--allow-destructive`; the map improves with every walk (legacy edges upgrade "
        "in place when re-driven). Manage with `aua memory show|path|update|forget` "
        "(`memory update --screen <name>` renames a badly-auto-named screen so `goto <name>` "
        "reads naturally)."
    )

    p.append("")
    p.append("## Worked examples")
    p.append("```bash")
    p.append("# Optional: warm daemon so every later call is ~tens of ms.")
    p.append("aua daemon start")
    p.append("")
    p.append("# See the screen. When the app is mapped the response already carries")
    p.append("# meta.known_screen + meta.known_routes + meta.suggested_gotos — act on those.")
    p.append("aua --format compact analyze")
    p.append("")
    p.append("# Jump straight to a remembered screen (drives + verifies each hop,")
    p.append("# including cross-app auth legs):")
    p.append('aua goto "image creator"')
    p.append('aua goto "settings" --plan          # just print the route, take no action')
    p.append("")
    p.append("# Replay a whole journey (authored or recorded) in ONE call:")
    p.append('aua flow run reset_account_google_login --param ACCOUNT="Engineering Team"')
    p.append("aua flow save reach_checkout --last 8   # materialize what you just did")
    p.append("")
    p.append("# Act by id. Every action returns the post-action screen by default, so the")
    p.append("# result already carries observation.elements with fresh ids — type → send is")
    p.append("# two calls, not three:")
    p.append('aua input 24 "a neon koala surfing a wave"   # result.observation has the send id')
    p.append("aua tap 25                          # send-button id, taken from that observation")
    p.append("aua wait --for-stable               # wait out image generation / loading")
    p.append("")
    p.append("# Reach an off-screen target; the scroll already returns what came into view:")
    p.append('aua scroll-to "Translate"')
    p.append("")
    p.append("# Cheap branch with no JSON parsing (exit 0 present / 1 absent):")
    p.append('aua has "Done" && echo present')
    p.append("```")

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
    p.append('                "known_screen", "known_routes", "suggested_gotos", "map_hint",')
    p.append('                "annotated_image", "device_serial" } }')
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
            "meta.known_screen + inline meta.known_routes/suggested_gotos/map_hint on revisit "
            '(ranked by recent navigation); `aua goto "<goal>"` drives a remembered route; '
            "durable skeleton only; values/secrets redacted."
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
                "known_routes",
                "suggested_gotos",
                "map_hint",
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
