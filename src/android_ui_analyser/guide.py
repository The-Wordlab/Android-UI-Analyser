"""The agent operating manual — a single canonical source (PRD §5 "Agent guide", §17b).

``aua guide`` prints this so a *future* agent (e.g. a fresh Claude Code session) learns
how to drive the tool: what it is, the recommended session protocol, how perception
self-routes, how memory works, the output schema, exit codes, and the key flags.

This module is the **single source of truth**. It renders three progressive layers:
- ``aua guide``              → the complete markdown manual
- ``aua guide --brief``      → a useful session-oriented field guide
- ``aua guide --emit-skill`` → a compact generated SKILL.md that points to both deeper layers

The generated skill is intentionally small enough to load on every Android task; detailed
reference material stays discoverable through the CLI instead of consuming agent context.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .projection import FIELD_ALIASES, TSV_DEFAULT_FIELDS

# Skill metadata. The description carries the trigger conditions that make Claude Code
# auto-activate the skill on Android-UI tasks — keep it stable across regenerations.
SKILL_NAME = "android-ui-analyser"
SKILL_DESCRIPTION = (
    "Drive, inspect, and verify Android app UIs on a device/emulator with the `aua` CLI. It "
    "returns stable element IDs and bounds, then acts by ID instead of guessed pixels. Use for "
    "Android device/emulator tasks: inspect a screen, act on a control, automate "
    "or debug a UI flow, verify a change, test offline/network or emulator microphone/voice "
    "input, or inspect/seed a debuggable app's SQLite database. Start goal-oriented work with "
    "`aua session start --goal`. AUA is hierarchy-first and falls back to OCR, detection, or "
    "grounding vision for Compose/Flutter/WebView/canvas/game screens the accessibility tree "
    "cannot describe."
)

DEFAULT_SKILL_PATH = Path(".claude/skills/android-ui-analyser/SKILL.md")

# --- structured data (drives both the prose tables and `--json`) ---------------------

SESSION_PROTOCOL: list[tuple[str, str]] = [
    (
        "Start from the user's goal",
        'Run `aua session start --goal "<what you must verify>"` first. It leases an available '
        "device, reuses the warm daemon, observes the foreground app once, and returns relevant "
        "capabilities plus one exact `recommended_call` in both CLI and MCP form. Its selection "
        "order is verified context-compatible `goto`, a matching safe flow, a proven deeplink, "
        "then a manual analyzed action. Explicit sequence words create durable ordered phases, "
        "so an online preparation remains before a later offline transition. Every result carries "
        "`goal_progress`; when its evidence completes the active checkpoint, add the returned "
        '`--phase-done phase_N="evidence"` to your next AUA call (MCP: `phase_done`). This '
        "advances the goal without another round trip. Deterministic offline evidence advances "
        "automatically. The returned compact observation is the current screen: reuse it "
        "and do not follow session start with analyze. The command is recommendation-first: "
        "when the foreground is unrelated, add `--app <package>` (alias `--package`, optional "
        "`--activity`) and session start launches it then reuses that folded observation. "
        "risky candidates are previewed and never receive authorization from the goal text. "
        "For deterministic acceptance proof, add `--contract <yaml>` and optionally "
        "`--artifacts-dir <dir> --evidence all --junit`. Authored checkpoints reuse flow "
        "assertions, require one fresh fingerprinted frame, and reject `--phase-done`. Every "
        "analyzed response carries `observation_contract` with `reusable` and "
        "`analyze_needed`. A contracted `session finish` stays active and returns "
        "`contract_incomplete` until all checkpoints, including UI cleanup, pass; only explicit "
        "`--allow-incomplete` bypasses that proof. Use `--wait-for-lease <seconds>` for bounded "
        "contention without switching or stealing the requested device. Once complete, "
        "`session candidate-flow NAME` previews the exact post-watermark action window; "
        "`--save` first requires an explicit `--reset-flow` and a successful replay. "
        "Use `aua session review` to see avoidable calls: `ok` means the review completed, "
        "while `run_ok` and `failures` describe the run without making the review itself "
        "another failure (`run_ok: null` means an older duplicated invocation had no provable "
        "caller-visible outcome). Treat its `accounting` as authoritative instead of estimating "
        "calls: `top_level_calls` counts caller-visible invocations and is partitioned into "
        "`lifecycle_calls` plus `task_calls`; `journal_events` also counts "
        "`folded_internal_events`, such as an action-bound wait. The review snapshot is computed "
        "before its current review or finish call is journaled, so `reporting_call_included` is "
        "false and `top_level_calls_including_reporting_call` adds that reporting call. For an "
        "intentional negative probe, declare the "
        "exact typed code on the same call with global `--expect-error CODE` (MCP: "
        "`expect_error`); only an exact match is excluded from run failures. End with the exact "
        "`cleanup_call` returned by session start: "
        "one `aua session finish` performs session-owned reversible cleanup and review. Do not "
        "run `network restore` separately first.",
    ),
    (
        "Let session start select or provision the device",
        "Do not list devices, start an emulator, set `AUA_OWNER`, or acquire a lease before "
        "goal work. `aua session start` probes every attached target, including leased ones so "
        "dead owner processes free immediately, and claims the first compatible free target for "
        "the calling agent process. If none matches, it selects a configured AVD from "
        "`--needs root,play,proxy` and boots it headless automatically. Use "
        "`--no-start-emulator` only when provisioning is forbidden. "
        "For microphone/voice-input tests add `--audio`; the normal unattended default uses "
        "`-no-audio` to avoid unnecessary host audio initialization. "
        "When the user needs to watch, add `--headed`; it is valid with an already attached "
        "visible emulator, and if AUA must start one its ownership is recorded so "
        "`aua session finish` can restore state, release the lease, and keep that AVD warm for "
        "the next session. The lease-gated idle watchdog retires it after "
        "`teardown.emulator_idle_stop_s` (20 minutes by default). Headless uses `-no-window` + "
        "**host GPU** on Mac — not "
        "SwiftShader; `--avd <name>` is an optional hard preference, not a discovery step. "
        "Prefer an already running device when one is attached; don't kill the user's headed "
        "emulator unless they asked. **Prefer a rootable Google APIs AVD over a Play Store "
        "image whenever the test does not specifically need Play services** — root is what "
        "unlocks the capabilities that are otherwise simply unavailable, and a Play image "
        "refuses `adb root`: HTTPS proxy / mock record (system CA), and the optional on-device "
        "helper that runs a long flow on the phone instead of one round trip per step. "
        "Create one once with `aua emulator recommend-proxy` / `ensure-proxy`; later sessions "
        "select it automatically with `--needs root,proxy`. "
        "Every AUA-started emulator checks its inherited Android proxy setting before app "
        "launch and automatically clears it only when it is both unowned and confirmed "
        "blackholed; reachable foreign proxies and AUA-owned proxies are preserved. "
        "A long automatic boot emits `AUA_PROGRESS` plus ten-second heartbeats on stderr while reserving "
        "stdout for the final result. If your shell yields a live process/session id, poll that "
        "same process — never issue a duplicate start. "
        "Parallel sessions serialize host-wide selection, allocate unique ports, and boot "
        "read-only AVD instances when the shared pool has no free match. After assignment, omit "
        "`--serial` from ordinary device commands and end with the returned `session finish` call.",
    ),
    (
        "Standalone/admin emulator starts require explicit stop",
        "`aua emulator start` is an administrative standalone operation, not the goal-session "
        "bootstrap. If you use it directly, the hard cleanup requirement is: "
        "`aua emulator stop --serial <yours>` (safest with parallel agents) or "
        "`AUA_OWNER=… aua emulator stop --mine` / `--avd <name>`. Orphaned headless AVDs burn "
        "CPU and battery. Do this even if the test failed. "
        "Safety nets (do not rely on them alone): idle watchdog auto-stops after "
        "`--idle-stop` seconds of no aua activity (default 900); MCP `emulator_start` "
        "tracks serials and stops them when the MCP process exits. "
        "`aua emulator status` shows what aua started (including `owner` / `port`).",
    ),
    (
        "Start the warm daemon",
        "`aua daemon start` — holds the device connection + loaded models warm so each later "
        "call is ~tens of ms instead of paying Python/connect startup. Optional; every command "
        "still works without it. For even lower host latency on hot commands once the daemon is "
        "up, build `native/aua-fast` (`make -C native/aua-fast install`) and use `aua-fast "
        "analyze|tap|has|…` — a tiny C client that speaks the daemon socket (falls back to "
        "`aua` if the daemon is down). See `docs/NATIVE_ROADMAP.md`. Unchanged screens short-"
        "circuit via `meta.via=hierarchy-unchanged` / `--format delta`; wait on any tree change "
        "with `aua wait-and-analyze --changed` (or MCP `wait_changed`); multi-device with `aua fanout`.",
    ),
    (
        "Observe once, then choose the highest-level action",
        "Reuse the observation returned by `session start`; when no goal session exists, start "
        "with `aua --format tsv analyze --fields id,text,rid,clickable`. Before doing "
        "more discovery, read its `# goto:`, `# flows:`, and `# aua asks:` lines. Use this "
        'decision order: (1) if `# goto:` offers the destination, run `aua goto "<goal>"`; '
        "it replays and verifies the route, (2) if a saved flow matches the repeated setup or "
        "journey, run `aua flow run <name>`, (3) only when neither covers the goal, try an "
        'offered `aua open-and-analyze "<deeplink>"` and check `verified` is `true` (not just '
        "`ok:true`) before trusting the screen it returns, "
        "then (4) use manual element actions. Do not inventory `about` or raw deeplinks first "
        "when the observation already offers the goal.",
    ),
    (
        "Read the app playbook when the task needs it",
        "`aua about` prints what the tool already learned about THIS app — a one-line "
        "description, login **recipes** (e.g. how to log in as a test/full user), useful "
        "**deeplinks**, and **notes** (quirks, e.g. a dialog to dismiss after login, or that "
        "the 'Catalog' tab is labelled Browse). Read it when you need a recipe, environment setup, "
        "an app quirk, or no inline route/flow answers the goal; it is a playbook, not a "
        "mandatory first round trip. As you learn things, teach it back with "
        '`aua remember --about "…" | --note "…" | --recipe NAME --note "…" | --deeplink URI --note "…"` '
        "so the next run starts even more informed. When the playbook says the launch is "
        "AMBIGUOUS, the build declares more than one MAIN/LAUNCHER Activity and a cold start may "
        "open a Dev Tools entry instead of the product — pin the right one once with "
        "`aua remember --launch-activity <Activity>` and every later `aua app launch` uses it. "
        "Check for an equivalent fact first rather "
        "than appending a near-duplicate; scope version/flag/locale/account-dependent claims, and "
        "when a new build contradicts one preserve its evidence with "
        "`aua knowledge stale <id>` before adding the replacement.",
    ),
    (
        "Use what memory already knows",
        "`aua map` (or `aua map --brief`) prints the app's known screens + routes — but you "
        "usually don't need to call it: every `analyze` already returns `meta.known_screen` plus "
        "inline `meta.known_routes` / `meta.suggested_gotos` / `meta.map_hint`; unresolved "
        "map questions arrive in `meta.research_tasks`. Act on those "
        'instead of re-exploring. `aua map --find "<goal>"` gives only a verified route to a '
        "target; provisional evidence is shown as no verified route, never as runnable steps. "
        "Feature-flag sets are separate contexts; use `--all-contexts` to compare variants and "
        "`--audit` to persist ambiguous names/routes as concrete research tasks.",
    ),
    (
        "Take shortcuts with deeplinks",
        '`aua open-and-analyze "<uri>"` fires a deeplink — jump straight to a screen or trigger an app '
        "action (e.g. set a feature flag) instead of tapping through the UI. Use one only when "
        "no verified `goto` or matching flow covers the goal. When an app has known deeplinks, "
        "every `analyze` offers the best ones inline in `meta.suggested_deeplinks` (e.g. `open "
        "myapp://home`). A delivered intent is not proof of arrival: `ok:true` only means `am "
        "start` succeeded. Check `verified` — `false` (or absent) means no destination change "
        "was confirmed (`stale_risk` explains why); only `true` means the screen actually "
        "moved. On `false`, return to `goto` or element actions instead of assuming you "
        "arrived. Some deeplinks "
        "need an app restart to take effect (`aua app stop <pkg>` + `aua app launch <pkg>`). "
        "Don't know the app's deeplinks? `aua explore mine <repo> --app <pkg>` harvests them "
        "from the source once (then they ride inline + show in `aua about`).",
    ),
    (
        "Inspect or seed app state through AUA, not hand-written adb",
        "For a debuggable build, `aua db list <pkg>` discovers private SQLite files, "
        '`aua db schema <pkg> <db>` describes them, and `aua db query <pkg> <db> "SELECT …"` '
        "returns bounded JSON rows without stopping the app, preserving its current UI. Android "
        "images often have no `sqlite3`, so AUA copies the database plus WAL through `run-as` "
        "and queries it read-only with host SQLite. Use `db query … --coherent` when transactional "
        "coherence is required; it stops/relaunches the app (`--no-restart` leaves it stopped). "
        "Data mutation is "
        '`aua db execute <pkg> <db> "UPDATE …" --yes`: it creates a restore point, runs one '
        "transaction, rejects schema/PRAGMA/ATTACH changes, checks foreign keys and integrity, "
        "then replaces the database without stale sidecars. Use `db backups` / "
        "`db restore … --yes` to roll back. Query/execute results can contain user data; "
        "request only the columns and rows the task needs. For human inspection, the "
        "per-device `aua dashboard` detail view has a database workspace backed by the "
        "same service and typed mutation/restore confirmations.",
    ),
    (
        "Index an app you don't know yet",
        "`aua explore plan` returns a risk-classified worklist — unresolved map questions first, "
        "then safe dead-end screens, then speculative deeplinks. External/destructive probes are "
        "labelled and a listed task never grants authorization for their side effects. Run safe "
        "tasks with normal `aua` commands; results auto-record into the map + playbook, and "
        "re-running the plan shows what's left. Use `aua map --audit --summary` for token-cheap "
        "health counts and full `--json` only when you need the evidence.",
    ),
    (
        "Feed research back and correct the map",
        "`aua knowledge add` stores an experience with source/agent/session/evidence so future "
        "runs inherit it. AUA also creates research tasks automatically when a new map entry is "
        "ambiguous or a route is provisional/unreplayable. **The cheapest of those is answered in "
        "passing.** A response may carry `meta.ask` — a `# aua asks: …` line under `--format tsv` "
        "— which is ONE question about the screen you are standing on, almost always *this name "
        "was generated, what is this screen actually for?*. Answer it by adding "
        '`--answers <task-id>="<name>"` to your NEXT command, whatever that command is: no extra '
        "round trip, no separate chore, and any unique tail of the id will do. Worth doing "
        'because the name you give is what `aua goto "<name>"` and `aua map --find` can then '
        "reach that screen by — for the rest of this run and for every run after it. Bigger "
        "corrections still go the long way: research `meta.research_tasks` (or run "
        "`aua reconcile plan`) in source/runtime, then submit the canonical JSON report. "
        "To settle a backlog of naming questions at once, answer them offline with "
        "`aua reconcile answers --app <pkg> <file>` — one transaction, one rollback id, "
        "and `--dry-run` shows what it would do. "
        "AUA does not spawn the research agent. `verdict=apply` commits every operation "
        "transactionally and returns "
        "a rollback id; `review` queues it and `reject` retains the feedback.",
    ),
    (
        "Jump to a known screen in one call",
        '`aua goto "<goal>"` replays the remembered steps of each route edge — by resource-id '
        "first, then label — verifying every hop. Cross-app auth legs are folded into one edge, "
        "but require explicit side-effect approval before replay. For a destination, prefer it "
        "over a raw "
        "deeplink whenever `suggested_gotos` lists the target: `goto` verifies each hop and "
        "arrival instead of merely delivering an intent. Known in-app hops skip OCR. If a hop "
        "first returns a mapped loading shell or visible progress state, goto waits briefly for "
        "the expected mapped destination before declaring `wrong_screen`; a loading frame is "
        "not route-divergence evidence. It retries OCR only when hierarchy cannot match a "
        "selector or verify arrival; transit screens keep automatic OCR. `--plan` prints the "
        "annotated route, including per-step risks, without acting. Before the first route step, "
        "AUA refuses deeplinks, cross-package actions, settings/data/environment mutation, app "
        "lifecycle changes, and other non-navigation effects with a visible preview; review it "
        "before re-running with `--allow-unsafe`. Steps matching `memory.destructive_labels` "
        "(delete/sign out/pay/…) separately require `--allow-destructive`; mixed routes require "
        "both flags. Prefer an authored flow for setup or mutation. On divergence it hands back "
        "the failing step, "
        "the remaining steps, and the current elements — finish that one step manually, then "
        're-run `aua goto "…" --from-here` to resume mid-edge (skips steps that already match '
        "the current screen; also covers mid-auth). Plain `aua goto` still starts from the "
        "current *screen* on the map; `--from-here` is for mid-*edge* (you already tapped some "
        "of the recorded steps yourself).",
    ),
    (
        "Replay whole journeys in one call (flows)",
        "A flow is a Maestro-style YAML journey you can AUTHOR directly (no walking needed) or "
        "record. There are two recorders and they take different inputs: `aua flow save` "
        "materializes YOUR OWN last N actions, while `aua demo start` / `aua demo stop --save "
        "<name>` records a journey a PERSON performs on the device, costing zero agent turns "
        "for the walk - offer it when a human knows a path you would have to hunt for. `demo` "
        "needs the on-device helper, nothing else may drive the device while it runs, and it "
        "combines the accessibility stream with the raw kernel touch stream, so a press "
        "Android never announced is still recovered by name (`recovered_from_touches`). What "
        "neither source explains is reported in `gaps`, and `--save` refuses a draft with any. "
        "`aua flow save <name> --last N` is preview-first and writes nothing. It shows "
        "the selected origin/context segment, any package/context boundary or selector warning, "
        "value-free `selector_resilience`, arrival proof, the authoritative path, whether it "
        "already exists, and an "
        "exact decision-complete `save_call` (including `--force` for a collision). A collision "
        "preview also returns `invalid_mode_probe` with the exact typed error code/call for "
        "intentionally checking `--force` without `--save`; do not guess it. The write still "
        "rechecks atomically; preview is never race authorization. Add `--save` only after "
        "review (typed values become "
        "required `${PARAM_n}` placeholders — fill them in the file); `--dry-run` remains a "
        "deprecated non-writing alias. New recordings choose one safe selector per action: a "
        "unique stable resource id, otherwise a unique non-PII content description (`desc:`), "
        "otherwise unique stable non-PII text. A capture with no safe selector is refused with "
        "edit/re-record guidance. "
        "`aua flow run <name> --param K=V` drives the whole journey — launch, taps, waits, "
        "rich `assert` count/text/state/structure checks (`within`, `same_parent_as`, "
        "`contains_all`), explicit-axis `assert_order` including normalized `reading` order, "
        "named `screenshot` "
        "checkpoints, cross-app auth, even `goto:` steps — and on divergence returns the failing "
        "step index + remaining steps; fix and resume with `--from-step N`. Flows live under "
        "`<memory.dir>/flows/<package>/*.yaml`, one directory per app, with app-agnostic (and "
        "older flat) flows directly in `flows/` — `aua flow list [--app PKG]|show|delete`. Two "
        "apps may own the same flow name, so run/show/delete also accept `<package>:<name>`, and "
        "each `flow list` entry carries the `ref` that loads it; a bare name matching two apps is "
        "refused with both named rather than guessed. Delete is idempotent and "
        "returns `status: already_absent` when cleanup is already complete. `--dry-run` previews. Use a "
        "flow for any setup you repeat (reset account, log in, reach the screen under test) — "
        "one call instead of a dozen. Recorded flows persist `context_id`, `arrival_status`, and "
        "`arrival_screen` only when the current destination is freshly recognized in the same "
        "origin/context; replay verifies that known-screen name. An unmapped destination stays "
        "explicitly unverified unless the immediately preceding analyzed action satisfied a "
        "privacy-safe positive `--until` on this exact package/context/frame; only that captured "
        "proof becomes authored `arrival:` with source `satisfied_action_until`. No label is "
        "fabricated. Authored legacy `arrival:` predicates remain supported. Give reusable "
        "flows goal-facing `aliases:`. CLI and MCP accept one saved name, flow file, or inline "
        "YAML body. `--artifacts-dir DIR --evidence failures --junit` writes a portable "
        "flow/result/manifest/report/screenshot/platform-diagnostics/JUnit bundle; use evidence `all` only when "
        "per-step pixels are worth the runtime cost. "
        "Offline journeys may use `network_offline`, `network_restore`, `network_profile`, or "
        "`network_profile_restore` "
        "steps; environment-changing flows are risk-previewed before automatic selection.",
    ),
    (
        "Optional: let a fast model recover or explore (opt-in)",
        "If `planner.enabled` is set (+ an API key like GEMINI_API_KEY), you get two "
        "extras. (1) On a `goto`/`flow` divergence, add `--assist` and a fast planner LLM "
        "tries to recover in the same call (dismiss a popup, find the moved element) "
        "before handing off — the divergence hint tells you when it's worth trying. "
        '(2) `aua navigate "<goal>"` drives to a goal with no prior map AND records the '
        "path, so the next `aua goto` is a free deterministic replay. It's OFF by default "
        "and never touches the fast path; destructive taps still need `--allow-destructive`.",
    ),
    (
        "Drive by element ID",
        "`aua --format compact analyze` → a list of elements each with an integer `id` + bounds. "
        "An integer is bound to that one observation frame; after a dynamic update prefer the "
        "element's stable `resource_id`/`stable_key` via `--rid` or `aua resolve`, rather than "
        "guessing that the same number still means the same control. Within the current frame: "
        '`aua tap-and-analyze <id>`, `aua input-and-analyze <id> "text"`, '
        "`aua swipe-and-analyze up`, `aua key-and-analyze back`. "
        'Use `aua has "<text>"` (exit 0/1) to branch cheaply without parsing JSON.',
    ),
    (
        "Ask for the columns you want — never post-process JSON",
        "**`aua --format tsv analyze` is the default way to look at a screen**: one element "
        "per line, tab-separated, `#`-commented summary on top, and status-bar/unlabelled "
        "noise already dropped (`--all` keeps everything). Narrow it in the same call instead "
        "of piping into a filter: `--fields id,text,rid,clickable` (`rid` = the short tail; "
        "`resource_id` = the full selector), `--where-text <substr>`, `--where-rid <substr>`, "
        "`--clickable`, `--region x1,y1,x2,y2` (header = `--region 0,0,1080,300 --clickable`), "
        "`--limit N`, `--nonempty`, `--no-system`, `--no-ime`, `--meta <csv>` / `--no-meta` (drops the "
        "diagnostics — element counts, tier, duration; the `# goto:` routes and any `# aua asks:` "
        "question SURVIVE it, because those are things to do rather than facts about the call). "
        "On View-based apps "
        "add `--no-wrappers` to drop the app's own id'd layout scaffolding (`app_bar`, "
        "`content_frame`) — inert, unlabelled boxes that wrap something; leaves and "
        "addressable containers stay. Filters of different "
        "kinds AND together, repeats of one kind OR together, and **ids are never renumbered** — "
        "the id in a filtered row is the id `aua tap-and-analyze` takes. The same flags work on "
        "`--format json|compact` when you want machine-readable output.",
    ),
    (
        "Read interaction state, don't screenshot it",
        "Every element carries `checkable`/`checked`/`selected`/`scrollable`/`long_clickable`/"
        "`password` alongside `clickable`/`enabled`/`focused`. So *is this switch on?* is "
        "`aua --format tsv analyze --where-rid settingsSwitch --fields id,checkable,checked` — not a "
        "screenshot you have to look at. `selected` tells you which tab is active; `scrollable` "
        "tells you which container actually scrolls. These are **tri-state**: `true`/`false` when "
        "the accessibility node reported it, **empty/null when genuinely unknown** (a "
        "vision-derived element has no a11y attributes), so off never masquerades as unknown.",
    ),
    (
        "Verify by resource-id, not just text",
        '`aua has --rid <id>` (or `aua has "<id>" --by id`) checks a resource-id (a bare tail '
        'like "containerDetail" works) — and it finds non-interactive **container** ids that '
        "`analyze` prunes from the element list, so it's the reliable way to assert you "
        "reached a screen (Maestro-style `assertVisible: id:`). Guard an action with the same "
        "selector you act with: `aua has --rid saveButton && aua tap-and-analyze --rid saveButton`. "
        "`wait --for <id> --by id` and "
        "`scroll-to <id> --by id` take `--by id` too. If a screen is WebView/Compose-backed "
        "and its result text isn't in the tree at all, read it with `analyze --source vision`.",
    ),
    (
        "Act, then read the screen the action gives back",
        "IDs are only valid until the screen changes. By default every state-changing action "
        "(tap/input/swipe/scroll-to/key) returns the next screen inline in `observation` "
        "(elements with fresh ids). This is the default agent contract: action + `observation` "
        "covers the normal readback path, so you should skip `analyze` unless you need another "
        "filtered view. Every action response now also includes `observation_present`, "
        "`known_screen`, `stable_elements`, `action_diff_summary`, and `note`, so callers can "
        "branch on that single payload. `type → tap send` is two calls, not three, and `goto` "
        "returns the destination's `elements` too. "
        "The MCP surface makes that contract visible in the method name: "
        "`tap_and_analyze`, `input_and_analyze`, `scroll_and_analyze`, and the corresponding "
        "names for every observed action. The ambiguous short MCP names are not exposed. On "
        "the CLI, prefer the matching `tap-and-analyze` / `input-and-analyze` names; these "
        "explicit forms force the observation even if `--no-observe` is supplied. "
        "Pass `--no-observe` to skip it on action-only sequences. Action `observation` waits "
        "for a pixel change + idle (animation-aware) before dumping the tree, and a screen "
        "whose content is still streaming in has to hold still for one confirming sample — so "
        "you get the *next* screen, not a mid-transition snapshot with the list body missing. "
        "When returning through nested screens, use `back-until-and-analyze '<known_screen>'` "
        "when the destination is mapped, or pass positive `rid:`/`text:`/`desc:` evidence. "
        "It re-resolves a labeled Back on every frame and stops rather than navigating again "
        "from an unrecognized mapped frame. If the first Back icon is unlabeled, "
        "pass its id from the current observation once with `--back-id <fresh-id>`—AUA still "
        "freshness-checks it and does not reuse that numeric id on later frames. "
        "A numeric tap id always names that exact node: AUA refuses to reinterpret a caption id "
        "as a sibling control. Use `aua target` and the acting control's id/rid when needed. "
        "The observation is **compact by default** "
        "(`id,text,desc,rid,clickable,enabled,checked,selected`, app nodes only); "
        "widen it with `--observe-fields all` or any field list. You therefore never need the "
        "`--no-observe` + `analyze` pair to get a cheap read — that pair costs two round trips "
        "for one screen. "
        "**That settle can only wait ~1.1s (max 1.6s).** A slower screen makes the action "
        "report `nothing changed` for a tap that did land, and `stale_risk` appears in "
        '`detail`. That is *not* evidence the tap missed: it cannot tell "no effect" from '
        '"not yet", so **never re-tap on it** — a second tap means a second submit. When you '
        "know what should come next, say so and the wait becomes evidence-based with your "
        'budget instead of the settle timer: `--until "rid:resultsPanel"` or `--until '
        '"text:Results,!text:Loading"` (with `--until-timeout MS`). An action-bound `--until` '
        "must include positive arrival evidence; an absence-only predicate belongs in a "
        "standalone await, which answers `absence-satisfied` rather than `satisfied` — proof "
        "that what you left is gone, never proof of what arrived. The response "
        "grammar ANDs comma-separated terms; escape a literal comma as "
        "`text:Hello\\, friend`. Before accepting a negated text miss, and before timing out "
        "a positive text term, AUA verifies hierarchy text with its available OCR path. The response "
        "then carries `await_outcome`: `satisfied` / `absence-satisfied` / `screen-changed` "
        "/ `timeout`. An "
        "action-bound wait can instead return `settled-unmet` when two stable, non-loading "
        "destination frames prove the positive predicate names the wrong screen; its "
        "`arrival_mismatch` gives stable replacement predicates without repeating the action. "
        "`await_terms` says which term is missing, and `unknown_selectors` names any unmet "
        "`rid:` the app map has never recorded on any screen — that one cannot arrive, so "
        "correct it from its `nearest` list instead of waiting again. Prefer "
        '`wait --for "<text>"` for known targets; wait/await observations accept `--no-meta` '
        "with the same meaning as analyze. Reserve `wait --for-stable` for "
        "generation / loading / video.",
    ),
    (
        "Wait on state, never sleep",
        '`aua wait-and-analyze --for "<text>"` waits for text to appear; `aua wait-and-analyze --for-stable` returns once '
        "the screen stops visually changing (grid pixel-hash; looping spinners/video are "
        "auto-masked so they don't block). `--after-change` additionally requires a first change, "
        "visual settle, and a bounded quiet confirmation; a later result that replaces a stable "
        "loading shell restarts settling. Prefer an explicit final goal over either generic wait; "
        "never fixed sleeps.",
    ),
    (
        "Detach only waits that may outlive one agent call",
        "For a slow read-only condition, start `aua job start await --predicate "
        "'rid:result,!text:Loading' --timeout-ms 180000`. It returns immediately with a durable "
        "`job_id`; reconnect with `aua job status <id> --recent-output`, wait at most ten seconds "
        "with `aua job wait <id>`, or cancel and briefly await a terminal acknowledgement with "
        "`aua job cancel <id>`. Lifecycle events persist with the job, so a reconnect explains "
        "whether it queued, ran, acknowledged cancellation, or was interrupted. "
        "The warm daemon remains responsive, but serializes every other device operation behind "
        "the job so no tap or analysis races it. `wait-stable`, `wait-changed`, and "
        "`wait-after-change` are also supported. Jobs do not detach mutating actions.",
    ),
    (
        "Wait on the backend when the screen cannot tell you",
        "`await` / `--until` terms are not limited to what is drawn. `net:<[METHOD ]PATH[=STATUS]>` "
        "waits for a completed HTTP exchange (`net:POST /v1/chat`, `net:/v1/chat=200`) — "
        "mitmproxy's response hook fires at *stream completion*, so it is the honest signal for a "
        "streamed chat turn; it needs `aua proxy start` running. `log:<substring>` matches logcat "
        "since the wait began and needs no proxy, but is only as good as what the app logs. "
        'Terms are ANDed, so `--until "net:POST /v1/chat,text:x ="` reads as *the backend '
        "replied and the screen shows it* — which matters because a streamed LaTeX answer reaches "
        "the hierarchy as U+FFFD, so no `text:` term alone can confirm it arrived. Both take a "
        "baseline when the wait starts, so the previous turn's response can never satisfy this "
        "one. Still not network idle: this app never is.",
    ),
    (
        "Use verified network isolation, not airplane mode",
        "Android may keep Wi-Fi active after `airplane on`, so that command is never proof the "
        "device is offline. Use `aua network offline --verify`: it snapshots airplane, Wi-Fi, "
        "and mobile-data controls, disables them, and refuses success while ConnectivityService "
        "still reports an active default transport. Inspect with `aua network status`, then "
        "always run `aua network restore`; the restore point is deleted only after read-back "
        "matches the saved controls and prior connectivity.",
    ),
    (
        "Use reversible profiles for constrained networks",
        "Run `aua network profile list`, then `profile apply wifi-only|cellular-only|slow|lossy`. "
        "Every profile saves the original conditions, verifies live evidence, and refuses to "
        "stack with offline mode or another profile; always finish with `network profile "
        "restore`. `slow` uses Android Emulator bandwidth/latency shaping. `lossy` adds outbound "
        "packet loss with `tc "
        "netem`, requires a rootable Google APIs AVD (`--needs root`), and refuses to replace an "
        "unknown existing traffic shaper.",
    ),
    (
        "One agent, one emulator — leases are automatic",
        'With several agents running at once, every one of them otherwise resolves to "the '
        "only/first device\" and they drive each other's screens; nothing errors, the results "
        "are just wrong. So each command **claims a lease** on the device it uses and keeps "
        "you on the same one (element ids, app state and the learned map are all per-device). "
        "Lease records live in one host-wide registry even when callers isolate per-run caches. "
        "Selection happens before transport selection, so each serial gets its own daemon "
        "socket and a warm daemon is forbidden from claiming a different device than it drives. "
        "You need do nothing: no lease command and no owner prompt. By default AUA derives the "
        "long-lived agent process, records its PID plus start token, and gives that normal owner "
        "exactly one sticky device "
        "across its short-lived shell/runner commands. On the next device request, a dead owner "
        "or reused PID is immediately treated as free and another agent is assigned "
        "automatically; it does not wait out the TTL. Explicit `--owner <agent>` / `$AUA_OWNER` "
        "remain friendly labels, while AUA transports the caller PID plus start token separately, "
        "so those leases are process-bound too—even through a warm daemon. "
        "Once leased, omit `--serial` from ordinary device commands: the lease is the routing "
        "source, and repeating a physical target can become stale after reassignment. Use an "
        "explicit serial only to select a user-requested target before acquisition or for an "
        "administrative/fanout command that intentionally names one device. Asking for a different "
        "target first returns `lease_switch_required` without changing either lease; only "
        "`aua lease acquire <new> --replace` acknowledges cleaning and releasing the old device. "
        "Fanout keeps this invariant by using one stable scoped logical owner per explicit target. "
        "To delegate the same running device without resetting it, the holder runs `aua lease "
        "transfer <serial>` and passes its one-time token to the child, which runs `aua lease "
        "accept <token>`. The source is frozen until accept, cancel-transfer, or the five-minute "
        "token expiry; goal-session ownership remains separate. That explicit pending transfer "
        "is the sole exception to immediate process-death release: its reservation survives a "
        "source crash only until the token expires, so the spawned child cannot lose the device. "
        "Ask for what the device must support with `aua session start --needs root,play,proxy`; "
        "AUA selects a capable free device or provisions a matching AVD automatically. "
        "**Exit 9 (`device_leased`) means another agent holds it.** If the serial was only a "
        "redundant stale pin, omit it and stay on your existing assignment. If the user explicitly "
        "requested that target, never redirect: wait, provision with user intent, or reconcile "
        "the holder identity. `aua lease list` shows who holds what, and `aua lease "
        "release` hands one back early. A crashed agent therefore blocks nobody and "
        "there is nothing to clean up.",
    ),
    (
        "Device changes reset themselves when your lease ends",
        "A proxy, verified-offline mode, a moved clock and disabled animations are *device-global* "
        "settings: they outlive your process. AUA journals how to undo each one before it makes "
        "the change, so the next agent never inherits a device reporting 'Offline' for reasons "
        "unrelated to the app. The undo runs when your lease goes (a detached watchdog does it "
        "even if no further `aua` command is ever run), so you do not have to remember. "
        "`aua teardown status` shows what is still pending and on which device; `aua teardown "
        "run` forces it now. An inherited device can also be pointed at a proxy *nothing* owns — a "
        "partial teardown leaves `http_proxy` set with the tunnel gone, so every app request "
        "fails with `ConnectException` and no ledger teardown can clean it up. AUA-started "
        "emulators detect and clear that confirmed black hole automatically before app launch; "
        "for devices started outside AUA, `aua proxy status` (and `aua doctor`) diagnoses it and "
        "`aua proxy stop` un-points it. You should still `aua network restore` / `aua clock set --restore` "
        "inside a test that depends on the restored state — the reset is a safety net for when "
        "you walk away, not a substitute for the assertion you actually wanted. Changes on a "
        "device someone else holds are reported, never undone.",
    ),
    (
        "Stop the daemon when done",
        "`aua --serial <serial> daemon stop` releases the warm connection. Pass the serial: a "
        "daemon started for a device listens on `daemon.sock.<serial>`, so an unpinned stop "
        "signals a socket nobody uses and reports what it did instead of a bare success. "
        "`aua daemon status` lists every live daemon with the exact command that stops it, and "
        "`aua daemon stop --all` ends all of them at once.",
    ),
]

# ``guide --brief`` is what a fresh agent can afford to read in the middle of a task. Keep this
# as a deliberately small decision loop rather than reprinting the full operating manual's every
# feature and exception. The generated skill is smaller again and links here progressively.
BRIEF_SESSION_PROTOCOL: list[tuple[str, str]] = [
    (
        "Start from the user's goal",
        'Run `aua session start --goal "<what you must verify>"`. It attaches and leases '
        "automatically, starts/reuses the warm transport, observes once, ranks a verified "
        "`goto`, matching saved flow, proven deeplink, or manual analyzed action in that order, "
        "and returns one exact CLI and MCP `recommended_call`. Follow that call instead of "
        "calling analyze again or inventorying commands. Ordered `goal_progress` accompanies "
        "every result; if the foreground is unrelated, add `--app <package>` / `--package` so "
        "the launch and its observation are folded into bootstrap. Carry its `--phase-done` / "
        "MCP `phase_done` checkpoint on your next call "
        'after evidence is visible, rather than spending a call on progress. Use `aua capabilities --goal "…"` only '
        "when you need another goal-specific capability, and finish reversible work with "
        "`aua session finish`. In its review, `top_level_calls` counts caller-visible invocations and "
        "equals `lifecycle_calls` + `task_calls`; `journal_events` additionally includes "
        "`folded_internal_events` such as an action-bound wait. The embedded snapshot precedes "
        "the current review/finish "
        "call, so `reporting_call_included` is false and "
        "`top_level_calls_including_reporting_call` adds it.",
    ),
    (
        "Attach automatically and clean up only what you started",
        "First call `session start`: it scans the host-wide pool, frees dead owners, matches "
        "capabilities, and selects or boots a target. Never list/start/acquire manually. The sticky "
        "lease stays implicit; omit `--serial` from ordinary commands. Never steal. Add "
        "`--needs root,play,proxy`, `--headed`, or `--audio` only when required. "
        "`session finish` cleans up.",
    ),
    (
        "Observe once and use stable selectors",
        "Reuse the compact observation returned by session start. Without a goal session, use "
        "`aua --format tsv analyze --fields id,text,rid,clickable`. Integer ids belong "
        "only to that observation frame. On dynamic screens, prefer `--rid <resource-id>` or an "
        "element's `stable_key`; after any state change, use the action's returned observation or "
        "`aua resolve <stable_key>` instead of replaying an old numeric id. A numeric id is fine "
        "only while its frame is still current. Numeric taps and long-presses refuse to redirect "
        "a caption into a sibling control subtree; name the actual acting control instead. "
        "Inline `--answers` / `--phase-done` are bookkeeping: an invalid annotation is returned "
        "as `annotation_warnings` and never cancels the requested device action.",
    ),
    (
        "Choose the highest-level safe navigation",
        'Use an offered verified `aua goto "<goal>"` first, then a matching `aua flow run`, then '
        "a known deeplink, and manual controls last. A delivered deeplink intent is not arrival: "
        "accept it only when the returned observation/activity proves the destination. Inspect a "
        "plan before any route that may delete, pay, send, sign out, mutate settings/data, use a "
        "deeplink, or leave the app. `goto` refuses those routes before its first step and names "
        "the required opt-in; exploration never supplies authorization for those effects.",
    ),
    (
        "Act and consume the returned screen",
        "Use `tap-and-analyze`, `input-and-analyze`, `swipe-and-analyze`, and `key-and-analyze`; "
        "their `observation` already contains fresh ids. Do not immediately call `analyze` again. "
        "Verify the exact depth the user named: an intermediate card/detail page with an "
        "`Open` control does not prove that a conversation, thread, document, or tool itself "
        "opens. Continue through the returned `next_actions` until the requested content and "
        "interactive affordance are visible. Never invent a rid that was not returned. "
        "For your action's expected result add `--until 'rid:resultCard'` or "
        "`--until 'text:Results,!text:Loading'`; escape a literal comma as "
        "`text:Hello\\, friend`. For a "
        "nested journey, return in one bounded call with `aua back-until-and-analyze "
        "'<known_screen>'` (or `'rid:<destination>'`) instead of replaying frame-local Back "
        "ids. A bare value is a mapped `known_screen`; text/rid/desc evidence keeps its prefix. For an "
        "unattached network-driven update use `wait-and-analyze --after-change --observe`. Prefer "
        "a positive final affordance over a generic spinner disappearance. If a daemon call "
        "reports `daemon_outcome_unknown`, never repeat the action: wait, then inspect one fresh "
        "screen. AUA treats a live busy daemon as the device owner and will not spawn or fall "
        "back to a competing controller; superseded daemon cleanup cannot remove its "
        "successor's ownership files. If a read-only wait may exceed one agent call, use `aua job "
        "start await --predicate '…'`; reconnect by its id with `job status`, or cancel it. Other "
        "device operations stay serialized until that job ends.",
    ),
    (
        "Let automatic perception escalate",
        "Hierarchy is fastest and supplies exact selectors. Leave source selection on auto so AUA "
        "adds OCR/vision only for a thin or opaque tree; paid grounding requires explicit `--deep`. "
        "Filter in the AUA call (`--where-rid`, `--where-text`, `--clickable`, `--region`) instead "
        "of piping a large JSON response through ad-hoc scripts.",
    ),
    (
        "Improve the map before probing shortcuts",
        "Read inline `known_screen`, routes, and research questions. Use "
        "`aua map --audit --summary` for health counts and `aua explore plan` for work ordered as "
        "map issues, safe dead ends, then speculative external intents. `goto` must verify every "
        "hop and stop on divergence; resume from the actual screen rather than replaying blind.",
    ),
    (
        "Keep the playbook concise and current",
        "Use `aua about` only when you need a recipe or quirk. Before `remember`/`knowledge add`, "
        "check for an existing equivalent fact and update its evidence instead of adding a "
        "near-duplicate. Treat version-, flag-, locale-, and account-dependent observations as "
        "scoped facts, not universal rules. When a new build contradicts one, preserve its "
        "history with `aua knowledge stale <id>` and add the replacement with fresh evidence. "
        "`about` and `orient` render only the current accepted projection; replacements deduplicate "
        "by recipe/deeplink identity and stale or wrong-version facts keep evidence but disappear.",
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
    (
        "1",
        "`has`: text not present · OR unexpected internal error (structured `internal_error` on stderr)",
    ),
    ("2", "usage error"),
    ("3", "no device / device error / `wait --for-stable` timeout"),
    ("4", "provider error (fallback chain exhausted)"),
    ("5", "config error"),
    ("6", "selector matched nothing (`--rid`/`--text`/`--desc`)"),
    ("7", "selector matched several candidates (disambiguate with `--index`/`--first`)"),
    ("8", "`aua expect-and-analyze` / `aua suite run` assertion failed"),
]

#: The smallest set of calls that takes a caller from "no idea" to a driven screen. Emitted by
#: the unknown-command error so that one wrong guess produces the orientation the guide would
#: have given — an agent that never read the manual reads this instead, once, and then knows.
ORIENTATION: tuple[tuple[str, str], ...] = (
    (
        'aua session start --goal "<what you must verify>"',
        "attach, observe once, and receive the safest exact next call plus cleanup",
    ),
    (
        "aua --format tsv analyze --fields id,text,rid,clickable",
        "READ as rows — `rid` is the app's resource-id (pass with --rid); `id` is this call's "
        "ordinal (positional, renumbered every analyze)",
    ),
    (
        'aua goto "<goal from # goto:>"',
        "use the offered verified route before a deeplink or manual taps",
    ),
    (
        "aua flow run <name from # flows:> --param K=V",
        "use a matching saved journey for repeated multi-step setup",
    ),
    (
        'aua open-and-analyze "<offered deeplink>"',
        "only when no goto/flow covers the goal; `ok:true` only means the intent was "
        "delivered — check `verified` (false/absent = no destination change confirmed) "
        "before trusting the screen it returns",
    ),
    (
        "aua tap-and-analyze --rid <resourceId> --until 'text:<label>'",
        "manual fallback: ACT, wait, and get the settled screen — one call",
    ),
    (
        "aua input-and-analyze --rid <resourceId> \"text\" --until 'rid:<result>,!text:Loading'",
        "TYPE — text is positional, and --until belongs HERE, not on a later analyze",
    ),
    (
        "aua back-until-and-analyze '<known_screen>' [--back-id <fresh-id>]",
        "RETURN through nested screens in one call; use rid:/text:/desc: when it is not mapped",
    ),
    (
        "aua job start await --predicate 'rid:<ready>,!text:Loading' --timeout-ms 180000",
        "DETACH only a long read-only wait; reconnect with `aua job status <job_id>` instead of restarting it",
    ),
    (
        'aua --answers <id>="<name>" <your next command>',
        "answer a `# aua asks:` line in passing — names that screen so `goto` / `map --find` "
        "can reach it, this run and every later one",
    ),
    ('aua has "Sign in"', "cheap presence check (exit 0 found / 1 not)"),
    (
        "aua app restart-and-analyze <pkg> --activity <activity>",
        "back to a known screen — no adb, no sleep",
    ),
    (
        'aua logcat --grep "FATAL EXCEPTION|ANR in" --since last-action',
        "WHEN A RESULT LOOKS WRONG — why the app died, without dropping to raw `adb`. Reach for "
        "this the moment a screen is not the app you were driving",
    ),
    (
        "aua guide --brief",
        "the manual, short form — the full `aua guide` is ~46KB and the loop above covers most "
        "tasks; reach for it only when something above did not answer you",
    ),
)

#: What agents type when they mean something else, mapped to what they meant. Click's built-in
#: suggestion is string distance over command names, which is actively misleading here: it sent
#: `tree` to `target` and `state` to `paste`, and offered nothing at all for `ui`, `dump` or
#: `elements`. These are answered, never aliased — see `_REMOVED_ACTION_ALIASES` in cli.py for
#: why a wrong name that quietly works is worse than one failed call.
COMMAND_SYNONYMS: dict[str, str] = {
    "screen": "analyze",
    "ui": "analyze",
    "dump": "analyze",
    "tree": "analyze",
    "elements": "analyze",
    "state": "analyze",
    "read": "analyze",
    "hierarchy": "analyze",
    "snapshot": "analyze",
    "page": "analyze",
    "view": "analyze",
    "press": "tap-and-analyze",
    "touch": "tap-and-analyze",
    "type": "input-and-analyze",
    "enter": "input-and-analyze",
    "fill": "input-and-analyze",
    "sleep": "wait-and-analyze",
    "pause": "wait-and-analyze",
    "find": "has",
    "exists": "has",
    "check": "has",
    "back": "key-and-analyze",
    "home": "key-and-analyze",
    "sideload": "install",
    "apk": "install",
    "adb-install": "install",
}


KEY_FLAGS: list[tuple[str, str]] = [
    (
        "global, BEFORE the subcommand",
        "`--format json|pretty|compact|tsv|delta|msgpack` (`delta`/`msgpack` = analyze; `tsv` "
        "also renders an action: its envelope becomes `#key=value` lines — `# change.text_added=…` "
        "is usually the whole verdict — followed by the observation's rows; "
        "`delta` omits elements when unchanged; `msgpack` is AUA1 binary/base64), "
        "`--serial`, `--config`, `--profile`, `--timeout`, `--log-level`, `--no-cache`, "
        '`--answers <task-id>="<name>"` (repeatable; answers the `meta.ask` question about the '
        "screen you are on, applied before the command itself runs), "
        "`--with-image` (session default: attach raw screenshots on analyze/actions — "
        "prefer off; use only when you must SEE pixels)",
    ),
    (
        "analyze",
        '`--source auto|hierarchy|vision`, `--query "<nl>"`, `--deep`, `--cheap`, '
        "`--strategy <tier>`, `--annotate [path]`, `--with-image [path]` (also save the raw "
        "screenshot; path lands in `meta.raw_image`), `--with-ocr/--no-ocr`",
    ),
    (
        "analyze — views (use these instead of post-processing JSON)",
        "`--fields <csv>` (`id,text,rid,desc,bounds,center,type,clickable,enabled,focused,"
        "checkable,checked,selected,scrollable,long_clickable,password,resource_id,parent,source,"
        "confidence`), `--nonempty`, `--no-system`, `--no-ime`, `--no-wrappers`, `--all`, "
        "`--where-text <substr>`, "
        "`--where-rid <substr>`, `--clickable`, `--region x1,y1,x2,y2`, `--limit N`, "
        "`--meta <csv>`, `--no-meta` — repeatable where it makes sense, and free of a device "
        "round-trip when the flags are wrong (bad name → exit 2 listing the valid ones)",
    ),
    (
        "screenshot",
        "`[PATH] | --out PATH`, `--region x1,y1,x2,y2` (crop before writing), `--scale <factor>`, "
        "`--max-width <px>`, `--annotate` (full-screen only) — crop/downscale when you must LOOK "
        "at something, so one header icon doesn't cost a 1080x2400 PNG in image tokens",
    ),
    (
        "daemon / orient",
        "`daemon start --quiet` skips the app-orientation blob (48 screens, mined deeplinks, "
        "notes); read it deliberately with `aua orient` instead — useful once per session, noise "
        "on every restart",
    ),
    (
        "job",
        "`job start await --predicate <terms> [--timeout-ms N] [--poll-ms N]` or start "
        "`wait-stable|wait-changed|wait-after-change`; reconnect with `job status <id> "
        "--recent-output`, make a "
        "bounded `job wait <id> --timeout-ms N` call (maximum 10000), `job cancel <id>`, or "
        "`job list`. Jobs are read-only and serialize other device calls while active.",
    ),
    (
        "emulator",
        "`emulator list|status|recommend-proxy|ensure-proxy [--name aua_proxy] [--api 30] "
        "[--force] [--start]|start [--avd NAME] [--headless|--windowed] [--gpu host|…] "
        "[--audio] [--parallel] [--port N] [--read-only] [--owner TAG] [--idle-stop 900] [--wait N]|"
        "stop [--serial emulator-5554|--avd NAME|--owner TAG|--mine|--all]` — boot headless "
        "for unattended verify (Mac defaults to `-gpu host`); **`--parallel` for multi-agent** "
        "(unique port + read-only + owner); **always stop yours** (`--serial` / "
        "`AUA_OWNER=… --mine`); idle watchdog auto-stops after `--idle-stop` as backup; "
        "`ensure-proxy` creates a small rootable google_apis AVD "
        "(HTTPS proxy system CA — Play Store AVDs refuse `adb root`)",
    ),
    (
        "mic",
        "`mic inject PCM-WAV [CONTROL-ID]` or `--rid/--text/--desc <control>`; "
        "`--control-mode hold` (default) keeps it down for pre/audio/post, while `toggle` "
        "uses one tap to start and one to stop. Toggle requires an enabled, clickable, "
        "initially-off control. The command returns the post-action observation. Input is "
        "uncompressed PCM WAV: U8/S16, mono/stereo, <=48 kHz, <=5 minutes. On macOS, `mic speak TEXT "
        "[--voice NAME] [--rate WPM]` uses `/usr/bin/say` and the same control path. "
        "Needs the `[audio]` extra and an emulator started with `--audio`; physical devices "
        "do not expose this API. `mic_delivery_uncertain` means samples may already have been "
        "delivered: inspect `error.result.observation` and never retry the voice action. If "
        "`mic_emulator_unavailable` occurs, inspect `aua devices` first. Emulator 36.4.10 "
        "permits one AUA injection attempt per boot; `mic_repeat_unsafe` requires a restart. "
        "`mic_delivered_release_failed` means audio arrived but control cleanup failed. "
        "`mic_toggle_start_uncertain` / `mic_toggle_stop_uncertain` mean recording may be "
        "active: protect privacy, inspect the forced observation, and never tap/retry blindly.",
    ),
    (
        "has",
        "`--by text|id|desc` (id finds pruned containers), `--match exact|contains|regex`, "
        "`--ignore-case`, `--ocr-fallback/--no-ocr-fallback`, `--timeout <ms>`",
    ),
    (
        "wait",
        '`--for "<text>"` (`--by id`, `--absent` = wait until it disappears), `--idle`, '
        "`--for-stable`, `--changed` (any hierarchy fingerprint change), `--interval`, "
        "`--settle`, `--timeout`, `--observe` (fresh ids, even on a miss). On timeout exit 3 "
        "with detail naming `--match` mode, fields searched, closest candidates — and a hint "
        "if the pattern looks like regex under `--match contains`",
    ),
    (
        "fanout",
        "`aua fanout [--serials a,b] [--parallel] <cmd…>` — run one subcommand on many "
        "devices (each gets `daemon.sock.<serial>`); gathers JSON per serial",
    ),
    (
        "open",
        "`<uri> [--package <pkg>]` — **pins the foreground package by default** so "
        "prod+dev installs never hit 'Open with…'; `--no-package-pin` to test the "
        "chooser; if a chooser still appears, errors naming the competing apps",
    ),
    (
        "resolve",
        "`<id|stable_key>` — remap a previous-frame id (or `rid:…` key) onto the current "
        "screen after IDs churn",
    ),
    (
        "app",
        "`exists|status <pkg>`, `launch <pkg> [--activity .Entry] [--clear --yes]`, "
        "`stop|kill|clear|grant`. `exists` exits 1 when absent; status is informational. "
        "`clear` / `launch --clear` wipe ALL app data (typically flags + login) — **requires "
        "`--yes` / `--yes-wipe-flags`**; re-apply flags afterwards",
    ),
    (
        "shell",
        "`COMMAND … [--shell-timeout 30]` — run one bounded read-only diagnostic on the "
        "leased target. Every argv item is quoted before Android's remote shell parses it. "
        "Stdout/stderr are each capped at 256 KiB; output reports truncation plus "
        "serial/argv/exit_code/mode. Unknown or mutating verbs are refused; use `--` before "
        "command flags, e.g. `aua shell -- logcat -d`",
    ),
    (
        "install",
        "`<app.apk> [--launch] [--reinstall | --fresh --yes] [--grant] [--package <pkg>]`. "
        "Never shell out to `adb install`. Idempotent: an app already present at the bundle's "
        "version is skipped, so re-running is one package query instead of a multi-second push; "
        "the bundle names its own package. `--launch` returns the landing screen, so install + "
        "open + observe is one call. `--reinstall` pushes anyway and keeps app data; `--fresh` "
        "uninstalls first — the only mode that survives a signing-key change "
        "(`INSTALL_FAILED_UPDATE_INCOMPATIBLE`) and the only one that wipes data. "
        "`aua emulator start --apk <app.apk> --launch` and `aua session start --apk <app.apk>` "
        "fold boot + install + launch into that same single call",
    ),
    (
        "db",
        "`list <pkg>`, `schema <pkg> <db> [--table NAME]`, "
        '`query <pkg> <db> "SELECT …" [--params JSON --limit N --coherent]`, '
        '`execute <pkg> <db> "UPDATE …" --yes`, `backup|backups|restore`. '
        "Queries preserve UI by default; `--coherent`, schema, and state-changing operations "
        "stop/relaunch the app. Execute backs up first and accepts data mutations only; restore "
        "preserves the current state as a new safety backup",
    ),
    (
        "clipboard / paste / copy",
        "`clipboard set|get`, `paste`, `copy --rid/--text/--desc` (Maestro copyTextFrom / "
        "pasteText / setClipboard)",
    ),
    (
        "erase",
        "`erase [ID] --chars N` (Maestro eraseText; omit ``--chars`` to clear the whole field)",
    ),
    (
        "location / orientation / airplane / network / media / mic / record / clock",
        "`location set LAT,LON`, `orientation set|get`, `airplane on|off|toggle`, "
        "`network status [--verify]|offline --verify|restore`, `network profile "
        "list|apply|status|restore` (all modes are saved, verified, and reversible), "
        "`media add PATH`, `mic inject PCM-WAV [CONTROL-ID] [--control-mode hold|toggle]`, "
        "`mic speak TEXT [CONTROL-ID]`, "
        "`record start|stop PATH`, `clock set --ms <unix-ms>` / "
        "`clock restore` (time travel invalidates auth — always restore)",
    ),
    (
        "logcat",
        "`logcat mark [NAME]`, `logcat --grep PAT [--since mark|last-action] [--tag T] "
        "[--json]` — bracket API/analytics verification around an action",
    ),
    (
        "suite",
        "`suite run PATH.yaml [--continue]` — AC checklist (has/expect/wait_for) with "
        "per-item pass/fail + summary; exit 8 if any fail",
    ),
    (
        "capture",
        "`capture status|last [--seconds N|--since last-action] [--region center]|"
        "export PATH.gif|explain [--llm]|on|off|prune|sidecar start|stop` — always-on "
        "rolling screencap with the daemon (deduped frames + diff summary / GIF); see "
        "`meta.capture_hint` / action `capture_hint` after fast transitions; suite failures "
        "attach `capture last --since last-action`. Sneak-peek a headless agent live: "
        "`aua dashboard` (separate process — enables capture via daemon or sidecar, "
        "opens http://127.0.0.1:8765 in a live device **grid** by default)",
    ),
    (
        "dashboard",
        "`dashboard [--serial …] [--grid|--detail] [--port 8765] [--no-open] [--poll-ms 500]` — "
        "localhost sneak-peek; defaults to a live device **grid** that opens with or without "
        "a device attached, discovers later emulators and shows lease/idle-watchdog "
        "state (click a tile for journal/map); the detail view has a proxy panel — health, "
        "live exchanges, which rules are armed and which fired, and click-a-request-to-arm "
        "a stub or rewrite from what it just saw; "
        "the detail view browses debuggable app "
        "databases, schema, bounded queries, restore points, and guarded writes; enables "
        "capture; does not stop the agent (Ctrl-C closes the dashboard only)",
    ),
    (
        "dev",
        "`dev show`, `dev anim off|restore`, `dev crashes on|off`, "
        "`dev profile ac|default` (AC: anim off + crashes on; always restore)",
    ),
    (
        "a11y",
        "`a11y scroll <id|--rid …> [--forward|--backward]`, "
        "`a11y action <id> CLICK|LONG_CLICK|SCROLL_FORWARD|…`; analyze `--no-ime`",
    ),
    (
        "flags",
        "`flags set <pkg> KEY=VAL…`, `flags apply file.yaml` — needs a `flags.templates` "
        "entry for the package (set-flags schemes are app-specific). Writes, then **verifies** "
        "against the app's shared_prefs (`applied`/`ignored`; a dropped key exits 8, "
        "`--no-verify` to skip) and **restarts** the app (`--no-restart` to skip) because "
        "flags read at cold start ignore a live-process override. A successful restart activates "
        "a deterministic map context carrying the verified flags. When `flags.prefs_files` or "
        "`flags.context_keys` is configured, ordinary `analyze` also discovers already-active "
        "experiment/treatment/variant/flag values and switches map context automatically.",
    ),
    (
        "proxy / mock",
        "`proxy start|stop`, `proxy status [--no-heal]`, "
        "`mock map METHOD PATH [--status N --body '{…}']`, "
        "`mock rewrite METHOD PATH [--host H --status N --header 'K: v' --set a.b=<json> "
        "--delete a.b --replace old=>new --times N]`, "
        "`mock record start|stop NAME`, `mock replay NAME` (optional `[proxy]` extra). "
        "**map vs rewrite**: `map` answers from the rule and the server never sees the "
        "request; `rewrite` lets the request through and patches the real response, which "
        "is how you reproduce a server-side condition (a 429, a missing field) you cannot "
        "trigger on demand. A `rewrite` with no `--host` and a catch-all path is refused: "
        "it would also intercept the platform's own connectivity probes and the device "
        "would just look offline. "
        "`proxy status` answers *is interception actually working* with a `state` of "
        "`unproxied` (clean device) / `healthy` / `degraded` (yours, broken) / `foreign` "
        "(someone else's working proxy — your mock rules are inert against it) / `blackholed` "
        "(pointed at a dead port: every app request fails with `ConnectException`, visible only "
        "in logcat, and exit 1). `ok` means this device's network path is sane; `intercepting` "
        "means traffic reaches a proxy *you* own — they are different questions. "
        "Mock rules are a host-side sidecar that outlives the command that armed it and can "
        "carry another session's stubs — run `mock list` before arming anything to see the "
        "live mode/owner/rules, and `mock clear` (or `mock rm <id>` for one) to start clean; "
        "`mock map` warns when it would append onto rules it did not create.",
    ),
    (
        "map",
        '`--app <pkg>`, `--brief`, `--screen <name>`, `--depth N`, `--find "<goal>"`, '
        "`--context <id>`, `--all-contexts`, `--audit [--summary]`, `--json`",
    ),
    (
        "goto",
        "`<goal>` (fuzzy), `--plan` (annotated route, no taps), `--max-steps N`, "
        "`--allow-unsafe` (after reviewing disclosed non-navigation effects), "
        "`--allow-destructive`, `--assist` (opt-in planner recovery), `--from-here` "
        "(resume mid-edge after a manual hop / divergence)",
    ),
    (
        "flow",
        "`run <name> [--param K=V] [--file PATH] [--dry-run] [--from-step N] "
        "[--no-allow-destructive] [--assist]`, `save <name> [--last N] [--save] "
        "[--force] [--dry-run (deprecated)]`, "
        "`list|show|delete`. Steps incl. `launch_app`/`stop_app`/`open_link`/`goto`/`flow` "
        "/ `dev_profile` / `a11y_scroll` / `flags_apply` / `prefs_write` (set the app's own "
        "`shared_prefs` on a debuggable build: `{file: settings.xml, values: {k: v}}` — the "
        "YAML type picks the Android type, the app is stopped for the write and relaunched "
        "unless `relaunch: false`; destructive, so `goto` never replays it, and the previous "
        "file is journalled for `aua teardown`) "
        "/ `proxy_start`/`stop` / `mock_replay` "
        "/ `network_offline`/`network_restore`/`network_profile`/`network_profile_restore` "
        "/ `wait_ms` (fixed delay in ms, e.g. for an async write to land — bounded by "
        "`perf.max_wait_ms` like every other wait (see `meta.caller.wait_ceiling_ms`); prefer "
        "`wait_for`/`wait_stable` when a UI "
        "condition can say it instead). Top-level `aliases:` help "
        "goal matching; `arrival_screen:` is mapped-screen proof for new recordings while legacy "
        "`arrival:` predicates remain supported (a `flow:` step runs a saved flow inline).",
    ),
    (
        "open / about / remember",
        "`open <uri>` deeplink; `about` app playbook; `remember …` teach it",
    ),
    (
        "knowledge / reconcile",
        "`knowledge list|show|add|stale`; `reconcile plan|answers|submit|status|apply|rollback` "
        "(external-agent JSON contract, transactional correction)",
    ),
    (
        "explore",
        "`mine <repo> --app <pkg>` harvests deeplink shortcuts from source into the "
        "playbook; `plan` returns a prioritized crawl worklist (probe deeplinks, expand "
        "dead-end screens) whose results auto-record",
    ),
    (
        "navigate (opt-in planner)",
        "`<goal>` (natural language), `--until <text>`, `--max-steps N`, "
        "`--allow-destructive`, `--save-flow <name>` — needs `planner.enabled`",
    ),
    (
        "observed actions (`*_and_analyze` MCP / `*-and-analyze` CLI)",
        "return the post-action screen inline (`observation`, fresh ids), and the explicit "
        "names cannot disable that readback. Prefer `hide-keyboard-and-analyze` over "
        "`key-and-analyze back` when the IME is covering the tree. For nested navigation use "
        "`back-until-and-analyze '<known_screen>'` for a mapped destination, or pass positive "
        "`rid:`/`text:`/`desc:` evidence; use `--back-id <fresh-id>` only when the first "
        "app-owned Back icon is unlabeled",
    ),
    (
        "logcat",
        "`aua logcat mark [NAME]` (default `default`; also auto-marks `last-action` immediately "
        "BEFORE every state-changing action, so `--since last-action` covers what the app "
        "logged in RESPONSE to it — that is the `act → what did it do` loop), "
        "`aua logcat [--grep REGEX] [--since MARK|last-action|30s] "
        "[--tag TAG] [--lines N] [--json]` — dump since the mark (default: last-action, else "
        "30s). Windows are in DEVICE time (logcat lines are device-stamped and emulator "
        "clocks drift from the host by seconds); `mark` reports `clock`, `host_unix_ms` and "
        "the measured `skew_ms` so drift is visible rather than silently eating your window",
    ),
    (
        "suite",
        "`aua suite run PATH|-- [--continue] [--json]` — YAML AC checklist of `has` / "
        "`expect` / `wait_for` checks; exit 0 all pass, 8 any fail (stop on first fail "
        "unless `--continue`)",
    ),
]


# Don't / Do / Why — agent operating contract (full guide + skill).
AGENT_BEST_PRACTICES_PERCEPTION: list[tuple[str, str, str]] = [
    (
        "Drive with raw `adb` + screenshots + pixel taps",
        "Drive with `aua` (`analyze` / `--rid` / `tap-and-analyze` / "
        "`input-and-analyze` / `has` / `wait-and-analyze`)",
        "Pixels break on density/layout; ids and resource-ids do not. Discovery via screenshots "
        "is minutes; aua discovery is seconds.",
    ),
    (
        "Hand-roll `adb exec-out run-as` + DB/WAL copies + host sqlite + push-back",
        "Use `aua db list|schema|query|execute|backup|restore`",
        "AUA targets the leased device/package, preserves UI for normal queries, offers coherent "
        "snapshots when needed, returns JSON, and makes every confirmed mutation recoverable.",
    ),
    (
        "`adb -s <serial> install -r <apk>` to get the build on the device",
        "`aua install <apk> --launch` (or `aua emulator start --apk <apk> --launch`)",
        "It targets the leased device, skips the push when that version is already installed, "
        "verifies the package manager actually registered it (adb can print Success when it did "
        "not), says so when the target is a `-read-only` emulator that discards the install, and "
        "returns the launched screen — so boot/install/launch/observe is one call, not four.",
    ),
    (
        "Use raw `adb -s … shell` for package checks or diagnostics",
        "Use `aua app exists|status <package>` or the bounded read-only `aua shell <argv…>`",
        "AUA selects and leases the target, reports the exact serial, returns structured output, "
        "quotes every argv item before Android's remote shell parses it, caps each output stream "
        "at 256 KiB, and refuses unknown/mutating verbs so raw shell cannot bypass confirmations "
        "or cleanup.",
    ),
    (
        "`analyze` after every `tap`/`input` (or always `--no-observe` then re-analyze)",
        "Use the explicit `*-and-analyze` action and consume its compact `observation`; narrow it with "
        '`--observe-fields`, and add `--until "<predicate>"` when the next screen is slow',
        "Post-action observation already has fresh ids. Extra analyzes double round trips. "
        "Measured on a 5-scenario run: 37 taps became 73 `analyze` + 37 `wait` calls this way, "
        "roughly 60% of wall-clock spent on avoidable round trips.",
    ),
    (
        "Act on ids from your previous response without checking whether that screen is still up",
        "Read `caller.previous_screen_gone` (on `meta.caller` for `analyze`); when it is true, "
        "use the ids in *this* response",
        "Every response reports your own think time (`caller.gap_ms`, `ema_ms`) and whether the "
        "screen your last call handed you has since been replaced — computed from fingerprints "
        "already in hand, so it costs nothing. Measured on one 13-call session: 75% of the "
        "elapsed time was the caller generating, mean 12.4s, and a screen arrived *during* one "
        "of those gaps. `caller.wait_ceiling_ms` is the cap on any single wait "
        "(`wait_ceiling_mode` says whether it was measured or pinned); it is deliberately short "
        "because another call is cheaper than a blocked session, so a wait that returns "
        "`timeout` means \"not yet\", never \"not there\".",
    ),
    (
        "Re-tap when an action reports `nothing changed` / `stale_risk`",
        "Re-read (or pass `--until`), never re-tap",
        "The settle gives up after ~1.1s, so a slow screen reports `unchanged` for a tap that "
        "landed. Re-tapping means a second submit / second purchase — the one failure this "
        "contract exists to prevent.",
    ),
    (
        "Force `--source vision` / `--with-ocr` / `--no-ocr` on every screen",
        "Leave `analyze` defaults alone; escalate vision only when the tree misses content "
        "(Compose/canvas/game, or `meta.lossy_text`)",
        "Parallel OCR is cheap (~100ms) and prevents empty/broken reads. The map skips OCR "
        "automatically once a screen has enough hierarchy-only evidence — do not second-guess it.",
    ),
    (
        "Guess coordinates or scrape `uiautomator dump` yourself",
        "Act with integer `id` from analyze, or stable `--rid <tail>` / `has --rid`",
        "That is the whole point of aua — selectors, not geometry.",
    ),
    (
        "Hand-roll `clipboard set` + `paste` (or IME keys) for typing speed",
        'Just `aua input-and-analyze <id|--rid …> "…"` — prefers one-shot `set_text`, then clipboard paste '
        "(clipboard restored), then IME keys",
        "`input-and-analyze` owns the fast path and readback; agents should not reinvent it.",
    ),
    (
        "Use `key back` to dismiss the keyboard",
        "`aua hide-keyboard-and-analyze`",
        "`key back` often navigates away from the screen instead of only closing the IME.",
    ),
]

AGENT_BEST_PRACTICES_MEMORY: list[tuple[str, str, str]] = [
    (
        "Re-explore an app from scratch every session",
        "Analyze once; follow `# goto` / `# flows`; read `aua about` only when you need a "
        "recipe, environment setup, or app quirk",
        "The map is the previous agent's gift. Ignoring it re-pays discovery cost.",
    ),
    (
        "Tap through 5 screens to reach a known destination",
        '`aua goto "<goal>"`; then a matching `aua flow run`; then a verified '
        "`open-and-analyze`; manual actions last",
        "Verified routes outrank delivered intents; flows collapse whole journeys.",
    ),
    (
        "After a goto handoff / mid-path manual hop, re-walk from the start or invent the rest",
        '`aua goto "<goal>" --from-here`',
        "Skips remembered steps that already match the current screen (mid-edge resume). "
        "Plain `goto` starts from the map's current *screen*; `--from-here` is for mid-*edge*.",
    ),
    (
        "Keep discoveries only in the chat transcript",
        "`aua remember` / `aua knowledge add` (and fix bad names with `memory update`)",
        "The next agent (or you tomorrow) will not see this chat — write it into the playbook.",
    ),
    (
        "Skip past a `# aua asks:` / `meta.ask` line as someone else's chore",
        'Answer it on your next call: `--answers <task-id>="<real name>"`',
        "It is one question about the screen in front of you, and you are the only one who can "
        "answer it. 970 had piled up at ~130/day because no answer had any way to arrive; each "
        "one you answer is a screen `goto` / `map --find` can reach by name from then on.",
    ),
    (
        "Start timed work before the daemon is warm",
        "`aua daemon start` (optional: `aua-fast` for hot commands)",
        "Cold Python/connect startup dwarfs hierarchy dump cost on short commands.",
    ),
]

AGENT_BEST_PRACTICES_SPEED: list[tuple[str, str, str]] = [
    (
        "`tap --rid x --no-observe` then `analyze`",
        "`tap-and-analyze --rid x`",
        "A tap RETURNS the resulting screen by default. The two-call habit doubled a lane's "
        "round trips: 66 taps followed by 52 analyzes.",
    ),
    (
        "`sleep 8` after an action",
        "`wait --for <text|id> --observe`",
        "Returns the moment it appears, and hands you the screen. A sleep is slower when "
        "short and wrong when the screen is not ready.",
    ),
    (
        "Extra `wait --for-stable` / sleeps between hops of `goto` / `flow run`",
        "Just run `goto` / `flow` — route replay already settles on the next known selector "
        "(`has` on rid/label) before falling back to pixel settle",
        "Smarter settle is built in; agent-side waits only add turns.",
    ),
    (
        "`wait --for-stable` after tapping send",
        "`wait --after-change`",
        "Nothing has changed yet, so the screen is already 'stable': --for-stable returned in "
        "1.6s with NO answer on screen; --after-change returned in 4.9s WITH it. Measured.",
    ),
    (
        "`sleep 30` for image generation",
        "`wait --after-change` (or `--for-stable --settle 1500`)",
        "Same trap, bigger waste. Wait on the condition, never the clock.",
    ),
    (
        "One shell call per assertion",
        "Group independent checks in one call",
        "Each extra call is another agent turn, ~6s of wall clock.",
    ),
    (
        "Screenshot every step",
        "Screenshot what you will cite",
        "68 screenshots in one lane; most were never referenced.",
    ),
    (
        "Trust hierarchy text containing `?`",
        "Re-read with `analyze --source vision`",
        "U+FFFD means the glyph never reached you. `meta.lossy_text` now flags it. A formula "
        "answer read as 'solve for <?>: <?>' in hierarchy; OCR read '2x = 8' correctly.",
    ),
    (
        "`--parallel` to prepare an AVD you will install into",
        "`--parallel --no-read-only`",
        "--parallel implies -read-only, so an `adb install` lands in a discarded overlay and "
        "reports Success. The app is simply gone after stop.",
    ),
    (
        "Block the suite on an LLM/chat reply you do not assert on",
        "Assert the UI affordance you need (`has --rid submitButton` / favorite), or mock the API",
        "Model latency is app time, not tool time — waiting for a full answer inflates every run.",
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
        "session already knowing the map. For debuggable builds, `aua db` also provides "
        "structured private-SQLite inspection, guarded mutation, and rollback without "
        "agent-authored adb pipelines."
    )
    p.append("")
    p.append("## Session protocol")
    protocol = BRIEF_SESSION_PROTOCOL if brief else SESSION_PROTOCOL
    for i, (title, body) in enumerate(protocol, 1):
        p.append(f"{i}. **{title}.** {body}")

    if brief:
        p.append("")
        p.append("## Agent best practices (short)")
        p.append(
            "**Do:** prefer stable `--rid`/`stable_key` on dynamic screens, use action "
            "`observation`, `wait --for` (never `sleep`), "
            "`about`/`goto`/`goto --from-here`/deeplinks/flows, leave OCR auto (map skips when "
            "sure), `aua input-and-analyze` (built-in fast typing + readback), use `aua db` "
            "for private SQLite state. "
            "**Don't:** raw `adb`+screenshots+pixels, `analyze` after every tap, force "
            "`--with-ocr`/`--no-ocr`/vision on every screen, re-walk a goto after a mid-path hop, "
            "reuse a numeric id after its frame changed, sleep between goto hops, `key back` to "
            "dismiss IME, hand-rolled clipboard paste. "
            "Full tables: `aua guide` (no `--brief`)."
        )
        p.append("")
        p.append("## Escalation (automatic)")
        p.append(
            "Perception climbs only as far as the question needs: "
            + " → ".join(t for t, _, _ in ESCALATION_LADDER)
            + ". Paid grounding (T4) is **never** entered automatically — pass `--deep`."
        )
        p.append("")
        p.append("## Optional local policy")
        p.append(
            "Off by default; never run `policy_suggestion`. In `session autopilot`, an "
            "authenticated adapter chooses IDs; the daemon executes guard-approved taps "
            "and re-observes. It returns `policy_handoff` on uncertainty. Zero/one "
            "bypass; bundled v10 serves two/three/four, advisory-capable, off by default. "
            "`aua policy status` is host-only."
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
    p.append("aua --format tsv analyze         # READ the screen: one element per line, no noise")
    p.append(
        "aua --format compact analyze     # same screen as JSON, when you need it machine-readable"
    )
    p.append('aua ask "describe this screen top-to-bottom"  # screenshot + element graph via VLM')
    p.append("aua tap-and-analyze 4            # act by id and receive the resulting screen")
    p.append('aua input-and-analyze 2 "hello@example.com"  # type and receive the resulting screen')
    p.append("aua --format tsv analyze         # only if you need a narrower/fresher filtered view")
    p.append("```")
    p.append(
        'Cheap presence check to branch on: `aua has "Sign in"` (exit 0 found / 1 not). '
        '`aua wait-and-analyze --for "Welcome"` polls until present; `aua wait-and-analyze --for-stable` returns once '
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
        "`meta.map_hint` / `meta.research_tasks`), ranked by your recent navigation so the screens "
        "you use most surface "
        'first; `aua goto "<goal>"` drives a remembered route in one call. **Cross-app auth '
        "legs (Google sign-in via Chrome/GMS, permission dialogs) fold into the origin app's "
        "route** and replay step by step — a redacted account row hands off for one manual tap, "
        "then re-running `goto` resumes. Before its first route step, replay refuses deeplinks, "
        "cross-package actions, settings/data/environment mutation, lifecycle changes, and other "
        "non-navigation effects with a visible risk preview; `--allow-unsafe` is a deliberate "
        "opt-in after review. Destructive steps (delete/sign out/…) separately require "
        "`--allow-destructive`. An auto-recorded route is provisional until the same "
        "transition is observed twice (cross-package transit supplies independent corroboration), "
        "and selectorless routes are rejected "
        "from `goto`. Schema v4 scopes screens/routes to automatically discovered deterministic "
        "feature-flag contexts; exact-context routes outrank trusted `legacy-default` fallbacks. "
        "Stable resource namespaces produce locale-independent destination names, while logical "
        "destinations group flag variants and loading/error/empty/ready states instead of "
        "acquiring numeric suffixes. `aua map --audit` persists source/runtime research tasks. "
        "Store agent feedback with `aua "
        "knowledge add`; exchange corrections through `aua reconcile plan|submit`, where "
        "`verdict=apply` is validated, snapshotted, committed atomically, and rollbackable. "
        "Manage with `aua memory show|path|update|forget` "
        "(`memory update --screen <name>` renames a badly-auto-named screen so `goto <name>` "
        "reads naturally)."
    )

    p.append("")
    p.append("## Headless / unattended verify")
    p.append(
        "When you shipped a change and just need confidence it works — and the user should "
        "**not** see an emulator window pop up — prefer a headless AVD:\n"
        "```bash\n"
        "aua session start --goal 'verify the change'\n"
        "# AUA selects a compatible free device or boots and leases one automatically\n"
        "# … drive the flow under test …\n"
        "aua session finish                   # cleans session-owned state/emulator\n"
        "```\n"
        "**Parallel agents on one host** use the same call (selection is serialized):\n"
        "```bash\n"
        "aua session start --goal 'record HTTPS' --needs root,proxy\n"
        "# … AUA may boot a unique read-only instance when every match is leased …\n"
        "aua session finish\n"
        "```\n"
        "Sneak-peek all of them: `aua dashboard` (live device **grid** by default).\n"
        "Headless on Mac uses **host GPU** (Metal). **Always stop AVDs you started** before "
        "ending — idle `--idle-stop` (default 900s) and MCP exit cleanup are backups only.\n"
        "For **proxy / mock HTTPS** (apps that only trust system CAs), Play Store AVDs "
        "will not work — create a small rootable one:\n"
        "```bash\n"
        "aua emulator recommend-proxy         # package + why (no download)\n"
        "aua emulator ensure-proxy            # one-time image setup\n"
        "aua session start --goal 'record HTTPS' --needs root,proxy\n"
        "aua proxy start\n"
        "aua session finish                    # cleanup when done\n"
        "```\n"
        "Analyze/tap/wait work identically; hierarchy + screenshots do not need a visible "
        "window. Never wipe or stop an emulator the user already had open unless they asked."
    )

    p.append("")
    p.append("## On-device helper (optional, rootable targets)")
    p.append(
        "AUA normally drives a flow one step at a time from the host: a round trip and a "
        "settle wait per step, about 430ms each. On a **rootable** target you can install a "
        "small AccessibilityService APK and hand it the whole run instead, so the steps "
        "execute on the device and settle locally.\n"
        "```bash\n"
        "aua helper status                    # installed / enabled / bound, and its version\n"
        "aua helper enable                    # install + switch on (needs `adb root`)\n"
        "aua helper tree                      # read the hierarchy without an adb dump\n"
        "aua helper watch --timeout-ms 5000   # stream screen-change events (with their text)\n"
        "aua helper remove                    # switch off and uninstall\n"
        "```\n"
        "**Off by default** (`helper.enabled: false`) — and that one switch is all there is: "
        "turn it on and AUA checks the target can run it, pushes the APK, enables the service "
        "and confirms it bound, once per device. A target that cannot root is noted and left "
        "alone rather than re-probed every run. Once enabled, `flow run` / `goto` hand "
        "the *leading stretch* of UI-only steps to the device and run the rest normally; "
        "anything it cannot do — proxy, network shaping, feature flags, launching apps, a "
        "nested flow — stops the handover and the host continues from that step. It runs "
        "`tap`, `long-press`, `input`, `clear`, `key` (back/home/recents), `swipe`, `scroll`, "
        "`scroll-to`, `tap-point`, `paste`, `wait-for`, `wait-stable` and the asserts; "
        "`hide-keyboard` deliberately stays on the host, because accessibility cannot send "
        "KEYCODE_ESCAPE and Back would finish the Activity. An absent, "
        "unbindable or failing helper changes nothing except the speed, so a run gives the "
        "same result either way.\n"
        "Measured end-to-end on an emulator: **2 steps 5.9s → 4.5s, 4 steps 9.7s → 5.3s**. "
        "There is a fixed handover cost, because Android suppresses accessibility services "
        "while uiautomator2 holds UiAutomation — but AUA resolves the target without "
        "connecting, so the helper is usually already bound and that cost is ~0.7s rather "
        "than ~2.8s. It engages from `helper.min_flow_steps` (default 2): one step cannot "
        "repay the handover, two can.\n"
        "Needs `adb root`, so a **Google APIs** AVD, not a Play Store image or a retail phone. "
        "On a target that cannot root, enable *AUA Helper* by hand once under "
        "Settings > Accessibility, or simply leave it off."
    )

    p.append("")
    p.append("## Emulator microphone input")
    p.append(
        "Install the optional transport (`pip install 'android-ui-analyser[audio]'`) and boot "
        "the selected AVD with audio enabled. AUA matches its runtime discovery record by adb "
        "serial and keeps the emulator bearer token out of output. Physical devices are refused "
        "before streaming.\n"
        "```bash\n"
        "aua emulator start --headless --audio\n"
        "aua mic inject sample.wav\n"
        "aua mic inject sample.wav --rid hold_to_talk --pre-roll-ms 300 --post-roll-ms 500\n"
        "aua mic inject sample.wav --rid record_button --control-mode toggle\n"
        'aua mic speak "Testing one two" --voice Samantha --rate 175 --rid hold_to_talk\n'
        "```\n"
        "WAV input is uncompressed unsigned 8-bit or little-endian signed 16-bit PCM, mono or "
        "stereo, at 48 kHz or less, up to five minutes. `mic speak` is macOS-only; elsewhere synthesize a compatible "
        "WAV and use `mic inject`. The stream uses emulator backpressure and waits for its close. "
        "A target defaults to push-to-talk `hold` (DOWN/pre/audio/post/UP). `--control-mode "
        "toggle` uses exactly one non-retrying tap to START and one to STOP at the same point; "
        "it requires an enabled, clickable, initially-off control. When the app does not expose "
        "checked/selected state, the caller must establish that initial-off precondition. AUA "
        "refuses audio or STOP if the foreground package no longer owns the screen. Toggle is "
        "best-effort unless the app exposes an active-state/STOP selector: use short media and "
        "require the control to remain actively recording through post-roll, because an app that "
        "auto-stops early could interpret AUA's final tap as a new START. An "
        "`INTERNAL` close becomes "
        "`mic_delivery_uncertain` with the post-action screen in `error.result.observation`: "
        "samples may already have arrived, so never repeat the voice action. Timeouts and "
        "unclassified RPC closes can also follow partial delivery, so inspect the UI before a "
        "new attempt. `mic_delivered_release_failed` means audio arrived but target-control "
        "cleanup failed; do not repeat it. `mic_toggle_start_uncertain` sends no audio and no "
        "blind STOP, while `mic_toggle_stop_uncertain` means START was confirmed but STOP is "
        "unknown; recording may be active in either case, so protect privacy, inspect the forced "
        "observation, and never tap/retry blindly. If "
        "`mic_emulator_unavailable` reports that the emulator exited or went offline, inspect "
        "`aua devices` and restart only that emulator with `--audio` when necessary. Android "
        "Emulator 36.4.10 is limited to one injection attempt per boot across all AUA workers; "
        "`mic_repeat_unsafe` means restart only that emulator before one new attempt."
    )

    p.append("")
    p.append("## Agent best practices (do / don't)")
    p.append("")
    p.append(
        "Other agents: treat this as the operating contract. Most wall-clock on a real run is "
        "**your** round trips, sleeps, and pixel guessing — not aua. Follow the **Do** column; "
        "the **Don't** column is how agents usually make aua look slow."
    )
    p.append("")
    p.append("### Perception & action")
    p.append(_md_table(["Don't", "Do", "Why"], AGENT_BEST_PRACTICES_PERCEPTION))
    p.append("")
    p.append("### Memory, map, and shortcuts")
    p.append(_md_table(["Don't", "Do", "Why"], AGENT_BEST_PRACTICES_MEMORY))
    p.append("")
    p.append("### Waiting & speed (measured)")
    p.append(
        "Measured on a real 6-scenario lane: 1348s wall clock, 239 aua calls, and only ~33s "
        "(2.5%) inside aua. Agent turns were ~48%; blind `sleep` burned 251s (19%). Optimise "
        "round trips, not aua internals."
    )
    p.append("")
    p.append(_md_table(["Don't", "Do", "Why"], AGENT_BEST_PRACTICES_SPEED))
    p.append("")
    p.append("## Worked examples")
    p.append("```bash")
    p.append("# No device attached? Boot headless so you don't bother the user:")
    p.append("aua emulator start --headless       # or: --avd pixel7")
    p.append("")
    p.append("# Optional: warm daemon so every later call is ~tens of ms.")
    p.append("aua daemon start --quiet            # `aua orient` prints the app playbook on demand")
    p.append("")
    p.append("# See the screen. When the app is mapped the response already carries")
    p.append("# meta.known_screen + meta.known_routes + meta.suggested_gotos — act on those.")
    p.append("aua --format tsv analyze")
    p.append("")
    p.append("# Just the header, just the tappable things (no JSON post-processing):")
    p.append("aua --format tsv analyze --region 0,0,1080,300 --clickable --fields id,desc,rid")
    p.append("")
    p.append("# Is that switch on? Read the boolean instead of looking at a screenshot:")
    p.append("aua --format tsv analyze --where-rid settingsSwitch --fields id,checkable,checked")
    p.append("")
    p.append("# Must you actually SEE something? Crop it — a full 1080x2400 PNG is expensive:")
    p.append("aua screenshot --region 0,0,1080,300 --out /tmp/header.png   # then read that file")
    p.append("")
    p.append("# Jump straight to a remembered screen (drives + verifies each hop,")
    p.append("# including cross-app auth legs):")
    p.append('aua goto "product detail"')
    p.append('aua goto "settings" --plan          # just print the route, take no action')
    p.append('aua goto "settings" --from-here     # resume mid-edge after a manual step')
    p.append("")
    p.append("# Replay a whole journey (authored or recorded) in ONE call:")
    p.append('aua flow run reset_account_google_login --param ACCOUNT="Engineering Team"')
    p.append("aua flow run smoke --artifacts-dir artifacts/smoke --evidence failures --junit")
    p.append("aua flow save reach_checkout --last 8         # preview only; writes nothing")
    p.append(
        "aua flow save reach_checkout --last 8 --save  # commit after reviewing proof/warnings"
    )
    p.append("")
    p.append("# Starter journey: open → tap → input → tap → wait → has → tap. Every action")
    p.append("# returns the post-action screen, so each id below comes from the previous call:")
    p.append('aua open-and-analyze "myapp://catalog"    # response carries observation + fresh ids')
    p.append("aua tap-and-analyze 24                    # id from the open response")
    p.append('aua input-and-analyze 25 "wireless"       # id from the tap response')
    p.append("aua tap-and-analyze 26                    # send-button id from that same response")
    p.append('aua wait-and-analyze --for "Results"       # confirm and receive the settled screen')
    p.append(
        "aua has --rid resultsPanel && echo present  # cheap branch, exit 0 present / 1 absent"
    )
    p.append("aua tap-and-analyze 31                    # continue on another read-back id")
    p.append("")
    p.append("# A wait longer than one agent call: start once, reconnect by id, never restart it:")
    p.append("aua job start await --predicate 'rid:answer,!text:Loading' --timeout-ms 180000")
    p.append("aua job status <job_id> --recent-output # or: job wait / job cancel")
    p.append("```")
    p.append("")
    p.append("An action response carries its own state, so a follow-up `analyze` is usually")
    p.append("unnecessary — reach for one when you need a *different* view (another region, OCR,")
    p.append("or a filtered projection), or when content was still streaming in on the first read.")
    p.append("```json")
    p.append("{")
    p.append('  "ok": true,')
    p.append('  "action": "tap",')
    p.append('  "observation_present": true,')
    p.append('  "known_screen": "chat",')
    p.append(
        '  "stable_elements": [{"id": 25, "stable_key": "compose_input"}, {"id": 26, "stable_key": "send"}],'
    )
    p.append(
        '  "action_diff_summary": {"added": 0, "removed": 0, "changed": 2, "prev_count": 17, "curr_count": 17},'
    )
    p.append('  "note": "No separate analyze needed; state is in observation."')
    p.append("}")
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
    p.append('                  "checkable", "checked", "selected",      // tri-state:')
    p.append('                  "scrollable", "long_clickable", "password",  // null = unknown')
    p.append('                  "source": "hierarchy|detection|ocr|grounding", "confidence" } ],')
    p.append('  "meta":     { "duration_ms", "tier_used", "path", "providers_used",')
    p.append('                "known_screen", "known_routes", "suggested_gotos", "research_tasks",')
    p.append(
        '                "flows": ["name(PARAM)"],          // aua flow run name --param PARAM=v'
    )
    p.append(
        '                "ask": {"id","about","q","how"},   // answer with --answers id="<name>"'
    )
    p.append('                "map_hint",')
    p.append('                "annotated_image", "raw_image", "device_serial" } }')
    p.append("```")
    p.append(
        "`compact` drops null/default fields for the smallest token footprint — except "
        "`checked` on a `checkable` node, where *off* is the answer you asked for."
    )
    p.append(
        "Don't post-process this by hand. `--format tsv` plus `--fields`/`--where-*`/`--region`/"
        "`--limit` gives you exactly the rows and columns you want in the same call (see the "
        "flag table above); `--all` turns tsv's implicit noise filtering off."
    )
    p.append("")
    p.append(
        "Action command responses always include a small contract wrapper so `analyze` is usually not needed:"
    )
    p.append("```json")
    p.append('{"ok": true,')
    p.append('  "action": "tap",')
    p.append('  "observation_present": true,')
    p.append('  "known_screen": "chat",')
    p.append('  "stable_elements": [')
    p.append('    {"id": 25, "stable_key": "compose_input"},')
    p.append('    {"id": 26, "stable_key": "send"}')
    p.append("  ],")
    p.append(
        '  "action_diff_summary": {"added": 0, "removed": 0, "changed": 2, "prev_count": 17, "curr_count": 17},'
    )
    p.append('  "note": "No separate analyze needed; state is in observation.",')
    p.append('  "observation": { "screen": {...}, "elements": [...], "meta": {...} }')
    p.append("}")
    p.append("```")
    p.append(
        "If `observation_present` is false, the action did not request a post-action read "
        "(`--no-observe` or unsupported action), so run `analyze` explicitly."
    )
    p.append(
        "Need the actual pixels too? `--with-image [path]` on `analyze` AND on every "
        "action (tap/input/swipe/scroll-to/key/open) saves the raw screenshot to a "
        "timestamped file and returns its path in `meta.raw_image` (on actions: inside "
        "`observation.meta`) — Read that file when you must SEE the screen (visual "
        "fidelity, images, charts) instead of just addressing it. Over MCP the image "
        "comes back inline as an image content block. **Default off.** Do not pass "
        "`--with-image` on every step — hierarchy/TSV is faster and cheaper; images erase "
        "the token advantage of acting by id."
    )

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
    p.append(
        "`aua ask` is provider-neutral: configure `grounding.chain: [gemini, openai]`. The factory "
        "tries that order and skips providers whose API-key env var is absent, so one config works "
        "with either key. Reverse the list to prefer OpenAI when both exist. On macOS, Apple "
        "Vision OCR and hierarchy capture fuse into one observation (parallel when OCR is forced; "
        "auto mode consults the map first). Screens the map has seen hierarchy-only enough times "
        "(and never needed OCR) skip OCR on later visits — cheaper analyze without risking "
        "unknown screens. Web content inside a Custom Tab stays visible to a plain `analyze`. "
        "Readings that only repeat text the tree already reports are withheld "
        "(`ocr.drop_redundant`); pixel-only text always survives. OCR works on a 720px preview "
        "and maps boxes back to original screen coordinates. Route replay settles on the next "
        "step's known selector when possible instead of a full pixel `wait_stable`."
    )
    p.append("")
    p.append("## Optional guarded FunctionGemma policy (off by default)")
    p.append(
        "Deterministic AUA code authors complete current-frame stable-selector "
        "tap calls, removes unsafe/unauthorized/destructive/stale/ambiguous/redundant choices, and "
        "gives the model only privacy-screened metadata plus opaque IDs. `shadow` exposes audit "
        "metadata without a call. `advisory` may expose a separate `policy_suggestion`, which never "
        "changes or executes `recommended_call`. The explicit `aua session autopilot` command is "
        "the only local execution lane: an authenticated advisory-capable selector chooses an "
        "opaque ID, then the warm AUA daemon revalidates and executes the corresponding trusted "
        "tap itself. It re-observes after every action and stops on stale/unknown outcomes, no "
        "progress, repeated calls, time/step limits, or work outside safe navigation taps. Zero "
        "candidates return a non-executing "
        "`policy_handoff`; reserved ID -1 is accepted from the model only when its authenticated "
        "manifest binds that protocol. Zero/one candidates bypass the model; the bundled v10 "
        "adapter runs for two, three, or four, the cardinalities its manifest authenticates. That "
        "manifest authenticates advisory, so the lane is reachable with the bundled adapter alone. "
        "It stays inert by configuration, not by a ceiling in the artifact: `policy.enabled` is "
        "false and `policy.mode` is off by default, so nothing loads until an operator turns it on."
    )
    p.append(
        "To enable it: from a checkout run `./install.sh --with-policy` to install the "
        "Apple-silicon runtime into the same environment as `aua`, then manually obtain "
        "the pinned external MLX base, then set `policy.enabled: true`, `policy.mode: advisory`, and "
        "`models.functiongemma.model_path` to its absolute directory. Leave "
        "`models.functiongemma.adapter_path` null to use the packaged v10 LoRA. AUA bundles only "
        "that separately licensed ~29 MB adapter; it does not bundle or automatically download the "
        "~543 MiB base, and the pinned revision is enforced by hash. `docs/LOCAL_POLICY_SETUP.md` "
        "is the full recipe, including the optional two-tier chain with a larger local reviewer. "
        "Run `aua policy status` for host-only dependency, artifact/hash, and daemon readiness "
        "without loading the model or touching Android. "
        "Measured by an independently authored probe — scenarios derived from the CLI surface, not "
        "from the training generators — v10 scores 0.600 at its best checkpoint and 0.471 mean over "
        "16, with refusal 18/38 and zero invalid outputs; on a device it made 5/5 navigations and "
        "2/2 refusals with zero wrong taps. It is NOT promoted: one seed, no live gate, and refusal "
        "swinging between 0 and 18 across checkpoints. Keep the deterministic guard, treat model "
        "refusal as unreliable, and do not read these numbers as autonomy or speed evidence. "
        "That separation matters: an in-house probe sharing its generator's phrasing reported 6/6 "
        "on a refusal capability independent measurement put at 0/144. Earlier generations show the "
        "same lesson — v3 scored 99.8535% synthetic then 62.5% on a production-serializer matrix "
        "(37.5-point target-ID and 54.17-point target-position gaps), and a failure-driven v4 "
        "reached 2,767/2,768 validation yet failed its independent gate on four unauthorized "
        "`session_finish`-over-`analyze_screen` choices and was never bundled. See the README and "
        "`experiments/functiongemma/` for licensing, reproduction, and evaluation."
    )
    return "\n".join(p) + "\n"


def render_brief() -> str:
    return render_markdown(brief=True)


def render_skill_markdown() -> str:
    """Compact triggered instructions; deeper guidance stays in the CLI manual."""
    return """# Android UI Analyser

Use `aua` for Android UI. Act on returned IDs or stable selectors, never pixels;
do not substitute raw `adb`.

## Operating loop

1. Start with `aua session start --goal "<what must be verified>"`; it selects/provisions a
   compatible target and leases it to the caller process. Add `--needs root,play,proxy` and
   `--app <package>` for an unrelated foreground app. Reuse its observation and follow its
   exact `recommended_call`; do not immediately re-analyze. `--contract` requires fresh
   proof and strict finish. `--artifacts-dir` records evidence; `--wait-for-lease` waits safely.
2. Prefer navigation in this order: verified `goto`, matching saved `flow`, proven deeplink,
   then a manual analyzed action. Preview risky routes; goal text never authorizes destructive,
   external, settings, data, payment, send, or sign-out effects.
3. Use analyzed actions and consume their returned `observation`; integer ids belong only to
   that frame. On dynamic screens prefer `--rid` or `stable_key`, resolving it again after a
   transition instead of replaying an old numeric id.
4. Fold arrival into the action with a positive predicate such as
   `--until 'rid:resultCard,!text:Loading'`. On `settled-unmet`, use its fresh destination and
   corrected predicate; never repeat the action. Use `await-and-analyze` for absence-only checks
   and `back-until-and-analyze` for nested return navigation.
5. Keep perception hierarchy-first. Filter in AUA (`--where-rid`, `--where-text`, `--clickable`,
   `--region`); use vision for opaque screens and `--deep` for grounding.
6. Carry `goal_progress.checkpoint` on the next call with `--phase-done` (MCP: `phase_done`)
   instead of spending a separate progress call. Use `aua job start await ...` only for a
   read-only wait that may outlive one agent call. If `daemon_outcome_unknown` appears, never
   repeat the action; wait, then inspect one fresh screen.
7. End with the returned cleanup call, normally `aua session finish`. Use `review.accounting`, not estimates:
   `top_level_calls` counts caller-visible invocations and equals `lifecycle_calls` + `task_calls`;
   `journal_events` adds `folded_internal_events` such as an action-bound wait. The snapshot excludes
   this review/finish (`reporting_call_included` is false); `top_level_calls_including_reporting_call` adds it.
8. After a contract passes, `session candidate-flow NAME --save` requires explicit
   `--reset-flow` and passing reset/replay.

Flow previews expose value-free `selector_resilience`. Trust an unmapped arrival only when its
source is `satisfied_action_until` from the preceding action's privacy-safe positive `--until`
on the same package/context/frame.

## Device and safety rules

- First call `session start`; never list/start devices, set `AUA_OWNER`, or acquire a lease.
  It scans leased targets, frees dead owners, and selects/provisions a capable match. One device
  stays implicit: omit `--serial`; switching or transfer is explicit.
- Use `--no-start-emulator` only when provisioning is forbidden; use `--headed` only when
  visibility is required. For voice input add `--audio`, then use `mic inject` or macOS `mic speak`;
  never repeat late-delivery or uncertain-toggle errors.
- Use `aua network offline --verify`; session cleanup restores it. Use guarded `aua db` for
  debuggable SQLite.
- A delivered deeplink, spinner disappearance, or unchanged short settle is not proof of the
  requested destination — check `verified`, not just `ok`. Verify the final interactive
  affordance the user named.
- Never execute `policy_suggestion`; `session autopilot` is off by default and **taps only** —
  never start it on a login or text entry. Short goal in the screen's own words (`Open Catalog`):
  a candidate sharing no goal word is refused. `policy_handoff` hands back.

## Load more only when needed

- Run `aua guide --brief` for the selector, wait, navigation, map, flow, lease, and recovery
  field guide.
- Run `aua capabilities --goal "<goal>"` for structured discovery without reading the manual.
- Run `aua guide` for command/flag tables, databases, proxy/mock, capture, maps,
  flow authoring, `aua helper`, troubleshooting, schema, and exit-code reference.
"""


def render_json() -> dict[str, object]:
    """Structured form of the manual for programmatic consumers."""
    from .capabilities import capability_manifest

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
            "Schema-v4 per-app map auto-recorded locally under memory.dir; runtime feature-flag "
            "contexts, semantic names/states, verified routes, provenance knowledge, and "
            "audit/reconciliation with rollback; read via `aua map`; meta.known_screen + inline "
            "meta.known_routes/suggested_gotos/map_hint/research_tasks on revisit "
            '(ranked by recent navigation); `aua goto "<goal>"` drives a remembered route; '
            "durable skeleton only; values/secrets redacted."
        ),
        "policy": (
            "OFF by default (policy.enabled false, mode off): nothing loads or consumes memory "
            "until an operator turns it on, and it is fine to offer. Bundled FunctionGemma v10 "
            "authenticates advisory, so no external adapter is needed — enabling it takes the "
            "Apple-silicon extra, the pinned external base (~543 MiB, never auto-downloaded, Gemma "
            "terms), and two config keys; recipe in docs/LOCAL_POLICY_SETUP.md, verify with `aua "
            "policy status`. Shadow is for local debugging only. Independently probed at 0.600 best "
            "checkpoint (0.471 mean) with refusal 18/38, and 5/5 device navigations with zero wrong "
            "taps, but NOT promoted: one seed, no live gate, refusal swinging 0-18. Expect frequent "
            "policy_handoff, never treat model refusal as a safety mechanism, and rely on the "
            "deterministic guard. A policy_suggestion is never executed manually or substituted for "
            "recommended_call. Explicit session_autopilot is the only execution lane: the daemon "
            "revalidates, executes, re-observes, and hands off at the first unsafe step. V10 serves "
            "two/three/four guard-approved opaque candidates; zero/one bypass the model."
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
                "checkable",
                "checked",
                "selected",
                "scrollable",
                "long_clickable",
                "password",
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
                "suggested_deeplinks",
                "research_tasks",
                "flows",
                "ask",
                "map_hint",
                "annotated_image",
                "raw_image",
                "device_serial",
            ],
        },
        "element_views": {
            "formats": ["json", "pretty", "compact", "tsv", "delta", "msgpack"],
            "field_aliases": sorted(FIELD_ALIASES),
            "tsv_default_fields": list(TSV_DEFAULT_FIELDS),
            "filters": [
                "--nonempty",
                "--no-system",
                "--no-wrappers",
                "--all",
                "--where-text",
                "--where-rid",
                "--clickable",
                "--region",
                "--limit",
                "--meta",
                "--no-meta",
            ],
        },
        "exit_codes": [{"code": c, "meaning": d} for c, d in EXIT_CODES],
        "key_flags": [{"scope": s, "flags": f} for s, f in KEY_FLAGS],
        "agent_best_practices": {
            "perception": [
                {"dont": a, "do": b, "why": c} for a, b, c in AGENT_BEST_PRACTICES_PERCEPTION
            ],
            "memory": [{"dont": a, "do": b, "why": c} for a, b, c in AGENT_BEST_PRACTICES_MEMORY],
            "speed": [{"dont": a, "do": b, "why": c} for a, b, c in AGENT_BEST_PRACTICES_SPEED],
        },
        "capabilities": capability_manifest(),
    }


# --------------------------------------------------------------------------- skill emit


def render_skill() -> str:
    """The compact generated SKILL.md; deeper manual layers remain available via ``aua``."""
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
    return "\n".join(front) + render_skill_markdown()


def emit_skill(path: str | Path | None = None) -> Path:
    """Write the generated SKILL.md to *path* (default `.claude/skills/.../SKILL.md`)."""
    target = Path(path) if path else DEFAULT_SKILL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_skill(), encoding="utf-8")
    return target


def render_codex_agent_metadata() -> str:
    """Deterministic Codex UI metadata shipped beside the same canonical skill body."""
    return (
        "interface:\n"
        '  display_name: "Android UI Analyser"\n'
        '  short_description: "Drive and verify Android apps with semantic UI evidence"\n'
        '  default_prompt: "Use $android-ui-analyser to verify the requested Android behavior efficiently."\n'
    )


def emit_skill_bundle(directory: str | Path) -> Path:
    """Install one generated skill body plus Codex metadata into a skill directory."""
    root = Path(directory)
    emit_skill(root / "SKILL.md")
    agents = root / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "openai.yaml").write_text(render_codex_agent_metadata(), encoding="utf-8")
    return root
