# PRD — `android-ui-analyser`

**A fast, configurable CLI that gives an AI agent structured "what's on screen and
where" for Android UI testing — hierarchy-first, with pluggable vision/OCR fallbacks
and selectable models (local or commercial).**

- Status: ready to build
- Background: see `docs/RESEARCH.md` (landscape) and `docs/ARCHITECTURE.md` (decisions)
- Working directory: the repository root

---

## 0. How this PRD is meant to be executed

This is a **single, complete deliverable**, not a phased rollout. It is written to be
handed to a long-running autonomous Claude Code session (e.g. `/goal`) that builds the
**entire** tool in one pass — it may run for hours. Do **not** stop at an MVP and ask
what to do next. Build everything specified here, wire up the tests, self-verify
against the acceptance criteria in §13, and only then report done.

The §16 task list is an internal build order for the run, **not** a set of separately
shippable phases. There is one milestone: the whole tool, working and tested.

> **Environment note:** the build machine may have **no Android device attached**.
> The build must therefore be fully completable and testable **without a device**,
> using unit tests with fixtures (sample hierarchy XML, mocked `uiautomator2`,
> stubbed providers). A device-dependent **smoke test** is documented for the human
> to run later (§13.2). Lack of a device is never a reason to leave the build
> incomplete.

---

## 1. Summary

`android-ui-analyser` (CLI name: **`aua`**) is a Python tool that reports the UI
elements on a connected Android device — each with a stable integer ID, type, text,
and bounding box — so an AI agent can drive UI tests by acting on IDs rather than
pixels. It reads the accessibility/view hierarchy first (fast, exact), and falls back
to image-based detection + OCR (and optionally a grounding VLM) on screens the
hierarchy cannot see. Every perception backend is pluggable and configurable, with
ordered fallback chains and a choice of local or commercial models.

It ships primarily as a **CLI** (driven by Claude Code over bash) with an **optional
MCP server** wrapper exposing the same capabilities. The perception logic lives in an
interface-agnostic **engine library** shared by both.

## 2. Goals

- **G1 — Speed.** Hierarchy `analyze` < 150 ms warm; local vision fallback < 600 ms.
- **G2 — Structured output.** Deterministic Set-of-Marks JSON the agent can act on.
- **G3 — Coverage.** Handle Compose/Flutter/WebView/canvas/game screens via the
  vision fallback when the hierarchy is empty.
- **G4 — Configurable.** Config file + env + flags; choose perception backends, OCR
  engines, detectors, and grounding models, with ordered fallbacks.
- **G5 — Pluggable models, local or commercial.** Built-in providers plus the ability
  to use a commercial multimodal model via an API key/setup.
- **G6 — Usable by hand and by agents.** Clean CLI ergonomics; optional MCP wrapper.
- **G7 — Shippable licensing.** Default configuration uses only commercially-usable
  components; AGPL/research-only options are opt-in with clear warnings.

## 3. Non-goals

- Not a test runner or assertion DSL (it is the *perception + action* layer; the
  agent or an external harness orchestrates tests).
- Not iOS (Android-only; architecture must not preclude a later iOS backend).
- Not a fork of Maestro/mobile-mcp/droidrun (we depend on `uiautomator2`, not fork).
- No bundled base or full-model weights. A small reviewed adapter may be bundled with its
  license, provenance manifest, and an explicit external-base setup contract.

## 4. Primary users & workflow

- **Primary:** an AI coding agent (Claude Code) testing an Android app.
- **Secondary:** a developer running `aua` directly to inspect a screen.

Typical loop:
1. `aua analyze` → JSON list of elements with IDs + boxes.
2. Agent decides an action → `aua tap-and-analyze 4` /
   `aua input-and-analyze 2 "hello"` / `aua swipe-and-analyze up`.
3. Repeat. `analyze` result is cached until a state-changing action invalidates it.

---

## 5. CLI specification (`aua`)

Global options (apply to all commands; override config):
- `--serial <id>` explicit target override. Each normal owner gets one automatic sticky lease, so
  ordinary commands omit it. A different target first refuses with `lease_switch_required`; the
  explicit `lease acquire <id> --replace` path cleans/releases the old device. `lease transfer`
  plus one-time `lease accept` delegates the same running device without teardown. Serial remains
  appropriate for initial selection, fanout, and targeted administration. Process death normally
  frees a lease immediately; an explicitly pending transfer is the sole bounded exception and
  reserves the target only until its five-minute token expires.
- `--config <path>` explicit config file
- `--format json|pretty|compact` output format (default `json`)
- `--profile <name>` named config profile (e.g. `local`, `cloud`)
- `--timeout <ms>` per-operation timeout
- `--log-level error|warn|info|debug` (default `warn`; logs go to **stderr**, JSON to **stdout**)
- `--no-cache` bypass the cached analyze result
- `--version`, `--help`

### Perception
- `aua analyze` → emit Set-of-Marks JSON (§8) for the current screen.
  - `--source auto|hierarchy|vision` force the perception path (default `auto` = gate-driven)
  - `--with-ocr / --no-ocr` include OCR text boxes in vision results
  - `--annotate [path]` also write an annotated screenshot (numbered boxes); default path under run dir
  - `--query "<instruction>"` return the single best-matching element ID. Resolves
    cheaply first — match the instruction against the hierarchy — and only escalates to
    the grounding provider if there's no confident match (see §6a). `--deep` forces
    escalation; `--cheap` forbids it.
- `aua screenshot [path]` → save a raw screenshot (PNG). `--annotate` to overlay marks.
- `aua inspect <id>` → print full attributes for one element from the last analyze.

### Quick checks (lightweight — do NOT run the full analyze pipeline)
- `aua has "<text>"` → is this text on screen right now? Exit `0` if present, `1` if
  not — ideal for an agent to branch on, and far cheaper than `analyze` (it returns an
  exit code, not a JSON blob). Prints a one-line result:
  `{"found": true, "source": "hierarchy", "bounds": [x1,y1,x2,y2]}`.
  - `--match exact|contains|regex` (default `contains`)
  - `--ignore-case`
  - `--ocr-fallback / --no-ocr-fallback` (default on): query the hierarchy first; only
    on a miss, OCR the screenshot and substring/regex-search the recognized text — this
    is where the fast macOS `apple_vision` OCR earns its keep. `--source
    hierarchy|vision|auto` forces the path.
  - `--timeout <ms>` poll until present or timeout (`0` = single instant check, default)

### Actions (all return the resulting screen; IDs come from the last `analyze`)
- `aua tap-and-analyze <id>` (alias: `click-and-analyze`)
- `aua long-press-and-analyze <id> [--ms 600]`
- `aua input-and-analyze <id> "<text>"` (focuses element, types; `--submit` sends IME action)
- `aua clear-and-analyze <id>`
- `aua swipe-and-analyze <up|down|left|right> [--from <id>] [--percent 50]` or
  `aua swipe-and-analyze --coords x1 y1 x2 y2`
- `aua scroll-to-and-analyze "<text|resource-id>"` (scroll until element appears or limit)
- `aua key-and-analyze <back|home|enter|recents|KEYCODE_*>`
- `aua wait-and-analyze --for "<text|resource-id>" [--timeout 5000]` /
  `aua wait-and-analyze --idle` /
  `aua wait-and-analyze --for-stable [--interval 200] [--settle 600] [--timeout 30000]`
  - `--for-stable` polls cheap screenshots and returns once the screen stops changing for `--settle` ms (a perceptual-hash "screen settled" check — no OCR, no hierarchy parse; works on opaque screens). Ideal for waiting on image generation / loading. Pairs with the daemon for tight, low-cost polling.

### Device & session
- `aua devices` → list attached devices (serial, model, android version, state)
- `aua app <foreground|launch <pkg>|stop <pkg>|current>`
- `aua install <app.apk> [--launch] [--reinstall|--fresh --yes]` → put a build on the device
  without a hand-rolled `adb install`; idempotent when that version is already installed.
  `aua emulator start --apk <app.apk> --launch` and `aua session start --apk <app.apk>` fold
  boot + install + launch + observe into a single call.
- `aua daemon <start|stop|status>` (§10)

### Private app databases (debuggable builds)
- `aua db list <pkg>` → list database primaries and WAL/SHM/journal sizes through
  `run-as`; no on-device `sqlite3` dependency.
- `aua db schema <pkg> <db> [--table NAME]` and `aua db query <pkg> <db> <SQL>
  [--params JSON] [--limit N]` → stop the app, copy a coherent main+sidecar snapshot,
  query with host Python SQLite under `query_only` + timeout, then relaunch by default.
  Results are bounded `{columns, rows, truncated}` JSON; blobs are base64 metadata.
- `aua db execute <pkg> <db> <SQL> --yes` → accept data mutation only
  (`INSERT|UPDATE|DELETE|REPLACE|WITH`), automatically persist a private restore point,
  execute one transaction, reject schema/PRAGMA/ATTACH/transaction control, verify schema
  stability + new foreign-key violations + `integrity_check`, consolidate WAL state,
  atomically replace the primary, remove stale sidecars, and relaunch. No confirmation means
  no stop/read/write side effect.
- `aua db backup|backups|restore` → explicit restore points scoped by device/package/database;
  restore requires `--yes` and first backs up the state it is about to replace.
- Database snapshots/backups can contain user data. They stay in AUA's private cache with
  restrictive permissions; journal entries redact SQL and bind parameters.
- The single-device dashboard detail view exposes the same list/schema/query/backup/execute/
  restore service. Query results are bounded; browser writes use a per-dashboard request token,
  and execute/restore require server-verified typed confirmation phrases before the engine is
  called.

### Config
- `aua config init` → write a commented default config to the user config path
- `aua config show [--effective]` → print config (effective = after precedence merge)
- `aua config path` → print resolved config file path
- `aua doctor` → check environment: adb present, device reachable, uiautomator2 agent
  installed, which OCR/detection/grounding providers are available & why (missing dep,
  missing key, unreachable endpoint). **Must never print secret values.**

### Memory / app map (§6b)
- `aua map [--app <pkg>] [--brief] [--screen <name>] [--depth N] [--find "<goal>"] [--context <id>|--all-contexts] [--audit] [--json]` → print the active flag context as a logical-screen outline, compare variants across all contexts, or emit structural issues and agent research questions. `--brief` = skeleton only; default = screens + key elements + context-grouped routes; `--find "image"` returns just the screen(s) + route to a target.
- `aua goto "<goal>" [--plan] [--max-steps N] [--allow-destructive]` → the navigation autopilot: resolve the (fuzzy) goal against the map, walk the shortest route from the current screen, and **replay each edge's recorded steps** (resource-id selector first, then label), verifying `known_screen` per hop. Cross-app auth legs (edges through `memory.transit_packages`) replay end-to-end with package-aware perception; steps matching `memory.destructive_labels` are refused without `--allow-destructive`. On divergence it exits `1` with the failing step, the remaining steps, and the current elements; a re-run **resumes**, even stranded mid-auth. `--plan` prints the annotated route (steps / replayable / legacy / destructive) without acting.
- `aua flow run [<name>|--file PATH|--yaml BODY] [--param K=V]… [--dry-run] [--from-step N] [--no-allow-destructive] [--assist] [--artifacts-dir DIR --evidence none|failures|all --junit]` / `aua flow save <name> [--last N] [--save] [--force]` / `aua flow list|show|delete` → **flows**: Maestro-style YAML journeys under `<memory.dir>/flows/`, authored directly or materialized from the newest homogeneous origin/context suffix of the session's recent actions. Save is preview-first and writes only with `--save`; the preview discloses scope/boundaries, selector safety, and mapped-screen arrival proof. New recordings select a unique stable resource id, then unique non-PII content description, then unique stable non-PII text; unsafe selectorless captures are refused. Typed values become required `${PARAM_n}` placeholders and are never persisted. A freshly recognized terminal screen is stored as typed `arrival_screen` proof and verified through known-screen recognition on replay; unmapped destinations remain explicitly unverified, while legacy `arrival:` predicates remain compatible. `flow run` replays the whole journey (launch, taps, input with `${PARAM}` substitution, key/swipe/scroll, waits, shared `expect`-style rich assertions including exact count/state, explicit horizontal/vertical order assertions, named screenshots, `goto:` steps, transit-package legs) through the same executor as `goto`, returning assertion detail and a resumable step index on divergence. Artifact mode emits a canonical unresolved `flow.yaml`, structured result/manifest, Markdown report, selected screenshot/observation evidence, supported-platform failure diagnostics, per-step evidence IDs/timings, and optional JUnit without overwriting a prior run. CLI, daemon, and MCP use the same engine path and source contract. Flows are deliberate authored intent → destructive steps allowed by default.
- `aua navigate "<goal>" [--until TEXT] [--max-steps N] [--allow-destructive] [--save-flow NAME]` → **opt-in autonomous navigation** (§7.3 planner; requires `planner.enabled`). A fast LLM drives to a natural-language goal with no prior map, recording the path into memory so a later `aua goto` replays it deterministically (the self-improvement flywheel); `--save-flow` also materializes it as a flow. `goto`/`flow run` gain `--assist` to invoke the same planner for one-call divergence recovery.
- `aua explore mine <repo> --app <pkg>` / `aua explore plan [--app <pkg>]` → **app indexing**. `mine` harvests deeplink shortcuts from the app's source (AndroidManifest intent-filters + `navDeepLink`/`uriPattern` literals for custom schemes; test/build sources skipped) into the playbook so the agent can `aua open-and-analyze` them. `plan` returns a prioritized crawl worklist (probe unprobed deeplinks, fill templated ones, expand dead-end screens) whose results auto-record — the agent runs it and re-runs until it converges (the "have the agent index the app" loop).
- `aua open-and-analyze <uri>` → open a deeplink and return the resulting screen; remembered in the playbook and marked probed.
- `aua about [--app <pkg>]` → print the app playbook (description, recipes, deeplinks, notes); `aua remember --about/--note/--recipe/--deeplink` teaches it.
- `aua knowledge list|show|add|stale` → provenance-bearing feedback and source/runtime facts, scoped by package/version/context.
- `aua reconcile plan|submit|status|apply|rollback` → external-agent research contract and transactional correction with snapshots and rollback.
- `aua memory show|path|update|forget [--app <pkg>] [--screen <name>]` → inspect / locate / force-record (or rename) the current screen / clear. Recording is automatic by default (§6b).

### Agent guide (self-documentation)
- `aua guide` (aliases `skill`, `agent`) → print the **agent operating manual** to stdout (markdown; `--json` for structured, `--brief` for short). It tells an agent everything needed to use the tool: what it is; the recommended **session protocol** — (1) `aua daemon start` for speed, (2) `aua map` to load the app's known layout before navigating, (3) drive with `analyze`/`has` and the `*-and-analyze` action commands acting on element **IDs**, (4) use `wait-and-analyze --for-stable`/`--for` instead of fixed sleeps, (5) `aua daemon stop` when done; how perception **self-routes** (the §6a escalation ladder — hierarchy→vision automatically; paid grounding only with `--deep`); how **memory** works (auto-recorded, read via `aua map`, `meta.known_screen`); the output schema; exit codes; and key global flags. This is the **single source of truth** that also generates `.claude/skills/android-ui-analyser/SKILL.md` (`aua guide --emit-skill [path]`), and the `aua --help` epilog points the agent to it.

### MCP
- `aua mcp` → run the MCP server over stdio, exposing the same tools (§11)

Exit codes: `0` success; `1` negative result (`has`: text absent; `goto`/`flow run`:
did not arrive / diverged); `2` usage error; `3` no device / device error; `4` provider
error after exhausting fallbacks; `5` config error. Errors print a structured object
to stderr: `{ "error": { "code", "message", "hint" } }` — actionable, in the tool's
voice (e.g. `"hint": "Set GEMINI_API_KEY or choose a local grounding provider."`).

---

## 6. The `analyze` pipeline (engine)

1. **Capture** current state over a warm `uiautomator2` connection: hierarchy XML +
   (lazily) a screenshot. Screenshot only taken if vision is needed or `--annotate`.
2. **Parse hierarchy** → element list (`hierarchy.py`): extract bounds, resource-id,
   text, content-desc, class (short name), clickable/enabled/focused; drop zero-area
   and fully off-screen nodes; compute centers; assign integer IDs in stable
   top-to-bottom, left-to-right order.
3. **Quality gate** (`gate.py`) decides if the hierarchy is sufficient. Rules in
   order, first match wins (all thresholds configurable):
   - usable element count below `gate.min_elements` (default 3), **or**
   - no node carries `text`/`content-desc` (likely custom-drawn), **or**
   - **hard**: the foreground *package/activity* matches `gate.vision_packages`
     (Flutter, game engines — genuinely opaque surfaces), **or**
   - ratio of clickable-with-label elements below `gate.min_labeled_ratio`, **or**
   - **soft**: an *element class* matches `gate.vision_packages` (e.g. a `WebView`
     node) AND the tree is weak — fewer than `gate.soft_min_elements` elements or a
     labeled fraction over ALL elements below `gate.soft_min_labeled_ratio`. A rich
     WebView tree (Google sign-in, most web content) stays on the fast hierarchy path.
   - `--source` overrides the gate.
4. **Vision fallback** (only if gate fails or `--source vision`):
   - **Detection** via the detection fallback chain → interactable boxes.
   - **OCR** via the OCR fallback chain → text boxes (unless `--no-ocr`).
   - **Merge**: dedupe overlapping boxes (IoU threshold), associate OCR text with
     detected boxes, assign synthetic IDs, set `source` accordingly.
5. **Grounding (optional, only for `--query`, and only after the cheap-first hierarchy
   match fails — see §6a)**: pass screenshot + instruction to the grounding chain; map
   the returned point/box to the nearest element ID.
6. **Emit** JSON (§8); cache the element list keyed by a screen signature so actions
   can resolve IDs; invalidate cache after any action command or on `--no-cache`.

> **Quick-check fast path (`has` / `wait --for`):** these do NOT run the full pipeline.
> They query the hierarchy directly for a text match (uiautomator2 selector;
> short-circuits on the first hit, ~tens of ms), and only OCR the screenshot as a
> fallback when the hierarchy has no match and `--ocr-fallback` is on. No element list,
> no ID assignment, no detection — just a boolean (plus the matched box if found).

---

## 6a. Adaptive perception — cost-aware routing (escalation ladder)

The tool must **match the method to the difficulty of the request** and pay only for
what the question needs. Perception is an ordered ladder from cheapest to most
expensive; the engine starts at the cheapest tier that *could* answer, runs it, and
**escalates only if that tier fails to produce a confident result** — bounded by config.

| Tier | Method | ~Cost | Answers |
|---|---|---|---|
| T0 | hierarchy text match (uiautomator2 selector) | ~tens of ms | "is this text/element present?" (`has`) |
| T1 | hierarchy selector locate (text / resource-id / xpath) | ~tens of ms | "give me THIS known element to act on" |
| T2 | hierarchy full parse → element list | ~50–150 ms | "what's on screen?" (`analyze`) |
| T3 | vision: detection + OCR (local) | ~150–600 ms | screens the hierarchy can't see (Compose-no-semantics, WebView, canvas, game) |
| T4 | grounding VLM (local or commercial) | ~0.5–6 s + $ | fuzzy / visual / semantic targets not resolvable above |

**Routing inputs (NO LLM is used to decide — that would defeat the purpose):**
1. **Intent from the command/verb** sets the entry tier: `has` → T0; `tap`/`find` of a
   known text/id → T1; `analyze` → T2; `analyze --query "<nl>"` → starts at T1/T2 (try
   to satisfy from the hierarchy first), NOT T4.
2. **A cheap query classifier** (regex/keyword heuristics) refines the entry tier:
   - looks like a resource-id (`pkg:id/...`) or an exact quoted literal → T0/T1
   - contains visual/relational language ("icon", "button", "top-right", "the X near
     Y", a color, "looks like") with no literal match available → candidate for T3/T4
3. **Result confidence drives escalation:** a tier that returns empty / not-found /
   low-confidence climbs one rung (if allowed); a confident hit short-circuits.

**Cheap-first for semantic queries (important):** `analyze --query "the Submit button"`
must FIRST try to satisfy the query from the hierarchy — extract salient tokens and
match against element `text` / `content-desc` / `resource-id`. Only if there is no
confident match does it escalate to the grounding VLM (T4). A well-instrumented screen
answers most "semantic" queries at T1/T2 for free.

**Bounds & overrides (cost safety):**
- `routing.auto_escalate` (default `true`) and `routing.max_tier` cap how far it climbs.
  Default ceiling is **T3 (local vision)**; T4 is entered only when the request is a
  semantic query AND grounding is enabled AND `max_tier >= grounding` (or `--deep`).
- The router **never silently escalates to a paid/commercial provider** — that tier
  must be explicitly enabled and within `max_tier`.
- `--strategy text|selector|hierarchy|vision|grounding|auto` pins a tier (supersedes
  `--source`); `--cheap` lowers the ceiling and `--deep` raises it for one call.
- `meta.tier_used` and `meta.providers_used` always report which rung actually ran, so
  callers can see when — and why — it climbed.

This formalizes behavior already implied elsewhere: the §6 quality gate is the T2→T3
rung, and `has --ocr-fallback` is the T0→T3 rung for a boolean check.

---

## 6b. App memory — persistent app maps (the tool's long-term knowledge)

The tool maintains a **persistent, per-app memory** on the local filesystem so an agent
starts each session already knowing the app's layout — what screens exist, what's on each,
and how to get from one to another (e.g. "the product list is Catalog tab → Books,
2 taps from the dashboard") — instead of re-discovering it every time. The tool builds and maintains
this map **itself** as it navigates; the agent just reads it back. Memory is a *prior*,
never a substitute for a live check when something may have changed.

### Storage (mirrors the `~/.claude` pattern)
A per-user directory, default `~/.android-ui-analyser/` (override `memory.dir`):
```
~/.android-ui-analyser/
  memory/
    <package>/                 # one folder per app, e.g. com.example.app/
      MAP.md                   # human- + AI-readable app map (what the agent reads)
      index.json               # schema v4: contexts, screens/states, trusted routes, tasks
      corrections/             # before snapshots + correction events (rollback)
      screens/<screen>.json    # optional per-screen element detail
```
One app = one folder; the tool may write/update **one or several files** per app.

### What a map contains
- **Identity & freshness:** package, app label, app version last seen, last-verified
  timestamp — so the agent knows how stale the map is.
- **Screens / sections:** each known screen gets a stable semantic name (`dashboard`, `catalog`,
  `product_list`, `product_detail`, …) plus: its activity, a **signature** (fingerprint of
  durable anchors — activity + a hash of stable resource-ids/labels) used to recognize the
  screen on revisit, the **perception tier** it needs (e.g. `product_detail` → hierarchy),
  its surface (`native`, `form`, `webview`, `canvas`), logical destination and state
  (`loading`, `error`, `empty`, `ready`), the key elements, and aliases. App-authored
  resource namespaces and inbound resource IDs outrank localized visible copy.
- **Feature-flag contexts:** a deterministic ID, exact flag set, app version, shell
  anchors, verification state, source, and runtime evidence. Screens and routes carry a context; an
  exact-context route wins over a legacy route with the same source/action.
- **Routes (navigation graph):** directed edges `screen --action--> screen`, e.g.
  `dashboard --tap nav "Catalog"--> catalog --tap "Books"--> books --tap "Example title"-->
  product_detail`. Auto-observed edges are `provisional` until independently observed again,
  then `verified`; edges without a durable selector are `rejected`. Only verified edges
  power `goto`, while provisional edges remain visible for map inspection/research.
- **Knowledge:** descriptions, notes, recipes, deeplinks, and source/runtime claims with
  package/version/context scope, status, agent/session provenance, and evidence.

### How the tool maintains it (by itself)
- **Auto-record (default on):** every `analyze` records/updates the current screen
  (signature, tier, key elements); every successful action (`tap`/`input`/…) that changes
  the screen records a **route edge** between the previous and new screen. Each edge
  carries structured **steps** (kind + resource-id tail + redacted label + package) so it
  is replayable, not just displayable; the human `action` string is derived from them.
  The map grows passively as the agent uses the tool — no extra calls. Post-action
  `observation` snapshots record too (recognition-only: they draw single-step edges when
  they land on a known screen, verify those edges immediately, and never create screens
  from a mid-transition frame).
- **Screen recognition:** on each `analyze` the tool computes the signature and, if it
  matches a known screen, sets `meta.known_screen` and attaches inline affordances —
  `meta.known_routes` (outgoing edges), `meta.suggested_gotos` (ranked ready-to-run
  targets), `meta.map_hint`, and open `meta.research_tasks` — the agent instantly knows
  where it is, what's reachable, and what map uncertainty needs source/runtime research.
- **Context recognition:** `flags set/apply` promotes verified flags after restart. In
  addition, when a package has `flags.prefs_files` or `flags.context_keys`, `analyze`
  privacy-filters current shared preferences to flag-like/exact keys and activates that
  verified context before screen recognition—even when another tool set the flags.
  Recognition first compares weighted
  resource/selected-state/content/text anchors inside that context, then falls back to
  trusted `legacy-default` screens. Existing v1/v2/v3 maps migrate without losing routes.
- **Cross-app transit:** screens in `memory.transit_packages` (Google auth via
  Chrome/GMS, permission dialogs) record into their own maps, but the **journey cursor
  stays on the origin app**, so an auth excursion returns as ONE replayable edge
  (`tap 'Continue with Example ID' … ⇢ (via com.android.chrome)`). IME/system packages
  (`memory.ignore_packages`) are never mapped and can't win the foreground vote.
- **Drift detection:** if the app version changed or weighted anchors diverge beyond
  `memory.drift_threshold`, the screen is flagged `stale` so the agent re-verifies;
  stable resource IDs can therefore preserve identity across localization changes.
  Pre-v2 (string-only) edges upgrade in place when re-walked.
- **Guarded replay:** `goto` refuses steps whose label matches
  `memory.destructive_labels` unless `--allow-destructive` — the map legitimately learns
  routes THROUGH destructive taps ("the way to the login screen is Delete").

### Map output & detail levels (the full picture, but token-aware) — CLI in §5
`aua map` prints a compact logical-screen outline for the active context. Variants and
states are grouped under their logical destination, and routes are grouped by context
instead of duplicated in a recursive tree. `--all-contexts` compares all variants;
`--find` is the focused query on top.
Detail is controlled so the agent loads only what it needs:
- `--brief` → skeleton only (screen tree + routes); smallest, load at session start.
- default → screens + their key/durable elements + routes.
- `--screen <name>` / `--depth N` → drill into one screen's full element detail.
- `--find "<goal>"` → just the screen(s) + route to a target. `--json` for any of these.
- `--context <id>` / `--all-contexts` → select one flag configuration or compare all.
- `--audit` → poor names, near-duplicate screens, stale entries, unverified contexts,
  provisional/unreplayable/conflicting routes, orphan references, and persisted
  source/runtime questions for an external research agent.
Example shape:
```
dashboard  (tier: hierarchy)
├─ nav: Catalog | Orders | Account
├─ recent-orders list (dynamic) -> order_detail
└─> catalog  (tap nav "Catalog")
    ├─ categories: Books, Music, Furniture, ...
    └─> books  (tap "Books")
        ├─ search field (EditText), sort, tabs: Featured|New
        └─> product_detail  (tap "Example title"; tier: hierarchy) — Add to cart, Share
```

### Static skeleton vs. dynamic content (why it stays small and fresh)
Memory stores the **durable skeleton** — screens, routes, and stable elements (tabs,
buttons, category names) — **not** volatile per-user data. A list such as recent orders is
recorded as a *shape* ("dashboard → recent-orders list (dynamic), each opens `order_detail`"),
**not** the literal "first chat, second chat…". Those items are live data: the agent
fetches them on demand with `analyze` when it actually needs them. This keeps the map
compact and always-fresh and avoids persisting user content (PII). So: the map gives the
full **structural** picture; live **contents** come from `analyze`.

### CLI (agent-facing) — see §5
`aua map [--brief|--screen|--depth|--find|--audit|--context|--all-contexts]` loads the map / answers "where is X and how do
I get there"; `aua goto "<goal>"` drives it (the autopilot); `aua flow run|save|list|show|
delete` replays whole journeys (Maestro-style, agent-authored or recorded);
`aua memory show|path|update|forget` inspects and manages it. `aua knowledge
list|show|add|stale` manages provenance-bearing experience while `remember`/`about`
remain backward compatible.

### Bidirectional research and correction
Map recording and `aua map --audit` automatically materialize `ResearchTask` objects
containing package/version/context, affected stable IDs, observations, conflicts, and
questions. They appear in `meta.research_tasks` and `MAP.md`; `aua reconcile plan` is the
explicit JSON export. The caller sends these to an external coding/runtime agent; AUA
never spawns a model. `reconcile submit` accepts a
`ResearchReport` containing agent/session, `apply|review|reject`, rationale, evidence,
knowledge, and typed correction operations. `apply` reports are committed automatically:
the operations run on a deep copy, structural invariants are checked, the old map is
snapshotted, and the replacement is written atomically. A correction event includes its
rollback ID. `review` queues the report without mutation; `reject` retains the feedback.
`reconcile status|apply|rollback` completes the lifecycle, with equivalent MCP tools.

### Privacy
Memory is **local-only**, never transmitted. The tool stores structure and durable labels,
**not** volatile or sensitive content: text in password / `FLAG_SECURE` / likely-PII fields
is redacted, and `EditText` *values* are stored as shape (e.g. `"<filled>"`) rather than
verbatim. `memory.redact` (default on) controls this.

---

## 7. Provider system (pluggable models + fallbacks)

Five provider kinds, each behind an abstract base class in `providers/base.py` and
resolved by `providers/registry.py`. Perception and planner chains try providers in
order; on exception, timeout, or empty result they advance to the next. The policy
boundary is narrower: deterministic code constructs and guards complete calls before
the configured selector sees an independent privacy-screened projection and returns one
opaque ID; AUA retains the authoritative call map.

### 7.1 Interfaces
- `OcrProvider.recognize(image) -> list[TextBox{text, bounds, confidence}]`
- `DetectionProvider.detect(image) -> list[Box{bounds, label?, interactable?, confidence}]`
- `GroundingProvider.locate(image, instruction) -> Point|Box` and/or
  `GroundingProvider.parse(image) -> list[Element]` (for VLMs that can do full parsing)
- `PlannerProvider.decide(objective, elements, image?) -> PlannerDecision{action, target_id?, text?, arg?}`
  — an LLM navigator that picks the next action (or `done`/`give-up`) from a compact
  element list (image attached only on weakly-labelled screens).
- `PolicyProvider.select(context) -> int|null` — choose one offered opaque candidate ID.
  The exact calls remain on the trusted side; a policy cannot author arguments, grant
  authorization, execute a call, or relax deterministic guards.

Each provider declares `name`, `is_available() -> (bool, reason)` (checks deps,
platform, keys, endpoint), and reads its settings from the resolved config.

### 7.2 Built-in providers (implement all)
**OCR**
- `apple_vision` — macOS only, via PyObjC `Vision` framework (no network). Default OCR
  on macOS. Must degrade gracefully (mark unavailable) on non-macOS.
- `rapidocr` — ONNXRuntime, cross-platform. Default OCR fallback / non-macOS default.
- `paddleocr` — PP-OCRv5; highest accuracy, slower.
- `tesseract` — via `pytesseract` (requires system tesseract); tiny.
- `easyocr` — optional.

**Detection**
- `omniparser` — OmniParser v2 **detection-only** (YOLOv8 `icon_detect`); skip the
  caption model for speed. **Emit an AGPL warning on first use** (the `icon_detect`
  weights are AGPL-3.0). Local; supports MPS/CUDA/CPU.
- `yolo` — generic Ultralytics YOLO with **user-supplied weights** path (e.g. a model
  fine-tuned on RICO/VINS). License-clean default once the user supplies weights.

**Grounding / analysis** (all optional; off by default)
- `local_vllm` — OpenAI-compatible endpoint (vLLM/Ollama/LM Studio/HF TGI); configure
  `base_url` + `model` (e.g. `Hcompany/Holo1.5-7B`, `Qwen/Qwen2.5-VL-7B-Instruct`).
- `openai` — GPT-5-class vision; key from `OPENAI_API_KEY`.
- `anthropic` — Claude vision; key from `ANTHROPIC_API_KEY`.
- `gemini` — Gemini vision; key from `GEMINI_API_KEY`.
- Commercial providers send the screenshot + a strict prompt instructing the model to
  return **only** JSON (element list or a single `{id|point|box}`); responses are
  parsed defensively (strip code fences, validate against the schema).

**Planner** (optional; off by default — powers `--assist` + `aua navigate`, §6b)
- `gemini_flash` — Gemini Flash Lite (`GEMINI_API_KEY`); the default. Given a goal + the
  compact on-screen element list (text-only; a screenshot is attached only when the
  screen is weakly labelled), it returns the next action to take. The `id` it may target
  is validated against the provided set (prompt-injection guard), its taps obey the
  destructive guard, and it is bounded by a per-recovery step cap. Never on the escalation
  ladder and never invoked on the happy path — gated by `planner.enabled` **and** an
  explicit per-call opt-in (`--assist`, or the `aua navigate` command).

**Guarded policy** (optional; `policy.enabled: false`, `mode: off` by default)
- `functiongemma` — local Apple-silicon MLX selector for AUA-authored tap candidates in an
  active verification phase. AUA ships the modified ~15.2 MB LoRA adapter under the included
  Gemma terms; the pinned ~543 MiB MLX base remains external and is never downloaded
  automatically. `shadow` returns audit metadata only; `advisory` may return a separate exact
  `policy_suggestion`, but never replaces or executes the deterministic recommendation. Bundled
  v3's authenticated manifest caps rollout at shadow, so advisory fails closed as
  `unsupported_mode` before inference.
- Guarding happens before and after inference. Zero candidates return a non-executing structured
  handoff in advisory mode; reserved ID `-1` is accepted from a model only when its authenticated
  manifest binds that handoff protocol. One action bypasses the model, and the frozen v3 adapter
  runs only on exactly four candidates; two/three are an unsupported cardinality. Its synthetic v3 static gate remains **FAIL** (one unauthorized and
  one redundant raw selection). A 96-case host-only engine-shaped production-serializer smoke also
  failed at 60/96 semantic accuracy (62.5%) with 37.5-point target-ID and 54.17-point target-position
  gaps, despite 100% protocol/offered-ID/provider agreement.
- A failure-driven v4 continuation fixed the untouched smoke (96/96), reached 2,767/2,768 validation
  (99.9639%, including 719/720 production-shaped cases), perfect held-out production choices at cardinalities
  2/3/4 (64/64, 144/144, 512/512), and a 4/4 clean closed loop. It was still rejected: the
  independent combined test was 2,764/2,768 (99.8555%) with 99.6875% critical accuracy, 100%
  parsing, zero redundant selections, and four unauthorized early `session_finish` choices in
  `sequence_recover_unknown` instead of
  `analyze_screen`. V4 remains ignored/not bundled, v3 remains shadow-only, and the next iteration
  requires independent recovery-focused training data and evaluation. Unguarded/autonomous
  execution remains out of scope.
- `aua session autopilot` / MCP `session_autopilot` is the one explicit bounded execution lane for
  a separately authenticated advisory-capable adapter. The selector still emits only an opaque ID;
  the warm AUA daemon revalidates and executes the trusted current-frame tap, consumes its folded
  observation, and repeats within step/time limits. It never relays the call through the parent
  agent. Unknown/stale outcomes, unchanged frames, repeated calls, handoff, input/toggle/wait/proof
  work, or exhausted budgets stop immediately and return the fresh screen. Bundled shadow-only v3
  cannot use this lane.

### 7.3 Adding a provider
Document (in README) the contract: subclass the relevant base, register via an entry
point or the registry's decorator, expose settings under `models.<name>` in config.
A new provider must require **zero** changes to the engine or CLI.

---

## 8. Output schema (canonical, versioned)

Top-level: `{ "schema_version": 1, "screen": {...}, "elements": [...], "meta": {...} }`

- `screen`: `{ width, height, package, activity, source }` where
  `source ∈ hierarchy|vision|mixed`.
- `elements[]`: `{ id:int, type:str, text:str|null, resource_id:str|null,
  content_desc:str|null, bounds:[x1,y1,x2,y2], center:[x,y], clickable:bool,
  enabled:bool, focused:bool, source: hierarchy|detection|ocr|grounding,
  confidence:float|null }`.
- `meta`: `{ duration_ms:int, tier_used: text|selector|hierarchy|vision|grounding,
  path: hierarchy|vision, providers_used:[...], known_screen:str|null,
  annotated_image:str|null, device_serial:str }`.

Formats: `json` (single line), `pretty` (indented), `compact` (drop null fields and
`enabled`/`focused`/`confidence` when default; smallest token footprint for agents).
The schema is defined with **pydantic** models in `schema.py` and is the single source
of truth (CLI, MCP, and tests all use it).

---

## 9. Configuration system

- **Format:** YAML. **Locations & precedence (highest first):**
  1. `--config <path>` / individual CLI flags
  2. environment variables (`AUA_*`, plus provider key vars like `OPENAI_API_KEY`)
  3. project config: nearest `.android-ui-analyser.yaml` walking up from CWD
  4. user config: `$XDG_CONFIG_HOME/android-ui-analyser/config.yaml`
     (default `~/.config/...`)
  5. built-in defaults
- **Profiles:** a config may define `profiles: { local: {...}, cloud: {...} }`;
  `--profile` deep-merges the chosen profile over the base.
- **Secrets:** never stored in config. Config references the **env var name**
  (`api_key_env: OPENAI_API_KEY`); the tool reads the value at runtime. `config show`
  and `doctor` mask/never print values.
- Loaded and validated by pydantic (`config.py`); invalid config → exit 5 with a
  precise message (which key, what was expected).

### Example `config.yaml`
```yaml
device:
  serial: null            # null = auto-detect
  backend: uiautomator2   # uiautomator2 | accessibility (future)
perception:
  gate:
    min_elements: 3
    min_labeled_ratio: 0.15
    vision_packages: ["io.flutter", "com.unity3d", "org.libsdl", "*.WebView"]
    soft_min_elements: 8        # element-CLASS matches escalate only on a weak tree
    soft_min_labeled_ratio: 0.3 # (package/activity matches always escalate)
routing:
  auto_escalate: true
  max_tier: vision          # text < selector < hierarchy < vision < grounding
  semantic_query_hierarchy_first: true   # satisfy NL queries from the hierarchy before any VLM
output:
  format: json
  annotate: false
ocr:
  enabled: true
  chain: [apple_vision, rapidocr]      # ordered fallback
detection:
  enabled: true
  chain: [yolo, omniparser]            # yolo first (license-clean) if weights present
grounding:
  enabled: false                       # opt-in
  chain: [local_vllm, gemini]
planner:
  enabled: false                       # opt-in LLM navigator (--assist / `aua navigate`)
  chain: [gemini_flash]
policy:
  enabled: false                       # off by default; never autonomous execution
  mode: off                            # off | shadow | advisory
  chain: [functiongemma]
  max_candidates: 4
models:
  yolo:        { weights: "~/models/ui-yolo.pt", device: mps, conf: 0.25 }
  omniparser:  { device: mps, accept_agpl: false }   # must be true to actually run
  rapidocr:    { lang: en }
  apple_vision:{ recognition_level: accurate }
  local_vllm:  { base_url: "http://localhost:8000/v1", model: "Hcompany/Holo1.5-7B" }
  openai:      { model: "gpt-5", api_key_env: OPENAI_API_KEY }
  anthropic:   { model: "claude-opus-4-8", api_key_env: ANTHROPIC_API_KEY }
  gemini:      { model: "gemini-2.5-pro", api_key_env: GEMINI_API_KEY }
  gemini_flash:{ model: "gemini-2.5-flash-lite", api_key_env: GEMINI_API_KEY }
  functiongemma:
    model_path: null                   # absolute path to pinned external MLX base
    adapter_path: null                 # null / bundled = packaged LoRA
    max_tokens: 24
daemon:
  enabled: true
  socket: "~/.cache/android-ui-analyser/daemon.sock"
memory:
  enabled: true
  auto_record: true        # record screens + route edges on every analyze/action
  dir: "~/.android-ui-analyser"
  drift_threshold: 0.3     # signature divergence that flags a screen stale
  redact: true             # never store secrets / PII / EditText values verbatim
  auto_research: true      # materialize audit questions and push them through analyze
  research_suggest_max: 3
  ignore_packages: ["com.android.systemui", "*inputmethod*"]   # never the foreground app
  transit_packages: ["com.google.android.gms", "com.android.chrome",
                     "com.android.permissioncontroller", "com.google.android.permissioncontroller"]
  destructive_labels: ["delete", "remove", "sign out", "log out", "logout", "pay", "buy",
                       "purchase", "subscribe", "unsubscribe", "uninstall", "format",
                       "erase", "reset", "deactivate"]
flags:
  auto_context: true
  prefs_files: {com.example.app.dev: "flag_overrides.xml"}
  # exact privacy allow-list (otherwise experiment/treatment/variant/flag-like keys only)
  context_keys: {com.example.app.dev: [catalog_experiment, services_treatment]}
```

> Note: model identifiers above are examples; the implementer should not hard-code a
> model's existence — read it from config and pass it through to the provider/endpoint.

---

## 10. Daemon mode (warm state)

To eliminate per-call cold-start, `aua daemon start` launches a small background
process holding the warm `uiautomator2` connection and the loaded vision models. The
CLI auto-detects a running daemon (via the configured unix socket) and forwards
`analyze`/action requests to it; otherwise it runs in-process (still correct, just
pays startup). `daemon stop`/`status` manage it. The daemon is **optional** — every
command must work without it. Protocol: newline-delimited JSON over a unix domain
socket (simple, local-only, no auth surface). The daemon must hot-reload nothing
sensitive and bind only to the socket (never a TCP port by default).

## 11. MCP wrapper (optional, build it)

`aua mcp` runs an MCP server (stdio) using the Python MCP SDK, exposing tools that map
1:1 to the engine: `analyze_screen(source?, with_ocr?, query?)`, `tap(id)`,
`long_press(id, ms?)`, `input(id, text, submit?)`, `swipe(direction|coords)`,
`scroll_to(text, match?, ignore_case?)`, `key(name)`, `wait(for?, idle?)`,
`wait_stable(interval?, settle?, timeout?)`, `has(text, match?, ignore_case?,
ocr_fallback?)`, `screenshot(annotate?)`, `inspect(id)`,
`goto(goal, plan?, max_steps?, allow_destructive?)`, `flow_run(name, params?, dry_run?,
from_step?, allow_destructive?)`, `list_devices()`, and the `database_list`,
`database_schema`, `database_query`, `database_execute`, `database_backup`,
`database_backups`, and `database_restore` equivalents. Tool results are the
same pydantic-validated JSON as the CLI. The MCP layer must be a **thin** adapter over
the engine — no perception logic of its own.

## 12. Tech stack & dependencies

- **Python 3.11+**, packaged with `pyproject.toml` (PEP 621). Installable via `pipx`.
- **CLI:** `typer` (or `click`). **Config/schema:** `pydantic` v2 + `pyyaml`.
- **Device:** `uiautomator2`. **Images:** `Pillow` (annotation), `numpy`.
- **HTTP (commercial/local providers):** `httpx`.
- **MCP:** the official `mcp` Python SDK.
- **Optional/extra deps grouped by provider** so a base install is light:
  `pip install android-ui-analyser[apple]` (pyobjc Vision), `[rapidocr]`, `[paddle]`,
  `[tesseract]`, `[easyocr]`, `[yolo]` (ultralytics + torch), `[omniparser]`, `[all]`.
- Providers must **lazy-import** heavy deps inside `is_available()` / on first use, so
  missing optional deps never break the core CLI.

## 13. Testing & acceptance criteria

### 13.1 Device-less (must all pass in CI / the build session, no phone required)
- **AC1** `pipx install .` (or `pip install -e .`) succeeds; `aua --help` and
  `aua --version` work.
- **AC2** `hierarchy.py` parses a set of **fixture XML files** (committed under
  `tests/fixtures/`, incl. a normal Views screen, a Compose-without-semantics screen,
  and an empty/canvas screen) into the exact expected element JSON (golden files).
- **AC3** The **quality gate** returns `vision` for the empty/canvas fixture and
  `hierarchy` for the normal fixture, per configured thresholds.
- **AC4** The **fallback chain runner**: given a stub chain `[fail, ok]`, it skips the
  failing provider and returns the second's result; given `[fail, fail]` it raises
  `ProviderError` and the CLI exits 4 with a structured error.
- **AC5** **Config precedence**: a test sets a default, overrides it via project file,
  env var, and flag, and asserts the effective value follows §9 precedence. Secrets
  referenced by env name are read correctly and **never** appear in `config show`.
- **AC6** **Commercial provider wiring** (mocked HTTP): with `grounding.chain:[openai]`
  and `OPENAI_API_KEY` set, `analyze --query "..."` builds the correct request, parses
  a JSON response (including when wrapped in code fences), and returns an element ID.
  With the key unset, the provider reports unavailable and the chain advances/errors
  with a clear hint.
- **AC7** **Schema**: all emitted JSON validates against the pydantic models; `compact`
  is a strict subset; `pretty` round-trips.
- **AC8** **MCP**: an in-process MCP client lists the tools and calls `analyze_screen`
  against a mocked device, receiving schema-valid JSON.
- **AC8b** **App databases**: a fake debuggable device serves a SQLite main/WAL/SHM
  snapshot; list/schema/query work without an on-device SQLite binary; query cannot write;
  execute requires confirmation, creates a restore point, mutates data while preserving the
  schema and foreign keys, removes stale sidecars, and restore recovers the original rows while
  preserving the replaced state as a second backup. CLI, daemon, and MCP route to the same
  engine methods, command journaling stores neither SQL literals nor bind values, and the
  dashboard database workspace enforces request-token plus typed-confirmation mutation guards.
- **AC9** **`aua doctor`** runs with no device and reports each subsystem's
  availability + reason, leaking no secrets.
- **AC10** Unit coverage for merge/dedup (IoU), ID assignment ordering, annotation
  image generation (assert boxes + labels drawn at expected coords on a synthetic image).
- **AC11** `has` quick-check: against the normal-screen fixture, `has "<known text>"`
  returns `found` via `hierarchy` with exit `0`; an absent string exits `1`; with the
  hierarchy stubbed to miss, a stubbed OCR result containing the text, and
  `--ocr-fallback`, it returns `found` via `ocr`. `--match` / `--ignore-case` / `regex`
  behave as specified.
- **AC12** Adaptive routing (§6a): an exact-literal `has` / locate resolves at the
  hierarchy tier and the vision providers are NEVER invoked (assert via a spy);
  `analyze --query "the Submit button"` against a fixture whose hierarchy contains a
  "Submit" element resolves without calling the (mocked) grounding provider; and with
  `max_tier: vision`, a semantic query that misses the hierarchy does NOT call the paid
  grounding provider — it reports not-found with `tier_used` ≤ `vision`.
- **AC13** App memory: after a scripted sequence of `analyze` + `tap` over fixture
  screens, `aua map` lists the visited screens with signatures and the route edges between
  them; revisiting a recorded screen sets `meta.known_screen`; a changed signature/version
  marks the screen `stale`; secrets / `EditText` values are redacted (never stored
  verbatim); all writes stay under `memory.dir`.
- **AC14** `wait --for-stable` returns once a (stubbed) screenshot stream stops changing
  for `--settle` ms, and times out with a clear error if it never stabilizes — without
  running OCR or hierarchy parse.
- **AC15** `aua guide` prints the agent manual covering the session protocol (daemon,
  `aua map`, ID-based actions, `wait --for-stable`), the escalation ladder, memory, the
  schema, and exit codes; `--json` / `--brief` work; `aua --help` references `aua guide`;
  and the generated `.claude/skills/android-ui-analyser/SKILL.md` is produced from the
  same source (no drift).
- **AC16** The optional policy is off by default and imports/loads no model on that path.
  Device-less tests prove candidate privacy and safety filtering, zero/one/four cardinality
  behavior, fail-closed parsing and stale-frame revalidation, separate shadow/advisory output,
  CLI/MCP `policy_status` parity, bundled-adapter provenance, and that the deterministic
  recommendation is never replaced or executed by the model.

### 13.2 Device smoke test (documented for the human; runs when a device/emulator is attached)
- `SMOKE.md` describes: start an emulator, `aua doctor`, `aua devices`, `aua analyze`
  on the launcher and on a sample app, `aua tap-and-analyze <id>`,
  `aua input-and-analyze <id> "text"`,
  `aua has "<text visible on screen>"` (and a string that isn't, to confirm exit 1),
  `aua analyze --source vision --annotate` on a Compose/Flutter/WebView/game screen,
  and (optional) `analyze --query` with a configured local or commercial grounding model.
  Include expected latencies (§Architecture budget) as a sanity check.

## 14. Non-functional requirements
- **Performance:** meet the §Architecture latency budget; `analyze` must lazily avoid
  taking a screenshot on the hierarchy happy path.
- **Reliability:** auto-reconnect to the device once on transient `uiautomator2`
  errors before failing; clear exit codes.
- **Security/privacy:** secrets only via env; local daemon socket only (no default TCP);
  never log secret values; OmniParser detection must require explicit `accept_agpl:
  true` before running and must be pinned to a non-vulnerable version range. Policy prompts
  contain only privacy-screened metadata and opaque IDs, never typed values, raw hierarchy,
  session/device identifiers, or trusted call values outside the explicit safe projection.
- **Logging:** structured logs to stderr, JSON results to stdout (so piping is clean).
- **Docs:** `README.md` (install, quickstart, CLAUDE.md snippet for Claude Code,
  provider matrix with license flags), `SMOKE.md`, inline docstrings.
- **Code quality:** typed throughout; `ruff` + `mypy` clean; formatted.

---

## 15. Project structure (target)

```
android-ui-analyser/
├── pyproject.toml
├── README.md
├── SMOKE.md
├── PRD.md
├── docs/
│   ├── RESEARCH.md
│   └── ARCHITECTURE.md
├── src/android_ui_analyser/
│   ├── __init__.py
│   ├── cli.py              # typer app + subcommands (thin; calls engine)
│   ├── config.py           # pydantic config models, loading, precedence, profiles
│   ├── schema.py           # Element/Screen/AnalyzeResult pydantic models (source of truth)
│   ├── engine.py           # analyze pipeline orchestration; action dispatch; cache
│   ├── policy.py           # dependency-free candidate guard + opaque-ID selection contract
│   ├── device.py           # uiautomator2 wrapper: warm connect, screenshot, input, reconnect
│   ├── hierarchy.py        # XML -> elements (bounds parse, filtering, ID assignment)
│   ├── gate.py             # quality-gate heuristics (configurable)
│   ├── merge.py            # IoU dedup, OCR<->box association, synthetic IDs
│   ├── annotate.py         # Set-of-Marks overlay image (Pillow)
│   ├── daemon.py           # unix-socket daemon + client transport
│   ├── mcp_server.py       # optional MCP wrapper over the engine
│   ├── resources/functiongemma/ # small separately licensed LoRA + manifest/notices
│   ├── errors.py           # typed errors + structured stderr emitter + exit codes
│   ├── memory.py           # persistent per-app map: record/recognize/drift + MAP.md & index.json
│   └── providers/
│       ├── base.py         # OcrProvider / DetectionProvider / GroundingProvider ABCs
│       ├── registry.py     # registration + ordered fallback-chain runner
│       ├── ocr/            # apple_vision.py, rapidocr.py, paddleocr.py, tesseract.py, easyocr.py
│       ├── detection/      # omniparser.py, yolo.py
│       └── grounding/      # local_vllm.py, openai.py, anthropic.py, gemini.py
└── tests/
    ├── fixtures/           # hierarchy XML samples + golden JSON + synthetic screenshots
    ├── test_hierarchy.py  test_gate.py  test_merge.py  test_config.py
    ├── test_chain.py  test_schema.py  test_providers_mocked.py  test_mcp.py
    └── conftest.py         # mocked uiautomator2 device, stub providers
```

## 16. Build task list (one milestone — internal order, not phases)

Do all of these in a single run; later items depend on earlier ones:
1. Scaffold package, `pyproject.toml` (with optional-dependency extras), tooling
   (`ruff`, `mypy`, `pytest`), `__init__`.
2. `schema.py` (pydantic models) — the contract everything else uses.
3. `config.py` — models, loading, precedence, profiles, secret-by-env, validation.
4. `errors.py` — typed errors, structured stderr, exit codes.
5. `hierarchy.py` + fixtures + golden tests (AC2).
6. `gate.py` + tests (AC3).
7. `device.py` — `uiautomator2` wrapper (warm connect, screenshot, input, reconnect),
   fully mockable; `conftest.py` mock device.
8. `providers/base.py` + `registry.py` chain runner + tests (AC4).
9. OCR providers (apple_vision, rapidocr first; then paddle/tesseract/easyocr).
10. Detection providers (yolo with user weights; omniparser detection-only + AGPL gate).
11. `merge.py` (IoU dedup, OCR association, synthetic IDs) + tests (AC10).
12. Grounding providers (local_vllm, openai, anthropic, gemini) with mocked HTTP +
    defensive JSON parsing + tests (AC6).
13. `annotate.py` + tests (AC10).
14. `engine.py` — wire the full pipeline (§6), caching, action dispatch, the quick-check
    fast path for `has` / `wait --for`, and the §6a cost-aware routing/escalation ladder
    (entry tier from intent + a cheap query classifier, confidence-based escalation,
    `max_tier` ceiling, never auto-escalating to a paid provider).
15. `cli.py` — all commands (§5), formats, `doctor`, exit codes (AC1, AC9).
16. `daemon.py` — socket daemon + client + auto-detect.
16b. `memory.py` — persistent per-app app-map: auto-record screens + route edges on each
    `analyze`/action, screen recognition (`meta.known_screen`) + drift detection +
    redaction, generate `MAP.md`/`index.json`, and the `aua map` / `aua memory …` commands
    (§6b). Also implement `wait --for-stable` (screenshot-settled detection).
17. `mcp_server.py` + in-process MCP test (AC8).
17b. `aua guide` — agent operating manual (markdown / `--json` / `--brief`) from a single
    canonical source that also emits `.claude/skills/android-ui-analyser/SKILL.md`
    (`--emit-skill`); reference it from the `aua --help` epilog (AC15).
18. `README.md` (incl. CLAUDE.md snippet + provider/license matrix) and `SMOKE.md`.
19. Run `ruff`, `mypy`, `pytest`; fix until **all** §13.1 acceptance criteria pass.
20. Final self-review against §17 Definition of Done.

## 17. Definition of Done
- All §13.1 acceptance criteria pass; `ruff` and `mypy` are clean.
- `aua` installs and runs; `aua doctor` works with no device and leaks no secrets.
- Default config is commercially-licensable (no AGPL/research components active);
  opting into OmniParser requires `accept_agpl: true`; FunctionGemma inference remains off.
- The MIT project license has an explicit carve-out for the packaged FunctionGemma Model
  Derivative, whose directory carries the Gemma agreement, prohibited-use policy, notices,
  modification statement, and frozen provenance. The base model is not bundled or auto-fetched.
- A developer can: add an API key via env + select a commercial grounding model in
  config and have `analyze --query` use it; swap OCR engines by editing one config
  line; run with or without the daemon.
- `README.md` lets a new user (and Claude Code via the CLAUDE.md snippet) start in
  minutes; `SMOKE.md` covers the on-device verification.
- The MCP wrapper exposes the same capabilities as the CLI.

## 18. Operational guidance for the implementing agent (`/goal`)
- Work entirely inside this repository.
- Use a local virtualenv; install dev + the lighter provider extras you can
  (`apple` on macOS, `rapidocr`, `yolo` if feasible). Heavy/large model downloads
  (OmniParser weights, a 7B VLM) are **not** required for the build — implement and
  unit-test those providers with mocks; document how to enable them.
- Assume **no Android device** is attached: rely on fixtures/mocks for all automated
  tests; never block the build waiting for a device.
- Prefer correctness + the acceptance tests over breadth of optional providers, but
  implement every provider listed (mocked where a live backend isn't available).
- Commit logically as you go. Keep the engine free of interface/provider specifics.
- When done, print a summary: what was built, test results, how to run the smoke test,
  and any provider that is stubbed/needs setup.

## 19. Risks, licensing & open questions
- **Licensing:** OmniParser `icon_detect` is AGPL-3.0 → gated behind `accept_agpl`;
  default detection is user-weights YOLO. Research-only models (Holo 3B/72B,
  Holo2-30B, UI-TARS-72B, Ferret-UI Lite) are out of scope for defaults.
- **FunctionGemma derivative:** the bundled adapter is subject to the Gemma Terms and prohibited-use
  policy rather than the repository's MIT license. Keep its notices and manifest with every
  distribution. The compatible pinned MLX base remains an explicit user download. Frozen v3's
  strict static and engine-shaped smoke gates are not green, so bundled v3 is shadow-only;
  the first recovery-focused v4 cycle also failed its independent combined safety test and remains
  ignored. Advisory stays disabled until an independently evaluated recovery-focused iteration
  passes without unauthorized selections.
- **Security:** OmniParser pre-2.0.1 carries CVE-2025-55322 → pin a safe version;
  daemon is unix-socket only.
- **Latency variance:** uiautomator2 idle-waits can spike on animated screens; expose
  a `--no-wait`/idle config and document it.
- **Compose coverage:** hierarchy quality depends on app instrumentation; the vision
  fallback is the safety net. An AccessibilityService backend is a documented future
  option for lower latency / better Compose coverage (not required now).
- **Open question for the human:** do you have (or want to train) a YOLO UI-detector
  checkpoint? If not, the license-clean detection path needs weights; until then,
  OmniParser (AGPL, opt-in) or a commercial vision provider fills the gap.
```
```
*End of PRD.*
