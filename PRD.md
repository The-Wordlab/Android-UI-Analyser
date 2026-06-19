# PRD — `android-ui-analyser`

**A fast, configurable CLI that gives an AI agent structured "what's on screen and
where" for Android UI testing — hierarchy-first, with pluggable vision/OCR fallbacks
and selectable models (local or commercial).**

- Status: ready to build
- Background: see `docs/RESEARCH.md` (landscape) and `docs/ARCHITECTURE.md` (decisions)
- Working directory: `/Users/luzia/repositories/ai/android-ui-analyser`

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
- No bundled model weights (providers download or call out; document how).

## 4. Primary users & workflow

- **Primary:** an AI coding agent (Claude Code) testing an Android app.
- **Secondary:** a developer running `aua` directly to inspect a screen.

Typical loop:
1. `aua analyze` → JSON list of elements with IDs + boxes.
2. Agent decides an action → `aua tap 4` / `aua input 2 "hello"` / `aua swipe up`.
3. Repeat. `analyze` result is cached until a state-changing action invalidates it.

---

## 5. CLI specification (`aua`)

Global options (apply to all commands; override config):
- `--serial <id>` target device (default: only/first device, else error listing devices)
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

### Actions (all take element IDs from the last `analyze`; coordinates computed by the tool)
- `aua tap <id>` (alias: `click`)
- `aua long-press <id> [--ms 600]`
- `aua input <id> "<text>"` (focuses element, types; `--submit` to send IME action)
- `aua clear <id>`
- `aua swipe <up|down|left|right> [--from <id>] [--percent 50]` or `aua swipe --coords x1 y1 x2 y2`
- `aua scroll-to "<text|resource-id>"` (scroll container until element appears or limit)
- `aua key <back|home|enter|recents|KEYCODE_*>`
- `aua wait --for "<text|resource-id>" [--timeout 5000]` / `aua wait --idle` / `aua wait --for-stable [--interval 200] [--settle 600] [--timeout 30000]`
  - `--for-stable` polls cheap screenshots and returns once the screen stops changing for `--settle` ms (a perceptual-hash "screen settled" check — no OCR, no hierarchy parse; works on opaque screens). Ideal for waiting on image generation / loading. Pairs with the daemon for tight, low-cost polling.

### Device & session
- `aua devices` → list attached devices (serial, model, android version, state)
- `aua app <foreground|launch <pkg>|stop <pkg>|current>`
- `aua daemon <start|stop|status>` (§10)

### Config
- `aua config init` → write a commented default config to the user config path
- `aua config show [--effective]` → print config (effective = after precedence merge)
- `aua config path` → print resolved config file path
- `aua doctor` → check environment: adb present, device reachable, uiautomator2 agent
  installed, which OCR/detection/grounding providers are available & why (missing dep,
  missing key, unreachable endpoint). **Must never print secret values.**

### Memory / app map (§6b)
- `aua map [--app <pkg>] [--brief] [--screen <name>] [--depth N] [--find "<goal>"] [--json]` → print the app's map. **With no query it prints the WHOLE app as a compact text tree** (every known screen, what's on it, routes between them) — not just a search result. `--brief` = skeleton only (screen tree + routes, smallest — load at session start); default = screens + key elements + routes; `--screen`/`--depth` drill into one screen; `--find "image"` returns just the screen(s) + route to a target. The agent reads this at session start to know the layout before navigating.
- `aua memory show|path|update|forget [--app <pkg>] [--screen <name>]` → inspect / locate / force-record the current screen / clear. Recording is automatic by default (§6b).

### Agent guide (self-documentation)
- `aua guide` (aliases `skill`, `agent`) → print the **agent operating manual** to stdout (markdown; `--json` for structured, `--brief` for short). It tells an agent everything needed to use the tool: what it is; the recommended **session protocol** — (1) `aua daemon start` for speed, (2) `aua map` to load the app's known layout before navigating, (3) drive with `analyze`/`has`/`tap`/`input`/`swipe` acting on element **IDs**, (4) use `wait --for-stable`/`--for` instead of fixed sleeps, (5) `aua daemon stop` when done; how perception **self-routes** (the §6a escalation ladder — hierarchy→vision automatically; paid grounding only with `--deep`); how **memory** works (auto-recorded, read via `aua map`, `meta.known_screen`); the output schema; exit codes; and key global flags. This is the **single source of truth** that also generates `.claude/skills/android-ui-analyser/SKILL.md` (`aua guide --emit-skill [path]`), and the `aua --help` epilog points the agent to it.

### MCP
- `aua mcp` → run the MCP server over stdio, exposing the same tools (§11)

Exit codes: `0` success; `2` usage error; `3` no device / device error; `4` provider
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
3. **Quality gate** (`gate.py`) decides if the hierarchy is sufficient. Default
   heuristics (all configurable thresholds):
   - usable element count below `gate.min_elements` (default 3), **or**
   - no node carries `text`/`content-desc` (likely custom-drawn), **or**
   - foreground package/class matches `gate.vision_packages` (e.g. Flutter views,
     known game engines, `WebView`), **or**
   - ratio of clickable-with-label elements below `gate.min_labeled_ratio`.
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
and how to get from one to another (e.g. "the image-creation tool is Apps tab → Images,
2 taps from home") — instead of re-discovering it every time. The tool builds and maintains
this map **itself** as it navigates; the agent just reads it back. Memory is a *prior*,
never a substitute for a live check when something may have changed.

### Storage (mirrors the `~/.claude` pattern)
A per-user directory, default `~/.android-ui-analyser/` (override `memory.dir`):
```
~/.android-ui-analyser/
  memory/
    <package>/                 # one folder per app, e.g. co.thewordlab.luzia/
      MAP.md                   # human- + AI-readable app map (what the agent reads)
      index.json               # machine index: screens, signatures, routes, freshness
      screens/<screen>.json    # optional per-screen element detail
```
One app = one folder; the tool may write/update **one or several files** per app.

### What a map contains
- **Identity & freshness:** package, app label, app version last seen, last-verified
  timestamp — so the agent knows how stale the map is.
- **Screens / sections:** each known screen gets a stable name (`home`, `apps_grid`,
  `image_create`, `image_result`, …) plus: its activity, a **signature** (fingerprint of
  durable anchors — activity + a hash of stable resource-ids/labels) used to recognize the
  screen on revisit, the **perception tier** it needs (e.g. `image_result` → vision/opaque),
  the key elements (nav targets, inputs, buttons), and free-text notes.
- **Routes (navigation graph):** directed edges `screen --action--> screen`, e.g.
  `home --tap nav "Apps"--> apps_grid --tap "Images"--> image_create --input+send-->
  image_result`. This yields "how many steps, and from where" for any target.
- **Notes / gotchas:** e.g. "CleverTap promo can overlay Apps; dismiss via X (vision)."

### How the tool maintains it (by itself)
- **Auto-record (default on):** every `analyze` records/updates the current screen
  (signature, tier, key elements); every successful action (`tap`/`input`/…) that changes
  the screen records a **route edge** between the previous and new screen. The map grows
  passively as the agent uses the tool — no extra calls.
- **Screen recognition:** on each `analyze` the tool computes the signature and, if it
  matches a known screen, sets `meta.known_screen` and can attach remembered routes — the
  agent instantly knows where it is and what's reachable.
- **Drift detection:** if the app version changed or a screen's signature diverges beyond
  `memory.drift_threshold`, the screen is flagged `stale` so the agent re-verifies;
  otherwise it trusts the map.

### Map output & detail levels (the full picture, but token-aware) — CLI in §5
`aua map` with no query prints the **whole app as a compact text tree** — every known
screen, what's on it, and the routes between them. `--find` is the focused query on top.
Detail is controlled so the agent loads only what it needs:
- `--brief` → skeleton only (screen tree + routes); smallest, load at session start.
- default → screens + their key/durable elements + routes.
- `--screen <name>` / `--depth N` → drill into one screen's full element detail.
- `--find "<goal>"` → just the screen(s) + route to a target. `--json` for any of these.
Example shape:
```
home  (tier: hierarchy)
├─ nav: Chat | World Cup | Ideas | Apps
├─ recent-chats list (dynamic) -> chat_thread
└─> Apps  (tap nav "Apps")
    ├─ tools: Anima, Photoedit, Mathematics, Images, Summarize, Games, ...
    └─> image_create  (tap "Images")
        ├─ prompt field (EditText), send, aspect (9:16), tabs: Create|Edit
        └─> image_result  (input + send; tier: vision) — Your image, Share
```

### Static skeleton vs. dynamic content (why it stays small and fresh)
Memory stores the **durable skeleton** — screens, routes, and stable elements (tabs,
buttons, tool names) — **not** volatile per-user data. A list such as recent chats is
recorded as a *shape* ("home → recent-chats list (dynamic), each opens `chat_thread`"),
**not** the literal "first chat, second chat…". Those items are live data: the agent
fetches them on demand with `analyze` when it actually needs them. This keeps the map
compact and always-fresh and avoids persisting user content (PII). So: the map gives the
full **structural** picture; live **contents** come from `analyze`.

### CLI (agent-facing) — see §5
`aua map [--brief|--screen|--depth|--find]` loads the map / answers "where is X and how do
I get there"; `aua memory show|path|update|forget` inspects and manages it.

### Privacy
Memory is **local-only**, never transmitted. The tool stores structure and durable labels,
**not** volatile or sensitive content: text in password / `FLAG_SECURE` / likely-PII fields
is redacted, and `EditText` *values* are stored as shape (e.g. `"<filled>"`) rather than
verbatim. `memory.redact` (default on) controls this.

---

## 7. Provider system (pluggable models + fallbacks)

Three provider kinds, each an abstract base class in `providers/base.py`, each
resolved through an **ordered fallback chain** by `providers/registry.py`. The chain
runner tries providers in order; on exception, timeout, or empty result it logs (to
stderr) and advances to the next; if all fail it raises a `ProviderError` (exit 4).

### 7.1 Interfaces
- `OcrProvider.recognize(image) -> list[TextBox{text, bounds, confidence}]`
- `DetectionProvider.detect(image) -> list[Box{bounds, label?, interactable?, confidence}]`
- `GroundingProvider.locate(image, instruction) -> Point|Box` and/or
  `GroundingProvider.parse(image) -> list[Element]` (for VLMs that can do full parsing)

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
models:
  yolo:        { weights: "~/models/ui-yolo.pt", device: mps, conf: 0.25 }
  omniparser:  { device: mps, accept_agpl: false }   # must be true to actually run
  rapidocr:    { lang: en }
  apple_vision:{ recognition_level: accurate }
  local_vllm:  { base_url: "http://localhost:8000/v1", model: "Hcompany/Holo1.5-7B" }
  openai:      { model: "gpt-5", api_key_env: OPENAI_API_KEY }
  anthropic:   { model: "claude-opus-4-8", api_key_env: ANTHROPIC_API_KEY }
  gemini:      { model: "gemini-2.5-pro", api_key_env: GEMINI_API_KEY }
daemon:
  enabled: true
  socket: "~/.cache/android-ui-analyser/daemon.sock"
memory:
  enabled: true
  auto_record: true        # record screens + route edges on every analyze/action
  dir: "~/.android-ui-analyser"
  drift_threshold: 0.3     # signature divergence that flags a screen stale
  redact: true             # never store secrets / PII / EditText values verbatim
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
`input(id, text, submit?)`, `swipe(direction|coords)`, `key(name)`,
`wait(for?, idle?)`, `has(text, match?, ignore_case?, ocr_fallback?)`, `screenshot(annotate?)`, `list_devices()`. Tool results are the
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

### 13.2 Device smoke test (documented for the human; runs when a device/emulator is attached)
- `SMOKE.md` describes: start an emulator, `aua doctor`, `aua devices`, `aua analyze`
  on the launcher and on a sample app, `aua tap <id>`, `aua input <id> "text"`,
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
  true` before running and must be pinned to a non-vulnerable version range.
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
│   ├── device.py           # uiautomator2 wrapper: warm connect, screenshot, input, reconnect
│   ├── hierarchy.py        # XML -> elements (bounds parse, filtering, ID assignment)
│   ├── gate.py             # quality-gate heuristics (configurable)
│   ├── merge.py            # IoU dedup, OCR<->box association, synthetic IDs
│   ├── annotate.py         # Set-of-Marks overlay image (Pillow)
│   ├── daemon.py           # unix-socket daemon + client transport
│   ├── mcp_server.py       # optional MCP wrapper over the engine
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
  opting into OmniParser requires `accept_agpl: true`.
- A developer can: add an API key via env + select a commercial grounding model in
  config and have `analyze --query` use it; swap OCR engines by editing one config
  line; run with or without the daemon.
- `README.md` lets a new user (and Claude Code via the CLAUDE.md snippet) start in
  minutes; `SMOKE.md` covers the on-device verification.
- The MCP wrapper exposes the same capabilities as the CLI.

## 18. Operational guidance for the implementing agent (`/goal`)
- Work entirely inside `/Users/luzia/repositories/ai/android-ui-analyser`.
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
