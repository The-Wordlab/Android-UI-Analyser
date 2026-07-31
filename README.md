# android-ui-analyser (`aua`)

`aua` is a fast, configurable CLI that gives an AI agent structured "what's on screen and where" for Android UI testing. It reads the accessibility/view hierarchy first — returning every element with a stable integer ID, type, text, and bounding box in tens of milliseconds — and falls back to image-based detection and OCR (and optionally a grounding VLM) only on screens the hierarchy cannot see (Compose without semantics, Flutter, WebViews, canvas, games). The agent acts on **integer IDs, not pixels**: `aua tap 4` and `aua input 2 "hello"` compute coordinates internally, eliminating coordinate hallucination and shrinking the token footprint to a compact JSON list.

> **Using Claude Code?** Install `aua` as a plugin in two lines — `/plugin marketplace add The-Wordlab/Android-UI-Analyser` then `/plugin install android-ui-analyser@the-wordlab` (plus a one-time CLI install). The skill then auto-activates on Android tasks in every project. Full details: [Use it from Claude Code](#use-it-from-claude-code-the-aua-skill).

---

## Requirements

`aua` is a Python CLI that talks to an Android device or emulator over **adb**, using [`uiautomator2`](https://github.com/openatx/uiautomator2). You need three things on the host:

| Requirement | Version | Why / how to get it |
|---|---|---|
| **Python** | **3.11 or newer** | Runs the CLI. Check with `python3 --version`. |
| **Android platform-tools (`adb`)** | any recent | `aua` discovers devices and `uiautomator2` drives them through `adb`. Must be on your `PATH`. ([install](#installing-adb-platform-tools)) |
| **An Android device or emulator** | Android 7.0 (API 24) or newer | The screen `aua` inspects — a running AVD emulator **or** a USB-attached phone with USB debugging enabled. ([setup](#connect-a-device-or-emulator)) |

You do **not** need Android Studio's IDE, Gradle, or the app's source code — `aua` works against any app already installed on the device, including release builds. (Android Studio is just the easiest way to obtain `adb` and an emulator.)

Optional, only for specific features:
- **`tesseract`** system binary — only if you enable the `tesseract` OCR extra.
- A **GPU** (CUDA / Apple Metal) — speeds up the `yolo`/`omniparser` detectors and local grounding, but everything also runs on CPU.
- **API keys** (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) — only if you opt into a commercial grounding provider (off by default).

### Installing `adb` (platform-tools)

`adb` ships in the Android SDK **platform-tools**. Get it any of these ways:

- **Android Studio** → *SDK Manager* installs it under `~/Library/Android/sdk/platform-tools` (macOS) or `~/Android/Sdk/platform-tools` (Linux).
- **Standalone download**: grab [platform-tools](https://developer.android.com/tools/releases/platform-tools) and unzip.
- **Homebrew (macOS)**: `brew install android-platform-tools`.
- **Linux (Debian/Ubuntu)**: `sudo apt install android-tools-adb`.

Then make sure it's on your `PATH` (macOS / Android Studio layout shown):

```bash
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"   # add to ~/.zshrc or ~/.bashrc
adb version   # confirm it resolves
```

---

## Install

Base install (macOS / Apple Silicon, recommended extras — Python 3.11+ per [Requirements](#requirements)):

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,apple,rapidocr]"
```

Or without uv:

```bash
pip install -e ".[dev,apple,rapidocr]"
```

Global install (no extras):

```bash
uv tool install .        # or:  pipx install .
```

### Put `aua` on your PATH (works from any directory)

Like `adb`, the `aua` binary must resolve from **any** directory — you (and your agents)
will call it from project folders, not from this repo:

- `uv tool install .` / `pipx install .` (run from the clone) install it globally to
  `~/.local/bin`. If a **new** shell still can't find `aua`, run `uv tool update-shell`
  or `pipx ensurepath` once, then open a new terminal.
- `./install.sh` does all of the above automatically (uv → pipx → venv fallback). If it
  printed the venv-fallback warning, `aua` exists **only** at `<repo>/.venv/bin/aua`:
  call it by that absolute path, add `<repo>/.venv/bin` to your `PATH`, or install
  `uv`/`pipx` and re-run `./install.sh`.
- Verify from your home directory:

```bash
cd ~ && command -v aua && aua --version
```

### Optional-dependency extras matrix

| Extra | Installs | Notes |
|---|---|---|
| `apple` | `pyobjc-framework-Vision` | Apple Vision OCR — **macOS only**, fastest OCR on Mac |
| `rapidocr` | `rapidocr-onnxruntime`, `onnxruntime` | Cross-platform ONNX OCR; default non-macOS OCR |
| `paddle` | `paddleocr`, `paddlepaddle` | PP-OCRv5; highest accuracy, slower |
| `tesseract` | `pytesseract` | Requires system `tesseract` binary |
| `easyocr` | `easyocr` | Optional OCR engine |
| `yolo` | `ultralytics`, `torch` | UI element detection with user-supplied weights |
| `omniparser` | `ultralytics`, `torch`, `huggingface-hub` | OmniParser detection — **AGPL-3.0, opt-in** |
| `proxy` | `mitmproxy` | Headless HTTPS mock / record / replay (`aua proxy`, `aua mock`) |
| `lxml` | `lxml` | Faster XML parse for huge hierarchy dumps |
| `dev` | pytest, ruff, mypy, respx | Development and test tooling |
| `all` | All of the above | Full install |

Heavy deps are **lazy-imported** — a missing optional extra never breaks the core CLI.

---

## Connect a device or emulator

`aua` drives whatever `adb` can see. Use an emulator or a physical device — either works.

### Option A — Emulator (AVD)

Prefer an already-running device when one is attached. Otherwise `aua` can boot (and, for proxy work, create) an AVD:

```bash
aua emulator list                   # marks Play Store vs rootable Google APIs
aua emulator start --headless       # -no-window; Mac/Windows use -gpu host (not CPU SwiftShader)
aua --format compact analyze
# … drive the flow …
aua emulator stop --mine            # agents: ALWAYS stop AVDs you started
# Safety net: headless auto-stops after --idle-stop seconds of no aua activity (default 900)

# Parallel agents on one host (each gets its own serial; tear down only yours):
aua emulator start --headless --parallel --avd Pixel_7
# → {serial, port, owner, …}; then: aua --serial <serial> analyze …
aua emulator stop --serial <serial> # or: AUA_OWNER=<owner> aua emulator stop --mine
```

**HTTPS proxy / mock** needs a *rootable* Google APIs image — Google Play AVDs refuse `adb root`, so the mitm CA cannot be installed as a system trust and HTTPS recording stays empty:

```bash
aua emulator recommend-proxy        # suggests a small package (no download)
aua emulator ensure-proxy --start   # downloads google_apis image + boots aua_proxy
aua --serial <serial> proxy start   # needs: pip/uv install with [proxy]
aua emulator stop --mine
```

You can still create AVDs by hand (`sdkmanager` / `avdmanager` / Android Studio). Prefer SDK `cmdline-tools/latest` under `$ANDROID_HOME` over stale Homebrew copies. On Mac, headless defaults to **host GPU** so fans stay quiet; override with `--gpu swiftshader` only for CI without a display.

### Option B — Physical device

1. On the phone, enable **Developer options** (tap *Settings → About phone → Build number* seven times), then turn on **USB debugging**.
2. Connect over USB and accept the **"Allow USB debugging"** prompt.
3. `adb devices` should now list it with state `device`.

### First run

On the first command against a device, `uiautomator2` automatically pushes a small helper agent (the uiautomator/ATX server) to it — there's nothing to install by hand, but that first call is slower while it sets up. Verify the whole chain end-to-end:

```bash
adb devices     # device appears as "device" (not "unauthorized" or "offline")
aua doctor      # checks: adb on PATH · uiautomator2 importable · devices reachable · provider readiness
aua devices     # aua's own device listing (serial, model, Android version)
```

`aua doctor` is the single command to run whenever something isn't working — it pinpoints which prerequisite is missing. See [Troubleshooting](#troubleshooting).

---

## Working on `aua` itself

Dependencies are locked (`uv.lock`) and the interpreter is pinned (`.python-version`), so
everyone resolves the same versions:

```bash
# The test suite asserts apple-vision + rapidocr ARE available and paddle/easyocr are NOT
# (tests/test_ocr.py), so sync exactly that set — `--all-extras` fails the
# "provider unavailable" tests, and syncing none fails the "provider available" ones.
uv sync --extra dev --extra apple --extra rapidocr --extra yolo

uv run pytest          # test
uv run ruff check src  # lint
uv run aua analyze     # run the CLI from THIS tree
```

Use `uv run`, not a hand-rolled `.venv`. `uv run` resolves the project from the working
directory, so inside a **git worktree** you get that worktree's source. An editable install in
a shared `.venv` points at whichever checkout created it, which silently tests the wrong tree —
that has produced real false conclusions (a flag reported "missing" that was present all along).

Two related traps:

* **The daemon caches loaded modules.** After changing source, `aua daemon stop` — otherwise a
  warm daemon keeps serving the old code.
* **`.githooks/pre-commit` resolves `aua` from `PATH`**, which in a worktree is the *main*
  checkout, so it can regenerate `SKILL.md` from the wrong tree. Regenerate manually
  (`uv run aua guide --emit-skill <path>`) and commit with `--no-verify` if it interferes.

## Quickstart

```bash
# Check environment: adb, device, uiautomator2 agent, provider availability
aua doctor

# List attached devices
aua devices

# Analyze the current screen → Set-of-Marks JSON
aua analyze

# Compact format (fewer tokens, drops null/default fields — best for agents)
aua --format compact analyze

# Unchanged-screen / binary dumps (warm daemon + native host path)
aua --format delta analyze          # skip payload when hierarchy hash unchanged
aua --format msgpack analyze        # compact binary frame (AUA1)

# One element per line, tab-separated, status-bar noise already dropped — the
# readable view; pick your own columns with --fields
aua --format tsv analyze
aua --format tsv analyze --fields id,text,rid,checked

# Ask a narrower question instead of filtering the JSON yourself
aua --format tsv analyze --region 0,0,1080,300 --clickable   # just the header
aua --format tsv analyze --where-rid settingsSwitch --fields id,checkable,checked

# Is "Sign in" visible right now? Exit 0 = yes, 1 = no
aua has "Sign in"

# Wait on state (don't sleep) — including any hierarchy change
aua wait --for "Sign in"
aua wait --changed                  # any tree fingerprint change

# Act on elements by ID from the last analyze
aua tap 4
aua input 2 "hello@example.com"
aua swipe up

# Multi-emulator: same command on several serials
# aua fanout --serials emulator-5554,emulator-5556 analyze

# Force the vision fallback + write an annotated screenshot (numbered boxes)
aua analyze --source vision --annotate

# Find the best-matching element for a natural-language description
# Tries the hierarchy first; escalates to grounding only if needed
aua analyze --query "the Submit button"
aua analyze --query "the Submit button" --deep    # force grounding escalation
aua analyze --query "the Submit button" --cheap   # forbid escalation beyond hierarchy
```

The **analyze → act → analyze** loop is the core workflow:
1. `aua analyze` returns elements with IDs.
2. The agent picks an ID and acts: `aua tap <id>` / `aua input <id> "text"`.
3. No manual re-analyze needed — by default each action returns the next screen inline (`observation`, with fresh IDs), folding step 1 into step 2. Use `--no-observe` to skip it on action-only sequences, or a plain `analyze` (after `aua wait --for-stable`) when the screen is still animating.

---

## Use it from Claude Code (the `aua` skill)

`aua` ships a **Claude Code skill** that teaches Claude Code how to drive the tool and **auto-activates whenever you ask it to test or inspect an Android app** — no prompt engineering, the skill's description carries the trigger.

Two pieces make it work, and both must be available *outside* this repo:
- the **skill** — installed via the **plugin** or at **user level**, so it activates in *every* project (a skill merely committed under a repo's `.claude/skills/` only activates while Claude Code is working *inside that repo*);
- the **`aua` CLI** — on your `PATH` **globally** (not a project venv), because the skill shells out to it.

Pick either install path.

### Option 1 — Install the plugin (recommended)

This repo is its own Claude Code **plugin marketplace**. In Claude Code, run:

```
/plugin marketplace add The-Wordlab/Android-UI-Analyser
/plugin install android-ui-analyser@the-wordlab
```

That installs the skill so it auto-activates in every project. The plugin can't install the Python CLI it drives, so also install the `aua` binary once, **globally**:

```bash
uv tool install "git+ssh://git@github.com/The-Wordlab/Android-UI-Analyser.git"
# or:  pipx install "git+ssh://git@github.com/The-Wordlab/Android-UI-Analyser.git"
```

Core hierarchy analysis needs no extras; for the vision fallback on Compose/Flutter/WebView screens, add an OCR engine (see the [extras matrix](#optional-dependency-extras-matrix)). Pull plugin updates later with `/plugin update android-ui-analyser@the-wordlab`.

### Option 2 — One-command bootstrap from a clone

Prefer to do everything (binary **and** skill) in one shot? Clone and run the idempotent [`install.sh`](install.sh):

```bash
git clone git@github.com:The-Wordlab/Android-UI-Analyser.git
cd Android-UI-Analyser
./install.sh
```

It installs `aua` globally (via `uv tool`/`pipx`, with a venv fallback), installs the skill at **user level** (`~/.claude/skills/`), and runs `aua doctor`. The repo's [`CLAUDE.md`](CLAUDE.md) is auto-loaded when Claude Code opens the clone, so you can also just tell a fresh session:

> "Clone `git@github.com:The-Wordlab/Android-UI-Analyser.git` and run its `install.sh` to set up the `aua` Android UI testing skill, then use it to &lt;your task&gt;."

### Then: connect a device

Either path leaves one thing to do: attach a [device or emulator](#connect-a-device-or-emulator). Run `aua doctor` until `adb` and `devices` show OK — after that, in **any** project, just ask Claude Code to test your app and the skill activates automatically.

### Keeping the skill current

The SKILL.md is **generated** from the same source as `aua guide`, so it never drifts from the CLI. After upgrading `aua`: plugin users get the refreshed skill via `/plugin update`; user-level installs re-run `aua guide --emit-skill ~/.claude/skills/android-ui-analyser/SKILL.md`.

### Prefer MCP?

`aua mcp` exposes the same actions over MCP (see [MCP server](#mcp-server)) — use it for non–Claude-Code clients, or alongside the skill. It's intentionally **not** bundled in the plugin: it needs the `aua` binary already installed, so bundling it would make every session fail to start the server until you've installed the CLI.

---

## Escalation ladder (cost-aware routing)

`aua` starts at the cheapest tier that could answer the question and escalates only when that tier returns no confident result. No LLM is used to route — routing is pure heuristics.

| Tier | Method | Latency | Used for |
|---|---|---|---|
| T0 | Hierarchy text match (selector) | ~tens of ms | `aua has "text"` |
| T1 | Hierarchy selector locate | ~tens of ms | Tap/find a known element |
| T2 | Hierarchy full parse → element list | ~50–150 ms | `aua analyze` |
| T3 | Vision: detection + OCR (local) | ~150–600 ms | Compose/Flutter/WebView/canvas |
| T4 | Grounding VLM (local or commercial) | ~0.5–6 s | Fuzzy / visual / semantic targets |

Key rules:
- Default ceiling is **T3 (local vision)**. T4 grounding is entered only when `grounding.enabled: true` and `max_tier: grounding` (or `--deep`).
- The router **never silently escalates to a paid/commercial provider** — that tier must be explicitly enabled.
- `meta.tier_used` and `meta.providers_used` always report which tier ran.
- `--cheap` lowers the ceiling; `--deep` raises it for one call.

---

## App memory & navigation

`aua` remembers each app's layout **locally** as you use it — every `analyze` records the current screen and every state-changing action records a route between screens. No extra calls, and nothing leaves your machine (stored under `memory.dir`, default `~/.android-ui-analyser`).

Because of that, `analyze` hands navigation affordances back to you **inline**, so you rarely need a separate `aua map` call:

| `meta` field | What it gives you |
|---|---|
| `known_screen` | The recognised screen name on a revisit (flagged `stale` if its signature or the app version drifted, so you re-verify) |
| `known_routes` | Outgoing routes from here, e.g. `["tap 'Apps' → apps"]` |
| `suggested_gotos` | Ranked, ready-to-run targets, e.g. `["goto image_creator"]` — ordered by what you've navigated to recently |
| `research_tasks` | Open map-quality questions for the calling agent to investigate in source/runtime and submit through `reconcile` |
| `map_hint` | A nudge like `"12 screens mapped — run aua map"` when there's a map but nothing actionable from the current screen |

### Jump to a known screen in one command

```bash
aua goto "image creator"      # replay the remembered steps: selector-matched, verified per hop
aua goto "settings" --plan    # print the annotated route (steps/replayable/destructive), don't act
aua goto "onboarding"         # cross-app auth legs (Google sign-in via Chrome) replay too
aua goto "login" --allow-destructive   # required when a step matches memory.destructive_labels
```

`goto` resolves the goal (fuzzy) against the map, walks the shortest **verified** route from the current screen, and replays each edge's recorded steps — matching by resource-id first, then label — re-checking `known_screen` after every hop. A newly auto-observed edge is provisional until the same transition is observed again (a passive observation landing on an already-known screen verifies it immediately); selectorless edges are retained for audit but rejected from navigation. Auth excursions through `memory.transit_packages` (Google sign-in in Chrome/GMS, permission dialogs) are recorded as **one edge on the origin app** and replay end-to-end; a step whose identity was redacted (e.g. an account row containing an email) hands off for one manual tap — then just re-run `goto`: it resumes mid-route, even mid-auth. Destructive steps (delete / sign out / pay / …) are refused without `--allow-destructive`. On divergence it stops and hands back the failing step, the remaining steps, the current screen, and its elements (exit `1`); it exits `0` on arrival, returning the destination's `elements` (fresh ids). It runs through the warm daemon too.

### Replay whole journeys in one call (flows)

A **flow** is a Maestro-style YAML journey — authored directly by you/your agent, or materialized from what you just did. The repeated setup path to the screen under test (reset account → log back in → reach onboarding) becomes one command:

```bash
aua flow run reset_account_google_login --param ACCOUNT="Engineering Team"
aua flow run smoke --dry-run          # print the resolved steps, act on nothing
aua flow run smoke --from-step 4      # resume after fixing a divergence
aua flow save reach_checkout --last 8 # materialize your recent actions into YAML
aua flow list · aua flow show <name> · aua flow delete <name>
```

Flows live flat under `<memory.dir>/flows/<name>.yaml` (they span packages by design — the auth leg is the point). Step vocabulary: `launch_app`, `tap` (by `id:` tail or `text:`), `input` (with `${PARAM}` substitution), `key`, `swipe`, `scroll_to`, `wait_for`, `wait_stable`, `assert_visible`, and `goto:` to compose map navigation. `flow save` never persists typed values — inputs become required `${PARAM_n}` placeholders you fill in the file. Flows are deliberate authored intent, so destructive steps run by default (`--no-allow-destructive` opts back into the guard). On divergence you get the failing step index, the remaining steps, and the current elements — fix, then `--from-step N`.

```yaml
# ~/.android-ui-analyser/flows/reset_account_google_login.yaml
name: reset_account_google_login
app: com.example.app.dev
params:
  ACCOUNT: "Engineering Team"
steps:
  - launch_app: com.example.app.dev
  - tap: {id: buttonSettings}          # unlabeled gear — id-tail selector
  - tap: "Account & Data"
  - tap: "Delete my account"
  - tap: "Delete"
  - wait_for: {text: "Continue with Google", timeout_ms: 15000}
  - tap: "Continue with Google"
  - tap: {text: "${ACCOUNT}", package: com.android.chrome}
  - tap: {text: "Continue", package: com.android.chrome}
  - wait_stable
```

### Inspect and manage the map

```bash
aua map                       # learned screens + routes for the current app
aua map --find "image"        # just the route to a target
aua map --context flags-catalog_experiment-…  # one verified feature-flag context
aua map --all-contexts        # compare variants across contexts
aua map --audit               # ambiguities + questions for a research agent
aua memory show|path|update|forget
aua memory update --screen login   # rename a badly-auto-named screen
```

Memory schema v4 treats a feature-flag set as part of UI identity. `flags set/apply`
activates a deterministic context after the app restarts. When `flags.prefs_files` or
`flags.context_keys` is configured, every `analyze` also reads the app's already-active,
privacy-filtered experiment/treatment/variant/flag values and switches context before
recording. Exact-context routes outrank compatible legacy routes; v1/v2/v3 maps migrate
without losing trusted edges. Stable resource namespaces produce locale-independent names,
and the map groups flag variants plus loading/error/empty/ready states under one logical
destination instead of filling the output with numeric/hash suffixes.

### Agent feedback and self-correction

Knowledge learned by a person, runtime probe, source inspection, or external agent can be
stored with provenance and context:

```bash
aua knowledge add --app com.example.app --kind claim \
  --text "Catalog uses catalogTabTOOLS under catalog_experiment=a" \
  --source agent --agent codex --evidence features/apps-hub/…
aua knowledge list --app com.example.app
aua knowledge stale <knowledge-id> --app com.example.app
```

The correction loop deliberately uses an external-agent contract: AUA automatically
materializes tasks when it sees weak names, stale/duplicate screens, unverified contexts,
provisional routes, or unreplayable/conflicting edges. It surfaces those tasks in
`analyze.meta.research_tasks` and `MAP.md`; the calling agent researches source/runtime and
feeds evidence back. AUA applies validated changes but does not choose or spawn a model.

```bash
aua reconcile plan --app com.example.app > tasks.json
# An external agent researches source/runtime using package, version, flags, and questions.
aua reconcile submit --app com.example.app report.json
aua reconcile status --app com.example.app
aua reconcile rollback --app com.example.app <rollback-id>
```

A report has `verdict: apply|review|reject`, evidence, knowledge, and typed operations
(`rename`, `alias`, `merge`, `split`, variant/state/context changes, route guards/replacement/
verification/rejection/deletion, knowledge upsert, or stale marking). `apply` is autonomous: AUA modifies a deep
copy, validates stable IDs and route references, snapshots the old map, commits atomically,
and returns a rollback ID. `review` is queued without mutation and `reject` is retained as
feedback. The same audit, reconciliation, and knowledge operations are available as MCP
tools.

**Privacy:** only the durable skeleton is stored (screen names, routes, stable elements). Dynamic lists are kept as a *shape*, and `EditText` values / secrets / PII are redacted (`<filled>` / `<redacted>`) — which is also why auto-learned edges never contain typed text, and why an account row may need one manual tap during replay.

### Optional: LLM assist (off by default)

Everything above is fully deterministic — no model runs during navigation. If you opt in, a **fast planner LLM** (default Gemini Flash Lite) plugs into the *edges* of that core to cut agent iterations further. It's off unless you enable it **and** ask for it:

```yaml
planner:
  enabled: true            # off by default; also needs the model's API key (GEMINI_API_KEY)
  chain: [gemini_flash]
```

```bash
aua goto "onboarding" --assist      # on a divergence, the planner tries to recover in-call
aua flow run reset --assist         # dismiss an unexpected popup mid-flow, then resume
aua navigate "open the image generator"          # drive to a goal with NO prior map…
aua navigate "reach checkout" --save-flow checkout   # …and record it as a reusable flow
```

Two roles: (1) **recovery** — when `goto`/`flow` diverge (an "Allow notifications?" popup, an A/B modal, a moved button), `--assist` lets the planner clear the blocker and continue in the same call instead of handing the whole screen back to you; the un-assisted handoff hint tells you when it's worth adding. (2) **`aua navigate "<goal>"`** — autonomous, map-free navigation that **records the path into memory**, so the next `aua goto <that screen>` is a free, deterministic replay (the expensive run happens once). The planner reads the compact element list (a cheap text call; a screenshot is attached only on unlabeled screens), may only act on an on-screen element, is bounded by `--max-steps`, and its taps still obey the destructive guard — so it never wanders into Delete/Pay without `--allow-destructive`.

### Tuning

```yaml
memory:
  suggest: true             # push known routes/gotos/map hints into analyze
  suggest_max: 4            # cap on suggested_gotos per analyze
  rank_half_life_days: 3.0  # recency decay for usage-based ranking
  auto_research: true       # create/surface research tasks for map uncertainty
  research_suggest_max: 3

flags:
  auto_context: true
  prefs_files:
    com.example.app.dev: "flag_overrides.xml"
  # Optional exact allow-list; otherwise only flag-like key names survive.
  context_keys:
    com.example.app.dev: [catalog_experiment, services_treatment]
```

---

## Configuration

### Precedence (highest to lowest)

1. Individual CLI flags (e.g. `--serial`, `--format`, `--no-cache`)
2. `--config <path>` explicit config file
3. Environment variables (`AUA_*` and provider key vars like `OPENAI_API_KEY`)
4. `--profile <name>` overlay (deep-merged over the base config)
5. Project config: nearest `.android-ui-analyser.yaml` walking up from CWD
6. User config: `$XDG_CONFIG_HOME/android-ui-analyser/config.yaml` (default `~/.config/...`)
7. Built-in defaults

### Secrets

**Secrets are never stored in config.** The config references the env-var **name** (`api_key_env: OPENAI_API_KEY`); the tool reads the value at runtime. `aua config show` and `aua doctor` never print secret values.

For convenience, keep a gitignored `.env` file and source it before running `aua`:

```bash
echo "GEMINI_API_KEY=..." >> .env
echo ".env" >> .gitignore
source .env
```

### Example config

```yaml
# .android-ui-analyser.yaml  (or aua config init to write the full commented version)
ocr:
  chain: [apple_vision, rapidocr]   # macOS: apple first; cross-platform: just [rapidocr]

grounding:
  enabled: true
  chain: [gemini]

models:
  gemini: { model: gemini-2.5-flash, api_key_env: GEMINI_API_KEY }
```

Swap a model with **one line**: change `ocr.chain: [apple_vision, rapidocr]` to `[rapidocr]`, or `grounding.chain: [gemini]` to `[openai]`.

### Profiles

```yaml
profiles:
  cloud:
    grounding: { enabled: true, chain: [gemini] }
  local:
    grounding: { enabled: false }
```

Activate with `aua --profile cloud analyze --query "the Submit button"`.

### Config commands

```bash
aua config init          # write commented default config to the user config path
aua config show          # print the current config (secrets masked)
aua config show --effective  # print after all precedence layers are merged
aua config path          # print the resolved config file path
```

---

## Provider / license matrix

### OCR

| Provider | Extra | Platform | License | Notes |
|---|---|---|---|---|
| `apple_vision` | `apple` | **macOS only** | Proprietary (native) | Fastest OCR on Mac (~100–500 ms); Neural Engine |
| `rapidocr` | `rapidocr` | Cross-platform | Apache-2.0 | Default non-macOS OCR; ONNX PaddleOCR |
| `paddleocr` | `paddle` | Cross-platform | Apache-2.0 | PP-OCRv5; highest accuracy |
| `tesseract` | `tesseract` | Cross-platform | Apache-2.0 | Requires system binary |
| `easyocr` | `easyocr` | Cross-platform | Apache-2.0 | — |

### Detection

| Provider | Extra | License | Notes |
|---|---|---|---|
| `yolo` | `yolo` | Apache-2.0 (Ultralytics) | User-supplied weights path; **license-clean default** |
| `omniparser` | `omniparser` | **AGPL-3.0** | OmniParser v2 detection-only; requires `accept_agpl: true` in config; CVE-2025-55322 patched in ≥2.0.1 — **never expose the OmniTool server** |

### Grounding (all opt-in, `grounding.enabled: false` by default)

| Provider | License | Config key | Notes |
|---|---|---|---|
| `local_vllm` | Apache-2.0 (Holo1.5-7B) | `local_vllm` | OpenAI-compatible endpoint; e.g. vLLM, Ollama, LM Studio |
| `openai` | Commercial | `openai` | GPT-class vision; key via `OPENAI_API_KEY` |
| `anthropic` | Commercial | `anthropic` | Claude vision; key via `ANTHROPIC_API_KEY` |
| `gemini` | Commercial | `gemini` | Gemini vision; key via `GEMINI_API_KEY` |

### Planner (opt-in, `planner.enabled: false` by default)

The fast LLM behind `--assist` and `aua navigate` (see [Optional: LLM assist](#optional-llm-assist-off-by-default)). Text-first (the element list rides in the prompt; a screenshot is attached only on unlabeled screens).

| Provider | License | Config key | Notes |
|---|---|---|---|
| `gemini_flash` | Commercial | `gemini_flash` | Default; Gemini Flash Lite, key via `GEMINI_API_KEY`. Swap the `planner.chain` for any provider you prefer. |

**Default config is commercially licensable.** No AGPL or research-only component is active out of the box. OmniParser requires explicit `accept_agpl: true`; grounding and the planner are off until you enable them.

---

## Adding a provider

Three steps, zero changes to `engine.py` or `cli.py`:

1. **Subclass** the relevant abstract base from `providers/base.py`:
   - `OcrProvider` — implement `recognize(image) -> list[TextBox]`
   - `DetectionProvider` — implement `detect(image) -> list[Box]`
   - `GroundingProvider` — implement `locate(image, instruction) -> Point|Box`

2. **Register** with the decorator from `providers/registry.py`:
   ```python
   from android_ui_analyser.providers.registry import register_ocr

   @register_ocr("my_ocr")
   class MyOcrProvider(OcrProvider):
       ...
   ```

3. **Add a `models.my_ocr` block** in your config:
   ```yaml
   models:
     my_ocr: { some_option: value }
   ```

Each provider must implement `is_available() -> tuple[bool, str]` to declare whether its dependencies and credentials are present.

---

## Daemon mode

The daemon holds a warm `uiautomator2` connection and loaded vision models, eliminating per-call cold-start overhead. The CLI auto-detects a running daemon via a unix socket and forwards requests to it; without a daemon it runs in-process (always correct, pays startup cost).

```bash
aua daemon start          # start the background daemon (+ the app orientation blob)
aua daemon start --quiet  # start it without the blob
aua daemon status         # check if running
aua daemon stop           # stop the daemon
aua orient                # the orientation blob on demand, any time
```

`daemon start` prints what the tool already knows about the foreground app — description,
screens, routes, mined deeplinks, login recipes, quirks. That is genuinely useful the first
time in a session and pure noise on every restart afterwards, so `--quiet` suppresses it and
`aua orient` prints it whenever you actually want it.

The daemon binds **only to a unix socket** (default `~/.cache/android-ui-analyser/daemon.sock`;
per-serial sockets when multiple devices are in play). No TCP port, no auth surface.

Optional: set `daemon.push_ws_port` in config to push `screen_changed` events over a
localhost WebSocket so agents can wait without polling. Unchanged screens short-circuit via
`meta.via=hierarchy-unchanged` / `--format delta` when `perf.skip_unchanged_analyze` is on.

### Optional: `aua-fast` (C thin client)

Once the daemon is up, each `aua …` call still pays Python/typer startup. `aua-fast` is a
~35 KB binary that speaks the same unix-socket JSON protocol and falls back to `aua` if the
daemon is down:

```bash
make -C native/aua-fast install   # → ~/.local/bin/aua-fast (also built by ./install.sh)
aua daemon start --quiet
aua-fast analyze
aua-fast tap 4
aua-fast has "Sign in"            # exit 0/1
```

Hot commands: `ping`, `analyze`, `devices`, `has`, `tap`, `key`, `input`, `swipe`, `wait`.
Everything else (and any unknown flags) exec's the full Python CLI. Host-side speedups
(incremental analyze, delta/msgpack, WS push, fanout, vision defaults) are summarized in
[`docs/NATIVE_ROADMAP.md`](docs/NATIVE_ROADMAP.md).

---

## Proxy / mock (HTTPS record & replay)

Optional extra: `pip`/`uv` install with `[proxy]` (pulls `mitmproxy`). Apps whose Network
Security Config trusts **system CAs only** need a rootable emulator and a system CA install
(see [Emulator](#option-a--emulator-avd)):

```bash
aua emulator ensure-proxy --start
aua --serial <serial> proxy start          # random high port + system CA by default
aua mock record start login_flow
# … drive the app …
aua mock record stop login_flow            # cassette under memory.dir/cassettes/
aua mock map GET /v1/foo --status 200 --body '{"ok":true}'
aua mock replay login_flow
aua proxy stop
aua emulator stop --mine
```

Empty cassettes almost always mean TLS rejected the forged cert (Play Store AVD / no system
CA) — not “no traffic”.

---

## Dashboard (sneak-peek headless runs)

Agents often drive a **headless** emulator with no window. To watch live without interrupting
them, run a separate process:

```bash
# One agent / one device:
aua dashboard --serial emulator-5554
# → opens http://127.0.0.1:8765  (enables capture via daemon or sidecar)

# Multiple parallel agents (auto grid of live screens):
aua dashboard            # several emulators online → tile grid
aua dashboard --grid     # force grid even with one device
# Click a tile → detail (journal / map / logcat) for that serial
```

The page live-polls capture frames + recent action marks. Ctrl-C stops only the dashboard.
If no warm daemon is present, aua starts the capture **sidecar** (single-device) or uses adb
screencap per tile (grid).

---

## MCP server

`aua mcp` runs an MCP server over stdio, exposing the same tools as the CLI. It is a thin adapter over the engine — no separate perception logic.

Tools include (non-exhaustive): `analyze_screen`, `tap`, `double_tap`, `long_press`, `input`,
`clear`, `swipe`, `scroll`, `scroll_to`, `key`, `wait`, `wait_stable`, `wait_changed`, `has`,
`expect`, `screenshot`, `inspect`, `goto`, `flow_run`, `navigate`, `list_devices`,
`emulator_list` / `emulator_status` / `emulator_start` / `emulator_stop` (stop before exit —
MCP also auto-stops emulators it started when the server process ends), `open_link`,
`app`, `resolve`, clipboard/paste/copy/erase, location/orientation/airplane/media/record/clock,
`capture_*`, `dev_profile`, `a11y_scroll`, `flags_apply`, map/`reconcile_*`/`knowledge_*`,
`proxy_start` / `proxy_stop` / `mock_replay`, `configure`.

Example MCP client config (Claude Desktop / `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "android-ui": {
      "command": "aua",
      "args": ["mcp"]
    }
  }
}
```

---

## CLAUDE.md snippet

> Most users should instead install the **skill** (see [Use it from Claude Code](#use-it-from-claude-code-the-aua-skill)) — it activates automatically and stays in sync with the CLI. Use the snippet below only if you prefer inline per-project instructions over the skill.

Paste this into your project's `CLAUDE.md` to teach Claude Code how to use `aua`:

```markdown
## Android UI testing with `aua`

Use `aua` to inspect and drive the connected Android device.

### Getting elements on screen

```bash
aua --format tsv analyze       # readable: one element per line, noise filtered out
aua --format compact analyze   # get element IDs (smaller token footprint)
aua analyze                    # full JSON with all fields
```

Elements are returned with stable integer IDs. Always re-analyze after a
state-changing action — IDs may change after navigation or screen transitions.

### Asking for fewer rows and columns (don't post-process the JSON)

Most of a screen is status-bar chrome and unlabelled containers, and you usually want
four columns out of eighteen. Say so in the same call:

```bash
aua --format tsv analyze                                  # id, text, rid, clickable
aua --format tsv analyze --fields id,text,rid,bounds      # your columns, your order
aua --format tsv analyze --all                            # keep the noise too
aua --format tsv analyze --region 0,0,1080,300 --clickable # the header's tap targets
aua --format tsv analyze --where-text "Browse"            # case-insensitive substring
aua --format tsv analyze --where-rid homeTab --limit 5
aua --format tsv analyze --no-meta                        # no `#` comment lines at all
aua --format compact analyze --fields id,rid --nonempty   # same views, JSON output
```

- `--fields` names: `id`, `type`, `text`, `rid` (short tail) / `resource_id` (full selector),
  `desc` / `content_desc`, `bounds`, `center`, `clickable`, `enabled`, `focused`, `checkable`,
  `checked`, `selected`, `scrollable`, `long_clickable`, `password`, `source`, `confidence`.
  A wrong name exits **2** and lists the valid ones — before touching the device.
- `--format tsv` implies `--nonempty --no-system` (drops rows with no text/id/desc, and
  status-bar chrome). `--all` opts out. JSON formats filter **nothing** unless you ask.
- `--no-wrappers` (opt-in) drops the app's **own** id'd layout scaffolding — `app_bar`,
  `content_frame`, `collapsing_toolbar`: nodes that are unlabelled, non-actionable, and wrap
  something else. Leaves stay (an unlabelled leaf is the icon you tap), and so do actionable
  or scrollable containers. Worth it on View-based apps, where such wrappers can be most of
  the top of the list; Compose apps barely notice. `--nonempty` does **not** drop them —
  they do have a resource-id, which is exactly what makes them addressable.
- Filters of different kinds AND together; repeating one kind ORs together
  (`--region A --region B` = either box).
- **IDs are never renumbered.** The id in a filtered row is the id `aua tap` takes.
- `--meta <csv>` / `--no-meta` control the metadata (the route/deeplink suggestions are
  worth reading once per session, not on every call).

### Reading interaction state (is that switch on?)

Every element carries the a11y interaction flags, so a boolean question needs no screenshot:

```bash
$ aua --format tsv analyze --where-rid switch_widget --fields id,type,checkable,checked
# screen=accessibility package=com.android.settings 1080x2400
# elements=24 shown=2 tier_used=hierarchy duration_ms=220
id	type	checkable	checked
16	Switch	true	true
19	Switch	true	false
```

`selected` tells you which tab is active; `scrollable` tells you which container really
scrolls. These are **tri-state**: `true`/`false` when the node reported the attribute, and
**empty/`null` when genuinely unknown** (a vision-derived element has no a11y attributes),
so *off* never masquerades as *unknown*.

### Acting on elements

```bash
aua tap <id>              # tap / click an element (returns the next screen by default)
aua input <id> "text"     # focus element and type; add --submit to send IME action
aua tap <id> --no-observe # act WITHOUT returning the new screen (skip the folded analyze)
aua swipe up              # swipe direction (up|down|left|right)
aua swipe --from <id>     # scroll a specific container
```

By default every action returns the post-action screen inline (`observation`, with fresh
ids), so `type → tap send` is two commands, not three. Add `--no-observe` to skip it.

### Quick checks (no full analyze needed)

```bash
aua has "Sign in"         # exit 0 if present, 1 if not — use to branch cheaply
aua has "Submit" --match exact
```

Prefer `aua has` over re-analyzing when you only need to confirm presence.

### Handling Compose / Flutter / WebView / game screens

When `analyze` returns few or no elements, the hierarchy is empty (common for
Compose without semantics, Flutter, or canvas apps). Use:

```bash
aua analyze --source vision --annotate
```

This runs detection + OCR and writes an annotated PNG (numbered boxes) to the
path shown in `meta.annotated_image`. Element IDs work the same way.

### When you must actually LOOK at the screen

Reading a full 1080x2400 PNG is expensive. Capture only the part you need:

```bash
aua screenshot --out /tmp/header.png --region 0,0,1080,300   # crop before writing
aua screenshot --out /tmp/small.png  --max-width 320         # downscale (never upscales)
aua screenshot --out /tmp/half.png   --scale 0.5
aua screenshot --annotate                                    # full screen + numbered marks
```

The written path comes back as `detail` (`aua screenshot … | jq -r .detail`). Regions are
screen pixels and are clamped to the screen; a box that misses it entirely exits **2**.
`--annotate` can't be combined with cropping — marks are placed in full-screen coordinates.

### Semantic / fuzzy target lookup

```bash
aua analyze --query "the Submit button"      # tries hierarchy first, grounding only if needed
aua analyze --query "the blue icon top-right" --deep   # force grounding escalation
```

### Navigating to known screens

`aua` remembers each app's screens and routes. Every `analyze` returns
`meta.suggested_gotos` (ranked, ready-to-run) and `meta.known_routes`. To jump to
a screen the tool has seen before, in one command:

```bash
aua goto "image creator"   # taps + verifies each hop along the remembered route
aua goto "settings" --plan # preview the route without acting
```

Prefer `goto` over manual tapping whenever your target is listed in `suggested_gotos`.

### Typical loop

1. `aua --format compact analyze` → read element IDs from JSON output.
2. `aua tap <id>` or `aua input <id> "text"`.
3. `aua has "<expected text>"` to confirm the transition.
4. `aua --format compact analyze` again for the new screen.
```

---

## Output schema summary

```json
{
  "schema_version": 1,
  "screen": { "width": 1080, "height": 2400, "package": "com.example", "activity": ".Main", "source": "hierarchy" },
  "elements": [
    {
      "id": 0,
      "type": "Button",
      "text": "Sign in",
      "resource_id": "com.example:id/sign_in",
      "content_desc": null,
      "bounds": [120, 1500, 960, 1610],
      "center": [540, 1555],
      "clickable": true,
      "enabled": true,
      "focused": false,
      "source": "hierarchy",
      "confidence": null
    }
  ],
  "meta": {
    "duration_ms": 42,
    "tier_used": "hierarchy",
    "path": "hierarchy",
    "providers_used": ["hierarchy"],
    "known_screen": "home",
    "known_routes": ["tap 'Apps' → apps"],
    "suggested_gotos": ["goto image_creator"],
    "map_hint": null,
    "annotated_image": null,
    "device_serial": "emulator-5554"
  }
}
```

`compact` format drops null fields and verbose defaults for the smallest token footprint.
`pretty` is indented JSON. `tsv` is one element per line. `delta` / `msgpack` are for warm-
daemon / native hot paths (unchanged hierarchy → tiny payload). All formats validate against
the same pydantic schema where applicable.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Negative result — `has`: text absent, or `goto`: route could not complete |
| `2` | Usage error (bad flags, missing argument) |
| `3` | No device / device error |
| `4` | Provider error — all fallbacks exhausted |
| `5` | Config error (invalid YAML, unknown key, bad value) |

Errors print a structured object to stderr: `{"error": {"code": ..., "message": ..., "hint": ...}}`.

---

## Command reference

Run `aua --help`, or `aua <command> --help` for any command. Global flags (`--format`, `--serial`, `--config`, `--profile`, `--timeout`, `--log-level`, `--no-cache`, `--with-image`) go **before** the subcommand.

| Command | What it does |
|---|---|
| `aua doctor` | Check environment: adb, uiautomator2, devices, provider readiness, emulator tooling |
| `aua devices` | List attached devices/emulators |
| `aua analyze` | Capture the screen → element list with IDs (the core command) |
| `aua resolve <id\|key>` | Remap a prior id / `stable_key` onto the current screen |
| `aua has "<text>"` | Exit 0 if text is on screen, 1 if not — cheap branch check |
| `aua expect …` | Assert visibility / state (exit 8 on failure) |
| `aua wait --for "<text>"` | Poll until text appears (`--idle` / `--for-stable` / `--changed`) |
| `aua fanout …` | Run a command across multiple `--serials` |
| `aua tap <id>` / `aua click <id>` | Tap an element by ID (also `--rid`/`--text`/`--desc`) |
| `aua double-tap <id>` | Double-tap an element |
| `aua long-press <id>` | Long-press an element by ID |
| `aua input <id> "text"` | Focus an element and type (`--submit` fires the IME action) |
| `aua clear <id>` / `aua erase` | Clear a field / backspace N chars |
| `aua hide-keyboard` | Dismiss the IME without navigating away |
| `aua swipe <up\|down\|left\|right>` | Swipe / scroll (`--from <id>` to scroll a container) |
| `aua scroll` / `aua scroll-to "<text>"` | Directional scroll / scroll until text is found |
| `aua key <back\|home\|enter\|…>` | Press a hardware/navigation key |
| `aua open <uri> [--app pkg]` | Open a deeplink (pin package to skip "Open with…") |
| `aua clipboard set\|get` / `paste` / `copy` | Clipboard helpers |
| `aua location set LAT,LON` | Mock GPS |
| `aua orientation set\|get` | Screen orientation |
| `aua airplane on\|off\|toggle` | Airplane mode |
| `aua media add PATH` | Push media into the gallery |
| `aua record start\|stop PATH` | Screen recording |
| `aua clock set --ms <unix-ms>` | Set device clock (emulator / rooted) |
| `aua screenshot [path]` | Save a raw screenshot (`--region` / `--scale`) |
| `aua inspect <id>` | Dump full details for one element |
| `aua app launch\|stop\|kill\|clear\|grant` | App control (`launch --clear` = clearState) |
| `aua emulator list\|status\|start\|stop` | Boot/stop AVDs (`--headless`, `--parallel`, `--gpu`, `--mine`/`--owner`) |
| `aua emulator recommend-proxy\|ensure-proxy` | Suggest/create a small rootable Google APIs AVD |
| `aua flags set\|apply` | Feature-flag writes with verify/restart |
| `aua proxy start\|stop` / `aua mock …` | HTTPS mitm record/map/replay (`[proxy]` extra) |
| `aua capture …` | Session capture / export / explain |
| `aua dashboard` | Live browser sneak-peek (`--grid` for multi-agent tiles; works with headless) |
| `aua logcat` / `aua suite` | Device-clock log windows / scripted suites |
| `aua dev` / `aua a11y` | Dev options helpers / a11y scroll |
| `aua map` | Show the active-context map (`--all-contexts`, `--audit`, or `--find "<goal>"`) |
| `aua goto "<goal>"` | Drive the remembered route to a known screen — taps + verifies each hop (`--plan` previews, `--max-steps N`) |
| `aua flow run\|save\|list\|…` | Maestro-style YAML journeys |
| `aua navigate "<goal>"` | Opt-in planner drive (needs `planner.enabled`) |
| `aua memory show\|path\|update\|forget` | Manage the per-app learned layout (`memory.backend: sqlite` optional) |
| `aua knowledge list\|show\|add\|stale` | Manage scoped, provenance-bearing learned facts |
| `aua reconcile plan\|submit\|status\|apply\|rollback` | Research and transactionally correct a map |
| `aua about` / `aua remember` / `aua orient` | App playbook + orientation blob |
| `aua config init\|show\|path` | Manage configuration |
| `aua daemon start\|status\|stop` | Manage the optional warm-state daemon |
| `aua guide` | Print the agent operating manual (`--emit-skill` writes the Claude Code skill) |
| `aua mcp` | Run the MCP server over stdio |

All action commands (`tap`, `long-press`, `input`, `clear`, `swipe`, `scroll-to`, `key`, …) **return the post-action screen inline by default** (an `observation` with fresh element IDs), so you rarely need a follow-up `analyze`. Pass **`--no-observe`** to skip it. Prefer **`aua --format tsv analyze`** when reading a screen; use global **`aua --with-image …`** only when you must see pixels.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aua doctor` shows **adb: FAIL / not found on PATH** | Install platform-tools and add them to `PATH` — see [Installing adb](#installing-adb-platform-tools). |
| `no device found` (exit 3) | Start an emulator or attach a phone; confirm with `adb devices`. Or `aua emulator start --headless`. |
| Device shows as **`unauthorized`** | Accept the "Allow USB debugging" prompt on the device. If it never appears: `adb kill-server && adb start-server`, then reconnect. |
| Device shows as **`offline`** | Re-plug the cable / cold-boot the emulator; `adb reconnect`. |
| `multiple devices attached` | Pass `--serial <id>` (get the id from `aua devices`). |
| First command is slow / times out | `uiautomator2` is pushing its helper agent on first connect — retry once it settles, then use `aua daemon start` to keep the connection warm. |
| `uiautomator2 is not installed` | Reinstall the package — `uiautomator2` is a base dependency, not an extra. |
| `analyze` returns few/no elements | The hierarchy is empty (Compose/Flutter/WebView/canvas). Force vision: `aua --format compact analyze --source vision --annotate`. |
| Typing does nothing on Android 14+ | Handled automatically (accessibility `set_text` on the focused field); make sure the field is actually focused first. |
| Headless emulator pegs CPU / fans | Old default was SwiftShader (CPU). Current Mac/Windows headless uses `-gpu host`. Stop orphans with `aua emulator stop --mine`. Headless also auto-stops after `--idle-stop` (default 900s) with no aua activity. |
| `proxy` / empty HTTPS cassettes | Need `[proxy]` extra + **rootable** Google APIs AVD (`aua emulator ensure-proxy`). Play Store images refuse `adb root` → system CA install fails → TLS handshake fails. |
| `sdkmanager` / `ensure-proxy` fails | Prefer `$ANDROID_HOME/cmdline-tools/latest/bin` over outdated Homebrew cmdline-tools. |

---

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions and the hierarchy-first thesis.
- [`docs/NATIVE_ROADMAP.md`](docs/NATIVE_ROADMAP.md) — `aua-fast`, delta/msgpack, WS push, fanout, vision defaults.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — landscape research behind the approach.
- [`PRD.md`](PRD.md) — the full product requirements document.
- [`SMOKE.md`](SMOKE.md) — manual smoke-test checklist against a live device.
- [`.claude/skills/android-ui-analyser/SKILL.md`](.claude/skills/android-ui-analyser/SKILL.md) — the operating manual an AI agent loads (also via `aua guide`).