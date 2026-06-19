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

Global install via pipx (no extras):

```bash
pipx install .
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
| `dev` | pytest, ruff, mypy, respx | Development and test tooling |
| `all` | All of the above | Full install |

Heavy deps are **lazy-imported** — a missing optional extra never breaks the core CLI.

---

## Connect a device or emulator

`aua` drives whatever `adb` can see. Use an emulator or a physical device — either works.

### Option A — Emulator (AVD)

With Android Studio (or the standalone command-line tools) installed, you can create and boot an emulator from the terminal:

```bash
emulator -list-avds                 # list existing AVDs

# No AVD yet? Create one. The system image is installed via Android Studio's SDK Manager
# or:  sdkmanager "system-images;android-34;google_apis;arm64-v8a"
avdmanager create avd -n pixel7 -k "system-images;android-34;google_apis;arm64-v8a" -d pixel_7

emulator -avd pixel7                # boot it
```

`emulator`, `avdmanager`, and `sdkmanager` live under `~/Library/Android/sdk/emulator` and `~/Library/Android/sdk/cmdline-tools/latest/bin` (macOS) — add those to your `PATH` too if you want them globally.

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

# Is "Sign in" visible right now? Exit 0 = yes, 1 = no
aua has "Sign in"

# Act on elements by ID from the last analyze
aua tap 4
aua input 2 "hello@example.com"
aua swipe up

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
3. Re-analyze — IDs may change after a state transition. Or add `--observe` to the action (e.g. `aua tap <id> --observe`) to get the post-action screen back inline, folding the re-analyze into the same call.

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
| `map_hint` | A nudge like `"12 screens mapped — run aua map"` when there's a map but nothing actionable from the current screen |

### Jump to a known screen in one command

```bash
aua goto "image creator"      # drive the remembered route: taps + verifies each hop
aua goto "settings" --plan    # print the route only, don't act
aua goto "checkout" --max-steps 12
```

`goto` resolves the goal (fuzzy) against the map, walks the shortest route from the **current** screen, and re-checks `known_screen` after every hop. If the route diverges it stops and hands back the remaining steps, the current screen, and the on-screen elements (exit `1`); it exits `0` once it arrives, returning the destination's `elements` (fresh ids). Either way you can keep going without a separate `analyze`. It runs through the warm daemon too.

### Inspect and manage the map

```bash
aua map                       # learned screens + routes for the current app
aua map --find "image"        # just the route to a target
aua memory show|path|update|forget
```

**Privacy:** only the durable skeleton is stored (screen names, routes, stable elements). Dynamic lists are kept as a *shape*, and `EditText` values / secrets / PII are redacted (`<filled>` / `<redacted>`).

### Tuning

```yaml
memory:
  suggest: true             # push known_routes/suggested_gotos/map_hint into analyze
  suggest_max: 4            # cap on suggested_gotos per analyze
  rank_half_life_days: 3.0  # recency decay for usage-based ranking
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

**Default config is commercially licensable.** No AGPL or research-only component is active out of the box. OmniParser requires explicit `accept_agpl: true`.

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
aua daemon start    # start the background daemon
aua daemon status   # check if running
aua daemon stop     # stop the daemon
```

The daemon binds **only to a unix socket** (default `~/.cache/android-ui-analyser/daemon.sock`). No TCP port, no auth surface.

---

## MCP server

`aua mcp` runs an MCP server over stdio, exposing the same tools as the CLI. It is a thin adapter over the engine — no separate perception logic.

Tools exposed: `analyze_screen`, `tap`, `input`, `swipe`, `key`, `wait`, `has`, `screenshot`, `list_devices`.

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
aua --format compact analyze   # get element IDs (smaller token footprint)
aua analyze                    # full JSON with all fields
```

Elements are returned with stable integer IDs. Always re-analyze after a
state-changing action — IDs may change after navigation or screen transitions.

### Acting on elements

```bash
aua tap <id>              # tap / click an element
aua input <id> "text"     # focus element and type; add --submit to send IME action
aua tap <id> --observe    # act AND return the new screen inline (skips a follow-up analyze)
aua swipe up              # swipe direction (up|down|left|right)
aua swipe --from <id>     # scroll a specific container
```

Any action accepts `--observe` to return the post-action screen (with fresh ids) in the
same call, so `type → tap send` is two commands, not three.

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

`compact` format drops null fields and verbose defaults for the smallest token footprint. `pretty` is indented JSON. All formats validate against the same pydantic schema.

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

Run `aua --help`, or `aua <command> --help` for any command. Global flags (`--format`, `--serial`, `--config`, `--profile`, `--timeout`, `--log-level`, `--no-cache`) go **before** the subcommand.

| Command | What it does |
|---|---|
| `aua doctor` | Check environment: adb, uiautomator2, devices, provider readiness |
| `aua devices` | List attached devices/emulators |
| `aua analyze` | Capture the screen → element list with IDs (the core command) |
| `aua has "<text>"` | Exit 0 if text is on screen, 1 if not — cheap branch check |
| `aua wait --for "<text>"` | Poll until text appears (or `--idle` / `--for-stable`) |
| `aua tap <id>` / `aua click <id>` | Tap an element by ID |
| `aua long-press <id>` | Long-press an element by ID |
| `aua input <id> "text"` | Focus an element and type (`--submit` fires the IME action) |
| `aua clear <id>` | Clear a text field |
| `aua swipe <up\|down\|left\|right>` | Swipe / scroll (`--from <id>` to scroll a container) |
| `aua scroll-to "<text>"` | Scroll until text is found |
| `aua key <back\|home\|enter\|…>` | Press a hardware/navigation key |
| `aua screenshot [path]` | Save a raw screenshot |
| `aua inspect <id>` | Dump full details for one element |
| `aua app <pkg>` | App control (launch/stop/current) |
| `aua map` | Show the learned map of the current app (`--find "<goal>"` for a route) |
| `aua goto "<goal>"` | Drive the remembered route to a known screen — taps + verifies each hop (`--plan` previews, `--max-steps N`) |
| `aua memory show\|path\|update\|forget` | Manage the per-app learned layout |
| `aua config init\|show\|path` | Manage configuration |
| `aua daemon start\|status\|stop` | Manage the optional warm-state daemon |
| `aua guide` | Print the agent operating manual (`--emit-skill` writes the Claude Code skill) |
| `aua mcp` | Run the MCP server over stdio |

All action commands (`tap`, `long-press`, `input`, `clear`, `swipe`, `scroll-to`, `key`) accept **`--observe`** to return the post-action screen inline (an `observation` with fresh element IDs), skipping a follow-up `analyze`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aua doctor` shows **adb: FAIL / not found on PATH** | Install platform-tools and add them to `PATH` — see [Installing adb](#installing-adb-platform-tools). |
| `no device found` (exit 3) | Start an emulator or attach a phone; confirm with `adb devices`. |
| Device shows as **`unauthorized`** | Accept the "Allow USB debugging" prompt on the device. If it never appears: `adb kill-server && adb start-server`, then reconnect. |
| Device shows as **`offline`** | Re-plug the cable / cold-boot the emulator; `adb reconnect`. |
| `multiple devices attached` | Pass `--serial <id>` (get the id from `aua devices`). |
| First command is slow / times out | `uiautomator2` is pushing its helper agent on first connect — retry once it settles, then use `aua daemon start` to keep the connection warm. |
| `uiautomator2 is not installed` | Reinstall the package — `uiautomator2` is a base dependency, not an extra. |
| `analyze` returns few/no elements | The hierarchy is empty (Compose/Flutter/WebView/canvas). Force vision: `aua --format compact analyze --source vision --annotate`. |
| Typing does nothing on Android 14+ | Handled automatically (accessibility `set_text` on the focused field); make sure the field is actually focused first. |

---

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design decisions and the hierarchy-first thesis.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — landscape research behind the approach.
- [`PRD.md`](PRD.md) — the full product requirements document.
- [`SMOKE.md`](SMOKE.md) — manual smoke-test checklist against a live device.
- [`.claude/skills/android-ui-analyser/SKILL.md`](.claude/skills/android-ui-analyser/SKILL.md) — the operating manual an AI agent loads (also via `aua guide`).
