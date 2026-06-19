# android-ui-analyser (`aua`)

`aua` is a fast, configurable CLI that gives an AI agent structured "what's on screen and where" for Android UI testing. It reads the accessibility/view hierarchy first — returning every element with a stable integer ID, type, text, and bounding box in tens of milliseconds — and falls back to image-based detection and OCR (and optionally a grounding VLM) only on screens the hierarchy cannot see (Compose without semantics, Flutter, WebViews, canvas, games). The agent acts on **integer IDs, not pixels**: `aua tap 4` and `aua input 2 "hello"` compute coordinates internally, eliminating coordinate hallucination and shrinking the token footprint to a compact JSON list.

---

## Install

**Python 3.11+ required.** Base install (macOS / Apple Silicon, recommended extras):

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
3. Re-analyze — IDs may change after a state transition.

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
aua swipe up              # swipe direction (up|down|left|right)
aua swipe --from <id>     # scroll a specific container
```

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
| `2` | Usage error (bad flags, missing argument) |
| `3` | No device / device error |
| `4` | Provider error — all fallbacks exhausted |
| `5` | Config error (invalid YAML, unknown key, bad value) |

Errors print a structured object to stderr: `{"error": {"code": ..., "message": ..., "hint": ...}}`.
