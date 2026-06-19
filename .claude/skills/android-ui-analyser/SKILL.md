---
name: android-ui-analyser
description: >-
  Drive and inspect an Android app's UI on a device/emulator with the `aua`
  (android-ui-analyser) CLI. Returns the screen as a list of elements with stable integer
  IDs + bounding boxes, then acts BY ID — tap/input/swipe/key — so you never guess pixel
  coordinates. Use whenever the task involves an Android device/emulator: "test the
  Android app", "what's on screen", "tap/type/swipe the X", "is <text> visible", "drive
  the emulator", automating or debugging an Android UI flow, or checking a screen after a
  change. Hierarchy-first (tens of ms); falls back to OCR/detection/grounding vision on
  Compose/Flutter/WebView/canvas/game screens the accessibility tree can't see.
---

# android-ui-analyser (`aua`) — drive Android UI by element ID

`aua` reports **what's on screen and where** so you act on **integer IDs, not pixels**.
The core loop: `analyze` → read element IDs → `tap`/`input`/`swipe` by ID → re-`analyze`.

## Running it

Use `aua` if it's on `PATH` (`command -v aua`). Otherwise it lives in the tool's venv:

```
/Users/luzia/repositories/ai/android-ui-analyser/.venv/bin/aua
```

For a clean global command: `pipx install /Users/luzia/repositories/ai/android-ui-analyser`.
Examples below say `aua`; substitute the venv path if it isn't on PATH.

**First, sanity-check the environment** (do this once per session):

```bash
aua doctor      # adb, device reachable, which OCR/detection/grounding providers are ready
aua devices     # attached devices (serial/model/android version)
```

## ⚠️ Flag placement (this bites people)

**Global** flags go BEFORE the subcommand; **command** flags go after:

```bash
aua --format compact --serial emulator-5554 analyze --source vision   # ✅
aua analyze --format compact                                          # ❌ "No such option"
```

- Global (before): `--format json|pretty|compact`, `--serial`, `--config`, `--profile`,
  `--timeout`, `--log-level`, `--no-cache`.
- Command (after): e.g. analyze's `--source/--query/--deep/--cheap/--strategy/--annotate/--with-ocr`,
  has's `--match/--ignore-case/--ocr-fallback`.

## The loop

```bash
aua --format compact analyze        # → JSON: elements[] with id, type, text, bounds, center, clickable
aua tap 4                           # tap element id 4 (alias: click)
aua input 2 "hello@example.com"     # focus id 2 and type (add --submit to fire IME action)
aua --format compact analyze        # RE-ANALYZE: ids are invalidated after any action
```

Read IDs from the latest `analyze`. **IDs are only valid until the next state-changing
action** — after a tap/input/swipe/key the screen changed, so re-`analyze` before acting again.
Use `--format compact` to keep the JSON small (drops null/default fields).

## Cheap presence check — branch on this, don't `analyze`

```bash
aua has "Sign in"                   # exit 0 = visible, exit 1 = not. One-line JSON result.
aua has "Loading" --ocr-fallback    # checks hierarchy, then OCRs the screenshot on a miss
aua has "submit" --ignore-case --match contains   # --match exact|contains|regex
```

`has` is far cheaper than `analyze` (no element list) — ideal for "did the screen change?"
branching and polling: `aua wait --for "Welcome" --timeout 5000`.

## Actions (all take IDs from the last analyze)

```bash
aua tap <id>            aua long-press <id> --ms 600       aua clear <id>
aua input <id> "text"   aua swipe up --percent 60          aua swipe --coords 540 1600 540 400
aua scroll-to "Privacy" aua key back|home|enter|recents|KEYCODE_TAB
aua wait --for "<text>" --timeout 5000      aua wait --idle
aua inspect <id>        # full attributes of one element from the last analyze
aua app launch <pkg> | stop <pkg> | current
```

## Hard screens (Compose / Flutter / WebView / games)

When the accessibility tree is empty/useless, force the vision fallback (detection + OCR):

```bash
aua --format compact analyze --source vision --annotate   # draws numbered boxes to a PNG
```

`meta.annotated_image` is the saved overlay path — open it to see the numbered marks.
`--source hierarchy|vision|auto` overrides the automatic quality gate.

## Find an element by natural language

```bash
aua analyze --query "the Submit button"          # tries the hierarchy first (free)
aua analyze --query "the gear icon top-right" --deep   # escalate to the grounding VLM
aua analyze --query "the blue banner" --cheap    # forbid escalation past the hierarchy
```

Returns the single best-matching element. By default the ceiling is **local vision**;
`--deep` is required to reach the (paid/commercial) grounding model — it never escalates
to a paid provider on its own. `meta.tier_used` tells you which rung actually ran
(`text` < `selector` < `hierarchy` < `vision` < `grounding`).

## Speed: warm daemon

A fresh CLI call pays Python startup + device connect. For a tight loop, start the warm
daemon once — later calls reuse the live connection (hierarchy `analyze` ~40 ms):

```bash
aua daemon start    # aua daemon status | stop
```

Every command works without it (just slower per call).

## Output schema (read these fields)

```json
{ "schema_version": 1,
  "screen":   { "width": 1080, "height": 2400, "package": "...", "activity": "...", "source": "hierarchy|vision|mixed" },
  "elements": [ { "id": 4, "type": "Button", "text": "Sign in", "resource_id": "...",
                  "content_desc": null, "bounds": [x1,y1,x2,y2], "center": [x,y],
                  "clickable": true, "enabled": true, "focused": false,
                  "source": "hierarchy|detection|ocr|grounding", "confidence": null } ],
  "meta":     { "duration_ms": 42, "tier_used": "hierarchy", "path": "hierarchy|vision",
                "providers_used": [...], "annotated_image": null, "device_serial": "..." } }
```

`compact` format drops nulls and default `enabled`/`focused`/`confidence`/`source`.

## Exit codes

`0` ok · `2` usage · `3` no device/device error · `4` provider error (chain exhausted) ·
`5` config error. (`has`: `0` found, `1` not found.) Errors print
`{"error":{"code","message","hint"}}` to **stderr**; JSON results go to **stdout** (pipe-clean).

## Config & providers (only if asked to change perception)

- Config: nearest `.android-ui-analyser.yaml` (project) → `~/.config/android-ui-analyser/config.yaml`
  (user). `aua config show` / `aua config path` / `aua config init`.
- Swap a model with one line, e.g. `ocr.chain: [apple_vision, rapidocr]` → `[rapidocr]`,
  or `grounding.chain: [gemini]` → `[openai]`.
- **Secrets are env-var names only** (`api_key_env: OPENAI_API_KEY`); set the env var, never
  paste keys into config. `doctor`/`config show` never print secret values.

## Gotchas

- Re-`analyze` after every action — IDs change when the screen changes.
- Global flags before the subcommand (see above).
- `--deep` is the only default route to the paid grounding model; default ceiling is local vision.
- Vision/OmniParser is fast only when warm (daemon); a cold call loads the model (~seconds).
- Prefer `aua has "..."` (exit code) over parsing `analyze` JSON for simple yes/no checks.
