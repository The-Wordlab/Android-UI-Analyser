# Smoke Test — on-device / emulator procedure

This test requires a connected Android device or a running emulator. It covers the full
`aua` pipeline end-to-end and confirms all perception paths, actions, and quick checks
work against a real device.

**Default config is license-clean** (no AGPL, no research-only, no paid providers active
unless you opt in). The optional grounding step at the end requires a configured key.

---

## Latency reference (from `docs/ARCHITECTURE.md`)

| Path | Target |
|---|---|
| Hierarchy `analyze` (warm) | < 150 ms |
| Local vision fallback (detect + OCR) | < 600 ms |
| Hosted 7B grounding VLM | 0.5–2 s |
| Commercial multimodal API | 2–6 s |

Use `--log-level info` or check `meta.duration_ms` in JSON output to compare against these targets.

---

## Step 1 — Start an emulator

```bash
# List available AVDs
emulator -list-avds

# Start one (substitute your AVD name)
emulator -avd Pixel_8_API_35 &

# Wait until the device is ready
adb wait-for-device
adb shell getprop sys.boot_completed   # should return "1"
```

Or use Android Studio's Device Manager to start an AVD.

---

## Step 2 — Environment check

```bash
aua doctor
```

Expected output: each subsystem listed with `ok` or a reason it is unavailable.
- `adb` must be present and on PATH.
- A device must appear as reachable.
- `uiautomator2` agent should show as installed (if not, `aua doctor` will hint how to install it).
- OCR/detection/grounding providers show availability based on installed extras and present keys.
- **No secret values should be printed.**

---

## Step 3 — List devices

```bash
aua devices
```

Expected: one row showing serial (e.g. `emulator-5554`), model, Android version, and state `device`.

---

## Step 4 — Analyze the launcher

Navigate to the home/launcher screen on the emulator, then:

```bash
aua --format pretty analyze
```

Expected:
- `meta.path` is `hierarchy` and `meta.tier_used` is `hierarchy`.
- `elements[]` contains several items with `source: "hierarchy"`.
- `meta.duration_ms` is well under 150 ms (warm, after first call).
- No vision providers are mentioned in `meta.providers_used`.

Also confirm compact format:

```bash
aua --format compact analyze
```

The output should be smaller — no null fields, no `enabled`/`focused`/`confidence` at default values.

---

## Step 5 — Analyze a sample app

Launch any installed app (e.g. the Settings app):

```bash
aua app launch com.android.settings
aua --format pretty analyze
```

Confirm elements appear with meaningful `text` or `content_desc` values and `clickable: true` on interactive items.

---

## Step 6 — Quick text check

Pick a string that is visible on the Settings screen (e.g. "Search settings" or "Network"):

```bash
# String that IS on screen — expect exit 0
aua has "Network"
echo "Exit code: $?"

# String that is NOT on screen — expect exit 1
aua has "ZZZ_NOT_ON_SCREEN_ZZZ"
echo "Exit code: $?"
```

Expected:
- First command: `{"found": true, "source": "hierarchy", "bounds": [...]}` and exit code `0`.
- Second command: `{"found": false}` and exit code `1`.

---

## Step 7 — Tap an element by ID

```bash
# Get element list
aua --format compact analyze

# Identify an ID for a tappable element (e.g. "Network & internet" in Settings)
# Suppose it is ID 3:
aua tap 3
```

Expected: the element is tapped; the screen navigates. Confirm with another `aua analyze`.

---

## Step 8 — Type text into an input field

Navigate to a screen with a text input (e.g. the search bar in Settings):

```bash
aua --format compact analyze
# Identify the search input element ID, e.g. ID 1
aua input 1 "wifi"
```

Expected: the text field is focused and "wifi" is typed.

Add `--submit` to send the IME action (Enter / Search):

```bash
aua input 1 "wifi" --submit
```

---

## Step 9 — Vision fallback on a Compose / Flutter / WebView / game screen

Navigate to a screen that uses Jetpack Compose without semantics, a Flutter view, a
WebView, or a canvas-drawn UI. If you do not have such an app, the system browser or a
Flutter demo app works.

```bash
aua analyze --source vision --annotate
```

Expected:
- `meta.path` is `vision` (or `mixed`).
- `meta.providers_used` lists detection and/or OCR providers.
- `meta.duration_ms` is under 600 ms for local detection + OCR.
- `meta.annotated_image` contains a path to an annotated PNG file.

Open the annotated PNG to verify numbered bounding boxes are drawn around detected elements.

---

## Step 10 — (Optional) Grounding with a configured VLM

This step requires a configured grounding provider. Example using Gemini:

```bash
export GEMINI_API_KEY="..."
```

Add to `.android-ui-analyser.yaml` (or pass `--config`):

```yaml
grounding:
  enabled: true
  chain: [gemini]
models:
  gemini: { model: gemini-2.5-flash, api_key_env: GEMINI_API_KEY }
```

Then:

```bash
aua analyze --query "the Settings search bar"
```

Expected:
- If the hierarchy contains a matching element, it resolves cheaply at T1/T2 (no VLM call).
- If not found in the hierarchy, escalates to Gemini grounding; `meta.tier_used` shows `grounding`.
- `meta.duration_ms` is 2–6 s for a commercial API.

Force grounding escalation regardless:

```bash
aua analyze --query "the Settings search bar" --deep
```

---

## Step 11 — Action loop end-to-end

```bash
# Start fresh on the launcher
aua key home

# Get elements
aua --format compact analyze

# Open Settings by tapping its icon (find the ID from analyze output)
aua tap <settings-icon-id>

# Confirm navigation
aua has "Search settings"

# Analyze the new screen
aua --format compact analyze
```

---

## Checklist

- [ ] `aua doctor` completes cleanly; no secrets printed
- [ ] `aua devices` lists the emulator
- [ ] `aua analyze` on the launcher returns hierarchy elements in < 150 ms warm
- [ ] `aua has "<present text>"` exits 0; `aua has "<absent text>"` exits 1
- [ ] `aua tap <id>` navigates the screen
- [ ] `aua input <id> "text"` types into a field
- [ ] `aua analyze --source vision --annotate` runs on a Compose/Flutter/WebView screen; annotated PNG is written
- [ ] (Optional) `aua analyze --query "..."` resolves via grounding when configured
