# android-ui-analyser (`aua`)

`aua` is a fast, configurable CLI that gives an AI agent structured "what's on screen and where" for Android UI testing. It reads the accessibility/view hierarchy first — returning every element with a stable integer ID, type, text, and bounding box in tens of milliseconds — and falls back to image-based detection and OCR (and optionally a grounding VLM) only on screens the hierarchy cannot see (Compose without semantics, Flutter, WebViews, canvas, games). The agent acts on **integer IDs, not pixels**: `aua tap-and-analyze 4` and `aua input-and-analyze 2 "hello"` compute coordinates internally, eliminating coordinate hallucination and shrinking the token footprint to a compact JSON list.

> **New here?** Start with [Installation help](#installation-help). Claude Code and Codex users
> can install one plugin that supplies the AUA skill and starts the matching released MCP server;
> no clone or permanent AUA install is required.

---

## Installation help

Choose the path that matches how you want to use AUA:

| You want to… | Recommended path | Is AUA permanently installed? |
|---|---|---|
| Use AUA from **Claude Code** | [Install the Claude Code plugin](#claude-code-plugin) | No |
| Use AUA from **Codex** | [Install the Codex plugin](#codex-plugin) | No |
| Run one CLI command or pin AUA in automation | [Run it with `uvx`](#run-the-cli-without-installing-aua) | No |
| Use the `aua` command everywhere or contribute | [Clone and bootstrap](#install-the-cli-permanently) | Yes |

The plugin paths are the easiest option for agents: each bundles the operating skill and starts
the local MCP server from the Git tag matching the plugin version.

### Prerequisites

`aua` is a Python CLI that talks to an Android device or emulator over **adb**, using [`uiautomator2`](https://github.com/openatx/uiautomator2). Every installation path needs these four things on the host:

| Requirement | Version | Why / how to get it |
|---|---|---|
| **Python** | **3.11 or newer** | Runs the CLI. Check with `python3 --version`. |
| **uv** | any recent | Runs AUA without a permanent install (`uvx`) and powers the Claude/Codex plugins. ([install](https://docs.astral.sh/uv/getting-started/installation/)) |
| **Android platform-tools (`adb`)** | any recent | `aua` discovers devices and `uiautomator2` drives them through `adb`. Must be on your `PATH`. ([install](#installing-adb-platform-tools)) |
| **An Android device or emulator** | Android 7.0 (API 24) or newer | The screen `aua` inspects — a running AVD emulator **or** a USB-attached phone with USB debugging enabled. ([setup](#connect-a-device-or-emulator)) |

You do **not** need Android Studio's IDE, Gradle, or the app's source code — `aua` works against any app already installed on the device, including release builds. (Android Studio is just the easiest way to obtain `adb` and an emulator.)

Optional, only for specific features:
- **`tesseract`** system binary — only if you enable the `tesseract` OCR extra.
- A **GPU** (CUDA / Apple Metal) — speeds up the `yolo`/`omniparser` detectors and local grounding, but everything also runs on CPU.
- **API keys** (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) — only if you opt into a commercial grounding provider (off by default).
- **Apple silicon + a separately downloaded MLX FunctionGemma base** — only if you opt into
  the local guarded next-call policy. The base is not bundled or downloaded by AUA.

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

### Claude Code plugin

Run these commands inside Claude Code:

```text
/plugin marketplace add The-Wordlab/Android-UI-Analyser
/plugin install android-ui-analyser@the-wordlab
```

Open a new Claude Code session after installation. The plugin supplies the generated AUA skill,
the selector-safety hook, and the MCP server configuration automatically.

### Codex plugin

Run these commands in a terminal:

```bash
codex plugin marketplace add The-Wordlab/Android-UI-Analyser
codex plugin add android-ui-analyser@the-wordlab
```

Open a new Codex session after installation. The plugin supplies the same generated AUA skill and
pinned MCP server as the Claude Code plugin. See OpenAI's
[marketplace CLI documentation](https://developers.openai.com/plugins/build/plugins#add-a-marketplace-from-the-cli)
for the underlying repository-marketplace flow.

### Run the CLI without installing AUA

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run a pinned release
directly from GitHub:

```bash
uvx --from \
  'android-ui-analyser[apple,rapidocr,audio] @ git+https://github.com/The-Wordlab/Android-UI-Analyser.git@v0.14.2' \
  aua --version
```

`uvx` creates an isolated environment and caches it for later calls. Replace `v0.14.2` with the
release you want; no clone, `git pull`, editable environment, or global `aua` executable is needed.
The command is `uvx --from … aua`, not `uv aua`: the shorter `uvx aua` becomes possible only after
the distribution is published to PyPI. Until then the explicit Git source keeps the release origin
and version visible.

For automation, replace `aua --version` with any AUA command while keeping the pinned `--from`
argument. The [latest GitHub Release](https://github.com/The-Wordlab/Android-UI-Analyser/releases/latest)
shows the current tag and its release notes.

### Install the CLI permanently

Clone the repository and run the idempotent bootstrap:

```bash
git clone https://github.com/The-Wordlab/Android-UI-Analyser.git
cd Android-UI-Analyser
./install.sh                       # add --with-policy for the optional local policy runtime
./install.sh --print-plan          # preview without changing anything
```

The script installs `aua` globally through `uv tool` or `pipx` (with a project-venv fallback),
installs equivalent user-level Claude Code and Codex skills, and runs `aua doctor`.

### Connect and verify

Start an emulator or attach an Android device with USB debugging enabled, then check that `adb`
can see it:

```bash
adb devices
```

For a permanent CLI install, verify everything and start goal-oriented work:

```bash
aua doctor
aua session start --goal "inspect the current screen"
```

For a plugin install, open a new Claude Code or Codex session and ask: **“Use AUA to list the
Android devices.”** The agent should use the plugin's MCP server; a global `aua` command is not
required. See [Connect a device or emulator](#connect-a-device-or-emulator) for physical-device,
AVD, Docker, and multi-agent setup.

### Update later

Claude Code plugin:

```text
/plugin update android-ui-analyser@the-wordlab
```

Codex plugin:

```bash
codex plugin marketplace upgrade the-wordlab
codex plugin add android-ui-analyser@the-wordlab
```

Permanent CLI installs can check the latest release with `aua update --check`; clone-based installs
upgrade by checking out the newer tag and re-running `./install.sh`. See
[Releases and updating](#releases-and-updating) for the automation exit-code contract.

### Common setup problems

| Symptom | Fix |
|---|---|
| `uvx: command not found` | [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/), then open a new terminal. |
| `adb: command not found` | Install Android platform-tools and follow [Installing `adb`](#installing-adb-platform-tools). |
| `adb devices` is empty | Boot an emulator, or enable USB debugging on the phone and accept its authorization prompt. |
| The plugin is installed but AUA is unavailable | Confirm `uvx --version` works, then open a new Claude Code or Codex session so the plugin's MCP server starts. |
| `./install.sh` succeeded but `aua` is not found | Run `uv tool update-shell` (or `pipx ensurepath`), then open a new terminal. The bootstrap prints an absolute fallback path if it had to use `.venv`. |

### Developer and source-install details

If you are developing AUA itself, use one of the following editable-install paths.

Base install (macOS / Apple Silicon, recommended extras — Python 3.11+ per
[Prerequisites](#prerequisites)):

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,apple,rapidocr,audio]"
```

Or without uv:

```bash
pip install -e ".[dev,apple,rapidocr,audio]"
```

Global install (no extras):

```bash
uv tool install .        # or:  pipx install .
```

**If you are going to edit the source, install it editable — once:**

```bash
uv tool install --force --reinstall --editable .
```

`aua` then resolves straight to your working tree, so every edit is live and there is no
re-install step to forget. Confirm with
`python -c "import android_ui_analyser.selectors as m; print(m.__file__)"` — it should print a
path inside your clone.

**If you install non-editable, re-installing needs `--reinstall`:**

```bash
uv tool install --force --reinstall .        # picks up your working tree
uv tool install --force .                    # does NOT - silently reuses a cached build
```

`--force` only means "replace the existing tool entry"; uv still resolves `.` from its build
cache, so the second command reports `Installed 1 executable: aua` and leaves the **old** code
in place. Nothing warns you. The failure mode is invisible: you edit, install, test, and measure
the previous build while believing it is the new one - which is how a working fix got recorded
as a fix that did nothing. If a change seems to have had no effect, verify what is actually
installed before doubting the change:

```bash
grep -c my_new_symbol "$(uv tool dir)"/android-ui-analyser/lib/python*/site-packages/android_ui_analyser/*.py
```

And restart the daemon afterwards (`aua daemon stop && aua daemon start`) - a running daemon
keeps serving the code and the config it started with, so a fresh install or an env override
does not reach it.

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
| `audio` | `grpcio` | Authenticated PCM injection into an Android Emulator microphone |
| `yolo` | `ultralytics`, `torch` | UI element detection with user-supplied weights |
| `omniparser` | `ultralytics`, `torch`, `huggingface-hub` | OmniParser detection — **AGPL-3.0, opt-in** |
| `functiongemma` | `mlx-lm` | Apple-silicon-only local policy runtime; base model stays external. Not in the default install — opt in with `./install.sh --with-policy` |
| `hybrid-policy` | `mlx-lm`, `mlx-vlm` | Adds the larger MLX-VLM policy reviewer; `./install.sh --with-policy=hybrid` |
| `proxy` | `mitmproxy` | Headless HTTPS mock / record / replay (`aua proxy`, `aua mock`) |
| `lxml` | `lxml` | Faster XML parse for huge hierarchy dumps |
| `dev` | pytest, ruff, mypy, respx | Development and test tooling |
| `all` | Perception, proxy, and XML extras | Excludes platform-specific `functiongemma`; add it explicitly |

Heavy deps are **lazy-imported** — a missing optional extra never breaks the core CLI. The two
policy extras are the only ones `install.sh` will not install unless asked (`--with-policy`), since
they are hundreds of megabytes of Apple-silicon-only ML runtime for a lane that is off by default
and needs a separately downloaded base model. When one is configured but absent, `aua doctor`
reports it with the exact install command instead of leaving autopilot to hand off every step.

## Releases and updating

AUA follows Semantic Versioning. Every release is a `vX.Y.Z` git tag with a matching
[GitHub Release](https://github.com/The-Wordlab/Android-UI-Analyser/releases); its notes come from
[CHANGELOG.md](CHANGELOG.md), so the release page says what changed without requiring a pull of
`main` or a read through commit history.

Check the installed version against the latest release without touching a device:

```bash
aua update --check          # installed vs latest, release notes, and the upgrade command
aua update --check --json   # one JSON object for automation
```

The exit code is `0` when current, `10` when an update is available, and `1` when the check could
not run. A CI gate can branch on that contract without parsing human text:

```bash
if aua update --check --json > /tmp/aua-update.json; then
  echo "AUA is up to date"
else
  case $? in
    10) jq -r '"new AUA: \(.latest) — \(.release_url)"' /tmp/aua-update.json ;;
    *)  cat /tmp/aua-update.json >&2; exit 1 ;;
  esac
fi
```

Without AUA installed, the equivalent release lookup is one API call:

```bash
curl -fsSL https://api.github.com/repos/The-Wordlab/Android-UI-Analyser/releases/latest \
  | jq -r .tag_name
```

Pin a clone-based install to a release instead of tracking `main`:

```bash
git fetch --tags
git checkout v0.14.2
./install.sh
```

Upgrade by checking out the newer tag and re-running `./install.sh`. Return to the moving branch
with `git checkout main && git pull && ./install.sh`. Maintainers cut releases using the procedure
in [docs/RELEASING.md](docs/RELEASING.md).

---

## Connect a device or emulator

`aua session start` selects from everything `adb` can see. It checks host-wide leases and target
capabilities, then leases a compatible free target to the calling agent process. If none is free,
it boots a matching configured AVD automatically.

### Option A — Emulator (AVD)

Agents do not list, start, name an owner, or acquire a lease before goal work:

```bash
aua session start --goal "verify the change"
# re-use the returned observation; later commands follow the automatic sticky lease
# … drive the flow …
aua session finish                  # restores state/releases lease; keeps its emulator warm

# Capability-aware selection/provisioning:
aua session start --goal "record HTTPS" --needs root,proxy
aua session start --goal "verify Play billing" --needs play
aua session start --goal "show the QA run" --headed
aua session start --goal "verify expand/collapse motion" --animations
```

Each AUA-started emulator checks the inherited Android proxy setting before app launch. A proxy
that is both unowned and confirmed blackholed (no reachable listener/tunnel) is cleared
automatically; reachable foreign proxies and AUA-owned proxies are left intact.

Long emulator boots emit `AUA_PROGRESS` records and ten-second heartbeats on stderr while keeping
stdout reserved for the single final JSON result. If an agent shell yields a live process/session
id, poll that same process; do not launch a duplicate `emulator start`.

**HTTPS proxy / mock** needs a *rootable* Google APIs image — Google Play AVDs refuse `adb root`, so the mitm CA cannot be installed as a system trust and HTTPS recording stays empty:

```bash
aua emulator recommend-proxy        # suggests a small package (no download)
aua emulator ensure-proxy           # one-time google_apis image setup
aua session start --goal "record HTTPS" --needs root,proxy
aua proxy start                     # needs: pip/uv install with [proxy]
aua session finish
```

You can still create AVDs by hand (`sdkmanager` / `avdmanager` / Android Studio). Prefer SDK `cmdline-tools/latest` under `$ANDROID_HOME` over stale Homebrew copies. On Mac, headless defaults to **host GPU** so fans stay quiet; override with `--gpu swiftshader` only for CI without a display.

### Emulator microphone input

`aua` can feed deterministic host audio into an app's microphone through the Android
Emulator's authenticated control API. This is emulator-only (not a USB phone), needs the
`audio` extra, and the AVD must be started with audio enabled:

```bash
aua session start --goal "verify voice input" --audio

aua mic inject sample.wav
aua mic inject sample.wav --rid hold_to_talk       # DOWN → audio → UP (default hold)
aua mic inject sample.wav --rid record_button --control-mode toggle  # tap start, then tap stop
aua mic inject sample.wav 7 --pre-roll-ms 300 --post-roll-ms 500

# macOS convenience: /usr/bin/say creates a temporary 44.1 kHz S16 mono WAV
aua mic speak "Testing one two" --voice Samantha --rate 175 --rid hold_to_talk
```

Input must be an uncompressed RIFF/WAVE file: unsigned 8-bit or little-endian signed 16-bit
PCM, mono or stereo, at 48 kHz or less, and no longer than five minutes. The default delivery
mode is server-backpressured. With a target, `--control-mode hold` remains the default:
DOWN → pre-roll → audio → post-roll → UP. `--control-mode toggle` instead sends one
non-retrying tap to start, waits/injects, then sends one non-retrying tap to the exact same
point to stop. Toggle mode requires a target that is enabled, clickable, and initially off;
when its active state is not exposed as `checked`/`selected`, the caller must establish that
precondition. Toggle mode is best-effort unless the app exposes an active-state/STOP selector:
use short media and require the control to remain actively recording through post-roll. If the
app auto-stops early (timeout, max duration, or recognition completion), the same final tap could
start a new recording instead. Each command then returns the post-action observation with fresh ids.

AUA discovers the endpoint by matching the selected `emulator-<port>` serial to the emulator's
`pid_*.ini` runtime record and sends its bearer token only as gRPC metadata; the token is never
printed. `mic speak` has a focused unsupported-host error off macOS; generate a compatible WAV
with another TTS tool and use `mic inject` there. `mic_delivery_uncertain` means the emulator
returned `INTERNAL` after accepting packets: samples may already have arrived, so inspect
`error.result.observation` and do not repeat the voice action. A timeout or unclassified RPC
close can also happen after partial delivery; its hint likewise requires inspecting the UI
before any new attempt. `mic_delivered_release_failed` means all audio arrived but the target
control gesture did not finish cleanly—do not repeat it; inspect the attached observation and
verify the control state. `mic_toggle_start_uncertain` means the one START tap may have landed:
AUA sends no audio and no blind compensating tap, because recording may be active.
`mic_toggle_stop_uncertain` means START was confirmed but STOP could not be sent or confirmed
safely. In either toggle uncertainty, protect nearby speech, inspect the forced observation,
and never tap or repeat audio blindly. AUA rechecks that the same foreground package owns the
screen before audio and STOP; if ownership changes it refuses to inject or tap stale coordinates.
If injection reports
`mic_emulator_unavailable`, run `aua devices`, then restart only that emulator with `--audio`
if it is offline or absent; never blindly retry either outcome. Android Emulator 36.4.10 has a
known repeat-stream crash, so AUA atomically permits only one injection attempt per emulator
boot on that build—even across workers with separate AUA cache directories. A second attempt
returns `mic_repeat_unsafe`; restart only that emulator and make the next attempt the sole
injection for the new boot.

### Option B — Physical device

1. On the phone, enable **Developer options** (tap *Settings → About phone → Build number* seven times), then turn on **USB debugging**.
2. Connect over USB and accept the **"Allow USB debugging"** prompt.
3. `adb devices` should now list it with state `device`.

### First run

On the first command against a device, `uiautomator2` automatically pushes a small helper agent (the uiautomator/ATX server) to it — there's nothing to install by hand, but that first call is slower while it sets up. Verify the whole chain end-to-end:

```bash
adb devices     # device appears as "device" (not "unauthorized" or "offline")
aua doctor      # checks: adb on PATH · uiautomator2 importable · devices reachable · provider readiness · installed skill is current
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
aua wait-and-analyze --for "Sign in"
aua wait-and-analyze --changed                  # any tree fingerprint change

# Act on elements by ID from the last analyze
aua tap-and-analyze 4
aua input-and-analyze 2 "hello@example.com"
aua swipe-and-analyze up

# Multi-emulator: same command on several serials
# aua fanout --serials emulator-5554,emulator-5556 analyze

# Force the vision fallback + write an annotated screenshot (numbered boxes)
aua analyze --source vision --annotate

# Find the best-matching element for a natural-language description
# Tries the hierarchy first; escalates to grounding only if needed
aua analyze --query "the Submit button"
aua analyze --query "the Submit button" --deep    # force grounding escalation
aua analyze --query "the Submit button" --cheap   # forbid escalation beyond hierarchy

# Ask a vision model about the whole screen. It receives screenshot + element graph,
# and returns structured regions, elements, graph ids, bounds, latency, and token usage.
aua ask "Describe this screen from top to bottom and say where each control is"
```

The **analyze → act → analyze** loop is the core workflow:
1. `aua analyze` returns elements with IDs.
2. The agent picks an ID and acts: `aua tap-and-analyze <id>` / `aua input-and-analyze <id> "text"`.
3. No manual re-analyze needed — by default each action returns the next screen inline (`observation`, with fresh IDs), folding step 1 into step 2.

The observation is **compact by default** (`id,text,rid,clickable`, app nodes only), so it is
also the cheapest way to read the new screen — widen it with `--observe-fields all` or any
field list. You should not need `--no-observe` followed by `analyze`; that pair costs two
round trips for one screen.

When the next screen is slow, say what you are waiting for and the action waits on evidence
instead of a fixed settle window:

```bash
aua tap-and-analyze <id> --until "rid:resultsPanel" --until-timeout 45000
aua tap-and-analyze <id> --until "text:Results,!text:Loading"    # terms are ANDed; ! means absent
aua tap-and-analyze <id> --until 'text:Hello\, explorer'         # \, is a literal comma in one value
aua tap-and-analyze <id> --until "net:POST /v1/chat,text:x ="    # backend replied AND the screen shows it
```

The response then carries `await_outcome` — `satisfied` / `screen-changed` / `timeout` — plus
per-term results, so a slow backend is distinguishable from a hang. Term prefixes are `text:`,
`rid:`, `desc:`, `net:` (a completed HTTP exchange, needs `aua proxy start`) and `log:`
(logcat since the wait began, no proxy needed). AUA keeps hierarchy polling cheap, then uses
available OCR before accepting `!text:` as visually absent or declaring a positive `text:` timeout.

> **Never re-tap on `nothing changed` / `stale_risk`.** The settle can only wait ~1.1s, so a
> slower screen reports "unchanged" for an action that landed. It cannot tell "no effect" from
> "not yet" — re-tapping means a second submit. Re-read, or pass `--until`.

### Several agents at once

With more than one agent running, leases use one host-wide registry even when every run has an
isolated cache. Start goal work with one command:

```bash
aua session start --goal "verify search"             # automatic process-bound lease
aua session start --goal "record HTTPS" --needs root,proxy
aua lease list                                    # administrative inspection only
```

`session start` probes all attached targets (including leased ones), discards a lease as soon as
its owner PID/start-token is dead, selects a compatible free target, or provisions a matching AVD.
Each normal owner gets exactly one device. After AUA assigns it, omit `--serial` from ordinary
commands: the lease is the routing source, and repeating a physical serial adds stale state.
Keep a serial for initial selection and intentional administration/fanout. Asking for another
device first fails with `lease_switch_required` and changes nothing; acknowledge the switch with
the administrative `aua lease acquire <new> --replace`, which cleans and releases the previous device.
`aua fanout` preserves the same rule by giving each explicitly targeted worker its own stable,
one-device logical owner scope.

An orchestrator can delegate the same running emulator without resetting it:

```bash
# orchestrator: freezes its device and returns a five-minute, one-time token
aua lease transfer <serial>
# spawned agent: atomically becomes the lease owner; later commands omit --serial
aua lease accept <token>
# orchestrator can abort an unaccepted offer without releasing the device
aua lease cancel-transfer <serial>
```

Lease transfer does not transfer goal-session or emulator-lifecycle ownership.

Asking for a device someone else holds fails with **exit 9 (`device_leased`)**. Drop a serial only
when it was a redundant stale pin and the existing assignment is acceptable. If the user named
that target, never redirect: wait, provision another device with user intent, or reconcile the
holder identity. `--needs` gets a capable device or a refusal, never a silent mismatch.

The owner label and caller process identity travel separately, including through the warm daemon,
so `AUA_OWNER` stays readable without becoming a 15-minute lock. A crashed agent is immediately
treated as gone; PID reuse is rejected by the recorded process-start token. The sole bounded
exception is an explicit pending transfer: its reservation survives a source crash for at most
five minutes so the spawned receiver cannot lose the device before accepting the token.
`--no-lease` opts out entirely for single-agent scripts.

---

## What the Claude Code and Codex plugins install

`aua` ships one generated skill plus a local MCP server. The skill auto-activates when you ask an
agent to test or inspect an Android app, and the plugin starts the matching released MCP server
through `uvx`.

The plugin path needs `uv`, `adb`, and a device/emulator. It does **not** need `aua` installed
globally: the MCP process runs from the Git tag matching the plugin version. The first start builds
and caches an isolated environment; later starts reuse it. Copy-paste setup and update commands
are collected in [Installation help](#installation-help).

The Claude Code plugin also installs a `PreToolUse` guard. It denies raw `adb` operations AUA already
covers (UI input, recording, screenshots, logs, app lifecycle, settings, and app data), gives the
exact AUA replacement, and asks for user approval on unknown raw-adb operations. Host build tools
such as Gradle are unaffected. AUA-evidence ffmpeg commands are redirected to the built-in
timestamped contact sheet.

[OpenAI's public plugin submission currently requires a remote MCP endpoint](https://developers.openai.com/plugins/guides/submit-claude-plugin),
so AUA's local stdio MCP plugin is distributed from this repository. A skills-only public listing
can be submitted separately without changing the repository plugin.

### Goal-first sessions and acceptance evidence

After [connecting a device or emulator](#connect-a-device-or-emulator), CLI users start with
`aua session start --goal "<what must be verified>"`. Plugin users call MCP `session_start`; the
server provides the same goal-first protocol during initialization.

For deterministic acceptance proof, attach an authored assertion contract and a portable
cross-command artifact directory:

```bash
aua session start --goal "sort the fixture and restore it" \
  --app dev.aua.fixture \
  --contract evals/agent_loop/contracts/classic-sort.yaml \
  --artifacts-dir artifacts/classic-sort --evidence all --junit \
  --wait-for-lease 30
```

Contract checkpoints reuse the flow assertion grammar (`assert`, `assert_order`, `within`,
`same_parent_as`, `contains_all`) and can only pass from one fresh fingerprinted observation;
`--phase-done` cannot complete them. `session finish` refuses to terminate while a contract is
incomplete unless `--allow-incomplete` is explicit. Every analyzed response includes a typed
`observation_contract` saying whether its frame is reusable or another analyze is required.
The bundle keeps redacted `calls.jsonl`, `manifest.json`, linked observation/screenshot evidence,
`report.md`, `result.json`, optional `junit.xml`, the canonical contract, and the candidate flow.

After every checkpoint passes, `aua session candidate-flow NAME` previews only the actions
recorded after the session watermark. Promotion requires deterministic reset and replay:

```bash
aua session candidate-flow classic-price --replay --save \
  --reset-flow evals/agent_loop/flows/reset-fixture.yaml
```

The reset and candidate must both pass; an existing saved flow is never overwritten.

### Keeping the skill current

The SKILL.md and Codex `agents/openai.yaml` are **generated** from the same source as `aua guide`,
so they do not drift from the CLI/MCP capability contract. After upgrading, re-run `./install.sh`;
`aua doctor` reports Claude and Codex skill drift separately. Use the
[release flow](#releases-and-updating) to choose the version before reinstalling.

### Prefer a different MCP client?

`aua mcp` exposes the same actions over stdio (see [MCP server](#mcp-server)). Point another MCP
client at a global `aua` install, or copy the repository's [`.mcp.json`](.mcp.json) configuration
to let `uvx` run the pinned release without installing AUA.

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
| `known_routes` | Outgoing routes from here, e.g. `["tap 'Catalog' → catalog"]` |
| `suggested_gotos` | Ranked, ready-to-run targets, e.g. `["goto product_detail"]` — ordered by what you've navigated to recently |
| `research_tasks` | Open map-quality questions for the calling agent to investigate in source/runtime and submit through `reconcile` |
| `map_hint` | A nudge like `"12 screens mapped — run aua map"` when there's a map but nothing actionable from the current screen |

### Jump to a known screen in one command

```bash
aua goto "product detail"     # replay the remembered steps: selector-matched, verified per hop
aua goto "settings" --plan    # print every step and risk; do not act
aua goto "onboarding"         # safe UI steps run; cross-app/deeplink effects return a refusal preview
aua goto "onboarding" --allow-unsafe  # only after reviewing that preview
aua goto "login" --allow-destructive   # required when a step matches memory.destructive_labels
```

`goto` resolves the goal (fuzzy) against the map, walks the shortest **verified** route from the current screen, and replays each edge's recorded steps — matching by resource-id first, then label — re-checking `known_screen` after every hop. Known in-app hops are hierarchy-only for speed; OCR runs only when a remembered selector is absent or hierarchy-only arrival verification fails. Transit/foreign screens keep automatic OCR. A newly auto-observed edge is provisional until the same transition is observed again; a cross-package transit journey supplies independent corroboration. Selectorless and conflicting edges are retained for audit but rejected from navigation.

An observed route proves that its actions preceded a destination; it does not prove those actions were navigation-only. `goto` therefore preflights the **whole** route before its first step. Ordinary semantic taps, scrolls, waits, and Back remain automatic. Deeplinks, external-package actions, settings/data/environment mutation, app lifecycle changes, and other actions with unproven side effects return `code: unsafe_route` plus the annotated `route`, `risks`, and `required_opt_in`; no route step has run. Re-run with `--allow-unsafe` only after reviewing that preview. Destructive labels (delete / sign out / pay / …) separately require `--allow-destructive`; a route containing both classes requires both flags. Prefer an explicitly authored flow for setup or mutation journeys.

Auth excursions through `memory.transit_packages` (Google sign-in in Chrome/GMS, permission dialogs) are recorded as **one edge on the origin app**, but their cross-package steps require that deliberate unsafe-route opt-in. A step whose identity was redacted (e.g. an account row containing an email) hands off for one manual tap — then re-run `goto --from-here` to resume mid-route, even mid-auth. On divergence it reports work already performed and replans from the actual recognized screen when another verified route exists, without replaying the failed edge. Reaching the target earlier than an obsolete intermediate is success. It exits `0` on arrival, returning the destination's `elements` (fresh ids). It runs through the warm daemon too.

### Replay whole journeys in one call (flows)

A **flow** is a Maestro-style YAML journey — authored directly by you/your agent, or materialized from what you just did. The repeated setup path to the screen under test (reset account → log back in → reach onboarding) becomes one command:

```bash
aua flow run reset_account_google_login --param ACCOUNT="Engineering Team"
aua flow run smoke --dry-run          # print the resolved steps, act on nothing
aua flow run smoke --from-step 4      # resume after fixing a divergence
aua flow run smoke --artifacts-dir artifacts/smoke --evidence failures --junit
aua flow save reach_checkout --last 8        # preview scope/selectors/proof; writes nothing
aua flow save reach_checkout --last 8 --save # save only after review
aua flow list · aua flow show <name> · aua flow delete <name>
```

Flows live under `<memory.dir>/flows/<package>/<name>.yaml`, filed by the app they start in (they still span transit packages by design — an auth leg stays with its origin app); a flow with no `app:` and every flow written before this layout stays directly in `flows/` and still loads. `aua flow list --app <package>` asks for one app's flows. Two apps may own the same flow name: `run`, `show`, and `delete` accept `<package>:<name>`, each `flow list` entry carries the `ref` that loads it, and a bare name matching two apps is refused with both candidates named instead of one being picked for you. Step vocabulary: `launch_app`, `tap` (by `id:`, `desc:`, or `text:`), `input` (with `${PARAM}` substitution), `key`, `swipe`, `scroll_to`, `wait_for`, `wait_stable`, `assert_visible`, rich `assert` predicates, explicit-axis `assert_order`, named `screenshot` checkpoints, and `goto:` to compose map navigation. `assert` shares the `expect` predicates (`exists`, `absent`, `count`, `text_is`, `text_contains`, `checked`, `enabled`, `selected`, `focused`) and adds canonical-tree relationships (`within`, `same_parent_as`, `contains_all`). `assert_order` accepts `axis: horizontal|vertical|reading`; `reading` follows the adapter's normalized structural traversal rather than guessed geometry. Parentless vision elements fail structural assertions explicitly instead of receiving inferred parentage. The same relationships are available from `expect-and-analyze`, suites, and MCP; `analyze --fields parent` exposes the evidence. `flow save` selects only the newest same-origin/context capture segment and is preview-first; only `--save` writes. Its result includes the authoritative path, current existence/collision status, and an exact `save_call` with `--force` when replacement is required. Collision previews also include `invalid_mode_probe`, the exact typed error code and CLI/MCP call for deliberately checking `--force` without `--save`; agents never need to guess. Saving still rechecks atomically, so a preview never authorizes a race. `flow delete` is idempotent and reports `status: already_absent` when cleanup was already complete. New recordings prefer a unique stable resource id, then a unique non-PII content description, then unique stable non-PII text, and refuse a step with no safe selector. The preview's value-free `selector_resilience` explains whether each selector is strong across frames and whether localization or positional ordering can break replay. It never persists typed values — inputs become required `${PARAM_n}` placeholders. A freshly recognized mapped destination is stored as `arrival_screen`. An unmapped destination remains explicit `arrival_status: unverified` unless the immediately preceding analyzed action satisfied a privacy-safe positive `--until` predicate on this exact package/context/frame; only then is that predicate promoted to authored `arrival:` proof with source `satisfied_action_until`. Flows are deliberate authored intent, so destructive steps run by default (`--no-allow-destructive` opts back into the guard). On divergence you get the failing step index, assertion detail, the remaining steps, and the current elements — fix, then `--from-step N`.

`--artifacts-dir DIR` writes a portable run bundle (`flow.yaml`, `result.json`, `manifest.json`, `report.md`, named screenshots, and observations). `--evidence failures` captures the failed frame and platform diagnostics by default; `all` captures every completed leaf step; `none` keeps only explicit `screenshot:` checkpoints. Add `--junit` for `junit.xml`. Reusing a non-empty directory creates a unique run subdirectory instead of overwriting evidence. CLI and MCP both accept exactly one source: a saved `name`, `file`, or inline `yaml` (`--yaml` on the CLI).

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
  - wait_for: {text: "Continue with Example ID", timeout_ms: 15000}
  - tap: "Continue with Example ID"
  - tap: {text: "${ACCOUNT}", package: com.android.chrome}
  - tap: {text: "Continue", package: com.android.chrome}
  - wait_stable
  - assert: {id: productTitle, count: 2}
  - assert: {id: consentSwitch, checked: true}
  - assert_order:
      axis: horizontal
      selectors: [{text: "Basic"}, {text: "Professional"}]
  - screenshot: catalog_sorted
```

### Inspect and manage the map

```bash
aua map                       # learned screens + routes for the current app
aua map --find "image"        # verified route only; provisional evidence is not replay advice
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
  --source agent --agent codex --evidence workspace-alerts
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
# Or add OPENAI_API_KEY instead. `.env` is already gitignored.
set -a; source .env; set +a   # export entries so `aua` and its daemon inherit them
```

### Example config

```yaml
# .android-ui-analyser.yaml  (or aua config init to write the full commented version)
ocr:
  chain: [apple_vision, rapidocr]   # macOS: apple first; cross-platform: just [rapidocr]
  augment_hierarchy: true           # fuse an Apple OCR pass into every hierarchy analyze
  drop_redundant: true              # withhold readings of text the hierarchy already reports

grounding:
  enabled: true
  # Ordered fallback. Missing keys are skipped automatically; reverse this list
  # if you prefer OpenAI whenever both keys exist.
  chain: [gemini, openai]

models:
  apple_vision: { recognition_level: accurate, max_width: 720 }
  gemini: { model: gemini-2.5-flash, api_key_env: GEMINI_API_KEY }
  openai: { model: gpt-5.6-luna, api_key_env: OPENAI_API_KEY,
            reasoning_effort: none, screen_image_detail: high,
            screen_preview_max_width: 720, screen_preview_jpeg_quality: 55 }
```

The grounding factory tries providers in chain order. A provider whose `api_key_env` is
missing is skipped, as are request failures or empty answers. With only `GEMINI_API_KEY`,
Gemini runs; with only `OPENAI_API_KEY`, Luna runs; with both, the first configured provider
wins. `aua ask "…"` uses the same provider-neutral interface and reports the provider/model
that answered. Apple Vision OCR remains local; it downsizes wide screenshots to 720px for
recognition and maps boxes back to original coordinates. Hierarchy and OCR run concurrently,
and the two are fused into one observation, so web content inside a Chrome Custom Tab is visible
to a plain `analyze`. Readings that merely repeat text the hierarchy already reports are
withheld (`ocr.drop_redundant`, default true): they cost tokens on every observation, they made
`tap --text` ambiguous, and a misread of text the tree had right is worse than no reading at
all. Pixel-only text - and any repair of lossy `U+FFFD` labels - always survives. Remote
screen-analysis calls use a compressed preview.

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
| `openai` | Commercial | `openai` | GPT-5.6 Luna vision by default; key via `OPENAI_API_KEY` |
| `anthropic` | Commercial | `anthropic` | Claude vision; key via `ANTHROPIC_API_KEY` |
| `gemini` | Commercial | `gemini` | Gemini vision; key via `GEMINI_API_KEY` |

### Planner (opt-in, `planner.enabled: false` by default)

The fast LLM behind `--assist` and `aua navigate` (see [Optional: LLM assist](#optional-llm-assist-off-by-default)). Text-first (the element list rides in the prompt; a screenshot is attached only on unlabeled screens).

| Provider | License | Config key | Notes |
|---|---|---|---|
| `gemini_flash` | Commercial | `gemini_flash` | Default; Gemini Flash Lite, key via `GEMINI_API_KEY`. Swap the `planner.chain` for any provider you prefer. |

### Guarded next-call policy (opt-in, `policy.enabled: false` and `mode: off` by default)

| Provider | Extra | Platform | License | Notes |
|---|---|---|---|---|
| `functiongemma` | `functiongemma` | Apple silicon | Gemma Terms (adapter); MLX runtime under its own licenses | Bundled ~29 MB v10 LoRA + external pinned MLX base; inert until `policy.enabled: true` |

The repository's code, docs, and training tools remain MIT-licensed. The modified FunctionGemma
adapter under `src/android_ui_analyser/resources/functiongemma/` is a separate model derivative
distributed under the included Gemma terms, prohibited-use policy, and notices.

**Default config remains commercially licensable and model-free at runtime.** No AGPL provider,
remote model, planner, or policy model is active out of the box. OmniParser requires explicit
`accept_agpl: true`; grounding, the planner, and the policy are off until enabled.

---

## Optional: guarded local FunctionGemma policy (off by default)

This is a deliberately narrow local controller for an active goal session, not an unrestricted
Android agent.
AUA deterministically constructs complete, current-frame, stable-selector tap calls, removes unsafe,
unauthorized, destructive, stale, ambiguous, and redundant choices, and gives the model only
an independently built, privacy-screened projection of candidate metadata keyed by opaque integer
IDs. AUA retains the authoritative call map. FunctionGemma cannot author arguments, grant
authorization, or waive cleanup. Its output never replaces AUA's deterministic
`recommended_call`:

- `shadow` records only policy audit metadata; it exposes no model-selected call.
- `advisory` asks the model for a selection; any returned `policy_suggestion` is still separate
  and unexecuted. The bundled v10 manifest authenticates advisory, so an operator who turns the
  policy on can reach this mode without supplying their own adapter.
- Zero eligible candidates produce a structured, non-executing `policy_handoff` in advisory mode.
  A model-selected handoff uses reserved ID `-1` and is accepted only when a pinned adapter manifest
  authenticates that prompt protocol. One action is selected deterministically without loading the
  model. The bundled v10 adapter is invoked for **two, three, or four** eligible candidates, the
  cardinalities its manifest authenticates; anything else reports `unsupported_cardinality` and
  fails closed.

For a separately authenticated adapter whose manifest permits advisory use, explicit
`aua session autopilot` changes the transport—not the trust boundary. The local model selects an
opaque candidate ID and the warm AUA daemon immediately revalidates and executes the corresponding
AUA-authored call itself; a parent agent does not relay that call. The daemon consumes the folded
post-action observation and repeats only while another safe navigation tap is available. It stops
and returns the fresh screen on model handoff, stale/unknown outcome, unchanged frame, repeated
call, input/toggle/wait/proof work, or the configured step/time limit. It currently executes only
guard-approved taps. `policy_suggestion` remains advice and must not be executed manually.

```bash
aua session autopilot --max-steps 6 --max-duration-ms 30000
```

The bundled v10 manifest authenticates advisory, so this lane is reachable without an
externally supplied adapter — but only after an operator turns the policy on. Nothing is loaded
and no memory or compute is spent while `policy.enabled` is `false`, which is the shipped default.
Shadow exists for developing and debugging the policy locally; it has no end-user purpose.

The checked-in adapter is 30,403,414 bytes. The compatible base is approximately 543 MiB and is
**not** in this repository, wheel, or an automatic download path. Review and accept the
[Gemma Terms](https://ai.google.dev/gemma/terms), then manually obtain the pinned external MLX
conversion and revision recorded in the bundled manifest:

```bash
uv pip install -e ".[functiongemma]"
hf download mlx-community/functiongemma-270m-it-bf16 \
  --revision bb327a9ad61044e1496a2bee2365a6b6a6684c72 \
  --local-dir /absolute/path/to/functiongemma-270m-it-bf16
```

The provider verifies the manifest's seven required base files and their aggregate digest
`76aabb2800b6b9e6da9160028dfb233bbfa723d8c33e21623022ca87a8fa9fd5`; unrelated snapshot files
do not affect that identity. Then turn the policy on explicitly — see
[docs/LOCAL_POLICY_SETUP.md](docs/LOCAL_POLICY_SETUP.md) for the full recipe, including the
two-model chain the device runs used:

```yaml
policy:
  enabled: true                   # false in shipped defaults; nothing loads until this is true
  mode: advisory                  # off | shadow | advisory
  chain: [functiongemma]
  max_candidates: 4

models:
  functiongemma:
    model_path: /absolute/path/to/functiongemma-270m-it-bf16
    adapter_path: null            # null selects AUA's packaged v10 LoRA
    max_tokens: 24
```

Check config, dependency, artifact hashes, and daemon compatibility without touching Android or
loading the model:

```bash
aua --config .android-ui-analyser.yaml policy status
```

#### What the bundled v10 adapter actually scores

Measured by an **independently authored** probe: its 150 jobs derive from the CLI surface rather
than from the training generators. This distinction is not pedantic. An in-house probe that shared
its generator's phrasing reported 6/6 on a refusal capability that independent measurement put at
**0/144** — it was scoring the phrasing, not the model.

| Adapter | Probe accuracy (150 jobs) | Refusal | Notes |
|---|---|---|---|
| v8 | 0.387 | — | previous generation |
| v9 | 0.620 | 6/38 | best accuracy of the Gemma line, but taps wrongly on a real device |
| **v10 (bundled)** | **0.600** best checkpoint, 0.471 mean over 16 | 18/38 | 0 invalid outputs |
| Qwen3-1.7B | 0.667 | 4/38 | ~2.3× the latency and ~2× the memory; kept as a `qwen3` profile |

On a real device v10 completed 5/5 navigations and 2/2 refusals with **0 wrong taps**, one refusal
under a deliberately rephrased goal. The shipped bundle reproduces the external checkpoint exactly
on the same 150 jobs (90/150, refusal 18/38, 0 invalid) at ~334 ms per call.

**This is not a promotion.** One seed, no live gate, and refusal is unstable across checkpoints
(0, 2, 0, 5, 4, 18, 0, 3, 8, …), so the 18/38 figure is a best checkpoint rather than a reliable
property. Safety does not rest on it: the deterministic guard removes unsafe, unauthorized,
destructive, stale, ambiguous, and redundant candidates before the model sees anything, and
revalidates before execution. No model in this line has learned refusal reliably enough to be
load-bearing.

Earlier generations, for the record: frozen v3 scored 2,045/2,048 (99.8535%) on synthetic held-out
data yet **failed** its strict static gate and then a production-serializer smoke at 60/96 (62.5%),
with accuracy swinging 37.5 points across target IDs and 54.17 points across target positions — the
gap between a synthetic score and a real one. A failure-driven v4 continuation also failed its
independent gate and was never bundled.

### Failure-driven v4 continuation (not promoted)

V4 learned the production serializer well: validation reached 2,767/2,768 (99.9639%), including
719/720 production-shaped validation cases; the untouched v3 smoke improved from 60/96 to 96/96;
held-out production choices passed at cardinalities two (64/64), three (144/144), and four
(512/512); and the fictional closed loop completed 4/4 cleanly.

It still **failed** the independent combined test: 2,764/2,768 correct (99.8555%), 99.6875%
critical accuracy, and 100% parse success, with zero redundant but four unauthorized selections.
All four were `sequence_recover_unknown`: the model ended the session early with `session_finish`
instead of observing the uncertain outcome with `analyze_screen`. V4 is therefore ignored and not
bundled. The next iteration needs independent recovery-focused data and an evaluation gate that
keeps this failure family isolated.

Reproduction source is checked in as the
[v4 production curriculum](experiments/functiongemma/production_curriculum.py),
[training configuration](experiments/functiongemma/train-lora-v4.yaml),
[static evaluator](experiments/functiongemma/evaluate.py),
[production smoke](experiments/functiongemma/run_production_smoke.py), and
[closed-loop runner](experiments/functiongemma/run_closed_loop.py). Generated datasets, adapters,
and detailed reports remain ignored and are not linked as repository artifacts.

The complete fictional-data generator, validator, MLX LoRA runner, static evaluator, and deterministic
closed-loop simulator are checked in under [`experiments/functiongemma/`](experiments/functiongemma/)
so later agents and contributors can reproduce or improve the adapter without device data.

---

## Adding a platform adapter

Android is the only built-in platform today, but the engine selects it through a platform
strategy. `device.platform`, `--platform`, or `AUA_PLATFORM` can select an installed adapter;
third-party packages register adapters through the `aua.platforms` Python entry-point group.
The choice is process/config scoped (not repeated on every command), and both target actions and
optional services are gated: another platform never silently falls back to ADB.
See [Platform adapters](docs/platform-plugins.md) for the contract and a minimal plugin skeleton.

## Adding a provider

Three steps, zero changes to `engine.py` or `cli.py`:

1. **Subclass** the relevant abstract base from `providers/base.py`:
   - `OcrProvider` — implement `recognize(image) -> list[TextBox]`
   - `DetectionProvider` — implement `detect(image) -> list[Box]`
   - `GroundingProvider` — implement `locate(image, instruction) -> Point|Box`; optionally
     implement `ask(image, question, elements) -> ScreenAnalysisResult` for `aua ask`
   - `PolicyProvider` — implement `select(context) -> int|None`; the context contains only
     privacy-screened projections of guard-approved candidates keyed by opaque IDs, and the
     provider never executes the authoritative call

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

`daemon start` prints a bounded current projection of what the tool already knows about the foreground app — description,
screens, routes, selected deeplinks, login recipes, quirks, plus counts and an `aua about` pointer when more exists. Stale, wrong-version, and superseded facts retain their evidence but are filtered from `about`/`orient`. That is genuinely useful the first
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

## Verified offline mode

Airplane mode alone does not prove an Android device is offline because Wi-Fi may remain active.
Use the reversible network workflow for offline scenarios:

```bash
aua network status
aua network offline --verify   # snapshots controls, disables transports, verifies read-back
# …exercise the offline scenario…
aua network restore            # restores and verifies the original controls/connectivity
```

`network offline` checks airplane mode, Wi-Fi, mobile-data state, and ConnectivityService's
active default network. It exits non-zero if a transport such as Wi-Fi, cellular, Ethernet, or
VPN remains active. The per-device restore point survives separate CLI invocations and is kept
after a failed verification; repeated offline calls never overwrite the original state. Restore
points are boot-aware so a recycled emulator serial cannot apply settings from an older boot.

For constrained-but-online scenarios, apply exactly one reversible profile:

```bash
aua network profile list
aua network profile apply wifi-only
aua network profile apply cellular-only
aua network profile apply slow                 # emulator EDGE bandwidth + 80-400 ms latency
aua session start --goal "verify packet loss" --needs root
aua network profile apply lossy --loss-percent 10
aua network profile status
aua network profile restore
```

Profiles never stack with offline mode or each other. `slow` uses the Android Emulator console
and fails on physical devices. `lossy` uses `tc netem` on the active interface, requires a
rootable Google APIs AVD, preserves whether adbd was originally rooted, and refuses to overwrite
an unfamiliar existing root qdisc. The loss applies to packets leaving the selected interface.
Restore points stay in place until live evidence confirms the
original conditions returned.

---

## Proxy / mock (HTTPS record & replay)

Optional extra: `pip`/`uv` install with `[proxy]` (pulls `mitmproxy`). Apps whose Network
Security Config trusts **system CAs only** need a rootable emulator and a system CA install
(see [Emulator](#option-a--emulator-avd)):

```bash
aua emulator ensure-proxy                  # one-time rootable image setup
aua session start --goal "record login HTTPS" --needs root,proxy
aua proxy start                            # random high port + system CA by default
aua mock record start login_flow
# … drive the app …
aua mock record stop login_flow            # cassette under memory.dir/cassettes/
aua mock map GET /v1/foo --status 200 --body '{"ok":true}'
aua mock replay login_flow
aua proxy stop
aua session finish
```

Empty cassettes almost always mean TLS rejected the forged cert (Play Store AVD / no system
CA) — not “no traffic”.

---

## Dashboard (sneak-peek headless runs)

Agents often drive a **headless** emulator with no window. The dashboard is a detached service on
one exact port, so it keeps running after the command returns and phone bookmarks remain valid:

```bash
# Start/reuse the dashboard. By default it is reachable at a name you can type:
aua dashboard start
# → publishes aua.local over mDNS, binds port 80, serves http://aua.local/ with no token
aua dashboard status
aua dashboard open
aua dashboard stop

# Phone on the same network (Android browsers cannot resolve .local):
aua dashboard qr
# → writes and opens a QR code for the phone URL

# Narrow it when you are not on a network you control:
aua dashboard start --local          # loopback only, no published name
aua dashboard start --auth           # keep the network bind, require an access token
aua dashboard start --name ""        # publish no name
aua dashboard start --port 48765     # pin an exact port

# Compatibility and foreground debugging:
aua dashboard                         # alias for `dashboard start`
aua dashboard start --detail          # focus the first online device
aua dashboard run                     # foreground; Ctrl-C stops it
```

**The default is open on your network.** `aua dashboard start` binds every interface, publishes
`aua.local`, and serves without a token, so the dashboard is something you type rather than
something you scan. Be clear about what that means: anything that can reach the port gets the whole
dashboard, which drives the device, streams logcat (routinely carrying auth tokens and user
identifiers), and queries app databases. Every start prints a warning saying so. Use it on a network
whose members you would hand your unlocked phone to — a home LAN, not an office guest SSID, a
coworking space, or a hotel — and use `--auth` or `--local` anywhere else.

Change the default once instead of typing a flag every time:

```yaml
# $XDG_CONFIG_HOME/android-ui-analyser/config.yaml
dashboard:
  name: aua        # mDNS name to publish; null or "" publishes nothing
  lan: true        # bind every interface rather than loopback only
  auth: false      # require an access token on network access
  port: null       # exact port; null means 80 with a name, else 48765
```

A typed flag always beats config, in both directions: `--auth` re-arms the token for one start and
`--no-auth` drops it, `--local`/`--lan` move the bind, `--name ""` publishes nothing.

Publishing the name needs no privilege — `dns-sd` on macOS and `avahi-publish` on Linux register a
host record as an ordinary user — so nothing edits `/etc/hosts` and nothing prompts for a password.
The record lives in a child process that dies with the dashboard, so a stopped dashboard leaves no
name pointing at a closed port; a host with no publisher simply keeps its IP URL. Resolving `.local`
is the *client's* job: macOS, iOS, and Windows 10+ do it natively, Android browsers do not — so a
phone still uses `aua dashboard qr`.

Port 80 is a preference, not a requirement: macOS lets an ordinary user bind it, Linux does not
without `CAP_NET_BIND_SERVICE`. When the default port cannot be bound the dashboard starts on 48765
and says so, but a port you pinned with `--port` is never quietly moved. Service mode never walks to
a *nearby* port — if the chosen port belongs to another process, startup fails and names the
collision — and `status`, `open`, `qr`, and `stop` follow the running dashboard's port without you
repeating it.

With `--auth`, every page, frame, journal, log, and control endpoint requires the generated access
token; the browser exchanges the one-time URL token for an HttpOnly cookie and removes it from the
address bar. Either way the dashboard speaks plain HTTP, so never expose the port through a router
or public Wi-Fi.

The page live-polls capture frames + recent action marks. In a device detail view, the
**Agent I/O journal** keeps rows compact while they stream; expand any row to inspect the full
agent request and AUA response. Full payloads load only on demand, and credentials, SQL, bind
parameters, typed input, microphone speech, and audio paths remain redacted. Events recorded by an
older AUA version expand to the compact payload that was retained at the time. After input or
microphone commands, free-form response strings are hidden as well: a UI can split or transform a
private value, so exact-string replacement alone is not a privacy boundary.

Grid tiles and device detail headers show the live lease holder separately from the owner that
started the emulator. They also show the AUA idle watchdog state and remaining auto-stop time.
Auto-stop is lease-gated: reaching the idle threshold never stops a device while a live agent still
holds its lease. Emulators AUA did not start report `auto-stop n/a`; an explicit `--idle-stop 0`
reports `auto-stop off`.

In the same detail view, the
**App database workspace** discovers databases for the foreground package, browses schema,
runs bounded read-only SQL, creates/lists restore points, and exposes guarded mutation and
restore actions. Mutation requires typing `MUTATE <database>`; restore requires typing
`RESTORE <backup-id>`. Both keep the same server-side backup and integrity protections as
the CLI. The phone layout prioritizes the live screen, enlarges touch targets, and keeps element
overlays tappable. If no warm daemon is present, aua starts the capture **sidecar** (single-device)
or uses platform screencap per tile (grid).

---

## App database inspection and mutation

For an installed **debuggable** build, use AUA instead of composing `adb run-as`, DB/WAL
copies, host SQLite, and push-back commands yourself:

```bash
aua db list com.example.app
aua db schema com.example.app app.db
aua db schema com.example.app app.db --table messages
aua db query com.example.app app.db \
  "SELECT id, state FROM messages WHERE chatId = :chat" \
  --params '{"chat":"abc"}' --limit 100
```

Android system images frequently omit the `sqlite3` executable. Read-only queries copy the
selected database plus its WAL through `run-as` while the app keeps running, then use Python
SQLite in read-only mode so the current screen and navigation state are preserved. Pass
`--coherent` when transactional coherence matters more than UI continuity; that mode stops the
package, includes WAL/SHM in the snapshot, and relaunches by default (`--no-restart` leaves it
stopped). Queries use SQLite `query_only`, a row limit, and a timeout; blobs are returned as
base64 metadata.

Data mutation is explicit and recoverable:

```bash
aua db execute com.example.app app.db \
  "UPDATE messages SET state = :state WHERE id = :id" \
  --params '{"state":"FAILED","id":"m1"}' --yes

aua db backups com.example.app app.db
aua db restore com.example.app app.db <backup-id> --yes
```

`execute` accepts data statements (`INSERT`, `UPDATE`, `DELETE`, `REPLACE`, `WITH`) and
refuses schema changes, PRAGMA, ATTACH, and transaction control. It always creates a restore
point first, executes one transaction, checks schema stability, foreign keys, and
`integrity_check`, consolidates WAL state, replaces the app database, removes stale sidecars,
then relaunches. `restore` first preserves the current state as another safety backup.
Restore points are device/package/database scoped under AUA's private cache and can contain
user data; query only the rows needed and handle backups accordingly.

---

## MCP server

`aua mcp` runs an MCP server over stdio, exposing the same tools as the CLI. It is a thin adapter over the engine — no separate perception logic.

Tools include (non-exhaustive): `analyze_screen`, `tap_and_analyze`,
`double_tap_and_analyze`, `long_press_and_analyze`, `input_and_analyze`,
`mic_inject_and_analyze`, `mic_speak_and_analyze`,
`clear_and_analyze`, `swipe_and_analyze`, `scroll_and_analyze`,
`scroll_to_and_analyze`, `key_and_analyze`, `wait_and_analyze`,
`wait_stable_and_analyze`, `wait_changed_and_analyze`, `has`,
`expect`, `screenshot`, `inspect`, `goto`, `flow_run`, `navigate`, `policy_status`, `list_devices`,
`emulator_list` / `emulator_status` / `emulator_start` / `emulator_stop` (stop before exit —
MCP also auto-stops emulators it started when the server process ends), `open_link_and_analyze`,
`app`, `install_app`, `resolve`, clipboard/paste/copy/erase,
location/orientation/airplane/network/network-profile/media/record/clock,
`capture_*`, `dev_profile`, `a11y_scroll_and_analyze`, `flags_apply_and_analyze`,
`database_list` / `database_schema` / `database_query` / `database_execute` /
`database_backup` / `database_backups` / `database_restore`,
map/`reconcile_*`/`knowledge_*`,
`proxy_start` / `proxy_stop` / `mock_replay`, `configure`,
`app_log_prefs_get` / `app_log_prefs_set`.

Example MCP client config (Claude Desktop / `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "android-ui": {
      "command": "uvx",
      "args": [
        "--quiet",
        "--from",
        "android-ui-analyser[apple,rapidocr,audio] @ git+https://github.com/The-Wordlab/Android-UI-Analyser.git@v0.14.2",
        "aua",
        "mcp"
      ]
    }
  }
}
```

---

## CLAUDE.md snippet

> Most users should instead install the **plugin** (see [Installation help](#installation-help)) — it activates automatically and keeps the skill and MCP server on the same release. Use the snippet below only if you prefer inline per-project instructions over the plugin.

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

Elements are returned with stable integer IDs. By default, action commands already return the next
screen in `observation` with fresh IDs, so you usually do not need a separate `analyze` immediately
after state-changing actions.

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
  `checked`, `selected`, `scrollable`, `long_clickable`, `password`, `parent`, `source`,
  `confidence`.
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
- **IDs are never renumbered.** The id in a filtered row is the id `aua tap-and-analyze` takes.
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
aua tap-and-analyze <id>              # tap / click an element (returns the next screen by default)
aua input-and-analyze <id> "text"     # focus element and type; add --submit to send IME action
aua tap-and-analyze <id> --no-observe # act WITHOUT returning the new screen (skip the folded analyze)
aua swipe-and-analyze up              # swipe direction (up|down|left|right)
aua swipe-and-analyze --from <id>     # scroll a specific container
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
aua goto "product detail"  # taps + verifies each hop along the remembered route
aua goto "settings" --plan # preview the route without acting
```

Prefer `goto` over manual tapping whenever your target is listed in `suggested_gotos`.

### Typical loop

1. `aua --format compact analyze` → read element IDs from JSON output.
2. `aua tap-and-analyze <id>` or `aua input-and-analyze <id> "text"`.
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
    "known_routes": ["tap 'Catalog' → catalog"],
    "suggested_gotos": ["goto product_detail"],
    "map_hint": null,
    "annotated_image": null,
    "device_serial": "emulator-5554",
    "device_locale": "es-ES"
  }
}
```

`meta.device_locale` is the device's UI locale — labels render in it, so match text you observed on screen (or use locale-proof resource-ids) rather than literals written in another language. On a text miss, `has`/`wait --for`/`scroll-to` echo the locale plus a hint — language-neutral, so a query in any language crossed with any device locale is caught. After `aua explore mine <repo>` has harvested the app's string resources (`values-*/strings.xml`), text lookups bridge locales automatically: `has "Edit basket"` finds "Editar cesta" on an es-ES device (any locale pair) and reports which string key and rendering matched.

`compact` format drops null fields and verbose defaults for the smallest token footprint.
`pretty` is indented JSON. `tsv` is one element per line. `delta` / `msgpack` are for warm-
daemon / native hot paths (unchanged hierarchy → tiny payload). All formats validate against
the same pydantic schema where applicable.

Action-command responses also carry a compact contract so `analyze` is usually not needed:

```json
{
  "ok": true,
  "action": "tap",
  "observation_present": true,
  "known_screen": "chat",
  "stable_elements": [
    { "id": 25, "stable_key": "compose_input" },
    { "id": 26, "stable_key": "send" }
  ],
  "action_diff_summary": {
    "added": 0,
    "removed": 0,
    "changed": 2,
    "prev_count": 17,
    "curr_count": 17
  },
  "note": "No separate analyze needed; state is in observation.",
  "observation": { ... },
  "capture_hint": null
}
```

If `observation_present` is false, the action was run as `--no-observe`; run a normal
`analyze` to refresh ids and state.

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
| `aua doctor` | Check environment plus separate Claude/Codex installed-skill freshness |
| `aua capabilities [--goal "…"]` | Discover the canonical CLI/MCP capability catalogue |
| `aua session start --goal "…"` | Observe once and return one exact next call; optionally add `--contract`, `--artifacts-dir`, `--wait-for-lease`, `--animations` |
| `aua session review\|finish` | Quantify calls; strict contracts keep finish active until proven (`--allow-incomplete` is explicit) |
| `aua session candidate-flow NAME` | Preview a proven action window; reset + replay before `--save` |
| `aua reach "<goal>" [--until …]` | Use verified goto then a matching safe flow, with semantic arrival proof |
| `aua devices` | List attached devices/emulators |
| `aua analyze` | Capture the screen → element list with IDs (the core command) |
| `aua resolve <id\|key>` | Remap a prior id / `stable_key` onto the current screen |
| `aua has "<text>"` | Exit 0 if text is on screen, 1 if not — cheap branch check |
| `aua expect-and-analyze …` | Assert visibility / state (exit 8 on failure) |
| `aua wait-and-analyze --for "<text>"` | Poll until text appears (`--idle` / `--for-stable` / `--changed`) |
| `aua await-and-analyze '<predicate>'` | Wait on ANDed terms — `text:` `rid:` `desc:` `net:` `log:`, `!` = absent. Reports `await_outcome`: satisfied / screen-changed / timeout |
| `aua lease list\|acquire\|release` | Who is driving which emulator (claimed automatically; `--needs root,play,proxy,animations`) |
| `aua fanout …` | Run a command across multiple `--serials` |
| `aua tap-and-analyze <id>` / `aua click-and-analyze <id>` | Tap an element by ID (also `--rid`/`--text`/`--desc`) |
| `aua double-tap-and-analyze <id>` | Double-tap an element |
| `aua long-press-and-analyze <id>` | Long-press an element by ID |
| `aua input-and-analyze <id> "text"` | Focus an element and type (`--submit` fires the IME action) |
| `aua clear-and-analyze <id>` / `aua erase-and-analyze` | Clear a field / backspace N chars |
| `aua hide-keyboard-and-analyze` | Dismiss the IME without navigating away |
| `aua swipe-and-analyze <up\|down\|left\|right>` | Swipe / scroll (`--from <id>` to scroll a container) |
| `aua scroll-and-analyze` / `aua scroll-to-and-analyze "<text>"` | Directional scroll / scroll until text is found |
| `aua key-and-analyze <back\|home\|enter\|…>` | Press a hardware/navigation key |
| `aua open-and-analyze <uri> [--app pkg]` | Open a deeplink (pin package to skip "Open with…") |
| `aua clipboard set\|get` / `paste` / `copy` | Clipboard helpers |
| `aua location set LAT,LON` | Mock GPS |
| `aua orientation set\|get` | Screen orientation |
| `aua airplane on\|off\|toggle` | Radio control only; not proof of offline connectivity |
| `aua network status\|offline\|restore` | Verified, saved, reversible network isolation |
| `aua network profile list\|apply\|status\|restore` | Reversible Wi-Fi, cellular, slow, and lossy conditions |
| `aua media add PATH` | Push media into the gallery |
| `aua record start\|stop PATH` | Screen recording; stop waits for and validates the finalized MP4 |
| `aua capture sheet PATH.png --since last-action` | Timestamped, evenly sampled transition contact sheet (no ffmpeg) |
| `aua mic inject PCM-WAV [CONTROL-ID]` | Inject emulator PCM; optional selector plus `--control-mode hold\|toggle` |
| `aua mic speak "TEXT" [CONTROL-ID]` | macOS `say` → temporary PCM WAV → the same hold/toggle path |
| `aua clock set --ms <unix-ms>` | Set device clock (emulator / rooted) |
| `aua screenshot [path]` | Save a raw screenshot (`--region` / `--scale`) |
| `aua inspect <id>` | Dump full details for one element |
| `aua app exists\|status <package>` | Read package presence/version from the leased target (`exists` exits 1 when absent) |
| `aua app launch\|stop\|kill\|clear\|grant` | App control (`launch --clear` = clearState) |
| `aua install <app.apk> [--launch]` | Install a build (skips the push when that version is already there); `--reinstall` keeps data, `--fresh --yes` wipes it |
| `aua shell COMMAND…` | Bounded read-only diagnostic on the leased target; argv is remote-shell-quoted, unknown/mutating verbs are refused, and each output stream is capped at 256 KiB |
| `aua db list\|schema\|query` | Structured private SQLite inspection for debuggable apps |
| `aua db execute\|backup\|backups\|restore` | Confirmed, backed-up data mutation and rollback |
| `aua emulator list\|status\|start\|stop` | Boot/stop AVDs (`--headless`, `--audio`, `--parallel`, `--gpu`, `--mine`/`--owner`); `start --apk <app.apk> --launch` boots, installs, and opens in one call |
| `aua emulator recommend-proxy\|ensure-proxy` | Suggest/create a small rootable Google APIs AVD |
| `aua flags set\|apply` | Feature-flag writes with verify/restart |
| `aua proxy start\|stop` / `aua mock …` | HTTPS mitm record/map/replay (`[proxy]` extra) |
| `aua capture …` | Session capture / export / explain |
| `aua helper status\|enable\|remove` | Optional on-device helper APK — runs a long flow on the device (rootable targets, off by default) |
| `aua dashboard start|status|open|qr|stop|run` | Persistent browser grid, by default open on your network at `http://aua.local/`; narrow it with `--auth`, `--local`, or `--name ""`; QR for phones |
| `aua logcat` / `aua suite` | Device-clock log windows / scripted suites |
| `aua logcat prefs show\|set\|reset` | Per-app, persisted `app_logs` preferences — ignored or only-wanted tags, priority set, line and per-tag caps |
| `aua dev` / `aua a11y` | Dev options helpers / a11y scroll |
| `aua map` | Show the active-context map (`--all-contexts`, `--audit`, or `--find "<goal>"`) |
| `aua goto "<goal>"` | Drive the remembered route to a known screen — taps + verifies each hop (`--plan` previews, `--max-steps N`) |
| `aua flow run\|save\|list\|…` | Maestro-style YAML journeys with rich assertions, named screenshots, artifact/JUnit bundles, and reversible network steps |
| `aua navigate "<goal>"` | Opt-in planner drive (needs `planner.enabled`) |
| `aua policy status` | Host-only readiness for optional guarded FunctionGemma advice |
| `aua memory show\|path\|update\|forget` | Manage the per-app learned layout (`memory.backend: sqlite` optional) |
| `aua knowledge list\|show\|add\|stale` | Manage scoped, provenance-bearing learned facts |
| `aua reconcile plan\|submit\|status\|apply\|rollback` | Research and transactionally correct a map |
| `aua about` / `aua remember` / `aua orient` | App playbook + orientation blob |
| `aua config init\|show\|path` | Manage configuration |
| `aua daemon start\|status\|stop` | Manage the optional warm-state daemon |
| `aua guide` | Print the canonical manual (`--emit-skill` / `--emit-codex-metadata`) |
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
| `multiple devices attached` | Automatic leasing normally prevents this. If leasing is intentionally disabled, pass `--serial <id>` from `aua devices`. |
| First command is slow / times out | `uiautomator2` is pushing its helper agent on first connect — retry once it settles, then use `aua daemon start` to keep the connection warm. |
| `uiautomator2 is not installed` | Reinstall the package — `uiautomator2` is a base dependency, not an extra. |
| `mic_grpc_unavailable` | Install the optional transport: `pip install 'android-ui-analyser[audio]'`. |
| `mic_audio_disabled` | Restart only the selected emulator with `aua emulator start --audio`; AUA refuses AVDs launched with `-no-audio`. |
| `mic_repeat_unsafe` | Emulator 36.4.10 already had its one safe stream attempt this boot. Do not retry; restart only that emulator with audio enabled. |
| `mic_delivery_uncertain` | Samples may already have arrived despite the emulator's `INTERNAL` close. Inspect `error.result.observation`; do not repeat the voice action. |
| `mic_delivered_release_failed` | Audio arrived, but target-control cleanup failed. Do not repeat it; inspect the attached observation and control state. |
| `mic_toggle_start_uncertain` | START may have landed; no audio or blind STOP was sent. Recording may be active—protect privacy and inspect the forced observation. |
| `mic_toggle_stop_uncertain` | START was confirmed but STOP is unconfirmed/unsafe. Do not retry blindly; inspect the forced observation and control state. |
| `mic_injection_timeout` / `mic_injection_failed` | Samples may already have arrived. Do not retry blindly; inspect the current UI first. |
| `mic_emulator_unavailable` / emulator goes offline during injection | Do not blindly retry. Check `aua devices`; restart only that emulator with audio enabled if it exited. |
| `analyze` returns few/no elements | The hierarchy is empty (Compose/Flutter/WebView/canvas). Force vision: `aua --format compact analyze --source vision --annotate`. |
| Typing does nothing / is slow on Android 14+ | `aua input-and-analyze` prefers accessibility `set_text`, then clipboard paste (restores clipboard), then IME keys. Focus the field first. |
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

---

## Author

Created and maintained by **Eiliya** — design, architecture, and direction. Parts of the implementation were written with AI coding assistants (Claude Code, Cursor); those contributions are recorded as `Co-Authored-By` trailers in the commit history.

Issues and ideas: [The-Wordlab/Android-UI-Analyser](https://github.com/The-Wordlab/Android-UI-Analyser/issues).
