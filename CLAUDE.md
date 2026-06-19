# android-ui-analyser (`aua`) — guide for Claude Code

This repo **is** the `aua` CLI: it gives an AI agent structured "what's on screen and where"
for an Android device/emulator, so you act on **integer element IDs, not pixels**.
Hierarchy-first (tens of ms), with OCR/detection/grounding vision fallbacks for screens the
accessibility tree can't see (Compose/Flutter/WebView/canvas/games).

## If you were handed this repo to USE the tool — set it up

Run the bootstrap. It installs the `aua` CLI **globally** and installs the Claude Code skill at
**user level** (`~/.claude/skills/`), so the skill auto-activates in **every** project, not just
this one:

```bash
./install.sh        # idempotent — installs aua + the skill, then runs `aua doctor`
```

Then connect an Android device or emulator (README → "Connect a device or emulator") and run
`aua doctor` until `adb` and `devices` are OK. After that, in **any** project, just drive the
app — the skill is active. The operating manual is `aua guide`.

Requirements: **Python 3.11+**, **`adb` on PATH** (Android SDK platform-tools), and a
**device/emulator** (Android 7.0+). See README → Requirements.

## If you're DEVELOPING the tool

- Dev install: `uv pip install -e ".[dev,apple,rapidocr]"` (or `pip`)
- Tests:       `.venv/bin/pytest` (or `uv run pytest`)
- Lint/types:  `.venv/bin/ruff check .` · `.venv/bin/mypy`
- **The SKILL.md is generated** — edit `src/android_ui_analyser/guide.py` (the single source),
  never the SKILL.md directly, then regenerate: `aua guide --emit-skill`
  (writes `.claude/skills/android-ui-analyser/SKILL.md`). This keeps the skill and the
  `aua guide` manual from drifting.
- Adding a perception provider: subclass in `providers/`, register with the decorator in
  `providers/registry.py`, add a `models.<name>` config block — no edits to `engine.py`/`cli.py`.
- Design rationale: `docs/ARCHITECTURE.md`. Full product spec: `PRD.md`.

## How the tool works (quick reference)

```bash
aua --format compact analyze   # → elements[] each with integer id + bounds
aua tap <id>                   # act by id (alias: click)
aua input <id> "text"          # focus + type (--submit fires the IME action)
aua swipe up · aua key back    # directional swipe / hardware key
aua has "<text>"               # exit 0 if present, 1 if not — cheap branch check
aua wait --for "<text>"        # wait on state, don't sleep
```

Re-analyze after every state-changing action — IDs are invalidated once the screen changes.
Full manual + flag placement rules: run `aua guide`, or read
`.claude/skills/android-ui-analyser/SKILL.md`.
