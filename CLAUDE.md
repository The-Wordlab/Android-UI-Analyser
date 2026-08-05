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

Then verify `aua` resolves from **anywhere** (it must work from any project directory, like `adb`):

```bash
cd ~ && command -v aua || { uv tool update-shell 2>/dev/null || pipx ensurepath; }  # then open a new shell
```

If it still doesn't resolve, `install.sh` fell back to the project venv — use
`<repo>/.venv/bin/aua` by absolute path (or install `uv`/`pipx` and re-run `./install.sh`).

Then connect an Android device or emulator (README → "Connect a device or emulator") and run
`aua doctor` until `adb` and `devices` are OK. After that, in **any** project, just drive the
app — the skill is active. The operating manual is `aua guide`.

Requirements: **Python 3.11+**, **`adb` on PATH** (Android SDK platform-tools), and a
**device/emulator** (Android 7.0+). See README → Requirements.

## If you're DEVELOPING the tool

- Dev install: `uv pip install -e ".[dev,apple,rapidocr]"` (or `pip`)
- Tests:       `.venv/bin/pytest` (or `uv run pytest`)
- Lint/types:  `.venv/bin/ruff check .` · `.venv/bin/mypy`
- Enable git hooks (once per clone): `git config core.hooksPath .githooks` — keeps the SKILL.md copies in sync on every commit
- **The SKILL.md is generated** — edit `src/android_ui_analyser/guide.py` (the single source),
  never a SKILL.md directly. There are **two** committed copies (project
  `.claude/skills/android-ui-analyser/SKILL.md` + plugin `skills/android-ui-analyser/SKILL.md`);
  the pre-commit hook (`.githooks/pre-commit`) regenerates and stages **both** from `guide.py`
  on every commit, so they can't drift. To regenerate by hand: `aua guide --emit-skill <path>`.
  When releasing a skill/CLI change, bump `version` in both `.claude-plugin/plugin.json` and
  `.claude-plugin/marketplace.json` so `/plugin update` picks it up.
- **Plugin/marketplace**: the repo is its own Claude Code plugin marketplace
  (`.claude-plugin/marketplace.json`, name `the-wordlab`) exposing the `android-ui-analyser`
  plugin (`.claude-plugin/plugin.json`). The MCP server (`aua mcp`) is intentionally not
  bundled — it needs the `aua` binary on PATH.
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

No separate `re-analyze` is required after every state-changing action. By default, each action
returns the post-action screen in `observation` with fresh IDs. Re-run `analyze` only when you
need a different view (`--fields`/`--where-*`, `source vision`, etc.).
Full manual + flag placement rules: run `aua guide`, or read
`.claude/skills/android-ui-analyser/SKILL.md`.
