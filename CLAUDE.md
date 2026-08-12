# android-ui-analyser (`aua`) — agent development guide

This repo **is** the `aua` CLI: it gives an AI agent structured "what's on screen and where"
for an Android device/emulator, so you act on **integer element IDs, not pixels**.
Hierarchy-first (tens of ms), with OCR/detection/grounding vision fallbacks for screens the
accessibility tree can't see (Compose/Flutter/WebView/canvas/games).

## If you were handed this repo to USE the tool — set it up

Run the bootstrap. It installs the `aua` CLI **globally** plus equivalent Claude Code and Codex
skills at user level, so the operating protocol is available in every project:

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
`aua doctor` until `adb` and `devices` are OK. Start runtime work with
`aua session start --goal "<what must be verified>"`; the same contract is MCP initialization
guidance. The operating manual is `aua guide --brief`.

Requirements: **Python 3.11+**, **`adb` on PATH** (Android SDK platform-tools), and a
**device/emulator** (Android 7.0+). See README → Requirements.

## If you're DEVELOPING the tool

- Dev install: `uv pip install -e ".[dev,apple,rapidocr]"` (or `pip`)
- Tests:       `.venv/bin/pytest` (or `uv run pytest`)
- Lint/types:  `.venv/bin/ruff check .` · `.venv/bin/mypy`
- **This is a public, app-agnostic repository.** Never commit private knowledge from a tested
  app: its name, package or private scheme, resource id, feature flag, screen copy/name, route,
  or other product detail—not in code, tests, fixtures, comments, docs, generated skills, or
  agent instructions. Use obviously fictional placeholders. Per-app knowledge belongs only in
  the user's config or local AUA memory. Run `tests/test_no_app_specific_refs.py` before publishing.
- Enable git hooks (once per clone): `git config core.hooksPath .githooks` — keeps the SKILL.md copies in sync on every commit
- **The agent guidance is generated** — edit `src/android_ui_analyser/guide.py` (the single source),
  never a SKILL.md directly. There are **two** committed copies (project
  `.claude/skills/android-ui-analyser/SKILL.md` + plugin `skills/android-ui-analyser/SKILL.md`);
  Codex UI metadata is `skills/android-ui-analyser/agents/openai.yaml`. The pre-commit hook
  regenerates and stages all three from `guide.py` on every commit. To regenerate by hand use
  `aua guide --emit-skill <path>` / `aua guide --emit-codex-metadata <path>`.
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
aua db list <pkg>              # discover private SQLite databases (debuggable builds)
aua db query <pkg> <db> "SELECT …"   # coherent host-side snapshot → JSON rows
aua db execute <pkg> <db> "UPDATE …" --yes  # backup + validate + replace + relaunch
```

Database access still uses adb internally, but agents should use the structured `aua db`
surface. Android images often omit `sqlite3`; AUA stops the app, copies the database plus
WAL/SHM through `run-as`, operates with host SQLite, and relaunches by default. Mutations are
data-only, require `--yes`, create a restore point, validate integrity/foreign keys, and remove
stale sidecars before launch. `aua db backups|restore` provides rollback.
The detail view in `aua dashboard` exposes the same database service for human inspection;
browser execute/restore actions add server-verified typed confirmation phrases.

No separate `re-analyze` is required after every state-changing action. By default, each action
returns the post-action screen in `observation` with fresh IDs. Re-run `analyze` only when you
need a different view (`--fields`/`--where-*`, `source vision`, etc.).
Full manual + flag placement rules: run `aua guide`, or read
`.claude/skills/android-ui-analyser/SKILL.md`.
