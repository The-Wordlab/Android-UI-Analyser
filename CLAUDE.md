# android-ui-analyser (`aua`) — agent development guide

This repo **is** the `aua` CLI: it gives an AI agent structured "what's on screen and where"
for an Android device/emulator, so you act on **stable element IDs, not pixels**.
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
**device/emulator** (Android 7.0+). See README → Installation help → Prerequisites.

## If you're DEVELOPING the tool

- Dev install: `uv pip install -e ".[dev,apple,rapidocr,audio]"` (or `pip`)
- Tests:       `.venv/bin/pytest` (or `uv run pytest`)
- Lint/types:  `.venv/bin/ruff check .` · `.venv/bin/mypy`
- **This is a public, app-agnostic repository.** Never commit private knowledge from a tested
  app: its name, package or private scheme, resource id, feature flag, screen copy/name, route,
  or other product detail—not in code, tests, fixtures, comments, docs, generated skills, or
  agent instructions. Use obviously fictional placeholders. Per-app knowledge belongs only in
  the user's config or local AUA memory. Run `tests/test_no_app_specific_refs.py` before publishing.
- Enable git hooks (once per clone): `git config core.hooksPath .githooks` — keeps the SKILL.md copies in sync on every commit
- **Worktrees live inside the repo, under `.worktree/`.** Never create one as a sibling of the
  repo (`../android-ui-analyser-wt-<topic>`): siblings escape this repo's `.gitignore`, clutter the
  parent directory, and get orphaned once the branch lands — each one still holding its own ~300 MB
  `.venv`. Create and retire them like this:

  ```bash
  git worktree add .worktree/<slug> -b <branch>   # new branch in a new worktree
  # …work, commit, land it…
  git worktree remove .worktree/<slug> && git branch -d <branch>
  ```

  `.worktree/` and `.claude/worktrees/` (the Claude Code harness path) are both gitignored, so
  either is fine; anything outside the repo root is not. `git worktree list` must only ever show
  the main checkout plus paths under it. To clean up leftovers: a branch whose
  `git log --oneline main..<branch>` is empty is fully merged — remove the worktree and delete the
  branch, nothing is lost. `tests/test_worktrees_stay_inside_the_repo.py` fails on a registered
  worktree outside the repo root.
- **The agent guidance is generated** — edit `src/android_ui_analyser/guide.py` (the single source),
  never a SKILL.md directly. There are **two** committed copies (project
  `.claude/skills/android-ui-analyser/SKILL.md` + plugin `skills/android-ui-analyser/SKILL.md`);
  Codex UI metadata is `skills/android-ui-analyser/agents/openai.yaml`. The pre-commit hook
  regenerates and stages all three from `guide.py` on every commit. To regenerate by hand use
  `aua guide --emit-skill <path>` / `aua guide --emit-codex-metadata <path>`.
- **Release every user-visible change deliberately.** The package/runtime version, both plugin
  manifests, Claude marketplace listing, the Git tag pinned by `.mcp.json`, and README install
  examples must agree;
  `tests/test_the_version_is_the_same_everywhere.py` fails if they drift. Add the user-visible note
  under `## [Unreleased]` in `CHANGELOG.md` in the same commit. Cut the version with
  `scripts/bump-version.sh`, then create the annotated tag;
  `.github/workflows/release.yml` verifies, tests, builds, and publishes it. Full procedure:
  `docs/RELEASING.md`.
- **Plugin/marketplace**: the repo is its own Claude Code and Codex marketplace
  (`.claude-plugin/marketplace.json`, name `the-wordlab`) exposing the `android-ui-analyser`
  plugin through `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`. Its shared
  `.mcp.json` starts the matching release with `uvx`; plugin users need `uv`, not `aua` on PATH.
- **`Engine` is one class spread over `engine.py` plus the `engine_*.py` domain modules.**
  `engine.py` keeps the constructor, properties, context managers and the device
  connect/lease core; every other method is a module-level function in the domain module its
  name suggests (`engine_flows.py`, `engine_navigation.py`, `engine_sessions.py`, …) whose first
  parameter is the `Engine`, bound back as a method in the class body
  (`flow_run = engine_flows.flow_run`). Add a new method in its domain module, then attach it in
  `Engine`; never add method bodies to `engine.py` itself. Helpers shared by several modules live
  in `engine_support.py`; a domain module may import `engine_support` but never `engine` or a
  sibling (that is a cycle). A test that patches a module constant the method reads must patch
  the module that holds the method: `sys.modules[Engine.flags_set.__module__]`. Rationale and the
  module map: `docs/ARCHITECTURE.md` → "Engine layout".
- Adding a perception provider: subclass in `providers/`, register with the decorator in
  `providers/registry.py`, add a `models.<name>` config block — no edits to `engine.py`/`cli.py`.
- Adding any device-facing feature: follow the platform boundary below. A new Android/ADB feature
  is incomplete until it has a platform-neutral contract and is reached through the selected
  adapter.
- **Changing persistent device state? Register the undo.** Anything the device keeps after your
  command returns — a `settings put`, a radio/network change, a moved clock, an enabled service,
  a host process the device points at — must be registered in
  `device_ledger.MUTATION_CATALOGUE` with an undo op, and recorded via
  `Engine.record_device_change(...)` **before** the device is touched. Write-ahead is the whole
  point: a crash after the record leaves a harmless redundant undo, a crash before it leaves a
  device nobody can clean. `tests/test_every_device_mutation_registers_an_undo.py` fails until
  you do this, and it will name your call site. If the change genuinely needs no undo, say so in
  the catalogue with `undo_op=None` and a reason. See `docs/device-teardown.md`.
- Design rationale: `docs/ARCHITECTURE.md`. Full product spec: `PRD.md`.

### Platform boundary — mandatory for new features

`PlatformAdapter` is the gateway between AUA's reusable agent layer (`analyze`, actions, `goto`,
history, maps, flows) and native automation tooling. Android is the only built-in adapter today and
remains the default, but new work must preserve the ability to plug in iOS, web, or another runtime.

- Do **not** add direct `adb`, `adbutils`, `uiautomator2`, emulator-console, `dumpsys`, `logcat`,
  `run-as`, or other platform SDK/tool calls to `engine.py`, CLI, MCP, daemon, or generic services.
- First define a platform-neutral method on `PlatformAdapter`, or a focused capability protocol
  owned and returned by it. The adapter may delegate to platform-owned runtime/services rather than
  accumulating every implementation in one file.
- Per-target operations belong on the semantic `Device` runtime returned by `connect`; host-wide or
  optional operations use `PlatformAdapter.capability(name)`. Capability names and their required
  structural members live in `platforms/services.py`; claiming an incomplete service is a typed
  configuration error.
- Put the Android implementation in `platforms/android.py` or an Android-only module reachable only
  through `AndroidPlatform`, and add the corresponding name to its `capabilities` set.
- Core layers call the selected adapter and use its capability contract. Never add new
  `if platform == "android"` branches there, bypass `PlatformFactory`, or silently fall back to ADB
  when another platform is selected. The explicitly marked legacy monkeypatch shim is migration
  debt and must not be extended.
- Unsupported optional features return a clear unsupported-capability error. Android-only commands
  still use this contract; being Android-only today is not an exception to the boundary.
- CLI and MCP must share the same engine implementation and error behavior.
- Every new platform operation needs a fake-adapter test proving the core has no Android dependency,
  plus an Android adapter regression test.

Raw ADB/native calls are confined to Android runtime/service implementation modules loaded by
`AndroidPlatform`; they are not patterns for generic callers. The architecture test
`tests/test_platform_boundary.py` prevents new core bypasses. The detailed plugin contract is in
`docs/platform-plugins.md`; repository-wide coding-agent instructions are also in `AGENTS.md`.

## How the tool works (quick reference)

```bash
aua --format compact analyze   # → elements[] each with a stable id + bounds
aua tap-and-analyze <id>       # act by id (e.g. rid:continue_btn) and get the resulting screen
aua input-and-analyze <id> "text"  # focus + type + resulting screen (--submit sends IME)
aua swipe-and-analyze up · aua key-and-analyze back
aua has "<text>"               # exit 0 if present, 1 if not — cheap branch check
aua wait-and-analyze --for "<text>"  # wait on state and return the satisfied screen
aua install <app.apk> --launch # put a build on the device and open it (no `adb install`)
aua app exists <package>       # package-manager presence/version on the leased target
aua shell pm path <package>    # leased, argv-quoted read-only diagnostic; 256 KiB per output stream
aua emulator start --apk <app.apk> --launch  # boot + install + launch in one call
aua db list <pkg>              # discover private SQLite databases (debuggable builds)
aua db query <pkg> <db> "SELECT …"   # live host-side snapshot; current UI is preserved
aua db query <pkg> <db> "SELECT …" --coherent  # stop app for transactional coherence
aua db execute <pkg> <db> "UPDATE …" --yes  # backup + validate + replace + relaunch
aua logcat prefs set --app <pkg> --ignore-tag <Tag> --lines 40  # persisted per-app app_logs filter
```

An **optional on-device helper APK** (`helper/`, shipped prebuilt at
`src/android_ui_analyser/data/aua-helper.apk`) can run the leading UI-only stretch of a flow on
the device instead of one host round trip per step — measured 13.9s → 5.6s on a 32-step run. It
is off by default (`helper.enabled`) and that single switch does everything — AUA probes
rootability cheaply, installs the APK and enables the service itself. It needs `adb root`,
and is reached only through the `device_agent` platform capability. Android suppresses accessibility services while
uiautomator2 holds UiAutomation, so the two never run at once: the offload hands the slot over
and takes it back, which costs ~2.9s and is why it only engages past `helper.min_flow_steps`.
Rebuild with `helper/tools/build.py`, which stamps the source digest into the APK, copies it
into `data/`, and is what `tests/test_helper_apk_matches_its_source.py` checks against — a
forgotten rebuild now fails the suite instead of shipping. `helper/tools/build.py --check`
answers the same question without building.

Database access still uses adb internally, but agents should use the structured `aua db`
surface. Android images often omit `sqlite3`; AUA stops the app, copies the database plus
WAL/SHM through `run-as`, operates with host SQLite, and relaunches by default. Mutations are
data-only, require `--yes`, create a restore point, validate integrity/foreign keys, and remove
stale sidecars before launch. `aua db backups|restore` provides rollback.
The detail view in `aua dashboard` exposes the same database service for human inspection;
browser execute/restore actions add server-verified typed confirmation phrases.

No separate `re-analyze` is required after every state-changing action. By default, each action
returns the post-action screen in `observation`. Re-run `analyze` only when you
need a different view (`--fields`/`--where-*`, `source vision`, etc.).
Full manual + flag placement rules: run `aua guide`, or read
`.claude/skills/android-ui-analyser/SKILL.md`.
