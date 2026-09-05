# Platform foundation v1

Status: implemented on `refactor/platform-foundation-v1`, rebased onto `713223a`; the 2026-09-05
Codex and Claude Fable 5.1 review corrections are being verified before merge.

This milestone gives AUA's existing platform strategy a versioned, enforceable boundary. Its
definition of done is deliberately stronger than "an adapter class exists": an independently
packaged, non-Android adapter must be able to implement an attached target and optional virtual
target without changing the Engine, CLI, MCP server, daemon, dashboard, teardown, or shared
storage code.

The milestone contains no iOS implementation and no Apple automation dependency. A strict fake
platform is the proof that the contract is genuinely external rather than an Android abstraction
renamed after the fact.

## Implemented result

- `android_ui_analyser.platforms` is the stable plugin facade. It exports API version 1, the
  adapter/runtime contracts, normalized app/target/geometry/diagnostic values, conformance
  helpers, capability names, and typed virtual-target request/result/service contracts.
- Installed strategies are discovered lazily through the `aua.platforms` entry-point group. Only
  the selected plugin is imported; duplicate names, load failures, incompatible API versions,
  invalid options, incomplete capabilities, and unsupported operations have stable typed errors.
- `TargetRuntime.target_id` is primary. `serial`, Android package/activity fields, AVD nouns, and
  established Android command payloads remain compatibility projections rather than requirements
  imposed on a new plugin.
- Shared hierarchy, screenshot, input, waits, verified scrolling, app-context transitions,
  diagnostics, flow execution, CLI/MCP routing, daemon/dashboard/capture workers, and teardown
  resolve the selected adapter instead of importing Android tooling.
- Target/app state is namespaced with `TargetRef(platform, target_id)` and
  `AppRef(platform, app_id)`. Existing untagged state remains Android-compatible; explicitly
  referenced caller-authored flow files may remain portable.
- Detached workers receive only the selected opaque option mapping through an inherited anonymous
  file descriptor. A local keyed digest identifies the complete mapping and named environment
  references without persisting their
  contents; changed or missing configuration fails closed instead of replaying against an unknown
  endpoint. Plugin option redaction covers nested/camel/snake secret names and credential-bearing
  URLs.
- Persistent target mutations have a catalogue entry, write-ahead registration, and neutral
  adapter replay, or an explicit `undo_op=None` rationale for an operation that has no truthful
  undo. Watchdogs and opportunistic reapers retain pending records when target or configuration
  identity cannot be proven.
- Virtual targets use typed definition, attached-target, and owned-instance identities. Session,
  CLI, MCP, cleanup, and Android emulator compatibility aliases share one Engine implementation.
- The strict external fixture imports only the public facade, is built as a valid wheel, is found
  by normal entry-point metadata in a fresh isolated process, and passes hierarchy/screenshot,
  non-identity geometry, tap, text, key, wait, verified-swipe, and unsupported-capability checks
  while Android imports and native subprocess execution are blocked.

Recovery deliberately stores a configuration fingerprint, not plugin credentials or a serialized
adapter. Therefore a target ID must be routable within the selected platform configuration.
Process-death recovery recreates the adapter from that same configuration; if it has changed or is
unavailable, AUA leaves the undo pending and tells the operator to restore with the original
configuration and its original local identity key. Empty option mappings need no key. A missing
or changed boot identity also leaves target-bound undos pending; a reboot is not evidence that
persisted settings, permissions, or files disappeared.

## Compatibility policy

- Android remains the built-in default and existing commands keep their behavior.
- Existing public names such as `serial`, `package`, `activity`, `android_version`, `--serial`,
  `--avd`, and `aua emulator` remain accepted. Neutral internal names are added first; aliases are
  removed only in a future major release, if ever.
- Existing serial-only state is treated as Android state. Migrations are read-compatible and never
  discard a pending undo or live lease.
- Optional capabilities remain optional. A platform need not reproduce Android-only proxy,
  database, feature-flag, microphone, helper-agent, or shell facilities.
- Missing optional operations fail with `platform_capability_unsupported`; an adapter that claims
  an incomplete contract fails with `platform_capability_invalid`. A non-Android selection never
  falls back to Android tooling.

## Target architecture

1. `PlatformAdapter` owns target discovery, connection, native-tree normalization, platform
   configuration validation, and host-wide capability services.
2. A neutral target runtime owns semantic per-target operations. Android's uiautomator2 runtime,
   key codes, shell grammar, paths, logcat, and ADB recovery live only in Android-owned modules.
3. Capability specifications declare their scope (adapter, runtime, or service) and complete
   structural surface. Shared code resolves a capability before invoking an optional operation.
4. `TargetRef(platform, target_id)` identifies target state everywhere, including detached
   processes and crash recovery. `AppRef(platform, app_id)` separates learned platform UI state.
5. `DisplayGeometry` defines the transform between native automation coordinates and AUA's
   canonical screenshot-pixel space. Elements, OCR/detection boxes, annotations, screenshots, and
   input all use that one canonical space.
6. CLI and MCP are transport adapters over one Engine operation. They do not independently invoke
   native runtimes or platform services.

## Implementation slices

### 1. Neutral identities, geometry, and plugin configuration

- Add neutral target/app/status/geometry value objects and compatibility projections.
- Add namespaced `platforms.<name>` option dictionaries. The selected adapter validates its own
  options; configuration continues to store references to secrets rather than secret values.
- Preserve existing Android config and JSON output.

### 2. Neutral runtime extraction

- Extract the target runtime contract into `platforms/runtime.py`.
- Move `Uiautomator2Device`, Android discovery/connect helpers, selector translation, key codes,
  shell fallbacks, and Android paths behind `AndroidPlatform`.
- Keep `android_ui_analyser.device` as a compatibility facade.
- Remove the legacy Engine `platform.name == "android"` connect/list monkeypatch route.

### 3. Enforceable capabilities and shared call paths

- Replace unchecked capability strings with scoped structural specifications.
- Validate adapter claims at construction and runtime claims after connection/at first use.
- Gate screenshot/input, keyboard, clipboard, accessibility, app lifecycle/status/files/links,
  location, orientation, airplane mode, media, recording, clock, diagnostics, shell, proxy, and
  every optional operation used by shared code.
- Route screenshots and diagnostics through adapter-owned semantic results. Android parses logcat;
  shared observation code consumes normalized diagnostic events.
- Audit all persistent mutations for write-ahead ledger registration before the device change.

### 4. Platform-scoped state and recovery

- Namespace leases, locks, ledgers, watchdogs, daemons, sidecars, journals, captures, sessions,
  dashboard/runtime caches, diagnostic marks, and device-specific backups by `TargetRef`.
- Carry selected platform and a non-secret effective-options fingerprint into detached processes.
- Store platform ownership in new ledger/lease metadata. Read legacy untagged records as Android.
- Sweep pending undo records through the adapter recorded for their target, never whichever adapter
  happens to run the next command.
- Namespace learned maps and flows by `AppRef` unless an artifact explicitly declares itself
  portable.

### 5. Neutral app and evidence semantics

- Use `AppContext(app_id, surface_id)` and neutral target status internally.
- Keep package/activity and Android transport-state spellings as compatibility projections.
- Keep Android launcher, chooser, IME, system-window, crash-dialog, and log parsing behind the
  Android adapter. Shared code consumes normalized outcomes.

### 6. Neutral virtual targets

- Add typed discover/select/start/status/stop/reclaim operations with an owned-instance token.
- Keep AVD images, Play/root/proxy selection, emulator flags, and proxy-image creation Android-only.
- Make session provisioning, CLI, and MCP call one Engine path. Existing emulator commands remain
  Android-compatible wrappers.

The public capability and command noun is `virtual_targets` / `virtual-target`. Historical
`virtual_devices`, `emulator`, `--avd`, `start_emulator`, and `emulator_*` names remain accepted
compatibility aliases. The reusable definition id, attached target id, and exact boot's opaque
instance token are separate values; rollback consumes only the instance token.

### 7. External conformance contract

- Provide a strict fake external platform with no Android-shaped convenience methods.
- Publish an attached-target profile covering discovery/connection, hierarchy, screenshot,
  scaled/rotated coordinates, input, waits, and verified scrolling. Exercise CLI/MCP parity,
  daemon/dashboard/capture/teardown identity, navigation/history/flows, app lifecycle, and optional
  provisioning in the repository integration matrix.
- Make Android imports and native commands fail during the fake-platform suite.
- Test identical target IDs on two platforms, legacy state migration, detached-process option
  identity, process-death recovery, typed unsupported operations, and owned provisioning rollback.
- Freeze `PLATFORM_API_VERSION = 1` only after the conformance suite passes, then document the
  public plugin surface and precise plugin-load/version errors.

## Readiness gates

### Attached-target gate

Result: the installed-wheel attached UI/input profile passed. Shared Engine workflows and
transports have separate integration coverage; they are not all exercised through that wheel.

An external adapter can discover and connect to an already-running target, analyze hierarchy and
visual sources, tap/type/swipe/wait, and use history/flows/goto without changing shared source. A
missing optional feature produces the same typed error through Engine, CLI, and MCP. Android tooling
is unavailable during the installed-wheel profile, which covers hierarchy, screenshot geometry,
tap/type/key/wait, and verified swipe. History/flows/goto use separate repository tests.

### Coexistence and recovery gate

Result: passed, including same-ID platform isolation, legacy Android reads, recorded-adapter replay,
and fail-closed configuration mismatches.

Two platforms may expose the same target ID without sharing a lease, daemon, journal, capture,
backup, or ledger. After simulated process death, each pending device mutation is replayed through
its recorded platform only. Legacy Android state remains readable and recoverable.

### Automatic-provisioning gate

Result: passed with the strict typed fake service and Android compatibility regressions.

A fake platform can select, start, lease, and stop exactly the virtual instance it created. Failure
rollback cannot stop a foreign instance with the same target ID. A platform without provisioning
can still use attached targets and receives a typed refusal if provisioning is requested.

### Plugin-only gate

Result: passed for the published attached-target profile. Optional provisioning and process
recovery have separate typed-fake integration tests, not one installed-wheel end-to-end scenario.
This does not require parity with Android-only services or remove Android dependencies from AUA.

The fake adapter is packaged and loaded through the `aua.platforms` entry point, supplies its own
configuration and native runtime, and passes the published conformance profiles with zero shared
source changes. This provides the implementation seam for a future iOS plugin, not proof of an
unimplemented native transport. A contributor must validate every optional capability they add;
the current evidence does not promise that all future native edge cases need zero core changes.

## Verification

Every slice must pass focused platform, boundary, config, Engine-domain, CLI, MCP, daemon, ledger,
and teardown tests; Ruff; Mypy; `git diff --check`; and the no-app-specific-reference guard. Moving
or adding a persistent operation also runs `test_every_device_mutation_registers_an_undo.py`.

Before handoff, run the full suite and a live Android regression covering unpinned session
start/finish, hierarchy and visual analysis, tap/input/wait, app install/launch/status, one reversible
environment mutation with explicit restore, and teardown with no pending ledger entries.

### Verification record — 2026-09-04

- Full repository suite: `.venv/bin/pytest` completed with 4,406 passed, 16 skipped, and 5 expected
  xfails (4,427 collected) in 94.14 seconds.
- Static gates: `.venv/bin/ruff check .`, `.venv/bin/mypy` (135 source files), and
  `git diff --check` passed.
- Architecture guards passed, including platform-boundary/native-import enforcement, registration
  of every device mutation, worktree placement, and the no-app-specific-reference scan.
- The external-plugin suite passed both in-process selection checks and the valid-wheel/fresh-
  process conformance run with Android modules and native subprocesses blocked.
- Live Android session `bbaa5041db534bb19b2961613360d4fc` passed on an unpinned, automatically
  provisioned `Medium_Phone` (`emulator-5554`). Evidence covered hierarchy plus mixed visual
  analysis, screenshot capture, Settings launch, semantic ID tap, text input and cleanup,
  positive/negative wait predicates, app status, an idempotent helper-APK install, orientation
  `natural -> landscape/left -> natural`, successful session finish, lease release, owned virtual-
  target handoff, and a final teardown status with zero pending device changes.
- This historical record predates the review corrections below and is not the final merge gate.

### Review corrections — 2026-09-05

Claude Code selected `claude-fable-5-1` for an independent read-only review of `713223a..d65e5d3`
(session `0e7e4922-2d6e-40df-9911-52790be9344e`). Its initial verdict was do not merge. The review
confirmed three fixes found independently and identified two additional recovery issues:

- Disjoint, reversible storage encoding prevents delimiter collisions, escaped-name collisions,
  path traversal, and plugin-id case folding on macOS while retaining Android's legacy paths.
- Entry-point aliases use bound subclasses without renaming live adapters.
- MCP exit cleanup retains the original strategy/configuration and exact instance token, calls
  the shared Engine exact-instance operation, and preserves transferred or replacement boots.
  Android starts now publish a fresh nonce token, not a reusable AVD/port stem; cleanup verifies
  the token and recorded process-start identity and re-reads metadata under the target fence.
- All-target teardown reports an unavailable recorded plugin and continues with other platforms.
- Empty configurations no longer depend on a local HMAC key. Nonempty configurations retain the
  fail-closed rule if the key is missing/corrupt, with a per-target report naming that key.
  An automatic or broad force bypass was intentionally rejected: configuration loss must not
  authorize replay against an unproven endpoint. Back up the key with pending ledgers.

Additional review hardening keeps unreadable/malformed ledgers from being overwritten, retains
undos when backups or recorded boot identity are unavailable, applies boot guards to target-facing
services as well as runtime calls, covers environment-reference values in option identity, and
prevents plugin configuration exceptions and nested credential values from leaking into output.

### Final verification — 2026-09-05

- Full suite at `93791aa`: 4,434 passed, 16 skipped, 5 expected xfails, 3 optional-provider warnings
  in 90.41 seconds. Ruff, Mypy (135 source files), and `git diff --check` passed.
- Fresh unpinned Android session `7c3c9d6faf74413ba08024176d652140` automatically provisioned a
  read-only `Medium_Phone` target. Public Settings UI checks covered launch/status, hierarchy and
  mixed analysis, semantic search tap, verified `Display` input and clear, folded waits, a
  visually inspected landscape screenshot, and `natural -> landscape/left -> natural` followed
  by explicit orientation restore. The helper APK install check was already-present/no-push;
  this was not evidence of a fresh APK transfer.
- A separate real MCP call started a parallel read-only target with a unique boot nonce, verified
  its owned status, exercised MCP exit cleanup, and observed that exact target disconnect. The
  original leased target remained available.
- Teardown reported zero pending device changes. Session finish returned `verdict=passed`,
  released its lease, and handed its owned target to the warm pool. Private runtime artifacts
  remain outside the repository under `/tmp/aua-platform-review-xtORZn`.
- Claude Fable 5.1 accepted the identity, cleanup, and key-loss safety decisions on follow-up,
  subject to final verification. It requested two operability fixes: an audited way to abandon
  unrecoverable stale records and per-file isolation/reporting of corrupt ledgers. Both are now
  implemented with Engine/CLI/MCP coverage. Explicit discard requires named keys, a reason and
  confirmation, refuses live leases, archives evidence first, and never loads a platform or
  touches a device. Proven-dead Android boot bookkeeping is also retired without signalling
  potentially recycled process/watchdog ids. Final recheck and updated suite count: pending.

## Claude review brief

Review `origin/main...refactor/platform-foundation-v1` and any explicitly identified corrective
working-tree diff as an architecture and safety review; do not add
iOS code. Treat a finding as blocking when it shows that a separately packaged adapter must change
shared AUA source for its declared capabilities, a non-Android path can reach Android tooling, or a
persistent mutation/recovery path can act on the wrong target/configuration or lose its undo.

Check these claims against source and tests:

1. The public facade and API/version/plugin-load errors are sufficient and do not require imports
   from internal Android or provider modules.
2. Runtime, adapter, and service capability scopes fail explicitly and CLI/MCP converge on Engine.
3. Canonical geometry, normalized elements/AppContext/diagnostics, and verified scrolling contain
   no Android grammar in shared code.
4. Same-named targets/apps remain isolated across every durable store and detached process.
5. Raw plugin options never enter argv, synthesized environment variables, state files, logs, or
   fingerprints; configuration mismatch remains pending and fail-closed.
6. Every device-retained mutation records before acting and can be replayed through the recorded
   adapter, or truthfully declares why no undo exists.
7. Virtual-target rollback consumes the exact owned instance token and cannot stop a foreign boot.
8. The installed-wheel fresh-process conformance proof and the live Android evidence justify the
   readiness-gate results above without claiming optional Android-feature parity.

## Deferred work

- An iOS adapter or any XCUITest, WebDriverAgent, Appium, or `simctl` integration.
- Renaming AUA, its distribution, or established public fields and commands.
- Requiring feature parity for Android-only optional services.
- Splitting the unconditional uiautomator2 dependency until second-platform packaging and installer
  compatibility are designed together.
- Unrelated Engine restructuring after the completed domain-module split.
