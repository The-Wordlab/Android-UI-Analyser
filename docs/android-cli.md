# Optional Android CLI tools in AUA

AUA can use Google's Android CLI for development, deployment and diagnostics while keeping its
existing Android, iOS and web automation backends. Installing Android CLI is optional. No ordinary
AUA command installs it or silently changes automation backends.

```bash
aua devtools status
aua devtools setup
aua devtools catalogue
aua devtools catalogue preview.render
```

`setup` downloads the official launcher into AUA's user-local tools directory, verifies its
version, and requires no administrator privileges. An existing `android` executable on PATH is
also supported. To select a particular installation or SDK, use a task configuration:

```yaml
platforms:
  android:
    cli_path: /absolute/path/to/android
    cli_sdk: /absolute/path/to/Android/sdk
```

The integration was developed against Android CLI **1.1.16717390**. `status` reports the installed
version and compatibility independently of whether an Android device is available. An absent or
incompatible binary produces an explicit error for optional operations; it does not disable AUA.

## Discover and run

```bash
aua devtools catalogue docs.search
aua devtools run docs.search "Compose accessibility"
aua devtools run info
aua devtools run build :app:assembleDebug --project /path/to/project --timeout 600
aua devtools run analyze /path/to/app.apk
aua devtools run studio.check
```

`catalogue OPERATION` is the authoritative argument/option contract for an installed AUA version.
Positional arguments are separate argv values; `--options` takes a JSON object, not shell code.
For example, a repeated native option uses a JSON array. The optional `--project` sets the child
process working directory without changing the agent's directory. Output includes operation,
provider, elapsed time, exit status, bounded stdout/stderr, parsed JSON when available, and artifact
paths. Truncation is explicit. A failure or timeout is never retried automatically.

| Family | Examples | Context |
|---|---|---|
| Project development | `create`, `describe`, `build` | Host project; builds execute project code |
| Package analysis | `analyze` | APK, AAB, AAR, JAR files |
| Compose | `preview.list`, `preview.render` | Host compilation; no emulator or IDE |
| SDK maintenance | `sdk.list`, `sdk.install`, `sdk.update`, `sdk.remove` | Host SDK |
| Documentation | `docs.search`, `docs.fetch` | Official Android knowledge service |
| Agent resources | `skills.list`, `skills.find`, `skills.add`, `init` | Explicit host configuration |
| Studio | `studio.check`, `studio.analyze-file`, `studio.find-usages`, `studio.version-lookup` | Running compatible Studio instance |
| Deployment | `install`, `run` | AUA's leased target |
| UI | `layout`, `input.tap`, `input.text`, `input.scroll`, `screen.capture` | AUA's leased target |
| Profiling | `profiler.systemtrace.*`, `profiler.heapdump.capture` | Leased target with durable cleanup |
| Field diagnostics | `signals.context`, `signals.crashes.*`, `signals.heaps.*` | Authorized Google account/project |
| Remote devices | `device.remote.*` | Explicit cloud reservations and account access |
| Virtual devices | `emulator.*` | Start/stop/delete retain AUA ownership checks |

See the [complete command reference](android-cli-reference.md) and live catalogue for supported
options, confirmation requirements, and restrictions. Cloud provisioning, SDK removal and account
changes are explicit administrative operations. AUA never creates a paid reservation while doing
ordinary UI testing. Provisioned definitions/reservations are retained administrative resources;
they are not described as temporary session state.

## Previews before deployment

```bash
aua preview list :app --project /path/to/project
aua preview render :app --project /path/to/project \
  --fqn com.example.demo.HomeScreenKt.HomePreview --output /path/to/evidence
aua preview inspect /path/to/evidence/HomePreview_0.json
```

Preview rendering **compiles the target**. It saves the cost of device installation and launch,
not the compilation itself. Component trees expose semantic properties, layout bounds, source
locations and modifiers. Agents can inspect these alongside PNGs, locate the source of a layout
problem, and run supported assertions against the saved tree. A preview is evidence for its
rendered configuration; it is not proof of navigation, networking, lifecycle or live accessibility.
See [Compose preview evidence](compose-previews.md).

## Profiling an ordinary AUA session

```bash
aua session start --goal 'Check the sample interaction and collect a trace' --apk /path/to/app.apk
aua profile start --package com.example.demo --output /path/to/interaction.perfetto-trace
# Perform the interaction using the current observation's element IDs.
aua profile stop
aua profile heap com.example.demo --output /path/to/memory.hprof
aua session finish
```

AUA assigns a session ID from the lease owner, platform, target and profiling family. It records
the exact cleanup action **before** prepare/start/capture. Explicit stop/cancel removes the undo
only after success; interruption or failure retains it for normal device teardown. Heap captures
and traces have different identities, so a successful heap capture cannot erase a trace's undo.
Native device/session overrides are rejected. Profiling support also depends on Android/API,
device and app configuration; the CLI's diagnostic output is preserved when prerequisites fail.

If a heap capture times out or is interrupted, inspect or cancel its owned diagnostic session
without changing an active trace:

```bash
aua profile status --heap
aua profile cancel --heap
```

The default `profile status` and `profile cancel` select the trace family. Generic calls use
`aua devtools run profiler.session.status --profile-kind heap` or
`aua devtools run profiler.session.terminate --profile-kind heap`. MCP uses the same
`development_tool` operations with `profile_kind: "heap"`. Inspect an uncertain capture before
starting another one; cancellation targets only the caller's selected profiling family.

## Deployment and input comparisons

The native AUA backend remains the default. Within an ordinary AUA device session, these calls
explicitly exercise the optional implementation:

```bash
aua devtools run install /path/to/app.apk --timeout 300
aua devtools run run /path/to/app.apk --timeout 300
aua devtools run layout
aua devtools run input.tap 'text=/^Continue$/,clickable=true'
```

Device calls use AUA's exact leased serial. Native UI inspection/input temporarily releases
UIAutomator2's exclusive Android UiAutomation slot; the following AUA observation reconnects.
Mutation results distinguish command delivery from a failed observation, preventing blind replay.
Optional `input.*` operations use Google's selectors from the current `layout`, not AUA's `el:`
handles. Native text selectors match substrings by default; the example uses an anchored regular
expression for an exact label. Ordinary `aua tap-and-analyze <el:id>` continues to use AUA's stable
IDs. See the [native selector syntax](https://developer.android.com/tools/agents/android-cli/commands/input_tap).
Neither an upstream speed claim nor fewer command names establishes a performance improvement:
compare cold/warm execution, changed/unchanged APKs, input handoff overhead and correctness on the
same fixture before changing defaults.

## Agent-executed journeys

Google's Journeys are instructions interpreted by the calling AI agent; Android CLI does not
contain a standalone `journey run` command or an embedded model. AUA's authored journeys likewise
keep the agent in control while providing ordered instructions, deterministic checks and explicit
visual assessments. They reuse AUA's goal-session state, target ownership and cleanup.

```bash
aua journey validate /path/to/journey.yaml
aua journey start /path/to/journey.yaml --apk /path/to/app.apk
aua journey status
# Act using the latest observation.
aua journey check
aua journey finish
```

An instruction is never treated as proof that it ran. Failed assertions do not advance the step.
Visual assessments reference the exact previously observed fingerprint and remain labeled
agent-assessed. See [journey format and evidence](journeys.md).

## CLI, MCP and adapter contract

MCP exposes `development_tools_status`, `development_tools_setup`, `development_tools_catalogue`
and `development_tool`, plus the `preview_*` and `journey_*` convenience tools. Every surface calls
the same Engine methods. Host-only work does not acquire a device; unsupported adapters fail with
an explicit capability error.

`development_tools` is a structural host-service protocol in `platforms/services.py`.
`device.development_tools` is the semantic per-target runtime contract. Android's implementation
lives in `platforms/android_cli.py` and is loaded by `AndroidPlatform`; core code never shells out
to `android`. Profiling cleanup is platform-neutral ledger data replayed through the runtime.
Argument values and options are redacted from AUA's developer-tool journal entries.
Executed checks and their limits are recorded in [validation evidence](android-cli-validation.md).

Official references: [overview](https://developer.android.com/tools/agents/android-cli),
[command catalog](https://developer.android.com/tools/agents/android-cli/commands),
[preview rendering](https://developer.android.com/tools/agents/android-cli/commands/preview_render),
[profiling](https://developer.android.com/tools/agents/android-cli/commands/profiler_systemtrace),
and [Journeys](https://developer.android.com/tools/agents/android-cli/journeys).
