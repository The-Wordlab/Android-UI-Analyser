# Optional Android CLI reference

AUA can use Google's Android CLI through the `development_tools` platform service.
Android CLI is optional. The ordinary Android, iOS, and web backends keep working
without it. The integration's command grammar was checked against **1.1.16717390**.
AUA rejects the unrelated legacy SDK command also named `android`.

## Install and discover

```sh
aua devtools status
aua devtools setup
aua devtools catalogue
```

`setup` explicitly downloads Google's launcher over HTTPS into the current user's
AUA tools directory. It verifies the executable's version before publishing the
installed file. It does not use sudo, run a downloaded shell script, or alter PATH.
The launcher may download Google's runtime components on first use.

To use an existing installation:

```yaml
platforms:
  android:
    cli_path: /absolute/path/to/android
    cli_sdk: /absolute/path/to/Android/sdk  # optional
```

`AUA_ANDROID_CLI` is another explicit executable override. With no override, AUA
prefers its managed installation, then discovers `android` on PATH. A configured
missing executable does not silently fall back to a different installation.

## Invocation and output

```sh
aua devtools run OPERATION [POSITIONAL_ARGUMENTS] \
  --options '{"named-option":"value"}' --project /path/to/project
```

Operation names and permitted options come from `catalogue`. Options are a JSON
object, using the exact names shown in the catalogue. Flags take booleans, value
options accept strings or finite numbers, and repeatable options accept arrays.
AUA validates options before touching a device. It does not accept arbitrary shell
commands, response files, or device-selection flags hidden in positional arguments.

Host operations need no connected device. Target operations use AUA's selected,
leased target. Profiling always names a session owned by that AUA caller; implicit
"the only active session" selection is never used. The caller cannot override
`device`, `session`, or `sdk` inside the options object. `signals` exposes its cloud
model filter as `device-filter`, since that filter is unrelated to target selection.
For session status/termination after an interrupted heap capture, pass AUA's
`--profile-kind heap` flag outside the options object; MCP uses `profile_kind: "heap"`.
The default selects the trace family. The shortcuts are `aua profile status --heap`
and `aua profile cancel --heap`; they leave an owned trace's cleanup record intact.

Results include `ok`, `operation`, `provider`, `target_id`, `duration_ms`, native
`exit_code`, structured `data` when available, and reported local `artifacts`.
Successful JSON appears once in `data`, rather than being duplicated in `stdout`.
Documentation search results expose titles, `kb://` URLs, and excerpts. Native
text diagnostics remain in `stdout`/`stderr`; each stream is capped at 256 KiB and
reports truncation explicitly. Errors retain these diagnostics. A timeout stops
the child process tree. Mutating commands are never retried automatically.

`--timeout` sets a finite command deadline, up to 3,600 seconds. Some operations
install tools, overwrite resources, or allocate remote reservations; their catalogue
entries state when the CLI requires `--yes`. Authentication uses Google's browser
sign-in and does not accept credentials through AUA options.

## Build, deploy, and previews

```sh
# Build accepts a Gradle task. :app alone is a module, not an assemble task.
aua devtools run build :app:assembleDebug --project /path/to/project --timeout 600

# Install/run and preview commands accept module/variant build references.
aua devtools run install :app --project /path/to/project --timeout 600
aua devtools run run :app --project /path/to/project --timeout 600

aua devtools run preview.list :app --project /path/to/project
aua devtools run preview.render :app --project /path/to/project \
  --options '{"fqn":["com.example.fixture.HomeKt.HomePreview"],"component-tree":true,"output":"/tmp/fixture-previews"}'

aua devtools run analyze /path/to/app.apk
aua devtools run docs.search 'Compose preview semantics'
```

A Compose preview **compiles its target**. It does not require installing or
launching the app on an emulator. The companion component tree can expose source
locations, semantics, modifiers, parameters, and bounds. A preview is evidence
about that rendered component, not proof that a full user journey worked.

The `build-args` array is the explicit build-system passthrough for build,
install/run, and preview operations. Its values follow the native `--` separator:

```sh
aua devtools run build :app:assembleDebug --project /path/to/project \
  --options '{"build-args":["--offline"]}'
```

## Ownership and capability boundaries

- `layout` and `input` operations hand over Android's exclusive UI automation slot
  through the selected runtime. Resulting UI actions still need AUA observation
  and assertion evidence. AUA records the helper's cleanup before execution, stops
  its exact instrumentation package afterward, and removes the helper APK only if
  it was absent beforehand. It also removes only new ADB forwards to that helper's
  exact socket, preserving preexisting mappings. Cleanup checks the recorded device boot and adapter
  configuration. A surviving record is replayed before reconnecting after a crash.
  A different preexisting helper APK is refused rather than overwritten. These UI
  helper lifecycle guarantees are verified for Android CLI `1.1.16717390`; a newer
  CLI needs a verified helper mapping before its `layout`/`input` commands run.
  The verified helper requests only normal permissions (`INTERNET`,
  `ACCESS_NETWORK_STATE`, and `QUERY_ALL_PACKAGES`), so installing it does not
  grant dangerous runtime permissions.
- Emulator start, stop, and removal use AUA's virtual-target routes, preserving its
  lifecycle ownership rules. Stop requires an explicit owned target ID (for example,
  `emulator-5554`). Removal does not expose the native `force` flag or delete a
  running target. They do not stop an arbitrary native CLI default.
- Profile preparation/start and heap capture have a write-ahead AUA undo record.
  Explicit stop/cancel cleans the exact owned session. Failed cleanup preserves
  the record for retry; already-finished sessions are idempotent.
- Remote device reservations remain until removed or expired. Creating a cloud
  reservation can incur provider charges; creation is explicit and is not claimed
  to be automatically rolled back. The provider does not accept a caller-assigned
  reservation ID, so a failed response can require inspection of its reservation
  list before another create attempt.
- `device.hub` may leave its mirroring window open after the command returns.
- Native `install-options` and `run --debug` are not passed through: arbitrary
  package-manager options and persistent debugger settings require their own AUA
  state contracts. Use AUA's ordinary permission-aware install/lifecycle controls.
- Android Studio operations require a running compatible IDE. Production signals
  and remote devices require an authenticated account with access to the selected
  project. AUA reports native failures rather than presenting unavailable data as
  a successful empty result.

## Operation catalogue

The 70 operations below are exposed by the adapter. `target` means a selected AUA
lease is required; `host` means the command does not select a local device.

| Operation | Scope | Purpose |
|---|---|---|
| `analyze` | host | Inspect package size, DEX content, and optimization results. |
| `auth.login` | host | Sign in to Google services in a browser. |
| `auth.logout` | host | Remove the current Google sign-in. |
| `auth.status` | host | Inspect the current Google sign-in. |
| `build` | host | Build a project target. |
| `completion` | host | Install shell completion. |
| `create` | host | Create an Android project from a template. |
| `describe` | host | Discover project targets and build artifact locations. |
| `device.hub` | target | Open a device mirroring and control window. |
| `device.remote.connect` | host | Connect a reserved remote device. |
| `device.remote.create` | host | Reserve a remote device. |
| `device.remote.disconnect` | host | Disconnect a remote device reservation. |
| `device.remote.extend` | host | Extend a remote device reservation. |
| `device.remote.list` | host | List remote device reservations. |
| `device.remote.models` | host | List remote device models. |
| `device.remote.projects` | host | List projects available for remote devices. |
| `device.remote.remove` | host | Remove a remote device reservation. |
| `docs.fetch` | host | Fetch an official knowledge-base article. |
| `docs.search` | host | Search the official Android knowledge base. |
| `emulator.create` | host | Create a virtual device definition. |
| `emulator.list` | host | List virtual device definitions. |
| `emulator.remove` | host | Delete a virtual device definition. |
| `emulator.start` | host | Boot an owned virtual target through AUA. |
| `emulator.stop` | host | Stop an AUA-owned target ID, such as emulator-5554. |
| `info` | host | Inspect the Android development environment. |
| `init` | host | Install Android CLI configuration and agent resources. |
| `input.key` | target | Send a navigation or hardware key. |
| `input.scroll` | target | Scroll a container toward an optional selector. |
| `input.swipe` | target | Swipe from a matching element. |
| `input.tap` | target | Tap a matching element. |
| `input.text` | target | Replace text in a matching input. |
| `install` | target | Deploy APKs or a build target. |
| `layout` | target | Read the live accessibility hierarchy. |
| `preview.list` | host | Discover Compose preview functions in a target. |
| `preview.render` | host | Compile and render Compose previews and component trees. |
| `profiler.heapdump.capture` | target | Capture a JVM heap snapshot. |
| `profiler.session.list` | host | List profiling sessions. |
| `profiler.session.status` | target | Inspect the owned profiling session. |
| `profiler.session.terminate` | target | Terminate the owned profiling session and restore its target. |
| `profiler.systemtrace.prepare` | target | Prepare an owned system-trace session. |
| `profiler.systemtrace.start` | target | Start an owned system-trace recording. |
| `profiler.systemtrace.stop` | target | Stop recording and collect the trace. |
| `run` | target | Deploy and launch APKs or a build target. |
| `screen.capture` | target | Capture and optionally annotate a device screenshot. |
| `screen.resolve` | host | Resolve annotated screenshot labels into coordinates. |
| `sdk.install` | host | Install Android SDK packages. |
| `sdk.list` | host | List installed or available SDK packages. |
| `sdk.remove` | host | Remove Android SDK packages. |
| `sdk.update` | host | Update Android SDK packages. |
| `signals.context` | host | Discover accessible Crashlytics projects and applications. |
| `signals.crashes.describe` | host | Inspect a production crash issue. |
| `signals.crashes.facets` | host | Group production crash issues by dimension. |
| `signals.crashes.list` | host | List production crash, ANR, and nonfatal issues. |
| `signals.crashes.samples` | host | Inspect event samples for a production issue. |
| `signals.heaps.describe` | host | Inspect a production heap-dump event. |
| `signals.heaps.download` | host | Download production heap traces. |
| `signals.heaps.list` | host | List production heap-dump events. |
| `skills.add` | host | Install an official Android skill. |
| `skills.find` | host | Search official Android skills. |
| `skills.list` | host | List official Android skills. |
| `skills.remove` | host | Remove an installed Android skill. |
| `skills.update` | host | Update installed Android skills. |
| `studio.analyze-file` | host | Analyze source using a running Android Studio instance. |
| `studio.check` | host | Discover running Android Studio instances. |
| `studio.find-declaration` | host | Locate symbol declarations using Android Studio. |
| `studio.find-usages` | host | Locate symbol usages using Android Studio. |
| `studio.open-file` | host | Open source in Android Studio. |
| `studio.render-compose-preview` | host | Render a Compose preview using Android Studio. |
| `studio.version-lookup` | host | Look up artifact versions using Android Studio. |
| `update` | host | Update the optional Android CLI installation. |

## Upstream references

The command metadata comes from Google's [command reference](https://developer.android.com/tools/agents/android-cli/commands)
and the downloaded launcher's help output. See the [release notes](https://developer.android.com/tools/agents/android-cli/release-notes),
[Compose preview rendering](https://developer.android.com/tools/agents/android-cli/commands/preview_render),
[system tracing](https://developer.android.com/tools/agents/android-cli/commands/profiler_systemtrace),
and [agent-executed Journeys](https://developer.android.com/tools/agents/android-cli/journeys)
for the corresponding upstream behavior. Journeys are instructions interpreted
by an external agent, not a native `android journey run` process.
