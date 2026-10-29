# Optional Android CLI integration: validation evidence

This page records executed checks for the optional developer tools and the surrounding AUA
workflows. A passing test establishes the behavior described here; it is not a general
performance claim or proof that every application behaves the same way.

## Native CLI and Compose checks

The optional Google Android CLI 1.1.16717390 was installed through `aua devtools setup`.
All device calls below used an AUA-owned Android emulator and a fictional Compose fixture;
all host calls used the optional adapter service. These checks ran on 2026-10-29.

| Executed check | Observed result |
|---|---|
| Project creation, description and `:app:assembleDebug` | Generated, described and built the fixture project |
| SDK inventory and APK analysis | Returned the installed SDK inventory and an analysis of the built APK |
| Documentation search | Returned structured titles, knowledge URLs and excerpts |
| Studio discovery | Detected a running IDE; no IDE project edits were performed |
| Preview list, render and inspect | Discovered the fixture preview and produced a PNG and semantic component tree without a device |
| Saved preview assertions | Expected component checks passed; a deliberately missing component failed |
| MCP preview workflow | A real MCP client listed 172 tools and called status/list/render/check; render returned an inline PNG and the counter assertion passed |
| Optional install/run | Deployed and launched the same fixture through Android CLI; AUA returned stable element IDs |
| Optional tap | Counter changed from zero to one, with fresh post-action evidence |
| Optional text entry | Input and echo both showed the supplied fixture text, with fresh evidence |
| Optional scroll-until | Reached the requested last row, visible in the fresh post-action observation |
| First-use helper on a fresh emulator | AUA package inspection confirmed the helper absent before and after a successful native layout call |
| Annotated screenshot | Saved a PNG and reported its path and size as a structured artifact |
| Trace around ordinary AUA interaction | Saved a nonempty Perfetto trace and cleared the owned profile cleanup entry |
| JVM heap capture | Saved a 48,851,825-byte HPROF file and cleared only the heap family's cleanup |
| Finite two-second trace | Returned a 442,404-byte trace artifact with `cleanup_pending: false` and no redundant stop recommendation |
| Trace cancellation after daemon restart | Reconnected and terminated the same caller-owned native session; cleanup was no longer pending |
| Session finish | Restored owned state, released the lease and stopped the exact AUA-started emulator, with no cleanup errors |

The native UI tests exposed a persistent instrumentation helper and an ADB forwarding connection.
AUA now records their baseline before invoking the tool, yields the exclusive UiAutomation slot,
and restores that baseline before resuming ordinary AUA observations. Recovery checks the exact
boot identity, configured SDK/tool context and verified helper package. Failed or unconfirmed
restoration keeps its durable undo. A nonzero native mutation exit remains an uncertain delivery
outcome: agents inspect state instead of repeating the action.

The development session included failed probes while fixing these issues. Its completed goal
verdict does not erase those journal failures or make `review.run_ok` true. Machine-local JSON,
preview images, trace/heap files and final session reports retain the executed evidence.

Cloud reservations, authenticated field diagnostics, SDK removal/update, IDE edits and every
possible project configuration were not exercised live. Their argument, capability, confirmation
and routing contracts are covered by tests; that is not an end-to-end claim about those services.
The UI-helper lifecycle is pinned to the verified CLI version, and optional operations never
replace AUA's default backend implicitly.

### Deployment timing on the same fixture

A warm-host comparison used the same unchanged 11.38 MB debug APK on the fresh emulator.
One warm-up round was discarded, followed by three measured rounds in the fixed order below.
Each row includes CLI startup, deployment/launch, observation and any additional read-only
analysis needed to obtain fresh evidence. The check required the fixture's initial counter
and OK button; an unchanged screen was never relabelled as fresh merely to improve timing.

| Deployment path | Median end-to-end | Three samples | Extra fresh readback |
|---|---:|---|---|
| AUA reinstall and launch | 2.376 s | 2.376 / 10.695 / 2.044 s | None |
| Android CLI standard run | 3.379 s | 3.379 / 3.390 / 2.719 s | One per run |
| Android CLI run with delta flag | 3.023 s | 3.023 / 2.968 / 3.710 s | One per run |
| AUA install-if-needed and launch | 2.296 s | 2.296 / 2.233 / 2.787 s | One per run |

The delta-flag calls all reported **Standard Install**, sending the full APK, so this experiment
shows no actual delta transfer. The slow reinstall sample spent 8.268 seconds in installation;
it is retained above. The small sample, fixed order, warm caches and single fixture do not support
a general speed ranking. These results support keeping the existing backend as the default and
making the alternative explicit. Changed APKs, larger apps and different devices need their own
measurements before selecting a deployment strategy.

## Agent journey on an Android emulator

An independent agent exercised a two-step journey on 2026-10-29 using a local, fictional
Compose fixture. The first step asserted a greeting and an OK button. The second required
the agent to inspect the screenshot and assess readability and spacing. All live interaction
used AUA, including target provisioning, stable element IDs, observation, and cleanup.

| Executed check | Observed result |
|---|---|
| Validate authored YAML | Accepted one assertion step and one visual-review step without a device action |
| Check from the launcher instead of the fixture | Both assertions failed; the first step remained active; CLI exit code 8 |
| Finish before completing the steps | Refused with `journey_incomplete`; the session stayed active; CLI exit code 8 |
| Open the fixture and check again | Both assertions passed on one fresh observation; only the assertion step completed |
| Tap the fresh stable ID for OK with a positive arrival predicate | Action succeeded; arrival was `satisfied`, evidence was fresh, and readiness was `ready` |
| Record a mismatched observation fingerprint | Rejected without advancing; CLI exit code 2 |
| Explicitly record a failed visual-review probe | The visual step remained active; CLI exit code 8 |
| Inspect the screenshot and record concrete visual facts | Completed the visual step as `agent_assessment`, with `verified: false` |
| Finish with `--stop-started-target` | Returned `agent_assessed_pass`; stopped only the exact emulator instance started for the journey, released its lease, and reported no cleanup errors |

The final result retained the assertion step's machine-verified proof and the visual step's
unverified agent judgement separately. The screenshot showed readable greeting and button
labels with space between them. The failed visual-review attempt was an explicit negative
test of progress handling, not a claim that the fixture had a visual defect.

The captured run had 11 top-level calls before its final reporting call, or 12 including that
call, plus one internal action-bound wait. Replaying its saved journal after the accounting
fix classified those 11 calls as three lifecycle calls and eight task calls. The review
reported zero avoidable calls. Its four deliberately induced failures remain failures in
the journal: the run did not declare expected-error probes, so `review.run_ok` remains false.
The completed journey verdict and the historical failure count describe different things.

### Friction found and addressed

- Parallel agents sharing a parent process need a distinct `AUA_CACHE__DIR` environment
  value on every invocation. A configuration-file `cache.dir` alone isolates artifacts,
  but does not provide the worker scope used for lease matching. The journey guide now
  explains this requirement. An initial overlapping test session was explicitly abandoned
  before the isolated run; no concurrent test actions were performed on that target.
- One provisioned emulator reached the transport but failed the Android/package-manager
  readiness deadline. AUA stopped that newly started instance and returned
  `emulator_boot_timeout`. A subsequent bootstrap succeeded. This was not counted as a
  successful journey run or a performance measurement.
- Journey reports now include the authored assertion criteria in the current step.
- Finished journeys no longer offer another check call. Explicit abandonment reports that
  cleanup finished while retaining an `incomplete` verdict.
- Journey start/status/finish now count as lifecycle calls. Assertion checks and visual
  records remain task calls; failed attempts are not relabelled as successes.
- CLI and MCP expose journey provisioning, headed execution, and artifact-directory
  options. MCP source schemas require exactly one of a file or inline YAML.

Saved evidence includes the authored YAML, command responses and exit outcomes, the inspected
screenshot, session and journey records, final cleanup report, and the offline accounting
replay. These machine-local artifacts are kept outside the repository.

## iOS physical-tap smoke check

The iOS adapter's physical-tap change was checked independently on an iOS simulator with
AXe 1.8.0. Its installed help advertised `tap --tap-style physical` as a touch-down/touch-up
delivery path.

An AUA session opened the system Settings app. The agent used a freshly observed General
element ID to navigate forward, then a freshly observed Settings back element ID to return.
Both actions returned satisfied arrival predicates, fresh evidence, and ready observations.
Session finish returned a passed verdict, released the lease, and reported no cleanup errors.
The session review counted four calls including finish and no avoidable calls.

The focused iOS platform/runtime, screenshot, and wait-query tests passed: 55 tests. The smoke
check confirms the two observed navigation actions; it does not establish compatibility with
older AXe versions or every simulator control. JSON action and finish evidence remains local.
