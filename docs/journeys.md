# Agent-executed journeys

A journey combines ordered natural-language instructions with explicit proof. The calling
agent reads a step, uses normal AUA actions to carry it out, then asks AUA to check assertions
or records a visual assessment tied to the observation it actually inspected. AUA does not
silently start another model or treat the instruction text as evidence that the task happened.

Google's [Journeys guidance](https://developer.android.com/tools/agents/android-cli/journeys)
describes this agent-driven testing approach. AUA makes the distinction between deterministic
checks and agent judgement explicit, and uses its existing sessions, leases, device undo
ledger and artifact handling for the run. Journeys themselves do not require Android CLI:
they can use any AUA platform with the required inspection/action capabilities.

## Author a journey

```yaml
version: 1
name: Catalog screen and visual review
package: com.example.catalog
constraints:
  - Do not purchase or delete anything.
steps:
  - id: catalog
    instruction: Open the Catalog screen from the app navigation.
    assertions:
      - assert: {text: Catalog, exists: true}
      - assert: {text: Loading, absent: true}
  - id: details
    instruction: Open the sample item's details.
    assertions:
      - assert: {rid: item_title, exists: true}
      - assert: {text: Continue, exists: true}
  - id: visual_layout
    instruction: Inspect the title and Continue button in the current screenshot.
    review: The item title and Continue button have clear spacing and readable labels.
```

Each step needs a unique ID, an instruction, and exactly one proof type:

- `assertions`: AUA's existing flow `assert` and `assert_order` grammar. All assertions must
  pass on one fresh frame. Actions such as `tap` are rejected in an assertion list.
- `review`: A question for the external agent to judge using the screen. Its answer is
  labelled `agent_assessment` and is never promoted to automated verification.

Constraints are passed back to the agent on every status response; they are instructions,
not an automatic permission-enforcement or action-planning system. Side effects still need
the same authorization as ordinary AUA use. App-specific journeys belong in the app project
or the user's local configuration, not in AUA's public repository.

## Run the journey

```bash
aua journey validate /path/to/catalog-journey.yaml
aua journey start /path/to/catalog-journey.yaml --apk /path/to/app-debug.apk
```

Validation checks the entire document and assertion grammar without touching a device.
Starting uses the existing session bootstrap, including target leasing, optional emulator
provisioning and app launch. It returns the initial observation and the exact current step.
The authored step order replaces inferred goal phases; AUA does not execute the natural
language instruction automatically.

Add `--artifacts-dir /path/to/evidence` to retain the session's full evidence bundle,
`--headed` to show a newly provisioned target, or `--no-provision-target` to require an
available target without starting another one. `--app` overrides the package in the YAML.

For parallel agents sharing one parent process, give each worker its own explicit cache
environment setting and keep it on every AUA invocation:

```bash
AUA_CACHE__DIR=/tmp/aua-catalog-worker aua journey start /path/to/catalog-journey.yaml
AUA_CACHE__DIR=/tmp/aua-catalog-worker aua journey check
```

The environment setting separates sibling workers while keeping the lease registry shared.
Setting only `cache.dir` in a configuration file isolates artifacts but does not distinguish
their lease identities. Do not invent an `AUA_OWNER` or select another worker's target.

Use fresh element IDs and action-bound waits as usual. Once the requested state is visible:

```bash
aua journey check
aua journey status
```

`status`, `check`, `record`, and `finish` accept `--session-id <id>` to select the run returned
by start. Status reads the saved session without acquiring a device, so a completed run can be
inspected after its emulator has stopped:

```bash
aua journey status --session-id '<session-id-from-start>'
```

`check` reads a fresh hierarchy and evaluates every assertion in the current step. A failure
returns the failed checks and observation, keeps that step active, and records the attempt.
It does not retry or run the instruction again. Two ordered assertion steps cannot be
completed using an unchanged observation fingerprint.

For a visual review, inspect the image from an AUA observation, retain its
`observation_contract.fingerprint` (or `observation.meta.fingerprint` in full JSON), and record
concrete facts:

```bash
aua journey record --fingerprint '<fingerprint-from-the-observation>' \
  --evidence 'The item title and Continue button have clear spacing; both labels are readable.'
```

Use `--failed` to record a failed visual review. A fingerprint that differs from the current AUA
observation, a stale frame, or a frame from another target is rejected. Recording reads the
saved evidence without acquiring a target or taking a replacement screenshot behind the agent's
back. Manual evidence cannot complete a step that requires assertions. Failed reviews leave
the step active; a later correction needs another explicit assessment.
Failed checks/reviews and an incomplete finish return CLI exit code 8 while preserving their
structured evidence. Invalid requests, including a mismatched fingerprint, return code 2.

## Finish and interpret the result

```bash
aua journey finish --stop-started-target
```

Finish reuses session cleanup and can stop the exact virtual target AUA started. It refuses
to finish an incomplete journey unless `--allow-incomplete` explicitly abandons it. The
result distinguishes:

| Journey verdict | Meaning |
|---|---|
| `asserted_pass` | Every step completed with automated assertion evidence |
| `agent_assessed_pass` | All steps completed, with at least one explicit agent judgement |
| `failed` | The latest attempt for the current step failed; the step remains active |
| `incomplete` | Required work remains or the run was explicitly abandoned |

`verified` is true only for an all-assertion completed run whose cleanup succeeded. Each
step carries its own proof type, observation fingerprint and evidence identity. The session
is the sole lifecycle/progress authority; a local companion file preserves the authored
instructions and failed attempts for review.

MCP exposes the same `journey_validate`, `journey_start`, `journey_status`, `journey_check`,
`journey_record` and `journey_finish` paths. Pass `yaml` instead of `file` for an inline authored
journey, and `session_id` when selecting an existing run. Visual records require `evidence`
and `fingerprint`; `passed=false` records a failed review. The start options are `artifacts_dir`,
`headed`, and `start_emulator` (`false` corresponds to CLI `--no-provision-target`). Finish with
`retain_started_target=false` to stop the exact target instance started for the journey.
